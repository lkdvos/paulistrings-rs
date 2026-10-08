//! [`PartitionedTruncation`] — truncation whose layer pass is collective.
//!
//! In partitioned mode each partition holds a *disjoint subset* of the terms
//! (ARCHITECTURE.md §Partitioning), so a [`TruncationPolicy`] splits cleanly
//! in two:
//!
//! - [`keep_term`](TruncationPolicy::keep_term) is per term and needs nothing: it runs inside the merge, on the complete summed coefficient of a key, and a key lives on exactly one partition.
//! - [`finalize_layer`](TruncationPolicy::finalize_layer) may be *global*, and a partition cannot see the global layer.
//!   [`ApproxTopN`] chooses an octave edge from the histogram of the whole layer — exactly the sum of the per-partition histograms — so one `allreduce` makes every partition choose the same edge, and the union of the retained sets is bit for bit the single-partition answer.
//!
//! Exact [`TopN`](crate::truncation::TopN) has no such reduction: the `n`-th
//! largest magnitude is a distributed *k*-th selection, not a sum. It is
//! rejected at compile time rather than approximated — see the trait docs.

use crate::pauli_sum::PauliSum;
use crate::truncation::builtin::{
    bucket_norms, collapse_to_target, octave_edge, octave_histogram, pick_slot, retain_at_or_above,
    APPROX_BINS, LOG_TARGET,
};
use crate::truncation::{
    And, ApproxTopN, BuiltinTruncation, CoefficientThreshold, CollapseSample, Or, TruncationPolicy,
    WeightCutoff,
};

use super::transport::Collectives;

/// A truncation policy usable in partitioned propagation: its layer finalization is collective.
///
/// # The contract
///
/// A truncation policy that finalizes a layer (`finalizes_layer() == true`) must implement [`PartitionedTruncation`]; the trait's default panics otherwise, since finalizing a layer with no collective form would let partitions diverge silently.
/// [`finalize_layer_partitioned`](Self::finalize_layer_partitioned) is called on every layer, on every partition, in lock-step: a collective is only well-defined if every partition issues the same calls in the same order, so an implementation must not skip a call on a partition that happens to have nothing to do, and must derive its decision only from all-reduced values so every partition applies the identical predicate to its own terms.
/// Under those two rules the retained set is exactly what a single partition holding the whole sum would have retained.
///
/// # Which policies implement it
///
/// [`CoefficientThreshold`] and [`WeightCutoff`] are per-term filters with no layer pass, so they take the default no-op body.
/// [`ApproxTopN`] all-reduces its octave histogram every layer, regardless of what the partition rows do.
/// [`CollapseSample`] all-reduces the global length every layer, and the per-partition norms on a layer that collapses.
/// [`And`] runs both sides in order, like [`And::finalize_layer`](TruncationPolicy::finalize_layer); [`Or`] runs neither, because its unpartitioned `finalize_layer` is the trait's no-op default rather than either child's, and the two must agree.
///
/// ```
/// use paulistrings::{And, ApproxTopN, CoefficientThreshold};
/// use paulistrings::PartitionedTruncation;
///
/// fn propagate_partitioned<T: PartitionedTruncation<1>>(_policy: T) {}
///
/// propagate_partitioned(And(CoefficientThreshold(1e-12), ApproxTopN(1_000)));
/// ```
///
/// # Why [`TopN`](crate::truncation::TopN) is rejected
///
/// ```compile_fail
/// use paulistrings::TopN;
/// use paulistrings::PartitionedTruncation;
///
/// fn propagate_partitioned<T: PartitionedTruncation<1>>(_policy: T) {}
///
/// // error[E0277]: the trait bound `TopN: PartitionedTruncation<1>` is not
/// // satisfied — exact top-n has no collective form yet.
/// propagate_partitioned(TopN(10));
/// ```
///
/// `TopN` needs the exact `n`-th largest `|c|²` of the whole layer — a distributed *k*-th selection, not a reduction — so it is rejected at compile time rather than approximated, which would silently change what `TopN` means.
/// Use [`ApproxTopN`] when `n` is a memory budget.
pub trait PartitionedTruncation<const W: usize>: TruncationPolicy<W> {
    /// The collective layer pass: `local` is this partition's slice of the layer, `coll` its view of the group.
    ///
    /// Called on every layer on every partition, in lock-step, for a policy whose [`finalizes_layer`](TruncationPolicy::finalizes_layer) is `true` — the driver skips the call entirely on one that answers `false` (see the trait docs).
    /// The default is no layer pass at all, which is correct exactly for the policies that have none, so it asserts that this is one of them rather than silently dropping a `finalize_layer` a caller was relying on.
    ///
    /// # Panics
    ///
    /// If the policy reports [`finalizes_layer()`](TruncationPolicy::finalizes_layer) and has not overridden this method.
    fn finalize_layer_partitioned(&self, _local: &mut PauliSum<W>, _coll: &dyn Collectives) {
        assert!(
            !self.finalizes_layer(),
            "{} has a layer finalization but no partitioned one. A layer pass \
             in partitioned mode has to be collective — every partition sees \
             only a disjoint slice of the layer — so the default no-op body \
             cannot stand in for it. Override `finalize_layer_partitioned` \
             (see `ApproxTopN`, which all-reduces its octave histogram), or \
             use `ApproxTopN` itself.",
            std::any::type_name_of_val(self),
        );
    }
}

/// Per-term only — the default no-op layer pass is the whole story.
impl<const W: usize> PartitionedTruncation<W> for CoefficientThreshold {}

/// Per-term only — the default no-op layer pass is the whole story.
impl<const W: usize> PartitionedTruncation<W> for WeightCutoff {}

impl<const W: usize, A, B> PartitionedTruncation<W> for And<A, B>
where
    A: PartitionedTruncation<W>,
    B: PartitionedTruncation<W>,
{
    /// Both sides, first then second — the order [`And::finalize_layer`](TruncationPolicy::finalize_layer) uses, and the same order on every partition, so the collectives stay in lock-step.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, coll: &dyn Collectives) {
        self.0.finalize_layer_partitioned(local, coll);
        self.1.finalize_layer_partitioned(local, coll);
    }
}

impl<const W: usize, A, B> PartitionedTruncation<W> for Or<A, B>
where
    A: PartitionedTruncation<W>,
    B: PartitionedTruncation<W>,
{
    /// Neither side — `Or`'s unpartitioned [`finalize_layer`](TruncationPolicy::finalize_layer) is the trait's no-op default rather than either child's, and the two paths have to agree.
    ///
    /// The children are still bounded by this trait: a composition is partitioned-safe only if its parts are, which keeps the bound meaningful if `Or` ever grows a layer pass.
    fn finalize_layer_partitioned(&self, _local: &mut PauliSum<W>, _coll: &dyn Collectives) {}
}

impl<const W: usize> PartitionedTruncation<W> for ApproxTopN {
    /// One `allreduce_sum_u64` of `[len, hist…]`, then the single-partition edge walk and retain.
    ///
    /// Octave populations are additive across a disjoint partition of the terms, and so is the term count, so the reduced buffer is exactly the histogram and length the single-partition path would have computed.
    /// `octave_edge` is a pure function of those two, so every partition reaches the same edge decision and retains its own terms against it, with no second round of communication.
    ///
    /// The length rides in slot 0 of the same buffer rather than in its own reduction.
    /// Neither early exit of the single-partition path is taken here, because neither test can be answered before the reduction — a partition that returned early would desynchronize the group.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, coll: &dyn Collectives) {
        let hist = octave_histogram(local);

        // [len, bin 0, bin 1, …]. `u32` counters widen to `u64` for the reduction: `P` partitions of up to `u32::MAX` terms each can overflow a `u32` bin, and the transport's reduction is `u64`.
        let mut packed = [0u64; 1 + APPROX_BINS];
        packed[0] = local.len() as u64;
        for (slot, &count) in packed[1..].iter_mut().zip(hist.iter()) {
            *slot = u64::from(count);
        }
        coll.allreduce_sum_u64(&mut packed);

        let total = packed[0] as usize;
        let edge = octave_edge(&packed[1..], total, self.0);
        retain_at_or_above(local, edge);
    }
}

impl<const W: usize> PartitionedTruncation<W> for CollapseSample {
    /// Two reductions, then a pick every partition makes identically: `[len, pass]` as integers, where only rank 0 contributes the pass index, and the per-partition `Σ|c|²` as a vector in which each partition fills its own slot, so every partition holds the same exact weights.
    /// The shared uniform picks a partition by weight and the offset left over picks within it, exactly as [`finalize_layer`](TruncationPolicy::finalize_layer) picks a bucket and then a term.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, coll: &dyn Collectives) {
        let rank = coll.rank() as usize;
        let call = if rank == 0 { self.next_call() } else { 0 };
        let mut head = [local.len() as u64, call];
        coll.allreduce_sum_u64(&mut head);
        let [len, call] = head;
        if len as usize <= self.cache {
            return;
        }

        let norms = bucket_norms(local);
        let mut weights = vec![0.0f64; coll.size() as usize];
        weights[rank] = norms.iter().sum();
        coll.allreduce_sum_f64(&mut weights);
        let total: f64 = weights.iter().sum();
        let target = self.target(call, total, len as usize);
        let (chosen, rest) = pick_slot(weights.iter().copied(), target)
            .expect("CollapseSample: a positive total has a positive partition");
        if rank == chosen {
            collapse_to_target(local, &norms, rest);
        } else {
            local.clear();
        }
        if rank == 0 {
            let n = self.count_collapse();
            log::debug!(
                target: LOG_TARGET,
                "collapse_sample: {len} terms over {} partitions, sum |c|^2 = {total:.6e}, \
                 collapsed to one on partition {chosen} (collapse {n})",
                coll.size(),
            );
        }
    }
}

impl<const W: usize> PartitionedTruncation<W> for BuiltinTruncation {
    /// Arm for arm the host [`finalize_layer`](TruncationPolicy::finalize_layer), through each builtin's own collective form.
    ///
    /// # Panics
    ///
    /// On a `TopN` that is reached, which has no collective form.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, coll: &dyn Collectives) {
        match self {
            Self::ApproxTopN(n) => ApproxTopN(*n).finalize_layer_partitioned(local, coll),
            Self::CollapseSample(s) => {
                <CollapseSample as PartitionedTruncation<W>>::finalize_layer_partitioned(
                    s, local, coll,
                )
            }
            Self::And(a, b) => {
                <Self as PartitionedTruncation<W>>::finalize_layer_partitioned(a, local, coll);
                <Self as PartitionedTruncation<W>>::finalize_layer_partitioned(b, local, coll);
            }
            Self::TopN(_) => panic!(
                "BuiltinTruncation::TopN has no partitioned layer pass: exact top-n is a distributed k-th selection; use ApproxTopN"
            ),
            Self::Keep | Self::Coeff(_) | Self::Weight(_) | Self::Or(_, _) => {}
        }
    }
}

#[cfg(test)]
mod tests;
