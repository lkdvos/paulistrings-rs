//! [`PartitionedTruncation`], truncation whose layer pass is collective, and its implementations for the builtin policies (ARCHITECTURE.md §Partitioning).

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
/// A policy whose `finalizes_layer()` is `true` must override [`finalize_layer_partitioned`](Self::finalize_layer_partitioned), which runs on every layer on every partition in lock-step.
/// An implementation must not skip a collective call on a partition with nothing to do, and must decide only from all-reduced values, so every partition applies the same predicate to its own terms.
///
/// Exact [`TopN`](crate::TopN) is rejected at compile time, a distributed `n`-th selection having no collective form; use [`ApproxTopN`] when `n` is a memory budget.
///
/// ```compile_fail
/// use paulistrings::TopN;
/// use paulistrings::PartitionedTruncation;
///
/// fn propagate_partitioned<T: PartitionedTruncation<1>>(_policy: T) {}
///
/// propagate_partitioned(TopN(10));
/// ```
pub trait PartitionedTruncation<const W: usize>: TruncationPolicy<W> {
    /// The collective layer pass over this partition's slice of the layer; the default is no pass at all.
    ///
    /// # Panics
    ///
    /// If the policy reports [`finalizes_layer()`](TruncationPolicy::finalizes_layer) and has not overridden this method.
    fn finalize_layer_partitioned(&self, _local: &mut PauliSum<W>, _collectives: &dyn Collectives) {
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

impl<const W: usize> PartitionedTruncation<W> for CoefficientThreshold {}

impl<const W: usize> PartitionedTruncation<W> for WeightCutoff {}

impl<const W: usize, A, B> PartitionedTruncation<W> for And<A, B>
where
    A: PartitionedTruncation<W>,
    B: PartitionedTruncation<W>,
{
    /// Both sides, in [`And::finalize_layer`](TruncationPolicy::finalize_layer)'s order.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, collectives: &dyn Collectives) {
        self.0.finalize_layer_partitioned(local, collectives);
        self.1.finalize_layer_partitioned(local, collectives);
    }
}

impl<const W: usize, A, B> PartitionedTruncation<W> for Or<A, B>
where
    A: PartitionedTruncation<W>,
    B: PartitionedTruncation<W>,
{
    /// Neither side, since `Or`'s unpartitioned [`finalize_layer`](TruncationPolicy::finalize_layer) is the no-op default and the two must agree.
    fn finalize_layer_partitioned(&self, _local: &mut PauliSum<W>, _collectives: &dyn Collectives) {
    }
}

impl<const W: usize> PartitionedTruncation<W> for ApproxTopN {
    /// One `allreduce_sum_u64` of `[len, histogram…]`, then the single-partition edge walk; no early exit, since that would desynchronize the group.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, collectives: &dyn Collectives) {
        let histogram = octave_histogram(local);

        // `u64`, since `P` partitions' `u32` bins can overflow a `u32`.
        let mut packed = [0u64; 1 + APPROX_BINS];
        packed[0] = local.len() as u64;
        for (slot, &count) in packed[1..].iter_mut().zip(histogram.iter()) {
            *slot = u64::from(count);
        }
        collectives.allreduce_sum_u64(&mut packed);

        let total = packed[0] as usize;
        let edge = octave_edge(&packed[1..], total, self.0);
        retain_at_or_above(local, edge);
    }
}

impl<const W: usize> PartitionedTruncation<W> for CollapseSample {
    /// Reduces `[len, pass]` (the pass from rank 0) and the per-partition `Σ|c|²`, then picks a partition by weight and a term within it, as [`finalize_layer`](TruncationPolicy::finalize_layer) picks a bucket and then a term.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, collectives: &dyn Collectives) {
        let rank = collectives.rank() as usize;
        let call = if rank == 0 { self.next_call() } else { 0 };
        let mut head = [local.len() as u64, call];
        collectives.allreduce_sum_u64(&mut head);
        let [len, call] = head;
        if len as usize <= self.cache {
            return;
        }

        let norms = bucket_norms(local);
        let mut weights = vec![0.0f64; collectives.size() as usize];
        weights[rank] = norms.iter().sum();
        collectives.allreduce_sum_f64(&mut weights);
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
            let collapses = self.count_collapse();
            log::debug!(
                target: LOG_TARGET,
                "collapse_sample: {len} terms over {} partitions, sum |c|^2 = {total:.6e}, \
                 collapsed to one on partition {chosen} (collapse {collapses})",
                collectives.size(),
            );
        }
    }
}

impl<const W: usize> PartitionedTruncation<W> for BuiltinTruncation {
    /// Arm for arm the host [`finalize_layer`](TruncationPolicy::finalize_layer); panics on a reached `TopN`.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, collectives: &dyn Collectives) {
        match self {
            Self::ApproxTopN(n) => ApproxTopN(*n).finalize_layer_partitioned(local, collectives),
            Self::CollapseSample(s) => {
                <CollapseSample as PartitionedTruncation<W>>::finalize_layer_partitioned(
                    s, local, collectives,
                )
            }
            Self::And(a, b) => {
                <Self as PartitionedTruncation<W>>::finalize_layer_partitioned(a, local, collectives);
                <Self as PartitionedTruncation<W>>::finalize_layer_partitioned(b, local, collectives);
            }
            Self::TopN(_) => panic!(
                "BuiltinTruncation::TopN has no partitioned layer pass: exact top-n is a distributed k-th selection; use ApproxTopN"
            ),
            Self::Keep | Self::Coefficient(_) | Self::Weight(_) | Self::Or(_, _) => {}
        }
    }
}

#[cfg(test)]
mod tests;
