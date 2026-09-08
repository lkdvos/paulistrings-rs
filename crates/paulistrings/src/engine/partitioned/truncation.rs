//! [`PartitionedTruncation`] — truncation whose layer pass is collective.
//!
//! In partitioned mode each partition holds a *disjoint subset* of the terms
//! (ARCHITECTURE.md §Partitioning), so a [`TruncationPolicy`] splits cleanly
//! in two:
//!
//! - [`keep_term`](TruncationPolicy::keep_term) is per term and needs nothing.
//!   It runs inside the merge, on the complete summed coefficient of a key,
//!   and a key lives on exactly one partition.
//! - [`finalize_layer`](TruncationPolicy::finalize_layer) may be *global*, and
//!   a partition cannot see the global layer. [`ApproxTopN`] chooses an octave
//!   edge from the histogram of the whole layer — which is exactly the sum of
//!   the per-partition histograms, so one `allreduce` makes every partition
//!   choose the *same* edge and apply it to its own terms. The union of the
//!   retained sets is then bit for bit the single-partition answer.
//!
//! Exact [`TopN`](crate::truncation::TopN) has no such reduction: the `n`-th
//! largest magnitude is a distributed *k*-th selection, not a sum. It is
//! rejected at compile time rather than approximated — see the trait docs.

use crate::pauli_sum::PauliSum;
use crate::truncation::builtin::{octave_edge, octave_histogram, retain_at_or_above, APPROX_BINS};
use crate::truncation::{
    And, ApproxTopN, CoefficientThreshold, Or, TruncationPolicy, WeightCutoff,
};

use super::transport::Collectives;

/// A truncation policy usable in partitioned propagation: its layer
/// finalization is collective.
///
/// # The contract
///
/// [`finalize_layer_partitioned`](Self::finalize_layer_partitioned) is called
/// on **every** layer, on **every** partition, in lock-step — regardless of
/// what [`finalizes_layer`](TruncationPolicy::finalizes_layer) says. That is
/// not an accident of the caller: a collective is only well-defined if every
/// partition issues the same collectives in the same order, so a policy is not
/// free to skip a layer on the partitions where it happens to have nothing to
/// do. The hint stays an optimization for the single-partition paths.
///
/// An implementation must therefore
///
/// 1. call the same collectives, in the same order, on every partition and
///    every layer (in particular, no early return before a collective on a
///    locally empty or locally short partition), and
/// 2. derive its decision **only** from all-reduced values, so that every
///    partition applies the identical predicate to its own terms.
///
/// Under those two rules the retained set is exactly the set a single
/// partition holding the whole sum would have retained.
///
/// # Which policies implement it
///
/// [`CoefficientThreshold`] and [`WeightCutoff`] are per-term filters with no
/// layer pass, so they take the default no-op body. [`ApproxTopN`] all-reduces
/// its octave histogram. [`And`] runs both sides in order, like
/// [`And::finalize_layer`](TruncationPolicy::finalize_layer); [`Or`] runs
/// neither, because its unpartitioned `finalize_layer` is the trait's no-op
/// default rather than either child's, and the two must agree.
///
/// ```
/// use paulistrings::truncation::{And, ApproxTopN, CoefficientThreshold};
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
/// use paulistrings::truncation::TopN;
/// use paulistrings::PartitionedTruncation;
///
/// fn propagate_partitioned<T: PartitionedTruncation<1>>(_policy: T) {}
///
/// // error[E0277]: the trait bound `TopN: PartitionedTruncation<1>` is not
/// // satisfied — exact top-n has no collective form yet.
/// propagate_partitioned(TopN(10));
/// ```
///
/// `TopN` needs the exact `n`-th largest `|c|²` of the whole layer. That is a
/// distributed *k*-th selection — an iterated search, several rounds of
/// communication, not one reduction — and an approximation would silently
/// change what `TopN` *means*, which is the one thing it exists to guarantee.
/// So there is no impl, and a partitioned run with `TopN` fails to compile
/// instead of quietly truncating per partition (which would keep `P·n` terms
/// and a different set on every thread count). Use [`ApproxTopN`] when `n` is
/// a memory budget; the distributed selection is a phase-6 follow-up.
pub trait PartitionedTruncation<const W: usize>: TruncationPolicy<W> {
    /// The collective layer pass: `local` is this partition's slice of the
    /// layer, `coll` its view of the group.
    ///
    /// Called on every layer on every partition, in lock-step. The default is
    /// no layer pass at all, which is correct exactly for the policies that
    /// have none — so it asserts that this is one of them rather than
    /// silently dropping a `finalize_layer` a caller was relying on.
    ///
    /// # Panics
    ///
    /// If the policy reports
    /// [`finalizes_layer()`](TruncationPolicy::finalizes_layer) and has not
    /// overridden this method.
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
    /// Both sides, first then second — the order
    /// [`And::finalize_layer`](TruncationPolicy::finalize_layer) uses, and the
    /// same order on every partition, so the collectives stay in lock-step.
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
    /// Neither side — `Or` combines only `keep_term`, and its
    /// [`finalize_layer`](TruncationPolicy::finalize_layer) is the trait's
    /// no-op default rather than either child's (the layer semantics of
    /// "either policy's finalize pass" are not well-defined). Forwarding here
    /// would make `Or(_, ApproxTopN(n))` truncate under partitioning and not
    /// truncate without it; the two paths have to agree.
    ///
    /// The children are still bounded by this trait: a composition is
    /// partitioned-safe only if its parts are, and that keeps the bound
    /// meaningful if `Or` ever grows a layer pass.
    fn finalize_layer_partitioned(&self, _local: &mut PauliSum<W>, _coll: &dyn Collectives) {}
}

impl<const W: usize> PartitionedTruncation<W> for ApproxTopN {
    /// One `allreduce_sum_u64` of `[len, hist…]`, then the single-partition
    /// edge walk and retain.
    ///
    /// Octave populations are additive across a disjoint partition of the
    /// terms, and so is the term count, so the reduced buffer is exactly the
    /// histogram and length the single-partition path would have computed.
    /// `octave_edge` is a pure function of those two, so every partition
    /// reaches the same edge decision and retains its own terms against it —
    /// no second round of communication, and the union is the
    /// single-partition set term for term.
    ///
    /// The length rides in slot 0 of the same buffer rather than in its own
    /// reduction: one collective per layer, and 8 bytes on a 16 KB message.
    /// Neither early exit of the single-partition path is taken here, because
    /// neither test can be answered before the reduction — a partition that
    /// returned early would desynchronize the group.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, coll: &dyn Collectives) {
        let hist = octave_histogram(local);

        // [len, bin 0, bin 1, …]. `u32` counters widen to `u64` for the
        // reduction: `P` partitions of up to `u32::MAX` terms each can
        // overflow a `u32` bin, and the transport's reduction is `u64`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket::PartitionRows;
    use crate::engine::partitioned::transport::InProcessTransport;
    use crate::test_support::{assert_same_terms, rand_sum_real, tie_heavy_sum};
    use num_complex::Complex64;

    /// Seed for every `PartitionRows` in this module's tests.
    const PSEED: u64 = 0x5EED_C0FFEE;

    /// Split `sum` into `1 << pbits` partitions, run the policy's collective
    /// layer pass on each in its own thread, and gather the result.
    ///
    /// One `std::thread::scope` per call with one thread per partition, which
    /// is what the in-process transport's blocking collectives need: every
    /// rank has to be able to make progress independently.
    fn partitioned_finalize<const W: usize, T>(
        policy: &T,
        sum: &PauliSum<W>,
        pbits: u8,
    ) -> PauliSum<W>
    where
        T: PartitionedTruncation<W>,
    {
        let rows = PartitionRows::<W>::from_seed(sum.num_qubits(), pbits, PSEED);
        let p = rows.num_partitions() as u32;
        let parts: Vec<PauliSum<W>> = (0..p).map(|r| sum.filter_partition(&rows, r)).collect();
        assert_eq!(
            parts.iter().map(PauliSum::len).sum::<usize>(),
            sum.len(),
            "the split must be a partition of the terms",
        );

        let group = InProcessTransport::group(p);
        let gathered: Vec<PauliSum<W>> = std::thread::scope(|scope| {
            let handles: Vec<_> = parts
                .into_iter()
                .zip(group)
                .map(|(mut local, transport)| {
                    scope.spawn(move || {
                        policy.finalize_layer_partitioned(&mut local, &transport);
                        (transport.rank(), local)
                    })
                })
                .collect();
            let mut slots: Vec<Option<PauliSum<W>>> = (0..p).map(|_| None).collect();
            for handle in handles {
                let (rank, local) = handle.join().expect("partition thread panicked");
                slots[rank as usize] = Some(local);
            }
            slots.into_iter().map(Option::unwrap).collect()
        });

        for local in &gathered {
            local.assert_invariants();
        }
        let merged = PauliSum::<W>::merge_partitions(gathered);
        merged.assert_invariants();
        merged
    }

    /// The whole point: partitioned finalization is *exactly* the
    /// single-partition one, for every `P`.
    fn assert_matches_single_partition<const W: usize, T>(
        policy: &T,
        input: &PauliSum<W>,
        pbits: u8,
        what: &str,
    ) where
        T: PartitionedTruncation<W>,
    {
        let mut want = input.clone();
        policy.finalize_layer(&mut want);

        let got = partitioned_finalize(policy, input, pbits);
        assert_eq!(
            got.len(),
            want.len(),
            "{what}: partitioned kept {} terms, single-partition {}",
            got.len(),
            want.len(),
        );
        assert_same_terms(&got, &want, what);
    }

    /// `W = 1`, random real coefficients: the retained set is the same at
    /// `P = 1, 2, 4` for a spread of `n`, including `n` well inside the sum,
    /// `n = 0`, and `n` past the end.
    #[test]
    fn approx_top_n_partitioned_matches_single_partition_w1() {
        let input = rand_sum_real::<1>(2000, 32, 0xA9C7);
        for pbits in [1u8, 2] {
            for n in [0usize, 1, 37, 300, 999, 1500, input.len(), input.len() + 10] {
                assert_matches_single_partition(
                    &ApproxTopN(n),
                    &input,
                    pbits,
                    &format!("w1 P=2^{pbits} n={n}"),
                );
            }
        }
    }

    /// `W = 2`, and a tie-heavy fixture: magnitudes 1, ½, ¼, ⅛ put one
    /// magnitude group per octave, so several of these `n` cut *inside* a tie
    /// band — the case where the edge choice actually matters and a
    /// per-partition decision would differ from a global one.
    #[test]
    fn approx_top_n_partitioned_matches_single_partition_w2_tie_heavy() {
        let input = tie_heavy_sum::<2>(2000, 100, 0x7135);
        // The four magnitude groups are ~500 terms each, so the cumulative
        // octave populations are ~500, ~1000, ~1500, ~2000: every one of these
        // `n` lands strictly between two of them.
        for pbits in [1u8, 2] {
            for n in [0usize, 3, 250, 700, 1200, 1900, 2500] {
                assert_matches_single_partition(
                    &ApproxTopN(n),
                    &input,
                    pbits,
                    &format!("w2 tie-heavy P=2^{pbits} n={n}"),
                );
            }
        }
    }

    /// A sum confined to one octave of `|c|²` is wiped to empty — the
    /// degenerate case documented on `ApproxTopN` — and it has to be wiped on
    /// *every* partition, from the global histogram, not decided locally.
    /// Magnitudes 1, 1⅛, 1¼, 1⅜ square into `[1, 2)`.
    #[test]
    fn approx_top_n_partitioned_wipes_a_single_octave_sum() {
        let mags = [1.0f64, 1.125, 1.25, 1.375];
        let input = PauliSum::<1>::from_sorted_columns(
            (0u64..mags.len() as u64).map(|i| [i]).collect(),
            vec![[0u64]; mags.len()],
            mags.iter().map(|&m| Complex64::new(m, 0.0)).collect(),
            32,
        );
        for pbits in [1u8, 2] {
            let got = partitioned_finalize(&ApproxTopN(3), &input, pbits);
            assert!(got.is_empty(), "P=2^{pbits}: one octave cannot be split");
            // …and `n >= len` is still the no-op it is single-partition.
            assert_matches_single_partition(
                &ApproxTopN(4),
                &input,
                pbits,
                &format!("single octave P=2^{pbits} n=4"),
            );
        }
    }

    /// A four-way split of a five-term sum leaves partitions empty (or very
    /// nearly). They must still enter the collective — an early return on
    /// "nothing here" would hang or desynchronize the group — and the merged
    /// result must still be the single-partition one.
    #[test]
    fn an_empty_partition_still_participates() {
        let mags = [1.0f64, 2.0, 4.0, 8.0, 16.0];
        let input = PauliSum::<1>::from_sorted_columns(
            (0u64..mags.len() as u64).map(|i| [i]).collect(),
            vec![[0u64]; mags.len()],
            mags.iter().map(|&m| Complex64::new(m, 0.0)).collect(),
            4,
        );
        let rows = PartitionRows::<1>::from_seed(input.num_qubits(), 2, PSEED);
        assert!(
            (0..4u32).any(|r| input.filter_partition(&rows, r).is_empty()),
            "the fixture must actually leave a partition empty",
        );
        for n in [1usize, 2, 3, 4] {
            assert_matches_single_partition(&ApproxTopN(n), &input, 2, &format!("tiny n={n}"));
        }
    }

    /// `And` forwards to both sides in order, so a threshold paired with
    /// `ApproxTopN` finalizes exactly as the same `And` does unpartitioned.
    #[test]
    fn and_composes_with_a_collective_finalization() {
        let input = rand_sum_real::<1>(1500, 32, 0xC0DE);
        let policy = And(CoefficientThreshold(1e-3), ApproxTopN(400));
        for pbits in [1u8, 2] {
            assert_matches_single_partition(&policy, &input, pbits, &format!("and P=2^{pbits}"));
        }
    }

    /// A per-term policy's layer pass is a no-op on every partition: nothing
    /// is dropped, nothing is communicated.
    #[test]
    fn a_per_term_policy_finalizes_to_a_no_op() {
        let input = rand_sum_real::<1>(600, 32, 0xF00D);
        for pbits in [1u8, 2] {
            let got = partitioned_finalize(&CoefficientThreshold(1e9), &input, pbits);
            assert_same_terms(&got, &input, &format!("threshold P=2^{pbits}"));
            let got = partitioned_finalize(&WeightCutoff(0), &input, pbits);
            assert_same_terms(&got, &input, &format!("weight P=2^{pbits}"));
            let got = partitioned_finalize(
                &And(CoefficientThreshold(1e9), WeightCutoff(0)),
                &input,
                pbits,
            );
            assert_same_terms(&got, &input, &format!("and P=2^{pbits}"));
        }
    }

    /// `Or` does not forward `finalize_layer` to either side, and its
    /// partitioned pass must not either — otherwise `Or(_, ApproxTopN)` would
    /// truncate under partitioning and not without it.
    #[test]
    fn or_forwards_no_layer_pass_either_way() {
        let input = rand_sum_real::<1>(600, 32, 0xB0B0);
        let policy = Or(CoefficientThreshold(1e-3), ApproxTopN(10));

        let mut unpartitioned = input.clone();
        policy.finalize_layer(&mut unpartitioned);
        assert_eq!(unpartitioned.len(), input.len(), "Or has no layer pass");

        for pbits in [1u8, 2] {
            assert_matches_single_partition(&policy, &input, pbits, &format!("or P=2^{pbits}"));
        }
    }

    /// The default body is a no-op *and* a tripwire: a policy with a real
    /// `finalize_layer` that forgets to write a collective one is caught at
    /// the first layer rather than silently losing its truncation.
    #[test]
    #[should_panic(expected = "ApproxTopN")]
    fn the_default_body_rejects_a_policy_that_finalizes_layers() {
        struct HalfDone;
        impl<const W: usize> TruncationPolicy<W> for HalfDone {
            fn finalize_layer(&self, sum: &mut PauliSum<W>) {
                sum.clear();
            }
        }
        impl<const W: usize> PartitionedTruncation<W> for HalfDone {}

        let group = InProcessTransport::group(1);
        let mut sum = rand_sum_real::<1>(10, 32, 0x1234);
        HalfDone.finalize_layer_partitioned(&mut sum, &group[0]);
    }
}
