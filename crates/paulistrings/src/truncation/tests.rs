use super::*;
use crate::engine::partitioned::transport::InProcessTransport;
use crate::pauli_sum::PartitionRows;
use crate::test_support::{assert_same_terms, rand_sum_real, tie_heavy_sum};
use crate::truncation::builtin::{
    And, ApproxTopN, CoefficientThreshold, CollapseSample, Or, TopN, WeightCutoff,
};
use num_complex::Complex64;

/// Seed for every `PartitionRows` in this module's tests.
const PSEED: u64 = 0x5EED_C0FFEE;

/// Split `sum` into `1 << pbits` partitions, run the policy's collective layer pass on each in its own thread, and gather the result.
fn partitioned_finalize<const W: usize, T>(policy: &T, sum: &PauliSum<W>, pbits: u8) -> PauliSum<W>
where
    T: TruncationPolicy<W>,
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

/// Partitioned finalization is exactly the single-partition one.
fn assert_matches_single_partition<const W: usize, T>(
    policy: &T,
    input: &PauliSum<W>,
    pbits: u8,
    what: &str,
) where
    T: TruncationPolicy<W>,
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

/// `W = 1`, random real coefficients: the retained set is the same at `P = 1, 2, 4` for a spread of `n`, including `n` well inside the sum, `n = 0`, and `n` past the end.
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

/// `W = 2` with magnitudes 1, ½, ¼, ⅛, one group per octave, so several `n` cut inside a tie band.
#[test]
fn approx_top_n_partitioned_matches_single_partition_w2_tie_heavy() {
    let input = tie_heavy_sum::<2>(2000, 100, 0x7135);
    // Cumulative octave populations are ~500, ~1000, ~1500, ~2000; every `n` lands strictly between two of them.
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

/// A sum confined to one octave of `|c|²` (magnitudes 1, 1⅛, 1¼, 1⅜) is wiped to empty on every partition.
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

/// A four-way split of a five-term sum leaves partitions (nearly) empty, which must still enter the collective.
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

/// `And` forwards to both sides in order, as it does unpartitioned.
#[test]
fn and_composes_with_a_collective_finalization() {
    let input = rand_sum_real::<1>(1500, 32, 0xC0DE);
    let policy = And(CoefficientThreshold(1e-3), ApproxTopN(400));
    for pbits in [1u8, 2] {
        assert_matches_single_partition(&policy, &input, pbits, &format!("and P=2^{pbits}"));
    }
}

/// A per-term policy's layer pass is a no-op on every partition: nothing is dropped, nothing is communicated.
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

/// `Or` forwards its partitioned pass to neither side, as it does unpartitioned.
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

/// At `P = 1` the collective pick is the unpartitioned one, seed for seed.
#[test]
fn collapse_sample_at_one_partition_is_the_unpartitioned_pick() {
    let input = crate::test_support::rand_sum::<1>(3000, 16, 0xC0C0);
    for seed in 0..20u64 {
        let mut want = input.clone();
        CollapseSample::new(50, seed).finalize_layer(&mut want);
        let got = partitioned_finalize(&CollapseSample::new(50, seed), &input, 0);
        assert_same_terms(&got, &want, &format!("seed {seed}"));
    }
}

/// Below the cache every partition keeps its share, even where a partition alone is far below it.
#[test]
fn collapse_sample_partitioned_is_a_no_op_up_to_the_global_cache() {
    let input = rand_sum_real::<1>(600, 32, 0xF00D);
    for pbits in [1u8, 2] {
        let policy = CollapseSample::new(input.len(), 3);
        let got = partitioned_finalize(&policy, &input, pbits);
        assert_same_terms(&got, &input, &format!("P=2^{pbits}"));
        assert_eq!(policy.collapses(), 0);
    }
}

/// Above the cache exactly one string survives across the group, counted once, and the picks over 3000 seeds match `|c|² / Σ|c|²`.
#[test]
fn collapse_sample_partitioned_draws_one_string_by_weight() {
    use crate::test_support::{
        assert_frequencies, collapsed_index, four_term_keys, weighted_four_term_sum,
        FOUR_TERM_WEIGHTS,
    };
    let keys = four_term_keys::<2>();
    let input = weighted_four_term_sum::<2>(8);
    for pbits in [1u8, 2] {
        let rows = PartitionRows::<2>::from_seed(8, pbits, PSEED);
        let owners: std::collections::HashSet<u32> =
            keys.iter().map(|p| rows.partition_of_pauli(p)).collect();
        assert!(
            owners.len() > 1,
            "P=2^{pbits}: the fixture must span partitions"
        );

        let mut counts = [0usize; 4];
        for seed in 0..3000u64 {
            let policy = CollapseSample::new(3, seed);
            let got = partitioned_finalize(&policy, &input, pbits);
            counts[collapsed_index(&got, &keys)] += 1;
            assert_eq!(policy.collapses(), 1, "one collapse, counted once");
        }
        assert_frequencies(&counts, &FOUR_TERM_WEIGHTS, &format!("P=2^{pbits}"));
    }
}

/// The default body panics above one partition for a policy with a real `finalize_layer` and no collective one.
#[test]
#[should_panic(expected = "ApproxTopN")]
fn the_default_body_rejects_a_policy_that_finalizes_layers() {
    let group = InProcessTransport::group(2);
    let mut sum = rand_sum_real::<1>(10, 32, 0x1234);
    HalfDone.finalize_layer_partitioned(&mut sum, &group[0]);
}

/// A policy with a layer pass and no collective form.
struct HalfDone;

impl<const W: usize> TruncationPolicy<W> for HalfDone {
    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        sum.clear();
    }
}

/// At one partition the default body is the plain layer pass.
#[test]
fn the_default_body_at_one_partition_is_the_plain_pass() {
    let group = InProcessTransport::group(1);
    let mut sum = rand_sum_real::<1>(10, 32, 0x1234);
    HalfDone.finalize_layer_partitioned(&mut sum, &group[0]);
    assert!(sum.is_empty());
}

/// Exact `TopN` runs at one partition and panics above it, naming `ApproxTopN`.
#[test]
fn exact_top_n_runs_at_one_partition_only() {
    let input = rand_sum_real::<1>(600, 32, 0x70B0);
    let mut want = input.clone();
    TopN(50).finalize_layer(&mut want);
    let got = partitioned_finalize(&TopN(50), &input, 0);
    assert_same_terms(&got, &want, "TopN at P = 1");

    let group = InProcessTransport::group(2);
    let mut local = input.clone();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        TopN(50).finalize_layer_partitioned(&mut local, &group[0])
    }))
    .expect_err("TopN above one partition must panic");
    let message = panic.downcast_ref::<String>().expect("a formatted message");
    assert!(
        message.contains("not yet supported") && message.contains("ApproxTopN"),
        "{message}"
    );
}

/// `supports_partitioned` is `false` exactly for a layer pass with no collective form, looking through `And` but not into `Or`.
#[test]
fn supports_partitioned_names_the_policies_a_partitioned_run_accepts() {
    fn supports<T: TruncationPolicy<1>>(policy: T) -> bool {
        policy.supports_partitioned()
    }
    assert!(supports(CoefficientThreshold(1e-3)));
    assert!(supports(WeightCutoff(3)));
    assert!(supports(ApproxTopN(10)));
    assert!(supports(CollapseSample::new(10, 1)));
    assert!(supports(And(CoefficientThreshold(1e-3), ApproxTopN(10))));
    assert!(supports(Or(TopN(10), CoefficientThreshold(1e-3))));
    assert!(!supports(TopN(10)));
    assert!(!supports(And(CoefficientThreshold(1e-3), TopN(10))));
    assert!(!supports(HalfDone));
}

/// The run-time check every partitioned driver makes before its first layer.
#[test]
fn assert_supports_partitions_rejects_exact_top_n_above_one_partition() {
    assert_supports_partitions::<1, _>(&TopN(10), 1);
    assert_supports_partitions::<1, _>(&ApproxTopN(10), 4);
    let panic = std::panic::catch_unwind(|| assert_supports_partitions::<1, _>(&TopN(10), 2))
        .expect_err("TopN on two partitions must panic");
    let message = panic.downcast_ref::<String>().expect("a formatted message");
    assert!(
        message.contains("not yet supported") && message.contains("ApproxTopN"),
        "{message}"
    );
}
