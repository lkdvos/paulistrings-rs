use super::*;
use crate::engine::partitioned::in_process::InProcessTransport;
use crate::pauli_sum::PartitionRows;
use crate::test_support::{
    assert_proportional_boundary_cut, assert_same_terms, rand_sum_real, skewed_cut_split,
    tie_heavy_sum,
};
use crate::truncation::builtin::{
    And, ApproxTopN, CoefficientThreshold, Or, PerPartition, TopN, WeightCutoff,
};
use crate::truncation::CollapseSample;
use num_complex::Complex64;

/// Seed for every `PartitionRows` in this module's tests.
const PSEED: u64 = 0x5EED_C0FFEE;

/// Split `sum` into `1 << pbits` partitions, run the policy's collective layer pass on each in its own thread, and gather the result.
fn partitioned_finalize<const W: usize, T>(policy: &T, sum: &PauliSum<W>, pbits: u8) -> PauliSum<W>
where
    T: TruncationPolicy<W>,
{
    run_partitioned(sum, pbits, |local, collectives| {
        policy.finalize_layer_partitioned(local, collectives)
    })
}

/// Split `sum` into `1 << pbits` partitions, run `pass` on each in its own thread over one in-process group, and gather the result.
pub(super) fn run_partitioned<const W: usize>(
    sum: &PauliSum<W>,
    pbits: u8,
    pass: impl Fn(&mut PauliSum<W>, &dyn Collectives) + Sync,
) -> PauliSum<W> {
    let merged = PauliSum::<W>::merge_partitions(run_on_ranks(split(sum, pbits), pass));
    merged.assert_invariants();
    merged
}

/// Run `pass` on each of `parts` in its own thread over one in-process group, returning the partitions in rank order.
fn run_on_ranks<const W: usize>(
    parts: Vec<PauliSum<W>>,
    pass: impl Fn(&mut PauliSum<W>, &dyn Collectives) + Sync,
) -> Vec<PauliSum<W>> {
    let p = parts.len() as u32;
    let group = InProcessTransport::group(p);
    let pass = &pass;
    let gathered: Vec<PauliSum<W>> = std::thread::scope(|scope| {
        let handles: Vec<_> = parts
            .into_iter()
            .zip(group)
            .map(|(mut local, transport)| {
                scope.spawn(move || {
                    pass(&mut local, &transport);
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
    gathered
}

/// `sum` split into `1 << pbits` partitions, in rank order.
fn split<const W: usize>(sum: &PauliSum<W>, pbits: u8) -> Vec<PauliSum<W>> {
    let rows = PartitionRows::<W>::from_seed(sum.num_qubits(), pbits, PSEED);
    let p = rows.num_partitions() as u32;
    let parts: Vec<PauliSum<W>> = (0..p).map(|r| sum.filter_partition(&rows, r)).collect();
    assert_eq!(
        parts.iter().map(PauliSum::len).sum::<usize>(),
        sum.len(),
        "the split must be a partition of the terms",
    );
    parts
}

/// Partitioned `ApproxTopN(n)` over [`split`] makes the cut [`assert_proportional_boundary_cut`] checks.
fn assert_partitioned_approx_top_n<const W: usize>(
    input: &PauliSum<W>,
    n: usize,
    pbits: u8,
    what: &str,
) {
    assert_approx_top_n_over(split(input, pbits), n, what);
}

/// Partitioned `ApproxTopN(n)` over `parts`, one per rank, makes the cut [`assert_proportional_boundary_cut`] checks.
fn assert_approx_top_n_over<const W: usize>(parts: Vec<PauliSum<W>>, n: usize, what: &str) {
    let got = run_on_ranks(parts.clone(), |local, collectives| {
        ApproxTopN(n).finalize_layer_partitioned(local, collectives)
    });
    assert_proportional_boundary_cut(&parts, &got, n, what);
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

/// `W = 1`, random real coefficients, `P = 2, 4`, for a spread of `n` including `n = 0` and `n` past the end.
#[test]
fn approx_top_n_partitioned_keeps_exactly_n_w1() {
    let input = rand_sum_real::<1>(2000, 32, 0xA9C7);
    for pbits in [1u8, 2] {
        for n in [0usize, 1, 37, 300, 999, 1500, input.len(), input.len() + 10] {
            assert_partitioned_approx_top_n(&input, n, pbits, &format!("w1 P=2^{pbits} n={n}"));
        }
    }
}

/// `W = 2` with magnitudes 1, ½, ¼, ⅛, one group per octave, so every `n` cuts inside a tie band.
#[test]
fn approx_top_n_partitioned_keeps_exactly_n_w2_tie_heavy() {
    let input = tie_heavy_sum::<2>(2000, 100, 0x7135);
    for pbits in [1u8, 2] {
        for n in [0usize, 3, 250, 700, 1200, 1900, 2500] {
            assert_partitioned_approx_top_n(
                &input,
                n,
                pbits,
                &format!("w2 tie-heavy P=2^{pbits} n={n}"),
            );
        }
    }
}

/// Cut rows over qubit blocks with partitions thinned unevenly: the boundary octave is split in proportion to each rank's share of it, not handed to the low ranks first.
#[test]
fn approx_top_n_splits_the_boundary_octave_across_cut_rows_in_proportion() {
    let w1 = skewed_cut_split(&rand_sum_real::<1>(4000, 32, 0xC07));
    let w2 = skewed_cut_split(&tie_heavy_sum::<2>(4000, 100, 0xC07));
    for n in [1usize, 37, 300, 999, 1500] {
        assert_approx_top_n_over(w1.clone(), n, &format!("w1 cut n={n}"));
        assert_approx_top_n_over(w2.clone(), n, &format!("w2 tie-heavy cut n={n}"));
    }
}

/// Two ranks share the quarter-bin fixture: rank 0 holds `|c|²` 1.9, 1.3, 1.0, 1.1 and rank 1 the rest.
/// At `n = 5` the boundary `[1.25, 1.5)` has one term on each rank and a quota of one, which the tie hands to rank 0; at `n = 8` the boundary `[1, 1.25)` has two terms on rank 0 and one on rank 1 for a quota of two, split ⌊4/3⌋ = 1 and ⌊2/3⌋ + 1 = 1 by the larger remainder.
#[test]
fn a_group_splits_a_quarter_bin_boundary_in_proportion() {
    use crate::test_support::sum_of_norm_sqrs;
    let parts = vec![
        sum_of_norm_sqrs(&[1.9, 1.3, 1.0, 1.1], 0),
        sum_of_norm_sqrs(&[1.8, 1.4, 1.2, 1.6, 0.9, 2.0], 16),
    ];
    let kept = |n: usize| -> Vec<Vec<f64>> {
        run_on_ranks(parts.clone(), |local, collectives| {
            ApproxTopN(n).finalize_layer_partitioned(local, collectives)
        })
        .iter()
        .map(|part| part.iter().map(|(_, _, c)| (c.norm_sqr() * 1e9).round() / 1e9).collect())
        .collect()
    };
    assert_eq!(kept(5), vec![vec![1.9, 1.3], vec![1.8, 1.6, 2.0]]);
    assert_eq!(kept(8), vec![vec![1.9, 1.3, 1.0], vec![1.8, 1.4, 1.2, 1.6, 2.0]]);
    for n in [1, 3, 7, 9] {
        assert_approx_top_n_over(parts.clone(), n, &format!("n={n}"));
    }
}

/// A sum confined to one octave of `|c|²` (magnitudes 1, 1⅛, 1¼, 1⅜) keeps `n` of it across partitions.
#[test]
fn approx_top_n_partitioned_splits_a_single_octave_sum() {
    let mags = [1.0f64, 1.125, 1.25, 1.375];
    let input = PauliSum::<1>::from_sorted_columns(
        (0u64..mags.len() as u64).map(|i| [i]).collect(),
        vec![[0u64]; mags.len()],
        mags.iter().map(|&m| Complex64::new(m, 0.0)).collect(),
        32,
    );
    for pbits in [1u8, 2] {
        for n in 0..=4 {
            assert_partitioned_approx_top_n(&input, n, pbits, &format!("P=2^{pbits} n={n}"));
        }
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
        let got = partitioned_finalize(&policy, &input, pbits);
        let want = partitioned_finalize(&ApproxTopN(400), &input, pbits);
        assert_same_terms(&got, &want, &format!("and P=2^{pbits}"));
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
        let got =
            partitioned_finalize(&And(CoefficientThreshold(1e9), WeightCutoff(0)), &input, pbits);
        assert_same_terms(&got, &input, &format!("and P=2^{pbits}"));
    }
}

/// An `Or` of filters has no layer pass, partitioned or not.
#[test]
fn an_or_of_filters_runs_no_layer_pass_either_way() {
    let input = rand_sum_real::<1>(600, 32, 0xB0B0);
    let policy = Or::new(CoefficientThreshold(1e-3), WeightCutoff(10));

    let mut unpartitioned = input.clone();
    policy.finalize_layer(&mut unpartitioned);
    assert_eq!(unpartitioned.len(), input.len(), "Or has no layer pass");

    for pbits in [1u8, 2] {
        assert_matches_single_partition(&policy, &input, pbits, &format!("or P=2^{pbits}"));
    }
}

/// At `P = 1` the collective collapse is the unpartitioned one, seed for seed.
#[test]
fn collapse_sample_at_one_partition_is_the_unpartitioned_collapse() {
    let input = crate::test_support::rand_sum::<1>(3000, 16, 0xC0C0);
    for seed in 0..20u64 {
        let mut want = input.clone();
        CollapseSample::new(50, 40, seed).finalize_layer(&mut want);
        let got = partitioned_finalize(&CollapseSample::new(50, 40, seed), &input, 0);
        assert_same_terms(&got, &want, &format!("seed {seed}"));
    }
}

/// Below the cache every partition keeps its share, even where a partition alone is far below it.
#[test]
fn collapse_sample_partitioned_is_a_no_op_up_to_the_global_cache() {
    let input = rand_sum_real::<1>(600, 32, 0xF00D);
    for pbits in [1u8, 2] {
        let policy = CollapseSample::new(input.len(), 3, 3);
        let got = partitioned_finalize(&policy, &input, pbits);
        assert_same_terms(&got, &input, &format!("P=2^{pbits}"));
        assert_eq!(policy.collapses(), 0);
    }
}

/// Above the cache the group draws exactly the systematic points of its terms laid end to end in rank order, then bucket order, counted once; one point leaves one string in the whole group.
fn check_partitioned_collapse<const W: usize>() {
    use crate::test_support::{assert_systematic_resample, rand_sum};
    let input = rand_sum::<W>(2000, 16, 0x5A3D + W as u64);
    for pbits in [1u8, 2] {
        let parts = split(&input, pbits);
        let terms: Vec<_> =
            parts.iter().flat_map(|part| part.iter().map(|(x, z, c)| (*x, *z, c))).collect();
        for (seed, samples) in [(0u64, 1usize), (1, 3), (2, 500), (3, 4000)] {
            let policy = CollapseSample::new(100, samples, seed);
            let got = partitioned_finalize(&policy, &input, pbits);
            let u = CollapseSample::new(0, 1, seed).uniform(0);
            let what = format!("W={W} P=2^{pbits} samples={samples}");
            assert_systematic_resample(&got, &terms, u, samples, 1e-12, &what);
            assert_eq!(policy.collapses(), 1, "{what}: one collapse, counted once");
        }
    }
}

#[test]
fn collapse_sample_partitioned_draws_the_global_systematic_points() {
    check_partitioned_collapse::<1>();
    check_partitioned_collapse::<2>();
}

/// Under `PerPartition` each partition draws `samples` points from its own slice, keeping its own `Σ|c|`.
#[test]
fn per_partition_collapse_sample_resamples_each_slice_alone() {
    let input = crate::test_support::rand_sum::<1>(1200, 16, 0xBEAD);
    let l1 = |sum: &PauliSum<1>| sum.iter().map(|(_, _, c)| c.norm()).sum::<f64>();
    for pbits in [1u8, 2] {
        let policy = PerPartition(CollapseSample::new(50, 7, 9));
        let got = partitioned_finalize(&policy, &input, pbits);
        let parts = 1usize << pbits;
        assert!(got.len() <= 7 * parts, "P=2^{pbits}: {} terms", got.len());
        for (part, slice) in split(&got, pbits).iter().zip(split(&input, pbits)) {
            assert!(part.len() <= 7);
            assert!((l1(part) - l1(&slice)).abs() < 1e-9 * l1(&slice), "P=2^{pbits}");
        }
        assert_eq!(policy.0.collapses() as usize, parts, "every partition's collapse counts");
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
    assert!(message.contains("not yet supported") && message.contains("ApproxTopN"), "{message}");
}

/// `supports_partitioned` is `false` exactly for a layer pass with no collective form, looking through `And`.
#[test]
fn supports_partitioned_names_the_policies_a_partitioned_run_accepts() {
    fn supports<T: TruncationPolicy<1>>(policy: T) -> bool {
        policy.supports_partitioned()
    }
    assert!(supports(CoefficientThreshold(1e-3)));
    assert!(supports(WeightCutoff(3)));
    assert!(supports(ApproxTopN(10)));
    assert!(supports(PerPartition(TopN(10))));
    assert!(supports(CollapseSample::new(10, 2, 1)));
    assert!(supports(And(CoefficientThreshold(1e-3), ApproxTopN(10))));
    assert!(supports(Or::new(WeightCutoff(10), CoefficientThreshold(1e-3))));
    assert!(!supports(TopN(10)));
    assert!(!supports(And(CoefficientThreshold(1e-3), TopN(10))));
    assert!(!supports(HalfDone));
}

/// The run-time check every partitioned driver makes before its first layer.
#[test]
fn assert_supports_partitions_rejects_exact_top_n_above_one_partition() {
    assert_supports_partitions::<1, num_complex::Complex64, _>(&TopN(10), 1);
    assert_supports_partitions::<1, num_complex::Complex64, _>(&ApproxTopN(10), 4);
    let panic = std::panic::catch_unwind(|| {
        assert_supports_partitions::<1, num_complex::Complex64, _>(&TopN(10), 2)
    })
    .expect_err("TopN on two partitions must panic");
    let message = panic.downcast_ref::<String>().expect("a formatted message");
    assert!(message.contains("not yet supported") && message.contains("ApproxTopN"), "{message}");
}

/// `PerPartition(policy)` is the policy's single-partition pass on each partition's own slice.
fn assert_per_partition<T: TruncationPolicy<1> + Copy + std::fmt::Debug>(
    policy: T,
    input: &PauliSum<1>,
    pbits: u8,
) {
    let mut parts = split(input, pbits);
    parts.iter_mut().for_each(|part| policy.finalize_layer(part));
    let want = PauliSum::<1>::merge_partitions(parts);
    let got = partitioned_finalize(&PerPartition(policy), input, pbits);
    assert_same_terms(&got, &want, &format!("{policy:?} P=2^{pbits}"));
}

#[test]
fn per_partition_runs_the_policy_on_each_slice_alone() {
    let input = rand_sum_real::<1>(2000, 32, 0x9A27);
    for pbits in [0u8, 1, 2] {
        for n in [0usize, 1, 300, input.len()] {
            assert_per_partition(TopN(n), &input, pbits);
            assert_per_partition(ApproxTopN(n), &input, pbits);
        }
    }
}

#[test]
fn every_coefficient_floor_is_honoured_by_keep_term() {
    use crate::truncation::BuiltinTruncation as B;
    let and = |a: B, b: B| B::And(Box::new(a), Box::new(b));
    let or = |a: B, b: B| B::or(a, b);
    let cases = [
        (B::Keep, -1.0),
        (B::Coefficient(0.25), 0.25),
        (B::Coefficient(-1.0), -1.0),
        (B::Weight(2), -1.0),
        (B::TopN(3), -1.0),
        (and(B::Coefficient(0.25), B::Weight(2)), 0.25),
        (and(B::Coefficient(0.25), B::Coefficient(0.5)), 0.5),
        (or(B::Coefficient(0.25), B::Coefficient(0.5)), 0.25),
        (or(B::Coefficient(0.25), B::Weight(2)), -1.0),
        (B::PerPartition(Box::new(B::Coefficient(0.5))), 0.5),
    ];
    let (x, z) = ([0b1011u64], [0b0110u64]);
    for (policy, want) in cases {
        let floor = <B as TruncationPolicy<1>>::coefficient_floor(&policy);
        assert_eq!(floor, want, "{policy:?}");
        for magnitude in [0.0, 0.1, 0.25, 0.3, 0.5, 0.7] {
            let c = Complex64::from_polar(magnitude, 0.7);
            if magnitude <= floor {
                assert!(!policy.keep_term(&x, &z, c), "{policy:?} keeps {magnitude}");
            }
        }
    }
    let lowered = And(CoefficientThreshold(0.25), WeightCutoff(2));
    assert_eq!(<_ as TruncationPolicy<1>>::coefficient_floor(&lowered), 0.25);
}
