use super::*;
use proptest::prelude::*;

/// At `ε = 0` a magnitude whose square underflows is dropped: `(1e-100)²` survives, `(1e-200)²` rounds to `0.0`.
#[test]
fn coefficient_threshold_drops_squares_that_underflow_to_zero() {
    let policy = CoefficientThreshold(0.0);
    assert!(<CoefficientThreshold as TruncationPolicy<1>>::keep_term(
        &policy,
        &[1],
        &[0],
        Complex64::new(1e-100, 0.0)
    ));
    assert!(!<CoefficientThreshold as TruncationPolicy<1>>::keep_term(
        &policy,
        &[1],
        &[0],
        Complex64::new(1e-200, 0.0)
    ));
    // An exact zero is dropped at ε = 0.
    assert!(!<CoefficientThreshold as TruncationPolicy<1>>::keep_term(
        &policy,
        &[1],
        &[0],
        Complex64::new(0.0, 0.0)
    ));
}

/// A negative threshold keeps everything, including an exact zero: the squared form must not invert `|c| > ε`'s vacuous truth for `ε < 0`.
#[test]
fn coefficient_threshold_negative_epsilon_keeps_everything() {
    let policy = CoefficientThreshold(-1.0);
    for c in [
        Complex64::new(0.0, 0.0),
        Complex64::new(1e-300, 0.0),
        Complex64::new(3.0, -4.0),
    ] {
        assert!(
            <CoefficientThreshold as TruncationPolicy<1>>::keep_term(&policy, &[1], &[0], c),
            "negative epsilon must keep {c}"
        );
    }
}

/// `finalizes_layer` must agree with which builtins override `finalize_layer`: only `TopN`, plus `And` inheriting from either side; `Or` never does.
#[test]
fn layer_finalize_hint_matches_the_builtins() {
    assert!(
        !<CoefficientThreshold as TruncationPolicy<1>>::finalizes_layer(&CoefficientThreshold(
            1e-9
        ))
    );
    assert!(!<WeightCutoff as TruncationPolicy<2>>::finalizes_layer(
        &WeightCutoff(3)
    ));
    assert!(<TopN as TruncationPolicy<1>>::finalizes_layer(&TopN(4)));

    let cheap = And(CoefficientThreshold(1e-9), WeightCutoff(2));
    assert!(!<_ as TruncationPolicy<1>>::finalizes_layer(&cheap));
    let with_topn = And(CoefficientThreshold(1e-9), TopN(4));
    assert!(<_ as TruncationPolicy<1>>::finalizes_layer(&with_topn));
    let topn_first = And(TopN(4), WeightCutoff(2));
    assert!(<_ as TruncationPolicy<1>>::finalizes_layer(&topn_first));

    let ored = Or::new(CoefficientThreshold(1e-9), WeightCutoff(4));
    assert!(!<_ as TruncationPolicy<1>>::finalizes_layer(&ored));
}

#[test]
#[should_panic(
    expected = "Or combines per-term filters only, but its second operand has a layer pass"
)]
fn or_with_top_n_panics_in_finalizes_layer() {
    let _ =
        <_ as TruncationPolicy<1>>::finalizes_layer(&Or::new(CoefficientThreshold(1e-9), TopN(4)));
}

#[test]
#[should_panic(expected = "Or combines per-term filters only, but both operands have")]
fn or_with_layer_passes_nested_in_and_panics_in_finalizes_layer() {
    let policy = Or::new(
        And(WeightCutoff(2), ApproxTopN(4)),
        And(CollapseSample::new(4, 1), CoefficientThreshold(0.1)),
    );
    let _ = <_ as TruncationPolicy<1>>::finalizes_layer(&policy);
}

/// The hint defaults to the conservative `true`, so a forgotten override still gets its layer pass run.
#[test]
fn layer_finalize_hint_defaults_to_conservative_true() {
    struct Silent;
    impl<const W: usize> TruncationPolicy<W> for Silent {}
    assert!(<_ as TruncationPolicy<1>>::finalizes_layer(&Silent));
}

/// `WeightCutoff(2)` keeps weights 0, 1, 2 and drops 3.
/// Identity I (weight 0), single X (1), XZ on qubits 0+1 (2) all kept;
/// X on q0 + Y on q1 + Z on q2 (3) dropped.
#[test]
fn weight_cutoff_keeps_below_or_equal() {
    let cut = WeightCutoff(2);
    // Identity: weight 0.
    assert!(<WeightCutoff as TruncationPolicy<1>>::keep_term(
        &cut,
        &[0],
        &[0],
        Complex64::new(1.0, 0.0)
    ));
    // X on q0: weight 1 (x bit set).
    assert!(<WeightCutoff as TruncationPolicy<1>>::keep_term(
        &cut,
        &[1],
        &[0],
        Complex64::new(1.0, 0.0)
    ));
    // X on q0, Z on q1: weight 2.
    assert!(<WeightCutoff as TruncationPolicy<1>>::keep_term(
        &cut,
        &[0b01],
        &[0b10],
        Complex64::new(1.0, 0.0)
    ));
    // X on q0, Y on q1 (x+z), Z on q2: weight 3, dropped.
    assert!(!<WeightCutoff as TruncationPolicy<1>>::keep_term(
        &cut,
        &[0b011],
        &[0b110],
        Complex64::new(1.0, 0.0)
    ));
}

/// `WeightCutoff(0)` keeps only the identity.
#[test]
fn weight_cutoff_zero_keeps_only_identity() {
    let cut = WeightCutoff(0);
    assert!(<WeightCutoff as TruncationPolicy<1>>::keep_term(
        &cut,
        &[0],
        &[0],
        Complex64::new(1.0, 0.0)
    ));
    // Any non-identity Pauli is dropped.
    assert!(!<WeightCutoff as TruncationPolicy<1>>::keep_term(
        &cut,
        &[1],
        &[0],
        Complex64::new(1.0, 0.0)
    ));
    assert!(!<WeightCutoff as TruncationPolicy<1>>::keep_term(
        &cut,
        &[0],
        &[1],
        Complex64::new(1.0, 0.0)
    ));
    assert!(!<WeightCutoff as TruncationPolicy<1>>::keep_term(
        &cut,
        &[1],
        &[1],
        Complex64::new(1.0, 0.0)
    ));
}

/// Weight counts across words; qubit 64 is word 1, bit 0.
#[test]
fn weight_cutoff_w2_word_boundary() {
    let cut = WeightCutoff(1);
    // X on qubit 64 alone: weight 1, kept.
    assert!(<WeightCutoff as TruncationPolicy<2>>::keep_term(
        &cut,
        &[0u64, 1u64],
        &[0u64, 0u64],
        Complex64::new(1.0, 0.0)
    ));
    // X on qubit 0 AND X on qubit 64: weight 2, dropped.
    assert!(!<WeightCutoff as TruncationPolicy<2>>::keep_term(
        &cut,
        &[1u64, 1u64],
        &[0u64, 0u64],
        Complex64::new(1.0, 0.0)
    ));
}

/// Ten distinct keys with decreasing |coeff| (10, 9, …, 1); `TopN(3)` keeps the three with magnitudes 10, 9, 8.
#[test]
fn top_n_keeps_largest_three_of_ten() {
    // Largest magnitudes first in key order.
    let mut sum = PauliSum::<1>::from_sorted_columns(
        (1u64..=10).map(|i| [i]).collect(),
        vec![[0u64]; 10],
        (1u64..=10)
            .rev()
            .map(|m| Complex64::new(m as f64, 0.0))
            .collect(),
        4,
    );
    sum.assert_invariants();
    TopN(3).finalize_layer(&mut sum);
    assert_eq!(sum.len(), 3);
    // Survivors: original magnitudes 10, 9, 8 → x = [1], [2], [3].
    let (x, _, c) = sum.to_arrays();
    assert_eq!(x, vec![[1u64], [2u64], [3u64]]);
    let magnitudes: Vec<f64> = c.iter().map(|c| c.norm()).collect();
    assert_eq!(magnitudes, vec![10.0, 9.0, 8.0]);
    sum.assert_invariants();
}

/// `TopN(N) where N >= len` is a no-op, checked at both `N > len` and `N == len` since the tie rule only engages on the `len > n` path.
#[test]
fn top_n_at_or_above_len_is_a_no_op() {
    let mut sum = PauliSum::<1>::from_sorted_columns(
        vec![[0], [0], [1]],
        vec![[0], [1], [0]],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ],
        1,
    );
    let (snapshot_x, snapshot_z, snapshot_c) = sum.to_arrays();
    TopN(5).finalize_layer(&mut sum);
    assert_eq!(
        sum.to_arrays(),
        (snapshot_x.clone(), snapshot_z.clone(), snapshot_c.clone())
    );
    TopN(3).finalize_layer(&mut sum);
    assert_eq!(sum.to_arrays(), (snapshot_x, snapshot_z, snapshot_c));

    // All-tied at exactly `n`: still a no-op, not a wipe.
    let mut tied = PauliSum::<1>::from_sorted_columns(
        vec![[0], [1], [2]],
        vec![[0]; 3],
        vec![Complex64::new(2.0, 0.0); 3],
        2,
    );
    TopN(3).finalize_layer(&mut tied);
    assert_eq!(tied.len(), 3);
    tied.assert_invariants();
}

/// With all magnitudes distinct, the tie group at rank `n` has size one and always fits, so `TopN(n)` retains exactly `n`.
#[test]
fn top_n_all_distinct_retains_exactly_n() {
    let magnitudes = [7.0f64, 1.0, 5.0, 3.0, 9.0, 2.0, 8.0, 4.0];
    let mut sum = PauliSum::<1>::from_sorted_columns(
        (0u64..8).map(|i| [i]).collect(),
        vec![[0u64]; 8],
        magnitudes.iter().map(|&m| Complex64::new(m, 0.0)).collect(),
        3,
    );
    sum.assert_invariants();
    TopN(5).finalize_layer(&mut sum);
    assert_eq!(sum.len(), 5, "all-distinct input must retain exactly n");
    let (_, _, c) = sum.to_arrays();
    let mut got: Vec<f64> = c.iter().map(|c| c.norm()).collect();
    got.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(got, vec![9.0, 8.0, 7.0, 5.0, 4.0]);
    sum.assert_invariants();
}

/// A tie group straddling the cut is split in bucket order: magnitudes 5, 4, 3, 3, 3, 2 with `n = 3` keeps 5, 4 and the first 3.
#[test]
fn top_n_splits_a_straddling_tie_group_in_bucket_order() {
    let magnitudes = [5.0f64, 4.0, 3.0, 3.0, 3.0, 2.0];
    let mut sum = PauliSum::<1>::from_sorted_columns(
        (0u64..6).map(|i| [i]).collect(),
        vec![[0u64]; 6],
        magnitudes.iter().map(|&m| Complex64::new(m, 0.0)).collect(),
        3,
    );
    TopN(3).finalize_layer(&mut sum);
    let (x, _, c) = sum.to_arrays();
    assert_eq!(x, vec![[0u64], [1u64], [2u64]]);
    let got: Vec<f64> = c.iter().map(|c| c.norm()).collect();
    assert_eq!(got, vec![5.0, 4.0, 3.0]);
    sum.assert_invariants();
}

/// A tie group that ends exactly at rank `n` fits and is kept whole: magnitudes 5, 4, 3, 3, 2, 1 with `n = 4` keeps all four of 5, 4, 3, 3.
#[test]
fn top_n_keeps_a_tie_group_that_fits_exactly() {
    let magnitudes = [5.0f64, 4.0, 3.0, 3.0, 2.0, 1.0];
    let mut sum = PauliSum::<1>::from_sorted_columns(
        (0u64..6).map(|i| [i]).collect(),
        vec![[0u64]; 6],
        magnitudes.iter().map(|&m| Complex64::new(m, 0.0)).collect(),
        3,
    );
    sum.assert_invariants();
    TopN(4).finalize_layer(&mut sum);
    assert_eq!(sum.len(), 4, "a group that fits must be kept in full");
    let (x, _, c) = sum.to_arrays();
    assert_eq!(x, vec![[0u64], [1u64], [2u64], [3u64]]);
    let got: Vec<f64> = c.iter().map(|c| c.norm()).collect();
    assert_eq!(got, vec![5.0, 4.0, 3.0, 3.0]);
    sum.assert_invariants();
}

/// An all-tied sum keeps exactly `n`, the first in bucket order.
#[test]
fn top_n_keeps_exactly_n_of_an_all_tied_sum() {
    // Same magnitude, different phases (fourth roots of unity so every norm is bitwise 2.0): a multiplet, not duplicates.
    let coefficients = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(-2.0, 0.0),
        Complex64::new(0.0, 2.0),
        Complex64::new(0.0, -2.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(-2.0, 0.0),
    ];
    let mut sum = PauliSum::<1>::from_sorted_columns(
        (0u64..6).map(|i| [i]).collect(),
        vec![[0u64]; 6],
        coefficients.clone(),
        3,
    );
    TopN(3).finalize_layer(&mut sum);
    let (x, _, c) = sum.to_arrays();
    assert_eq!(x, vec![[0u64], [1u64], [2u64]]);
    assert_eq!(c, coefficients[..3].to_vec());
    sum.assert_invariants();
}

/// Magnitudes below the square-underflow floor (`≈1.57e-162`) tie at zero: six terms at 1e-200..6e-200 with `n = 3` keep the first three in bucket order, not the largest three.
#[test]
fn top_n_treats_underflowing_squares_as_one_tie_group() {
    let mut sum = PauliSum::<1>::from_sorted_columns(
        (0u64..6).map(|i| [i]).collect(),
        vec![[0u64]; 6],
        (1..=6)
            .map(|i| Complex64::new(i as f64 * 1e-200, 0.0))
            .collect(),
        3,
    );
    TopN(3).finalize_layer(&mut sum);
    let (x, _, c) = sum.to_arrays();
    assert_eq!(x, vec![[0u64], [1u64], [2u64]]);
    assert_eq!(
        c,
        (1..=3)
            .map(|i| Complex64::new(i as f64 * 1e-200, 0.0))
            .collect::<Vec<_>>()
    );
    sum.assert_invariants();
}

/// A cut inside an underflowing tail keeps the representable terms and fills the rest from the tail in bucket order: magnitudes 3, 2, 1 plus five terms at 5e-200..1e-200 with `n = 5`.
#[test]
fn top_n_fills_from_an_underflowing_tail_in_bucket_order() {
    let mut sum = PauliSum::<1>::from_sorted_columns(
        (0u64..8).map(|i| [i]).collect(),
        vec![[0u64]; 8],
        vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(5e-200, 0.0),
            Complex64::new(4e-200, 0.0),
            Complex64::new(3e-200, 0.0),
            Complex64::new(2e-200, 0.0),
            Complex64::new(1e-200, 0.0),
        ],
        3,
    );
    TopN(5).finalize_layer(&mut sum);
    let (x, _, c) = sum.to_arrays();
    assert_eq!(x, (0u64..5).map(|i| [i]).collect::<Vec<_>>());
    assert_eq!(
        c,
        vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(5e-200, 0.0),
            Complex64::new(4e-200, 0.0),
        ]
    );
    sum.assert_invariants();
}

/// On multi-bucket tie-dense sums, `TopN(n)` keeps exactly the stable-sort transcription of the rule, at every bucket count.
#[test]
fn top_n_matches_the_stable_sort_rule_across_bucket_counts() {
    let input = crate::test_support::tie_heavy_sum::<1>(2000, 8, 0x7136);
    for bits in [0u8, 2, 5, 9] {
        let hashed = input.clone().with_hash(Gf2Hash::<1>::new(8, bits, 0x99));
        for n in [0usize, 1, 3, 250, 500, 700, 1000, 1999, 2000, 2500] {
            let mut sum = hashed.clone();
            TopN(n).finalize_layer(&mut sum);
            sum.assert_invariants();
            let want = first_n_by_rank(&storage_triples(&hashed), n, magnitude_rank);
            assert_eq!(sum.len(), n.min(input.len()), "bits={bits} n={n}");
            assert_eq!(storage_triples(&sum), want, "bits={bits} n={n}");
        }
    }
}

/// A smaller sum finalized after a larger one on the same thread reads only its own prefix of the pooled buffer, not the stale tail.
#[test]
fn a_smaller_layer_after_a_larger_one_reads_only_its_own_prefix() {
    let mut big = PauliSum::<1>::from_sorted_columns(
        (0u64..20).map(|i| [i]).collect(),
        vec![[0u64]; 20],
        (1..=20).map(|m| Complex64::new(m as f64, 0.0)).collect(),
        5,
    );
    TopN(5).finalize_layer(&mut big);
    assert_eq!(big.len(), 5, "first layer: magnitudes 16..=20 survive");

    // Six terms, every magnitude below the previous layer's threshold.
    // Sixteenths, so the literals below are exact in binary.
    let mut small = PauliSum::<1>::from_sorted_columns(
        (0u64..6).map(|i| [i]).collect(),
        vec![[0u64]; 6],
        (1..=6)
            .map(|m| Complex64::new(m as f64 / 16.0, 0.0))
            .collect(),
        3,
    );
    TopN(3).finalize_layer(&mut small);
    small.assert_invariants();
    let (x, _, c) = small.to_arrays();
    assert_eq!(x, vec![[3u64], [4u64], [5u64]]);
    assert_eq!(
        c,
        vec![
            Complex64::new(0.25, 0.0),
            Complex64::new(0.3125, 0.0),
            Complex64::new(0.375, 0.0),
        ]
    );
}

/// `finalize_layer` inside a rayon job, where work-stealing may re-enter it.
#[test]
fn finalize_layer_runs_inside_a_rayon_job() {
    use crate::test_support::rand_sum;
    let sums: Vec<PauliSum<1>> = (0..16)
        .map(|k| rand_sum::<1>(2000, 10, 0xF1A5 + k))
        .collect();
    let want: Vec<usize> = sums.iter().map(|s| s.len().min(500)).collect();
    let got: Vec<usize> = sums
        .into_par_iter()
        .map(|mut s| {
            TopN(500).finalize_layer(&mut s);
            s.assert_invariants();
            s.len()
        })
        .collect();
    assert_eq!(got, want, "every sum must truncate to n on a worker thread");
}

/// `TopN(0)` empties the sum.
#[test]
fn top_n_zero_empties_sum() {
    let mut sum = PauliSum::<1>::from_sorted_columns(
        vec![[0], [1]],
        vec![[1], [0]],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        1,
    );
    TopN(0).finalize_layer(&mut sum);
    assert!(sum.is_empty());
    sum.assert_invariants();
}

/// Largest coefficients sit at the end of the sort order; survivors must still come back in (x, z) sort order, not magnitude order.
#[test]
fn top_n_preserves_sort_order() {
    // Five keys, magnitudes 1, 2, 3, 4, 5 (back-loaded).
    let mut sum = PauliSum::<1>::from_sorted_columns(
        vec![[1], [2], [3], [4], [5]],
        vec![[0]; 5],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
        ],
        4,
    );
    sum.assert_invariants();
    TopN(3).finalize_layer(&mut sum);
    assert_eq!(sum.len(), 3);
    // Survivors: magnitudes 5, 4, 3, i.e. keys [5], [4], [3]; sort order preservation means they come back as [3], [4], [5].
    let (x, _, c) = sum.to_arrays();
    assert_eq!(x, vec![[3u64], [4u64], [5u64]]);
    assert_eq!(
        c,
        vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
        ]
    );
    sum.assert_invariants();
}

/// A `W = 1` sum of `magnitudes.len()` distinct keys (`x = i`, `z = 0`) with the given real coefficients, single-bucket and already in key order.
fn sum_of_mags(magnitudes: &[f64]) -> PauliSum<1> {
    PauliSum::<1>::from_sorted_columns(
        (0u64..magnitudes.len() as u64).map(|i| [i]).collect(),
        vec![[0u64]; magnitudes.len()],
        magnitudes.iter().map(|&m| Complex64::new(m, 0.0)).collect(),
        32,
    )
}

/// The surviving magnitudes, in canonical order.
fn kept_mags<const W: usize>(sum: &PauliSum<W>) -> Vec<f64> {
    sum.iter().map(|(_, _, c)| c.norm()).collect()
}

/// Octave of `|c|²`, i.e. the bin `ApproxTopN` histograms into, derived here from the definition rather than from the implementation.
fn octave(c: Complex64) -> usize {
    (c.norm_sqr().to_bits() >> 52) as usize
}

/// Every octave above the boundary is kept whole and the boundary octave fills the rest in bucket order, so exactly `n` survive — hand-tabulated for four octaves of populations 1, 2, 3, 4 (keys in magnitude order, one bucket).
#[test]
fn approx_top_n_keeps_exactly_n_filling_the_boundary_octave_in_bucket_order() {
    let magnitudes = [8.0f64, 4.0, 4.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0];
    for n in 1usize..10 {
        let mut sum = sum_of_mags(&magnitudes);
        ApproxTopN(n).finalize_layer(&mut sum);
        sum.assert_invariants();
        let keys: Vec<u64> = sum.iter().map(|(x, _, _)| x[0]).collect();
        assert_eq!(keys, (0..n as u64).collect::<Vec<_>>(), "n={n}");
    }
    // `n >= len` is a no-op, like every other policy's.
    let mut sum = sum_of_mags(&magnitudes);
    ApproxTopN(10).finalize_layer(&mut sum);
    assert_eq!(sum.len(), 10);
}

/// Within the boundary octave bucket order decides, not magnitude: keys carry magnitudes 2, 1, 1.3, 1.1, 1.2 and `n = 3` keeps 2 and the first two of the octave `[1, 2)` of `|c|²`, 1 and 1.3.
#[test]
fn approx_top_n_does_not_rank_inside_the_boundary_octave() {
    let mut sum = sum_of_mags(&[2.0, 1.0, 1.3, 1.1, 1.2]);
    ApproxTopN(3).finalize_layer(&mut sum);
    assert_eq!(kept_mags(&sum), vec![2.0, 1.0, 1.3]);
}

/// When the cut lands exactly on an octave boundary (`n` = 1, 3, or 6, the fixture's cumulative populations), the approximation is no approximation: both policies return the same top `n`.
#[test]
fn approx_top_n_matches_top_n_when_the_histogram_resolves_exactly() {
    let magnitudes = [8.0f64, 4.0, 4.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0];
    for n in [1usize, 3, 6] {
        let mut approx = sum_of_mags(&magnitudes);
        ApproxTopN(n).finalize_layer(&mut approx);
        let mut exact = sum_of_mags(&magnitudes);
        TopN(n).finalize_layer(&mut exact);
        assert_eq!(exact.len(), n, "n={n}: TopN must resolve exactly here");
        assert_eq!(
            approx.to_arrays(),
            exact.to_arrays(),
            "n={n}: the two policies must agree term for term"
        );
    }
}

/// Larger `n` keeps a superset: the octave edge can only move down as `n` grows, so retained sets nest.
#[test]
fn approx_top_n_is_monotone_in_n() {
    let input = crate::test_support::rand_sum::<1>(2000, 10, 0xA9C7);
    let mut previous: Option<std::collections::HashSet<(u64, u64)>> = None;
    for n in [1usize, 5, 50, 300, 700, 1300, 1900] {
        let mut sum = input.clone();
        ApproxTopN(n).finalize_layer(&mut sum);
        sum.assert_invariants();
        // `(x, z)` — `rand_sum` draws both, so `x` alone is not a key.
        let keys: std::collections::HashSet<(u64, u64)> =
            sum.iter().map(|(x, z, _)| (x[0], z[0])).collect();
        if let Some(prev) = &previous {
            assert!(
                prev.is_subset(&keys),
                "n={n}: the kept set must be a superset of every smaller n's"
            );
        }
        previous = Some(keys);
    }
}

/// The `≈n` contract, checked against a bound derived from the input:
/// `kept <= n`, and `kept > n - p` where `p` is the population of the highest *excluded* octave. Equivalently `kept + p > n`: the next octave down would have overshot.
#[test]
fn approx_top_n_shortfall_is_bounded_by_one_octave() {
    let input = crate::test_support::rand_sum::<1>(3000, 10, 0xB0117);
    let len = input.len();
    for n in [1usize, 17, 200, 900, 2000, len - 1] {
        let mut sum = input.clone();
        ApproxTopN(n).finalize_layer(&mut sum);
        let kept = sum.len();
        assert!(kept <= n, "n={n}: kept {kept} exceeds the bound");

        // The highest excluded octave's population in the input is the slack.
        let survivors: std::collections::HashSet<(u64, u64)> =
            sum.iter().map(|(x, z, _)| (x[0], z[0])).collect();
        let dropped_octaves: Vec<usize> = input
            .iter()
            .filter(|(x, z, _)| !survivors.contains(&(x[0], z[0])))
            .map(|(_, _, c)| octave(c))
            .collect();
        let p = match dropped_octaves.iter().copied().max() {
            None => 0,
            Some(top) => input.iter().filter(|(_, _, c)| octave(*c) == top).count(),
        };
        assert!(
            kept + p > n,
            "n={n}: kept {kept} + excluded octave {p} must overshoot n, \
             else that octave should have been kept"
        );
    }
}

/// On multi-bucket tie-dense sums `ApproxTopN(n)` keeps exactly the stable-sort transcription of the rule, at every bucket count: whole octaves above the boundary, the first of the boundary octave in bucket order.
#[test]
fn approx_top_n_matches_the_stable_sort_rule_across_bucket_counts() {
    let input = crate::test_support::tie_heavy_sum::<1>(2000, 8, 0x7135);
    for bits in [0u8, 2, 5, 9] {
        let hashed = input.clone().with_hash(Gf2Hash::<1>::new(8, bits, 0x99));
        for n in [1usize, 3, 250, 500, 700, 1200, 1900, input.len() - 1] {
            let mut sum = hashed.clone();
            ApproxTopN(n).finalize_layer(&mut sum);
            sum.assert_invariants();
            assert_eq!(sum.len(), n, "bits={bits} n={n}");
            let want = first_n_by_rank(&storage_triples(&hashed), n, octave_rank);
            assert_eq!(storage_triples(&sum), want, "bits={bits} n={n}");
        }
    }
}

/// One octave holding everything keeps the first `n` in bucket order, the smallest magnitudes included.
#[test]
fn approx_top_n_keeps_the_first_n_of_a_single_octave_sum() {
    let magnitudes = [1.0f64, 1.125, 1.25, 1.375];
    let mut sum = sum_of_mags(&magnitudes);
    ApproxTopN(3).finalize_layer(&mut sum);
    assert_eq!(kept_mags(&sum), vec![1.0, 1.125, 1.25]);
    sum.assert_invariants();
    // …and it is a no-op at n >= len, as always.
    let mut sum = sum_of_mags(&magnitudes);
    ApproxTopN(4).finalize_layer(&mut sum);
    assert_eq!(sum.len(), 4);
}

/// `ApproxTopN(0)` empties the sum.
#[test]
fn approx_top_n_zero_empties_sum() {
    let mut sum = sum_of_mags(&[1.0, 2.0]);
    ApproxTopN(0).finalize_layer(&mut sum);
    assert!(sum.is_empty());
    sum.assert_invariants();
}

/// `W = 2`: the const-generic surface, on keys that straddle the word boundary.
#[test]
fn approx_top_n_w2() {
    let build = || {
        PauliSum::<2>::from_sorted_columns(
            vec![[0, 1], [0, 2], [1, 0], [2, 0]],
            vec![[0, 0]; 4],
            vec![
                Complex64::new(4.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            128,
        )
    };
    let mut sum = build();
    ApproxTopN(2).finalize_layer(&mut sum);
    assert_eq!(kept_mags(&sum), vec![4.0, 2.0]);
    sum.assert_invariants();

    let mut sum = build();
    ApproxTopN(3).finalize_layer(&mut sum);
    assert_eq!(kept_mags(&sum), vec![4.0, 2.0, 2.0]);
    sum.assert_invariants();
}

/// A complex coefficient is ranked by `re² + im²` like everywhere else.
#[test]
fn approx_top_n_ranks_complex_coefficients_by_squared_magnitude() {
    let build = || {
        PauliSum::<1>::from_sorted_columns(
            vec![[0], [1], [2]],
            vec![[0]; 3],
            vec![
                // |c|² = 25 → octave [16, 32)
                Complex64::new(3.0, 4.0),
                // |c|² = 36 → octave [32, 64), the largest
                Complex64::new(0.0, 6.0),
                // |c|² = 4 → octave [4, 8)
                Complex64::new(-2.0, 0.0),
            ],
            8,
        )
    };
    let mut sum = build();
    ApproxTopN(1).finalize_layer(&mut sum);
    assert_eq!(kept_mags(&sum), vec![6.0], "only 6i fits in one slot");
    sum.assert_invariants();

    // Two slots take both of the top two octaves; magnitudes come back in key order, not magnitude order.
    let mut sum = build();
    ApproxTopN(2).finalize_layer(&mut sum);
    assert_eq!(kept_mags(&sum), vec![5.0, 6.0]);
    sum.assert_invariants();
}

proptest! {
    /// The whole contract over tie-dense magnitude multisets: exactly `min(n, len)` survive, and they are the stable-sort transcription of the rule.
    /// Magnitudes are small integers so squares collide into few octaves and the interesting branches are hit often.
    #[test]
    fn approx_top_n_keeps_exactly_n_on_tie_dense_multisets(
        values in proptest::collection::vec(1u32..40u32, 1..48),
        n in 0usize..48,
    ) {
        let magnitudes: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
        let input = sum_of_mags(&magnitudes);
        let mut sum = input.clone();
        ApproxTopN(n).finalize_layer(&mut sum);
        prop_assert_eq!(sum.len(), n.min(magnitudes.len()));
        prop_assert_eq!(
            storage_triples(&sum),
            first_n_by_rank(&storage_triples(&input), n, octave_rank)
        );
        let mut exact = input.clone();
        TopN(n).finalize_layer(&mut exact);
        prop_assert_eq!(
            storage_triples(&exact),
            first_n_by_rank(&storage_triples(&input), n, magnitude_rank)
        );
    }
}

use crate::pauli_sum::Gf2Hash;
use crate::test_support::{
    assert_frequencies, collapsed_index, first_n_by_rank, four_term_keys, magnitude_rank,
    octave_rank, rand_sum, storage_triples, weighted_four_term_sum, FOUR_TERM_WEIGHTS,
};

/// Weights `[0, 1, 0, 2]`: slot 1 owns `[0, 1)`, slot 3 owns `[1, 3)`, and a target at or past 3 falls back to slot 3.
#[test]
fn pick_slot_walks_the_running_total() {
    let w = [0.0, 1.0, 0.0, 2.0];
    assert_eq!(pick_slot(w, 0.0), Some((1, 0.0)));
    assert_eq!(pick_slot(w, 0.75), Some((1, 0.75)));
    assert_eq!(pick_slot(w, 1.0), Some((3, 0.0)));
    assert_eq!(pick_slot(w, 2.5), Some((3, 1.5)));
    assert_eq!(pick_slot(w, 3.0), Some((3, 2.0)));
    assert_eq!(pick_slot([0.0, 0.0], 0.0), None);
}

#[test]
fn collapse_sample_is_a_no_op_up_to_the_cache() {
    let input = weighted_four_term_sum::<1>(8);
    for cache in [4usize, 5, 100] {
        let policy = CollapseSample::new(cache, 7);
        let mut sum = input.clone();
        policy.finalize_layer(&mut sum);
        assert_eq!(sum.to_arrays(), input.to_arrays(), "cache {cache}");
        assert_eq!(policy.collapses(), 0);
    }
    assert!(<_ as TruncationPolicy<1>>::finalizes_layer(
        &CollapseSample::new(4, 7)
    ));
}

#[test]
fn collapse_sample_leaves_one_input_string_with_unit_coefficient() {
    let keys = four_term_keys::<1>();
    let policy = CollapseSample::new(3, 11);
    for pass in 1..=5u64 {
        let mut sum = weighted_four_term_sum::<1>(8);
        policy.finalize_layer(&mut sum);
        sum.assert_invariants();
        collapsed_index(&sum, &keys);
        assert_eq!(policy.collapses(), pass);
    }
}

/// Picks over 5000 seeds match `|c|² / Σ|c|²`, in one bucket and spread over four, at both widths.
fn check_pick_frequencies<const W: usize>(bits: u8) {
    let keys = four_term_keys::<W>();
    let input = weighted_four_term_sum::<W>(8).with_hash(Gf2Hash::new(8, bits, 0xB0C4));
    if bits > 0 {
        let used = (0..input.num_buckets())
            .filter(|&b| input.bucket_len(b) > 0)
            .count();
        assert!(used > 1, "the fixture must span several buckets");
    }
    let mut counts = [0usize; 4];
    for seed in 0..5000u64 {
        let mut sum = input.clone();
        CollapseSample::new(3, seed).finalize_layer(&mut sum);
        counts[collapsed_index(&sum, &keys)] += 1;
    }
    assert_frequencies(&counts, &FOUR_TERM_WEIGHTS, &format!("W={W} bits={bits}"));
}

#[test]
fn collapse_sample_draws_by_squared_magnitude() {
    check_pick_frequencies::<1>(0);
    check_pick_frequencies::<1>(2);
    check_pick_frequencies::<2>(0);
    check_pick_frequencies::<2>(2);
}

/// The draw is keyed by `(seed, pass)`, so a trajectory repeats from its seed, and a local pool of one or eight threads picks the same string.
#[test]
fn collapse_sample_is_reproducible_and_thread_count_independent() {
    let input = rand_sum::<1>(4000, 16, 0xC011).with_hash(Gf2Hash::new(16, 5, 0x1234));
    let pick = |threads: usize, seed: u64| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let policy = CollapseSample::new(100, seed);
            (0..3)
                .map(|_| {
                    let mut sum = input.clone();
                    policy.finalize_layer(&mut sum);
                    sum.to_arrays()
                })
                .collect::<Vec<_>>()
        })
    };
    let mut distinct = std::collections::HashSet::new();
    for seed in 0..8u64 {
        let one = pick(1, seed);
        assert_eq!(one, pick(8, seed), "seed {seed}");
        assert_eq!(one, pick(1, seed), "seed {seed} repeats");
        for arrays in &one {
            distinct.insert(arrays.0[0]);
        }
    }
    assert!(distinct.len() > 4, "passes and seeds must draw differently");
}

#[test]
#[should_panic(expected = "cannot draw by weight")]
fn collapse_sample_rejects_an_all_zero_sum() {
    let mut sum = PauliSum::<1>::from_sorted_columns(
        vec![[1], [2]],
        vec![[0], [0]],
        vec![Complex64::new(0.0, 0.0); 2],
        4,
    );
    CollapseSample::new(1, 0).finalize_layer(&mut sum);
}

/// `And` requires both policies to accept. Pair a coeff threshold with a weight cutoff; only terms passing *both* survive.
#[test]
fn and_requires_both_keep() {
    let policy = And(CoefficientThreshold(0.5), WeightCutoff(1));
    // (X, 1.0): |c|=1.0 > 0.5 ✓, weight=1 ≤ 1 ✓ → kept.
    assert!(<And<_, _> as TruncationPolicy<1>>::keep_term(
        &policy,
        &[1],
        &[0],
        Complex64::new(1.0, 0.0)
    ));
    // (X, 0.1): |c|=0.1 ≤ 0.5 ✗ → dropped.
    assert!(!<And<_, _> as TruncationPolicy<1>>::keep_term(
        &policy,
        &[1],
        &[0],
        Complex64::new(0.1, 0.0)
    ));
    // (XZ, 1.0): weight 2 > 1 ✗ → dropped.
    assert!(!<And<_, _> as TruncationPolicy<1>>::keep_term(
        &policy,
        &[0b01],
        &[0b10],
        Complex64::new(1.0, 0.0)
    ));
}

/// `Or` accepts if *either* policy accepts.
#[test]
fn or_keeps_if_either() {
    let policy = Or::new(CoefficientThreshold(0.5), WeightCutoff(0));
    // (I, 0.1): |c| fails (0.1 ≤ 0.5), but weight=0 passes → kept.
    assert!(<Or<_, _> as TruncationPolicy<1>>::keep_term(
        &policy,
        &[0],
        &[0],
        Complex64::new(0.1, 0.0)
    ));
    // (X, 1.0): weight fails, but |c|=1.0 > 0.5 → kept.
    assert!(<Or<_, _> as TruncationPolicy<1>>::keep_term(
        &policy,
        &[1],
        &[0],
        Complex64::new(1.0, 0.0)
    ));
    // (X, 0.1): both fail → dropped.
    assert!(!<Or<_, _> as TruncationPolicy<1>>::keep_term(
        &policy,
        &[1],
        &[0],
        Complex64::new(0.1, 0.0)
    ));
}
