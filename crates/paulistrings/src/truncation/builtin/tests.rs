use super::*;
use num_complex::Complex64;
use proptest::prelude::*;

/// A 256 KiB worker stack holds the histogram's reduction at 32 threads, a debug build included; an overflow aborts the binary on a thread named after this test.
#[test]
fn octave_histogram_fits_a_small_worker_stack() {
    let sum = crate::test_support::rand_sum::<1>(20_000, 64, 0x57AC);
    assert!(sum.num_buckets() >= 128, "too few buckets to split 32 threads × 4 ways");
    let want = octave_histogram(&sum);
    assert_eq!(want.iter().map(|&count| count as usize).sum::<usize>(), sum.len());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(32)
        .stack_size(256 << 10)
        .thread_name(|i| format!("octave_histogram_fits_a_small_worker_stack-{i}"))
        .build()
        .expect("pool");
    pool.install(|| {
        for _ in 0..50 {
            assert_eq!(octave_histogram(&sum), want);
        }
    });
}

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
    for c in [Complex64::new(0.0, 0.0), Complex64::new(1e-300, 0.0), Complex64::new(3.0, -4.0)] {
        assert!(
            <CoefficientThreshold as TruncationPolicy<1>>::keep_term(&policy, &[1], &[0], c),
            "negative epsilon must keep {c}"
        );
    }
}

/// `finalizes_layer` must agree with which builtins override `finalize_layer`: only `TopN`, plus `And` inheriting from either side; `Or` never does.
#[test]
fn layer_finalize_hint_matches_the_builtins() {
    assert!(!<CoefficientThreshold as TruncationPolicy<1>>::finalizes_layer(
        &CoefficientThreshold(1e-9)
    ));
    assert!(!<WeightCutoff as TruncationPolicy<2>>::finalizes_layer(&WeightCutoff(3)));
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
        And(crate::truncation::CollapseSample::new(4, 1, 1), CoefficientThreshold(0.1)),
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

/// Magnitudes below the square-underflow floor (`≈1.57e-162`) tie at zero: six terms at 1e-200..6e-200 with `n = 3` keep the first three in bucket order, not the largest three.
#[test]
fn top_n_treats_underflowing_squares_as_one_tie_group() {
    let mut sum = PauliSum::<1>::from_sorted_columns(
        (0u64..6).map(|i| [i]).collect(),
        vec![[0u64]; 6],
        (1..=6).map(|i| Complex64::new(i as f64 * 1e-200, 0.0)).collect(),
        3,
    );
    TopN(3).finalize_layer(&mut sum);
    let (x, _, c) = sum.to_arrays();
    assert_eq!(x, vec![[0u64], [1u64], [2u64]]);
    assert_eq!(c, (1..=3).map(|i| Complex64::new(i as f64 * 1e-200, 0.0)).collect::<Vec<_>>());
    sum.assert_invariants();
}

/// A NaN sorts below every number: of 3, NaN, 2, 1 at `n = 2` exactly 3 and 2 survive, and of NaN, NaN, 1 only the 1.
#[test]
fn top_n_never_keeps_a_nan_coefficient() {
    let real = |v: f64| Complex64::new(v, 0.0);
    for (coefficients, want) in [
        (vec![3.0, f64::NAN, 2.0, 1.0], vec![[0u64], [2u64]]),
        (vec![f64::NAN, f64::NAN, 1.0], vec![[2u64]]),
    ] {
        let len = coefficients.len() as u64;
        let mut sum = PauliSum::<1>::from_sorted_columns(
            (0..len).map(|i| [i]).collect(),
            vec![[0u64]; len as usize],
            coefficients.into_iter().map(real).collect(),
            3,
        );
        TopN(2).finalize_layer(&mut sum);
        sum.assert_invariants();
        let (mut x, _, _) = sum.to_arrays();
        x.sort();
        assert_eq!(x, want);
    }
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
        (1..=6).map(|m| Complex64::new(m as f64 / 16.0, 0.0)).collect(),
        3,
    );
    TopN(3).finalize_layer(&mut small);
    small.assert_invariants();
    let (x, _, c) = small.to_arrays();
    assert_eq!(x, vec![[3u64], [4u64], [5u64]]);
    assert_eq!(
        c,
        vec![Complex64::new(0.25, 0.0), Complex64::new(0.3125, 0.0), Complex64::new(0.375, 0.0),]
    );
}

/// `finalize_layer` inside a rayon job, where work-stealing may re-enter it.
#[test]
fn finalize_layer_runs_inside_a_rayon_job() {
    use crate::test_support::rand_sum;
    let sums: Vec<PauliSum<1>> = (0..16).map(|k| rand_sum::<1>(2000, 10, 0xF1A5 + k)).collect();
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

/// The bin `ApproxTopN` histograms `|c|²` into, its exponent and top two mantissa bits, derived here from the definition rather than from the implementation.
fn quarter_octave(c: Complex64) -> usize {
    (c.norm_sqr().to_bits() >> 50) as usize
}

/// Bins are quarter-octaves of `|c|²` numbered from the bottom: four per `f64` exponent, `1023 · 4` the bin of `[1, 1.25)`.
#[test]
fn approx_bin_is_the_exponent_and_two_mantissa_bits() {
    assert_eq!(APPROX_BINS, 2048 * 4);
    let cases = [
        (0.0, 0),
        (1.0, 4092),
        (1.2499, 4092),
        (1.25, 4093),
        (1.5, 4094),
        (1.99, 4095),
        (2.0, 4096),
        (0.75, 4090),
        (f64::INFINITY, 8188),
    ];
    for (norm_sqr, bin) in cases {
        assert_eq!(approx_bin(norm_sqr), bin, "|c|² = {norm_sqr}");
    }
    assert_eq!(
        approx_bin(-f64::NAN),
        APPROX_BINS - 1,
        "a NaN with its sign bit shares the top bin"
    );
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
/// `kept <= n`, and `kept > n - p` where `p` is the population of the highest *excluded* bin. Equivalently `kept + p > n`: the next bin down would have overshot.
#[test]
fn approx_top_n_shortfall_is_bounded_by_one_bin() {
    let input = crate::test_support::rand_sum::<1>(3000, 10, 0xB0117);
    let len = input.len();
    for n in [1usize, 17, 200, 900, 2000, len - 1] {
        let mut sum = input.clone();
        ApproxTopN(n).finalize_layer(&mut sum);
        let kept = sum.len();
        assert!(kept <= n, "n={n}: kept {kept} exceeds the bound");

        // The highest excluded bin's population in the input is the slack.
        let survivors: std::collections::HashSet<(u64, u64)> =
            sum.iter().map(|(x, z, _)| (x[0], z[0])).collect();
        let dropped_bins: Vec<usize> = input
            .iter()
            .filter(|(x, z, _)| !survivors.contains(&(x[0], z[0])))
            .map(|(_, _, c)| quarter_octave(c))
            .collect();
        let p = match dropped_bins.iter().copied().max() {
            None => 0,
            Some(top) => input.iter().filter(|(_, _, c)| quarter_octave(*c) == top).count(),
        };
        assert!(
            kept + p > n,
            "n={n}: kept {kept} + excluded bin {p} must overshoot n, \
             else that bin should have been kept"
        );
    }
}

/// On multi-bucket tie-dense sums `ApproxTopN(n)` keeps exactly the stable-sort transcription of the rule, at every bucket count: whole octaves above the boundary, the first of the boundary octave in bucket order.
#[test]
fn approx_top_n_matches_the_stable_sort_rule_across_bucket_counts() {
    let input = crate::test_support::tie_heavy_sum::<1>(2000, 8, 0x7135);
    for bits in [0u8, 2, 5, 9] {
        let hashed = input.clone().with_hash(Gf2Hash::<1>::new(8, bits, 0x99));
        for n in [0usize, 1, 3, 250, 500, 700, 1200, 1900, input.len() - 1] {
            let mut sum = hashed.clone();
            ApproxTopN(n).finalize_layer(&mut sum);
            sum.assert_invariants();
            assert_eq!(sum.len(), n, "bits={bits} n={n}");
            let want = first_n_by_rank(&storage_triples(&hashed), n, octave_rank);
            assert_eq!(storage_triples(&sum), want, "bits={bits} n={n}");
        }
    }
}

/// A boundary inside one octave keeps the octave's upper quarter-bins whole and fills from the boundary bin in bucket order; an overfull top bin never empties the sum.
#[test]
fn approx_top_n_cuts_inside_an_octave_at_a_quarter_bin() {
    // |c|² 4 and 4.84 share the bin [4, 5) of the octave [4, 8), 5.76 is in [5, 6), 6.76 in [6, 7), and 0.25 lies below.
    let magnitudes = [2.0f64, 2.2, 0.5, 2.4, 2.6];
    let mut sum = sum_of_mags(&magnitudes);
    ApproxTopN(3).finalize_layer(&mut sum);
    sum.assert_invariants();
    assert_eq!(kept_mags(&sum), vec![2.0, 2.4, 2.6]);
    // |c|² 4.0, 4.41 and 4.84 all lie in [4, 5).
    let mut sum = sum_of_mags(&[2.0, 2.1, 2.2]);
    ApproxTopN(2).finalize_layer(&mut sum);
    assert_eq!(kept_mags(&sum), vec![2.0, 2.1]);
}

/// The boundary bin is a quarter of the octave: `n = 5` keeps 2.0, both terms of `[1.75, 2)`, 1.6 and one of `[1.25, 1.5)`, where an octave cut would have kept the first four of `[1, 2)` in bucket order.
#[test]
fn approx_top_n_keeps_every_bin_above_a_quarter_octave_boundary() {
    use crate::test_support::{sum_of_norm_sqrs, QUARTER_BIN_NORM_SQRS};
    let input = sum_of_norm_sqrs(&QUARTER_BIN_NORM_SQRS, 0);
    let kept_norm_sqrs = |n: usize| -> Vec<f64> {
        let mut sum = input.clone();
        ApproxTopN(n).finalize_layer(&mut sum);
        sum.assert_invariants();
        sum.iter().map(|(_, _, c)| (c.norm_sqr() * 1e9).round() / 1e9).collect()
    };
    assert_eq!(kept_norm_sqrs(5), vec![1.3, 1.6, 1.9, 1.8, 2.0]);
    assert_eq!(kept_norm_sqrs(8), vec![1.0, 1.1, 1.3, 1.4, 1.6, 1.9, 1.8, 2.0]);
    assert_eq!(kept_norm_sqrs(9), vec![1.0, 1.1, 1.3, 1.2, 1.4, 1.6, 1.9, 1.8, 2.0]);
}

/// The kept part of the boundary bin is not taken from either edge: over a uniform `|c|²`, the kept boundary terms' mean equals the dropped ones' to a hundredth of the bin, and every dropped term is within one quarter-octave of every kept one.
#[test]
fn approx_top_n_keeps_a_centred_sample_of_the_boundary_bin() {
    let input = crate::test_support::one_octave_sum::<1>(20_000, 64, 0xCE47);
    let n = 7_500;
    let mut sum = input.clone();
    ApproxTopN(n).finalize_layer(&mut sum);
    assert_eq!(sum.len(), n);
    let kept: std::collections::HashSet<(u64, u64)> =
        sum.iter().map(|(x, z, _)| (x[0], z[0])).collect();
    let (mut at_kept, mut at_dropped) = (Vec::new(), Vec::new());
    let (mut lowest_kept, mut highest_dropped) = (f64::INFINITY, 0.0f64);
    for (x, z, c) in input.iter() {
        let norm_sqr = c.norm_sqr();
        let stays = kept.contains(&(x[0], z[0]));
        if stays {
            lowest_kept = lowest_kept.min(norm_sqr);
        } else {
            highest_dropped = highest_dropped.max(norm_sqr);
        }
        if (1.5..1.75).contains(&norm_sqr) {
            if stays { &mut at_kept } else { &mut at_dropped }.push(norm_sqr);
        }
    }
    assert!(lowest_kept >= 1.5 && highest_dropped < 1.75, "the cut is the bin [1.5, 1.75)");
    assert!(at_kept.len() > 1_000 && at_dropped.len() > 1_000, "the quota splits the bin");
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let offset = (mean(&at_kept) - mean(&at_dropped)) / 0.25;
    assert!(offset.abs() < 0.01, "kept minus dropped boundary mean is {offset} of the bin");
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
use crate::test_support::{first_n_by_rank, magnitude_rank, octave_rank, storage_triples};

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

/// A split hands each bucket what is left of the quota after the buckets before it, on sparse and dense counts across the chunk edges.
#[test]
fn a_quota_split_hands_out_the_quota_in_bucket_order() {
    let mut rng = crate::test_support::Xs64::new(0x5917);
    for buckets in [0usize, 1, 5, 1023, 1024, 1025, 4100] {
        for spread in [1u64, 2, 9] {
            let counts: Vec<usize> =
                (0..buckets).map(|_| (rng.next_u64() % (spread + 1)) as usize).collect();
            let total: usize = counts.iter().sum();
            for quota in [0, 1, total / 3, total / 2, total.saturating_sub(1), total, total + 5] {
                let split = QuotaSplit::new(|b| counts[b], buckets, total, quota);
                let label = format!("buckets={buckets} spread={spread} quota={quota}");
                let mut remaining = quota;
                for (b, &count) in counts.iter().enumerate() {
                    let want = count.min(remaining);
                    remaining -= want;
                    assert_eq!(split.quota(b).min(count), want, "{label}: bucket {b}");
                }
            }
        }
    }
}

/// Each rank's [`local_boundary_quota`] on an in-process group, `counts[r]` boundary terms on rank `r`.
fn boundary_quotas(counts: &[usize], quota: usize) -> Vec<usize> {
    crate::engine::partitioned::in_process::on_every_rank(counts.len() as u32, |transport| {
        local_boundary_quota(transport, counts[transport.rank() as usize], quota)
    })
}

/// Hand-computed splits: floors of `quota · count / total`, the remainder to the largest fractions, ties to the lower rank.
#[test]
fn a_boundary_quota_splits_across_ranks_in_proportion_to_their_counts() {
    let cases: [(&[usize], usize, &[usize]); 10] = [
        (&[1, 3], 1, &[0, 1]),
        (&[3, 1], 1, &[1, 0]),
        (&[2, 3, 5], 7, &[1, 2, 4]),
        (&[5, 3, 2], 7, &[4, 2, 1]),
        (&[10, 30, 0, 60], 50, &[5, 15, 0, 30]),
        (&[1, 1, 1, 1], 2, &[1, 1, 0, 0]),
        (&[0, 0, 4, 0], 3, &[0, 0, 3, 0]),
        (&[7, 9], 0, &[0, 0]),
        (&[7, 9], 16, &[7, 9]),
        (&[400, 100, 300, 200], 999, &[399, 100, 300, 200]),
    ];
    for (counts, quota, want) in cases {
        assert_eq!(boundary_quotas(counts, quota), want, "counts {counts:?} quota {quota}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Every rank's share is at most its count and within one of its proportional share, and the shares add up to the quota.
    #[test]
    fn a_boundary_quota_split_is_exact_and_proportional(
        counts in prop::collection::vec(0usize..1_000, 1..=4),
        fraction in 0.0f64..=1.0,
    ) {
        let total: usize = counts.iter().sum();
        let quota = (fraction * total as f64) as usize;
        let shares = boundary_quotas(&counts, quota);
        prop_assert_eq!(shares.iter().sum::<usize>(), quota);
        for (&share, &count) in shares.iter().zip(&counts) {
            prop_assert!(share <= count);
            let deviation = (share * total) as i64 - (count * quota) as i64;
            prop_assert!(deviation.unsigned_abs() < total.max(1) as u64, "{shares:?} of {counts:?} at {quota}");
        }
    }
}
