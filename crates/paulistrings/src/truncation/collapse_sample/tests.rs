use super::*;
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::Gf2Hash;
use crate::test_support::{
    assert_systematic_resample, assert_terms_close, coefficient_tolerance, rand_sum, rand_sum_real,
    weighted_four_term_sum,
};
use num_complex::Complex64;

/// The largest `u` [`Rng::uniform`] returns.
const LAST_UNIFORM: f64 = 1.0 - 1.0 / (1u64 << 53) as f64;

/// `X`-strings with `x[0] = 1, 2, 3, …` carrying `coefficients` in that (iteration) order; a zero coefficient is kept as a term.
fn ordered_sum<const W: usize, C: Coefficient>(coefficients: &[C]) -> PauliSum<W, C> {
    let key = |i: usize| {
        let mut p = PauliString::<W>::identity();
        p.x[0] = i as u64 + 1;
        p
    };
    let mut acc = BuildAccumulator::<W, C>::new(8);
    for i in 0..coefficients.len() {
        acc.add_term(key(i), C::ONE);
    }
    let mut sum = acc.finalize();
    sum.update_per_bucket(|_| {
        |x: &[u64; W], _: &[u64; W], _: C| Some(coefficients[x[0] as usize - 1])
    });
    let order: Vec<u64> = sum.iter().map(|(x, _, _)| x[0]).collect();
    assert_eq!(order, (1..=coefficients.len() as u64).collect::<Vec<_>>(), "iteration order");
    sum
}

/// `sum` after one systematic resample at offset `u` with `samples` points.
fn resampled<const W: usize, C: Coefficient>(
    sum: &PauliSum<W, C>,
    u: f64,
    samples: usize,
) -> PauliSum<W, C> {
    let mut out = sum.clone();
    let norms = bucket_l1_norms(&out);
    let total = running_sum(norms.iter().copied());
    systematic_resample(&mut out, &norms, 0.0, SystematicPoints { u, samples, total });
    out.assert_invariants();
    out
}

/// The coefficients of `sum` by `x[0]`, the [`ordered_sum`] index plus one.
fn by_index<const W: usize>(sum: &PauliSum<W, f64>) -> Vec<(u64, f64)> {
    let mut out: Vec<(u64, f64)> = sum.iter().map(|(x, _, c)| (x[0], c)).collect();
    out.sort_by_key(|&(i, _)| i);
    out
}

/// `|c| = [1, 0, 2.5, 0.5]` with signs `[+, ·, −, +]`: `‖O‖₁ = 4`, so four points sit `h = 1` apart and the stretches are `[0, 1)`, `[1, 1)`, `[1, 3.5)`, `[3.5, 4)`.
fn check_hand_picks<const W: usize>() {
    let sum = ordered_sum::<W, f64>(&[1.0, 0.0, -2.5, 0.5]);
    // Points 0.25, 1.25, 2.25, 3.25: three land in the third stretch and merge into one term worth 3h.
    assert_eq!(by_index(&resampled(&sum, 0.25, 4)), [(1, 1.0), (3, -3.0)]);
    // Points 0, 1, 2, 3: the point at 1 skips the empty stretch for the later one.
    assert_eq!(by_index(&resampled(&sum, 0.0, 4)), [(1, 1.0), (3, -3.0)]);
    // Points 0.5, 1.5, 2.5, 3.5: the point at 3.5 belongs to the fourth stretch.
    assert_eq!(by_index(&resampled(&sum, 0.5, 4)), [(1, 1.0), (3, -2.0), (4, 1.0)]);
    // Two points at 0 and 2, worth h = 2 each.
    assert_eq!(by_index(&resampled(&sum, 0.0, 2)), [(1, 2.0), (3, -2.0)]);
    // Eight points at 0.25, 0.75, …: every |c| is a whole number of h = 0.5, so this draw returns the sum.
    assert_eq!(by_index(&resampled(&sum, 0.5, 8)), [(1, 1.0), (3, -2.5), (4, 0.5)]);
}

#[test]
fn systematic_picks_match_hand_values() {
    check_hand_picks::<1>();
    check_hand_picks::<2>();
}

/// `|c| = [1, 2, 0]`, three points at the largest `u`: `(u + 2)·1` rounds to `3 = ‖O‖₁`, which belongs to the second term, the last with a nonzero stretch, never to the trailing zero.
fn check_a_point_rounded_onto_the_total<const W: usize>() {
    assert_eq!(LAST_UNIFORM + 2.0, 3.0, "the third point must round onto the total");
    let sum = ordered_sum::<W, f64>(&[1.0, -2.0, 0.0]);
    assert_eq!(by_index(&resampled(&sum, LAST_UNIFORM, 3)), [(1, 1.0), (2, -2.0)]);
}

#[test]
fn a_point_rounded_onto_the_total_goes_to_the_last_nonzero_stretch() {
    check_a_point_rounded_onto_the_total::<1>();
    check_a_point_rounded_onto_the_total::<2>();
}

/// Over many buckets the walk agrees with a search per point over the flattened sum, and every coefficient keeps its sign or phase.
fn check_against_the_reference<const W: usize, C: Coefficient>(input: PauliSum<W, C>) {
    let input = input.with_hash(Gf2Hash::new(16, 5, 0x1234));
    let used = (0..input.num_buckets()).filter(|&b| input.bucket_len(b) > 0).count();
    assert!(used > 16, "the fixture must span many buckets");
    let terms: Vec<_> = input.iter().map(|(x, z, c)| (*x, *z, c)).collect();
    for (seed, samples) in [(1u64, 1usize), (2, 7), (3, 300), (4, 2000), (5, 5000)] {
        let u = Rng::from_key(&[seed]).uniform();
        let got = resampled(&input, u, samples);
        let what = format!("{} W={W} samples={samples}", C::NAME);
        assert_systematic_resample(&got, &terms, u, samples, coefficient_tolerance::<C>(), &what);
    }
}

#[test]
fn bucketed_resample_matches_a_search_per_point_and_keeps_every_phase() {
    check_against_the_reference::<1, Complex64>(rand_sum::<1>(3000, 16, 0xC011));
    check_against_the_reference::<2, Complex64>(rand_sum::<2>(3000, 16, 0xC012));
    let real = |sum: PauliSum<1>| sum.cast::<f64>(1e-12).expect("real coefficients");
    check_against_the_reference::<1, f64>(real(rand_sum_real::<1>(3000, 16, 0xC013)));
    let real = |sum: PauliSum<2>| sum.cast::<f64>(1e-12).expect("real coefficients");
    check_against_the_reference::<2, f64>(real(rand_sum_real::<2>(3000, 16, 0xC014)));
    let narrow = rand_sum::<1>(3000, 16, 0xC015).cast::<num_complex::Complex32>(0.0).unwrap();
    check_against_the_reference::<1, num_complex::Complex32>(narrow);
}

/// The mean of 4000 collapses of a four-term sum to two points is the sum, to within five standard errors in each part of each coefficient.
#[test]
fn a_collapse_is_unbiased() {
    let input = weighted_four_term_sum::<1>(8);
    let trials = 4000u64;
    let mut moments = std::collections::HashMap::<[u64; 2], [f64; 4]>::new();
    for seed in 0..trials {
        let mut sum = input.clone();
        CollapseSample::new(3, 2, seed).finalize_layer(&mut sum);
        assert!(sum.len() <= 2);
        for (x, z, c) in sum.iter() {
            let entry = moments.entry([x[0], z[0]]).or_default();
            *entry =
                [entry[0] + c.re, entry[1] + c.re * c.re, entry[2] + c.im, entry[3] + c.im * c.im];
        }
    }
    let n = trials as f64;
    for (x, z, c) in input.iter() {
        let [re, re2, im, im2] = moments[&[x[0], z[0]]];
        for (sum, squares, want, part) in [(re, re2, c.re, "re"), (im, im2, c.im, "im")] {
            let mean = sum / n;
            let error = ((squares / n - mean * mean) / n).sqrt();
            assert!(
                (mean - want).abs() < 5.0 * error.max(1e-12),
                "{x:?}/{z:?} {part}: mean {mean} vs {want}, standard error {error}",
            );
        }
    }
}

#[test]
fn collapse_sample_is_a_no_op_up_to_the_cache() {
    let input = weighted_four_term_sum::<1>(8);
    for cache in [4usize, 5, 100] {
        let policy = CollapseSample::new(cache, 1, 7);
        let mut sum = input.clone();
        policy.finalize_layer(&mut sum);
        assert_eq!(sum.to_arrays(), input.to_arrays(), "cache {cache}");
        assert_eq!(policy.collapses(), 0);
    }
    assert!(<_ as TruncationPolicy<1>>::finalizes_layer(&CollapseSample::new(4, 1, 7)));
}

/// Pass `k` draws at `u = uniform(k)`, counting every pass and every collapse.
#[test]
fn each_pass_draws_at_its_own_offset() {
    let input = rand_sum::<1>(400, 16, 0xD1CE).with_hash(Gf2Hash::new(16, 3, 0x77));
    let terms: Vec<_> = input.iter().map(|(x, z, c)| (*x, *z, c)).collect();
    let policy = CollapseSample::new(100, 50, 11);
    let mut distinct = std::collections::HashSet::new();
    for pass in 0..6u64 {
        let mut sum = input.clone();
        policy.finalize_layer(&mut sum);
        let u = CollapseSample::new(0, 1, 11).uniform(pass);
        assert_systematic_resample(&sum, &terms, u, 50, 1e-12, &format!("pass {pass}"));
        assert_eq!(policy.collapses(), pass + 1);
        distinct.insert(format!("{:?}", sum.to_arrays().0));
    }
    assert!(distinct.len() > 1, "passes must draw afresh");
}

/// A trajectory repeats from its seed whatever the thread count.
#[test]
fn collapse_sample_draw_is_a_function_of_the_seed() {
    let input = rand_sum::<1>(4000, 16, 0xC011).with_hash(Gf2Hash::new(16, 5, 0x1234));
    let run = |threads: usize, seed: u64| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let policy = CollapseSample::new(100, 64, seed);
            (0..3)
                .map(|_| {
                    let mut sum = input.clone();
                    policy.finalize_layer(&mut sum);
                    sum
                })
                .collect::<Vec<_>>()
        })
    };
    for seed in 0..4u64 {
        let one = run(1, seed);
        for (other, what) in [(run(8, seed), "eight threads"), (run(1, seed), "repeat")] {
            for (pass, (a, b)) in one.iter().zip(&other).enumerate() {
                assert_terms_close(b, a, 1e-12, &format!("seed {seed} pass {pass} {what}"));
            }
        }
    }
}

#[test]
#[should_panic(expected = "cannot draw by |c|")]
fn collapse_sample_rejects_an_all_zero_sum() {
    let mut sum = ordered_sum::<1, f64>(&[0.0, 0.0]);
    CollapseSample::new(1, 1, 0).finalize_layer(&mut sum);
}

#[test]
#[should_panic(expected = "samples must be at least 1")]
fn collapse_sample_rejects_zero_samples() {
    let _ = CollapseSample::new(1, 0, 0);
}

/// A lone partition checks the cache once per run of local layers within `RUN_GROWTH × cache`: the collapsing circuit, whose rotations run, takes fewer passes than layers, and the last pass still leaves at most `cache` terms.
#[test]
fn a_run_defers_the_collapse_check_to_its_end() {
    use crate::test_support::{collapsing_circuit, z0_sum};
    let circuit = collapsing_circuit();
    let policy = CollapseSample::new(6, 4, 3);
    let out = crate::propagate(&circuit, z0_sum(), &policy, crate::Direction::Heisenberg);
    let passes = policy.calls.load(Ordering::Relaxed);
    assert!(passes < circuit.channels.len() as u64, "{passes} passes");
    assert!(policy.collapses() >= 1);
    assert!(out.len() <= 6, "{} terms", out.len());
    assert_eq!(<CollapseSample as TruncationPolicy<1>>::run_length_limit(&policy), 48);
}
