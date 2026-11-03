use super::{raw_outputs, IdentityChannel, HEAVY_HEX_127_EDGES};
use crate::channel::Channel;
use crate::pauli_string::PauliString;
use num_complex::Complex64;

#[test]
fn identity_channel_has_empty_support() {
    assert_eq!(<IdentityChannel as Channel<1>>::support(&IdentityChannel), [0u64; 1]);
    assert_eq!(<IdentityChannel as Channel<2>>::support(&IdentityChannel), [0u64; 2]);
}

#[test]
fn identity_channel_emits_its_input_unchanged() {
    let mut input = PauliString::<2>::x(0);
    let _ = input.mul_assign(&PauliString::<2>::y(1));
    let _ = input.mul_assign(&PauliString::<2>::z(70));
    let coeff = Complex64::new(2.0, 3.0);
    for adjoint in [false, true] {
        assert_eq!(raw_outputs(&IdentityChannel, adjoint, input, coeff), vec![(input, coeff)]);
    }
}

/// Pins the transcribed copy: 144 undirected edges over qubits `0..126`, sorted and unique as `(lo, hi)`, degree histogram 2 × 1, 89 × 2, 36 × 3.
#[test]
fn heavy_hex_127_edges_match_the_source_lattice() {
    assert_eq!(HEAVY_HEX_127_EDGES.len(), 144);
    let mut degree = [0usize; 127];
    let mut prev = (0usize, 0usize);
    for (i, &(a, b)) in HEAVY_HEX_127_EDGES.iter().enumerate() {
        assert!(a < b, "edge {i} is not (lo, hi): ({a}, {b})");
        assert!(b < 127, "edge {i} names qubit {b} outside 0..126");
        if i > 0 {
            assert!(prev < (a, b), "edge {i} breaks the sorted-unique order");
        }
        prev = (a, b);
        degree[a] += 1;
        degree[b] += 1;
    }
    let mut histogram = [0usize; 4];
    for d in degree {
        histogram[d] += 1;
    }
    assert_eq!(histogram, [0, 2, 89, 36]);
}

mod runs_agree {
    use crate::pauli_string::PauliString;
    use crate::pauli_sum::accumulator::BuildAccumulator;
    use crate::pauli_sum::PauliSum;
    use crate::test_support::{assert_runs_agree, assert_runs_agree_count_only, DiscardMeter};
    use crate::truncation::{And, ApproxTopN, CoefficientThreshold, TopN};
    use num_complex::Complex64;

    /// Term `k` is `X` on the bits of `k + 1`, so every index is its own key.
    fn sum_of(terms: &[(usize, f64)]) -> PauliSum<1> {
        let mut accumulator = BuildAccumulator::<1>::new(16);
        for &(k, magnitude) in terms {
            let mut p = PauliString::<1>::identity();
            p.x[0] = k as u64 + 1;
            accumulator.add_term(p, Complex64::from_polar(magnitude, k as f64));
        }
        accumulator.finalize()
    }

    /// Ten distinct magnitudes from 8 down, then ten tied at 0.6, then ten from 0.1 down.
    fn input() -> Vec<(usize, f64)> {
        let above = (0..10).map(|k| (k, 8.0 - k as f64 * 0.5));
        let ties = (10..20).map(|k| (k, 0.6));
        let below = (20..30).map(|k| (k, 0.1 - (k - 20) as f64 * 0.005));
        above.chain(ties).chain(below).collect()
    }

    fn kept(indices: impl IntoIterator<Item = usize>) -> PauliSum<1> {
        let input = input();
        sum_of(&indices.into_iter().map(|k| input[k]).collect::<Vec<_>>())
    }

    /// `TopN(15)` of [`input`], and the norm its cut dropped.
    fn want() -> (PauliSum<1>, f64) {
        let mut sum = sum_of(&input());
        let meter = DiscardMeter::new(&TopN(15));
        crate::TruncationPolicy::<1>::finalize_layer(&meter, &mut sum);
        assert_eq!(sum.len(), 15);
        (sum, meter.discarded())
    }

    #[test]
    fn the_meter_reports_the_dropped_norm() {
        let dropped: f64 = input()[15..].iter().map(|(_, m)| m * m).sum();
        assert!((want().1 - dropped.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn a_different_choice_of_boundary_ties_passes() {
        let (want, discarded) = want();
        let got = kept((0..10).chain(15..20));
        assert_runs_agree(TopN(15), discarded, &got, &want, 1e-12, "ties");
        let policy = And(CoefficientThreshold(1e-9), ApproxTopN(15));
        assert_runs_agree(policy, discarded, &got, &want, 1e-12, "octave");
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn a_sign_flipped_on_a_large_term_fails() {
        let (want, discarded) = want();
        let mut terms: Vec<_> = input()[..15].to_vec();
        terms[0].1 = -8.0;
        terms[14] = (19, 0.6);
        let got = sum_of(&terms);
        assert_runs_agree(TopN(15), discarded, &got, &want, 1e-12, "sign");
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn a_large_term_dropped_fails() {
        let (want, discarded) = want();
        let got = kept((1..10).chain(10..16));
        assert_runs_agree(TopN(15), discarded, &got, &want, 1e-12, "dropped");
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn a_wrong_boundary_coefficient_fails() {
        let (want, discarded) = want();
        let mut terms: Vec<_> = input()[..10].to_vec();
        terms.extend((15..20).map(|k| (k, 0.6)));
        terms[14].1 = 5.6;
        let got = sum_of(&terms);
        assert_runs_agree(TopN(15), discarded, &got, &want, 1e-12, "boundary");
    }

    #[test]
    #[should_panic(expected = "term count")]
    fn a_different_count_fails() {
        let (want, discarded) = want();
        assert_runs_agree(TopN(15), discarded, &kept(0..14), &want, 1e-12, "count");
    }

    #[test]
    #[should_panic(expected = "too heavy")]
    fn a_too_heavy_cut_fails() {
        let (want, _) = want();
        let got = kept((0..10).chain(15..20));
        assert_runs_agree(TopN(15), 10.0, &got, &want, 1e-12, "heavy");
    }

    #[test]
    #[should_panic(expected = "key mismatch")]
    fn different_keys_without_a_layer_pass_fail() {
        let (want, discarded) = want();
        let got = kept((0..10).chain(15..20));
        let policy = CoefficientThreshold(1e-9);
        assert_runs_agree(policy, discarded, &got, &want, 1e-12, "no cut");
    }

    #[test]
    fn a_heavy_cut_compares_counts_only() {
        let (want, _) = want();
        let got = kept((0..10).chain(15..20));
        assert_runs_agree_count_only(TopN(15), 10.0, &got, &want, "heavy");
    }

    /// `want = {k0: 1, k1: 0.5}` against `got = {k0: 1, k2: 0.5}`: `1 − cos = 0.25 / 1.25 = 0.2`, and the bound `2Δ²/1.25` reaches it at `Δ = √0.125`, below the sign-flip limit `Δ < 1`.
    fn swapped_small_term() -> (PauliSum<1>, PauliSum<1>) {
        (sum_of(&[(0, 1.0), (2, 0.5)]), sum_of(&[(0, 1.0), (1, 0.5)]))
    }

    #[test]
    fn a_distance_just_inside_the_bound_passes() {
        let (got, want) = swapped_small_term();
        let discarded = 1.01 * 0.125f64.sqrt();
        assert_runs_agree(TopN(2), discarded, &got, &want, 0.0, "inside");
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn a_distance_just_outside_the_bound_fails() {
        let (got, want) = swapped_small_term();
        let discarded = 0.99 * 0.125f64.sqrt();
        assert_runs_agree(TopN(2), discarded, &got, &want, 0.0, "outside");
    }

    #[test]
    #[should_panic(expected = "non-finite")]
    fn a_count_only_check_rejects_a_nan() {
        let (want, _) = want();
        let mut terms: Vec<_> = input()[..15].to_vec();
        terms[14].1 = f64::NAN;
        assert_runs_agree_count_only(TopN(15), 10.0, &sum_of(&terms), &want, "nan");
    }

    #[test]
    #[should_panic(expected = "light enough")]
    fn a_light_cut_is_refused_by_the_count_only_check() {
        let (want, discarded) = want();
        let got = kept((0..10).chain(15..20));
        assert_runs_agree_count_only(TopN(15), discarded, &got, &want, "light");
    }

    #[test]
    #[should_panic(expected = "the last layer must cut")]
    fn a_count_below_n_fails() {
        let (want, _) = want();
        let got = kept((0..10).chain(15..20));
        assert_runs_agree_count_only(TopN(16), 10.0, &got, &want, "below n");
    }

    #[test]
    #[should_panic(expected = "bare TopN or ApproxTopN")]
    fn a_count_only_check_needs_a_bare_cut() {
        let (want, _) = want();
        let got = kept((0..10).chain(15..20));
        let policy = And(CoefficientThreshold(1e-9), ApproxTopN(15));
        assert_runs_agree_count_only(policy, 10.0, &got, &want, "and");
    }
}

mod structural_diff {
    use crate::pauli_string::PauliString;
    use crate::pauli_sum::accumulator::BuildAccumulator;
    use crate::pauli_sum::PauliSum;
    use crate::test_support::{bucket_len, structural_diff, with_hash, StructuralDiff};
    use crate::Gf2Hash;
    use num_complex::Complex64;

    /// One bucket holding each `(k, c)` as the key whose last x-word is `k`, so keys order by `k` at every `W`.
    fn sum_of<const W: usize>(terms: &[(u64, f64)]) -> PauliSum<W> {
        let mut accumulator = BuildAccumulator::<W>::new(64 * W);
        for &(k, c) in terms {
            let mut p = PauliString::<W>::identity();
            p.x[W - 1] = k;
            accumulator.add_term(p, Complex64::new(c, 0.0));
        }
        accumulator.finalize()
    }

    const BASE: [(u64, f64); 4] = [(1, 1.0), (2, 2.0), (4, 3.0), (8, 4.0)];

    type Terms = &'static [(u64, f64)];

    /// `(after, [terms_after, changed, inserted, deleted, shift])` against `BASE`, shift counted by hand from the first index where the key sequences differ.
    const CASES: [(Terms, [usize; 5]); 8] = [
        (&BASE, [4, 0, 0, 0, 0]),
        (&[(1, 1.0), (2, -2.0), (4, 3.0), (8, 0.5)], [4, 2, 0, 0, 0]),
        (&[(1, 1.0), (2, 2.0), (3, 9.0), (4, 3.0), (8, 4.0)], [5, 0, 1, 0, 3]),
        (&[(1, 1.0), (4, 3.0), (8, 4.0)], [3, 0, 0, 1, 2]),
        (&[(0, 7.0), (1, 1.0), (2, 2.0), (4, 3.0), (8, 4.0)], [5, 0, 1, 0, 5]),
        (&[(1, 1.0), (2, 2.0), (4, 3.0), (8, 4.0), (16, 5.0)], [5, 0, 1, 0, 1]),
        (&[(1, 1.0), (2, 2.0), (4, 3.0)], [3, 0, 0, 1, 0]),
        (&[(1, 1.0), (3, 2.0), (4, 3.0), (8, -4.0)], [4, 1, 1, 1, 3]),
    ];

    fn one_bucket<const W: usize>() {
        for (after, [terms_after, changed, inserted, deleted, shift]) in CASES {
            let want = StructuralDiff { terms: 4, terms_after, changed, inserted, deleted, shift };
            assert_eq!(
                structural_diff(&sum_of::<W>(&BASE), &sum_of::<W>(after)),
                want,
                "{after:?}"
            );
        }
    }

    /// A key below every other in one of two buckets shifts exactly that bucket.
    fn shift_sums_over_buckets<const W: usize>() {
        let keys: Vec<(u64, f64)> = (1..=16).map(|k| (k << 1, k as f64)).collect();
        let hash = Gf2Hash::<W>::new(64 * W, 1, 0x5EED);
        let before = with_hash(sum_of::<W>(&keys), hash.clone());
        let after = with_hash(sum_of::<W>(&[keys.as_slice(), &[(1, 0.25)]].concat()), hash.clone());
        let mut p = PauliString::<W>::identity();
        p.x[W - 1] = 1;
        let shift = bucket_len(&after, hash.bucket_of(&p.x, &p.z) as usize);
        let want =
            StructuralDiff { terms: 16, terms_after: 17, inserted: 1, shift, ..Default::default() };
        assert_eq!(structural_diff(&before, &after), want);
    }

    #[test]
    fn hand_counted_diffs() {
        one_bucket::<1>();
        one_bucket::<2>();
        shift_sums_over_buckets::<1>();
        shift_sums_over_buckets::<2>();
    }

    #[test]
    #[should_panic(expected = "one bucket hash")]
    fn sums_under_different_hashes_are_refused() {
        let before = sum_of::<1>(&BASE);
        let after = with_hash(sum_of::<1>(&BASE), Gf2Hash::<1>::new(64, 1, 0x5EED));
        let _ = structural_diff(&before, &after);
    }
}
