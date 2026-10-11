use super::{raw_outputs, IdentityChannel, HEAVY_HEX_127_EDGES};
use crate::channel::Channel;
use crate::pauli_string::PauliString;
use num_complex::Complex64;

#[test]
fn identity_channel_has_empty_support() {
    assert_eq!(
        <IdentityChannel as Channel<1>>::support(&IdentityChannel),
        [0u64; 1]
    );
    assert_eq!(
        <IdentityChannel as Channel<2>>::support(&IdentityChannel),
        [0u64; 2]
    );
}

#[test]
fn identity_channel_emits_its_input_unchanged() {
    let mut input = PauliString::<2>::x(0);
    let _ = input.mul_assign(&PauliString::<2>::y(1));
    let _ = input.mul_assign(&PauliString::<2>::z(70));
    let coeff = Complex64::new(2.0, 3.0);
    for adjoint in [false, true] {
        assert_eq!(
            raw_outputs(&IdentityChannel, adjoint, input, coeff),
            vec![(input, coeff)]
        );
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
    use crate::test_support::assert_runs_agree;
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

    /// `TopN(15)` with the five boundary slots filled from different ties.
    #[test]
    fn top_n_accepts_different_ties() {
        let want = kept((0..10).chain(10..15));
        let got = kept((0..10).chain(15..20));
        assert_runs_agree(TopN(15), 1, &got, &want, 1e-12, "ties");
    }

    #[test]
    #[should_panic(expected = "smallest kept")]
    fn top_n_rejects_dropping_the_smallest_kept() {
        let want = kept((0..10).chain(10..15));
        let got = kept((0..10).chain(10..13).chain([20, 21]));
        assert_runs_agree(TopN(15), 1, &got, &want, 1e-12, "dropped ties");
    }

    #[test]
    #[should_panic(expected = "above the boundary")]
    fn top_n_rejects_swapping_a_term_above_the_boundary() {
        let want = kept((0..10).chain(10..15));
        let mut terms = input()[1..15].to_vec();
        terms.push((30, 8.0));
        let got = sum_of(&terms);
        assert_runs_agree(TopN(15), 1, &got, &want, 1e-12, "swapped top");
    }

    /// The ties and the `0.1 .. 0.055` tail: `ApproxTopN(15)` cuts in the octave `[0.25, 0.5)` of `|c|²`.
    #[test]
    fn approx_top_n_accepts_different_boundary_octave_terms() {
        let want = kept((0..10).chain(10..15));
        let got = kept((0..10).chain(15..20));
        let policy = And(CoefficientThreshold(1e-9), ApproxTopN(15));
        assert_runs_agree(policy, 1, &got, &want, 1e-12, "octave");
    }

    #[test]
    #[should_panic(expected = "smallest kept")]
    fn approx_top_n_rejects_dropping_the_boundary_octave() {
        let want = kept((0..10).chain(10..15));
        let got = kept((0..10).chain(20..25));
        assert_runs_agree(ApproxTopN(15), 1, &got, &want, 1e-12, "dropped octave");
    }

    #[test]
    #[should_panic(expected = "above the boundary")]
    fn approx_top_n_rejects_swapping_a_term_above_the_boundary_octave() {
        let want = kept((0..10).chain(10..15));
        let got = kept((0..9).chain(10..16));
        assert_runs_agree(ApproxTopN(15), 1, &got, &want, 1e-12, "swapped top");
    }

    /// After two cuts the survivors evolve apart, so only the count is compared.
    #[test]
    fn a_repeated_cut_compares_the_count() {
        let want = kept((0..10).chain(10..15));
        let got = kept((0..10).chain(20..25));
        assert_runs_agree(TopN(15), 2, &got, &want, 1e-12, "two layers");
    }
}
