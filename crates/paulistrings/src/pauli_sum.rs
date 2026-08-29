//! [`PauliSum<W>`] — weighted sum of Pauli strings in structure-of-arrays form.
//!
//! Storage is per-bucket parallel `Vec<[u64; W]>` columns for the `x` and `z`
//! parts plus a `Vec<Complex64>` for coefficients, partitioned by a
//! GF(2)-linear hash ([`Gf2Hash`](crate::bucket::Gf2Hash)). SoA is chosen so
//! coefficient-only and key-only scans get full cache utilization, and so each
//! `Vec` maps directly to a GPU device buffer.
//!
//! # Canonical order
//!
//! **Terms are ordered by (bucket index `h(x, z)` ascending, then
//! lexicographic `(x, z)` key within a bucket).** [`PauliSum::iter`] and
//! [`PauliSum::to_arrays`] produce exactly this order, and no two entries
//! share a key. Every public operation preserves the invariant or returns a
//! fresh [`PauliSum`] that does.
//!
//! A single-bucket sum's canonical order is plain lexicographic `(x, z)` —
//! `h ≡ 0` — and every sum of at most 1024 terms
//! ([`DEFAULT_TARGET_BUCKET_LEN`](crate::bucket::DEFAULT_TARGET_BUCKET_LEN))
//! is single-bucket when built through [`BuildAccumulator`], so small sums
//! always come out lex-sorted. Larger sums interleave their buckets in an
//! `H`-dependent order; compare them by key ([`PauliSum::get`],
//! [`PauliSum::iter`]) rather than by position.
//!
//! Build a [`PauliSum`] from unsorted inputs via [`BuildAccumulator`]; once
//! built, combine sums with [`PauliSum::add`] or scale coefficients with
//! [`PauliSum::scale`].
//!
//! See v0.1 design doc §3.2 and the v0.2 design doc §4 (storage), plus v0.3
//! §4 (the single bucketed representation).
//!
//! # Examples
//!
//! Construct the observable `Z₀ + 0.5·X₁` on two qubits via
//! [`BuildAccumulator`], then merge in a second sum.
//!
//! ```
//! use paulistrings::{BuildAccumulator, PauliString, PauliSum, Phase};
//! use num_complex::Complex64;
//!
//! let mut acc = BuildAccumulator::<1>::new(2);
//! acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
//! acc.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(0.5, 0.0));
//! let a = acc.finalize();
//! assert_eq!(a.len(), 2);
//!
//! let mut acc2 = BuildAccumulator::<1>::new(2);
//! acc2.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(-0.25, 0.0));
//! let b = acc2.finalize();
//!
//! let merged = a.add(&b);
//! assert_eq!(merged.len(), 2); // Z₀ + 0.25·X₁
//! ```
//!
//! [`BuildAccumulator`]: crate::BuildAccumulator

#![allow(unused)]

use num_complex::Complex64;

use crate::pauli_string::PauliString;

pub use crate::bucket::sum::PauliSum;

/// A uniform single-qubit product state, for
/// [`PauliSum::expectation_product_state`].
///
/// Each variant names the single-qubit Pauli whose `+1` eigenstate is taken on
/// every qubit. These are the states quench experiments actually start from, and
/// each one makes the expectation a masked scan rather than a simulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductState {
    /// `|+…+⟩`, the `+1` eigenstate of `X` on every qubit.
    XPlus,
    /// `|+i…+i⟩`, the `+1` eigenstate of `Y` on every qubit.
    YPlus,
    /// `|0…0⟩`, the `+1` eigenstate of `Z` on every qubit.
    ZPlus,
}

#[cfg(test)]
impl<const W: usize> PauliSum<W> {
    /// Test-only helper: build a `PauliSum<W>` from `(pauli_str, coeff)`
    /// pairs. Each `pauli_str` is a sequence of `I/X/Y/Z` characters where
    /// index `i` of the string corresponds to qubit `i`. `Y` characters
    /// fold one factor of `i` into the coefficient — the bitstring image
    /// of `Y_canonical` is `(x=1, z=1)` with an implicit `i` factor, so
    /// `Y_canonical = i · (x=1, z=1)`.
    ///
    /// `num_qubits` is taken from the length of the first string; all
    /// other strings must match. Routes through `BuildAccumulator`, so
    /// duplicate keys sum and exact-zero coefficients are dropped.
    pub(crate) fn from_strings(terms: &[(&str, Complex64)]) -> Self {
        use crate::phase::Phase;
        assert!(!terms.is_empty(), "from_strings requires at least one term");
        let num_qubits = terms[0].0.len();
        assert!(num_qubits <= 64 * W, "num_qubits must fit in W*64 bits");
        let mut acc = crate::accumulator::BuildAccumulator::<W>::new(num_qubits);
        for (s, c) in terms {
            assert_eq!(
                s.len(),
                num_qubits,
                "all pauli strings must have the same length",
            );
            let mut x = [0u64; W];
            let mut z = [0u64; W];
            let mut phase = Phase::ONE;
            for (i, ch) in s.chars().enumerate() {
                let word = i / 64;
                let bit = 1u64 << (i % 64);
                match ch {
                    'I' => {}
                    'X' => x[word] |= bit,
                    'Z' => z[word] |= bit,
                    'Y' => {
                        x[word] |= bit;
                        z[word] |= bit;
                        phase += Phase::I;
                    }
                    other => panic!("unexpected Pauli char {:?} (expected I/X/Y/Z)", other),
                }
            }
            let p = PauliString::<W> { x, z };
            acc.add_term(p, phase, *c);
        }
        acc.finalize()
    }
}

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;

    // ---- overlap / expectation (v0.2 B.10) ----
    //
    // Before this there was no way to get a *number* out of a propagated sum:
    // examples/ising_2d_quench.rs hand-rolled its own observable against the raw
    // SoA columns, a gap CLAUDE.md flagged.

    fn b10_build<const W: usize>(
        num_qubits: usize,
        terms: &[(PauliString<W>, Complex64)],
    ) -> PauliSum<W> {
        let mut acc =
            crate::accumulator::BuildAccumulator::<W>::with_capacity(num_qubits, terms.len());
        for &(pp, c) in terms {
            acc.add_term(pp, crate::phase::Phase::ONE, c);
        }
        acc.finalize()
    }

    #[test]
    fn overlap_with_self_is_the_squared_norm() {
        let a = b10_build::<1>(
            8,
            &[
                (PauliString::<1>::x(0), Complex64::new(2.0, 0.0)),
                (PauliString::<1>::z(3), Complex64::new(0.0, 3.0)),
            ],
        );
        assert!((a.overlap(&a) - Complex64::new(13.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn overlap_is_conjugate_symmetric() {
        let a = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(1.0, 2.0))]);
        let b = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(3.0, -1.0))]);
        let ab = a.overlap(&b);
        let ba = b.overlap(&a);
        assert!((ab - ba.conj()).norm() < 1e-12, "{ab} vs conj({ba})");
    }

    #[test]
    fn overlap_of_disjoint_supports_is_zero() {
        let a = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(1.0, 0.0))]);
        let b = b10_build::<1>(8, &[(PauliString::<1>::z(5), Complex64::new(1.0, 0.0))]);
        assert!(a.overlap(&b).norm() < 1e-12);
    }

    #[test]
    fn overlap_only_counts_shared_keys() {
        let a = b10_build::<1>(
            8,
            &[
                (PauliString::<1>::x(0), Complex64::new(2.0, 0.0)),
                (PauliString::<1>::y(1), Complex64::new(5.0, 0.0)),
            ],
        );
        let b = b10_build::<1>(
            8,
            &[
                (PauliString::<1>::x(0), Complex64::new(3.0, 0.0)),
                (PauliString::<1>::z(2), Complex64::new(7.0, 0.0)),
            ],
        );
        assert!((a.overlap(&b) - Complex64::new(6.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn overlap_handles_an_empty_operand() {
        let a = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(1.0, 0.0))]);
        let empty = PauliSum::<1>::empty(8);
        assert!(a.overlap(&empty).norm() < 1e-12);
        assert!(empty.overlap(&a).norm() < 1e-12);
    }

    #[test]
    fn overlap_across_a_word_boundary_w2() {
        let a = b10_build::<2>(
            128,
            &[
                (PauliString::<2>::x(3), Complex64::new(1.0, 0.0)),
                (PauliString::<2>::z(70), Complex64::new(2.0, 0.0)),
            ],
        );
        let b = b10_build::<2>(128, &[(PauliString::<2>::z(70), Complex64::new(4.0, 0.0))]);
        assert!((a.overlap(&b) - Complex64::new(8.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn identity_coefficient_picks_out_the_trace() {
        let a = b10_build::<1>(
            8,
            &[
                (PauliString::<1>::identity(), Complex64::new(1.5, 0.0)),
                (PauliString::<1>::x(0), Complex64::new(9.0, 0.0)),
            ],
        );
        assert!((a.identity_coefficient() - Complex64::new(1.5, 0.0)).norm() < 1e-12);
        let b = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(9.0, 0.0))]);
        assert!(b.identity_coefficient().norm() < 1e-12);
    }

    #[test]
    fn expectation_of_single_paulis_in_each_product_state() {
        let cases = [
            (PauliString::<1>::identity(), 1.0, 1.0, 1.0),
            (PauliString::<1>::x(0), 1.0, 0.0, 0.0),
            (PauliString::<1>::y(0), 0.0, 1.0, 0.0),
            (PauliString::<1>::z(0), 0.0, 0.0, 1.0),
        ];
        for (pp, ex, ey, ez) in cases {
            let s = b10_build::<1>(8, &[(pp, Complex64::new(1.0, 0.0))]);
            assert!(
                (s.expectation_product_state(ProductState::XPlus).re - ex).abs() < 1e-12,
                "XPlus for {pp:?}",
            );
            assert!(
                (s.expectation_product_state(ProductState::YPlus).re - ey).abs() < 1e-12,
                "YPlus for {pp:?}",
            );
            assert!(
                (s.expectation_product_state(ProductState::ZPlus).re - ez).abs() < 1e-12,
                "ZPlus for {pp:?}",
            );
        }
    }

    #[test]
    fn expectation_of_multi_qubit_products() {
        let mut xx = PauliString::<1>::x(0);
        xx.mul_assign(&PauliString::<1>::x(1));
        let mut xz = PauliString::<1>::x(0);
        xz.mul_assign(&PauliString::<1>::z(1));
        let mut yy = PauliString::<1>::y(0);
        yy.mul_assign(&PauliString::<1>::y(1));

        let s = b10_build::<1>(
            8,
            &[
                (xx, Complex64::new(1.0, 0.0)),
                (xz, Complex64::new(10.0, 0.0)),
                (yy, Complex64::new(100.0, 0.0)),
            ],
        );
        assert!((s.expectation_product_state(ProductState::XPlus).re - 1.0).abs() < 1e-12);
        assert!((s.expectation_product_state(ProductState::YPlus).re - 100.0).abs() < 1e-12);
        assert!(s.expectation_product_state(ProductState::ZPlus).re.abs() < 1e-12);
    }

    #[test]
    fn expectation_is_linear_and_keeps_the_imaginary_part() {
        let s = b10_build::<1>(
            8,
            &[
                (PauliString::<1>::x(0), Complex64::new(1.0, 2.0)),
                (PauliString::<1>::x(1), Complex64::new(3.0, -5.0)),
            ],
        );
        let e = s.expectation_product_state(ProductState::XPlus);
        assert!((e - Complex64::new(4.0, -3.0)).norm() < 1e-12);
    }

    #[test]
    fn expectation_across_a_word_boundary_w2() {
        let s = b10_build::<2>(
            128,
            &[
                (PauliString::<2>::x(70), Complex64::new(2.0, 0.0)),
                (PauliString::<2>::z(70), Complex64::new(9.0, 0.0)),
            ],
        );
        assert!((s.expectation_product_state(ProductState::XPlus).re - 2.0).abs() < 1e-12);
        assert!((s.expectation_product_state(ProductState::ZPlus).re - 9.0).abs() < 1e-12);
    }

    /// The new API must reproduce the observable
    /// `examples/ising_2d_quench.rs` hand-rolled, which is why it exists.
    #[test]
    fn expectation_xplus_matches_the_hand_rolled_reference() {
        let mut rng = 0x2468u64 | 1;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut acc = crate::accumulator::BuildAccumulator::<1>::with_capacity(16, 500);
        for _ in 0..500 {
            let pp = PauliString::<1> {
                x: [next() & 0xFFFF],
                z: [next() & 0xFFFF],
            };
            let c = Complex64::new((next() as i64 as f64) / (i64::MAX as f64), 0.0);
            acc.add_term(pp, crate::phase::Phase::ONE, c);
        }
        let sum = acc.finalize();

        let mut want = 0.0f64;
        for i in 0..sum.len() {
            if sum.bucket(0).1[i] == [0u64] {
                want += sum.bucket(0).2[i].re;
            }
        }
        let got = sum.expectation_product_state(ProductState::XPlus).re;
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn assert_invariants_accepts_bits_within_num_qubits() {
        // num_qubits=50, single term with X on qubit 49 (in range).
        let sum = PauliSum::<1>::from_sorted_columns(
            vec![[1u64 << 49]],
            vec![[0u64; 1]],
            vec![Complex64::new(1.0, 0.0)],
            50,
        );
        sum.assert_invariants();
    }

    #[test]
    #[should_panic(expected = "exceeds num_qubits")]
    fn assert_invariants_rejects_bit_beyond_num_qubits() {
        // num_qubits=50, but X bit set at qubit 50 — must panic.
        let sum = PauliSum::<1>::from_sorted_columns(
            vec![[1u64 << 50]],
            vec![[0u64; 1]],
            vec![Complex64::new(1.0, 0.0)],
            50,
        );
        sum.assert_invariants();
    }

    #[test]
    #[should_panic(expected = "exceeds num_qubits")]
    fn assert_invariants_rejects_z_bit_beyond_num_qubits() {
        // Same as above but on the Z-part: invariant must check both parts.
        let sum = PauliSum::<1>::from_sorted_columns(
            vec![[0u64; 1]],
            vec![[1u64 << 60]],
            vec![Complex64::new(1.0, 0.0)],
            50,
        );
        sum.assert_invariants();
    }

    #[test]
    #[should_panic(expected = "exceeds num_qubits")]
    fn assert_invariants_rejects_bit_in_unused_word() {
        // num_qubits=64 (one full word), W=2. Bit on qubit 64 lives in word 1
        // and is therefore out of range.
        let sum = PauliSum::<2>::from_sorted_columns(
            vec![[0u64, 1u64]],
            vec![[0u64; 2]],
            vec![Complex64::new(1.0, 0.0)],
            64,
        );
        sum.assert_invariants();
    }

    // --- Slice 2.1: keyed lookup (get) -----------------------------------

    /// Three-term `PauliSum<1>` with sorted, distinct keys `K0 < K1 < K2`.
    fn three_term_sum_w1() -> PauliSum<1> {
        // K0 = (x=0, z=1), K1 = (x=1, z=0), K2 = (x=1, z=2). Sorted by lex
        // on (x, z): K0 has smallest x; K1, K2 share x but K1 has smaller z.
        PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64], [1u64]],
            vec![[1u64], [0u64], [2u64]],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            4,
        )
    }

    #[test]
    fn get_on_empty_is_none() {
        let s = PauliSum::<1>::empty(4);
        assert_eq!(s.get(&[0u64], &[0u64]), None);
    }

    #[test]
    fn get_hits_every_key_and_misses_between() {
        let s = three_term_sum_w1();
        assert_eq!(s.get(&[0u64], &[1u64]), Some(Complex64::new(1.0, 0.0)));
        assert_eq!(s.get(&[1u64], &[0u64]), Some(Complex64::new(2.0, 0.0)));
        assert_eq!(s.get(&[1u64], &[2u64]), Some(Complex64::new(3.0, 0.0)));
        // Below the smallest, in a gap, and above the largest key.
        assert_eq!(s.get(&[0u64], &[0u64]), None);
        assert_eq!(s.get(&[1u64], &[1u64]), None);
        assert_eq!(s.get(&[2u64], &[0u64]), None);
    }

    #[test]
    fn canonical_order_is_lex_x_before_z_on_a_single_bucket() {
        // Two terms with K_a=(x=0, z=5) and K_b=(x=1, z=0). Despite z_a > z_b,
        // x_a < x_b, so K_a < K_b in the canonical (lex) order of a
        // single-bucket sum. A lex-on-x-only order would invert this.
        let s = PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64]],
            vec![[5u64], [0u64]],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            4,
        );
        s.assert_invariants();
        let (x, z, _) = s.to_arrays();
        assert_eq!((x[0], z[0]), ([0u64], [5u64]));
        assert_eq!((x[1], z[1]), ([1u64], [0u64]));
    }

    #[test]
    fn get_w2_hit_across_word_boundary() {
        // W=2 sum, key bits live in word 1.
        let s = PauliSum::<2>::from_sorted_columns(
            vec![[0u64, 1u64], [0u64, 1u64], [0u64, 2u64]],
            vec![[0u64, 0u64], [0u64, 4u64], [0u64, 0u64]],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            128,
        );
        s.assert_invariants();
        assert_eq!(
            s.get(&[0u64, 1u64], &[0u64, 4u64]),
            Some(Complex64::new(2.0, 0.0))
        );
        assert_eq!(
            s.get(&[0u64, 2u64], &[0u64, 0u64]),
            Some(Complex64::new(3.0, 0.0))
        );
        assert_eq!(s.get(&[0u64, 1u64], &[0u64, 1u64]), None);
    }

    // --- Slice 2.2: scale() ----------------------------------------------

    #[test]
    fn scale_by_zero_zeros_all_coeffs() {
        let mut s = three_term_sum_w1();
        s.scale(Complex64::new(0.0, 0.0));
        assert_eq!(s.len(), 3);
        for (_, _, c) in s.iter() {
            assert_eq!(c, Complex64::new(0.0, 0.0));
        }
        s.assert_invariants();
    }

    #[test]
    fn scale_by_one_is_identity() {
        let mut s = three_term_sum_w1();
        let (_, _, before) = s.to_arrays();
        s.scale(Complex64::new(1.0, 0.0));
        assert_eq!(s.to_arrays().2, before);
    }

    #[test]
    fn scale_by_i_rotates_phases() {
        let mut s = PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64]],
            vec![[1u64], [0u64]],
            vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, -3.0)],
            4,
        );
        s.scale(Complex64::new(0.0, 1.0));
        // (2 + 0i) * i = 0 + 2i; (0 - 3i) * i = 3 + 0i.
        assert_eq!(s.bucket(0).2[0], Complex64::new(0.0, 2.0));
        assert_eq!(s.bucket(0).2[1], Complex64::new(3.0, 0.0));
    }

    #[test]
    fn scale_preserves_sort_invariant() {
        let mut s = three_term_sum_w1();
        s.scale(Complex64::new(2.5, -0.5));
        s.assert_invariants();
    }

    // --- Slice 2.3: truncate_by_magnitude() ------------------------------

    #[test]
    fn truncate_eps_zero_is_noop_on_nonzero_terms() {
        let mut s = three_term_sum_w1();
        let before_len = s.len();
        s.truncate_by_magnitude(0.0);
        assert_eq!(s.len(), before_len);
        s.assert_invariants();
    }

    #[test]
    fn truncate_eps_above_max_empties() {
        let mut s = three_term_sum_w1();
        s.truncate_by_magnitude(10.0);
        assert!(s.is_empty());
        s.assert_invariants();
    }

    #[test]
    fn truncate_mixed_drops_only_below_threshold() {
        // Four sorted terms with magnitudes [0.1, 0.5, 1.0, 0.05]; eps=0.2
        // keeps 0.5 and 1.0 (originally at indices 1 and 2).
        let mut s = PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [0u64], [1u64], [1u64]],
            vec![[1u64], [2u64], [0u64], [1u64]],
            vec![
                Complex64::new(0.1, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.05, 0.0),
            ],
            4,
        );
        s.assert_invariants();
        s.truncate_by_magnitude(0.2);
        assert_eq!(s.len(), 2);
        assert_eq!(s.bucket(0).0[0], [0u64]);
        assert_eq!(s.bucket(0).1[0], [2u64]);
        assert_eq!(s.bucket(0).2[0], Complex64::new(0.5, 0.0));
        assert_eq!(s.bucket(0).0[1], [1u64]);
        assert_eq!(s.bucket(0).1[1], [0u64]);
        assert_eq!(s.bucket(0).2[1], Complex64::new(1.0, 0.0));
        s.assert_invariants();
    }

    #[test]
    fn truncate_drops_exact_zero_at_eps_zero() {
        // Include an exact (0+0i) term; eps=0 should drop only that one.
        let mut s = PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64], [1u64]],
            vec![[1u64], [0u64], [2u64]],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            4,
        );
        s.truncate_by_magnitude(0.0);
        assert_eq!(s.len(), 2);
        assert_eq!(s.bucket(0).0[0], [0u64]);
        assert_eq!(s.bucket(0).0[1], [1u64]);
        assert_eq!(s.bucket(0).1[1], [2u64]);
        s.assert_invariants();
    }

    #[test]
    fn truncate_w2_preserves_sort() {
        let mut s = PauliSum::<2>::from_sorted_columns(
            vec![[0u64, 0u64], [0u64, 1u64], [1u64, 0u64]],
            vec![[0u64, 1u64], [0u64, 0u64], [0u64, 0u64]],
            vec![
                Complex64::new(0.01, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.005, 0.0),
            ],
            128,
        );
        s.assert_invariants();
        s.truncate_by_magnitude(0.1);
        assert_eq!(s.len(), 1);
        assert_eq!(s.bucket(0).0[0], [0u64, 1u64]);
        s.assert_invariants();
    }

    // --- Slice 2.4: add() ------------------------------------------------

    #[test]
    fn add_empty_left_is_other() {
        let a = PauliSum::<1>::empty(4);
        let b = three_term_sum_w1();
        let r = a.add(&b);
        assert_eq!(r.len(), 3);
        assert_eq!(r.to_arrays(), b.to_arrays());
        r.assert_invariants();
    }

    #[test]
    fn add_empty_right_is_self() {
        let a = three_term_sum_w1();
        let b = PauliSum::<1>::empty(4);
        let r = a.add(&b);
        assert_eq!(r.len(), 3);
        assert_eq!(r.to_arrays(), a.to_arrays());
        r.assert_invariants();
    }

    #[test]
    fn add_disjoint_keys_interleaves_in_sort_order() {
        // a has K0=(0,1), K2=(1,2); b has K1=(1,0), K3=(2,0).
        // Lex sort across the union: (0,1) < (1,0) < (1,2) < (2,0).
        let a = PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64]],
            vec![[1u64], [2u64]],
            vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
            4,
        );
        let b = PauliSum::<1>::from_sorted_columns(
            vec![[1u64], [2u64]],
            vec![[0u64], [0u64]],
            vec![Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)],
            4,
        );
        let r = a.add(&b);
        assert_eq!(r.len(), 4);
        let (rx, rz, rc) = r.to_arrays();
        assert_eq!(rx, vec![[0u64], [1u64], [1u64], [2u64]]);
        assert_eq!(rz, vec![[1u64], [0u64], [2u64], [0u64]]);
        assert_eq!(
            rc,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ]
        );
        r.assert_invariants();
    }

    #[test]
    fn add_equal_keys_sum_coeffs() {
        let a = three_term_sum_w1();
        let r = a.add(&a);
        assert_eq!(r.len(), 3);
        assert_eq!(r.to_arrays().0, a.to_arrays().0);
        assert_eq!(r.to_arrays().1, a.to_arrays().1);
        for k in 0..3 {
            assert_eq!(
                r.bucket(0).2[k],
                a.bucket(0).2[k] * Complex64::new(2.0, 0.0)
            );
        }
        r.assert_invariants();
    }

    #[test]
    fn add_cancellation_drops_term() {
        let a = PauliSum::<1>::from_sorted_columns(
            vec![[1u64]],
            vec![[0u64]],
            vec![Complex64::new(1.0, 0.0)],
            4,
        );
        let b = PauliSum::<1>::from_sorted_columns(
            vec![[1u64]],
            vec![[0u64]],
            vec![Complex64::new(-1.0, 0.0)],
            4,
        );
        let r = a.add(&b);
        assert!(r.is_empty());
        r.assert_invariants();
    }

    #[test]
    fn add_mixed_cancellation_and_merge() {
        // a = {K1: 1, K2: 2, K3: 3}, b = {K1: -1, K2: 0.5, K4: 4}
        // K1 cancels, K2 sums to 2.5, K3 unique to a, K4 unique to b.
        let a = PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64], [2u64]],
            vec![[0u64], [0u64], [0u64]],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            4,
        );
        let b = PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64], [3u64]],
            vec![[0u64], [0u64], [0u64]],
            vec![
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(4.0, 0.0),
            ],
            4,
        );
        let r = a.add(&b);
        assert_eq!(r.len(), 3);
        let (rx, rz, rc) = r.to_arrays();
        assert_eq!(rx, vec![[1u64], [2u64], [3u64]]);
        assert_eq!(rz, vec![[0u64], [0u64], [0u64]]);
        assert_eq!(
            rc,
            vec![
                Complex64::new(2.5, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ]
        );
        r.assert_invariants();
    }

    #[test]
    fn add_w2_across_word_boundary() {
        let a = PauliSum::<2>::from_sorted_columns(
            vec![[0u64, 1u64], [0u64, 2u64]],
            vec![[0u64, 0u64], [0u64, 0u64]],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            128,
        );
        let b = PauliSum::<2>::from_sorted_columns(
            vec![[0u64, 1u64], [0u64, 4u64]],
            vec![[0u64, 0u64], [0u64, 0u64]],
            vec![Complex64::new(0.5, 0.0), Complex64::new(7.0, 0.0)],
            128,
        );
        let r = a.add(&b);
        assert_eq!(r.len(), 3);
        assert_eq!(
            r.to_arrays().0,
            vec![[0u64, 1u64], [0u64, 2u64], [0u64, 4u64]]
        );
        assert_eq!(r.bucket(0).2[0], Complex64::new(1.5, 0.0));
        assert_eq!(r.bucket(0).2[1], Complex64::new(2.0, 0.0));
        assert_eq!(r.bucket(0).2[2], Complex64::new(7.0, 0.0));
        r.assert_invariants();
    }

    // --- Slice 3.2: PauliSum::from_strings test helper -------------------

    #[test]
    fn from_strings_single_x_term() {
        let s = PauliSum::<1>::from_strings(&[("XII", Complex64::new(1.0, 0.0))]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.num_qubits(), 3);
        assert_eq!(s.bucket(0).0[0], [0b001u64]);
        assert_eq!(s.bucket(0).1[0], [0u64]);
        assert_eq!(s.bucket(0).2[0], Complex64::new(1.0, 0.0));
        s.assert_invariants();
    }

    #[test]
    fn from_strings_x_z_combined() {
        // "XZI": X on qubit 0, Z on qubit 1, I on qubit 2.
        let s = PauliSum::<1>::from_strings(&[("XZI", Complex64::new(1.0, 0.0))]);
        assert_eq!(s.bucket(0).0[0], [0b001u64]);
        assert_eq!(s.bucket(0).1[0], [0b010u64]);
        s.assert_invariants();
    }

    #[test]
    fn from_strings_y_includes_i_phase() {
        // Y_canonical = i · (x=1, z=1). Caller writes coeff=1, stored is i.
        let s = PauliSum::<1>::from_strings(&[("Y", Complex64::new(1.0, 0.0))]);
        assert_eq!(s.bucket(0).0[0], [1u64]);
        assert_eq!(s.bucket(0).1[0], [1u64]);
        assert_eq!(s.bucket(0).2[0], Complex64::new(0.0, 1.0));
    }

    #[test]
    fn from_strings_yy_phase_minus_one() {
        // i^2 = -1.
        let s = PauliSum::<1>::from_strings(&[("YY", Complex64::new(1.0, 0.0))]);
        assert_eq!(s.bucket(0).2[0], Complex64::new(-1.0, 0.0));
    }

    #[test]
    fn from_strings_yyy_phase_minus_i() {
        // i^3 = -i.
        let s = PauliSum::<1>::from_strings(&[("YYY", Complex64::new(1.0, 0.0))]);
        assert_eq!(s.bucket(0).2[0], Complex64::new(0.0, -1.0));
    }

    #[test]
    fn from_strings_yyyy_phase_one() {
        // i^4 = 1.
        let s = PauliSum::<1>::from_strings(&[("YYYY", Complex64::new(1.0, 0.0))]);
        assert_eq!(s.bucket(0).2[0], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn from_strings_dedup_sums_coeffs() {
        let s = PauliSum::<1>::from_strings(&[
            ("XI", Complex64::new(1.0, 0.0)),
            ("XI", Complex64::new(0.5, -0.25)),
        ]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.bucket(0).2[0], Complex64::new(1.5, -0.25));
        s.assert_invariants();
    }

    #[test]
    fn from_strings_cancellation_drops_term() {
        let s = PauliSum::<1>::from_strings(&[
            ("XI", Complex64::new(1.0, 0.0)),
            ("XI", Complex64::new(-1.0, 0.0)),
            ("ZI", Complex64::new(2.0, 0.0)),
        ]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.bucket(0).0[0], [0u64]);
        assert_eq!(s.bucket(0).1[0], [1u64]);
        assert_eq!(s.bucket(0).2[0], Complex64::new(2.0, 0.0));
        s.assert_invariants();
    }

    #[test]
    fn from_strings_sorts_lex_keys() {
        // Insert out of order: ZI=(0,1), XI=(1,0), YI=(1,1) — lex sorted is
        // ZI < XI < YI.
        let s = PauliSum::<1>::from_strings(&[
            ("YI", Complex64::new(1.0, 0.0)),
            ("ZI", Complex64::new(2.0, 0.0)),
            ("XI", Complex64::new(3.0, 0.0)),
        ]);
        assert_eq!(s.len(), 3);
        assert_eq!((s.bucket(0).0[0], s.bucket(0).1[0]), ([0u64], [1u64])); // ZI
        assert_eq!((s.bucket(0).0[1], s.bucket(0).1[1]), ([1u64], [0u64])); // XI
        assert_eq!((s.bucket(0).0[2], s.bucket(0).1[2]), ([1u64], [1u64])); // YI (with i factor)
        assert_eq!(s.bucket(0).2[0], Complex64::new(2.0, 0.0));
        assert_eq!(s.bucket(0).2[1], Complex64::new(3.0, 0.0));
        assert_eq!(s.bucket(0).2[2], Complex64::new(0.0, 1.0));
        s.assert_invariants();
    }

    #[test]
    fn from_strings_w2_qubit_64() {
        // 65-character string: X at index 64 lands in word 1.
        let mut s_chars: String = "I".repeat(65);
        // Replace index 64 with 'X'.
        unsafe {
            let bytes = s_chars.as_bytes_mut();
            bytes[64] = b'X';
        }
        let s = PauliSum::<2>::from_strings(&[(s_chars.as_str(), Complex64::new(1.0, 0.0))]);
        assert_eq!(s.num_qubits(), 65);
        assert_eq!(s.bucket(0).0[0], [0u64, 1u64]);
        assert_eq!(s.bucket(0).1[0], [0u64, 0u64]);
        s.assert_invariants();
    }

    #[test]
    #[should_panic(expected = "unexpected Pauli char")]
    fn from_strings_panics_on_invalid_char() {
        let _ = PauliSum::<1>::from_strings(&[("AB", Complex64::new(1.0, 0.0))]);
    }

    #[test]
    #[should_panic(expected = "all pauli strings must have the same length")]
    fn from_strings_panics_on_length_mismatch() {
        let _ = PauliSum::<1>::from_strings(&[
            ("XI", Complex64::new(1.0, 0.0)),
            ("XII", Complex64::new(1.0, 0.0)),
        ]);
    }
}

#[cfg(all(test, debug_assertions))]
mod props {
    use super::*;
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    /// Build a sorted, deduplicated `PauliSum<2>` from random `(x, z, coeff)`
    /// triples. Uses `BTreeMap` keyed on `(x, z)` to enforce the sorted /
    /// unique invariant before SoA materialization. Coefficients are kept
    /// small (`re, im ∈ [-4, 4]`) so the property assertions don't run into
    /// FP cancellation noise. Length capped at 8 — sufficient to exercise
    /// merge interleaving without blowing up shrinking time.
    fn arb_pauli_sum_w2() -> impl Strategy<Value = PauliSum<2>> {
        prop::collection::vec(
            (
                any::<u64>(),
                any::<u64>(),
                any::<u64>(),
                any::<u64>(),
                -4.0f64..4.0,
                -4.0f64..4.0,
            ),
            0..8,
        )
        .prop_map(|entries| {
            let mut map: BTreeMap<([u64; 2], [u64; 2]), Complex64> = BTreeMap::new();
            for (x0, x1, z0, z1, re, im) in entries {
                map.insert(([x0, x1], [z0, z1]), Complex64::new(re, im));
            }
            let mut x = Vec::with_capacity(map.len());
            let mut z = Vec::with_capacity(map.len());
            let mut coeff = Vec::with_capacity(map.len());
            for ((kx, kz), c) in map {
                x.push(kx);
                z.push(kz);
                coeff.push(c);
            }
            PauliSum::<2>::from_sorted_columns(x, z, coeff, 128)
        })
    }

    proptest! {
        #[test]
        fn add_is_associative(
            a in arb_pauli_sum_w2(),
            b in arb_pauli_sum_w2(),
            c in arb_pauli_sum_w2(),
        ) {
            let left = a.add(&b).add(&c);
            let right = a.add(&b.add(&c));
            left.assert_invariants();
            right.assert_invariants();
            let (lx, lz, lc) = left.to_arrays();
            let (rx, rz, rc) = right.to_arrays();
            prop_assert_eq!(lx, rx);
            prop_assert_eq!(lz, rz);
            prop_assert_eq!(lc.len(), rc.len());
            for k in 0..lc.len() {
                let diff = lc[k] - rc[k];
                prop_assert!(
                    diff.norm() <= 1e-12,
                    "coeff mismatch at idx {}: lhs={:?} rhs={:?}",
                    k, lc[k], rc[k]
                );
            }
        }
    }
}
