//! `PauliSum<W>` — weighted sum of Pauli strings in structure-of-arrays form.
//!
//! See design doc §3.2.
//!
//! Invariant: parallel arrays `x` and `z` are sorted in lexicographic order as
//! a single key, and no two entries share a key. Every public operation
//! either preserves this invariant or returns a fresh `PauliSum` that does.

#![allow(unused)]

use num_complex::Complex64;

use crate::pauli_string::PauliString;

/// Weighted sum of Pauli operators, stored SoA, sorted and deduplicated.
#[derive(Clone, Debug, Default)]
pub struct PauliSum<const W: usize> {
    pub(crate) x: Vec<[u64; W]>,
    pub(crate) z: Vec<[u64; W]>,
    pub(crate) coeff: Vec<Complex64>,
    pub(crate) num_qubits: usize,
}

impl<const W: usize> PauliSum<W> {
    /// Empty sum on `num_qubits` qubits. Caller is responsible for ensuring
    /// `num_qubits <= 64 * W`.
    pub fn empty(num_qubits: usize) -> Self {
        debug_assert!(num_qubits <= 64 * W);
        Self {
            x: Vec::new(),
            z: Vec::new(),
            coeff: Vec::new(),
            num_qubits,
        }
    }

    /// Number of qubits this sum is defined over.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of non-identity terms after deduplication.
    #[inline]
    pub fn len(&self) -> usize {
        self.coeff.len()
    }

    /// `true` iff the sum has no terms.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coeff.is_empty()
    }

    /// Read-only view of the X-part column.
    #[inline]
    pub fn x(&self) -> &[[u64; W]] {
        &self.x
    }

    /// Read-only view of the Z-part column.
    #[inline]
    pub fn z(&self) -> &[[u64; W]] {
        &self.z
    }

    /// Read-only view of the coefficient column.
    #[inline]
    pub fn coeff(&self) -> &[Complex64] {
        &self.coeff
    }

    /// Sum of two `PauliSum`s. Linear-time merge; preserves the sorted invariant.
    /// Terms whose coefficients sum to exactly `0+0i` are dropped.
    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.num_qubits, other.num_qubits);
        let n_a = self.x.len();
        let n_b = other.x.len();
        let cap = n_a + n_b;
        let mut x = Vec::with_capacity(cap);
        let mut z = Vec::with_capacity(cap);
        let mut coeff = Vec::with_capacity(cap);
        let zero = Complex64::new(0.0, 0.0);
        let (mut i, mut j) = (0usize, 0usize);
        while i < n_a && j < n_b {
            match (&self.x[i], &self.z[i]).cmp(&(&other.x[j], &other.z[j])) {
                std::cmp::Ordering::Less => {
                    x.push(self.x[i]);
                    z.push(self.z[i]);
                    coeff.push(self.coeff[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    x.push(other.x[j]);
                    z.push(other.z[j]);
                    coeff.push(other.coeff[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let c = self.coeff[i] + other.coeff[j];
                    if c != zero {
                        x.push(self.x[i]);
                        z.push(self.z[i]);
                        coeff.push(c);
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < n_a {
            x.push(self.x[i]);
            z.push(self.z[i]);
            coeff.push(self.coeff[i]);
            i += 1;
        }
        while j < n_b {
            x.push(other.x[j]);
            z.push(other.z[j]);
            coeff.push(other.coeff[j]);
            j += 1;
        }
        Self {
            x,
            z,
            coeff,
            num_qubits: self.num_qubits,
        }
    }

    /// Multiply every coefficient by `c` in place.
    pub fn scale(&mut self, c: Complex64) {
        for coeff in self.coeff.iter_mut() {
            *coeff *= c;
        }
    }

    /// Locate a Pauli key by binary search; returns `Ok(idx)` if present,
    /// `Err(idx)` for the insertion point otherwise.
    pub fn find(&self, x: &[u64; W], z: &[u64; W]) -> Result<usize, usize> {
        let mut lo = 0;
        let mut hi = self.x.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match (&self.x[mid], &self.z[mid]).cmp(&(x, z)) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Err(lo)
    }

    /// Drop terms whose coefficient magnitude is `<= eps`. Preserves sort.
    pub fn truncate_by_magnitude(&mut self, eps: f64) {
        let n = self.coeff.len();
        let mut w = 0;
        for r in 0..n {
            if self.coeff[r].norm() > eps {
                if w != r {
                    self.x[w] = self.x[r];
                    self.z[w] = self.z[r];
                    self.coeff[w] = self.coeff[r];
                }
                w += 1;
            }
        }
        self.x.truncate(w);
        self.z.truncate(w);
        self.coeff.truncate(w);
    }

    /// Debug-only invariant check. No-op in release builds.
    #[cfg(debug_assertions)]
    pub fn assert_invariants(&self) {
        assert_eq!(self.x.len(), self.z.len());
        assert_eq!(self.x.len(), self.coeff.len());
        for i in 0..self.x.len() {
            let term = PauliString::<W> {
                x: self.x[i],
                z: self.z[i],
            };
            assert!(
                term.is_within(self.num_qubits),
                "PauliSum term {} has bits beyond num_qubits={}",
                i,
                self.num_qubits,
            );
        }
        for i in 1..self.x.len() {
            let prev = (&self.x[i - 1], &self.z[i - 1]);
            let cur = (&self.x[i], &self.z[i]);
            assert!(prev < cur, "PauliSum out of order at {}", i);
        }
    }
}

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;

    #[test]
    fn assert_invariants_accepts_bits_within_num_qubits() {
        // num_qubits=50, single term with X on qubit 49 (in range).
        let sum = PauliSum::<1> {
            x: vec![[1u64 << 49]],
            z: vec![[0u64; 1]],
            coeff: vec![Complex64::new(1.0, 0.0)],
            num_qubits: 50,
        };
        sum.assert_invariants();
    }

    #[test]
    #[should_panic(expected = "beyond num_qubits")]
    fn assert_invariants_rejects_bit_beyond_num_qubits() {
        // num_qubits=50, but X bit set at qubit 50 — must panic.
        let sum = PauliSum::<1> {
            x: vec![[1u64 << 50]],
            z: vec![[0u64; 1]],
            coeff: vec![Complex64::new(1.0, 0.0)],
            num_qubits: 50,
        };
        sum.assert_invariants();
    }

    #[test]
    #[should_panic(expected = "beyond num_qubits")]
    fn assert_invariants_rejects_z_bit_beyond_num_qubits() {
        // Same as above but on the Z-part: invariant must check both parts.
        let sum = PauliSum::<1> {
            x: vec![[0u64; 1]],
            z: vec![[1u64 << 60]],
            coeff: vec![Complex64::new(1.0, 0.0)],
            num_qubits: 50,
        };
        sum.assert_invariants();
    }

    #[test]
    #[should_panic(expected = "beyond num_qubits")]
    fn assert_invariants_rejects_bit_in_unused_word() {
        // num_qubits=64 (one full word), W=2. Bit on qubit 64 lives in word 1
        // and is therefore out of range.
        let sum = PauliSum::<2> {
            x: vec![[0u64, 1u64]],
            z: vec![[0u64; 2]],
            coeff: vec![Complex64::new(1.0, 0.0)],
            num_qubits: 64,
        };
        sum.assert_invariants();
    }

    // --- Slice 2.1: find() -----------------------------------------------

    /// Three-term `PauliSum<1>` with sorted, distinct keys `K0 < K1 < K2`.
    /// Used as the fixture for `find` hit/miss tests.
    fn three_term_sum_w1() -> PauliSum<1> {
        // K0 = (x=0, z=1), K1 = (x=1, z=0), K2 = (x=1, z=2). Sorted by lex
        // on (x, z): K0 has smallest x; K1, K2 share x but K1 has smaller z.
        PauliSum::<1> {
            x: vec![[0u64], [1u64], [1u64]],
            z: vec![[1u64], [0u64], [2u64]],
            coeff: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            num_qubits: 4,
        }
    }

    #[test]
    fn find_on_empty_returns_err_zero() {
        let s = PauliSum::<1>::empty(4);
        assert_eq!(s.find(&[0u64], &[0u64]), Err(0));
    }

    #[test]
    fn find_hit_at_index_zero() {
        let s = three_term_sum_w1();
        assert_eq!(s.find(&[0u64], &[1u64]), Ok(0));
    }

    #[test]
    fn find_hit_in_middle() {
        let s = three_term_sum_w1();
        assert_eq!(s.find(&[1u64], &[0u64]), Ok(1));
    }

    #[test]
    fn find_hit_at_last() {
        let s = three_term_sum_w1();
        assert_eq!(s.find(&[1u64], &[2u64]), Ok(2));
    }

    #[test]
    fn find_miss_below_min_returns_err_zero() {
        let s = three_term_sum_w1();
        // Identity (0, 0) is below K0=(0, 1) under lex.
        assert_eq!(s.find(&[0u64], &[0u64]), Err(0));
    }

    #[test]
    fn find_miss_in_gap_returns_insertion_point() {
        let s = three_term_sum_w1();
        // (1, 1) sits between K1=(1,0) and K2=(1,2): insertion point is 2.
        assert_eq!(s.find(&[1u64], &[1u64]), Err(2));
    }

    #[test]
    fn find_miss_above_max_returns_err_len() {
        let s = three_term_sum_w1();
        // (2, 0) is above all keys (largest x).
        assert_eq!(s.find(&[2u64], &[0u64]), Err(3));
    }

    #[test]
    fn find_lex_orders_x_before_z() {
        // Two terms with K_a=(x=0, z=5) and K_b=(x=1, z=0). Despite z_a > z_b,
        // x_a < x_b, so K_a < K_b. A lex-on-x-only impl would invert this.
        let s = PauliSum::<1> {
            x: vec![[0u64], [1u64]],
            z: vec![[5u64], [0u64]],
            coeff: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            num_qubits: 4,
        };
        assert_eq!(s.find(&[0u64], &[5u64]), Ok(0));
        assert_eq!(s.find(&[1u64], &[0u64]), Ok(1));
        // (0, 6) is between them under lex(x, z): same x as K_a but larger z.
        assert_eq!(s.find(&[0u64], &[6u64]), Err(1));
    }

    #[test]
    fn find_w2_hit_across_word_boundary() {
        // W=2 sum, key bits live in word 1.
        let s = PauliSum::<2> {
            x: vec![[0u64, 1u64], [0u64, 1u64], [0u64, 2u64]],
            z: vec![[0u64, 0u64], [0u64, 4u64], [0u64, 0u64]],
            coeff: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            num_qubits: 128,
        };
        s.assert_invariants();
        assert_eq!(s.find(&[0u64, 1u64], &[0u64, 4u64]), Ok(1));
        assert_eq!(s.find(&[0u64, 2u64], &[0u64, 0u64]), Ok(2));
        // Miss between idx 0 and 1: same x, z between 0 and 4.
        assert_eq!(s.find(&[0u64, 1u64], &[0u64, 1u64]), Err(1));
    }

    // --- Slice 2.2: scale() ----------------------------------------------

    #[test]
    fn scale_by_zero_zeros_all_coeffs() {
        let mut s = three_term_sum_w1();
        s.scale(Complex64::new(0.0, 0.0));
        assert_eq!(s.len(), 3);
        for c in s.coeff() {
            assert_eq!(*c, Complex64::new(0.0, 0.0));
        }
        s.assert_invariants();
    }

    #[test]
    fn scale_by_one_is_identity() {
        let mut s = three_term_sum_w1();
        let before: Vec<Complex64> = s.coeff().to_vec();
        s.scale(Complex64::new(1.0, 0.0));
        assert_eq!(s.coeff(), before.as_slice());
    }

    #[test]
    fn scale_by_i_rotates_phases() {
        let mut s = PauliSum::<1> {
            x: vec![[0u64], [1u64]],
            z: vec![[1u64], [0u64]],
            coeff: vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, -3.0)],
            num_qubits: 4,
        };
        s.scale(Complex64::new(0.0, 1.0));
        // (2 + 0i) * i = 0 + 2i; (0 - 3i) * i = 3 + 0i.
        assert_eq!(s.coeff()[0], Complex64::new(0.0, 2.0));
        assert_eq!(s.coeff()[1], Complex64::new(3.0, 0.0));
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
        let mut s = PauliSum::<1> {
            x: vec![[0u64], [0u64], [1u64], [1u64]],
            z: vec![[1u64], [2u64], [0u64], [1u64]],
            coeff: vec![
                Complex64::new(0.1, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.05, 0.0),
            ],
            num_qubits: 4,
        };
        s.assert_invariants();
        s.truncate_by_magnitude(0.2);
        assert_eq!(s.len(), 2);
        assert_eq!(s.x()[0], [0u64]);
        assert_eq!(s.z()[0], [2u64]);
        assert_eq!(s.coeff()[0], Complex64::new(0.5, 0.0));
        assert_eq!(s.x()[1], [1u64]);
        assert_eq!(s.z()[1], [0u64]);
        assert_eq!(s.coeff()[1], Complex64::new(1.0, 0.0));
        s.assert_invariants();
    }

    #[test]
    fn truncate_drops_exact_zero_at_eps_zero() {
        // Include an exact (0+0i) term; eps=0 should drop only that one.
        let mut s = PauliSum::<1> {
            x: vec![[0u64], [1u64], [1u64]],
            z: vec![[1u64], [0u64], [2u64]],
            coeff: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
            ],
            num_qubits: 4,
        };
        s.truncate_by_magnitude(0.0);
        assert_eq!(s.len(), 2);
        assert_eq!(s.x()[0], [0u64]);
        assert_eq!(s.x()[1], [1u64]);
        assert_eq!(s.z()[1], [2u64]);
        s.assert_invariants();
    }

    #[test]
    fn truncate_w2_preserves_sort() {
        let mut s = PauliSum::<2> {
            x: vec![[0u64, 0u64], [0u64, 1u64], [1u64, 0u64]],
            z: vec![[0u64, 1u64], [0u64, 0u64], [0u64, 0u64]],
            coeff: vec![
                Complex64::new(0.01, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.005, 0.0),
            ],
            num_qubits: 128,
        };
        s.assert_invariants();
        s.truncate_by_magnitude(0.1);
        assert_eq!(s.len(), 1);
        assert_eq!(s.x()[0], [0u64, 1u64]);
        s.assert_invariants();
    }

    // --- Slice 2.4: add() ------------------------------------------------

    #[test]
    fn add_empty_left_is_other() {
        let a = PauliSum::<1>::empty(4);
        let b = three_term_sum_w1();
        let r = a.add(&b);
        assert_eq!(r.len(), 3);
        assert_eq!(r.x(), b.x());
        assert_eq!(r.z(), b.z());
        assert_eq!(r.coeff(), b.coeff());
        r.assert_invariants();
    }

    #[test]
    fn add_empty_right_is_self() {
        let a = three_term_sum_w1();
        let b = PauliSum::<1>::empty(4);
        let r = a.add(&b);
        assert_eq!(r.len(), 3);
        assert_eq!(r.x(), a.x());
        assert_eq!(r.z(), a.z());
        assert_eq!(r.coeff(), a.coeff());
        r.assert_invariants();
    }

    #[test]
    fn add_disjoint_keys_interleaves_in_sort_order() {
        // a has K0=(0,1), K2=(1,2); b has K1=(1,0), K3=(2,0).
        // Lex sort across the union: (0,1) < (1,0) < (1,2) < (2,0).
        let a = PauliSum::<1> {
            x: vec![[0u64], [1u64]],
            z: vec![[1u64], [2u64]],
            coeff: vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
            num_qubits: 4,
        };
        let b = PauliSum::<1> {
            x: vec![[1u64], [2u64]],
            z: vec![[0u64], [0u64]],
            coeff: vec![Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)],
            num_qubits: 4,
        };
        let r = a.add(&b);
        assert_eq!(r.len(), 4);
        assert_eq!(r.x(), &[[0u64], [1u64], [1u64], [2u64]][..]);
        assert_eq!(r.z(), &[[1u64], [0u64], [2u64], [0u64]][..]);
        assert_eq!(
            r.coeff(),
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ][..]
        );
        r.assert_invariants();
    }

    #[test]
    fn add_equal_keys_sum_coeffs() {
        let a = three_term_sum_w1();
        let r = a.add(&a);
        assert_eq!(r.len(), 3);
        assert_eq!(r.x(), a.x());
        assert_eq!(r.z(), a.z());
        for k in 0..3 {
            assert_eq!(r.coeff()[k], a.coeff()[k] * Complex64::new(2.0, 0.0));
        }
        r.assert_invariants();
    }

    #[test]
    fn add_cancellation_drops_term() {
        let a = PauliSum::<1> {
            x: vec![[1u64]],
            z: vec![[0u64]],
            coeff: vec![Complex64::new(1.0, 0.0)],
            num_qubits: 4,
        };
        let b = PauliSum::<1> {
            x: vec![[1u64]],
            z: vec![[0u64]],
            coeff: vec![Complex64::new(-1.0, 0.0)],
            num_qubits: 4,
        };
        let r = a.add(&b);
        assert!(r.is_empty());
        r.assert_invariants();
    }

    #[test]
    fn add_mixed_cancellation_and_merge() {
        // a = {K1: 1, K2: 2, K3: 3}, b = {K1: -1, K2: 0.5, K4: 4}
        // K1 cancels, K2 sums to 2.5, K3 unique to a, K4 unique to b.
        let a = PauliSum::<1> {
            x: vec![[0u64], [1u64], [2u64]],
            z: vec![[0u64], [0u64], [0u64]],
            coeff: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            num_qubits: 4,
        };
        let b = PauliSum::<1> {
            x: vec![[0u64], [1u64], [3u64]],
            z: vec![[0u64], [0u64], [0u64]],
            coeff: vec![
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(4.0, 0.0),
            ],
            num_qubits: 4,
        };
        let r = a.add(&b);
        assert_eq!(r.len(), 3);
        assert_eq!(r.x(), &[[1u64], [2u64], [3u64]][..]);
        assert_eq!(r.z(), &[[0u64], [0u64], [0u64]][..]);
        assert_eq!(
            r.coeff(),
            &[
                Complex64::new(2.5, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ][..]
        );
        r.assert_invariants();
    }

    #[test]
    fn add_w2_across_word_boundary() {
        let a = PauliSum::<2> {
            x: vec![[0u64, 1u64], [0u64, 2u64]],
            z: vec![[0u64, 0u64], [0u64, 0u64]],
            coeff: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            num_qubits: 128,
        };
        let b = PauliSum::<2> {
            x: vec![[0u64, 1u64], [0u64, 4u64]],
            z: vec![[0u64, 0u64], [0u64, 0u64]],
            coeff: vec![Complex64::new(0.5, 0.0), Complex64::new(7.0, 0.0)],
            num_qubits: 128,
        };
        let r = a.add(&b);
        assert_eq!(r.len(), 3);
        assert_eq!(
            r.x(),
            &[[0u64, 1u64], [0u64, 2u64], [0u64, 4u64]][..]
        );
        assert_eq!(r.coeff()[0], Complex64::new(1.5, 0.0));
        assert_eq!(r.coeff()[1], Complex64::new(2.0, 0.0));
        assert_eq!(r.coeff()[2], Complex64::new(7.0, 0.0));
        r.assert_invariants();
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
            PauliSum::<2> {
                x,
                z,
                coeff,
                num_qubits: 128,
            }
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
            prop_assert_eq!(left.x(), right.x());
            prop_assert_eq!(left.z(), right.z());
            prop_assert_eq!(left.coeff().len(), right.coeff().len());
            for k in 0..left.coeff().len() {
                let diff = left.coeff()[k] - right.coeff()[k];
                prop_assert!(
                    diff.norm() <= 1e-12,
                    "coeff mismatch at idx {}: lhs={:?} rhs={:?}",
                    k, left.coeff()[k], right.coeff()[k]
                );
            }
        }
    }
}
