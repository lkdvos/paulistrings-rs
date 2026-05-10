//! Built-in truncation policies and combinators. See §7.

#![allow(unused)]

use super::TruncationPolicy;
use crate::pauli_sum::PauliSum;
use num_complex::Complex64;

/// Drop terms whose coefficient magnitude falls below `epsilon`.
pub struct CoefficientThreshold(pub f64);

impl<const W: usize> TruncationPolicy<W> for CoefficientThreshold {
    #[inline]
    fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], c: Complex64) -> bool {
        c.norm() > self.0
    }
}

/// Drop terms whose Pauli weight (number of non-identity qubits) exceeds `k`.
pub struct WeightCutoff(pub u32);

impl<const W: usize> TruncationPolicy<W> for WeightCutoff {
    #[inline]
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], _c: Complex64) -> bool {
        let weight: u32 = (0..W).map(|i| (x[i] | z[i]).count_ones()).sum();
        weight <= self.0
    }
}

/// Retain only the `n` terms with largest coefficient magnitude. Implemented
/// as a `finalize_layer` partial sort (no per-term filter).
pub struct TopN(pub usize);

impl<const W: usize> TruncationPolicy<W> for TopN {
    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        let n = self.0;
        let len = sum.coeff.len();
        if len <= n {
            return;
        }
        if n == 0 {
            sum.x.clear();
            sum.z.clear();
            sum.coeff.clear();
            return;
        }
        let mut perm: Vec<usize> = (0..len).collect();
        // Partition descending by |coeff|: indices [0..n) hold the n largest.
        perm.select_nth_unstable_by(n - 1, |&a, &b| {
            sum.coeff[b]
                .norm()
                .partial_cmp(&sum.coeff[a].norm())
                .unwrap()
        });
        perm.truncate(n);
        // The survivors of an already-sorted sum are still sorted once we
        // restore their original index order.
        perm.sort_unstable();
        let new_x: Vec<[u64; W]> = perm.iter().map(|&i| sum.x[i]).collect();
        let new_z: Vec<[u64; W]> = perm.iter().map(|&i| sum.z[i]).collect();
        let new_c: Vec<Complex64> = perm.iter().map(|&i| sum.coeff[i]).collect();
        sum.x = new_x;
        sum.z = new_z;
        sum.coeff = new_c;
    }
}

/// Logical AND of two policies — both must accept.
pub struct And<A, B>(pub A, pub B);

impl<const W: usize, A, B> TruncationPolicy<W> for And<A, B>
where
    A: TruncationPolicy<W>,
    B: TruncationPolicy<W>,
{
    #[inline]
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
        self.0.keep_term(x, z, c) && self.1.keep_term(x, z, c)
    }

    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        self.0.finalize_layer(sum);
        self.1.finalize_layer(sum);
    }
}

/// Logical OR of two policies — either accepting is enough.
pub struct Or<A, B>(pub A, pub B);

impl<const W: usize, A, B> TruncationPolicy<W> for Or<A, B>
where
    A: TruncationPolicy<W>,
    B: TruncationPolicy<W>,
{
    #[inline]
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
        self.0.keep_term(x, z, c) || self.1.keep_term(x, z, c)
    }
}

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;

    /// Slice 7.2: `WeightCutoff(2)` keeps weights 0, 1, 2 and drops 3.
    /// Identity I (weight 0), single X (1), XZ on qubits 0+1 (2) all kept;
    /// X on q0 + Y on q1 + Z on q2 (3) dropped.
    #[test]
    fn weight_cutoff_keeps_below_or_equal() {
        let cut = WeightCutoff(2);
        // Identity: weight 0.
        assert!(<WeightCutoff as TruncationPolicy<1>>::keep_term(
            &cut, &[0], &[0], Complex64::new(1.0, 0.0)
        ));
        // X on q0: weight 1 (x bit set).
        assert!(<WeightCutoff as TruncationPolicy<1>>::keep_term(
            &cut, &[1], &[0], Complex64::new(1.0, 0.0)
        ));
        // X on q0, Z on q1: weight 2.
        assert!(<WeightCutoff as TruncationPolicy<1>>::keep_term(
            &cut, &[0b01], &[0b10], Complex64::new(1.0, 0.0)
        ));
        // X on q0, Y on q1 (x+z), Z on q2: weight 3, dropped.
        assert!(!<WeightCutoff as TruncationPolicy<1>>::keep_term(
            &cut, &[0b011], &[0b110], Complex64::new(1.0, 0.0)
        ));
    }

    /// Slice 7.2: `WeightCutoff(0)` keeps only the identity.
    #[test]
    fn weight_cutoff_zero_keeps_only_identity() {
        let cut = WeightCutoff(0);
        assert!(<WeightCutoff as TruncationPolicy<1>>::keep_term(
            &cut, &[0], &[0], Complex64::new(1.0, 0.0)
        ));
        // Any non-identity Pauli is dropped.
        assert!(!<WeightCutoff as TruncationPolicy<1>>::keep_term(
            &cut, &[1], &[0], Complex64::new(1.0, 0.0)
        ));
        assert!(!<WeightCutoff as TruncationPolicy<1>>::keep_term(
            &cut, &[0], &[1], Complex64::new(1.0, 0.0)
        ));
        assert!(!<WeightCutoff as TruncationPolicy<1>>::keep_term(
            &cut, &[1], &[1], Complex64::new(1.0, 0.0)
        ));
    }

    /// Slice 7.2: multi-word popcount. Qubit 64 lives in word 1, bit 0.
    #[test]
    fn weight_cutoff_w2_word_boundary() {
        let cut = WeightCutoff(1);
        // X on qubit 64 alone: weight 1, kept.
        assert!(<WeightCutoff as TruncationPolicy<2>>::keep_term(
            &cut, &[0u64, 1u64], &[0u64, 0u64], Complex64::new(1.0, 0.0)
        ));
        // X on qubit 0 AND X on qubit 64: weight 2, dropped.
        assert!(!<WeightCutoff as TruncationPolicy<2>>::keep_term(
            &cut, &[1u64, 1u64], &[0u64, 0u64], Complex64::new(1.0, 0.0)
        ));
    }

    /// Slice 7.3: ten distinct keys with decreasing |coeff| (10, 9, …, 1);
    /// `TopN(3)` keeps the three with magnitudes 10, 9, 8.
    #[test]
    fn top_n_keeps_largest_three_of_ten() {
        // Ten distinct (x, z) keys: x ∈ {0..10}, z=0. Sorted by x ascending.
        let mut sum = PauliSum::<1> {
            x: (1u64..=10).map(|i| [i]).collect(),
            z: vec![[0u64]; 10],
            // Magnitudes 10, 9, 8, ... in same order. So largest sit at the
            // *front* of the sort order. We'll exercise back-loaded magnitudes
            // in a separate test.
            coeff: (1u64..=10)
                .rev()
                .map(|m| Complex64::new(m as f64, 0.0))
                .collect(),
            num_qubits: 4,
        };
        sum.assert_invariants();
        TopN(3).finalize_layer(&mut sum);
        assert_eq!(sum.len(), 3);
        // Survivors are the keys whose original magnitudes were 10, 9, 8 →
        // they sat at indices 0, 1, 2 → x = [1], [2], [3].
        assert_eq!(sum.x(), &[[1u64], [2u64], [3u64]]);
        let mags: Vec<f64> = sum.coeff().iter().map(|c| c.norm()).collect();
        assert_eq!(mags, vec![10.0, 9.0, 8.0]);
        sum.assert_invariants();
    }

    /// Slice 7.3: `TopN(N) where N >= len` is a no-op.
    #[test]
    fn top_n_no_op_when_n_ge_len() {
        let mut sum = PauliSum::<1> {
            x: vec![[0], [0], [1]],
            z: vec![[0], [1], [0]],
            coeff: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            num_qubits: 1,
        };
        let snapshot_x = sum.x().to_vec();
        let snapshot_z = sum.z().to_vec();
        let snapshot_c = sum.coeff().to_vec();
        TopN(5).finalize_layer(&mut sum);
        assert_eq!(sum.x(), snapshot_x.as_slice());
        assert_eq!(sum.z(), snapshot_z.as_slice());
        assert_eq!(sum.coeff(), snapshot_c.as_slice());
    }

    /// Slice 7.3: `TopN(0)` empties the sum.
    #[test]
    fn top_n_zero_empties_sum() {
        let mut sum = PauliSum::<1> {
            x: vec![[0], [1]],
            z: vec![[1], [0]],
            coeff: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            num_qubits: 1,
        };
        TopN(0).finalize_layer(&mut sum);
        assert!(sum.is_empty());
        sum.assert_invariants();
    }

    /// Slice 7.3: largest coefficients sit at the *end* of the sort order;
    /// the survivors must still be in (x, z) sort order, not magnitude order.
    #[test]
    fn top_n_preserves_sort_order() {
        // Five keys, magnitudes 1, 2, 3, 4, 5 (back-loaded).
        let mut sum = PauliSum::<1> {
            x: vec![[1], [2], [3], [4], [5]],
            z: vec![[0]; 5],
            coeff: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
            ],
            num_qubits: 4,
        };
        sum.assert_invariants();
        TopN(3).finalize_layer(&mut sum);
        assert_eq!(sum.len(), 3);
        // Survivors: magnitudes 5, 4, 3 — keys x=[5], [4], [3] in the
        // original. Sort-order preservation means [3], [4], [5].
        assert_eq!(sum.x(), &[[3u64], [4u64], [5u64]]);
        assert_eq!(
            sum.coeff(),
            &[
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
            ]
        );
        sum.assert_invariants();
    }

    /// `And` requires both policies to accept. Pair a coeff threshold with
    /// a weight cutoff; only terms passing *both* survive.
    #[test]
    fn and_requires_both_keep() {
        let policy = And(CoefficientThreshold(0.5), WeightCutoff(1));
        // (X, 1.0): |c|=1.0 > 0.5 ✓, weight=1 ≤ 1 ✓ → kept.
        assert!(<And<_, _> as TruncationPolicy<1>>::keep_term(
            &policy, &[1], &[0], Complex64::new(1.0, 0.0)
        ));
        // (X, 0.1): |c|=0.1 ≤ 0.5 ✗ → dropped.
        assert!(!<And<_, _> as TruncationPolicy<1>>::keep_term(
            &policy, &[1], &[0], Complex64::new(0.1, 0.0)
        ));
        // (XZ, 1.0): weight 2 > 1 ✗ → dropped.
        assert!(!<And<_, _> as TruncationPolicy<1>>::keep_term(
            &policy, &[0b01], &[0b10], Complex64::new(1.0, 0.0)
        ));
    }

    /// `Or` accepts if *either* policy accepts.
    #[test]
    fn or_keeps_if_either() {
        let policy = Or(CoefficientThreshold(0.5), WeightCutoff(0));
        // (I, 0.1): |c| fails (0.1 ≤ 0.5), but weight=0 passes → kept.
        assert!(<Or<_, _> as TruncationPolicy<1>>::keep_term(
            &policy, &[0], &[0], Complex64::new(0.1, 0.0)
        ));
        // (X, 1.0): weight fails, but |c|=1.0 > 0.5 → kept.
        assert!(<Or<_, _> as TruncationPolicy<1>>::keep_term(
            &policy, &[1], &[0], Complex64::new(1.0, 0.0)
        ));
        // (X, 0.1): both fail → dropped.
        assert!(!<Or<_, _> as TruncationPolicy<1>>::keep_term(
            &policy, &[1], &[0], Complex64::new(0.1, 0.0)
        ));
    }
}
