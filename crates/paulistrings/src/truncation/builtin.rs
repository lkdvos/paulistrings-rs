//! Built-in truncation policies and combinators. See design doc §7.

#![allow(unused)]

use super::TruncationPolicy;
use crate::pauli_sum::PauliSum;
use num_complex::Complex64;
use rayon::prelude::*;

/// Drop terms whose coefficient magnitude is at most `epsilon`.
///
/// # Examples
///
/// ```
/// use paulistrings::truncation::CoefficientThreshold;
/// let policy = CoefficientThreshold(1e-9);
/// # let _ = policy;
/// ```
pub struct CoefficientThreshold(
    /// Magnitude threshold. Terms with `|coeff| <= epsilon` are dropped.
    pub f64,
);

impl<const W: usize> TruncationPolicy<W> for CoefficientThreshold {
    #[inline]
    fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], c: Complex64) -> bool {
        c.norm() > self.0
    }
}

/// Drop terms whose Pauli weight (number of non-identity qubits) exceeds `k`.
///
/// # Examples
///
/// ```
/// use paulistrings::truncation::WeightCutoff;
/// let policy = WeightCutoff(4);
/// # let _ = policy;
/// ```
pub struct WeightCutoff(
    /// Maximum allowed Pauli weight. Terms with weight `> k` are dropped.
    pub u32,
);

impl<const W: usize> TruncationPolicy<W> for WeightCutoff {
    #[inline]
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], _c: Complex64) -> bool {
        let weight: u32 = (0..W).map(|i| (x[i] | z[i]).count_ones()).sum();
        weight <= self.0
    }
}

/// Retain only the `n` terms with largest coefficient magnitude. Implemented
/// as a `finalize_layer` partial sort (no per-term filter).
///
/// # Examples
///
/// ```
/// use paulistrings::truncation::TopN;
/// let policy = TopN(1_000_000);
/// # let _ = policy;
/// ```
pub struct TopN(
    /// Number of terms to retain. Terms outside the top-`n` by magnitude
    /// are dropped at layer finalization.
    pub usize,
);

impl<const W: usize> TruncationPolicy<W> for TopN {
    /// Bucket-native top-`n` selection.
    ///
    /// One `O(n)` magnitude scan, one `select_nth_unstable` for the threshold,
    /// then an in-place compaction per bucket. Per-bucket filtering preserves
    /// within-bucket order automatically, so the canonical-order invariant
    /// holds with no re-sort. At a single bucket this degenerates to a plain
    /// partial selection over the lex-sorted sum.
    ///
    /// # Tie-breaking
    ///
    /// `|coeff|` descending, then **key** ascending — a total order (keys are
    /// globally unique), so the retained set is well defined even when the cut
    /// falls inside a group of exactly equal magnitudes, and it is independent
    /// of the partition (bucket position never enters the comparison). Ties
    /// are not a corner case: any symmetric Hamiltonian gives many terms
    /// exactly equal coefficients by lattice symmetry, so `TopN` routinely
    /// cuts through a large tie group. The `then_with` is lazy, so the key
    /// comparison costs nothing when magnitudes differ.
    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        let n = self.0;
        if sum.len() <= n {
            return;
        }
        if n == 0 {
            sum.clear();
            return;
        }

        // Flat numbering over buckets, so a survivor can be marked in O(1).
        let nb = sum.num_buckets();
        let mut offsets: Vec<usize> = Vec::with_capacity(nb + 1);
        let mut total = 0usize;
        for b in 0..nb {
            offsets.push(total);
            total += sum.bucket_len(b);
        }
        offsets.push(total);

        // (norm, bucket, index-within-bucket).
        //
        // Built in parallel. The comparator below is a *total* order -- norm,
        // then key, and keys are globally unique -- so the order in which
        // `ranked` is assembled cannot affect which terms survive. Rayon's
        // `collect` is order-preserving anyway, but the correctness argument does
        // not rely on that.
        let ranked_per_bucket: Vec<Vec<(f64, u32, u32)>> = (0..nb)
            .into_par_iter()
            .map(|b| {
                let (_, _, coeff) = sum.bucket(b);
                coeff
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (c.norm(), b as u32, i as u32))
                    .collect()
            })
            .collect();
        let mut ranked: Vec<(f64, u32, u32)> = Vec::with_capacity(total);
        for chunk in ranked_per_bucket {
            ranked.extend_from_slice(&chunk);
        }

        {
            let view = &*sum;
            ranked.select_nth_unstable_by(n - 1, |a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then_with(|| {
                        let (ax, az, _) = view.bucket(a.1 as usize);
                        let (bx, bz, _) = view.bucket(b.1 as usize);
                        let ai = a.2 as usize;
                        let bi = b.2 as usize;
                        (&ax[ai], &az[ai]).cmp(&(&bx[bi], &bz[bi]))
                    })
            });
        }
        ranked.truncate(n);

        let mut keep = vec![false; total];
        for &(_, b, i) in ranked.iter() {
            keep[offsets[b as usize] + i as usize] = true;
        }

        // Compaction is per-bucket and in place, so it parallelizes directly.
        let keep_ref = &keep;
        let offsets_ref = &offsets;
        sum.buckets_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(b, cols)| {
                let base = offsets_ref[b];
                let len = cols.len();
                let mut write = 0usize;
                for i in 0..len {
                    if !keep_ref[base + i] {
                        continue;
                    }
                    cols.x[write] = cols.x[i];
                    cols.z[write] = cols.z[i];
                    cols.coeff[write] = cols.coeff[i];
                    write += 1;
                }
                cols.x.truncate(write);
                cols.z.truncate(write);
                cols.coeff.truncate(write);
            });
        sum.recount();
    }
}

/// Logical AND of two policies — both must accept.
///
/// # Examples
///
/// ```
/// use paulistrings::truncation::{And, CoefficientThreshold, WeightCutoff};
/// let policy = And(CoefficientThreshold(1e-6), WeightCutoff(4));
/// # let _ = policy;
/// ```
pub struct And<A, B>(
    /// First policy. `keep_term` and `finalize_layer` both consult this first.
    pub A,
    /// Second policy.
    pub B,
);

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
///
/// Only `keep_term` is combined disjunctively; `finalize_layer` falls through
/// to the trait default (no-op) because the layer-finalization semantics of
/// "either policy's finalize pass" are not well-defined.
///
/// # Examples
///
/// ```
/// use paulistrings::truncation::{Or, CoefficientThreshold, WeightCutoff};
/// // Keep a term if |coeff| > 0.1 OR weight == 0 (identity).
/// let policy = Or(CoefficientThreshold(0.1), WeightCutoff(0));
/// # let _ = policy;
/// ```
pub struct Or<A, B>(
    /// First policy.
    pub A,
    /// Second policy.
    pub B,
);

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

    /// Slice 7.2: `WeightCutoff(0)` keeps only the identity.
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

    /// Slice 7.2: multi-word popcount. Qubit 64 lives in word 1, bit 0.
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

    /// Slice 7.3: ten distinct keys with decreasing |coeff| (10, 9, …, 1);
    /// `TopN(3)` keeps the three with magnitudes 10, 9, 8.
    #[test]
    fn top_n_keeps_largest_three_of_ten() {
        // Ten distinct (x, z) keys: x ∈ {0..10}, z=0. Sorted by x ascending.
        // Magnitudes 10, 9, 8, ... in same order, so the largest sit at the
        // *front* of the sort order; back-loaded magnitudes are exercised in a
        // separate test.
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
        // Survivors are the keys whose original magnitudes were 10, 9, 8 →
        // they sat at indices 0, 1, 2 → x = [1], [2], [3].
        let (x, _, c) = sum.to_arrays();
        assert_eq!(x, vec![[1u64], [2u64], [3u64]]);
        let mags: Vec<f64> = c.iter().map(|c| c.norm()).collect();
        assert_eq!(mags, vec![10.0, 9.0, 8.0]);
        sum.assert_invariants();
    }

    /// Slice 7.3: `TopN(N) where N >= len` is a no-op.
    #[test]
    fn top_n_no_op_when_n_ge_len() {
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
        assert_eq!(sum.to_arrays(), (snapshot_x, snapshot_z, snapshot_c));
    }

    /// Slice 7.3: `TopN(0)` empties the sum.
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

    /// Slice 7.3: largest coefficients sit at the *end* of the sort order;
    /// the survivors must still be in (x, z) sort order, not magnitude order.
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
        // Survivors: magnitudes 5, 4, 3 — keys x=[5], [4], [3] in the
        // original. Sort-order preservation means [3], [4], [5].
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

    /// `And` requires both policies to accept. Pair a coeff threshold with
    /// a weight cutoff; only terms passing *both* survive.
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
        let policy = Or(CoefficientThreshold(0.5), WeightCutoff(0));
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
}
