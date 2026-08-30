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

/// Retain **at most** `n` terms by coefficient magnitude, never splitting a
/// group of exactly equal magnitudes. Implemented as a `finalize_layer`
/// partial selection (no per-term filter).
///
/// # Semantics
///
/// Let `t` be the `n`-th largest magnitude in a sum of more than `n` terms.
///
/// 1. Every term with `|c| > t` is kept.
/// 2. The tie group at `|c| == t` is kept **iff it fits entirely**, i.e. iff
///    `count(|c| > t) + count(|c| == t) <= n`. Otherwise the *whole* group is
///    discarded.
///
/// So `TopN(n)` retains exactly `n` terms whenever the cut lands on a group
/// boundary — in particular whenever all magnitudes are distinct, which is the
/// generic case — and fewer than `n` when a group straddles the cut. It never
/// retains more than `n`, so the memory bound it exists to provide still
/// holds. `len <= n` is a no-op.
///
/// # Why whole groups
///
/// Terms related by a symmetry of the Hamiltonian carry *exactly* equal
/// coefficients. Keeping an arbitrary subset of such a multiplet — which is
/// what any tiebreak on the Pauli key does, since lexicographic key order has
/// nothing to do with the symmetry — yields a truncated operator that is no
/// longer symmetric. Discarding the multiplet whole keeps the symmetry intact.
/// This is not a corner case: the 2D Ising example cuts through large tie
/// groups on essentially every layer.
///
/// Because the rule reads magnitudes only, the retained *set* is a pure
/// function of the magnitude multiset — independent of the bucket partition,
/// the hash seed, and the thread count.
///
/// # ⚠ A fully degenerate sum is wiped to empty
///
/// **If every candidate ties at the threshold, truncation keeps nothing.**
/// With all magnitudes equal, `t` is the maximum, step 1 keeps zero terms, and
/// the single group of size `len > n` cannot fit — so `TopN(n)` empties the
/// sum. This is deliberate: the alternative (keeping the group anyway) would
/// let `TopN(n)` retain unboundedly more than `n`, destroying its purpose as a
/// memory bound. Pair `TopN` with [`CoefficientThreshold`] via [`And`], or
/// pick `n` at least as large as the expected multiplet size, if that outcome
/// would be wrong for your workload.
///
/// # Examples
///
/// ```
/// use paulistrings::truncation::TopN;
/// let policy = TopN(1_000_000);
/// # let _ = policy;
/// ```
pub struct TopN(
    /// Upper bound on the number of terms to retain. Terms below the
    /// magnitude threshold — and any tie group at the threshold that does not
    /// fit within the bound — are dropped at layer finalization.
    pub usize,
);

impl<const W: usize> TruncationPolicy<W> for TopN {
    /// Bucket-native top-`n` selection; see the type docs for the rule.
    ///
    /// Three `O(n)` passes and one `select_nth_unstable`: gather the
    /// magnitudes, select the threshold `t` at rank `n`, count the terms above
    /// and at `t`, then compact each bucket in place against the resulting
    /// predicate. Per-bucket filtering preserves within-bucket order
    /// automatically, so the canonical-order invariant holds with no re-sort.
    /// At a single bucket this degenerates to a plain partial selection over
    /// the lex-sorted sum.
    ///
    /// The predicate is `|c| > t || (fits && |c| == t)`, so no keys are read
    /// during selection or compaction at all — which is both why the result is
    /// partition-independent and why there is no comparator to pay for on
    /// tie-dense data.
    ///
    /// Exact `f64` equality against `t` is the right test here rather than a
    /// tolerance: a symmetry multiplet's magnitudes are bitwise equal, having
    /// been produced by the same arithmetic. Two coefficients that merely
    /// round to nearby values are different magnitudes and are ranked as such.
    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        let n = self.0;
        if sum.len() <= n {
            return;
        }
        if n == 0 {
            sum.clear();
            return;
        }

        // Magnitudes only; `select_nth_unstable` permutes them, which is fine
        // because every later step reads this as a multiset.
        let nb = sum.num_buckets();
        let view = &*sum;
        let mut mags: Vec<f64> = (0..nb)
            .into_par_iter()
            .flat_map_iter(|b| view.bucket(b).2.iter().map(|c| c.norm()))
            .collect();

        // `t` = the n-th largest magnitude.
        mags.select_nth_unstable_by(n - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal)
        });
        let t = mags[n - 1];

        // One pass for both counts. `count_gt < n <= count_gt + count_eq` by
        // construction, so the group fits exactly when the sum equals `n`; the
        // inequality is written out anyway because it is the stated rule.
        let (count_gt, count_eq) = mags
            .par_iter()
            .map(|&m| (usize::from(m > t), usize::from(m == t)))
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        let keep_tied = count_gt + count_eq <= n;

        // Compaction is per-bucket and in place, so it parallelizes directly.
        sum.buckets_mut().par_iter_mut().for_each(|cols| {
            let len = cols.len();
            let mut write = 0usize;
            for i in 0..len {
                let m = cols.coeff[i].norm();
                if !(m > t || (keep_tied && m == t)) {
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
    ///
    /// Checked at `N > len` *and* at `N == len`, because the tie rule only
    /// engages on the `len > n` path — an all-tied sum of exactly `n` terms
    /// must survive untouched rather than being wiped.
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

    /// §3: with all magnitudes distinct, the tie group at rank `n` has size
    /// one and therefore always fits — so `TopN(n)` retains **exactly** `n`,
    /// exactly as it did before the tie rule landed. This is the property that
    /// keeps every generic workload's behaviour unchanged.
    #[test]
    fn top_n_all_distinct_retains_exactly_n() {
        let mags = [7.0f64, 1.0, 5.0, 3.0, 9.0, 2.0, 8.0, 4.0];
        let mut sum = PauliSum::<1>::from_sorted_columns(
            (0u64..8).map(|i| [i]).collect(),
            vec![[0u64]; 8],
            mags.iter().map(|&m| Complex64::new(m, 0.0)).collect(),
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

    /// §3: a tie group straddling the cut is discarded **whole**, so fewer than
    /// `n` terms survive and the retained count equals `count(|c| > t)`.
    ///
    /// Magnitudes 5, 4, 3, 3, 3, 2 with `n = 3`: the threshold `t` is 3, two
    /// terms beat it, and the three-member group at 3 does not fit in the one
    /// remaining slot. Splitting it would keep an arbitrary member — the
    /// symmetry-breaking this rule exists to prevent.
    #[test]
    fn top_n_discards_a_straddling_tie_group_entirely() {
        let mags = [5.0f64, 4.0, 3.0, 3.0, 3.0, 2.0];
        let mut sum = PauliSum::<1>::from_sorted_columns(
            (0u64..6).map(|i| [i]).collect(),
            vec![[0u64]; 6],
            mags.iter().map(|&m| Complex64::new(m, 0.0)).collect(),
            3,
        );
        sum.assert_invariants();
        TopN(3).finalize_layer(&mut sum);
        assert_eq!(sum.len(), 2, "straddling group must be dropped whole");
        let (x, _, c) = sum.to_arrays();
        assert_eq!(x, vec![[0u64], [1u64]]);
        let got: Vec<f64> = c.iter().map(|c| c.norm()).collect();
        assert_eq!(got, vec![5.0, 4.0]);
        assert!(
            !got.contains(&3.0),
            "no member of the straddling group may survive"
        );
        sum.assert_invariants();
    }

    /// §3: a tie group that ends exactly at rank `n` fits, so it is kept in
    /// full and exactly `n` terms survive.
    ///
    /// Magnitudes 5, 4, 3, 3, 2, 1 with `n = 4`: `t = 3`, `count(> t) = 2`,
    /// `count(== t) = 2`, and `2 + 2 <= 4`.
    #[test]
    fn top_n_keeps_a_tie_group_that_fits_exactly() {
        let mags = [5.0f64, 4.0, 3.0, 3.0, 2.0, 1.0];
        let mut sum = PauliSum::<1>::from_sorted_columns(
            (0u64..6).map(|i| [i]).collect(),
            vec![[0u64]; 6],
            mags.iter().map(|&m| Complex64::new(m, 0.0)).collect(),
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

    /// §3, the loud edge case: if **every** candidate ties at the threshold,
    /// the group cannot fit and the whole sum is discarded. Documented on
    /// [`TopN`] itself; pinned here so it can never regress silently.
    #[test]
    fn top_n_wipes_an_all_tied_sum_to_empty() {
        let mut sum = PauliSum::<1>::from_sorted_columns(
            (0u64..6).map(|i| [i]).collect(),
            vec![[0u64]; 6],
            // Same magnitude, different phases: a multiplet, not duplicates.
            // Phases are the fourth roots of unity so every norm is *bitwise*
            // 2.0 — `from_polar` at a generic angle would not be.
            vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(0.0, 2.0),
                Complex64::new(0.0, -2.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(-2.0, 0.0),
            ],
            3,
        );
        sum.assert_invariants();
        TopN(3).finalize_layer(&mut sum);
        assert!(
            sum.is_empty(),
            "an all-tied sum is wiped: t is the maximum, nothing exceeds it, \
             and the single group of size 6 does not fit in 3"
        );
        sum.assert_invariants();
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
