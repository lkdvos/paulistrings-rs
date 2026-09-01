//! Built-in truncation policies and combinators. See ARCHITECTURE.md §Truncation.

use super::TruncationPolicy;
use crate::pauli_sum::PauliSum;
use num_complex::Complex64;
use rayon::prelude::*;
use std::cell::RefCell;

thread_local! {
    /// Reusable squared-magnitude buffer for [`TopN::finalize_layer`], the
    /// one place in the crate that needs an array of the whole layer's
    /// coefficients. At `m = 1.5e6` terms a fresh one is 12 MB of allocation,
    /// first-touch page faults and (via rayon's unindexed `collect`) a double
    /// write — **per layer**, against a `finalize_layer` already measured at
    /// 61-71% of layer wall time
    /// (`research/notes/2026-09-01-large-m-phase-breakdown.md` §6).
    ///
    /// `finalize_layer` runs on whichever thread drives `propagate`, so this
    /// is one buffer per such thread. It is **borrowed out with `take()` and
    /// returned at the end**, never held across the parallel sections: rayon
    /// work-steals on a blocked thread, so a nested `propagate` (several
    /// observables under one `par_iter`) can re-enter `finalize_layer` on this
    /// very thread. Re-entering then finds an empty slot and allocates, which
    /// is correct and merely unoptimized; a held `RefCell` borrow would
    /// instead panic.
    ///
    /// The buffer is never shrunk, so a thread retains 8 B per term of the
    /// largest layer it has ever finalized until it exits — small against the
    /// ~100 B/term the sum itself costs, and it is the point of the cache.
    static MAGS: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Drop terms whose coefficient magnitude is at most `epsilon`.
///
/// # The test is `|c|² > ε²`, not `|c| > ε`
///
/// `Complex64::norm()` is `hypot`, a libm call that measured at **11.8–14.4
/// ns per merged term** on the reference host — nearly doubling the cost of
/// the merge phase all by itself
/// (`research/notes/2026-09-01-large-m-phase-breakdown.md` §6). Since
/// `x ↦ x²` is strictly increasing on `[0, ∞)`, comparing squares decides
/// the same predicate for a few arithmetic instructions instead, and
/// `norm_sqr()` is `re·re + im·im`.
///
/// Two riders follow from working in squared space, both accepted rather
/// than guarded (the correctness bar is floating-point tolerance):
///
/// - **Underflow.** `|c|² ` loses precision below `|c| ≈ 1.49e-154` and
///   rounds to `0.0` below `|c| ≈ 1.57e-162`. So `CoefficientThreshold(0.0)`
///   — "drop only the exact zeros" — also drops magnitudes under
///   `≈1.57e-162`, and any `ε` in the subnormal-square band resolves
///   coarsely. Every such coefficient is numerically zero.
/// - **Overflow.** `ε > ≈1.34e154` squares to `+∞`, so nothing is kept; the
///   unsquared test kept only `|c| > ε`, which for a finite sum is also
///   nothing.
///
/// A negative `ε` still keeps everything, as `|c| > ε` does.
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
        let eps = self.0;
        // `|c| > ε ⟺ |c|² > ε²` for ε >= 0; see the type docs for why the
        // squared form is the one that ships. A negative ε keeps everything,
        // which squaring would otherwise invert. Both `eps` and the loop-
        // invariant `eps * eps` hoist out of the merge's inner loop: `&self`
        // is `noalias readonly` and `CoefficientThreshold` has no interior
        // mutability. NaN drops everything either way.
        eps < 0.0 || c.norm_sqr() > eps * eps
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
/// # Ranked on `|c|²`
///
/// The implementation never computes `|c|`: it selects and compares
/// `norm_sqr()` against `t²`, because `Complex64::norm()` is `hypot` and
/// `finalize_layer` called it **twice per candidate** — ~24 of its ~56
/// ns/term (`research/notes/2026-09-01-large-m-phase-breakdown.md` §6).
/// Squaring preserves the order of finite magnitudes, so ranks 1..n are the
/// same ranks; what shifts slightly is which magnitudes count as *equal*,
/// i.e. the boundaries of the tie group:
///
/// - A symmetry multiplet — the thing the tie rule exists for — stays intact.
///   Its members differ by a sign or a power of `i`, and `re² + im²` is
///   invariant under both (negation is exact, and swapping the two squares
///   is exact because addition commutes), so bitwise-equal magnitudes remain
///   bitwise-equal squares.
/// - Two magnitudes that differ by ~1 ulp *may* square to the same `f64` and
///   so be treated as one group, or (with different `re`/`im` splits) tie
///   under `hypot` yet differ by an ulp when squared. Both are inside
///   floating-point tolerance and neither can break the `≤ n` bound: `t²` is
///   the `n`-th largest square, so `count(> t²) ≤ n - 1` by construction.
/// - **Underflow.** `|c|² ` rounds to `0.0` below `|c| ≈ 1.57e-162` (and is
///   subnormal below `≈1.49e-154`), so magnitudes in that band collapse into
///   one tie group. Consequences: a sum whose magnitudes *all* underflow is
///   wiped (it is zero to any tolerance — see the degenerate case below), and
///   a cut landing inside an underflowing *tail* drops that tail whole and
///   keeps the representable terms, which is the better of the two outcomes.
///   Overflow (`|c| > ≈1.34e154`) collapses the same way at `+∞`.
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
    /// Three `O(n)` passes and one `select_nth_unstable`: gather the squared
    /// magnitudes, select the threshold `t²` at rank `n`, count the terms
    /// above and at `t²`, then compact each bucket in place against the
    /// resulting predicate. Per-bucket filtering preserves within-bucket order
    /// automatically, so the canonical-order invariant holds with no re-sort.
    /// At a single bucket this degenerates to a plain partial selection over
    /// the lex-sorted sum.
    ///
    /// The predicate is `|c|² > t² || (fits && |c|² == t²)`, so no keys are
    /// read during selection or compaction at all — which is both why the
    /// result is partition-independent and why there is no comparator to pay
    /// for on tie-dense data. Squares rather than magnitudes because `norm()`
    /// is `hypot`; see the type docs' "Ranked on `|c|²`".
    ///
    /// Exact `f64` equality against `t²` is the right test here rather than a
    /// tolerance: a symmetry multiplet's magnitudes are bitwise equal, having
    /// been produced by the same arithmetic, and squaring preserves that.
    /// Two coefficients that merely round to nearby values are different
    /// magnitudes and are ranked as such.
    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        let n = self.0;
        if sum.len() <= n {
            return;
        }
        if n == 0 {
            sum.clear();
            return;
        }

        // Squared magnitudes only; `select_nth_unstable` permutes them, which
        // is fine because every later step reads this as a multiset. The
        // buffer is borrowed out of the thread-local pool rather than
        // allocated per layer (see `MAGS`), and only `[..total]` of it is
        // ever live — the tail is the previous, larger layer's stale data.
        let total = sum.len();
        let mut buf = MAGS.take();
        if buf.len() < total {
            buf.resize(total, 0.0);
        }
        let mags = &mut buf[..total];
        {
            // One `&mut [f64]` per bucket, carved off in bucket order, so the
            // fill writes each square exactly once. (`collect()` on an
            // unindexed parallel iterator cannot: rayon has to build a
            // per-thread `Vec` per split and concatenate, which writes every
            // magnitude twice and reallocates as it grows.) The handle vector
            // is 16 B per bucket against the buffer's 8 B per *term*, i.e.
            // ~1/500 of it at the default 1024-term bucket target.
            let view = &*sum;
            let nb = view.num_buckets();
            let mut handles: Vec<&mut [f64]> = Vec::with_capacity(nb);
            let mut rest: &mut [f64] = mags;
            for b in 0..nb {
                let (head, tail) = rest.split_at_mut(view.bucket_len(b));
                handles.push(head);
                rest = tail;
            }
            debug_assert!(rest.is_empty(), "bucket lengths must sum to len()");
            handles.into_par_iter().enumerate().for_each(|(b, dst)| {
                for (d, c) in dst.iter_mut().zip(view.bucket(b).2.iter()) {
                    *d = c.norm_sqr();
                }
            });
        }

        // `t2` = the n-th largest squared magnitude.
        mags.select_nth_unstable_by(n - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal)
        });
        let t2 = mags[n - 1];

        // The tie group fits **iff no element after the pivot equals `t2`**,
        // which is the stated `count(> t2) + count(== t2) <= n` rule with the
        // selection's own partition substituted in. Writing `e_pre`/`e_suf`
        // for the count of elements equal to `t2` before/after index `n - 1`:
        // everything before is `>= t2` and everything after is `<= t2`, so
        // `count_gt = (n - 1) - e_pre` and `count_eq = e_pre + 1 + e_suf`,
        // whose sum is `n + e_suf`. So the two forms agree exactly, and this
        // one reads the `len - n` suffix instead of all `len` elements and
        // stops at the first tie it finds. `top_n_matches_the_reference_rule_
        // on_tied_magnitudes` checks the equivalence against the literal rule
        // across straddling and fitting cuts.
        let keep_tied = !mags[n..].par_iter().any(|&m| m == t2);

        MAGS.set(buf);

        // Compaction is per-bucket and in place, so it parallelizes directly.
        sum.buckets_mut().par_iter_mut().for_each(|cols| {
            let len = cols.len();
            let mut write = 0usize;
            for i in 0..len {
                let m = cols.coeff[i].norm_sqr();
                if !(m > t2 || (keep_tied && m == t2)) {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `CoefficientThreshold` compares `|c|²` against `ε²`, so a magnitude
    /// whose *square* underflows to zero is indistinguishable from an exact
    /// zero. At `ε = 0` — "drop only the exact zeros" — that means every
    /// magnitude below `≈1.57e-162` is dropped as well.
    ///
    /// `(1e-100)² = 1e-200` is a normal `f64` and survives; `(1e-200)² ` is
    /// below the smallest subnormal (`4.94e-324`) and rounds to `0.0`, which
    /// is not `> 0.0`.
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
        // An exact zero is dropped at ε = 0, exactly as it was before.
        assert!(!<CoefficientThreshold as TruncationPolicy<1>>::keep_term(
            &policy,
            &[1],
            &[0],
            Complex64::new(0.0, 0.0)
        ));
    }

    /// A negative threshold keeps *everything*, including an exact zero —
    /// `|c| > ε` is vacuously true for `ε < 0` and the squared form must not
    /// silently invert that (`ε² > 0` would drop the zero).
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

    /// multi-word popcount. Qubit 64 lives in word 1, bit 0.
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

    /// ten distinct keys with decreasing |coeff| (10, 9, …, 1);
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

    /// `TopN(N) where N >= len` is a no-op.
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

    /// With all magnitudes distinct, the tie group at rank `n` has size one
    /// and therefore always fits — so `TopN(n)` retains **exactly** `n`. This
    /// is the property that keeps every generic workload's behaviour matching
    /// a naive top-`n` selection.
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

    /// A tie group straddling the cut is discarded **whole**, so fewer than
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

    /// A tie group that ends exactly at rank `n` fits, so it is kept in
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

    /// The loud edge case: if **every** candidate ties at the threshold,
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

    /// Selection ranks on `|c|²`, so magnitudes below the square-underflow
    /// floor (`≈1.57e-162`) all collapse to `0.0` and become **one tie
    /// group** however distinct they were.
    ///
    /// Magnitudes 1e-200, 2e-200, …, 6e-200 with `n = 3`: every square is
    /// `0.0`, so `t² = 0`, nothing exceeds it, and the single group of six
    /// does not fit in three — the documented all-tied wipe. Such a sum is
    /// zero to any tolerance, which is why this is accepted rather than
    /// guarded.
    #[test]
    fn top_n_wipes_a_sum_whose_squares_all_underflow() {
        let mut sum = PauliSum::<1>::from_sorted_columns(
            (0u64..6).map(|i| [i]).collect(),
            vec![[0u64]; 6],
            (1..=6)
                .map(|i| Complex64::new(i as f64 * 1e-200, 0.0))
                .collect(),
            3,
        );
        sum.assert_invariants();
        TopN(3).finalize_layer(&mut sum);
        assert!(
            sum.is_empty(),
            "squares all underflow to 0.0, so the whole sum is one tie group"
        );
        sum.assert_invariants();
    }

    /// The useful half of the same edge: when the cut falls *inside* an
    /// underflowing tail, the tail is dropped whole and the representable
    /// terms are kept — a strictly better outcome than padding the result
    /// with numerical noise.
    ///
    /// Magnitudes 3, 2, 1 and then five terms at 1e-200 with `n = 5`: the
    /// 5th largest square is `0.0`, `count(> 0) = 3`, `count(== 0) = 5`, and
    /// `3 + 5 > 5`, so the underflow group is discarded whole and exactly the
    /// three representable terms survive.
    #[test]
    fn top_n_drops_an_underflowing_tail_and_keeps_the_rest() {
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
        sum.assert_invariants();
        TopN(5).finalize_layer(&mut sum);
        assert_eq!(sum.len(), 3, "the underflowing tail must go whole");
        let (x, _, c) = sum.to_arrays();
        assert_eq!(x, vec![[0u64], [1u64], [2u64]]);
        assert_eq!(
            c,
            vec![
                Complex64::new(3.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.0),
            ]
        );
        sum.assert_invariants();
    }

    /// The squared-magnitude buffer is pooled per thread and never shrunk, so
    /// a *smaller* sum finalized after a larger one must read only its own
    /// `[..len]` prefix. This is the guard for that: the second sum's
    /// magnitudes all sit below the first sum's threshold, so a stale tail
    /// leaking into the selection would pick `t2` from the previous layer and
    /// wipe the second sum instead of truncating it.
    ///
    /// Both calls run on the test's own thread, in order, which is exactly
    /// the reuse pattern a Trotter driver produces.
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

    /// `finalize_layer` must work when it is itself called from inside a
    /// rayon job — the shape a caller propagating several observables in
    /// parallel produces. Its own `par_iter` sections then run nested, and a
    /// blocked worker may steal a sibling task, re-entering `finalize_layer`
    /// on a thread that is already inside one.
    ///
    /// This cannot *force* that interleaving, so it is a smoke test, not a
    /// proof: what it does pin is that the magnitude buffer is pooled in a
    /// form that tolerates it (borrowed out with `take`, never held as a live
    /// `RefCell` borrow across a parallel section, which would panic).
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

    /// largest coefficients sit at the *end* of the sort order;
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
