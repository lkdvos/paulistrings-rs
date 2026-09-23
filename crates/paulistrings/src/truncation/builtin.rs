//! Built-in truncation policies and combinators. See ARCHITECTURE.md §Truncation.

use super::{DeviceKeep, TruncationPolicy};
use crate::pauli_sum::PauliSum;
use num_complex::Complex64;
use rayon::prelude::*;
use std::cell::RefCell;

thread_local! {
    /// Reusable squared-magnitude scratch buffer for [`TopN::finalize_layer`], pooled per thread to avoid a per-layer allocation.
    /// Borrowed out with `take()` and returned after use, never held across a parallel section: rayon can work-steal a nested `propagate` onto this thread, which would re-enter `finalize_layer` and panic on a held `RefCell` borrow instead of just allocating a fresh buffer.
    /// Never shrunk, so a thread retains a buffer sized to the largest layer it has finalized.
    static MAGS: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Drop terms whose coefficient magnitude is at most `epsilon`.
///
/// Compares `|c|² > ε²` rather than `|c| > ε`: `Complex64::norm()` is a `hypot` call, and squaring is strictly increasing on `[0, ∞)` so it decides the same predicate for a few arithmetic instructions instead.
/// Two riders follow from working in squared space, both accepted rather than guarded since the correctness bar is floating-point tolerance: `|c|²` rounds to `0.0` below `|c| ≈ 1.57e-162`, so `CoefficientThreshold(0.0)` also drops magnitudes under that bound (all numerically zero); and `ε > ≈1.34e154` squares to `+∞`, keeping nothing, which agrees with the unsquared test on a finite sum.
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
        // `|c| > ε ⟺ |c|² > ε²` for ε >= 0, guarded so a negative ε still keeps everything.
        // NaN drops everything either way.
        eps < 0.0 || c.norm_sqr() > eps * eps
    }

    /// Per-term only — no layer pass.
    fn finalizes_layer(&self) -> bool {
        false
    }

    fn device_policy(&self) -> Option<DeviceKeep> {
        Some(DeviceKeep::Coeff(self.0))
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

    /// Per-term only — no layer pass.
    fn finalizes_layer(&self) -> bool {
        false
    }

    fn device_policy(&self) -> Option<DeviceKeep> {
        Some(DeviceKeep::Weight(self.0))
    }
}

/// Retain **at most** `n` terms by coefficient magnitude, never splitting a group of exactly equal magnitudes.
/// Implemented as a `finalize_layer` partial selection; there is no per-term filter.
///
/// # Semantics
///
/// Let `t` be the `n`-th largest magnitude in a sum of more than `n` terms.
/// Every term with `|c| > t` is kept, and the tie group at `|c| == t` is kept iff it fits entirely, i.e. `count(|c| > t) + count(|c| == t) <= n` — otherwise the whole group is discarded.
/// So `TopN(n)` retains exactly `n` terms when the cut lands on a group boundary (the generic case, since magnitudes are usually distinct), fewer when a group straddles the cut, and never more. `len <= n` is a no-op.
///
/// # Why whole groups
///
/// Terms related by a symmetry of the Hamiltonian carry exactly equal coefficients — a multiplet.
/// Keeping an arbitrary subset of a multiplet, which is what any tiebreak on the Pauli key does since key order has nothing to do with the symmetry, yields a truncated operator that is no longer symmetric — so the whole group is discarded instead.
/// Because the rule reads magnitudes only, the retained set is a pure function of the magnitude multiset, independent of the bucket partition, the hash seed, and the thread count.
///
/// # Ranked on `|c|²`
///
/// Selection compares `norm_sqr()` against `t²` rather than `norm()` against `t`, since `Complex64::norm()` is a `hypot` call.
/// Squaring preserves the order of finite magnitudes and preserves bitwise-equal magnitudes as bitwise-equal squares, so a symmetry multiplet stays intact; magnitudes below `|c| ≈ 1.57e-162` all square to `0.0` and so collapse into one tie group, and magnitudes above `≈1.34e154` collapse the same way at `+∞`.
///
/// # ⚠ A fully degenerate sum is wiped to empty
///
/// If every candidate ties at the threshold, truncation keeps nothing: `t` is the maximum, no term beats it, and the one tie group of size `len > n` cannot fit.
/// The alternative — keeping the group anyway — would let `TopN(n)` retain unboundedly more than `n`, destroying its purpose as a memory bound.
/// Pair `TopN` with [`CoefficientThreshold`] via [`And`], or pick `n` at least as large as the expected multiplet size, if that outcome would be wrong for your workload.
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
    /// Three `O(n)` passes and one `select_nth_unstable`: gather the squared magnitudes, select the threshold `t²` at rank `n`, count the terms above and at `t²`, then compact each bucket in place against the resulting predicate.
    /// Per-bucket filtering preserves within-bucket order automatically, so the canonical-order invariant holds with no re-sort.
    /// The predicate `|c|² > t² || (fits && |c|² == t²)` reads no keys, which is why the result is partition-independent.
    /// Exact `f64` equality against `t²` is the right test here rather than a tolerance: a symmetry multiplet's magnitudes are bitwise equal, and squaring preserves that.
    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        let n = self.0;
        if sum.len() <= n {
            return;
        }
        if n == 0 {
            sum.clear();
            return;
        }

        // `select_nth_unstable` permutes the squared magnitudes, which is fine since every later step reads this as a multiset.
        // The buffer is pooled (see `MAGS`); only `[..total]` is ever live, the tail is stale data from a previous, larger layer.
        let total = sum.len();
        let mut buf = MAGS.take();
        if buf.len() < total {
            buf.resize(total, 0.0);
        }
        let mags = &mut buf[..total];
        {
            // One `&mut [f64]` per bucket, carved off in bucket order, so the fill writes each square exactly once.
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

        // The tie group fits iff no element after the pivot equals `t2` — equivalent to `count(> t2) + count(== t2) <= n` but only reads the `len - n` suffix.
        // `top_n_matches_the_reference_rule_on_tied_magnitudes` checks the equivalence against the literal rule.
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

/// Number of bins in [`ApproxTopN`]'s histogram: one per `f64` binade, the full range of the 11-bit biased exponent.
/// `+∞` and `NaN` share the top bin; every subnormal and `0.0` share bin 0.
/// Sized so the bin counter array (`2048 × 4 B = 8 KB`) is L1-resident; adding mantissa bits for a finer threshold would push it out of L1.
pub(crate) const APPROX_BINS: usize = 2048;

/// Retain **approximately** `n` terms: at most `n`, and more than `n - p` where `p` is the population of the coarsest octave that did not fit.
/// The cheap sibling of [`TopN`] — no selection, no candidate array, no tie rule.
///
/// # Semantics
///
/// Bins every term by the octave of `|c|²` (a factor of 2 in `|c|²`, `√2` in `|c|`) and lets `S_k` be the population of bin `k` and above.
/// The kept set is `{ |c|² ≥ 2^(k*-1023) }` for the lowest `k*` with `S_k* ≤ n`, so `kept ≤ n` always, `kept > n - p` for `p` the next octave's population, and the retained set is a union of whole octaves.
/// The shortfall against `n` is thus bounded by how many terms sit inside one `√2`-wide band around the cut.
///
/// Cheaper than [`TopN`] — a histogram pass plus a compaction, no per-layer allocation, no selection — at the cost of retaining `(n - p, n]` terms instead of exactly `n`.
/// [`TopN`] remains the default and the one to use when the retained count matters; reach for this when `n` is a memory budget and a little slack in the term count is cheaper than the selection.
/// Like [`TopN`] it needs no tie rule: equal magnitudes share an octave, so a symmetry multiplet can never be split.
///
/// # ⚠ A sum inside a single octave is wiped to empty
///
/// If every magnitude lands in one octave of `|c|²` and `len > n`, nothing is kept — [`TopN`]'s all-tied wipe, with "tied" widened to a factor of `√2` in `|c|`.
/// Pair with [`CoefficientThreshold`] via [`And`], or use [`TopN`], if that outcome would be wrong for your workload.
///
/// # Examples
///
/// ```
/// use paulistrings::truncation::ApproxTopN;
/// let policy = ApproxTopN(1_000_000);
/// # let _ = policy;
/// ```
pub struct ApproxTopN(
    /// Target term count, and a hard upper bound on what is retained. Terms
    /// below the chosen octave edge are dropped at layer finalization.
    pub usize,
);

/// What [`octave_edge`] decided a layer's histogram means for the terms.
///
/// Factored out so the single-partition and partitioned paths share one decision rule; the histogram fed to it is local in one case and all-reduced in the other, and downstream is always a per-sum [`retain_at_or_above`] needing no further communication.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum EdgeDecision {
    /// The whole sum fits within `n`: nothing is dropped.
    KeepAll,
    /// Nothing fits — `n == 0`, or the top octave alone overshoots `n`.
    Clear,
    /// Keep the terms with `norm_sqr() >= threshold`.
    AtOrAbove {
        /// The lower edge of the lowest octave of `|c|²` that still fits.
        threshold: f64,
        /// How many terms survive across the whole sum (global, in partitioned mode).
        kept: usize,
    },
}

/// Population of each octave of `|c|²` over the whole sum.
///
/// The bin index is `norm_sqr().to_bits() >> 52`: the high 12 bits of a non-negative `f64`'s bit pattern are a `log₂` bucketing.
/// Accumulated over a fixed number of tasks (four per worker) so the number of accumulators is bounded by the thread count rather than by rayon's splitting.
pub(crate) fn octave_histogram<const W: usize>(sum: &PauliSum<W>) -> [u32; APPROX_BINS] {
    // A bin counter is `u32`; a sum of 2^32 terms is >100 GB of columns.
    debug_assert!(sum.len() <= u32::MAX as usize, "len exceeds bin counters");
    let nb = sum.num_buckets();
    let tasks = (rayon::current_num_threads() * 4).clamp(1, nb);
    (0..tasks)
        .into_par_iter()
        .map(|t| {
            let mut h = [0u32; APPROX_BINS];
            for b in (nb * t / tasks)..(nb * (t + 1) / tasks) {
                for c in sum.bucket(b).2 {
                    h[(c.norm_sqr().to_bits() >> 52) as usize] += 1;
                }
            }
            h
        })
        .reduce(
            || [0u32; APPROX_BINS],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += y;
                }
                a
            },
        )
}

/// The octave edge [`ApproxTopN`] retains against, from a histogram of the whole sum.
///
/// `hist` is [`octave_histogram`]'s output, locally or summed across partitions (`total_len` summed to match); generic in the counter type so an all-reduced `[u64]` needs no narrowing copy.
/// Walks down from the top bin while the running count still fits, since `S_k` is non-increasing in `k` so the first overshoot is the boundary.
pub(crate) fn octave_edge<C>(hist: &[C], total_len: usize, n: usize) -> EdgeDecision
where
    C: Copy + Into<u64>,
{
    debug_assert_eq!(hist.len(), APPROX_BINS, "octave_edge: wrong histogram size");
    if total_len <= n {
        return EdgeDecision::KeepAll;
    }
    if n == 0 {
        return EdgeDecision::Clear;
    }

    let mut kept = 0usize;
    let mut edge = APPROX_BINS;
    for bin in (0..APPROX_BINS).rev() {
        let next = kept + hist[bin].into() as usize;
        if next > n {
            break;
        }
        kept = next;
        edge = bin;
    }
    if kept == 0 {
        // Even the top octave alone overshoots `n`.
        return EdgeDecision::Clear;
    }
    EdgeDecision::AtOrAbove {
        threshold: f64::from_bits((edge as u64) << 52),
        kept,
    }
}

/// Apply an [`EdgeDecision`] to one sum — the only step that touches terms.
///
/// Per-sum and communication-free: every partition of a distributed sum applies the same decision to its own terms since the predicate reads only a term's own coefficient.
pub(crate) fn retain_at_or_above<const W: usize>(sum: &mut PauliSum<W>, edge: EdgeDecision) {
    match edge {
        EdgeDecision::KeepAll => {}
        EdgeDecision::Clear => sum.clear(),
        EdgeDecision::AtOrAbove { threshold, .. } => {
            sum.retain(|_, _, c| c.norm_sqr() >= threshold);
        }
    }
}

impl<const W: usize> TruncationPolicy<W> for ApproxTopN {
    /// Two `O(n)` passes and no selection: histogram the octaves of `|c|²` (`octave_histogram`), walk the bins down to the last edge that still fits in `n` (`octave_edge`), then [`PauliSum::retain`] against it (`retain_at_or_above`).
    /// The two early exits below skip the histogram when the term count already settles the answer; the partitioned sibling, whose global length is not known locally, goes through the histogram regardless — see [`PartitionedTruncation`](crate::PartitionedTruncation).
    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        let n = self.0;
        let total = sum.len();
        if total <= n {
            return;
        }
        if n == 0 {
            sum.clear();
            return;
        }

        let edge = octave_edge(&octave_histogram(sum), total, n);
        retain_at_or_above(sum, edge);
        if let EdgeDecision::AtOrAbove { kept, .. } = edge {
            debug_assert_eq!(sum.len(), kept, "histogram and predicate disagree");
        }
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

    /// Both sides' layer passes run, so either one wanting a layer means the
    /// composition does.
    fn finalizes_layer(&self) -> bool {
        self.0.finalizes_layer() || self.1.finalizes_layer()
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

    /// `Or` does not forward `finalize_layer` to either side (see the type
    /// docs), so it has no layer pass regardless of what its children answer.
    fn finalizes_layer(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

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

    /// A negative threshold keeps everything, including an exact zero: the squared form must not invert `|c| > ε`'s vacuous truth for `ε < 0`.
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

    /// `finalizes_layer` must agree with which builtins override `finalize_layer`: only `TopN`, plus `And` inheriting from either side; `Or` never does.
    #[test]
    fn layer_finalize_hint_matches_the_builtins() {
        assert!(
            !<CoefficientThreshold as TruncationPolicy<1>>::finalizes_layer(&CoefficientThreshold(
                1e-9
            ))
        );
        assert!(!<WeightCutoff as TruncationPolicy<2>>::finalizes_layer(
            &WeightCutoff(3)
        ));
        assert!(<TopN as TruncationPolicy<1>>::finalizes_layer(&TopN(4)));

        let cheap = And(CoefficientThreshold(1e-9), WeightCutoff(2));
        assert!(!<_ as TruncationPolicy<1>>::finalizes_layer(&cheap));
        let with_topn = And(CoefficientThreshold(1e-9), TopN(4));
        assert!(<_ as TruncationPolicy<1>>::finalizes_layer(&with_topn));
        let topn_first = And(TopN(4), WeightCutoff(2));
        assert!(<_ as TruncationPolicy<1>>::finalizes_layer(&topn_first));

        // `Or` does not forward `finalize_layer` to either side, so it has no
        // layer pass however its children answer.
        let ored = Or(CoefficientThreshold(1e-9), TopN(4));
        assert!(!<_ as TruncationPolicy<1>>::finalizes_layer(&ored));
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

    /// Ten distinct keys with decreasing |coeff| (10, 9, …, 1); `TopN(3)` keeps the three with magnitudes 10, 9, 8.
    #[test]
    fn top_n_keeps_largest_three_of_ten() {
        // Largest magnitudes sit at the front of the sort order; back-loaded magnitudes are exercised separately.
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
        // Survivors: original magnitudes 10, 9, 8 → x = [1], [2], [3].
        let (x, _, c) = sum.to_arrays();
        assert_eq!(x, vec![[1u64], [2u64], [3u64]]);
        let mags: Vec<f64> = c.iter().map(|c| c.norm()).collect();
        assert_eq!(mags, vec![10.0, 9.0, 8.0]);
        sum.assert_invariants();
    }

    /// `TopN(N) where N >= len` is a no-op, checked at both `N > len` and `N == len` since the tie rule only engages on the `len > n` path.
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

    /// With all magnitudes distinct, the tie group at rank `n` has size one and always fits, so `TopN(n)` retains exactly `n`.
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

    /// A tie group straddling the cut is discarded whole: magnitudes 5, 4, 3, 3, 3, 2 with `n = 3` keeps only 5 and 4, since the three-member group at 3 does not fit in the one remaining slot.
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

    /// A tie group that ends exactly at rank `n` fits and is kept whole: magnitudes 5, 4, 3, 3, 2, 1 with `n = 4` keeps all four of 5, 4, 3, 3.
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

    /// If every candidate ties at the threshold, the group cannot fit and the whole sum is discarded — the case documented on [`TopN`] itself.
    #[test]
    fn top_n_wipes_an_all_tied_sum_to_empty() {
        let mut sum = PauliSum::<1>::from_sorted_columns(
            (0u64..6).map(|i| [i]).collect(),
            vec![[0u64]; 6],
            // Same magnitude, different phases (fourth roots of unity so every norm is bitwise 2.0): a multiplet, not duplicates.
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

    /// Magnitudes below the square-underflow floor (`≈1.57e-162`) collapse to one tie group: six terms at 1e-200..6e-200 with `n = 3` all square to `0.0` and the group of six does not fit in three, so the sum is wiped.
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

    /// When the cut falls inside an underflowing tail, the tail is dropped whole and the representable terms are kept: magnitudes 3, 2, 1 plus five terms at 1e-200 with `n = 5` keeps only the three representable terms.
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

    /// Smoke test for `finalize_layer` called from inside a rayon job (a caller propagating several observables in parallel), where a blocked worker may re-enter `finalize_layer` via work-stealing.
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

    /// Largest coefficients sit at the end of the sort order; survivors must still come back in (x, z) sort order, not magnitude order.
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
        // Survivors: magnitudes 5, 4, 3, i.e. keys [5], [4], [3]; sort order preservation means they come back as [3], [4], [5].
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

    // -----------------------------------------------------------------
    // ApproxTopN
    // -----------------------------------------------------------------

    /// A `W = 1` sum of `mags.len()` distinct keys (`x = i`, `z = 0`) with the
    /// given real coefficients, single-bucket and already in key order.
    fn sum_of_mags(mags: &[f64]) -> PauliSum<1> {
        PauliSum::<1>::from_sorted_columns(
            (0u64..mags.len() as u64).map(|i| [i]).collect(),
            vec![[0u64]; mags.len()],
            mags.iter().map(|&m| Complex64::new(m, 0.0)).collect(),
            32,
        )
    }

    /// The surviving magnitudes, in canonical order.
    fn kept_mags<const W: usize>(sum: &PauliSum<W>) -> Vec<f64> {
        sum.iter().map(|(_, _, c)| c.norm()).collect()
    }

    /// Octave of `|c|²`, i.e. the bin `ApproxTopN` histograms into, derived
    /// here from the definition rather than from the implementation.
    fn octave(c: Complex64) -> usize {
        (c.norm_sqr().to_bits() >> 52) as usize
    }

    /// The threshold can only land on an octave boundary of `|c|²`, so the retained count is a cumulative octave population, not `n` — hand-tabulated here for the four-octave fixture below (populations 1, 2, 3, 4; cumulative 1, 3, 6, 10 from the top).
    #[test]
    fn approx_top_n_keeps_a_cumulative_octave_population() {
        let mags = [8.0f64, 4.0, 4.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0];
        for (n, want) in [
            (1usize, 1usize),
            (2, 1),
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 6),
            (7, 6),
            (8, 6),
            (9, 6),
        ] {
            let mut sum = sum_of_mags(&mags);
            ApproxTopN(n).finalize_layer(&mut sum);
            sum.assert_invariants();
            assert_eq!(sum.len(), want, "n={n}");
            // Whatever survives is the largest `want` magnitudes.
            let mut got = kept_mags(&sum);
            got.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let mut all = mags.to_vec();
            all.sort_by(|a, b| b.partial_cmp(a).unwrap());
            assert_eq!(got, all[..want].to_vec(), "n={n}");
        }
        // `n >= len` is a no-op, like every other policy's.
        let mut sum = sum_of_mags(&mags);
        ApproxTopN(10).finalize_layer(&mut sum);
        assert_eq!(sum.len(), 10);
    }

    /// When the cut lands exactly on an octave boundary (`n` = 1, 3, or 6, the fixture's cumulative populations), the approximation is no approximation: both policies return the same top `n`.
    #[test]
    fn approx_top_n_matches_top_n_when_the_histogram_resolves_exactly() {
        let mags = [8.0f64, 4.0, 4.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0];
        for n in [1usize, 3, 6] {
            let mut approx = sum_of_mags(&mags);
            ApproxTopN(n).finalize_layer(&mut approx);
            let mut exact = sum_of_mags(&mags);
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
    /// `kept <= n`, and `kept > n - p` where `p` is the population of the
    /// highest *excluded* octave. Equivalently `kept + p > n`: the next octave
    /// down would have overshot.
    #[test]
    fn approx_top_n_shortfall_is_bounded_by_one_octave() {
        let input = crate::test_support::rand_sum::<1>(3000, 10, 0xB0117);
        let len = input.len();
        for n in [1usize, 17, 200, 900, 2000, len - 1] {
            let mut sum = input.clone();
            ApproxTopN(n).finalize_layer(&mut sum);
            let kept = sum.len();
            assert!(kept <= n, "n={n}: kept {kept} exceeds the bound");

            // The highest excluded octave's population in the input is the slack.
            let survivors: std::collections::HashSet<(u64, u64)> =
                sum.iter().map(|(x, z, _)| (x[0], z[0])).collect();
            let dropped_octaves: Vec<usize> = input
                .iter()
                .filter(|(x, z, _)| !survivors.contains(&(x[0], z[0])))
                .map(|(_, _, c)| octave(c))
                .collect();
            let p = match dropped_octaves.iter().copied().max() {
                None => 0,
                Some(top) => input.iter().filter(|(_, _, c)| octave(*c) == top).count(),
            };
            assert!(
                kept + p > n,
                "n={n}: kept {kept} + excluded octave {p} must overshoot n, \
                 else that octave should have been kept"
            );
        }
    }

    /// Equal magnitudes share an octave, so `ApproxTopN` cannot split a symmetry multiplet: `tie_heavy_sum`'s magnitudes (1, ½, ¼, ⅛) each land in their own octave, so every retained set must be a union of whole magnitude groups.
    #[test]
    fn approx_top_n_never_splits_a_tie_group() {
        let input = crate::test_support::tie_heavy_sum::<1>(2000, 8, 0x7135);
        for n in [3usize, 250, 700, 1200, 1900] {
            let mut sum = input.clone();
            ApproxTopN(n).finalize_layer(&mut sum);
            sum.assert_invariants();
            for mag in [1.0f64, 0.5, 0.25, 0.125] {
                let want = input.iter().filter(|(_, _, c)| c.norm() == mag).count();
                let got = sum.iter().filter(|(_, _, c)| c.norm() == mag).count();
                assert!(
                    got == 0 || got == want,
                    "n={n}: magnitude {mag} group is split, {got} of {want} kept"
                );
            }
        }
    }

    /// One octave holding everything resolves like [`TopN`]'s all-tied sum: the bound wins and the sum is wiped.
    #[test]
    fn approx_top_n_wipes_a_single_octave_sum() {
        let mags = [1.0f64, 1.125, 1.25, 1.375];
        let mut sum = sum_of_mags(&mags);
        ApproxTopN(3).finalize_layer(&mut sum);
        assert!(
            sum.is_empty(),
            "one octave cannot be split, so nothing fits"
        );
        sum.assert_invariants();
        // …and it is a no-op at n >= len, as always.
        let mut sum = sum_of_mags(&mags);
        ApproxTopN(4).finalize_layer(&mut sum);
        assert_eq!(sum.len(), 4);
    }

    /// `ApproxTopN(0)` empties the sum.
    #[test]
    fn approx_top_n_zero_empties_sum() {
        let mut sum = sum_of_mags(&[1.0, 2.0]);
        ApproxTopN(0).finalize_layer(&mut sum);
        assert!(sum.is_empty());
        sum.assert_invariants();
    }

    /// `W = 2`: the const-generic surface, on keys that straddle the word boundary.
    #[test]
    fn approx_top_n_w2() {
        let build = || {
            PauliSum::<2>::from_sorted_columns(
                vec![[0, 1], [0, 2], [1, 0], [2, 0]],
                vec![[0, 0]; 4],
                vec![
                    Complex64::new(4.0, 0.0),
                    Complex64::new(2.0, 0.0),
                    Complex64::new(2.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
                128,
            )
        };
        let mut sum = build();
        ApproxTopN(2).finalize_layer(&mut sum);
        assert_eq!(kept_mags(&sum), vec![4.0]);
        sum.assert_invariants();

        let mut sum = build();
        ApproxTopN(3).finalize_layer(&mut sum);
        assert_eq!(kept_mags(&sum), vec![4.0, 2.0, 2.0]);
        sum.assert_invariants();
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

        // Two slots take both of the top two octaves; magnitudes come back in
        // key order, not magnitude order.
        let mut sum = build();
        ApproxTopN(2).finalize_layer(&mut sum);
        assert_eq!(kept_mags(&sum), vec![5.0, 6.0]);
        sum.assert_invariants();
    }

    proptest! {
        /// The whole contract over tie-dense magnitude multisets: `kept <= n`, the kept set is a union of whole top octaves, and the shortfall bound `kept + p > n` holds for `p` the population of the highest excluded octave.
        /// Magnitudes are small integers so squares collide into few octaves and the interesting branches are hit often.
        #[test]
        fn approx_top_n_thresholds_on_an_octave_edge(
            values in proptest::collection::vec(1u32..40u32, 1..48),
            n in 1usize..48,
        ) {
            let mags: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
            let mut sum = sum_of_mags(&mags);
            ApproxTopN(n).finalize_layer(&mut sum);

            // Key `i` carries `mags[i]`, so a key identifies its magnitude.
            let survivors: std::collections::HashSet<u64> =
                sum.iter().map(|(x, _, _)| x[0]).collect();
            let oct = |m: f64| (m * m).to_bits() >> 52;

            if mags.len() <= n {
                prop_assert_eq!(survivors.len(), mags.len(), "n >= len must be a no-op");
                return Ok(());
            }
            prop_assert!(survivors.len() <= n, "kept {} > n {}", survivors.len(), n);

            // Every kept octave is kept whole and outranks every dropped one.
            let dropped_top = mags
                .iter()
                .enumerate()
                .filter(|(i, _)| !survivors.contains(&(*i as u64)))
                .map(|(_, &m)| oct(m))
                .max();
            let kept_low = mags
                .iter()
                .enumerate()
                .filter(|(i, _)| survivors.contains(&(*i as u64)))
                .map(|(_, &m)| oct(m))
                .min();
            if let (Some(d), Some(k)) = (dropped_top, kept_low) {
                prop_assert!(d < k, "dropped octave {} is not below kept octave {}", d, k);
            }

            // Shortfall bound: including the next octave down would overshoot.
            if let Some(d) = dropped_top {
                let p = mags.iter().filter(|&&m| oct(m) == d).count();
                prop_assert!(
                    survivors.len() + p > n,
                    "kept {} + octave {} must exceed n {}",
                    survivors.len(), p, n,
                );
            }
        }
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
