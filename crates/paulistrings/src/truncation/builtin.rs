//! Built-in truncation policies and combinators. See ARCHITECTURE.md §Truncation.

use super::TruncationPolicy;
use crate::pauli_sum::PauliSum;
use crate::rng::Rng;
use num_complex::Complex64;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

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
/// use paulistrings::CoefficientThreshold;
/// let policy = CoefficientThreshold(1e-9);
/// # let _ = policy;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
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
}

/// Drop terms whose Pauli weight (number of non-identity qubits) exceeds `k`.
///
/// # Examples
///
/// ```
/// use paulistrings::WeightCutoff;
/// let policy = WeightCutoff(4);
/// # let _ = policy;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
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
/// use paulistrings::TopN;
/// let policy = TopN(1_000_000);
/// # let _ = policy;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
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
/// use paulistrings::ApproxTopN;
/// let policy = ApproxTopN(1_000_000);
/// # let _ = policy;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
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

/// `log` target for the sampling policies' per-collapse records, shared with [`propagate`](crate::propagate).
pub(crate) const LOG_TARGET: &str = "paulistrings::propagate";

/// Replace the sum by **one** Pauli string, drawn with probability `|c|² / Σ|c|²`, whenever it holds more than `cache` terms.
///
/// The survivor's coefficient is set to `1`: the sum is normalized before the draw and only the survivor's string is carried forward, so each collapse restarts growth from a single term.
/// One seed is one Monte Carlo trajectory; statistics come from many seeds.
///
/// # Reproducibility
///
/// The draw at the `k`-th layer pass is a pure function of `(seed, k)`, so a trajectory is reproducible and does not depend on the thread count.
/// The pass counter lives in the policy, so reusing one policy object across runs continues its sequence rather than repeating it.
///
/// # Partitioned
///
/// Every partition enters two reductions per collapse (the global length, then the per-partition norms), all draw the same uniform, and the one partition owning the chosen string keeps it while the rest clear.
/// At `P = 1` the pick is the unpartitioned one; across partition counts the cumulative order differs, so trajectories agree in distribution, not string for string.
/// The pass and [`collapses`](Self::collapses) counters advance on rank 0's policy object only.
///
/// # Examples
///
/// ```
/// use paulistrings::CollapseSample;
/// use paulistrings::TruncationPolicy;
/// use paulistrings::{BuildAccumulator, PauliString, Phase};
/// use num_complex::Complex64;
///
/// let mut acc = BuildAccumulator::<1>::new(2);
/// acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(0.6, 0.0));
/// acc.add_term(PauliString::<1>::z(1), Phase::ONE, Complex64::new(0.8, 0.0));
/// let mut sum = acc.finalize();
///
/// let policy = CollapseSample::new(1, 42);
/// policy.finalize_layer(&mut sum);
/// assert_eq!(sum.len(), 1);
/// assert_eq!(sum.iter().next().unwrap().2, Complex64::new(1.0, 0.0));
/// assert_eq!(policy.collapses(), 1);
/// ```
#[derive(Debug)]
pub struct CollapseSample {
    /// Largest sum left untouched; one more term triggers a collapse.
    pub cache: usize,
    /// Trajectory seed.
    pub seed: u64,
    /// Layer passes seen, the second key word of each draw.
    calls: AtomicU64,
    /// Collapses performed.
    collapses: AtomicU64,
}

impl CollapseSample {
    /// A fresh trajectory: collapse above `cache` terms, draws keyed by `seed`.
    pub fn new(cache: usize, seed: u64) -> Self {
        Self {
            cache,
            seed,
            calls: AtomicU64::new(0),
            collapses: AtomicU64::new(0),
        }
    }

    /// Collapses performed so far (rank 0's count in partitioned mode).
    pub fn collapses(&self) -> u64 {
        self.collapses.load(Ordering::Relaxed)
    }

    /// Claim the next pass index.
    pub(crate) fn next_call(&self) -> u64 {
        self.calls.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn count_collapse(&self) -> u64 {
        self.collapses.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// `u · Σ|c|²` for pass `call`, the point of the cumulative weight the pick lands on.
    ///
    /// # Panics
    ///
    /// If the total is zero or not finite, since no string can then be drawn by weight.
    pub(crate) fn target(&self, call: u64, total: f64, len: usize) -> f64 {
        assert!(
            total.is_finite() && total > 0.0,
            "CollapseSample: cannot draw by weight from {len} terms with Σ|c|² = {total}",
        );
        Rng::from_key(&[self.seed, call]).uniform() * total
    }
}

/// `Σ|c|²` of each bucket, in bucket order.
pub(crate) fn bucket_norms<const W: usize>(sum: &PauliSum<W>) -> Vec<f64> {
    (0..sum.num_buckets())
        .into_par_iter()
        .map(|b| sum.bucket(b).2.iter().map(|c| c.norm_sqr()).sum())
        .collect()
}

/// The slot whose stretch of the running total contains `target`, and `target`'s offset into that stretch.
/// A `target` that rounding carries past the end lands on the last positive slot; `None` only when no slot is positive.
pub(crate) fn pick_slot(
    weights: impl IntoIterator<Item = f64>,
    target: f64,
) -> Option<(usize, f64)> {
    let mut cum = 0.0f64;
    let mut last = None;
    for (i, w) in weights.into_iter().enumerate() {
        if w > 0.0 {
            if target < cum + w {
                return Some((i, target - cum));
            }
            last = Some((i, target - cum));
        }
        cum += w;
    }
    last
}

/// Reduce `sum` to the single string at `target` of its cumulative weight, with coefficient `1`; `norms` is [`bucket_norms`] of `sum`.
pub(crate) fn collapse_to_target<const W: usize>(
    sum: &mut PauliSum<W>,
    norms: &[f64],
    target: f64,
) {
    let (b, rest) = pick_slot(norms.iter().copied(), target)
        .expect("collapse_to_target: no bucket has positive weight");
    let (pos, _) = pick_slot(sum.bucket(b).2.iter().map(|c| c.norm_sqr()), rest)
        .expect("collapse_to_target: a positive bucket has a positive term");
    let (x, z) = (sum.bucket(b).0[pos], sum.bucket(b).1[pos]);
    sum.clear();
    sum.buckets_mut()[b].push(x, z, Complex64::new(1.0, 0.0));
    sum.recount();
}

/// Equal when the same trajectory stands at the same pass: same `cache`, `seed` and counters.
impl PartialEq for CollapseSample {
    fn eq(&self, other: &Self) -> bool {
        self.cache == other.cache
            && self.seed == other.seed
            && self.calls.load(Ordering::Relaxed) == other.calls.load(Ordering::Relaxed)
            && self.collapses() == other.collapses()
    }
}

impl<const W: usize> TruncationPolicy<W> for CollapseSample {
    /// One norms pass and one uniform: pick the bucket by its share of `Σ|c|²`, then the term within it.
    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        let call = self.next_call();
        let len = sum.len();
        if len <= self.cache {
            return;
        }
        let norms = bucket_norms(sum);
        let total: f64 = norms.iter().sum();
        collapse_to_target(sum, &norms, self.target(call, total, len));
        let n = self.count_collapse();
        log::debug!(
            target: LOG_TARGET,
            "collapse_sample: {len} terms, sum |c|^2 = {total:.6e}, collapsed to one (collapse {n})",
        );
    }
}

/// Logical AND of two policies — both must accept.
///
/// # Examples
///
/// ```
/// use paulistrings::{And, CoefficientThreshold, WeightCutoff};
/// let policy = And(CoefficientThreshold(1e-6), WeightCutoff(4));
/// # let _ = policy;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
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
/// use paulistrings::{Or, CoefficientThreshold, WeightCutoff};
/// // Keep a term if |coeff| > 0.1 OR weight == 0 (identity).
/// let policy = Or(CoefficientThreshold(0.1), WeightCutoff(0));
/// # let _ = policy;
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
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
mod tests;
