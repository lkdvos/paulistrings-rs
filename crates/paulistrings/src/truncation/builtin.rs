//! The built-in truncation policies and combinators, and the helpers their partitioned and device forms share.

use super::TruncationPolicy;
use crate::pauli_sum::PauliSum;
use crate::rng::Rng;
use num_complex::Complex64;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! {
    /// [`TopN`]'s per-thread squared-magnitude buffer.
    // Taken out and put back, never borrowed across a parallel section: rayon can steal a nested `propagate` onto this thread and re-enter `finalize_layer`.
    static MAGS: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

/// Drop terms whose coefficient magnitude is at most `epsilon`.
///
/// The test is `|c|² > ε²`, so magnitudes below about `1.6e-162` count as zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoefficientThreshold(
    /// Magnitude threshold. Terms with `|coeff| <= epsilon` are dropped.
    pub f64,
);

impl<const W: usize> TruncationPolicy<W> for CoefficientThreshold {
    #[inline]
    fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], c: Complex64) -> bool {
        let eps = self.0;
        eps < 0.0 || c.norm_sqr() > eps * eps
    }

    fn finalizes_layer(&self) -> bool {
        false
    }
}

/// Drop terms whose Pauli weight (number of non-identity qubits) exceeds `k`.
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

    fn finalizes_layer(&self) -> bool {
        false
    }
}

/// Retain at most `n` terms by coefficient magnitude, never splitting a group of equal magnitudes.
///
/// With `t` the `n`-th largest magnitude, every term above `t` is kept and the group at `t` is kept only if it fits whole, since equal magnitudes are usually a symmetry multiplet.
/// A sum whose magnitudes all tie is therefore wiped to empty; pair with [`CoefficientThreshold`] via [`And`] if that would be wrong.
/// Magnitudes are compared squared, so those below about `1.6e-162` form one tie group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TopN(
    /// Upper bound on the retained term count.
    pub usize,
);

impl<const W: usize> TruncationPolicy<W> for TopN {
    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        let n = self.0;
        if sum.len() <= n {
            return;
        }
        if n == 0 {
            sum.clear();
            return;
        }

        let total = sum.len();
        let mut buf = MAGS.take();
        if buf.len() < total {
            buf.resize(total, 0.0);
        }
        let mags = &mut buf[..total];
        {
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

        mags.select_nth_unstable_by(n - 1, |a, b| {
            b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal)
        });
        let t2 = mags[n - 1];

        // The tie group fits iff nothing after the pivot equals `t2`.
        let keep_tied = !mags[n..].par_iter().any(|&m| m == t2);

        MAGS.set(buf);

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

/// Bins of [`ApproxTopN`]'s histogram, one per `f64` exponent.
pub(crate) const APPROX_BINS: usize = 2048;

/// Retain at most `n` terms, cutting at an octave of `|c|²` instead of selecting exactly.
///
/// The kept set is every octave of `|c|²` from the top down while the running count fits in `n`, so it falls short of `n` by less than the population of the first octave that does not fit.
/// Equal magnitudes share an octave and are never split; a sum inside a single octave is wiped to empty, as an all-tied sum is under [`TopN`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApproxTopN(
    /// Upper bound on the retained term count.
    pub usize,
);

/// [`octave_edge`]'s decision, shared by the host, partitioned and device paths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum EdgeDecision {
    /// The whole sum fits within `n`: nothing is dropped.
    KeepAll,
    /// Nothing fits — `n == 0`, or the top octave alone overshoots `n`.
    Clear,
    /// Keep the terms with `norm_sqr() >= threshold`.
    AtOrAbove {
        /// Lower edge of the lowest octave of `|c|²` that fits.
        threshold: f64,
        /// Terms kept across the whole sum.
        kept: usize,
    },
}

/// Population of each octave of `|c|²`, binned by the exponent bits `norm_sqr().to_bits() >> 52`.
pub(crate) fn octave_histogram<const W: usize>(sum: &PauliSum<W>) -> [u32; APPROX_BINS] {
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

/// The octave edge [`ApproxTopN`] retains against, from a histogram of the whole sum, local or all-reduced.
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
        return EdgeDecision::Clear;
    }
    EdgeDecision::AtOrAbove {
        threshold: f64::from_bits((edge as u64) << 52),
        kept,
    }
}

/// Apply an [`EdgeDecision`] to one sum or partition.
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

/// `log` target shared with [`propagate`](crate::propagate).
pub(crate) const LOG_TARGET: &str = "paulistrings::propagate";

/// Replace the sum by one Pauli string, drawn with probability `|c|² / Σ|c|²` and given coefficient `1`, whenever it holds more than `cache` terms.
///
/// One seed is one Monte Carlo trajectory: the draw at the `k`-th layer pass is a function of `(seed, k)` only, and the pass counter lives in the policy, so reusing a policy continues its sequence.
/// Partitioned runs agree with unpartitioned ones in distribution, not string for string, and count passes and collapses on rank 0's policy only.
/// A collapse panics if the sum's total weight is zero or not finite.
#[derive(Debug)]
pub struct CollapseSample {
    /// Largest sum left untouched.
    pub cache: usize,
    /// Trajectory seed.
    pub seed: u64,
    /// Layer passes seen.
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

    /// The point of the cumulative weight pass `call` draws; panics unless `total` is positive and finite.
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

/// The slot whose stretch of the running total contains `target`, and the offset into it; past the end, the last positive slot.
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

/// Equal when `cache`, `seed` and both counters are.
impl PartialEq for CollapseSample {
    fn eq(&self, other: &Self) -> bool {
        self.cache == other.cache
            && self.seed == other.seed
            && self.calls.load(Ordering::Relaxed) == other.calls.load(Ordering::Relaxed)
            && self.collapses() == other.collapses()
    }
}

impl<const W: usize> TruncationPolicy<W> for CollapseSample {
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

/// Both policies: a term is kept if both keep it, and both layer passes run, first then second.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct And<A, B>(
    /// First policy.
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

    fn finalizes_layer(&self) -> bool {
        self.0.finalizes_layer() || self.1.finalizes_layer()
    }
}

/// Either policy: a term is kept if either keeps it, and neither layer pass runs.
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

    fn finalizes_layer(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests;
