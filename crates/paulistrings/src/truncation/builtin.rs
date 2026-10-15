//! The built-in truncation policies and combinators, and the helpers their partitioned and device forms share.

use super::TruncationPolicy;
use crate::collectives::Collectives;
use crate::engine::LOG_TARGET;
use crate::pauli_sum::PauliSum;
use crate::rng::Rng;
use num_complex::Complex64;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! {
    /// [`TopN`]'s per-thread squared-magnitude buffer.
    // Taken out and put back, never borrowed across a parallel section: rayon can steal a nested `propagate` onto this thread and re-enter `finalize_layer`.
    static MAGNITUDES: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
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
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], _c: Complex64) -> bool {
        let weight: u32 = (0..W).map(|i| (x[i] | z[i]).count_ones()).sum();
        weight <= self.0
    }

    fn finalizes_layer(&self) -> bool {
        false
    }
}

/// Retain the `n` terms of largest coefficient magnitude.
///
/// With `t` the `n`-th largest magnitude, every term above `t` is kept, plus as many terms at exactly `t` as fit in `n`, the first in bucket order; which tied terms survive is not a semantic promise.
/// A sum of more than `n` terms keeps exactly `n`, and `n == 0` clears; a NaN coefficient is never kept, so a sum with fewer than `n` other terms keeps those.
/// Magnitudes are compared squared, so those below about `1.6e-162` tie at zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TopN(
    /// The retained term count.
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
        let mut buffer = MAGNITUDES.take();
        if buffer.len() < total {
            buffer.resize(total, 0.0);
        }
        let magnitudes = &mut buffer[..total];
        {
            let view = &*sum;
            let num_buckets = view.num_buckets();
            let mut handles: Vec<&mut [f64]> = Vec::with_capacity(num_buckets);
            let mut rest: &mut [f64] = magnitudes;
            for b in 0..num_buckets {
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

        // A NaN sorts below every number, so it becomes the boundary only when fewer than `n` terms are finite.
        magnitudes.select_nth_unstable_by(n - 1, |a, b| {
            b.partial_cmp(a)
                .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
        });
        let boundary = magnitudes[n - 1];
        // Everything above the pivot sits before it after the selection.
        let above = magnitudes[..n - 1]
            .par_iter()
            .filter(|&&m| m > boundary)
            .count();

        MAGNITUDES.set(buffer);

        retain_with_boundary_quota(
            sum,
            |m| {
                if m.is_nan() {
                    core::cmp::Ordering::Less
                } else {
                    // Unordered only against a NaN boundary, which every finite term is above.
                    m.partial_cmp(&boundary)
                        .unwrap_or(core::cmp::Ordering::Greater)
                }
            },
            n - above,
        );
    }

    /// Panics above one partition, where the `n`-th largest of a split sum has no collective form yet.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, collectives: &dyn Collectives) {
        assert!(
            collectives.size() == 1,
            "TopN({}) on {} partitions: partitioned exact TopN is not yet supported; ApproxTopN \
             is the partitioned alternative.",
            self.0,
            collectives.size(),
        );
        self.finalize_layer(local);
    }
}

/// How many boundary terms each bucket keeps: `quota` handed out in bucket order, each bucket taking at most its own count.
fn bucket_quotas(boundary_counts: &[usize], quota: usize) -> Vec<usize> {
    let mut remaining = quota;
    boundary_counts
        .iter()
        .map(|&count| {
            let taken = count.min(remaining);
            remaining -= taken;
            taken
        })
        .collect()
}

/// Keep every term `classify` puts above the boundary (`Greater`) and the first `quota` it puts at it (`Equal`), in bucket order.
pub(crate) fn retain_with_boundary_quota<const W: usize>(
    sum: &mut PauliSum<W>,
    classify: impl Fn(f64) -> core::cmp::Ordering + Sync,
    quota: usize,
) {
    use core::cmp::Ordering;
    let boundary_counts: Vec<usize> = (0..sum.num_buckets())
        .into_par_iter()
        .map(|b| {
            sum.bucket(b)
                .2
                .iter()
                .filter(|c| classify(c.norm_sqr()) == Ordering::Equal)
                .count()
        })
        .collect();
    let quotas = bucket_quotas(&boundary_counts, quota);
    sum.retain_per_bucket(|b| {
        let mut remaining = quotas[b];
        let classify = &classify;
        move |_: &[u64; W], _: &[u64; W], c: Complex64| match classify(c.norm_sqr()) {
            Ordering::Greater => true,
            Ordering::Equal if remaining > 0 => {
                remaining -= 1;
                true
            }
            Ordering::Equal | Ordering::Less => false,
        }
    });
}

/// Bins of [`ApproxTopN`]'s histogram, one per `f64` exponent.
pub(crate) const APPROX_BINS: usize = 2048;

/// Retain `n` terms, cutting at an octave of `|c|²` instead of selecting exactly.
///
/// Every octave of `|c|²` that fits whole is kept from the top down; the first that does not fit, the boundary octave, contributes as many terms as fit in `n`, the first in bucket order (rank order across partitions).
/// A sum of more than `n` terms keeps exactly `n` (fewer only if coefficients are NaN), and `n == 0` clears.
/// Which boundary-octave terms survive is not a semantic promise.
/// Partition-exact: a split sum's octave histogram is the sum of its partitions', so every partition cuts at the boundary one partition would.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApproxTopN(
    /// The retained term count.
    pub usize,
);

/// [`octave_edge`]'s decision, shared by the host, partitioned and device paths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum EdgeDecision {
    /// The whole sum fits within `n`: nothing is dropped.
    KeepAll,
    /// `n == 0`.
    Clear,
    /// Keep every term above octave `bin` and `quota` terms of octave `bin`, `kept_above + quota == n` in all.
    Boundary {
        /// The boundary octave, `norm_sqr().to_bits() >> 52`.
        bin: usize,
        /// Terms above the boundary octave, across the whole sum.
        kept_above: usize,
        /// Boundary-octave terms kept, across the whole sum.
        quota: usize,
    },
}

/// The octave of `|c|²`, [`APPROX_BINS`] wide.
fn octave(norm_sqr: f64) -> usize {
    (norm_sqr.to_bits() >> 52) as usize
}

/// `|c|²` against the boundary octave `bin`; a NaN is below every octave.
pub(crate) fn classify_octave(norm_sqr: f64, bin: usize) -> core::cmp::Ordering {
    if norm_sqr.is_nan() {
        core::cmp::Ordering::Less
    } else {
        octave(norm_sqr).cmp(&bin)
    }
}

/// Population of each octave of `|c|²`, binned by the exponent bits `norm_sqr().to_bits() >> 52`.
pub(crate) fn octave_histogram<const W: usize>(sum: &PauliSum<W>) -> [u32; APPROX_BINS] {
    debug_assert!(sum.len() <= u32::MAX as usize, "len exceeds bin counters");
    let num_buckets = sum.num_buckets();
    let tasks = (rayon::current_num_threads() * 4).clamp(1, num_buckets);
    (0..tasks)
        .into_par_iter()
        .map(|t| {
            let mut h = [0u32; APPROX_BINS];
            for b in (num_buckets * t / tasks)..(num_buckets * (t + 1) / tasks) {
                for c in sum.bucket(b).2 {
                    h[octave(c.norm_sqr())] += 1;
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

/// The boundary octave [`ApproxTopN`] cuts at, from a histogram of the whole sum, local or all-reduced.
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

    let mut kept_above = 0usize;
    for bin in (0..APPROX_BINS).rev() {
        let population = hist[bin].into() as usize;
        if kept_above + population > n {
            return EdgeDecision::Boundary {
                bin,
                kept_above,
                quota: n - kept_above,
            };
        }
        kept_above += population;
    }
    unreachable!("a histogram of more than n terms overflows n at some octave")
}

/// Apply an [`EdgeDecision`] to one sum or partition, keeping `local_quota` boundary-octave terms.
pub(crate) fn retain_at_edge<const W: usize>(
    sum: &mut PauliSum<W>,
    edge: EdgeDecision,
    local_quota: usize,
) {
    match edge {
        EdgeDecision::KeepAll => {}
        EdgeDecision::Clear => sum.clear(),
        EdgeDecision::Boundary { bin, .. } => {
            retain_with_boundary_quota(sum, |m| classify_octave(m, bin), local_quota);
        }
    }
}

/// This partition's share of a boundary quota: what is left of `quota` after the lower ranks' boundary terms, at most `local_count`.
///
/// One `allreduce_sum_u64` of a length-`P` vector holding each rank's `local_count` in its slot; every rank calls it with the same `quota`.
pub(crate) fn local_boundary_quota(
    collectives: &dyn Collectives,
    local_count: usize,
    quota: usize,
) -> usize {
    let rank = collectives.rank() as usize;
    let mut counts = vec![0u64; collectives.size() as usize];
    counts[rank] = local_count as u64;
    collectives.allreduce_sum_u64(&mut counts);
    let below: u64 = counts[..rank].iter().sum();
    (quota as u64).saturating_sub(below).min(local_count as u64) as usize
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
        let quota = match edge {
            EdgeDecision::Boundary { quota, .. } => quota,
            EdgeDecision::KeepAll | EdgeDecision::Clear => 0,
        };
        retain_at_edge(sum, edge, quota);
    }

    /// One `allreduce_sum_u64` of `[len, histogram…]`, the single-partition edge walk, and at a boundary one more of the per-rank boundary counts for this rank's quota; no early exit, since that would desynchronize the group.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, collectives: &dyn Collectives) {
        if collectives.size() == 1 {
            self.finalize_layer(local);
            return;
        }
        let histogram = octave_histogram(local);

        // `u64`, since `P` partitions' `u32` bins can overflow a `u32`.
        let mut packed = [0u64; 1 + APPROX_BINS];
        packed[0] = local.len() as u64;
        for (slot, &count) in packed[1..].iter_mut().zip(histogram.iter()) {
            *slot = u64::from(count);
        }
        collectives.allreduce_sum_u64(&mut packed);

        let total = packed[0] as usize;
        let edge = octave_edge(&packed[1..], total, self.0);
        let local_quota = match edge {
            EdgeDecision::Boundary { bin, quota, .. } => {
                local_boundary_quota(collectives, histogram[bin] as usize, quota)
            }
            EdgeDecision::KeepAll | EdgeDecision::Clear => 0,
        };
        retain_at_edge(local, edge, local_quota);
    }

    fn supports_partitioned(&self) -> bool {
        true
    }
}

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
    let mut cumulative = 0.0f64;
    let mut last = None;
    for (i, w) in weights.into_iter().enumerate() {
        if w > 0.0 {
            if target < cumulative + w {
                return Some((i, target - cumulative));
            }
            last = Some((i, target - cumulative));
        }
        cumulative += w;
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

    /// Reduces `[len, pass]` (the pass from rank 0) and the per-partition `Σ|c|²`, then picks a partition by weight and a term within it, as [`finalize_layer`](TruncationPolicy::finalize_layer) picks a bucket and then a term.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, collectives: &dyn Collectives) {
        if collectives.size() == 1 {
            self.finalize_layer(local);
            return;
        }
        let rank = collectives.rank() as usize;
        let call = if rank == 0 { self.next_call() } else { 0 };
        let mut head = [local.len() as u64, call];
        collectives.allreduce_sum_u64(&mut head);
        let [len, call] = head;
        if len as usize <= self.cache {
            return;
        }

        let norms = bucket_norms(local);
        let mut weights = vec![0.0f64; collectives.size() as usize];
        weights[rank] = norms.iter().sum();
        collectives.allreduce_sum_f64(&mut weights);
        let total: f64 = weights.iter().sum();
        let target = self.target(call, total, len as usize);
        let (chosen, rest) = pick_slot(weights.iter().copied(), target)
            .expect("CollapseSample: a positive total has a positive partition");
        if rank == chosen {
            collapse_to_target(local, &norms, rest);
        } else {
            local.clear();
        }
        if rank == 0 {
            let collapses = self.count_collapse();
            log::debug!(
                target: LOG_TARGET,
                "collapse_sample: {len} terms over {} partitions, sum |c|^2 = {total:.6e}, \
                 collapsed to one on partition {chosen} (collapse {collapses})",
                collectives.size(),
            );
        }
    }

    fn supports_partitioned(&self) -> bool {
        true
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

    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, collectives: &dyn Collectives) {
        self.0.finalize_layer_partitioned(local, collectives);
        self.1.finalize_layer_partitioned(local, collectives);
    }

    fn supports_partitioned(&self) -> bool {
        self.0.supports_partitioned() && self.1.supports_partitioned()
    }
}

/// Either of two per-term filters: a term is kept if either keeps it.
///
/// `Or` combines per-term filters only: an operand with a layer pass ([`TopN`], [`ApproxTopN`], [`CollapseSample`], or an [`And`] containing one) is rejected, since "either keeps" has no meaning for a pass over the whole sum.
/// Use [`And`] to add a filter to a layer pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Or<A, B>(pub(crate) A, pub(crate) B);

impl<A, B> Or<A, B> {
    /// Either of `first` and `second`.
    ///
    /// The operands are checked in [`finalizes_layer`](TruncationPolicy::finalizes_layer), which every driver calls before the first layer, since a layer pass is a property of the policy at a width `W` and `new` has no `W`.
    pub fn new(first: A, second: B) -> Self {
        Self(first, second)
    }
}

/// Why an `Or` of these operands is rejected, or `None` if both are per-term filters.
pub(crate) fn or_operand_error(first_has_pass: bool, second_has_pass: bool) -> Option<String> {
    let which = match (first_has_pass, second_has_pass) {
        (false, false) => return None,
        (true, false) => "its first operand has",
        (false, true) => "its second operand has",
        (true, true) => "both operands have",
    };
    Some(format!(
        "Or combines per-term filters only, but {which} a layer pass (TopN, ApproxTopN or \
         CollapseSample, alone or inside an And); use And to add a filter to a layer pass"
    ))
}

impl<const W: usize, A, B> TruncationPolicy<W> for Or<A, B>
where
    A: TruncationPolicy<W>,
    B: TruncationPolicy<W>,
{
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
        self.0.keep_term(x, z, c) || self.1.keep_term(x, z, c)
    }

    /// `false`; panics if either operand has a layer pass.
    fn finalizes_layer(&self) -> bool {
        if let Some(message) = or_operand_error(self.0.finalizes_layer(), self.1.finalizes_layer())
        {
            panic!("{message}");
        }
        false
    }
}

#[cfg(test)]
mod tests;
