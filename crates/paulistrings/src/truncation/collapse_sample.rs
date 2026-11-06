//! [`CollapseSample`], the sampling truncation: a systematic resample in proportion to `|c|` whenever the sum outgrows its cache.

use super::TruncationPolicy;
use crate::coefficient::Coefficient;
use crate::collectives::Collectives;
use crate::engine::LOG_TARGET;
use crate::pauli_sum::PauliSum;
use crate::rng::Rng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(doc)]
use super::PerPartition;

/// Whenever the sum holds more than `cache` terms after a layer, replace it by `samples` systematic draws in proportion to `|c|`, an unbiased estimate of it.
///
/// With `‖O‖₁ = Σ|c|`, the terms are laid end to end in iteration order, term `i` owning the half-open stretch `[S_i, S_i + |c_i|)` of `[0, ‖O‖₁)`; one `u ∈ [0, 1)` per collapse places the points `t_k = (u + k)·h` for `k < samples`, with `h = ‖O‖₁ / samples`.
/// A term holding `m ≥ 1` points survives with coefficient `m·h·c/|c|`, every other term is dropped, and so the expectation of the result is the sum.
/// A point on a boundary belongs to the later stretch, a point that rounding puts at or past `‖O‖₁` to the last term with a nonzero stretch, and a zero coefficient is never drawn.
/// Every term is drawn `⌊samples·|c|/‖O‖₁⌋` or `⌈…⌉` times up to rounding, so the result holds at most `min(samples, len)` terms; a collapse resamples even when `samples` is at least the length.
/// `‖O‖₁`, the stretches and the points are `f64` whatever the coefficient type.
///
/// One seed is one Monte Carlo trajectory: `u` at the `k`-th layer pass is a function of `(seed, k)` only, and the pass counter lives in the policy, so reusing a policy continues its sequence.
/// Partitioned, the stretches run in rank order and then bucket order, and each partition takes exactly the points that fall in its own: the group draws `samples` points, every term's multiplicity has its unpartitioned law, and a partition with no point clears; passes and collapses count on rank 0's policy only.
/// Inside a [`PerPartition`] each partition instead draws `samples` points from its own slice whenever that slice exceeds `cache`.
/// A collapse panics if `‖O‖₁` is zero or not finite.
///
/// A lone partition may apply a run of local layers (consecutive one-to-one channels, or rotations the engine joins) as one pass and check the cache once, at the run's end, while the run's worst case, its input length times the product of its layers' fanouts, stays within [`RUN_GROWTH`](Self::RUN_GROWTH)` × cache`.
/// Every collapse is unbiased given the sum it draws from, so every schedule of them is, and the run's estimate keeps the expectation of the layer-by-layer one; fewer passes take fewer pass indices, so the trajectory a seed names depends on the runs taken.
#[derive(Debug)]
pub struct CollapseSample {
    /// Largest sum left untouched.
    pub cache: usize,
    /// Points drawn per collapse, at least one.
    pub samples: usize,
    /// Trajectory seed.
    pub seed: u64,
    /// Layer passes seen.
    calls: AtomicU64,
    /// Collapses performed.
    collapses: AtomicU64,
}

impl CollapseSample {
    /// How far past `cache` a run of local layers may grow, worst case, before its one collapse check.
    pub const RUN_GROWTH: usize = 8;

    /// A fresh trajectory: above `cache` terms, collapse to `samples` draws keyed by `seed`.
    ///
    /// # Panics
    ///
    /// If `samples` is zero.
    pub fn new(cache: usize, samples: usize, seed: u64) -> Self {
        assert!(samples >= 1, "CollapseSample: samples must be at least 1");
        Self { cache, samples, seed, calls: AtomicU64::new(0), collapses: AtomicU64::new(0) }
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

    /// The offset `u` of pass `call`'s points.
    pub(crate) fn uniform(&self, call: u64) -> f64 {
        Rng::from_key(&[self.seed, call]).uniform()
    }
}

/// Panics unless `total`, the `‖O‖₁` of `len` terms, is positive and finite.
pub(crate) fn assert_drawable(total: f64, len: usize) {
    assert!(
        total.is_finite() && total > 0.0,
        "CollapseSample: cannot draw by |c| from {len} terms with Σ|c| = {total}",
    );
}

/// `values` summed left to right from `0.0`, the order every stretch boundary of a collapse is accumulated in.
pub(crate) fn running_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0, |total, value| total + value)
}

/// `Σ|c|` of each bucket, in bucket order.
pub(crate) fn bucket_l1_norms<const W: usize, C: Coefficient>(sum: &PauliSum<W, C>) -> Vec<f64> {
    (0..sum.num_buckets())
        .into_par_iter()
        .map(|b| running_sum(sum.bucket(b).2.iter().map(|c| c.magnitude())))
        .collect()
}

/// The points `t_k = (u + k)·h`, `k < samples`, of one collapse over `[0, total)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SystematicPoints {
    pub(crate) u: f64,
    pub(crate) samples: usize,
    pub(crate) total: f64,
}

impl SystematicPoints {
    fn step(&self) -> f64 {
        self.total / self.samples as f64
    }

    fn point(&self, k: usize) -> f64 {
        (self.u + k as f64) * self.step()
    }

    /// How many points lie below `boundary`, all of them once it reaches `total`; the points are nondecreasing in `k`.
    fn below(&self, boundary: f64) -> usize {
        if boundary >= self.total {
            return self.samples;
        }
        let (mut lo, mut hi) = (0, self.samples);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.point(mid) < boundary {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// `weight · c/|c|` in `f64`, given `magnitude = |c| > 0`.
fn drawn<C: Coefficient>(c: C, magnitude: f64, weight: f64) -> C {
    C::from_complex64(c.to_complex64() / magnitude * weight, f64::INFINITY)
        .expect("a real coefficient scaled by a real factor stays real")
}

/// The systematic resample [`CollapseSample`] documents, of a sum whose stretches start at `offset` of `points.total`; `norms` is [`bucket_l1_norms`] of `sum`.
pub(crate) fn systematic_resample<const W: usize, C: Coefficient>(
    sum: &mut PauliSum<W, C>,
    norms: &[f64],
    offset: f64,
    points: SystematicPoints,
) {
    let mut starts = Vec::with_capacity(norms.len() + 1);
    let mut prefix = 0.0f64;
    for &norm in norms {
        starts.push(prefix);
        prefix += norm;
    }
    starts.push(prefix);
    // Both passes form a boundary as `offset + (bucket start + running Σ|c|)`, so a bucket's last stretch ends exactly where the next bucket's first begins.
    let first: Vec<usize> = starts.par_iter().map(|&start| points.below(offset + start)).collect();
    let step = points.step();
    sum.update_per_bucket(|b| {
        let (bucket_start, end) = (starts[b], first[b + 1]);
        let mut next = first[b];
        let mut running = 0.0f64;
        move |_: &[u64; W], _: &[u64; W], c: C| {
            if next == end {
                return None;
            }
            let magnitude = c.magnitude();
            running += magnitude;
            let boundary = offset + (bucket_start + running);
            let mut stop = end;
            if boundary < points.total {
                stop = next;
                while stop < end && points.point(stop) < boundary {
                    stop += 1;
                }
            }
            let multiplicity = stop - next;
            next = stop;
            (multiplicity > 0).then(|| drawn(c, magnitude, multiplicity as f64 * step))
        }
    });
}

/// Equal when `cache`, `samples`, `seed` and both counters are.
impl PartialEq for CollapseSample {
    fn eq(&self, other: &Self) -> bool {
        self.cache == other.cache
            && self.samples == other.samples
            && self.seed == other.seed
            && self.calls.load(Ordering::Relaxed) == other.calls.load(Ordering::Relaxed)
            && self.collapses() == other.collapses()
    }
}

impl<const W: usize, C: Coefficient> TruncationPolicy<W, C> for CollapseSample {
    fn finalize_layer(&self, sum: &mut PauliSum<W, C>) {
        let call = self.next_call();
        let len = sum.len();
        if len <= self.cache {
            return;
        }
        let norms = bucket_l1_norms(sum);
        let total = running_sum(norms.iter().copied());
        assert_drawable(total, len);
        let points = SystematicPoints { u: self.uniform(call), samples: self.samples, total };
        systematic_resample(sum, &norms, 0.0, points);
        let n = self.count_collapse();
        log::debug!(
            target: LOG_TARGET,
            "collapse_sample: {len} terms, sum |c| = {total:.6e}, {} draws kept {} (collapse {n})",
            self.samples,
            sum.len(),
        );
    }

    /// Reduces `[len, pass]` (the pass from rank 0) and the per-partition `Σ|c|`, then resamples this partition's stretch of the group's points, as [`finalize_layer`](TruncationPolicy::finalize_layer) resamples each bucket's.
    fn finalize_layer_partitioned(
        &self,
        local: &mut PauliSum<W, C>,
        collectives: &dyn Collectives,
    ) {
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

        let norms = bucket_l1_norms(local);
        let mut weights = vec![0.0f64; collectives.size() as usize];
        weights[rank] = running_sum(norms.iter().copied());
        collectives.allreduce_sum_f64(&mut weights);
        let offset = running_sum(weights[..rank].iter().copied());
        let total = running_sum(weights.iter().copied());
        assert_drawable(total, len as usize);
        let points = SystematicPoints { u: self.uniform(call), samples: self.samples, total };
        systematic_resample(local, &norms, offset, points);
        if rank == 0 {
            let collapses = self.count_collapse();
            log::debug!(
                target: LOG_TARGET,
                "collapse_sample: {len} terms over {} partitions, sum |c| = {total:.6e}, \
                 {} draws (collapse {collapses})",
                collectives.size(),
                self.samples,
            );
        }
    }

    fn supports_partitioned(&self) -> bool {
        true
    }

    fn run_length_limit(&self) -> usize {
        self.cache.saturating_mul(Self::RUN_GROWTH)
    }
}

#[cfg(test)]
mod tests;
