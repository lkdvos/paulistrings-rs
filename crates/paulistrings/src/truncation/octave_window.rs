//! [`ApproxTopN`]'s layer pass from per-bucket octave counts the coset loop takes while each bucket is still in cache, so the pass itself only reads the sum to retain.
//!
//! Contract: [`finalize_from_window`] keeps exactly the set [`approx_top_n_pass`] keeps, and falls back to it when the boundary octave lies outside the window.

#[cfg(doc)]
use super::builtin::ApproxTopN;
use super::builtin::{
    approx_top_n_pass, bucket_quotas, classify_octave, local_boundary_quota, octave,
    retain_with_bucket_quotas, EdgeDecision, APPROX_BINS,
};
use crate::coefficient::Coefficient;
use crate::collectives::Collectives;
use crate::pauli_sum::PauliSum;

/// Octaves a window resolves.
pub(crate) const WINDOW_OCTAVES: usize = 64;

/// A bucket's counts: below the window, one per window octave, above the window.
pub(crate) type BucketOctaves = [u32; WINDOW_OCTAVES + 2];

/// The [`WINDOW_OCTAVES`] octaves of `|c|²` counted per bucket, from `base` up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OctaveWindow {
    base: usize,
}

impl OctaveWindow {
    /// The window centred on `bin`, clamped into the octave range.
    pub(crate) fn around(bin: usize) -> Self {
        Self { base: bin.saturating_sub(WINDOW_OCTAVES / 2).min(APPROX_BINS - WINDOW_OCTAVES) }
    }

    /// The slot of octave `bin` in [`BucketOctaves`].
    #[inline(always)]
    fn slot(&self, bin: usize) -> usize {
        (bin + 1).saturating_sub(self.base).min(WINDOW_OCTAVES + 1)
    }

    /// Count `coefficients` into `counts`, which it overwrites, binned as `octave_histogram` bins them.
    pub(crate) fn count<C: Coefficient>(&self, coefficients: &[C], counts: &mut BucketOctaves) {
        *counts = [0; WINDOW_OCTAVES + 2];
        for c in coefficients {
            counts[self.slot(octave(c.norm_sqr()))] += 1;
        }
    }

    /// `octave_edge` of a sum whose counts, summed over every bucket and partition, are `totals`; `None` when the boundary is outside the window.
    fn edge(&self, totals: &[u64], total_len: usize, n: usize) -> Option<EdgeDecision> {
        debug_assert_eq!(totals.len(), WINDOW_OCTAVES + 2);
        if total_len <= n {
            return Some(EdgeDecision::KeepAll);
        }
        if n == 0 {
            return Some(EdgeDecision::Clear);
        }
        let mut kept_above = totals[WINDOW_OCTAVES + 1] as usize;
        if kept_above > n {
            return None;
        }
        for offset in (0..WINDOW_OCTAVES).rev() {
            let population = totals[1 + offset] as usize;
            if kept_above + population > n {
                let bin = self.base + offset;
                return Some(EdgeDecision::Boundary { bin, kept_above, quota: n - kept_above });
            }
            kept_above += population;
        }
        None
    }
}

/// [`approx_top_n_pass`] over a partition whose bucket `b` holds `per_bucket[b]` under `window`, returning the decision it applied.
///
/// One `allreduce_sum_u64` of `[len, window counts…]`; at an in-window boundary one more of the per-rank boundary counts, and outside the window the fallback's collectives, chosen alike on every partition from the reduced counts.
pub(crate) fn finalize_from_window<const W: usize, C: Coefficient>(
    local: &mut PauliSum<W, C>,
    collectives: &dyn Collectives,
    n: usize,
    window: OctaveWindow,
    per_bucket: &[BucketOctaves],
) -> EdgeDecision {
    debug_assert_eq!(per_bucket.len(), local.num_buckets());
    let mut packed = [0u64; WINDOW_OCTAVES + 3];
    packed[0] = local.len() as u64;
    for counts in per_bucket {
        for (slot, &count) in packed[1..].iter_mut().zip(counts) {
            *slot += u64::from(count);
        }
    }
    debug_assert_eq!(packed[1..].iter().sum::<u64>(), packed[0], "the window counts every term");
    let solo = collectives.size() == 1;
    if !solo {
        collectives.allreduce_sum_u64(&mut packed);
    }
    let Some(edge) = window.edge(&packed[1..], packed[0] as usize, n) else {
        return approx_top_n_pass(local, collectives, n);
    };
    match edge {
        EdgeDecision::KeepAll => {}
        EdgeDecision::Clear => local.clear(),
        EdgeDecision::Boundary { bin, quota, .. } => {
            let slot = window.slot(bin);
            let boundary_counts: Vec<usize> =
                per_bucket.iter().map(|counts| counts[slot] as usize).collect();
            let local_quota = if solo {
                quota
            } else {
                local_boundary_quota(collectives, boundary_counts.iter().sum(), quota)
            };
            retain_with_bucket_quotas(
                local,
                |m| classify_octave(m, bin),
                &bucket_quotas(&boundary_counts, local_quota),
            );
        }
    }
    edge
}

#[cfg(test)]
mod tests;
