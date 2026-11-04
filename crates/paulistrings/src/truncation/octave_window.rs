//! [`ApproxTopN`]'s layer pass decided from per-bucket octave counts the coset loop takes, and the [`BoundaryCut`] it decides, which the engine applies bucket by bucket.
//!
//! Contract: [`decide_from_window`] decides the cut [`approx_top_n_pass`] makes, or answers `None` when the boundary octave lies outside the window, and a [`BoundaryCut`] applied bucket by bucket keeps the same set as applied at once.

#[cfg(doc)]
use super::builtin::{approx_top_n_pass, ApproxTopN};
use super::builtin::{
    boundary_filter, bucket_quotas, classify_octave, local_boundary_quota, octave,
    octave_edge_within, EdgeDecision, APPROX_BINS,
};
use crate::coefficient::Coefficient;
use crate::collectives::Collectives;
use crate::pauli_sum::storage::BucketColumns;
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

    /// Overwrite `counts` with `coefficients`' counts.
    pub(crate) fn count<C: Coefficient>(&self, coefficients: &[C], counts: &mut BucketOctaves) {
        *counts = [0; WINDOW_OCTAVES + 2];
        for c in coefficients {
            counts[self.slot(octave(c.norm_sqr()))] += 1;
        }
    }
}

/// A boundary cut decided but not yet applied: bucket `b` keeps its terms above octave `bin` and the first `quotas[b]` at it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BoundaryCut {
    pub(crate) bin: usize,
    pub(crate) quotas: Vec<usize>,
    /// The partition's terms after the cut, fewer only by NaN coefficients.
    pub(crate) kept: usize,
}

impl BoundaryCut {
    fn filter<const W: usize, C: Coefficient>(
        &self,
        b: usize,
    ) -> impl FnMut(&[u64; W], &[u64; W], C) -> bool {
        let bin = self.bin;
        boundary_filter(move |m| classify_octave(m, bin), self.quotas[b])
    }

    pub(crate) fn apply<const W: usize, C: Coefficient>(&self, local: &mut PauliSum<W, C>) {
        local.retain_per_bucket(|b| self.filter(b));
        debug_assert!(local.len() <= self.kept, "a cut keeps at most `kept`");
    }

    /// Apply the cut to bucket `b` alone, as [`apply`](Self::apply) would.
    pub(crate) fn apply_to_bucket<const W: usize, C: Coefficient>(
        &self,
        b: usize,
        columns: &mut BucketColumns<W, C>,
    ) {
        columns.retain(self.filter(b));
    }
}

/// [`approx_top_n_pass`]'s decision and whole-sum length for a partition whose bucket `b` holds `per_bucket[b]` under `window`, with the cut at a boundary, without touching the sum; `None` when the boundary octave lies outside the window.
///
/// One `allreduce_sum_u64` of `[len, window counts…]`, and at an in-window boundary one more of the per-rank boundary counts; every partition answers `None` alike, from the reduced counts.
pub(crate) fn decide_from_window(
    local_len: usize,
    collectives: &dyn Collectives,
    n: usize,
    window: OctaveWindow,
    per_bucket: &[BucketOctaves],
) -> Option<(EdgeDecision, usize, Option<BoundaryCut>)> {
    let mut packed = [0u64; WINDOW_OCTAVES + 3];
    packed[0] = local_len as u64;
    for counts in per_bucket {
        for (slot, &count) in packed[1..].iter_mut().zip(counts) {
            *slot += u64::from(count);
        }
    }
    debug_assert_eq!(packed[1..].iter().sum::<u64>(), packed[0], "the window counts every term");
    if collectives.size() > 1 {
        collectives.allreduce_sum_u64(&mut packed);
    }
    let total = packed[0] as usize;
    let above = packed[WINDOW_OCTAVES + 2] as usize;
    let edge = octave_edge_within(&packed[2..WINDOW_OCTAVES + 2], window.base, above, total, n)?;
    let cut = match edge {
        EdgeDecision::Boundary { bin, quota, .. } => {
            let slot = window.slot(bin);
            let boundary_counts: Vec<usize> =
                per_bucket.iter().map(|counts| counts[slot] as usize).collect();
            let local_quota =
                local_boundary_quota(collectives, boundary_counts.iter().sum(), quota);
            let local_above: usize = per_bucket
                .iter()
                .map(|counts| counts[slot + 1..].iter().map(|&count| count as usize).sum::<usize>())
                .sum();
            Some(BoundaryCut {
                bin,
                quotas: bucket_quotas(&boundary_counts, local_quota),
                kept: local_above + local_quota,
            })
        }
        EdgeDecision::KeepAll | EdgeDecision::Clear => None,
    };
    Some((edge, total, cut))
}

#[cfg(test)]
mod tests;
