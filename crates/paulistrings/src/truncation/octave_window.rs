//! [`ApproxTopN`]'s layer pass decided from per-bucket octave counts the coset loop takes, and the [`BoundaryCut`] it decides, which the engine applies bucket by bucket.
//!
//! Contract: [`decide_from_window`] decides the cut [`approx_top_n_pass`] makes, or answers `None` when the boundary octave lies outside the window, and a [`BoundaryCut`] applied bucket by bucket keeps the same set as applied at once.

#[cfg(doc)]
use super::builtin::{approx_top_n_pass, ApproxTopN};
use super::builtin::{
    boundary_filter, classify_octave, local_boundary_quota, octave, octave_edge_within,
    EdgeDecision, QuotaSplit, APPROX_BINS,
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
    pub(crate) fn slot(&self, bin: usize) -> usize {
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

/// Buckets' [`BucketOctaves`] summed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WindowTotals(pub(crate) [u64; WINDOW_OCTAVES + 2]);

impl Default for WindowTotals {
    fn default() -> Self {
        Self([0; WINDOW_OCTAVES + 2])
    }
}

impl WindowTotals {
    pub(crate) fn of(per_bucket: &[BucketOctaves]) -> Self {
        let mut totals = Self::default();
        for counts in per_bucket {
            totals.add(counts);
        }
        totals
    }

    #[inline]
    pub(crate) fn add(&mut self, counts: &BucketOctaves) {
        for (total, &count) in self.0.iter_mut().zip(counts) {
            *total += u64::from(count);
        }
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        for (total, &count) in self.0.iter_mut().zip(&other.0) {
            *total += count;
        }
    }
}

/// A boundary cut decided but not yet applied: bucket `b` keeps its terms above octave `bin` and the first [`split.quota(b)`](QuotaSplit::quota) at it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BoundaryCut {
    pub(crate) bin: usize,
    pub(crate) split: QuotaSplit,
    /// The partition's terms after the cut, fewer only by NaN coefficients.
    pub(crate) kept: usize,
}

impl BoundaryCut {
    /// Bucket `b`'s keep test, to be asked of its terms in bucket order, as [`apply_to_bucket`](Self::apply_to_bucket) asks it.
    pub(crate) fn filter<const W: usize, C: Coefficient>(
        &self,
        b: usize,
    ) -> impl FnMut(&[u64; W], &[u64; W], C) -> bool {
        let bin = self.bin;
        boundary_filter(move |m| classify_octave(m, bin), self.split.quota(b))
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

/// [`approx_top_n_pass`]'s decision, the whole sum's length and, at a boundary, this partition's cut, from the window counts alone; `None` when the boundary octave lies outside the window.
///
/// `totals` sums `per_bucket`, whose boundary slot only places the split.
/// One `allreduce_sum_u64` of `[len, window counts…]`, and at an in-window boundary one more of the per-rank boundary counts; every partition answers `None` alike, from the reduced counts.
pub(crate) fn decide_from_window(
    local_len: usize,
    collectives: &dyn Collectives,
    n: usize,
    window: OctaveWindow,
    per_bucket: &[BucketOctaves],
    totals: &WindowTotals,
) -> Option<(EdgeDecision, usize, Option<BoundaryCut>)> {
    debug_assert_eq!(*totals, WindowTotals::of(per_bucket), "the totals sum the bucket counts");
    let mut packed = [0u64; WINDOW_OCTAVES + 3];
    packed[0] = local_len as u64;
    packed[1..].copy_from_slice(&totals.0);
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
            let local_count = totals.0[slot] as usize;
            let local_quota = local_boundary_quota(collectives, local_count, quota);
            let local_above = totals.0[slot + 1..].iter().sum::<u64>() as usize;
            let split = QuotaSplit::new(
                |b| per_bucket[b][slot] as usize,
                per_bucket.len(),
                local_count,
                local_quota,
            );
            Some(BoundaryCut { bin, split, kept: local_above + local_quota })
        }
        EdgeDecision::KeepAll | EdgeDecision::Clear => None,
    };
    Some((edge, total, cut))
}

#[cfg(test)]
mod tests;
