//! [`ApproxTopN`]'s layer pass from per-bucket octave counts the coset loop takes while each bucket is still in cache, so the pass itself only reads the sum to retain.
//!
//! Contract: [`finalize_from_window`] keeps exactly the set [`approx_top_n_pass`] keeps, and falls back to it when the boundary octave lies outside the window; a [`BoundaryCut`] applied bucket by bucket keeps the same set as applied at once.

#[cfg(doc)]
use super::builtin::ApproxTopN;
use super::builtin::{
    bucket_quotas, classify_octave, local_boundary_quota, octave, retain_with_bucket_quotas,
    EdgeDecision, APPROX_BINS,
};
use crate::coefficient::Coefficient;
use crate::collectives::Collectives;
use crate::pauli_sum::storage::BucketColumns;
use crate::pauli_sum::PauliSum;
use core::cmp::Ordering;

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

/// A boundary cut decided but not yet applied: bucket `b` keeps its terms above octave `bin` and the first `quotas[b]` at it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BoundaryCut {
    pub(crate) bin: usize,
    pub(crate) quotas: Vec<usize>,
    /// The partition's terms after the cut, fewer only by NaN coefficients.
    pub(crate) kept: usize,
}

impl BoundaryCut {
    /// Apply the cut to the whole partition.
    pub(crate) fn apply<const W: usize, C: Coefficient>(&self, local: &mut PauliSum<W, C>) {
        let bin = self.bin;
        retain_with_bucket_quotas(local, |m| classify_octave(m, bin), &self.quotas);
        debug_assert!(local.len() <= self.kept, "a cut keeps at most `kept`");
    }

    /// Apply the cut to bucket `b` alone, as [`apply`](Self::apply) would.
    pub(crate) fn apply_to_bucket<const W: usize, C: Coefficient>(
        &self,
        b: usize,
        columns: &mut BucketColumns<W, C>,
    ) {
        let mut remaining = self.quotas[b];
        let mut kept = 0usize;
        for row in 0..columns.len() {
            let (x, z, c) = (columns.x[row], columns.z[row], columns.coeff[row]);
            let keep = match classify_octave(c.norm_sqr(), self.bin) {
                Ordering::Greater => true,
                Ordering::Equal if remaining > 0 => {
                    remaining -= 1;
                    true
                }
                Ordering::Equal | Ordering::Less => false,
            };
            if keep {
                columns.x[kept] = x;
                columns.z[kept] = z;
                columns.coeff[kept] = c;
                kept += 1;
            }
        }
        columns.x.truncate(kept);
        columns.z.truncate(kept);
        columns.coeff.truncate(kept);
    }
}

/// What [`decide_from_window`] leaves to do.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum WindowDecision {
    /// Nothing is dropped.
    KeepAll,
    /// Every term is dropped.
    Clear,
    Cut(BoundaryCut),
}

/// [`approx_top_n_pass`]'s decision for a partition whose bucket `b` holds `per_bucket[b]` under `window`, without touching the sum; `None` when the boundary octave lies outside the window.
///
/// One `allreduce_sum_u64` of `[len, window counts…]`, and at an in-window boundary one more of the per-rank boundary counts; every partition answers `None` alike, from the reduced counts.
pub(crate) fn decide_from_window(
    local_len: usize,
    collectives: &dyn Collectives,
    n: usize,
    window: OctaveWindow,
    per_bucket: &[BucketOctaves],
) -> Option<WindowDecision> {
    let mut packed = [0u64; WINDOW_OCTAVES + 3];
    packed[0] = local_len as u64;
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
    let decision = match window.edge(&packed[1..], packed[0] as usize, n)? {
        EdgeDecision::KeepAll => WindowDecision::KeepAll,
        EdgeDecision::Clear => WindowDecision::Clear,
        EdgeDecision::Boundary { bin, quota, .. } => {
            let slot = window.slot(bin);
            let boundary_counts: Vec<usize> =
                per_bucket.iter().map(|counts| counts[slot] as usize).collect();
            let local_quota = if solo {
                quota
            } else {
                local_boundary_quota(collectives, boundary_counts.iter().sum(), quota)
            };
            let above: usize = per_bucket
                .iter()
                .map(|counts| counts[slot + 1..].iter().map(|&count| count as usize).sum::<usize>())
                .sum();
            WindowDecision::Cut(BoundaryCut {
                bin,
                quotas: bucket_quotas(&boundary_counts, local_quota),
                kept: above + local_quota,
            })
        }
    };
    Some(decision)
}

/// [`approx_top_n_pass`] from the window counts: [`decide_from_window`], applied at once, or the plain pass outside the window.
#[cfg(test)]
pub(crate) fn finalize_from_window<const W: usize, C: Coefficient>(
    local: &mut PauliSum<W, C>,
    collectives: &dyn Collectives,
    n: usize,
    window: OctaveWindow,
    per_bucket: &[BucketOctaves],
) {
    debug_assert_eq!(per_bucket.len(), local.num_buckets());
    match decide_from_window(local.len(), collectives, n, window, per_bucket) {
        None => {
            super::builtin::approx_top_n_pass(local, collectives, n);
        }
        Some(WindowDecision::KeepAll) => {}
        Some(WindowDecision::Clear) => local.clear(),
        Some(WindowDecision::Cut(cut)) => cut.apply(local),
    }
}

#[cfg(test)]
mod tests;
