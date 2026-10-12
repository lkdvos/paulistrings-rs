//! The per-run sort kernels and the fused two-stream merge, the bucketed engine's inner loop (ARCHITECTURE.md §Engine).
//!
//! Both sorts leave a run ascending in lex `(x, z)` with duplicates allowed, as a permutation of its rows; equal-key order is unspecified, so they agree only to floating-point tolerance.

use num_complex::Complex64;

use crate::channel::prepared::LocalPtm;
use crate::pauli_sum::storage::{BucketColumns, ColumnsRef};
use crate::truncation::TruncationPolicy;

/// Worker-persistent scratch for the per-run sorts.
#[derive(Clone, Debug, Default)]
pub(crate) struct SortScratch<const W: usize> {
    permutation: Vec<u32>,
    /// The radix kernel's `(surrogate << 32) | row index` records.
    packed: Vec<u64>,
    /// The radix kernel's double buffer for `packed`.
    spare: Vec<u64>,
    staging_x: Vec<[u64; W]>,
    staging_z: Vec<[u64; W]>,
    staging_coefficient: Vec<Complex64>,
}

/// Sort the columns in place by the key `(x, z)` alone, by comparison through a permutation.
// Must stay the adaptive stable `sort_by`, whose run detection merges the run's presorted delta streams (ARCHITECTURE.md §Engine).
// Not `sort_unstable_by`: research/FINDINGS.md §The `engine/merge.rs` `#[inline]` folklore
fn sort_by_comparison<const W: usize>(
    columns: &mut BucketColumns<W>,
    scratch: &mut SortScratch<W>,
) {
    let BucketColumns { x, z, coeff: c } = columns;
    let len = x.len();
    debug_assert_eq!(len, z.len());
    debug_assert_eq!(len, c.len());
    assert!(
        len <= u32::MAX as usize,
        "sort: run of {len} rows exceeds u32 row indices"
    );
    if len < 2 {
        return;
    }
    scratch.permutation.clear();
    scratch.permutation.extend(0..len as u32);
    scratch.permutation.sort_by(|&a, &b| {
        x[a as usize]
            .cmp(&x[b as usize])
            .then_with(|| z[a as usize].cmp(&z[b as usize]))
    });
    scratch.staging_x.clear();
    scratch
        .staging_x
        .extend(scratch.permutation.iter().map(|&i| x[i as usize]));
    scratch.staging_z.clear();
    scratch
        .staging_z
        .extend(scratch.permutation.iter().map(|&i| z[i as usize]));
    scratch.staging_coefficient.clear();
    scratch
        .staging_coefficient
        .extend(scratch.permutation.iter().map(|&i| c[i as usize]));
    std::mem::swap(x, &mut scratch.staging_x);
    std::mem::swap(z, &mut scratch.staging_z);
    std::mem::swap(c, &mut scratch.staging_coefficient);
}

/// The per-run sort kernel, chosen once per layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SortKernel {
    /// [`sort_by_comparison`], the default and the only kernel a sparse-PTM layer sees.
    Comparison,
    /// [`sort_by_radix`], for runs of many ascending streams with heavily duplicated keys.
    Radix,
}

impl SortKernel {
    /// The kernel for a layer with `rest_streams` rest streams; `rows_per_key` (the channel's [`rest_rows_per_key`]) is evaluated only when the second arm needs it.
    pub(crate) fn for_layer(rest_streams: usize, rows_per_key: impl FnOnce() -> f64) -> Self {
        if rest_streams >= RADIX_MIN_REST_STREAMS
            || (rest_streams >= RADIX_MIN_DISJOINT_STREAMS
                && rows_per_key() < RADIX_MAX_REST_ROWS_PER_KEY)
        {
            Self::Radix
        } else {
            Self::Comparison
        }
    }

    /// Sort `columns` in place by the key `(x, z)` alone.
    pub(crate) fn sort<const W: usize>(
        self,
        columns: &mut BucketColumns<W>,
        scratch: &mut SortScratch<W>,
    ) {
        match self {
            Self::Comparison => sort_by_comparison(columns, scratch),
            Self::Radix => sort_by_radix(columns, scratch),
        }
    }
}

/// Rest streams at or above which a layer uses [`SortKernel::Radix`]; research/FINDINGS.md §Radix sort kernel for dense PTMs.
const RADIX_MIN_REST_STREAMS: usize = 8;

/// The radix gate's second arm: the largest `rest_rows_per_key` still treated as disjoint streams; research/FINDINGS.md §Constant recalibration and the radix gate's second arm.
const RADIX_MAX_REST_ROWS_PER_KEY: f64 = 2.0;

/// The radix gate's second arm: the minimum rest-stream count for disjoint streams.
const RADIX_MIN_DISJOINT_STREAMS: usize = 3;

/// Mean rest rows per output key that gets any, the radix gate's second arm; ask it of the channel's own PTM, never a restricted one.
pub(crate) fn rest_rows_per_key<const W: usize>(ptm: &LocalPtm<W>) -> f64 {
    let zero = Complex64::new(0.0, 0.0);
    let rest_start = usize::from(ptm.has_identity());
    let dim = 1usize << (2 * ptm.k());
    let (mut rows, mut keys) = (0u32, 0u32);
    for output in 0..dim {
        let rows_here = ptm.deltas()[rest_start..]
            .iter()
            .filter(|delta| delta.amplitude[output ^ delta.local_delta as usize] != zero)
            .count() as u32;
        rows += rows_here;
        keys += u32::from(rows_here > 0);
    }
    if keys == 0 {
        0.0
    } else {
        f64::from(rows) / f64::from(keys)
    }
}

/// Surrogate width the radix kernel sorts on, enough to separate duplicate-key groups rather than every key.
const RADIX_SURROGATE_BITS: u32 = 16;
/// Digit width per radix pass.
const RADIX_DIGIT_BITS: u32 = 8;
const RADIX_BUCKETS: usize = 1 << RADIX_DIGIT_BITS;
/// Below this many discriminating bits in the surrogate window the comparison kernel runs instead.
const RADIX_MIN_WINDOW_BITS: u32 = 8;

/// Word `k` of the lex key `(x, z)`, word 0 the most significant.
fn key_word<const W: usize>(x: &[[u64; W]], z: &[[u64; W]], word: usize, row: usize) -> u64 {
    if word < W {
        x[row][word]
    } else {
        z[row][word - W]
    }
}

/// The first key word the rows disagree on, its highest disagreeing bit and the disagreeing bits; `None` if all keys are equal.
fn discriminating_window<const W: usize>(
    x: &[[u64; W]],
    z: &[[u64; W]],
) -> Option<(usize, u32, u64)> {
    let len = x.len();
    for word in 0..2 * W {
        let mut any = 0u64;
        let mut all = !0u64;
        for row in 0..len {
            let value = key_word(x, z, word, row);
            any |= value;
            all &= value;
        }
        let diff = any & !all;
        if diff != 0 {
            return Some((word, 63 - diff.leading_zeros(), diff));
        }
    }
    None
}

/// Sort the columns in place by the key `(x, z)` alone, by radix on a 16-bit surrogate of the first disagreeing key word.
// Every row shares the bits above the window, so the surrogate is monotone in the key and only leaves ties, which the fixup orders on the full key.
fn sort_by_radix<const W: usize>(columns: &mut BucketColumns<W>, scratch: &mut SortScratch<W>) {
    let BucketColumns { x, z, coeff: c } = columns;
    let len = x.len();
    debug_assert_eq!(len, z.len());
    debug_assert_eq!(len, c.len());
    assert!(
        len <= u32::MAX as usize,
        "sort: run of {len} rows exceeds u32 row indices"
    );
    if len < 2 {
        return;
    }
    let Some((word, high_bit, diff)) = discriminating_window(x, z) else {
        return;
    };
    let shift = (high_bit + 1).saturating_sub(RADIX_SURROGATE_BITS);
    let mask = (1u64 << RADIX_SURROGATE_BITS) - 1;
    if ((diff >> shift) & mask).count_ones() < RADIX_MIN_WINDOW_BITS {
        sort_by_comparison(columns, scratch);
        return;
    }

    let SortScratch {
        packed,
        spare,
        staging_x,
        staging_z,
        staging_coefficient,
        ..
    } = scratch;
    packed.clear();
    packed.extend((0..len).map(|i| (((key_word(x, z, word, i) >> shift) & mask) << 32) | i as u64));
    spare.resize(len, 0);

    let mut digit = 0u32;
    while digit * RADIX_DIGIT_BITS < RADIX_SURROGATE_BITS {
        let digit_shift = 32 + digit * RADIX_DIGIT_BITS;
        let mut count = [0u32; RADIX_BUCKETS + 1];
        for &record in packed.iter() {
            count[(((record >> digit_shift) as usize) & (RADIX_BUCKETS - 1)) + 1] += 1;
        }
        // A constant digit contributes no ordering: skip its scatter.
        if count[1..].iter().filter(|&&tally| tally != 0).count() > 1 {
            for slot in 1..=RADIX_BUCKETS {
                count[slot] += count[slot - 1];
            }
            for &record in packed.iter() {
                let bucket = ((record >> digit_shift) as usize) & (RADIX_BUCKETS - 1);
                spare[count[bucket] as usize] = record;
                count[bucket] += 1;
            }
            std::mem::swap(packed, spare);
        }
        digit += 1;
    }

    // Fixup: rows sharing a surrogate are ordered on the full key.
    let mut i = 0usize;
    while i < len {
        let surrogate = packed[i] >> 32;
        let mut j = i + 1;
        while j < len && packed[j] >> 32 == surrogate {
            j += 1;
        }
        if j - i > 1 {
            packed[i..j].sort_by(|&left, &right| {
                let (row_a, row_b) = (left as u32 as usize, right as u32 as usize);
                x[row_a]
                    .cmp(&x[row_b])
                    .then_with(|| z[row_a].cmp(&z[row_b]))
            });
        }
        i = j;
    }

    staging_x.clear();
    staging_x.extend(packed.iter().map(|&record| x[record as u32 as usize]));
    staging_z.clear();
    staging_z.extend(packed.iter().map(|&record| z[record as u32 as usize]));
    staging_coefficient.clear();
    staging_coefficient.extend(packed.iter().map(|&record| c[record as u32 as usize]));
    std::mem::swap(x, staging_x);
    std::mem::swap(z, staging_z);
    std::mem::swap(c, staging_coefficient);
}

/// Merge the strictly ascending `identity` stream with the sorted `rest` stream into `out`, summing equal keys and applying `keep_term` to each full sum.
// Signed-zero contract: exact-zero rows are summed like any other, and the only zero test is on the final sum.
// Three loops, so neither stream's bound is tested in the main walk (ARCHITECTURE.md §Engine).
// Not segment copies: research/FINDINGS.md §Segment-copy merge
// Not a branchless walk: research/FINDINGS.md §Branchless `merge2_into`
pub(crate) fn merge_identity_and_rest<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    identity: ColumnsRef<'_, W>,
    rest: ColumnsRef<'_, W>,
    out: &mut BucketColumns<W>,
    policy: &T,
) {
    let ColumnsRef {
        x: a_x,
        z: a_z,
        coeff: a_c,
    } = identity;
    let ColumnsRef {
        x: b_x,
        z: b_z,
        coeff: b_c,
    } = rest;
    let (a_len, b_len) = (a_c.len(), b_c.len());
    debug_assert_eq!(a_len, a_x.len());
    debug_assert_eq!(a_len, a_z.len());
    debug_assert_eq!(b_len, b_x.len());
    debug_assert_eq!(b_len, b_z.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a_len && j < b_len {
        // On a tie `a` seeds the sum; since `a` is unique, only `b` rows can extend the segment.
        let take_a = (a_x[i], a_z[i]) <= (b_x[j], b_z[j]);
        let (key_x, key_z, mut sum) = if take_a {
            debug_assert!(
                i == 0 || (a_x[i - 1], a_z[i - 1]) < (a_x[i], a_z[i]),
                "merge_identity_and_rest: identity stream must be strictly ascending at {i}",
            );
            let t = (a_x[i], a_z[i], a_c[i]);
            i += 1;
            t
        } else {
            debug_assert!(
                j == 0 || (b_x[j - 1], b_z[j - 1]) <= (b_x[j], b_z[j]),
                "merge_identity_and_rest: rest stream must be sorted at {j}",
            );
            let t = (b_x[j], b_z[j], b_c[j]);
            j += 1;
            t
        };
        while j < b_len && b_x[j] == key_x && b_z[j] == key_z {
            sum += b_c[j];
            j += 1;
        }
        emit_if_kept(out, policy, key_x, key_z, sum);
    }
    while i < a_len {
        let (key_x, key_z, sum) = (a_x[i], a_z[i], a_c[i]);
        i += 1;
        emit_if_kept(out, policy, key_x, key_z, sum);
    }
    while j < b_len {
        let (key_x, key_z) = (b_x[j], b_z[j]);
        let mut sum = b_c[j];
        j += 1;
        while j < b_len && b_x[j] == key_x && b_z[j] == key_z {
            sum += b_c[j];
            j += 1;
        }
        emit_if_kept(out, policy, key_x, key_z, sum);
    }
}

/// Push one reduced key unless its sum is exactly zero or `keep_term` drops it.
#[inline(always)]
fn emit_if_kept<const W: usize, T: TruncationPolicy<W> + ?Sized>(
    out: &mut BucketColumns<W>,
    policy: &T,
    key_x: [u64; W],
    key_z: [u64; W],
    sum: Complex64,
) {
    if sum != Complex64::new(0.0, 0.0) && policy.keep_term(&key_x, &key_z, sum) {
        out.x.push(key_x);
        out.z.push(key_z);
        out.coeff.push(sum);
    }
}

#[cfg(test)]
mod tests;
