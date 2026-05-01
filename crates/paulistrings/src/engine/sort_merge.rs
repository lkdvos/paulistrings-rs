//! Sort-merge propagation pipeline: scan → bucket → merge. See §5.

#![allow(unused)]

use crate::channel::Channel;
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

/// Empirical threshold below which a hashmap-based fast path beats sort-merge
/// (§8.3). Subject to benchmarking.
pub const SMALL_SUM_THRESHOLD: usize = 4096;

/// Apply a single channel to a `PauliSum`, producing the next layer.
///
/// Implements the three-phase pipeline:
///   1. **Scan** — `n_in × MAX_FANOUT` data-parallel channel applications.
///   2. **Bucket** — partition by support bits into `2^(2|support|)` buckets.
///   3. **Merge** — segmented reduction with truncation folded in.
pub fn apply_layer<const W: usize, C, T>(
    _input: &PauliSum<W>,
    _channel: &C,
    _policy: &T,
) -> PauliSum<W>
where
    C: Channel<W>,
    T: TruncationPolicy<W>,
{
    todo!("§5: scan-bucket-merge; consider fast path when input.len() < SMALL_SUM_THRESHOLD")
}

/// Phase 1 of `apply_layer`: write each input's outputs into a flat,
/// pre-allocated `n_in × MAX_FANOUT` scratch buffer.
pub(crate) fn scan_phase<const W: usize, C: Channel<W>>(
    _input: &PauliSum<W>,
    _channel: &C,
    /* scratch: SoA buffer of size n_in * MAX_FANOUT */
) {
    todo!("§5: rayon parallel iter; thread-local writes into disjoint buffer regions")
}

/// Phase 2: partition the scratch buffer into `2^(2|support|)` buckets.
pub(crate) fn bucket_phase<const W: usize>(
    /* scratch in/out, support bits, bucket count */
) {
    todo!("§5: index by support bits; in-place partition or out-of-place scatter (TBD)")
}

/// Phase 3: segmented reduction with `keep_term` folded in.
pub(crate) fn merge_phase<const W: usize, T: TruncationPolicy<W>>(
    /* sorted scratch, output PauliSum */
    _policy: &T,
) {
    todo!("§5: linear pass; combine adjacent equal keys; drop terms rejected by keep_term")
}
