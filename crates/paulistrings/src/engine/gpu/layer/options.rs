//! The device layer's knobs: [`GpuLayerOptions`] and the [`GpuBucketPolicy`] that sizes a layer's bucket count.

use crate::channel::prepared::Prepared;
use crate::pauli_sum::hash::B_MAX_BITS;
use crate::pauli_sum::storage::desired_bits;

/// Records per fused block the default bucket policy aims for.
pub const DEFAULT_RECORDS_PER_BLOCK: usize = 4096;

/// Default cap on the loose output arena, in bytes.
pub const DEFAULT_ARENA_BYTES: usize = 4 << 30;

/// How the device chooses its bucket count before a layer; grow-only either way, as [`PauliSum::rebucket`](crate::PauliSum::rebucket).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBucketPolicy {
    /// `fanout × terms per bucket ≈ target`, so a fused block sees about `target` records whatever the channel's fanout.
    RecordsPerBlock(usize),
    /// A fixed `desired_bits(len, target, 1)`, independent of the channel.
    TermsPerBucket(usize),
}

impl Default for GpuBucketPolicy {
    fn default() -> Self {
        GpuBucketPolicy::RecordsPerBlock(DEFAULT_RECORDS_PER_BLOCK)
    }
}

/// Knobs of the device layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuLayerOptions {
    /// The bucket count the device refines to before a layer.
    pub bucket_policy: GpuBucketPolicy,
    /// Bytes the loose output arena may hold; positions are batched so no batch's pre-dedup rows exceed it.
    pub arena_bytes: usize,
    /// Bucket bits a layer may refine to before an oversize block is [`GpuError::Unsupported`](crate::gpu::GpuError::Unsupported); `B_MAX_BITS` by default.
    pub max_bits: u8,
    /// Merge one partner's exported rows by key on the sender before the exchange (ARCHITECTURE.md §Partitioning); on by default.
    pub premerge: bool,
    /// Bytes the received rows of a remote layer may hold on the device at once, at `(2W + 3) × 8` per row.
    ///
    /// The receive then moves in chunks of destination positions, a power of two of them, each merged by the fused layer before the next arrives (ARCHITECTURE.md §Partitioning); a chunk exceeds the cap only when one position alone does.
    /// Unbounded, one chunk, unless `PAULISTRINGS_GPU_EXCHANGE_BYTES` names a byte count (`K`, `M` and `G` suffixes are binary).
    pub exchange_bytes: usize,
    /// Run a local layer whose table is a permutation of keys (every Clifford's) by the scatter path, K12–K14, instead of the fused layer; on by default.
    pub clifford: bool,
}

/// The default of [`GpuLayerOptions::exchange_bytes`], read once per process.
fn exchange_bytes_default() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        parse_bytes(
            std::env::var("PAULISTRINGS_GPU_EXCHANGE_BYTES")
                .ok()
                .as_deref(),
        )
    })
}

/// A positive byte count with an optional binary `K`/`M`/`G` suffix; anything else is unbounded.
pub(super) fn parse_bytes(raw: Option<&str>) -> usize {
    let Some(raw) = raw.map(str::trim) else {
        return usize::MAX;
    };
    let (digits, shift) = match raw.as_bytes().last() {
        Some(b'K' | b'k') => (&raw[..raw.len() - 1], 10),
        Some(b'M' | b'm') => (&raw[..raw.len() - 1], 20),
        Some(b'G' | b'g') => (&raw[..raw.len() - 1], 30),
        _ => (raw, 0),
    };
    digits
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
        .and_then(|n| n.checked_mul(1usize << shift))
        .unwrap_or(usize::MAX)
}

impl Default for GpuLayerOptions {
    fn default() -> Self {
        Self {
            bucket_policy: GpuBucketPolicy::default(),
            arena_bytes: DEFAULT_ARENA_BYTES,
            max_bits: B_MAX_BITS,
            premerge: true,
            exchange_bytes: exchange_bytes_default(),
            clifford: true,
        }
    }
}

/// The bucket bits a layer of `fanout` entries over `len` terms wants under `policy`, never below `current`.
pub(crate) fn gpu_desired_bits(
    len: usize,
    fanout: usize,
    policy: GpuBucketPolicy,
    current: u8,
) -> u8 {
    let (rows, target) = match policy {
        GpuBucketPolicy::TermsPerBucket(target) => (len, target),
        GpuBucketPolicy::RecordsPerBlock(target) => (len.saturating_mul(fanout.max(1)), target),
    };
    desired_bits(rows, target.max(1), 1)
        .max(current)
        .min(B_MAX_BITS)
}

/// The entries `prep` emits records for, local and received alike: the fanout [`gpu_desired_bits`] sizes a block by.
pub(crate) fn prepared_fanout<const W: usize>(prep: &Prepared<W>) -> usize {
    match prep {
        Prepared::Local(ptm) => ptm
            .deltas()
            .iter()
            .filter(|d| {
                d.amp
                    .iter()
                    .any(|a| *a != num_complex::Complex64::new(0.0, 0.0))
            })
            .count(),
        Prepared::Rotation(_) => 2,
    }
}
