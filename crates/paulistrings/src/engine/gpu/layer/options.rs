//! The device layer's knobs, [`GpuLayerOptions`], and the bucket count they size a layer at.

use crate::channel::prepared::Prepared;
use crate::engine::gpu::GPU_LOG_TARGET;
use crate::engine::knob::Knob;
use crate::pauli_sum::hash::B_MAX_BITS;
use crate::pauli_sum::storage::desired_bits;

/// Records per fused block the default bucket count aims for.
pub const DEFAULT_RECORDS_PER_BLOCK: usize = 4096;

/// Default cap on the loose output arena, in bytes.
pub const DEFAULT_ARENA_BYTES: usize = 4 << 30;

/// Knobs of the device layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuLayerOptions {
    /// Records a fused block aims for: the device refines before a layer until `fanout × terms per bucket` is about this many, grow-only as on the host; at least 1.
    pub records_per_block: usize,
    /// Bytes the loose output arena may hold; positions are batched so no batch's pre-dedup rows exceed it; at least 1.
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
    /// Run a local layer whose table is a permutation of keys (every Clifford's) by the scatter path (`k_permute_*`) instead of the fused layer; on by default.
    pub clifford: bool,
}

/// `PAULISTRINGS_GPU_EXCHANGE_BYTES`, the default of [`GpuLayerOptions::exchange_bytes`]; unbounded when unset.
static EXCHANGE_BYTES: Knob<usize> = Knob::new(
    "PAULISTRINGS_GPU_EXCHANGE_BYTES",
    GPU_LOG_TARGET,
    parse_bytes,
    || usize::MAX,
);

/// A positive byte count with an optional binary `K`/`M`/`G` suffix.
pub(super) fn parse_bytes(text: &str) -> Option<usize> {
    let (digits, shift) = match text.as_bytes().last() {
        Some(b'K' | b'k') => (&text[..text.len() - 1], 10),
        Some(b'M' | b'm') => (&text[..text.len() - 1], 20),
        Some(b'G' | b'g') => (&text[..text.len() - 1], 30),
        _ => (text, 0),
    };
    digits
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
        .and_then(|n| n.checked_mul(1usize << shift))
}

impl Default for GpuLayerOptions {
    fn default() -> Self {
        Self {
            records_per_block: DEFAULT_RECORDS_PER_BLOCK,
            arena_bytes: DEFAULT_ARENA_BYTES,
            max_bits: B_MAX_BITS,
            premerge: true,
            exchange_bytes: *EXCHANGE_BYTES.get(),
            clifford: true,
        }
    }
}

impl GpuLayerOptions {
    /// `Err` with the message `set_layer_options` and a device scatter panic with unless `records_per_block >= 1`, `arena_bytes >= 1` and `max_bits <= B_MAX_BITS`.
    pub fn validate(&self) -> Result<(), String> {
        if self.records_per_block == 0 {
            return Err("records_per_block must be at least 1, got 0".to_string());
        }
        if self.arena_bytes == 0 {
            return Err("arena_bytes must be at least 1, got 0".to_string());
        }
        if self.max_bits > B_MAX_BITS {
            return Err(format!(
                "max_bits must be at most {B_MAX_BITS}, got {}",
                self.max_bits
            ));
        }
        Ok(())
    }

    /// Panics with [`validate`](Self::validate)'s message.
    pub(crate) fn assert_valid(&self) {
        if let Err(message) = self.validate() {
            panic!("{message}");
        }
    }
}

/// The bucket bits a layer of `fanout` entries over `len` terms wants at `records_per_block`, never below `current`.
pub(crate) fn gpu_desired_bits(
    len: usize,
    fanout: usize,
    records_per_block: usize,
    current: u8,
) -> u8 {
    desired_bits(
        len.saturating_mul(fanout.max(1)),
        records_per_block.max(1),
        1,
    )
    .max(current)
    .min(B_MAX_BITS)
}

/// The entries `prepared` emits records for, local and received alike: the fanout [`gpu_desired_bits`] sizes a block by.
pub(crate) fn prepared_fanout<const W: usize>(prepared: &Prepared<W>) -> usize {
    match prepared {
        Prepared::Local(ptm) => ptm.emitting_entries(),
        Prepared::Rotation(_) => 2,
    }
}
