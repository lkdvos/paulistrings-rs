//! The export pass: rows a layer generates that belong to another partition, gathered into one [`ExchangeBlock`] per remote delta.
//!
//! A remote delta (see [`PartitionPlan`]) moves every row it produces to one partner, so export is a pure per-delta gather with no per-term routing decision.
//! Each block is CSR in the receiver's destination-coset order ([`ChunkMap`](super::transport::ChunkMap)), which is what makes a receiving coset's rows contiguous so the transfer can be cut into chunks.
//! Two parallel passes over the local buckets: count rows per (remote delta, source bucket), then fill each block position by position.
//! The row arithmetic itself is not reimplemented here: see [`DeltaEntry::emit`](crate::channel::prepared::DeltaEntry::emit) and [`RotationPrep::emit_gen`](crate::channel::prepared::RotationPrep::emit_gen).

use num_complex::Complex64;
use rayon::prelude::*;

use super::plan::PartitionPlan;
use super::transport::{ChunkMap, ExchangeBlock, PartnerPayload};
use crate::channel::prepared::{DeltaEntry, LocalPtm, Prepared, RotationPrep};
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::PauliSum;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Rows below which a block's fill stays on one thread.
///
/// The fill splits a block's position range in two and hands the halves to `rayon::join`; this threshold is purely about not paying task overhead for a handful of rows.
const FILL_PARALLEL_MIN_ROWS: usize = 4096;

/// Reusable scratch for [`export_layer`], held by the caller across layers so a layer allocates nothing after the first.
///
/// [`Self::counts`] is the pass-1 count table in bucket-major order (`counts[β * K + k]` is remote delta `k`'s row count from source bucket `β`), the layout that lets pass 1 run as one disjoint `par_chunks_mut(K)` over buckets.
/// [`Self::block_counts`] is that layout transposed into a per-block column for [`ExchangeBlock::set_counts`].
#[derive(Debug)]
pub(crate) struct ExportScratch<const W: usize> {
    /// Pass-1 counts, bucket-major: `counts[β * K + k]`.
    counts: Vec<u32>,
    /// One block's counts by destination position, gathered out of
    /// [`Self::counts`].
    block_counts: Vec<u32>,
    /// One block's source bucket per destination position:
    /// `src_of[p] = map.bucket_at(p) ^ bd`. Built once per block and read by
    /// both the count transpose and the fill.
    src_of: Vec<u32>,
    /// Payloads not currently in flight, with their block columns intact.
    ///
    /// The layer's own pool ([`Transport::exchange_layer`](super::transport::Transport::exchange_layer)): both outgoing and incoming payloads are drawn from here and returned when the layer is done with them, so a steady-state remote layer allocates nothing at all.
    pub(crate) pool: Vec<PartnerPayload<W>>,
}

// Hand-written because `#[derive(Default)]` would demand `W: Default`.
impl<const W: usize> Default for ExportScratch<W> {
    fn default() -> Self {
        Self {
            counts: Vec::new(),
            block_counts: Vec::new(),
            src_of: Vec::new(),
            pool: Vec::new(),
        }
    }
}

/// What one layer's export costs, by partner rank (length is the group size).
///
/// Rows and bytes are what this partition *sent*; a partner it sent nothing to has zeros.
/// The bytes are the wire footprint ([`ExchangeBlock::bytes`]) of the blocks as they stand, so an in-process transport reports the same number an MPI one would move.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExportCounts {
    /// Rows sent to each partner.
    pub rows_to: Vec<u64>,
    /// Wire bytes sent to each partner.
    pub bytes_to: Vec<u64>,
}

/// One remote delta's row-level emitter: the tabulated entry, or the wide rotation's generator pass.
///
/// Exists so the two prepared forms share the count and fill loops.
enum RowEmitter<'p, const W: usize> {
    /// A tabulated delta of a [`Prepared::Local`] layer.
    Tabulated {
        ptm: &'p LocalPtm<W>,
        entry: &'p DeltaEntry<W>,
    },
    /// The generator pass of a [`Prepared::Rotation`] layer; the identity pass is never remote.
    Generator { prep: &'p RotationPrep<W> },
}

impl<const W: usize> RowEmitter<'_, W> {
    /// Whether every term emits a row, so a bucket's count is its length.
    ///
    /// Dense over the *active* support patterns only: `amp` is sized `LOCAL_DIM` but the channel populates `4^k` entries.
    /// A rotation's generator pass is never dense, since a commuting term emits nothing.
    fn is_dense(&self) -> bool {
        match self {
            RowEmitter::Tabulated { ptm, entry } => {
                let dim = 1usize << (2 * ptm.k());
                entry.amp[..dim].iter().all(|a| *a != ZERO)
            }
            RowEmitter::Generator { .. } => false,
        }
    }

    /// The row this delta emits for one term, or `None` when it emits none.
    #[inline]
    fn emit(
        &self,
        x: &[u64; W],
        z: &[u64; W],
        c: Complex64,
    ) -> Option<([u64; W], [u64; W], Complex64)> {
        match self {
            RowEmitter::Tabulated { ptm, entry } => entry.emit(ptm.support_bits(x, z), x, z, c),
            RowEmitter::Generator { prep } => prep.emit_gen(x, z, c),
        }
    }
}

/// Build this partition's outgoing payloads for one layer.
///
/// Returns one slot per partner rank: `Some(payload)` for every partner the plan names, `None` for the rest (including this partition's own slot).
/// A partner's [`PartnerPayload::blocks`] is one block per remote delta destined for it, in ascending remote-delta index, *including blocks with no rows*: the receiver indexes blocks positionally.
pub(crate) fn export_layer<const W: usize>(
    local: &PauliSum<W>,
    prep: &Prepared<W>,
    plan: &PartitionPlan,
    size: u32,
    map: &ChunkMap,
    scratch: &mut ExportScratch<W>,
) -> (Vec<Option<PartnerPayload<W>>>, ExportCounts) {
    let k = plan.remote.len();
    let nb = local.num_buckets();
    let mut send: Vec<Option<PartnerPayload<W>>> = (0..size).map(|_| None).collect();
    let mut counts = ExportCounts {
        rows_to: vec![0; size as usize],
        bytes_to: vec![0; size as usize],
    };
    if k == 0 {
        return (send, counts);
    }

    let emitters: Vec<RowEmitter<'_, W>> = plan
        .remote
        .iter()
        .map(|r| match prep {
            Prepared::Local(ptm) => RowEmitter::Tabulated {
                ptm,
                entry: &ptm.deltas()[r.entry],
            },
            Prepared::Rotation(p) => {
                debug_assert_eq!(
                    r.entry, 1,
                    "a rotation's only remote entry is the generator pass",
                );
                RowEmitter::Generator { prep: p }
            }
        })
        .collect();
    let dense: Vec<bool> = emitters.iter().map(RowEmitter::is_dense).collect();

    // Pass 1. Bucket-major counts, so each bucket owns one contiguous chunk and the pass needs no synchronization.
    scratch.counts.clear();
    scratch.counts.resize(nb * k, 0);
    scratch
        .counts
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(b, slot)| count_bucket(local, prep, plan, &dense, b, slot));

    // Pass 2, one block per remote delta, written into payloads taken from the pool: the fill writes by index into storage that is neither allocated nor zeroed here.
    debug_assert_eq!(
        map.positions(),
        nb,
        "the chunk map is built for a different bucket count than the sum has",
    );
    scratch.block_counts.clear();
    scratch.block_counts.resize(nb, 0);
    scratch.src_of.clear();
    scratch.src_of.resize(nb, 0);
    let mut blocks_used = vec![0usize; size as usize];
    for (i, r) in plan.remote.iter().enumerate() {
        // Destination-coset order: position `p` of the receiver is filled from this partition's bucket `bucket_at(p) ^ bd`.
        for p in 0..nb {
            let src = map.bucket_at(p as u32) ^ r.bucket_delta;
            scratch.src_of[p] = src;
            scratch.block_counts[p] = scratch.counts[src as usize * k + i];
        }
        let q = r.partner as usize;
        let payload = send[q].get_or_insert_with(|| scratch.pool.pop().unwrap_or_default());
        let j = blocks_used[q];
        blocks_used[q] += 1;
        if payload.blocks.len() <= j {
            payload
                .blocks
                .resize_with(j + 1, ExchangeBlock::<W>::default);
        }
        let block = &mut payload.blocks[j];
        block.set_counts(r.entry as u32, &scratch.block_counts);
        let rows = block.rows();
        {
            let ExchangeBlock {
                ref offsets,
                ref mut x,
                ref mut z,
                ref mut coeff,
                ..
            } = *block;
            let cols = BlockCols {
                x: &mut x[..rows],
                z: &mut z[..rows],
                c: &mut coeff[..rows],
            };
            fill_range(local, &emitters[i], offsets, &scratch.src_of, 0, nb, cols);
        }
        counts.rows_to[q] += rows as u64;
        counts.bytes_to[q] += block.bytes() as u64;
    }
    // A payload out of the pool may have carried more blocks than this layer has remote deltas for it; the extras must not travel.
    for (q, payload) in send.iter_mut().enumerate() {
        if let Some(payload) = payload {
            payload.blocks.truncate(blocks_used[q]);
        }
    }

    (send, counts)
}

/// Count one source bucket's exported rows, one slot per remote delta.
///
/// The support pattern is computed once per term and reused across every sparse entry; dense entries are filled from the bucket length without touching a term at all.
fn count_bucket<const W: usize>(
    local: &PauliSum<W>,
    prep: &Prepared<W>,
    plan: &PartitionPlan,
    dense: &[bool],
    b: usize,
    out: &mut [u32],
) {
    let (bx, bz, bc) = local.bucket(b);
    let n = bc.len();
    match prep {
        Prepared::Local(ptm) => {
            let mut all_dense = true;
            for (k, &d) in dense.iter().enumerate() {
                if d {
                    out[k] = n as u32;
                } else {
                    out[k] = 0;
                    all_dense = false;
                }
            }
            if all_dense {
                return;
            }
            for t in 0..n {
                let s = ptm.support_bits(&bx[t], &bz[t]);
                for (k, r) in plan.remote.iter().enumerate() {
                    if !dense[k] && ptm.deltas()[r.entry].amp[s] != ZERO {
                        out[k] += 1;
                    }
                }
            }
        }
        Prepared::Rotation(p) => {
            debug_assert_eq!(out.len(), 1, "a rotation has at most one remote delta");
            let mut anti = 0u32;
            for t in 0..n {
                let v = PauliString::<W> { x: bx[t], z: bz[t] };
                if !v.commutes_with(&p.gen) {
                    anti += 1;
                }
            }
            out[0] = anti;
        }
    }
}

/// Fill destination positions `lo..hi` of one block.
///
/// `src_of[p]` is the source bucket position `p` draws from.
/// Splitting the range at `mid` splits the columns at `offsets[mid] - offsets[lo]`, and the two halves are disjoint by construction, so a block's contents are independent of the thread count.
fn fill_range<const W: usize>(
    local: &PauliSum<W>,
    emitter: &RowEmitter<'_, W>,
    offsets: &[u32],
    src_of: &[u32],
    lo: usize,
    hi: usize,
    cols: BlockCols<'_, W>,
) {
    debug_assert_eq!(cols.len(), (offsets[hi] - offsets[lo]) as usize);
    if hi - lo > 1 && cols.len() > FILL_PARALLEL_MIN_ROWS {
        let mid = lo + (hi - lo) / 2;
        let (head, tail) = cols.split_at((offsets[mid] - offsets[lo]) as usize);
        rayon::join(
            || fill_range(local, emitter, offsets, src_of, lo, mid, head),
            || fill_range(local, emitter, offsets, src_of, mid, hi, tail),
        );
        return;
    }
    let BlockCols { x, z, c } = cols;
    let mut w = 0usize;
    for &src in &src_of[lo..hi] {
        let (bx, bz, bc) = local.bucket(src as usize);
        for t in 0..bc.len() {
            if let Some((kx, kz, kc)) = emitter.emit(&bx[t], &bz[t], bc[t]) {
                x[w] = kx;
                z[w] = kz;
                c[w] = kc;
                w += 1;
            }
        }
    }
    debug_assert_eq!(
        w,
        x.len(),
        "export: pass 2 filled {w} rows where pass 1 counted {}",
        x.len(),
    );
}

/// One block's three columns, restricted to a source-bucket range.
///
/// A bundle, not an abstraction: keeps [`fill_range`]'s recursive signature inside clippy's argument budget.
struct BlockCols<'a, const W: usize> {
    x: &'a mut [[u64; W]],
    z: &'a mut [[u64; W]],
    c: &'a mut [Complex64],
}

impl<'a, const W: usize> BlockCols<'a, W> {
    /// Rows in the range.
    fn len(&self) -> usize {
        self.c.len()
    }

    /// Split every column at the same row, consuming the borrow.
    fn split_at(self, at: usize) -> (BlockCols<'a, W>, BlockCols<'a, W>) {
        let (x0, x1) = self.x.split_at_mut(at);
        let (z0, z1) = self.z.split_at_mut(at);
        let (c0, c1) = self.c.split_at_mut(at);
        (
            BlockCols {
                x: x0,
                z: z0,
                c: c0,
            },
            BlockCols {
                x: x1,
                z: z1,
                c: c1,
            },
        )
    }
}

/// Every exported row belongs to the partner it is addressed to.
///
/// Debug builds only, `O(exported rows)`.
/// A failure means either the plan misclassified a delta or the caller's `local` sum was not the pure partition it claims to be.
#[cfg(debug_assertions)]
pub(super) fn debug_assert_exported_partitions<const W: usize>(
    send: &[Option<PartnerPayload<W>>],
    rows: &crate::pauli_sum::hash::PartitionRows<W>,
) {
    for (q, payload) in send.iter().enumerate() {
        let Some(payload) = payload else { continue };
        for block in &payload.blocks {
            for i in 0..block.rows() {
                let part = rows.partition_of(&block.x[i], &block.z[i]);
                debug_assert_eq!(
                    part, q as u32,
                    "export: row {i} of the block for entry {} is addressed to partition {q} but \
                     belongs to partition {part}",
                    block.header.entry,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests;
