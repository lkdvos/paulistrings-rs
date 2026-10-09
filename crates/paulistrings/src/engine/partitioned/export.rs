//! The export pass: a layer's rows that belong to another partition, gathered into one [`ExchangeBlock`] per remote delta (ARCHITECTURE.md §Partitioning).

use num_complex::Complex64;
use rayon::prelude::*;

use super::plan::PartitionPlan;
use super::transport::{ChunkMap, ExchangeBlock, PartnerPayload};
use crate::channel::prepared::{DeltaEntry, LocalPtm, Prepared, PreparedRotation};
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::PauliSum;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Rows below which a block's fill stays on one thread.
const FILL_PARALLEL_MIN_ROWS: usize = 4096;

/// Reusable scratch for [`export_layer`], so a steady-state layer allocates nothing.
#[derive(Debug, Default)]
pub(crate) struct ExportScratch<const W: usize> {
    /// Pass-1 counts, bucket-major: `counts[β * K + k]`.
    counts: Vec<u32>,
    /// One block's counts by destination position.
    block_counts: Vec<u32>,
    /// `source_of[p] = map.bucket_at(p) ^ bd`.
    source_of: Vec<u32>,
    /// Payloads not in flight, columns intact; outgoing and incoming payloads are both drawn from here.
    pub(crate) pool: Vec<PartnerPayload<W>>,
}

/// What one layer's export sent, indexed by partner rank.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExportCounts {
    pub rows_to: Vec<u64>,
    /// Wire bytes ([`ExchangeBlock::bytes`]), the same under every transport.
    pub bytes_to: Vec<u64>,
}

/// One remote delta's row-level emitter: the tabulated entry, or the wide rotation's generator pass.
enum RowEmitter<'p, const W: usize> {
    Tabulated {
        ptm: &'p LocalPtm<W>,
        entry: &'p DeltaEntry<W>,
    },
    /// The identity pass of a rotation is never remote.
    Generator { rotation: &'p PreparedRotation<W> },
}

impl<const W: usize> RowEmitter<'_, W> {
    /// Whether every term emits a row; `amplitude` is sized `LOCAL_DIM` but only `4^k` entries are populated.
    fn is_dense(&self) -> bool {
        match self {
            RowEmitter::Tabulated { ptm, entry } => {
                let dim = 1usize << (2 * ptm.k());
                entry.amplitude[..dim].iter().all(|a| *a != ZERO)
            }
            RowEmitter::Generator { .. } => false,
        }
    }

    #[inline]
    fn emit(
        &self,
        x: &[u64; W],
        z: &[u64; W],
        c: Complex64,
    ) -> Option<([u64; W], [u64; W], Complex64)> {
        match self {
            RowEmitter::Tabulated { ptm, entry } => entry.emit(ptm.support_bits(x, z), x, z, c),
            RowEmitter::Generator { rotation } => rotation.emit_generator(x, z, c),
        }
    }
}

/// This partition's outgoing payloads for one layer, one slot per partner rank.
///
/// A partner's blocks are one per remote delta destined for it in ascending order, empty ones included, since the receiver indexes blocks positionally.
pub(crate) fn export_layer<const W: usize>(
    local: &PauliSum<W>,
    prepared: &Prepared<W>,
    plan: &PartitionPlan,
    size: u32,
    map: &ChunkMap,
    scratch: &mut ExportScratch<W>,
) -> (Vec<Option<PartnerPayload<W>>>, ExportCounts) {
    let k = plan.remote.len();
    let num_buckets = local.num_buckets();
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
        .map(|r| match prepared {
            Prepared::Local(ptm) => RowEmitter::Tabulated {
                ptm,
                entry: &ptm.deltas()[r.entry],
            },
            Prepared::Rotation(rotation) => {
                debug_assert_eq!(
                    r.entry, 1,
                    "a rotation's only remote entry is the generator pass",
                );
                RowEmitter::Generator { rotation }
            }
        })
        .collect();
    let dense: Vec<bool> = emitters.iter().map(RowEmitter::is_dense).collect();

    // Pass 1: bucket-major counts, `counts[β * k + i]`, so each bucket owns one disjoint chunk.
    scratch.counts.clear();
    scratch.counts.resize(num_buckets * k, 0);
    scratch
        .counts
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(b, slot)| count_bucket(local, prepared, plan, &dense, b, slot));

    // Pass 2: one block per remote delta, written by index into pooled storage that is neither allocated nor zeroed here.
    debug_assert_eq!(
        map.positions(),
        num_buckets,
        "the chunk map is built for a different bucket count than the sum has",
    );
    scratch.block_counts.clear();
    scratch.block_counts.resize(num_buckets, 0);
    scratch.source_of.clear();
    scratch.source_of.resize(num_buckets, 0);
    let mut blocks_used = vec![0usize; size as usize];
    for (i, r) in plan.remote.iter().enumerate() {
        for p in 0..num_buckets {
            let source = map.bucket_at(p as u32) ^ r.bucket_delta;
            scratch.source_of[p] = source;
            scratch.block_counts[p] = scratch.counts[source as usize * k + i];
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
            let columns = BlockColumns {
                x: &mut x[..rows],
                z: &mut z[..rows],
                c: &mut coeff[..rows],
            };
            fill_range(
                local,
                &emitters[i],
                offsets,
                &scratch.source_of,
                0,
                num_buckets,
                columns,
            );
        }
        counts.rows_to[q] += rows as u64;
        counts.bytes_to[q] += block.bytes() as u64;
    }
    // A pooled payload may carry more blocks than this layer fills; the extras must not travel.
    for (q, payload) in send.iter_mut().enumerate() {
        if let Some(payload) = payload {
            payload.blocks.truncate(blocks_used[q]);
        }
    }

    (send, counts)
}

/// Count one source bucket's exported rows, one slot per remote delta.
fn count_bucket<const W: usize>(
    local: &PauliSum<W>,
    prepared: &Prepared<W>,
    plan: &PartitionPlan,
    dense: &[bool],
    b: usize,
    out: &mut [u32],
) {
    let (bx, bz, bc) = local.bucket(b);
    let n = bc.len();
    match prepared {
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
                    if !dense[k] && ptm.deltas()[r.entry].amplitude[s] != ZERO {
                        out[k] += 1;
                    }
                }
            }
        }
        Prepared::Rotation(rotation) => {
            debug_assert_eq!(out.len(), 1, "a rotation has at most one remote delta");
            let mut anticommuting = 0u32;
            for t in 0..n {
                let v = PauliString::<W> { x: bx[t], z: bz[t] };
                if !v.commutes_with(&rotation.generator) {
                    anticommuting += 1;
                }
            }
            out[0] = anticommuting;
        }
    }
}

/// Fill destination positions `lo..hi` of one block; the halves of a split are disjoint, so the contents are independent of the thread count.
fn fill_range<const W: usize>(
    local: &PauliSum<W>,
    emitter: &RowEmitter<'_, W>,
    offsets: &[u32],
    source_of: &[u32],
    lo: usize,
    hi: usize,
    columns: BlockColumns<'_, W>,
) {
    debug_assert_eq!(columns.len(), (offsets[hi] - offsets[lo]) as usize);
    if hi - lo > 1 && columns.len() > FILL_PARALLEL_MIN_ROWS {
        let mid = lo + (hi - lo) / 2;
        let (head, tail) = columns.split_at((offsets[mid] - offsets[lo]) as usize);
        rayon::join(
            || fill_range(local, emitter, offsets, source_of, lo, mid, head),
            || fill_range(local, emitter, offsets, source_of, mid, hi, tail),
        );
        return;
    }
    let BlockColumns { x, z, c } = columns;
    let mut w = 0usize;
    for &source in &source_of[lo..hi] {
        let (bx, bz, bc) = local.bucket(source as usize);
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
struct BlockColumns<'a, const W: usize> {
    x: &'a mut [[u64; W]],
    z: &'a mut [[u64; W]],
    c: &'a mut [Complex64],
}

impl<'a, const W: usize> BlockColumns<'a, W> {
    fn len(&self) -> usize {
        self.c.len()
    }

    fn split_at(self, at: usize) -> (BlockColumns<'a, W>, BlockColumns<'a, W>) {
        let (x0, x1) = self.x.split_at_mut(at);
        let (z0, z1) = self.z.split_at_mut(at);
        let (c0, c1) = self.c.split_at_mut(at);
        (
            BlockColumns {
                x: x0,
                z: z0,
                c: c0,
            },
            BlockColumns {
                x: x1,
                z: z1,
                c: c1,
            },
        )
    }
}

/// Every exported row belongs to the partner it is addressed to.
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
