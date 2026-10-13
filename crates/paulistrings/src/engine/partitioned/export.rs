//! The export pass: a layer's rows that belong to another partition, gathered into one [`ExchangeBlock`] per remote delta (ARCHITECTURE.md §Partitioning).

use num_complex::Complex64;
use rayon::prelude::*;

use super::layer::LayerExchangeCounts;
use super::plan::PartitionPlan;
use super::transport::{ExchangeBlock, PartnerPayload};
use crate::channel::prepared::{DeltaEntry, LocalPtm, Prepared, PreparedRotation};
use crate::engine::coset::CosetOrder;
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::{ColumnsMut, PauliSum};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Rows below which a block's fill stays on one thread.
const FILL_PARALLEL_MIN_ROWS: usize = 4096;

/// Reusable scratch for [`export_layer`], so a steady-state layer allocates nothing.
#[derive(Debug, Default)]
pub(crate) struct ExportScratch<const W: usize> {
    /// Pass-1 counts, bucket-major: `counts[β * num_remote + i]`.
    counts: Vec<u32>,
    /// One block's counts by destination position.
    block_counts: Vec<u32>,
    /// `source_of[position] = order.bucket_at(position) ^ bucket_delta`.
    source_of: Vec<u32>,
    /// Payloads not in flight, columns intact; outgoing and incoming payloads are both drawn from here.
    pub(crate) pool: Vec<PartnerPayload<W>>,
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
    fn emit(
        &self,
        x: &[u64; W],
        z: &[u64; W],
        coeff: Complex64,
    ) -> Option<([u64; W], [u64; W], Complex64)> {
        match self {
            RowEmitter::Tabulated { ptm, entry } => entry.emit(ptm.support_bits(x, z), x, z, coeff),
            RowEmitter::Generator { rotation } => rotation.emit_generator(x, z, coeff),
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
    order: &CosetOrder,
    scratch: &mut ExportScratch<W>,
) -> (Vec<Option<PartnerPayload<W>>>, LayerExchangeCounts) {
    let num_remote = plan.remote.len();
    let num_buckets = local.num_buckets();
    let mut send: Vec<Option<PartnerPayload<W>>> = (0..size).map(|_| None).collect();
    let mut counts = LayerExchangeCounts::none(size);
    if num_remote == 0 {
        return (send, counts);
    }

    let emitters: Vec<RowEmitter<'_, W>> = plan
        .remote
        .iter()
        .map(|remote| match prepared {
            Prepared::Local(ptm) => RowEmitter::Tabulated {
                ptm,
                entry: &ptm.deltas()[remote.entry],
            },
            Prepared::Rotation(rotation) => {
                debug_assert_eq!(
                    remote.entry, 1,
                    "a rotation's only remote entry is the generator pass",
                );
                RowEmitter::Generator { rotation }
            }
        })
        .collect();
    // Whether every term emits a row, which the generator pass never guarantees.
    let dense: Vec<bool> = plan
        .remote
        .iter()
        .map(|remote| match prepared {
            Prepared::Local(ptm) => ptm.entry_is_dense(remote.entry),
            Prepared::Rotation(_) => false,
        })
        .collect();

    // Pass 1: bucket-major counts, `counts[β * num_remote + i]`, so each bucket owns one disjoint chunk.
    scratch.counts.clear();
    scratch.counts.resize(num_buckets * num_remote, 0);
    scratch
        .counts
        .par_chunks_mut(num_remote)
        .enumerate()
        .for_each(|(bucket, slot)| count_bucket(local, prepared, plan, &dense, bucket, slot));

    // Pass 2: one block per remote delta, written by index into pooled storage that is neither allocated nor zeroed here.
    debug_assert_eq!(
        order.positions(),
        num_buckets,
        "the coset order is built for a different bucket count than the sum has",
    );
    scratch.block_counts.clear();
    scratch.block_counts.resize(num_buckets, 0);
    scratch.source_of.clear();
    scratch.source_of.resize(num_buckets, 0);
    let mut blocks_used = vec![0usize; size as usize];
    for (i, remote) in plan.remote.iter().enumerate() {
        for position in 0..num_buckets {
            let source = order.bucket_at(position as u32) ^ remote.bucket_delta;
            scratch.source_of[position] = source;
            scratch.block_counts[position] = scratch.counts[source as usize * num_remote + i];
        }
        let partner = remote.partner as usize;
        let payload = send[partner].get_or_insert_with(|| scratch.pool.pop().unwrap_or_default());
        let block_index = plan.sent_block_index(i);
        blocks_used[partner] = block_index + 1;
        if payload.blocks.len() <= block_index {
            payload
                .blocks
                .resize_with(block_index + 1, ExchangeBlock::<W>::default);
        }
        let block = &mut payload.blocks[block_index];
        block.set_counts(remote.entry as u32, &scratch.block_counts);
        let rows = block.rows();
        {
            let ExchangeBlock {
                ref offsets,
                ref mut x,
                ref mut z,
                ref mut coeff,
                ..
            } = *block;
            let columns = ColumnsMut {
                x: &mut x[..rows],
                z: &mut z[..rows],
                coeff: &mut coeff[..rows],
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
        counts.rows_sent[partner] += rows as u64;
        counts.bytes_sent[partner] += block.bytes() as u64;
    }
    // A pooled payload may carry more blocks than this layer fills; the extras must not travel.
    for (partner, payload) in send.iter_mut().enumerate() {
        if let Some(payload) = payload {
            payload.blocks.truncate(blocks_used[partner]);
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
    bucket: usize,
    out: &mut [u32],
) {
    let (bucket_x, bucket_z, bucket_coeff) = local.bucket(bucket);
    let len = bucket_coeff.len();
    match prepared {
        Prepared::Local(ptm) => {
            let mut all_dense = true;
            for (i, &is_dense) in dense.iter().enumerate() {
                if is_dense {
                    out[i] = len as u32;
                } else {
                    out[i] = 0;
                    all_dense = false;
                }
            }
            if all_dense {
                return;
            }
            for term in 0..len {
                let support = ptm.support_bits(&bucket_x[term], &bucket_z[term]);
                for (i, remote) in plan.remote.iter().enumerate() {
                    if !dense[i] && ptm.deltas()[remote.entry].amplitude[support] != ZERO {
                        out[i] += 1;
                    }
                }
            }
        }
        Prepared::Rotation(rotation) => {
            debug_assert_eq!(out.len(), 1, "a rotation has at most one remote delta");
            let mut anticommuting = 0u32;
            for term in 0..len {
                let string = PauliString::<W> {
                    x: bucket_x[term],
                    z: bucket_z[term],
                };
                if !string.commutes_with(&rotation.generator) {
                    anticommuting += 1;
                }
            }
            out[0] = anticommuting;
        }
    }
}

/// Fill destination positions `start..end` of one block; the halves of a split are disjoint, so the contents are independent of the thread count.
fn fill_range<const W: usize>(
    local: &PauliSum<W>,
    emitter: &RowEmitter<'_, W>,
    offsets: &[u32],
    source_of: &[u32],
    start: usize,
    end: usize,
    columns: ColumnsMut<'_, W>,
) {
    debug_assert_eq!(columns.len(), (offsets[end] - offsets[start]) as usize);
    if end - start > 1 && columns.len() > FILL_PARALLEL_MIN_ROWS {
        let middle = start + (end - start) / 2;
        let (head, tail) = columns.split_at((offsets[middle] - offsets[start]) as usize);
        rayon::join(
            || fill_range(local, emitter, offsets, source_of, start, middle, head),
            || fill_range(local, emitter, offsets, source_of, middle, end, tail),
        );
        return;
    }
    let ColumnsMut { x, z, coeff } = columns;
    let mut written = 0usize;
    for &source in &source_of[start..end] {
        let (bucket_x, bucket_z, bucket_coeff) = local.bucket(source as usize);
        for term in 0..bucket_coeff.len() {
            if let Some((out_x, out_z, out_coeff)) =
                emitter.emit(&bucket_x[term], &bucket_z[term], bucket_coeff[term])
            {
                x[written] = out_x;
                z[written] = out_z;
                coeff[written] = out_coeff;
                written += 1;
            }
        }
    }
    debug_assert_eq!(
        written,
        x.len(),
        "export: pass 2 filled {written} rows where pass 1 counted {}",
        x.len(),
    );
}

/// Every exported row belongs to the partner it is addressed to.
#[cfg(debug_assertions)]
pub(super) fn debug_assert_exported_partitions<const W: usize>(
    send: &[Option<PartnerPayload<W>>],
    rows: &crate::pauli_sum::hash::PartitionRows<W>,
) {
    for (partner, payload) in send.iter().enumerate() {
        let Some(payload) = payload else { continue };
        for block in &payload.blocks {
            for i in 0..block.rows() {
                let owner = rows.partition_of(&block.x[i], &block.z[i]);
                debug_assert_eq!(
                    owner, partner as u32,
                    "export: row {i} of the block for entry {} is addressed to partition {partner} \
                     but belongs to partition {owner}",
                    block.header.entry,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests;
