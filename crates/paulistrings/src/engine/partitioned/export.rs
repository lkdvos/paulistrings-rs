//! The export pass: a layer's rows that belong to another partition, gathered into one [`ExchangeBlock`] per remote delta.
//!
//! A row of remote delta `d` from local bucket `β` belongs to partition `rank ⊕ part(d)`, bucket `β ⊕ h(d)`, so the sender fills segment `p` from its bucket `bucket_at(p) ⊕ h(d)` and the block arrives in the receiver's coset order.
//! A partner gets one block per remote delta to it, in ascending plan order and empty ones included, since the receiver walks them in lockstep with its own plan.

use rayon::prelude::*;

use crate::coefficient::Coefficient;

use super::exchange_block::{ExchangeBlock, PartnerPayload};
use super::layer::LayerExchangeCounts;
use crate::channel::prepared::{DeltaEntry, LocalPtm, Prepared, PreparedRotation};
use crate::engine::coset::CosetOrder;
use crate::engine::plan::PartitionPlan;
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::{ColumnsMut, PauliSum};

/// Scanned input terms below which a block's fill stays on one thread.
const FILL_PARALLEL_MIN_TERMS: usize = 16384;

/// Reusable scratch for [`prepare_export`], so a steady-state layer allocates nothing.
#[derive(Debug, Default)]
pub(crate) struct ExportScratch {
    /// Pass-1 counts, bucket-major: `counts[β * num_remote + i]`.
    counts: Vec<u32>,
    /// One block's counts by destination position.
    block_counts: Vec<u32>,
    /// Remote delta `i`'s `source_of[position] = order.bucket_at(position) ^ bucket_delta`, at `i * num_buckets`.
    source_of: Vec<u32>,
    /// Remote delta `i`'s CSR offsets by destination position, at `i * (num_buckets + 1)`.
    offsets: Vec<u32>,
    /// Remote delta `i`'s prefix sums of the input terms its fill scans, at `i * (num_buckets + 1)`; a position that exports nothing scans nothing.
    work_prefix: Vec<usize>,
}

/// One remote delta's row-level emitter: the tabulated entry, or the wide rotation's generator pass.
enum RowEmitter<'p, const W: usize, C: Coefficient> {
    Tabulated {
        ptm: &'p LocalPtm<W, C>,
        entry: &'p DeltaEntry<W, C>,
    },
    /// The identity pass of a rotation is never remote.
    Generator {
        rotation: &'p PreparedRotation<W, C::Real>,
    },
}

impl<const W: usize, C: Coefficient> RowEmitter<'_, W, C> {
    fn emit(&self, x: &[u64; W], z: &[u64; W], coeff: C) -> Option<([u64; W], [u64; W], C)> {
        match self {
            RowEmitter::Tabulated { ptm, entry } => entry.emit(ptm.support_bits(x, z), x, z, coeff),
            RowEmitter::Generator { rotation } => rotation.emit_generator(x, z, coeff),
        }
    }
}

/// This partition's outgoing payloads for one layer, filled whole.
#[cfg(test)]
pub(crate) fn export_layer<const W: usize, C: Coefficient>(
    local: &PauliSum<W, C>,
    prepared: &Prepared<W, C>,
    plan: &PartitionPlan,
    size: u32,
    order: &CosetOrder,
    scratch: &mut ExportScratch,
) -> (Vec<Option<PartnerPayload<W, C>>>, LayerExchangeCounts) {
    export_layer_with(local, prepared, plan, size, order, scratch, FILL_PARALLEL_MIN_TERMS)
}

/// [`export_layer`] with the fill's leaf size in scanned input terms.
#[cfg(test)]
fn export_layer_with<const W: usize, C: Coefficient>(
    local: &PauliSum<W, C>,
    prepared: &Prepared<W, C>,
    plan: &PartitionPlan,
    size: u32,
    order: &CosetOrder,
    scratch: &mut ExportScratch,
    min_terms: usize,
) -> (Vec<Option<PartnerPayload<W, C>>>, LayerExchangeCounts) {
    use crate::engine::transport::{chunk_views, ChunkMap, Payload};
    let (mut send, counts, fill) = prepare_export_with(
        local,
        prepared,
        plan,
        size,
        order,
        scratch,
        &mut Vec::new(),
        min_terms,
    );
    let map = ChunkMap::of_order(order, 1);
    let mut bulk: Vec<Option<Vec<Vec<&mut [u8]>>>> = send
        .iter_mut()
        .map(|payload| payload.as_mut().map(|payload| payload.split_for_send(&map).1))
        .collect();
    if plan.has_remote() {
        fill.chunk(local, 0, &map, &mut chunk_views(&mut bulk, 0));
    }
    drop(bulk);
    (send, counts)
}

/// This partition's outgoing payloads for one layer, one slot per partner rank, sized and indexed but not yet written; the [`ExportFill`] writes them chunk by chunk.
///
/// A partner's blocks are one per remote delta destined for it in ascending order, empty ones included, since the receiver indexes blocks positionally.
pub(crate) fn prepare_export<'a, const W: usize, C: Coefficient>(
    local: &PauliSum<W, C>,
    prepared: &'a Prepared<W, C>,
    plan: &PartitionPlan,
    size: u32,
    order: &CosetOrder,
    scratch: &'a mut ExportScratch,
    pool: &mut Vec<PartnerPayload<W, C>>,
) -> (Vec<Option<PartnerPayload<W, C>>>, LayerExchangeCounts, ExportFill<'a, W, C>) {
    prepare_export_with(local, prepared, plan, size, order, scratch, pool, FILL_PARALLEL_MIN_TERMS)
}

/// [`prepare_export`] with the fill's leaf size in scanned input terms.
#[allow(clippy::too_many_arguments)]
fn prepare_export_with<'a, const W: usize, C: Coefficient>(
    local: &PauliSum<W, C>,
    prepared: &'a Prepared<W, C>,
    plan: &PartitionPlan,
    size: u32,
    order: &CosetOrder,
    scratch: &'a mut ExportScratch,
    pool: &mut Vec<PartnerPayload<W, C>>,
    min_terms: usize,
) -> (Vec<Option<PartnerPayload<W, C>>>, LayerExchangeCounts, ExportFill<'a, W, C>) {
    let num_remote = plan.remote.len();
    let num_buckets = local.num_buckets();
    let mut send: Vec<Option<PartnerPayload<W, C>>> = (0..size).map(|_| None).collect();
    let mut counts = LayerExchangeCounts::none(size);
    let mut blocks_of: Vec<Vec<usize>> = vec![Vec::new(); size as usize];

    let emitters: Vec<RowEmitter<'a, W, C>> = plan
        .remote
        .iter()
        .map(|remote| match prepared {
            Prepared::Local(ptm) => {
                RowEmitter::Tabulated { ptm, entry: &ptm.deltas()[remote.entry] }
            }
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
    if num_remote > 0 {
        scratch
            .counts
            .par_chunks_mut(num_remote)
            .enumerate()
            .for_each(|(bucket, slot)| count_bucket(local, prepared, plan, &dense, bucket, slot));
    }

    // One block per remote delta, sized in pooled storage that is neither allocated nor zeroed here.
    debug_assert_eq!(
        order.positions(),
        num_buckets,
        "the coset order is built for a different bucket count than the sum has",
    );
    scratch.block_counts.clear();
    scratch.block_counts.resize(num_buckets, 0);
    scratch.source_of.clear();
    scratch.source_of.resize(num_remote * num_buckets, 0);
    scratch.offsets.clear();
    scratch.offsets.resize(num_remote * (num_buckets + 1), 0);
    scratch.work_prefix.clear();
    scratch.work_prefix.resize(num_remote * (num_buckets + 1), 0);
    for (i, remote) in plan.remote.iter().enumerate() {
        let source_of = &mut scratch.source_of[i * num_buckets..(i + 1) * num_buckets];
        let work_prefix =
            &mut scratch.work_prefix[i * (num_buckets + 1)..(i + 1) * (num_buckets + 1)];
        let mut work = 0usize;
        for position in 0..num_buckets {
            let source = order.bucket_at(position as u32) ^ remote.bucket_delta;
            let count = scratch.counts[source as usize * num_remote + i];
            source_of[position] = source;
            scratch.block_counts[position] = count;
            if count != 0 {
                work += local.bucket_len(source as usize);
            }
            work_prefix[position + 1] = work;
        }
        let partner = remote.partner as usize;
        let payload = send[partner].get_or_insert_with(|| pool.pop().unwrap_or_default());
        let block_index = plan.sent_block_index(i);
        debug_assert_eq!(
            block_index,
            blocks_of[partner].len(),
            "a partner's blocks follow the plan's remote order",
        );
        blocks_of[partner].push(i);
        if payload.blocks.len() <= block_index {
            payload.blocks.resize_with(block_index + 1, ExchangeBlock::<W, C>::default);
        }
        let block = &mut payload.blocks[block_index];
        block.set_counts(remote.entry as u32, &scratch.block_counts);
        scratch.offsets[i * (num_buckets + 1)..(i + 1) * (num_buckets + 1)]
            .copy_from_slice(&block.offsets);
        counts.rows_sent[partner] += block.rows() as u64;
        counts.bytes_sent[partner] += block.bytes() as u64;
    }
    // A pooled payload may carry more blocks than this layer fills; the extras must not travel.
    for (partner, payload) in send.iter_mut().enumerate() {
        if let Some(payload) = payload {
            payload.blocks.truncate(blocks_of[partner].len());
        }
    }

    let scratch: &'a ExportScratch = scratch;
    let fill = ExportFill {
        emitters,
        source_of: &scratch.source_of,
        offsets: &scratch.offsets,
        work_prefix: &scratch.work_prefix,
        blocks_of,
        num_buckets,
        min_terms,
    };
    (send, counts, fill)
}

/// Writes the rows of a layer's outgoing blocks, one chunk of destination positions at a time, from the counts [`prepare_export`] sized them by.
pub(crate) struct ExportFill<'a, const W: usize, C: Coefficient> {
    emitters: Vec<RowEmitter<'a, W, C>>,
    source_of: &'a [u32],
    offsets: &'a [u32],
    work_prefix: &'a [usize],
    /// Remote-delta indices in block order, by partner.
    blocks_of: Vec<Vec<usize>>,
    num_buckets: usize,
    min_terms: usize,
}

impl<const W: usize, C: Coefficient> ExportFill<'_, W, C> {
    /// Write chunk `k` of `map` from `local`, the sum [`prepare_export`] counted, into `views`, partner `q`'s being the x, z and coeff chunk views of its blocks in order, as [`chunk_views`](crate::engine::transport::chunk_views) cuts them.
    pub(crate) fn chunk(
        &self,
        local: &PauliSum<W, C>,
        k: usize,
        map: &crate::engine::transport::ChunkMap,
        views: &mut [Option<Vec<&mut [u8]>>],
    ) {
        let (start, end) = (map.bound(k) as usize, map.bound(k + 1) as usize);
        let positions = self.num_buckets + 1;
        for (partner, blocks) in self.blocks_of.iter().enumerate() {
            if blocks.is_empty() {
                continue;
            }
            let parts = views[partner].as_mut().expect("chunk views for every partner sent to");
            debug_assert_eq!(parts.len(), 3 * blocks.len(), "x, z and coeff per block");
            for (block, &i) in blocks.iter().enumerate() {
                let [x, z, coeff] = &mut parts[3 * block..3 * block + 3] else {
                    unreachable!("three parts per block")
                };
                let (x, x_rest) = bytemuck::cast_slice_mut::<u8, u64>(x).as_chunks_mut::<W>();
                let (z, z_rest) = bytemuck::cast_slice_mut::<u8, u64>(z).as_chunks_mut::<W>();
                debug_assert!(x_rest.is_empty() && z_rest.is_empty(), "whole keys per chunk");
                let coeff = bytemuck::cast_slice_mut::<u8, C>(coeff);
                let fill = Fill {
                    local,
                    emitter: &self.emitters[i],
                    offsets: &self.offsets[i * positions..(i + 1) * positions],
                    source_of: &self.source_of[i * self.num_buckets..(i + 1) * self.num_buckets],
                    work_prefix: &self.work_prefix[i * positions..(i + 1) * positions],
                    min_terms: self.min_terms,
                };
                fill.range(start, end, ColumnsMut { x, z, coeff });
            }
        }
    }
}

/// Count one source bucket's exported rows, one slot per remote delta.
fn count_bucket<const W: usize, C: Coefficient>(
    local: &PauliSum<W, C>,
    prepared: &Prepared<W, C>,
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
                    if !dense[i] && ptm.deltas()[remote.entry].amplitude[support] != C::ZERO {
                        out[i] += 1;
                    }
                }
            }
        }
        Prepared::Rotation(rotation) => {
            debug_assert_eq!(out.len(), 1, "a rotation has at most one remote delta");
            let mut anticommuting = 0u32;
            for term in 0..len {
                let string = PauliString::<W> { x: bucket_x[term], z: bucket_z[term] };
                if !string.commutes_with(&rotation.generator) {
                    anticommuting += 1;
                }
            }
            out[0] = anticommuting;
        }
    }
}

/// One block's pass-2 inputs, shared by every task of its fill.
struct Fill<'a, 'p, const W: usize, C: Coefficient> {
    local: &'a PauliSum<W, C>,
    emitter: &'a RowEmitter<'p, W, C>,
    offsets: &'a [u32],
    source_of: &'a [u32],
    work_prefix: &'a [usize],
    /// Scanned input terms below which a range stays on one thread.
    min_terms: usize,
}

impl<const W: usize, C: Coefficient> Fill<'_, '_, W, C> {
    /// Fill destination positions `start..end`; the halves of a split are disjoint, so the contents are independent of the thread count.
    ///
    /// The split balances scanned input terms rather than emitted rows, since bucket bits can decide the emit test for whole buckets and leave most of the scan where few rows land.
    fn range(&self, start: usize, end: usize, columns: ColumnsMut<'_, W, C>) {
        let offsets = self.offsets;
        debug_assert_eq!(columns.len(), (offsets[end] - offsets[start]) as usize);
        if end - start > 1 && self.work_prefix[end] - self.work_prefix[start] > self.min_terms {
            let middle = split_at_half_work(self.work_prefix, start, end);
            let (head, tail) = columns.split_at((offsets[middle] - offsets[start]) as usize);
            rayon::join(|| self.range(start, middle, head), || self.range(middle, end, tail));
            return;
        }
        let ColumnsMut { x, z, coeff } = columns;
        let mut written = 0usize;
        for position in start..end {
            if offsets[position] == offsets[position + 1] {
                continue;
            }
            let (bucket_x, bucket_z, bucket_coeff) =
                self.local.bucket(self.source_of[position] as usize);
            for term in 0..bucket_coeff.len() {
                if let Some((out_x, out_z, out_coeff)) =
                    self.emitter.emit(&bucket_x[term], &bucket_z[term], bucket_coeff[term])
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
}

/// The position in `start + 1..end` where the scan work of `start..end`, read from its prefix sums, first reaches half.
fn split_at_half_work(work_prefix: &[usize], start: usize, end: usize) -> usize {
    let target = work_prefix[start] + (work_prefix[end] - work_prefix[start]) / 2;
    let middle = start + 1 + work_prefix[start + 1..end].partition_point(|&work| work < target);
    middle.min(end - 1)
}

/// Every row of a written chunk belongs to the partner it is addressed to; `views` as [`ExportFill::chunk`] takes them.
#[cfg(debug_assertions)]
pub(super) fn debug_assert_chunk_partitions<const W: usize>(
    views: &[Option<Vec<&mut [u8]>>],
    rows: &crate::pauli_sum::hash::PartitionRows<W>,
) {
    for (partner, parts) in views.iter().enumerate() {
        let Some(parts) = parts else { continue };
        for block in parts.chunks_exact(3) {
            let x = bytemuck::cast_slice::<u8, u64>(block[0]).as_chunks::<W>().0;
            let z = bytemuck::cast_slice::<u8, u64>(block[1]).as_chunks::<W>().0;
            for (row, (x, z)) in x.iter().zip(z).enumerate() {
                let owner = rows.partition_of(x, z);
                debug_assert_eq!(
                    owner, partner as u32,
                    "export: chunk row {row} is addressed to partition {partner} but belongs to \
                     partition {owner}",
                );
            }
        }
    }
}

#[cfg(test)]
mod tests;
