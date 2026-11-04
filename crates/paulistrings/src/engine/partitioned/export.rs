//! The export pass: a layer's rows that belong to another partition, gathered into one [`ExchangeBlock`] per remote delta.
//!
//! A row of remote delta `d` from local bucket `β` belongs to partition `rank ⊕ part(d)`, bucket `β ⊕ h(d)`, so the sender fills segment `p` from its bucket `bucket_at(p) ⊕ h(d)` and the block arrives in the receiver's coset order.
//! A partner gets one block per remote delta to it, in ascending plan order and empty ones included, since the receiver walks them in lockstep with its own plan.
//! A pending `ApproxTopN` cut is not applied here: both passes skip the terms it drops, and the coset loop applies it as it swaps each bucket in.
//! Pass 1 reads the counts the last coset loop took ahead where they answer (`LayerOctaves::rows_ahead`), and the bucket otherwise.

use crate::coefficient::Coefficient;
use rayon::prelude::*;

use super::exchange_block::{ExchangeBlock, PartnerPayload};
use super::layer::LayerExchangeCounts;
use crate::channel::prepared::{DeltaEntry, LocalPtm, Prepared, PreparedRotation};
use crate::engine::bucketed::LayerOctaves;
use crate::engine::coset::CosetOrder;
use crate::engine::plan::PartitionPlan;
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::{ColumnsMut, PauliSum};
use crate::truncation::octave_window::BoundaryCut;

/// Scanned input terms below which a block's fill stays on one thread.
const FILL_PARALLEL_MIN_TERMS: usize = 16384;

/// Reusable scratch for [`export_layer`], so a steady-state layer allocates nothing.
#[derive(Debug, Default)]
pub(crate) struct ExportScratch<const W: usize, C: Coefficient = num_complex::Complex64> {
    /// Pass-1 counts, bucket-major: `counts[β * num_remote + i]`.
    counts: Vec<u32>,
    /// One block's counts by destination position.
    block_counts: Vec<u32>,
    /// `source_of[position] = order.bucket_at(position) ^ bucket_delta`.
    source_of: Vec<u32>,
    /// One block's prefix sums of input terms its fill scans, by destination position; a position that exports nothing scans nothing.
    work_prefix: Vec<usize>,
    /// Payloads not in flight, columns intact; outgoing and incoming payloads are both drawn from here.
    pub(crate) pool: Vec<PartnerPayload<W, C>>,
}

/// One remote delta's row-level emitter: the tabulated entry, or the wide rotation's generator pass.
enum RowEmitter<'p, const W: usize, C: Coefficient = num_complex::Complex64> {
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

/// This partition's outgoing payloads for one layer, one slot per partner rank, from `local` as its pending cut in `octaves` leaves it.
///
/// A partner's blocks are one per remote delta destined for it in ascending order, empty ones included, since the receiver indexes blocks positionally.
pub(crate) fn export_layer<const W: usize, C: Coefficient>(
    local: &PauliSum<W, C>,
    octaves: &LayerOctaves,
    prepared: &Prepared<W, C>,
    plan: &PartitionPlan,
    size: u32,
    order: &CosetOrder,
    scratch: &mut ExportScratch<W, C>,
) -> (Vec<Option<PartnerPayload<W, C>>>, LayerExchangeCounts) {
    export_layer_with(local, octaves, prepared, plan, size, order, scratch, FILL_PARALLEL_MIN_TERMS)
}

/// [`export_layer`] with the fill's leaf size in scanned input terms.
#[allow(clippy::too_many_arguments)]
fn export_layer_with<const W: usize, C: Coefficient>(
    local: &PauliSum<W, C>,
    octaves: &LayerOctaves,
    prepared: &Prepared<W, C>,
    plan: &PartitionPlan,
    size: u32,
    order: &CosetOrder,
    scratch: &mut ExportScratch<W, C>,
    min_terms: usize,
) -> (Vec<Option<PartnerPayload<W, C>>>, LayerExchangeCounts) {
    let num_remote = plan.remote.len();
    let num_buckets = local.num_buckets();
    let cut = octaves.pending();
    let ahead = num_remote == 1 && octaves.has_ahead();
    let mut send: Vec<Option<PartnerPayload<W, C>>> = (0..size).map(|_| None).collect();
    let mut counts = LayerExchangeCounts::none(size);
    if num_remote == 0 {
        return (send, counts);
    }

    let emitters: Vec<RowEmitter<'_, W, C>> = plan
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
    let all_dense =
        matches!(prepared, Prepared::Local(_)) && dense.iter().all(|&is_dense| is_dense);

    // Pass 1: bucket-major counts, `counts[β * num_remote + i]`, so each bucket owns one disjoint chunk.
    scratch.counts.clear();
    scratch.counts.resize(num_buckets * num_remote, 0);
    scratch.counts.par_chunks_mut(num_remote).enumerate().for_each(|(bucket, slot)| {
        let counted = |slot: &mut [u32]| {
            let terms = local.bucket(bucket);
            match cut {
                None if all_dense => slot.fill(terms.2.len() as u32),
                None => count_bucket(terms, prepared, plan, &dense, |_, _, _| true, slot),
                Some(cut) => count_bucket(terms, prepared, plan, &dense, cut.filter(bucket), slot),
            }
        };
        match octaves.rows_ahead(bucket).filter(|_| ahead) {
            Some(rows) => {
                slot[0] = rows;
                #[cfg(debug_assertions)]
                {
                    let mut recounted = [0u32];
                    counted(&mut recounted);
                    debug_assert_eq!(
                        rows, recounted[0],
                        "bucket {bucket}: the counts ahead disagree"
                    );
                }
            }
            None => counted(slot),
        }
    });

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
    scratch.work_prefix.clear();
    scratch.work_prefix.resize(num_buckets + 1, 0);
    let mut blocks_used = vec![0usize; size as usize];
    for (i, remote) in plan.remote.iter().enumerate() {
        let mut work = 0usize;
        for position in 0..num_buckets {
            let source = order.bucket_at(position as u32) ^ remote.bucket_delta;
            let count = scratch.counts[source as usize * num_remote + i];
            scratch.source_of[position] = source;
            scratch.block_counts[position] = count;
            if count != 0 {
                work += local.bucket_len(source as usize);
            }
            scratch.work_prefix[position + 1] = work;
        }
        let partner = remote.partner as usize;
        let payload = send[partner].get_or_insert_with(|| scratch.pool.pop().unwrap_or_default());
        let block_index = plan.sent_block_index(i);
        blocks_used[partner] = block_index + 1;
        if payload.blocks.len() <= block_index {
            payload.blocks.resize_with(block_index + 1, ExchangeBlock::<W, C>::default);
        }
        let block = &mut payload.blocks[block_index];
        block.set_counts(remote.entry as u32, &scratch.block_counts);
        let rows = block.rows();
        {
            let ExchangeBlock { ref offsets, ref mut x, ref mut z, ref mut coeff, .. } = *block;
            let columns =
                ColumnsMut { x: &mut x[..rows], z: &mut z[..rows], coeff: &mut coeff[..rows] };
            let fill = Fill {
                local,
                cut,
                emitter: &emitters[i],
                offsets,
                source_of: &scratch.source_of,
                work_prefix: &scratch.work_prefix,
                min_terms,
            };
            fill.range(0, num_buckets, columns);
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

/// Count one source bucket's exported rows among the terms `keep` keeps, one slot per remote delta.
fn count_bucket<const W: usize, C: Coefficient>(
    (bucket_x, bucket_z, bucket_coeff): (&[[u64; W]], &[[u64; W]], &[C]),
    prepared: &Prepared<W, C>,
    plan: &PartitionPlan,
    dense: &[bool],
    mut keep: impl FnMut(&[u64; W], &[u64; W], C) -> bool,
    out: &mut [u32],
) {
    let len = bucket_coeff.len();
    match prepared {
        Prepared::Local(ptm) => {
            let all_dense = dense.iter().all(|&is_dense| is_dense);
            out.fill(0);
            let mut kept = 0u32;
            for term in 0..len {
                let survives = keep(&bucket_x[term], &bucket_z[term], bucket_coeff[term]);
                kept += u32::from(survives);
                if all_dense {
                    continue;
                }
                let support = ptm.support_bits(&bucket_x[term], &bucket_z[term]);
                for (i, remote) in plan.remote.iter().enumerate() {
                    let emits = ptm.deltas()[remote.entry].amplitude[support] != C::ZERO;
                    out[i] += u32::from(survives & !dense[i] & emits);
                }
            }
            for (slot, &is_dense) in out.iter_mut().zip(dense) {
                if is_dense {
                    *slot = kept;
                }
            }
        }
        Prepared::Rotation(rotation) => {
            debug_assert_eq!(out.len(), 1, "a rotation has at most one remote delta");
            let mut anticommuting = 0u32;
            for term in 0..len {
                let survives = keep(&bucket_x[term], &bucket_z[term], bucket_coeff[term]);
                let string = PauliString::<W> { x: bucket_x[term], z: bucket_z[term] };
                anticommuting += u32::from(survives & !string.commutes_with(&rotation.generator));
            }
            out[0] = anticommuting;
        }
    }
}

/// One block's pass-2 inputs, shared by every task of its fill.
struct Fill<'a, 'p, const W: usize, C: Coefficient> {
    local: &'a PauliSum<W, C>,
    cut: Option<&'a BoundaryCut>,
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
            let source = self.source_of[position] as usize;
            let out = ColumnsMut {
                x: &mut x[written..],
                z: &mut z[written..],
                coeff: &mut coeff[written..],
            };
            written += match self.cut {
                None => self.fill_bucket(source, |_, _, _| true, out),
                Some(cut) => self.fill_bucket(source, cut.filter(source), out),
            };
        }
        debug_assert_eq!(
            written,
            x.len(),
            "export: pass 2 filled {written} rows where pass 1 counted {}",
            x.len(),
        );
    }
}

impl<const W: usize, C: Coefficient> Fill<'_, '_, W, C> {
    /// Write bucket `source`'s rows among the terms `keep` keeps to the head of `out`; returns how many.
    fn fill_bucket(
        &self,
        source: usize,
        mut keep: impl FnMut(&[u64; W], &[u64; W], C) -> bool,
        out: ColumnsMut<'_, W, C>,
    ) -> usize {
        let ColumnsMut { x, z, coeff } = out;
        let (bucket_x, bucket_z, bucket_coeff) = self.local.bucket(source);
        let mut written = 0usize;
        for term in 0..bucket_coeff.len() {
            let survives = keep(&bucket_x[term], &bucket_z[term], bucket_coeff[term]);
            if let (true, Some((out_x, out_z, out_coeff))) =
                (survives, self.emitter.emit(&bucket_x[term], &bucket_z[term], bucket_coeff[term]))
            {
                x[written] = out_x;
                z[written] = out_z;
                coeff[written] = out_coeff;
                written += 1;
            }
        }
        written
    }
}

/// The position in `start + 1..end` where the scan work of `start..end`, read from its prefix sums, first reaches half.
fn split_at_half_work(work_prefix: &[usize], start: usize, end: usize) -> usize {
    let target = work_prefix[start] + (work_prefix[end] - work_prefix[start]) / 2;
    let middle = start + 1 + work_prefix[start + 1..end].partition_point(|&work| work < target);
    middle.min(end - 1)
}

/// Every exported row belongs to the partner it is addressed to.
#[cfg(debug_assertions)]
pub(super) fn debug_assert_exported_partitions<const W: usize, C: Coefficient>(
    send: &[Option<PartnerPayload<W, C>>],
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
