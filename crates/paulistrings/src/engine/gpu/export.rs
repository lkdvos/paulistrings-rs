//! K10 export, the exchange over a device wire, and the chunked receive of one device layer. See ARCHITECTURE.md §Partitioning.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};

use super::columns::grow;
use super::error::GpuError;
use super::layer::{timed_transfer, LayerScratch, TableBuffers, Transfer};
use super::module::{thread_per, warp_per_bucket, MAX_BUCKET_LEN};
use super::payload::DevicePayload;
use super::prepared::DevicePrepared;
use super::scan::exclusive_scan;
use super::sum::GpuSum;
use super::wire::{schedule, BlockSkeletons, DeviceWire, ScheduledOp, Skeleton};
use crate::engine::partitioned::layer::LayerExchangeCounts;
use crate::engine::partitioned::plan::PartitionPlan;
use crate::engine::partitioned::transport::{ChunkMap, Transport};

mod premerge;
mod receive;

use premerge::{premerge_groups, premerge_partner, PremergeScratch};
use receive::{discard_received, position_chunks, recv_cap_rows, stage_receive, wire_ready};
pub(crate) use receive::{finish_receive, receive_chunk};

/// Grow-only device and host buffers of the export and receive passes, kept between layers.
pub(crate) struct DeviceExport<const W: usize> {
    counts: CudaSlice<u32>,
    offsets: CudaSlice<u32>,
    /// Received blocks, concatenated: `received_offsets` is `K × (B + 1)` CSR offsets, `received_base[k]` block `k`'s row base.
    pub(crate) received_offsets: CudaSlice<u32>,
    pub(crate) received_base: CudaSlice<u32>,
    pub(crate) received_x: CudaSlice<u64>,
    pub(crate) received_z: CudaSlice<u64>,
    pub(crate) received_coefficient: CudaSlice<f64>,
    pub(crate) received_fingerprint: CudaSlice<u64>,
    offsets_host: Vec<u32>,
    base_host: Vec<u32>,
    /// The longest received segment of the current layer; must fit the tag's offset field.
    pub(crate) received_max_segment: usize,
    pub(crate) received_rows: usize,
    premerge: PremergeScratch,
    stream: Arc<CudaStream>,
    /// The group's device wire, `Some` in every group of more than one partition.
    pub(crate) wire: Option<Arc<dyn DeviceWire>>,
    /// Send payloads not in flight.
    sends: Vec<DevicePayload<W>>,
    /// Skeleton payloads not in flight.
    skeletons: Vec<BlockSkeletons<W>>,
    /// Test hook: the next remote layer's receive growth fails as out of memory.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fail_recv_growth: bool,
    /// The current layer's received rows still to move into `received_*`.
    pub(crate) pending: Option<PendingRecv<W>>,
    /// Test hook: the next chunked receive fails as out of memory after moving this chunk.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fail_after_chunk: Option<usize>,
}

/// A remote layer's transfers not yet made, chunk `c` (positions `map.bound(c)..map.bound(c + 1)`) moving just before the fused layer reaches it (ARCHITECTURE.md §Partitioning).
pub(crate) struct PendingRecv<const W: usize> {
    pub(crate) map: ChunkMap,
    /// The first chunk not yet moved.
    next: usize,
    send: Vec<Option<DevicePayload<W>>>,
    /// This partition's block for remote delta `k` is `send[own[k].0].blocks[own[k].1]`.
    own: Vec<(usize, usize)>,
    ops: Vec<ScheduledOp>,
    /// Set once a group failed to post or complete, after which none posts.
    failed: bool,
}

impl<const W: usize> DeviceExport<W> {
    pub(crate) fn new(sum: &GpuSum<W>) -> Result<Self, GpuError> {
        let stream = &sum.stream;
        Ok(Self {
            counts: stream.alloc_zeros(1)?,
            offsets: stream.alloc_zeros(2)?,
            received_offsets: stream.alloc_zeros(2)?,
            received_base: stream.alloc_zeros(16)?,
            received_x: stream.alloc_zeros(W)?,
            received_z: stream.alloc_zeros(W)?,
            received_coefficient: stream.alloc_zeros(2)?,
            received_fingerprint: stream.alloc_zeros(1)?,
            offsets_host: Vec::new(),
            base_host: Vec::new(),
            received_max_segment: 0,
            received_rows: 0,
            premerge: PremergeScratch::new(stream, W)?,
            stream: stream.clone(),
            wire: None,
            sends: Vec::new(),
            skeletons: Vec::new(),
            #[cfg(any(test, feature = "test-utils"))]
            fail_recv_growth: false,
            pending: None,
            #[cfg(any(test, feature = "test-utils"))]
            fail_after_chunk: None,
        })
    }

    /// A pooled skeleton payload, emptied.
    fn skeleton(&mut self) -> BlockSkeletons<W> {
        let mut skeleton = self.skeletons.pop().unwrap_or_default();
        skeleton.blocks.clear();
        skeleton
    }
}

/// The exchange a partition owes its partners when its own layer failed before exporting: one empty skeleton per remote delta the plan names and a no vote, so every receiver finishes the layer and the error surfaces after the loop.
pub(crate) fn pair_empty_exchange<const W: usize, X: Transport>(
    transport: &X,
    plan: &PartitionPlan,
    bits: u8,
    export: &mut DeviceExport<W>,
) {
    let mut send: Vec<Option<BlockSkeletons<W>>> = (0..transport.size()).map(|_| None).collect();
    for r in &plan.remote {
        let q = r.partner as usize;
        if send[q].is_none() {
            send[q] = Some(export.skeleton());
        }
        send[q]
            .as_mut()
            .expect("just set")
            .push_empty(r.entry as u32, 1 << bits);
    }
    let recv = transport.exchange(send, &mut export.skeletons);
    export.skeletons.extend(recv.into_iter().flatten());
    vote(transport, None);
}

/// **Collective** go/no-go and chunk count: `ready` is this rank's `Some(log2 chunks)`, the result the largest such count when every rank is ready, which fits every rank since a finer power-of-two cut never grows a chunk.
fn vote<X: Transport>(transport: &X, ready: Option<u8>) -> Option<u8> {
    let (rank, size) = (transport.rank() as usize, transport.size() as usize);
    let mut votes = vec![0u64; 2 * size];
    votes[rank] = u64::from(ready.is_none());
    votes[size + rank] = u64::from(ready.unwrap_or(0));
    transport.allreduce_sum_u64(&mut votes);
    let chunks = votes[size..].iter().copied().max().unwrap_or(0) as u8;
    votes[..size].iter().all(|&v| v == 0).then_some(chunks)
}

/// One remote layer's exchange after K1 (ARCHITECTURE.md §Partitioning): K10's blocks, their skeletons and the vote, leaving on a yes the pending receive [`receive_chunk`] moves; on any no nobody posts and a rank that could not receive returns its error.
pub(crate) fn exchange_rows<const W: usize, X: Transport>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    plan: &PartitionPlan,
    scratch: &mut LayerScratch<W>,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    let size = transport.size();
    #[cfg(feature = "phase-timing")]
    let t_export = std::time::Instant::now();
    let (send, mut counts) = match export_blocks_device(sum, table, plan, scratch, size) {
        Ok(v) => v,
        Err(e) => {
            pair_empty_exchange::<W, X>(transport, plan, sum.hash.bits(), &mut scratch.export);
            return Err(e);
        }
    };
    #[cfg(feature = "phase-timing")]
    {
        scratch.laps.export_ns += t_export.elapsed().as_nanos() as u64;
        scratch.laps.rows_exported += counts.rows_sent.iter().sum::<u64>();
    }
    let b = sum.hash.num_buckets();
    let cap_rows = recv_cap_rows::<W>(scratch.options.exchange_bytes);
    let export = &mut scratch.export;
    let transfer_ns = &mut scratch.transfer_ns;
    #[cfg(feature = "phase-timing")]
    let t_exchange = std::time::Instant::now();
    let skeletons: Vec<Option<BlockSkeletons<W>>> = send
        .iter()
        .map(|slot| {
            slot.as_ref().map(|payload| {
                let mut skeleton = export.skeleton();
                skeleton.fill_from(&payload.blocks);
                skeleton
            })
        })
        .collect();
    let recv = transport.exchange(skeletons, &mut export.skeletons);
    let (rows, ops) = {
        let blocks = paired_blocks(plan, &recv);
        let ready = stage_receive(sum, export, &blocks, cap_rows, transfer_ns)
            .and_then(|log2| wire_ready(export).map(|()| log2));
        match vote(transport, ready.as_ref().ok().copied()) {
            Some(log2) => {
                let map = position_chunks(sum.hash.bits(), b, log2);
                let own = own_at(plan);
                let partners: Vec<u32> = plan.remote.iter().map(|r| r.partner).collect();
                let own_off: Vec<&[u32]> = own
                    .iter()
                    .map(|&(q, j)| {
                        send[q]
                            .as_ref()
                            .expect("a payload for every partner")
                            .blocks[j]
                            .offsets
                            .as_slice()
                    })
                    .collect();
                let received_offsets: Vec<&[u32]> =
                    blocks.iter().map(|b| b.offsets.as_slice()).collect();
                let ops = schedule(&partners, &own_off, &received_offsets, &map);
                (Ok(export.received_rows as u64), Some((map, own, ops)))
            }
            None => (ready.and_then(|_| discard_received(export)), None),
        }
    };
    export.skeletons.extend(recv.into_iter().flatten());
    match ops {
        Some((map, own, ops)) => {
            export.pending = Some(PendingRecv {
                map,
                next: 0,
                send,
                own,
                ops,
                failed: false,
            });
        }
        None => export.sends.extend(send.into_iter().flatten()),
    }
    #[cfg(feature = "phase-timing")]
    {
        scratch.laps.exchange_ns += t_exchange.elapsed().as_nanos() as u64;
    }
    let rows_received = rows?;
    #[cfg(feature = "phase-timing")]
    {
        scratch.laps.recv_rows += rows_received;
    }
    counts.rows_received = rows_received;
    counts.remote_deltas = plan.remote.len();
    Ok(counts)
}

/// Remote delta `k`'s `(partner, index)` among the partner's received blocks, in plan order: the `j`-th of `remote_for_partner(q)` is the `j`-th block of `recv[q]`, as `RecvRows::new`.
fn paired_at(plan: &PartitionPlan) -> Vec<(usize, usize)> {
    plan.remote
        .iter()
        .map(|r| {
            let j = plan
                .remote_for_partner(r.partner)
                .position(|other| other.entry == r.entry)
                .expect("a remote delta is in its own partner's list");
            (r.partner as usize, j)
        })
        .collect()
}

/// Remote delta `k`'s `(partner, index)` among this rank's send blocks, which the export fills in plan order per partner.
fn own_at(plan: &PartitionPlan) -> Vec<(usize, usize)> {
    let mut used = std::collections::HashMap::<u32, usize>::new();
    plan.remote
        .iter()
        .map(|r| {
            let j = used.entry(r.partner).or_default();
            *j += 1;
            (r.partner as usize, *j - 1)
        })
        .collect()
}

/// The received skeleton per remote delta in plan order, by [`paired_at`].
fn paired_blocks<'a, const W: usize>(
    plan: &PartitionPlan,
    recv: &'a [Option<BlockSkeletons<W>>],
) -> Vec<&'a Skeleton> {
    paired_at(plan)
        .into_iter()
        .map(|(q, j)| {
            recv[q]
                .as_ref()
                .and_then(|payload| payload.blocks.get(j))
                .unwrap_or_else(|| panic!("partition {q} sent no block {j} for a remote delta"))
        })
        .collect()
}

/// K10's count and scan for entry `e` at bucket delta `bd`, leaving the CSR offsets in `scratch.export.offsets` and their copy in `offsets`; returns the block's rows.
fn export_offsets<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    scratch: &mut LayerScratch<W>,
    bd: u32,
    e: u32,
    offsets: &mut Vec<u32>,
) -> Result<usize, GpuError> {
    let stream = &sum.stream;
    let kernels = &sum.kernels;
    let b = sum.hash.num_buckets();
    let (e32, b32) = (table.entries as u32, b as u32);
    // SAFETY: arguments match `k_export_counts` in export.cu; `counts` holds `b` entries.
    unsafe {
        stream
            .launch_builder(&kernels.export_counts)
            .arg(&scratch.counts)
            .arg(&scratch.bucket_at)
            .arg(&bd)
            .arg(&e)
            .arg(&e32)
            .arg(&b32)
            .arg(&mut scratch.export.counts)
            .launch(thread_per(b, 1024))?;
    }
    exclusive_scan(
        stream,
        kernels,
        &scratch.export.counts.slice(0..b),
        &mut scratch.export.offsets.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut scratch.totals,
    )?;
    offsets.clear();
    offsets.resize(b + 1, 0);
    let device_offsets = &scratch.export.offsets;
    timed_transfer(stream, &mut scratch.transfer_ns, Transfer::D2h, || {
        stream.memcpy_dtoh(&device_offsets.slice(0..b + 1), &mut offsets[..])?;
        stream.synchronize()?;
        Ok(())
    })?;
    Ok(offsets[b] as usize)
}

/// The scratch buffers K10's fill reads, borrowed apart from the columns it writes.
struct FillContext<'a> {
    bucket_at: &'a CudaSlice<u32>,
    table: &'a TableBuffers,
    offsets: &'a CudaSlice<u32>,
}

/// K10's fill of entry `e` at bucket delta `bd` into `(x, z, c)`, which hold the rows `context.offsets` names.
#[allow(clippy::too_many_arguments)]
fn export_fill<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    context: FillContext<'_>,
    bd: u32,
    e: u32,
    x: &mut CudaSlice<u64>,
    z: &mut CudaSlice<u64>,
    c: &mut CudaSlice<f64>,
) -> Result<(), GpuError> {
    let stream = &sum.stream;
    let b = sum.hash.num_buckets();
    let (e32, b32) = (table.entries as u32, b as u32);
    // SAFETY: arguments match `k_export_fill` in export.cu; the columns hold the block's rows and `context.offsets` its CSR.
    unsafe {
        stream
            .launch_builder(&sum.kernels.export_fill)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lens)
            .arg(context.bucket_at)
            .arg(&table.mode)
            .arg(&e32)
            .arg(&table.kq)
            .arg(&table.q0)
            .arg(&table.q1)
            .arg(&table.rot_cos)
            .arg(&table.rot_sin)
            .arg(&context.table.amp)
            .arg(&context.table.mask)
            .arg(&context.table.nz)
            .arg(&bd)
            .arg(&e)
            .arg(&b32)
            .arg(context.offsets)
            .arg(x)
            .arg(z)
            .arg(c)
            .launch(warp_per_bucket(b))?;
    }
    Ok(())
}

/// K10 for every remote delta into device blocks: one block per delta into pooled [`DevicePayload`]s, in ascending remote-delta index per partner.
fn export_blocks_device<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    plan: &PartitionPlan,
    scratch: &mut LayerScratch<W>,
    size: u32,
) -> Result<(Vec<Option<DevicePayload<W>>>, LayerExchangeCounts), GpuError> {
    let mut send: Vec<Option<DevicePayload<W>>> = (0..size).map(|_| None).collect();
    let mut counts = LayerExchangeCounts::none(size);
    let b = sum.hash.num_buckets();
    let stream: Arc<CudaStream> = sum.stream.clone();
    let ordinal = sum.device();
    grow(&stream, &mut scratch.export.counts, b, ordinal)?;
    grow(&stream, &mut scratch.export.offsets, b + 1, ordinal)?;
    for r in &plan.remote {
        let q = r.partner as usize;
        if send[q].is_none() {
            send[q] = Some(scratch.export.sends.pop().unwrap_or_default());
        }
    }
    let groups = premerge_groups(sum, table, plan, scratch, size)?;
    let mut merged = vec![false; size as usize];
    for (q, entries) in groups.iter().enumerate() {
        if entries.is_empty() {
            continue;
        }
        let payload = send[q].as_mut().expect("a payload for every partner");
        payload.block_mut(entries.len() - 1, &stream)?;
        let blocks = &mut payload.blocks[..entries.len()];
        if let Some(rows) = premerge_partner(sum, table, entries, scratch, blocks)? {
            merged[q] = true;
            counts.rows_sent[q] += rows;
            counts.bytes_sent[q] += blocks.iter().map(|b| b.bytes() as u64).sum::<u64>();
        }
    }
    let mut blocks_used = vec![0usize; size as usize];
    for r in &plan.remote {
        let q = r.partner as usize;
        let payload = send[q].as_mut().expect("a payload for every partner");
        let j = blocks_used[q];
        blocks_used[q] += 1;
        if merged[q] {
            continue;
        }
        let block = payload.block_mut(j, &stream)?;
        let (bd, e) = (r.bucket_delta, r.entry as u32);
        let t0 = scratch.event(sum)?;
        let rows = export_offsets(sum, table, scratch, bd, e, &mut block.offsets)?;
        block.set_header(e, b);
        if rows > 0 {
            block.grow(&stream, rows, ordinal)?;
            let context = FillContext {
                bucket_at: &scratch.bucket_at,
                table: &scratch.table,
                offsets: &scratch.export.offsets,
            };
            export_fill(
                sum,
                table,
                context,
                bd,
                e,
                &mut block.x,
                &mut block.z,
                &mut block.coefficient,
            )?;
        }
        scratch.lap(sum, t0, |m| &mut m.export)?;
        counts.rows_sent[q] += rows as u64;
        counts.bytes_sent[q] += block.bytes() as u64;
    }
    for (q, payload) in send.iter_mut().enumerate() {
        if let Some(payload) = payload {
            payload.blocks.truncate(blocks_used[q]);
        }
    }
    #[cfg(feature = "phase-timing")]
    stream.synchronize()?;
    Ok((send, counts))
}

/// Device bytes one received row holds in `received_*`: both key columns, the coefficient and the fingerprint.
pub(crate) const fn recv_row_bytes<const W: usize>() -> usize {
    (2 * W + 3) * std::mem::size_of::<u64>()
}

/// Rows the tag's offset field can address in one received segment.
pub(crate) const MAX_RECV_SEGMENT: usize = MAX_BUCKET_LEN;

#[cfg(test)]
mod tests;
