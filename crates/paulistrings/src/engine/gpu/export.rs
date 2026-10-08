//! K10 export, the exchange over a device wire, and the chunked receive of one device layer. See ARCHITECTURE.md §Partitioning.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};

use super::columns::{grow, DeviceColumns};
use super::error::GpuError;
use super::layer::{
    arena_batches, fused_variant, launch_fused, xfer, FusedOut, FusedRecv, FusedTable,
    LayerScratch, TableBufs, Xfer, XferNs,
};
use super::module::{thread_per, warp_per_bucket, MAX_BUCKET_LEN};
use super::payload::{DeviceBlock, DevicePayload};
use super::prepared::DevicePrepared;
use super::scan::exclusive_scan;
use super::sum::{launch_fingerprint, GpuSum};
use super::truncation::KeepProgram;
use super::wire::{
    schedule, BlockSkeletons, DeviceWire, ScheduledOp, Skeleton, WireColumn, WireGroup, WireOpKind,
};
use crate::engine::coset::Gf2Span;
use crate::engine::partitioned::layer::LayerExchangeCounts;
use crate::engine::partitioned::plan::PartitionPlan;
use crate::engine::partitioned::transport::{ChunkMap, Transport};

/// Grow-only device and host buffers of the export and receive passes, kept between layers.
pub(crate) struct DeviceExport<const W: usize> {
    counts: CudaSlice<u32>,
    off: CudaSlice<u32>,
    /// Received blocks, concatenated: `off` is `K × (B + 1)` CSR offsets, `base[k]` the first row of block `k`.
    pub(crate) recv_off: CudaSlice<u32>,
    pub(crate) recv_base: CudaSlice<u32>,
    pub(crate) recv_x: CudaSlice<u64>,
    pub(crate) recv_z: CudaSlice<u64>,
    pub(crate) recv_c: CudaSlice<f64>,
    pub(crate) recv_g: CudaSlice<u64>,
    off_host: Vec<u32>,
    base_host: Vec<u32>,
    /// The longest received segment of the current layer; must fit the tag's offset field.
    pub(crate) recv_max_segment: usize,
    pub(crate) recv_rows: usize,
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
    /// The current layer's received rows still to move into `recv_*`.
    pub(crate) pending: Option<PendingRecv<W>>,
    /// Test hook: the next chunked receive fails as out of memory after moving this chunk.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fail_after_chunk: Option<usize>,
}

/// A remote layer's transfers not yet made: chunk `c`, positions `map.bound(c)..map.bound(c + 1)`, moves just before the fused layer reaches it, so `recv_*` holds one chunk at a time (ARCHITECTURE.md §Partitioning).
/// `send` holds this partition's blocks, its block for remote delta `k` at `own[k]`; `failed` once a group failed to post or complete, after which none posts.
pub(crate) struct PendingRecv<const W: usize> {
    pub(crate) map: ChunkMap,
    /// The first chunk not yet moved.
    next: usize,
    send: Vec<Option<DevicePayload<W>>>,
    own: Vec<(usize, usize)>,
    ops: Vec<ScheduledOp>,
    failed: bool,
}

/// Grow-only buffers of the sender-side merge: one partner's sub-table, its counts and CSR, and the split of its merged rows over the partner's blocks.
struct PremergeScratch {
    table: TableBufs,
    sel: CudaSlice<u32>,
    cnt: CudaSlice<u32>,
    rows: CudaSlice<u32>,
    start: CudaSlice<u32>,
    out_len_pos: CudaSlice<u32>,
    lens: CudaSlice<u32>,
    loff: CudaSlice<u32>,
    fallback: CudaSlice<u32>,
    tot: CudaSlice<u32>,
    start_host: Vec<u32>,
    loff_host: Vec<u32>,
}

impl PremergeScratch {
    fn new(s: &Arc<CudaStream>, w: usize) -> Result<Self, GpuError> {
        Ok(Self {
            table: TableBufs::new(s, w)?,
            sel: s.alloc_zeros(16)?,
            cnt: s.alloc_zeros(1)?,
            rows: s.alloc_zeros(1)?,
            start: s.alloc_zeros(2)?,
            out_len_pos: s.alloc_zeros(1)?,
            lens: s.alloc_zeros(1)?,
            loff: s.alloc_zeros(2)?,
            fallback: s.alloc_zeros(2)?,
            tot: s.alloc_zeros(2)?,
            start_host: Vec::new(),
            loff_host: Vec::new(),
        })
    }
}

impl<const W: usize> DeviceExport<W> {
    pub(crate) fn new(sum: &GpuSum<W>) -> Result<Self, GpuError> {
        let s = &sum.stream;
        Ok(Self {
            counts: s.alloc_zeros(1)?,
            off: s.alloc_zeros(2)?,
            recv_off: s.alloc_zeros(2)?,
            recv_base: s.alloc_zeros(16)?,
            recv_x: s.alloc_zeros(W)?,
            recv_z: s.alloc_zeros(W)?,
            recv_c: s.alloc_zeros(2)?,
            recv_g: s.alloc_zeros(1)?,
            off_host: Vec::new(),
            base_host: Vec::new(),
            recv_max_segment: 0,
            recv_rows: 0,
            premerge: PremergeScratch::new(s, W)?,
            stream: s.clone(),
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
        let mut s = self.skeletons.pop().unwrap_or_default();
        s.blocks.clear();
        s
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

/// The exchange's go/no-go and chunk count: `ready` is `Some(log2 chunks)` when this rank can receive in that many, and the result the largest such count when every rank of the group can, which fits every rank's buffers since a finer power-of-two cut never grows a chunk.
/// **Collective**: one `allreduce_sum_u64` of `2 × size` words, rank `r`'s slot `r` set when it is not ready and slot `size + r` its count.
fn vote<X: Transport>(transport: &X, ready: Option<u8>) -> Option<u8> {
    let (rank, size) = (transport.rank() as usize, transport.size() as usize);
    let mut votes = vec![0u64; 2 * size];
    votes[rank] = u64::from(ready.is_none());
    votes[size + rank] = u64::from(ready.unwrap_or(0));
    transport.allreduce_sum_u64(&mut votes);
    let chunks = votes[size..].iter().copied().max().unwrap_or(0) as u8;
    votes[..size].iter().all(|&v| v == 0).then_some(chunks)
}

/// One remote layer's exchange (ARCHITECTURE.md §Partitioning): K10's blocks, their skeletons over `transport` and the vote; on a yes the pending receive that [`receive_chunk`] moves chunk by chunk.
/// Runs after K1 has filled `cnt` at the agreed bucket count; a rank that cannot receive votes no and returns its error, and on any no nobody posts and a ready rank takes every received block as empty.
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
    let xfer_ns = &mut scratch.xfer_ns;
    #[cfg(feature = "phase-timing")]
    let t_exchange = std::time::Instant::now();
    let skeletons: Vec<Option<BlockSkeletons<W>>> = send
        .iter()
        .map(|p| {
            p.as_ref().map(|p| {
                let mut s = export.skeleton();
                s.fill_from(&p.blocks);
                s
            })
        })
        .collect();
    let recv = transport.exchange(skeletons, &mut export.skeletons);
    let (rows, ops) = {
        let blocks = paired_blocks(plan, &recv);
        let ready = stage_receive(sum, export, &blocks, cap_rows, xfer_ns)
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
                let recv_off: Vec<&[u32]> = blocks.iter().map(|b| b.offsets.as_slice()).collect();
                let ops = schedule(&partners, &own_off, &recv_off, &map);
                (Ok(export.recv_rows as u64), Some((map, own, ops)))
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

/// The receive layout from the skeletons in plan order, checked against the segment cap before any row moves; sizes `recv_*` for this rank's own chunk count, which it returns as `log2`.
fn stage_receive<const W: usize>(
    sum: &GpuSum<W>,
    export: &mut DeviceExport<W>,
    blocks: &[&Skeleton],
    cap_rows: usize,
    xfer_ns: &mut XferNs,
) -> Result<u8, GpuError> {
    let b = sum.hash.num_buckets();
    lay_out(
        export,
        b,
        blocks.iter().map(|bl| {
            (
                bl.header.num_buckets,
                &bl.offsets[..],
                bl.header.rows as usize,
            )
        }),
    );
    if export.recv_max_segment > MAX_RECV_SEGMENT {
        return Err(GpuError::Unsupported(
            "a fused-layer block or a received segment exceeds the record cap at the agreed bucket count",
        ));
    }
    #[cfg(any(test, feature = "test-utils"))]
    if std::mem::take(&mut export.fail_recv_growth) {
        return Err(GpuError::OutOfMemory {
            device: sum.device(),
            bytes: (export.recv_rows * recv_row_bytes::<W>()) as u64,
        });
    }
    let (log2, most) = recv_chunks(&export.off_host, b, cap_rows);
    size_receive(sum, export, most, xfer_ns)?;
    Ok(log2)
}

/// Whether this rank's wire can carry the group, so a failure it can predict is a no vote rather than an error after a yes.
fn wire_ready<const W: usize>(export: &DeviceExport<W>) -> Result<(), GpuError> {
    match &export.wire {
        None => Err(GpuError::Unsupported(
            "a remote layer on a partition without a device wire",
        )),
        Some(wire) if !wire.is_healthy() => Err(GpuError::Wire(
            "an exchange on a failed or aborted device wire",
        )),
        Some(_) => Ok(()),
    }
}

/// A ready rank's side of a no vote: every received block is empty.
fn discard_received<const W: usize>(export: &mut DeviceExport<W>) -> Result<u64, GpuError> {
    let n = export.off_host.len();
    export
        .stream
        .memset_zeros(&mut export.recv_off.slice_mut(0..n))?;
    export.recv_rows = 0;
    export.recv_max_segment = 0;
    Ok(0)
}

/// Device bytes one received row holds in `recv_*`: both key columns, the coefficient and the fingerprint.
pub(crate) const fn recv_row_bytes<const W: usize>() -> usize {
    (2 * W + 3) * std::mem::size_of::<u64>()
}

/// Received rows a chunk may hold under a cap of `bytes`.
fn recv_cap_rows<const W: usize>(bytes: usize) -> usize {
    (bytes / recv_row_bytes::<W>()).max(1)
}

/// `off_host`, `recv_max_segment` and `recv_rows` from received blocks in plan order, each `(positions, CSR offsets, rows)`.
fn lay_out<'a, const W: usize>(
    export: &mut DeviceExport<W>,
    b: usize,
    blocks: impl Iterator<Item = (u32, &'a [u32], usize)>,
) {
    let mut total = 0usize;
    let mut max_seg = 0usize;
    export.off_host.clear();
    for (num_buckets, offsets, rows) in blocks {
        assert_eq!(
            num_buckets as usize, b,
            "a partner sent a block indexed by {num_buckets} buckets where this partition has {b}"
        );
        export.off_host.extend_from_slice(&offsets[..b + 1]);
        max_seg = offsets[..b + 1]
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0)
            .max(max_seg);
        total += rows;
    }
    export.recv_max_segment = max_seg;
    export.recv_rows = total;
}

/// Received rows over every block in positions `lo..hi` of the concatenated `K × (b + 1)` offsets.
fn rows_between(off: &[u32], b: usize, lo: usize, hi: usize) -> usize {
    off.chunks_exact(b + 1)
        .map(|o| (o[hi] - o[lo]) as usize)
        .sum()
}

/// The largest chunk's received rows when `b` positions are cut into `2^log2` equal chunks.
fn chunk_max(off: &[u32], b: usize, log2: u8) -> usize {
    let step = b >> log2;
    (0..1usize << log2)
        .map(|c| rows_between(off, b, c * step, (c + 1) * step))
        .max()
        .unwrap_or(0)
}

/// The fewest chunks, a power of two up to one per position, whose received rows each fit `cap_rows`, as `(log2 chunks, rows of the largest)`.
/// A power of two because such a cut refines every coarser one, so a group agreeing on the largest count its members asked for never grows anyone's chunk.
fn recv_chunks(off: &[u32], b: usize, cap_rows: usize) -> (u8, usize) {
    let mut log2 = 0u8;
    loop {
        let most = chunk_max(off, b, log2);
        if most <= cap_rows || 1usize << log2 >= b {
            return (log2, most);
        }
        log2 += 1;
    }
}

/// `2^log2` equal chunks of `b` positions in the fused layer's position order.
fn position_chunks(bits: u8, b: usize, log2: u8) -> ChunkMap {
    let mut map = ChunkMap::default();
    map.rebuild(&Gf2Span::new(&[0], bits), b, 1 << log2);
    debug_assert_eq!(map.chunks(), 1 << log2);
    map
}

/// Room for `rows` received rows in `recv_*`, and the offsets in `recv_off`.
fn size_receive<const W: usize>(
    sum: &GpuSum<W>,
    export: &mut DeviceExport<W>,
    rows: usize,
    xfer_ns: &mut XferNs,
) -> Result<(), GpuError> {
    let s = &sum.stream;
    let o = sum.device();
    let DeviceExport {
        recv_off,
        recv_x,
        recv_z,
        recv_c,
        recv_g,
        off_host,
        ..
    } = export;
    grow(s, recv_off, off_host.len().max(2), o)?;
    grow(s, recv_x, rows.max(1) * W, o)?;
    grow(s, recv_z, rows.max(1) * W, o)?;
    grow(s, recv_c, 2 * rows.max(1), o)?;
    grow(s, recv_g, rows.max(1), o)?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    xfer(xfer_ns, Xfer::H2d, || {
        s.memcpy_htod(&off_host[..], &mut recv_off.slice_mut(0..off_host.len()))?;
        Ok(())
    })
}

/// Positions `lo..hi` of every received block laid end to end from row 0 of `recv_*` in plan order, returning their rows, with in `base` the value that makes the fused layer's `base[k] + off[k][p]` land there.
/// That base is `start - off[k][lo]` in wrapping `u32` arithmetic, which the kernel's own `u32` sum undoes for every `p` in the chunk.
fn chunk_layout(off: &[u32], b: usize, lo: usize, hi: usize, base: &mut Vec<u32>) -> usize {
    base.clear();
    base.resize(16, 0);
    let mut at = 0usize;
    for (k, o) in off.chunks_exact(b + 1).enumerate() {
        base[k] = (at as u32).wrapping_sub(o[lo]);
        at += (o[hi] - o[lo]) as usize;
    }
    at
}

/// Move chunk `c` of the pending receive into `recv_*`, fingerprinted, with its bases in `recv_base`; returns its rows.
/// Called in chunk order before the fused layer's first batch in the chunk.
pub(crate) fn receive_chunk<const W: usize>(
    sum: &GpuSum<W>,
    scratch: &mut LayerScratch<W>,
    c: usize,
) -> Result<usize, GpuError> {
    #[cfg(feature = "phase-timing")]
    let t = std::time::Instant::now();
    let export = &mut scratch.export;
    let mut pending = export.pending.take().expect("a pending receive");
    debug_assert_eq!(pending.next, c);
    let r = move_chunk(sum, export, &mut pending, c, true, &mut scratch.xfer_ns);
    export.pending = Some(pending);
    #[cfg(feature = "phase-timing")]
    {
        scratch.laps.exchange_ns += t.elapsed().as_nanos() as u64;
    }
    #[cfg(any(test, feature = "test-utils"))]
    if r.is_ok() && scratch.export.fail_after_chunk == Some(c) {
        scratch.export.fail_after_chunk = None;
        return Err(GpuError::OutOfMemory {
            device: sum.device(),
            bytes: 0,
        });
    }
    r
}

/// Chunk `c` of `pending` into `recv_*` through one wire group; with `keep` its rows are fingerprinted and its bases uploaded, without it the transfer only completes, as a failed layer's must.
fn move_chunk<const W: usize>(
    sum: &GpuSum<W>,
    export: &mut DeviceExport<W>,
    pending: &mut PendingRecv<W>,
    c: usize,
    keep: bool,
    xfer_ns: &mut XferNs,
) -> Result<usize, GpuError> {
    let b = sum.hash.num_buckets();
    let s = &sum.stream;
    let (lo, hi) = (
        pending.map.bound(c) as usize,
        pending.map.bound(c + 1) as usize,
    );
    let n = chunk_layout(&export.off_host, b, lo, hi, &mut export.base_host);
    let wire = export.wire.clone().ok_or(GpuError::Unsupported(
        "a remote layer on a partition without a device wire",
    ))?;
    let DeviceExport {
        recv_base,
        recv_x,
        recv_z,
        recv_c,
        recv_g,
        base_host,
        ..
    } = export;
    let PendingRecv { send, own, ops, .. } = &*pending;
    // `schedule` lists every send before every receive, each chunk-major.
    let (sends, recvs) = ops.split_at(ops.partition_point(|op| op.kind == WireOpKind::Send));
    let of_chunk = |ops: &'_ [ScheduledOp]| -> std::ops::Range<usize> {
        ops.partition_point(|op| op.chunk < c)..ops.partition_point(|op| op.chunk <= c)
    };
    let (sends, recvs) = (&sends[of_chunk(sends)], &recvs[of_chunk(recvs)]);
    let mut group = WireGroup::new();
    for op in sends {
        let (q, j) = own[op.k];
        let block = &send[q]
            .as_ref()
            .expect("a payload for every partner")
            .blocks[j];
        let (r0, r1) = op.rows;
        let e = op.column.elems_per_row::<W>();
        match op.column {
            WireColumn::X => group.send(block.x.slice(r0 * e..r1 * e), op.peer, s),
            WireColumn::Z => group.send(block.z.slice(r0 * e..r1 * e), op.peer, s),
            WireColumn::Coeff => group.send(block.c.slice(r0 * e..r1 * e), op.peer, s),
        }
    }
    let recv_parts = |column: WireColumn| -> Vec<(usize, u32)> {
        let e = column.elems_per_row::<W>();
        recvs
            .iter()
            .filter(|op| op.column == column)
            .map(|op| ((op.rows.1 - op.rows.0) * e, op.peer))
            .collect()
    };
    group.recv_parts(recv_x.slice_mut(0..n * W), &recv_parts(WireColumn::X), s);
    group.recv_parts(recv_z.slice_mut(0..n * W), &recv_parts(WireColumn::Z), s);
    group.recv_parts(
        recv_c.slice_mut(0..2 * n),
        &recv_parts(WireColumn::Coeff),
        s,
    );
    if let Err(e) = group.post(&*wire).and_then(|()| wire.wait(s)) {
        pending.failed = true;
        return Err(e);
    }
    pending.next = c + 1;
    if !keep {
        return Ok(n);
    }
    launch_fingerprint(
        s,
        &sum.kernels,
        &sum.fp_rows,
        (&*recv_x, &*recv_z),
        recv_g,
        n,
    )?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    xfer(xfer_ns, Xfer::H2d, || {
        s.memcpy_htod(&base_host[..], recv_base)?;
        Ok(())
    })?;
    Ok(n)
}

/// End the current layer's pending receive, if any: chunks a failure left unmoved still complete their transfers, rows discarded, so every peer's receive completes, unless a group already failed; then the send payloads return to the pool.
pub(crate) fn finish_receive<const W: usize>(
    sum: &GpuSum<W>,
    scratch: &mut LayerScratch<W>,
) -> Result<(), GpuError> {
    let export = &mut scratch.export;
    let Some(mut pending) = export.pending.take() else {
        return Ok(());
    };
    let mut drained = Ok(());
    for c in pending.next..pending.map.chunks() {
        if pending.failed {
            break;
        }
        if let Err(e) = move_chunk(sum, export, &mut pending, c, false, &mut scratch.xfer_ns) {
            drained = Err(e);
            break;
        }
    }
    let synced = sum.stream.synchronize().map_err(GpuError::from);
    // A failed group's peers may still read these, but a failed partition never exports again, so they are only freed with it.
    export.sends.extend(pending.send.into_iter().flatten());
    drained.and(synced)
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

/// K10's count and scan for entry `e` at bucket delta `bd`, leaving the CSR offsets in `scratch.export.off` and their copy in `offsets`; returns the block's rows.
fn export_offsets<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    scratch: &mut LayerScratch<W>,
    bd: u32,
    e: u32,
    offsets: &mut Vec<u32>,
) -> Result<usize, GpuError> {
    let s = &sum.stream;
    let k = &sum.kernels;
    let b = sum.hash.num_buckets();
    let (e32, b32) = (table.entries as u32, b as u32);
    // SAFETY: arguments match `k_export_counts` in export.cu; `counts` holds `b` entries.
    unsafe {
        s.launch_builder(&k.export_counts)
            .arg(&scratch.cnt)
            .arg(&scratch.bucket_at)
            .arg(&bd)
            .arg(&e)
            .arg(&e32)
            .arg(&b32)
            .arg(&mut scratch.export.counts)
            .launch(thread_per(b, 1024))?;
    }
    exclusive_scan(
        s,
        k,
        &scratch.export.counts.slice(0..b),
        &mut scratch.export.off.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut scratch.tot_a,
    )?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    offsets.clear();
    offsets.resize(b + 1, 0);
    let off = &scratch.export.off;
    xfer(&mut scratch.xfer_ns, Xfer::D2h, || {
        s.memcpy_dtoh(&off.slice(0..b + 1), &mut offsets[..])?;
        s.synchronize()?;
        Ok(())
    })?;
    Ok(offsets[b] as usize)
}

/// The scratch buffers K10's fill reads, borrowed apart from the columns it writes.
struct FillCtx<'a> {
    bucket_at: &'a CudaSlice<u32>,
    table: &'a TableBufs,
    off: &'a CudaSlice<u32>,
}

/// K10's fill of entry `e` at bucket delta `bd` into `(x, z, c)`, which hold the rows `ctx.off` names.
#[allow(clippy::too_many_arguments)]
fn export_fill<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    ctx: FillCtx<'_>,
    bd: u32,
    e: u32,
    x: &mut CudaSlice<u64>,
    z: &mut CudaSlice<u64>,
    c: &mut CudaSlice<f64>,
) -> Result<(), GpuError> {
    let s = &sum.stream;
    let b = sum.hash.num_buckets();
    let (e32, b32) = (table.entries as u32, b as u32);
    // SAFETY: arguments match `k_export_fill` in export.cu; the columns hold the block's rows and `off` its offsets.
    unsafe {
        s.launch_builder(&sum.kernels.export_fill)
            .arg(&sum.cols.x)
            .arg(&sum.cols.z)
            .arg(&sum.cols.coeff)
            .arg(&sum.cols.start)
            .arg(&sum.cols.lens)
            .arg(ctx.bucket_at)
            .arg(&table.mode)
            .arg(&e32)
            .arg(&table.kq)
            .arg(&table.q0)
            .arg(&table.q1)
            .arg(&table.rot_cos)
            .arg(&table.rot_sin)
            .arg(&ctx.table.amp)
            .arg(&ctx.table.mask)
            .arg(&ctx.table.nz)
            .arg(&bd)
            .arg(&e)
            .arg(&b32)
            .arg(ctx.off)
            .arg(x)
            .arg(z)
            .arg(c)
            .launch(warp_per_bucket(b))?;
    }
    Ok(())
}

/// K10 for every remote delta into device blocks: one block per delta into pooled [`DevicePayload`]s, in ascending remote-delta index per partner.
pub(crate) fn export_blocks_device<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    plan: &PartitionPlan,
    scratch: &mut LayerScratch<W>,
    size: u32,
) -> Result<(Vec<Option<DevicePayload<W>>>, LayerExchangeCounts), GpuError> {
    let mut send: Vec<Option<DevicePayload<W>>> = (0..size).map(|_| None).collect();
    let mut counts = LayerExchangeCounts::none(size);
    let b = sum.hash.num_buckets();
    let s: Arc<CudaStream> = sum.stream.clone();
    let o = sum.device();
    grow(&s, &mut scratch.export.counts, b, o)?;
    grow(&s, &mut scratch.export.off, b + 1, o)?;
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
        payload.block_mut(entries.len() - 1, &s)?;
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
        let block = payload.block_mut(j, &s)?;
        let (bd, e) = (r.bucket_delta, r.entry as u32);
        let t0 = scratch.event(sum)?;
        let rows = export_offsets(sum, table, scratch, bd, e, &mut block.offsets)?;
        block.set_header(e, b);
        if rows > 0 {
            block.grow(&s, rows, o)?;
            let ctx = FillCtx {
                bucket_at: &scratch.bucket_at,
                table: &scratch.table,
                off: &scratch.export.off,
            };
            export_fill(
                sum,
                table,
                ctx,
                bd,
                e,
                &mut block.x,
                &mut block.z,
                &mut block.c,
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
    s.synchronize()?;
    Ok((send, counts))
}

/// Elements `exclusive_scan` handles, so the split's `K × positions` offsets must fit.
const SCAN_LIMIT: usize = 1 << 24;

/// Per partner, the remote entries the sender-side merge applies to, empty where it does not: at least two entries that can emit one key (`DevicePrepared::entries_can_collide`) and every source bucket inside the tag's offset field.
fn premerge_groups<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    plan: &PartitionPlan,
    scratch: &mut LayerScratch<W>,
    size: u32,
) -> Result<Vec<Vec<usize>>, GpuError> {
    let mut groups = vec![Vec::new(); size as usize];
    if !scratch.options.premerge {
        return Ok(groups);
    }
    for r in &plan.remote {
        groups[r.partner as usize].push(r.entry);
    }
    let b = sum.hash.num_buckets();
    for g in &mut groups {
        if g.len() < 2 || g.len() * b > SCAN_LIMIT || !table.entries_can_collide(g) {
            g.clear();
        }
    }
    if groups.iter().all(Vec::is_empty) {
        return Ok(groups);
    }
    // K3's tag addresses 4096 rows of a source bucket; a longer one fails the layer after the exchange, and must not reach K3 before it.
    if scratch.longest_bucket(sum)? as usize > MAX_BUCKET_LEN {
        groups.iter_mut().for_each(Vec::clear);
    }
    Ok(groups)
}

/// The sender-side merge of one partner's blocks (ARCHITECTURE.md §Partitioning): K3 over the partner's sub-table under the keep-everything program, each position's rows split back over the blocks within their unmerged counts so every segment still fits the receiver's tag.
/// `blocks` are the partner's in its remote-delta order; returns the rows written, or `None` having written nothing when a position's records exceed the fused kernel's cap.
fn premerge_partner<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    entries: &[usize],
    scratch: &mut LayerScratch<W>,
    blocks: &mut [DeviceBlock<W>],
) -> Result<Option<u64>, GpuError> {
    let s: Arc<CudaStream> = sum.stream.clone();
    let k = sum.kernels.clone();
    let o = sum.device();
    let b = sum.hash.num_buckets();
    let nk = entries.len();
    let sub = table.restrict(entries);
    let mut sel = [0u32; 16];
    for (j, &e) in entries.iter().enumerate() {
        sel[j] = e as u32;
    }
    let (b32, k32, e32) = (b as u32, nk as u32, table.entries as u32);
    let t0 = scratch.event(sum)?;
    {
        let pm = &mut scratch.export.premerge;
        grow(&s, &mut pm.cnt, b * nk, o)?;
        grow(&s, &mut pm.rows, b, o)?;
        grow(&s, &mut pm.start, b + 1, o)?;
        grow(&s, &mut pm.out_len_pos, b, o)?;
        #[cfg(feature = "phase-timing")]
        s.synchronize()?;
        xfer(&mut scratch.xfer_ns, Xfer::H2d, || {
            pm.table.upload(&s, &sub)?;
            s.memcpy_htod(&sel[..], &mut pm.sel)?;
            Ok(())
        })?;
        // SAFETY: arguments match `k_premerge_counts` in export.cu; `cnt` holds the layer's `b × E` counts and `pm.cnt` room for `b × K`.
        unsafe {
            s.launch_builder(&k.premerge_counts)
                .arg(&scratch.cnt)
                .arg(&e32)
                .arg(&pm.sel)
                .arg(&k32)
                .arg(&b32)
                .arg(&mut pm.cnt)
                .launch(thread_per(b * nk, 256))?;
        }
        // SAFETY: arguments match `k_rows` in count.cu; every `rem` entry is `NO_REMOTE`, so `recv_off` is never read.
        unsafe {
            s.launch_builder(&k.rows)
                .arg(&pm.cnt)
                .arg(&scratch.bucket_at)
                .arg(&pm.table.bd)
                .arg(&pm.table.rem)
                .arg(&scratch.export.recv_off)
                .arg(&mut pm.rows)
                .arg(&b32)
                .arg(&k32)
                .launch(thread_per(b, 1024))?;
        }
        exclusive_scan(
            &s,
            &k,
            &pm.rows.slice(0..b),
            &mut pm.start.slice_mut(0..b + 1),
            b,
            &mut scratch.scan,
            &mut pm.tot,
        )?;
    }
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    let (total, most) = {
        let pm = &mut scratch.export.premerge;
        pm.start_host.resize(b + 1, 0);
        let (start, start_host, tot) = (&pm.start, &mut pm.start_host, &pm.tot);
        xfer(&mut scratch.xfer_ns, Xfer::D2h, || {
            let v = s.clone_dtoh(tot)?;
            s.memcpy_dtoh(&start.slice(0..b + 1), &mut start_host[..])?;
            s.synchronize()?;
            Ok((v[0] as u64, v[1] as usize))
        })?
    };
    let cap = k.layer_cap();
    if most > cap {
        scratch.lap(sum, t0, |m| &mut m.export)?;
        return Ok(None);
    }
    let (func, _, smem) = fused_variant(&k, W, most, sub.dense);
    let (batches, max_batch_rows) = arena_batches::<W>(
        &scratch.export.premerge.start_host,
        scratch.options.arena_bytes,
        cap,
        &[],
    );
    let mut arena = match scratch.arena.take() {
        Some(a) => a,
        None => DeviceColumns::<W>::with_capacity(&s, o, max_batch_rows, 1)?,
    };
    arena.len = 0;
    arena.buckets = 0;
    arena.reserve(max_batch_rows, 1)?;
    let mut running = vec![0u32; nk];
    for bl in blocks.iter_mut() {
        bl.offsets.clear();
        bl.offsets.resize(b + 1, 0);
    }
    let r = premerge_batches(
        sum,
        &sub,
        scratch,
        &mut arena,
        &batches,
        (func, smem),
        &mut running,
        blocks,
    );
    scratch.arena = Some(arena);
    r?;
    let mut merged = 0u64;
    for (j, &e) in entries.iter().enumerate() {
        let rows = running[j];
        merged += u64::from(rows);
        blocks[j].offsets[b] = rows;
        blocks[j].set_header(e as u32, b);
    }
    let fb = s.clone_dtoh(&scratch.export.premerge.fallback)?;
    s.synchronize()?;
    scratch.counters.fallback_hi += fb[0];
    scratch.counters.fallback_key += fb[1];
    scratch.counters.rows_premerged += total - merged;
    scratch.lap(sum, t0, |m| &mut m.export)?;
    Ok(Some(merged))
}

/// The merge's batch loop: K3 into the arena, the split and its scan, the block offsets, and the copy of every block's share.
#[allow(clippy::too_many_arguments)]
fn premerge_batches<const W: usize>(
    sum: &GpuSum<W>,
    sub: &DevicePrepared<W>,
    scratch: &mut LayerScratch<W>,
    arena: &mut DeviceColumns<W>,
    batches: &[(usize, usize)],
    kernel: (&cudarc::driver::CudaFunction, u32),
    running: &mut [u32],
    blocks: &mut [DeviceBlock<W>],
) -> Result<(), GpuError> {
    let s = &sum.stream;
    let k = &sum.kernels;
    let o = sum.device();
    let nk = running.len();
    let k32 = nk as u32;
    let LayerScratch {
        bucket_at,
        scan,
        export,
        xfer_ns,
        ..
    } = scratch;
    let DeviceExport {
        premerge: pm,
        recv_off,
        recv_base,
        recv_x,
        recv_z,
        recv_c,
        recv_g,
        ..
    } = export;
    s.memset_zeros(&mut pm.fallback)?;
    for &(p0, p1) in batches {
        let n = p1 - p0;
        let (p0u, n32) = (p0 as u32, n as u32);
        let fused = FusedTable {
            cnt: &pm.cnt,
            seg_start: &pm.start,
            table: &pm.table,
        };
        let recv = FusedRecv {
            off: recv_off,
            base: recv_base,
            x: recv_x,
            z: recv_z,
            c: recv_c,
            g: recv_g,
        };
        let written = FusedOut {
            arena: &mut *arena,
            out_len_pos: &mut pm.out_len_pos,
            fallback: &mut pm.fallback,
        };
        launch_fused(
            sum,
            sub,
            &fused,
            &recv,
            bucket_at,
            &KeepProgram::KEEP,
            kernel,
            (p0u, n32),
            written,
        )?;
        grow(s, &mut pm.lens, nk * n, o)?;
        grow(s, &mut pm.loff, nk * n + 1, o)?;
        // SAFETY: arguments match `k_premerge_split` in export.cu; `lens` holds `K × n` entries.
        unsafe {
            s.launch_builder(&k.premerge_split)
                .arg(&pm.out_len_pos)
                .arg(&pm.cnt)
                .arg(&*bucket_at)
                .arg(&pm.table.bd)
                .arg(&k32)
                .arg(&p0u)
                .arg(&n32)
                .arg(&mut pm.lens)
                .launch(thread_per(n, 256))?;
        }
        exclusive_scan(
            s,
            k,
            &pm.lens.slice(0..nk * n),
            &mut pm.loff.slice_mut(0..nk * n + 1),
            nk * n,
            scan,
            &mut pm.tot,
        )?;
        #[cfg(feature = "phase-timing")]
        s.synchronize()?;
        pm.loff_host.resize(nk * n + 1, 0);
        {
            let (loff, loff_host) = (&pm.loff, &mut pm.loff_host);
            xfer(xfer_ns, Xfer::D2h, || {
                s.memcpy_dtoh(&loff.slice(0..nk * n + 1), &mut loff_host[..])?;
                s.synchronize()?;
                Ok(())
            })?;
        }
        let lo = &pm.loff_host;
        let share = |j: usize| (lo[j * n] as usize, (lo[(j + 1) * n] - lo[j * n]) as usize);
        let set_offsets = |off: &mut [u32], j: usize, run: u32| {
            for i in 0..n {
                off[p0 + i] = run + lo[j * n + i] - lo[j * n];
            }
        };
        let copy = |j: usize,
                    dst0: u32,
                    (x, z, c): (
            &mut CudaSlice<u64>,
            &mut CudaSlice<u64>,
            &mut CudaSlice<f64>,
        )|
         -> Result<(), GpuError> {
            let j32 = j as u32;
            // SAFETY: arguments match `k_premerge_copy` in export.cu; the destination holds `dst0` plus block `j`'s share of the batch.
            unsafe {
                s.launch_builder(&k.premerge_copy)
                    .arg(&arena.x)
                    .arg(&arena.z)
                    .arg(&arena.coeff)
                    .arg(&pm.start)
                    .arg(&pm.lens)
                    .arg(&pm.loff)
                    .arg(&j32)
                    .arg(&p0u)
                    .arg(&n32)
                    .arg(&dst0)
                    .arg(x)
                    .arg(z)
                    .arg(c)
                    .launch(warp_per_bucket(n))?;
            }
            Ok(())
        };
        for (j, bl) in blocks.iter_mut().enumerate() {
            let (_, rows) = share(j);
            set_offsets(&mut bl.offsets, j, running[j]);
            if rows > 0 {
                let live = running[j] as usize;
                bl.reserve_keep(s, live, live + rows, o)?;
                let DeviceBlock { x, z, c, .. } = bl;
                copy(j, running[j], (x, z, c))?;
            }
            running[j] += rows as u32;
        }
    }
    Ok(())
}

/// Rows the tag's offset field can address in one received segment.
pub(crate) const MAX_RECV_SEGMENT: usize = MAX_BUCKET_LEN;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::clifford::Clifford2Q;
    use crate::channel::{Channel, GeneralUnitary2Q};
    use crate::engine::gpu::layer::GpuLayerOptions;
    use crate::engine::gpu::DEFAULT_ARENA_BYTES;
    use crate::engine::partitioned::export::{export_layer, ExportScratch};
    use crate::engine::partitioned::transport::{ExchangeBlock, PartnerPayload};
    use crate::pauli_sum::hash::{Gf2Hash, PartitionRows, B_MAX_BITS};
    use crate::pauli_sum::PauliSum;
    use crate::test_support::{haar_su4_matrix, rand_sum, zz_rotation};
    use num_complex::Complex64;

    /// The three channels of the export fixtures: a dense SU(4), a Clifford and a rotation.
    fn fixture_channels<const W: usize>() -> Vec<(&'static str, Box<dyn Channel<W>>)> {
        vec![
            (
                "su4",
                Box::new(GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix())),
            ),
            ("cnot", Box::new(Clifford2Q::cnot(1, 3))),
            ("zz", Box::new(zz_rotation::<W>(0, 2, 0.3))),
        ]
    }

    fn su4<const W: usize>() -> GeneralUnitary2Q {
        GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix())
    }

    /// Every support pattern of qubits 0 and 1 under each of `base`'s keys, so an SU(4) on `(0, 1)` makes one partner's remote rows collide.
    fn dense_on_01<const W: usize>(base: &PauliSum<W>) -> PauliSum<W> {
        let mut acc = crate::pauli_sum::accumulator::BuildAccumulator::<W>::new(base.num_qubits());
        for (x, z, c) in base.iter() {
            for s in 0..16u64 {
                let (mut x, mut z) = (*x, *z);
                x[0] = (x[0] & !0b11) | (s & 0b11);
                z[0] = (z[0] & !0b11) | (s >> 2);
                acc.add_term(
                    crate::pauli_string::PauliString::<W> { x, z },
                    crate::phase::Phase::ONE,
                    c * (1.0 + s as f64),
                );
            }
        }
        acc.finalize()
    }

    /// One rank's layer: the host's `export_layer` and the device's `export_blocks_device` of the same share.
    struct Exported<const W: usize> {
        plan: PartitionPlan,
        nb: usize,
        want: Vec<Option<PartnerPayload<W>>>,
        want_rows: Vec<u64>,
        dev: GpuSum<W>,
        scratch: LayerScratch<W>,
        got: Vec<Option<DevicePayload<W>>>,
        got_counts: LayerExchangeCounts,
    }

    /// Rank `rank`'s share of `input` under `rows` through `ch`, exported on the host and on the device (sender-side merge `premerge`, arena `arena_bytes`, NVRTC options `nvrtc`); `None` when the rank has no remote delta.
    fn export_rank<const W: usize>(
        input: &PauliSum<W>,
        rows: &PartitionRows<W>,
        ch: &dyn Channel<W>,
        rank: u32,
        (premerge, arena_bytes, nvrtc): (bool, usize, &[String]),
    ) -> Option<Exported<W>> {
        let size = rows.num_partitions() as u32;
        let local = input.filter_partition(rows, rank);
        let prep = ch.prepare(local.hash(), false).expect("prepared");
        let plan = PartitionPlan::new(&prep, rows, rank);
        if !plan.has_remote() {
            return None;
        }
        let nb = local.num_buckets();
        let mut map = ChunkMap::default();
        map.rebuild(
            &Gf2Span::new(&plan.local_bucket_deltas, local.hash().bits()),
            nb,
            1,
        );
        let (want, want_counts) = export_layer(
            &local,
            &prep,
            &plan,
            size,
            &map,
            &mut ExportScratch::default(),
        );
        let dev = GpuSum::from_host_with_options(&local, 0, nvrtc).expect("upload");
        let options = GpuLayerOptions {
            premerge,
            arena_bytes,
            ..GpuLayerOptions::default()
        };
        let mut scratch = LayerScratch::new(&dev, options).expect("scratch");
        let table = DevicePrepared::new(&prep, dev.hash(), &dev.fp, &plan.remote);
        scratch.upload_table(&dev, &table).expect("table");
        scratch.count_local(&dev, &table).expect("count");
        let (got, got_counts) =
            export_blocks_device(&dev, &table, &plan, &mut scratch, size).expect("export");
        Some(Exported {
            plan,
            nb,
            want,
            want_rows: want_counts.rows_to,
            dev,
            scratch,
            got,
            got_counts,
        })
    }

    /// A device block's columns on the host.
    fn download<const W: usize>(
        s: &Arc<CudaStream>,
        block: &DeviceBlock<W>,
    ) -> (Vec<u64>, Vec<u64>, Vec<f64>) {
        let n = block.rows();
        let x = s.clone_dtoh(&block.x.slice(0..n * W)).unwrap();
        let z = s.clone_dtoh(&block.z.slice(0..n * W)).unwrap();
        let c = s.clone_dtoh(&block.c.slice(0..2 * n)).unwrap();
        s.synchronize().unwrap();
        (x, z, c)
    }

    /// The device payloads equal the host payloads bitwise: presence, headers, offsets and the three columns.
    fn assert_payloads_eq<const W: usize>(e: &Exported<W>, what: &str) {
        assert_eq!(e.got_counts.rows_sent, e.want_rows, "{what}: rows");
        assert_eq!(e.got.len(), e.want.len());
        for (q, (g, w)) in e.got.iter().zip(&e.want).enumerate() {
            match (g, w) {
                (None, None) => {}
                (Some(g), Some(w)) => {
                    assert_eq!(g.blocks.len(), w.blocks.len(), "{what}: blocks to {q}");
                    for (j, (gb, wb)) in g.blocks.iter().zip(&w.blocks).enumerate() {
                        assert_eq!(gb.header, wb.header, "{what}: header {j} to {q}");
                        assert_eq!(gb.offsets, wb.offsets, "{what}: offsets {j} to {q}");
                        let (x, z, c) = download(&e.dev.stream, gb);
                        let (wx, wz, wc) = wb.cols();
                        assert_eq!(x, wx.as_flattened(), "{what}: x {j} to {q}");
                        assert_eq!(z, wz.as_flattened(), "{what}: z {j} to {q}");
                        assert_eq!(
                            c,
                            bytemuck::cast_slice::<Complex64, f64>(wc),
                            "{what}: coeff {j} to {q}"
                        );
                    }
                }
                _ => panic!("{what}: payload presence to {q} differs"),
            }
        }
    }

    /// Unmerged, the device payloads equal `export_layer`'s blocks bitwise, and the SU(4) ships several blocks to one partner.
    fn device_export_matches_host<const W: usize>(num_qubits: usize, seed: u64, pbits: u8) {
        let input = rand_sum::<W>(6000, num_qubits, seed);
        let rows = PartitionRows::<W>::from_seed(num_qubits, pbits, seed ^ 0x77);
        let (mut exported, mut multi_block_payloads) = (0usize, 0);
        for (name, ch) in &fixture_channels::<W>() {
            for rank in 0..rows.num_partitions() as u32 {
                let off = (false, DEFAULT_ARENA_BYTES, &[][..]);
                let Some(e) = export_rank(&input, &rows, ch.as_ref(), rank, off) else {
                    continue;
                };
                assert_payloads_eq(&e, &format!("W={W} {name} rank {rank}"));
                let blocks = e.got.iter().flatten().map(|p| p.blocks.len());
                multi_block_payloads += blocks.clone().filter(|&n| n >= 2).count();
                exported += blocks.sum::<usize>();
            }
        }
        assert!(exported > 0, "the fixture must export something");
        assert!(
            multi_block_payloads > 0,
            "the SU(4) layer must ship several remote deltas to one partner"
        );
    }

    #[test]
    fn device_payloads_match_export_layer_bitwise() {
        crate::require_cuda!();
        device_export_matches_host::<1>(12, 0xE7, 1);
        device_export_matches_host::<1>(12, 0xE8, 2);
        device_export_matches_host::<2>(100, 0xE9, 1);
        device_export_matches_host::<2>(100, 0xEA, 2);
    }

    /// Two blocks over eight positions receiving `[1, 0, 2, 4, 0, 3, 1, 1]` rows: each cap takes the fewest power-of-two chunks that fit it, and one position alone may exceed it.
    #[test]
    fn the_receive_takes_the_fewest_power_of_two_chunks_under_its_cap() {
        let off: Vec<u32> = [[0u32, 1, 1, 3, 3, 3, 6, 7, 8], [0, 0, 0, 0, 4, 4, 4, 4, 4]].concat();
        let b = 8;
        assert_eq!(rows_between(&off, b, 0, 8), 12);
        assert_eq!(rows_between(&off, b, 3, 4), 4);
        assert_eq!(
            (0..=3).map(|j| chunk_max(&off, b, j)).collect::<Vec<_>>(),
            vec![12, 7, 6, 4],
            "a finer power-of-two cut never grows the largest chunk"
        );
        assert_eq!(recv_chunks(&off, b, usize::MAX), (0, 12));
        assert_eq!(recv_chunks(&off, b, 12), (0, 12));
        assert_eq!(recv_chunks(&off, b, 7), (1, 7));
        assert_eq!(recv_chunks(&off, b, 6), (2, 6));
        assert_eq!(recv_chunks(&off, b, 3), (3, 4));
        assert_eq!(
            recv_chunks(&[0, 5], 1, 1),
            (0, 5),
            "one position is one chunk"
        );
        let mut base = Vec::new();
        assert_eq!(chunk_layout(&off, b, 2, 4, &mut base), 6);
        assert_eq!(base[0].wrapping_add(off[2]), 0);
        assert_eq!(base[1].wrapping_add(off[9 + 3]), 2);
        assert_eq!(base[1].wrapping_add(off[9 + 4]), 6);
    }

    /// Every rank learns one verdict and the largest chunk count any ready rank asked for; one rank not ready is a no everywhere.
    #[test]
    fn the_vote_agrees_the_largest_chunk_count() {
        use crate::engine::partitioned::transport::{Collectives, InProcessTransport};
        let run = |asks: [Option<u8>; 4]| -> Vec<Option<u8>> {
            let group = InProcessTransport::group(4);
            std::thread::scope(|s| {
                let hs: Vec<_> = group
                    .into_iter()
                    .map(|t| s.spawn(move || vote(&t, asks[t.rank() as usize])))
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            })
        };
        assert_eq!(run([Some(1), Some(3), Some(0), Some(2)]), vec![Some(3); 4]);
        assert_eq!(run([Some(0); 4]), vec![Some(0); 4]);
        assert_eq!(run([Some(2), None, Some(0), Some(5)]), vec![None; 4]);
    }

    /// A partition with no terms still ships one empty block per remote delta, the host's headers and offsets.
    #[test]
    fn an_empty_partition_ships_empty_blocks() {
        crate::require_cuda!();
        let seed = 0xE55u64;
        let input = PauliSum::<1>::empty_with_hash(12, Gf2Hash::<1>::new(12, 3, seed));
        let rows = PartitionRows::<1>::from_seed(12, 1, seed ^ 0x77);
        let off = (false, DEFAULT_ARENA_BYTES, &[][..]);
        let e = export_rank(&input, &rows, &su4::<1>(), 0, off).expect("the SU(4) must cross");
        assert_payloads_eq(&e, "empty");
        let payload = e.got[1].as_ref().expect("a device payload for the partner");
        assert_eq!(payload.blocks.len(), e.plan.remote.len());
        assert!(payload
            .blocks
            .iter()
            .all(|b| b.rows() == 0 && b.offsets.len() == 9));
    }

    type Key<const W: usize> = ([u64; W], [u64; W]);

    /// Per destination position, one partner's rows keyed by `(x, z)` with their coefficients summed across the partner's blocks.
    type PositionSums<const W: usize> = Vec<std::collections::BTreeMap<Key<W>, Complex64>>;

    /// One block's `(offsets, x, z, coeff)`, live rows only.
    type BlockCols<const W: usize> = (Vec<u32>, Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>);

    /// The per-position key sums of `blocks`, and the largest number of times one key occurs at one position.
    fn position_sums<const W: usize>(
        blocks: &[BlockCols<W>],
        b: usize,
    ) -> (PositionSums<W>, usize) {
        let mut out: PositionSums<W> = vec![Default::default(); b];
        let mut seen: Vec<std::collections::BTreeMap<Key<W>, usize>> = vec![Default::default(); b];
        let mut most = 0;
        for (off, x, z, c) in blocks {
            for p in 0..b {
                for i in off[p] as usize..off[p + 1] as usize {
                    *out[p]
                        .entry((x[i], z[i]))
                        .or_insert(Complex64::new(0.0, 0.0)) += c[i];
                    let n = seen[p].entry((x[i], z[i])).or_insert(0);
                    *n += 1;
                    most = most.max(*n);
                }
            }
        }
        (out, most)
    }

    /// Two per-position key sums agree to `tol`, a key absent on one side counting as zero.
    fn assert_position_sums_close<const W: usize>(
        got: &PositionSums<W>,
        want: &PositionSums<W>,
        tol: f64,
        what: &str,
    ) {
        let zero = Complex64::new(0.0, 0.0);
        for (p, (g, w)) in got.iter().zip(want).enumerate() {
            for (k, gc) in g {
                let wc = w.get(k).copied().unwrap_or(zero);
                assert!(
                    (gc - wc).norm() <= tol,
                    "{what}: position {p} key {k:?}: {gc} vs {wc}"
                );
            }
            for (k, wc) in w {
                if !g.contains_key(k) {
                    assert!(
                        wc.norm() <= tol,
                        "{what}: position {p} lost key {k:?} ({wc})"
                    );
                }
            }
        }
    }

    fn host_block_cols<const W: usize>(b: &ExchangeBlock<W>) -> BlockCols<W> {
        let (x, z, c) = b.cols();
        (b.offsets.clone(), x.to_vec(), z.to_vec(), c.to_vec())
    }

    fn device_block_cols<const W: usize>(s: &Arc<CudaStream>, b: &DeviceBlock<W>) -> BlockCols<W> {
        let (x, z, c) = download(s, b);
        let x: Vec<[u64; W]> = x.chunks(W).map(|w| w.try_into().unwrap()).collect();
        let z: Vec<[u64; W]> = z.chunks(W).map(|w| w.try_into().unwrap()).collect();
        let c: Vec<Complex64> = c.chunks(2).map(|p| Complex64::new(p[0], p[1])).collect();
        (b.offsets.clone(), x, z, c)
    }

    /// With the sender-side merge on, the device payloads carry per position what `export_layer` carries summed by key, no key twice across one partner's blocks, and no segment longer than its unmerged one.
    /// Returns `(rows sent, unmerged rows, fallback blocks)` per channel name, summed over ranks.
    fn premerge_matches_host_by_key<const W: usize>(
        input: &PauliSum<W>,
        rows: &PartitionRows<W>,
        arena_bytes: usize,
        nvrtc: &[String],
    ) -> Vec<(&'static str, u64, u64, (u32, u32))> {
        let mut totals = Vec::new();
        for (name, ch) in &fixture_channels::<W>() {
            let (mut sent, mut unmerged, mut fallbacks) = (0u64, 0u64, (0u32, 0u32));
            for rank in 0..rows.num_partitions() as u32 {
                let on = (true, arena_bytes, nvrtc);
                let Some(e) = export_rank(input, rows, ch.as_ref(), rank, on) else {
                    continue;
                };
                let (nb, what) = (
                    e.nb,
                    format!("W={W} {name} rank {rank} arena={arena_bytes}"),
                );
                for (q, w) in e.want.iter().enumerate() {
                    let (Some(w), Some(g)) = (w, &e.got[q]) else {
                        assert!(w.is_none() && e.got[q].is_none(), "{what}: presence to {q}");
                        continue;
                    };
                    assert_eq!(g.blocks.len(), w.blocks.len(), "{what}: blocks to {q}");
                    let g: Vec<BlockCols<W>> = g
                        .blocks
                        .iter()
                        .map(|b| {
                            assert_eq!(b.header.num_buckets as usize, nb);
                            assert_eq!(b.header.rows, b.offsets[nb]);
                            device_block_cols(&e.dev.stream, b)
                        })
                        .collect();
                    for (j, (gb, wb)) in g.iter().zip(&w.blocks).enumerate() {
                        assert_eq!(gb.0.len(), nb + 1, "{what}: offsets {j} to {q}");
                        for p in 0..nb {
                            assert!(
                                gb.0[p + 1] - gb.0[p] <= wb.offsets[p + 1] - wb.offsets[p],
                                "{what}: block {j} to {q} grew at position {p}"
                            );
                        }
                    }
                    let wcols: Vec<_> = w.blocks.iter().map(host_block_cols).collect();
                    let (want_sums, _) = position_sums(&wcols, nb);
                    let (got_sums, most) = position_sums(&g, nb);
                    assert_position_sums_close(&got_sums, &want_sums, 1e-12, &what);
                    if w.blocks.len() >= 2 && *name == "su4" {
                        assert_eq!(most, 1, "{what}: a key twice across the blocks to {q}");
                    }
                }
                let shipped: usize = e
                    .got
                    .iter()
                    .flatten()
                    .flat_map(|p| &p.blocks)
                    .map(DeviceBlock::rows)
                    .sum();
                assert_eq!(
                    e.got_counts.rows_sent.iter().sum::<u64>(),
                    shipped as u64,
                    "{what}: counted rows"
                );
                sent += shipped as u64;
                unmerged += e.want_rows.iter().sum::<u64>();
                fallbacks.0 += e.scratch.counters.fallback_hi;
                fallbacks.1 += e.scratch.counters.fallback_key;
            }
            totals.push((*name, sent, unmerged, fallbacks));
        }
        totals
    }

    fn premerge_fixture<const W: usize>(num_qubits: usize, seed: u64, pbits: u8) {
        let input = dense_on_01(&rand_sum::<W>(400, num_qubits, seed));
        let rows = PartitionRows::<W>::from_seed(num_qubits, pbits, seed ^ 0x77);
        // The SU(4) ships under half its unmerged rows; the Clifford and the rotation have nothing to merge.
        let check_shrink = |name: &str, sent: u64, unmerged: u64| {
            if name == "su4" {
                assert!(
                    sent * 2 < unmerged,
                    "P={}: su4 sent {sent} of {unmerged} rows",
                    1 << pbits
                );
            } else {
                assert_eq!(sent, unmerged, "{name}: nothing to merge");
            }
        };
        for arena in [DEFAULT_ARENA_BYTES, 1] {
            for (name, sent, unmerged, fallbacks) in
                premerge_matches_host_by_key(&input, &rows, arena, &[])
            {
                check_shrink(name, sent, unmerged);
                assert_eq!(
                    fallbacks,
                    (0, 0),
                    "{name}: a 64-bit fingerprint needs no fallback"
                );
            }
        }
        // Every merge block takes a fallback: the `g_hi32` passes with the low word cleared, the full-key sort with no fingerprint at all.
        for (opt, full_key) in [("-DFP_ZERO_LO", false), ("-DFP_BITS=0", true)] {
            let nvrtc = [opt.to_string()];
            for (name, sent, unmerged, fallbacks) in
                premerge_matches_host_by_key(&input, &rows, DEFAULT_ARENA_BYTES, &nvrtc)
            {
                check_shrink(name, sent, unmerged);
                if name == "su4" {
                    let taken = if full_key { fallbacks.1 } else { fallbacks.0 };
                    assert!(taken > 0, "{opt}: su4 merged without the expected fallback");
                }
            }
        }
    }

    #[test]
    fn premerged_blocks_match_export_layer_by_key_and_shrink() {
        crate::require_cuda!();
        premerge_fixture::<1>(12, 0xF7, 1);
        premerge_fixture::<1>(12, 0xF8, 2);
        premerge_fixture::<2>(100, 0xF9, 1);
        premerge_fixture::<2>(100, 0xFA, 2);
    }

    /// Two terms whose remote rows to one key cancel exactly: the merged export drops the key, the unmerged one ships both rows.
    /// Unit coefficients against equal-magnitude amplitudes keep every product exact whatever the kernel's FMA contraction.
    #[test]
    fn exactly_cancelling_remote_rows_are_not_shipped() {
        crate::require_cuda!();
        use crate::channel::prepared::Prepared;
        use crate::test_support::sqrt_swap_matrix;
        let nq = 8;
        let hash = crate::pauli_sum::hash::Gf2Hash::<1>::new(nq, 2, 0xCA);
        // The row reads x on qubit 0, so a delta flipping qubit 0's x-bit is remote.
        let rows = PartitionRows::<1>::from_rows(nq, vec![[0b1u64]], vec![[0u64]]);
        let ch = GeneralUnitary2Q::from_matrix(0, 1, sqrt_swap_matrix());
        let prep = ch.prepare(&hash, false).expect("prepared");
        let Prepared::Local(ptm) = &prep else {
            unreachable!()
        };
        let plan = PartitionPlan::new(&prep, &rows, 0);
        let d = ptm.deltas();
        let zero = Complex64::new(0.0, 0.0);
        let mut pick = None;
        'outer: for (i, a) in plan.remote.iter().enumerate() {
            for b in &plan.remote[i + 1..] {
                let (da, db) = (&d[a.entry], &d[b.entry]);
                for sa in (0..16usize).step_by(2) {
                    let sb = sa ^ (da.local_delta ^ db.local_delta) as usize;
                    let (aa, ab) = (da.amp[sa], db.amp[sb]);
                    if sb & 1 == 0 && aa != zero && (aa == ab || aa == -ab) {
                        pick = Some((a.entry, sa, b.entry, sb));
                        break 'outer;
                    }
                }
            }
        }
        let (ea, sa, eb, sb) =
            pick.expect("sqrt(SWAP) has two colliding remote entries of equal magnitude");
        let key = |s: usize| {
            let x = (s & 1) as u64 | (((s >> 2) & 1) as u64) << 1 | 1 << 5;
            let z = ((s >> 1) & 1) as u64 | (((s >> 3) & 1) as u64) << 1;
            crate::pauli_string::PauliString::<1> { x: [x], z: [z] }
        };
        let (aa, ab) = (d[ea].amp[sa], d[eb].amp[sb]);
        let cb = if aa == ab { -1.0 } else { 1.0 };
        let mut acc = crate::pauli_sum::accumulator::BuildAccumulator::<1>::new(nq);
        acc.add_term(key(sa), crate::phase::Phase::ONE, Complex64::new(1.0, 0.0));
        acc.add_term(key(sb), crate::phase::Phase::ONE, Complex64::new(cb, 0.0));
        let local = acc.finalize().with_hash(hash.clone());
        let (ka, kb) = (key(sa), key(sb));
        assert_eq!(rows.partition_of(&ka.x, &ka.z), 0);
        assert_eq!(rows.partition_of(&kb.x, &kb.z), 0);
        let (tx, tz) = (ka.x[0] ^ d[ea].mask_x[0], ka.z[0] ^ d[ea].mask_z[0]);
        assert_eq!(
            (tx, tz),
            (kb.x[0] ^ d[eb].mask_x[0], kb.z[0] ^ d[eb].mask_z[0])
        );
        let dev = GpuSum::from_host(&local, 0).expect("upload");
        let table = DevicePrepared::new(&prep, &hash, &dev.fp, &plan.remote);
        let shipped = |premerge: bool| {
            let options = GpuLayerOptions {
                premerge,
                ..GpuLayerOptions::default()
            };
            let mut scratch = LayerScratch::new(&dev, options).expect("scratch");
            scratch.upload_table(&dev, &table).expect("table");
            scratch.count_local(&dev, &table).expect("count");
            let (got, counts) =
                export_blocks_device(&dev, &table, &plan, &mut scratch, 2).expect("export");
            let keys: Vec<(u64, u64)> = got[1]
                .as_ref()
                .unwrap()
                .blocks
                .iter()
                .flat_map(|b| {
                    let (x, z, _) = download(&dev.stream, b);
                    x.into_iter().zip(z).collect::<Vec<_>>()
                })
                .collect();
            (keys, counts.rows_sent[1])
        };
        let (plain, plain_rows) = shipped(false);
        assert_eq!(plain.iter().filter(|k| **k == (tx, tz)).count(), 2);
        let (merged, merged_rows) = shipped(true);
        assert!(!merged.contains(&(tx, tz)), "the cancelled key travelled");
        assert!(
            merged_rows + 2 <= plain_rows,
            "{merged_rows} vs {plain_rows}"
        );
    }

    /// With the merge on, every rank of `input` under `rows` through an SU(4) falls back to the unmerged export, bitwise `export_layer`'s.
    fn assert_unmerged_fallback(input: &PauliSum<1>, rows: &PartitionRows<1>, nvrtc: &[String]) {
        let mut remote_layers = 0;
        for rank in 0..rows.num_partitions() as u32 {
            let on = (true, DEFAULT_ARENA_BYTES, nvrtc);
            let Some(e) = export_rank(input, rows, &su4::<1>(), rank, on) else {
                continue;
            };
            remote_layers += 1;
            assert_eq!(
                e.scratch.counters.rows_premerged, 0,
                "rank {rank}: no rows were premerged"
            );
            assert_payloads_eq(&e, &format!("rank {rank}"));
        }
        assert!(remote_layers > 0, "the fixture must export something");
    }

    /// A source bucket longer than [`MAX_BUCKET_LEN`] clears every merge group, the tag-overflow guard; one bucket of 12000 rows stays over the limit after the split halves it.
    #[test]
    fn oversize_source_bucket_falls_back_to_unmerged_export() {
        crate::require_cuda!();
        let input = rand_sum::<1>(12_000, 12, 0xB16).with_hash(Gf2Hash::<1>::new(12, 0, 0xB16));
        let rows = PartitionRows::<1>::from_seed(12, 1, 0xB16 ^ 0x77);
        assert!(input.filter_partition(&rows, 0).bucket(0).2.len() > MAX_BUCKET_LEN);
        assert_unmerged_fallback(&input, &rows, &[]);
    }

    /// A merge position whose records exceed the fused kernel's record cap: `premerge_partner` writes nothing and returns `None`.
    /// `-DTEST_SHARED_LIMIT` shrinks the cap to the smallest variant so a bucket under [`MAX_BUCKET_LEN`] exceeds it.
    #[test]
    fn oversize_merge_position_falls_back_to_unmerged_export() {
        crate::require_cuda!();
        let input = rand_sum::<1>(6000, 12, 0xCA9).with_hash(Gf2Hash::<1>::new(12, 0, 0xCA9));
        let rows = PartitionRows::<1>::from_seed(12, 1, 0xCA9 ^ 0x77);
        assert!(input.filter_partition(&rows, 0).bucket(0).2.len() <= MAX_BUCKET_LEN);
        assert_unmerged_fallback(&input, &rows, &["-DTEST_SHARED_LIMIT=17000".to_string()]);
    }

    /// A partner group whose `entries × buckets` exceeds [`SCAN_LIMIT`] is cleared before K3: `B_MAX_BITS` buckets times the SU(4)'s eight-entry group already does.
    #[test]
    fn oversize_partner_group_falls_back_to_unmerged_export() {
        crate::require_cuda!();
        let base = rand_sum::<1>(400, 32, 0xB5CA);
        let input = dense_on_01(&base).with_hash(Gf2Hash::<1>::new(32, B_MAX_BITS, 0xB5CA));
        assert!(input.num_buckets() * 8 > SCAN_LIMIT);
        assert_unmerged_fallback(
            &input,
            &PartitionRows::<1>::from_seed(32, 1, 0xB5CA ^ 0x77),
            &[],
        );
    }
}
