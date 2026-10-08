//! The receive side of a device exchange: staging, sizing and moving the received rows chunk by chunk.

use crate::engine::coset::Gf2Span;
use crate::engine::gpu::columns::grow;
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::layer::{xfer, LayerScratch, Xfer, XferNs};
use crate::engine::gpu::sum::{launch_fingerprint, GpuSum};
use crate::engine::gpu::wire::{ScheduledOp, Skeleton, WireColumn, WireGroup, WireOpKind};
use crate::engine::partitioned::transport::ChunkMap;

use super::{recv_row_bytes, DeviceExport, PendingRecv, MAX_RECV_SEGMENT};

/// The receive layout from the skeletons in plan order, checked against the segment cap before any row moves; sizes `recv_*` for this rank's own chunk count, which it returns as `log2`.
pub(super) fn stage_receive<const W: usize>(
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
pub(super) fn wire_ready<const W: usize>(export: &DeviceExport<W>) -> Result<(), GpuError> {
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
pub(super) fn discard_received<const W: usize>(
    export: &mut DeviceExport<W>,
) -> Result<u64, GpuError> {
    let n = export.off_host.len();
    export
        .stream
        .memset_zeros(&mut export.recv_off.slice_mut(0..n))?;
    export.recv_rows = 0;
    export.recv_max_segment = 0;
    Ok(0)
}

/// Received rows a chunk may hold under a cap of `bytes`.
pub(super) fn recv_cap_rows<const W: usize>(bytes: usize) -> usize {
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
pub(super) fn rows_between(off: &[u32], b: usize, lo: usize, hi: usize) -> usize {
    off.chunks_exact(b + 1)
        .map(|o| (o[hi] - o[lo]) as usize)
        .sum()
}

/// The largest chunk's received rows when `b` positions are cut into `2^log2` equal chunks.
pub(super) fn chunk_max(off: &[u32], b: usize, log2: u8) -> usize {
    let step = b >> log2;
    (0..1usize << log2)
        .map(|c| rows_between(off, b, c * step, (c + 1) * step))
        .max()
        .unwrap_or(0)
}

/// The fewest chunks, a power of two up to one per position, whose received rows each fit `cap_rows`, as `(log2 chunks, rows of the largest)`.
/// A power of two because such a cut refines every coarser one, so a group agreeing on the largest count its members asked for never grows anyone's chunk.
pub(super) fn recv_chunks(off: &[u32], b: usize, cap_rows: usize) -> (u8, usize) {
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
pub(super) fn position_chunks(bits: u8, b: usize, log2: u8) -> ChunkMap {
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
pub(super) fn chunk_layout(
    off: &[u32],
    b: usize,
    lo: usize,
    hi: usize,
    base: &mut Vec<u32>,
) -> usize {
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
