//! The receive side of a device exchange: staging, sizing and moving the received rows chunk by chunk.

use crate::engine::coset::Gf2Span;
use crate::engine::gpu::columns::grow;
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::layer::{timed_transfer, LayerScratch, Transfer, TransferNs};
use crate::engine::gpu::sum::{launch_fingerprint, GpuSum};
use crate::engine::gpu::wire::{ScheduledOp, Skeleton, WireColumn, WireGroup, WireOpKind};
use crate::engine::partitioned::transport::ChunkMap;

use super::{recv_row_bytes, DeviceExport, PendingRecv, MAX_RECV_SEGMENT};

/// The receive layout from the skeletons in plan order, checked against the segment cap before any row moves; sizes `received_*` for this rank's own chunk count, which it returns as `log2`.
pub(super) fn stage_receive<const W: usize>(
    sum: &GpuSum<W>,
    export: &mut DeviceExport<W>,
    blocks: &[&Skeleton],
    cap_rows: usize,
    transfer_ns: &mut TransferNs,
) -> Result<u8, GpuError> {
    let b = sum.hash.num_buckets();
    lay_out(
        export,
        b,
        blocks.iter().map(|block| {
            (
                block.header.num_buckets,
                &block.offsets[..],
                block.header.rows as usize,
            )
        }),
    );
    if export.received_max_segment > MAX_RECV_SEGMENT {
        return Err(GpuError::Unsupported(
            "a fused-layer block or a received segment exceeds the record cap at the agreed bucket count",
        ));
    }
    #[cfg(any(test, feature = "test-utils"))]
    if std::mem::take(&mut export.fail_recv_growth) {
        return Err(GpuError::OutOfMemory {
            device: sum.device(),
            bytes: (export.received_rows * recv_row_bytes::<W>()) as u64,
        });
    }
    let (log2, most) = recv_chunks(&export.offsets_host, b, cap_rows);
    size_receive(sum, export, most, transfer_ns)?;
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
    let n = export.offsets_host.len();
    export
        .stream
        .memset_zeros(&mut export.received_offsets.slice_mut(0..n))?;
    export.received_rows = 0;
    export.received_max_segment = 0;
    Ok(0)
}

/// Received rows a chunk may hold under a cap of `bytes`.
pub(super) fn recv_cap_rows<const W: usize>(bytes: usize) -> usize {
    (bytes / recv_row_bytes::<W>()).max(1)
}

/// `off_host`, `received_max_segment` and `received_rows` from received blocks in plan order, each `(positions, CSR offsets, rows)`.
fn lay_out<'a, const W: usize>(
    export: &mut DeviceExport<W>,
    b: usize,
    blocks: impl Iterator<Item = (u32, &'a [u32], usize)>,
) {
    let mut total = 0usize;
    let mut max_segment = 0usize;
    export.offsets_host.clear();
    for (num_buckets, offsets, rows) in blocks {
        assert_eq!(
            num_buckets as usize, b,
            "a partner sent a block indexed by {num_buckets} buckets where this partition has {b}"
        );
        export.offsets_host.extend_from_slice(&offsets[..b + 1]);
        max_segment = offsets[..b + 1]
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0)
            .max(max_segment);
        total += rows;
    }
    export.received_max_segment = max_segment;
    export.received_rows = total;
}

/// Received rows over every block in positions `lo..hi` of the concatenated `K × (b + 1)` offsets.
pub(super) fn rows_between(offsets: &[u32], b: usize, lo: usize, hi: usize) -> usize {
    offsets
        .chunks_exact(b + 1)
        .map(|block| (block[hi] - block[lo]) as usize)
        .sum()
}

/// The largest chunk's received rows when `b` positions are cut into `2^log2` equal chunks.
pub(super) fn chunk_max(offsets: &[u32], b: usize, log2: u8) -> usize {
    let step = b >> log2;
    (0..1usize << log2)
        .map(|c| rows_between(offsets, b, c * step, (c + 1) * step))
        .max()
        .unwrap_or(0)
}

/// The fewest power-of-two chunks, at most one per position, whose received rows each fit `cap_rows`, as `(log2 chunks, rows of the largest)`.
// A power of two because such a cut refines every coarser one, so the group's agreed maximum never grows anyone's chunk.
pub(super) fn recv_chunks(offsets: &[u32], b: usize, cap_rows: usize) -> (u8, usize) {
    let mut log2 = 0u8;
    loop {
        let most = chunk_max(offsets, b, log2);
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

/// Room for `rows` received rows in `received_*`, and the offsets in `received_offsets`.
fn size_receive<const W: usize>(
    sum: &GpuSum<W>,
    export: &mut DeviceExport<W>,
    rows: usize,
    transfer_ns: &mut TransferNs,
) -> Result<(), GpuError> {
    let stream = &sum.stream;
    let ordinal = sum.device();
    let DeviceExport {
        received_offsets,
        received_x,
        received_z,
        received_coefficient,
        received_fingerprint,
        offsets_host,
        ..
    } = export;
    grow(stream, received_offsets, offsets_host.len().max(2), ordinal)?;
    grow(stream, received_x, rows.max(1) * W, ordinal)?;
    grow(stream, received_z, rows.max(1) * W, ordinal)?;
    grow(stream, received_coefficient, 2 * rows.max(1), ordinal)?;
    grow(stream, received_fingerprint, rows.max(1), ordinal)?;
    timed_transfer(stream, transfer_ns, Transfer::H2d, || {
        stream.memcpy_htod(
            &offsets_host[..],
            &mut received_offsets.slice_mut(0..offsets_host.len()),
        )?;
        Ok(())
    })
}

/// Positions `lo..hi` of every received block laid end to end from row 0, returning their rows, with `base[k]` the wrapping `start - off[k][lo]` the kernel's own `u32` sum `base[k] + off[k][p]` undoes.
pub(super) fn chunk_layout(
    offsets: &[u32],
    b: usize,
    lo: usize,
    hi: usize,
    base: &mut Vec<u32>,
) -> usize {
    base.clear();
    base.resize(16, 0);
    let mut at = 0usize;
    for (k, block) in offsets.chunks_exact(b + 1).enumerate() {
        base[k] = (at as u32).wrapping_sub(block[lo]);
        at += (block[hi] - block[lo]) as usize;
    }
    at
}

/// Move chunk `c` of the pending receive into `received_*`, fingerprinted, with its bases in `received_base`; called in chunk order before the chunk's first batch.
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
    let r = move_chunk(sum, export, &mut pending, c, true, &mut scratch.transfer_ns);
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

/// Chunk `c` of `pending` into `received_*` through one wire group; with `keep` its rows are fingerprinted and its bases uploaded, without it the transfer only completes, as a failed layer's must.
fn move_chunk<const W: usize>(
    sum: &GpuSum<W>,
    export: &mut DeviceExport<W>,
    pending: &mut PendingRecv<W>,
    c: usize,
    keep: bool,
    transfer_ns: &mut TransferNs,
) -> Result<usize, GpuError> {
    let b = sum.hash.num_buckets();
    let stream = &sum.stream;
    let (lo, hi) = (
        pending.map.bound(c) as usize,
        pending.map.bound(c + 1) as usize,
    );
    let n = chunk_layout(&export.offsets_host, b, lo, hi, &mut export.base_host);
    let wire = export.wire.clone().ok_or(GpuError::Unsupported(
        "a remote layer on a partition without a device wire",
    ))?;
    let DeviceExport {
        received_base,
        received_x,
        received_z,
        received_coefficient,
        received_fingerprint,
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
        let (q, j) = own[op.delta];
        let block = &send[q]
            .as_ref()
            .expect("a payload for every partner")
            .blocks[j];
        let (r0, r1) = op.rows;
        let e = op.column.elems_per_row::<W>();
        match op.column {
            WireColumn::X => group.send(block.x.slice(r0 * e..r1 * e), op.peer, stream),
            WireColumn::Z => group.send(block.z.slice(r0 * e..r1 * e), op.peer, stream),
            WireColumn::Coefficient => {
                group.send(block.coefficient.slice(r0 * e..r1 * e), op.peer, stream)
            }
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
    group.recv_parts(
        received_x.slice_mut(0..n * W),
        &recv_parts(WireColumn::X),
        stream,
    );
    group.recv_parts(
        received_z.slice_mut(0..n * W),
        &recv_parts(WireColumn::Z),
        stream,
    );
    group.recv_parts(
        received_coefficient.slice_mut(0..2 * n),
        &recv_parts(WireColumn::Coefficient),
        stream,
    );
    if let Err(e) = group.post(&*wire).and_then(|()| wire.wait(stream)) {
        pending.failed = true;
        return Err(e);
    }
    pending.next = c + 1;
    if !keep {
        return Ok(n);
    }
    launch_fingerprint(
        stream,
        &sum.kernels,
        &sum.fingerprint_rows,
        (&*received_x, &*received_z),
        received_fingerprint,
        n,
    )?;
    timed_transfer(stream, transfer_ns, Transfer::H2d, || {
        stream.memcpy_htod(&base_host[..], received_base)?;
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
        if let Err(e) = move_chunk(
            sum,
            export,
            &mut pending,
            c,
            false,
            &mut scratch.transfer_ns,
        ) {
            drained = Err(e);
            break;
        }
    }
    let synced = sum.stream.synchronize().map_err(GpuError::from);
    // A failed group's peers may still read these, but a failed partition never exports again, so they are only freed with it.
    export.sends.extend(pending.send.into_iter().flatten());
    drained.and(synced)
}
