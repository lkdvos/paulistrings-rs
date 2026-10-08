//! The sender-side merge: one partner's exported rows merged by key before the exchange (ARCHITECTURE.md §Partitioning).

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};

use crate::engine::gpu::columns::{grow, DeviceColumns};
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::layer::{
    arena_batches, fused_variant, launch_fused, xfer, FusedOut, FusedRecv, FusedTable,
    LayerScratch, TableBuffers, Xfer,
};
use crate::engine::gpu::module::{thread_per, warp_per_bucket, MAX_BUCKET_LEN};
use crate::engine::gpu::payload::DeviceBlock;
use crate::engine::gpu::prepared::DevicePrepared;
use crate::engine::gpu::scan::exclusive_scan;
use crate::engine::gpu::sum::GpuSum;
use crate::engine::gpu::truncation::KeepProgram;
use crate::engine::partitioned::plan::PartitionPlan;

use super::DeviceExport;

/// Grow-only buffers of the sender-side merge: one partner's sub-table, its counts and CSR, and the split of its merged rows over the partner's blocks.
pub(super) struct PremergeScratch {
    table: TableBuffers,
    selected: CudaSlice<u32>,
    counts: CudaSlice<u32>,
    rows: CudaSlice<u32>,
    start: CudaSlice<u32>,
    out_len_pos: CudaSlice<u32>,
    lens: CudaSlice<u32>,
    split_offsets: CudaSlice<u32>,
    fallback: CudaSlice<u32>,
    totals: CudaSlice<u32>,
    start_host: Vec<u32>,
    split_offsets_host: Vec<u32>,
}

impl PremergeScratch {
    pub(super) fn new(stream: &Arc<CudaStream>, w: usize) -> Result<Self, GpuError> {
        Ok(Self {
            table: TableBuffers::new(stream, w)?,
            selected: stream.alloc_zeros(16)?,
            counts: stream.alloc_zeros(1)?,
            rows: stream.alloc_zeros(1)?,
            start: stream.alloc_zeros(2)?,
            out_len_pos: stream.alloc_zeros(1)?,
            lens: stream.alloc_zeros(1)?,
            split_offsets: stream.alloc_zeros(2)?,
            fallback: stream.alloc_zeros(2)?,
            totals: stream.alloc_zeros(2)?,
            start_host: Vec::new(),
            split_offsets_host: Vec::new(),
        })
    }
}

/// Elements `exclusive_scan` handles, so the split's `K × positions` offsets must fit.
pub(super) const SCAN_LIMIT: usize = 1 << 24;

/// Per partner, the remote entries the sender-side merge applies to, empty where it does not: at least two entries that can emit one key (`DevicePrepared::entries_can_collide`) and every source bucket inside the tag's offset field.
pub(super) fn premerge_groups<const W: usize>(
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

/// K3 over one partner's sub-table, each position's rows split back over its `blocks` within their unmerged counts so every segment still fits the receiver's tag; `None`, having written nothing, when a position exceeds the fused kernel's cap.
pub(super) fn premerge_partner<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    entries: &[usize],
    scratch: &mut LayerScratch<W>,
    blocks: &mut [DeviceBlock<W>],
) -> Result<Option<u64>, GpuError> {
    let stream: Arc<CudaStream> = sum.stream.clone();
    let kernels = sum.kernels.clone();
    let ordinal = sum.device();
    let b = sum.hash.num_buckets();
    let nk = entries.len();
    let sub = table.restrict(entries);
    let mut selected = [0u32; 16];
    for (j, &e) in entries.iter().enumerate() {
        selected[j] = e as u32;
    }
    let (b32, k32, e32) = (b as u32, nk as u32, table.entries as u32);
    let t0 = scratch.event(sum)?;
    {
        let premerge = &mut scratch.export.premerge;
        grow(&stream, &mut premerge.counts, b * nk, ordinal)?;
        grow(&stream, &mut premerge.rows, b, ordinal)?;
        grow(&stream, &mut premerge.start, b + 1, ordinal)?;
        grow(&stream, &mut premerge.out_len_pos, b, ordinal)?;
        #[cfg(feature = "phase-timing")]
        stream.synchronize()?;
        xfer(&mut scratch.xfer_ns, Xfer::H2d, || {
            premerge.table.upload(&stream, &sub)?;
            stream.memcpy_htod(&selected[..], &mut premerge.selected)?;
            Ok(())
        })?;
        // SAFETY: arguments match `k_premerge_counts` in export.cu; `scratch.counts` holds the layer's `b × E` counts and `premerge.counts` room for `b × K`.
        unsafe {
            stream
                .launch_builder(&kernels.premerge_counts)
                .arg(&scratch.counts)
                .arg(&e32)
                .arg(&premerge.selected)
                .arg(&k32)
                .arg(&b32)
                .arg(&mut premerge.counts)
                .launch(thread_per(b * nk, 256))?;
        }
        // SAFETY: arguments match `k_rows` in count.cu; every `rem` entry is `NO_REMOTE`, so `recv_off` is never read.
        unsafe {
            stream
                .launch_builder(&kernels.rows)
                .arg(&premerge.counts)
                .arg(&scratch.bucket_at)
                .arg(&premerge.table.bucket_delta)
                .arg(&premerge.table.rem)
                .arg(&scratch.export.recv_off)
                .arg(&mut premerge.rows)
                .arg(&b32)
                .arg(&k32)
                .launch(thread_per(b, 1024))?;
        }
        exclusive_scan(
            &stream,
            &kernels,
            &premerge.rows.slice(0..b),
            &mut premerge.start.slice_mut(0..b + 1),
            b,
            &mut scratch.scan,
            &mut premerge.totals,
        )?;
    }
    #[cfg(feature = "phase-timing")]
    stream.synchronize()?;
    let (total, most) = {
        let premerge = &mut scratch.export.premerge;
        premerge.start_host.resize(b + 1, 0);
        let (start, start_host, totals) =
            (&premerge.start, &mut premerge.start_host, &premerge.totals);
        xfer(&mut scratch.xfer_ns, Xfer::D2h, || {
            let v = stream.clone_dtoh(totals)?;
            stream.memcpy_dtoh(&start.slice(0..b + 1), &mut start_host[..])?;
            stream.synchronize()?;
            Ok((v[0] as u64, v[1] as usize))
        })?
    };
    let record_cap = kernels.layer_cap();
    if most > record_cap {
        scratch.lap(sum, t0, |m| &mut m.export)?;
        return Ok(None);
    }
    let (func, _, smem) = fused_variant(&kernels, W, most, sub.dense);
    let (batches, max_batch_rows) = arena_batches::<W>(
        &scratch.export.premerge.start_host,
        scratch.options.arena_bytes,
        record_cap,
        &[],
    );
    let mut arena = match scratch.arena.take() {
        Some(a) => a,
        None => DeviceColumns::<W>::with_capacity(&stream, ordinal, max_batch_rows, 1)?,
    };
    arena.len = 0;
    arena.buckets = 0;
    arena.reserve(max_batch_rows, 1)?;
    let mut running = vec![0u32; nk];
    for block in blocks.iter_mut() {
        block.offsets.clear();
        block.offsets.resize(b + 1, 0);
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
    let fallbacks = stream.clone_dtoh(&scratch.export.premerge.fallback)?;
    stream.synchronize()?;
    scratch.counters.fallback_hi += fallbacks[0];
    scratch.counters.fallback_key += fallbacks[1];
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
    let stream = &sum.stream;
    let kernels = &sum.kernels;
    let ordinal = sum.device();
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
        premerge,
        recv_off,
        recv_base,
        recv_x,
        recv_z,
        recv_c,
        recv_g,
        ..
    } = export;
    stream.memset_zeros(&mut premerge.fallback)?;
    for &(p0, p1) in batches {
        let n = p1 - p0;
        let (p0u, n32) = (p0 as u32, n as u32);
        let fused = FusedTable {
            counts: &premerge.counts,
            segment_start: &premerge.start,
            table: &premerge.table,
        };
        let recv = FusedRecv {
            offsets: recv_off,
            base: recv_base,
            x: recv_x,
            z: recv_z,
            c: recv_c,
            g: recv_g,
        };
        let written = FusedOut {
            arena: &mut *arena,
            out_len_pos: &mut premerge.out_len_pos,
            fallback: &mut premerge.fallback,
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
        grow(stream, &mut premerge.lens, nk * n, ordinal)?;
        grow(stream, &mut premerge.split_offsets, nk * n + 1, ordinal)?;
        // SAFETY: arguments match `k_premerge_split` in export.cu; `lens` holds `K × n` entries.
        unsafe {
            stream
                .launch_builder(&kernels.premerge_split)
                .arg(&premerge.out_len_pos)
                .arg(&premerge.counts)
                .arg(&*bucket_at)
                .arg(&premerge.table.bucket_delta)
                .arg(&k32)
                .arg(&p0u)
                .arg(&n32)
                .arg(&mut premerge.lens)
                .launch(thread_per(n, 256))?;
        }
        exclusive_scan(
            stream,
            kernels,
            &premerge.lens.slice(0..nk * n),
            &mut premerge.split_offsets.slice_mut(0..nk * n + 1),
            nk * n,
            scan,
            &mut premerge.totals,
        )?;
        #[cfg(feature = "phase-timing")]
        stream.synchronize()?;
        premerge.split_offsets_host.resize(nk * n + 1, 0);
        {
            let (split_offsets, split_offsets_host) =
                (&premerge.split_offsets, &mut premerge.split_offsets_host);
            xfer(xfer_ns, Xfer::D2h, || {
                stream.memcpy_dtoh(
                    &split_offsets.slice(0..nk * n + 1),
                    &mut split_offsets_host[..],
                )?;
                stream.synchronize()?;
                Ok(())
            })?;
        }
        let split = &premerge.split_offsets_host;
        let share = |j: usize| {
            (
                split[j * n] as usize,
                (split[(j + 1) * n] - split[j * n]) as usize,
            )
        };
        let set_offsets = |block_offsets: &mut [u32], j: usize, run: u32| {
            for i in 0..n {
                block_offsets[p0 + i] = run + split[j * n + i] - split[j * n];
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
                stream
                    .launch_builder(&kernels.premerge_copy)
                    .arg(&arena.x)
                    .arg(&arena.z)
                    .arg(&arena.coeff)
                    .arg(&premerge.start)
                    .arg(&premerge.lens)
                    .arg(&premerge.split_offsets)
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
        for (j, block) in blocks.iter_mut().enumerate() {
            let (_, rows) = share(j);
            set_offsets(&mut block.offsets, j, running[j]);
            if rows > 0 {
                let live = running[j] as usize;
                block.reserve_keep(stream, live, live + rows, ordinal)?;
                let DeviceBlock { x, z, c, .. } = block;
                copy(j, running[j], (x, z, c))?;
            }
            running[j] += rows as u32;
        }
    }
    Ok(())
}
