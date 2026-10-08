//! The sender-side merge: one partner's exported rows merged by key before the exchange (ARCHITECTURE.md §Partitioning).

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};

use crate::engine::gpu::columns::{grow, DeviceColumns};
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::layer::{
    arena_batches, fused_variant, launch_fused, xfer, FusedOut, FusedRecv, FusedTable,
    LayerScratch, TableBufs, Xfer,
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
    pub(super) fn new(s: &Arc<CudaStream>, w: usize) -> Result<Self, GpuError> {
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

/// The sender-side merge of one partner's blocks (ARCHITECTURE.md §Partitioning): K3 over the partner's sub-table under the keep-everything program, each position's rows split back over the blocks within their unmerged counts so every segment still fits the receiver's tag.
/// `blocks` are the partner's in its remote-delta order; returns the rows written, or `None` having written nothing when a position's records exceed the fused kernel's cap.
pub(super) fn premerge_partner<const W: usize>(
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
