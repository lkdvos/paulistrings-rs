//! The two layer paths that bypass the fused kernel: the key-preserving rescale (K5) and the permutation scatter (K12–K14).

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::*;
use crate::engine::gpu::columns::{grow, DeviceColumns};
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::module::{layer_threads, thread_per, warp_per_bucket};
use crate::engine::gpu::prepared::DevicePrepared;
use crate::engine::gpu::scan::exclusive_scan;
use crate::engine::gpu::sum::GpuSum;
use crate::engine::gpu::truncation::KeepProgram;

/// K5: the key-preserving fast path, into the spare columns at the input's offsets.
pub(super) fn rescale_device<const W: usize>(
    sum: &mut GpuSum<W>,
    table: &DevicePrepared<W>,
    keep: &KeepProgram,
    scratch: &mut LayerScratch<W>,
) -> Result<(), GpuError> {
    let s = sum.stream.clone();
    let k = sum.kernels.clone();
    let o = sum.device();
    let b = sum.hash.num_buckets();
    let extent = scratch.extent.max(sum.len());
    let mut out = match sum.spare.take() {
        Some(spare) => spare,
        None => DeviceColumns::<W>::with_capacity(&s, o, extent, b)?,
    };
    out.len = 0;
    out.buckets = 0;
    out.reserve(extent, b)?;
    grow(&s, &mut scratch.dst_off, b + 1, o)?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    let amp = &mut scratch.table.amp;
    xfer(&mut scratch.xfer_ns, Xfer::H2d, || {
        s.memcpy_htod(&table.amp, amp)?;
        Ok(())
    })?;
    let b32 = b as u32;
    let t0 = scratch.event(sum)?;
    // SAFETY: arguments match `k_rescale` in rescale.cu; `out` has room for the input's extent.
    unsafe {
        s.launch_builder(&k.rescale)
            .arg(&sum.cols.x)
            .arg(&sum.cols.z)
            .arg(&sum.cols.coeff)
            .arg(&sum.cols.g)
            .arg(&sum.cols.start)
            .arg(&sum.cols.lens)
            .arg(&b32)
            .arg(&table.kq)
            .arg(&table.q0)
            .arg(&table.q1)
            .arg(&scratch.table.amp)
            .arg(keep)
            .arg(&mut out.x)
            .arg(&mut out.z)
            .arg(&mut out.coeff)
            .arg(&mut out.g)
            .arg(&mut out.start)
            .arg(&mut out.lens)
            .launch(warp_per_bucket(b))?;
    }
    exclusive_scan(
        &s,
        &k,
        &out.lens.slice(0..b),
        &mut scratch.dst_off.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut scratch.tot_a,
    )?;
    scratch.lap(sum, t0, |m| &mut m.rescale)?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    let tot = &scratch.tot_a;
    let total = xfer(&mut scratch.xfer_ns, Xfer::D2h, || {
        let v = s.clone_dtoh(tot)?;
        s.synchronize()?;
        Ok(v[0])
    })?;
    out.len = total as usize;
    out.buckets = b;
    sum.spare = Some(std::mem::replace(&mut sum.cols, out));
    scratch.counters.rescaled = true;
    sum.debug_check();
    Ok(())
}

/// K12's and K14's block width: the average source bucket rounded up to a power of two, between two warps and the fused layer's width for `w`.
pub(in crate::engine::gpu) fn perm_threads(len: usize, buckets: usize, w: usize) -> u32 {
    let avg = len.div_ceil(buckets.max(1));
    (avg.max(64).next_power_of_two() as u32).min(layer_threads(w))
}

/// K12–K14: the permutation path, into the spare columns as a tight CSR in bucket order.
pub(super) fn permute_device<const W: usize>(
    sum: &mut GpuSum<W>,
    table: &DevicePrepared<W>,
    keep: &KeepProgram,
    scratch: &mut LayerScratch<W>,
) -> Result<(), GpuError> {
    let s = sum.stream.clone();
    let k = sum.kernels.clone();
    let o = sum.device();
    let b = sum.hash.num_buckets();
    let n_in = sum.len();
    let (b32, e32) = (b as u32, table.entries as u32);
    grow(&s, &mut scratch.cnt, b * table.entries, o)?;
    let mut out = match sum.spare.take() {
        Some(spare) => spare,
        None => DeviceColumns::<W>::with_capacity(&s, o, n_in, b)?,
    };
    out.len = 0;
    out.buckets = 0;
    out.reserve(n_in, b)?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    let (bufs, entry_of) = (&mut scratch.table, &mut scratch.entry_of);
    xfer(&mut scratch.xfer_ns, Xfer::H2d, || {
        s.memcpy_htod(&table.amp, &mut bufs.amp)?;
        s.memcpy_htod(&table.mask, &mut bufs.mask)?;
        s.memcpy_htod(&table.bucket_delta, &mut bufs.bd)?;
        s.memcpy_htod(&table.gm, &mut bufs.gm)?;
        s.memcpy_htod(&table.entry_of, entry_of)?;
        Ok(())
    })?;
    let block_per_bucket = LaunchConfig {
        grid_dim: (b32, 1, 1),
        block_dim: (perm_threads(n_in, b, W), 1, 1),
        shared_mem_bytes: 0,
    };
    let t0 = scratch.event(sum)?;
    // SAFETY: arguments match `k_perm_count` in permute.cu; `cnt` holds `b * e` entries.
    unsafe {
        s.launch_builder(&k.perm_count)
            .arg(&sum.cols.x)
            .arg(&sum.cols.z)
            .arg(&sum.cols.coeff)
            .arg(&sum.cols.start)
            .arg(&sum.cols.lens)
            .arg(&table.mode)
            .arg(&e32)
            .arg(&table.kq)
            .arg(&table.q0)
            .arg(&table.q1)
            .arg(&table.rot_cos)
            .arg(&table.rot_sin)
            .arg(&scratch.table.amp)
            .arg(&scratch.table.mask)
            .arg(&scratch.table.nz)
            .arg(&scratch.entry_of)
            .arg(keep)
            .arg(&mut scratch.cnt)
            .launch(block_per_bucket)?;
    }
    scratch.lap(sum, t0, |m| &mut m.count)?;
    let t1 = scratch.event(sum)?;
    // SAFETY: arguments match `k_perm_lens`; `out.lens` holds `b` entries.
    unsafe {
        s.launch_builder(&k.perm_lens)
            .arg(&scratch.cnt)
            .arg(&scratch.table.bd)
            .arg(&b32)
            .arg(&e32)
            .arg(&mut out.lens)
            .launch(thread_per(b, 256))?;
    }
    exclusive_scan(
        &s,
        &k,
        &out.lens.slice(0..b),
        &mut out.start.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut scratch.tot_a,
    )?;
    scratch.lap(sum, t1, |m| &mut m.sizes)?;
    let t2 = scratch.event(sum)?;
    // SAFETY: arguments match `k_perm_scatter`; `out` has room for every input row, the scan's total at most.
    unsafe {
        s.launch_builder(&k.perm_scatter)
            .arg(&sum.cols.x)
            .arg(&sum.cols.z)
            .arg(&sum.cols.coeff)
            .arg(&sum.cols.g)
            .arg(&sum.cols.start)
            .arg(&sum.cols.lens)
            .arg(&table.mode)
            .arg(&e32)
            .arg(&table.kq)
            .arg(&table.q0)
            .arg(&table.q1)
            .arg(&table.rot_cos)
            .arg(&table.rot_sin)
            .arg(&scratch.table.amp)
            .arg(&scratch.table.mask)
            .arg(&scratch.table.nz)
            .arg(&scratch.entry_of)
            .arg(&scratch.table.bd)
            .arg(&scratch.table.gm)
            .arg(&scratch.cnt)
            .arg(&out.start)
            .arg(keep)
            .arg(&mut out.x)
            .arg(&mut out.z)
            .arg(&mut out.coeff)
            .arg(&mut out.g)
            .launch(block_per_bucket)?;
    }
    scratch.lap(sum, t2, |m| &mut m.permute)?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    let tot = &scratch.tot_a;
    let total = xfer(&mut scratch.xfer_ns, Xfer::D2h, || {
        let v = s.clone_dtoh(tot)?;
        s.synchronize()?;
        Ok(v[0])
    })?;
    out.len = total as usize;
    out.buckets = b;
    sum.spare = Some(std::mem::replace(&mut sum.cols, out));
    scratch.extent = sum.len();
    scratch.counters.bits = sum.hash.bits();
    scratch.counters.records = n_in as u64;
    scratch.counters.permuted = true;
    sum.debug_check();
    Ok(())
}
