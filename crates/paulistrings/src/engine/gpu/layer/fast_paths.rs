//! The two layer paths that bypass the fused kernel: the key-preserving rescale and the permutation scatter.

use cudarc::driver::PushKernelArg;

use super::*;
use crate::engine::gpu::columns::grow;
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::length::Growth;
use crate::engine::gpu::module::{launch_grid, layer_threads, thread_per, warp_per_bucket};
use crate::engine::gpu::prepared::{DevicePrepared, NO_ENTRY, NO_REMOTE};
use crate::engine::gpu::scan::exclusive_scan;
use crate::engine::gpu::sum::GpuSum;
use crate::engine::gpu::truncation::KeepProgram;
use crate::engine::timing::timed;

/// `k_rescale`: the key-preserving fast path, into the spare columns at the input's offsets.
pub(super) fn rescale_device<const W: usize, C: Coefficient>(
    sum: &mut GpuSum<W, C>,
    table: &DevicePrepared<W, C>,
    keep: &KeepProgram,
    scratch: &mut LayerScratch<W, C>,
) -> Result<(), GpuError> {
    let stream = sum.stream.clone();
    let kernels = sum.kernels.clone();
    let ordinal = sum.device();
    let b = sum.hash.num_buckets();
    let extent = scratch.extent.max(sum.len());
    let mut out = sum.take_spare(extent, b)?;
    grow(&stream, &mut scratch.destination_offsets, b + 1, ordinal)?;
    let amplitude = &mut scratch.table.amplitude;
    timed_transfer(&stream, &mut scratch.stats, Transfer::H2d, || {
        stream.memcpy_htod(bytemuck::cast_slice::<C, u8>(&table.amplitude), amplitude)?;
        Ok(())
    })?;
    let b32 = b as u32;
    timed!(let t0 = scratch.event(sum)?);
    // SAFETY: arguments match `k_rescale` in rescale.cu; `out` has room for the input's extent.
    unsafe {
        stream
            .launch_builder(&kernels.rescale)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.fingerprint)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&b32)
            .arg(&table.support_size)
            .arg(&table.qubit_0)
            .arg(&table.qubit_1)
            .arg(&scratch.table.amplitude)
            .arg(keep)
            .arg(&mut out.x)
            .arg(&mut out.z)
            .arg(&mut out.coeff)
            .arg(&mut out.fingerprint)
            .arg(&mut out.start)
            .arg(&mut out.lengths)
            .launch(warp_per_bucket(b))?;
    }
    exclusive_scan(
        &stream,
        &kernels,
        &out.lengths.slice(0..b),
        &mut scratch.destination_offsets.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut scratch.totals,
    )?;
    timed!(scratch.lap(sum, t0, |stats| &mut stats.rescale_ns)?);
    sum.swap_in(out, sum.len());
    sum.record_step(Growth::Shrink { cap: usize::MAX }, Some(&scratch.totals))?;
    scratch.record.rescaled = true;
    sum.debug_check();
    Ok(())
}

/// `k_permute_count`'s and `k_permute_scatter`'s block width: the average source bucket rounded up to a power of two, between two warps and the fused layer's width for `w`.
pub(in crate::engine::gpu) fn permute_threads(len: usize, buckets: usize, w: usize) -> u32 {
    let avg = len.div_ceil(buckets.max(1));
    (avg.max(64).next_power_of_two() as u32).min(layer_threads(w))
}

/// `k_permute_*`: the permutation path, into the spare columns as a tight CSR in bucket order.
/// With received entries, the local rows of those entries have gone to their partners, and the rows received in `scratch.export`, one chunk over every position, follow each bucket's local rows.
pub(super) fn permute_device<const W: usize, C: Coefficient>(
    sum: &mut GpuSum<W, C>,
    table: &DevicePrepared<W, C>,
    keep: &KeepProgram,
    scratch: &mut LayerScratch<W, C>,
) -> Result<(), GpuError> {
    let stream = sum.stream.clone();
    let kernels = sum.kernels.clone();
    let ordinal = sum.device();
    let b = sum.hash.num_buckets();
    let received = table.received_entries;
    let n_in = sum.len();
    let rows_in = n_in + scratch.export.received_rows;
    let (b32, e32, k32) = (b as u32, table.entries as u32, received as u32);
    grow(&stream, &mut scratch.counts, b * table.entries, ordinal)?;
    grow(&stream, &mut scratch.rows, b, ordinal)?;
    let mut out = sum.take_spare(rows_in, b)?;
    let entry_of: Vec<u32> = table
        .entry_of
        .iter()
        .map(|&e| match e {
            NO_ENTRY => NO_ENTRY,
            e if table.received_slot[e as usize] != NO_REMOTE => NO_ENTRY,
            e => e,
        })
        .collect();
    let (buffers, entry_of_device) = (&mut scratch.table, &mut scratch.entry_of);
    timed_transfer(&stream, &mut scratch.stats, Transfer::H2d, || {
        stream
            .memcpy_htod(bytemuck::cast_slice::<C, u8>(&table.amplitude), &mut buffers.amplitude)?;
        stream.memcpy_htod(&table.mask, &mut buffers.mask)?;
        stream.memcpy_htod(&table.bucket_delta, &mut buffers.bucket_delta)?;
        stream.memcpy_htod(&table.fingerprint_delta, &mut buffers.fingerprint_delta)?;
        stream.memcpy_htod(&entry_of[..], entry_of_device)?;
        Ok(())
    })?;
    let block_per_bucket = launch_grid(b32, permute_threads(n_in, b, W), 0);
    timed!(let t0 = scratch.event(sum)?);
    // SAFETY: arguments match `k_permute_count` in permute.cu; `counts` holds `b * e` entries.
    unsafe {
        stream
            .launch_builder(&kernels.permute_count)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&table.mode)
            .arg(&e32)
            .arg(&table.support_size)
            .arg(&table.qubit_0)
            .arg(&table.qubit_1)
            .arg(&table.rotation_cos)
            .arg(&table.rotation_sin)
            .arg(&scratch.table.amplitude)
            .arg(&scratch.table.mask)
            .arg(&scratch.table.nonzero_patterns)
            .arg(&scratch.entry_of)
            .arg(keep)
            .arg(&mut scratch.counts)
            .launch(block_per_bucket)?;
    }
    timed!(scratch.lap(sum, t0, |stats| &mut stats.gather_ns)?);
    timed!(let t1 = scratch.event(sum)?);
    let local_lengths = if received == 0 { &mut out.lengths } else { &mut scratch.rows };
    // SAFETY: arguments match `k_permute_lengths`; `out.lengths` and `scratch.rows` hold `b` entries.
    unsafe {
        stream
            .launch_builder(&kernels.permute_lengths)
            .arg(&scratch.counts)
            .arg(&scratch.table.bucket_delta)
            .arg(&b32)
            .arg(&e32)
            .arg(local_lengths)
            .launch(thread_per(b, 256))?;
    }
    if received > 0 {
        let export = &scratch.export;
        // SAFETY: arguments match `k_permute_received_lengths`; `received_offsets` holds `K × (b + 1)` offsets into the received columns from `received_base`, and `bucket_at` maps every position of the current count.
        unsafe {
            stream
                .launch_builder(&kernels.permute_received_lengths)
                .arg(&export.received_offsets)
                .arg(&export.received_base)
                .arg(&k32)
                .arg(&b32)
                .arg(&export.received_x)
                .arg(&export.received_z)
                .arg(&export.received_coefficient)
                .arg(keep)
                .arg(&scratch.bucket_at)
                .arg(&scratch.rows)
                .arg(&mut out.lengths)
                .launch(warp_per_bucket(b))?;
        }
    }
    exclusive_scan(
        &stream,
        &kernels,
        &out.lengths.slice(0..b),
        &mut out.start.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut scratch.totals,
    )?;
    timed!(scratch.lap(sum, t1, |stats| &mut stats.gather_ns)?);
    timed!(let t2 = scratch.event(sum)?);
    // SAFETY: arguments match `k_permute_scatter`; `out` has room for every input row, the scan's total at most.
    unsafe {
        stream
            .launch_builder(&kernels.permute_scatter)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.fingerprint)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&table.mode)
            .arg(&e32)
            .arg(&table.support_size)
            .arg(&table.qubit_0)
            .arg(&table.qubit_1)
            .arg(&table.rotation_cos)
            .arg(&table.rotation_sin)
            .arg(&scratch.table.amplitude)
            .arg(&scratch.table.mask)
            .arg(&scratch.table.nonzero_patterns)
            .arg(&scratch.entry_of)
            .arg(&scratch.table.bucket_delta)
            .arg(&scratch.table.fingerprint_delta)
            .arg(&scratch.counts)
            .arg(&out.start)
            .arg(keep)
            .arg(&mut out.x)
            .arg(&mut out.z)
            .arg(&mut out.coeff)
            .arg(&mut out.fingerprint)
            .launch(block_per_bucket)?;
    }
    if received > 0 {
        let export = &scratch.export;
        // SAFETY: arguments match `k_permute_received_scatter`; every surviving received row lands after its bucket's local rows, inside `out.start`'s segment, which counted it.
        unsafe {
            stream
                .launch_builder(&kernels.permute_received_scatter)
                .arg(&export.received_offsets)
                .arg(&export.received_base)
                .arg(&k32)
                .arg(&b32)
                .arg(&export.received_x)
                .arg(&export.received_z)
                .arg(&export.received_coefficient)
                .arg(&export.received_fingerprint)
                .arg(keep)
                .arg(&scratch.bucket_at)
                .arg(&scratch.rows)
                .arg(&out.start)
                .arg(&mut out.x)
                .arg(&mut out.z)
                .arg(&mut out.coeff)
                .arg(&mut out.fingerprint)
                .launch(warp_per_bucket(b))?;
        }
    }
    timed!(scratch.lap(sum, t2, |stats| &mut stats.merge_ns)?);
    sum.swap_in(out, rows_in);
    sum.record_step(Growth::Regroup, Some(&scratch.totals))?;
    scratch.extent = sum.len();
    scratch.record.records = rows_in as u64;
    scratch.record.permuted = true;
    sum.debug_check();
    Ok(())
}
