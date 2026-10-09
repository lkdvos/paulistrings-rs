//! Launching the fused layer kernel (K3): position batches that fit the arena, the kernel variant, and the launch itself.

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use super::TableBuffers;
use crate::engine::gpu::columns::DeviceColumns;
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::module::{layer_shared_bytes, layer_threads, KernelSet};
use crate::engine::gpu::prepared::DevicePrepared;
use crate::engine::gpu::sum::GpuSum;
use crate::engine::gpu::truncation::KeepProgram;

/// Contiguous position ranges whose pre-dedup rows (`seg`, `b + 1` CSR offsets) fit an arena of `arena_bytes`, each starting a new range at every position of the ascending `starts`, and the largest range's rows.
pub(in crate::engine::gpu) fn arena_batches<const W: usize>(
    segment_start: &[u32],
    arena_bytes: usize,
    record_cap: usize,
    starts: &[usize],
) -> (Vec<(usize, usize)>, usize) {
    let b = segment_start.len() - 1;
    let cap_rows = (arena_bytes / DeviceColumns::<W>::BYTES_PER_TERM).max(record_cap);
    let mut batches: Vec<(usize, usize)> = Vec::new();
    let mut starts = starts.iter().copied().peekable();
    let mut p0 = 0usize;
    while p0 < b {
        while starts.next_if(|&c| c <= p0).is_some() {}
        let end = starts.peek().copied().unwrap_or(b).min(b);
        let mut p1 = p0 + 1;
        while p1 < end && (segment_start[p1 + 1] - segment_start[p0]) as usize <= cap_rows {
            p1 += 1;
        }
        batches.push((p0, p1));
        p0 = p1;
    }
    let max_rows = batches
        .iter()
        .map(|&(a, c)| (segment_start[c] - segment_start[a]) as usize)
        .max()
        .unwrap_or(0);
    (batches, max_rows)
}

/// The fused-layer variant for blocks of up to `records_max` records: the kernel, its record capacity and its dynamic shared bytes.
pub(in crate::engine::gpu) fn fused_variant(
    kernels: &KernelSet,
    w: usize,
    records_max: usize,
    dense: bool,
) -> (&CudaFunction, usize, u32) {
    let threads = layer_threads(w) as usize;
    let record_capacity = records_max.max(threads).next_power_of_two();
    let variant = &kernels.layer[(record_capacity / threads).trailing_zeros() as usize];
    debug_assert_eq!(variant.items * threads, record_capacity);
    let func = if dense {
        &variant.segscan
    } else {
        &variant.serial
    };
    (
        func,
        record_capacity,
        layer_shared_bytes(record_capacity, w),
    )
}

/// The table-side buffers one fused-layer launch reads: the counts and segment starts sized for `table`'s upload.
pub(in crate::engine::gpu) struct FusedTable<'a> {
    pub(in crate::engine::gpu) counts: &'a CudaSlice<u32>,
    pub(in crate::engine::gpu) segment_start: &'a CudaSlice<u32>,
    pub(in crate::engine::gpu) table: &'a TableBuffers,
}

/// The concatenated received blocks a fused-layer launch reads for its received entries.
pub(in crate::engine::gpu) struct FusedRecv<'a> {
    pub(in crate::engine::gpu) offsets: &'a CudaSlice<u32>,
    pub(in crate::engine::gpu) base: &'a CudaSlice<u32>,
    pub(in crate::engine::gpu) x: &'a CudaSlice<u64>,
    pub(in crate::engine::gpu) z: &'a CudaSlice<u64>,
    pub(in crate::engine::gpu) coefficient: &'a CudaSlice<f64>,
    pub(in crate::engine::gpu) g: &'a CudaSlice<u64>,
}

/// What a fused-layer launch writes: the loose arena, rows per position, and the fallback counters.
pub(in crate::engine::gpu) struct FusedOut<'a, const W: usize> {
    pub(in crate::engine::gpu) arena: &'a mut DeviceColumns<W>,
    pub(in crate::engine::gpu) out_len_pos: &'a mut CudaSlice<u32>,
    pub(in crate::engine::gpu) fallback: &'a mut CudaSlice<u32>,
}

/// K3 over positions `p0..p0 + num_blocks` with `kernel = (function, shared bytes)` from [`fused_variant`].
#[allow(clippy::too_many_arguments)]
pub(in crate::engine::gpu) fn launch_fused<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    fused: &FusedTable<'_>,
    received: &FusedRecv<'_>,
    bucket_at: &CudaSlice<u32>,
    keep: &KeepProgram,
    kernel: (&CudaFunction, u32),
    (p0, num_blocks): (u32, u32),
    out: FusedOut<'_, W>,
) -> Result<(), GpuError> {
    let (func, smem) = kernel;
    let e32 = table.entries as u32;
    let b32 = sum.hash.num_buckets() as u32;
    let FusedOut {
        arena,
        out_len_pos,
        fallback,
    } = out;
    // SAFETY: arguments match the `LAYER_KERNEL` signature in layer.cu; the arena holds the batch's pre-dedup rows, `out_len_pos` has `b` entries, `fused` is `table`'s upload with `counts`/`segment_start` sized for it, and the received columns hold every row `table.rem` names.
    unsafe {
        sum.stream
            .launch_builder(func)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.g)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lens)
            .arg(bucket_at)
            .arg(fused.counts)
            .arg(fused.segment_start)
            .arg(&table.mode)
            .arg(&e32)
            .arg(&table.kq)
            .arg(&table.q0)
            .arg(&table.q1)
            .arg(&table.rot_cos)
            .arg(&table.rot_sin)
            .arg(&fused.table.amp)
            .arg(&fused.table.mask)
            .arg(&fused.table.nz)
            .arg(&fused.table.bucket_delta)
            .arg(&fused.table.gm)
            .arg(&fused.table.rem)
            .arg(received.offsets)
            .arg(received.base)
            .arg(&b32)
            .arg(received.x)
            .arg(received.z)
            .arg(received.coefficient)
            .arg(received.g)
            .arg(keep)
            .arg(&p0)
            .arg(&mut arena.x)
            .arg(&mut arena.z)
            .arg(&mut arena.coeff)
            .arg(&mut arena.g)
            .arg(out_len_pos)
            .arg(fallback)
            .launch(LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (layer_threads(W), 1, 1),
                shared_mem_bytes: smem,
            })?;
    }
    Ok(())
}
