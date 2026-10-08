//! Launching the fused layer kernel (K3): position batches that fit the arena, the kernel variant, and the launch itself.

use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use super::TableBufs;
use crate::engine::gpu::columns::DeviceColumns;
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::module::{layer_shared_bytes, layer_threads, KernelSet};
use crate::engine::gpu::prepared::DevicePrepared;
use crate::engine::gpu::sum::GpuSum;
use crate::engine::gpu::truncation::KeepProgram;

/// Contiguous position ranges whose pre-dedup rows (`seg`, `b + 1` CSR offsets) fit an arena of `arena_bytes`, each starting a new range at every position of the ascending `starts`, and the largest range's rows.
pub(in crate::engine::gpu) fn arena_batches<const W: usize>(
    seg: &[u32],
    arena_bytes: usize,
    cap: usize,
    starts: &[usize],
) -> (Vec<(usize, usize)>, usize) {
    let b = seg.len() - 1;
    let cap_rows = (arena_bytes / DeviceColumns::<W>::BYTES_PER_TERM).max(cap);
    let mut batches: Vec<(usize, usize)> = Vec::new();
    let mut starts = starts.iter().copied().peekable();
    let mut p0 = 0usize;
    while p0 < b {
        while starts.next_if(|&c| c <= p0).is_some() {}
        let end = starts.peek().copied().unwrap_or(b).min(b);
        let mut p1 = p0 + 1;
        while p1 < end && (seg[p1 + 1] - seg[p0]) as usize <= cap_rows {
            p1 += 1;
        }
        batches.push((p0, p1));
        p0 = p1;
    }
    let max_rows = batches
        .iter()
        .map(|&(a, c)| (seg[c] - seg[a]) as usize)
        .max()
        .unwrap_or(0);
    (batches, max_rows)
}

/// The fused-layer variant for blocks of up to `records_max` records: the kernel, its record capacity and its dynamic shared bytes.
pub(in crate::engine::gpu) fn fused_variant(
    k: &KernelSet,
    w: usize,
    records_max: usize,
    dense: bool,
) -> (&CudaFunction, usize, u32) {
    let threads = layer_threads(w) as usize;
    let n_cap = records_max.max(threads).next_power_of_two();
    let variant = &k.layer[(n_cap / threads).trailing_zeros() as usize];
    debug_assert_eq!(variant.items * threads, n_cap);
    let func = if dense {
        &variant.segscan
    } else {
        &variant.serial
    };
    (func, n_cap, layer_shared_bytes(n_cap, w))
}

/// The table-side buffers one fused-layer launch reads: the counts and segment starts sized for `table`'s upload.
pub(in crate::engine::gpu) struct FusedTable<'a> {
    pub(in crate::engine::gpu) cnt: &'a CudaSlice<u32>,
    pub(in crate::engine::gpu) seg_start: &'a CudaSlice<u32>,
    pub(in crate::engine::gpu) table: &'a TableBufs,
}

/// The concatenated received blocks a fused-layer launch reads for its received entries.
pub(in crate::engine::gpu) struct FusedRecv<'a> {
    pub(in crate::engine::gpu) off: &'a CudaSlice<u32>,
    pub(in crate::engine::gpu) base: &'a CudaSlice<u32>,
    pub(in crate::engine::gpu) x: &'a CudaSlice<u64>,
    pub(in crate::engine::gpu) z: &'a CudaSlice<u64>,
    pub(in crate::engine::gpu) c: &'a CudaSlice<f64>,
    pub(in crate::engine::gpu) g: &'a CudaSlice<u64>,
}

/// What a fused-layer launch writes: the loose arena, rows per position, and the fallback counters.
pub(in crate::engine::gpu) struct FusedOut<'a, const W: usize> {
    pub(in crate::engine::gpu) arena: &'a mut DeviceColumns<W>,
    pub(in crate::engine::gpu) out_len_pos: &'a mut CudaSlice<u32>,
    pub(in crate::engine::gpu) fallback: &'a mut CudaSlice<u32>,
}

/// K3 over positions `p0..p0 + nblk` with `kernel = (function, shared bytes)` from [`fused_variant`].
#[allow(clippy::too_many_arguments)]
pub(in crate::engine::gpu) fn launch_fused<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    t: &FusedTable<'_>,
    r: &FusedRecv<'_>,
    bucket_at: &CudaSlice<u32>,
    keep: &KeepProgram,
    kernel: (&CudaFunction, u32),
    (p0, nblk): (u32, u32),
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
    // SAFETY: arguments match the `LAYER_KERNEL` signature in layer.cu; the arena holds the batch's pre-dedup rows, `out_len_pos` has `b` entries, `t` is `table`'s upload with `cnt`/`seg_start` sized for it, and the received columns hold every row `t.rem` names.
    unsafe {
        sum.stream
            .launch_builder(func)
            .arg(&sum.cols.x)
            .arg(&sum.cols.z)
            .arg(&sum.cols.coeff)
            .arg(&sum.cols.g)
            .arg(&sum.cols.start)
            .arg(&sum.cols.lens)
            .arg(bucket_at)
            .arg(t.cnt)
            .arg(t.seg_start)
            .arg(&table.mode)
            .arg(&e32)
            .arg(&table.kq)
            .arg(&table.q0)
            .arg(&table.q1)
            .arg(&table.rot_cos)
            .arg(&table.rot_sin)
            .arg(&t.table.amp)
            .arg(&t.table.mask)
            .arg(&t.table.nz)
            .arg(&t.table.bd)
            .arg(&t.table.gm)
            .arg(&t.table.rem)
            .arg(r.off)
            .arg(r.base)
            .arg(&b32)
            .arg(r.x)
            .arg(r.z)
            .arg(r.c)
            .arg(r.g)
            .arg(keep)
            .arg(&p0)
            .arg(&mut arena.x)
            .arg(&mut arena.z)
            .arg(&mut arena.coeff)
            .arg(&mut arena.g)
            .arg(out_len_pos)
            .arg(fallback)
            .launch(LaunchConfig {
                grid_dim: (nblk, 1, 1),
                block_dim: (layer_threads(W), 1, 1),
                shared_mem_bytes: smem,
            })?;
    }
    Ok(())
}
