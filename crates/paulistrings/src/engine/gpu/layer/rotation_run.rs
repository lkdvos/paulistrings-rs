//! A rotation run applied in one pass over the sum (`kernels/rotation_run.cu`), every layer inside shared memory.
//!
//! Contracts: the result agrees with the run's layers applied one by one to floating-point tolerance, `keep` asked after every layer about every string that layer changed and after the first about every string, as `apply_rotation_run` does (engine/bucketed/rotation_layer.rs); the Clifford arm drops its identity residues as the host's rotation layer does.
//! A layer moves a string only within its class, its key off the run's qubits, so a block holding whole classes needs no other block; apply blocks pack consecutive class buckets, one that outgrows shared memory reruns over tables in global memory, and only one that outgrows [`WIDE_MAX_ROWS`] leaves the sum untouched for the caller to run the layers one by one.

use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};

use super::run::refine_to_target;
use super::{timed_transfer, LayerScratch, Transfer};
use crate::channel::prepared::{LocalPtm, Prepared, LOCAL_DIM};
use crate::coefficient::Coefficient;
use crate::engine::gpu::columns::{grow, DeviceColumns};
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::module::{thread_per, RR_ROWS, RR_THREADS, RR_WIDE_THREADS};
use crate::engine::gpu::scan::exclusive_scan;
use crate::engine::gpu::sum::GpuSum;
use crate::engine::gpu::truncation::KeepProgram;
use crate::engine::timing::timed;

/// Most layers and qubits a fused run takes; `RR_MAX_LAYERS` and `RR_MAX_QUBITS` in `kernels/rotation_run.cu`.
const MAX_LAYERS: usize = 8;
const MAX_QUBITS: usize = 16;

/// Block width of the per-bucket passes.
const BUCKET_THREADS: u32 = 256;

/// Strings per class bucket the class-bucket count aims for, touched or not; apply blocks pack consecutive class buckets.
const CLASS_TARGET: usize = 32;

/// Touched strings an apply block aims for, a quarter of its input half, so a run's growth seldom outgrows it.
const BLOCK_ROWS: usize = RR_ROWS / 8;

/// Most rows an apply block's global tables hold before the run falls back to its layers one by one.
const WIDE_MAX_ROWS: usize = 1 << 22;

/// Most bytes one launch of the wide pass gives its blocks' tables; a longer overflow list runs in several launches.
const WIDE_MAX_BYTES: usize = 1 << 30;

/// `RunInfo` in `kernels/rotation_run.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct RotationRunInfo {
    layers: u32,
    qubits_len: u32,
    qubits: [u32; MAX_QUBITS],
    low: [u32; MAX_LAYERS],
    high: [u32; MAX_LAYERS],
    two: [u32; MAX_LAYERS],
    flip: [u32; MAX_LAYERS],
    active: [u32; MAX_LAYERS],
    clifford: [u32; MAX_LAYERS],
    bucket: [u32; 2 * MAX_QUBITS],
    fingerprint: [u64; 2 * MAX_QUBITS],
}

const _: () = assert!(std::mem::size_of::<RotationRunInfo>() == 648);

// SAFETY: plain `repr(C)` data with the field order, types and padding of `RunInfo` in kernels/rotation_run.cu.
unsafe impl DeviceRepr for RotationRunInfo {}

/// The two entries of a rotation table, `(identity, generator)`, when `ptm` has that shape: a pattern the generator skips keeps its string exactly.
fn rotation_entries<const W: usize, C: Coefficient>(
    ptm: &LocalPtm<W, C>,
) -> Option<(&[C; LOCAL_DIM], &crate::channel::prepared::DeltaEntry<W, C>)> {
    let [identity, generator] = ptm.deltas() else { return None };
    if identity.local_delta != 0 {
        return None;
    }
    let dim = 1usize << (2 * ptm.k());
    let delta = generator.local_delta as usize;
    for s in 0..dim {
        if generator.amplitude[s] == C::ZERO {
            if identity.amplitude[s] != C::ONE {
                return None;
            }
        } else if generator.amplitude[s ^ delta] == C::ZERO {
            return None;
        }
    }
    Some((&identity.amplitude, generator))
}

/// `layers` as the kernels take them, with each layer's identity then generator amplitudes, or `None` when one is not a rotation table or the run is too wide.
fn run_info<const W: usize, C: Coefficient>(
    layers: &[Prepared<W, C>],
    sum: &GpuSum<W, C>,
) -> Option<(RotationRunInfo, Vec<C>)> {
    if layers.len() > MAX_LAYERS {
        return None;
    }
    let mut info = RotationRunInfo { layers: layers.len() as u32, ..Default::default() };
    let mut qubits: Vec<usize> = Vec::new();
    let mut amplitude = Vec::with_capacity(layers.len() * 2 * LOCAL_DIM);
    for (i, layer) in layers.iter().enumerate() {
        let Prepared::Local(ptm) = layer else { return None };
        let (identity, generator) = rotation_entries(ptm)?;
        let mut shift = [0u32; 2];
        for (j, q) in ptm.qubits().enumerate() {
            let index = match qubits.iter().position(|&known| known == q) {
                Some(index) => index,
                None => {
                    qubits.push(q);
                    qubits.len() - 1
                }
            };
            shift[j] = 2 * index as u32;
        }
        if qubits.len() > MAX_QUBITS {
            return None;
        }
        let two = ptm.k() == 2;
        let d = u32::from(generator.local_delta);
        info.low[i] = shift[0];
        info.high[i] = if two { shift[1] } else { shift[0] };
        info.two[i] = u32::from(two);
        info.flip[i] = (d & 3) << shift[0] | if two { (d >> 2 & 3) << shift[1] } else { 0 };
        let dim = 1usize << (2 * ptm.k());
        let mut clifford = true;
        for (s, (a0, a1)) in identity.iter().zip(&generator.amplitude).take(dim).enumerate() {
            if *a1 != C::ZERO {
                info.active[i] |= 1 << s;
                // `cos θ` evaluated at a quarter turn is a rounding residue of `sin θ`, as on the host.
                clifford &= a0.norm_sqr() <= f64::EPSILON * f64::EPSILON * a1.norm_sqr();
            }
        }
        info.clifford[i] = u32::from(clifford);
        amplitude.extend_from_slice(identity);
        amplitude.extend_from_slice(&generator.amplitude);
    }
    info.qubits_len = qubits.len() as u32;
    for (j, &q) in qubits.iter().enumerate() {
        info.qubits[j] = q as u32;
        for (column, bit) in [(0usize, 2 * j), (1, 2 * j + 1)] {
            let (mut mask_x, mut mask_z) = ([0u64; W], [0u64; W]);
            let mask = if column == 0 { &mut mask_x } else { &mut mask_z };
            mask[q / 64] = 1u64 << (q % 64);
            info.bucket[bit] = sum.hash.bucket_of(&mask_x, &mask_z);
            info.fingerprint[bit] = sum.fingerprints.fingerprint(&mask_x, &mask_z);
        }
    }
    Some((info, amplitude))
}

/// The fused run's grow-only device buffers.
pub(crate) struct RotationRunScratch<const W: usize, C: Coefficient> {
    /// The touched strings by class bucket, and each one's bucket.
    gathered: DeviceColumns<W, C>,
    gathered_bucket: CudaSlice<u32>,
    /// The survivors of every class bucket, and each one's output bucket.
    arena: DeviceColumns<W, C>,
    arena_bucket: CudaSlice<u32>,
    class_count: CudaSlice<u32>,
    class_start: CudaSlice<u32>,
    /// Each apply block's first class bucket, `k_rotation_run_ranges`.
    block_lo: CudaSlice<u32>,
    inert: CudaSlice<u32>,
    total: CudaSlice<u32>,
    cursor: CudaSlice<u32>,
    /// `[arena rows claimed, overflowed class buckets listed in overflow[0], in overflow[1], their largest input]`.
    counters: CudaSlice<u32>,
    /// The class buckets one apply pass could not hold, for the next wide pass.
    overflow: [CudaSlice<u32>; 2],
    /// The wide pass's per-block tables.
    tables: CudaSlice<u8>,
    amplitude: CudaSlice<u8>,
}

impl<const W: usize, C: Coefficient> RotationRunScratch<W, C> {
    fn new(sum: &GpuSum<W, C>) -> Result<Self, GpuError> {
        let (stream, ordinal) = (&sum.stream, sum.device());
        Ok(Self {
            gathered: DeviceColumns::with_capacity(stream, ordinal, 1, 1)?,
            gathered_bucket: stream.alloc_zeros(1)?,
            arena: DeviceColumns::with_capacity(stream, ordinal, 1, 1)?,
            arena_bucket: stream.alloc_zeros(1)?,
            class_count: stream.alloc_zeros(2)?,
            class_start: stream.alloc_zeros(2)?,
            block_lo: stream.alloc_zeros(2)?,
            inert: stream.alloc_zeros(1)?,
            total: stream.alloc_zeros(1)?,
            cursor: stream.alloc_zeros(1)?,
            counters: stream.alloc_zeros(4)?,
            overflow: [stream.alloc_zeros(1)?, stream.alloc_zeros(1)?],
            tables: stream.alloc_zeros(1)?,
            amplitude: stream.alloc_zeros(2 * MAX_LAYERS * LOCAL_DIM * 16)?,
        })
    }
}

/// Table bytes per row of a class bucket, `RR_ROW_BYTES` in `kernels/rotation_run.cu`.
fn row_bytes(coefficient_bytes: usize) -> usize {
    2 * coefficient_bytes + 24
}

/// Dynamic shared bytes of `k_rotation_run_apply`, `RR_SHARED_BYTES` in `kernels/rotation_run.cu`.
pub(crate) fn apply_shared_bytes(coefficient_bytes: usize) -> u32 {
    (RR_ROWS * row_bytes(coefficient_bytes)) as u32
}

/// Rows a wide pass may give an apply block: four times the largest input that overflowed and four times the previous pass's rows, a power of two; each block takes four times its own input, at least four times the previous pass's rows.
fn wide_rows(largest_input: u32, previous: usize) -> usize {
    (4 * largest_input as usize).max(4 * previous).next_power_of_two()
}

/// Apply the rotation run `layers` to `sum` in one pass, into the spare columns; `Ok(false)`, the sum untouched, when a layer is not a rotation table or a class bucket outgrows its block.
pub(crate) fn rotation_run_fused<const W: usize, C: Coefficient>(
    sum: &mut GpuSum<W, C>,
    layers: &[Prepared<W, C>],
    keep: &KeepProgram,
    target_bits: u8,
    scratch: &mut LayerScratch<W, C>,
) -> Result<bool, GpuError> {
    refine_to_target(sum, scratch, target_bits)?;
    let Some((info, amplitude)) = run_info(layers, sum) else { return Ok(false) };
    let stream = sum.stream.clone();
    let kernels = sum.kernels.clone();
    let ordinal = sum.device();
    let b = sum.hash.num_buckets();
    let n_in = sum.len();
    let class_bits =
        n_in.div_ceil(CLASS_TARGET).max(1).next_power_of_two().trailing_zeros().min(24);
    let classes = 1usize << class_bits;
    let mut buffers = match scratch.rotation_run.take() {
        Some(buffers) => buffers,
        None => RotationRunScratch::new(sum)?,
    };
    grow(&stream, &mut buffers.class_count, classes, ordinal)?;
    grow(&stream, &mut buffers.class_start, classes + 1, ordinal)?;
    grow(&stream, &mut buffers.inert, b, ordinal)?;
    grow(&stream, &mut buffers.total, b, ordinal)?;
    grow(&stream, &mut buffers.cursor, b.max(classes), ordinal)?;
    buffers.gathered.len = 0;
    buffers.gathered.reserve(n_in, 1)?;
    grow(&stream, &mut buffers.gathered_bucket, n_in, ordinal)?;
    let amplitude = bytemuck::cast_slice::<C, u8>(&amplitude);
    timed_transfer(&stream, &mut scratch.stats, Transfer::H2d, || {
        stream.memcpy_htod(amplitude, &mut buffers.amplitude.slice_mut(0..amplitude.len()))?;
        Ok(())
    })?;
    let (class_bits32, b32) = (class_bits, b as u32);
    let per_bucket = LaunchConfig {
        grid_dim: (b32, 1, 1),
        block_dim: (BUCKET_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    timed!(let t0 = scratch.event(sum)?);
    stream.memset_zeros(&mut buffers.class_count.slice_mut(0..classes))?;
    // SAFETY: arguments match `k_rotation_run_classify` in rotation_run.cu; `class_count` holds `2^class_bits` zeroed counters and `inert` one per bucket.
    unsafe {
        stream
            .launch_builder(&kernels.rotation_run_classify)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.fingerprint)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&info)
            .arg(keep)
            .arg(&class_bits32)
            .arg(&mut buffers.class_count)
            .arg(&mut buffers.inert)
            .launch(per_bucket)?;
    }
    exclusive_scan(
        &stream,
        &kernels,
        &buffers.class_count.slice(0..classes),
        &mut buffers.class_start.slice_mut(0..classes + 1),
        classes,
        &mut scratch.scan,
        &mut scratch.totals,
    )?;
    stream.memcpy_dtod(
        &buffers.class_start.slice(0..classes),
        &mut buffers.cursor.slice_mut(0..classes),
    )?;
    // SAFETY: arguments match `k_rotation_run_gather`; `cursor` starts at `class_start`, and `gathered` has room for every input row.
    unsafe {
        let gathered = &mut buffers.gathered;
        stream
            .launch_builder(&kernels.rotation_run_gather)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.fingerprint)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&info)
            .arg(&class_bits32)
            .arg(&mut buffers.cursor)
            .arg(&mut gathered.x)
            .arg(&mut gathered.z)
            .arg(&mut gathered.coeff)
            .arg(&mut gathered.fingerprint)
            .arg(&mut buffers.gathered_bucket)
            .launch(per_bucket)?;
    }
    let totals = &scratch.totals;
    let touched = timed_transfer(&stream, &mut scratch.stats, Transfer::D2h, || {
        let total = stream.clone_dtoh(totals)?;
        stream.synchronize()?;
        Ok(total[0] as usize)
    })?;
    let blocks = touched.div_ceil(BLOCK_ROWS);
    grow(&stream, &mut buffers.block_lo, blocks + 1, ordinal)?;
    for list in &mut buffers.overflow {
        grow(&stream, list, blocks.max(1), ordinal)?;
    }
    if blocks > 0 {
        let (classes32, per_block, blocks32) = (classes as u32, BLOCK_ROWS as u32, blocks as u32);
        // SAFETY: arguments match `k_rotation_run_ranges`; `class_start` holds `classes + 1` offsets ending at `touched`, and `block_lo` holds `blocks + 1` entries.
        unsafe {
            stream
                .launch_builder(&kernels.rotation_run_ranges)
                .arg(&buffers.class_start)
                .arg(&classes32)
                .arg(&per_block)
                .arg(&blocks32)
                .arg(&mut buffers.block_lo)
                .launch(thread_per(classes + 1, 256))?;
        }
    }
    timed!(scratch.lap(sum, t0, |stats| &mut stats.gather_ns)?);

    let mut out = match sum.spare.take() {
        Some(spare) => spare,
        None => DeviceColumns::<W, C>::with_capacity(&stream, ordinal, n_in, b)?,
    };
    out.len = 0;
    out.buckets = 0;
    out.reserve(out.term_capacity(), b)?;
    if buffers.arena.term_capacity() < n_in {
        buffers.arena.len = 0;
        buffers.arena.reserve(n_in, 1)?;
        grow(&stream, &mut buffers.arena_bucket, n_in, ordinal)?;
    }
    let shared = apply_shared_bytes(DeviceColumns::<W, C>::COEFFICIENT_BYTES);
    let (claimed, total) = loop {
        let arena_cap = buffers.arena.term_capacity() as u32;
        timed!(let t1 = scratch.event(sum)?);
        stream.memcpy_dtod(&buffers.inert.slice(0..b), &mut buffers.total.slice_mut(0..b))?;
        stream.memset_zeros(&mut buffers.counters)?;
        if blocks > 0 {
            // SAFETY: arguments match `k_rotation_run_apply`; `class_start` holds `classes + 1` offsets into `gathered`, `block_lo` the `blocks + 1` ranges over them, the arena holds `arena_cap` rows, `total` one counter per bucket, and the dynamic shared bytes are the kernel's `RR_SHARED_BYTES`.
            unsafe {
                let arena = &mut buffers.arena;
                stream
                    .launch_builder(&kernels.rotation_run_apply)
                    .arg(&buffers.gathered.x)
                    .arg(&buffers.gathered.z)
                    .arg(&buffers.gathered.coeff)
                    .arg(&buffers.gathered.fingerprint)
                    .arg(&buffers.gathered_bucket)
                    .arg(&buffers.class_start)
                    .arg(&buffers.block_lo)
                    .arg(&info)
                    .arg(&buffers.amplitude)
                    .arg(keep)
                    .arg(&mut arena.x)
                    .arg(&mut arena.z)
                    .arg(&mut arena.coeff)
                    .arg(&mut arena.fingerprint)
                    .arg(&mut buffers.arena_bucket)
                    .arg(&arena_cap)
                    .arg(&mut buffers.counters)
                    .arg(&mut buffers.total)
                    .arg(&mut buffers.overflow[0])
                    .launch(LaunchConfig {
                        grid_dim: (blocks as u32, 1, 1),
                        block_dim: (RR_THREADS, 1, 1),
                        shared_mem_bytes: shared,
                    })?;
            }
        }
        let (mut counters, mut total) = scan_totals(sum, &buffers, &mut out, scratch)?;
        if counters[1] != 0 {
            match apply_wide(sum, &mut buffers, &info, keep, arena_cap, counters, scratch)? {
                Some(widened) => counters = widened,
                None => {
                    sum.spare = Some(out);
                    scratch.rotation_run = Some(buffers);
                    scratch.resolve(sum)?;
                    return Ok(false);
                }
            }
            total = scan_totals(sum, &buffers, &mut out, scratch)?.1;
        }
        timed!(scratch.lap(sum, t1, |stats| &mut stats.merge_ns)?);
        if counters[0] as usize > buffers.arena.term_capacity() {
            let need = counters[0] as usize;
            buffers.arena.len = 0;
            buffers.arena.reserve(need.max(2 * buffers.arena.term_capacity()), 1)?;
            grow(&stream, &mut buffers.arena_bucket, buffers.arena.term_capacity(), ordinal)?;
            continue;
        }
        break (counters[0], total);
    };
    out.reserve(total as usize, b)?;
    timed!(let t2 = scratch.event(sum)?);
    // SAFETY: arguments match `k_rotation_run_place`; `out` holds the scan's `total` rows at the starts the scan wrote, and `cursor` one entry per bucket.
    unsafe {
        stream
            .launch_builder(&kernels.rotation_run_place)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.fingerprint)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&info)
            .arg(keep)
            .arg(&buffers.inert)
            .arg(&buffers.total)
            .arg(&out.start)
            .arg(&mut out.lengths)
            .arg(&mut buffers.cursor)
            .arg(&mut out.x)
            .arg(&mut out.z)
            .arg(&mut out.coeff)
            .arg(&mut out.fingerprint)
            .launch(per_bucket)?;
    }
    if claimed > 0 {
        // SAFETY: arguments match `k_rotation_run_scatter`; the arena holds `claimed` rows, each bucket's cursor sits past its untouched rows, and the scan sized every bucket for its arena rows.
        unsafe {
            let arena = &buffers.arena;
            stream
                .launch_builder(&kernels.rotation_run_scatter)
                .arg(&arena.x)
                .arg(&arena.z)
                .arg(&arena.coeff)
                .arg(&arena.fingerprint)
                .arg(&buffers.arena_bucket)
                .arg(&claimed)
                .arg(&mut buffers.cursor)
                .arg(&mut out.x)
                .arg(&mut out.z)
                .arg(&mut out.coeff)
                .arg(&mut out.fingerprint)
                .launch(thread_per(claimed as usize, 256))?;
        }
    }
    timed!(scratch.lap(sum, t2, |stats| &mut stats.compact_ns)?);
    out.len = total as usize;
    out.buckets = b;
    scratch.rotation_run = Some(buffers);
    sum.spare = Some(std::mem::replace(&mut sum.columns, out));
    scratch.extent = sum.len();
    scratch.record.records = n_in as u64;
    scratch.record.joined = true;
    sum.debug_check();
    scratch.resolve(sum)?;
    Ok(true)
}

/// The exclusive scan of the output bucket totals into `out.start`, and the counters with the scan's total.
fn scan_totals<const W: usize, C: Coefficient>(
    sum: &GpuSum<W, C>,
    buffers: &RotationRunScratch<W, C>,
    out: &mut DeviceColumns<W, C>,
    scratch: &mut LayerScratch<W, C>,
) -> Result<(Vec<u32>, u32), GpuError> {
    let (stream, b) = (&sum.stream, sum.hash.num_buckets());
    exclusive_scan(
        stream,
        &sum.kernels,
        &buffers.total.slice(0..b),
        &mut out.start.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut scratch.totals,
    )?;
    let (counters, totals) = (&buffers.counters, &scratch.totals);
    timed_transfer(stream, &mut scratch.stats, Transfer::D2h, || {
        let counters = stream.clone_dtoh(counters)?;
        let total = stream.clone_dtoh(totals)?;
        stream.synchronize()?;
        Ok((counters, total[0]))
    })
}

/// Rerun the apply blocks the apply pass listed in `overflow[0]` over global tables, each pass's overflows again at four times the rows; the final counters, or `None` once a block would outgrow [`WIDE_MAX_ROWS`].
fn apply_wide<const W: usize, C: Coefficient>(
    sum: &GpuSum<W, C>,
    buffers: &mut RotationRunScratch<W, C>,
    info: &RotationRunInfo,
    keep: &KeepProgram,
    arena_cap: u32,
    mut counters: Vec<u32>,
    scratch: &mut LayerScratch<W, C>,
) -> Result<Option<Vec<u32>>, GpuError> {
    let (stream, ordinal) = (&sum.stream, sum.device());
    let row_bytes = row_bytes(DeviceColumns::<W, C>::COEFFICIENT_BYTES);
    let (mut listed, mut rows) = (1usize, RR_ROWS);
    loop {
        let pending = counters[listed] as usize;
        if pending == 0 {
            return Ok(Some(counters));
        }
        let min_rows = 4 * rows;
        rows = wide_rows(counters[3], rows);
        if rows > WIDE_MAX_ROWS {
            return Ok(None);
        }
        let per_launch = (WIDE_MAX_BYTES / (rows * row_bytes)).clamp(1, pending);
        grow(stream, &mut buffers.tables, per_launch * rows * row_bytes, ordinal)?;
        let next = 3 - listed;
        stream.memset_zeros(&mut buffers.counters.slice_mut(next..next + 1))?;
        let (rows32, min_rows32, next32) = (rows as u32, min_rows as u32, next as u32);
        let [first, second] = &mut buffers.overflow;
        let (overflow_in, overflow_out) =
            if listed == 1 { (first, second) } else { (second, first) };
        for start in (0..pending).step_by(per_launch) {
            let (start32, grid) = (start as u32, per_launch.min(pending - start) as u32);
            // SAFETY: arguments match `k_rotation_run_apply_wide` in rotation_run.cu; `overflow_in` lists `pending` apply blocks, `tables` holds `rows` rows of `RR_ROW_BYTES` for each of the `grid` launched, and `overflow_out` has room for every apply block.
            unsafe {
                let arena = &mut buffers.arena;
                stream
                    .launch_builder(&sum.kernels.rotation_run_apply_wide)
                    .arg(&buffers.gathered.x)
                    .arg(&buffers.gathered.z)
                    .arg(&buffers.gathered.coeff)
                    .arg(&buffers.gathered.fingerprint)
                    .arg(&buffers.gathered_bucket)
                    .arg(&buffers.class_start)
                    .arg(&buffers.block_lo)
                    .arg(info)
                    .arg(&buffers.amplitude)
                    .arg(keep)
                    .arg(&mut arena.x)
                    .arg(&mut arena.z)
                    .arg(&mut arena.coeff)
                    .arg(&mut arena.fingerprint)
                    .arg(&mut buffers.arena_bucket)
                    .arg(&arena_cap)
                    .arg(&mut buffers.counters)
                    .arg(&mut buffers.total)
                    .arg(&*overflow_in)
                    .arg(&start32)
                    .arg(&mut buffers.tables)
                    .arg(&rows32)
                    .arg(&min_rows32)
                    .arg(&next32)
                    .arg(&mut *overflow_out)
                    .launch(LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (RR_WIDE_THREADS, 1, 1),
                        shared_mem_bytes: 0,
                    })?;
            }
        }
        counters = timed_transfer(stream, &mut scratch.stats, Transfer::D2h, || {
            let counters = stream.clone_dtoh(&buffers.counters)?;
            stream.synchronize()?;
            Ok(counters)
        })?;
        listed = next;
    }
}

#[cfg(test)]
mod tests;
