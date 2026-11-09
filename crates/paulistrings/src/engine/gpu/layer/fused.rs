//! The fused layer: position batches that fit the arena, the kernel variant, its launch and the compaction into the output.

use std::sync::Arc;

use crate::coefficient::Coefficient;
use cudarc::driver::{CudaFunction, CudaSlice, PushKernelArg};

use super::{read_back, timed_transfer, LayerScratch, TableBuffers, Transfer};
use crate::engine::gpu::columns::{grow, DeviceColumns};
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::export::receive_chunk;
use crate::engine::gpu::module::{
    launch_grid, layer_shared_bytes, layer_threads, thread_per, KernelSet,
};
use crate::engine::gpu::prepared::DevicePrepared;
use crate::engine::gpu::scan::exclusive_scan;
use crate::engine::gpu::sum::GpuSum;
use crate::engine::gpu::truncation::KeepProgram;
use crate::engine::timing::timed;

/// Contiguous position ranges whose pre-dedup rows (`seg`, `b + 1` CSR offsets) fit an arena of `arena_bytes`, each starting a new range at every position of the ascending `starts`, and the largest range's rows.
pub(in crate::engine::gpu) fn arena_batches<const W: usize, C: Coefficient>(
    segment_start: &[u32],
    arena_bytes: usize,
    record_cap: usize,
    starts: &[usize],
) -> (Vec<(usize, usize)>, usize) {
    let b = segment_start.len() - 1;
    let cap_rows = (arena_bytes / DeviceColumns::<W, C>::BYTES_PER_TERM).max(record_cap);
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

/// The fused layer over every position, after the count table: position batches that fit the arena, each launched and compacted into the output, a pending receive taken chunk by chunk; the previous columns become `sum.spare`.
pub(super) fn fused_layer<const W: usize, C: Coefficient>(
    sum: &mut GpuSum<W, C>,
    scratch: &mut LayerScratch<W, C>,
    table: &DevicePrepared<W, C>,
    keep: &KeepProgram,
    records: u32,
    records_max: u32,
) -> Result<(), GpuError> {
    let b = sum.hash.num_buckets();
    let record_cap = sum.kernels.layer_cap();
    let kernels: Arc<KernelSet> = sum.kernels.clone();
    let (func, record_capacity, smem) =
        fused_variant(&kernels, W, records_max as usize, table.dense);
    scratch.record.record_capacity = record_capacity as u32;

    // Batches: contiguous position ranges whose pre-dedup rows fit the arena, never straddling a chunk of the pending receive.
    let chunk_starts: Vec<usize> =
        scratch.export.pending.as_ref().map_or_else(Vec::new, |p| {
            (0..p.map.chunks()).map(|c| p.map.bound(c) as usize).collect()
        });
    let (batches, max_batch_rows) = arena_batches::<W, C>(
        &scratch.segment_start_host,
        scratch.options.arena_bytes,
        record_cap,
        &chunk_starts,
    );
    scratch.record.batches = batches.len() as u32;
    scratch.receive_chunks = chunk_starts.len() as u32;

    let stream = sum.stream.clone();
    let ordinal = sum.device();
    let mut arena = match scratch.arena.take() {
        Some(a) => a,
        None => DeviceColumns::<W, C>::with_capacity(&stream, ordinal, max_batch_rows, 1)?,
    };
    arena.len = 0;
    arena.buckets = 0;
    arena.reserve(max_batch_rows, 1)?;
    let mut out = sum.take_spare(0, b)?;
    grow(&stream, &mut scratch.rows_per_position, b, ordinal)?;
    stream.memset_zeros(&mut scratch.fallback)?;
    // A lone batch whose pre-dedup rows already fit the output compacts at offset 0 without its total, which is read once after the compaction.
    let deferred_total = batches.len() == 1
        && (scratch.segment_start_host[b] - scratch.segment_start_host[0]) as usize
            <= out.term_capacity();
    let mut running = 0u32;
    let mut next_chunk = 0usize;
    for &(p0, p1) in &batches {
        if chunk_starts.get(next_chunk) == Some(&p0) {
            receive_chunk(sum, scratch, next_chunk)?;
            next_chunk += 1;
        }
        let num_blocks = (p1 - p0) as u32;
        let p0u = p0 as u32;
        timed!(let t0 = scratch.event(sum)?);
        let fused = FusedTable {
            counts: &scratch.counts,
            segment_start: &scratch.segment_start,
            table: &scratch.table,
        };
        let received = FusedReceived {
            offsets: &scratch.export.received_offsets,
            base: &scratch.export.received_base,
            x: &scratch.export.received_x,
            z: &scratch.export.received_z,
            coefficient: &scratch.export.received_coefficient,
            fingerprint: &scratch.export.received_fingerprint,
        };
        let written = FusedOut {
            arena: &mut arena,
            rows_per_position: &mut scratch.rows_per_position,
            fallback: &mut scratch.fallback,
        };
        launch_fused(
            sum,
            table,
            &fused,
            &received,
            &scratch.bucket_at,
            keep,
            (func, smem),
            (p0u, num_blocks),
            written,
        )?;
        timed!(scratch.lap(sum, t0, |stats| &mut stats.merge_ns)?);
        let n = p1 - p0;
        timed!(let t1 = scratch.event(sum)?);
        exclusive_scan(
            &stream,
            &kernels,
            &scratch.rows_per_position.slice(p0..p1),
            &mut scratch.destination_offsets.slice_mut(0..n + 1),
            n,
            &mut scratch.scan,
            &mut scratch.totals,
        )?;
        let batch_out = if deferred_total {
            0
        } else {
            let batch_out = read_back(&stream, &mut scratch.stats, &scratch.totals)?[0];
            out.len = running as usize;
            let need = (running + batch_out) as usize;
            if need > out.term_capacity() {
                let grown = (2 * out.term_capacity()).max(need).min((records as usize).max(need));
                out.reserve(grown, b)?;
            }
            batch_out
        };
        // SAFETY: arguments match `k_compact` in compact.cu; `out` holds `running + batch_out` terms.
        unsafe {
            stream
                .launch_builder(&kernels.compact)
                .arg(&arena.x)
                .arg(&arena.z)
                .arg(&arena.coeff)
                .arg(&arena.fingerprint)
                .arg(&scratch.segment_start)
                .arg(&scratch.destination_offsets)
                .arg(&scratch.rows_per_position)
                .arg(&scratch.bucket_at)
                .arg(&p0u)
                .arg(&running)
                .arg(&mut out.x)
                .arg(&mut out.z)
                .arg(&mut out.coeff)
                .arg(&mut out.fingerprint)
                .arg(&mut out.start)
                .arg(&mut out.lengths)
                .launch(thread_per(num_blocks as usize * 256, 256))?;
        }
        timed!(scratch.lap(sum, t1, |stats| &mut stats.compact_ns)?);
        running += batch_out;
    }
    debug_assert_eq!(next_chunk, chunk_starts.len());
    let (totals, fallback) = (&scratch.totals, &scratch.fallback);
    let (total, fallbacks) = timed_transfer(&stream, &mut scratch.stats, Transfer::D2h, || {
        let total = if deferred_total { stream.clone_dtoh(totals)?[0] } else { 0 };
        let fallbacks = stream.clone_dtoh(fallback)?;
        stream.synchronize()?;
        Ok((total, fallbacks))
    })?;
    running += total;
    scratch.record.high_word_fallbacks += fallbacks[0];
    scratch.record.full_key_fallbacks += fallbacks[1];
    scratch.arena = Some(arena);
    sum.swap_in(out, running as usize);
    sum.record_exact(running as usize, None);
    scratch.extent = sum.len();
    sum.debug_check();
    Ok(())
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
    let func = if dense { &variant.segscan } else { &variant.serial };
    (func, record_capacity, layer_shared_bytes(record_capacity, w, kernels.coefficient))
}

/// The table-side buffers one fused-layer launch reads: the counts and segment starts sized for `table`'s upload.
pub(in crate::engine::gpu) struct FusedTable<'a> {
    pub(in crate::engine::gpu) counts: &'a CudaSlice<u32>,
    pub(in crate::engine::gpu) segment_start: &'a CudaSlice<u32>,
    pub(in crate::engine::gpu) table: &'a TableBuffers,
}

/// The concatenated received blocks a fused-layer launch reads for its received entries.
pub(in crate::engine::gpu) struct FusedReceived<'a> {
    pub(in crate::engine::gpu) offsets: &'a CudaSlice<u32>,
    pub(in crate::engine::gpu) base: &'a CudaSlice<u32>,
    pub(in crate::engine::gpu) x: &'a CudaSlice<u64>,
    pub(in crate::engine::gpu) z: &'a CudaSlice<u64>,
    pub(in crate::engine::gpu) coefficient: &'a CudaSlice<u8>,
    pub(in crate::engine::gpu) fingerprint: &'a CudaSlice<u64>,
}

/// What a fused-layer launch writes: the loose arena, rows per position, and the fallback counters.
pub(in crate::engine::gpu) struct FusedOut<'a, const W: usize, C: Coefficient> {
    pub(in crate::engine::gpu) arena: &'a mut DeviceColumns<W, C>,
    pub(in crate::engine::gpu) rows_per_position: &'a mut CudaSlice<u32>,
    pub(in crate::engine::gpu) fallback: &'a mut CudaSlice<u32>,
}

/// The fused layer over positions `p0..p0 + num_blocks` with `kernel = (function, shared bytes)` from [`fused_variant`].
#[allow(clippy::too_many_arguments)]
pub(in crate::engine::gpu) fn launch_fused<const W: usize, C: Coefficient>(
    sum: &GpuSum<W, C>,
    table: &DevicePrepared<W, C>,
    fused: &FusedTable<'_>,
    received: &FusedReceived<'_>,
    bucket_at: &CudaSlice<u32>,
    keep: &KeepProgram,
    kernel: (&CudaFunction, u32),
    (p0, num_blocks): (u32, u32),
    out: FusedOut<'_, W, C>,
) -> Result<(), GpuError> {
    let (func, smem) = kernel;
    let e32 = table.entries as u32;
    let b32 = sum.hash.num_buckets() as u32;
    let FusedOut { arena, rows_per_position, fallback } = out;
    // SAFETY: arguments match the `LAYER_KERNEL` signature in layer.cu; the arena holds the batch's pre-dedup rows, `rows_per_position` has `b` entries, `fused` is `table`'s upload with `counts`/`segment_start` sized for it, and the received columns hold every row `table.received_slot` names.
    unsafe {
        sum.stream
            .launch_builder(func)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.fingerprint)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(bucket_at)
            .arg(fused.counts)
            .arg(fused.segment_start)
            .arg(&table.mode)
            .arg(&e32)
            .arg(&table.support_size)
            .arg(&table.qubit_0)
            .arg(&table.qubit_1)
            .arg(&table.rotation_cos)
            .arg(&table.rotation_sin)
            .arg(&fused.table.amplitude)
            .arg(&fused.table.mask)
            .arg(&fused.table.nonzero_patterns)
            .arg(&fused.table.bucket_delta)
            .arg(&fused.table.fingerprint_delta)
            .arg(&fused.table.received_slot)
            .arg(received.offsets)
            .arg(received.base)
            .arg(&b32)
            .arg(received.x)
            .arg(received.z)
            .arg(received.coefficient)
            .arg(received.fingerprint)
            .arg(keep)
            .arg(&p0)
            .arg(&mut arena.x)
            .arg(&mut arena.z)
            .arg(&mut arena.coeff)
            .arg(&mut arena.fingerprint)
            .arg(rows_per_position)
            .arg(fallback)
            .launch(launch_grid(num_blocks, layer_threads(W), smem))?;
    }
    Ok(())
}
