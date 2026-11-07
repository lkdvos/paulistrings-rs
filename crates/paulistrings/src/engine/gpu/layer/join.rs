//! The join layer (`k_join` in `kernels/join.cu`): its table size, its launch, and its host flow straight into the output columns.

use crate::coefficient::Coefficient;
use cudarc::driver::PushKernelArg;

use crate::engine::gpu::module::thread_per;
use crate::engine::gpu::scan::exclusive_scan;

use super::LayerScratch;
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::length::Growth;
use crate::engine::gpu::module::{launch_grid, JOIN_THREADS};
use crate::engine::gpu::prepared::{DevicePrepared, JoinTable};
use crate::engine::gpu::sum::GpuSum;
use crate::engine::gpu::truncation::KeepProgram;
use crate::engine::timing::timed;
use crate::truncation::builtin::APPROX_BINS;

impl<const W: usize, C: Coefficient> LayerScratch<W, C> {
    /// `k_join_rows` and its scan into `segment_start`: each position's bound, the lengths of the two source buckets it reads, `2 · len` rows in all; nothing is read back.
    /// It replaces `k_count` and `k_rows` for a join, at the cost of output room for the rows the partner entry does not emit.
    pub(super) fn join_sizes(&mut self, sum: &GpuSum<W, C>) -> Result<(), GpuError> {
        let stream = &sum.stream;
        let b = sum.hash.num_buckets();
        let b32 = b as u32;
        timed!(let t = self.event(sum)?);
        // SAFETY: arguments match `k_join_rows` in join.cu; `bucket_at` and `rows` hold `b` entries and `bucket_delta` the table's two.
        unsafe {
            stream
                .launch_builder(&sum.kernels.join_rows)
                .arg(&sum.columns.lengths)
                .arg(&self.bucket_at)
                .arg(&self.table.bucket_delta)
                .arg(&mut self.rows)
                .arg(&b32)
                .arg(&mut self.join_totals)
                .launch(thread_per(b, 1024))?;
        }
        exclusive_scan(
            stream,
            &sum.kernels,
            &self.rows.slice(0..b),
            &mut self.segment_start.slice_mut(0..b + 1),
            b,
            &mut self.scan,
            &mut self.totals,
        )?;
        timed!(self.lap(sum, t, |stats| &mut stats.gather_ns)?);
        Ok(())
    }
}

/// Apply `table`'s join to `sum` over every position, each output bucket written at its position's offset in `scratch.segment_start` from [`LayerScratch::join_sizes`] (at most `records` rows in all), for source buckets of up to `longest` rows; the previous columns become `sum.spare`.
/// The output keeps a gap after each bucket, so its extent is at most `records` and no compaction runs; its length and longest bucket stay on the device, published to the sum's length tracker.
pub(super) fn join_layer<const W: usize, C: Coefficient>(
    sum: &mut GpuSum<W, C>,
    scratch: &mut LayerScratch<W, C>,
    table: &DevicePrepared<W, C>,
    join: JoinTable,
    keep: &KeepProgram,
    longest: u32,
    records: u32,
) -> Result<(), GpuError> {
    let stream = sum.stream.clone();
    let b = sum.hash.num_buckets();
    let mut out = sum.take_spare(0, b)?;
    let need = records as usize;
    if need > out.term_capacity() {
        // A growing sum would otherwise reallocate at nearly every join; growing by half again keeps that logarithmic, up to twice the policy's ceiling on a layer's input.
        let roomy = (out.term_capacity() + out.term_capacity() / 2)
            .min(scratch.ceiling.map_or(usize::MAX, |ceiling| ceiling.saturating_mul(2)))
            .max(need);
        if out.reserve(roomy, b).is_err() {
            out.reserve(need, b)?;
        }
    }
    out.reserve(need, b)?;
    // The largest block's table, at most half full.
    let table_bits = (2 * longest.max(16)).next_power_of_two().trailing_zeros();
    // The next pass's histogram comes out of the join itself (`LayerScratch::histogram_fresh`).
    let histogram = scratch.join_histogram;
    if histogram && !scratch.histogram_clean {
        stream.memset_zeros(&mut scratch.histogram)?;
    }
    let histogram_bytes = if histogram { 4 * APPROX_BINS as u32 } else { 0 };
    timed!(let t0 = scratch.event(sum)?);
    // SAFETY: arguments match `k_join` in join.cu; `out` holds `records` terms and `b` buckets, `segment_start` holds `b + 1` offsets ending at `records`, and no source bucket exceeds `longest` rows, so the `1 << table_bits` slots stay at most half full.
    unsafe {
        stream
            .launch_builder(&sum.kernels.join)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.fingerprint)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&scratch.bucket_at)
            .arg(&scratch.segment_start)
            .arg(&table.mode)
            .arg(&table.support_size)
            .arg(&table.qubit_0)
            .arg(&table.qubit_1)
            .arg(&table.rotation_cos)
            .arg(&table.rotation_sin)
            .arg(&scratch.table.amplitude)
            .arg(&scratch.table.mask)
            .arg(&scratch.table.nonzero_patterns)
            .arg(&scratch.table.bucket_delta)
            .arg(&scratch.table.fingerprint_delta)
            .arg(&join.base_entry)
            .arg(&join.hit_patterns)
            .arg(&table_bits)
            .arg(keep)
            .arg(&mut out.x)
            .arg(&mut out.z)
            .arg(&mut out.coeff)
            .arg(&mut out.fingerprint)
            .arg(&mut out.start)
            .arg(&mut out.lengths)
            .arg(&mut scratch.join_totals)
            .arg(&u32::from(histogram))
            .arg(&mut scratch.histogram)
            .launch(launch_grid(b as u32, JOIN_THREADS, (4 << table_bits) + histogram_bytes))?;
    }
    scratch.histogram_clean &= !histogram;
    scratch.histogram_fresh = histogram;
    timed!(scratch.lap(sum, t0, |stats| &mut stats.merge_ns)?);
    sum.swap_in(out, records as usize);
    scratch.extent = records as usize;
    sum.record_step(Growth::Join, Some(&scratch.join_totals))?;
    sum.debug_check();
    Ok(())
}
