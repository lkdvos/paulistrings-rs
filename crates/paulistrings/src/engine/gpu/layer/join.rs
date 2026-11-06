//! The join layer (`k_join` in `kernels/join.cu`): its table size, its launch, and its host flow straight into the output columns.

use crate::coefficient::Coefficient;
use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::engine::gpu::module::thread_per;
use crate::engine::gpu::scan::exclusive_scan;

use super::{timed_transfer, LayerScratch, Transfer};
use crate::engine::gpu::columns::DeviceColumns;
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::module::JOIN_THREADS;
use crate::engine::gpu::prepared::{DevicePrepared, JoinTable};
use crate::engine::gpu::sum::GpuSum;
use crate::engine::gpu::truncation::KeepProgram;
use crate::engine::timing::timed;

/// Slots of the join layer's table, as `log2`, for source buckets of up to `longest` rows: at most half full.
pub(in crate::engine::gpu) fn join_table_bits(longest: u32) -> u32 {
    (2 * longest.max(16)).next_power_of_two().trailing_zeros()
}

impl<const W: usize, C: Coefficient> LayerScratch<W, C> {
    /// `k_join_rows` and its scan into `segment_start`: each position's bound, the lengths of the two source buckets it reads; returns `(records, longest local bucket)` behind one synchronization.
    /// It replaces `k_count` and `k_rows` for a join, at the cost of output room for the rows the partner entry does not emit.
    pub(super) fn join_sizes(&mut self, sum: &GpuSum<W, C>) -> Result<(u32, u32), GpuError> {
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
        let known = self.longest;
        if known.is_none() {
            self.scan_lens(sum)?;
        }
        timed!(self.lap(sum, t, |stats| &mut stats.gather_ns)?);
        let (totals, length_totals) = (&self.totals, &self.length_totals);
        let (records, longest) = timed_transfer(stream, &mut self.stats, Transfer::D2h, || {
            let records = stream.clone_dtoh(totals)?[0];
            let longest = match known {
                Some(l) => l,
                None => stream.clone_dtoh(length_totals)?[1],
            };
            stream.synchronize()?;
            Ok((records, longest))
        })?;
        self.longest = Some(longest);
        Ok((records, longest))
    }
}

/// Apply `table`'s join to `sum` over every position, each output bucket written at its position's offset in `scratch.segment_start` from [`LayerScratch::join_sizes`] (`records` rows in all), for source buckets of up to `longest` rows; the previous columns become `sum.spare`.
/// The output keeps a gap after each bucket, so its extent is `records` and no compaction runs.
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
    let mut out = match sum.spare.take() {
        Some(spare) => spare,
        None => DeviceColumns::<W, C>::with_capacity(&stream, sum.device(), records as usize, b)?,
    };
    out.len = 0;
    out.buckets = 0;
    let need = records as usize;
    if need > out.term_capacity() {
        // A growing sum would otherwise reallocate at nearly every join; growing by half again keeps that logarithmic.
        let roomy = need.max(out.term_capacity() + out.term_capacity() / 2);
        if out.reserve(roomy, b).is_err() {
            out.reserve(need, b)?;
        }
    }
    out.reserve(need, b)?;
    stream.memset_zeros(&mut scratch.totals)?;
    let table_bits = join_table_bits(longest);
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
            .arg(&mut scratch.totals)
            .launch(LaunchConfig {
                grid_dim: (b as u32, 1, 1),
                block_dim: (JOIN_THREADS, 1, 1),
                shared_mem_bytes: 4 << table_bits,
            })?;
    }
    timed!(scratch.lap(sum, t0, |stats| &mut stats.merge_ns)?);
    let totals = &scratch.totals;
    let total = timed_transfer(&stream, &mut scratch.stats, Transfer::D2h, || {
        let v = stream.clone_dtoh(totals)?;
        stream.synchronize()?;
        Ok(v[0])
    })?;
    out.len = total as usize;
    out.buckets = b;
    sum.spare = Some(std::mem::replace(&mut sum.columns, out));
    scratch.extent = records as usize;
    sum.debug_check();
    Ok(())
}
