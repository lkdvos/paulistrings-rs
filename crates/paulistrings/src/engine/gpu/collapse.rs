//! [`CollapseSample`]'s layer pass on device: the systematic resample of `kernels/collapse.cu`, on one device or over a group's global prefix.
//!
//! The stretches are `|c|` rounded to a fixed point below `2^62` of the total, so every prefix is exact and the points of a bucket, of a partition and of the whole group partition `0..samples` without a rounding gap; the picks agree with the host's `f64` stretches up to ties at a boundary.
//! On one device the pass reads back two words behind one synchronization; a group adds one synchronization and one `f64` and one `u64` all-reduce.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};

use super::columns::grow;
use super::error::GpuError;
use super::finalize::whole_sum_launch;
use super::layer::{read_back, LayerScratch};
use super::module::{launch_grid, thread_per, warp_per_bucket};
use super::sum::GpuSum;
use crate::coefficient::Coefficient;
use crate::collectives::Collectives;
use crate::engine::timing::timed;
use crate::engine::LOG_TARGET;
use crate::truncation::collapse_sample::{assert_drawable, running_sum};
use crate::truncation::CollapseSample;

/// Values a scan block covers; must match `COLLAPSE_SCAN_BLOCK` in `kernels/collapse.cu`.
const SCAN_BLOCK: usize = 4096;

/// Grow-only buffers of one collapse.
pub(crate) struct CollapseScratch {
    /// The `f64` estimate of `Σ|c|` the fixed-point shift is taken from.
    total: CudaSlice<f64>,
    weights: CudaSlice<u64>,
    /// The exclusive scan of `weights`, `buckets + 1` entries.
    offsets: CudaSlice<u64>,
    block_sums: CudaSlice<u64>,
    /// `[offset, whole]` of `kernels/collapse.cu`.
    frame: CudaSlice<u64>,
    kept: CudaSlice<u64>,
}

impl CollapseScratch {
    pub(crate) fn new(stream: &Arc<CudaStream>) -> Result<Self, GpuError> {
        Ok(Self {
            total: stream.alloc_zeros(1)?,
            weights: stream.alloc_zeros(1)?,
            offsets: stream.alloc_zeros(2)?,
            block_sums: stream.alloc_zeros(1)?,
            frame: stream.alloc_zeros(2)?,
            kept: stream.alloc_zeros(1)?,
        })
    }
}

/// Enqueue `total = Σ|c|` of `sum`.
fn enqueue_total<const W: usize, C: Coefficient>(
    sum: &GpuSum<W, C>,
    scratch: &mut CollapseScratch,
) -> Result<(), GpuError> {
    let buckets = sum.hash.num_buckets();
    sum.stream.memset_zeros(&mut scratch.total)?;
    let buckets_u32 = buckets as u32;
    // SAFETY: arguments match `k_collapse_total` in collapse.cu; `total` holds one entry.
    unsafe {
        sum.stream
            .launch_builder(&sum.kernels.collapse_total)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&buckets_u32)
            .arg(&mut scratch.total)
            .launch(whole_sum_launch(buckets))?;
    }
    Ok(())
}

/// Enqueue the per-bucket fixed-point weights and their exclusive scan into `offsets`, which also sets `frame` to this sum alone, `[0, Σ weights]`.
fn enqueue_offsets<const W: usize, C: Coefficient>(
    sum: &GpuSum<W, C>,
    scratch: &mut CollapseScratch,
) -> Result<(), GpuError> {
    let (stream, kernels) = (&sum.stream, &sum.kernels);
    let buckets = sum.hash.num_buckets();
    let blocks = buckets.div_ceil(SCAN_BLOCK).max(1);
    if blocks > SCAN_BLOCK {
        return Err(GpuError::Unsupported("a collapse over more than 2^24 buckets"));
    }
    let ordinal = sum.device();
    grow(stream, &mut scratch.weights, buckets, ordinal)?;
    grow(stream, &mut scratch.offsets, buckets + 1, ordinal)?;
    grow(stream, &mut scratch.block_sums, blocks, ordinal)?;
    let (buckets_u32, blocks_u32) = (buckets as u32, blocks as u32);
    // SAFETY: arguments match the `k_collapse_*` signatures in collapse.cu; `weights` holds `buckets`, `offsets` `buckets + 1`, `block_sums` `blocks` and `frame` two entries.
    unsafe {
        stream
            .launch_builder(&kernels.collapse_weights)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&buckets_u32)
            .arg(&scratch.total)
            .arg(&mut scratch.weights)
            .launch(warp_per_bucket(buckets))?;
        stream
            .launch_builder(&kernels.collapse_scan_block)
            .arg(&scratch.weights)
            .arg(&mut scratch.offsets)
            .arg(&mut scratch.block_sums)
            .arg(&buckets_u32)
            .launch(launch_grid(blocks_u32, 1024, 0))?;
        stream
            .launch_builder(&kernels.collapse_scan_single)
            .arg(&mut scratch.block_sums)
            .arg(&blocks_u32)
            .arg(&mut scratch.offsets)
            .arg(&buckets_u32)
            .arg(&mut scratch.frame)
            .launch(launch_grid(1, 1024, 0))?;
        stream
            .launch_builder(&kernels.collapse_scan_add)
            .arg(&mut scratch.offsets)
            .arg(&scratch.block_sums)
            .arg(&buckets_u32)
            .launch(thread_per(buckets, 1024))?;
    }
    Ok(())
}

/// Enqueue the picks of the `samples` points at offset `u` against `frame` into the spare columns, then read back `(total, kept)` behind one synchronization and swap the picks in, keeping each bucket's start offset.
/// The swap waits for the read, so a sum whose `Σ|c|` is not drawable panics untouched, as the host's collapse does; more than `u32::MAX` points is [`GpuError::Unsupported`], the sum untouched.
fn pick<const W: usize, C: Coefficient>(
    sum: &mut GpuSum<W, C>,
    scratch: &mut LayerScratch<W, C>,
    u: f64,
    samples: usize,
    len: usize,
) -> Result<(), GpuError> {
    let samples = u32::try_from(samples)
        .map_err(|_| GpuError::Unsupported("a collapse of more than u32::MAX draws"))?;
    let stream = sum.stream.clone();
    let kernels = sum.kernels.clone();
    let buckets = sum.hash.num_buckets();
    let extent = scratch.extent.max(sum.len());
    let mut out = sum.take_spare(extent, buckets)?;
    let collapse = &mut scratch.collapse;
    stream.memset_zeros(&mut collapse.kept)?;
    let buckets_u32 = buckets as u32;
    // SAFETY: arguments match `k_collapse_pick` in collapse.cu; `out` has room for the input's extent, a bucket's picks never outnumber its terms, and `offsets` holds `buckets + 1` entries.
    unsafe {
        stream
            .launch_builder(&kernels.collapse_pick)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.fingerprint)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&buckets_u32)
            .arg(&collapse.offsets)
            .arg(&collapse.total)
            .arg(&collapse.frame)
            .arg(&u)
            .arg(&samples)
            .arg(&mut out.x)
            .arg(&mut out.z)
            .arg(&mut out.coeff)
            .arg(&mut out.fingerprint)
            .arg(&mut out.start)
            .arg(&mut out.lengths)
            .arg(&mut collapse.kept)
            .launch(warp_per_bucket(buckets))?;
    }
    let total = stream.clone_dtoh(&collapse.total)?;
    let kept = stream.clone_dtoh(&collapse.kept)?;
    stream.synchronize()?;
    if !(total[0].is_finite() && total[0] > 0.0) {
        sum.spare = Some(out);
        assert_drawable(total[0], len);
        unreachable!("assert_drawable panics on a total that is not positive and finite");
    }
    sum.swap_in(out, kept[0] as usize);
    sum.record_exact(kept[0] as usize, None);
    sum.debug_check();
    Ok(())
}

/// The systematic resample of `sum` alone at offset `u` with `samples` points; the panic of a non-drawable `Σ|c|` as the host's.
pub(crate) fn resample_device<const W: usize, C: Coefficient>(
    sum: &mut GpuSum<W, C>,
    scratch: &mut LayerScratch<W, C>,
    u: f64,
    samples: usize,
) -> Result<(), GpuError> {
    let len = sum.len();
    timed!(let started = scratch.event(sum)?);
    enqueue_total(sum, &mut scratch.collapse)?;
    enqueue_offsets(sum, &mut scratch.collapse)?;
    pick(sum, scratch, u, samples, len)?;
    timed!(scratch.lap(sum, started, |stats| &mut stats.truncate_ns)?);
    Ok(())
}

/// One `CollapseSample` layer pass, the host's [`finalize_layer`](crate::TruncationPolicy::finalize_layer) on one device and its [`finalize_layer_partitioned`](crate::TruncationPolicy::finalize_layer_partitioned) over a group.
/// `parts` is `None` on a partition that already failed; above one partition it still enters every reduction, with zeros, so the group stays in lock-step.
pub(crate) fn collapse_sample_device<const W: usize, C: Coefficient>(
    parts: Option<(&mut GpuSum<W, C>, &mut LayerScratch<W, C>)>,
    policy: &CollapseSample,
    collectives: &dyn Collectives,
) -> Result<(), GpuError> {
    if collectives.size() == 1 {
        let Some((sum, scratch)) = parts else {
            return Ok(());
        };
        let call = policy.next_call();
        if sum.len() > policy.cache {
            sum.settle_lengths()?;
        }
        let len = sum.len();
        if len <= policy.cache {
            return Ok(());
        }
        resample_device(sum, scratch, policy.uniform(call), policy.samples)?;
        let n = policy.count_collapse();
        log::debug!(
            target: LOG_TARGET,
            "collapse_sample (device): {len} terms, {} draws kept {} (collapse {n})",
            policy.samples,
            sum.len(),
        );
        return Ok(());
    }
    group_collapse(parts, policy, collectives)
}

/// [`collapse_sample_device`] above one partition: `[len, pass]`, then the per-partition `Σ|c|` estimates that fix the group's shift, then the per-partition fixed-point totals that place this partition's stretch.
fn group_collapse<const W: usize, C: Coefficient>(
    mut parts: Option<(&mut GpuSum<W, C>, &mut LayerScratch<W, C>)>,
    policy: &CollapseSample,
    collectives: &dyn Collectives,
) -> Result<(), GpuError> {
    let (rank, size) = (collectives.rank() as usize, collectives.size() as usize);
    let call = if rank == 0 { policy.next_call() } else { 0 };
    let settled = match parts.as_mut() {
        Some((sum, _)) => sum.settle_lengths(),
        None => Ok(()),
    };
    let local_len = parts.as_ref().map_or(0, |(sum, _)| sum.len());
    let mut head = [local_len as u64, call];
    collectives.allreduce_sum_u64(&mut head);
    let [len, call] = head;
    if len as usize <= policy.cache {
        return Ok(());
    }

    let mut failure = Ok(());
    let mut estimates = vec![0.0f64; size];
    if let Some((sum, scratch)) = parts.as_mut() {
        let total = enqueue_total(sum, &mut scratch.collapse)
            .and_then(|()| read_back(&sum.stream, &mut scratch.stats, &scratch.collapse.total));
        match total {
            Ok(total) => estimates[rank] = total[0],
            Err(e) => failure = Err(e),
        }
    }
    collectives.allreduce_sum_f64(&mut estimates);
    let estimate = running_sum(estimates.iter().copied());
    assert_drawable(estimate, len as usize);

    let mut wholes = vec![0u64; size];
    if let (Some((sum, scratch)), Ok(())) = (parts.as_mut(), &failure) {
        // This partition's fixed-point total under the group's estimate; its bucket offsets stay for the pick.
        let frame = sum
            .stream
            .memcpy_htod(&[estimate], &mut scratch.collapse.total)
            .map_err(GpuError::from)
            .and_then(|()| enqueue_offsets(sum, &mut scratch.collapse))
            .and_then(|()| read_back(&sum.stream, &mut scratch.stats, &scratch.collapse.frame));
        match frame {
            Ok(frame) => wholes[rank] = frame[1],
            Err(e) => failure = Err(e),
        }
    }
    collectives.allreduce_sum_u64(&mut wholes);
    settled?;
    failure?;
    let Some((sum, scratch)) = parts else {
        return Ok(());
    };
    let frame = [wholes[..rank].iter().sum::<u64>(), wholes.iter().sum::<u64>()];
    sum.stream.memcpy_htod(&frame, &mut scratch.collapse.frame)?;
    let local_len = sum.len();
    pick(sum, scratch, policy.uniform(call), policy.samples, local_len)?;
    if rank == 0 {
        let collapses = policy.count_collapse();
        log::debug!(
            target: LOG_TARGET,
            "collapse_sample (device): {len} terms over {size} partitions, sum |c| ~ {estimate:.6e}, {} draws (collapse {collapses})",
            policy.samples,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;
