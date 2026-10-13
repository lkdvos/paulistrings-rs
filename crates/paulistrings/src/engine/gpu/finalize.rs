//! [`ApproxTopN`](crate::truncation::ApproxTopN)'s layer pass on device by octave histogram, and exact [`TopN`](crate::truncation::TopN)'s on one device by radix-select.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::columns::{grow, DeviceColumns};
use super::error::GpuError;
use super::layer::LayerScratch;
use super::module::warp_per_bucket;
use super::scan::exclusive_scan;
use super::sum::GpuSum;
use crate::collectives::Collectives;
use crate::engine::timing::timed;
use crate::truncation::builtin::{local_boundary_quota, octave_edge, EdgeDecision, APPROX_BINS};

/// Blocks the histogram launches at most; each block folds its warps' buckets into one shared histogram.
const HISTOGRAM_BLOCKS: usize = 1024;

/// The launch shape the histogram and radix-select passes share: warps spread over buckets, capped at `HISTOGRAM_BLOCKS`.
fn whole_sum_launch(buckets: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (buckets.div_ceil(8).clamp(1, HISTOGRAM_BLOCKS) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `[len, bin 0, …, bin 2047]` of `sum`, the packed layout of the host's collective `ApproxTopN` pass.
fn octave_histogram_device<const W: usize>(
    sum: &GpuSum<W>,
    scratch: &mut LayerScratch<W>,
) -> Result<Vec<u64>, GpuError> {
    let stream = sum.stream.clone();
    let buckets = sum.hash.num_buckets();
    stream.memset_zeros(&mut scratch.histogram)?;
    let buckets_u32 = buckets as u32;
    timed!(let started = scratch.event(sum)?);
    // SAFETY: arguments match `k_octave_histogram` in truncate.cu; `histogram` holds `1 + APPROX_BINS` entries.
    unsafe {
        stream
            .launch_builder(&sum.kernels.octave_histogram)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&buckets_u32)
            .arg(&mut scratch.histogram)
            .launch(whole_sum_launch(buckets))?;
    }
    timed!(scratch.lap(sum, started, |stats| &mut stats.truncate_ns)?);
    let packed = stream.clone_dtoh(&scratch.histogram)?;
    stream.synchronize()?;
    debug_assert_eq!(packed[0] as usize, sum.len(), "histogram term count");
    Ok(packed)
}

/// Clear `sum` on device.
fn clear_device<const W: usize>(sum: &mut GpuSum<W>) -> Result<(), GpuError> {
    let buckets = sum.hash.num_buckets();
    sum.stream
        .memset_zeros(&mut sum.columns.lengths.slice_mut(0..buckets))?;
    sum.columns.len = 0;
    sum.debug_check();
    Ok(())
}

/// Apply `edge` to `sum` in place of [`retain_at_edge`](crate::truncation::builtin::retain_at_edge), keeping `local_quota` boundary-octave terms, each bucket's order and its start offset.
fn retain_at_edge_device<const W: usize>(
    sum: &mut GpuSum<W>,
    scratch: &mut LayerScratch<W>,
    edge: EdgeDecision,
    local_quota: usize,
) -> Result<(), GpuError> {
    match edge {
        EdgeDecision::KeepAll => Ok(()),
        EdgeDecision::Clear => clear_device(sum),
        EdgeDecision::Boundary { bin, .. } => {
            retain_device(sum, scratch, bin as u64, 52, local_quota as u64)
        }
    }
}

/// One `ApproxTopN(n)` layer pass, exactly the host's collective one: histogram, one `allreduce_sum_u64` of `[len, histogram…]`, the edge, at a boundary one more of the per-rank boundary counts, retain.
/// `sum` is `None` on a partition that already failed; it still enters both reductions, with zeros, so the group stays in lock-step.
pub(crate) fn approx_top_n_device<const W: usize>(
    sum: Option<(&mut GpuSum<W>, &mut LayerScratch<W>)>,
    n: usize,
    collectives: &dyn Collectives,
) -> Result<(), GpuError> {
    let Some((sum, scratch)) = sum else {
        let mut packed = [0u64; 1 + APPROX_BINS];
        collectives.allreduce_sum_u64(&mut packed);
        if let EdgeDecision::Boundary { quota, .. } =
            octave_edge(&packed[1..], packed[0] as usize, n)
        {
            if collectives.size() > 1 {
                local_boundary_quota(collectives, 0, quota);
            }
        }
        return Ok(());
    };
    let local = octave_histogram_device(sum, scratch);
    let mut packed = match &local {
        Ok(histogram) => histogram.clone(),
        Err(_) => vec![0u64; 1 + APPROX_BINS],
    };
    collectives.allreduce_sum_u64(&mut packed);
    let edge = octave_edge(&packed[1..], packed[0] as usize, n);
    let local_quota = match edge {
        EdgeDecision::Boundary { bin, quota, .. } if collectives.size() > 1 => {
            let local_count = local.as_ref().map_or(0, |histogram| histogram[1 + bin]);
            local_boundary_quota(collectives, local_count as usize, quota)
        }
        EdgeDecision::Boundary { quota, .. } => quota,
        EdgeDecision::KeepAll | EdgeDecision::Clear => 0,
    };
    local?;
    retain_at_edge_device(sum, scratch, edge, local_quota)
}

/// The radix-select: the exact bit pattern of the `n`-th largest `|c|²` (1-indexed from the top), one 8-bit digit per pass from the top byte down, and its rank among the terms tied at it, which is how many of them `TopN(n)` keeps.
fn radix_select_bits<const W: usize>(
    sum: &GpuSum<W>,
    scratch: &mut LayerScratch<W>,
    n: usize,
) -> Result<(u64, u64), GpuError> {
    let stream = sum.stream.clone();
    let buckets = sum.hash.num_buckets() as u32;
    let mut prefix: u64 = 0;
    let mut remaining: u64 = n as u64;
    for pass in 0u32..8 {
        let shift = 56 - 8 * pass;
        let fixed_mask: u64 = if pass == 0 {
            0
        } else {
            u64::MAX << (64 - 8 * pass)
        };
        stream.memset_zeros(&mut scratch.radix_histogram)?;
        // SAFETY: arguments match `k_radix_histogram` in truncate.cu; `radix_histogram` holds 256 entries.
        unsafe {
            stream
                .launch_builder(&sum.kernels.radix_histogram)
                .arg(&sum.columns.coeff)
                .arg(&sum.columns.start)
                .arg(&sum.columns.lengths)
                .arg(&buckets)
                .arg(&fixed_mask)
                .arg(&prefix)
                .arg(&shift)
                .arg(&mut scratch.radix_histogram)
                .launch(whole_sum_launch(buckets as usize))?;
        }
        let histogram = stream.clone_dtoh(&scratch.radix_histogram)?;
        stream.synchronize()?;
        let mut cumulative = 0u64;
        let mut chosen: Option<u32> = None;
        for digit in (0..256u32).rev() {
            let count = histogram[digit as usize];
            if cumulative + count >= remaining {
                chosen = Some(digit);
                break;
            }
            cumulative += count;
        }
        let chosen = chosen.expect("radix select: remaining exceeds the population");
        let group = histogram[chosen as usize];
        prefix |= (u64::from(chosen)) << shift;
        remaining -= cumulative;
        if pass == 7 {
            return Ok((prefix, remaining));
        }
        // A digit of population one names one term, whose remaining bits are read directly.
        if group == 1 {
            let mask = fixed_mask | (0xFFu64 << shift);
            stream.memset_zeros(&mut scratch.radix_out.slice_mut(0..1))?;
            // SAFETY: arguments match `k_radix_extract` in truncate.cu; exactly one term matches `mask`/`prefix`.
            unsafe {
                stream
                    .launch_builder(&sum.kernels.radix_extract)
                    .arg(&sum.columns.coeff)
                    .arg(&sum.columns.start)
                    .arg(&sum.columns.lengths)
                    .arg(&buckets)
                    .arg(&mask)
                    .arg(&prefix)
                    .arg(&mut scratch.radix_out)
                    .launch(whole_sum_launch(buckets as usize))?;
            }
            let out = stream.clone_dtoh(&scratch.radix_out.slice(0..1))?;
            stream.synchronize()?;
            return Ok((out[0], remaining));
        }
    }
    unreachable!("the loop returns by pass 7")
}

/// Keep the terms whose `|c|²` bits shifted right by `shift` exceed `boundary`, plus the first `quota` equal to it in bucket order, in each bucket's order and at its start offset.
/// The quota is split by a per-bucket count and an exclusive scan, so the kept set is deterministic.
fn retain_device<const W: usize>(
    sum: &mut GpuSum<W>,
    scratch: &mut LayerScratch<W>,
    boundary: u64,
    shift: u32,
    quota: u64,
) -> Result<(), GpuError> {
    let stream = sum.stream.clone();
    let kernels = sum.kernels.clone();
    let ordinal = sum.device();
    let buckets = sum.hash.num_buckets();
    let extent = scratch.extent.max(sum.len());
    let mut out = match sum.spare.take() {
        Some(spare) => spare,
        None => DeviceColumns::<W>::with_capacity(&stream, ordinal, extent, buckets)?,
    };
    out.len = 0;
    out.buckets = 0;
    out.reserve(extent, buckets)?;
    grow(&stream, &mut scratch.counts, buckets, ordinal)?;
    grow(
        &stream,
        &mut scratch.destination_offsets,
        buckets + 1,
        ordinal,
    )?;
    let buckets_u32 = buckets as u32;
    timed!(let started = scratch.event(sum)?);
    // SAFETY: arguments match `k_boundary_counts` in truncate.cu; `counts` holds `buckets` entries.
    unsafe {
        stream
            .launch_builder(&kernels.boundary_counts)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&buckets_u32)
            .arg(&boundary)
            .arg(&shift)
            .arg(&mut scratch.counts)
            .launch(warp_per_bucket(buckets))?;
    }
    // `destination_offsets` holds the boundary terms before each bucket until `k_retain` has read it, then the output offsets.
    exclusive_scan(
        &stream,
        &kernels,
        &scratch.counts.slice(0..buckets),
        &mut scratch.destination_offsets.slice_mut(0..buckets + 1),
        buckets,
        &mut scratch.scan,
        &mut scratch.totals,
    )?;
    // SAFETY: arguments match `k_retain` in truncate.cu; `out` has room for the input's extent and `destination_offsets` holds `buckets + 1` entries.
    unsafe {
        stream
            .launch_builder(&kernels.retain)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.fingerprint)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&buckets_u32)
            .arg(&boundary)
            .arg(&shift)
            .arg(&scratch.destination_offsets)
            .arg(&quota)
            .arg(&mut out.x)
            .arg(&mut out.z)
            .arg(&mut out.coeff)
            .arg(&mut out.fingerprint)
            .arg(&mut out.start)
            .arg(&mut out.lengths)
            .launch(warp_per_bucket(buckets))?;
    }
    exclusive_scan(
        &stream,
        &kernels,
        &out.lengths.slice(0..buckets),
        &mut scratch.destination_offsets.slice_mut(0..buckets + 1),
        buckets,
        &mut scratch.scan,
        &mut scratch.totals,
    )?;
    timed!(scratch.lap(sum, started, |stats| &mut stats.truncate_ns)?);
    let total = stream.clone_dtoh(&scratch.totals)?[0];
    stream.synchronize()?;
    out.len = total as usize;
    out.buckets = buckets;
    sum.spare = Some(std::mem::replace(&mut sum.columns, out));
    sum.debug_check();
    Ok(())
}

/// One `TopN(n)` layer pass on one device: [`radix_select_bits`] gives the `n`-th largest `|c|²` and how many terms tied at it fit, then [`retain_device`].
pub(crate) fn top_n_device<const W: usize>(
    sum: &mut GpuSum<W>,
    scratch: &mut LayerScratch<W>,
    n: usize,
) -> Result<(), GpuError> {
    if sum.len() <= n {
        return Ok(());
    }
    if n == 0 {
        return clear_device(sum);
    }
    timed!(let started = scratch.event(sum)?);
    let (threshold, tied_kept) = radix_select_bits(sum, scratch, n)?;
    timed!(scratch.lap(sum, started, |stats| &mut stats.truncate_ns)?);
    retain_device(sum, scratch, threshold, 0, tied_kept)
}

#[cfg(test)]
mod tests;
