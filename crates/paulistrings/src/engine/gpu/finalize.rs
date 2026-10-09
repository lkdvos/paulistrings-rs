//! K7, [`ApproxTopN`](crate::truncation::ApproxTopN)'s layer pass on device, and K8, exact [`TopN`](crate::truncation::TopN)'s on one device.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::columns::{grow, DeviceColumns};
use super::error::GpuError;
use super::layer::LayerScratch;
use super::module::warp_per_bucket;
use super::scan::exclusive_scan;
use super::sum::GpuSum;
use crate::collectives::Collectives;
use crate::truncation::builtin::{octave_edge, EdgeDecision, APPROX_BINS};

/// Blocks the histogram launches at most; each block folds its warps' buckets into one shared histogram.
const HIST_BLOCKS: usize = 1024;

/// The launch shape K7 and K8's whole-sum passes share: warps spread over buckets, capped at `HIST_BLOCKS`.
fn whole_sum_launch(b: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (b.div_ceil(8).clamp(1, HIST_BLOCKS) as u32, 1, 1),
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
    let b = sum.hash.num_buckets();
    stream.memset_zeros(&mut scratch.hist)?;
    let b32 = b as u32;
    let started = scratch.event(sum)?;
    // SAFETY: arguments match `k_octave_hist` in truncate.cu; `hist` holds `1 + APPROX_BINS` entries.
    unsafe {
        stream
            .launch_builder(&sum.kernels.octave_histogram)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lens)
            .arg(&b32)
            .arg(&mut scratch.hist)
            .launch(whole_sum_launch(b))?;
    }
    scratch.lap(sum, started, |m| &mut m.truncate)?;
    let packed = stream.clone_dtoh(&scratch.hist)?;
    stream.synchronize()?;
    debug_assert_eq!(packed[0] as usize, sum.len(), "histogram term count");
    Ok(packed)
}

/// Apply `edge` to `sum` in place of [`retain_at_or_above`](crate::truncation::builtin::retain_at_or_above), keeping each bucket's order and start offset.
fn retain_at_or_above_device<const W: usize>(
    sum: &mut GpuSum<W>,
    scratch: &mut LayerScratch<W>,
    edge: EdgeDecision,
) -> Result<(), GpuError> {
    match edge {
        EdgeDecision::KeepAll => Ok(()),
        EdgeDecision::Clear => {
            let b = sum.hash.num_buckets();
            sum.stream
                .memset_zeros(&mut sum.columns.lens.slice_mut(0..b))?;
            sum.columns.len = 0;
            sum.debug_check();
            Ok(())
        }
        // A non-negative threshold orders as its bits, so `>= threshold` is `> bits || == bits`.
        EdgeDecision::AtOrAbove { threshold, .. } => {
            retain_device(sum, scratch, threshold.to_bits(), true)
        }
    }
}

/// One `ApproxTopN(n)` layer pass, exactly the host's collective one: histogram, one `allreduce_sum_u64` of `[len, hist…]`, edge, retain.
/// `sum` is `None` on a partition that already failed; it still enters the reduction, with zeros, so the group stays in lock-step.
pub(crate) fn approx_top_n_device<const W: usize>(
    sum: Option<(&mut GpuSum<W>, &mut LayerScratch<W>)>,
    n: usize,
    collectives: &dyn Collectives,
) -> Result<(), GpuError> {
    let Some((sum, scratch)) = sum else {
        collectives.allreduce_sum_u64(&mut [0u64; 1 + APPROX_BINS]);
        return Ok(());
    };
    let local = octave_histogram_device(sum, scratch);
    let mut packed = match &local {
        Ok(histogram) => histogram.clone(),
        Err(_) => vec![0u64; 1 + APPROX_BINS],
    };
    collectives.allreduce_sum_u64(&mut packed);
    local?;
    let edge = octave_edge(&packed[1..], packed[0] as usize, n);
    retain_at_or_above_device(sum, scratch, edge)
}

/// K8's radix-select: the exact bit pattern of the `n`-th largest `|c|²` (1-indexed from the top), one 8-bit digit per pass from the top byte down.
fn radix_select_bits<const W: usize>(
    sum: &GpuSum<W>,
    scratch: &mut LayerScratch<W>,
    n: usize,
) -> Result<u64, GpuError> {
    let stream = sum.stream.clone();
    let b = sum.hash.num_buckets() as u32;
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
        // SAFETY: arguments match `k_radix_hist` in truncate.cu; `radix_histogram` holds 256 entries.
        unsafe {
            stream
                .launch_builder(&sum.kernels.radix_histogram)
                .arg(&sum.columns.coeff)
                .arg(&sum.columns.start)
                .arg(&sum.columns.lens)
                .arg(&b)
                .arg(&fixed_mask)
                .arg(&prefix)
                .arg(&shift)
                .arg(&mut scratch.radix_histogram)
                .launch(whole_sum_launch(b as usize))?;
        }
        let hist = stream.clone_dtoh(&scratch.radix_histogram)?;
        stream.synchronize()?;
        let mut cumulative = 0u64;
        let mut chosen: Option<u32> = None;
        for digit in (0..256u32).rev() {
            let count = hist[digit as usize];
            if cumulative + count >= remaining {
                chosen = Some(digit);
                break;
            }
            cumulative += count;
        }
        let chosen = chosen.expect("radix select: remaining exceeds the population");
        let group = hist[chosen as usize];
        prefix |= (u64::from(chosen)) << shift;
        remaining -= cumulative;
        if pass == 7 {
            return Ok(prefix);
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
                    .arg(&sum.columns.lens)
                    .arg(&b)
                    .arg(&mask)
                    .arg(&prefix)
                    .arg(&mut scratch.radix_out)
                    .launch(whole_sum_launch(b as usize))?;
            }
            let out = stream.clone_dtoh(&scratch.radix_out.slice(0..1))?;
            stream.synchronize()?;
            return Ok(out[0]);
        }
    }
    unreachable!("the loop returns by pass 7")
}

/// Keep the terms whose `|c|²` bits exceed `threshold`, and the tie group at `threshold` when `keep_tied`, in each bucket's order and at its start offset.
fn retain_device<const W: usize>(
    sum: &mut GpuSum<W>,
    scratch: &mut LayerScratch<W>,
    threshold: u64,
    keep_tied: bool,
) -> Result<(), GpuError> {
    let stream = sum.stream.clone();
    let kernels = sum.kernels.clone();
    let ordinal = sum.device();
    let b = sum.hash.num_buckets();
    let extent = scratch.extent.max(sum.len());
    let mut out = match sum.spare.take() {
        Some(spare) => spare,
        None => DeviceColumns::<W>::with_capacity(&stream, ordinal, extent, b)?,
    };
    out.len = 0;
    out.buckets = 0;
    out.reserve(extent, b)?;
    grow(&stream, &mut scratch.destination_offsets, b + 1, ordinal)?;
    let b32 = b as u32;
    let keep_tied_u32 = u32::from(keep_tied);
    let started = scratch.event(sum)?;
    // SAFETY: arguments match `k_retain` in truncate.cu; `out` has room for the input's extent.
    unsafe {
        stream
            .launch_builder(&kernels.retain)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.g)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lens)
            .arg(&b32)
            .arg(&threshold)
            .arg(&keep_tied_u32)
            .arg(&mut out.x)
            .arg(&mut out.z)
            .arg(&mut out.coeff)
            .arg(&mut out.g)
            .arg(&mut out.start)
            .arg(&mut out.lens)
            .launch(warp_per_bucket(b))?;
    }
    exclusive_scan(
        &stream,
        &kernels,
        &out.lens.slice(0..b),
        &mut scratch.destination_offsets.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut scratch.totals,
    )?;
    scratch.lap(sum, started, |m| &mut m.truncate)?;
    let total = stream.clone_dtoh(&scratch.totals)?[0];
    stream.synchronize()?;
    out.len = total as usize;
    out.buckets = b;
    sum.spare = Some(std::mem::replace(&mut sum.columns, out));
    sum.debug_check();
    Ok(())
}

/// One `TopN(n)` layer pass on one device: [`radix_select_bits`], the two counts the tie rule needs, then [`retain_device`].
pub(crate) fn top_n_device<const W: usize>(
    sum: &mut GpuSum<W>,
    scratch: &mut LayerScratch<W>,
    n: usize,
) -> Result<(), GpuError> {
    let total = sum.len();
    if total <= n {
        return Ok(());
    }
    if n == 0 {
        return retain_at_or_above_device(sum, scratch, EdgeDecision::Clear);
    }
    let threshold = radix_select_bits(sum, scratch, n)?;
    let stream = sum.stream.clone();
    let b = sum.hash.num_buckets() as u32;
    stream.memset_zeros(&mut scratch.radix_out)?;
    let started = scratch.event(sum)?;
    // SAFETY: arguments match `k_topn_counts` in truncate.cu; `radix_out` holds 2 entries.
    unsafe {
        stream
            .launch_builder(&sum.kernels.topn_counts)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lens)
            .arg(&b)
            .arg(&threshold)
            .arg(&mut scratch.radix_out)
            .launch(whole_sum_launch(b as usize))?;
    }
    scratch.lap(sum, started, |m| &mut m.truncate)?;
    let counts = stream.clone_dtoh(&scratch.radix_out.slice(0..2))?;
    stream.synchronize()?;
    let (above, equal) = (counts[0], counts[1]);
    retain_device(sum, scratch, threshold, above + equal <= n as u64)
}

#[cfg(test)]
mod tests;
