//! K7, [`ApproxTopN`](crate::truncation::ApproxTopN)'s layer pass on device: the octave histogram, the host edge walk after one `allreduce_sum_u64`, and the retain.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::columns::DeviceColumns;
use super::error::GpuError;
use super::layer::{grow, warp_per_bucket, LayerScratch};
use super::scan::exclusive_scan_with_max;
use super::sum::GpuSum;
use crate::engine::partitioned::transport::Collectives;
use crate::truncation::builtin::{octave_edge, EdgeDecision, APPROX_BINS};

/// Blocks the histogram launches at most; each block folds its warps' buckets into one shared histogram.
const HIST_BLOCKS: usize = 1024;

/// `[len, bin 0, …, bin 2047]` of `sum`, the packed layout of the host's collective `ApproxTopN` pass.
pub(crate) fn octave_histogram_device<const W: usize>(
    sum: &GpuSum<W>,
    scratch: &mut LayerScratch<W>,
) -> Result<Vec<u64>, GpuError> {
    let s = sum.stream.clone();
    let b = sum.hash.num_buckets();
    s.memset_zeros(&mut scratch.hist)?;
    let b32 = b as u32;
    let t0 = scratch.event(sum)?;
    // SAFETY: arguments match `k_octave_hist` in truncate.cu; `hist` holds `1 + APPROX_BINS` entries.
    unsafe {
        s.launch_builder(&sum.kernels.octave_hist)
            .arg(&sum.cols.coeff)
            .arg(&sum.cols.start)
            .arg(&sum.cols.lens)
            .arg(&b32)
            .arg(&mut scratch.hist)
            .launch(LaunchConfig {
                grid_dim: (b.div_ceil(8).clamp(1, HIST_BLOCKS) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    scratch.lap(sum, t0, |m| &mut m.truncate)?;
    let packed = s.clone_dtoh(&scratch.hist)?;
    s.synchronize()?;
    debug_assert_eq!(packed[0] as usize, sum.len(), "histogram term count");
    Ok(packed)
}

/// Apply `edge` to `sum` in place of [`retain_at_or_above`](crate::truncation::builtin::retain_at_or_above), keeping each bucket's order and start offset.
pub(crate) fn retain_at_or_above_device<const W: usize>(
    sum: &mut GpuSum<W>,
    scratch: &mut LayerScratch<W>,
    edge: EdgeDecision,
) -> Result<(), GpuError> {
    let threshold = match edge {
        EdgeDecision::KeepAll => return Ok(()),
        EdgeDecision::Clear => {
            let b = sum.hash.num_buckets();
            sum.stream
                .memset_zeros(&mut sum.cols.lens.slice_mut(0..b))?;
            sum.cols.len = 0;
            sum.debug_check();
            return Ok(());
        }
        EdgeDecision::AtOrAbove { threshold, .. } => threshold,
    };
    let s = sum.stream.clone();
    let k = sum.kernels.clone();
    let o = sum.device();
    let b = sum.hash.num_buckets();
    let extent = scratch.extent.max(sum.len());
    let mut out = match sum.spare.take() {
        Some(spare) => spare,
        None => DeviceColumns::<W>::with_capacity(&s, o, extent, b)?,
    };
    out.len = 0;
    out.buckets = 0;
    out.reserve(extent, b)?;
    grow(&s, &mut scratch.dst_off, b + 1, o)?;
    let b32 = b as u32;
    let t0 = scratch.event(sum)?;
    // SAFETY: arguments match `k_retain` in truncate.cu; `out` has room for the input's extent.
    unsafe {
        s.launch_builder(&k.retain)
            .arg(&sum.cols.x)
            .arg(&sum.cols.z)
            .arg(&sum.cols.coeff)
            .arg(&sum.cols.g)
            .arg(&sum.cols.start)
            .arg(&sum.cols.lens)
            .arg(&b32)
            .arg(&threshold)
            .arg(&mut out.x)
            .arg(&mut out.z)
            .arg(&mut out.coeff)
            .arg(&mut out.g)
            .arg(&mut out.start)
            .arg(&mut out.lens)
            .launch(warp_per_bucket(b))?;
    }
    let tot = exclusive_scan_with_max(
        &s,
        &k,
        &out.lens.slice(0..b),
        &mut scratch.dst_off.slice_mut(0..b + 1),
        b,
    )?;
    scratch.lap(sum, t0, |m| &mut m.truncate)?;
    let total = s.clone_dtoh(&tot)?[0];
    s.synchronize()?;
    out.len = total as usize;
    out.buckets = b;
    sum.spare = Some(std::mem::replace(&mut sum.cols, out));
    sum.debug_check();
    Ok(())
}

/// One `ApproxTopN(n)` layer pass, exactly the host's collective one: histogram, one `allreduce_sum_u64` of `[len, hist…]`, edge, retain.
///
/// `sum` is `None` on a partition that already failed; it still enters the reduction, with zeros, so the group stays in lock-step.
pub(crate) fn approx_top_n_device<const W: usize>(
    sum: Option<(&mut GpuSum<W>, &mut LayerScratch<W>)>,
    n: usize,
    coll: &dyn Collectives,
) -> Result<(), GpuError> {
    let Some((sum, scratch)) = sum else {
        coll.allreduce_sum_u64(&mut [0u64; 1 + APPROX_BINS]);
        return Ok(());
    };
    let local = octave_histogram_device(sum, scratch);
    let mut packed = match &local {
        Ok(h) => h.clone(),
        Err(_) => vec![0u64; 1 + APPROX_BINS],
    };
    coll.allreduce_sum_u64(&mut packed);
    local?;
    let edge = octave_edge(&packed[1..], packed[0] as usize, n);
    retain_at_or_above_device(sum, scratch, edge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::gpu::layer::GpuLayerOptions;
    use crate::pauli_sum::PauliSum;
    use crate::test_support::{rand_sum, tie_heavy_sum};
    use crate::truncation::builtin::{octave_histogram, retain_at_or_above};
    use num_complex::Complex64;

    fn device<const W: usize>(sum: &PauliSum<W>) -> (GpuSum<W>, LayerScratch<W>) {
        let dev = GpuSum::from_host(sum, 0).expect("upload");
        let scratch = LayerScratch::new(&dev, GpuLayerOptions::default()).expect("scratch");
        (dev, scratch)
    }

    fn fixtures<const W: usize>() -> Vec<(&'static str, PauliSum<W>)> {
        let nq = 64 * W;
        vec![
            ("rand 1e5", rand_sum::<W>(100_000, nq, 0x0C7A)),
            ("tie heavy", tie_heavy_sum::<W>(20_000, nq, 0x7135)),
            ("empty", PauliSum::<W>::empty(nq)),
        ]
    }

    fn histogram_matches<const W: usize>() {
        for (what, sum) in fixtures::<W>() {
            let (dev, mut scratch) = device(&sum);
            let got = octave_histogram_device(&dev, &mut scratch).expect("histogram");
            let want = octave_histogram(&sum);
            assert_eq!(got.len(), 1 + APPROX_BINS, "{what}");
            assert_eq!(got[0] as usize, sum.len(), "{what}: term count");
            let want: Vec<u64> = want.iter().map(|&c| u64::from(c)).collect();
            assert_eq!(got[1..], want[..], "{what}: bins");
        }
    }

    #[test]
    fn histogram_is_bitwise_the_hosts_w1() {
        crate::require_cuda!();
        histogram_matches::<1>();
    }

    #[test]
    fn histogram_is_bitwise_the_hosts_w2() {
        crate::require_cuda!();
        histogram_matches::<2>();
    }

    /// Complex coefficients, a subnormal square, and an exact power of two sitting on a bin edge.
    #[test]
    fn histogram_bins_edge_values_as_the_host() {
        crate::require_cuda!();
        let sum = PauliSum::<1>::from_sorted_columns(
            (0u64..5).map(|i| [i]).collect(),
            vec![[0u64]; 5],
            vec![
                Complex64::new(3.0, 4.0),
                Complex64::new(0.0, 2.0),
                Complex64::new(1e-160, 0.0),
                Complex64::new(1e-200, 1e-200),
                Complex64::new(-0.5, 0.5),
            ],
            8,
        );
        let (dev, mut scratch) = device(&sum);
        let got = octave_histogram_device(&dev, &mut scratch).expect("histogram");
        let want = octave_histogram(&sum);
        for (bin, &c) in want.iter().enumerate() {
            assert_eq!(got[1 + bin], u64::from(c), "bin {bin}");
        }
    }

    fn retain_matches<const W: usize>() {
        for (what, sum) in fixtures::<W>() {
            let len = sum.len();
            let hist = octave_histogram(&sum);
            for n in [0usize, 1, len / 3, len / 2, len.saturating_sub(1), len] {
                let edge = octave_edge(&hist, len, n);
                let mut want = sum.clone();
                retain_at_or_above(&mut want, edge);
                let (mut dev, mut scratch) = device(&sum);
                retain_at_or_above_device(&mut dev, &mut scratch, edge).expect("retain");
                assert_eq!(dev.len(), want.len(), "{what} n={n}: len");
                dev.assert_invariants_device()
                    .unwrap_or_else(|e| panic!("{what} n={n}: {e}"));
                let got = dev.to_host().expect("download");
                assert_eq!(got.to_arrays(), want.to_arrays(), "{what} n={n}: terms");
                if let EdgeDecision::AtOrAbove { kept, .. } = edge {
                    assert_eq!(dev.len(), kept, "{what} n={n}: histogram and predicate");
                }
            }
        }
    }

    #[test]
    fn retain_equals_the_hosts_term_for_term_w1() {
        crate::require_cuda!();
        retain_matches::<1>();
    }

    #[test]
    fn retain_equals_the_hosts_term_for_term_w2() {
        crate::require_cuda!();
        retain_matches::<2>();
    }

    /// A second retain reads the first one's loose layout (buckets at their old offsets, gaps between them) and reuses its spare columns.
    #[test]
    fn retain_twice_reuses_the_spare_and_stays_valid() {
        crate::require_cuda!();
        let sum = rand_sum::<1>(30_000, 64, 0x2E7);
        let (mut dev, mut scratch) = device(&sum);
        let mut host = sum.clone();
        for n in [20_000usize, 12_000] {
            let edge = octave_edge(&octave_histogram(&host), host.len(), n);
            retain_at_or_above(&mut host, edge);
            retain_at_or_above_device(&mut dev, &mut scratch, edge).expect("retain");
        }
        assert!(
            host.len() < 12_000 && !host.is_empty(),
            "both passes must cut without wiping"
        );
        assert_eq!(dev.to_host().unwrap().to_arrays(), host.to_arrays());
    }
}
