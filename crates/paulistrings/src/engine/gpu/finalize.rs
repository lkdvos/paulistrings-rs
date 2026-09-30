//! K7, [`ApproxTopN`](crate::truncation::ApproxTopN)'s layer pass on device, and K8, exact [`TopN`](crate::truncation::TopN)'s on one device.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::columns::{grow, DeviceColumns};
use super::error::GpuError;
use super::layer::LayerScratch;
use super::module::warp_per_bucket;
use super::scan::exclusive_scan;
use super::sum::GpuSum;
use crate::engine::partitioned::transport::Collectives;
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
            .launch(whole_sum_launch(b))?;
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
    match edge {
        EdgeDecision::KeepAll => Ok(()),
        EdgeDecision::Clear => {
            let b = sum.hash.num_buckets();
            sum.stream
                .memset_zeros(&mut sum.cols.lens.slice_mut(0..b))?;
            sum.cols.len = 0;
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

/// K8's radix-select: the exact bit pattern of the `n`-th largest `|c|²` (1-indexed from the top), one 8-bit digit per pass from the top byte down, `TopN::finalize_layer`'s `select_nth_unstable_by` at the bit level.
/// A digit whose population is `1` names one term, whose remaining bits `k_radix_extract` reads directly.
fn radix_select_bits<const W: usize>(
    sum: &GpuSum<W>,
    scratch: &mut LayerScratch<W>,
    n: usize,
) -> Result<u64, GpuError> {
    let s = sum.stream.clone();
    let b = sum.hash.num_buckets() as u32;
    let mut prefix: u64 = 0;
    let mut remaining: u64 = n as u64;
    for p in 0u32..8 {
        let shift = 56 - 8 * p;
        let fixed_mask: u64 = if p == 0 { 0 } else { u64::MAX << (64 - 8 * p) };
        s.memset_zeros(&mut scratch.radix_hist)?;
        // SAFETY: arguments match `k_radix_hist` in truncate.cu; `radix_hist` holds 256 entries.
        unsafe {
            s.launch_builder(&sum.kernels.radix_hist)
                .arg(&sum.cols.coeff)
                .arg(&sum.cols.start)
                .arg(&sum.cols.lens)
                .arg(&b)
                .arg(&fixed_mask)
                .arg(&prefix)
                .arg(&shift)
                .arg(&mut scratch.radix_hist)
                .launch(whole_sum_launch(b as usize))?;
        }
        let hist = s.clone_dtoh(&scratch.radix_hist)?;
        s.synchronize()?;
        let mut cum = 0u64;
        let mut chosen: Option<u32> = None;
        for d in (0..256u32).rev() {
            let count = hist[d as usize];
            if cum + count >= remaining {
                chosen = Some(d);
                break;
            }
            cum += count;
        }
        let chosen = chosen.expect("radix select: remaining exceeds the population");
        let group = hist[chosen as usize];
        prefix |= (u64::from(chosen)) << shift;
        remaining -= cum;
        if p == 7 {
            return Ok(prefix);
        }
        if group == 1 {
            let mask = fixed_mask | (0xFFu64 << shift);
            s.memset_zeros(&mut scratch.radix_out.slice_mut(0..1))?;
            // SAFETY: arguments match `k_radix_extract` in truncate.cu; exactly one term matches `mask`/`prefix`.
            unsafe {
                s.launch_builder(&sum.kernels.radix_extract)
                    .arg(&sum.cols.coeff)
                    .arg(&sum.cols.start)
                    .arg(&sum.cols.lens)
                    .arg(&b)
                    .arg(&mask)
                    .arg(&prefix)
                    .arg(&mut scratch.radix_out)
                    .launch(whole_sum_launch(b as usize))?;
            }
            let out = s.clone_dtoh(&scratch.radix_out.slice(0..1))?;
            s.synchronize()?;
            return Ok(out[0]);
        }
    }
    unreachable!("the loop returns by pass 7")
}

/// Keep the terms whose `|c|²` bits exceed `t2`, and the tie group at `t2` when `keep_tied`, in each bucket's order and at its start offset.
fn retain_device<const W: usize>(
    sum: &mut GpuSum<W>,
    scratch: &mut LayerScratch<W>,
    t2: u64,
    keep_tied: bool,
) -> Result<(), GpuError> {
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
    let keep_tied_u32 = u32::from(keep_tied);
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
            .arg(&t2)
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
        &s,
        &k,
        &out.lens.slice(0..b),
        &mut scratch.dst_off.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut scratch.tot_a,
    )?;
    scratch.lap(sum, t0, |m| &mut m.truncate)?;
    let total = s.clone_dtoh(&scratch.tot_a)?[0];
    s.synchronize()?;
    out.len = total as usize;
    out.buckets = b;
    sum.spare = Some(std::mem::replace(&mut sum.cols, out));
    sum.debug_check();
    Ok(())
}

/// One `TopN(n)` layer pass on one device: [`radix_select_bits`], the two global counts the tie rule needs, then [`retain_device`], `TopN::finalize_layer`'s three passes.
/// A group member never calls it (`DevicePartition::finalize_layer`), since the group's `n`-th largest has no collective form.
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
    let t2 = radix_select_bits(sum, scratch, n)?;
    let s = sum.stream.clone();
    let b = sum.hash.num_buckets() as u32;
    s.memset_zeros(&mut scratch.radix_out)?;
    let t0 = scratch.event(sum)?;
    // SAFETY: arguments match `k_topn_counts` in truncate.cu; `radix_out` holds 2 entries.
    unsafe {
        s.launch_builder(&sum.kernels.topn_counts)
            .arg(&sum.cols.coeff)
            .arg(&sum.cols.start)
            .arg(&sum.cols.lens)
            .arg(&b)
            .arg(&t2)
            .arg(&mut scratch.radix_out)
            .launch(whole_sum_launch(b as usize))?;
    }
    scratch.lap(sum, t0, |m| &mut m.truncate)?;
    let counts = s.clone_dtoh(&scratch.radix_out.slice(0..2))?;
    s.synchronize()?;
    let (above, equal) = (counts[0], counts[1]);
    retain_device(sum, scratch, t2, above + equal <= n as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::gpu::layer::GpuLayerOptions;
    use crate::pauli_sum::PauliSum;
    use crate::test_support::{rand_sum, tie_heavy_sum};
    use crate::truncation::builtin::{octave_histogram, retain_at_or_above};
    use crate::TruncationPolicy;
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
    fn histogram_is_bitwise_the_hosts() {
        crate::require_cuda!();
        histogram_matches::<1>();
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
    fn retain_equals_the_hosts_term_for_term() {
        crate::require_cuda!();
        retain_matches::<1>();
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

    fn top_n_matches<const W: usize>() {
        use crate::truncation::TopN;
        for (what, sum) in fixtures::<W>() {
            let len = sum.len();
            for n in [
                0usize,
                1,
                len / 3,
                len / 2,
                len.saturating_sub(1),
                len,
                len + 5,
            ] {
                let mut want = sum.clone();
                TopN(n).finalize_layer(&mut want);
                let (mut dev, mut scratch) = device(&sum);
                top_n_device(&mut dev, &mut scratch, n).expect("top n");
                assert_eq!(dev.len(), want.len(), "{what} n={n}: len");
                dev.assert_invariants_device()
                    .unwrap_or_else(|e| panic!("{what} n={n}: {e}"));
                let got = dev.to_host().expect("download");
                assert_eq!(got.to_arrays(), want.to_arrays(), "{what} n={n}: terms");
            }
        }
    }

    #[test]
    fn top_n_equals_the_hosts_term_for_term() {
        crate::require_cuda!();
        top_n_matches::<1>();
        top_n_matches::<2>();
    }

    /// `TopN::finalize_layer`'s tie rule on device: a group straddling the cut is dropped whole, one ending exactly at `n` is kept, an all-tied sum is wiped, and distinct magnitudes keep exactly `n`.
    #[test]
    fn top_n_tie_rules_match_the_host_on_device() {
        crate::require_cuda!();
        use crate::truncation::TopN;
        let six = |c: [(f64, f64); 6]| {
            PauliSum::<1>::from_sorted_columns(
                (0u64..6).map(|i| [i]).collect(),
                vec![[0u64]; 6],
                c.iter().map(|&(re, im)| Complex64::new(re, im)).collect(),
                3,
            )
        };
        let real = |m: [f64; 6]| six(m.map(|m| (m, 0.0)));
        let cases = [
            ("straddling", real([5.0, 4.0, 3.0, 3.0, 3.0, 2.0]), 3, 2),
            ("fits exactly", real([5.0, 4.0, 3.0, 3.0, 2.0, 1.0]), 4, 4),
            (
                "all tied",
                six([
                    (2.0, 0.0),
                    (-2.0, 0.0),
                    (0.0, 2.0),
                    (0.0, -2.0),
                    (2.0, 0.0),
                    (-2.0, 0.0),
                ]),
                3,
                0,
            ),
            ("all distinct", rand_sum::<1>(5_000, 32, 0xD157), 1234, 1234),
        ];
        for (what, sum, n, kept) in cases {
            let mut want = sum.clone();
            TopN(n).finalize_layer(&mut want);
            assert_eq!(want.len(), kept, "{what}: host");
            let (mut dev, mut scratch) = device(&sum);
            top_n_device(&mut dev, &mut scratch, n).expect("top n");
            dev.assert_invariants_device().unwrap();
            assert_eq!(
                dev.to_host().unwrap().to_arrays(),
                want.to_arrays(),
                "{what}"
            );
        }
    }
}
