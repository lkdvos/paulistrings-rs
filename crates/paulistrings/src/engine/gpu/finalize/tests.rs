use super::*;
use crate::engine::gpu::layer::GpuLayerOptions;
use crate::engine::transport::SoloTransport;
use crate::pauli_sum::PauliSum;
use crate::test_support::{rand_sum, tie_heavy_sum};
use crate::truncation::builtin::{octave_histogram, retain_at_edge};
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
fn histogram_is_the_hosts() {
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

/// The whole-sum boundary quota of `edge`, which a single partition keeps in full.
fn quota_of(edge: EdgeDecision) -> usize {
    match edge {
        EdgeDecision::Boundary { quota, .. } => quota,
        EdgeDecision::KeepAll | EdgeDecision::Clear => 0,
    }
}

/// `got` is `input` cut at `edge`: every term above the boundary octave and `quota` of those in it, none below, each with its input coefficient.
fn assert_cut<const W: usize>(
    got: &PauliSum<W>,
    input: &PauliSum<W>,
    edge: EdgeDecision,
    quota: usize,
    what: &str,
) {
    let mut above = input.clone();
    let mut through = input.clone();
    if let EdgeDecision::Boundary { bin, .. } = edge {
        retain_at_edge(&mut above, edge, 0);
        retain_at_edge(
            &mut through,
            EdgeDecision::Boundary { bin, kept_above: 0, quota: usize::MAX },
            usize::MAX,
        );
    } else {
        retain_at_edge(&mut above, edge, quota);
        retain_at_edge(&mut through, edge, quota);
    }
    assert_eq!(got.len(), above.len() + quota.min(through.len() - above.len()), "{what}: len");
    let kept: std::collections::HashMap<_, _> = got.iter().map(|(x, z, c)| ((*x, *z), c)).collect();
    for (x, z, c) in above.iter() {
        assert_eq!(kept.get(&(*x, *z)), Some(&c), "{what}: a term above the edge");
    }
    let through: std::collections::HashMap<_, _> =
        through.iter().map(|(x, z, c)| ((*x, *z), c)).collect();
    for (key, c) in &kept {
        assert_eq!(through.get(key), Some(c), "{what}: a kept term at or above the edge");
    }
}

/// One bucket of `n` terms whose octaves interleave, so most of the kept rows sit behind dropped ones and the move queues fill.
fn one_long_bucket<const W: usize>(n: u64) -> PauliSum<W> {
    PauliSum::<W>::from_sorted_columns(
        (0..n).map(|i| std::array::from_fn(|w| if w == 0 { i } else { 0 })).collect(),
        vec![[0u64; W]; n as usize],
        (0..n)
            .map(|i| {
                Complex64::new(0.5f64.powi((i * 7 % 13) as i32) * (1.0 + (i % 5) as f64 / 8.0), 0.0)
            })
            .collect(),
        64 * W,
    )
}

fn cut_matches<const W: usize>() {
    let mut cases = fixtures::<W>();
    cases.push(("one long bucket", one_long_bucket::<W>(5_000)));
    for (what, sum) in cases {
        let len = sum.len();
        let hist = octave_histogram(&sum);
        for n in [0usize, 1, len / 3, len / 2, len.saturating_sub(1), len] {
            let edge = octave_edge(&hist, len, n);
            let quota = quota_of(edge);
            let (mut dev, mut scratch) = device(&sum);
            let histogram = octave_histogram_device(&dev, &mut scratch).expect("histogram");
            retain_at_edge_device(&mut dev, &mut scratch, edge, quota, &histogram).expect("cut");
            dev.assert_invariants_device().unwrap_or_else(|e| panic!("{what} n={n}: {e}"));
            let got = dev.to_host().expect("download");
            assert_eq!(got.len(), dev.len(), "{what} n={n}: the kept count is the buckets'");
            let mut want = sum.clone();
            retain_at_edge(&mut want, edge, quota);
            assert_eq!(got.to_arrays(), want.to_arrays(), "{what} n={n}: the host's terms");
            if let EdgeDecision::Boundary { .. } = edge {
                assert_eq!(dev.len(), n, "{what} n={n}: exactly n kept");
            }
        }
    }
}

#[test]
fn the_cut_keeps_the_hosts_terms() {
    crate::require_cuda!();
    cut_matches::<1>();
    cut_matches::<2>();
}

fn real_cut_matches<const W: usize>() {
    let cases = [
        ("rand real", crate::test_support::rand_sum_real::<W>(50_000, 64 * W, 0x2EA1)),
        ("one long bucket", one_long_bucket::<W>(5_000)),
    ];
    for (what, sum) in cases {
        let real = sum.cast::<f64>(0.0).expect("real coefficients");
        let len = sum.len();
        for n in [1, len / 3, len / 2, len - 1] {
            let mut dev = GpuSum::<W, f64>::from_host(&real, 0).expect("upload");
            let mut scratch = LayerScratch::new(&dev, GpuLayerOptions::default()).expect("scratch");
            let histogram = octave_histogram_device(&dev, &mut scratch).expect("histogram");
            let edge = octave_edge(&histogram[1..], len, n);
            retain_at_edge_device(&mut dev, &mut scratch, edge, quota_of(edge), &histogram)
                .expect("cut");
            dev.assert_invariants_device().unwrap_or_else(|e| panic!("{what} n={n}: {e}"));
            let got = dev.to_host().expect("download").cast::<Complex64>(0.0).expect("complex");
            assert_eq!((got.len(), dev.len()), (n, n), "{what} n={n}: exactly n kept");
            let mut want = sum.clone();
            retain_at_edge(&mut want, edge, quota_of(edge));
            assert_eq!(got.to_arrays(), want.to_arrays(), "real {what} n={n}: the host's terms");
        }
    }
}

#[test]
fn the_cut_on_real_coefficients_keeps_the_same_terms() {
    crate::require_cuda!();
    real_cut_matches::<1>();
    real_cut_matches::<2>();
}

/// A second cut reads the first one's loose layout (buckets at their old offsets, gaps after them) in place; the first reordered rows within buckets, so which boundary terms stay is checked as a set.
#[test]
fn a_second_cut_reads_the_first_ones_layout() {
    crate::require_cuda!();
    let sum = rand_sum::<1>(30_000, 64, 0x2E7);
    let (mut dev, mut scratch) = device(&sum);
    for n in [20_000usize, 12_000] {
        let before = dev.to_host().unwrap();
        let histogram = octave_histogram_device(&dev, &mut scratch).expect("histogram");
        let edge = octave_edge(&histogram[1..], dev.len(), n);
        retain_at_edge_device(&mut dev, &mut scratch, edge, quota_of(edge), &histogram)
            .expect("cut");
        dev.assert_invariants_device().unwrap();
        let got = dev.to_host().unwrap();
        assert_eq!(got.len(), n, "exactly n kept");
        assert_eq!(dev.len(), n, "exactly n counted");
        assert_cut(&got, &before, edge, quota_of(edge), &format!("n={n}"));
    }
}

/// A NaN shares the top octave's histogram bin but never stays, so the kept count is read back: 40 of 64 terms with one NaN keep 39.
#[test]
fn a_nan_in_the_top_bin_is_cut_and_not_counted() {
    crate::require_cuda!();
    let mut c: Vec<Complex64> = (0..64).map(|i| Complex64::new(1.0 + i as f64, 0.0)).collect();
    c[5] = Complex64::new(f64::NAN, 0.0);
    c[9] = Complex64::new(f64::INFINITY, 0.0);
    let sum = PauliSum::<1>::from_sorted_columns(
        (0u64..64).map(|i| [i]).collect(),
        vec![[0u64]; 64],
        c,
        8,
    );
    let (mut dev, mut scratch) = device(&sum);
    let histogram = octave_histogram_device(&dev, &mut scratch).expect("histogram");
    let edge = octave_edge(&histogram[1..], dev.len(), 40);
    retain_at_edge_device(&mut dev, &mut scratch, edge, quota_of(edge), &histogram).expect("cut");
    dev.assert_invariants_device().unwrap();
    let got = dev.to_host().unwrap();
    assert_eq!(got.len(), 39, "the NaN counts toward n but is cut");
    assert_eq!(dev.len(), 39, "the read-back count");
    assert_cut(&got, &sum, edge, quota_of(edge), "nan");
    assert!(got.iter().all(|(_, _, c)| !c.re.is_nan()), "the NaN is cut");
    assert!(got.iter().any(|(_, _, c)| c.re.is_infinite()), "the infinity stays");
}

fn top_n_matches<const W: usize>() {
    use crate::truncation::TopN;
    for (what, sum) in fixtures::<W>() {
        let len = sum.len();
        for n in [0usize, 1, len / 3, len / 2, len.saturating_sub(1), len, len + 5] {
            let mut want = sum.clone();
            TopN(n).finalize_layer(&mut want);
            let (mut dev, mut scratch) = device(&sum);
            top_n_device(&mut dev, &mut scratch, n).expect("top n");
            assert_eq!(dev.len(), want.len(), "{what} n={n}: len");
            dev.assert_invariants_device().unwrap_or_else(|e| panic!("{what} n={n}: {e}"));
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

/// `TopN::finalize_layer`'s tie rule on device: a group straddling the cut is split in bucket order, one ending exactly at `n` is kept, an all-tied sum keeps its first `n`, and distinct magnitudes keep exactly `n`.
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
        ("straddling", real([5.0, 4.0, 3.0, 3.0, 3.0, 2.0]), 3, 3),
        ("fits exactly", real([5.0, 4.0, 3.0, 3.0, 2.0, 1.0]), 4, 4),
        (
            "all tied",
            six([(2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0), (2.0, 0.0), (-2.0, 0.0)]),
            3,
            3,
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
        assert_eq!(dev.to_host().unwrap().to_arrays(), want.to_arrays(), "{what}");
    }
}

fn device_decided_cut_matches<const W: usize>() {
    let mut cases = fixtures::<W>();
    cases.push(("one long bucket", one_long_bucket::<W>(5_000)));
    for (what, sum) in cases {
        let len = sum.len();
        let hist = octave_histogram(&sum);
        for n in [1usize, len / 3, len / 2, len.saturating_sub(1), len, len + 5] {
            let edge = octave_edge(&hist, len, n);
            let (mut dev, mut scratch) = device(&sum);
            approx_top_n_device(Some((&mut dev, &mut scratch)), n, &SoloTransport).expect("cut");
            dev.assert_invariants_device().unwrap_or_else(|e| panic!("{what} n={n}: {e}"));
            let got = dev.to_host().expect("download");
            let mut want = sum.clone();
            retain_at_edge(&mut want, edge, quota_of(edge));
            assert_eq!(got.to_arrays(), want.to_arrays(), "{what} n={n}: the host's terms");
            if let EdgeDecision::Boundary { bin, quota, .. } = edge {
                let decision = dev.stream.clone_dtoh(&scratch.cut_decision).expect("decision");
                assert_eq!(&decision[..3], &[0, bin as u64, quota as u64], "{what} n={n}");
            }
        }
    }
}

/// A lone device decides `ApproxTopN`'s edge on the device and keeps exactly the host's terms.
#[test]
fn the_device_decided_cut_keeps_the_hosts_terms() {
    crate::require_cuda!();
    device_decided_cut_matches::<1>();
    device_decided_cut_matches::<2>();
}
