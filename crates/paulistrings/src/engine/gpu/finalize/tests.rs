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
