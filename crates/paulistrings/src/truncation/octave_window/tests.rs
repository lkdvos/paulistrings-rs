use super::*;
use crate::pauli_sum::hash::Gf2Hash;
use crate::test_support::{canonical_triples, rand_sum_real, storage_triples, tie_heavy_sum};
use crate::truncation::builtin::{approx_top_n_pass, octave_edge, octave_histogram, ApproxTopN};
use crate::truncation::tests::run_partitioned;
use crate::truncation::TruncationPolicy;

/// Every bucket's window counts.
fn per_bucket<const W: usize>(sum: &PauliSum<W>, window: OctaveWindow) -> Vec<BucketOctaves> {
    (0..sum.num_buckets())
        .map(|b| {
            let mut counts = [0; WINDOW_OCTAVES + 2];
            window.count(sum.bucket(b).2, &mut counts);
            counts
        })
        .collect()
}

/// The boundary octave `ApproxTopN(n)` cuts `sum` at, which the fixtures must have.
fn boundary_bin<const W: usize>(sum: &PauliSum<W>, n: usize) -> usize {
    match octave_edge(&octave_histogram(sum), sum.len(), n) {
        EdgeDecision::Boundary { bin, .. } => bin,
        other => panic!("the fixture must cut inside an octave, got {other:?}"),
    }
}

/// Windows that hold `bin`, sit wholly above it, and wholly below it.
fn windows_around(bin: usize) -> [OctaveWindow; 3] {
    [
        OctaveWindow::around(bin),
        OctaveWindow { base: bin + 1 },
        OctaveWindow { base: bin - WINDOW_OCTAVES },
    ]
}

/// `sum` with `1 << bits` buckets.
fn hashed<const W: usize>(sum: PauliSum<W>, bits: u8) -> PauliSum<W> {
    let num_qubits = sum.num_qubits();
    sum.with_hash(Gf2Hash::<W>::new(num_qubits, bits, 0x99))
}

#[test]
fn window_counts_agree_with_the_histogram() {
    let sum = hashed(rand_sum_real::<1>(3000, 32, 0x0C7A), 4);
    let histogram = octave_histogram(&sum);
    for base in [0usize, 900, 1000, 1010, APPROX_BINS - WINDOW_OCTAVES] {
        let window = OctaveWindow { base };
        let mut totals = [0u64; WINDOW_OCTAVES + 2];
        for counts in per_bucket(&sum, window) {
            for (total, count) in totals.iter_mut().zip(counts) {
                *total += u64::from(count);
            }
        }
        let bins = |range: std::ops::Range<usize>| -> u64 {
            histogram[range].iter().map(|&count| u64::from(count)).sum()
        };
        assert_eq!(totals[0], bins(0..base), "base={base}: below");
        for offset in 0..WINDOW_OCTAVES {
            assert_eq!(
                totals[1 + offset],
                u64::from(histogram[base + offset]),
                "base={base}: octave {}",
                base + offset
            );
        }
        assert_eq!(
            totals[WINDOW_OCTAVES + 1],
            bins(base + WINDOW_OCTAVES..APPROX_BINS),
            "base={base}: above"
        );
    }
}

#[test]
fn around_clamps_into_the_octave_range() {
    assert_eq!(OctaveWindow::around(3).base, 0);
    assert_eq!(OctaveWindow::around(1023).base, 1023 - WINDOW_OCTAVES / 2);
    assert_eq!(OctaveWindow::around(APPROX_BINS - 1).base, APPROX_BINS - WINDOW_OCTAVES);
}

/// [`approx_top_n_pass`] from the window counts: the window's cut, applied bucket by bucket and checked against the whole-sum application and `kept`, or the plain pass outside the window.
fn finalize_from_window<const W: usize>(
    local: &mut PauliSum<W>,
    collectives: &dyn Collectives,
    n: usize,
    window: OctaveWindow,
) {
    let counts = per_bucket(local, window);
    match decide_from_window(
        local.len(),
        collectives,
        n,
        window,
        &counts,
        &WindowTotals::of(&counts),
    ) {
        None => {
            approx_top_n_pass(local, collectives, n);
        }
        Some((EdgeDecision::Clear, ..)) => local.clear(),
        Some((_, _, None)) => {}
        Some((_, _, Some(cut))) => {
            let mut whole = local.clone();
            cut.apply(&mut whole);
            for (b, columns) in local.buckets_mut().iter_mut().enumerate() {
                cut.apply_to_bucket(b, columns);
            }
            local.recount();
            assert_eq!(storage_triples(local), storage_triples(&whole), "per bucket vs whole");
            assert_eq!(local.len(), cut.kept, "kept");
        }
    }
}

/// On 1, 2 and 4 partitions, every window, holding the boundary, above it or below it, keeps exactly the plain pass's set.
fn assert_window_pass_is_the_plain_pass<const W: usize>(num_qubits: usize) {
    let random = hashed(rand_sum_real::<W>(2000, num_qubits, 0xA9C7), 4);
    let ties = hashed(tie_heavy_sum::<W>(2000, num_qubits, 0x7135), 4);
    for (input, what) in [(&random, "random"), (&ties, "tie-heavy")] {
        for pbits in [0u8, 1, 2] {
            for n in [0usize, 3, 37, 300, 999, 1500, 1900, 2000, 2010] {
                let want = run_partitioned(input, pbits, |local, collectives| {
                    ApproxTopN(n).finalize_layer_partitioned(local, collectives)
                });
                let windows: Vec<OctaveWindow> = if n < input.len() && n > 0 {
                    windows_around(boundary_bin(input, n)).to_vec()
                } else {
                    vec![OctaveWindow::around(1000)]
                };
                for window in windows {
                    let got = run_partitioned(input, pbits, |local, collectives| {
                        finalize_from_window(local, collectives, n, window);
                    });
                    let label = format!("W={W} {what} P=2^{pbits} n={n} base={}", window.base);
                    assert_eq!(got.len(), n.min(input.len()), "{label}: kept count");
                    assert_eq!(canonical_triples(&got), canonical_triples(&want), "{label}");
                }
            }
        }
    }
}

#[test]
fn the_window_pass_is_the_plain_pass() {
    assert_window_pass_is_the_plain_pass::<1>(32);
    assert_window_pass_is_the_plain_pass::<2>(100);
}

/// A bucket's filter keeps every term `classify_octave` puts above the boundary and the first `quota` at it, a NaN never.
#[test]
fn a_bucket_filter_keeps_what_classify_octave_keeps() {
    use num_complex::Complex64;
    let bin = approx_bin(1.0);
    let cut = BoundaryCut { bin, split: QuotaSplit { bucket: 0, remaining: 2 }, kept: 0 };
    // The boundary bin is [1, 1.25): 1.3 is above it in the same octave, 0.9 below.
    let norms = [f64::NAN, 1.0, 4.0, 1.2, 0.25, 1.3, 1.0, 0.9, f64::INFINITY, f64::NAN, 1.0];
    let want = [false, true, true, true, false, true, false, false, true, false, false];
    let mut filter = cut.filter::<1, Complex64>(0);
    for (&norm_sqr, &keep) in norms.iter().zip(&want) {
        let c = Complex64::new(norm_sqr.sqrt(), 0.0);
        assert_eq!(filter(&[0], &[0], c), keep, "|c|² = {norm_sqr}");
    }
}
