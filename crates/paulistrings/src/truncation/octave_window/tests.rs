use super::*;
use crate::engine::partitioned::in_process::InProcessTransport;
use crate::pauli_sum::hash::Gf2Hash;
use crate::test_support::{canonical_triples, rand_sum_real, storage_triples, tie_heavy_sum};
use crate::truncation::builtin::{octave_histogram, ApproxTopN};
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
    match super::super::builtin::octave_edge(&octave_histogram(sum), sum.len(), n) {
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

/// Unpartitioned, every window keeps exactly `ApproxTopN`'s own set in the same storage order, inside the window or through the fallback.
fn assert_single_partition_agrees<const W: usize>(input: &PauliSum<W>, ns: &[usize], what: &str) {
    for &n in ns {
        let mut want = input.clone();
        ApproxTopN(n).finalize_layer(&mut want);
        let windows: Vec<OctaveWindow> = if n < input.len() && n > 0 {
            windows_around(boundary_bin(input, n)).to_vec()
        } else {
            vec![OctaveWindow::around(1000)]
        };
        for window in windows {
            let mut got = input.clone();
            let counts = per_bucket(&got, window);
            let solo = InProcessTransport::group(1).pop().expect("one partition");
            finalize_from_window(&mut got, &solo, n, window, &counts);
            got.assert_invariants();
            let label = format!("{what} n={n} base={}", window.base);
            assert_eq!(got.len(), n.min(input.len()), "{label}: kept count");
            assert_eq!(storage_triples(&got), storage_triples(&want), "{label}: kept set");
        }
    }
}

#[test]
fn finalize_from_window_is_the_plain_pass_w1() {
    let input = hashed(rand_sum_real::<1>(2000, 32, 0xA9C7), 5);
    assert_single_partition_agrees(&input, &[0, 1, 37, 300, 999, 1500, 2000, 2010], "w1");
}

#[test]
fn finalize_from_window_is_the_plain_pass_w2_tie_heavy() {
    let input = hashed(tie_heavy_sum::<2>(2000, 100, 0x7135), 4);
    assert_single_partition_agrees(&input, &[0, 3, 250, 700, 1200, 1900, 2500], "w2 tie-heavy");
}

/// Across partitions the window pass keeps exactly the partitioned `ApproxTopN` set, every partition choosing the window path or the fallback alike.
#[test]
fn finalize_from_window_partitioned_is_the_plain_pass() {
    let random = hashed(rand_sum_real::<1>(2000, 32, 0xA9C7), 4);
    let ties = hashed(tie_heavy_sum::<1>(2000, 32, 0x7135), 4);
    for (input, what) in [(&random, "random"), (&ties, "tie-heavy")] {
        for pbits in [1u8, 2] {
            for n in [0usize, 3, 300, 999, 1500, 2000, 2010] {
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
                        let counts = per_bucket(local, window);
                        finalize_from_window(local, collectives, n, window, &counts);
                    });
                    let label = format!("{what} P=2^{pbits} n={n} base={}", window.base);
                    assert_eq!(got.len(), n.min(input.len()), "{label}: kept count");
                    assert_eq!(canonical_triples(&got), canonical_triples(&want), "{label}");
                }
            }
        }
    }
}

/// A cut applied bucket by bucket keeps what the whole-sum application keeps, and `kept` counts it.
#[test]
fn a_cut_applies_alike_per_bucket_and_whole() {
    let input = hashed(tie_heavy_sum::<1>(2000, 32, 0x7135), 4);
    for n in [3usize, 700, 1200, 1900] {
        let window = OctaveWindow::around(boundary_bin(&input, n));
        let solo = InProcessTransport::group(1).pop().expect("one partition");
        let Some(WindowDecision::Cut(cut)) =
            decide_from_window(input.len(), &solo, n, window, &per_bucket(&input, window))
        else {
            panic!("n={n}: the window must resolve the boundary");
        };
        assert_eq!(cut.kept, n, "n={n}: kept");
        let mut whole = input.clone();
        cut.apply(&mut whole);
        let mut each = input.clone();
        for (b, columns) in each.buckets_mut().iter_mut().enumerate() {
            cut.apply_to_bucket(b, columns);
        }
        each.recount();
        assert_eq!(storage_triples(&each), storage_triples(&whole), "n={n}");
    }
}
