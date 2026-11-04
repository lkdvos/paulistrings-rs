use super::*;
use crate::pauli_sum::hash::Gf2Hash;
use crate::test_support::{canonical_triples, rand_sum_real, storage_triples, tie_heavy_sum, Xs64};
use crate::truncation::builtin::{
    approx_top_n_pass, bucket_quotas, octave_edge, octave_histogram, ApproxTopN,
};
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

/// The window counts summed over every bucket.
fn totals_of(per_bucket: &[BucketOctaves]) -> WindowTotals {
    let mut totals = WindowTotals::default();
    for counts in per_bucket {
        totals.add(counts);
    }
    totals
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
    match decide_from_window(local.len(), collectives, n, window, &counts, &totals_of(&counts)) {
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
    let bin = octave(1.0);
    let cut = BoundaryCut { bin, split: 0, split_quota: 2, kept: 0 };
    let norms = [f64::NAN, 1.0, 4.0, 1.5, 0.25, 1.0, f64::INFINITY, f64::NAN, 1.0];
    let want = [false, true, true, true, false, false, true, false, false];
    let mut filter = cut.filter::<1, Complex64>(0);
    for (&norm_sqr, &keep) in norms.iter().zip(&want) {
        let c = Complex64::new(norm_sqr.sqrt(), 0.0);
        assert_eq!(filter(&[0], &[0], c), keep, "|c|² = {norm_sqr}");
    }
}

/// A decision with its cut as `(bin, quota per bucket, kept)`.
type SerialDecision = (EdgeDecision, usize, Option<(usize, Vec<usize>, usize)>);

/// The decision with every bucket's quota spelled out, as a serial pass over every bucket's counts takes it.
fn serial_decision(
    local_len: usize,
    collectives: &dyn Collectives,
    n: usize,
    window: OctaveWindow,
    per_bucket: &[BucketOctaves],
) -> Option<SerialDecision> {
    let mut packed = [0u64; WINDOW_OCTAVES + 3];
    packed[0] = local_len as u64;
    for counts in per_bucket {
        for (slot, &count) in packed[1..].iter_mut().zip(counts) {
            *slot += u64::from(count);
        }
    }
    if collectives.size() > 1 {
        collectives.allreduce_sum_u64(&mut packed);
    }
    let total = packed[0] as usize;
    let above = packed[WINDOW_OCTAVES + 2] as usize;
    let edge = octave_edge_within(&packed[2..WINDOW_OCTAVES + 2], window.base, above, total, n)?;
    let cut = match edge {
        EdgeDecision::Boundary { bin, quota, .. } => {
            let slot = window.slot(bin);
            let boundary_counts: Vec<usize> =
                per_bucket.iter().map(|counts| counts[slot] as usize).collect();
            let local_quota =
                local_boundary_quota(collectives, boundary_counts.iter().sum(), quota);
            let local_above: usize = per_bucket
                .iter()
                .map(|counts| counts[slot + 1..].iter().map(|&count| count as usize).sum::<usize>())
                .sum();
            Some((bin, bucket_quotas(&boundary_counts, local_quota), local_above + local_quota))
        }
        EdgeDecision::KeepAll | EdgeDecision::Clear => None,
    };
    Some((edge, total, cut))
}

/// On 1, 2 and 4 partitions, few and many buckets, the decision from the totals and the split is the serial decision, bucket for bucket.
#[test]
fn the_decision_from_totals_is_the_serial_decision() {
    for bits in [4u8, 11] {
        let random = hashed(rand_sum_real::<1>(3000, 32, 0x51D3), bits);
        let ties = hashed(tie_heavy_sum::<1>(3000, 32, 0x7135), bits);
        for (input, what) in [(&random, "random"), (&ties, "tie-heavy")] {
            for pbits in [0u8, 1, 2] {
                for n in [0usize, 1, 37, 999, 1500, 2999, 3000] {
                    let windows: Vec<OctaveWindow> = if n < input.len() && n > 0 {
                        windows_around(boundary_bin(input, n)).to_vec()
                    } else {
                        vec![OctaveWindow::around(1000)]
                    };
                    for window in windows {
                        let label =
                            format!("{what} bits={bits} P=2^{pbits} n={n} base={}", window.base);
                        run_partitioned(input, pbits, |local, collectives| {
                            let counts = per_bucket(local, window);
                            let want =
                                serial_decision(local.len(), collectives, n, window, &counts);
                            let got = decide_from_window(
                                local.len(),
                                collectives,
                                n,
                                window,
                                &counts,
                                &totals_of(&counts),
                            );
                            let got = got.map(|(edge, total, cut)| {
                                let cut = cut.map(|cut| {
                                    let slot = window.slot(cut.bin);
                                    let quotas = counts
                                        .iter()
                                        .enumerate()
                                        .map(|(b, counts)| cut.quota(b).min(counts[slot] as usize))
                                        .collect();
                                    (cut.bin, quotas, cut.kept)
                                });
                                (edge, total, cut)
                            });
                            assert_eq!(got, want, "{label}");
                        });
                    }
                }
            }
        }
    }
}

/// The split hands each bucket the serial quota rule's share, on sparse, tie-heavy and dense counts across the chunk edges.
#[test]
fn the_split_is_the_serial_quota_rule() {
    let mut rng = Xs64::new(0x5917);
    for buckets in [0usize, 1, 5, 1023, 1024, 1025, 4100] {
        for spread in [1u64, 2, 9] {
            let counts: Vec<usize> =
                (0..buckets).map(|_| (rng.next_u64() % (spread + 1)) as usize).collect();
            let total: usize = counts.iter().sum();
            let quotas = [0, 1, total / 3, total / 2, total.saturating_sub(1), total, total + 5];
            for quota in quotas {
                let want = bucket_quotas(&counts, quota);
                let (split, split_quota) = quota_split(|b| counts[b], buckets, total, quota);
                let cut = BoundaryCut { bin: 0, split, split_quota, kept: 0 };
                for (b, (&count, &want)) in counts.iter().zip(&want).enumerate() {
                    assert_eq!(
                        cut.quota(b).min(count),
                        want,
                        "buckets={buckets} spread={spread} quota={quota}: bucket {b}"
                    );
                }
            }
        }
    }
}
