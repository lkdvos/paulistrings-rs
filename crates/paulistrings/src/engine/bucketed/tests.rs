use super::coset_fill::{gather_local_input_major, gather_local_output_major, nonzero, GatherRun};
use super::*;
use crate::channel::rotation::PauliRotation;
use crate::channel::Channel;
use crate::channel::Prepare;
use crate::channel::PtmChannel;
use crate::engine::merge::{merge_identity_and_rest, rest_rows_per_key, SortKernel, SortScratch};
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::storage::ColumnsRef;
use crate::pauli_sum::PauliSum;
use crate::test_support::IdentityChannel;
use crate::truncation::builtin::{And, CoefficientThreshold, WeightCutoff};

// Re-exported so the sibling test modules reach the fixtures through `super::tests`.
pub(super) use crate::test_support::{
    assert_same_terms, assert_terms_close, canonical_triples, naive_apply_layer, rand_sum,
};

/// A discarded row leaves the lengths and every published row untouched, even in the spare `+ 1` slot.
#[test]
fn push_if_publishes_only_kept_rows_and_never_overruns() {
    let mut run: GatherRun<1> = GatherRun::default();
    // Exactly two countable rows for the rest stream.
    run.reset(0, 0, 2);
    let c = |v: f64| Complex64::new(v, 0.0);
    run.push_if(false, [7], [7], c(9.0)); // discarded at len 0
    assert_eq!(run.rest.x.len(), 0);
    run.push_if(true, [1], [2], c(1.0));
    run.push_if(false, [7], [7], c(9.0)); // discarded at len 1
    run.push_if(true, [3], [4], c(2.0)); // the last countable row
    assert_eq!(run.rest.x.as_slice(), &[[1], [3]]);
    assert_eq!(run.rest.z.as_slice(), &[[2], [4]]);
    assert_eq!(run.rest.coeff.as_slice(), &[c(1.0), c(2.0)]);
    // The identity stream answers the same contract.
    run.reset(1, 1, 0);
    run.push_identity_if(false, [5], [6], c(3.0));
    assert_eq!(run.identity.x.len(), 0);
    run.push_identity_if(true, [5], [6], c(3.0));
    assert_eq!(run.identity.x.as_slice(), &[[5]]);
    assert_eq!(run.identity.coeff.as_slice(), &[c(3.0)]);
}

/// `nonzero` agrees with `!= ZERO`, signed zeros and NaN included.
#[test]
fn nonzero_agrees_with_complex_inequality() {
    for a in [
        Complex64::new(0.0, 0.0),
        Complex64::new(-0.0, 0.0),
        Complex64::new(0.0, -0.0),
        Complex64::new(-0.0, -0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(0.0, f64::NAN),
        Complex64::new(f64::MIN_POSITIVE, 0.0),
    ] {
        assert_eq!(nonzero(a), a != ZERO, "disagreement at {a}");
    }
}

const TOL: f64 = 1e-11;

pub(super) struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

/// A Haar-random SU(4), the dense-PTM two-qubit gate.
pub(super) fn haar_su4(q0: usize, q1: usize) -> PtmChannel {
    PtmChannel::unitary_2q(q0, q1, crate::test_support::haar_su4_matrix())
}

/// The trace's state machine, independent of any propagation: `None` means off, `enable` is idempotent and non-destructive, `take` drains but stays on.
#[test]
fn trace_is_opt_in_and_drains_on_take() {
    use crate::engine::trace::{PartitionLayerRecord, PartitionTrace};

    let mut scratch = LayerScratch::<1>::new();
    assert!(scratch.take_trace().is_none(), "off by default");

    let record = PartitionLayerRecord {
        circuit_index: 3,
        ..PartitionLayerRecord::default()
    };
    scratch.enable_trace();
    scratch.trace.as_mut().unwrap().layers.push(record.clone());
    scratch.enable_trace(); // idempotent: must not clear the record
    assert_eq!(
        scratch.take_trace(),
        Some(PartitionTrace {
            layers: vec![record]
        })
    );
    assert_eq!(scratch.take_trace(), Some(PartitionTrace::default()));
}

/// Run one layer through the bucketed engine, converting in and out.
pub(super) fn bucketed_layer<const W: usize, C, T>(
    input: &PauliSum<W>,
    channel: &C,
    policy: &T,
    adjoint: bool,
    bits: u8,
    seed: u64,
) -> PauliSum<W>
where
    C: Channel<W> + ?Sized,
    T: TruncationPolicy<W> + ?Sized,
{
    let hash = Gf2Hash::<W>::new(input.num_qubits(), bits, seed);
    let mut b = input.clone().with_hash(hash);
    let prepared = channel
        .prepare(b.hash(), adjoint)
        .expect("channel could not be prepared");
    let mut scratch = LayerScratch::<W>::new();
    apply_layer_bucketed(&mut b, &prepared, policy, &mut scratch);
    b
}

/// Keys must match exactly; coefficients only to tolerance.
pub(super) fn assert_sums_close<const W: usize>(got: &PauliSum<W>, want: &PauliSum<W>, what: &str) {
    assert_terms_close(got, want, TOL, what);
}

// ---- hand-checked behaviour ----

#[test]
fn h_conjugates_z_to_x() {
    let mut accumulator = BuildAccumulator::<1>::with_capacity(4, 1);
    accumulator.add_term(PauliString::<1>::z(0), Complex64::new(1.0, 0.0));
    let input = accumulator.finalize();
    let out = bucketed_layer(&input, &PtmChannel::hadamard(0), &AlwaysKeep, false, 4, 0x1);
    assert_eq!(out.len(), 1);
    let (x, z, c) = out.iter().next().unwrap();
    assert_eq!(*x, [1]);
    assert_eq!(*z, [0]);
    assert!((c - Complex64::new(1.0, 0.0)).norm() < TOL);
}

#[test]
fn cnot_propagates_z_on_the_control() {
    let mut accumulator = BuildAccumulator::<1>::with_capacity(4, 1);
    accumulator.add_term(PauliString::<1>::z(1), Complex64::new(1.0, 0.0));
    let input = accumulator.finalize();
    // I⊗Z under CNOT(0 -> 1) becomes Z⊗Z.
    let out = bucketed_layer(&input, &PtmChannel::cnot(0, 1), &AlwaysKeep, false, 4, 0x1);
    assert_eq!(out.len(), 1);
    let (x, z, _) = out.iter().next().unwrap();
    assert_eq!(*z, [0b11]);
    assert_eq!(*x, [0]);
}

#[test]
fn a_rotation_fans_out_to_two_terms() {
    let mut accumulator = BuildAccumulator::<1>::with_capacity(4, 1);
    accumulator.add_term(PauliString::<1>::x(0), Complex64::new(1.0, 0.0));
    let input = accumulator.finalize();
    let rot = PauliRotation::new(PauliString::<1>::z(0), std::f64::consts::FRAC_PI_3);
    let out = bucketed_layer(&input, &rot, &AlwaysKeep, false, 4, 0x1);
    // cos(pi/3)*X + sin(pi/3)*(i * X * Z) = 0.5*X - 0.866*Y
    assert_eq!(out.len(), 2);
    let want = naive_apply_layer(&input, &rot, &AlwaysKeep, false);
    assert_sums_close(&out, &want, "rotation fanout");
}

// ---- the differential test against the naive oracle ----

/// The engine's primary correctness net: every built-in channel against `naive_apply_layer`, over both occupancy regimes, bucket counts, directions and three policies.
#[test]
fn differential_against_the_naive_oracle_w1_dense_collisions() {
    // 8 qubits, so 2000 terms collide heavily and the merge has real duplicates to combine.
    let input = rand_sum::<1>(2000, 8, 0xC0FFEE);
    let channels = crate::test_support::differential_channels_w1();

    for (name, channel) in &channels {
        let channel_ref: &dyn Channel<1> = channel.as_ref();
        for &adjoint in &[false, true] {
            let want = naive_apply_layer(&input, channel_ref, &AlwaysKeep, adjoint);
            for &bits in &[0u8, 11] {
                let got = bucketed_layer(&input, channel_ref, &AlwaysKeep, adjoint, bits, 0xABCD);
                assert_terms_close(
                    &got,
                    &want,
                    TOL,
                    &format!("{name} adjoint={adjoint} bits={bits}"),
                );
            }
        }
    }
}

#[test]
fn differential_against_the_naive_oracle_w2_sparse() {
    // The other regime: wide keys, few collisions, word-boundary supports.
    let input = rand_sum::<2>(1000, 128, 0xBEEF);
    let channels = crate::test_support::differential_channels_w2();
    for (name, channel) in &channels {
        let channel_ref: &dyn Channel<2> = channel.as_ref();
        for &adjoint in &[false, true] {
            let want = naive_apply_layer(&input, channel_ref, &AlwaysKeep, adjoint);
            for &bits in &[2u8, 9] {
                let got = bucketed_layer(&input, channel_ref, &AlwaysKeep, adjoint, bits, 0xABCD);
                assert_terms_close(
                    &got,
                    &want,
                    TOL,
                    &format!("{name} adjoint={adjoint} bits={bits}"),
                );
            }
        }
    }
}

#[test]
fn differential_with_truncation_policies() {
    let input = rand_sum::<1>(1500, 8, 0xF00D);
    // Thresholds far from the coefficient scale, so rounding cannot cross a cutoff.
    let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
    let cnot = PtmChannel::cnot(1, 5);

    for bits in [0u8, 4, 9] {
        let got = bucketed_layer(&input, &rot, &CoefficientThreshold(1e-9), false, bits, 0x11);
        let want = naive_apply_layer(&input, &rot, &CoefficientThreshold(1e-9), false);
        assert_terms_close(&got, &want, TOL, &format!("threshold bits={bits}"));

        let got = bucketed_layer(&input, &rot, &WeightCutoff(4), false, bits, 0x11);
        let want = naive_apply_layer(&input, &rot, &WeightCutoff(4), false);
        assert_terms_close(&got, &want, TOL, &format!("weight bits={bits}"));

        let policy = And(CoefficientThreshold(1e-9), WeightCutoff(5));
        let got = bucketed_layer(&input, &cnot, &policy, false, bits, 0x11);
        let want = naive_apply_layer(&input, &cnot, &policy, false);
        assert_terms_close(&got, &want, TOL, &format!("and bits={bits}"));
    }
}

#[test]
fn keep_term_sees_the_summed_coefficient() {
    // At theta = pi/2, X and Y land on one key with nearly cancelling weights, which the threshold must drop.
    let mut accumulator = BuildAccumulator::<1>::with_capacity(4, 2);
    accumulator.add_term(PauliString::<1>::x(0), Complex64::new(0.5, 0.0));
    accumulator.add_term(PauliString::<1>::y(0), Complex64::new(-0.4999999, 0.0));
    let input = accumulator.finalize();
    let rot = PauliRotation::new(PauliString::<1>::z(0), std::f64::consts::FRAC_PI_2);
    for bits in [0u8, 3, 7] {
        let policy = CoefficientThreshold(1e-6);
        let got = bucketed_layer(&input, &rot, &policy, false, bits, 0x21);
        let want = naive_apply_layer(&input, &rot, &policy, false);
        assert_terms_close(&got, &want, TOL, &format!("post-sum threshold bits={bits}"));
    }
}

// ---- the key-preserving fast path ----

#[test]
fn rescale_fast_path_agrees_with_the_general_path() {
    // Depolarizing/Dephasing/Pauli take `rescale_in_place`; the oracle has no such special case.
    let input = rand_sum::<1>(1500, 8, 0x5A5A);
    let chans: Vec<(&str, Box<dyn Channel<1>>)> = vec![
        ("identity", Box::new(IdentityChannel::new())),
        ("depolarizing", Box::new(PtmChannel::depolarizing(3, 0.11))),
        ("dephasing", Box::new(PtmChannel::dephasing(3, 0.11))),
        ("pauli_z", Box::new(PtmChannel::pauli_z(3))),
    ];
    for (name, channel) in &chans {
        let channel_ref: &dyn Channel<1> = channel.as_ref();
        for bits in [0u8, 4, 8] {
            let got = bucketed_layer(&input, channel_ref, &AlwaysKeep, false, bits, 0x31);
            let want = naive_apply_layer(&input, channel_ref, &AlwaysKeep, false);
            assert_terms_close(&got, &want, TOL, &format!("{name} bits={bits}"));
        }
    }
}

#[test]
fn rescale_fast_path_still_applies_truncation() {
    let input = rand_sum::<1>(1500, 8, 0x5A5B);
    let depol = PtmChannel::depolarizing(3, 0.11);
    for bits in [0u8, 5] {
        let policy = And(CoefficientThreshold(0.3), WeightCutoff(4));
        let got = bucketed_layer(&input, &depol, &policy, false, bits, 0x41);
        let want = naive_apply_layer(&input, &depol, &policy, false);
        assert_terms_close(&got, &want, TOL, &format!("truncated rescale bits={bits}"));
        assert!(got.len() < input.len(), "truncation dropped nothing");
    }
}

/// At `p = 3/4` depolarizing scales X, Y and Z by exactly zero, and the rescale drops those terms rather than storing zeros.
#[test]
fn a_rescale_to_exactly_zero_drops_the_term() {
    let build = |terms: &[(PauliString<1>, f64)]| {
        let mut accumulator = BuildAccumulator::<1>::with_capacity(4, terms.len());
        for &(p, c) in terms {
            accumulator.add_term(p, Complex64::new(c, 0.0));
        }
        accumulator.finalize()
    };
    let input = build(&[
        (PauliString::<1>::identity(), 2.0),
        (PauliString::<1>::x(0), 1.0),
        (PauliString::<1>::z(1), 0.5),
    ]);
    let want = build(&[
        (PauliString::<1>::identity(), 2.0),
        (PauliString::<1>::z(1), 0.5),
    ]);
    let depolarize = PtmChannel::depolarizing(0, 0.75);
    for bits in [0u8, 2] {
        let got = bucketed_layer(&input, &depolarize, &AlwaysKeep, false, bits, 0x51);
        got.assert_invariants();
        assert_terms_close(&got, &want, TOL, &format!("bits={bits}"));
    }
}

// ---- agreement across bucket counts, seeds and gather orders ----

/// sqrt(SWAP): a wide delta set merging three or more contributions per key, the only regime where summation order is observable.
fn sqrt_swap_w1(a: usize, b: usize) -> PtmChannel {
    let h = Complex64::new(0.5, 0.5);
    let hc = Complex64::new(0.5, -0.5);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    PtmChannel::unitary_2q(
        a,
        b,
        [
            [one, zero, zero, zero],
            [zero, h, hc, zero],
            [zero, hc, h, zero],
            [zero, zero, zero, one],
        ],
    )
}

#[test]
fn output_agrees_across_bucket_counts_to_fp_tolerance() {
    // A different bucket count can reorder a key's summands, so agreement is to tolerance; a dense two-qubit unitary is the channel that exercises it.
    let input = rand_sum::<1>(2000, 8, 0x9001);
    let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
    let cnot = PtmChannel::cnot(1, 5);
    let gu2q = sqrt_swap_w1(1, 5);
    for channel in [
        &rot as &dyn Channel<1>,
        &cnot as &dyn Channel<1>,
        &gu2q as &dyn Channel<1>,
    ] {
        let reference = bucketed_layer(&input, channel, &AlwaysKeep, false, 0, 0x51);
        for bits in [1u8, 2, 3, 5, 8, 11] {
            let got = bucketed_layer(&input, channel, &AlwaysKeep, false, bits, 0x51);
            assert_terms_close(&got, &reference, TOL, &format!("bits={bits}"));
        }
    }
}

#[test]
fn output_agrees_across_hash_seeds_to_fp_tolerance() {
    // A different `H` regroups terms; tolerance, for the same reason.
    let input = rand_sum::<1>(2000, 8, 0x9002);
    let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
    let gu2q = sqrt_swap_w1(1, 5);
    for channel in [&rot as &dyn Channel<1>, &gu2q as &dyn Channel<1>] {
        let reference = bucketed_layer(&input, channel, &AlwaysKeep, false, 6, 1);
        for seed in [2u64, 3, 5, 8, 13, 21] {
            let got = bucketed_layer(&input, channel, &AlwaysKeep, false, 6, seed);
            assert_terms_close(&got, &reference, TOL, &format!("seed={seed}"));
        }
    }
}

#[test]
fn local_gather_orders_agree_to_fp_tolerance() {
    // The two gather orders emit the same rows per run in different orders, so merged runs agree to tolerance.
    // sqrt-SWAP's span has rank exactly 2 under any hash.
    let input = rand_sum::<1>(2000, 8, 0xAB12);
    let gu2q = sqrt_swap_w1(1, 5);
    let hash = Gf2Hash::<1>::new(8, 5, 0x77);
    let sum = input.clone().with_hash(hash);
    let prepared = gu2q.prepare(sum.hash(), false).unwrap();
    let Prepared::Local(ptm) = &prepared else {
        panic!("gu2q prepares to a Local plan");
    };
    let span = Gf2Span::new(&prepared.bucket_deltas(), sum.hash().bits());
    assert!(
        span.dimension() >= 2,
        "want a multi-member coset so the two visit orders actually differ; got r={}",
        span.dimension()
    );
    let coordinates: Vec<u32> = ptm
        .deltas()
        .iter()
        .map(|d| span.coordinate_of(d.bucket_delta))
        .collect();
    let m = span.coset_size();

    // Assemble the rank-0 coset's member columns, ascending by coordinate.
    let mut old: Vec<BucketColumns<1>> = (0..m).map(|_| BucketColumns::default()).collect();
    for beta in 0..sum.num_buckets() as u32 {
        let p = span.permuted_index(beta) as usize;
        if p < m {
            let (bx, bz, bc) = sum.bucket(beta as usize);
            old[p] = BucketColumns {
                x: bx.to_vec(),
                z: bz.to_vec(),
                coeff: bc.to_vec(),
            };
        }
    }

    let has_identity = ptm.deltas().first().is_some_and(|d| d.local_delta == 0);
    // gu2q's identity is dense: only id coefficients are gathered and the merge borrows the source keys.
    let dim = 1usize << (2 * ptm.k());
    let dense_identity =
        has_identity && ptm.deltas()[0].amplitude[..dim].iter().all(|a| *a != ZERO);
    assert!(dense_identity, "gu2q's identity amplitude must be dense");
    let gather = |output_major: bool| {
        let mut runs: Vec<GatherRun<1>> = (0..m).map(|_| GatherRun::default()).collect();
        for (j, run) in runs.iter_mut().enumerate() {
            let mut cap_id = 0usize;
            let mut cap_rest = 0usize;
            for (e, &c) in coordinates.iter().enumerate() {
                let l = old[j ^ c as usize].len();
                if has_identity && e == 0 {
                    cap_id += l;
                } else {
                    cap_rest += l;
                }
            }
            run.reset(0, cap_id, cap_rest);
        }
        if output_major {
            gather_local_output_major(&old, &mut runs, ptm, &coordinates, has_identity, true);
        } else {
            gather_local_input_major(&old, &mut runs, ptm, &coordinates, has_identity, true);
        }
        let mut scratch = SortScratch::<1>::default();
        for run in runs.iter_mut() {
            SortKernel::Comparison.sort(&mut run.rest, &mut scratch);
        }
        runs
    };
    let a = gather(false);
    let b = gather(true);
    assert!(
        a.iter().map(GatherRun::len).sum::<usize>() > 0,
        "gather produced nothing — the coset assembly is wrong"
    );
    // Merge before comparing: equal keys may land in either order under the key-only sort.
    let merge_run =
        |j: usize, run: &GatherRun<1>| -> (Vec<[u64; 1]>, Vec<[u64; 1]>, Vec<Complex64>) {
            let src = &old[j];
            assert_eq!(src.len(), run.identity.len(), "dense id must be 1:1");
            let identity = ColumnsRef {
                x: &src.x,
                z: &src.z,
                coeff: &run.identity.coeff,
            };
            let mut merged = BucketColumns::default();
            merge_identity_and_rest::<1, AlwaysKeep>(
                identity,
                run.rest.view(),
                &mut merged,
                &AlwaysKeep,
            );
            (merged.x, merged.z, merged.coeff)
        };
    for (j, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
        let (max, maz, mac) = merge_run(j, ra);
        let (mbx, mbz, mbc) = merge_run(j, rb);
        assert_eq!(max, mbx, "run {j}: merged keys (x) diverge");
        assert_eq!(maz, mbz, "run {j}: merged keys (z) diverge");
        assert_eq!(mac.len(), mbc.len(), "run {j}: merged term count diverges");
        for (i, (ca, cb)) in mac.iter().zip(mbc.iter()).enumerate() {
            let d = (ca - cb).norm();
            assert!(
                d < TOL,
                "run {j} term {i}: merged coefficients {ca} vs {cb} (delta {d:e})"
            );
        }
    }
}

/// Pins the dense/sparse identity classification per built-in.
#[test]
fn identity_density_classification() {
    let hash = Gf2Hash::<1>::new(12, 6, 0xD1CE);
    let check = |channel: &dyn Channel<1>, want: bool, label: &str| {
        let prepared = channel.prepare(&hash, false).unwrap();
        let span = Gf2Span::new(&prepared.bucket_deltas(), 6);
        match DeltaPlan::new(&prepared, &span, LayerKnobs::default()) {
            DeltaPlan::Local { dense_identity, .. } => {
                assert_eq!(dense_identity, want, "{label}")
            }
            DeltaPlan::Rotation { .. } => panic!("{label}: expected a Local plan"),
        }
    };
    // Dense: every source row emits an id row.
    check(&sqrt_swap_w1(1, 5), true, "gu2q");
    check(
        &PauliRotation::new(PauliString::<1>::z(2), 0.3),
        true,
        "rot_z",
    );
    check(
        &PtmChannel::amplitude_damping(5, 0.3),
        true,
        "amplitude_damping",
    );
    // Sparse: the id amplitude vanishes on some patterns (CNOT: 12 of 16; H: 2 of 4).
    check(&PtmChannel::cnot(1, 4), false, "cnot");
    check(&PtmChannel::hadamard(3), false, "h");
}

/// Pins per built-in which sort kernel its layer gets and the two quantities that decide it.
#[test]
fn radix_sort_kernel_is_selected_only_for_dense_ptms() {
    let hash = Gf2Hash::<1>::new(12, 8, 0xD1CE);
    let rest_streams = |channel: &dyn Channel<1>, label: &str| -> (usize, bool, f64) {
        let prepared = channel.prepare(&hash, false).unwrap();
        let span = Gf2Span::new(&prepared.bucket_deltas(), 8);
        match DeltaPlan::new(&prepared, &span, LayerKnobs::default()) {
            DeltaPlan::Local {
                ptm,
                has_identity,
                sort_kernel,
                ..
            } => (
                ptm.deltas().len() - has_identity as usize,
                sort_kernel == SortKernel::Radix,
                rest_rows_per_key(ptm),
            ),
            // A wide rotation has one rest stream and never takes the radix kernel.
            DeltaPlan::Rotation { .. } => {
                assert!(
                    label.starts_with("rot"),
                    "{label}: unexpected Rotation plan"
                );
                (1, false, 0.0)
            }
        }
    };
    // (label, rest streams, rest_rows_per_key, radix).
    let expect: &[(&str, usize, f64, bool)] = &[
        ("haar_su4", 15, 14.0, true),
        ("gu2q_sqrt_swap", 3, 3.0, false),
        ("cnot", 3, 1.0, true),
        ("cz", 3, 1.0, true),
        ("swap", 3, 1.0, true),
        ("h", 1, 1.0, false),
        ("s", 1, 1.0, false),
        ("rot_zz", 1, 1.0, false),
        ("depolarizing", 0, 0.0, false),
    ];
    let mut selected = Vec::new();
    let cases: Vec<(&str, Box<dyn Channel<1>>)> = vec![
        ("haar_su4", Box::new(haar_su4(1, 5))),
        ("gu2q_sqrt_swap", Box::new(sqrt_swap_w1(1, 5))),
        ("cnot", Box::new(PtmChannel::cnot(1, 5))),
        ("cz", Box::new(PtmChannel::cz(1, 5))),
        ("swap", Box::new(PtmChannel::swap(1, 5))),
        ("h", Box::new(PtmChannel::hadamard(3))),
        ("s", Box::new(PtmChannel::phase_gate(3))),
        (
            "rot_zz",
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<1>::z(1);
                    g.mul_assign(&PauliString::<1>::z(5));
                    g
                },
                0.3,
            )),
        ),
        ("depolarizing", Box::new(PtmChannel::depolarizing(3, 0.05))),
    ];
    assert_eq!(cases.len(), expect.len());
    for ((label, channel), &(want_label, want_streams, want_rpk, want_radix)) in
        cases.iter().zip(expect)
    {
        assert_eq!(*label, want_label);
        let (streams, radix, rpk) = rest_streams(channel.as_ref(), label);
        assert_eq!(streams, want_streams, "{label}: rest streams");
        assert!(
            (rpk - want_rpk).abs() < 1e-9,
            "{label}: rest_rows_per_key {rpk}, want {want_rpk}",
        );
        assert_eq!(
            radix, want_radix,
            "{label}: {streams} rest streams, {rpk} rows per key, radix = {radix}",
        );
        if radix {
            selected.push(*label);
        }
    }
    // Arm one: only the dense SU(4). Arm two: the two-qubit Cliffords; sqrt(SWAP) has `cnot`'s stream count but overlapping streams.
    assert_eq!(
        selected,
        vec!["haar_su4", "cnot", "cz", "swap"],
        "the radix gate fired on an unexpected set of channels",
    );
}

/// A partition's restricted PTM can look more disjoint than the channel, so the plan must read the channel-wide overlap from `LayerKnobs`.
#[test]
fn a_partitioned_plan_reads_the_channel_wide_overlap() {
    let hash = Gf2Hash::<1>::new(12, 8, 0xD1CE);
    let gu2q = sqrt_swap_w1(1, 5);
    let prepared = gu2q.prepare(&hash, false).unwrap();
    let Prepared::Local(full) = &prepared else {
        panic!("sqrt(SWAP) prepares to a Local plan");
    };
    assert!(
        (rest_rows_per_key(full) - 3.0).abs() < 1e-9,
        "the channel-wide value is the one the gate must see",
    );

    // The identity and one rest entry: the most misleading local view.
    let mut keep = vec![false; full.deltas().len()];
    keep[0] = true;
    keep[1] = true;
    let local = Prepared::Local(full.retain_entries(&keep));
    let Prepared::Local(restricted) = &local else {
        unreachable!()
    };
    assert!(
        (rest_rows_per_key(restricted) - 1.0).abs() < 1e-9,
        "the restricted view must be the misleading one, or this test is vacuous",
    );

    let span = Gf2Span::new(&local.bucket_deltas(), 8);
    let radix = |knobs: LayerKnobs<'_>| match DeltaPlan::new(&local, &span, knobs) {
        DeltaPlan::Local { sort_kernel, .. } => sort_kernel == SortKernel::Radix,
        DeltaPlan::Rotation { .. } => unreachable!(),
    };
    // What the partitioned layer actually passes.
    assert!(
        !radix(LayerKnobs {
            rest_streams: Some(3),
            rows_per_key: Some(rest_rows_per_key(full)),
            ..LayerKnobs::default()
        }),
        "a partition of a fan-out channel must keep the comparison kernel",
    );
    // The count override alone is not enough.
    assert!(
        radix(LayerKnobs {
            rest_streams: Some(3),
            ..LayerKnobs::default()
        }),
        "the restricted PTM really does mislead the overlap arm",
    );
}

// ---- multi-layer, staying bucketed ----

#[test]
fn many_layers_without_converting_out() {
    // Convert in once, run many layers, convert out once, against the oracle.
    let input = rand_sum::<1>(800, 8, 0x7001);
    let chans: Vec<Box<dyn Channel<1>>> = vec![
        Box::new(PtmChannel::hadamard(0)),
        Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.3)),
        Box::new(PtmChannel::cnot(1, 5)),
        Box::new(PtmChannel::depolarizing(3, 0.05)),
        Box::new(PtmChannel::phase_gate(6)),
        Box::new(PauliRotation::new(
            {
                let mut g = PauliString::<1>::z(1);
                g.mul_assign(&PauliString::<1>::z(4));
                g
            },
            0.2,
        )),
    ];

    let mut want = input.clone();
    for channel in &chans {
        want = naive_apply_layer(&want, channel.as_ref(), &AlwaysKeep, false);
    }

    let hash = Gf2Hash::<1>::new(8, 5, 0x77);
    let mut b = input.clone().with_hash(hash);
    let mut scratch = LayerScratch::<1>::new();
    for channel in &chans {
        let prepared = channel.prepare(b.hash(), false).unwrap();
        apply_layer_bucketed(&mut b, &prepared, &AlwaysKeep, &mut scratch);
    }
    let got = b;
    assert_terms_close(&got, &want, TOL, "six layers");
}

#[test]
fn layers_survive_a_rebucket_in_between() {
    let input = rand_sum::<1>(800, 8, 0x7002);
    let h = PtmChannel::hadamard(0);
    let rot = PauliRotation::new(PauliString::<1>::z(2), 0.3);

    let want = naive_apply_layer(
        &naive_apply_layer(&input, &h, &AlwaysKeep, false),
        &rot,
        &AlwaysKeep,
        false,
    );

    let hash = Gf2Hash::<1>::new(8, 2, 0x77);
    let mut b = input.clone().with_hash(hash);
    let mut scratch = LayerScratch::<1>::new();

    let prepared = h.prepare(b.hash(), false).unwrap();
    apply_layer_bucketed(&mut b, &prepared, &AlwaysKeep, &mut scratch);
    b.rebucket(32, 1);
    let prepared = rot.prepare(b.hash(), false).unwrap();
    apply_layer_bucketed(&mut b, &prepared, &AlwaysKeep, &mut scratch);

    assert_terms_close(&b, &want, TOL, "layer, rebucket, layer");
}

#[test]
fn an_empty_sum_survives_a_layer() {
    let input = PauliSum::<1>::empty(8);
    let rot = PauliRotation::new(PauliString::<1>::z(2), 0.3);
    let out = bucketed_layer(&input, &rot, &AlwaysKeep, false, 4, 0x1);
    assert!(out.is_empty());
}

/// At `bits = 0` the whole sum is one coset on the ordinary code path.
#[test]
fn single_bucket_sum_is_one_serial_coset() {
    let input = rand_sum::<1>(600, 8, 0xB1);
    for channel in [
        Box::new(PauliRotation::new(
            {
                let mut g = PauliString::<1>::z(1);
                g.mul_assign(&PauliString::<1>::z(5));
                g
            },
            0.37,
        )) as Box<dyn Channel<1>>,
        Box::new(PtmChannel::cnot(2, 6)),
    ] {
        let got = bucketed_layer(&input, channel.as_ref(), &AlwaysKeep, false, 0, 0xEE);
        assert_eq!(got.num_buckets(), 1);
        let want = naive_apply_layer(&input, channel.as_ref(), &AlwaysKeep, false);
        // Tolerance: the oracle sums equal keys in hash-map order.
        assert_terms_close(&got, &want, TOL, "bits=0 single coset");
    }
}

/// A wide rotation whose generator hashes to bucket delta 0 (`r = 0`).
#[test]
fn wide_rotation_with_colliding_bucket_delta() {
    // Weight 4 > MAX_LOCAL_SUPPORT, so it prepares as `Prepared::Rotation`.
    let mut gen = PauliString::<1>::z(0);
    for q in [2, 4, 6] {
        gen.mul_assign(&PauliString::<1>::x(q));
    }
    let rot = PauliRotation::new(gen, 0.53);
    let input = rand_sum::<1>(800, 8, 0xC0111);

    // A seed whose 3-bit hash sends the generator's key delta to bucket 0.
    let bits = 3u8;
    let mut chosen = None;
    for seed in 0u64..4096 {
        let hash = Gf2Hash::<1>::new(8, bits, seed);
        if hash.bucket_of(&gen.x, &gen.z) == 0 {
            chosen = Some(seed);
            break;
        }
    }
    let seed = chosen.expect("no seed with H·P = 0 in 4096 tries");

    let hash = Gf2Hash::<1>::new(8, bits, seed);
    let mut b = input.clone().with_hash(hash);
    let prepared = rot.prepare(b.hash(), false).unwrap();
    match &prepared {
        Prepared::Rotation(r) => {
            assert_eq!(
                r.bucket_delta_generator, r.bucket_delta_identity,
                "seed search failed to produce the collision"
            );
        }
        _ => panic!("weight-4 rotation must prepare as Rotation"),
    }
    let mut scratch = LayerScratch::<1>::new();
    apply_layer_bucketed(&mut b, &prepared, &AlwaysKeep, &mut scratch);

    let want = naive_apply_layer(&input, &rot, &AlwaysKeep, false);
    // Tolerance, not bitwise: the oracle sums equal keys in hashmap order.
    assert_terms_close(&b, &want, TOL, "H·P = 0 collision");
}

/// One `LayerScratch` serves layers of every prepared shape back to back.
#[test]
fn in_place_layers_share_one_scratch_across_channel_types() {
    let input = rand_sum::<2>(1500, 10, 0x5CA7C4);
    let rot = PauliRotation::new(
        {
            let mut g = PauliString::<2>::z(1);
            g.mul_assign(&PauliString::<2>::x(7));
            g
        },
        0.29,
    );
    let cnot = PtmChannel::cnot(3, 8);
    let h = Complex64::new(0.5, 0.5);
    let hc = Complex64::new(0.5, -0.5);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let gu2q = PtmChannel::unitary_2q(
        2,
        6,
        [
            [one, zero, zero, zero],
            [zero, h, hc, zero],
            [zero, hc, h, zero],
            [zero, zero, zero, one],
        ],
    );
    let channels: [&dyn Channel<2>; 3] = [&rot, &cnot, &gu2q];

    let hash = Gf2Hash::<2>::new(10, 5, 0xD00D);
    let mut b = input.clone().with_hash(hash);
    let mut scratch = LayerScratch::<2>::new();
    let mut want = input;
    for channel in channels {
        let prepared = channel.prepare(b.hash(), false).unwrap();
        apply_layer_bucketed(&mut b, &prepared, &AlwaysKeep, &mut scratch);
        want = naive_apply_layer(&want, channel, &AlwaysKeep, false);
    }
    // Tolerance, not bitwise: the oracle sums equal keys in hashmap order.
    assert_terms_close(&b, &want, TOL, "rot → cnot → gu2q through one scratch");
}

/// Once the working set stops growing, a layer allocates nothing.
#[test]
fn capacity_stabilizes_across_repeated_layers() {
    let input = rand_sum::<1>(2000, 10, 0xCAFE);
    let hash = Gf2Hash::<1>::new(10, 4, 0xF00);
    let mut b = input.with_hash(hash);
    let hgate = PtmChannel::hadamard(3);
    let prepared = hgate.prepare(b.hash(), false).unwrap();
    let mut scratch = LayerScratch::<1>::new();

    let total_capacity = |s: &PauliSum<1>, scratch: &LayerScratch<1>| -> usize {
        let bucket_cap: usize = (0..s.num_buckets())
            .map(|i| {
                let (x, _, _) = s.bucket(i);
                // Bucket capacity is not observable through the slice view, so count lengths there and real capacities in the scratch.
                x.len()
            })
            .sum();
        let old_cap: usize = scratch.task.old.iter().map(|c| c.x.capacity()).sum();
        let run_cap: usize = scratch.task.runs.iter().map(|r| r.rest.x.capacity()).sum();
        let sort_cap = scratch.task.sort.total_capacity();
        bucket_cap
            + old_cap
            + run_cap
            + sort_cap
            + scratch.order.capacity()
            + scratch.staging.capacity()
    };

    let mut snapshots = Vec::new();
    for _ in 0..4 {
        apply_layer_bucketed(&mut b, &prepared, &AlwaysKeep, &mut scratch);
        snapshots.push(total_capacity(&b, &scratch));
    }
    assert_eq!(
        snapshots[2], snapshots[3],
        "scratch/bucket footprint still growing at layer 4: {snapshots:?}"
    );
}

/// A channel whose delta set `{0, a, b}` is not XOR-closed, so only the span's cosets partition the buckets.
#[test]
fn coset_path_is_correct_for_a_non_subspace_delta_set() {
    struct ThreeDeltas;
    impl<const W: usize> Channel<W> for ThreeDeltas {
        fn apply_adjoint(
            &self,
            input_x: &[u64; W],
            input_z: &[u64; W],
            coeff: Complex64,
            out: &mut crate::channel::OutputBuffer<W>,
        ) {
            self.apply(input_x, input_z, coeff, out);
        }
        fn support(&self) -> [u64; W] {
            crate::channel::support_mask(&[0, 1])
        }
        fn apply(
            &self,
            input_x: &[u64; W],
            input_z: &[u64; W],
            coeff: Complex64,
            out: &mut crate::channel::OutputBuffer<W>,
        ) {
            // v (0.5) + v⊕x₀ (0.3) + v⊕x₁ (0.2), so `a ⊕ b = x₀x₁` is never emitted.
            out.push(*input_x, *input_z, coeff * 0.5);
            let mut xa = *input_x;
            xa[0] ^= 1;
            out.push(xa, *input_z, coeff * 0.3);
            let mut xb = *input_x;
            xb[0] ^= 2;
            out.push(xb, *input_z, coeff * 0.2);
        }
    }

    let channel = ThreeDeltas;
    let input = rand_sum::<1>(1200, 8, 0xAB5EA7);
    let want = naive_apply_layer(&input, &channel, &AlwaysKeep, false);
    for bits in [0u8, 2, 5] {
        let got = bucketed_layer(&input, &channel, &AlwaysKeep, false, bits, 0x7EA);
        // Tolerance: three summands per key, summed by the oracle in hash-map order.
        assert_terms_close(
            &got,
            &want,
            TOL,
            &format!("non-subspace deltas, bits={bits}"),
        );
    }
}

/// The [`ExtraRows`] hook: rows handed to a layer from outside.
mod extra_rows_tests {
    use super::*;
    use super::{assert_terms_close, bucketed_layer, AlwaysKeep};
    use crate::channel::rotation::PauliRotation;
    use crate::channel::Channel;
    use crate::channel::PtmChannel;
    use crate::pauli_string::PauliString;
    use crate::pauli_sum::accumulator::BuildAccumulator;
    use crate::pauli_sum::hash::Gf2Hash;
    use crate::pauli_sum::PauliSum;

    use crate::test_support::{naive_apply_layer, rand_sum};
    use crate::truncation::builtin::CoefficientThreshold;
    use std::collections::{HashMap, HashSet};

    const TOL: f64 = 1e-11;

    /// One row to inject: key columns plus coefficient.
    type Row<const W: usize> = ([u64; W], [u64; W], Complex64);

    /// Rows to inject, keyed by original output bucket and filed by [`Self::push`] from the layer's own hash.
    #[derive(Default)]
    struct Injected<const W: usize> {
        rows: HashMap<u32, Vec<Row<W>>>,
    }

    impl<const W: usize> Injected<W> {
        fn push(&mut self, hash: &Gf2Hash<W>, x: [u64; W], z: [u64; W], c: Complex64) {
            let beta = hash.bucket_of(&x, &z);
            self.rows.entry(beta).or_default().push((x, z, c));
        }

        fn all(&self) -> impl Iterator<Item = &Row<W>> {
            self.rows.values().flatten()
        }
    }

    impl<const W: usize> ExtraRows<W> for Injected<W> {
        const NEEDS_BETA: bool = true;

        fn count(&self, beta: u32) -> usize {
            self.rows.get(&beta).map_or(0, Vec::len)
        }

        fn append_into(&self, beta: u32, rest: &mut BucketColumns<W>) {
            let Some(rows) = self.rows.get(&beta) else {
                return;
            };
            for &(x, z, c) in rows {
                rest.x.push(x);
                rest.z.push(z);
                rest.coeff.push(c);
            }
        }
    }

    /// One layer through [`apply_layer_bucketed_with`], at a fixed partition.
    fn layer_with_extra<const W: usize, T, X>(
        input: &PauliSum<W>,
        channel: &dyn Channel<W>,
        policy: &T,
        bits: u8,
        seed: u64,
        extra: &X,
    ) -> PauliSum<W>
    where
        T: TruncationPolicy<W> + ?Sized,
        X: ExtraRows<W> + Sync,
    {
        let hash = Gf2Hash::<W>::new(input.num_qubits(), bits, seed);
        let mut b = input.clone().with_hash(hash);
        let prepared = channel
            .prepare(b.hash(), false)
            .expect("channel could not be prepared");
        let mut scratch = LayerScratch::<W>::new();
        apply_layer_bucketed_with(
            &mut b,
            &prepared,
            policy,
            &mut scratch,
            extra,
            LayerKnobs::default(),
        );
        b
    }

    /// The oracle: the naive layer plus the injected rows, summed per key and only then filtered.
    fn expected<const W: usize, T>(
        input: &PauliSum<W>,
        channel: &dyn Channel<W>,
        policy: &T,
        injected: &Injected<W>,
    ) -> PauliSum<W>
    where
        T: TruncationPolicy<W> + ?Sized,
    {
        let base = naive_apply_layer(input, channel, &AlwaysKeep, false);
        let mut map: HashMap<([u64; W], [u64; W]), Complex64> = HashMap::new();
        for (x, z, c) in base.iter() {
            *map.entry((*x, *z)).or_insert(ZERO) += c;
        }
        for &(x, z, c) in injected.all() {
            *map.entry((x, z)).or_insert(ZERO) += c;
        }
        let mut accumulator = BuildAccumulator::<W>::with_capacity(input.num_qubits(), map.len());
        for ((x, z), c) in map {
            if c == ZERO || !policy.keep_term(&x, &z, c) {
                continue;
            }
            accumulator.add_term(PauliString::<W> { x, z }, c);
        }
        accumulator.finalize()
    }

    /// A handful of rows for a layer: some on keys the fixture already carries (colliding with the local output, so they must be summed), some on keys it does not.
    fn injection_for(input: &PauliSum<1>, hash: &Gf2Hash<1>) -> Injected<1> {
        let mut injected = Injected::<1>::default();
        let mut seen: HashSet<([u64; 1], [u64; 1])> = HashSet::new();
        for (i, (x, z, _)) in input.iter().enumerate() {
            if i % 137 == 0 && seen.insert((*x, *z)) {
                injected.push(hash, *x, *z, Complex64::new(0.25, -0.5));
            }
        }
        // 8-qubit fixture, so keys stay inside the low byte.
        for k in 0u64..5 {
            let x = [(0xA5u64 ^ k.wrapping_mul(31)) & 0xFF];
            let z = [(0x3Cu64 ^ k.wrapping_mul(17)) & 0xFF];
            if seen.insert((x, z)) {
                injected.push(hash, x, z, Complex64::new(-0.75, 0.125));
            }
        }
        injected
    }

    /// Injected rows reach the output bucket named by their original index, under both the permuted and the identity handle layouts, and are deduplicated against the local output rather than appended beside it.
    #[test]
    fn injected_rows_land_in_their_original_bucket_and_are_summed() {
        let input = rand_sum::<1>(600, 8, 0xE47A);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
        let cnot = PtmChannel::cnot(1, 5);
        // At `bits = 5` both channels permute non-trivially and run the parallel chunk loop (16 and 8 cosets).
        let cases: [(&str, &dyn Channel<1>, u8, usize); 4] = [
            ("rot bits=5", &rot, 5, 1),
            ("cnot bits=5", &cnot, 5, 2),
            ("rot bits=0", &rot, 0, 0),
            ("cnot bits=0", &cnot, 0, 0),
        ];
        for (label, channel, bits, want_r) in cases {
            // Pick the hash for the intended span rank, so a colliding draw cannot degrade a `bits = 5` case.
            let seed = (0u64..4096)
                .find(|&s| {
                    let hash = Gf2Hash::<1>::new(8, bits, s);
                    let prepared = channel.prepare(&hash, false).unwrap();
                    Gf2Span::new(&prepared.bucket_deltas(), bits).dimension() == want_r
                })
                .unwrap_or_else(|| panic!("{label}: no seed of rank {want_r} in 4096 tries"));
            let hash = Gf2Hash::<1>::new(8, bits, seed);

            let injected = injection_for(&input, &hash);
            assert!(
                injected.all().count() >= 6,
                "{label}: too few injected rows"
            );

            // Non-vacuity: both the summed and the inserted case occur.
            let plain = bucketed_layer(&input, channel, &AlwaysKeep, false, bits, seed);
            let collided = injected
                .all()
                .filter(|(x, z, _)| plain.get(x, z).is_some())
                .count();
            let fresh = injected.all().count() - collided;
            assert!(
                collided > 0 && fresh > 0,
                "{label}: want both colliding and fresh keys (collided={collided} fresh={fresh})"
            );

            let got = layer_with_extra(&input, channel, &AlwaysKeep, bits, seed, &injected);
            let want = expected(&input, channel, &AlwaysKeep, &injected);
            assert_eq!(got.len(), plain.len() + fresh, "{label}: term count");
            assert_terms_close(&got, &want, TOL, label);
        }
    }

    /// The coset loop checks that each append lands in the rest stream, row for row as `count` promised.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "received rows where ExtraRows::count promised")]
    fn an_append_beyond_its_count_is_caught() {
        /// An [`ExtraRows`] whose `count` promises fewer rows than `append_into` pushes.
        struct Undercounted<const W: usize>(Injected<W>);

        impl<const W: usize> ExtraRows<W> for Undercounted<W> {
            const NEEDS_BETA: bool = true;

            fn count(&self, beta: u32) -> usize {
                self.0.count(beta).saturating_sub(1)
            }

            fn append_into(&self, beta: u32, rest: &mut BucketColumns<W>) {
                self.0.append_into(beta, rest)
            }
        }

        let input = rand_sum::<1>(600, 8, 0xE47C);
        let cnot = PtmChannel::cnot(1, 5);
        let hash = Gf2Hash::<1>::new(8, 3, 0x51);
        let injected = Undercounted(injection_for(&input, &hash));
        let _ = layer_with_extra(&input, &cnot, &AlwaysKeep, 3, 0x51, &injected);
    }

    /// An empty `Injected` (`NEEDS_BETA = true`) gives exactly what `NoExtra` gives.
    #[test]
    fn an_empty_injection_changes_nothing() {
        let input = rand_sum::<1>(600, 8, 0xE47B);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
        let cnot = PtmChannel::cnot(1, 5);
        for channel in [&rot as &dyn Channel<1>, &cnot as &dyn Channel<1>] {
            for bits in [0u8, 2, 5] {
                let empty = Injected::<1>::default();
                let got = layer_with_extra(&input, channel, &AlwaysKeep, bits, 0x7A58, &empty);
                let plain = bucketed_layer(&input, channel, &AlwaysKeep, false, bits, 0x7A58);
                assert_eq!(
                    got.to_arrays(),
                    plain.to_arrays(),
                    "bits={bits}: the empty hook perturbed the layer"
                );
            }
        }
    }

    /// `keep_term` runs on the sum of the local and the injected contribution, not on either alone: two coefficients that each clear the threshold can cancel to below it.
    #[test]
    fn keep_term_sees_local_plus_injected() {
        let mut accumulator = BuildAccumulator::<1>::with_capacity(8, 2);
        accumulator.add_term(PauliString::<1>::z(0), Complex64::new(0.5, 0.0));
        accumulator.add_term(PauliString::<1>::z(1), Complex64::new(1.0, 0.0));
        let input = accumulator.finalize();
        // H on qubit 0 sends Z₀ → X₀ with amplitude 1 and leaves Z₁ alone.
        let h = PtmChannel::hadamard(0);
        let x0 = PauliString::<1>::x(0);
        let z1 = PauliString::<1>::z(1);

        for bits in [0u8, 3] {
            let hash = Gf2Hash::<1>::new(8, bits, 0x9F);
            let mut injected = Injected::<1>::default();
            injected.push(&hash, x0.x, x0.z, Complex64::new(-0.4999999, 0.0));

            let policy = CoefficientThreshold(1e-6);
            let got = layer_with_extra(&input, &h, &policy, bits, 0x9F, &injected);
            assert!(
                got.get(&x0.x, &x0.z).is_none(),
                "bits={bits}: a term that only survives unsummed was kept"
            );
            assert_eq!(got.len(), 1, "bits={bits}");
            assert!(
                (got.get(&z1.x, &z1.z).unwrap() - Complex64::new(1.0, 0.0)).norm() < TOL,
                "bits={bits}: the untouched term must survive"
            );

            // Without the threshold the residue is the sum of the two contributions.
            let kept = layer_with_extra(&input, &h, &AlwaysKeep, bits, 0x9F, &injected);
            let want = expected(&input, &h, &AlwaysKeep, &injected);
            assert_terms_close(&kept, &want, TOL, &format!("no threshold bits={bits}"));
            assert!(
                (kept.get(&x0.x, &x0.z).unwrap() - Complex64::new(1.0000000000287557e-7, 0.0))
                    .norm()
                    < 1e-18,
                "bits={bits}: residue {:?}",
                kept.get(&x0.x, &x0.z),
            );
        }
    }
}

mod finalize_tests {
    use super::*;
    use super::{assert_same_terms, assert_terms_close, naive_apply_layer, rand_sum};
    use crate::channel::rotation::PauliRotation;
    use crate::channel::Channel;
    use crate::channel::PtmChannel;
    use crate::pauli_string::PauliString;
    use crate::pauli_sum::hash::Gf2Hash;
    use crate::pauli_sum::PauliSum;
    use crate::truncation::builtin::{And, CoefficientThreshold, TopN};

    #[test]
    fn and_runs_both_finalizers_bucketed() {
        // TopN(n) twice with different n must behave like the tighter one.
        let input = rand_sum::<1>(1000, 8, 0x7777);
        let policy = And(TopN(400), TopN(120));
        let mut flat = input.clone();
        policy.finalize_layer(&mut flat);

        let hash = Gf2Hash::<1>::new(8, 5, 0x99);
        let mut b = input.clone().with_hash(hash);
        policy.finalize_layer(&mut b);
        b.assert_invariants();
        let got = b;
        assert_eq!(got.len(), 120);
        assert_same_terms(&got, &flat, "and of two top_n");
    }

    /// A custom `finalize_layer` via `retain` acts on the bucketed sum, independent of the partition.
    #[test]
    fn a_custom_finalizer_runs_on_the_bucketed_sum() {
        /// Drops every term with negative real part.
        struct DropNegativeReal;
        impl<const W: usize> TruncationPolicy<W> for DropNegativeReal {
            fn finalize_layer(&self, sum: &mut PauliSum<W>) {
                sum.retain(|_x, _z, c| c.re >= 0.0);
            }
        }

        let input = rand_sum::<1>(800, 8, 0x9999);
        let mut flat = input.clone();
        DropNegativeReal.finalize_layer(&mut flat);
        assert!(
            flat.len() < input.len(),
            "the custom policy dropped nothing"
        );

        for bits in [0u8, 3, 7] {
            let hash = Gf2Hash::<1>::new(8, bits, 0x99);
            let mut b = input.clone().with_hash(hash);
            DropNegativeReal.finalize_layer(&mut b);
            b.assert_invariants();
            let got = b;
            assert_same_terms(&got, &flat, &format!("bits={bits}"));
        }
    }

    /// Layer then finalize, repeatedly, as `propagate` does.
    #[test]
    fn interleaved_layers_and_finalizers_match_the_naive_sequence() {
        let input = rand_sum::<1>(1200, 8, 0xAAAA);
        let policy = And(CoefficientThreshold(1e-9), TopN(300));
        let chans: Vec<Box<dyn Channel<1>>> = vec![
            Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.37)),
            Box::new(PtmChannel::hadamard(0)),
            Box::new(PauliRotation::new(PauliString::<1>::x(5), 0.21)),
        ];

        let mut want = input.clone();
        for channel in &chans {
            want = naive_apply_layer(&want, channel.as_ref(), &policy, false);
            policy.finalize_layer(&mut want);
        }

        let hash = Gf2Hash::<1>::new(8, 5, 0xBB);
        let mut b = input.clone().with_hash(hash);
        let mut scratch = LayerScratch::<1>::new();
        for channel in &chans {
            let prepared = channel.prepare(b.hash(), false).unwrap();
            apply_layer_bucketed(&mut b, &prepared, &policy, &mut scratch);
            policy.finalize_layer(&mut b);
        }
        let got = b;

        assert_terms_close(&got, &want, 1e-11, "3 truncated layers");
    }
}

mod tie_tests {
    use super::canonical_triples;
    use super::*;
    use crate::pauli_sum::hash::Gf2Hash;
    use crate::pauli_sum::PauliSum;
    use crate::test_support::{
        first_n_by_rank, magnitude_rank, octave_rank, storage_triples, tie_heavy_sum,
    };
    use crate::truncation::builtin::{ApproxTopN, TopN};

    /// The kept terms whose `rank` is strictly above the `n`-th largest rank of `input`, in key order.
    fn above_the_boundary<const W: usize>(
        input: &PauliSum<W>,
        kept: &PauliSum<W>,
        n: usize,
        rank: fn(Complex64) -> u64,
    ) -> Vec<([u64; W], [u64; W], Complex64)> {
        let mut ranks: Vec<u64> = input.iter().map(|(_, _, c)| rank(c)).collect();
        ranks.sort_unstable_by(|a, b| b.cmp(a));
        let boundary = ranks[n - 1];
        canonical_triples(kept)
            .into_iter()
            .filter(|t| rank(t.2) > boundary)
            .collect()
    }

    /// `TopN(n)` keeps exactly `n` of a tie-dense sum at every bucket count and hash seed: the same terms above the boundary, and the stable-sort transcription of the rule in each storage order.
    #[test]
    fn top_n_keeps_exactly_n_at_every_bucket_count_and_seed() {
        let input = tie_heavy_sum::<1>(2000, 8, 0x7135);
        // 700 cuts inside the group of magnitude-0.5 terms; 1000 near a group boundary.
        for n in [3usize, 700, 1000, 1900] {
            let reference = {
                let mut b = input.clone().with_hash(Gf2Hash::<1>::new(8, 0, 0x99));
                TopN(n).finalize_layer(&mut b);
                b
            };
            for (bits, seed) in [(1u8, 0x99u64), (2, 0x99), (5, 1), (5, 13), (9, 0x99)] {
                let hashed = input.clone().with_hash(Gf2Hash::<1>::new(8, bits, seed));
                let mut b = hashed.clone();
                TopN(n).finalize_layer(&mut b);
                b.assert_invariants();
                let what = format!("n={n} bits={bits} seed={seed}");
                assert_eq!(b.len(), n, "{what}: kept count");
                assert_eq!(
                    storage_triples(&b),
                    first_n_by_rank(&storage_triples(&hashed), n, magnitude_rank),
                    "{what}: rule"
                );
                assert_eq!(
                    above_the_boundary(&input, &b, n, magnitude_rank),
                    above_the_boundary(&input, &reference, n, magnitude_rank),
                    "{what}: terms above the boundary"
                );
            }
        }
    }

    /// The same for `ApproxTopN(n)`, whose boundary is an octave; `n = 700` cuts between the first two of `tie_heavy_sum`'s four octaves.
    #[test]
    fn approx_top_n_keeps_exactly_n_at_every_bucket_count_and_seed() {
        let input = tie_heavy_sum::<1>(2000, 8, 0x7135);
        let n = 700;
        let reference = {
            let mut b = input.clone().with_hash(Gf2Hash::<1>::new(8, 0, 0x99));
            ApproxTopN(n).finalize_layer(&mut b);
            b
        };
        assert_eq!(
            above_the_boundary(&input, &reference, n, octave_rank).len(),
            input.iter().filter(|(_, _, c)| c.norm() == 1.0).count(),
            "the fixture must cut between the top two octaves"
        );
        for (bits, seed) in [(1u8, 0x99u64), (2, 0x99), (4, 3), (6, 13), (9, 0x99)] {
            let hashed = input.clone().with_hash(Gf2Hash::<1>::new(8, bits, seed));
            let mut b = hashed.clone();
            ApproxTopN(n).finalize_layer(&mut b);
            b.assert_invariants();
            let what = format!("bits={bits} seed={seed}");
            assert_eq!(b.len(), n, "{what}: kept count");
            assert_eq!(
                storage_triples(&b),
                first_n_by_rank(&storage_triples(&hashed), n, octave_rank),
                "{what}: rule"
            );
            assert_eq!(
                above_the_boundary(&input, &b, n, octave_rank),
                above_the_boundary(&input, &reference, n, octave_rank),
                "{what}: terms above the boundary"
            );
        }
    }
}
