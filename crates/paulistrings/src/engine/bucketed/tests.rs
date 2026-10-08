use super::coset_fill::{gather_local_input_major, gather_local_output_major, nonzero, GatherRun};
use super::*;
use crate::channel::clifford::{Clifford1Q, Clifford2Q};
use crate::channel::identity::IdentityChannel;
use crate::channel::noise::{AmplitudeDamping, Dephasing, Depolarizing};
use crate::channel::rotation::PauliRotation;
use crate::channel::Channel;
use crate::engine::merge::{merge2_into, sort_rows_with_scratch, SortScratch};
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::PauliSum;
use crate::phase::Phase;
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
    assert_eq!(run.x.len(), 0);
    run.push_if(true, [1], [2], c(1.0));
    run.push_if(false, [7], [7], c(9.0)); // discarded at len 1
    run.push_if(true, [3], [4], c(2.0)); // the last countable row
    assert_eq!(run.x.as_slice(), &[[1], [3]]);
    assert_eq!(run.z.as_slice(), &[[2], [4]]);
    assert_eq!(run.coeff.as_slice(), &[c(1.0), c(2.0)]);
    // The identity stream answers the same contract.
    run.reset(1, 1, 0);
    run.push_id_if(false, [5], [6], c(3.0));
    assert_eq!(run.id_x.len(), 0);
    run.push_id_if(true, [5], [6], c(3.0));
    assert_eq!(run.id_x.as_slice(), &[[5]]);
    assert_eq!(run.id_coeff.as_slice(), &[c(3.0)]);
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
pub(super) fn haar_su4(q0: u32, q1: u32) -> crate::channel::GeneralUnitary2Q {
    crate::channel::GeneralUnitary2Q::from_matrix(q0, q1, crate::test_support::haar_su4_matrix())
}

/// The term trace's state machine, independent of any propagation: `None` means off, `enable` is idempotent and non-destructive, `take` drains but stays on.
#[test]
fn term_trace_is_opt_in_and_drains_on_take() {
    let mut scratch = LayerScratch::<1>::new();
    assert!(scratch.take_term_trace().is_none(), "off by default");

    scratch.enable_term_trace();
    scratch.term_trace.as_mut().unwrap().terms_in.push(7);
    scratch.enable_term_trace(); // idempotent: must not clear the 7
    assert_eq!(
        scratch.take_term_trace(),
        Some(TermTrace {
            terms_in: vec![7],
            terms_out: vec![],
        })
    );
    assert_eq!(scratch.take_term_trace(), Some(TermTrace::default()));
}

/// The gate trace's state machine mirrors [`TermTrace`]'s: off by default, `enable` idempotent and non-destructive, `take` drains but stays on.
#[test]
fn gate_trace_is_opt_in_and_drains_on_take() {
    let mut scratch = LayerScratch::<1>::new();
    assert!(scratch.take_gate_trace().is_none(), "off by default");

    scratch.enable_gate_trace();
    scratch.gate_trace.as_mut().unwrap().circuit_index.push(3);
    scratch.enable_gate_trace(); // idempotent: must not clear the 3
    assert_eq!(
        scratch.take_gate_trace(),
        Some(GateTrace {
            circuit_index: vec![3],
            ..GateTrace::default()
        })
    );
    assert_eq!(scratch.take_gate_trace(), Some(GateTrace::default()));
}

/// `peak_terms` is the between-layer maximum, which includes the first `terms_in`.
#[test]
fn peak_terms_spans_the_first_input_and_every_output() {
    assert_eq!(TermTrace::default().peak_terms(), None);
    assert_eq!(
        TermTrace {
            terms_in: vec![9, 4],
            terms_out: vec![4, 6],
        }
        .peak_terms(),
        Some(9),
        "a shrinking first layer keeps the input as the peak"
    );
    assert_eq!(
        TermTrace {
            terms_in: vec![1, 5],
            terms_out: vec![5, 3],
        }
        .peak_terms(),
        Some(5)
    );
}

/// Run one layer through the bucketed engine, converting in and out.
pub(super) fn bucketed_layer<const W: usize, C, T>(
    input: &PauliSum<W>,
    ch: &C,
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
    let prep = ch
        .prepare(b.hash(), adjoint)
        .expect("channel could not be prepared");
    let mut scratch = LayerScratch::<W>::new();
    apply_layer_bucketed(&mut b, &prep, policy, &mut scratch);
    b
}

/// Keys must match exactly; coefficients only to tolerance.
pub(super) fn assert_sums_close<const W: usize>(got: &PauliSum<W>, want: &PauliSum<W>, what: &str) {
    assert_terms_close(got, want, TOL, what);
}

// ---- hand-checked behaviour ----

#[test]
fn h_conjugates_z_to_x() {
    let mut acc = BuildAccumulator::<1>::with_capacity(4, 1);
    acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
    let input = acc.finalize();
    let out = bucketed_layer(&input, &Clifford1Q::h(0), &AlwaysKeep, false, 4, 0x1);
    assert_eq!(out.len(), 1);
    let (x, z, c) = out.iter().next().unwrap();
    assert_eq!(*x, [1]);
    assert_eq!(*z, [0]);
    assert!((c - Complex64::new(1.0, 0.0)).norm() < TOL);
}

#[test]
fn cnot_propagates_z_on_the_control() {
    let mut acc = BuildAccumulator::<1>::with_capacity(4, 1);
    acc.add_term(PauliString::<1>::z(1), Phase::ONE, Complex64::new(1.0, 0.0));
    let input = acc.finalize();
    // I⊗Z under CNOT(0 -> 1) becomes Z⊗Z.
    let out = bucketed_layer(&input, &Clifford2Q::cnot(0, 1), &AlwaysKeep, false, 4, 0x1);
    assert_eq!(out.len(), 1);
    let (x, z, _) = out.iter().next().unwrap();
    assert_eq!(*z, [0b11]);
    assert_eq!(*x, [0]);
}

#[test]
fn a_rotation_fans_out_to_two_terms() {
    let mut acc = BuildAccumulator::<1>::with_capacity(4, 1);
    acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(1.0, 0.0));
    let input = acc.finalize();
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

    for (name, ch) in &channels {
        let cr: &dyn Channel<1> = ch.as_ref();
        for &adjoint in &[false, true] {
            for &bits in &[0u8, 1, 3, 6, 11] {
                let want = naive_apply_layer(&input, cr, &AlwaysKeep, adjoint);
                let got = bucketed_layer(&input, cr, &AlwaysKeep, adjoint, bits, 0xABCD);
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
    let input = rand_sum::<2>(3000, 128, 0xBEEF);
    let channels = crate::test_support::differential_channels_w2();
    for (name, ch) in &channels {
        let cr: &dyn Channel<2> = ch.as_ref();
        for &adjoint in &[false, true] {
            for &bits in &[2u8, 5, 9] {
                let want = naive_apply_layer(&input, cr, &AlwaysKeep, adjoint);
                let got = bucketed_layer(&input, cr, &AlwaysKeep, adjoint, bits, 0xABCD);
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
    let cnot = Clifford2Q::cnot(1, 5);

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
    let mut acc = BuildAccumulator::<1>::with_capacity(4, 2);
    acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(0.5, 0.0));
    acc.add_term(
        PauliString::<1>::y(0),
        Phase::ONE,
        Complex64::new(-0.4999999, 0.0),
    );
    let input = acc.finalize();
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
        (
            "depolarizing",
            Box::new(Depolarizing {
                support: [3],
                p: 0.11,
            }),
        ),
        (
            "dephasing",
            Box::new(Dephasing {
                support: [3],
                p: 0.11,
            }),
        ),
        ("pauli_z", Box::new(Clifford1Q::z(3))),
    ];
    for (name, ch) in &chans {
        let cr: &dyn Channel<1> = ch.as_ref();
        for bits in [0u8, 4, 8] {
            let got = bucketed_layer(&input, cr, &AlwaysKeep, false, bits, 0x31);
            let want = naive_apply_layer(&input, cr, &AlwaysKeep, false);
            assert_terms_close(&got, &want, TOL, &format!("{name} bits={bits}"));
        }
    }
}

#[test]
fn rescale_fast_path_still_applies_truncation() {
    let input = rand_sum::<1>(1500, 8, 0x5A5B);
    let depol = Depolarizing {
        support: [3],
        p: 0.11,
    };
    for bits in [0u8, 5] {
        let policy = And(CoefficientThreshold(0.3), WeightCutoff(4));
        let got = bucketed_layer(&input, &depol, &policy, false, bits, 0x41);
        let want = naive_apply_layer(&input, &depol, &policy, false);
        assert_terms_close(&got, &want, TOL, &format!("truncated rescale bits={bits}"));
        assert!(got.len() < input.len(), "truncation dropped nothing");
    }
}

// ---- determinism ----

/// sqrt(SWAP): a wide delta set merging three or more contributions per key, the only regime where summation order is observable.
fn sqrt_swap_w1(a: u32, b: u32) -> crate::channel::GeneralUnitary2Q {
    let h = Complex64::new(0.5, 0.5);
    let hc = Complex64::new(0.5, -0.5);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    crate::channel::GeneralUnitary2Q::from_matrix(
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
    // A different bucket count can reorder a key's summands, so agreement is to tolerance; GeneralUnitary2Q is the channel that exercises it.
    let input = rand_sum::<1>(2000, 8, 0x9001);
    let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
    let cnot = Clifford2Q::cnot(1, 5);
    let gu2q = sqrt_swap_w1(1, 5);
    for ch in [
        &rot as &dyn Channel<1>,
        &cnot as &dyn Channel<1>,
        &gu2q as &dyn Channel<1>,
    ] {
        let reference = bucketed_layer(&input, ch, &AlwaysKeep, false, 0, 0x51);
        for bits in [1u8, 2, 3, 5, 8, 11] {
            let got = bucketed_layer(&input, ch, &AlwaysKeep, false, bits, 0x51);
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
    for ch in [&rot as &dyn Channel<1>, &gu2q as &dyn Channel<1>] {
        let reference = bucketed_layer(&input, ch, &AlwaysKeep, false, 6, 1);
        for seed in [2u64, 3, 5, 8, 13, 21] {
            let got = bucketed_layer(&input, ch, &AlwaysKeep, false, 6, seed);
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
    let prep = gu2q.prepare(sum.hash(), false).unwrap();
    let Prepared::Local(ptm) = &prep else {
        panic!("gu2q prepares to a Local plan");
    };
    let span = Gf2Span::new(&prep.bucket_deltas(), sum.hash().bits());
    assert!(
        span.r() >= 2,
        "want a multi-member coset so the two visit orders actually differ; got r={}",
        span.r()
    );
    let coords: Vec<u32> = ptm
        .deltas()
        .iter()
        .map(|d| span.coord_of(d.bucket_delta))
        .collect();
    let m = span.coset_size();

    // Assemble the rank-0 coset's member columns, ascending by coordinate.
    let mut old: Vec<BucketCols<1>> = (0..m).map(|_| BucketCols::default()).collect();
    for beta in 0..sum.num_buckets() as u32 {
        let p = span.perm_index(beta) as usize;
        if p < m {
            let (bx, bz, bc) = sum.bucket(beta as usize);
            old[p] = BucketCols {
                x: bx.to_vec(),
                z: bz.to_vec(),
                coeff: bc.to_vec(),
            };
        }
    }

    let has_identity = ptm.deltas().first().is_some_and(|d| d.local_delta == 0);
    // gu2q's identity is dense: only id coefficients are gathered and the merge borrows the source keys.
    let dim = 1usize << (2 * ptm.k());
    let dense_identity = has_identity && ptm.deltas()[0].amp[..dim].iter().all(|a| *a != ZERO);
    assert!(dense_identity, "gu2q's identity amplitude must be dense");
    let gather = |output_major: bool| {
        let mut runs: Vec<GatherRun<1>> = (0..m).map(|_| GatherRun::default()).collect();
        for (j, run) in runs.iter_mut().enumerate() {
            let mut cap_id = 0usize;
            let mut cap_rest = 0usize;
            for (e, &c) in coords.iter().enumerate() {
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
            gather_local_output_major(&old, &mut runs, ptm, &coords, has_identity, true);
        } else {
            gather_local_input_major(&old, &mut runs, ptm, &coords, has_identity, true);
        }
        let mut scratch = SortScratch::<1>::default();
        for run in runs.iter_mut() {
            sort_rows_with_scratch(&mut run.x, &mut run.z, &mut run.coeff, &mut scratch);
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
            assert_eq!(src.len(), run.id_coeff.len(), "dense id must be 1:1");
            let mut mx = Vec::new();
            let mut mz = Vec::new();
            let mut mc = Vec::new();
            merge2_into::<1, AlwaysKeep>(
                &src.x,
                &src.z,
                &run.id_coeff,
                &run.x,
                &run.z,
                &run.coeff,
                &mut mx,
                &mut mz,
                &mut mc,
                &AlwaysKeep,
            );
            (mx, mz, mc)
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
    let check = |ch: &dyn Channel<1>, want: bool, label: &str| {
        let prep = ch.prepare(&hash, false).unwrap();
        let span = Gf2Span::new(&prep.bucket_deltas(), 6);
        match DeltaPlan::new(&prep, &span, LayerKnobs::default()) {
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
        &AmplitudeDamping {
            support: [5],
            gamma: 0.3,
        },
        true,
        "amplitude_damping",
    );
    // Sparse: the id amplitude vanishes on some patterns (CNOT: 12 of 16; H: 2 of 4).
    check(&Clifford2Q::cnot(1, 4), false, "cnot");
    check(&Clifford1Q::h(3), false, "h");
}

/// Pins per built-in which sort kernel its layer gets and the two quantities that decide it.
#[test]
fn radix_sort_kernel_is_selected_only_for_dense_ptms() {
    let hash = Gf2Hash::<1>::new(12, 8, 0xD1CE);
    let rest_streams = |ch: &dyn Channel<1>, label: &str| -> (usize, bool, f64) {
        let prep = ch.prepare(&hash, false).unwrap();
        let span = Gf2Span::new(&prep.bucket_deltas(), 8);
        match DeltaPlan::new(&prep, &span, LayerKnobs::default()) {
            DeltaPlan::Local {
                ptm,
                has_identity,
                radix_sort,
                ..
            } => (
                ptm.deltas().len() - has_identity as usize,
                radix_sort,
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
        ("cnot", Box::new(Clifford2Q::cnot(1, 5))),
        ("cz", Box::new(Clifford2Q::cz(1, 5))),
        ("swap", Box::new(Clifford2Q::swap(1, 5))),
        ("h", Box::new(Clifford1Q::h(3))),
        ("s", Box::new(Clifford1Q::s(3))),
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
        (
            "depolarizing",
            Box::new(Depolarizing {
                support: [3],
                p: 0.05,
            }),
        ),
    ];
    assert_eq!(cases.len(), expect.len());
    for ((label, ch), &(want_label, want_streams, want_rpk, want_radix)) in cases.iter().zip(expect)
    {
        assert_eq!(*label, want_label);
        let (streams, radix, rpk) = rest_streams(ch.as_ref(), label);
        assert_eq!(streams, want_streams, "{label}: rest streams");
        assert!(
            (rpk - want_rpk).abs() < 1e-9,
            "{label}: rest_rows_per_key {rpk}, want {want_rpk}",
        );
        assert_eq!(
            radix, want_radix,
            "{label}: {streams} rest streams, {rpk} rows per key, radix_sort = {radix}",
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
    let prep = gu2q.prepare(&hash, false).unwrap();
    let Prepared::Local(full) = &prep else {
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
        DeltaPlan::Local { radix_sort, .. } => radix_sort,
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
        Box::new(Clifford1Q::h(0)),
        Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.3)),
        Box::new(Clifford2Q::cnot(1, 5)),
        Box::new(Depolarizing {
            support: [3],
            p: 0.05,
        }),
        Box::new(Clifford1Q::s(6)),
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
    for ch in &chans {
        want = naive_apply_layer(&want, ch.as_ref(), &AlwaysKeep, false);
    }

    let hash = Gf2Hash::<1>::new(8, 5, 0x77);
    let mut b = input.clone().with_hash(hash);
    let mut scratch = LayerScratch::<1>::new();
    for ch in &chans {
        let prep = ch.prepare(b.hash(), false).unwrap();
        apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);
    }
    let got = b;
    assert_terms_close(&got, &want, TOL, "six layers");
}

#[test]
fn layers_survive_a_rebucket_in_between() {
    let input = rand_sum::<1>(800, 8, 0x7002);
    let h = Clifford1Q::h(0);
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

    let prep = h.prepare(b.hash(), false).unwrap();
    apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);
    b.rebucket(32, 1);
    let prep = rot.prepare(b.hash(), false).unwrap();
    apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);

    assert_terms_close(&b, &want, TOL, "layer, rebucket, layer");
}

// ---- the fingerprint net ----

/// FNV-1a over the eight little-endian bytes of one `u64`, inline so the pinned fingerprints keep their mix.
fn fnv_fold(h: u64, v: u64) -> u64 {
    let mut h = h;
    for b in v.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A digest of the sum's exact bits in canonical key order, blind to the partition.
fn layer_fingerprint<const W: usize>(s: &PauliSum<W>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    h = fnv_fold(h, s.len() as u64);
    for (x, z, c) in canonical_triples(s) {
        for &w in x.iter().chain(z.iter()) {
            h = fnv_fold(h, w);
        }
        h = fnv_fold(h, c.re.to_bits());
        h = fnv_fold(h, c.im.to_bits());
    }
    h
}

/// The channels in the fingerprint net: one per prepared-path shape (Cliffords, a dense two-qubit unitary, weight-2 and weight-4 rotations, the key-preserving rescale path, and amplitude damping).
fn fingerprint_channels() -> Vec<(&'static str, Box<dyn Channel<2>>)> {
    vec![
        ("clifford1q_h", Box::new(Clifford1Q::h(3))),
        ("clifford2q_cnot", Box::new(Clifford2Q::cnot(1, 5))),
        ("clifford2q_swap", Box::new(Clifford2Q::swap(1, 5))),
        (
            // sqrt(SWAP): non-Clifford with a wide delta set.
            "general_unitary2q",
            Box::new({
                let h = Complex64::new(0.5, 0.5);
                let hc = Complex64::new(0.5, -0.5);
                let one = Complex64::new(1.0, 0.0);
                let zero = Complex64::new(0.0, 0.0);
                crate::channel::GeneralUnitary2Q::from_matrix(
                    1,
                    5,
                    [
                        [one, zero, zero, zero],
                        [zero, h, hc, zero],
                        [zero, hc, h, zero],
                        [zero, zero, zero, one],
                    ],
                )
            }),
        ),
        (
            "rotation_zz",
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<2>::z(1);
                    g.mul_assign(&PauliString::<2>::z(6));
                    g
                },
                0.41,
            )),
        ),
        (
            // Weight 4 > MAX_LOCAL_SUPPORT, so this takes `gather_rotation`.
            "rotation_w4",
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<2>::z(0);
                    for q in [2u32, 4, 7] {
                        g.mul_assign(&PauliString::<2>::x(q));
                    }
                    g
                },
                0.41,
            )),
        ),
        (
            "depolarizing",
            Box::new(Depolarizing {
                support: [2],
                p: 0.07,
            }),
        ),
        (
            "amp_damping",
            Box::new(AmplitudeDamping {
                support: [2],
                gamma: 0.3,
            }),
        ),
    ]
}

/// Every `(channel, direction, bits)` fingerprint, in `fingerprint_channels` order.
const LAYER_FINGERPRINTS: &[(&str, bool, u8, u64)] = &[
    ("clifford1q_h", false, 2, 0x8a01_7283_1dac_9905),
    ("clifford1q_h", false, 5, 0x8a01_7283_1dac_9905),
    ("clifford1q_h", true, 2, 0x8a01_7283_1dac_9905),
    ("clifford1q_h", true, 5, 0x8a01_7283_1dac_9905),
    ("clifford2q_cnot", false, 2, 0x8d22_5efb_4856_044f),
    ("clifford2q_cnot", false, 5, 0x8d22_5efb_4856_044f),
    ("clifford2q_cnot", true, 2, 0x8d22_5efb_4856_044f),
    ("clifford2q_cnot", true, 5, 0x8d22_5efb_4856_044f),
    ("clifford2q_swap", false, 2, 0x5fe9_a80d_62af_1da9),
    ("clifford2q_swap", false, 5, 0x5fe9_a80d_62af_1da9),
    ("clifford2q_swap", true, 2, 0x5fe9_a80d_62af_1da9),
    ("clifford2q_swap", true, 5, 0x5fe9_a80d_62af_1da9),
    ("general_unitary2q", false, 2, 0x6a89_211e_1337_0d4b),
    ("general_unitary2q", false, 5, 0x6a89_211e_1337_0d4b),
    ("general_unitary2q", true, 2, 0x54b3_481c_3682_b7db),
    ("general_unitary2q", true, 5, 0x54b3_481c_3682_b7db),
    ("rotation_zz", false, 2, 0x79b5_287d_69fe_3049),
    ("rotation_zz", false, 5, 0x79b5_287d_69fe_3049),
    ("rotation_zz", true, 2, 0x0888_9337_8137_9549),
    ("rotation_zz", true, 5, 0x0888_9337_8137_9549),
    ("rotation_w4", false, 2, 0xd22c_2678_5d1a_6ec7),
    ("rotation_w4", false, 5, 0xd22c_2678_5d1a_6ec7),
    ("rotation_w4", true, 2, 0xda87_ea29_d292_f0c7),
    ("rotation_w4", true, 5, 0xda87_ea29_d292_f0c7),
    ("depolarizing", false, 2, 0x0c2d_0f88_a7cb_3051),
    ("depolarizing", false, 5, 0x0c2d_0f88_a7cb_3051),
    ("depolarizing", true, 2, 0x0c2d_0f88_a7cb_3051),
    ("depolarizing", true, 5, 0x0c2d_0f88_a7cb_3051),
    ("amp_damping", false, 2, 0x8b0f_59fb_c452_c0bf),
    ("amp_damping", false, 5, 0x8b0f_59fb_c452_c0bf),
    ("amp_damping", true, 2, 0xd3cf_d844_cd3d_2be8),
    ("amp_damping", true, 5, 0xd3cf_d844_cd3d_2be8),
];

/// Exact-bit tripwire for one layer across every prepared-path shape; regenerate the literals when a change is correct to tolerance.
#[test]
fn layer_fingerprints_are_stable() {
    let input = rand_sum::<2>(2000, 10, 0xC05E7);
    let channels = fingerprint_channels();
    let mut got: Vec<(&str, bool, u8, u64)> = Vec::new();
    for (name, ch) in &channels {
        let cr: &dyn Channel<2> = ch.as_ref();
        for &adjoint in &[false, true] {
            for &bits in &[2u8, 5] {
                let out = bucketed_layer(&input, cr, &AlwaysKeep, adjoint, bits, 0xF17E);
                got.push((name, adjoint, bits, layer_fingerprint(&out)));
            }
        }
    }

    // Printed for re-pinning; run with `--nocapture`.
    for &(name, adjoint, bits, fp) in &got {
        println!("(\"{name}\", {adjoint}, {bits}, {fp:#018x}),");
    }

    assert_eq!(
        got.len(),
        LAYER_FINGERPRINTS.len(),
        "the net and the pinned table cover different cases"
    );
    for (g, w) in got.iter().zip(LAYER_FINGERPRINTS.iter()) {
        assert_eq!(
            (g.0, g.1, g.2),
            (w.0, w.1, w.2),
            "the net and the pinned table are out of order"
        );
        assert_eq!(
            g.3, w.3,
            "fingerprint changed for {} adjoint={} bits={}: {:#018x} != {:#018x}",
            g.0, g.1, g.2, g.3, w.3,
        );
    }
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
    for ch in [
        Box::new(PauliRotation::new(
            {
                let mut g = PauliString::<1>::z(1);
                g.mul_assign(&PauliString::<1>::z(5));
                g
            },
            0.37,
        )) as Box<dyn Channel<1>>,
        Box::new(Clifford2Q::cnot(2, 6)),
    ] {
        let got = bucketed_layer(&input, ch.as_ref(), &AlwaysKeep, false, 0, 0xEE);
        assert_eq!(got.num_buckets(), 1);
        let want = naive_apply_layer(&input, ch.as_ref(), &AlwaysKeep, false);
        // Tolerance: the oracle sums equal keys in hash-map order.
        assert_terms_close(&got, &want, TOL, "bits=0 single coset");
    }
}

/// A wide rotation whose generator hashes to bucket delta 0 (`r = 0`); at most two summands per key, so this stays bitwise.
#[test]
fn wide_rotation_with_colliding_bucket_delta() {
    // Weight 4 > MAX_LOCAL_SUPPORT, so it prepares as `Prepared::Rotation`.
    let mut gen = PauliString::<1>::z(0);
    for q in [2u32, 4, 6] {
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
    let prep = rot.prepare(b.hash(), false).unwrap();
    match &prep {
        Prepared::Rotation(r) => {
            assert_eq!(
                r.bucket_delta_gen, r.bucket_delta_identity,
                "seed search failed to produce the collision"
            );
        }
        _ => panic!("weight-4 rotation must prepare as Rotation"),
    }
    let mut scratch = LayerScratch::<1>::new();
    apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);

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
    let cnot = Clifford2Q::cnot(3, 8);
    let h = Complex64::new(0.5, 0.5);
    let hc = Complex64::new(0.5, -0.5);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let gu2q = crate::channel::GeneralUnitary2Q::from_matrix(
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
    for ch in channels {
        let prep = ch.prepare(b.hash(), false).unwrap();
        apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);
        want = naive_apply_layer(&want, ch, &AlwaysKeep, false);
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
    let hgate = Clifford1Q::h(3);
    let prep = hgate.prepare(b.hash(), false).unwrap();
    let mut scratch = LayerScratch::<1>::new();

    let total_capacity = |s: &PauliSum<1>, sc: &LayerScratch<1>| -> usize {
        let bucket_cap: usize = (0..s.num_buckets())
            .map(|i| {
                let (x, _, _) = s.bucket(i);
                // Bucket capacity is not observable through the slice view, so count lengths there and real capacities in the scratch.
                x.len()
            })
            .sum();
        let old_cap: usize = sc.task.old.iter().map(|c| c.x.capacity()).sum();
        let run_cap: usize = sc.task.runs.iter().map(|r| r.x.capacity()).sum();
        let sort_cap = sc.task.sort.total_capacity();
        bucket_cap + old_cap + run_cap + sort_cap + sc.perm.capacity() + sc.staging.capacity()
    };

    let mut snapshots = Vec::new();
    for _ in 0..4 {
        apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);
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
        fn max_fanout(&self) -> usize {
            3
        }
        fn support(&self) -> [u64; W] {
            crate::channel::support_mask(&[0, 1])
        }
        fn apply(
            &self,
            input_x: &[u64; W],
            input_z: &[u64; W],
            coeff: Complex64,
            out: &mut crate::channel::OutputBuffer<'_, W>,
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

    let ch = ThreeDeltas;
    let input = rand_sum::<1>(1200, 8, 0xAB5EA7);
    let want = naive_apply_layer(&input, &ch, &AlwaysKeep, false);
    for bits in [0u8, 2, 5] {
        let got = bucketed_layer(&input, &ch, &AlwaysKeep, false, bits, 0x7EA);
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
    use crate::channel::clifford::{Clifford1Q, Clifford2Q};
    use crate::channel::rotation::PauliRotation;
    use crate::channel::Channel;
    use crate::pauli_string::PauliString;
    use crate::pauli_sum::accumulator::BuildAccumulator;
    use crate::pauli_sum::hash::Gf2Hash;
    use crate::pauli_sum::PauliSum;
    use crate::phase::Phase;
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

        fn append_into(
            &self,
            beta: u32,
            x: &mut Vec<[u64; W]>,
            z: &mut Vec<[u64; W]>,
            c: &mut Vec<Complex64>,
        ) {
            let Some(rows) = self.rows.get(&beta) else {
                return;
            };
            for &(rx, rz, rc) in rows {
                x.push(rx);
                z.push(rz);
                c.push(rc);
            }
        }
    }

    /// One layer through [`apply_layer_bucketed_with`], at a fixed partition.
    fn layer_with_extra<const W: usize, T, X>(
        input: &PauliSum<W>,
        ch: &dyn Channel<W>,
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
        let prep = ch
            .prepare(b.hash(), false)
            .expect("channel could not be prepared");
        let mut scratch = LayerScratch::<W>::new();
        apply_layer_bucketed_with(
            &mut b,
            &prep,
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
        ch: &dyn Channel<W>,
        policy: &T,
        injected: &Injected<W>,
    ) -> PauliSum<W>
    where
        T: TruncationPolicy<W> + ?Sized,
    {
        let base = naive_apply_layer(input, ch, &AlwaysKeep, false);
        let mut map: HashMap<([u64; W], [u64; W]), Complex64> = HashMap::new();
        for (x, z, c) in base.iter() {
            *map.entry((*x, *z)).or_insert(ZERO) += c;
        }
        for &(x, z, c) in injected.all() {
            *map.entry((x, z)).or_insert(ZERO) += c;
        }
        let mut acc = BuildAccumulator::<W>::with_capacity(input.num_qubits(), map.len());
        for ((x, z), c) in map {
            if c == ZERO || !policy.keep_term(&x, &z, c) {
                continue;
            }
            acc.add_term(PauliString::<W> { x, z }, Phase::ONE, c);
        }
        acc.finalize()
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
        let cnot = Clifford2Q::cnot(1, 5);
        // At `bits = 5` both channels permute non-trivially and run the parallel chunk loop (16 and 8 cosets).
        let cases: [(&str, &dyn Channel<1>, u8, usize); 4] = [
            ("rot bits=5", &rot, 5, 1),
            ("cnot bits=5", &cnot, 5, 2),
            ("rot bits=0", &rot, 0, 0),
            ("cnot bits=0", &cnot, 0, 0),
        ];
        for (label, ch, bits, want_r) in cases {
            // Pick the hash for the intended span rank, so a colliding draw cannot degrade a `bits = 5` case.
            let seed = (0u64..4096)
                .find(|&s| {
                    let hash = Gf2Hash::<1>::new(8, bits, s);
                    let prep = ch.prepare(&hash, false).unwrap();
                    Gf2Span::new(&prep.bucket_deltas(), bits).r() == want_r
                })
                .unwrap_or_else(|| panic!("{label}: no seed of rank {want_r} in 4096 tries"));
            let hash = Gf2Hash::<1>::new(8, bits, seed);

            let injected = injection_for(&input, &hash);
            assert!(
                injected.all().count() >= 6,
                "{label}: too few injected rows"
            );

            // Non-vacuity: both the summed and the inserted case occur.
            let plain = bucketed_layer(&input, ch, &AlwaysKeep, false, bits, seed);
            let collided = injected
                .all()
                .filter(|(x, z, _)| plain.get(x, z).is_some())
                .count();
            let fresh = injected.all().count() - collided;
            assert!(
                collided > 0 && fresh > 0,
                "{label}: want both colliding and fresh keys (collided={collided} fresh={fresh})"
            );

            let got = layer_with_extra(&input, ch, &AlwaysKeep, bits, seed, &injected);
            let want = expected(&input, ch, &AlwaysKeep, &injected);
            assert_eq!(got.len(), plain.len() + fresh, "{label}: term count");
            assert_terms_close(&got, &want, TOL, label);
        }
    }

    /// An empty `Injected` (`NEEDS_BETA = true`) gives exactly what `NoExtra` gives.
    #[test]
    fn an_empty_injection_changes_nothing() {
        let input = rand_sum::<1>(600, 8, 0xE47B);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
        let cnot = Clifford2Q::cnot(1, 5);
        for ch in [&rot as &dyn Channel<1>, &cnot as &dyn Channel<1>] {
            for bits in [0u8, 2, 5] {
                let empty = Injected::<1>::default();
                let got = layer_with_extra(&input, ch, &AlwaysKeep, bits, 0x7A58, &empty);
                let plain = bucketed_layer(&input, ch, &AlwaysKeep, false, bits, 0x7A58);
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
        let mut acc = BuildAccumulator::<1>::with_capacity(8, 2);
        acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(0.5, 0.0));
        acc.add_term(PauliString::<1>::z(1), Phase::ONE, Complex64::new(1.0, 0.0));
        let input = acc.finalize();
        // H on qubit 0 sends Z₀ → X₀ with amplitude 1 and leaves Z₁ alone.
        let h = Clifford1Q::h(0);
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
    use crate::channel::clifford::Clifford1Q;
    use crate::channel::rotation::PauliRotation;
    use crate::channel::Channel;
    use crate::pauli_string::PauliString;
    use crate::pauli_sum::hash::Gf2Hash;
    use crate::pauli_sum::PauliSum;
    use crate::truncation::builtin::{And, CoefficientThreshold, Or, TopN, WeightCutoff};

    /// `TopN` keeps exactly `n` terms, the same set as the flat implementation absent magnitude ties.
    #[test]
    fn top_n_bucketed_matches_the_flat_implementation() {
        let input = rand_sum::<1>(2000, 8, 0x1234);
        for n in [1usize, 7, 100, 999, 1999, 5000] {
            let policy = TopN(n);
            let mut flat = input.clone();
            policy.finalize_layer(&mut flat);

            for bits in [0u8, 3, 6, 10] {
                let hash = Gf2Hash::<1>::new(8, bits, 0x99);
                let mut b = input.clone().with_hash(hash);
                policy.finalize_layer(&mut b);
                b.assert_invariants();
                let got = b;
                assert_same_terms(&got, &flat, &format!("n={n} bits={bits}"));
            }
        }
    }

    #[test]
    fn top_n_bucketed_keeps_exactly_n_and_the_largest() {
        let input = rand_sum::<1>(1000, 8, 0x4321);
        let hash = Gf2Hash::<1>::new(8, 5, 0x99);
        let mut b = input.clone().with_hash(hash);
        TopN(50).finalize_layer(&mut b);
        assert_eq!(b.len(), 50);
        let got = b;

        // Every retained magnitude must be >= every dropped one.
        let mut all: Vec<f64> = input.iter().map(|(_, _, c)| c.norm()).collect();
        all.sort_by(|a, c| c.partial_cmp(a).unwrap());
        let cutoff = all[49];
        for (_, _, c) in got.iter() {
            assert!(c.norm() >= cutoff - 1e-15, "kept a below-cutoff term");
        }
    }

    #[test]
    fn top_n_zero_clears_and_preserves_the_invariant() {
        let input = rand_sum::<1>(500, 8, 0x5555);
        let hash = Gf2Hash::<1>::new(8, 4, 0x99);
        let mut b = input.clone().with_hash(hash);
        TopN(0).finalize_layer(&mut b);
        b.assert_invariants();
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
    }

    #[test]
    fn top_n_above_the_length_is_a_no_op() {
        // `rand_sum` dedups, so compare against the realized length.
        let input = rand_sum::<1>(300, 8, 0x6666);
        let hash = Gf2Hash::<1>::new(8, 4, 0x99);
        let mut b = input.clone().with_hash(hash);
        TopN(10_000).finalize_layer(&mut b);
        assert_eq!(b.len(), input.len());
        let got = b;
        assert_same_terms(&got, &input, "top_n above length");
    }

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

    #[test]
    fn threshold_and_weight_and_or_finalizers_are_no_ops() {
        // These have no layer pass, so `finalize_layer` must leave the sum untouched.
        let input = rand_sum::<1>(500, 8, 0x8888);
        let hash = Gf2Hash::<1>::new(8, 4, 0x99);
        for tag in 0..3 {
            let mut b = input.clone().with_hash(hash.clone());
            match tag {
                0 => CoefficientThreshold(0.5).finalize_layer(&mut b),
                1 => WeightCutoff(2).finalize_layer(&mut b),
                _ => Or(CoefficientThreshold(0.5), WeightCutoff(2)).finalize_layer(&mut b),
            }
            assert_eq!(b.len(), input.len(), "tag {tag} changed the sum");
        }
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
            Box::new(Clifford1Q::h(0)),
            Box::new(PauliRotation::new(PauliString::<1>::x(5), 0.21)),
        ];

        let mut want = input.clone();
        for ch in &chans {
            want = naive_apply_layer(&want, ch.as_ref(), &policy, false);
            policy.finalize_layer(&mut want);
        }

        let hash = Gf2Hash::<1>::new(8, 5, 0xBB);
        let mut b = input.clone().with_hash(hash);
        let mut scratch = LayerScratch::<1>::new();
        for ch in &chans {
            let prep = ch.prepare(b.hash(), false).unwrap();
            apply_layer_bucketed(&mut b, &prep, &policy, &mut scratch);
            policy.finalize_layer(&mut b);
        }
        let got = b;

        assert_terms_close(&got, &want, 1e-11, "3 truncated layers");
    }
}

mod tie_tests {
    use crate::pauli_string::PauliString;
    /// Byte-identical output across thread counts at a fixed bucket count.
    #[test]
    fn parallel_output_is_byte_identical_across_thread_counts() {
        use crate::channel::rotation::PauliRotation;
        use crate::channel::Channel;

        let input = rand_sum::<1>(4000, 10, 0xC1C1);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.37);
        let cnot = crate::channel::clifford::Clifford2Q::cnot(1, 5);

        for ch in [&rot as &dyn Channel<1>, &cnot as &dyn Channel<1>] {
            // 64 buckets, well above MIN_COSETS_FOR_PARALLEL.
            let run = |threads: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("pool")
                    .install(|| {
                        let hash = Gf2Hash::<1>::new(10, 6, 0xC1);
                        let mut b = input.clone().with_hash(hash);
                        let prep = ch.prepare(b.hash(), false).unwrap();
                        let mut scratch = LayerScratch::<1>::new();
                        apply_layer_bucketed(
                            &mut b,
                            &prep,
                            &super::tests::AlwaysKeep,
                            &mut scratch,
                        );
                        b
                    })
            };
            let reference = run(1);
            for threads in [2usize, 4, 8, 16, 32] {
                let got = run(threads);
                assert_eq!(got.len(), reference.len(), "threads={threads}");
                // Same fixed hash on both sides, so column equality is the bitwise statement.
                assert_eq!(
                    got.to_arrays(),
                    reference.to_arrays(),
                    "threads={threads}: output is not byte-identical",
                );
            }
        }
    }

    /// The in-place rescale path is parallel too, and must give the same answer.
    #[test]
    fn parallel_rescale_is_byte_identical_across_thread_counts() {
        use crate::channel::noise::Depolarizing;
        use crate::channel::Channel;

        let input = rand_sum::<1>(4000, 10, 0xC1C2);
        let depol = Depolarizing {
            support: [3],
            p: 0.11,
        };
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| {
                    let hash = Gf2Hash::<1>::new(10, 6, 0xC2);
                    let mut b = input.clone().with_hash(hash);
                    let prep = Channel::<1>::prepare(&depol, b.hash(), false).unwrap();
                    let mut scratch = LayerScratch::<1>::new();
                    apply_layer_bucketed(&mut b, &prep, &super::tests::AlwaysKeep, &mut scratch);
                    b
                })
        };
        let reference = run(1);
        for threads in [2usize, 8, 32] {
            let got = run(threads);
            // Identical fixed hash on both sides: canonical order is shared.
            assert_eq!(
                got.to_arrays().2,
                reference.to_arrays().2,
                "threads={threads}"
            );
        }
    }

    use super::*;
    use super::{assert_same_terms, canonical_triples};
    use crate::pauli_sum::hash::Gf2Hash;
    use crate::pauli_sum::PauliSum;
    use crate::test_support::{rand_sum, tie_heavy_sum};
    use crate::truncation::builtin::TopN;

    /// A naive transcription of the `TopN` tie-group rule (ARCHITECTURE.md §Truncation), sharing no code with production.
    fn top_n_reference<const W: usize>(
        sum: &PauliSum<W>,
        n: usize,
    ) -> Vec<([u64; W], [u64; W], Complex64)> {
        let mut triples = canonical_triples(sum);
        if triples.len() <= n {
            return triples;
        }
        if n == 0 {
            return Vec::new();
        }
        let mut mags: Vec<f64> = triples.iter().map(|t| t.2.norm()).collect();
        mags.sort_by(|a, b| b.partial_cmp(a).expect("no NaN magnitudes"));
        // `t` = the n-th largest magnitude.
        let t = mags[n - 1];
        let count_gt = mags.iter().filter(|&&m| m > t).count();
        let count_eq = mags.iter().filter(|&&m| m == t).count();
        // Keep the tie group iff it fits entirely.
        let keep_tied = count_gt + count_eq <= n;
        triples.retain(|(_, _, c)| {
            let m = c.norm();
            m > t || (keep_tied && m == t)
        });
        triples
    }

    /// The retained set depends only on the magnitude multiset, never on the bucket partition; checked on tie-dense data.
    #[test]
    fn top_n_is_bucket_count_independent_on_tied_magnitudes() {
        let input = tie_heavy_sum::<1>(2000, 8, 0x7135);
        let n = 700; // cuts inside the group of magnitude-0.5 terms
        let reference = {
            let hash = Gf2Hash::<1>::new(8, 0, 0x99);
            let mut b = input.clone().with_hash(hash);
            TopN(n).finalize_layer(&mut b);
            b
        };
        for bits in [1u8, 2, 4, 6, 9] {
            let hash = Gf2Hash::<1>::new(8, bits, 0x99);
            let mut b = input.clone().with_hash(hash);
            TopN(n).finalize_layer(&mut b);
            let got = b;
            assert_same_terms(
                &got,
                &reference,
                &format!("bits={bits}: TopN kept a different set of tied terms"),
            );
        }
    }

    /// `finalize_layer` matches the `TopN` tie-group rule for `n` both straddling and exactly on group boundaries.
    #[test]
    fn top_n_matches_the_reference_rule_on_tied_magnitudes() {
        let input = tie_heavy_sum::<1>(2000, 8, 0x7136);
        let len = input.len();

        // Cumulative magnitude-group sizes, descending: each is a cut exactly on a group boundary.
        let mut mags: Vec<f64> = input.iter().map(|(_, _, c)| c.norm()).collect();
        mags.sort_by(|a, b| b.partial_cmp(a).expect("no NaN magnitudes"));
        let boundaries: Vec<usize> = (1..mags.len())
            .filter(|&i| mags[i] != mags[i - 1])
            .collect();
        assert!(
            boundaries.len() >= 2,
            "fixture must have several magnitude groups, got {}",
            boundaries.len() + 1
        );

        let mut sweep = vec![3usize, 250, 700, 1200, 1900];
        sweep.extend_from_slice(&boundaries);
        let mut saw_straddle = false;
        let mut saw_fit = false;

        for n in sweep {
            let want = top_n_reference(&input, n);
            // A straddling group is dropped whole, so fewer than `n` survive; a fitting group leaves exactly `n`.
            assert!(want.len() <= n, "n={n}: rule must retain at most n");
            if want.len() < n {
                saw_straddle = true;
            } else {
                saw_fit = true;
            }

            let policy = TopN(n);
            for bits in [0u8, 2, 5, 9] {
                let hash = Gf2Hash::<1>::new(8, bits, 0x99);
                let mut b = input.clone().with_hash(hash);
                policy.finalize_layer(&mut b);
                b.assert_invariants();
                assert_eq!(
                    canonical_triples(&b),
                    want,
                    "n={n} bits={bits} (len={len}): retained set differs from the \
                     reference rule",
                );
            }
        }

        assert!(
            saw_straddle,
            "the n sweep no longer covers a straddling tie group"
        );
        assert!(
            saw_fit,
            "the n sweep no longer covers a tie group that fits"
        );
    }

    /// The same across hash seeds.
    #[test]
    fn top_n_is_hash_seed_independent_on_tied_magnitudes() {
        let input = tie_heavy_sum::<1>(2000, 8, 0x7123);
        let n = 700;
        let reference = {
            let hash = Gf2Hash::<1>::new(8, 5, 1);
            let mut b = input.clone().with_hash(hash);
            TopN(n).finalize_layer(&mut b);
            b
        };
        for seed in [2u64, 3, 5, 8, 13] {
            let hash = Gf2Hash::<1>::new(8, 5, seed);
            let mut b = input.clone().with_hash(hash);
            TopN(n).finalize_layer(&mut b);
            let got = b;
            assert_same_terms(
                &got,
                &reference,
                &format!("seed={seed}: different set kept"),
            );
        }
    }

    /// `ApproxTopN`'s retained set is independent of bucket count and hash seed; `n = 700` cuts between the first two of `tie_heavy_sum`'s four equal octaves.
    #[test]
    fn approx_top_n_is_partition_independent_on_tied_magnitudes() {
        use crate::truncation::builtin::ApproxTopN;
        let input = tie_heavy_sum::<1>(2000, 8, 0x7135);
        let n = 700;
        let reference = {
            let mut b = input.clone().with_hash(Gf2Hash::<1>::new(8, 0, 0x99));
            ApproxTopN(n).finalize_layer(&mut b);
            b
        };
        assert_eq!(
            reference.len(),
            input.iter().filter(|(_, _, c)| c.norm() == 1.0).count(),
            "the fixture must cut between the top two octaves"
        );
        for (bits, seed) in [(1u8, 0x99u64), (2, 0x99), (4, 3), (6, 13), (9, 0x99)] {
            let mut b = input.clone().with_hash(Gf2Hash::<1>::new(8, bits, seed));
            ApproxTopN(n).finalize_layer(&mut b);
            b.assert_invariants();
            assert_same_terms(
                &b,
                &reference,
                &format!("bits={bits} seed={seed}: ApproxTopN kept a different set"),
            );
        }
    }

    /// A straddling tie group leaves no member behind in any bucket.
    #[test]
    fn top_n_drops_a_straddling_group_from_every_bucket() {
        let input = tie_heavy_sum::<1>(2000, 8, 0x71C0);
        let n = 700;
        // t is the 700th largest magnitude; the group at t straddles the cut.
        let mut mags: Vec<f64> = input.iter().map(|(_, _, c)| c.norm()).collect();
        mags.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let t = mags[n - 1];
        let count_gt = mags.iter().filter(|&&m| m > t).count();
        let count_eq = mags.iter().filter(|&&m| m == t).count();
        assert!(
            count_gt + count_eq > n,
            "fixture no longer straddles: gt={count_gt} eq={count_eq} n={n}"
        );

        for bits in [0u8, 3, 7] {
            let hash = Gf2Hash::<1>::new(8, bits, 0x99);
            let mut b = input.clone().with_hash(hash);
            TopN(n).finalize_layer(&mut b);
            b.assert_invariants();
            assert_eq!(
                b.len(),
                count_gt,
                "bits={bits}: retained count must be exactly count(|c| > t)"
            );
            for nb in 0..b.num_buckets() {
                let (_, _, coeff) = b.bucket(nb);
                assert!(
                    coeff.iter().all(|c| c.norm() > t),
                    "bits={bits} bucket={nb}: a member of the discarded group survived"
                );
            }
        }
    }
}
