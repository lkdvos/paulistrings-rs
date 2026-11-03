use std::f64::consts::FRAC_PI_2;

use super::super::{rotation_layers_taken, with_general_path};
use super::*;
use crate::channel::{Channel, PauliRotation, Prepare, PtmChannel};
use crate::engine::bucketed::{apply_layer_bucketed, LayerScratch};
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::PauliSum;
use crate::test_support::{
    assert_runs_agree, assert_terms_close, coarsen, propagate_discarding, rand_sum, zz_rotation,
    KeepAll, Xs64,
};
use crate::truncation::{ApproxTopN, BuiltinTruncation, CoefficientThreshold, TopN, WeightCutoff};
use crate::{Circuit, Direction};

const TOL: f64 = 1e-12;

/// The single-qubit X, Y, Z and the two-qubit ZZ, XY, XZ generators, on qubits `a` and `b`.
fn generators<const W: usize>(a: usize, b: usize) -> Vec<(String, PauliString<W>)> {
    type P<const W: usize> = PauliString<W>;
    vec![
        (format!("X{a}"), P::x(a)),
        (format!("Y{a}"), P::y(a)),
        (format!("Z{a}"), P::z(a)),
        (format!("Z{a}Z{b}"), P::z(a).product(&P::z(b)).0),
        (format!("X{a}Y{b}"), P::x(a).product(&P::y(b)).0),
        (format!("X{a}Z{b}"), P::x(a).product(&P::z(b)).0),
    ]
}

/// The first hash whose `h(g)` is zero or not, as asked.
fn hash_for<const W: usize>(
    num_qubits: usize,
    bits: u8,
    g: &PauliString<W>,
    zero: bool,
) -> Gf2Hash<W> {
    (1..10_000u64)
        .map(|seed| Gf2Hash::<W>::new(num_qubits, bits, seed))
        .find(|hash| (hash.bucket_of(&g.x, &g.z) == 0) == zero)
        .expect("a hash with the requested h(g)")
}

/// `n` random terms supported on `qubits` only, so partners hit existing keys often.
fn clustered_sum<const W: usize>(
    n: usize,
    num_qubits: usize,
    qubits: &[usize],
    seed: u64,
) -> PauliSum<W> {
    let mut rng = Xs64::new(seed);
    let mut accumulator = BuildAccumulator::<W>::with_capacity(num_qubits, n);
    for _ in 0..n {
        let mut p = PauliString::<W>::identity();
        for &q in qubits {
            let bits = rng.next_u64();
            if bits & 1 == 1 {
                p.x[q / 64] |= 1 << (q % 64);
            }
            if bits & 2 == 2 {
                p.z[q / 64] |= 1 << (q % 64);
            }
        }
        let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        let im = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        accumulator.add_term(p, Complex64::new(re, im));
    }
    accumulator.finalize()
}

#[derive(Clone, Copy, Debug)]
enum Spare {
    /// Every column exactly full, so a bucket gaining partners falls back.
    Tight,
    /// Room for every partner, so every bucket merges in place.
    Roomy,
}

fn set_spare<const W: usize>(sum: &mut PauliSum<W>, spare: Spare) {
    for bucket in sum.buckets_mut() {
        match spare {
            Spare::Tight => {
                bucket.x.shrink_to_fit();
                bucket.z.shrink_to_fit();
                bucket.coeff.shrink_to_fit();
            }
            Spare::Roomy => {
                let room = bucket.len() + 1;
                bucket.x.reserve(room);
                bucket.z.reserve(room);
                bucket.coeff.reserve(room);
            }
        }
    }
}

/// `(in place, fallback)` merges over every slot of `scratch`.
fn merges<const W: usize>(scratch: &LayerScratch<W>) -> (usize, usize) {
    let mut total = scratch.task.rotation.merges;
    for slot in &scratch.workers {
        let (in_place, fallback) = slot.lock().unwrap().rotation.merges;
        total.0 += in_place;
        total.1 += fallback;
    }
    total
}

/// One layer of `channel` and the policy's layer pass, on a copy of `input`.
fn one_layer<const W: usize>(
    input: &PauliSum<W>,
    channel: &dyn Channel<W>,
    policy: &BuiltinTruncation,
    adjoint: bool,
    spare: Spare,
) -> (PauliSum<W>, LayerScratch<W>) {
    let mut sum = input.clone();
    set_spare(&mut sum, spare);
    let prepared = channel.prepare(sum.hash(), adjoint).expect("prepares");
    let mut scratch = LayerScratch::new();
    apply_layer_bucketed(&mut sum, &prepared, policy, &mut scratch);
    <BuiltinTruncation as crate::TruncationPolicy<W>>::finalize_layer(policy, &mut sum);
    (sum, scratch)
}

/// The rotation layer and the general path on one layer; panics unless the rotation layer ran and the two agree.
fn assert_paths_agree<const W: usize>(
    input: &PauliSum<W>,
    channel: &dyn Channel<W>,
    policy: &BuiltinTruncation,
    adjoint: bool,
    spare: Spare,
    what: &str,
) -> (usize, usize) {
    let taken = rotation_layers_taken();
    let (got, scratch) = one_layer(input, channel, policy, adjoint, spare);
    assert_eq!(rotation_layers_taken(), taken + 1, "{what}: the rotation layer did not run");
    let (want, _) = with_general_path(|| one_layer(input, channel, policy, adjoint, spare));
    assert_eq!(
        rotation_layers_taken(),
        taken + 1,
        "{what}: the general path took the rotation layer"
    );
    got.assert_invariants();
    assert_terms_close(&got, &want, TOL, what);
    merges(&scratch)
}

fn policies(len: usize) -> Vec<BuiltinTruncation> {
    vec![
        KeepAll.into(),
        CoefficientThreshold(0.3).into(),
        WeightCutoff(3).into(),
        TopN(len * 2 / 3).into(),
        ApproxTopN(len / 2).into(),
    ]
}

/// Every generator, `h(g)` zero and not, both directions and spare capacities, random and Clifford angles, every policy, dense and sparse sums.
fn assert_generators_agree<const W: usize>(sums: &[(&str, PauliSum<W>)], a: usize, b: usize) {
    let mut rng = Xs64::new(0xA11CE ^ W as u64);
    for (generator_name, g) in generators::<W>(a, b) {
        let random = (rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * 6.0;
        for theta in [random, FRAC_PI_2] {
            let rotation = PauliRotation::new(g, theta);
            for zero in [true, false] {
                for (sum_name, sum) in sums {
                    let input = sum.clone().with_hash(hash_for(sum.num_qubits(), 5, &g, zero));
                    for policy in policies(input.len()) {
                        // The adjoint runs roomy, so dropping policies reach the in-place merge's shift.
                        for (adjoint, spare) in [(false, Spare::Tight), (true, Spare::Roomy)] {
                            let what = format!(
                                "{generator_name} θ={theta} h(g)=0:{zero} {sum_name} {policy:?} adjoint={adjoint} {spare:?}"
                            );
                            assert_paths_agree(&input, &rotation, &policy, adjoint, spare, &what);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn agrees_with_the_general_path_w1() {
    let sums = [
        ("dense", clustered_sum::<1>(1500, 12, &[2, 3, 5, 9, 11], 0xD1)),
        ("sparse", rand_sum::<1>(1500, 40, 0x5A)),
    ];
    assert_generators_agree(&sums, 3, 9);
}

#[test]
fn agrees_with_the_general_path_w2() {
    let sums = [
        ("dense", clustered_sum::<2>(1500, 100, &[1, 63, 64, 70, 99], 0xD2)),
        ("sparse", rand_sum::<2>(1500, 128, 0x5B)),
    ];
    // Across the word boundary.
    assert_generators_agree(&sums, 63, 64);
    assert_generators_agree(&sums, 1, 70);
}

/// One bucket, a mid count, more buckets than terms per bucket, and a coarsened hash.
#[test]
fn agrees_at_every_bucket_count() {
    let base = clustered_sum::<2>(2000, 80, &[0, 1, 40, 64, 79], 0xB0);
    let g = PauliString::<2>::z(1).product(&PauliString::<2>::z(64)).0;
    let rotation = PauliRotation::new(g, 0.7);
    for bits in [0u8, 3, 9] {
        let input = base.clone().with_hash(Gf2Hash::new(80, bits, 0x77));
        for policy in policies(input.len()) {
            assert_paths_agree(
                &input,
                &rotation,
                &policy,
                false,
                Spare::Tight,
                &format!("bits={bits} {policy:?}"),
            );
        }
    }
    let mut coarsened = base.with_hash(Gf2Hash::new(80, 6, 0x78));
    coarsen(&mut coarsened);
    assert_eq!(coarsened.hash().bits(), 5);
    assert_paths_agree(&coarsened, &rotation, &KeepAll.into(), false, Spare::Tight, "coarsened");
}

/// Tight columns fall back to the scratch merge, roomy ones merge in place, and both agree with the general path.
#[test]
fn merges_in_place_exactly_when_the_spare_capacity_allows() {
    let base = clustered_sum::<1>(3000, 16, &[0, 4, 7, 8, 15], 0x5E);
    for (name, g) in generators::<1>(4, 8) {
        for zero in [true, false] {
            let input = base.clone().with_hash(hash_for(16, 4, &g, zero));
            let rotation = PauliRotation::new(g, 0.4);
            let what = format!("{name} h(g)=0:{zero}");
            let (in_place, fallback) =
                assert_paths_agree(&input, &rotation, &KeepAll.into(), false, Spare::Tight, &what);
            assert!(fallback > 0, "{what}: tight columns must fall back ({in_place} in place)");
            let (in_place, fallback) =
                assert_paths_agree(&input, &rotation, &KeepAll.into(), false, Spare::Roomy, &what);
            assert_eq!(fallback, 0, "{what}: roomy columns must merge in place");
            assert_eq!(in_place, input.num_buckets(), "{what}: one merge per bucket");
        }
    }
}

/// A rotation built as a dense unitary keeps `cos(π/2)` as a residue of order `ε`, which the general path keeps and the Clifford arm drops.
#[test]
fn the_clifford_arm_drops_quarter_turn_residues() {
    let half = std::f64::consts::FRAC_PI_4;
    let (c, s) = (Complex64::new(half.cos(), 0.0), Complex64::new(0.0, -half.sin()));
    // exp(-i·(π/2)·X/2), entry by entry, with no snapping.
    let rx = [[c, s], [s, c]];
    let channel = PtmChannel::unitary_1q(3, rx);
    let input = clustered_sum::<1>(400, 8, &[1, 3, 5], 0xE5).with_hash(Gf2Hash::new(8, 2, 0x9));
    let prepared = Prepare::<1>::prepare(&channel, input.hash(), false).unwrap();
    let crate::channel::prepared::Prepared::Local(ptm) = &prepared else { unreachable!() };
    let layer = RotationLayer::of(ptm).expect("a quarter-turn unitary has the rotation shape");
    assert!(layer.clifford, "the residue is within ε of the partner amplitude");

    let (got, _) = one_layer(&input, &channel, &KeepAll.into(), false, Spare::Tight);
    let (want, _) =
        with_general_path(|| one_layer(&input, &channel, &KeepAll.into(), false, Spare::Tight));
    assert_eq!(got.len(), input.len(), "a Clifford layer is a permutation");
    let residues: Vec<_> = want.iter().filter(|t| t.2.norm() < 1e-14).collect();
    assert_eq!(
        want.len(),
        got.len() + residues.len(),
        "the general path keeps one residue per moved string"
    );
    let want_without_residues = {
        let mut trimmed = want.clone();
        trimmed.retain(|_, _, c| c.norm() >= 1e-14);
        trimmed
    };
    assert_terms_close(&got, &want_without_residues, TOL, "residues aside");
}

/// `exp(-iθG/2)` conjugation by hand, with `XZ = -iY` and `ZX = iY`.
#[test]
fn hand_computed_phases() {
    let (sin, cos) = 0.3f64.sin_cos();
    let sum = |terms: &[(&str, f64)]| {
        let terms: Vec<_> = terms.iter().map(|&(p, c)| (p, Complex64::new(c, 0.0))).collect();
        PauliSum::<1>::from_strings(&terms)
    };
    type P = PauliString<1>;
    let cases = [
        // X under Z: cos·X + sin·i·(XZ) = cos·X + sin·Y.
        ("X by Z", "XI", P::z(0), vec![("XI", cos), ("YI", sin)]),
        // Z under X: cos·Z + sin·i·(ZX) = cos·Z − sin·Y.
        ("Z by X", "ZI", P::x(0), vec![("ZI", cos), ("YI", -sin)]),
        // X₀ under Z₀Z₁: cos·X₀ + sin·i·(XZ)₀Z₁ = cos·X₀ + sin·Y₀Z₁.
        ("X0 by Z0Z1", "XI", P::z(0).product(&P::z(1)).0, vec![("XI", cos), ("YZ", sin)]),
        // Commuting: untouched.
        ("Z by Z", "ZI", P::z(0), vec![("ZI", 1.0)]),
    ];
    for (what, input, g, want) in cases {
        let rotation = PauliRotation::new(g, 0.3);
        for spare in [Spare::Tight, Spare::Roomy] {
            let (got, _) =
                one_layer(&sum(&[(input, 1.0)]), &rotation, &KeepAll.into(), false, spare);
            assert_terms_close(&got, &sum(&want), TOL, what);
        }
    }
    // A quarter turn about Z takes X to Y exactly, one term.
    let quarter = PauliRotation::new(P::z(0), FRAC_PI_2);
    let (got, _) = one_layer(&sum(&[("XI", 1.0)]), &quarter, &KeepAll.into(), false, Spare::Tight);
    assert_terms_close(&got, &sum(&[("YI", 1.0)]), f64::MIN_POSITIVE, "quarter turn");
}

/// Tables of another shape keep the general path.
#[test]
fn only_rotation_shaped_tables_qualify() {
    let hash = Gf2Hash::<1>::new(8, 3, 0x1);
    let shape = |channel: &dyn Channel<1>| {
        let prepared = channel.prepare(&hash, false).unwrap();
        let crate::channel::prepared::Prepared::Local(ptm) = &prepared else { unreachable!() };
        RotationLayer::of(ptm).map(|layer| layer.clifford)
    };
    assert_eq!(shape(&PauliRotation::new(PauliString::z(2), 0.4)), Some(false));
    assert_eq!(shape(&PauliRotation::new(PauliString::y(2), FRAC_PI_2)), Some(true));
    assert_eq!(shape(&PtmChannel::hadamard(2)), None);
    assert_eq!(shape(&PtmChannel::cnot(2, 3)), None);
    assert_eq!(shape(&PtmChannel::amplitude_damping(2, 0.3)), None);
    assert_eq!(shape(&PtmChannel::depolarizing(2, 0.1)), None);
}

/// Multi-layer runs through the driver, which rebuckets and coarsens between layers, and every built-in policy's layer pass.
#[test]
fn propagated_rotation_circuits_agree() {
    let mut circuit = Circuit::<1>::new(7);
    for step in 0..6 {
        for q in 0..7 {
            circuit.push(PauliRotation::new(PauliString::x(q), 0.1 + 0.01 * step as f64));
        }
        for q in 0..6 {
            circuit.push(zz_rotation(q, q + 1, 0.1));
        }
        circuit.push(PauliRotation::new(PauliString::y(step), FRAC_PI_2));
    }
    let input = {
        let mut accumulator = BuildAccumulator::<1>::new(7);
        accumulator.add_term(PauliString::z(3), Complex64::new(1.0, 0.0));
        accumulator.add_term(PauliString::x(5), Complex64::new(0.0, 0.5));
        accumulator.finalize()
    };
    let policies: [BuiltinTruncation; 5] = [
        KeepAll.into(),
        CoefficientThreshold(1e-3).into(),
        WeightCutoff(4).into(),
        TopN(600).into(),
        ApproxTopN(600).into(),
    ];
    for policy in policies {
        for direction in [Direction::Forward, Direction::Heisenberg] {
            let taken = rotation_layers_taken();
            let (got, _) = propagate_discarding(&circuit, input.clone(), &policy, direction);
            assert_eq!(rotation_layers_taken() - taken, circuit.channels.len());
            let (want, discarded) = with_general_path(|| {
                propagate_discarding(&circuit, input.clone(), &policy, direction)
            });
            if matches!(policy, BuiltinTruncation::TopN(_) | BuiltinTruncation::ApproxTopN(_)) {
                assert_eq!(want.len(), 600, "the layer pass must cut");
            }
            assert_runs_agree(
                policy.clone(),
                discarded,
                &got,
                &want,
                1e-10,
                &format!("{policy:?} {direction:?}"),
            );
        }
    }
}

/// Local rotation layers of a two-partition run take the rotation layer on each partition, a remote one the general path; the gathered result matches one partition's general path.
#[test]
fn partitioned_local_layers_agree_at_two_partitions() {
    use crate::engine::partitioned::{PartitionedSum, ScatterOptions, ScatterRows};
    use crate::test_support::{partition_rows, unpinned_scatter};

    let mut circuit = Circuit::<1>::new(8);
    for q in 0..7 {
        circuit.push(PauliRotation::new(PauliString::x(q), 0.35));
        circuit.push(zz_rotation(q, q + 1, 0.25));
    }
    // Remote under rows reading X₇ and Z₇.
    circuit.push(PauliRotation::new(PauliString::y(7), 0.5));
    let input = clustered_sum::<1>(600, 8, &(0..8).collect::<Vec<_>>(), 0x9A);
    let rows = partition_rows::<1>(8, vec![[1 << 7]], vec![[1 << 7]]);
    let policy = CoefficientThreshold(1e-4);
    for direction in [Direction::Forward, Direction::Heisenberg] {
        let options = ScatterOptions {
            rows: ScatterRows::Explicit(rows.clone()),
            ..unpinned_scatter(2, 2, 0)
        };
        let mut split = PartitionedSum::scatter_with(input.clone(), options);
        split.propagate(&circuit, &policy, direction);
        let got = split.into_gathered();
        let want =
            with_general_path(|| crate::propagate(&circuit, input.clone(), &policy, direction));
        assert_terms_close(&got, &want, 1e-10, &format!("{direction:?}"));
    }
}
