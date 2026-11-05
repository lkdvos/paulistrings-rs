use num_complex::Complex64;
use std::f64::consts::FRAC_PI_2;

use super::merge::{merge_backward_in_place, merge_sparse_in_place};
use super::*;
use crate::channel::{Channel, PauliRotation, Prepare, PtmChannel};
use crate::engine::bucketed::{apply_layer_bucketed, LayerScratch};
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::PauliSum;
use crate::test_support::{
    assert_terms_close, coarsen, naive_apply_layer, rand_sum, zz_rotation, KeepAll, Xs64,
};
use crate::truncation::{ApproxTopN, BuiltinTruncation, CoefficientThreshold, TopN, WeightCutoff};
use crate::PropagateOptions;
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
    let mut total = scratch.task.rotation_merges;
    for slot in &scratch.workers {
        let (in_place, fallback) = slot.lock().unwrap().rotation_merges;
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

/// The naive oracle's layer and the policy's layer pass.
fn oracle<const W: usize>(
    input: &PauliSum<W>,
    channel: &dyn Channel<W>,
    policy: &BuiltinTruncation,
    adjoint: bool,
) -> PauliSum<W> {
    let mut want = naive_apply_layer(input, channel, policy, adjoint);
    <BuiltinTruncation as crate::TruncationPolicy<W>>::finalize_layer(policy, &mut want);
    want
}

/// One layer against the oracle; panics unless the rotation layer ran and the two agree, and returns its merges.
fn assert_paths_agree<const W: usize>(
    input: &PauliSum<W>,
    channel: &dyn Channel<W>,
    policy: &BuiltinTruncation,
    adjoint: bool,
    spare: Spare,
    what: &str,
) -> (usize, usize) {
    let (got, scratch) = one_layer(input, channel, policy, adjoint, spare);
    let merges = merges(&scratch);
    assert!(merges.0 + merges.1 > 0, "{what}: the rotation layer did not run");
    got.assert_invariants();
    assert_terms_close(&got, &oracle(input, channel, policy, adjoint), TOL, what);
    merges
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

/// Hand-merged runs: with room for every partner the hits go uncounted and the bucket shifts down over them, without room the merge falls back, and both give the same terms.
#[test]
fn merges_hits_and_drops_with_and_without_room() {
    let columns = |terms: &[(u64, f64)]| {
        let mut columns = BucketColumns::<1>::default();
        for &(k, c) in terms {
            columns.push([k], [0], Complex64::new(c, 0.0));
        }
        columns
    };
    let bucket = [(1, 1.0), (2, 2.0), (4, 3.0), (8, 4.0)];
    type Terms = &'static [(u64, f64)];
    let cases: [(Terms, Terms, bool); 2] = [
        // Every partner a hit, so it merges in place even without spare room.
        (
            &[(1, 10.0), (2, 20.0), (4, 30.0), (8, 40.0)],
            &[(1, 11.0), (2, 22.0), (4, 33.0), (8, 44.0)],
            true,
        ),
        // Two new keys, and a hit cancelling to an exact zero.
        (
            &[(0, 5.0), (2, -2.0), (5, 6.0)],
            &[(0, 5.0), (1, 1.0), (4, 3.0), (5, 6.0), (8, 4.0)],
            false,
        ),
    ];
    for (partners, want, in_place_without_room) in cases {
        for room in [true, false] {
            let mut got = columns(&bucket);
            if room {
                got.x.reserve(partners.len());
                got.z.reserve(partners.len());
                got.coeff.reserve(partners.len());
            } else {
                got.x.shrink_to_fit();
                got.z.shrink_to_fit();
                got.coeff.shrink_to_fit();
            }
            let partners = columns(partners);
            let mut merged = BucketColumns::default();
            let in_place = merge_partners(&mut got, partners.view(), &KeepAll, &mut merged, None);
            assert_eq!(in_place, room || in_place_without_room, "{partners:?} room={room}");
            assert_eq!(got.x, columns(want).x, "{partners:?} room={room}");
            assert_eq!(got.coeff, columns(want).coeff, "{partners:?} room={room}");
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
    let want = oracle(&input, &channel, &KeepAll.into(), false);
    assert_eq!(got.len(), input.len(), "a Clifford layer is a permutation");
    let residues: Vec<_> = want.iter().filter(|t| t.2.norm() < 1e-14).collect();
    assert_eq!(
        want.len(),
        got.len() + residues.len(),
        "the oracle keeps one residue per moved string"
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

/// Multi-layer runs through the driver, which rebuckets and coarsens between layers, against the oracle stepped layer by layer.
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
    let input = rand_sum::<1>(40, 7, 0x7E);
    let policies: [BuiltinTruncation; 3] =
        [KeepAll.into(), CoefficientThreshold(1e-3).into(), WeightCutoff(4).into()];
    for policy in policies {
        for direction in [Direction::Forward, Direction::Heisenberg] {
            let got = crate::propagate(&circuit, input.clone(), &policy, direction);
            let adjoint = direction == Direction::Heisenberg;
            let mut channels: Vec<_> = circuit.channels.iter().collect();
            if adjoint {
                channels.reverse();
            }
            let want = channels.into_iter().fold(input.clone(), |sum, channel| {
                oracle(&sum, channel.as_ref(), &policy, adjoint)
            });
            assert_terms_close(&got, &want, 1e-10, &format!("{policy:?} {direction:?}"));
        }
    }
}

/// Local rotation layers of a two-partition run take the rotation layer on each partition, a remote one the general path; the gathered result matches one partition's.
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
        let want = crate::propagate(&circuit, input.clone(), &policy, direction);
        assert_terms_close(&got, &want, 1e-10, &format!("{direction:?}"));
    }
}

/// A run of rotation layers in one coset loop against the same layers applied one by one, over hashes whose run spans have every dimension from 0 to the run's length.
fn assert_runs_agree_with_single_layers<const W: usize>(
    input: &PauliSum<W>,
    generators: &[PauliString<W>],
) {
    let mut rng = Xs64::new(0x5EED ^ W as u64);
    let rotations: Vec<PauliRotation<W>> = generators
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let theta = if i % 3 == 2 {
                FRAC_PI_2
            } else {
                (rng.next_u64() as f64 / u64::MAX as f64 - 0.5) * 3.0
            };
            PauliRotation::new(*g, theta)
        })
        .collect();
    let policies: [BuiltinTruncation; 3] =
        [KeepAll.into(), CoefficientThreshold(0.05).into(), WeightCutoff(4).into()];
    for bits in [0u8, 1, 3, 6] {
        for seed in 1..4u64 {
            let sum = input.clone().with_hash(Gf2Hash::<W>::new(input.num_qubits(), bits, seed));
            for policy in &policies {
                for adjoint in [false, true] {
                    let prepared: Vec<_> = rotations
                        .iter()
                        .map(|r| r.prepare(sum.hash(), adjoint).expect("prepares"))
                        .collect();
                    let mut want = sum.clone();
                    let mut scratch = LayerScratch::new();
                    for layer in &prepared {
                        apply_layer_bucketed(&mut want, layer, policy, &mut scratch);
                    }
                    let mut got = sum.clone();
                    apply_rotation_run(&mut got, &prepared, policy, &mut LayerScratch::new());
                    got.assert_invariants();
                    let what =
                        format!("W={W} bits={bits} seed={seed} {policy:?} adjoint={adjoint}");
                    assert_terms_close(&got, &want, TOL, &what);
                }
            }
        }
    }
}

#[test]
fn rotation_runs_agree_with_single_layers_w1() {
    let input = clustered_sum::<1>(1200, 12, &[1, 2, 3, 5, 8], 0xB1);
    type P = PauliString<1>;
    let generators = [
        P::x(2),
        P::z(2).product(&P::z(3)).0,
        P::y(5),
        P::x(8),
        P::x(1).product(&P::y(2)).0,
        P::z(3),
    ];
    assert_runs_agree_with_single_layers(&input, &generators);
}

#[test]
fn rotation_runs_agree_with_single_layers_w2() {
    let input = clustered_sum::<2>(1200, 100, &[3, 40, 63, 64, 70, 99], 0xB2);
    type P = PauliString<2>;
    let generators = [
        P::x(63),
        P::z(63).product(&P::z(64)).0,
        P::y(70),
        P::x(3),
        P::x(40).product(&P::y(99)).0,
        P::x(63),
    ];
    assert_runs_agree_with_single_layers(&input, &generators);
}

/// `(unread, unmerged)` members over every slot of `scratch`.
fn skips<const W: usize>(scratch: &LayerScratch<W>) -> (usize, usize) {
    let mut total = scratch.task.rotation_skips;
    for slot in &scratch.workers {
        let (unread, unmerged) = slot.lock().unwrap().rotation_skips;
        total.0 += unread;
        total.1 += unmerged;
    }
    total
}

/// A kicked-Ising-like circuit through the driver, against the same layers applied one by one: the bucket count lets rotation runs form, and the light cone of a sum clustered on `cluster` leaves most gates touching no string of most buckets.
///
/// The input holds strings each policy drops, which the first layer must drop though no gate changes them.
fn assert_propagated_runs_agree<const W: usize>(num_qubits: usize, cluster: &[usize]) {
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in 0..num_qubits {
        let theta = if q % 5 == 0 { FRAC_PI_2 } else { 0.3 + 0.01 * q as f64 };
        circuit.push(PauliRotation::new(PauliString::x(q), theta));
    }
    for q in 0..num_qubits - 1 {
        circuit.push(zz_rotation(q, q + 1, if q % 3 == 0 { FRAC_PI_2 } else { 0.2 }));
    }
    let input = clustered_sum::<W>(4000, num_qubits, cluster, 0x1C ^ W as u64);
    let options = PropagateOptions { target_bucket_len: 32, ..PropagateOptions::default() };
    let policies: [BuiltinTruncation; 2] =
        [CoefficientThreshold(0.05).into(), WeightCutoff(4).into()];
    for policy in policies {
        for direction in [Direction::Forward, Direction::Heisenberg] {
            let mut scratch = LayerScratch::new();
            let got = crate::propagate_with(
                &circuit,
                input.clone(),
                &policy,
                direction,
                &mut scratch,
                options,
            );
            let adjoint = direction == Direction::Heisenberg;
            let mut channels: Vec<_> = circuit.channels.iter().collect();
            if adjoint {
                channels.reverse();
            }
            let mut want = input.clone();
            let mut layer_scratch = LayerScratch::new();
            for channel in channels {
                let prepared = channel.prepare(want.hash(), adjoint).expect("prepares");
                apply_layer_bucketed(&mut want, &prepared, &policy, &mut layer_scratch);
            }
            let what = format!("W={W} {policy:?} {direction:?}");
            assert_terms_close(&got, &want, 1e-10, &what);
            assert!(scratch.rotation_run_layers > 0, "{what}: no rotation run");
            let (unread, unmerged) = skips(&scratch);
            assert!(unread > 0 && unmerged > 0, "{what}: no member skipped ({unread}, {unmerged})");
        }
    }
}

#[test]
fn propagated_runs_and_light_cone_skips_agree_w1() {
    assert_propagated_runs_agree::<1>(40, &[18, 19, 20, 21, 22, 23]);
}

#[test]
fn propagated_runs_and_light_cone_skips_agree_w2() {
    assert_propagated_runs_agree::<2>(80, &[60, 62, 63, 64, 65, 67]);
}

/// A sorted bucket of `len` distinct keys on `W` words, every coefficient of magnitude at least 1, so all pass a 0.5 threshold.
fn sparse_case<const W: usize>(len: usize, rng: &mut Xs64) -> BucketColumns<W> {
    let mut keys: Vec<([u64; W], [u64; W])> = (0..len * 2)
        .map(|_| {
            (
                core::array::from_fn(|_| rng.next_u64() % 64),
                core::array::from_fn(|_| rng.next_u64() % 4),
            )
        })
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys.truncate(len);
    let mut bucket = BucketColumns::<W>::default();
    for (x, z) in keys {
        let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
        bucket.push(x, z, Complex64::new(sign * (1.0 + (rng.next_u64() % 8) as f64), 0.0));
    }
    bucket
}

/// The sparse merge against the per-row merge, on buckets whose dirty rows were rescaled (some below the threshold, some to zero) and whose partners mix hits and new keys.
fn assert_sparse_merge_agrees<const W: usize>() {
    let mut rng = Xs64::new(0x5BA ^ W as u64);
    let policy = CoefficientThreshold(0.5);
    for case in 0..300 {
        let len = 1 + (rng.next_u64() % 200) as usize;
        let mut bucket = sparse_case::<W>(len, &mut rng);
        let len = bucket.len();
        let mut dirty: Vec<u32> =
            (0..len as u32).filter(|_| rng.next_u64().is_multiple_of(9)).collect();
        dirty.dedup();
        for &row in &dirty {
            let scale = [0.0, 0.1, 0.7, -2.0][(rng.next_u64() % 4) as usize];
            bucket.coeff[row as usize] *= scale;
        }
        let packed: Vec<u32> = dirty.iter().map(|&row| row << 4 | (row & 15)).collect();
        // Partners: some on dirty rows' keys, some new.
        let mut partner_keys: Vec<([u64; W], [u64; W])> = dirty
            .iter()
            .filter(|_| rng.next_u64() & 1 == 0)
            .map(|&row| (bucket.x[row as usize], bucket.z[row as usize]))
            .collect();
        for _ in 0..(rng.next_u64() % 12) {
            partner_keys.push((
                core::array::from_fn(|_| rng.next_u64() % 64),
                core::array::from_fn(|_| 4 + rng.next_u64() % 4),
            ));
        }
        partner_keys.sort_unstable();
        partner_keys.dedup();
        let mut partners = BucketColumns::<W>::default();
        for (x, z) in partner_keys {
            partners.push(
                x,
                z,
                Complex64::new([0.3, 0.6, -1.5][(rng.next_u64() % 3) as usize], 0.0),
            );
        }
        let mut want = bucket.clone();
        want.x.reserve(partners.len());
        want.z.reserve(partners.len());
        want.coeff.reserve(partners.len());
        let total = want.len() + partners.len();
        merge_backward_in_place(&mut want, partners.view(), &policy, total);
        let mut got = bucket.clone();
        got.x.reserve(partners.len());
        got.z.reserve(partners.len());
        got.coeff.reserve(partners.len());
        merge_sparse_in_place(&mut got, partners.view(), &packed, &policy, total);
        assert_eq!(got.x, want.x, "W={W} case {case}");
        assert_eq!(got.z, want.z, "W={W} case {case}");
        assert_eq!(got.coeff, want.coeff, "W={W} case {case}");
    }
}

#[test]
fn sparse_merge_agrees_with_the_per_row_merge() {
    assert_sparse_merge_agrees::<1>();
    assert_sparse_merge_agrees::<2>();
}
