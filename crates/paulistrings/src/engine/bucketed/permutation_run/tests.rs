use num_complex::Complex64;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use super::*;
use crate::channel::{Channel, PauliRotation, Prepare, PtmChannel};
use crate::engine::bucketed::{apply_layer_bucketed, LayerScratch};
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::hash::Gf2Hash;
use crate::test_support::{
    assert_terms_close, low_weight_sum, rand_sum, random_clifford_circuit, zz_rotation, KeepAll,
    Xs64,
};
use crate::truncation::{BuiltinTruncation, CoefficientThreshold, WeightCutoff};
use crate::Circuit;

const TOL: f64 = 1e-12;

fn step_of<const W: usize>(
    channel: &dyn Channel<W>,
    hash: &Gf2Hash<W>,
) -> Option<PermutationStep<W, Complex64>> {
    PermutationStep::of(&channel.prepare(hash, true).expect("a built-in prepares"))
}

#[test]
fn classifies_one_to_one_channels() {
    let hash = Gf2Hash::<1>::new(8, 3, 7);
    let accepted: Vec<(&str, Box<dyn Channel<1>>)> = vec![
        ("ZZ quarter turn", Box::new(zz_rotation::<1>(1, 4, FRAC_PI_2))),
        ("ZZ minus quarter turn", Box::new(zz_rotation::<1>(1, 4, -FRAC_PI_2))),
        ("X quarter turn", Box::new(PauliRotation::new(PauliString::<1>::x(2), FRAC_PI_2))),
        ("H", Box::new(PtmChannel::hadamard(0))),
        ("S", Box::new(PtmChannel::phase_gate(3))),
        ("Y", Box::new(PtmChannel::pauli_y(5))),
        ("CNOT", Box::new(PtmChannel::cnot(2, 6))),
        ("CZ", Box::new(PtmChannel::cz(6, 2))),
        ("SWAP", Box::new(PtmChannel::swap(0, 7))),
        ("dephasing", Box::new(PtmChannel::dephasing(1, 0.2))),
    ];
    for (name, channel) in &accepted {
        assert!(step_of(channel.as_ref(), &hash).is_some(), "{name} is one-to-one");
    }
    let refused: Vec<(&str, Box<dyn Channel<1>>)> = vec![
        ("ZZ eighth turn", Box::new(zz_rotation::<1>(1, 4, FRAC_PI_4))),
        ("X eighth turn", Box::new(PauliRotation::new(PauliString::<1>::x(2), -FRAC_PI_4))),
        ("amplitude damping", Box::new(PtmChannel::amplitude_damping(1, 0.3))),
    ];
    for (name, channel) in &refused {
        assert!(step_of(channel.as_ref(), &hash).is_none(), "{name} splits strings");
    }
}

/// Every channel of `circuit`, prepared under `sum`'s hash in Heisenberg order, applied one layer at a time.
fn per_layer<const W: usize, T: TruncationPolicy<W>>(
    circuit: &Circuit<W>,
    mut sum: PauliSum<W>,
    policy: &T,
) -> PauliSum<W> {
    let mut scratch = LayerScratch::<W>::new();
    for channel in circuit.channels.iter().rev() {
        let prepared = channel.prepare(sum.hash(), true).expect("a built-in prepares");
        apply_layer_bucketed(&mut sum, &prepared, policy, &mut scratch);
    }
    sum
}

fn as_run<const W: usize, T: TruncationPolicy<W>>(
    circuit: &Circuit<W>,
    mut sum: PauliSum<W>,
    policy: &T,
) -> PauliSum<W> {
    let steps: Vec<_> = circuit
        .channels
        .iter()
        .rev()
        .map(|channel| step_of(channel.as_ref(), sum.hash()).expect("a one-to-one channel"))
        .collect();
    apply_permutation_run(&mut sum, &steps, policy, &mut PermutationScratch::default());
    sum.assert_invariants();
    sum
}

/// Cliffords and quarter-turn rotations about one-, two-qubit and `Y`-bearing generators on `num_qubits` qubits.
fn clifford_mix<const W: usize>(num_qubits: usize, layers: usize, seed: u64) -> Circuit<W> {
    let mut circuit = random_clifford_circuit::<W>(num_qubits, layers, seed);
    let mut rng = Xs64::new(seed ^ 0x5eed);
    for _ in 0..layers {
        let a = (rng.next_u64() % num_qubits as u64) as usize;
        let b = (a + 1 + (rng.next_u64() % (num_qubits as u64 - 1)) as usize) % num_qubits;
        let theta = if rng.next_u64().is_multiple_of(2) { FRAC_PI_2 } else { -FRAC_PI_2 };
        let generator = match rng.next_u64() % 3 {
            0 => PauliString::<W>::z(a).product(&PauliString::z(b)).0,
            1 => PauliString::<W>::y(a).product(&PauliString::x(b)).0,
            _ => PauliString::<W>::y(a),
        };
        circuit.push(PauliRotation::new(generator, theta));
    }
    circuit
}

/// Runs against the per-layer path under every built-in per-term policy, the weight cut dropping terms mid-run.
fn agrees_with_per_layer<const W: usize>(num_qubits: usize) {
    let policies: [BuiltinTruncation; 3] = [
        KeepAll.into(),
        WeightCutoff(num_qubits as u32 * 3 / 4).into(),
        CoefficientThreshold(0.5).into(),
    ];
    for (terms, bits, seed) in [(300, 2, 1u64), (20_000, 6, 2), (20_000, 0, 3)] {
        let mut sum = rand_sum::<W>(terms, num_qubits, seed);
        sum = sum.with_hash(Gf2Hash::new(num_qubits, bits, seed));
        let circuit = clifford_mix::<W>(num_qubits, 40, seed);
        for policy in &policies {
            let want = per_layer(&circuit, sum.clone(), policy);
            let got = as_run(&circuit, sum.clone(), policy);
            let what = format!("W={W}, {terms} terms, {bits} bits, {policy:?}");
            assert_terms_close(&got, &want, TOL, &what);
            if matches!(policy, BuiltinTruncation::Weight(_)) && terms > 300 {
                assert!(want.len() < sum.len(), "{what}: the weight cut drops terms mid-run");
            }
        }
    }
}

#[test]
fn a_run_agrees_with_the_per_layer_path_w1() {
    agrees_with_per_layer::<1>(40);
}

#[test]
fn a_run_agrees_with_the_per_layer_path_w2() {
    agrees_with_per_layer::<2>(100);
}

#[test]
fn two_quarter_turns_negate_an_anticommuting_y_string() {
    // R(π/2)² = R(π) maps an anticommuting `P` to `cos(π)·P = −P`, and four quarter turns are the identity.
    let hash = Gf2Hash::<2>::new(70, 2, 3);
    let zz = zz_rotation::<2>(3, 66, FRAC_PI_2);
    let step = step_of(&zz, &hash).expect("a quarter turn");
    let y = PauliString::<2>::y(3).product(&PauliString::x(40)).0;
    let commuting = PauliString::<2>::y(3).product(&PauliString::y(66)).0;
    let mut accumulator = BuildAccumulator::<2>::with_capacity(70, 2);
    accumulator.add_term(y, Complex64::new(0.25, -0.5));
    accumulator.add_term(commuting, Complex64::new(2.0, 0.0));
    let mut sum = accumulator.finalize().with_hash(hash);
    let mut scratch = PermutationScratch::default();
    apply_permutation_run(&mut sum, &[step.clone(), step.clone()], &KeepAll, &mut scratch);
    assert_eq!(sum.len(), 2);
    assert_eq!(sum.get(&y.x, &y.z), Some(Complex64::new(-0.25, 0.5)));
    assert_eq!(sum.get(&commuting.x, &commuting.z), Some(Complex64::new(2.0, 0.0)));
    apply_permutation_run(&mut sum, &[step.clone(), step], &KeepAll, &mut scratch);
    assert_eq!(sum.get(&y.x, &y.z), Some(Complex64::new(0.25, -0.5)));
}

#[test]
fn low_weight_strings_over_several_segments_agree_with_the_per_layer_path() {
    for (bits, seed) in [(5, 4u64), (0, 5)] {
        let sum =
            low_weight_sum::<2>(20_000, 100, 4, seed).with_hash(Gf2Hash::new(100, bits, seed));
        let circuit = clifford_mix::<2>(100, 300, seed);
        assert!(circuit.channels.len() > 2 * SEGMENT_STEPS, "the run spans three segments");
        for policy in [WeightCutoff(12), WeightCutoff(100)] {
            let want = per_layer(&circuit, sum.clone(), &policy);
            let got = as_run(&circuit, sum.clone(), &policy);
            assert_terms_close(&got, &want, TOL, &format!("{bits} bits, {policy:?}"));
        }
    }
}

#[test]
fn a_string_no_step_touches_still_meets_keep_term() {
    // Every step acts on qubits 0 and 1, which the heavy string leaves alone; the per-layer merge asks `keep_term` of it after each step.
    let hash = Gf2Hash::<1>::new(20, 0, 1);
    let light = PauliString::<1>::x(0);
    let mut heavy = PauliString::<1>::identity();
    for q in 2..10 {
        heavy.x[0] |= 1 << q;
    }
    let mut accumulator = BuildAccumulator::<1>::with_capacity(20, 2);
    accumulator.add_term(light, Complex64::new(1.0, 0.0));
    accumulator.add_term(heavy, Complex64::new(1.0, 0.0));
    let sum = accumulator.finalize().with_hash(hash.clone());
    let mut circuit = Circuit::<1>::new(20);
    circuit.push(PtmChannel::cnot(0, 1));
    circuit.push(PtmChannel::hadamard(1));
    let got = as_run(&circuit, sum.clone(), &WeightCutoff(4));
    assert_terms_close(&got, &per_layer(&circuit, sum, &WeightCutoff(4)), TOL, "weight 4");
    assert_eq!(got.len(), 1, "the weight-8 string is cut");
}
