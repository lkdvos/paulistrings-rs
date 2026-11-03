use super::*;
use crate::test_support::{
    assert_terms_close, differential_channels_w1, differential_channels_w2, rand_sum,
    random_circuit, KeepAll,
};
use crate::truncation::CoefficientThreshold;
use crate::{propagate_with, Channel, LayerScratch, PauliSum, PropagateOptions};

const SIZINGS: [HashOptions; 3] = [
    HashOptions { target_bucket_len: usize::MAX / 2, min_buckets: 1 },
    HashOptions { target_bucket_len: 64, min_buckets: 16 },
    HashOptions { target_bucket_len: 8, min_buckets: 16 },
];

fn sorted_reference<const W: usize, T: TruncationPolicy<W>>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    direction: Direction,
) -> PauliSum<W> {
    let options = PropagateOptions { target_bucket_len: 64, min_buckets: 16 };
    propagate_with(circuit, sum.clone(), policy, direction, &mut LayerScratch::new(), options)
}

fn hashed<const W: usize, T: TruncationPolicy<W>>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    direction: Direction,
    options: HashOptions,
) -> PauliSum<W> {
    let mut hashed = HashSum::from_pauli_sum(sum);
    propagate_hashed(circuit, &mut hashed, policy, direction, &mut HashScratch::new(), options);
    assert_eq!(hashed.len(), hashed.buckets.iter().map(|t| t.len()).sum::<usize>());
    for (b, table) in hashed.buckets.iter().enumerate() {
        for e in table.entries() {
            assert_eq!(hashed.hash.bucket_of(&e.x, &e.z) as usize, b, "entry in the wrong bucket");
        }
    }
    hashed.to_pauli_sum()
}

fn agree_on<const W: usize>(channels: Vec<(&'static str, Box<dyn Channel<W>>)>, num_qubits: usize) {
    let sum = rand_sum::<W>(3000, num_qubits, 7);
    for (name, channel) in channels {
        let mut circuit = Circuit::<W>::new(num_qubits);
        circuit.push_boxed(channel);
        for direction in [Direction::Forward, Direction::Heisenberg] {
            let want = sorted_reference(&circuit, &sum, &KeepAll, direction);
            for options in SIZINGS {
                let got = hashed(&circuit, &sum, &KeepAll, direction, options);
                assert_terms_close(&got, &want, 1e-12, &format!("{name} {options:?}"));
            }
        }
    }
}

#[test]
fn every_differential_channel_agrees_with_the_sorted_engine_w1() {
    agree_on::<1>(differential_channels_w1(), 12);
}

#[test]
fn every_differential_channel_agrees_with_the_sorted_engine_w2() {
    agree_on::<2>(differential_channels_w2(), 128);
}

fn random_circuits_agree<const W: usize>(num_qubits: usize) {
    for seed in 0..3 {
        let circuit = random_circuit::<W>(num_qubits, 16, seed, true);
        let sum = rand_sum::<W>(300, num_qubits, seed + 100);
        let policy = CoefficientThreshold(1e-3);
        let want = sorted_reference(&circuit, &sum, &policy, Direction::Heisenberg);
        for options in SIZINGS {
            let got = hashed(&circuit, &sum, &policy, Direction::Heisenberg, options);
            assert_terms_close(&got, &want, 1e-10, &format!("seed {seed} {options:?}"));
        }
    }
}

#[test]
fn random_truncated_circuits_agree_with_the_sorted_engine_w1() {
    random_circuits_agree::<1>(10);
}

#[test]
fn random_truncated_circuits_agree_with_the_sorted_engine_w2() {
    random_circuits_agree::<2>(90);
}

/// Kicked Ising with a wide rotation and a Clifford mixed in, under a threshold that cuts every layer: the fused kernel's pass-1 truncation boundary.
fn rotation_clifford_circuit<const W: usize>(num_qubits: usize, steps: usize) -> Circuit<W> {
    use crate::test_support::zz_rotation;
    use crate::{PauliRotation, PauliString, PtmChannel};
    let mut c = Circuit::<W>::new(num_qubits);
    for step in 0..steps {
        for a in 0..num_qubits - 1 {
            c.push(zz_rotation::<W>(a, a + 1, 0.3 + 0.1 * step as f64));
        }
        for q in 0..num_qubits {
            c.push(PauliRotation::<W>::new(PauliString::<W>::x(q), 0.7));
        }
        let mut wide = PauliString::<W>::x(0);
        wide.mul_assign(&PauliString::<W>::z(2));
        wide.mul_assign(&PauliString::<W>::y(4));
        c.push(PauliRotation::<W>::new(wide, 0.45));
        c.push(PtmChannel::cnot(1, 3));
        c.push(PtmChannel::hadamard(2));
        c.push(PtmChannel::phase_gate(5));
    }
    c
}

fn truncated_rotations_agree<const W: usize>(num_qubits: usize) {
    let circuit = rotation_clifford_circuit::<W>(num_qubits, 3);
    let sum = rand_sum::<W>(60, num_qubits, 11);
    for threshold in [1e-2, 3e-3] {
        let policy = CoefficientThreshold(threshold);
        let want = sorted_reference(&circuit, &sum, &policy, Direction::Heisenberg);
        for options in SIZINGS {
            let got = hashed(&circuit, &sum, &policy, Direction::Heisenberg, options);
            assert_terms_close(&got, &want, 1e-10, &format!("threshold {threshold} {options:?}"));
        }
    }
}

#[test]
fn truncated_rotation_and_clifford_circuits_agree_with_the_sorted_engine_w1() {
    truncated_rotations_agree::<1>(10);
}

#[test]
fn truncated_rotation_and_clifford_circuits_agree_with_the_sorted_engine_w2() {
    truncated_rotations_agree::<2>(70);
}
