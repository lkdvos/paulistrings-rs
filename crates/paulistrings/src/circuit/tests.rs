use super::*;
use crate::channel::{PauliRotation, PtmChannel};
use crate::pauli_string::PauliString;

#[test]
fn push_keeps_order_and_counts() {
    let mut circuit = Circuit::<2>::new(100);
    circuit.push(PtmChannel::hadamard(0));
    circuit.push(PtmChannel::cnot(99, 64));
    circuit.push_boxed(Box::new(PtmChannel::depolarizing(5, 0.1)));
    assert_eq!(circuit.num_qubits(), 100);
    assert_eq!(circuit.len(), 3);
    let names: Vec<_> = circuit.channels().iter().map(|c| c.debug_name()).collect();
    assert_eq!(names, ["hadamard", "cnot", "depolarizing"]);
}

#[test]
#[should_panic(expected = "Circuit::push: cnot acts on qubit 8, outside this 8-qubit circuit")]
fn push_rejects_a_support_past_num_qubits() {
    let mut circuit = Circuit::<1>::new(8);
    circuit.push(PtmChannel::cnot(8, 0));
}

#[test]
#[should_panic(
    expected = "Circuit::push: PauliRotation acts on qubit 70, outside this 70-qubit circuit"
)]
fn push_rejects_a_wide_generator_past_num_qubits_in_a_later_word() {
    let mut generator = PauliString::<2>::z(3);
    let _ = generator.mul_assign(&PauliString::<2>::x(70));
    let mut circuit = Circuit::<2>::new(70);
    circuit.push(PauliRotation::new(generator, 0.1));
}

#[test]
fn an_empty_support_fits_any_circuit() {
    let mut circuit = Circuit::<1>::new(0);
    circuit.push(PauliRotation::new(PauliString::<1>::identity(), 0.1));
    assert_eq!(circuit.len(), 1);
}

#[test]
fn deltas_are_the_key_shifts_each_channel_applies() {
    let mut zz = PauliString::<2>::z(3);
    let _ = zz.mul_assign(&PauliString::<2>::z(70));
    let mut circuit = Circuit::<2>::new(72);
    circuit.push(PauliRotation::new(PauliString::<2>::x(5), 0.3));
    circuit.push(PauliRotation::new(zz, 0.3));
    circuit.push(PtmChannel::hadamard(64));
    circuit.push(PauliRotation::new(PauliString::<2>::x(5), 0.0));
    let deltas = circuit.deltas();
    let key = |p: PauliString<2>| (p.x, p.z);
    // A rotation shifts by its generator; the identity shift is left out.
    assert_eq!(deltas[0], Some(vec![key(PauliString::x(5))]));
    let (zz_x, zz_z) = (deltas[1].as_ref().unwrap()[0].0, deltas[1].as_ref().unwrap()[0].1);
    assert_eq!((zz_x, zz_z[0], zz_z[1]), ([0, 0], 1 << 3, 1 << 6));
    // Hadamard swaps X and Z and keeps Y, so its one nonzero shift is x ⊕ z on its qubit.
    let mut y = PauliString::<2>::x(64);
    let _ = y.mul_assign(&PauliString::<2>::z(64));
    assert_eq!(deltas[2], Some(vec![key(y)]));
    // A zero angle on one qubit tabulates as the identity, which shifts nothing, as the partition plan sees it.
    assert_eq!(deltas[3], Some(Vec::new()));
}
