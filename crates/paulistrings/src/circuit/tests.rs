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
