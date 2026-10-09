use num_complex::Complex64;

use crate::channel::{support_mask, Channel, Clifford1Q, OutputBuffer};
use crate::circuit::Circuit;
use crate::engine::bucketed::LayerScratch;
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::truncation::TruncationPolicy;

use super::{propagate, propagate_with, Direction, PropagateOptions};

struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

/// Support on three qubits, so `prepare` returns `None`.
struct ThreeQubits;
impl<const W: usize> Channel<W> for ThreeQubits {
    fn max_fanout(&self) -> usize {
        1
    }
    fn support(&self) -> [u64; W] {
        support_mask(&[0, 1, 2])
    }
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        out.push(*input_x, *input_z, coeff);
    }
}

#[test]
#[should_panic(expected = "Channel::prepare declined")]
fn an_unpreparable_channel_panics() {
    let mut accumulator = BuildAccumulator::<1>::with_capacity(8, 1);
    accumulator.add_term(PauliString::<1>::z(0), Complex64::new(1.0, 0.0));
    let sum = accumulator.finalize();

    let mut circuit = Circuit::<1>::new(8);
    circuit.push(ThreeQubits);
    let _ = propagate(&circuit, sum, &AlwaysKeep, Direction::Forward);
}

#[test]
fn gate_trace_forward_indices_match_circuit_order() {
    let mut accumulator = BuildAccumulator::<1>::with_capacity(8, 1);
    accumulator.add_term(PauliString::<1>::z(0), Complex64::new(1.0, 0.0));
    let sum = accumulator.finalize();

    let mut circuit = Circuit::<1>::new(1);
    circuit.push(Clifford1Q::h(0));
    circuit.push(Clifford1Q::s(0));

    let mut scratch = LayerScratch::<1>::new();
    scratch.enable_gate_trace();
    let _ = propagate_with(
        &circuit,
        sum,
        &AlwaysKeep,
        Direction::Forward,
        &mut scratch,
        PropagateOptions::default(),
    );
    let trace = scratch.take_gate_trace().unwrap();

    assert_eq!(trace.application_index, vec![0, 1]);
    assert_eq!(trace.circuit_index, vec![0, 1]);
    assert_eq!(trace.gate_name.len(), 2);
    assert_eq!(trace.terms_in, vec![1, 1]);
    assert_eq!(trace.terms_out, vec![1, 1]);
    assert_eq!(trace.nanos.len(), 2);
}

#[test]
fn gate_trace_heisenberg_reverses_circuit_index_not_application_index() {
    let mut accumulator = BuildAccumulator::<1>::with_capacity(8, 1);
    accumulator.add_term(PauliString::<1>::z(0), Complex64::new(1.0, 0.0));
    let sum = accumulator.finalize();

    let mut circuit = Circuit::<1>::new(1);
    circuit.push(Clifford1Q::h(0));
    circuit.push(Clifford1Q::s(0));

    let mut scratch = LayerScratch::<1>::new();
    scratch.enable_gate_trace();
    let _ = propagate_with(
        &circuit,
        sum,
        &AlwaysKeep,
        Direction::Heisenberg,
        &mut scratch,
        PropagateOptions::default(),
    );
    let trace = scratch.take_gate_trace().unwrap();

    // Applied in loop order 0, 1, but the circuit ran channel 1 (`s`) first.
    assert_eq!(trace.application_index, vec![0, 1]);
    assert_eq!(trace.circuit_index, vec![1, 0]);
}

#[test]
fn gate_trace_stays_empty_when_not_enabled() {
    let mut accumulator = BuildAccumulator::<1>::with_capacity(8, 1);
    accumulator.add_term(PauliString::<1>::z(0), Complex64::new(1.0, 0.0));
    let sum = accumulator.finalize();

    let mut circuit = Circuit::<1>::new(1);
    circuit.push(Clifford1Q::h(0));

    let mut scratch = LayerScratch::<1>::new();
    let _ = propagate_with(
        &circuit,
        sum,
        &AlwaysKeep,
        Direction::Forward,
        &mut scratch,
        PropagateOptions::default(),
    );
    assert!(scratch.take_gate_trace().is_none());
}
