use num_complex::Complex64;

use crate::channel::{support_mask, Channel, OutputBuffer};
use crate::circuit::Circuit;
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::truncation::TruncationPolicy;

use super::{propagate, Direction};

struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

/// Support on three qubits, so `prepare` returns `None`.
struct ThreeQubits;
impl<const W: usize> Channel<W> for ThreeQubits {
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
        support_mask(&[0, 1, 2])
    }
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<W>,
    ) {
        out.push(*input_x, *input_z, coeff);
    }
}

#[test]
#[should_panic(expected = "unsupported except for PauliRotation")]
fn an_unpreparable_channel_panics() {
    let mut accumulator = BuildAccumulator::<1>::with_capacity(8, 1);
    accumulator.add_term(PauliString::<1>::z(0), Complex64::new(1.0, 0.0));
    let sum = accumulator.finalize();

    let mut circuit = Circuit::<1>::new(8);
    circuit.push(ThreeQubits);
    let _ = propagate(&circuit, sum, &AlwaysKeep, Direction::Forward);
}

#[test]
fn progress_scope_names_the_partitions() {
    use super::progress_scope;
    assert_eq!(progress_scope(None, 1, None), "1 partition");
    assert_eq!(
        progress_scope(None, 4, Some("0:unpinned x2")),
        "4 partitions [0:unpinned x2]"
    );
    assert_eq!(
        progress_scope(Some(2), 4, Some("0:0-7x8")),
        "partition 2/4 [0:0-7x8]"
    );
}
