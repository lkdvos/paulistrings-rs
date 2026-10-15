use super::*;

use crate::channel::rotation::PauliRotation;
use crate::channel::PtmChannel;
use crate::pauli_string::PauliString;
use crate::test_support::zz_rotation;

fn hash(num_qubits: usize) -> Gf2Hash<1> {
    Gf2Hash::<1>::new(num_qubits, 4, 0xB00C)
}

const NQ: usize = 128;

/// One partition row that is `Z` on `qubit`: `part(v)` is the z-bit of `v` there.
fn z_row(qubit: usize) -> PartitionRows<2> {
    let mut rz = [0u64; 2];
    rz[qubit / 64] = 1u64 << (qubit % 64);
    PartitionRows::<2>::from_rows(NQ, vec![[0u64; 2]], vec![rz])
}

#[test]
fn circuit_generators_merges_duplicate_masks_and_counts_layers() {
    let mut c = Circuit::<1>::new(4);
    c.push(PauliRotation::new(PauliString::<1>::x(0), 0.3));
    c.push(zz_rotation::<1>(0, 1, 0.2));
    c.push(PauliRotation::new(PauliString::<1>::x(0), 0.7));
    let gens = circuit_generators(&c, &hash(4), false);

    // Two distinct masks: X_0 (twice) and Z_0·Z_1 (once); identity deltas are dropped.
    assert_eq!(gens.len(), 2);
    assert_eq!(gens[0].mask_x, [0b0001]);
    assert_eq!(gens[0].mask_z, [0]);
    assert_eq!(gens[0].weight, 2.0);
    assert_eq!(gens[0].name, "PauliRotation");
    assert_eq!(gens[1].mask_x, [0]);
    assert_eq!(gens[1].mask_z, [0b0011]);
    assert_eq!(gens[1].weight, 1.0);
}

#[test]
fn circuit_generators_walks_layers_in_application_order() {
    let mut c = Circuit::<1>::new(4);
    c.push(PtmChannel::hadamard(0)); // delta X_0 Z_0
    c.push(PtmChannel::phase_gate(1)); // delta Z_1
    let forward = circuit_generators(&c, &hash(4), false);
    let adjoint = circuit_generators(&c, &hash(4), true);
    assert_eq!(forward.len(), 2);
    assert_eq!((forward[0].mask_x, forward[0].mask_z), ([0b1], [0b1]));
    assert_eq!((forward[1].mask_x, forward[1].mask_z), ([0], [0b10]));
    // Reverse order, same masks (an adjoint's delta set is the same subspace).
    assert_eq!((adjoint[0].mask_x, adjoint[0].mask_z), ([0], [0b10]));
    assert_eq!((adjoint[1].mask_x, adjoint[1].mask_z), ([0b1], [0b1]));
}

#[test]
fn circuit_generators_of_an_empty_circuit_is_empty() {
    assert!(circuit_generators(&Circuit::<1>::new(4), &hash(4), false).is_empty());
}

#[test]
fn count_remote_deltas_reports_layers_in_application_order() {
    // Two layers with hand-checkable delta sets under a Z-row on qubit 3:
    //   H(3):  deltas {0, XZ_3} -> part = {0, 1}  -> (1 local, 1 remote)
    //   S(9):  deltas {0,  Z_9} -> part = {0, 0}  -> (2 local, 0 remote)
    // (S maps X -> Y and Y -> -X, so its only nonzero delta is a z-bit.)
    let rows = z_row(3);
    let h = Gf2Hash::<2>::new(NQ, 8, 0xB00C);
    let mut circuit = Circuit::<2>::new(NQ);
    circuit.push(PtmChannel::hadamard(3));
    circuit.push(PtmChannel::phase_gate(9));

    assert_eq!(count_remote_deltas(&circuit, &h, &rows, false), vec![(1, 1), (2, 0)],);
    // Heisenberg order is the reverse. `S` is not self-adjoint, but its adjoint's delta set is the same subspace, so the counts do not move.
    assert_eq!(count_remote_deltas(&circuit, &h, &rows, true), vec![(2, 0), (1, 1)],);
}

#[test]
fn count_remote_deltas_without_rows_is_all_local() {
    let rows = PartitionRows::<2>::none(NQ);
    let h = Gf2Hash::<2>::new(NQ, 8, 0xB00C);
    let mut circuit = Circuit::<2>::new(NQ);
    circuit.push(PtmChannel::cnot(1, 4));
    circuit.push(PtmChannel::hadamard(3));
    assert_eq!(count_remote_deltas(&circuit, &h, &rows, false), vec![(4, 0), (2, 0)],);
}
