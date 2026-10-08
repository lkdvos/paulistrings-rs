use super::*;

use crate::channel::clifford::Clifford1Q;
use crate::channel::rotation::PauliRotation;
use crate::pauli_string::PauliString;
use crate::test_support::zz_rotation;

fn hash(num_qubits: usize) -> Gf2Hash<1> {
    Gf2Hash::<1>::new(num_qubits, 4, 0xB00C)
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
    c.push(Clifford1Q::h(0)); // delta X_0 Z_0
    c.push(Clifford1Q::s(1)); // delta Z_1
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
