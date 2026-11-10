use super::*;
use crate::channel::rotation::PauliRotation;
use crate::channel::PtmChannel;
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::test_support::{kicked_ising, square_lattice};

/// Breadth-first from the centre of the 3 × 3 square lattice.
const CENTRE_ORDER: [usize; 9] = [4, 1, 3, 5, 7, 0, 2, 6, 8];

fn sum_of<const W: usize>(strings: &[PauliString<W>], num_qubits: usize) -> PauliSum<W> {
    let mut accumulator = BuildAccumulator::<W>::new(num_qubits);
    for string in strings {
        accumulator.add_term(*string, num_complex::Complex64::new(1.0, 0.0));
    }
    accumulator.finalize()
}

fn string<const W: usize>(x: &[usize], z: &[usize], num_qubits: usize) -> PauliString<W> {
    PauliString {
        x: qubit_mask(x.iter().copied(), num_qubits),
        z: qubit_mask(z.iter().copied(), num_qubits),
    }
}

#[test]
fn nearest_first_walks_a_square_lattice_breadth_first() {
    let bonds = square_lattice(3, 3);
    assert_eq!(PartitionRowPolicy::nearest_first(9, &bonds, &[4]), CENTRE_ORDER);
    assert_eq!(PartitionRowPolicy::nearest_first(9, &bonds, &[0]), [0, 1, 3, 2, 4, 6, 5, 7, 8]);
    // Several sites start together, in ascending order whatever order they come in.
    assert_eq!(PartitionRowPolicy::nearest_first(9, &bonds, &[8, 0]), [0, 8, 1, 3, 5, 7, 2, 4, 6]);
    // A qubit no edge reaches comes last.
    assert_eq!(PartitionRowPolicy::nearest_first(10, &bonds, &[4]), [4, 1, 3, 5, 7, 0, 2, 6, 8, 9]);
}

#[test]
fn parity_rows_read_one_axis_of_consecutive_groups() {
    let bonds = square_lattice(3, 3);
    let rows_of =
        |axis| PartitionRowPolicy::parity_around(axis, 9, &bonds, &[4], 2).rows::<1>(9, 2, 0);
    let groups = vec![[1 << 4 | 1 << 1], [1 << 3 | 1 << 5]];
    let zero = vec![[0u64]; 2];
    assert_eq!(rows_of(PauliAxis::X), PartitionRows::from_rows(9, groups.clone(), zero.clone()));
    assert_eq!(rows_of(PauliAxis::Z), PartitionRows::from_rows(9, zero, groups.clone()));
    assert_eq!(rows_of(PauliAxis::Y), PartitionRows::from_rows(9, groups.clone(), groups));

    let rows = rows_of(PauliAxis::X);
    let label = |s: PauliString<1>| rows.partition_of(&s.x, &s.z);
    assert_eq!(label(string(&[4], &[], 9)), 1);
    assert_eq!(label(string(&[3], &[], 9)), 2);
    // Y_1 X_5: x-bits in both groups.
    assert_eq!(label(string(&[1, 5], &[1], 9)), 3);
    // X_1 X_4 is even in the first group; Z-type strings read nothing.
    assert_eq!(label(string(&[1, 4], &[], 9)), 0);
    assert_eq!(label(string(&[], &[4, 1, 3], 9)), 0);
}

#[test]
fn parity_rows_cross_a_word_boundary() {
    let order: Vec<usize> = (62..96).collect();
    let rows = PartitionRowPolicy::parity(PauliAxis::Z, order, 2).rows::<2>(96, 2, 0);
    let (rows_x, rows_z) = rows.rows();
    assert_eq!(rows_x, [[0, 0], [0, 0]]);
    assert_eq!(rows_z, [[1 << 62 | 1 << 63, 0], [0, 1 << 0 | 1 << 1]]);
}

/// The contract on `parity`: `X` rows leave every `ZZ` bond local and make an `R_X` remote exactly on the rows' qubits.
#[test]
fn x_parity_rows_keep_every_zz_bond_local() {
    let bonds = square_lattice(3, 3);
    let circuit = kicked_ising::<1>(9, &bonds, 0.3, 0.7, 1);
    let rows =
        PartitionRowPolicy::parity_around(PauliAxis::X, 9, &bonds, &[4], 2).rows::<1>(9, 2, 0);
    let remote: Vec<bool> = circuit
        .deltas()
        .into_iter()
        .map(|shifts| shifts.unwrap().iter().any(|(x, z)| rows.partition_of(x, z) != 0))
        .collect();
    let kicks_remote: Vec<usize> = (0..9).filter(|&q| remote[q]).collect();
    assert_eq!(kicks_remote, [1, 3, 4, 5]);
    assert!(remote[9..].iter().all(|&remote| !remote), "a ZZ bond is remote");
}

#[test]
fn blocks_label_contiguous_runs_of_the_order() {
    let blocks = |axis| PartitionRowPolicy::blocks(axis, (0..8).collect()).rows::<1>(8, 2, 0);
    // Runs {0, 1}, {2, 3}, {4, 5}, {6, 7}: row 0 reads runs 1 and 3, row 1 runs 2 and 3.
    let labelled = vec![[0b1100_1100], [0b1111_0000]];
    let zero = vec![[0u64]; 2];
    assert_eq!(blocks(PauliAxis::Z), PartitionRows::from_rows(8, zero.clone(), labelled.clone()));
    assert_eq!(
        blocks(PauliAxis::Z),
        PartitionRows::cut(8, &[vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]])
    );
    assert_eq!(blocks(PauliAxis::X), PartitionRows::from_rows(8, labelled, zero));
    // Nine qubits over four runs: 3, 2, 2 and 2.
    let uneven = PartitionRowPolicy::blocks(PauliAxis::Z, (0..9).collect()).rows::<1>(9, 2, 0);
    assert_eq!(uneven, PartitionRows::cut(9, &[vec![0, 1, 2], vec![3, 4], vec![5, 6], vec![7, 8]]));
}

#[test]
#[should_panic(expected = "2 parity rows of 5 qubits need 10 qubits, the order holds 9")]
fn parity_rows_need_enough_qubits() {
    PartitionRowPolicy::parity(PauliAxis::X, CENTRE_ORDER.to_vec(), 5).rows::<1>(9, 2, 0);
}

#[test]
#[should_panic(expected = "at least one qubit per row")]
fn parity_rows_need_a_group() {
    PartitionRowPolicy::parity(PauliAxis::X, CENTRE_ORDER.to_vec(), 0).rows::<1>(9, 1, 0);
}

#[test]
#[should_panic(expected = "the row order repeats qubit 4")]
fn a_row_order_repeats_no_qubit() {
    PartitionRowPolicy::parity(PauliAxis::X, vec![4, 1, 4], 1).rows::<1>(9, 1, 0);
}

#[test]
fn for_run_gives_a_local_observable_parity_rows_off_the_entanglers_axis() {
    let bonds = square_lattice(3, 3);
    let circuit = kicked_ising::<1>(9, &bonds, 0.3, 0.7, 2);
    let site = sum_of(&[string::<1>(&[], &[4], 9)], 9);
    assert_eq!(
        PartitionRowPolicy::for_run(&site, &circuit),
        PartitionRowPolicy::parity(PauliAxis::X, CENTRE_ORDER.to_vec(), 2)
    );
    // A bond observable starts the order from both of its qubits.
    let bond = sum_of(&[string::<1>(&[], &[0, 1], 9)], 9);
    assert_eq!(
        PartitionRowPolicy::for_run(&bond, &circuit),
        PartitionRowPolicy::parity(PauliAxis::X, vec![0, 1, 3, 2, 4, 6, 5, 7, 8], 2)
    );
}

#[test]
fn for_run_gives_an_extensive_observable_block_rows_on_the_entanglers_axis() {
    let bonds = square_lattice(3, 3);
    let circuit = kicked_ising::<1>(9, &bonds, 0.3, 0.7, 2);
    let strings: Vec<PauliString<1>> = (0..9).map(|q| string(&[], &[q], 9)).collect();
    assert_eq!(
        PartitionRowPolicy::for_run(&sum_of(&strings, 9), &circuit),
        PartitionRowPolicy::blocks(PauliAxis::Z, (0..9).collect())
    );
}

/// The kicked-Ising circuit with `X` and `Z` exchanged: `R_XX` bonds and `R_Z` kicks.
fn exchanged_kicked_ising(bonds: &[(usize, usize)]) -> Circuit<1> {
    let mut circuit = Circuit::<1>::new(9);
    for q in 0..9 {
        circuit.push(PauliRotation::new(string::<1>(&[], &[q], 9), 0.3));
    }
    for &(a, b) in bonds {
        circuit.push(PauliRotation::new(string::<1>(&[a, b], &[], 9), 0.7));
    }
    circuit
}

#[test]
fn for_run_follows_the_entanglers_axis() {
    let bonds = square_lattice(3, 3);
    let circuit = exchanged_kicked_ising(&bonds);
    let site = sum_of(&[string::<1>(&[4], &[], 9)], 9);
    assert_eq!(
        PartitionRowPolicy::for_run(&site, &circuit),
        PartitionRowPolicy::parity(PauliAxis::Z, CENTRE_ORDER.to_vec(), 2)
    );
    let strings: Vec<PauliString<1>> = (0..9).map(|q| string(&[q], &[], 9)).collect();
    assert_eq!(
        PartitionRowPolicy::for_run(&sum_of(&strings, 9), &circuit),
        PartitionRowPolicy::blocks(PauliAxis::X, (0..9).collect())
    );
}

#[test]
fn for_run_falls_back_to_seeded_rows_under_mixed_entanglers() {
    let bonds = square_lattice(3, 3);
    let mut circuit = kicked_ising::<1>(9, &bonds, 0.3, 0.7, 1);
    circuit.push(PtmChannel::cnot(0, 1));
    let site = sum_of(&[string::<1>(&[], &[4], 9)], 9);
    assert_eq!(PartitionRowPolicy::for_run(&site, &circuit), PartitionRowPolicy::Seeded(None));
}

/// With no channel setting an x-bit, x-parity rows would keep every term in the observable's partition.
#[test]
fn for_run_gives_a_diagonal_circuit_block_rows() {
    let bonds = square_lattice(3, 3);
    let mut circuit = Circuit::<1>::new(9);
    for q in 0..9 {
        circuit.push(PauliRotation::new(string::<1>(&[], &[q], 9), 0.3));
    }
    for &(a, b) in &bonds {
        circuit.push(PauliRotation::new(string::<1>(&[], &[a, b], 9), 0.7));
    }
    let site = sum_of(&[string::<1>(&[4], &[], 9)], 9);
    assert_eq!(
        PartitionRowPolicy::for_run(&site, &circuit),
        PartitionRowPolicy::blocks(PauliAxis::Z, (0..9).collect())
    );
}
