//! The Clifford constructors of [`PtmChannel`], stored as signed permutations.

use super::{assert_distinct, PtmChannel, PtmKind, PtmStorage};
use crate::channel::prepared::LOCAL_DIM;
use crate::channel::{read_pauli, write_pauli};
use crate::pauli_string::PauliString;
use crate::phase::Phase;

impl PtmChannel {
    /// Hadamard: `X → Z, Z → X, Y → −Y`.
    pub fn hadamard(qubit: usize) -> Self {
        Self::clifford_1q(qubit, [0, 2, 1, 3], 0b1000, PtmKind::Hadamard)
    }

    /// Phase gate `S = diag(1, i)`: `X → Y, Z → Z, Y → −X`.
    pub fn phase_gate(qubit: usize) -> Self {
        Self::clifford_1q(qubit, [0, 3, 2, 1], 0b1000, PtmKind::PhaseGate)
    }

    /// Pauli-X gate: `X → X, Z → −Z, Y → −Y`.
    pub fn pauli_x(qubit: usize) -> Self {
        Self::clifford_1q(qubit, [0, 1, 2, 3], 0b1100, PtmKind::PauliX)
    }

    /// Pauli-Y gate: `X → −X, Z → −Z, Y → Y`.
    pub fn pauli_y(qubit: usize) -> Self {
        Self::clifford_1q(qubit, [0, 1, 2, 3], 0b0110, PtmKind::PauliY)
    }

    /// Pauli-Z gate: `X → −X, Z → Z, Y → −Y`.
    pub fn pauli_z(qubit: usize) -> Self {
        Self::clifford_1q(qubit, [0, 1, 2, 3], 0b1010, PtmKind::PauliZ)
    }

    /// CNOT: `X⊗I → X⊗X, I⊗X → I⊗X, Z⊗I → Z⊗I, I⊗Z → Z⊗Z`, control first.
    ///
    /// # Panics
    ///
    /// If `control == target`.
    pub fn cnot(control: usize, target: usize) -> Self {
        assert_distinct("cnot", control, target);
        Self::clifford_2q(
            [control, target],
            PtmKind::Cnot,
            [
                pack(1, 0, 1, 0),
                pack(0, 1, 0, 0),
                pack(0, 0, 1, 0),
                pack(0, 1, 0, 1),
            ],
        )
    }

    /// CZ: `X⊗I → X⊗Z, I⊗X → Z⊗X, Z⊗I → Z⊗I, I⊗Z → I⊗Z`.
    ///
    /// # Panics
    ///
    /// If `q0 == q1`.
    pub fn cz(q0: usize, q1: usize) -> Self {
        assert_distinct("cz", q0, q1);
        Self::clifford_2q(
            [q0, q1],
            PtmKind::Cz,
            [
                pack(1, 0, 0, 1),
                pack(0, 1, 0, 0),
                pack(0, 1, 1, 0),
                pack(0, 0, 0, 1),
            ],
        )
    }

    /// SWAP: `P⊗Q → Q⊗P`.
    ///
    /// # Panics
    ///
    /// If `q0 == q1`.
    pub fn swap(q0: usize, q1: usize) -> Self {
        assert_distinct("swap", q0, q1);
        Self::clifford_2q(
            [q0, q1],
            PtmKind::Swap,
            [
                pack(0, 0, 1, 0),
                pack(0, 0, 0, 1),
                pack(1, 0, 0, 0),
                pack(0, 1, 0, 0),
            ],
        )
    }

    fn clifford_1q(qubit: usize, image: [u8; 4], negated: u16, kind: PtmKind) -> Self {
        let mut padded = [0u8; LOCAL_DIM];
        padded[..4].copy_from_slice(&image);
        Self::new(
            &[qubit],
            PtmStorage::SignedPermutation {
                image: padded,
                negated,
            },
            kind,
        )
    }

    /// The signed permutation from the images of `X₀, Z₀, X₁, Z₁`, multiplied out per input.
    fn clifford_2q(qubits: [usize; 2], kind: PtmKind, generator_images: [u8; 4]) -> Self {
        let mut image = [0u8; LOCAL_DIM];
        let mut negated = 0u16;
        for (index, slot) in image.iter_mut().enumerate() {
            // The image of `X^{x0} Z^{z0} X^{x1} Z^{z1}`; each input `Y = i · X · Z` adds an `i`.
            let mut product = PauliString::<1>::identity();
            let mut phase = Phase::ONE;
            for (generator, &generator_image) in generator_images.iter().enumerate() {
                if (index >> generator) & 1 == 1 {
                    phase += product.mul_assign(&unpack(generator_image));
                }
            }
            let y_count =
                ((index & 1) & ((index >> 1) & 1)) + (((index >> 2) & 1) & ((index >> 3) & 1));
            phase += Phase::new(y_count as u8);
            debug_assert!(
                phase == Phase::ONE || phase == Phase::MINUS_ONE,
                "a Clifford maps a Hermitian Pauli to ± a Hermitian Pauli"
            );
            *slot = (read_pauli(&product.x, &product.z, 0, 0)
                | (read_pauli(&product.x, &product.z, 0, 1) << 2)) as u8;
            if phase == Phase::MINUS_ONE {
                negated |= 1 << index;
            }
        }
        Self::new(
            &qubits,
            PtmStorage::SignedPermutation { image, negated },
            kind,
        )
    }
}

/// Pack `(x0, z0, x1, z1)` into a two-qubit Pauli index.
const fn pack(x0: u8, z0: u8, x1: u8, z1: u8) -> u8 {
    x0 | (z0 << 1) | (x1 << 2) | (z1 << 3)
}

/// A two-qubit Pauli index as a Pauli string on qubits 0 and 1.
fn unpack(packed: u8) -> PauliString<1> {
    let mut string = PauliString::<1>::identity();
    write_pauli(&mut string.x, &mut string.z, 0, 0, 1, (packed & 3) as usize);
    write_pauli(
        &mut string.x,
        &mut string.z,
        0,
        1,
        2,
        ((packed >> 2) & 3) as usize,
    );
    string
}
