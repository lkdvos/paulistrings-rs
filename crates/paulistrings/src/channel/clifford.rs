//! Clifford gates (table-driven, branchless). See §6.

#![allow(unused)]

use super::{Channel, OutputBuffer};
use num_complex::Complex64;

/// Single-qubit Clifford gate, parameterized by its symplectic 2x2 matrix.
pub struct Clifford1Q {
    /// Single qubit this gate acts on. Held as a `[u32; 1]` so `support()`
    /// can return a slice without allocation.
    pub support: [u32; 1],
    /// Symplectic image: `[xx, xz, zx, zz]` over GF(2).
    pub symplectic: [u8; 4],
    /// Sign byte for the Pauli image (one of `1, i, -1, -i` for each of X, Z).
    pub phase: [u8; 2],
}

impl Channel<1> for Clifford1Q {
    fn max_fanout(&self) -> usize {
        1
    }

    fn support(&self) -> &[u32] {
        &self.support
    }

    fn apply(
        &self,
        _input_x: &[u64; 1],
        _input_z: &[u64; 1],
        _coeff: Complex64,
        _out: &mut OutputBuffer<'_, 1>,
    ) {
        todo!("§6: symplectic-matrix lookup on the support qubit, branchless")
    }
}

/// Two-qubit Clifford gate, parameterized by its 4x4 symplectic matrix.
pub struct Clifford2Q {
    pub support: [u32; 2],
    pub symplectic: [u8; 16],
    pub phase: [u8; 4],
}

impl Channel<1> for Clifford2Q {
    fn max_fanout(&self) -> usize {
        1
    }

    fn support(&self) -> &[u32] {
        &self.support
    }

    fn apply(
        &self,
        _input_x: &[u64; 1],
        _input_z: &[u64; 1],
        _coeff: Complex64,
        _out: &mut OutputBuffer<'_, 1>,
    ) {
        todo!("§6: 4x4 symplectic-matrix lookup on the two support qubits")
    }
}
