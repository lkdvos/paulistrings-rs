//! General unitaries pre-decomposed into Pauli expansions. See §6.

#![allow(unused)]

use super::{Channel, OutputBuffer};
use num_complex::Complex64;

/// Generic 1-qubit unitary, stored as the Pauli expansion of its
/// Heisenberg-picture action on `{I, X, Y, Z}` at the support qubit.
///
/// `MAX_FANOUT = 4` since each input Pauli on the support can map to a sum
/// over all four basis Paulis.
pub struct GeneralUnitary1Q {
    pub support: [u32; 1],
    /// 4x4 table: rows indexed by input Pauli (II, X, Z, Y in symplectic
    /// order), columns by output Pauli, entries are complex coefficients.
    pub table: [[Complex64; 4]; 4],
}

impl Channel<1> for GeneralUnitary1Q {
    fn max_fanout(&self) -> usize {
        4
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
        todo!("§6: index `table` by input bits at support; emit nonzero output Paulis")
    }
}

/// Generic 2-qubit unitary, stored as a 16x16 Pauli-expansion table.
pub struct GeneralUnitary2Q {
    pub support: [u32; 2],
    pub table: [[Complex64; 16]; 16],
}

impl Channel<1> for GeneralUnitary2Q {
    fn max_fanout(&self) -> usize {
        16
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
        todo!("§6: index `table` by 4 input bits at support; emit nonzero output Paulis")
    }
}
