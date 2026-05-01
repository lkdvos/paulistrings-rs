//! Noise channels: Depolarizing, Dephasing, AmplitudeDamping. See §6.

#![allow(unused)]

use super::{Channel, OutputBuffer};
use num_complex::Complex64;

/// Single-qubit depolarizing noise with error probability `p`.
///
/// In the Heisenberg picture this is just a coefficient rescaling: the
/// identity is preserved, every non-identity Pauli on the support is
/// multiplied by `1 - 4p/3`.
pub struct Depolarizing {
    pub support: [u32; 1],
    pub p: f64,
}

impl Channel<1> for Depolarizing {
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
        todo!("§6: scale coeff by 1 if input bit at support is I, else 1 - 4p/3")
    }
}

/// Single-qubit dephasing noise with error probability `p`. Coefficient
/// rescaling: factor is `1 - 2p` if there is an X-component on the support.
pub struct Dephasing {
    pub support: [u32; 1],
    pub p: f64,
}

impl Channel<1> for Dephasing {
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
        todo!("§6: scale coeff by 1 - 2p iff support qubit has X-component (X or Y)")
    }
}

/// Single-qubit amplitude damping with parameter `gamma`.
///
/// The only noise in the built-in set with genuine fan-out > 1: maps
/// `{I, X, Y, Z}` on the support to short Pauli combinations.
pub struct AmplitudeDamping {
    pub support: [u32; 1],
    pub gamma: f64,
}

impl Channel<1> for AmplitudeDamping {
    fn max_fanout(&self) -> usize {
        2
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
        todo!("§6: dispatch on support Pauli; for Z emit (1-gamma)·Z + gamma·I, etc.")
    }
}
