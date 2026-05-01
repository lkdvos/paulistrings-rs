//! Pauli rotation `exp(-i * theta * P / 2)`. See §6.

#![allow(unused)]

use super::{Channel, OutputBuffer};
use num_complex::Complex64;

/// A rotation `U = exp(-i * theta * P / 2)`.
///
/// In the Heisenberg picture, conjugation by `U` either leaves the input
/// invariant (if `[input, P] = 0`) or maps it to `cos(theta) * input +
/// sin(theta) * i * input * P`. Hence `MAX_FANOUT = 2`.
pub struct PauliRotation<const W: usize> {
    pub support: Vec<u32>,
    /// X-part of the generator P, restricted to `support`.
    pub gen_x: [u64; W],
    /// Z-part of the generator P, restricted to `support`.
    pub gen_z: [u64; W],
    pub theta: f64,
}

impl<const W: usize> Channel<W> for PauliRotation<W> {
    fn max_fanout(&self) -> usize {
        2
    }

    fn support(&self) -> &[u32] {
        &self.support
    }

    fn apply(
        &self,
        _input_x: &[u64; W],
        _input_z: &[u64; W],
        _coeff: Complex64,
        _out: &mut OutputBuffer<'_, W>,
    ) {
        todo!("§6: branch on commutation parity; emit cos(theta)·input and sin(theta)·i·input·P")
    }
}
