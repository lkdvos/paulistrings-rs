//! The noise channels: Pauli-diagonal rescalings and amplitude damping.

use super::{qubit_loc, read_pauli, set_bit, support_mask, Channel, OutputBuffer};
use num_complex::Complex64;

/// Multiply the coefficient by `scale` when the support qubit's Pauli index is `affected`.
#[inline]
fn rescale_on_support<const W: usize>(
    support: u32,
    scale: f64,
    affected: impl FnOnce(usize) -> bool,
    input_x: &[u64; W],
    input_z: &[u64; W],
    coeff: Complex64,
    out: &mut OutputBuffer<'_, W>,
) {
    let q = support as usize;
    debug_assert!(q < 64 * W);
    let (word, bit, _mask) = qubit_loc(q);
    let idx = read_pauli(input_x, input_z, word, bit);
    let s = if affected(idx) { scale } else { 1.0 };
    out.push(*input_x, *input_z, coeff * s);
}

/// Single-qubit depolarizing noise with error probability `p`: every non-identity Pauli on the support is scaled by `1 - 4p/3`.
pub struct Depolarizing {
    /// The single qubit this channel acts on.
    pub support: [u32; 1],
    /// Error probability `p ∈ [0, 1]`.
    pub p: f64,
}

impl<const W: usize> Channel<W> for Depolarizing {
    #[inline]
    fn max_fanout(&self) -> usize {
        1
    }

    #[inline]
    fn support(&self) -> [u64; W] {
        support_mask(&self.support)
    }

    #[inline]
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        rescale_on_support(
            self.support[0],
            1.0 - 4.0 * self.p / 3.0,
            |idx| idx != 0,
            input_x,
            input_z,
            coeff,
            out,
        );
    }
}

/// Single-qubit dephasing noise with error probability `p`: `X` and `Y` on the support are scaled by `1 - 2p`.
pub struct Dephasing {
    /// The single qubit this channel acts on.
    pub support: [u32; 1],
    /// Error probability `p ∈ [0, 1]`.
    pub p: f64,
}

impl<const W: usize> Channel<W> for Dephasing {
    #[inline]
    fn max_fanout(&self) -> usize {
        1
    }

    #[inline]
    fn support(&self) -> [u64; W] {
        support_mask(&self.support)
    }

    #[inline]
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        rescale_on_support(
            self.support[0],
            1.0 - 2.0 * self.p,
            |idx| idx & 1 == 1,
            input_x,
            input_z,
            coeff,
            out,
        );
    }
}

/// Single-qubit Pauli channel `E(ρ) = (1-px-py-pz)ρ + px·XρX + py·YρY + pz·ZρZ`.
///
/// It scales `X` by `1 - 2(py + pz)`, `Y` by `1 - 2(px + pz)` and `Z` by `1 - 2(px + py)`, so `(p/3, p/3, p/3)` is [`Depolarizing`] and `(0, 0, p)` is [`Dephasing`].
pub struct PauliChannel {
    /// The single qubit this channel acts on.
    pub support: [u32; 1],
    /// Probability of an `X` error.
    pub px: f64,
    /// Probability of a `Y` error.
    pub py: f64,
    /// Probability of a `Z` error.
    pub pz: f64,
}

impl<const W: usize> Channel<W> for PauliChannel {
    #[inline]
    fn max_fanout(&self) -> usize {
        1
    }

    #[inline]
    fn support(&self) -> [u64; W] {
        support_mask(&self.support)
    }

    #[inline]
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        let q = self.support[0] as usize;
        debug_assert!(q < 64 * W);
        let (word, bit, _mask) = qubit_loc(q);
        let idx = read_pauli(input_x, input_z, word, bit);
        let s = match idx {
            0 => 1.0,
            1 => 1.0 - 2.0 * (self.py + self.pz),
            2 => 1.0 - 2.0 * (self.px + self.py),
            3 => 1.0 - 2.0 * (self.px + self.pz),
            _ => unreachable!(),
        };
        out.push(*input_x, *input_z, coeff * s);
    }
}

/// Two-qubit depolarizing noise `E(ρ) = (1-p)ρ + (p/15)·Σ_k P_k ρ P_k` over the 15 non-identity two-qubit Paulis.
///
/// Every Pauli that is non-identity on the support is scaled by `1 - 16p/15`.
pub struct Depolarizing2Q {
    /// The two qubits this channel acts on, which must differ.
    pub support: [u32; 2],
    /// Error probability `p ∈ [0, 1]`.
    pub p: f64,
}

impl<const W: usize> Channel<W> for Depolarizing2Q {
    #[inline]
    fn max_fanout(&self) -> usize {
        1
    }

    #[inline]
    fn support(&self) -> [u64; W] {
        support_mask(&self.support)
    }

    #[inline]
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        debug_assert!(
            self.support[0] != self.support[1],
            "Depolarizing2Q support qubits must differ (both are {})",
            self.support[0]
        );
        let mut touches_pair = false;
        for &q in &self.support {
            let q = q as usize;
            debug_assert!(q < 64 * W);
            let (word, bit, _mask) = qubit_loc(q);
            touches_pair |= read_pauli(input_x, input_z, word, bit) != 0;
        }
        let s = if touches_pair {
            1.0 - 16.0 * self.p / 15.0
        } else {
            1.0
        };
        out.push(*input_x, *input_z, coeff * s);
    }
}

/// Single-qubit amplitude damping with Kraus operators `K_0 = |0⟩⟨0| + √(1-γ)|1⟩⟨1|`, `K_1 = √γ |0⟩⟨1|`.
///
/// Not self-adjoint: `apply` is the Schrödinger map (`I → I + γZ`, `Z → (1-γ)Z`) and `apply_adjoint` its Heisenberg dual (`I → I`, `Z → (1-γ)Z + γI`); both scale `X` and `Y` by `√(1-γ)`.
pub struct AmplitudeDamping {
    /// The single qubit this channel acts on.
    pub support: [u32; 1],
    /// Damping parameter `γ ∈ [0, 1]`.
    pub gamma: f64,
}

impl<const W: usize> Channel<W> for AmplitudeDamping {
    #[inline]
    fn max_fanout(&self) -> usize {
        2
    }

    #[inline]
    fn support(&self) -> [u64; W] {
        support_mask(&self.support)
    }

    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        let q = self.support[0] as usize;
        debug_assert!(q < 64 * W);
        let (word, bit, mask) = qubit_loc(q);
        let idx = read_pauli(input_x, input_z, word, bit);
        match idx {
            0 => {
                out.push(*input_x, *input_z, coeff);
                let mut nz = *input_z;
                set_bit(&mut nz, word, mask, true);
                out.push(*input_x, nz, coeff * self.gamma);
            }
            1 | 3 => {
                let scale = (1.0 - self.gamma).sqrt();
                out.push(*input_x, *input_z, coeff * scale);
            }
            2 => {
                out.push(*input_x, *input_z, coeff * (1.0 - self.gamma));
            }
            _ => unreachable!(),
        }
    }

    fn apply_adjoint(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        let q = self.support[0] as usize;
        debug_assert!(q < 64 * W);
        let (word, bit, mask) = qubit_loc(q);
        let idx = read_pauli(input_x, input_z, word, bit);
        match idx {
            0 => {
                out.push(*input_x, *input_z, coeff);
            }
            1 | 3 => {
                let scale = (1.0 - self.gamma).sqrt();
                out.push(*input_x, *input_z, coeff * scale);
            }
            2 => {
                out.push(*input_x, *input_z, coeff * (1.0 - self.gamma));
                let mut nz = *input_z;
                set_bit(&mut nz, word, mask, false);
                out.push(*input_x, nz, coeff * self.gamma);
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests;
