//! One- and two-qubit Clifford gates as precomputed conjugation tables.

use super::{qubit_loc, read_pauli, support_mask, write_pauli, Channel, OutputBuffer};
use crate::phase::Phase;
use num_complex::Complex64;

/// Single-qubit Clifford gate stored as a 4-entry conjugation table.
///
/// Paulis are indexed `x | (z << 1)`, so `I = 0, X = 1, Z = 2, Y = 3`; `G · P_i · G† = phase[i] · P_{out_pauli[i]}`.
#[derive(Clone, Copy, Debug)]
pub struct Clifford1Q {
    /// The qubit this gate acts on.
    pub support: [u32; 1],
    /// Output Pauli index per input Pauli index; `out_pauli[0]` is `0`.
    pub out_pauli: [u8; 4],
    /// Phase per input Pauli index; `phase[0]` is [`Phase::ONE`].
    pub phase: [Phase; 4],
}

impl Clifford1Q {
    /// Hadamard. Conjugation: `I → I, X → Z, Z → X, Y → −Y`.
    pub fn h(qubit: u32) -> Self {
        Self {
            support: [qubit],
            out_pauli: [0, 2, 1, 3],
            phase: [Phase::ONE, Phase::ONE, Phase::ONE, Phase::MINUS_ONE],
        }
    }

    /// Phase gate `S = diag(1, i)`. Conjugation: `I → I, X → Y, Z → Z, Y → −X`.
    pub fn s(qubit: u32) -> Self {
        Self {
            support: [qubit],
            out_pauli: [0, 3, 2, 1],
            phase: [Phase::ONE, Phase::ONE, Phase::ONE, Phase::MINUS_ONE],
        }
    }

    /// Pauli-X gate. Conjugation: `I → I, X → X, Z → −Z, Y → −Y`.
    pub fn x(qubit: u32) -> Self {
        Self {
            support: [qubit],
            out_pauli: [0, 1, 2, 3],
            phase: [Phase::ONE, Phase::ONE, Phase::MINUS_ONE, Phase::MINUS_ONE],
        }
    }

    /// Pauli-Y gate. Conjugation: `I → I, X → −X, Z → −Z, Y → Y`.
    pub fn y(qubit: u32) -> Self {
        Self {
            support: [qubit],
            out_pauli: [0, 1, 2, 3],
            phase: [Phase::ONE, Phase::MINUS_ONE, Phase::MINUS_ONE, Phase::ONE],
        }
    }

    /// Pauli-Z gate. Conjugation: `I → I, X → −X, Z → Z, Y → −Y`.
    pub fn z(qubit: u32) -> Self {
        Self {
            support: [qubit],
            out_pauli: [0, 1, 2, 3],
            phase: [Phase::ONE, Phase::MINUS_ONE, Phase::ONE, Phase::MINUS_ONE],
        }
    }

    /// The adjoint gate `G†`: if `G P_a G† = c_a · P_{f(a)}` then `G† P_{f(a)} G = c_a* · P_a`.
    pub fn adjoint(&self) -> Self {
        let mut out_pauli = [0u8; 4];
        let mut phase = [Phase::ONE; 4];
        for input_idx in 0u8..4 {
            let f_a = self.out_pauli[input_idx as usize] as usize;
            let c_a = self.phase[input_idx as usize];
            out_pauli[f_a] = input_idx;
            phase[f_a] = Phase::new((4 - c_a.exponent()) & 3);
        }
        Self {
            support: self.support,
            out_pauli,
            phase,
        }
    }
}

impl Clifford1Q {
    /// Body of `apply` and `apply_adjoint`, which differ only in the table.
    #[inline]
    fn apply_table<const W: usize>(
        &self,
        out_pauli: &[u8; 4],
        phase: &[Phase; 4],
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        let q = self.support[0] as usize;
        debug_assert!(q < 64 * W);
        let (word, bit, mask) = qubit_loc(q);
        let idx = read_pauli(input_x, input_z, word, bit);
        let op = out_pauli[idx] as usize;
        let mut nx = *input_x;
        let mut nz = *input_z;
        write_pauli(&mut nx, &mut nz, word, bit, mask, op);
        out.push(nx, nz, phase[idx].apply(coeff));
    }
}

impl<const W: usize> Channel<W> for Clifford1Q {
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
        self.apply_table(&self.out_pauli, &self.phase, input_x, input_z, coeff, out);
    }

    #[inline]
    fn apply_adjoint(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        let adj = self.adjoint();
        self.apply_table(&adj.out_pauli, &adj.phase, input_x, input_z, coeff, out);
    }
}

/// Two-qubit Clifford gate stored as a 16-entry conjugation table.
///
/// Paulis are indexed `x0 | (z0 << 1) | (x1 << 2) | (z1 << 3)`, qubit 0 being `support[0]`; `G · P_i · G† = phase[i] · P_{out_pauli[i]}`.
#[derive(Clone, Copy, Debug)]
pub struct Clifford2Q {
    /// The two qubits this gate acts on, control first for CNOT.
    pub support: [u32; 2],
    /// Output Pauli index per input Pauli index; `out_pauli[0]` is `0`.
    pub out_pauli: [u8; 16],
    /// Phase per input Pauli index; `phase[0]` is [`Phase::ONE`].
    pub phase: [Phase; 16],
}

impl Clifford2Q {
    /// CNOT with `control` and `target`: `X⊗I → X⊗X, I⊗X → I⊗X, Z⊗I → Z⊗I, I⊗Z → Z⊗Z`.
    pub fn cnot(control: u32, target: u32) -> Self {
        Self::from_2q_generators(
            [control, target],
            (pack4(1, 0, 1, 0), Phase::ONE),
            (pack4(0, 1, 0, 0), Phase::ONE),
            (pack4(0, 0, 1, 0), Phase::ONE),
            (pack4(0, 1, 0, 1), Phase::ONE),
        )
    }

    /// CZ on `q0` and `q1`: `X⊗I → X⊗Z, I⊗X → Z⊗X, Z⊗I → Z⊗I, I⊗Z → I⊗Z`.
    pub fn cz(q0: u32, q1: u32) -> Self {
        Self::from_2q_generators(
            [q0, q1],
            (pack4(1, 0, 0, 1), Phase::ONE),
            (pack4(0, 1, 0, 0), Phase::ONE),
            (pack4(0, 1, 1, 0), Phase::ONE),
            (pack4(0, 0, 0, 1), Phase::ONE),
        )
    }

    /// SWAP on `q0` and `q1`: `P⊗Q → Q⊗P`.
    pub fn swap(q0: u32, q1: u32) -> Self {
        Self::from_2q_generators(
            [q0, q1],
            (pack4(0, 0, 1, 0), Phase::ONE),
            (pack4(0, 0, 0, 1), Phase::ONE),
            (pack4(1, 0, 0, 0), Phase::ONE),
            (pack4(0, 1, 0, 0), Phase::ONE),
        )
    }

    /// The 16-entry table from the images of `X₀, Z₀, X₁, Z₁`, multiplied out per input.
    fn from_2q_generators(
        support: [u32; 2],
        x0_image: (u8, Phase),
        z0_image: (u8, Phase),
        x1_image: (u8, Phase),
        z1_image: (u8, Phase),
    ) -> Self {
        let gens = [x0_image, z0_image, x1_image, z1_image];
        let mut out_pauli = [0u8; 16];
        let mut phase = [Phase::ONE; 16];
        for idx in 0..16usize {
            let bits = [
                (idx & 1) as u8,
                ((idx >> 1) & 1) as u8,
                ((idx >> 2) & 1) as u8,
                ((idx >> 3) & 1) as u8,
            ];

            // The product is the image of `X^{x0} Z^{z0} X^{x1} Z^{z1}`; each input `Y = i · X · Z` adds an `i` below.
            let mut acc_x = [0u64; 1];
            let mut acc_z = [0u64; 1];
            let mut acc_phase = Phase::ONE;
            for (b, (img, ph)) in bits.iter().zip(gens.iter()) {
                if *b == 1 {
                    let (gx, gz) = unpack4_to_word(*img);
                    let mut acc = crate::pauli_string::PauliString::<1> { x: acc_x, z: acc_z };
                    let other = crate::pauli_string::PauliString::<1> { x: gx, z: gz };
                    let mul_phase = acc.mul_assign(&other);
                    acc_x = acc.x;
                    acc_z = acc.z;
                    acc_phase = acc_phase + mul_phase + *ph;
                }
            }
            let y_count = (bits[0] & bits[1]) + (bits[2] & bits[3]);
            acc_phase += Phase::new(y_count);
            out_pauli[idx] = pack4_from_word(acc_x, acc_z);
            phase[idx] = acc_phase;
        }
        Self {
            support,
            out_pauli,
            phase,
        }
    }
}

/// Pack `(x0, z0, x1, z1)` into a two-qubit Pauli index.
const fn pack4(x0: u8, z0: u8, x1: u8, z1: u8) -> u8 {
    (x0 & 1) | ((z0 & 1) << 1) | ((x1 & 1) << 2) | ((z1 & 1) << 3)
}

/// A two-qubit Pauli index as one-word `(x, z)` on qubits 0 and 1.
fn unpack4_to_word(packed: u8) -> ([u64; 1], [u64; 1]) {
    let mut x = [0u64; 1];
    let mut z = [0u64; 1];
    write_pauli(&mut x, &mut z, 0, 0, 1, (packed & 3) as usize);
    write_pauli(&mut x, &mut z, 0, 1, 2, ((packed >> 2) & 3) as usize);
    (x, z)
}

/// Inverse of [`unpack4_to_word`].
fn pack4_from_word(x: [u64; 1], z: [u64; 1]) -> u8 {
    let p0 = read_pauli(&x, &z, 0, 0) as u8;
    let p1 = read_pauli(&x, &z, 0, 1) as u8;
    p0 | (p1 << 2)
}

impl<const W: usize> Channel<W> for Clifford2Q {
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
        let q0 = self.support[0] as usize;
        let q1 = self.support[1] as usize;
        debug_assert!(q0 < 64 * W);
        debug_assert!(q1 < 64 * W);
        debug_assert!(q0 != q1);

        let (w0, b0, m0) = qubit_loc(q0);
        let (w1, b1, m1) = qubit_loc(q1);
        let idx =
            read_pauli(input_x, input_z, w0, b0) | (read_pauli(input_x, input_z, w1, b1) << 2);

        let op = self.out_pauli[idx] as usize;

        let mut nx = *input_x;
        let mut nz = *input_z;
        write_pauli(&mut nx, &mut nz, w0, b0, m0, op & 3);
        write_pauli(&mut nx, &mut nz, w1, b1, m1, (op >> 2) & 3);

        out.push(nx, nz, self.phase[idx].apply(coeff));
    }
}

#[cfg(test)]
mod tests;
