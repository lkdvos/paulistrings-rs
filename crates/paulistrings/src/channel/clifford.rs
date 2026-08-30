//! Clifford gates (table-driven, branchless). See §6.
//!
//! A Clifford `G` conjugates each single-qubit Pauli `P` to `± P'` for some
//! Pauli `P'`. We precompute the full lookup table at construction so
//! `apply` is a single indexed read on the support qubit(s) — no runtime
//! Pauli multiplication.
//!
//! Encoding: a single-qubit Pauli is indexed by `(x_bit | (z_bit << 1))` —
//! `I = 0, X = 1, Z = 2, Y = 3`. The output Pauli uses the same packing.

use super::{support_mask, Channel, OutputBuffer};
use crate::phase::Phase;
use num_complex::Complex64;

/// Single-qubit Clifford gate stored as a 4-entry conjugation table.
///
/// `out_pauli[i]` and `phase[i]` give the result of `G · P_i · G†` for the
/// four input Paulis (indexed as above): the new packed Pauli on the
/// support qubit and the `i^k` phase to fold into the coefficient.
///
/// Generic over `W` so the same gate value can act on any Pauli width.
#[derive(Clone, Copy, Debug)]
pub struct Clifford1Q {
    /// Single qubit this gate acts on. Held as `[u32; 1]` so `support()`
    /// returns a slice without allocation.
    pub support: [u32; 1],
    /// Output Pauli bits per input Pauli. Same packing as the index:
    /// `(out_x | (out_z << 1))`. `out_pauli[0]` is always `0` (`I → I`).
    pub out_pauli: [u8; 4],
    /// Phase factor (`i^k`) per input Pauli. `phase[0]` is always
    /// `Phase::ONE`.
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

    /// Phase gate `S = diag(1, i)`. Conjugation:
    /// `I → I, X → Y, Z → Z, Y → −X`.
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

    /// Conjugation table for `G†`. Inverts the Pauli permutation and conjugates
    /// the per-input phases: if `G P_a G† = c_a · P_{f(a)}` then
    /// `G† P_{f(a)} G = c_a* · P_a`.
    ///
    /// Self-inverse 1Q Cliffords (H, X, Y, Z) round-trip to themselves; `S`
    /// returns `S†` (a distinct gate).
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
    /// Shared body of `apply` and `apply_adjoint`. The two paths differ
    /// only in which lookup table to use; pulling the lookup out of
    /// `Channel::apply` keeps the bit-fiddling unduplicated.
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
        let word = q / 64;
        let bit = q % 64;
        let mask = 1u64 << bit;
        let x_bit = ((input_x[word] >> bit) & 1) as u8;
        let z_bit = ((input_z[word] >> bit) & 1) as u8;
        let idx = (x_bit | (z_bit << 1)) as usize;
        let op = out_pauli[idx];
        let ox = (op & 1) as u64;
        let oz = ((op >> 1) & 1) as u64;
        let mut nx = *input_x;
        let mut nz = *input_z;
        nx[word] = (nx[word] & !mask) | (ox << bit);
        nz[word] = (nz[word] & !mask) | (oz << bit);
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
/// Index encoding: low 2 bits select the input Pauli on `support[0]`,
/// high 2 bits select it on `support[1]`. Each `out_pauli[i]` packs four
/// bits `(ox0 | (oz0 << 1) | (ox1 << 2) | (oz1 << 3))`.
#[derive(Clone, Copy, Debug)]
pub struct Clifford2Q {
    /// The two qubits this gate acts on. Convention: `support[0]` is the
    /// "first" qubit (e.g. CNOT control), `support[1]` the second.
    pub support: [u32; 2],
    /// Output Pauli bits per input. 16 entries indexed by
    /// `(x0 | (z0 << 1) | (x1 << 2) | (z1 << 3))`; each entry packs the
    /// output bits in the same layout. `out_pauli[0]` is always `0`.
    pub out_pauli: [u8; 16],
    /// Phase factor (`i^k`) per input. `phase[0]` is always [`Phase::ONE`].
    pub phase: [Phase; 16],
}

impl Clifford2Q {
    /// CNOT with `control` and `target`. Conjugation generators:
    /// `X⊗I → X⊗X, I⊗X → I⊗X, Z⊗I → Z⊗I, I⊗Z → Z⊗Z` (all phase `+1`).
    /// The full 16-entry table follows by linearity over Pauli products.
    pub fn cnot(control: u32, target: u32) -> Self {
        Self::from_2q_generators(
            [control, target],
            // X⊗I → X⊗X
            (pack4(1, 0, 1, 0), Phase::ONE),
            // Z⊗I → Z⊗I
            (pack4(0, 1, 0, 0), Phase::ONE),
            // I⊗X → I⊗X
            (pack4(0, 0, 1, 0), Phase::ONE),
            // I⊗Z → Z⊗Z
            (pack4(0, 1, 0, 1), Phase::ONE),
        )
    }

    /// CZ on `q0` and `q1`. Conjugation generators:
    /// `X⊗I → X⊗Z, I⊗X → Z⊗X, Z⊗I → Z⊗I, I⊗Z → I⊗Z` (all phase `+1`).
    pub fn cz(q0: u32, q1: u32) -> Self {
        Self::from_2q_generators(
            [q0, q1],
            // X⊗I → X⊗Z
            (pack4(1, 0, 0, 1), Phase::ONE),
            // Z⊗I → Z⊗I
            (pack4(0, 1, 0, 0), Phase::ONE),
            // I⊗X → Z⊗X
            (pack4(0, 1, 1, 0), Phase::ONE),
            // I⊗Z → I⊗Z
            (pack4(0, 0, 0, 1), Phase::ONE),
        )
    }

    /// SWAP on `q0` and `q1`. Conjugation: `(P ⊗ Q) → (Q ⊗ P)` for all
    /// `P, Q` (all phase `+1`).
    pub fn swap(q0: u32, q1: u32) -> Self {
        Self::from_2q_generators(
            [q0, q1],
            // X⊗I → I⊗X
            (pack4(0, 0, 1, 0), Phase::ONE),
            // Z⊗I → I⊗Z
            (pack4(0, 0, 0, 1), Phase::ONE),
            // I⊗X → X⊗I
            (pack4(1, 0, 0, 0), Phase::ONE),
            // I⊗Z → Z⊗I
            (pack4(0, 1, 0, 0), Phase::ONE),
        )
    }

    /// Build a 2Q Clifford table from the four single-Pauli generators
    /// `(X₀, Z₀, X₁, Z₁)`. For each of the 16 inputs we multiply the
    /// corresponding generator outputs together (using `PauliString`
    /// multiplication on a single word) and accumulate the `i^k` phase.
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
            // Decompose the input across the four generators.
            let bits = [
                (idx & 1) as u8,        // x0
                ((idx >> 1) & 1) as u8, // z0
                ((idx >> 2) & 1) as u8, // x1
                ((idx >> 3) & 1) as u8, // z1
            ];

            // Multiply the four generator images in X₀ Z₀ X₁ Z₁ order.
            // The result is the image of `X^{x0} Z^{z0} X^{x1} Z^{z1}`,
            // which is *not* the same as the Hermitian input Pauli when
            // any qubit holds Y: `Y = i · X · Z`, so each Y in the input
            // contributes an extra `i` factor that we must add to the
            // accumulated phase below.
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
            // Add `i` for each Y in the input (qubits where both x and z
            // bits are set). The output bits encode their Hermitian Pauli
            // directly — no symmetric cancellation factor on the output
            // side.
            let y_count = (bits[0] & bits[1]) + (bits[2] & bits[3]);
            acc_phase += Phase::new(y_count);
            out_pauli[idx] = pack4_from_word(acc_x, acc_z, support);
            phase[idx] = acc_phase;
        }
        Self {
            support,
            out_pauli,
            phase,
        }
    }
}

/// Pack `(x0, z0, x1, z1)` bits into the 4-bit `out_pauli` encoding.
const fn pack4(x0: u8, z0: u8, x1: u8, z1: u8) -> u8 {
    (x0 & 1) | ((z0 & 1) << 1) | ((x1 & 1) << 2) | ((z1 & 1) << 3)
}

/// Convert a packed 4-bit Pauli on the two support qubits back into a
/// `PauliString<1>`-style `(x, z)` word pair, with the support qubits at
/// positions 0 and 1 of the word. Used only inside table construction.
fn unpack4_to_word(packed: u8) -> ([u64; 1], [u64; 1]) {
    let x0 = (packed & 1) as u64;
    let z0 = ((packed >> 1) & 1) as u64;
    let x1 = ((packed >> 2) & 1) as u64;
    let z1 = ((packed >> 3) & 1) as u64;
    let x = x0 | (x1 << 1);
    let z = z0 | (z1 << 1);
    ([x], [z])
}

/// Inverse of `unpack4_to_word`: reads bits 0 and 1 of a single-word
/// `(x, z)` pair and packs them back into the 4-bit encoding. The
/// `support` argument is unused at runtime — kept for future-proofing
/// once the helper moves out of test-only construction.
fn pack4_from_word(x: [u64; 1], z: [u64; 1], _support: [u32; 2]) -> u8 {
    let x0 = (x[0] & 1) as u8;
    let z0 = (z[0] & 1) as u8;
    let x1 = ((x[0] >> 1) & 1) as u8;
    let z1 = ((z[0] >> 1) & 1) as u8;
    pack4(x0, z0, x1, z1)
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

        let w0 = q0 / 64;
        let b0 = q0 % 64;
        let w1 = q1 / 64;
        let b1 = q1 % 64;
        let m0 = 1u64 << b0;
        let m1 = 1u64 << b1;

        let x0 = ((input_x[w0] >> b0) & 1) as u8;
        let z0 = ((input_z[w0] >> b0) & 1) as u8;
        let x1 = ((input_x[w1] >> b1) & 1) as u8;
        let z1 = ((input_z[w1] >> b1) & 1) as u8;
        let idx = (x0 | (z0 << 1) | (x1 << 2) | (z1 << 3)) as usize;

        let op = self.out_pauli[idx];
        let ox0 = (op & 1) as u64;
        let oz0 = ((op >> 1) & 1) as u64;
        let ox1 = ((op >> 2) & 1) as u64;
        let oz1 = ((op >> 3) & 1) as u64;

        let mut nx = *input_x;
        let mut nz = *input_z;
        nx[w0] = (nx[w0] & !m0) | (ox0 << b0);
        nz[w0] = (nz[w0] & !m0) | (oz0 << b0);
        nx[w1] = (nx[w1] & !m1) | (ox1 << b1);
        nz[w1] = (nz[w1] & !m1) | (oz1 << b1);

        out.push(nx, nz, self.phase[idx].apply(coeff));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli_string::PauliString;

    #[allow(clippy::type_complexity)]
    fn alloc_buf<const W: usize>() -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>, usize) {
        (
            vec![[0u64; W]; 1],
            vec![[0u64; W]; 1],
            vec![Complex64::new(0.0, 0.0); 1],
            0usize,
        )
    }

    /// Apply `gate` to `input` with coefficient 1 and return `(output, coeff)`.
    fn apply_1q<const W: usize>(
        gate: &Clifford1Q,
        input: &PauliString<W>,
    ) -> (PauliString<W>, Complex64) {
        let (mut x, mut z, mut c, mut len) = alloc_buf::<W>();
        let mut buf = OutputBuffer::<W> {
            x: &mut x,
            z: &mut z,
            coeff: &mut c,
            len: &mut len,
        };
        gate.apply(&input.x, &input.z, Complex64::new(1.0, 0.0), &mut buf);
        assert_eq!(*buf.len, 1);
        let out = PauliString::<W> { x: x[0], z: z[0] };
        (out, c[0])
    }

    fn apply_2q<const W: usize>(
        gate: &Clifford2Q,
        input: &PauliString<W>,
    ) -> (PauliString<W>, Complex64) {
        let (mut x, mut z, mut c, mut len) = alloc_buf::<W>();
        let mut buf = OutputBuffer::<W> {
            x: &mut x,
            z: &mut z,
            coeff: &mut c,
            len: &mut len,
        };
        gate.apply(&input.x, &input.z, Complex64::new(1.0, 0.0), &mut buf);
        assert_eq!(*buf.len, 1);
        let out = PauliString::<W> { x: x[0], z: z[0] };
        (out, c[0])
    }

    fn pauli_y<const W: usize>(qubit: u32) -> PauliString<W> {
        PauliString::<W>::y(qubit)
    }

    // ---- Slice 4.3: Clifford1Q tables ----

    #[test]
    fn h_on_qubit_0_w1() {
        let h = Clifford1Q::h(0);
        // I → I, +1
        let (out, c) = apply_1q::<1>(&h, &PauliString::<1>::identity());
        assert_eq!(out, PauliString::<1>::identity());
        assert_eq!(c, Complex64::new(1.0, 0.0));
        // X → Z, +1
        let (out, c) = apply_1q::<1>(&h, &PauliString::<1>::x(0));
        assert_eq!(out, PauliString::<1>::z(0));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        // Z → X, +1
        let (out, c) = apply_1q::<1>(&h, &PauliString::<1>::z(0));
        assert_eq!(out, PauliString::<1>::x(0));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        // Y → -Y, -1
        let (out, c) = apply_1q::<1>(&h, &pauli_y::<1>(0));
        assert_eq!(out, pauli_y::<1>(0));
        assert_eq!(c, Complex64::new(-1.0, 0.0));
    }

    #[test]
    fn s_on_qubit_0_w1() {
        let s = Clifford1Q::s(0);
        // X → Y, +1
        let (out, c) = apply_1q::<1>(&s, &PauliString::<1>::x(0));
        assert_eq!(out, pauli_y::<1>(0));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        // Z → Z, +1
        let (out, c) = apply_1q::<1>(&s, &PauliString::<1>::z(0));
        assert_eq!(out, PauliString::<1>::z(0));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        // Y → -X, -1
        let (out, c) = apply_1q::<1>(&s, &pauli_y::<1>(0));
        assert_eq!(out, PauliString::<1>::x(0));
        assert_eq!(c, Complex64::new(-1.0, 0.0));
    }

    #[test]
    fn x_gate_w1() {
        let g = Clifford1Q::x(0);
        // X → X, Z → -Z, Y → -Y
        let (o, c) = apply_1q::<1>(&g, &PauliString::<1>::x(0));
        assert_eq!(o, PauliString::<1>::x(0));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        let (o, c) = apply_1q::<1>(&g, &PauliString::<1>::z(0));
        assert_eq!(o, PauliString::<1>::z(0));
        assert_eq!(c, Complex64::new(-1.0, 0.0));
        let (o, c) = apply_1q::<1>(&g, &pauli_y::<1>(0));
        assert_eq!(o, pauli_y::<1>(0));
        assert_eq!(c, Complex64::new(-1.0, 0.0));
    }

    #[test]
    fn y_gate_w1() {
        let g = Clifford1Q::y(0);
        let (o, c) = apply_1q::<1>(&g, &PauliString::<1>::x(0));
        assert_eq!(o, PauliString::<1>::x(0));
        assert_eq!(c, Complex64::new(-1.0, 0.0));
        let (o, c) = apply_1q::<1>(&g, &PauliString::<1>::z(0));
        assert_eq!(o, PauliString::<1>::z(0));
        assert_eq!(c, Complex64::new(-1.0, 0.0));
        let (o, c) = apply_1q::<1>(&g, &pauli_y::<1>(0));
        assert_eq!(o, pauli_y::<1>(0));
        assert_eq!(c, Complex64::new(1.0, 0.0));
    }

    #[test]
    fn z_gate_w1() {
        let g = Clifford1Q::z(0);
        let (o, c) = apply_1q::<1>(&g, &PauliString::<1>::x(0));
        assert_eq!(o, PauliString::<1>::x(0));
        assert_eq!(c, Complex64::new(-1.0, 0.0));
        let (o, c) = apply_1q::<1>(&g, &PauliString::<1>::z(0));
        assert_eq!(o, PauliString::<1>::z(0));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        let (o, c) = apply_1q::<1>(&g, &pauli_y::<1>(0));
        assert_eq!(o, pauli_y::<1>(0));
        assert_eq!(c, Complex64::new(-1.0, 0.0));
    }

    #[test]
    fn h_on_qubit_64_w2_word_boundary() {
        let h = Clifford1Q::h(64);
        // X(64) → Z(64), bits live in word 1.
        let (out, c) = apply_1q::<2>(&h, &PauliString::<2>::x(64));
        assert_eq!(out, PauliString::<2>::z(64));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        assert_eq!(out.x[0], 0); // word 0 untouched
        assert_eq!(out.z[0], 0);
        // Z(64) → X(64).
        let (out, _c) = apply_1q::<2>(&h, &PauliString::<2>::z(64));
        assert_eq!(out, PauliString::<2>::x(64));
    }

    #[test]
    fn support_outside_bits_untouched_w2() {
        // Build X(0) · X(70) and apply H(70). Bit 0 must stay X, bit 70
        // must become Z, and the coefficient stays +1.
        let mut input = PauliString::<2>::x(0);
        let _ = input.mul_assign(&PauliString::<2>::x(70));
        let h = Clifford1Q::h(70);
        let (out, c) = apply_1q::<2>(&h, &input);

        // Expected: X(0) · Z(70).
        let mut expected = PauliString::<2>::x(0);
        let _ = expected.mul_assign(&PauliString::<2>::z(70));
        assert_eq!(out, expected);
        assert_eq!(c, Complex64::new(1.0, 0.0));
    }

    #[test]
    fn h_squared_is_identity() {
        // Apply H twice; the result is the input with phase +1.
        let h = Clifford1Q::h(0);
        for input in [
            PauliString::<1>::identity(),
            PauliString::<1>::x(0),
            PauliString::<1>::z(0),
            pauli_y::<1>(0),
        ] {
            let (mid, c1) = apply_1q::<1>(&h, &input);
            let (out, c2) = apply_1q::<1>(&h, &mid);
            assert_eq!(out, input, "H·H should be identity on {:?}", input);
            assert_eq!(
                c1 * c2,
                Complex64::new(1.0, 0.0),
                "phase should square to +1"
            );
        }
    }

    // ---- Slice 6.5: Clifford1Q adjoint table ----

    /// Self-adjoint 1Q Cliffords round-trip through `adjoint()` to themselves.
    #[test]
    fn h_x_y_z_are_self_adjoint() {
        for gate in [
            Clifford1Q::h(0),
            Clifford1Q::x(0),
            Clifford1Q::y(0),
            Clifford1Q::z(0),
        ] {
            let adj = gate.adjoint();
            assert_eq!(adj.out_pauli, gate.out_pauli);
            assert_eq!(adj.phase, gate.phase);
        }
    }

    /// `S` is not self-adjoint: its adjoint table differs in the phase
    /// pattern, but `(S†)† = S` (involution).
    #[test]
    fn s_adjoint_inverts_table_and_is_involutive() {
        let s = Clifford1Q::s(0);
        let s_dag = s.adjoint();
        // Forward: I→I(+1), X→Y(+1), Z→Z(+1), Y→-X(-1)
        // Adjoint: I→I(+1), X→-Y(-1), Z→Z(+1), Y→X(+1)
        assert_eq!(s_dag.out_pauli, [0, 3, 2, 1]);
        assert_eq!(
            s_dag.phase,
            [Phase::ONE, Phase::MINUS_ONE, Phase::ONE, Phase::ONE]
        );
        // Involution: (S†)† = S.
        let s_again = s_dag.adjoint();
        assert_eq!(s_again.out_pauli, s.out_pauli);
        assert_eq!(s_again.phase, s.phase);
    }

    /// `apply_adjoint` on `S` followed by `apply` on `S` round-trips X.
    #[test]
    fn s_apply_then_apply_adjoint_round_trips() {
        let s = Clifford1Q::s(0);
        let x_in = PauliString::<1>::x(0);
        let (mid, c1) = apply_1q::<1>(&s, &x_in);
        // mid = Y. Now apply S†.
        let (mut bx, mut bz, mut bc, mut len) = alloc_buf::<1>();
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        s.apply_adjoint(&mid.x, &mid.z, c1, &mut buf);
        assert_eq!(*buf.len, 1);
        assert_eq!(bx[0], x_in.x);
        assert_eq!(bz[0], x_in.z);
        assert_eq!(bc[0], Complex64::new(1.0, 0.0));
    }

    // ---- Slice 4.4: Clifford2Q tables ----

    /// Build `P_a ⊗ P_b` on qubits `(q0, q1)` of a `PauliString<W>` using
    /// `mul_assign`, where `pa` and `pb` are 2-bit single-qubit Pauli
    /// codes (`I=0, X=1, Z=2, Y=3`).
    fn tensor<const W: usize>(q0: u32, q1: u32, pa: u8, pb: u8) -> PauliString<W> {
        let mut p = PauliString::<W>::identity();
        let put = |p: &mut PauliString<W>, q: u32, code: u8| {
            let g = match code {
                0 => return,
                1 => PauliString::<W>::x(q),
                2 => PauliString::<W>::z(q),
                3 => PauliString::<W>::y(q),
                _ => unreachable!(),
            };
            // Y on a fresh identity contributes phase 0 (Y = (1,1) bits, no
            // pre-existing X·Z to worry about), and the partial products are
            // on disjoint qubits, so phases are always +1 here.
            let _ = p.mul_assign(&g);
        };
        put(&mut p, q0, pa);
        put(&mut p, q1, pb);
        p
    }

    #[test]
    fn cnot_generator_rules_w1() {
        let cnot = Clifford2Q::cnot(0, 1);
        // X⊗I → X⊗X
        let (o, c) = apply_2q::<1>(&cnot, &tensor::<1>(0, 1, 1, 0));
        assert_eq!(o, tensor::<1>(0, 1, 1, 1));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        // I⊗X → I⊗X
        let (o, c) = apply_2q::<1>(&cnot, &tensor::<1>(0, 1, 0, 1));
        assert_eq!(o, tensor::<1>(0, 1, 0, 1));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        // Z⊗I → Z⊗I
        let (o, c) = apply_2q::<1>(&cnot, &tensor::<1>(0, 1, 2, 0));
        assert_eq!(o, tensor::<1>(0, 1, 2, 0));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        // I⊗Z → Z⊗Z
        let (o, c) = apply_2q::<1>(&cnot, &tensor::<1>(0, 1, 0, 2));
        assert_eq!(o, tensor::<1>(0, 1, 2, 2));
        assert_eq!(c, Complex64::new(1.0, 0.0));
    }

    /// Reference for `CNOT (P⊗Q) CNOT†` derived from generator rules and
    /// `PauliString::mul_assign`. Each input Pauli on each qubit maps:
    ///   qubit 0:  I→I,  X→X⊗X,  Z→Z⊗I,  Y→i·X·Z → Y⊗X (with phase from XZ·X)
    ///   qubit 1:  I→I,  X→I⊗X,  Z→Z⊗Z,  Y→i·X·Z → Z⊗Y
    /// We compute the image as the product of these two qubit images and
    /// fold in any phase the multiplication picks up.
    fn cnot_reference<const W: usize>(
        control: u32,
        target: u32,
        pa: u8,
        pb: u8,
    ) -> (PauliString<W>, Phase) {
        // Image of `pa` on `control`: a `PauliString<W>` plus a phase.
        let (img_a, ph_a) = match pa {
            0 => (PauliString::<W>::identity(), Phase::ONE),
            1 => {
                // X⊗X on (control, target).
                let mut p = PauliString::<W>::x(control);
                let _ = p.mul_assign(&PauliString::<W>::x(target));
                (p, Phase::ONE)
            }
            2 => (PauliString::<W>::z(control), Phase::ONE),
            3 => {
                // Y → i · X · Z (decompose `pa = Y` into X·Z; image of X
                // is X⊗X, image of Z is Z⊗I; product picks up phase from
                // mul_assign, then multiply by `i` for the Y-decomposition).
                let mut p = PauliString::<W>::x(control);
                let _ = p.mul_assign(&PauliString::<W>::x(target));
                let mut q = PauliString::<W>::z(control);
                let mp = p.mul_assign(&q);
                // Y = i · X · Z, so the image is i · (X-image)(Z-image).
                (p, Phase::I + mp)
            }
            _ => unreachable!(),
        };
        let (img_b, ph_b) = match pb {
            0 => (PauliString::<W>::identity(), Phase::ONE),
            1 => (PauliString::<W>::x(target), Phase::ONE),
            2 => {
                // I⊗Z → Z⊗Z.
                let mut p = PauliString::<W>::z(control);
                let _ = p.mul_assign(&PauliString::<W>::z(target));
                (p, Phase::ONE)
            }
            3 => {
                // I⊗Y → I⊗(i·X·Z) → i · (I⊗X) · (Z⊗Z) = i · X(target) · Z(c) · Z(t).
                let mut p = PauliString::<W>::x(target);
                let mut q = PauliString::<W>::z(control);
                let _ = q.mul_assign(&PauliString::<W>::z(target));
                let mp = p.mul_assign(&q);
                (p, Phase::I + mp)
            }
            _ => unreachable!(),
        };
        // Combine the two images: image of `pa ⊗ pb` is image(pa) · image(pb)
        // since they commute as operators on different input qubits — but
        // the *image* operators may overlap, so phases come from mul_assign.
        let mut prod = img_a;
        let mp = prod.mul_assign(&img_b);
        (prod, ph_a + ph_b + mp)
    }

    #[test]
    fn cnot_full_table_w1() {
        let cnot = Clifford2Q::cnot(0, 1);
        for pa in 0u8..4 {
            for pb in 0u8..4 {
                let input = tensor::<1>(0, 1, pa, pb);
                let (got_out, got_c) = apply_2q::<1>(&cnot, &input);
                let (exp_out, exp_phase) = cnot_reference::<1>(0, 1, pa, pb);
                assert_eq!(
                    got_out, exp_out,
                    "CNOT output mismatch on (pa={}, pb={})",
                    pa, pb
                );
                assert_eq!(
                    got_c,
                    exp_phase.to_complex(),
                    "CNOT phase mismatch on (pa={}, pb={})",
                    pa,
                    pb
                );
            }
        }
    }

    #[test]
    fn cnot_word_boundary_w2() {
        // Control on qubit 63 (last bit of word 0), target on qubit 64
        // (first bit of word 1) — exercises the per-qubit word/bit math
        // independently for each support qubit.
        let cnot = Clifford2Q::cnot(63, 64);
        // X⊗X on (63, 64) → X⊗I (per CNOT rules: X⊗X → X⊗I).
        let input = tensor::<2>(63, 64, 1, 1);
        let (out, c) = apply_2q::<2>(&cnot, &input);
        let expected = tensor::<2>(63, 64, 1, 0);
        assert_eq!(out, expected);
        assert_eq!(c, Complex64::new(1.0, 0.0));
        // Y⊗X on (63, 64). Reference computes the image.
        let input = tensor::<2>(63, 64, 3, 1);
        let (got_out, got_c) = apply_2q::<2>(&cnot, &input);
        let (exp_out, exp_phase) = cnot_reference::<2>(63, 64, 3, 1);
        assert_eq!(got_out, exp_out);
        assert_eq!(got_c, exp_phase.to_complex());
    }

    #[test]
    fn cz_symmetry_w1() {
        // CZ is symmetric in its qubits.
        let cz_ab = Clifford2Q::cz(0, 1);
        let cz_ba = Clifford2Q::cz(1, 0);
        for pa in 0u8..4 {
            for pb in 0u8..4 {
                let input = tensor::<1>(0, 1, pa, pb);
                let (a, ca) = apply_2q::<1>(&cz_ab, &input);
                let (b, cb) = apply_2q::<1>(&cz_ba, &input);
                assert_eq!(a, b, "CZ(0,1) and CZ(1,0) disagree on ({}, {})", pa, pb);
                assert_eq!(ca, cb);
            }
        }
    }

    #[test]
    fn cz_generator_rules_w1() {
        let cz = Clifford2Q::cz(0, 1);
        // X⊗I → X⊗Z
        let (o, c) = apply_2q::<1>(&cz, &tensor::<1>(0, 1, 1, 0));
        assert_eq!(o, tensor::<1>(0, 1, 1, 2));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        // I⊗X → Z⊗X
        let (o, c) = apply_2q::<1>(&cz, &tensor::<1>(0, 1, 0, 1));
        assert_eq!(o, tensor::<1>(0, 1, 2, 1));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        // Z⊗I → Z⊗I
        let (o, c) = apply_2q::<1>(&cz, &tensor::<1>(0, 1, 2, 0));
        assert_eq!(o, tensor::<1>(0, 1, 2, 0));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        // I⊗Z → I⊗Z
        let (o, c) = apply_2q::<1>(&cz, &tensor::<1>(0, 1, 0, 2));
        assert_eq!(o, tensor::<1>(0, 1, 0, 2));
        assert_eq!(c, Complex64::new(1.0, 0.0));
    }

    #[test]
    fn swap_table_w1() {
        let swap = Clifford2Q::swap(0, 1);
        // II → II, XI → IX, IY → YI, XZ → ZX
        let (o, c) = apply_2q::<1>(&swap, &tensor::<1>(0, 1, 0, 0));
        assert_eq!(o, tensor::<1>(0, 1, 0, 0));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        let (o, c) = apply_2q::<1>(&swap, &tensor::<1>(0, 1, 1, 0));
        assert_eq!(o, tensor::<1>(0, 1, 0, 1));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        let (o, c) = apply_2q::<1>(&swap, &tensor::<1>(0, 1, 0, 3));
        assert_eq!(o, tensor::<1>(0, 1, 3, 0));
        assert_eq!(c, Complex64::new(1.0, 0.0));
        let (o, c) = apply_2q::<1>(&swap, &tensor::<1>(0, 1, 1, 2));
        assert_eq!(o, tensor::<1>(0, 1, 2, 1));
        assert_eq!(c, Complex64::new(1.0, 0.0));
    }
}
