//! Noise channels: Depolarizing, Dephasing, AmplitudeDamping. See §6.

#![allow(unused)]

use super::{Channel, OutputBuffer};
use num_complex::Complex64;

/// Single-qubit depolarizing noise with error probability `p`.
///
/// In the Heisenberg picture this is just a coefficient rescaling: the
/// identity on the support qubit is preserved unchanged; every non-identity
/// Pauli on the support is multiplied by `1 - 4p/3`. Self-adjoint, so the
/// default `apply_adjoint` from the trait is correct.
pub struct Depolarizing {
    pub support: [u32; 1],
    pub p: f64,
}

impl<const W: usize> Channel<W> for Depolarizing {
    #[inline]
    fn max_fanout(&self) -> usize {
        1
    }

    #[inline]
    fn support(&self) -> &[u32] {
        &self.support
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
        let word = q / 64;
        let bit = q % 64;
        let x_bit = (input_x[word] >> bit) & 1;
        let z_bit = (input_z[word] >> bit) & 1;
        let scale = if (x_bit | z_bit) == 0 {
            1.0
        } else {
            1.0 - 4.0 * self.p / 3.0
        };
        out.push(*input_x, *input_z, coeff * scale);
    }
}

/// Single-qubit dephasing noise with error probability `p`.
///
/// Heisenberg dual: `E*(P) = (1-p) P + p Z P Z`. So I and Z are preserved;
/// X and Y are scaled by `1 - 2p` (`Z X Z = -X`, `Z Y Z = -Y`). Equivalently,
/// the scale fires iff the support qubit's `x_bit` is set. Self-adjoint.
pub struct Dephasing {
    pub support: [u32; 1],
    pub p: f64,
}

impl<const W: usize> Channel<W> for Dephasing {
    #[inline]
    fn max_fanout(&self) -> usize {
        1
    }

    #[inline]
    fn support(&self) -> &[u32] {
        &self.support
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
        let word = q / 64;
        let bit = q % 64;
        let x_bit = (input_x[word] >> bit) & 1;
        let scale = if x_bit == 1 { 1.0 - 2.0 * self.p } else { 1.0 };
        out.push(*input_x, *input_z, coeff * scale);
    }
}

/// Single-qubit amplitude damping with parameter `gamma`.
///
/// The only noise in the built-in set with genuine fan-out > 1. Heisenberg
/// dual via Kraus operators `E_0 = |0⟩⟨0| + √(1-γ)|1⟩⟨1|` and
/// `E_1 = √γ |0⟩⟨1|`:
///
/// - `I → I`
/// - `X → √(1-γ) X`
/// - `Y → √(1-γ) Y`
/// - `Z → (1-γ) Z + γ I`   (the only fanout-2 case)
///
/// Not self-adjoint in general — the Heisenberg adjoint of amplitude damping
/// is not amplitude damping itself. v0.1 uses the default `apply_adjoint =
/// apply` placeholder; users propagating in `Direction::Heisenberg` mode
/// with this channel should be aware of that approximation. (The
/// non-Heisenberg use-case is the common one and is exact.)
pub struct AmplitudeDamping {
    pub support: [u32; 1],
    pub gamma: f64,
}

impl<const W: usize> Channel<W> for AmplitudeDamping {
    #[inline]
    fn max_fanout(&self) -> usize {
        2
    }

    #[inline]
    fn support(&self) -> &[u32] {
        &self.support
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
        let word = q / 64;
        let bit = q % 64;
        let mask = 1u64 << bit;
        let x_bit = (input_x[word] >> bit) & 1;
        let z_bit = (input_z[word] >> bit) & 1;
        match (x_bit, z_bit) {
            (0, 0) => {
                // I → I.
                out.push(*input_x, *input_z, coeff);
            }
            (1, _) => {
                // X or Y → √(1-γ) · same.
                let scale = (1.0 - self.gamma).sqrt();
                out.push(*input_x, *input_z, coeff * scale);
            }
            (0, 1) => {
                // Z → (1-γ) Z + γ I. Emit Z first (matches the order in the
                // doc-comment), then I (with the support's z-bit cleared).
                out.push(*input_x, *input_z, coeff * (1.0 - self.gamma));
                let mut nz = *input_z;
                nz[word] &= !mask;
                out.push(*input_x, nz, coeff * self.gamma);
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli_string::PauliString;

    const TOL: f64 = 1e-12;

    #[allow(clippy::type_complexity)]
    fn make_buf<const W: usize>(
        cap: usize,
    ) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>, usize) {
        (
            vec![[0u64; W]; cap],
            vec![[0u64; W]; cap],
            vec![Complex64::new(0.0, 0.0); cap],
            0,
        )
    }

    fn approx_eq(a: Complex64, b: Complex64, tol: f64) -> bool {
        (a - b).norm() <= tol
    }

    /// Identity on the support qubit is preserved exactly — coefficient
    /// rescaling does not touch the I sector.
    #[test]
    fn depolarizing_passes_identity_through() {
        let ch = Depolarizing { support: [0], p: 0.1 };
        let p = PauliString::<1>::identity();
        let (mut bx, mut bz, mut bc, mut len) = make_buf::<1>(1);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        <Depolarizing as Channel<1>>::apply(&ch, &p.x, &p.z, Complex64::new(2.0, 0.0), &mut buf);
        assert_eq!(len, 1);
        assert_eq!(bx[0], p.x);
        assert_eq!(bz[0], p.z);
        assert!(approx_eq(bc[0], Complex64::new(2.0, 0.0), TOL));
    }

    /// X, Y, Z on the support qubit each get scaled by `1 - 4p/3`.
    #[test]
    fn depolarizing_scales_xyz() {
        let p = 0.15;
        let ch = Depolarizing { support: [0], p };
        let scale = 1.0 - 4.0 * p / 3.0;
        for pauli in [PauliString::<1>::x(0), PauliString::<1>::y(0), PauliString::<1>::z(0)] {
            let (mut bx, mut bz, mut bc, mut len) = make_buf::<1>(1);
            let mut buf = OutputBuffer::<1> {
                x: &mut bx,
                z: &mut bz,
                coeff: &mut bc,
                len: &mut len,
            };
            <Depolarizing as Channel<1>>::apply(
                &ch,
                &pauli.x,
                &pauli.z,
                Complex64::new(1.0, 0.0),
                &mut buf,
            );
            assert_eq!(len, 1);
            assert_eq!(bx[0], pauli.x);
            assert_eq!(bz[0], pauli.z);
            assert!(approx_eq(bc[0], Complex64::new(scale, 0.0), TOL));
        }
    }

    /// Off-support qubits are ignored: a Z on qubit 1 with the channel
    /// supported on qubit 0 leaves the coefficient untouched (the support
    /// qubit is in I-state).
    #[test]
    fn depolarizing_off_support_is_no_op() {
        let ch = Depolarizing { support: [0], p: 0.2 };
        let p = PauliString::<1>::z(1);
        let (mut bx, mut bz, mut bc, mut len) = make_buf::<1>(1);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        <Depolarizing as Channel<1>>::apply(&ch, &p.x, &p.z, Complex64::new(3.0, 0.0), &mut buf);
        assert_eq!(len, 1);
        assert!(approx_eq(bc[0], Complex64::new(3.0, 0.0), TOL));
    }

    /// W=2: support qubit lives in word 1 (qubit 64+). The scale must
    /// trigger when bits at qubit 64 are non-identity.
    #[test]
    fn depolarizing_w2_word_boundary() {
        let p = 0.25;
        let ch = Depolarizing { support: [64], p };
        let scale = 1.0 - 4.0 * p / 3.0;
        // X on qubit 64 → word-1 x-bit set.
        let pauli = PauliString::<2>::x(64);
        let (mut bx, mut bz, mut bc, mut len) = make_buf::<2>(1);
        let mut buf = OutputBuffer::<2> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        <Depolarizing as Channel<2>>::apply(
            &ch,
            &pauli.x,
            &pauli.z,
            Complex64::new(1.0, 0.0),
            &mut buf,
        );
        assert_eq!(len, 1);
        assert_eq!(bx[0], pauli.x);
        assert_eq!(bz[0], pauli.z);
        assert!(approx_eq(bc[0], Complex64::new(scale, 0.0), TOL));
    }

    /// I and Z commute with Z, so dephasing leaves their coefficients alone.
    #[test]
    fn dephasing_preserves_i_and_z() {
        let ch = Dephasing { support: [0], p: 0.3 };
        for pauli in [PauliString::<1>::identity(), PauliString::<1>::z(0)] {
            let (mut bx, mut bz, mut bc, mut len) = make_buf::<1>(1);
            let mut buf = OutputBuffer::<1> {
                x: &mut bx,
                z: &mut bz,
                coeff: &mut bc,
                len: &mut len,
            };
            <Dephasing as Channel<1>>::apply(
                &ch,
                &pauli.x,
                &pauli.z,
                Complex64::new(2.5, 0.0),
                &mut buf,
            );
            assert_eq!(len, 1);
            assert_eq!(bx[0], pauli.x);
            assert_eq!(bz[0], pauli.z);
            assert!(approx_eq(bc[0], Complex64::new(2.5, 0.0), TOL));
        }
    }

    /// X and Y both anticommute with Z, so dephasing scales them by 1 - 2p.
    #[test]
    fn dephasing_scales_x_and_y() {
        let p = 0.2;
        let ch = Dephasing { support: [0], p };
        let scale = 1.0 - 2.0 * p;
        for pauli in [PauliString::<1>::x(0), PauliString::<1>::y(0)] {
            let (mut bx, mut bz, mut bc, mut len) = make_buf::<1>(1);
            let mut buf = OutputBuffer::<1> {
                x: &mut bx,
                z: &mut bz,
                coeff: &mut bc,
                len: &mut len,
            };
            <Dephasing as Channel<1>>::apply(
                &ch,
                &pauli.x,
                &pauli.z,
                Complex64::new(1.0, 0.0),
                &mut buf,
            );
            assert_eq!(len, 1);
            assert_eq!(bx[0], pauli.x);
            assert_eq!(bz[0], pauli.z);
            assert!(approx_eq(bc[0], Complex64::new(scale, 0.0), TOL));
        }
    }

    /// W=2: dephasing on qubit 64 scales an X@64 by 1 - 2p.
    #[test]
    fn dephasing_w2_word_boundary() {
        let p = 0.4;
        let ch = Dephasing { support: [64], p };
        let scale = 1.0 - 2.0 * p;
        let pauli = PauliString::<2>::x(64);
        let (mut bx, mut bz, mut bc, mut len) = make_buf::<2>(1);
        let mut buf = OutputBuffer::<2> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        <Dephasing as Channel<2>>::apply(
            &ch,
            &pauli.x,
            &pauli.z,
            Complex64::new(1.0, 0.0),
            &mut buf,
        );
        assert_eq!(len, 1);
        assert!(approx_eq(bc[0], Complex64::new(scale, 0.0), TOL));
    }

    /// I on the support is fixed by amplitude damping (fanout 1, coeff
    /// preserved).
    #[test]
    fn amplitude_damping_passes_identity_through() {
        let ch = AmplitudeDamping { support: [0], gamma: 0.3 };
        let p = PauliString::<1>::identity();
        let (mut bx, mut bz, mut bc, mut len) = make_buf::<1>(2);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        <AmplitudeDamping as Channel<1>>::apply(
            &ch,
            &p.x,
            &p.z,
            Complex64::new(2.0, 0.0),
            &mut buf,
        );
        assert_eq!(len, 1);
        assert_eq!(bx[0], p.x);
        assert_eq!(bz[0], p.z);
        assert!(approx_eq(bc[0], Complex64::new(2.0, 0.0), TOL));
    }

    /// X and Y on the support each get scaled by √(1-γ), fanout 1.
    #[test]
    fn amplitude_damping_scales_x_and_y_by_sqrt() {
        let gamma = 0.2;
        let ch = AmplitudeDamping { support: [0], gamma };
        let scale = (1.0 - gamma).sqrt();
        for pauli in [PauliString::<1>::x(0), PauliString::<1>::y(0)] {
            let (mut bx, mut bz, mut bc, mut len) = make_buf::<1>(2);
            let mut buf = OutputBuffer::<1> {
                x: &mut bx,
                z: &mut bz,
                coeff: &mut bc,
                len: &mut len,
            };
            <AmplitudeDamping as Channel<1>>::apply(
                &ch,
                &pauli.x,
                &pauli.z,
                Complex64::new(1.0, 0.0),
                &mut buf,
            );
            assert_eq!(len, 1);
            assert_eq!(bx[0], pauli.x);
            assert_eq!(bz[0], pauli.z);
            assert!(approx_eq(bc[0], Complex64::new(scale, 0.0), TOL));
        }
    }

    /// Z on the support fans out to `(1-γ)·Z + γ·I`. The first emit is the
    /// Z term, the second is the I term (z-bit cleared on the support
    /// qubit).
    #[test]
    fn amplitude_damping_z_fans_out_to_z_plus_i() {
        let gamma = 0.25;
        let ch = AmplitudeDamping { support: [0], gamma };
        let p = PauliString::<1>::z(0);
        let (mut bx, mut bz, mut bc, mut len) = make_buf::<1>(2);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        <AmplitudeDamping as Channel<1>>::apply(
            &ch,
            &p.x,
            &p.z,
            Complex64::new(1.0, 0.0),
            &mut buf,
        );
        assert_eq!(len, 2);
        // First: (1-γ)·Z (z-bit kept).
        assert_eq!(bx[0], p.x);
        assert_eq!(bz[0], p.z);
        assert!(approx_eq(bc[0], Complex64::new(1.0 - gamma, 0.0), TOL));
        // Second: γ·I (z-bit cleared).
        let id = PauliString::<1>::identity();
        assert_eq!(bx[1], id.x);
        assert_eq!(bz[1], id.z);
        assert!(approx_eq(bc[1], Complex64::new(gamma, 0.0), TOL));
    }

    /// W=2: Z on qubit 64 fans out to (1-γ)·Z@64 + γ·I, with the I term's
    /// z-bit cleared in word 1 only.
    #[test]
    fn amplitude_damping_w2_word_boundary() {
        let gamma = 0.4;
        let ch = AmplitudeDamping { support: [64], gamma };
        let p = PauliString::<2>::z(64);
        let (mut bx, mut bz, mut bc, mut len) = make_buf::<2>(2);
        let mut buf = OutputBuffer::<2> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        <AmplitudeDamping as Channel<2>>::apply(
            &ch,
            &p.x,
            &p.z,
            Complex64::new(1.0, 0.0),
            &mut buf,
        );
        assert_eq!(len, 2);
        assert_eq!(bx[0], p.x);
        assert_eq!(bz[0], p.z);
        assert!(approx_eq(bc[0], Complex64::new(1.0 - gamma, 0.0), TOL));
        // I term: word 1 z-bit cleared.
        assert_eq!(bx[1], [0u64, 0u64]);
        assert_eq!(bz[1], [0u64, 0u64]);
        assert!(approx_eq(bc[1], Complex64::new(gamma, 0.0), TOL));
    }
}
