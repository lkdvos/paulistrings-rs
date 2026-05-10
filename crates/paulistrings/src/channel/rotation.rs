//! Pauli rotation `exp(-i * theta * P / 2)`. See §6.

use super::{Channel, OutputBuffer};
use crate::pauli_string::PauliString;
use crate::phase::Phase;
use num_complex::Complex64;

/// A rotation `U = exp(-i · θ · P / 2)`.
///
/// In the Heisenberg picture, conjugation by `U` either leaves the input
/// invariant (if `[input, P] = 0`) or maps it to `cos(θ) · input +
/// sin(θ) · i · input · P`. Hence `MAX_FANOUT = 2`.
///
/// # Examples
///
/// ```
/// use paulistrings::channel::PauliRotation;
/// use paulistrings::PauliString;
///
/// let gen = PauliString::<1>::z(0);
/// let rot = PauliRotation::<1> {
///     support: vec![0],
///     gen_x: gen.x,
///     gen_z: gen.z,
///     theta: std::f64::consts::FRAC_PI_4,
/// };
/// # let _ = rot;
/// ```
pub struct PauliRotation<const W: usize> {
    /// Qubits the generator `P` acts on (the channel's support).
    pub support: Vec<u32>,
    /// X-part of the generator `P`, restricted to `support`.
    pub gen_x: [u64; W],
    /// Z-part of the generator `P`, restricted to `support`.
    pub gen_z: [u64; W],
    /// Rotation angle in radians.
    pub theta: f64,
}

impl<const W: usize> PauliRotation<W> {
    /// Shared body of `apply` and `apply_adjoint`. The adjoint of
    /// `exp(-i·θ·P/2)` is `exp(+i·θ·P/2) = exp(-i·(-θ)·P/2)`, so it's the
    /// same conjugation rule with `theta → -theta`.
    #[inline]
    fn apply_with_theta(
        &self,
        theta: f64,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        let input = PauliString::<W> {
            x: *input_x,
            z: *input_z,
        };
        let gen = PauliString::<W> {
            x: self.gen_x,
            z: self.gen_z,
        };

        if input.commutes_with(&gen) {
            out.push(*input_x, *input_z, coeff);
            return;
        }

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Term 1: cos(theta) * Q
        out.push(*input_x, *input_z, coeff * cos_t);

        // Term 2: i * sin(theta) * Q * P. mul_assign returns the i^k phase
        // arising from the Pauli algebra; the leading `i` is folded in via
        // `Phase::I + phase`.
        let mut prod = input;
        let phase = prod.mul_assign(&gen);
        let total_phase = Phase::I + phase;
        out.push(prod.x, prod.z, total_phase.apply(coeff) * sin_t);
    }
}

impl<const W: usize> Channel<W> for PauliRotation<W> {
    #[inline]
    fn max_fanout(&self) -> usize {
        2
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
        self.apply_with_theta(self.theta, input_x, input_z, coeff, out);
    }

    #[inline]
    fn apply_adjoint(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        self.apply_with_theta(-self.theta, input_x, input_z, coeff, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[allow(clippy::type_complexity)]
    fn alloc_bufs<const W: usize>(
        n: usize,
    ) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>, usize) {
        (
            vec![[0u64; W]; n],
            vec![[0u64; W]; n],
            vec![Complex64::new(0.0, 0.0); n],
            0usize,
        )
    }

    fn approx_eq(a: Complex64, b: Complex64, tol: f64) -> bool {
        (a - b).norm() <= tol
    }

    /// `theta = 0` and the input/generator anticommute: the fanout-2 branch
    /// runs but `sin(0) = 0` makes the second term vanish.
    #[test]
    fn theta_zero_anticommuting_w1() {
        let q = PauliString::<1>::x(0);
        let p = PauliString::<1>::z(0);
        let rot = PauliRotation::<1> {
            support: vec![0],
            gen_x: p.x,
            gen_z: p.z,
            theta: 0.0,
        };
        let c = Complex64::new(2.0, 3.0);
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        rot.apply(&q.x, &q.z, c, &mut buf);
        assert_eq!(*buf.len, 2);
        assert_eq!(bx[0], q.x);
        assert_eq!(bz[0], q.z);
        assert!(approx_eq(bc[0], c, TOL));
        // Bits of the second term are X * Z = Y; coefficient is 0.
        let y = PauliString::<1>::y(0);
        assert_eq!(bx[1], y.x);
        assert_eq!(bz[1], y.z);
        assert!(approx_eq(bc[1], Complex64::new(0.0, 0.0), TOL));
    }

    /// `theta = 0` with a commuting generator: fanout-1, output is input.
    #[test]
    fn theta_zero_commuting_w1() {
        let q = PauliString::<1>::z(0);
        let p = PauliString::<1>::z(0);
        let rot = PauliRotation::<1> {
            support: vec![0],
            gen_x: p.x,
            gen_z: p.z,
            theta: 0.0,
        };
        let c = Complex64::new(2.0, 3.0);
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        rot.apply(&q.x, &q.z, c, &mut buf);
        assert_eq!(*buf.len, 1);
        assert_eq!(bx[0], q.x);
        assert_eq!(bz[0], q.z);
        assert_eq!(bc[0], c);
    }

    /// Slice spec: rotation by π around `Z` flips `X → −X` (sign in the coeff).
    #[test]
    fn pi_z_flips_x_to_minus_x_w1() {
        let q = PauliString::<1>::x(0);
        let p = PauliString::<1>::z(0);
        let rot = PauliRotation::<1> {
            support: vec![0],
            gen_x: p.x,
            gen_z: p.z,
            theta: std::f64::consts::PI,
        };
        let c = Complex64::new(1.0, 0.0);
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        rot.apply(&q.x, &q.z, c, &mut buf);
        assert_eq!(*buf.len, 2);
        assert_eq!(bx[0], q.x);
        assert_eq!(bz[0], q.z);
        assert!(approx_eq(bc[0], Complex64::new(-1.0, 0.0), TOL));
        let y = PauliString::<1>::y(0);
        assert_eq!(bx[1], y.x);
        assert_eq!(bz[1], y.z);
        assert!(approx_eq(bc[1], Complex64::new(0.0, 0.0), TOL));
    }

    /// Identity input commutes with every generator: fanout-1, output is input.
    #[test]
    fn commuting_case_is_fanout_one_w1() {
        let q = PauliString::<1>::identity();
        let p = PauliString::<1>::z(0);
        let rot = PauliRotation::<1> {
            support: vec![0],
            gen_x: p.x,
            gen_z: p.z,
            theta: std::f64::consts::FRAC_PI_4,
        };
        let c = Complex64::new(0.5, 0.25);
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        rot.apply(&q.x, &q.z, c, &mut buf);
        assert_eq!(*buf.len, 1);
        assert_eq!(bx[0], q.x);
        assert_eq!(bz[0], q.z);
        assert_eq!(bc[0], c);
    }

    /// Anticommuting case at a generic angle: cos·Q + sin·Y, with both
    /// coefficients pinned numerically.
    #[test]
    fn anticommuting_case_is_fanout_two_w1() {
        let q = PauliString::<1>::x(0);
        let p = PauliString::<1>::z(0);
        let theta = std::f64::consts::FRAC_PI_3;
        let rot = PauliRotation::<1> {
            support: vec![0],
            gen_x: p.x,
            gen_z: p.z,
            theta,
        };
        let c = Complex64::new(1.0, 0.0);
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        rot.apply(&q.x, &q.z, c, &mut buf);
        assert_eq!(*buf.len, 2);
        assert_eq!(bx[0], q.x);
        assert_eq!(bz[0], q.z);
        assert!(approx_eq(bc[0], Complex64::new(theta.cos(), 0.0), TOL));
        let y = PauliString::<1>::y(0);
        assert_eq!(bx[1], y.x);
        assert_eq!(bz[1], y.z);
        assert!(approx_eq(bc[1], Complex64::new(theta.sin(), 0.0), TOL));
    }

    /// Catches a sign error in the multiplication direction. With Q=Z, P=X:
    /// `cos(π/2)·Z + i sin(π/2) · ZX = i · iY = −Y`.
    #[test]
    fn pi_over_two_x_rotates_z_to_minus_y_w1() {
        let q = PauliString::<1>::z(0);
        let p = PauliString::<1>::x(0);
        let rot = PauliRotation::<1> {
            support: vec![0],
            gen_x: p.x,
            gen_z: p.z,
            theta: std::f64::consts::FRAC_PI_2,
        };
        let c = Complex64::new(1.0, 0.0);
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        rot.apply(&q.x, &q.z, c, &mut buf);
        assert_eq!(*buf.len, 2);
        assert_eq!(bx[0], q.x);
        assert_eq!(bz[0], q.z);
        assert!(approx_eq(bc[0], Complex64::new(0.0, 0.0), TOL));
        let y = PauliString::<1>::y(0);
        assert_eq!(bx[1], y.x);
        assert_eq!(bz[1], y.z);
        assert!(approx_eq(bc[1], Complex64::new(-1.0, 0.0), TOL));
    }

    /// Catches a sign error in the `Phase::I + phase` step. With Q=Y, P=Z:
    /// `Y · Z = +iX` (mul_assign delta = 1), so the total phase factor is
    /// `i · i = −1`, giving `(X, −sin(π/2)·c) = (X, −c)`.
    #[test]
    fn phase_from_mul_assign_is_folded_w1() {
        let q = PauliString::<1>::y(0);
        let p = PauliString::<1>::z(0);
        let rot = PauliRotation::<1> {
            support: vec![0],
            gen_x: p.x,
            gen_z: p.z,
            theta: std::f64::consts::FRAC_PI_2,
        };
        let c = Complex64::new(1.0, 0.0);
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        rot.apply(&q.x, &q.z, c, &mut buf);
        assert_eq!(*buf.len, 2);
        assert_eq!(bx[0], q.x);
        assert_eq!(bz[0], q.z);
        assert!(approx_eq(bc[0], Complex64::new(0.0, 0.0), TOL));
        let xp = PauliString::<1>::x(0);
        assert_eq!(bx[1], xp.x);
        assert_eq!(bz[1], xp.z);
        assert!(approx_eq(bc[1], Complex64::new(-1.0, 0.0), TOL));
    }

    /// Multi-word: input on word 0, generator on word 1. Disjoint support →
    /// commute → fanout-1.
    #[test]
    fn multi_word_disjoint_support_commutes_w2() {
        let q = PauliString::<2>::x(0);
        let p = PauliString::<2>::z(64);
        let rot = PauliRotation::<2> {
            support: vec![64],
            gen_x: p.x,
            gen_z: p.z,
            theta: std::f64::consts::FRAC_PI_4,
        };
        let c = Complex64::new(1.0, 0.0);
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<2>(2);
        let mut buf = OutputBuffer::<2> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        rot.apply(&q.x, &q.z, c, &mut buf);
        assert_eq!(*buf.len, 1);
        assert_eq!(bx[0], q.x);
        assert_eq!(bz[0], q.z);
        assert_eq!(bc[0], c);
    }

    /// Multi-word: anticommuting bits land in word 1; word 0 stays zero.
    #[test]
    fn multi_word_anticommute_in_word_1_w2() {
        let q = PauliString::<2>::x(64);
        let p = PauliString::<2>::z(64);
        let theta = std::f64::consts::FRAC_PI_3;
        let rot = PauliRotation::<2> {
            support: vec![64],
            gen_x: p.x,
            gen_z: p.z,
            theta,
        };
        let c = Complex64::new(1.0, 0.0);
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<2>(2);
        let mut buf = OutputBuffer::<2> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        rot.apply(&q.x, &q.z, c, &mut buf);
        assert_eq!(*buf.len, 2);
        assert_eq!(bx[0], q.x);
        assert_eq!(bz[0], q.z);
        assert!(approx_eq(bc[0], Complex64::new(theta.cos(), 0.0), TOL));
        let y = PauliString::<2>::y(64);
        assert_eq!(bx[1], y.x);
        assert_eq!(bz[1], y.z);
        assert_eq!(bx[1][0], 0u64);
        assert_eq!(bz[1][0], 0u64);
        assert!(approx_eq(bc[1], Complex64::new(theta.sin(), 0.0), TOL));
    }

    /// The engine drives apply repeatedly against the same buffer; back-to-back
    /// calls must reuse storage without growing the backing vecs.
    #[test]
    fn reuse_buffer_across_calls() {
        let cap = 2;
        let mut bx: Vec<[u64; 1]> = vec![[0u64; 1]; cap];
        let mut bz: Vec<[u64; 1]> = vec![[0u64; 1]; cap];
        let mut bc: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); cap];
        let p = PauliString::<1>::z(0);
        let rot = PauliRotation::<1> {
            support: vec![0],
            gen_x: p.x,
            gen_z: p.z,
            theta: std::f64::consts::FRAC_PI_3,
        };
        let q = PauliString::<1>::x(0);
        for _ in 0..3 {
            let mut len = 0usize;
            let mut buf = OutputBuffer::<1> {
                x: &mut bx,
                z: &mut bz,
                coeff: &mut bc,
                len: &mut len,
            };
            rot.apply(&q.x, &q.z, Complex64::new(1.0, 0.0), &mut buf);
            assert_eq!(*buf.len, 2);
        }
        assert_eq!(bx.capacity(), cap);
        assert_eq!(bz.capacity(), cap);
        assert_eq!(bc.capacity(), cap);
    }
}
