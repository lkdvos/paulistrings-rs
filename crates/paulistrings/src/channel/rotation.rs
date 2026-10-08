//! [`PauliRotation`], the rotation `exp(-i·θ·P/2)` about a Pauli generator.

use super::prepared::{Prepared, RotationPrep, MAX_LOCAL_SUPPORT};
use super::{Channel, OutputBuffer};
use crate::pauli_string::PauliString;
use crate::pauli_sum::hash::Gf2Hash;
use crate::phase::Phase;
use num_complex::Complex64;

/// A rotation `U = exp(-i · θ · P / 2)` about a Pauli generator `P` of any weight.
///
/// The support is derived from the generator: a qubit belongs to it iff `P` is non-identity there.
#[derive(Clone, Debug)]
pub struct PauliRotation<const W: usize> {
    /// X-part of the generator `P`.
    gen_x: [u64; W],
    /// Z-part of the generator `P`.
    gen_z: [u64; W],
    /// Rotation angle in radians.
    theta: f64,
}

impl<const W: usize> PauliRotation<W> {
    /// A rotation by `theta` radians about `gen`.
    pub fn new(gen: PauliString<W>, theta: f64) -> Self {
        Self {
            gen_x: gen.x,
            gen_z: gen.z,
            theta,
        }
    }

    /// The generator `P`.
    #[inline]
    pub fn generator(&self) -> PauliString<W> {
        PauliString::<W> {
            x: self.gen_x,
            z: self.gen_z,
        }
    }

    /// The rotation angle in radians.
    #[inline]
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Number of qubits in the support, i.e. the generator's Pauli weight.
    #[inline]
    pub fn weight(&self) -> usize {
        self.generator().weight() as usize
    }

    /// Body of `apply` and `apply_adjoint`, the adjoint being the rotation by `-θ`.
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

        let (sin_t, cos_t) = sin_cos(theta);

        out.push(*input_x, *input_z, coeff * cos_t);

        // `i · sin θ · Q · P`, the leading `i` folded into the product's phase.
        let mut prod = input;
        let phase = prod.mul_assign(&gen);
        let total_phase = Phase::I + phase;
        out.push(prod.x, prod.z, total_phase.apply(coeff) * sin_t);
    }
}

/// `theta.sin_cos()` with quarter turns snapped to exact `0`/`±1`, so a rotation by `k·π/2` stays fanout 1.
fn sin_cos(theta: f64) -> (f64, f64) {
    const SNAP: f64 = 1e-15;
    match theta.sin_cos() {
        (s, c) if s.abs() < SNAP => (0.0, c.signum()),
        (s, c) if c.abs() < SNAP => (s.signum(), 0.0),
        sc => sc,
    }
}

impl<const W: usize> Channel<W> for PauliRotation<W> {
    #[inline]
    fn max_fanout(&self) -> usize {
        2
    }

    #[inline]
    fn support(&self) -> [u64; W] {
        core::array::from_fn(|w| self.gen_x[w] | self.gen_z[w])
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

    /// The tabulated default up to `MAX_LOCAL_SUPPORT`, `Prepared::Rotation` above it.
    fn prepare(&self, hash: &Gf2Hash<W>, adjoint: bool) -> Option<Prepared<W>> {
        if self.weight() <= MAX_LOCAL_SUPPORT {
            return Prepared::derive_local(self, hash, adjoint);
        }
        let (sin, cos) = sin_cos(if adjoint { -self.theta } else { self.theta });
        let gen = self.generator();
        Some(Prepared::Rotation(RotationPrep {
            gen,
            cos,
            sin,
            bucket_delta_identity: 0,
            bucket_delta_gen: hash.bucket_of_pauli(&gen),
        }))
    }
}

#[cfg(test)]
mod tests;
