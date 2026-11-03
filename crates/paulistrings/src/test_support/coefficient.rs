//! Fixtures for runs in every [`Coefficient`] type.

use num_complex::Complex64;

use crate::{support_mask, Channel, Circuit, Coefficient, OutputBuffer, PtmChannel};

/// The agreement a run in `C` owes the `Complex64` reference: double precision is exact up to summation order, single precision rounds every product.
pub fn coefficient_tolerance<C: Coefficient>() -> f64 {
    if core::mem::size_of::<C::Real>() == 8 {
        1e-12
    } else {
        2e-5
    }
}

/// `X ↦ i·X` on one qubit: maps the Hermitian `X` to an anti-Hermitian operator, so its transfer table is imaginary.
pub struct ImaginaryPhase(pub usize);

impl ImaginaryPhase {
    fn push(
        &self,
        x: &[u64; 1],
        z: &[u64; 1],
        phase: Complex64,
        coeff: Complex64,
        out: &mut OutputBuffer<1>,
    ) {
        let on_x = (x[0] >> self.0) & 1 == 1 && (z[0] >> self.0) & 1 == 0;
        out.push(*x, *z, if on_x { coeff * phase } else { coeff });
    }
}

impl Channel<1> for ImaginaryPhase {
    fn support(&self) -> [u64; 1] {
        support_mask(&[self.0])
    }

    fn apply(&self, x: &[u64; 1], z: &[u64; 1], coeff: Complex64, out: &mut OutputBuffer<1>) {
        self.push(x, z, Complex64::i(), coeff, out);
    }

    fn apply_adjoint(
        &self,
        x: &[u64; 1],
        z: &[u64; 1],
        coeff: Complex64,
        out: &mut OutputBuffer<1>,
    ) {
        self.push(x, z, -Complex64::i(), coeff, out);
    }
}

/// A Hadamard on qubit 0, then [`ImaginaryPhase`] on qubit 1, on four qubits: a circuit no real coefficient type can carry.
pub fn non_hermitian_circuit() -> Circuit<1> {
    let mut circuit = Circuit::<1>::new(4);
    circuit.push(PtmChannel::hadamard(0));
    circuit.push(ImaginaryPhase(1));
    circuit
}
