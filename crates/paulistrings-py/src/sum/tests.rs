use super::*;
use crate::circuit::CircuitImpl;
use paulistrings::{Channel, OutputBuffer, PtmChannel};

/// `X ↦ i·X` on one qubit: maps a Hermitian operator to an anti-Hermitian one, which no gate or noise factory the bindings expose can build.
struct ImaginaryPhase(usize);

impl Channel<1> for ImaginaryPhase {
    fn support(&self) -> [u64; 1] {
        paulistrings::support_mask(&[self.0])
    }

    fn apply(&self, x: &[u64; 1], z: &[u64; 1], coeff: Complex64, out: &mut OutputBuffer<1>) {
        let on_x = x[0] >> self.0 & 1 == 1 && z[0] >> self.0 & 1 == 0;
        out.push(*x, *z, if on_x { coeff * Complex64::i() } else { coeff });
    }

    fn apply_adjoint(
        &self,
        x: &[u64; 1],
        z: &[u64; 1],
        coeff: Complex64,
        out: &mut OutputBuffer<1>,
    ) {
        let on_x = x[0] >> self.0 & 1 == 1 && z[0] >> self.0 & 1 == 0;
        out.push(*x, *z, if on_x { -coeff * Complex64::i() } else { coeff });
    }
}

fn circuit(hermitian: bool) -> CircuitImpl {
    let mut circuit = CoreCircuit::<1>::new(2);
    circuit.push(PtmChannel::hadamard(0));
    if !hermitian {
        circuit.push(ImaginaryPhase(1));
    }
    CircuitImpl::W1(circuit)
}

#[test]
fn a_real_dtype_refuses_a_channel_that_breaks_hermiticity() {
    for dtype in [Dtype::F64, Dtype::F32] {
        let message = hermiticity_refusal("PauliSum", dtype, &circuit(false))
            .expect("a real dtype cannot carry an imaginary transfer table");
        assert!(message.contains(dtype.name()), "{message}");
        assert!(message.contains("complex128"), "{message}");
    }
}

#[test]
fn a_complex_dtype_or_a_hermitian_circuit_passes() {
    for dtype in [Dtype::C128, Dtype::C64] {
        assert_eq!(hermiticity_refusal("PauliSum", dtype, &circuit(false)), None);
    }
    for dtype in [Dtype::F64, Dtype::F32, Dtype::C128, Dtype::C64] {
        assert_eq!(hermiticity_refusal("PauliSum", dtype, &circuit(true)), None);
    }
}
