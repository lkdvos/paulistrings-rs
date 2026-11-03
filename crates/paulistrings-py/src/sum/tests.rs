use super::*;
use crate::circuit::CircuitImpl;
use paulistrings::test_support::{non_hermitian_circuit, random_circuit};

#[test]
fn only_a_real_dtype_refuses_a_channel_that_breaks_hermiticity() {
    let broken = CircuitImpl::W1(non_hermitian_circuit());
    let hermitian = CircuitImpl::W1(random_circuit::<1>(4, 20, 1, true));
    for dtype in [Dtype::F64, Dtype::F32] {
        let message =
            hermiticity_refusal("PauliSum", dtype, &broken).expect("a real dtype refuses");
        assert!(message.contains(dtype.name()) && message.contains("complex128"), "{message}");
    }
    for dtype in [Dtype::C128, Dtype::C64] {
        assert_eq!(hermiticity_refusal("PauliSum", dtype, &broken), None);
    }
    for dtype in [Dtype::F64, Dtype::F32, Dtype::C128, Dtype::C64] {
        assert_eq!(hermiticity_refusal("PauliSum", dtype, &hermitian), None);
    }
}
