use super::*;
use crate::pauli_string::PauliString;
use crate::test_support::raw_outputs;

#[test]
fn support_is_empty() {
    let id = IdentityChannel::new();
    assert_eq!(<IdentityChannel as Channel<1>>::support(&id), [0u64; 1]);
    assert_eq!(<IdentityChannel as Channel<2>>::support(&id), [0u64; 2]);
}

#[test]
fn apply_emits_input_unchanged_w1() {
    // Build XYZ on qubits 0,1,2 by composition.
    let mut input = PauliString::<1>::x(0);
    let _ = input.mul_assign(&PauliString::<1>::y(1));
    let _ = input.mul_assign(&PauliString::<1>::z(2));
    let coeff = Complex64::new(2.0, 3.0);
    for adjoint in [false, true] {
        assert_eq!(
            raw_outputs(&IdentityChannel::new(), adjoint, input, coeff),
            vec![(input, coeff)]
        );
    }
}

#[test]
fn apply_emits_input_unchanged_w2() {
    // X on qubit 70 lands in word 1 — exercises the multi-word case.
    let input = PauliString::<2>::x(70);
    let coeff = Complex64::new(-1.5, 0.25);
    for adjoint in [false, true] {
        assert_eq!(
            raw_outputs(&IdentityChannel::new(), adjoint, input, coeff),
            vec![(input, coeff)]
        );
    }
}
