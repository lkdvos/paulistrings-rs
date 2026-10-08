use super::*;
use num_complex::Complex64;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// `X` and `Z` anticommute, so `[X, Z] = 2·X·Z = -2i·Y` and `{X, Z} = 0`.
#[test]
fn commutator_of_x_and_z_is_minus_two_i_y() {
    fn check<const W: usize>() {
        let x = PauliString::<W>::x(0);
        let z = PauliString::<W>::z(0);

        let (product, coeff) = x.commutator(&z);
        assert_eq!(product, PauliString::<W>::y(0));
        assert_eq!(coeff, Complex64::new(0.0, -2.0));

        let (product, coeff) = x.anticommutator(&z);
        assert_eq!(product, PauliString::<W>::y(0));
        assert_eq!(coeff, ZERO);
    }
    check::<1>();
    check::<2>();
}

/// Reversing the operands flips the commutator's sign: `[Z, X] = 2·Z·X = +2i·Y`.
#[test]
fn commutator_is_antisymmetric_on_x_and_z() {
    let (product, coeff) = PauliString::<1>::z(0).commutator(&PauliString::<1>::x(0));
    assert_eq!(product, PauliString::<1>::y(0));
    assert_eq!(coeff, Complex64::new(0.0, 2.0));
}

/// `X⊗I` and `I⊗X` commute, so `[P, Q] = 0` and `{P, Q} = 2·P·Q = 2·X⊗X`.
#[test]
fn commuting_strings_have_a_zero_commutator_and_twice_the_product() {
    fn check<const W: usize>() {
        let a = PauliString::<W>::x(0);
        let b = PauliString::<W>::x(1);
        let mut xx = PauliString::<W>::x(0);
        xx.x[0] |= 1u64 << 1;

        let (product, coeff) = a.commutator(&b);
        assert_eq!(product, xx);
        assert_eq!(coeff, ZERO);

        let (product, coeff) = a.anticommutator(&b);
        assert_eq!(product, xx);
        assert_eq!(coeff, Complex64::new(2.0, 0.0));
    }
    check::<1>();
    check::<2>();
}

/// Identity commutes with everything, so its anticommutator carries the whole `2·P`.
#[test]
fn identity_anticommutator_is_twice_the_other_string() {
    let id = PauliString::<2>::identity();
    let y = PauliString::<2>::y(64);

    let (product, coeff) = id.commutator(&y);
    assert_eq!(product, y);
    assert_eq!(coeff, ZERO);

    let (product, coeff) = id.anticommutator(&y);
    assert_eq!(product, y);
    assert_eq!(coeff, Complex64::new(2.0, 0.0));
}

/// The same `X`/`Z` case on qubit 64, i.e. entirely in the second word.
#[test]
fn commutator_multi_word() {
    let (product, coeff) = PauliString::<2>::x(64).commutator(&PauliString::<2>::z(64));
    assert_eq!(product, PauliString::<2>::y(64));
    assert_eq!(coeff, Complex64::new(0.0, -2.0));
}

mod props {
    use super::*;
    use proptest::prelude::*;

    fn arb_pauli_w2() -> impl Strategy<Value = PauliString<2>> {
        (any::<u64>(), any::<u64>(), any::<u64>(), any::<u64>()).prop_map(|(x0, x1, z0, z1)| {
            PauliString::<2> {
                x: [x0, x1],
                z: [z0, z1],
            }
        })
    }

    proptest! {
        /// `[P, Q] + {P, Q} = 2·P·Q` — the defining decomposition, and the reason exactly one of the two is nonzero.
        #[test]
        fn commutator_plus_anticommutator_is_twice_the_product(
            a in arb_pauli_w2(),
            b in arb_pauli_w2(),
        ) {
            let (product, phase) = a.mul(&b);
            let (comm_product, comm) = a.commutator(&b);
            let (anti_product, anti) = a.anticommutator(&b);

            prop_assert_eq!(comm_product, product);
            prop_assert_eq!(anti_product, product);
            prop_assert_eq!(comm + anti, phase.to_complex() * 2.0);
            prop_assert!((comm == ZERO) != (anti == ZERO));
        }
    }
}
