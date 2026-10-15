use super::*;
use crate::test_support::Xs64;
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
        (any::<u64>(), any::<u64>(), any::<u64>(), any::<u64>())
            .prop_map(|(x0, x1, z0, z1)| PauliString::<2> { x: [x0, x1], z: [z0, z1] })
    }

    proptest! {
        /// `[P, Q] + {P, Q} = 2·P·Q` — the defining decomposition, and the reason exactly one of the two is nonzero.
        #[test]
        fn commutator_plus_anticommutator_is_twice_the_product(
            a in arb_pauli_w2(),
            b in arb_pauli_w2(),
        ) {
            let (product, phase) = a.product(&b);
            let (comm_product, comm) = a.commutator(&b);
            let (anti_product, anti) = a.anticommutator(&b);

            prop_assert_eq!(comm_product, product);
            prop_assert_eq!(anti_product, product);
            prop_assert_eq!(comm + anti, phase.to_complex() * 2.0);
            prop_assert!((comm == ZERO) != (anti == ZERO));
        }

        /// `(a·b)·c = a·(b·c)`, strings and accumulated phases both.
        #[test]
        fn mul_is_associative(
            a in arb_pauli_w2(),
            b in arb_pauli_w2(),
            c in arb_pauli_w2(),
        ) {
            let mut left = a;
            let p1 = left.mul_assign(&b);
            let p2 = left.mul_assign(&c);

            let mut bc = b;
            let q1 = bc.mul_assign(&c);
            let mut right = a;
            let q2 = right.mul_assign(&bc);

            prop_assert_eq!(left.x, right.x);
            prop_assert_eq!(left.z, right.z);
            prop_assert_eq!(p1 + p2, q1 + q2);
        }
    }
}

#[test]
fn qubit_mask_packs_across_words() {
    assert_eq!(qubit_mask::<2>([0, 63, 64, 69], 70), [1 | 1 << 63, 1 | 1 << 5]);
}

#[test]
#[should_panic(expected = "qubit 70 is outside the 70-qubit register")]
fn qubit_mask_rejects_a_qubit_outside_the_register() {
    qubit_mask::<2>([70], 70);
}

#[test]
fn pauli_string_layout_has_no_padding() {
    use std::mem::size_of;
    assert_eq!(size_of::<PauliString<1>>(), 16);
    assert_eq!(size_of::<PauliString<2>>(), 32);
    assert_eq!(size_of::<PauliString<4>>(), 64);
}

#[test]
fn identity_has_zero_bits_and_weight() {
    let p: PauliString<2> = PauliString::identity();
    assert_eq!(p.x, [0u64; 2]);
    assert_eq!(p.z, [0u64; 2]);
    assert_eq!(p.weight(), 0);
}

/// Weight counts qubits, not bits, across words.
#[test]
fn weight_multi_word() {
    let mut p: PauliString<2> = PauliString::identity();
    p.x[0] = 1u64 << 7;
    p.z[1] = 1u64 << 2;
    assert_eq!(p.weight(), 2);
}

#[test]
fn x_constructor_sets_correct_bit() {
    let p = PauliString::<1>::x(3);
    assert_eq!(p.x[0], 1u64 << 3);
    assert_eq!(p.z[0], 0);
    assert_eq!(p.weight(), 1);
}

/// `Y` sets both bits of its qubit and still has weight one.
#[test]
fn y_constructor_sets_both_bits() {
    let p = PauliString::<1>::y(5);
    assert_eq!(p.x[0], 1u64 << 5);
    assert_eq!(p.z[0], 1u64 << 5);
    assert_eq!(p.weight(), 1);
}

#[test]
fn z_constructor_sets_z_bit() {
    let p = PauliString::<1>::z(7);
    assert_eq!(p.x[0], 0);
    assert_eq!(p.z[0], 1u64 << 7);
    assert_eq!(p.weight(), 1);
}

#[test]
fn constructor_crosses_word_boundary() {
    let p = PauliString::<2>::x(64);
    assert_eq!(p.x[0], 0);
    assert_eq!(p.x[1], 1u64 << 0);
    assert_eq!(p.z, [0u64; 2]);
    assert_eq!(p.weight(), 1);
}

#[test]
fn single_qubit_commutation() {
    let (x, y, z) = (PauliString::<1>::x(0), PauliString::<1>::y(0), PauliString::<1>::z(0));
    assert!(!x.commutes_with(&z));
    assert!(!x.commutes_with(&y));
    assert!(x.commutes_with(&x));
    let id = PauliString::<1>::identity();
    assert!(id.commutes_with(&x) && id.commutes_with(&y) && id.commutes_with(&z));
}

/// `X⊗X` against `Z⊗Z`: each qubit anticommutes, so the parity is even and the strings commute.
#[test]
fn xx_commutes_with_zz() {
    let mut xx = PauliString::<1>::x(0);
    xx.x[0] |= 1u64 << 1;
    let mut zz = PauliString::<1>::z(0);
    zz.z[0] |= 1u64 << 1;
    assert!(xx.commutes_with(&zz));
}

/// `X₀` and `Z₆₄` have disjoint support and commute; `X₆₄` and `Z₆₄` anticommute.
#[test]
fn commutes_with_multi_word() {
    let z64 = PauliString::<2>::z(64);
    assert!(PauliString::<2>::x(0).commutes_with(&z64));
    assert!(!PauliString::<2>::x(64).commutes_with(&z64));
}

/// `X·Z = -iY`: bits `(1, 1)` and phase `i³`.
#[test]
fn x_times_z_gives_minus_i_y() {
    let mut p = PauliString::<1>::x(0);
    let phase = p.mul_assign(&PauliString::<1>::z(0));
    assert_eq!((p.x[0], p.z[0]), (1, 1));
    assert_eq!(phase, Phase::MINUS_I);
}

/// `Z·X = iY`.
#[test]
fn z_times_x_gives_plus_i_y() {
    let mut p = PauliString::<1>::z(0);
    let phase = p.mul_assign(&PauliString::<1>::x(0));
    assert_eq!((p.x[0], p.z[0]), (1, 1));
    assert_eq!(phase, Phase::I);
}

/// `X·Y = iZ`.
#[test]
fn x_times_y_gives_i_z() {
    let mut p = PauliString::<1>::x(0);
    let phase = p.mul_assign(&PauliString::<1>::y(0));
    assert_eq!((p.x[0], p.z[0]), (0, 1));
    assert_eq!(phase, Phase::I);
}

/// `X² = Y² = I` with phase one.
#[test]
fn squares_are_the_identity() {
    for p in [PauliString::<1>::x(0), PauliString::<1>::y(0)] {
        let mut q = p;
        let phase = q.mul_assign(&p);
        assert_eq!(q, PauliString::<1>::identity());
        assert_eq!(phase, Phase::ONE);
    }
}

/// `product` returns what `mul_assign` leaves in place, for `X·Z = -iY`.
#[test]
fn product_matches_mul_assign() {
    let (p, phase) = PauliString::<1>::x(0).product(&PauliString::<1>::z(0));
    assert_eq!((p.x[0], p.z[0]), (1, 1));
    assert_eq!(phase, Phase::MINUS_I);
}

/// A caller chains products by adding the returned phases: `i` then `X·Z = -iY` totals `1`.
#[test]
fn mul_phase_accumulates_at_call_site() {
    let mut p = PauliString::<1>::x(0);
    let mut total = Phase::I;
    total += p.mul_assign(&PauliString::<1>::z(0));
    assert_eq!((p.x[0], p.z[0]), (1, 1));
    assert_eq!(total, Phase::ONE);
}

/// `X₆₄·Z₆₄ = -iY₆₄`, entirely in word 1.
#[test]
fn mul_multi_word_no_overflow() {
    let mut p = PauliString::<2>::x(64);
    let phase = p.mul_assign(&PauliString::<2>::z(64));
    assert_eq!((p.x, p.z), ([0, 1], [0, 1]));
    assert_eq!(phase, Phase::MINUS_I);
}

#[test]
fn is_within_zero_qubits_only_identity() {
    assert!(PauliString::<1>::identity().is_within(0));
    assert!(PauliString::<2>::identity().is_within(0));
    assert!(!PauliString::<1>::x(0).is_within(0));
    assert!(!PauliString::<1>::z(0).is_within(0));
}

/// At `num_qubits = 64·W` every bit is in range.
#[test]
fn is_within_full_capacity_accepts_anything() {
    let p = PauliString::<1> { x: [!0u64], z: [!0u64] };
    assert!(p.is_within(64));
    let q = PauliString::<2> { x: [!0u64; 2], z: [!0u64; 2] };
    assert!(q.is_within(128));
}

/// At `num_qubits = 50`, bit 49 is the highest valid one.
#[test]
fn is_within_boundary_inside_word() {
    for p in [PauliString::<1>::x, PauliString::<1>::z, PauliString::<1>::y] {
        assert!(p(49).is_within(50));
        assert!(!p(50).is_within(50));
    }
}

/// At `num_qubits = 64` with `W = 2`, word 0 may be anything and a single bit in word 1 trips it.
#[test]
fn is_within_word_boundary() {
    let p = PauliString::<2> { x: [!0u64, 0], z: [!0u64, 0] };
    assert!(p.is_within(64));
    assert!(!PauliString::<2>::x(64).is_within(64));
    assert!(!PauliString::<2>::z(64).is_within(64));
}

/// At `num_qubits = 100` with `W = 2`, bit 99 is valid and bit 100 is not.
#[test]
fn is_within_partial_in_second_word() {
    for p in [PauliString::<2>::x, PauliString::<2>::z] {
        assert!(p(99).is_within(100));
        assert!(!p(100).is_within(100));
    }
}

/// `commutes_with`'s XOR-fold of the masked words under one `count_ones` agrees with one popcount per word at every width.
/// `parity(a) ^ parity(b) == parity(a ^ b)` is an exact integer identity, so the bar is equality.
#[test]
fn xor_fold_commutes_matches_per_word_popcount() {
    fn naive<const W: usize>(a: &PauliString<W>, b: &PauliString<W>) -> bool {
        let mut parity: u32 = 0;
        for i in 0..W {
            parity ^= (a.x[i] & b.z[i]).count_ones();
            parity ^= (a.z[i] & b.x[i]).count_ones();
        }
        parity & 1 == 0
    }

    fn check<const W: usize>(seed: u64) {
        let mut rng = Xs64::new(seed);
        for _ in 0..2000 {
            let mut p = PauliString::<W>::identity();
            let mut q = PauliString::<W>::identity();
            // Low-weight keys, as weight truncation produces, plus one dense word each.
            for _ in 0..3 {
                let b = rng.next_u64();
                p.x[(b as usize >> 3) % W] |= 1u64 << (b % 64);
                let c = rng.next_u64();
                q.z[(c as usize >> 3) % W] |= 1u64 << (c % 64);
            }
            p.z[0] = rng.next_u64();
            q.x[0] = rng.next_u64();
            assert_eq!(p.commutes_with(&q), naive(&p, &q), "W={W}");
        }
    }

    check::<1>(0xA11CE);
    check::<2>(0xB0B);
    check::<4>(0xC0FFEE);
    check::<8>(0xD00D);
    check::<16>(0xFEED);
}
