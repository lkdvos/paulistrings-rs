//! Integration tests for `PauliString` and `PauliSum`.
//!
//! Placeholder while the core algebra is being implemented (§13 step 1).

use paulistrings::{PauliString, Phase};

#[test]
fn identity_has_zero_bits() {
    let p: PauliString<1> = PauliString::identity();
    assert_eq!(p.x, [0u64; 1]);
    assert_eq!(p.z, [0u64; 1]);
}

#[test]
fn pauli_string_layout_has_no_padding() {
    use std::mem::size_of;
    assert_eq!(size_of::<PauliString<1>>(), 16);
    assert_eq!(size_of::<PauliString<2>>(), 32);
    assert_eq!(size_of::<PauliString<4>>(), 64);
}

#[test]
fn weight_of_identity_is_zero() {
    let p: PauliString<2> = PauliString::identity();
    assert_eq!(p.weight(), 0);
}

#[test]
fn weight_of_single_x_is_one() {
    let mut p: PauliString<1> = PauliString::identity();
    p.x[0] = 1u64 << 3;
    assert_eq!(p.weight(), 1);
}

#[test]
fn weight_of_single_y_is_one() {
    // Y has both x and z bits set on the same qubit; weight counts qubits, not bits.
    let mut p: PauliString<1> = PauliString::identity();
    p.x[0] = 1u64 << 5;
    p.z[0] = 1u64 << 5;
    assert_eq!(p.weight(), 1);
}

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
fn x_anticommutes_with_z_same_qubit() {
    let x = PauliString::<1>::x(0);
    let z = PauliString::<1>::z(0);
    assert!(!x.commutes_with(&z));
}

#[test]
fn x_commutes_with_x() {
    let x = PauliString::<1>::x(0);
    assert!(x.commutes_with(&x));
}

#[test]
fn x_anticommutes_with_y_same_qubit() {
    let x = PauliString::<1>::x(0);
    let y = PauliString::<1>::y(0);
    assert!(!x.commutes_with(&y));
}

#[test]
fn identity_commutes_with_everything() {
    let id = PauliString::<1>::identity();
    assert!(id.commutes_with(&PauliString::<1>::x(0)));
    assert!(id.commutes_with(&PauliString::<1>::y(0)));
    assert!(id.commutes_with(&PauliString::<1>::z(0)));
}

#[test]
fn xx_commutes_with_zz() {
    // X⊗X on qubits {0,1} vs Z⊗Z on qubits {0,1}: each pair anticommutes,
    // so the parity is even and the strings commute.
    let mut xx = PauliString::<1>::x(0);
    xx.x[0] |= 1u64 << 1;
    let mut zz = PauliString::<1>::z(0);
    zz.z[0] |= 1u64 << 1;
    assert!(xx.commutes_with(&zz));
}

#[test]
fn x_times_z_gives_minus_i_y() {
    // X·Z = -iY  →  bits (1,1) and phase i^3 = -i.
    let mut p = PauliString::<1>::x(0);
    let phase = p.mul_assign(&PauliString::<1>::z(0));
    assert_eq!(p.x[0], 1);
    assert_eq!(p.z[0], 1);
    assert_eq!(phase, Phase::MINUS_I);
}

#[test]
fn z_times_x_gives_plus_i_y() {
    // Z·X = iY  →  bits (1,1) and phase i.
    let mut p = PauliString::<1>::z(0);
    let phase = p.mul_assign(&PauliString::<1>::x(0));
    assert_eq!(p.x[0], 1);
    assert_eq!(p.z[0], 1);
    assert_eq!(phase, Phase::I);
}

#[test]
fn y_squared_is_identity() {
    let mut p = PauliString::<1>::y(0);
    let phase = p.mul_assign(&PauliString::<1>::y(0));
    assert_eq!(p.x[0], 0);
    assert_eq!(p.z[0], 0);
    assert_eq!(phase, Phase::ONE);
}

#[test]
fn x_squared_is_identity() {
    let mut p = PauliString::<1>::x(0);
    let phase = p.mul_assign(&PauliString::<1>::x(0));
    assert_eq!(p.x[0], 0);
    assert_eq!(p.z[0], 0);
    assert_eq!(phase, Phase::ONE);
}

#[test]
fn x_times_y_gives_i_z() {
    // X·Y = iZ  →  bits (0,1) and phase i.
    let mut p = PauliString::<1>::x(0);
    let phase = p.mul_assign(&PauliString::<1>::y(0));
    assert_eq!(p.x[0], 0);
    assert_eq!(p.z[0], 1);
    assert_eq!(phase, Phase::I);
}

#[test]
fn mul_value_returning_matches_in_place() {
    // The `mul` value-returning variant produces the same (string, phase)
    // pair as the in-place form, for `X · Z = -iY`.
    let (p, phase) = PauliString::<1>::x(0).mul(&PauliString::<1>::z(0));
    assert_eq!(p.x[0], 1);
    assert_eq!(p.z[0], 1);
    assert_eq!(phase, Phase::MINUS_I);
}

#[test]
fn mul_phase_accumulates_at_call_site() {
    // Caller chains two multiplications and combines the returned phases.
    // With a pre-existing factor of `i`, then `X · Z = -iY` (phase -i):
    // total phase = i + (-i) = 1 (Phase::ONE), result Y bits.
    let mut p = PauliString::<1>::x(0);
    let mut total = Phase::I;
    total += p.mul_assign(&PauliString::<1>::z(0));
    assert_eq!(p.x[0], 1);
    assert_eq!(p.z[0], 1);
    assert_eq!(total, Phase::ONE);
}

#[test]
fn mul_multi_word_no_overflow() {
    // X on qubit 64 (word 1) times Z on qubit 64: should give -iY on word 1.
    let mut p = PauliString::<2>::x(64);
    let phase = p.mul_assign(&PauliString::<2>::z(64));
    assert_eq!(p.x[0], 0);
    assert_eq!(p.z[0], 0);
    assert_eq!(p.x[1], 1);
    assert_eq!(p.z[1], 1);
    assert_eq!(phase, Phase::MINUS_I);
}

#[test]
fn commutes_with_multi_word() {
    // X on qubit 0, Z on qubit 64 (in word 1) — disjoint support, must commute.
    let x0 = PauliString::<2>::x(0);
    let z64 = PauliString::<2>::z(64);
    assert!(x0.commutes_with(&z64));
    // X on qubit 64 vs Z on qubit 64 — anticommute.
    let x64 = PauliString::<2>::x(64);
    assert!(!x64.commutes_with(&z64));
}

#[test]
fn is_within_identity_holds_for_any_count() {
    let id1 = PauliString::<1>::identity();
    assert!(id1.is_within(0));
    assert!(id1.is_within(1));
    assert!(id1.is_within(64));
    let id2 = PauliString::<2>::identity();
    assert!(id2.is_within(0));
    assert!(id2.is_within(50));
    assert!(id2.is_within(128));
}

#[test]
fn is_within_zero_qubits_only_identity() {
    let id = PauliString::<1>::identity();
    assert!(id.is_within(0));
    let x0 = PauliString::<1>::x(0);
    assert!(!x0.is_within(0));
    let z0 = PauliString::<1>::z(0);
    assert!(!z0.is_within(0));
}

#[test]
fn is_within_full_capacity_accepts_anything() {
    // num_qubits = 64*W: every bit is in range.
    let mut p: PauliString<1> = PauliString::identity();
    p.x[0] = !0u64;
    p.z[0] = !0u64;
    assert!(p.is_within(64));
    let mut q: PauliString<2> = PauliString::identity();
    q.x = [!0u64; 2];
    q.z = [!0u64; 2];
    assert!(q.is_within(128));
}

#[test]
fn is_within_boundary_inside_word() {
    // num_qubits = 50 → bit 49 is the highest valid; bit 50 is the first invalid.
    assert!(PauliString::<1>::x(49).is_within(50));
    assert!(!PauliString::<1>::x(50).is_within(50));
    assert!(PauliString::<1>::z(49).is_within(50));
    assert!(!PauliString::<1>::z(50).is_within(50));
    assert!(PauliString::<1>::y(49).is_within(50));
    assert!(!PauliString::<1>::y(50).is_within(50));
}

#[test]
fn is_within_word_boundary() {
    // num_qubits = 64 with W=2: word 0 may be anything, word 1 must be empty.
    let mut p: PauliString<2> = PauliString::identity();
    p.x[0] = !0u64;
    p.z[0] = !0u64;
    assert!(p.is_within(64));

    // A single bit in word 1 trips it.
    assert!(!PauliString::<2>::x(64).is_within(64));
    assert!(!PauliString::<2>::z(64).is_within(64));
}

#[test]
fn is_within_partial_in_second_word() {
    // num_qubits = 100 with W=2: bit 99 ok, bit 100 not.
    assert!(PauliString::<2>::x(99).is_within(100));
    assert!(!PauliString::<2>::x(100).is_within(100));
    assert!(PauliString::<2>::z(99).is_within(100));
    assert!(!PauliString::<2>::z(100).is_within(100));
}

mod props {
    use super::PauliString;
    use proptest::prelude::*;

    fn arb_pauli_w2() -> impl Strategy<Value = PauliString<2>> {
        (any::<u64>(), any::<u64>(), any::<u64>(), any::<u64>()).prop_map(
            |(x0, x1, z0, z1)| PauliString::<2> {
                x: [x0, x1],
                z: [z0, z1],
            },
        )
    }

    proptest! {
        #[test]
        fn mul_is_associative(
            a in arb_pauli_w2(),
            b in arb_pauli_w2(),
            c in arb_pauli_w2(),
        ) {
            // Left-associated: ((a·b)·c). Track cumulative phase at each step.
            let mut left = a;
            let p1 = left.mul_assign(&b);
            let p2 = left.mul_assign(&c);
            let left_phase = p1 + p2;

            // Right-associated: (a·(b·c)).
            let mut bc = b;
            let q1 = bc.mul_assign(&c);
            let mut right = a;
            let q2 = right.mul_assign(&bc);
            let right_phase = q1 + q2;

            prop_assert_eq!(left.x, right.x);
            prop_assert_eq!(left.z, right.z);
            prop_assert_eq!(left_phase, right_phase);
        }
    }
}
