//! Integration tests for `PauliString` and `PauliSum`.
//!
//! Placeholder while the core algebra is being implemented (§13 step 1).

use paulistrings::PauliString;

#[test]
fn identity_has_phase_zero_and_zero_bits() {
    let p: PauliString<1> = PauliString::identity();
    assert_eq!(p.phase, 0);
    assert_eq!(p.x, [0u64; 1]);
    assert_eq!(p.z, [0u64; 1]);
}

#[test]
#[ignore = "TODO: enable once PauliString::weight is implemented (§3.1)"]
fn weight_of_identity_is_zero() {
    let p: PauliString<2> = PauliString::identity();
    assert_eq!(p.weight(), 0);
}
