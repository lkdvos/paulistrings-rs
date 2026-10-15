use super::*;

#[test]
fn constants_have_expected_exponents() {
    assert_eq!(Phase::ONE.exponent(), 0);
    assert_eq!(Phase::I.exponent(), 1);
    assert_eq!(Phase::MINUS_ONE.exponent(), 2);
    assert_eq!(Phase::MINUS_I.exponent(), 3);
}

#[test]
fn new_reduces_mod_4() {
    assert_eq!(Phase::new(0), Phase::ONE);
    assert_eq!(Phase::new(1), Phase::I);
    assert_eq!(Phase::new(4), Phase::ONE);
    assert_eq!(Phase::new(5), Phase::I);
    assert_eq!(Phase::new(255), Phase::MINUS_I); // 255 & 3 == 3
}

#[test]
fn to_complex_matches_i_powers() {
    assert_eq!(Phase::ONE.to_complex(), Complex64::new(1.0, 0.0));
    assert_eq!(Phase::I.to_complex(), Complex64::new(0.0, 1.0));
    assert_eq!(Phase::MINUS_ONE.to_complex(), Complex64::new(-1.0, 0.0));
    assert_eq!(Phase::MINUS_I.to_complex(), Complex64::new(0.0, -1.0));
}

#[test]
fn apply_agrees_with_to_complex_times_c() {
    let c = Complex64::new(2.0, 3.0);
    for p in [Phase::ONE, Phase::I, Phase::MINUS_ONE, Phase::MINUS_I] {
        assert_eq!(p.apply(c), p.to_complex() * c);
    }
}

#[test]
fn add_wraps_mod_4() {
    assert_eq!(Phase::I + Phase::I, Phase::MINUS_ONE);
    assert_eq!(Phase::MINUS_ONE + Phase::I, Phase::MINUS_I);
    assert_eq!(Phase::MINUS_I + Phase::I, Phase::ONE);
    assert_eq!(Phase::MINUS_I + Phase::MINUS_I, Phase::MINUS_ONE);
}

#[test]
fn add_assign_wraps_mod_4() {
    let mut p = Phase::I;
    p += Phase::I;
    assert_eq!(p, Phase::MINUS_ONE);
    p += Phase::MINUS_I;
    assert_eq!(p, Phase::I);
}

#[test]
fn repr_transparent_size_is_one_byte() {
    use std::mem::size_of;
    assert_eq!(size_of::<Phase>(), 1);
}
