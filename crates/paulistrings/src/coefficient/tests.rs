use num_complex::{Complex32, Complex64};

use super::{Coefficient, REAL_TOLERANCE};
use crate::phase::Phase;

fn even_phases_negate_or_keep<C: Coefficient>() {
    let c = C::from_f64(1.5);
    assert_eq!(c.mul_phase(Phase::ONE), c);
    assert_eq!(c.mul_phase(Phase::MINUS_ONE), -c);
    assert_eq!(c.mul_phase(Phase::MINUS_ONE).to_complex64(), Complex64::new(-1.5, 0.0));
}

#[test]
fn even_phases_on_every_type() {
    even_phases_negate_or_keep::<f64>();
    even_phases_negate_or_keep::<f32>();
    even_phases_negate_or_keep::<Complex64>();
    even_phases_negate_or_keep::<Complex32>();
}

#[test]
fn odd_phases_rotate_complex_coefficients() {
    let c = Complex64::new(2.0, 3.0);
    // i·(2 + 3i) = -3 + 2i; -i·(2 + 3i) = 3 - 2i.
    assert_eq!(c.mul_phase(Phase::I), Complex64::new(-3.0, 2.0));
    assert_eq!(c.mul_phase(Phase::MINUS_I), Complex64::new(3.0, -2.0));
    assert_eq!(Complex32::new(2.0, 3.0).mul_phase(Phase::I), Complex32::new(-3.0, 2.0));
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "odd phase on a real coefficient")]
fn odd_phase_on_real_coefficient_panics_in_debug() {
    let _ = 1.0f64.mul_phase(Phase::I);
}

#[test]
fn real_types_reject_imaginary_parts() {
    let imaginary = Complex64::new(0.5, 1e-3);
    assert_eq!(f64::from_complex64(imaginary, REAL_TOLERANCE), None);
    assert_eq!(f32::from_complex64(imaginary, REAL_TOLERANCE), None);
    assert_eq!(f64::from_complex64(Complex64::new(0.5, 1e-14), REAL_TOLERANCE), Some(0.5));
    assert_eq!(
        Complex32::from_complex64(imaginary, REAL_TOLERANCE),
        Some(Complex32::new(0.5, 1e-3))
    );
}

#[test]
fn norm_sqr_and_conj() {
    assert_eq!((-3.0f32).norm_sqr(), 9.0);
    assert_eq!(Complex64::new(3.0, 4.0).norm_sqr(), 25.0);
    assert_eq!(Complex64::new(3.0, 4.0).conj(), Complex64::new(3.0, -4.0));
    assert_eq!(2.0f64.conj(), 2.0);
}

#[test]
fn rotation_scale_uses_the_real_type() {
    let c = Complex32::new(1.0, -2.0) * <Complex32 as Coefficient>::real(0.5);
    assert_eq!(c, Complex32::new(0.5, -1.0));
    assert_eq!(4.0f32 * <f32 as Coefficient>::real(0.25), 1.0);
}
