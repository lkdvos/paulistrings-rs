use num_complex::{Complex32, Complex64};

use super::{Coefficient, REAL_TOLERANCE};
use crate::phase::Phase;

const PHASES: [Phase; 4] = [Phase::ONE, Phase::I, Phase::MINUS_ONE, Phase::MINUS_I];

#[test]
fn mul_phase_is_multiplication_by_i_to_the_k() {
    let c = Complex64::new(2.0, 3.0);
    for p in PHASES {
        assert_eq!(c.mul_phase(p), p.to_complex() * c);
        let narrow = Complex32::new(2.0, 3.0).mul_phase(p);
        assert_eq!(narrow.to_complex64(), p.to_complex() * c);
    }
    for p in [Phase::ONE, Phase::MINUS_ONE] {
        assert_eq!(1.5f64.mul_phase(p), (p.to_complex() * 1.5).re);
        assert_eq!(1.5f32.mul_phase(p).to_complex64(), p.to_complex() * 1.5);
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "odd phase on a real coefficient")]
fn odd_phase_on_real_coefficient_panics_in_debug() {
    let _ = 1.0f64.mul_phase(Phase::I);
}

#[test]
fn conversions_norms_and_conjugates() {
    let imaginary = Complex64::new(0.5, 1e-3);
    assert_eq!(f64::from_complex64(imaginary, REAL_TOLERANCE), None);
    assert_eq!(f32::from_complex64(imaginary, REAL_TOLERANCE), None);
    assert_eq!(f64::from_complex64(Complex64::new(0.5, 1e-14), REAL_TOLERANCE), Some(0.5));
    assert_eq!(Complex32::from_complex64(imaginary, 0.0), Some(Complex32::new(0.5, 1e-3)));
    assert_eq!((-3.0f32).norm_sqr(), 9.0);
    assert_eq!(Complex64::new(3.0, 4.0).norm_sqr(), 25.0);
    assert_eq!(Coefficient::conj(Complex64::new(3.0, 4.0)), Complex64::new(3.0, -4.0));
    assert_eq!(
        Complex32::new(1.0, -2.0) * <Complex32 as Coefficient>::real(0.5),
        Complex32::new(0.5, -1.0)
    );
}
