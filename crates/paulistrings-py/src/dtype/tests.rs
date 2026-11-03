use super::*;

const ALL: [Dtype; 4] = [Dtype::F64, Dtype::F32, Dtype::C128, Dtype::C64];

#[test]
fn promotion_is_numpys_result_type() {
    use Dtype::*;
    let expected = [
        [F64, F64, C128, C128],
        [F64, F32, C128, C64],
        [C128, C128, C128, C128],
        [C128, C64, C128, C64],
    ];
    for (i, a) in ALL.into_iter().enumerate() {
        for (j, b) in ALL.into_iter().enumerate() {
            assert_eq!(a.promote(b), expected[i][j], "{a:?} + {b:?}");
        }
    }
}

#[test]
fn a_scalar_is_real_exactly_for_a_real_type() {
    assert_eq!(Scalar::of(0.5f32), Scalar::Real(0.5));
    assert_eq!(Scalar::of(-2.0f64), Scalar::Real(-2.0));
    assert_eq!(Scalar::of(Complex32::new(1.0, 0.0)), Scalar::Complex(Complex64::new(1.0, 0.0)));
    assert_eq!(Scalar::of(Complex64::new(0.0, 1.0)), Scalar::Complex(Complex64::new(0.0, 1.0)));
}

#[test]
fn formatting_keeps_the_types_own_precision() {
    assert_eq!(0.1f32.format(), "0.1");
    assert_eq!(0.1f64.format(), "0.1");
    assert_eq!(Complex32::new(0.1, -0.5).format(), "(0.1-0.5j)");
    assert_eq!(Complex64::new(0.25, 0.0).format(), "0.25");
}

#[test]
fn every_type_names_its_own_dtype() {
    assert_eq!(<f64 as PyCoefficient>::DTYPE.name(), "float64");
    assert_eq!(<f32 as PyCoefficient>::DTYPE.name(), "float32");
    assert_eq!(<Complex64 as PyCoefficient>::DTYPE.name(), "complex128");
    assert_eq!(<Complex32 as PyCoefficient>::DTYPE.name(), "complex64");
    assert!(ALL.into_iter().filter(|d| d.is_real()).eq([Dtype::F64, Dtype::F32]));
}
