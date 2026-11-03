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
fn formatting_keeps_the_types_own_precision() {
    assert_eq!(0.1f32.format(), "0.1");
    assert_eq!(0.1f64.format(), "0.1");
    assert_eq!(Complex32::new(0.1, -0.5).format(), "(0.1-0.5j)");
    assert_eq!(Complex64::new(0.25, 0.0).format(), "0.25");
}
