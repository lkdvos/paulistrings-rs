use super::*;
use crate::channel::rotation::PauliRotation;
use crate::circuit::Circuit;
use crate::engine::{propagate, Direction};
use crate::pauli_string::PauliString;
use crate::test_support::{rand_sum_on, KeepAll};
use proptest::prelude::*;

const DELTA: f64 = 0.3;

/// Label index `i` is qubit `i`.
fn sum_of<const W: usize>(terms: &[(&str, f64)]) -> PauliSum<W> {
    let terms: Vec<(&str, Complex64)> = terms
        .iter()
        .map(|&(l, c)| (l, Complex64::new(c, 0.0)))
        .collect();
    PauliSum::<W>::from_strings(&terms)
}

/// `2⁻ⁿ Tr(A† V† A V)` by brute force: `V† A V` materialized through `propagate`, then [`PauliSum::overlap`].
fn materialized<const W: usize>(
    a: &PauliSum<W>,
    sites: &[usize],
    delta: f64,
    axis: RotationAxis,
) -> Complex64 {
    let mut circuit = Circuit::<W>::new(a.num_qubits());
    for &q in sites {
        let g = match axis {
            RotationAxis::Z => PauliString::<W>::z(q as u32),
            RotationAxis::X => PauliString::<W>::x(q as u32),
        };
        circuit.push(PauliRotation::new(g, 2.0 * delta));
    }
    let b = propagate(&circuit, a.clone(), &KeepAll, Direction::Heisenberg);
    a.clone().with_hash(b.hash().clone()).overlap(&b)
}

#[test]
fn single_qubit_hand_values() {
    let c = (2.0 * DELTA).cos();
    for (label, axis, sites, want) in [
        ("X", RotationAxis::Z, 0, c),
        ("Y", RotationAxis::Z, 0, c),
        ("Z", RotationAxis::Z, 0, 1.0),
        ("Z", RotationAxis::X, 0, c),
        ("Y", RotationAxis::X, 0, c),
        ("X", RotationAxis::X, 0, 1.0),
        // A generator off the string's support is invisible.
        ("XI", RotationAxis::Z, 1, 1.0),
    ] {
        let got = sum_of::<1>(&[(label, 1.0)]).rotated_overlap(&[sites], DELTA, axis);
        assert!(
            (got - want).abs() < 1e-15,
            "{label} {axis:?}: {got} vs {want}"
        );
    }
}

/// `XX` and `YY` form one class on sites `{0, 1}` (Z axis), with `R[XX, YY] = R[YY, XX] = sin² 2δ`, so `a XX + b YY` gives `(a² + b²) cos² 2δ + 2ab sin² 2δ`.
/// `XX + YY` commutes with `Z ⊗ Z` rotations and returns its norm `2`; `XY − YX` does too, through the negative `R` entries.
/// On the X axis `ZZ` and `YY` share a class, as do `ZY` and `YZ`.
#[test]
fn two_qubit_class_hand_values() {
    let (s, c) = (2.0 * DELTA).sin_cos();
    let (z, x) = (RotationAxis::Z, RotationAxis::X);
    let cases = [
        ([("XX", 0.7), ("YY", -0.4)], z, 0.65 * c * c - 0.56 * s * s),
        ([("XX", 1.0), ("YY", 1.0)], z, 2.0),
        ([("XY", 1.0), ("YX", -1.0)], z, 2.0),
        ([("XX", 1.0), ("YY", -1.0)], z, 2.0 * (4.0 * DELTA).cos()),
        ([("ZZ", 1.0), ("YY", 1.0)], x, 2.0),
        ([("ZY", 1.0), ("YZ", -1.0)], x, 2.0),
        ([("ZY", 1.0), ("YZ", 1.0)], x, 2.0 * (4.0 * DELTA).cos()),
    ];
    for (terms, axis, want) in cases {
        let got = sum_of::<1>(&terms).rotated_overlap(&[0, 1], DELTA, axis);
        assert!(
            (got - want).abs() < 1e-14,
            "{terms:?} {axis:?}: {got} vs {want}"
        );
    }
}

/// Complex coefficients so the sign convention of every off-diagonal entry shows in the imaginary part too.
fn check_against_materialized<const W: usize>(num_qubits: usize, window: &[u32]) {
    for (seed, sites) in [
        (1u64, vec![window[0] as usize, window[2] as usize]),
        (2, window.iter().map(|&q| q as usize).collect::<Vec<_>>()),
        (3, vec![window[1] as usize]),
    ] {
        let a = rand_sum_on::<W>(300, num_qubits, window, seed);
        for axis in [RotationAxis::Z, RotationAxis::X] {
            let want = materialized(&a, &sites, DELTA, axis);
            let got = a.rotated_overlap_complex(&sites, DELTA, axis);
            assert!(
                (got - want).norm() < 1e-10 * want.norm().max(1.0),
                "W={W} seed {seed} {axis:?} sites {sites:?}: {got} vs {want}",
            );
        }
    }
}

#[test]
fn agrees_with_materializing_the_rotation() {
    check_against_materialized::<1>(8, &[1, 2, 3, 4, 6]);
    check_against_materialized::<2>(70, &[62, 63, 64, 65, 67]);
}

/// With no class of two, the overlap is the histogram's diagonal sum: `aX0 + bZ0 + cY0Y1 + dI` on sites `{0, 1}`.
#[test]
fn histogram_hand_values_and_the_diagonal_limit() {
    let (a, b, c, d) = (0.5, -0.25, 0.75, 0.125);
    let sum = sum_of::<1>(&[("XI", a), ("ZI", b), ("YY", c), ("II", d)]);
    let w = sum.anticommute_histogram(&[0, 1], RotationAxis::Z);
    assert_eq!(w, vec![b * b + d * d, a * a, c * c]);
    let w = sum.anticommute_histogram(&[0, 1], RotationAxis::X);
    assert_eq!(w, vec![a * a + d * d, b * b, c * c]);

    for axis in [RotationAxis::Z, RotationAxis::X] {
        let w = sum.anticommute_histogram(&[0, 1], axis);
        let norm: f64 = w.iter().sum();
        let diag = diagonal_echo(&w, DELTA) * norm;
        let exact = sum.rotated_overlap(&[0, 1], DELTA, axis);
        assert!((diag - exact).abs() < 1e-15, "{axis:?}: {diag} vs {exact}");
    }
    // X_q under Z rotations keeps cos 2δ of itself.
    let w = sum_of::<1>(&[("X", 1.0)]).anticommute_histogram(&[0], RotationAxis::Z);
    assert!((diagonal_echo(&w, DELTA) - (2.0 * DELTA).cos()).abs() < 1e-15);
}

/// Bucketing is invisible to both read-outs, at both widths.
#[test]
fn read_outs_are_partition_independent() {
    use crate::pauli_sum::Gf2Hash;
    let a = rand_sum_on::<2>(400, 70, &[0, 5, 63, 64, 66], 9);
    let sites = [5usize, 63, 64];
    let spread = a.clone().with_hash(Gf2Hash::new(70, 4, 0xABC));
    assert!(spread.num_buckets() > 1);
    for axis in [RotationAxis::Z, RotationAxis::X] {
        let (x, y) = (
            a.rotated_overlap(&sites, DELTA, axis),
            spread.rotated_overlap(&sites, DELTA, axis),
        );
        assert!((x - y).abs() < 1e-12, "{axis:?}: {x} vs {y}");
        let (hx, hy) = (
            a.anticommute_histogram(&sites, axis),
            spread.anticommute_histogram(&sites, axis),
        );
        for (u, v) in hx.iter().zip(&hy) {
            assert!((u - v).abs() < 1e-12, "{axis:?}: {hx:?} vs {hy:?}");
        }
    }
}

#[test]
fn flip_mask_names_the_other_half_of_the_sites() {
    assert_eq!(
        RotationAxis::Z.flip_mask::<2>(&[1, 64]),
        ([0, 0], [0b10, 1])
    );
    assert_eq!(RotationAxis::X.flip_mask::<1>(&[0, 3]), ([0b1001], [0]));
}

#[test]
#[should_panic(expected = "listed twice")]
fn a_repeated_site_is_rejected() {
    sum_of::<1>(&[("XX", 1.0)]).rotated_overlap(&[1, 1], DELTA, RotationAxis::Z);
}

#[test]
#[should_panic(expected = "outside")]
fn an_out_of_range_site_is_rejected() {
    sum_of::<1>(&[("XX", 1.0)]).anticommute_histogram(&[2], RotationAxis::X);
}

proptest! {
    /// At `δ = 0`, `V` is the identity and the echo is `Σ|c|²`.
    #[test]
    fn zero_angle_is_the_squared_norm(seed in 0u64..1000, z_axis in any::<bool>()) {
        let a = rand_sum_on::<1>(120, 8, &[0, 1, 2, 5, 7], seed);
        let axis = if z_axis { RotationAxis::Z } else { RotationAxis::X };
        let norm: f64 = a.iter().map(|(_, _, c)| c.norm_sqr()).sum();
        let got = a.rotated_overlap(&[0, 2, 5, 7], 0.0, axis);
        prop_assert!((got - norm).abs() < 1e-12 * norm.max(1.0), "{} vs {}", got, norm);
    }
}
