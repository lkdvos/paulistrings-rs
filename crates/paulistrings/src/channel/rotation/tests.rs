// ---- support derivation ----
//
// These pin that the support is derived from the generator: a mismatch would silently miscompile the sort order.

#[test]
fn support_is_derived_from_a_weight_one_generator() {
    let rot = PauliRotation::new(PauliString::<1>::z(7), 0.3);
    assert_eq!(Channel::<1>::support(&rot), [1u64 << 7]);
    assert_eq!(rot.weight(), 1);
}

#[test]
fn support_is_derived_from_a_weight_two_generator() {
    let mut gen = PauliString::<1>::z(2);
    gen.mul_assign(&PauliString::<1>::z(5));
    let rot = PauliRotation::new(gen, 0.3);
    assert_eq!(Channel::<1>::support(&rot), [(1u64 << 2) | (1u64 << 5)]);
    assert_eq!(rot.weight(), 2);
}

#[test]
fn a_y_generator_is_one_qubit_of_support_not_two() {
    // Y sets both the x-bit and the z-bit of a single qubit; counting bits instead of qubits would report weight 2 and extract the wrong support.
    let rot = PauliRotation::new(PauliString::<1>::y(11), 0.3);
    assert_eq!(Channel::<1>::support(&rot), [1u64 << 11]);
    assert_eq!(rot.weight(), 1);
}

#[test]
fn support_crosses_a_word_boundary_w2() {
    let mut gen = PauliString::<2>::x(5);
    gen.mul_assign(&PauliString::<2>::z(70));
    gen.mul_assign(&PauliString::<2>::y(64));
    let rot = PauliRotation::new(gen, 0.3);
    // Qubit 70 (word 1, bit 6) lands in mask[1], alongside qubit 64 (bit 0).
    assert_eq!(
        Channel::<2>::support(&rot),
        [1u64 << 5, (1u64 << 0) | (1u64 << 6)]
    );
    assert_eq!(rot.weight(), 3);
}

#[test]
fn support_is_deduplicated_across_x_and_z() {
    // Mixed x/z on overlapping qubits: q3 gets both an X and a Z (making a Y), so it must appear as a single set bit, not double-counted.
    let mut gen = PauliString::<1>::x(3);
    gen.mul_assign(&PauliString::<1>::z(3));
    gen.mul_assign(&PauliString::<1>::x(1));
    let rot = PauliRotation::new(gen, 0.3);
    assert_eq!(Channel::<1>::support(&rot), [(1u64 << 1) | (1u64 << 3)]);
    assert_eq!(rot.weight(), 2);
}

#[test]
fn an_identity_generator_has_empty_support() {
    // Degenerate but representable: exp(-i*theta*I/2) is a global phase, so
    // it commutes with everything and fanout collapses to 1.
    let rot = PauliRotation::new(PauliString::<1>::identity(), 0.3);
    assert_eq!(Channel::<1>::support(&rot), [0u64]);
    assert_eq!(rot.weight(), 0);
}

/// The mask form directly: bit `q` set in `support()` iff qubit `q` is
/// non-identity in the generator (`gen_x[w] | gen_z[w]` per word).
#[test]
fn rotation_support_is_generator_mask() {
    let mut gen = PauliString::<2>::z(3);
    gen.mul_assign(&PauliString::<2>::x(70));
    let rot = PauliRotation::new(gen, 0.7);
    assert_eq!(
        Channel::<2>::support(&rot),
        [gen.x[0] | gen.z[0], gen.x[1] | gen.z[1]]
    );
}

/// `weight()` is the popcount of the generator's support mask, at any generator weight (including above `MAX_LOCAL_SUPPORT`).
#[test]
fn rotation_weight_is_popcount() {
    for n in 0..=5u32 {
        let mut gen = PauliString::<1>::identity();
        for q in 0..n {
            gen.mul_assign(&PauliString::<1>::z(q * 10));
        }
        let rot = PauliRotation::new(gen, 0.1);
        let mask = Channel::<1>::support(&rot);
        let popcount: u32 = mask.iter().map(|w| w.count_ones()).sum();
        assert_eq!(rot.weight(), popcount as usize, "n={n}");
        assert_eq!(rot.weight(), n as usize, "n={n}");
    }
}

#[test]
fn accessors_round_trip_the_generator_and_angle() {
    let gen = PauliString::<2>::y(65);
    let rot = PauliRotation::new(gen, -1.25);
    assert_eq!(rot.generator(), gen);
    assert_eq!(rot.theta(), -1.25);
}

use super::*;
use crate::test_support::{alloc_bufs, approx_eq};

const TOL: f64 = 1e-12;

/// `theta = 0` and the input/generator anticommute: the fanout-2 branch runs but `sin(0) = 0` makes the second term vanish.
#[test]
fn theta_zero_anticommuting_w1() {
    let q = PauliString::<1>::x(0);
    let p = PauliString::<1>::z(0);
    let rot = PauliRotation::new(p, 0.0);
    let c = Complex64::new(2.0, 3.0);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
    let mut buf = OutputBuffer::<1> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    rot.apply(&q.x, &q.z, c, &mut buf);
    assert_eq!(*buf.len, 2);
    assert_eq!(bx[0], q.x);
    assert_eq!(bz[0], q.z);
    assert!(approx_eq(bc[0], c, TOL));
    // Bits of the second term are X * Z = Y; coefficient is 0.
    let y = PauliString::<1>::y(0);
    assert_eq!(bx[1], y.x);
    assert_eq!(bz[1], y.z);
    assert!(approx_eq(bc[1], Complex64::new(0.0, 0.0), TOL));
}

/// `theta = 0` with a commuting generator: fanout-1, output is input.
#[test]
fn theta_zero_commuting_w1() {
    let q = PauliString::<1>::z(0);
    let p = PauliString::<1>::z(0);
    let rot = PauliRotation::new(p, 0.0);
    let c = Complex64::new(2.0, 3.0);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
    let mut buf = OutputBuffer::<1> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    rot.apply(&q.x, &q.z, c, &mut buf);
    assert_eq!(*buf.len, 1);
    assert_eq!(bx[0], q.x);
    assert_eq!(bz[0], q.z);
    assert_eq!(bc[0], c);
}

/// Slice spec: rotation by π around `Z` flips `X → −X` (sign in the coeff).
#[test]
fn pi_z_flips_x_to_minus_x_w1() {
    let q = PauliString::<1>::x(0);
    let p = PauliString::<1>::z(0);
    let rot = PauliRotation::new(p, std::f64::consts::PI);
    let c = Complex64::new(1.0, 0.0);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
    let mut buf = OutputBuffer::<1> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    rot.apply(&q.x, &q.z, c, &mut buf);
    assert_eq!(*buf.len, 2);
    assert_eq!(bx[0], q.x);
    assert_eq!(bz[0], q.z);
    assert!(approx_eq(bc[0], Complex64::new(-1.0, 0.0), TOL));
    let y = PauliString::<1>::y(0);
    assert_eq!(bx[1], y.x);
    assert_eq!(bz[1], y.z);
    assert!(approx_eq(bc[1], Complex64::new(0.0, 0.0), TOL));
}

/// Identity input commutes with every generator: fanout-1, output is input.
#[test]
fn commuting_case_is_fanout_one_w1() {
    let q = PauliString::<1>::identity();
    let p = PauliString::<1>::z(0);
    let rot = PauliRotation::new(p, std::f64::consts::FRAC_PI_4);
    let c = Complex64::new(0.5, 0.25);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
    let mut buf = OutputBuffer::<1> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    rot.apply(&q.x, &q.z, c, &mut buf);
    assert_eq!(*buf.len, 1);
    assert_eq!(bx[0], q.x);
    assert_eq!(bz[0], q.z);
    assert_eq!(bc[0], c);
}

/// Anticommuting case at a generic angle: cos·Q + sin·Y, with both coefficients pinned numerically.
#[test]
fn anticommuting_case_is_fanout_two_w1() {
    let q = PauliString::<1>::x(0);
    let p = PauliString::<1>::z(0);
    let theta = std::f64::consts::FRAC_PI_3;
    let rot = PauliRotation::new(p, theta);
    let c = Complex64::new(1.0, 0.0);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
    let mut buf = OutputBuffer::<1> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    rot.apply(&q.x, &q.z, c, &mut buf);
    assert_eq!(*buf.len, 2);
    assert_eq!(bx[0], q.x);
    assert_eq!(bz[0], q.z);
    assert!(approx_eq(bc[0], Complex64::new(theta.cos(), 0.0), TOL));
    let y = PauliString::<1>::y(0);
    assert_eq!(bx[1], y.x);
    assert_eq!(bz[1], y.z);
    assert!(approx_eq(bc[1], Complex64::new(theta.sin(), 0.0), TOL));
}

/// Catches a sign error in the multiplication direction. With Q=Z, P=X: `cos(π/2)·Z + i sin(π/2) · ZX = i · iY = −Y`.
#[test]
fn pi_over_two_x_rotates_z_to_minus_y_w1() {
    let q = PauliString::<1>::z(0);
    let p = PauliString::<1>::x(0);
    let rot = PauliRotation::new(p, std::f64::consts::FRAC_PI_2);
    let c = Complex64::new(1.0, 0.0);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
    let mut buf = OutputBuffer::<1> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    rot.apply(&q.x, &q.z, c, &mut buf);
    assert_eq!(*buf.len, 2);
    assert_eq!(bx[0], q.x);
    assert_eq!(bz[0], q.z);
    assert!(approx_eq(bc[0], Complex64::new(0.0, 0.0), TOL));
    let y = PauliString::<1>::y(0);
    assert_eq!(bx[1], y.x);
    assert_eq!(bz[1], y.z);
    assert!(approx_eq(bc[1], Complex64::new(-1.0, 0.0), TOL));
}

/// Catches a sign error in the `Phase::I + phase` step. With Q=Y, P=Z: `Y · Z = +iX` (mul_assign delta = 1), so the total phase factor is `i · i = −1`, giving `(X, −sin(π/2)·c) = (X, −c)`.
#[test]
fn phase_from_mul_assign_is_folded_w1() {
    let q = PauliString::<1>::y(0);
    let p = PauliString::<1>::z(0);
    let rot = PauliRotation::new(p, std::f64::consts::FRAC_PI_2);
    let c = Complex64::new(1.0, 0.0);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
    let mut buf = OutputBuffer::<1> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    rot.apply(&q.x, &q.z, c, &mut buf);
    assert_eq!(*buf.len, 2);
    assert_eq!(bx[0], q.x);
    assert_eq!(bz[0], q.z);
    assert!(approx_eq(bc[0], Complex64::new(0.0, 0.0), TOL));
    let xp = PauliString::<1>::x(0);
    assert_eq!(bx[1], xp.x);
    assert_eq!(bz[1], xp.z);
    assert!(approx_eq(bc[1], Complex64::new(-1.0, 0.0), TOL));
}

/// Multi-word: input on word 0, generator on word 1. Disjoint support → commute → fanout-1.
#[test]
fn multi_word_disjoint_support_commutes_w2() {
    let q = PauliString::<2>::x(0);
    let p = PauliString::<2>::z(64);
    let rot = PauliRotation::new(p, std::f64::consts::FRAC_PI_4);
    let c = Complex64::new(1.0, 0.0);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<2>(2);
    let mut buf = OutputBuffer::<2> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    rot.apply(&q.x, &q.z, c, &mut buf);
    assert_eq!(*buf.len, 1);
    assert_eq!(bx[0], q.x);
    assert_eq!(bz[0], q.z);
    assert_eq!(bc[0], c);
}

/// Multi-word: anticommuting bits land in word 1; word 0 stays zero.
#[test]
fn multi_word_anticommute_in_word_1_w2() {
    let q = PauliString::<2>::x(64);
    let p = PauliString::<2>::z(64);
    let theta = std::f64::consts::FRAC_PI_3;
    let rot = PauliRotation::new(p, theta);
    let c = Complex64::new(1.0, 0.0);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<2>(2);
    let mut buf = OutputBuffer::<2> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    rot.apply(&q.x, &q.z, c, &mut buf);
    assert_eq!(*buf.len, 2);
    assert_eq!(bx[0], q.x);
    assert_eq!(bz[0], q.z);
    assert!(approx_eq(bc[0], Complex64::new(theta.cos(), 0.0), TOL));
    let y = PauliString::<2>::y(64);
    assert_eq!(bx[1], y.x);
    assert_eq!(bz[1], y.z);
    assert_eq!(bx[1][0], 0u64);
    assert_eq!(bz[1][0], 0u64);
    assert!(approx_eq(bc[1], Complex64::new(theta.sin(), 0.0), TOL));
}

/// A quarter-turn is an exact Clifford through both prepare paths: one output term, coefficient exactly `±1`.
#[test]
fn quarter_turns_are_fanout_one() {
    use crate::test_support::KeepAll;
    use crate::{propagate, Circuit, Direction, PauliSum};
    use std::f64::consts::{FRAC_PI_2, PI};
    let (x0, z) = (PauliString::<1>::x(0), PauliString::<1>::z);
    let mut zzz = z(0);
    zzz.mul_assign(&z(1));
    zzz.mul_assign(&z(2));
    for gen in [z(0), zzz] {
        let mut y = x0;
        y.mul_assign(&gen);
        for (theta, want, c) in [(FRAC_PI_2, y, 1.0), (PI, x0, -1.0), (-FRAC_PI_2, y, -1.0)] {
            let mut circuit = Circuit::<1>::new(3);
            circuit.push(PauliRotation::new(gen, theta));
            let input = PauliSum::from_strings(&[("XII", Complex64::new(1.0, 0.0))]);
            let out = propagate(&circuit, input, &KeepAll, Direction::Forward);
            assert_eq!(out.len(), 1, "{gen:?} at {theta}");
            assert_eq!(out.get(&want.x, &want.z), Some(Complex64::new(c, 0.0)));
        }
    }
}

/// The engine drives apply repeatedly against the same buffer; back-to-back calls must reuse storage without growing the backing vecs.
#[test]
fn reuse_buffer_across_calls() {
    let cap = 2;
    let mut bx: Vec<[u64; 1]> = vec![[0u64; 1]; cap];
    let mut bz: Vec<[u64; 1]> = vec![[0u64; 1]; cap];
    let mut bc: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); cap];
    let p = PauliString::<1>::z(0);
    let rot = PauliRotation::new(p, std::f64::consts::FRAC_PI_3);
    let q = PauliString::<1>::x(0);
    for _ in 0..3 {
        let mut len = 0usize;
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        rot.apply(&q.x, &q.z, Complex64::new(1.0, 0.0), &mut buf);
        assert_eq!(*buf.len, 2);
    }
    assert_eq!(bx.capacity(), cap);
    assert_eq!(bz.capacity(), cap);
    assert_eq!(bc.capacity(), cap);
}
