use num_complex::Complex64;
use proptest::prelude::*;

use crate::coefficient::Coefficient;
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::PauliSum;
use crate::test_support::{assert_terms_close, low_weight_sum, rand_sum};

const I: Complex64 = Complex64::new(0.0, 1.0);

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn real<const W: usize>(sum: &PauliSum<W>) -> PauliSum<W, f64> {
    let mut accumulator = BuildAccumulator::<W, f64>::new(sum.num_qubits());
    for (x, z, coeff) in sum.iter() {
        accumulator.add_term(PauliString { x: *x, z: *z }, coeff.re);
    }
    accumulator.finalize()
}

fn widen<const W: usize, C: Coefficient>(sum: &PauliSum<W, C>) -> PauliSum<W> {
    sum.cast::<Complex64>(0.0).expect("widening never fails")
}

/// The textbook double loop: `P_a·P_b` for every pair, accumulated by key, with `weight(P_a, P_b)` the factor the rule puts on `a·b`.
fn reference<const W: usize, C: Coefficient>(
    a: &PauliSum<W, C>,
    b: &PauliSum<W, C>,
    weight: impl Fn(PauliString<W>, &PauliString<W>) -> (PauliString<W>, Complex64),
) -> PauliSum<W> {
    let mut accumulator = BuildAccumulator::<W>::new(a.num_qubits());
    for (ax, az, ac) in a.iter() {
        for (bx, bz, bc) in b.iter() {
            let (key, factor) =
                weight(PauliString { x: *ax, z: *az }, &PauliString { x: *bx, z: *bz });
            if factor != Complex64::new(0.0, 0.0) {
                accumulator.add_term(key, ac.to_complex64() * bc.to_complex64() * factor);
            }
        }
    }
    accumulator.finalize()
}

fn reference_product<const W: usize, C: Coefficient>(
    a: &PauliSum<W, C>,
    b: &PauliSum<W, C>,
) -> PauliSum<W> {
    reference(a, b, |p, q| {
        let (key, phase) = p.product(q);
        (key, phase.to_complex())
    })
}

fn reference_liouvillian<const W: usize, C: Coefficient>(
    a: &PauliSum<W, C>,
    b: &PauliSum<W, C>,
) -> PauliSum<W> {
    reference(a, b, |p, q| {
        let (key, factor) = p.commutator(q);
        (key, I * factor)
    })
}

fn reference_anticommutator<const W: usize, C: Coefficient>(
    a: &PauliSum<W, C>,
    b: &PauliSum<W, C>,
) -> PauliSum<W> {
    reference(a, b, |p, q| p.anticommutator(q))
}

#[test]
fn the_liouvillian_of_z_on_x_is_minus_two_y() {
    // ZX = iY, so i[Z, X] = i·2iY = -2Y.
    let z = real(&PauliSum::<1>::from_strings(&[("Z", c(1.0, 0.0))]));
    let x = real(&PauliSum::<1>::from_strings(&[("X", c(1.0, 0.0))]));
    let got = z.liouvillian(&x);
    assert_eq!(got.len(), 1);
    assert_eq!(got.get(&[1], &[1]), Some(-2.0));
}

#[test]
fn x_times_z_is_minus_i_y() {
    let x = PauliSum::<2>::from_strings(&[("IX", c(1.0, 0.0))]);
    let z = PauliSum::<2>::from_strings(&[("IZ", c(1.0, 0.0))]);
    let got = x.product(&z);
    assert_eq!(got.len(), 1);
    assert_eq!(got.get(&[0b10, 0], &[0b10, 0]), Some(-I));
}

#[test]
fn the_anticommutator_keeps_commuting_pairs_only() {
    // {X, X} = 2I, {X, Z} = 0, {XI, IZ} = 2·XZ.
    let a = real(&PauliSum::<1>::from_strings(&[("XI", c(1.0, 0.0))]));
    let b = real(&PauliSum::<1>::from_strings(&[
        ("XI", c(3.0, 0.0)),
        ("ZI", c(5.0, 0.0)),
        ("IZ", c(7.0, 0.0)),
    ]));
    let got = a.anticommutator(&b);
    assert_eq!(got.len(), 2);
    assert_eq!(got.identity_coefficient(), 6.0);
    assert_eq!(got.get(&[0b01], &[0b10]), Some(14.0));
}

#[test]
fn a_crossing_word_boundary_keeps_its_phase() {
    // Qubit 70 lives in the second word: i[Y_70, Z_70] = i·2iX = -2X.
    let mut y = BuildAccumulator::<2, f64>::new(80);
    y.add_term(PauliString::<2>::y(70), 1.0);
    let mut z = BuildAccumulator::<2, f64>::new(80);
    z.add_term(PauliString::<2>::z(70), 1.0);
    let got = y.finalize().liouvillian(&z.finalize());
    let x = PauliString::<2>::x(70);
    assert_eq!(got.len(), 1);
    assert_eq!(got.get(&x.x, &x.z), Some(-2.0));
}

#[test]
fn commuting_operands_have_an_empty_liouvillian() {
    let h = real(&low_weight_sum::<1>(200, 12, 3, 5));
    // Pairs (a, b) and (b, a) cancel exactly, but several such pairs sharing a key cancel only to rounding.
    let got = h.liouvillian(&h);
    assert!(got.iter().all(|(_, _, c)| c.abs() < 1e-14), "[H, H] left a term above rounding");
}

#[test]
#[should_panic(expected = "real coefficient type")]
fn a_real_product_is_rejected() {
    let a = real(&rand_sum::<1>(10, 6, 1));
    let _ = a.product(&a);
}

#[test]
#[should_panic(expected = "num_qubits mismatch")]
fn mismatched_qubit_counts_are_rejected() {
    let a = rand_sum::<1>(10, 6, 1);
    let b = rand_sum::<1>(10, 7, 2);
    let _ = a.product(&b);
}

/// Every rule against the double loop, for an operand pair in both orders and both coefficient families.
fn assert_matches_reference<const W: usize>(a: &PauliSum<W>, b: &PauliSum<W>, what: &str) {
    for (left, right, order) in [(a, b, "a·b"), (b, a, "b·a")] {
        let what = format!("{what}, {order}");
        let got = left.product(right);
        got.assert_invariants();
        assert_terms_close(
            &got,
            &reference_product(left, right),
            1e-12,
            &format!("{what}: product"),
        );
        assert_terms_close(
            &left.liouvillian(right),
            &reference_liouvillian(left, right),
            1e-12,
            &format!("{what}: complex liouvillian"),
        );
        let (left, right) = (real(left), real(right));
        let got = left.liouvillian(&right);
        got.assert_invariants();
        assert_terms_close(
            &widen(&got),
            &reference_liouvillian(&left, &right),
            1e-12,
            &format!("{what}: real liouvillian"),
        );
        let got = left.anticommutator(&right);
        got.assert_invariants();
        assert_terms_close(
            &widen(&got),
            &reference_anticommutator(&left, &right),
            1e-12,
            &format!("{what}: real anticommutator"),
        );
    }
}

fn regimes<const W: usize>() {
    let num_qubits = if W == 1 { 40 } else { 100 };
    // A small generator against a many-bucket operand: the Lanczos regime.
    let generator = low_weight_sum::<W>(20, num_qubits, 2, 11);
    let operand = low_weight_sum::<W>(10_000, num_qubits, 5, 12);
    assert!(operand.num_buckets() > 1, "the operand should span several buckets");
    assert_matches_reference(&generator, &operand, "imbalanced");
    // Two single-bucket operands whose pair count asks for more output buckets than either has.
    let left = low_weight_sum::<W>(250, 10, 4, 13);
    let right = low_weight_sum::<W>(250, 10, 4, 14);
    assert_eq!(left.num_buckets(), 1);
    assert!(left.product(&right).num_buckets() > 1, "the pair count should ask for more buckets");
    assert_matches_reference(&left, &right, "balanced");
    // Dense keys on few qubits, so nearly every output key is hit many times.
    let dense = rand_sum::<W>(120, 6, 15);
    assert_matches_reference(&dense, &rand_sum::<W>(100, 6, 16), "colliding");
}

#[test]
fn every_rule_matches_the_double_loop_w1() {
    regimes::<1>();
}

#[test]
fn every_rule_matches_the_double_loop_w2() {
    regimes::<2>();
}

#[test]
fn generator_terms_above_the_index_weight_scan_their_bucket() {
    // Dense keys on 128 qubits weigh about 96, above `SCAN_ABOVE_WEIGHT`.
    let heavy = rand_sum::<2>(20, 128, 31);
    let operand = low_weight_sum::<2>(10_000, 128, 6, 32);
    assert!(heavy
        .iter()
        .all(|(x, z, _)| (0..2).map(|w| (x[w] | z[w]).count_ones()).sum::<u32>() > 64));
    assert_matches_reference(&heavy, &operand, "heavy generator");
}

#[test]
fn an_empty_operand_gives_an_empty_result() {
    let a = rand_sum::<1>(50, 8, 3);
    let empty = PauliSum::<1>::empty(8);
    assert!(a.product(&empty).is_empty());
    assert!(empty.liouvillian(&a).is_empty());
}

#[test]
fn the_output_is_bucketed_for_its_size() {
    let generator = low_weight_sum::<1>(60, 40, 2, 21);
    let operand = low_weight_sum::<1>(20_000, 40, 4, 22);
    let got = generator.liouvillian(&operand);
    let want = crate::test_support::rebucketed_bits(
        got.len(),
        got.hash().bits(),
        crate::test_support::DEFAULT_TARGET_BUCKET_LEN,
        crate::test_support::DEFAULT_MIN_BUCKETS,
    );
    assert_eq!(got.hash().bits(), want);
}

fn assert_laws<const W: usize>(a: &PauliSum<W>, b: &PauliSum<W>) {
    let ab = a.product(b);
    let ba = b.product(a);
    // [A, B] = AB − BA = −i·L_A(B), and {A, B} = AB + BA.
    let mut minus_ba = ba.clone();
    minus_ba.scale(-Complex64::new(1.0, 0.0));
    let mut commutator = a.liouvillian(b);
    commutator.scale(-I);
    assert_terms_close(&drop_tiny(&ab.add(&minus_ba)), &drop_tiny(&commutator), 1e-12, "AB − BA");
    assert_terms_close(
        &drop_tiny(&ab.add(&ba)),
        &drop_tiny(&a.anticommutator(b)),
        1e-12,
        "AB + BA",
    );
    // L_A(B) = −L_B(A).
    let mut swapped = b.liouvillian(a);
    swapped.scale(-Complex64::new(1.0, 0.0));
    assert_terms_close(&a.liouvillian(b), &swapped, 1e-12, "antisymmetry");
}

/// Sums that cancel in a different order leave residues of a few ulps where the other side has an exact zero.
fn drop_tiny<const W: usize>(sum: &PauliSum<W>) -> PauliSum<W> {
    let mut out = sum.clone();
    out.retain(|_, _, c| c.norm() > 1e-13);
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn the_algebraic_laws_hold_w1(seed in any::<u64>(), n_a in 1usize..80, n_b in 1usize..80) {
        assert_laws(&low_weight_sum::<1>(n_a, 9, 3, seed), &low_weight_sum::<1>(n_b, 9, 3, seed ^ 0xABCD));
    }

    #[test]
    fn the_algebraic_laws_hold_w2(seed in any::<u64>(), n_a in 1usize..80, n_b in 1usize..80) {
        assert_laws(&low_weight_sum::<2>(n_a, 70, 3, seed), &low_weight_sum::<2>(n_b, 70, 3, seed ^ 0xABCD));
    }
}
