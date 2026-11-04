use num_complex::Complex64;

use crate::pauli_sum::PauliSum;
use crate::test_support::{assert_terms_close, drop_tiny, low_weight_sum};
use crate::truncation::{
    And, BuiltinTruncation, CoefficientThreshold, TopN, TruncationPolicy, WeightCutoff,
};

fn operands() -> (PauliSum<1>, PauliSum<1>) {
    (low_weight_sum::<1>(30, 30, 2, 41), low_weight_sum::<1>(5_000, 30, 5, 42))
}

fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

fn zero() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

/// `pairing(h, o)` under `policy` into an empty destination.
fn fresh(
    add: impl FnOnce(&mut PauliSum<1>, &PauliSum<1>, &PauliSum<1>),
    h: &PauliSum<1>,
    o: &PauliSum<1>,
) -> PauliSum<1> {
    let mut out = PauliSum::empty(h.num_qubits());
    add(&mut out, h, o);
    out
}

/// The untruncated result with `policy` applied afterwards: its filter on every term, then its layer pass.
fn truncated_afterwards<P: TruncationPolicy<1>>(mut sum: PauliSum<1>, policy: &P) -> PauliSum<1> {
    sum.retain(|x, z, c| policy.keep_term(x, z, c));
    if policy.finalizes_layer() {
        policy.finalize_layer(&mut sum);
    }
    sum
}

#[test]
fn a_filter_keeps_exactly_the_terms_it_accepts() {
    let (h, o) = operands();
    let policy = And(CoefficientThreshold(0.5), WeightCutoff(6));
    let got = fresh(|y, h, o| y.add_liouvillian(h, o, one(), zero(), &policy), &h, &o);
    got.assert_invariants();
    assert!(got.len() < h.liouvillian(&o).len());
    assert_terms_close(&got, &truncated_afterwards(h.liouvillian(&o), &policy), 1e-12, "filter");
}

#[test]
fn a_layer_pass_runs_once_on_the_result() {
    let (h, o) = operands();
    let got = fresh(|y, h, o| y.add_anticommutator(h, o, one(), zero(), &TopN(500)), &h, &o);
    got.assert_invariants();
    assert_eq!(got.len(), 500);
    assert_terms_close(
        &got,
        &truncated_afterwards(h.anticommutator(&o), &TopN(500)),
        1e-12,
        "top-n",
    );
}

#[test]
fn keep_leaves_the_product_untouched() {
    let (h, o) = operands();
    assert_terms_close(
        &fresh(|y, h, o| y.add_product(h, o, one(), zero(), &BuiltinTruncation::Keep), &h, &o),
        &h.product(&o),
        1e-15,
        "keep",
    );
}

#[test]
#[should_panic(expected = "real coefficient type")]
fn a_real_product_with_a_policy_is_rejected() {
    let (h, _) = operands();
    let h = h.cast::<f64>(f64::INFINITY).expect("any tolerance admits");
    let mut y = PauliSum::<1, f64>::empty(h.num_qubits());
    y.add_product(&h, &h, 1.0, 0.0, &BuiltinTruncation::Keep);
}

#[test]
fn the_threshold_sees_summed_coefficients() {
    // i[Z, 0.2·X] = -0.4·Y and i[X, -0.2·Z] = -0.4·Y: each below 0.5, summed to -0.8.
    let h = PauliSum::<1>::from_strings(&[
        ("Z", Complex64::new(1.0, 0.0)),
        ("X", Complex64::new(1.0, 0.0)),
    ]);
    let o = PauliSum::<1>::from_strings(&[
        ("X", Complex64::new(0.2, 0.0)),
        ("Z", Complex64::new(-0.2, 0.0)),
    ]);
    let threshold = |t: f64| {
        fresh(|y, h, o| y.add_liouvillian(h, o, one(), zero(), &CoefficientThreshold(t)), &h, &o)
    };
    let kept = threshold(0.5);
    assert_eq!(kept.len(), 1);
    assert!((kept.get(&[1], &[1]).unwrap() - Complex64::new(-0.8, 0.0)).norm() < 1e-15);
    assert!(threshold(0.9).is_empty());
}

#[test]
fn the_destination_is_scaled_and_summed_in() {
    let (h, o) = operands();
    let (alpha, beta) = (Complex64::new(0.5, -0.25), Complex64::new(-1.5, 0.0));
    let y = low_weight_sum::<1>(4_000, 30, 5, 43);
    let mut got = y.clone();
    got.add_liouvillian(&h, &o, alpha, beta, &BuiltinTruncation::Keep);
    got.assert_invariants();
    let mut want = h.liouvillian(&o);
    want.scale(alpha);
    let mut scaled = y.clone();
    scaled.scale(beta);
    let want = want.add(&scaled);
    assert_terms_close(&drop_tiny(&got), &drop_tiny(&want), 1e-12, "alpha L + beta y");
}

#[test]
fn a_destination_under_other_rows_or_bucket_counts_is_realigned() {
    let (h, o) = operands();
    let y = low_weight_sum::<1>(3_000, 30, 5, 44);
    let mut want = h.anticommutator(&o);
    want = want.add(&y);
    for mut dest in [y.clone().with_hash(crate::Gf2Hash::new(30, 1, 7)), y.clone()] {
        dest.refine();
        dest.add_anticommutator(&h, &o, one(), one(), &BuiltinTruncation::Keep);
        dest.assert_invariants();
        assert_terms_close(&drop_tiny(&dest), &drop_tiny(&want), 1e-12, "realigned");
    }
}

#[test]
fn beta_zero_ignores_the_destination() {
    let (h, o) = operands();
    let mut y = low_weight_sum::<1>(100, 30, 3, 45);
    y.scale(Complex64::new(f64::NAN, 0.0));
    y.add_liouvillian(&h, &o, one(), zero(), &BuiltinTruncation::Keep);
    assert_terms_close(&y, &h.liouvillian(&o), 1e-15, "beta = 0");
}

#[test]
fn the_filter_sees_the_destination_in_the_sum() {
    // i[Z, X] = -2·Y; a destination holding 1.9·Y leaves -0.1·Y, below the threshold, and its own 0.3·X stays.
    let z = PauliSum::<1>::from_strings(&[("Z", one())]);
    let x = PauliSum::<1>::from_strings(&[("X", one())]);
    let mut y = PauliSum::<1>::from_strings(&[
        ("Y", Complex64::new(1.9, 0.0)),
        ("X", Complex64::new(0.3, 0.0)),
    ]);
    y.add_liouvillian(&z, &x, one(), one(), &CoefficientThreshold(0.2));
    assert_eq!(y.len(), 1);
    assert!((y.get(&[1], &[0]).unwrap() - Complex64::new(0.3, 0.0)).norm() < 1e-15);
}
