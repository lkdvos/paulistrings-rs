use num_complex::Complex64;

use crate::pauli_sum::PauliSum;
use crate::test_support::{assert_terms_close, low_weight_sum};
use crate::truncation::{
    And, BuiltinTruncation, CoefficientThreshold, TopN, TruncationPolicy, WeightCutoff,
};

fn operands() -> (PauliSum<1>, PauliSum<1>) {
    (low_weight_sum::<1>(30, 30, 2, 41), low_weight_sum::<1>(5_000, 30, 5, 42))
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
    let got = h.liouvillian_with(&o, &policy);
    got.assert_invariants();
    assert!(got.len() < h.liouvillian(&o).len());
    assert_terms_close(&got, &truncated_afterwards(h.liouvillian(&o), &policy), 1e-12, "filter");
}

#[test]
fn a_layer_pass_runs_once_on_the_result() {
    let (h, o) = operands();
    let got = h.anticommutator_with(&o, &TopN(500));
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
        &h.product_with(&o, &BuiltinTruncation::Keep),
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
    let _ = h.product_with(&h, &BuiltinTruncation::Keep);
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
    let kept = h.liouvillian_with(&o, &CoefficientThreshold(0.5));
    assert_eq!(kept.len(), 1);
    assert!((kept.get(&[1], &[1]).unwrap() - Complex64::new(-0.8, 0.0)).norm() < 1e-15);
    assert!(h.liouvillian_with(&o, &CoefficientThreshold(0.9)).is_empty());
}
