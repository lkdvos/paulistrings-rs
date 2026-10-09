use super::BuiltinTruncation as T;
use super::*;
use crate::engine::partitioned::transport::InProcessTransport;
use crate::test_support::{and, assert_same_terms, or, rand_sum_real, KeepAll};
use crate::truncation::{And, Or};

/// Weights 0..3 on one word, and coefficients on both sides of 0.1 and 0.5.
fn grid() -> Vec<([u64; 1], [u64; 1], Complex64)> {
    let keys = [
        ([0u64], [0u64]),
        ([1], [0]),
        ([0b01], [0b10]),
        ([0b011], [0b110]),
    ];
    let cs = [0.0, 0.05, 0.1, 0.3, 0.5, 2.0].map(|r| Complex64::new(r, -r / 3.0));
    keys.iter()
        .flat_map(|&(x, z)| cs.iter().map(move |&c| (x, z, c)))
        .collect()
}

fn assert_keeps_like<P: TruncationPolicy<1>>(tree: &T, builtin: &P, what: &str) {
    for (x, z, c) in grid() {
        assert_eq!(
            <T as TruncationPolicy<1>>::keep_term(tree, &x, &z, c),
            builtin.keep_term(&x, &z, c),
            "{what}: x={x:?} z={z:?} c={c}"
        );
    }
    assert_eq!(
        <T as TruncationPolicy<1>>::finalizes_layer(tree),
        builtin.finalizes_layer(),
        "{what}: finalizes_layer"
    );
}

#[test]
fn keep_term_and_the_layer_hint_equal_the_builtin_each_variant_names() {
    assert_keeps_like(&T::Keep, &KeepAll, "keep");
    for eps in [-1.0, 0.0, 0.1, 0.5] {
        assert_keeps_like(&T::Coefficient(eps), &CoefficientThreshold(eps), "coeff");
    }
    for k in [0u32, 1, 2, 3] {
        assert_keeps_like(&T::Weight(k), &WeightCutoff(k), "weight");
    }
    assert_keeps_like(&T::TopN(3), &TopN(3), "topn");
    assert_keeps_like(&T::ApproxTopN(3), &ApproxTopN(3), "approx");
    assert_keeps_like(
        &and(T::Coefficient(0.1), T::Weight(1)),
        &And(CoefficientThreshold(0.1), WeightCutoff(1)),
        "and",
    );
    assert_keeps_like(
        &or(T::Coefficient(0.5), T::Weight(0)),
        &Or(CoefficientThreshold(0.5), WeightCutoff(0)),
        "or",
    );
    assert_keeps_like(
        &or(and(T::Coefficient(0.1), T::ApproxTopN(9)), T::Weight(1)),
        &Or(
            And(CoefficientThreshold(0.1), ApproxTopN(9)),
            WeightCutoff(1),
        ),
        "nested",
    );
}

/// `true` iff a `TopN`/`ApproxTopN` is reachable through `And` alone.
#[test]
fn finalizes_layer_truth_table() {
    let f = |t: &T| <T as TruncationPolicy<2>>::finalizes_layer(t);
    assert!(!f(&T::Keep));
    assert!(!f(&T::Coefficient(1e-3)));
    assert!(!f(&T::Weight(4)));
    assert!(f(&T::TopN(4)));
    assert!(f(&T::ApproxTopN(4)));
    assert!(f(&and(T::Coefficient(1e-3), T::ApproxTopN(4))));
    assert!(f(&and(T::ApproxTopN(4), T::Weight(2))));
    assert!(f(&and(T::ApproxTopN(4), T::ApproxTopN(2))));
    assert!(f(&and(T::Keep, and(T::Coefficient(0.1), T::TopN(3)))));
    assert!(!f(&and(T::Coefficient(1e-3), T::Weight(2))));
    assert!(!f(&or(T::ApproxTopN(4), T::Coefficient(1e-3))));
    assert!(!f(&or(T::TopN(4), T::ApproxTopN(4))));
    assert!(!f(&and(or(T::TopN(4), T::Keep), T::Coefficient(0.5))));
    assert!(f(&T::from(CollapseSample::new(4, 1))));
    assert!(f(&and(
        T::Coefficient(1e-3),
        T::from(CollapseSample::new(4, 1))
    )));
    assert!(!f(&or(
        T::from(CollapseSample::new(4, 1)),
        T::Coefficient(1e-3)
    )));
}

#[test]
fn every_builtin_converts_node_for_node() {
    let tree = and(T::Coefficient(1e-3), or(T::ApproxTopN(7), T::Weight(2)));
    let builtin = And(
        CoefficientThreshold(1e-3),
        Or(ApproxTopN(7), WeightCutoff(2)),
    );
    assert_eq!(T::from(&builtin), tree);
    assert_eq!(T::from(builtin), tree);
    assert_eq!(T::from(TopN(10)), T::TopN(10));
    assert_eq!(T::from(&tree), tree);
    assert_eq!(T::from(KeepAll), T::Keep);
}

/// A collapse through the tree draws the string the builtin draws at the same pass, and the handle is shared rather than copied by `Clone`.
#[test]
fn collapse_sample_shares_one_trajectory_with_the_builtin() {
    let input = rand_sum_real::<1>(200, 16, 0xC011);
    let mut want = input.clone();
    CollapseSample::new(10, 7).finalize_layer(&mut want);
    let tree = and(T::Coefficient(0.0), T::from(CollapseSample::new(10, 7)));
    let twin = tree.clone();
    let mut got = input.clone();
    <T as TruncationPolicy<1>>::finalize_layer(&tree, &mut got);
    assert_same_terms(&got, &want, "host");
    assert!(tree.contains_collapse_sample());
    assert!(!tree.contains_exact_top_n());
    let T::And(_, leaf) = &twin else {
        unreachable!()
    };
    let T::CollapseSample(s) = &**leaf else {
        unreachable!()
    };
    assert_eq!(s.collapses(), 1, "the clone shares the pass counter");

    let group = InProcessTransport::group(1);
    let mut got = input.clone();
    let tree = T::from(CollapseSample::new(10, 7));
    <T as TruncationPolicy<1>>::finalize_layer_partitioned(&tree, &mut got, &group[0]);
    assert_same_terms(&got, &want, "partitioned P=1");
}

#[test]
fn contains_exact_top_n_looks_through_or() {
    assert!(T::TopN(1).contains_exact_top_n());
    assert!(or(T::Coefficient(0.1), T::TopN(1)).contains_exact_top_n());
    assert!(and(T::Keep, and(T::TopN(1), T::Weight(2))).contains_exact_top_n());
    assert!(!and(T::ApproxTopN(1), T::Weight(2)).contains_exact_top_n());
}

/// The layer pass, host and partitioned at `P = 1, 2`, equals the builtin composition's.
#[test]
fn finalize_layer_equals_the_builtin_composition() {
    let input = rand_sum_real::<1>(1500, 32, 0xB117);
    let tree = and(
        T::Coefficient(1e-3),
        and(T::ApproxTopN(900), T::ApproxTopN(400)),
    );
    let builtin = And(
        CoefficientThreshold(1e-3),
        And(ApproxTopN(900), ApproxTopN(400)),
    );
    let mut want = input.clone();
    builtin.finalize_layer(&mut want);
    assert!(want.len() < input.len(), "the fixture must truncate");
    let mut got = input.clone();
    <T as TruncationPolicy<1>>::finalize_layer(&tree, &mut got);
    assert_same_terms(&got, &want, "host");

    let ored = or(T::ApproxTopN(5), T::Coefficient(0.5));
    let mut untouched = input.clone();
    <T as TruncationPolicy<1>>::finalize_layer(&ored, &mut untouched);
    assert_eq!(untouched.len(), input.len(), "Or runs no layer pass");

    let group = InProcessTransport::group(1);
    let mut got = input.clone();
    <T as TruncationPolicy<1>>::finalize_layer_partitioned(&tree, &mut got, &group[0]);
    assert_same_terms(&got, &want, "partitioned P=1");
}

#[test]
#[should_panic(expected = "not yet supported")]
fn a_reached_top_n_panics_above_one_partition() {
    let group = InProcessTransport::group(2);
    let mut sum = rand_sum_real::<1>(10, 32, 0x1);
    let tree = and(T::Coefficient(0.1), T::TopN(3));
    <T as TruncationPolicy<1>>::finalize_layer_partitioned(&tree, &mut sum, &group[0]);
}

/// `supports_partitioned` follows the tree: an exact `TopN` under `And` refuses, one under `Or` never runs.
#[test]
fn supports_partitioned_reflects_the_tree() {
    let supports = |tree: &T| <T as TruncationPolicy<1>>::supports_partitioned(tree);
    assert!(supports(&T::Keep));
    assert!(supports(&and(T::Coefficient(0.1), T::ApproxTopN(3))));
    assert!(supports(&or(T::Coefficient(0.1), T::TopN(3))));
    assert!(!supports(&T::TopN(3)));
    assert!(!supports(&and(
        T::Coefficient(0.1),
        and(T::Weight(2), T::TopN(3))
    )));
}
