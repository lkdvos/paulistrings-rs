use super::*;
use paulistrings::CollapseSample;
use std::any::type_name_of_val;
use BuiltinTruncation as T;

/// The filter a tree lowers to, `None` for the tree itself.
fn filter_of(tree: &BuiltinTruncation) -> Option<String> {
    match Lowered::of(tree) {
        Lowered::Coefficient(lowered) => Some(format!("{:?}", lowered.filter)),
        Lowered::CoefficientAndWeight(lowered) => Some(format!("{:?}", lowered.filter)),
        Lowered::Tree(_) => None,
    }
}

fn and(a: BuiltinTruncation, b: BuiltinTruncation) -> BuiltinTruncation {
    T::And(Box::new(a), Box::new(b))
}

fn sampler() -> BuiltinTruncation {
    T::from(CollapseSample::new(4, 7))
}

#[test]
fn trees_lower_to_their_conjunction_of_filters() {
    let keep_all = Some("CoefficientThreshold(-1.0)".to_string());
    for passes_only in
        [T::Keep, T::TopN(9), T::ApproxTopN(9), sampler(), and(T::ApproxTopN(9), sampler())]
    {
        assert_eq!(filter_of(&passes_only), keep_all, "{passes_only:?}");
    }
    let coefficient = Some("CoefficientThreshold(0.001)".to_string());
    for tree in [
        T::Coefficient(1e-3),
        and(T::Coefficient(1e-3), T::ApproxTopN(9)),
        and(T::TopN(9), T::Coefficient(1e-3)),
        and(sampler(), and(T::Keep, T::Coefficient(1e-3))),
    ] {
        assert_eq!(filter_of(&tree), coefficient, "{tree:?}");
    }
    assert_eq!(
        filter_of(&T::Weight(3)).as_deref(),
        Some("And(CoefficientThreshold(-1.0), WeightCutoff(3))")
    );
    assert_eq!(
        filter_of(&and(and(T::Weight(3), T::ApproxTopN(9)), T::Coefficient(1e-3))).as_deref(),
        Some("And(CoefficientThreshold(0.001), WeightCutoff(3))")
    );
}

#[test]
fn trees_without_a_lowered_filter_run_as_the_tree() {
    for tree in [
        T::or(T::Coefficient(1e-3), T::Weight(3)),
        and(T::or(T::Coefficient(1e-3), T::Weight(3)), T::ApproxTopN(9)),
        and(T::Coefficient(1e-3), T::Coefficient(1e-4)),
        and(T::Weight(3), T::Weight(4)),
    ] {
        assert_eq!(filter_of(&tree), None, "{tree:?}");
    }
}

#[test]
fn the_layer_passes_and_partitioned_support_are_the_trees() {
    let tree = and(T::Coefficient(1e-3), T::TopN(9));
    let Lowered::Coefficient(lowered) = Lowered::of(&tree) else {
        panic!("a coefficient and a pass lower");
    };
    assert!(<WithFilter<_> as TruncationPolicy<1>>::finalizes_layer(&lowered));
    assert!(!<WithFilter<_> as TruncationPolicy<1>>::supports_partitioned(&lowered));
    let lowered_keep = Lowered::of(&T::Keep);
    let Lowered::Coefficient(lowered_keep) = lowered_keep else {
        panic!("Keep lowers");
    };
    assert!(!<WithFilter<_> as TruncationPolicy<1>>::finalizes_layer(&lowered_keep));
}

#[test]
fn the_macro_runs_each_arm_as_its_own_type() {
    let name =
        |tree: &BuiltinTruncation| with_lowered_policy!(tree, |policy| type_name_of_val(policy));
    assert!(name(&T::ApproxTopN(9))
        .contains("WithFilter<'_, paulistrings::truncation::builtin::CoefficientThreshold>"));
    assert!(name(&T::Weight(3)).contains("WithFilter<'_, paulistrings::truncation::builtin::And<"));
    assert_eq!(
        name(&T::or(T::Coefficient(1e-3), T::Weight(3))),
        "paulistrings::truncation::tree::BuiltinTruncation"
    );
}
