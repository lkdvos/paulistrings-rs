//! [`BuiltinTruncation`], every builtin policy and combinator as one runtime value.

use super::builtin::{ApproxTopN, CoefficientThreshold, TopN, WeightCutoff};
use super::TruncationPolicy;
use crate::engine::partitioned::transport::Collectives;
use crate::engine::partitioned::truncation::PartitionedTruncation;
use crate::pauli_sum::PauliSum;
use num_complex::Complex64;

/// The builtin truncation policies as one value-level tree, the form a backend lowers.
///
/// Every variant delegates to the builtin type it names, so a `BuiltinTruncation` truncates exactly as the corresponding composition of [`CoefficientThreshold`], [`WeightCutoff`], [`TopN`], [`ApproxTopN`], [`And`](super::And) and [`Or`](super::Or) does, on the host and in partitioned mode.
/// [`TruncationPolicy::device_policy`] returns one for every builtin, which is how the CUDA backend reads a policy.
///
/// `Or` combines per-term filters only and runs neither side's layer pass, matching [`Or`](super::Or); `And` runs both, first then second.
/// In partitioned mode an exact `TopN` whose layer pass would run panics, since it has no collective form (see [`PartitionedTruncation`]).
///
/// ```
/// use paulistrings::truncation::BuiltinTruncation as T;
/// use paulistrings::TruncationPolicy;
///
/// let policy = T::And(Box::new(T::Coeff(1e-9)), Box::new(T::ApproxTopN(1_000)));
/// assert!(<_ as TruncationPolicy<1>>::finalizes_layer(&policy));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum BuiltinTruncation {
    /// No filtering; exact zeros are still dropped by the merge.
    Keep,
    /// [`CoefficientThreshold`]`(eps)`.
    Coeff(f64),
    /// [`WeightCutoff`]`(k)`.
    Weight(u32),
    /// [`TopN`]`(n)`.
    TopN(usize),
    /// [`ApproxTopN`]`(n)`.
    ApproxTopN(usize),
    /// [`And`](super::And) of two policies.
    And(Box<BuiltinTruncation>, Box<BuiltinTruncation>),
    /// [`Or`](super::Or) of two policies.
    Or(Box<BuiltinTruncation>, Box<BuiltinTruncation>),
}

impl BuiltinTruncation {
    /// Whether an exact [`TopN`] appears anywhere in the tree, `Or` branches included.
    pub fn contains_exact_top_n(&self) -> bool {
        match self {
            Self::TopN(_) => true,
            Self::And(a, b) | Self::Or(a, b) => {
                a.contains_exact_top_n() || b.contains_exact_top_n()
            }
            Self::Keep | Self::Coeff(_) | Self::Weight(_) | Self::ApproxTopN(_) => false,
        }
    }
}

impl<const W: usize> TruncationPolicy<W> for BuiltinTruncation {
    #[inline]
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
        match self {
            Self::Keep => true,
            Self::Coeff(eps) => CoefficientThreshold(*eps).keep_term(x, z, c),
            Self::Weight(k) => WeightCutoff(*k).keep_term(x, z, c),
            Self::TopN(n) => <TopN as TruncationPolicy<W>>::keep_term(&TopN(*n), x, z, c),
            Self::ApproxTopN(n) => {
                <ApproxTopN as TruncationPolicy<W>>::keep_term(&ApproxTopN(*n), x, z, c)
            }
            Self::And(a, b) => a.keep_term(x, z, c) && b.keep_term(x, z, c),
            Self::Or(a, b) => a.keep_term(x, z, c) || b.keep_term(x, z, c),
        }
    }

    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        match self {
            Self::TopN(n) => TopN(*n).finalize_layer(sum),
            Self::ApproxTopN(n) => ApproxTopN(*n).finalize_layer(sum),
            Self::And(a, b) => {
                a.finalize_layer(sum);
                b.finalize_layer(sum);
            }
            Self::Keep | Self::Coeff(_) | Self::Weight(_) | Self::Or(_, _) => {}
        }
    }

    fn finalizes_layer(&self) -> bool {
        match self {
            Self::TopN(_) | Self::ApproxTopN(_) => true,
            Self::And(a, b) => {
                <Self as TruncationPolicy<W>>::finalizes_layer(a)
                    || <Self as TruncationPolicy<W>>::finalizes_layer(b)
            }
            Self::Keep | Self::Coeff(_) | Self::Weight(_) | Self::Or(_, _) => false,
        }
    }

    fn device_policy(&self) -> Option<BuiltinTruncation> {
        Some(self.clone())
    }
}

impl<const W: usize> PartitionedTruncation<W> for BuiltinTruncation {
    /// Arm for arm the host [`finalize_layer`](TruncationPolicy::finalize_layer), through each builtin's own collective form.
    ///
    /// # Panics
    ///
    /// On a `TopN` that is reached, which has no collective form.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, coll: &dyn Collectives) {
        match self {
            Self::ApproxTopN(n) => ApproxTopN(*n).finalize_layer_partitioned(local, coll),
            Self::And(a, b) => {
                <Self as PartitionedTruncation<W>>::finalize_layer_partitioned(a, local, coll);
                <Self as PartitionedTruncation<W>>::finalize_layer_partitioned(b, local, coll);
            }
            Self::TopN(_) => panic!(
                "BuiltinTruncation::TopN has no partitioned layer pass: exact top-n is a distributed k-th selection; use ApproxTopN"
            ),
            Self::Keep | Self::Coeff(_) | Self::Weight(_) | Self::Or(_, _) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BuiltinTruncation as T;
    use super::*;
    use crate::engine::partitioned::transport::InProcessTransport;
    use crate::test_support::{assert_same_terms, rand_sum_real, KeepAll};
    use crate::truncation::{And, Or};

    fn and(a: T, b: T) -> T {
        T::And(Box::new(a), Box::new(b))
    }

    fn or(a: T, b: T) -> T {
        T::Or(Box::new(a), Box::new(b))
    }

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
            assert_keeps_like(&T::Coeff(eps), &CoefficientThreshold(eps), "coeff");
        }
        for k in [0u32, 1, 2, 3] {
            assert_keeps_like(&T::Weight(k), &WeightCutoff(k), "weight");
        }
        assert_keeps_like(&T::TopN(3), &TopN(3), "topn");
        assert_keeps_like(&T::ApproxTopN(3), &ApproxTopN(3), "approx");
        assert_keeps_like(
            &and(T::Coeff(0.1), T::Weight(1)),
            &And(CoefficientThreshold(0.1), WeightCutoff(1)),
            "and",
        );
        assert_keeps_like(
            &or(T::Coeff(0.5), T::Weight(0)),
            &Or(CoefficientThreshold(0.5), WeightCutoff(0)),
            "or",
        );
        assert_keeps_like(
            &or(and(T::Coeff(0.1), T::ApproxTopN(9)), T::Weight(1)),
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
        assert!(!f(&T::Coeff(1e-3)));
        assert!(!f(&T::Weight(4)));
        assert!(f(&T::TopN(4)));
        assert!(f(&T::ApproxTopN(4)));
        assert!(f(&and(T::Coeff(1e-3), T::ApproxTopN(4))));
        assert!(f(&and(T::ApproxTopN(4), T::Weight(2))));
        assert!(f(&and(T::ApproxTopN(4), T::ApproxTopN(2))));
        assert!(f(&and(T::Keep, and(T::Coeff(0.1), T::TopN(3)))));
        assert!(!f(&and(T::Coeff(1e-3), T::Weight(2))));
        assert!(!f(&or(T::ApproxTopN(4), T::Coeff(1e-3))));
        assert!(!f(&or(T::TopN(4), T::ApproxTopN(4))));
        assert!(!f(&and(or(T::TopN(4), T::Keep), T::Coeff(0.5))));
    }

    #[test]
    fn device_policy_of_a_tree_is_itself_and_every_builtin_lowers() {
        let tree = and(T::Coeff(1e-3), or(T::ApproxTopN(7), T::Weight(2)));
        assert_eq!(
            <T as TruncationPolicy<1>>::device_policy(&tree),
            Some(tree.clone())
        );
        let lowered = <_ as TruncationPolicy<1>>::device_policy(&And(
            CoefficientThreshold(1e-3),
            Or(ApproxTopN(7), WeightCutoff(2)),
        ));
        assert_eq!(lowered, Some(tree));
        assert_eq!(
            <_ as TruncationPolicy<2>>::device_policy(&TopN(10)),
            Some(T::TopN(10))
        );
        assert_eq!(
            <_ as TruncationPolicy<1>>::device_policy(&KeepAll),
            Some(T::Keep)
        );
        struct Custom;
        impl<const W: usize> TruncationPolicy<W> for Custom {}
        assert_eq!(<_ as TruncationPolicy<1>>::device_policy(&Custom), None);
        assert_eq!(
            <_ as TruncationPolicy<1>>::device_policy(&And(CoefficientThreshold(0.1), Custom)),
            None
        );
    }

    #[test]
    fn contains_exact_top_n_looks_through_or() {
        assert!(T::TopN(1).contains_exact_top_n());
        assert!(or(T::Coeff(0.1), T::TopN(1)).contains_exact_top_n());
        assert!(and(T::Keep, and(T::TopN(1), T::Weight(2))).contains_exact_top_n());
        assert!(!and(T::ApproxTopN(1), T::Weight(2)).contains_exact_top_n());
    }

    /// The layer pass, host and partitioned at `P = 1, 2`, equals the builtin composition's.
    #[test]
    fn finalize_layer_equals_the_builtin_composition() {
        let input = rand_sum_real::<1>(1500, 32, 0xB117);
        let tree = and(T::Coeff(1e-3), and(T::ApproxTopN(900), T::ApproxTopN(400)));
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

        let ored = or(T::ApproxTopN(5), T::Coeff(0.5));
        let mut untouched = input.clone();
        <T as TruncationPolicy<1>>::finalize_layer(&ored, &mut untouched);
        assert_eq!(untouched.len(), input.len(), "Or runs no layer pass");

        let group = InProcessTransport::group(1);
        let mut got = input.clone();
        <T as PartitionedTruncation<1>>::finalize_layer_partitioned(&tree, &mut got, &group[0]);
        assert_same_terms(&got, &want, "partitioned P=1");
    }

    #[test]
    #[should_panic(expected = "no partitioned layer pass")]
    fn a_reached_top_n_panics_in_partitioned_mode() {
        let group = InProcessTransport::group(1);
        let mut sum = rand_sum_real::<1>(10, 32, 0x1);
        let tree = and(T::Coeff(0.1), T::TopN(3));
        <T as PartitionedTruncation<1>>::finalize_layer_partitioned(&tree, &mut sum, &group[0]);
    }
}
