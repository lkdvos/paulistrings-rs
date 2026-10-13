//! Lowering of a [`BuiltinTruncation`] tree's per-term filter onto a generic policy, so a host propagate runs a monomorphized `keep_term` inside the merge instead of the tree's enum match.

use num_complex::Complex64;
use paulistrings::{
    And, BuiltinTruncation, CoefficientThreshold, Collectives, PauliSum, TruncationPolicy,
    WeightCutoff,
};
use pyo3::prelude::*;
use std::sync::Mutex;

/// `tree` with its per-term filter replaced by the generic `filter` it is equal to; the layer passes still run through `tree`, once per layer.
pub(crate) struct WithFilter<'a, F> {
    filter: F,
    tree: &'a BuiltinTruncation,
}

impl<const W: usize, F: TruncationPolicy<W>> TruncationPolicy<W> for WithFilter<'_, F> {
    #[inline]
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
        self.filter.keep_term(x, z, c)
    }

    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        <BuiltinTruncation as TruncationPolicy<W>>::finalize_layer(self.tree, sum)
    }

    fn finalizes_layer(&self) -> bool {
        <BuiltinTruncation as TruncationPolicy<W>>::finalizes_layer(self.tree)
    }

    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, collectives: &dyn Collectives) {
        <BuiltinTruncation as TruncationPolicy<W>>::finalize_layer_partitioned(
            self.tree,
            local,
            collectives,
        )
    }

    fn supports_partitioned(&self) -> bool {
        self.tree.supports_partitioned()
    }
}

/// The policy a host propagate runs a tree as.
///
/// Two generic filters cover every tree whose `And`-reachable leaves hold at most one `Coefficient` and one `Weight` and no `Or`; `Keep` and the layer passes keep every term, so a missing filter is `CoefficientThreshold(-1.0)`, which keeps every term too.
/// Each further filter type is one more engine per width and partitioning, which is what keeps this list short.
pub(crate) enum Lowered<'a> {
    Coefficient(WithFilter<'a, CoefficientThreshold>),
    CoefficientAndWeight(WithFilter<'a, And<CoefficientThreshold, WeightCutoff>>),
    Tree(&'a BuiltinTruncation),
}

impl<'a> Lowered<'a> {
    pub(crate) fn of(tree: &'a BuiltinTruncation) -> Self {
        let (mut coefficient, mut weight) = (None, None);
        if !filter_leaves(tree, &mut coefficient, &mut weight) {
            return Self::Tree(tree);
        }
        let coefficient = CoefficientThreshold(coefficient.unwrap_or(-1.0));
        match weight {
            None => Self::Coefficient(WithFilter {
                filter: coefficient,
                tree,
            }),
            Some(k) => Self::CoefficientAndWeight(WithFilter {
                filter: And(coefficient, WeightCutoff(k)),
                tree,
            }),
        }
    }
}

/// Collects the `Coefficient` and `Weight` leaves under `tree`'s `And`s; `false` on an `Or` or a second leaf of either kind.
fn filter_leaves(
    tree: &BuiltinTruncation,
    coefficient: &mut Option<f64>,
    weight: &mut Option<u32>,
) -> bool {
    use BuiltinTruncation as T;
    match tree {
        T::Keep | T::TopN(_) | T::ApproxTopN(_) | T::CollapseSample(_) => true,
        T::Coefficient(eps) => coefficient.replace(*eps).is_none(),
        T::Weight(k) => weight.replace(*k).is_none(),
        T::And(a, b) => {
            filter_leaves(a, coefficient, weight) && filter_leaves(b, coefficient, weight)
        }
        T::Or(..) => false,
    }
}

/// The policy type of the last host propagate in this process, for the `_last_host_policy` test hook.
static LAST_HOST_POLICY: Mutex<&str> = Mutex::new("");

/// Records `P` as the last host policy type and hands `policy` back.
pub(crate) fn record<P>(policy: &P) -> &P {
    *LAST_HOST_POLICY.lock().unwrap_or_else(|e| e.into_inner()) = std::any::type_name::<P>();
    policy
}

/// Test hook: the Rust type the last host propagate in this process ran its policy as.
#[pyfunction]
pub(crate) fn _last_host_policy() -> &'static str {
    *LAST_HOST_POLICY.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests;
