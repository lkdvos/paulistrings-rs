//! `PyTruncation` — opaque, width-erased truncation policy handle over a core [`BuiltinTruncation`] tree.
//! Free factories `truncation.coeff/weight/topn/approx_topn/collapse_sample(...)` return one; `&`/`|` compose via `And`/`Or`.

use paulistrings::{BuiltinTruncation, CollapseSample};
use pyo3::prelude::*;
use std::sync::Arc;

/// Why a policy containing an exact `topn` cannot run partitioned, worded for the Python `NotImplementedError`.
pub(crate) const TOPN_PARTITIONED_MSG: &str =
    "exact truncation.topn is not supported in partitioned mode; use \
     truncation.approx_topn or partitions=None";

/// Why a policy containing a `collapse_sample` cannot run on a CUDA device, worded for the Python `NotImplementedError`.
#[cfg(feature = "cuda")]
pub(crate) const COLLAPSE_SAMPLE_DEVICE_MSG: &str =
    "truncation.collapse_sample has no device form; run it on the host (device=None)";

/// Collapses since `before`, both read from `policy` with [`collapse_count`].
pub(crate) fn collapses_since(policy: &BuiltinTruncation, before: Option<u64>) -> Option<u64> {
    collapse_count(policy)
        .zip(before)
        .map(|(after, before)| after - before)
}

/// Collapses performed so far by the distinct samplers in `tree`, or `None` if it has none.
/// A sampler reached twice (`s & s`) is counted once. In a partitioned or distributed run only rank 0's objects count.
pub(crate) fn collapse_count(tree: &BuiltinTruncation) -> Option<u64> {
    fn walk<'a>(tree: &'a BuiltinTruncation, out: &mut Vec<&'a Arc<CollapseSample>>) {
        match tree {
            BuiltinTruncation::CollapseSample(sampler) => {
                if !out.iter().any(|seen| Arc::ptr_eq(seen, sampler)) {
                    out.push(sampler);
                }
            }
            BuiltinTruncation::And(a, b) | BuiltinTruncation::Or(a, b) => {
                walk(a, out);
                walk(b, out);
            }
            BuiltinTruncation::Keep
            | BuiltinTruncation::Coefficient(_)
            | BuiltinTruncation::Weight(_)
            | BuiltinTruncation::TopN(_)
            | BuiltinTruncation::ApproxTopN(_) => {}
        }
    }
    let mut samplers = Vec::new();
    walk(tree, &mut samplers);
    (!samplers.is_empty()).then(|| samplers.iter().map(|s| s.collapses()).sum())
}

/// Opaque truncation-policy handle exposed to Python.
#[pyclass(module = "paulistrings._paulistrings", name = "Truncation")]
#[derive(Clone)]
pub struct PyTruncation {
    pub(crate) tree: BuiltinTruncation,
}

impl PyTruncation {
    pub fn new(tree: BuiltinTruncation) -> Self {
        Self { tree }
    }

    /// The tree `policy=` names, `Keep` for `None`.
    pub(crate) fn tree_of(policy: Option<&PyTruncation>) -> BuiltinTruncation {
        policy.map_or(BuiltinTruncation::Keep, |p| p.tree.clone())
    }
}

#[pymethods]
impl PyTruncation {
    fn __and__(&self, other: &PyTruncation) -> PyTruncation {
        PyTruncation::new(BuiltinTruncation::And(
            Box::new(self.tree.clone()),
            Box::new(other.tree.clone()),
        ))
    }

    fn __or__(&self, other: &PyTruncation) -> PyTruncation {
        PyTruncation::new(BuiltinTruncation::Or(
            Box::new(self.tree.clone()),
            Box::new(other.tree.clone()),
        ))
    }

    fn __repr__(&self) -> String {
        format!("Truncation({})", tree_repr(&self.tree))
    }
}

/// The tree's `Debug` form with the Python spelling `Coeff` for [`BuiltinTruncation::Coefficient`].
fn tree_repr(tree: &BuiltinTruncation) -> String {
    match tree {
        BuiltinTruncation::Coefficient(eps) => format!("Coeff({eps:?})"),
        BuiltinTruncation::And(a, b) => format!("And({}, {})", tree_repr(a), tree_repr(b)),
        BuiltinTruncation::Or(a, b) => format!("Or({}, {})", tree_repr(a), tree_repr(b)),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paulistrings::BuildAccumulator;
    use paulistrings::PauliString;

    use paulistrings::TruncationPolicy;

    /// `n` single-site `X` strings with coefficient 1.
    fn x_sum(n: usize) -> paulistrings::PauliSum<1> {
        let mut acc = BuildAccumulator::<1>::new(n);
        for q in 0..n {
            acc.add_term(
                PauliString::<1>::x(q),
                num_complex::Complex64::new(1.0, 0.0),
            );
        }
        acc.finalize()
    }

    #[test]
    fn collapse_count_sums_distinct_samplers() {
        let sampler = |seed| BuiltinTruncation::from(CollapseSample::new(1, seed));
        assert_eq!(collapse_count(&BuiltinTruncation::Keep), None);
        assert_eq!(collapse_count(&BuiltinTruncation::ApproxTopN(3)), None);
        let (a, b) = (sampler(0), sampler(1));
        for tree in [&a, &b, &b] {
            <BuiltinTruncation as TruncationPolicy<1>>::finalize_layer(tree, &mut x_sum(40));
        }
        let both = BuiltinTruncation::Or(Box::new(a.clone()), Box::new(b.clone()));
        assert_eq!(collapse_count(&both), Some(3));
        let twice = BuiltinTruncation::And(Box::new(a.clone()), Box::new(a));
        assert_eq!(collapse_count(&twice), Some(1));
    }
}
