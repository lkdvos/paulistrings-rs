//! `PyTruncation` — opaque, width-erased truncation policy handle over a core [`BuiltinTruncation`] tree.
//! Free factories `truncation.coeff/weight/topn/approx_topn(...)` return one; `&`/`|` compose via `And`/`Or`.

use paulistrings::truncation::BuiltinTruncation;
use pyo3::prelude::*;

/// Why a policy containing an exact `topn` cannot run partitioned, worded for the Python `NotImplementedError`.
pub(crate) const TOPN_PARTITIONED_MSG: &str =
    "exact truncation.topn is not supported in partitioned mode; use \
     truncation.approx_topn or partitions=None";

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
        format!("Truncation({:?})", self.tree)
    }
}
