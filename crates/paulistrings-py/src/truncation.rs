//! `paulistrings._paulistrings.truncation` submodule: policy factories. See §7, §11.
//!
//! Python composition is via the `&` and `|` operators on the returned objects.

use crate::truncation_spec::{PolicySpec, PyTruncation};
use pyo3::prelude::*;

#[pyfunction]
fn coeff(epsilon: f64) -> PyTruncation {
    PyTruncation::new(PolicySpec::Coeff(epsilon))
}

#[pyfunction]
fn weight(k: u32) -> PyTruncation {
    PyTruncation::new(PolicySpec::Weight(k))
}

/// Keep at most ``n`` terms by coefficient magnitude after each layer.
///
/// Terms are never split across a group of exactly equal magnitudes: let ``t``
/// be the n-th largest magnitude; everything above ``t`` is kept, and the tie
/// group at ``t`` is kept only if it fits within ``n`` in full, otherwise it is
/// dropped whole. So the result has at most ``n`` terms, and exactly ``n``
/// whenever the cut lands on a group boundary (in particular when all
/// magnitudes are distinct).
///
/// Equal magnitudes come from symmetry-related terms, and keeping an arbitrary
/// subset of such a multiplet breaks the symmetry of the truncated operator.
///
/// Note the degenerate case: if *every* candidate ties at the threshold, this
/// keeps nothing and the sum becomes empty. Combine with ``coeff`` via ``&``,
/// or raise ``n`` above the expected multiplet size, if that matters.
#[pyfunction]
fn topn(n: usize) -> PyTruncation {
    PyTruncation::new(PolicySpec::TopN(n))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coeff, m)?)?;
    m.add_function(wrap_pyfunction!(weight, m)?)?;
    m.add_function(wrap_pyfunction!(topn, m)?)?;
    Ok(())
}
