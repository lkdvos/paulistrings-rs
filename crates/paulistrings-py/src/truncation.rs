//! `paulistrings._paulistrings.truncation` submodule: policy factories. See
//! ARCHITECTURE.md §Truncation and ARCHITECTURE.md §Python-Bindings.
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
/// Never splits a tie group: let ``t`` be the n-th largest magnitude; everything above ``t`` is kept, and the group at ``t`` is kept only if it fits within ``n`` whole, otherwise dropped whole. Equal magnitudes are symmetry-related terms, and keeping an arbitrary subset of such a multiplet would break the symmetry of the truncated operator.
/// Degenerate case: if every candidate ties at the threshold, this keeps nothing. Combine with ``coeff`` via ``&``, or raise ``n``, if that matters.
#[pyfunction]
fn topn(n: usize) -> PyTruncation {
    PyTruncation::new(PolicySpec::TopN(n))
}

/// Keep approximately ``n`` terms after each layer — the cheap sibling of ``topn``, opt-in and never a default.
///
/// Terms are binned by the octave of ``|c|**2``, and whole octaves are kept from the top down while they still fit in ``n``. So: (1) at most ``n`` is kept, always; (2) the shortfall is bounded by the population of the coarsest excluded octave; (3) a tie group is always kept whole, since equal magnitudes share an octave (``topn`` instead drops a tie group whole if it does not fit).
/// The retained set is a pure function of the magnitude multiset, like ``topn``: independent of bucket partition, hash seed, and thread count. Reach for this when ``n`` is a memory budget and a little slack in the term count is cheaper than exact selection; keep ``topn`` when the retained count itself matters.
/// Degenerate case: if every magnitude lands in one octave and there are more than ``n`` terms, this keeps nothing. Combine with ``coeff`` via ``&``, or use ``topn``, if that matters.
#[pyfunction]
fn approx_topn(n: usize) -> PyTruncation {
    PyTruncation::new(PolicySpec::ApproxTopN(n))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coeff, m)?)?;
    m.add_function(wrap_pyfunction!(weight, m)?)?;
    m.add_function(wrap_pyfunction!(topn, m)?)?;
    m.add_function(wrap_pyfunction!(approx_topn, m)?)?;
    Ok(())
}
