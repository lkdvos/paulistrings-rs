//! `paulistrings._paulistrings.truncation` submodule: policy factories. See
//! ARCHITECTURE.md §Truncation and ARCHITECTURE.md §Python-Bindings.
//!
//! Python composition is via the `&` and `|` operators on the returned objects.

use crate::truncation_spec::PyTruncation;
use paulistrings::truncation::{BuiltinTruncation, CollapseSample};
use pyo3::prelude::*;

#[pyfunction]
fn coeff(epsilon: f64) -> PyTruncation {
    PyTruncation::new(BuiltinTruncation::Coeff(epsilon))
}

#[pyfunction]
fn weight(k: u32) -> PyTruncation {
    PyTruncation::new(BuiltinTruncation::Weight(k))
}

/// Keep at most ``n`` terms by coefficient magnitude after each layer.
///
/// Never splits a tie group: let ``t`` be the n-th largest magnitude; everything above ``t`` is kept, and the group at ``t`` is kept only if it fits within ``n`` whole, otherwise dropped whole. Equal magnitudes are symmetry-related terms, and keeping an arbitrary subset of such a multiplet would break the symmetry of the truncated operator.
/// Degenerate case: if every candidate ties at the threshold, this keeps nothing. Combine with ``coeff`` via ``&``, or raise ``n``, if that matters.
#[pyfunction]
fn topn(n: usize) -> PyTruncation {
    PyTruncation::new(BuiltinTruncation::TopN(n))
}

/// Keep approximately ``n`` terms after each layer — the cheap sibling of ``topn``, opt-in and never a default.
///
/// Terms are binned by the octave of ``|c|**2``, and whole octaves are kept from the top down while they still fit in ``n``. So: (1) at most ``n`` is kept, always; (2) the shortfall is bounded by the population of the coarsest excluded octave; (3) a tie group is always kept whole, since equal magnitudes share an octave (``topn`` instead drops a tie group whole if it does not fit).
/// The retained set is a pure function of the magnitude multiset, like ``topn``: independent of bucket partition, hash seed, and thread count. Reach for this when ``n`` is a memory budget and a little slack in the term count is cheaper than exact selection; keep ``topn`` when the retained count itself matters.
/// Degenerate case: if every magnitude lands in one octave and there are more than ``n`` terms, this keeps nothing. Combine with ``coeff`` via ``&``, or use ``topn``, if that matters.
#[pyfunction]
fn approx_topn(n: usize) -> PyTruncation {
    PyTruncation::new(BuiltinTruncation::ApproxTopN(n))
}

/// Replace the sum by **one** Pauli string with coefficient ``1``, drawn with probability ``|c|**2 / sum |c|**2``, after any layer that leaves more than ``cache`` terms.
///
/// One ``seed`` is one Monte Carlo trajectory: the draw at the ``k``-th layer pass is a pure function of ``(seed, k)``, independent of the thread count, so ``collapse_sample(cache, seed)`` repeats a trajectory exactly.
/// The returned object *is* that trajectory: propagating with it again, or with a policy composed from it via ``&``/``|``, continues its random sequence rather than restarting it. Call the factory again to start over.
/// Runs under ``partitions=`` and ``comm=`` too, where every partition draws the same pick; at one partition the trajectory is the unpartitioned one, across partition counts only its distribution agrees. ``PropagationStats.collapses`` counts the collapses of one call.
#[pyfunction]
fn collapse_sample(cache: usize, seed: u64) -> PyTruncation {
    PyTruncation::new(BuiltinTruncation::from(CollapseSample::new(cache, seed)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coeff, m)?)?;
    m.add_function(wrap_pyfunction!(weight, m)?)?;
    m.add_function(wrap_pyfunction!(topn, m)?)?;
    m.add_function(wrap_pyfunction!(approx_topn, m)?)?;
    m.add_function(wrap_pyfunction!(collapse_sample, m)?)?;
    Ok(())
}
