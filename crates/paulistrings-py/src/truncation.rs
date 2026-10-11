//! `paulistrings._paulistrings.truncation` submodule: policy factories. See
//! ARCHITECTURE.md §Truncation and ARCHITECTURE.md §Python-Bindings.
//!
//! Python composition is via the `&` and `|` operators on the returned objects; `|` combines per-term filters only.

use crate::truncation_spec::PyTruncation;
use paulistrings::{BuiltinTruncation, CollapseSample};
use pyo3::prelude::*;

#[pyfunction]
fn coeff(epsilon: f64) -> PyTruncation {
    PyTruncation::new(BuiltinTruncation::Coefficient(epsilon))
}

#[pyfunction]
fn weight(k: u32) -> PyTruncation {
    PyTruncation::new(BuiltinTruncation::Weight(k))
}

/// Keep the ``n`` terms of largest coefficient magnitude after each layer; a sum of at most ``n`` terms is untouched.
///
/// Let ``t`` be the n-th largest magnitude: everything above ``t`` is kept, plus as many terms at exactly ``t`` as fit, the first in storage (bucket) order. Which tied terms survive is not part of the contract.
#[pyfunction]
fn topn(n: usize) -> PyTruncation {
    PyTruncation::new(BuiltinTruncation::TopN(n))
}

/// Keep ``n`` terms after each layer, ranked by octave of ``|c|**2`` — the cheap sibling of ``topn``, and the one that runs partitioned.
///
/// Whole octaves are kept from the top down while they fit in ``n``, and the next octave fills the rest in storage order (rank order across partitions, then bucket order), so exactly ``n`` survive whenever the sum holds more, though not necessarily the ``n`` largest.
/// Which boundary-octave terms survive depends on bucketing and partitioning and is not part of the contract.
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
