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
