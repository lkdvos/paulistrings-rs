//! `paulistrings._paulistrings.truncation` submodule: policy factories. See §7, §11.
//!
//! Python composition is via the `&` and `|` operators on the returned objects.

#![allow(unused)]

use pyo3::prelude::*;

#[pyfunction]
fn coeff(_epsilon: f64) -> PyResult<PyObject> {
    todo!("§11: return PyTruncation wrapping CoefficientThreshold")
}

#[pyfunction]
fn weight(_k: u32) -> PyResult<PyObject> {
    todo!("§11: return PyTruncation wrapping WeightCutoff")
}

#[pyfunction]
fn topn(_n: usize) -> PyResult<PyObject> {
    todo!("§11: return PyTruncation wrapping TopN")
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coeff, m)?)?;
    m.add_function(wrap_pyfunction!(weight, m)?)?;
    m.add_function(wrap_pyfunction!(topn, m)?)?;
    Ok(())
}
