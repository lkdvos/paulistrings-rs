//! `paulistrings._paulistrings.noise` submodule: noise channel factories. See §11.

#![allow(unused)]

use pyo3::prelude::*;

#[pyfunction]
fn depolarize(_p: f64, _qubit: u32) -> PyResult<PyObject> {
    todo!("§11: return an opaque PyChannel wrapping Depolarizing")
}

#[pyfunction]
fn dephase(_p: f64, _qubit: u32) -> PyResult<PyObject> {
    todo!("§11: return an opaque PyChannel wrapping Dephasing")
}

#[pyfunction]
fn amplitude_damping(_gamma: f64, _qubit: u32) -> PyResult<PyObject> {
    todo!("§11: return an opaque PyChannel wrapping AmplitudeDamping")
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(depolarize, m)?)?;
    m.add_function(wrap_pyfunction!(dephase, m)?)?;
    m.add_function(wrap_pyfunction!(amplitude_damping, m)?)?;
    Ok(())
}
