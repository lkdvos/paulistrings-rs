//! `paulistrings._paulistrings.gates` submodule: gate factories. See §11.

#![allow(unused)]

use pyo3::prelude::*;

/// Free-function form of `Circuit.h(...)`. Returns an opaque channel handle
/// suitable for `Circuit.append(...)`.
#[pyfunction]
fn h(_qubit: u32) -> PyResult<PyObject> {
    todo!("§11: return an opaque PyChannel wrapping a Hadamard Clifford1Q")
}

#[pyfunction]
fn cnot(_control: u32, _target: u32) -> PyResult<PyObject> {
    todo!("§11: return an opaque PyChannel wrapping a CNOT Clifford2Q")
}

#[pyfunction]
fn rz(_theta: f64, _qubit: u32) -> PyResult<PyObject> {
    todo!("§11: return an opaque PyChannel wrapping a PauliRotation about Z")
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(h, m)?)?;
    m.add_function(wrap_pyfunction!(cnot, m)?)?;
    m.add_function(wrap_pyfunction!(rz, m)?)?;
    Ok(())
}
