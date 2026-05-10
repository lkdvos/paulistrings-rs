//! `paulistrings._paulistrings.gates` submodule: gate factories. See §11.

use crate::channel_spec::{ChannelSpec, PyChannel};
use pyo3::prelude::*;

#[pyfunction]
fn h(qubit: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::H { qubit })
}

#[pyfunction]
fn s(qubit: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::S { qubit })
}

#[pyfunction]
fn x(qubit: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::X { qubit })
}

#[pyfunction]
fn y(qubit: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::Y { qubit })
}

#[pyfunction]
fn z(qubit: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::Z { qubit })
}

#[pyfunction]
fn cnot(control: u32, target: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::Cnot { control, target })
}

#[pyfunction]
fn cz(q0: u32, q1: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::Cz { q0, q1 })
}

#[pyfunction]
fn swap(q0: u32, q1: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::Swap { q0, q1 })
}

#[pyfunction]
fn rz(theta: f64, qubit: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::Rz { theta, qubit })
}

#[pyfunction]
fn rx(theta: f64, qubit: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::Rx { theta, qubit })
}

#[pyfunction]
fn ry(theta: f64, qubit: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::Ry { theta, qubit })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(h, m)?)?;
    m.add_function(wrap_pyfunction!(s, m)?)?;
    m.add_function(wrap_pyfunction!(x, m)?)?;
    m.add_function(wrap_pyfunction!(y, m)?)?;
    m.add_function(wrap_pyfunction!(z, m)?)?;
    m.add_function(wrap_pyfunction!(cnot, m)?)?;
    m.add_function(wrap_pyfunction!(cz, m)?)?;
    m.add_function(wrap_pyfunction!(swap, m)?)?;
    m.add_function(wrap_pyfunction!(rz, m)?)?;
    m.add_function(wrap_pyfunction!(rx, m)?)?;
    m.add_function(wrap_pyfunction!(ry, m)?)?;
    Ok(())
}
