//! `paulistrings._paulistrings.noise` submodule: noise channel factories. See §11.

use crate::channel_spec::{ChannelSpec, PyChannel};
use pyo3::prelude::*;

#[pyfunction]
fn depolarize(p: f64, qubit: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::Depolarize { p, qubit })
}

#[pyfunction]
fn dephase(p: f64, qubit: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::Dephase { p, qubit })
}

#[pyfunction]
fn amplitude_damping(gamma: f64, qubit: u32) -> PyChannel {
    PyChannel::new(ChannelSpec::AmplitudeDamping { gamma, qubit })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(depolarize, m)?)?;
    m.add_function(wrap_pyfunction!(dephase, m)?)?;
    m.add_function(wrap_pyfunction!(amplitude_damping, m)?)?;
    Ok(())
}
