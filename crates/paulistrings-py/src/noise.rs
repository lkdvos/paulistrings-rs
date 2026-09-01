//! `paulistrings._paulistrings.noise` submodule: noise channel factories. See ARCHITECTURE.md §Python-Bindings.

use crate::channel_spec::{ChannelSpec, PyChannel};
use pyo3::exceptions::PyValueError;
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

/// Shared by `noise.pauli_channel` and `Circuit.pauli_channel`.
///
/// The three probabilities must be a sub-probability distribution: the fourth
/// weight is `1 - px - py - pz`, the "no error" branch, so a negative one would
/// make the channel non-physical and its Heisenberg dual a rescale by a factor
/// outside `[-1, 1]` — coefficients would grow layer over layer with nothing to
/// flag it.
pub(crate) fn pauli_channel_spec(px: f64, py: f64, pz: f64, qubit: u32) -> PyResult<ChannelSpec> {
    for (name, p) in [("px", px), ("py", py), ("pz", pz)] {
        if p < 0.0 {
            return Err(PyValueError::new_err(format!(
                "pauli_channel: {name} must be non-negative (got {p})"
            )));
        }
    }
    let total = px + py + pz;
    if total > 1.0 {
        return Err(PyValueError::new_err(format!(
            "pauli_channel: px + py + pz must be at most 1 (got {total})"
        )));
    }
    Ok(ChannelSpec::PauliChannel { px, py, pz, qubit })
}

/// Shared by `noise.depolarize2` and `Circuit.depolarize2`.
pub(crate) fn depolarize2_spec(p: f64, q0: u32, q1: u32) -> PyResult<ChannelSpec> {
    if !(0.0..=1.0).contains(&p) {
        return Err(PyValueError::new_err(format!(
            "depolarize2: p must be between 0 and 1 (got {p})"
        )));
    }
    // An overlapping pair would declare a two-qubit support over one qubit,
    // which the engine's local-PTM derivation mis-tabulates rather than
    // rejecting.
    if q0 == q1 {
        return Err(PyValueError::new_err(format!(
            "depolarize2: the two qubit indices must differ (both are {q0})"
        )));
    }
    Ok(ChannelSpec::Depolarize2 { p, q0, q1 })
}

/// A general single-qubit Pauli channel:
/// `E(ρ) = (1-px-py-pz)ρ + px·XρX + py·YρY + pz·ZρZ`.
///
/// In the Heisenberg picture a pure coefficient rescaling — `I → 1`,
/// `X → 1 - 2(py+pz)`, `Y → 1 - 2(px+pz)`, `Z → 1 - 2(px+py)` — and
/// self-adjoint, so `direction="heisenberg"` applies the same factors.
///
/// `pauli_channel(p/3, p/3, p/3, q)` is `depolarize(p, q)` and
/// `pauli_channel(0, 0, p, q)` is `dephase(p, q)`.
#[pyfunction]
fn pauli_channel(px: f64, py: f64, pz: f64, qubit: u32) -> PyResult<PyChannel> {
    Ok(PyChannel::new(pauli_channel_spec(px, py, pz, qubit)?))
}

/// Uniform two-qubit depolarizing noise: probability `p` spread evenly over the
/// 15 non-identity two-qubit Paulis on the pair `(q0, q1)`.
///
/// Heisenberg dual: identity on the pair is preserved, anything else is scaled
/// by `1 - 16p/15` — the same factor whether the Pauli is non-identity on one of
/// the pair or on both. Self-adjoint.
#[pyfunction]
fn depolarize2(p: f64, q0: u32, q1: u32) -> PyResult<PyChannel> {
    Ok(PyChannel::new(depolarize2_spec(p, q0, q1)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(depolarize, m)?)?;
    m.add_function(wrap_pyfunction!(dephase, m)?)?;
    m.add_function(wrap_pyfunction!(amplitude_damping, m)?)?;
    m.add_function(wrap_pyfunction!(pauli_channel, m)?)?;
    m.add_function(wrap_pyfunction!(depolarize2, m)?)?;
    Ok(())
}
