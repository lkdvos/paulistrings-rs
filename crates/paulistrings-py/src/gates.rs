//! `paulistrings._paulistrings.gates` submodule: gate factories. See §11.

use crate::channel_spec::{ChannelSpec, PyChannel};
use num_complex::Complex64;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Read an `n x n` complex matrix from a NumPy array, checking shape and
/// unitarity.
///
/// The unitarity check is not pedantry: a non-unitary matrix silently produces a
/// non-physical channel whose Pauli-transfer matrix is not norm-preserving, and
/// the error would only show up as drifting coefficients many layers later.
fn read_unitary<const N: usize>(
    a: PyReadonlyArray2<'_, Complex64>,
    what: &str,
) -> PyResult<[[Complex64; N]; N]> {
    let view = a.as_array();
    if view.shape() != [N, N] {
        return Err(PyValueError::new_err(format!(
            "{what} expects a {N}x{N} complex matrix, got shape {:?}",
            view.shape(),
        )));
    }
    let mut u = [[Complex64::new(0.0, 0.0); N]; N];
    for i in 0..N {
        for j in 0..N {
            u[i][j] = view[[i, j]];
        }
    }
    // U U^dagger == I to within tolerance.
    for i in 0..N {
        for j in 0..N {
            let acc: Complex64 = u[i]
                .iter()
                .zip(u[j].iter())
                .map(|(a, b)| a * b.conj())
                .sum();
            let want = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            if (acc - want).norm() > 1e-9 {
                return Err(PyValueError::new_err(format!(
                    "{what}: matrix is not unitary (U U* deviates from the identity \
                     by {:.3e} at [{i}][{j}])",
                    (acc - want).norm(),
                )));
            }
        }
    }
    Ok(u)
}

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

/// Shared by `gates.unitary_1q` and `Circuit.unitary_1q`.
pub(crate) fn unitary_1q_spec(
    qubit: u32,
    matrix: PyReadonlyArray2<'_, Complex64>,
) -> PyResult<ChannelSpec> {
    let m = read_unitary::<2>(matrix, "unitary_1q")?;
    Ok(ChannelSpec::Unitary1Q { qubit, matrix: m })
}

/// Shared by `gates.unitary_2q` and `Circuit.unitary_2q`.
pub(crate) fn unitary_2q_spec(
    q0: u32,
    q1: u32,
    matrix: PyReadonlyArray2<'_, Complex64>,
) -> PyResult<ChannelSpec> {
    if q0 == q1 {
        return Err(PyValueError::new_err("unitary_2q: q0 and q1 must differ"));
    }
    let m = read_unitary::<4>(matrix, "unitary_2q")?;
    Ok(ChannelSpec::Unitary2Q { q0, q1, matrix: m })
}

/// An arbitrary single-qubit unitary from its 2x2 matrix.
#[pyfunction]
fn unitary_1q(qubit: u32, matrix: PyReadonlyArray2<'_, Complex64>) -> PyResult<PyChannel> {
    Ok(PyChannel::new(unitary_1q_spec(qubit, matrix)?))
}

/// An arbitrary two-qubit unitary from its 4x4 matrix.
///
/// `q0` is the more significant tensor factor, i.e. the matrix acts on
/// `|q0 q1>`.
#[pyfunction]
fn unitary_2q(q0: u32, q1: u32, matrix: PyReadonlyArray2<'_, Complex64>) -> PyResult<PyChannel> {
    Ok(PyChannel::new(unitary_2q_spec(q0, q1, matrix)?))
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
    m.add_function(wrap_pyfunction!(unitary_1q, m)?)?;
    m.add_function(wrap_pyfunction!(unitary_2q, m)?)?;
    Ok(())
}
