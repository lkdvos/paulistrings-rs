//! `paulistrings._paulistrings.gates` submodule: gate factories. See ARCHITECTURE.md §Python-Bindings.

use crate::channel_spec::{Axis, ChannelSpec, PyChannel};
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

/// Reject a two-qubit gate whose two indices coincide.
///
/// Such a "gate" is not a typo the core can absorb: the prepared local
/// Pauli-transfer matrix would be derived over a one-qubit support declared as
/// two, so the result is silently wrong rather than merely odd. `unitary_2q` has
/// always refused it; the named Clifford pairs now do too.
fn distinct_pair(name: &str, q0: u32, q1: u32) -> PyResult<()> {
    if q0 == q1 {
        return Err(PyValueError::new_err(format!(
            "{name}: the two qubit indices must differ (both are {q0})"
        )));
    }
    Ok(())
}

/// Shared by `gates.cnot` and `Circuit.cnot`.
pub(crate) fn cnot_spec(control: u32, target: u32) -> PyResult<ChannelSpec> {
    distinct_pair("cnot", control, target)?;
    Ok(ChannelSpec::Cnot { control, target })
}

/// Shared by `gates.cz` and `Circuit.cz`.
pub(crate) fn cz_spec(q0: u32, q1: u32) -> PyResult<ChannelSpec> {
    distinct_pair("cz", q0, q1)?;
    Ok(ChannelSpec::Cz { q0, q1 })
}

/// Shared by `gates.swap` and `Circuit.swap`.
pub(crate) fn swap_spec(q0: u32, q1: u32) -> PyResult<ChannelSpec> {
    distinct_pair("swap", q0, q1)?;
    Ok(ChannelSpec::Swap { q0, q1 })
}

#[pyfunction]
fn cnot(control: u32, target: u32) -> PyResult<PyChannel> {
    Ok(PyChannel::new(cnot_spec(control, target)?))
}

#[pyfunction]
fn cz(q0: u32, q1: u32) -> PyResult<PyChannel> {
    Ok(PyChannel::new(cz_spec(q0, q1)?))
}

#[pyfunction]
fn swap(q0: u32, q1: u32) -> PyResult<PyChannel> {
    Ok(PyChannel::new(swap_spec(q0, q1)?))
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

/// Shared by `gates.pauli_rotation` and `Circuit.pauli_rotation`.
///
/// The compact form: `pauli[k]` is the Pauli acting on `qubits[k]`, identity
/// everywhere else. Full-length `IXYZ` strings are deliberately not accepted —
/// the suite circuits address 127-qubit lattices, where a full-length string is
/// unreadable and a miscount is silent.
pub(crate) fn pauli_rotation_spec(
    pauli: &str,
    qubits: &[u32],
    theta: f64,
) -> PyResult<ChannelSpec> {
    if pauli.chars().count() != qubits.len() {
        return Err(PyValueError::new_err(format!(
            "pauli_rotation: pauli and qubits must have the same length \
             (got {} characters and {} qubits)",
            pauli.chars().count(),
            qubits.len(),
        )));
    }
    if qubits.is_empty() {
        return Err(PyValueError::new_err(
            "pauli_rotation: pauli and qubits must be non-empty",
        ));
    }
    let mut paulis = Vec::with_capacity(qubits.len());
    for (ch, &qubit) in pauli.chars().zip(qubits.iter()) {
        // 'I' is rejected along with everything else: an identity position is
        // expressed by leaving the qubit out, so allowing it would give two
        // spellings of the same channel.
        let axis = match ch {
            'X' => Axis::X,
            'Y' => Axis::Y,
            'Z' => Axis::Z,
            other => {
                return Err(PyValueError::new_err(format!(
                    "pauli_rotation: unexpected Pauli character {other:?} \
                     (expected X/Y/Z; identity positions are expressed by omission)"
                )));
            }
        };
        paulis.push((qubit, axis));
    }
    // Quadratic, but the generator is a handful of qubits and this runs once per
    // circuit push, never per term. A repeated index would silently halve the
    // generator's weight (the two bit-plane writes would collide).
    for i in 0..paulis.len() {
        for j in (i + 1)..paulis.len() {
            if paulis[i].0 == paulis[j].0 {
                return Err(PyValueError::new_err(format!(
                    "pauli_rotation: qubits must be distinct (index {} appears twice)",
                    paulis[i].0
                )));
            }
        }
    }
    Ok(ChannelSpec::PauliRotationN { theta, paulis })
}

/// A rotation `exp(-i·θ·P/2)` about a Pauli string of any weight.
///
/// `P` is `pauli[0]` on `qubits[0]`, `pauli[1]` on `qubits[1]`, ..., identity
/// elsewhere. `pauli_rotation("X", [q], theta)` is `rx(theta, q)`;
/// `pauli_rotation("ZZ", [i, j], -pi/2)` is the kicked-Ising Clifford-point bond.
///
/// The argument order is `(what, where, how much)`, which diverges from
/// `rz(theta, qubit)` on purpose: for a multi-qubit generator it reads correctly,
/// and putting the two string-ish arguments first makes an accidental
/// transposition a `TypeError` instead of a silent angle/qubit swap.
///
/// Qubit indices are checked against the circuit width when the channel is
/// appended, not here — a factory-made `Channel` is width-agnostic by design.
#[pyfunction]
fn pauli_rotation(pauli: &str, qubits: Vec<u32>, theta: f64) -> PyResult<PyChannel> {
    Ok(PyChannel::new(pauli_rotation_spec(pauli, &qubits, theta)?))
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
    m.add_function(wrap_pyfunction!(pauli_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(unitary_1q, m)?)?;
    m.add_function(wrap_pyfunction!(unitary_2q, m)?)?;
    Ok(())
}
