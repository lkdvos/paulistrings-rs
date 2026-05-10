//! Python `PauliSum` class with width-monomorphized backing storage. See §4, §11.

#![allow(unused)]

use crate::truncation_spec::{PolicySpec, PyTruncation, SpecPolicy};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use paulistrings::accumulator::BuildAccumulator;
use paulistrings::pauli_string::PauliString;
use paulistrings::phase::Phase;
use paulistrings::{propagate, Direction, PauliSum as CorePauliSum};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict};

/// Width-dispatch enum. The Python boundary picks the smallest width that
/// fits `num_qubits` and stores the appropriately monomorphized `PauliSum`.
pub enum PauliSumImpl {
    W1(CorePauliSum<1>),
    W2(CorePauliSum<2>),
    W4(CorePauliSum<4>),
    W8(CorePauliSum<8>),
    W16(CorePauliSum<16>),
}

impl PauliSumImpl {
    /// Pick the smallest supported width for `num_qubits`. Returns `None` if
    /// `num_qubits` exceeds the largest monomorphized width (1024 qubits).
    pub fn empty_for(num_qubits: usize) -> Option<Self> {
        match num_qubits {
            0..=64 => Some(Self::W1(CorePauliSum::empty(num_qubits))),
            65..=128 => Some(Self::W2(CorePauliSum::empty(num_qubits))),
            129..=256 => Some(Self::W4(CorePauliSum::empty(num_qubits))),
            257..=512 => Some(Self::W8(CorePauliSum::empty(num_qubits))),
            513..=1024 => Some(Self::W16(CorePauliSum::empty(num_qubits))),
            _ => None,
        }
    }

    pub fn num_qubits(&self) -> usize {
        match self {
            Self::W1(s) => s.num_qubits(),
            Self::W2(s) => s.num_qubits(),
            Self::W4(s) => s.num_qubits(),
            Self::W8(s) => s.num_qubits(),
            Self::W16(s) => s.num_qubits(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::W1(s) => s.len(),
            Self::W2(s) => s.len(),
            Self::W4(s) => s.len(),
            Self::W8(s) => s.len(),
            Self::W16(s) => s.len(),
        }
    }

    /// Snapshot of the coefficient column.
    pub fn coeffs(&self) -> Vec<Complex64> {
        match self {
            Self::W1(s) => s.coeff().to_vec(),
            Self::W2(s) => s.coeff().to_vec(),
            Self::W4(s) => s.coeff().to_vec(),
            Self::W8(s) => s.coeff().to_vec(),
            Self::W16(s) => s.coeff().to_vec(),
        }
    }

    /// `(width, x_flat, z_flat)` snapshot of the SoA columns. Both `x_flat`
    /// and `z_flat` have length `len() * width`, and `width` is the active
    /// monomorphization's `W`. Caller reshapes to `(len, width)`.
    pub fn xz_flat(&self) -> (usize, Vec<u64>, Vec<u64>) {
        fn flatten<const W: usize>(rows: &[[u64; W]]) -> Vec<u64> {
            // SAFETY-equivalent: flat-copy via iteration. The W is small (≤16)
            // and the array length is `len()`; this is not on the hot path.
            let mut out = Vec::with_capacity(rows.len() * W);
            for r in rows {
                out.extend_from_slice(r);
            }
            out
        }
        match self {
            Self::W1(s) => (1, flatten(s.x()), flatten(s.z())),
            Self::W2(s) => (2, flatten(s.x()), flatten(s.z())),
            Self::W4(s) => (4, flatten(s.x()), flatten(s.z())),
            Self::W8(s) => (8, flatten(s.x()), flatten(s.z())),
            Self::W16(s) => (16, flatten(s.x()), flatten(s.z())),
        }
    }

    /// Build from a `{pauli_string: coefficient}` Python dict at the requested
    /// width. The width must already match `num_qubits` (caller's job).
    pub fn from_strings_dict(num_qubits: usize, terms: &Bound<'_, PyDict>) -> PyResult<Self> {
        // The match arms call into the generic helper, which monomorphizes
        // the parser per width. Slice 10.2 will replace this match with a
        // macro that also covers from_strings, propagate, etc.
        match num_qubits {
            0..=64 => Ok(Self::W1(parse_terms::<1>(num_qubits, terms)?)),
            65..=128 => Ok(Self::W2(parse_terms::<2>(num_qubits, terms)?)),
            129..=256 => Ok(Self::W4(parse_terms::<4>(num_qubits, terms)?)),
            257..=512 => Ok(Self::W8(parse_terms::<8>(num_qubits, terms)?)),
            513..=1024 => Ok(Self::W16(parse_terms::<16>(num_qubits, terms)?)),
            _ => Err(PyValueError::new_err(
                "num_qubits exceeds largest monomorphized width (1024)",
            )),
        }
    }
}

/// Build a `PauliSum<W>` from a `{pauli_string: coefficient}` Python dict.
///
/// Pauli-string format matches the test helper in `pauli_sum.rs`: the
/// character at index `i` describes qubit `i`. `Y` is `Y_canonical`, i.e.
/// `i · (x=1, z=1)`, with the `i` factor folded into the coefficient.
fn parse_terms<const W: usize>(
    num_qubits: usize,
    terms: &Bound<'_, PyDict>,
) -> PyResult<CorePauliSum<W>> {
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, terms.len());
    for (key, val) in terms.iter() {
        let s: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("PauliSum.from_strings keys must be str"))?;
        if s.len() != num_qubits {
            return Err(PyValueError::new_err(format!(
                "Pauli string {:?} has length {}, expected {} (length must match num_qubits)",
                s,
                s.len(),
                num_qubits
            )));
        }
        let c = extract_complex(&val)?;
        let mut x = [0u64; W];
        let mut z = [0u64; W];
        let mut phase = Phase::ONE;
        for (i, ch) in s.chars().enumerate() {
            let word = i / 64;
            let bit = 1u64 << (i % 64);
            match ch {
                'I' => {}
                'X' => x[word] |= bit,
                'Z' => z[word] |= bit,
                'Y' => {
                    x[word] |= bit;
                    z[word] |= bit;
                    phase += Phase::I;
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unexpected Pauli character {:?} (expected I/X/Y/Z)",
                        other
                    )));
                }
            }
        }
        acc.add_term(PauliString::<W> { x, z }, phase, c);
    }
    Ok(acc.finalize())
}

/// Extract a Python complex/float/int into `Complex64`.
fn extract_complex(val: &Bound<'_, PyAny>) -> PyResult<Complex64> {
    if let Ok(c) = val.downcast::<PyComplex>() {
        return Ok(Complex64::new(c.real(), c.imag()));
    }
    if let Ok(f) = val.extract::<f64>() {
        return Ok(Complex64::new(f, 0.0));
    }
    Err(PyTypeError::new_err(
        "expected complex, float, or int coefficient",
    ))
}

#[pyclass(module = "paulistrings._paulistrings", name = "PauliSum")]
pub struct PauliSum {
    pub(crate) inner: PauliSumImpl,
}

#[pymethods]
impl PauliSum {
    /// Empty Pauli sum on `num_qubits` qubits.
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        PauliSumImpl::empty_for(num_qubits)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                PyValueError::new_err("num_qubits exceeds largest monomorphized width (1024)")
            })
    }

    /// Build from a `{pauli_string: coefficient}` dict. See §11.
    #[classmethod]
    fn from_strings(
        _cls: &Bound<'_, pyo3::types::PyType>,
        terms: &Bound<'_, PyDict>,
        num_qubits: usize,
    ) -> PyResult<Self> {
        let inner = PauliSumImpl::from_strings_dict(num_qubits, terms)?;
        Ok(Self { inner })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Snapshot of the coefficient column as a list of Python complex values.
    fn coefficients(&self) -> Vec<Complex64> {
        self.inner.coeffs()
    }

    /// Snapshot of the coefficient column as a 1-D NumPy `complex128` array.
    fn coefficients_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
        self.inner.coeffs().into_pyarray_bound(py)
    }

    /// Snapshot of the X-part column as a 2-D NumPy `uint64` array of shape
    /// `(len, W)` where `W` is the monomorphized width chosen for this sum.
    /// One row per term; column `j` holds the bit-word covering qubits
    /// `64*j .. 64*(j+1)`.
    fn x_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        let (w, x_flat, _z_flat) = self.inner.xz_flat();
        let n = x_flat.len() / w;
        x_flat
            .into_pyarray_bound(py)
            .reshape([n, w])
            .expect("flat length is n*w by construction")
    }

    /// Snapshot of the Z-part column as a 2-D NumPy `uint64` array. See
    /// `x_array` for the layout.
    fn z_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        let (w, _x_flat, z_flat) = self.inner.xz_flat();
        let n = z_flat.len() / w;
        z_flat
            .into_pyarray_bound(py)
            .reshape([n, w])
            .expect("flat length is n*w by construction")
    }

    /// Active monomorphized width `W` (number of `u64` words per term).
    /// Useful when paired with `x_array` / `z_array` for downstream bit-twiddling.
    #[getter]
    fn width(&self) -> usize {
        self.inner.xz_flat().0
    }

    /// Propagate `self` through `circuit`. See §8.1, §11.
    ///
    /// `direction`: `"forward"` (default) or `"heisenberg"`. `policy` is an
    /// optional `Truncation` from the `truncation` submodule; if `None`, no
    /// per-term filtering is applied (the engine's merge phase still drops
    /// exact-zero terms).
    #[pyo3(signature = (circuit, policy=None, direction=None))]
    fn propagate(
        &self,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
    ) -> PyResult<Self> {
        let dir = match direction.unwrap_or("forward") {
            "forward" => Direction::Forward,
            "heisenberg" => Direction::Heisenberg,
            other => {
                return Err(PyValueError::new_err(format!(
                    "direction must be 'forward' or 'heisenberg', got {:?}",
                    other
                )));
            }
        };
        if self.inner.num_qubits() != circuit.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "PauliSum.num_qubits ({}) != Circuit.num_qubits ({})",
                self.inner.num_qubits(),
                circuit.inner.num_qubits()
            )));
        }
        let no_op = PolicySpec::NoOp;
        let spec: &PolicySpec = match policy {
            Some(p) => &p.spec,
            None => &no_op,
        };
        let inner = match (&self.inner, &circuit.inner) {
            (PauliSumImpl::W1(s), crate::circuit::CircuitImpl::W1(c)) => {
                PauliSumImpl::W1(propagate(c, s.clone(), &SpecPolicy::<1>(spec), dir))
            }
            (PauliSumImpl::W2(s), crate::circuit::CircuitImpl::W2(c)) => {
                PauliSumImpl::W2(propagate(c, s.clone(), &SpecPolicy::<2>(spec), dir))
            }
            (PauliSumImpl::W4(s), crate::circuit::CircuitImpl::W4(c)) => {
                PauliSumImpl::W4(propagate(c, s.clone(), &SpecPolicy::<4>(spec), dir))
            }
            (PauliSumImpl::W8(s), crate::circuit::CircuitImpl::W8(c)) => {
                PauliSumImpl::W8(propagate(c, s.clone(), &SpecPolicy::<8>(spec), dir))
            }
            (PauliSumImpl::W16(s), crate::circuit::CircuitImpl::W16(c)) => {
                PauliSumImpl::W16(propagate(c, s.clone(), &SpecPolicy::<16>(spec), dir))
            }
            _ => {
                // Same num_qubits but different widths is impossible because
                // both width pickers map num_qubits to the same arm.
                return Err(PyValueError::new_err(
                    "internal: PauliSum and Circuit width mismatch",
                ));
            }
        };
        Ok(Self { inner })
    }
}
