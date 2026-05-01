//! Python `PauliSum` class with width-monomorphized backing storage. See §4, §11.

#![allow(unused)]

use paulistrings::PauliSum as CorePauliSum;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
                pyo3::exceptions::PyValueError::new_err(
                    "num_qubits exceeds largest monomorphized width (1024)",
                )
            })
    }

    /// Build from a `{pauli_string: coefficient}` dict. See §11.
    #[classmethod]
    fn from_strings(
        _cls: &Bound<'_, pyo3::types::PyType>,
        _terms: &Bound<'_, PyDict>,
        _num_qubits: usize,
    ) -> PyResult<Self> {
        todo!("§11: parse Pauli strings, build via BuildAccumulator, finalize")
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Propagate `self` through `circuit`. See §8.1, §11.
    #[pyo3(signature = (_circuit, _policy=None, _direction=None))]
    fn propagate(
        &self,
        _circuit: &crate::circuit::Circuit,
        _policy: Option<PyObject>,
        _direction: Option<&str>,
    ) -> PyResult<Self> {
        todo!("§11: dispatch on PauliSumImpl width; call core::propagate")
    }
}
