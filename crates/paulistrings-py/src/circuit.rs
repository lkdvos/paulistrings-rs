//! Python `Circuit` class. See §11.

#![allow(unused)]

use paulistrings::Circuit as CoreCircuit;
use pyo3::prelude::*;

pub enum CircuitImpl {
    W1(CoreCircuit<1>),
    W2(CoreCircuit<2>),
    W4(CoreCircuit<4>),
    W8(CoreCircuit<8>),
    W16(CoreCircuit<16>),
}

impl CircuitImpl {
    pub fn new_for(num_qubits: usize) -> Option<Self> {
        match num_qubits {
            0..=64 => Some(Self::W1(CoreCircuit::new(num_qubits))),
            65..=128 => Some(Self::W2(CoreCircuit::new(num_qubits))),
            129..=256 => Some(Self::W4(CoreCircuit::new(num_qubits))),
            257..=512 => Some(Self::W8(CoreCircuit::new(num_qubits))),
            513..=1024 => Some(Self::W16(CoreCircuit::new(num_qubits))),
            _ => None,
        }
    }

    pub fn num_qubits(&self) -> usize {
        match self {
            Self::W1(c) => c.num_qubits,
            Self::W2(c) => c.num_qubits,
            Self::W4(c) => c.num_qubits,
            Self::W8(c) => c.num_qubits,
            Self::W16(c) => c.num_qubits,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::W1(c) => c.len(),
            Self::W2(c) => c.len(),
            Self::W4(c) => c.len(),
            Self::W8(c) => c.len(),
            Self::W16(c) => c.len(),
        }
    }
}

#[pyclass(module = "paulistrings._paulistrings", name = "Circuit")]
pub struct Circuit {
    pub(crate) inner: CircuitImpl,
}

#[pymethods]
impl Circuit {
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        CircuitImpl::new_for(num_qubits)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "num_qubits exceeds largest monomorphized width (1024)",
                )
            })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn h(&mut self, _qubit: u32) -> PyResult<()> {
        todo!("§11: append a Hadamard Clifford1Q to inner")
    }

    fn cnot(&mut self, _control: u32, _target: u32) -> PyResult<()> {
        todo!("§11: append a CNOT Clifford2Q to inner")
    }

    fn rz(&mut self, _theta: f64, _qubit: u32) -> PyResult<()> {
        todo!("§11: append a PauliRotation about Z to inner")
    }

    fn depolarize(&mut self, _p: f64, _qubits: Vec<u32>) -> PyResult<()> {
        todo!("§11: append one Depolarizing per qubit (or a multi-qubit form)")
    }
}
