//! Python `Circuit` class. See §11.

use crate::channel_spec::{ChannelSpec, PyChannel};
use paulistrings::Circuit as CoreCircuit;
use pyo3::exceptions::PyValueError;
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
        for_num_qubits!(num_qubits, |W| CoreCircuit::<W>::new(num_qubits))
    }

    pub fn num_qubits(&self) -> usize {
        for_each_width!(self, |c| c.num_qubits)
    }

    pub fn len(&self) -> usize {
        for_each_width!(self, |c| c.len())
    }

    /// Push a width-erased channel spec onto the underlying circuit.
    pub fn push_spec(&mut self, spec: &ChannelSpec) {
        for_each_width!(self, |c| spec.push_into(c))
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
                PyValueError::new_err("num_qubits exceeds largest monomorphized width (1024)")
            })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Append a `Channel` produced by the `gates`/`noise` factories.
    fn append(&mut self, channel: &PyChannel) {
        self.inner.push_spec(&channel.spec);
    }

    fn h(&mut self, qubit: u32) {
        self.inner.push_spec(&ChannelSpec::H { qubit });
    }

    fn s(&mut self, qubit: u32) {
        self.inner.push_spec(&ChannelSpec::S { qubit });
    }

    fn x(&mut self, qubit: u32) {
        self.inner.push_spec(&ChannelSpec::X { qubit });
    }

    fn y(&mut self, qubit: u32) {
        self.inner.push_spec(&ChannelSpec::Y { qubit });
    }

    fn z(&mut self, qubit: u32) {
        self.inner.push_spec(&ChannelSpec::Z { qubit });
    }

    fn cnot(&mut self, control: u32, target: u32) {
        self.inner.push_spec(&ChannelSpec::Cnot { control, target });
    }

    fn cz(&mut self, q0: u32, q1: u32) {
        self.inner.push_spec(&ChannelSpec::Cz { q0, q1 });
    }

    fn swap(&mut self, q0: u32, q1: u32) {
        self.inner.push_spec(&ChannelSpec::Swap { q0, q1 });
    }

    fn rz(&mut self, theta: f64, qubit: u32) {
        self.inner.push_spec(&ChannelSpec::Rz { theta, qubit });
    }

    fn rx(&mut self, theta: f64, qubit: u32) {
        self.inner.push_spec(&ChannelSpec::Rx { theta, qubit });
    }

    fn ry(&mut self, theta: f64, qubit: u32) {
        self.inner.push_spec(&ChannelSpec::Ry { theta, qubit });
    }

    fn depolarize(&mut self, p: f64, qubits: Vec<u32>) {
        for qubit in qubits {
            self.inner.push_spec(&ChannelSpec::Depolarize { p, qubit });
        }
    }

    fn dephase(&mut self, p: f64, qubits: Vec<u32>) {
        for qubit in qubits {
            self.inner.push_spec(&ChannelSpec::Dephase { p, qubit });
        }
    }

    fn amplitude_damping(&mut self, gamma: f64, qubits: Vec<u32>) {
        for qubit in qubits {
            self.inner
                .push_spec(&ChannelSpec::AmplitudeDamping { gamma, qubit });
        }
    }

    fn unitary_1q(
        &mut self,
        qubit: u32,
        matrix: numpy::PyReadonlyArray2<'_, num_complex::Complex64>,
    ) -> pyo3::PyResult<()> {
        let ch = crate::gates::unitary_1q_spec(qubit, matrix)?;
        self.inner.push_spec(&ch);
        Ok(())
    }

    fn unitary_2q(
        &mut self,
        q0: u32,
        q1: u32,
        matrix: numpy::PyReadonlyArray2<'_, num_complex::Complex64>,
    ) -> pyo3::PyResult<()> {
        let ch = crate::gates::unitary_2q_spec(q0, q1, matrix)?;
        self.inner.push_spec(&ch);
        Ok(())
    }
}
