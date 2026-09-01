//! Python `Circuit` class. See ARCHITECTURE.md §Python-Bindings.

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

    /// Push a width-erased channel spec onto the underlying circuit, rejecting
    /// any qubit index the circuit cannot address.
    ///
    /// This is the *only* place a qubit index meets a concrete width: the
    /// `gates`/`noise` factories build width-agnostic specs, so an out-of-range
    /// index can only be caught here. Before the check existed, an index past
    /// `num_qubits` but inside the monomorphized band (e.g. qubit 70 on a
    /// 3-qubit circuit at `W = 1`... or worse, past `64 · W`) either produced
    /// silently wrong results or tripped a `debug_assert` deep in the core.
    pub fn push_spec(&mut self, spec: &ChannelSpec) -> PyResult<()> {
        let n = self.num_qubits();
        let max_qubit = spec.max_qubit();
        if (max_qubit as usize) >= n {
            return Err(PyValueError::new_err(format!(
                "qubit index {max_qubit} is out of range for a {n}-qubit circuit"
            )));
        }
        for_each_width!(self, |c| spec.push_into(c));
        Ok(())
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
    fn append(&mut self, channel: &PyChannel) -> PyResult<()> {
        self.inner.push_spec(&channel.spec)
    }

    fn h(&mut self, qubit: u32) -> PyResult<()> {
        self.inner.push_spec(&ChannelSpec::H { qubit })
    }

    fn s(&mut self, qubit: u32) -> PyResult<()> {
        self.inner.push_spec(&ChannelSpec::S { qubit })
    }

    fn x(&mut self, qubit: u32) -> PyResult<()> {
        self.inner.push_spec(&ChannelSpec::X { qubit })
    }

    fn y(&mut self, qubit: u32) -> PyResult<()> {
        self.inner.push_spec(&ChannelSpec::Y { qubit })
    }

    fn z(&mut self, qubit: u32) -> PyResult<()> {
        self.inner.push_spec(&ChannelSpec::Z { qubit })
    }

    fn cnot(&mut self, control: u32, target: u32) -> PyResult<()> {
        self.inner
            .push_spec(&crate::gates::cnot_spec(control, target)?)
    }

    fn cz(&mut self, q0: u32, q1: u32) -> PyResult<()> {
        self.inner.push_spec(&crate::gates::cz_spec(q0, q1)?)
    }

    fn swap(&mut self, q0: u32, q1: u32) -> PyResult<()> {
        self.inner.push_spec(&crate::gates::swap_spec(q0, q1)?)
    }

    fn rz(&mut self, theta: f64, qubit: u32) -> PyResult<()> {
        self.inner.push_spec(&ChannelSpec::Rz { theta, qubit })
    }

    fn rx(&mut self, theta: f64, qubit: u32) -> PyResult<()> {
        self.inner.push_spec(&ChannelSpec::Rx { theta, qubit })
    }

    fn ry(&mut self, theta: f64, qubit: u32) -> PyResult<()> {
        self.inner.push_spec(&ChannelSpec::Ry { theta, qubit })
    }

    /// A rotation `exp(-i·θ·P/2)` about a Pauli string of any weight; see
    /// `gates.pauli_rotation` for the argument convention.
    fn pauli_rotation(&mut self, pauli: &str, qubits: Vec<u32>, theta: f64) -> PyResult<()> {
        let spec = crate::gates::pauli_rotation_spec(pauli, &qubits, theta)?;
        self.inner.push_spec(&spec)
    }

    fn depolarize(&mut self, p: f64, qubits: Vec<u32>) -> PyResult<()> {
        for qubit in qubits {
            self.inner
                .push_spec(&ChannelSpec::Depolarize { p, qubit })?;
        }
        Ok(())
    }

    fn dephase(&mut self, p: f64, qubits: Vec<u32>) -> PyResult<()> {
        for qubit in qubits {
            self.inner.push_spec(&ChannelSpec::Dephase { p, qubit })?;
        }
        Ok(())
    }

    fn amplitude_damping(&mut self, gamma: f64, qubits: Vec<u32>) -> PyResult<()> {
        for qubit in qubits {
            self.inner
                .push_spec(&ChannelSpec::AmplitudeDamping { gamma, qubit })?;
        }
        Ok(())
    }

    /// A general single-qubit Pauli channel on each of `qubits`; see
    /// `noise.pauli_channel` for the semantics.
    fn pauli_channel(&mut self, px: f64, py: f64, pz: f64, qubits: Vec<u32>) -> PyResult<()> {
        for qubit in qubits {
            let spec = crate::noise::pauli_channel_spec(px, py, pz, qubit)?;
            self.inner.push_spec(&spec)?;
        }
        Ok(())
    }

    /// Uniform two-qubit depolarizing noise on each `(q0, q1)` pair; see
    /// `noise.depolarize2` for the semantics.
    fn depolarize2(&mut self, p: f64, pairs: Vec<(u32, u32)>) -> PyResult<()> {
        for (q0, q1) in pairs {
            let spec = crate::noise::depolarize2_spec(p, q0, q1)?;
            self.inner.push_spec(&spec)?;
        }
        Ok(())
    }

    fn unitary_1q(
        &mut self,
        qubit: u32,
        matrix: numpy::PyReadonlyArray2<'_, num_complex::Complex64>,
    ) -> PyResult<()> {
        let ch = crate::gates::unitary_1q_spec(qubit, matrix)?;
        self.inner.push_spec(&ch)
    }

    fn unitary_2q(
        &mut self,
        q0: u32,
        q1: u32,
        matrix: numpy::PyReadonlyArray2<'_, num_complex::Complex64>,
    ) -> PyResult<()> {
        let ch = crate::gates::unitary_2q_spec(q0, q1, matrix)?;
        self.inner.push_spec(&ch)
    }
}
