//! Python `Circuit` class. See ARCHITECTURE.md §Python-Bindings.

use crate::channel_spec::{ChannelSpec, PyChannel};
use paulistrings::Circuit as CoreCircuit;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};

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
    /// The width-erased spec of every channel pushed, in order.
    ///
    /// The core `Circuit<W>` stores materialized channels and has no way to
    /// hand a gate back out, so the specs are kept alongside it. They are what
    /// `gates`, slicing, `extend` and `adjoint` all read; `inner` and `specs`
    /// are appended to together (`push`) and never diverge.
    ///
    /// The cost is one spec per channel — dominated by `Unitary2Q`'s 4x4 matrix
    /// at ~256 bytes, negligible against the materialized channel's own
    /// Pauli-transfer matrix.
    specs: Vec<ChannelSpec>,
}

impl Circuit {
    /// Materialize `spec` onto the core circuit and record it.
    ///
    /// Not a `#[pymethods]` member on purpose: `Circuit` deliberately exposes no
    /// Python-level `push` — channels go in through `append` or a named method.
    fn push(&mut self, spec: ChannelSpec) -> PyResult<()> {
        self.inner.push_spec(&spec)?;
        self.specs.push(spec);
        Ok(())
    }

    /// An empty circuit of the same width as `self`.
    fn empty_like(&self) -> PyResult<Self> {
        Self::new(self.inner.num_qubits())
    }

    /// Reject composing two circuits of different widths.
    fn require_same_width(&self, other: &Self, what: &str) -> PyResult<()> {
        let (n, m) = (self.inner.num_qubits(), other.inner.num_qubits());
        if n != m {
            return Err(PyValueError::new_err(format!(
                "Circuit.{what}: cannot compose a {n}-qubit circuit with a \
                 {m}-qubit circuit"
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl Circuit {
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        CircuitImpl::new_for(num_qubits)
            .map(|inner| Self {
                inner,
                specs: Vec::new(),
            })
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

    /// The channel list as task-JSON schema v1 gate objects.
    ///
    /// One dict per channel, in application order — so the list index *is* the
    /// channel index, and a broadcast call like `depolarize(p, [0, 1])` shows up
    /// as the two channels it pushed. Keys: `name`, `qubits` (a list of ints in
    /// the gate's own argument order), plus whichever of `theta` / `pauli` /
    /// `p` / `gamma` / `px`,`py`,`pz` / `matrix` that name carries. A `matrix` is
    /// nested rows of `[re, im]` pairs, so the list is JSON-native as it stands:
    ///
    /// ```python
    /// json.dumps({"version": 1, "n_qubits": c.num_qubits,
    ///             "circuit": {"gates": c.gates}, "run": {"direction": "heisenberg"}})
    /// ```
    ///
    /// `paulistrings.interop.circuit_from_json({"gates": c.gates}, c.num_qubits)`
    /// rebuilds an identical circuit — the round trip is pinned by
    /// `test_circuit_introspection.py`.
    ///
    /// Each call builds fresh dicts; mutating them does not touch the circuit.
    #[getter]
    fn gates<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.specs.iter().map(|s| s.to_gate_dict(py)).collect()
    }

    /// `circuit[i]` is the channel at `i`; `circuit[a:b]` is a new `Circuit` of
    /// the selected channels, at the same width.
    ///
    /// Negative indices and every slice form work as they do for a list,
    /// including a negative step (`circuit[::-1]` reverses the channel order —
    /// note that reversing is *not* the adjoint; see `adjoint()`).
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if let Ok(slice) = key.downcast::<PySlice>() {
            let indices = slice.indices(self.specs.len() as isize)?;
            let mut out = self.empty_like()?;
            let mut i = indices.start;
            for _ in 0..indices.slicelength {
                out.push(self.specs[i as usize].clone())?;
                i += indices.step;
            }
            return Ok(Py::new(py, out)?.into_any());
        }
        let raw: isize = key.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "Circuit indices must be integers or slices, not {}",
                key.get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default(),
            ))
        })?;
        let len = self.specs.len() as isize;
        let i = if raw < 0 { raw + len } else { raw };
        if i < 0 || i >= len {
            return Err(PyIndexError::new_err("Circuit index out of range"));
        }
        Ok(Py::new(py, PyChannel::new(self.specs[i as usize].clone()))?.into_any())
    }

    /// Append every channel of `other` to this circuit, in place.
    ///
    /// Both circuits must have the same `num_qubits`; a circuit's width is part
    /// of its identity, and silently widening a short circuit into a wide one is
    /// exactly the kind of mistake this would otherwise hide.
    ///
    /// Takes the receiver as a `Bound` rather than `&mut self` so that
    /// `c.extend(c)` — a legitimate "repeat this circuit once" — can be served
    /// by cloning the spec list before borrowing mutably, instead of tripping
    /// pyo3's "already mutably borrowed".
    fn extend(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        let specs = if slf.as_any().is(other.as_any()) {
            slf.borrow().specs.clone()
        } else {
            let o = other.borrow();
            slf.borrow().require_same_width(&o, "extend")?;
            o.specs.clone()
        };
        let mut me = slf.borrow_mut();
        for spec in specs {
            me.push(spec)?;
        }
        Ok(())
    }

    /// `a + b`: a new circuit applying `a`'s channels then `b`'s. Neither
    /// operand is modified.
    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.require_same_width(other, "__add__")?;
        let mut out = self.empty_like()?;
        for spec in self.specs.iter().chain(other.specs.iter()) {
            out.push(spec.clone())?;
        }
        Ok(out)
    }

    /// The adjoint circuit: reversed channel order, each gate replaced by its
    /// dagger.
    ///
    /// Its **forward** application equals this circuit's Heisenberg
    /// application, i.e. for any observable
    ///
    /// ```python
    /// obs.propagate(c, direction="heisenberg") == obs.propagate(c.adjoint(), direction="forward")
    /// ```
    ///
    /// to floating-point tolerance (pinned in `test_circuit_introspection.py`).
    /// Per gate: `rz`/`rx`/`ry`/`pauli_rotation` negate `theta`, `s` becomes
    /// `sdg` (and back), `unitary_1q`/`unitary_2q` take the conjugate transpose,
    /// and `h`/`x`/`y`/`z`/`cnot`/`cz`/`swap` are self-adjoint.
    ///
    /// A non-unitary channel raises `ValueError` naming the channel: a noise
    /// channel's dual is not its time-reversal, so there is no inverse to return
    /// (see `ChannelSpec::adjoint`).
    fn adjoint(&self) -> PyResult<Self> {
        let mut out = self.empty_like()?;
        for spec in self.specs.iter().rev() {
            out.push(spec.adjoint()?)?;
        }
        Ok(out)
    }

    /// Append a `Channel` produced by the `gates`/`noise` factories.
    fn append(&mut self, channel: &PyChannel) -> PyResult<()> {
        self.push(channel.spec.clone())
    }

    fn h(&mut self, qubit: u32) -> PyResult<()> {
        self.push(ChannelSpec::H { qubit })
    }

    fn s(&mut self, qubit: u32) -> PyResult<()> {
        self.push(ChannelSpec::S { qubit })
    }

    /// `S^dagger`; see `gates.sdg`.
    fn sdg(&mut self, qubit: u32) -> PyResult<()> {
        self.push(ChannelSpec::Sdg { qubit })
    }

    fn x(&mut self, qubit: u32) -> PyResult<()> {
        self.push(ChannelSpec::X { qubit })
    }

    fn y(&mut self, qubit: u32) -> PyResult<()> {
        self.push(ChannelSpec::Y { qubit })
    }

    fn z(&mut self, qubit: u32) -> PyResult<()> {
        self.push(ChannelSpec::Z { qubit })
    }

    fn cnot(&mut self, control: u32, target: u32) -> PyResult<()> {
        self.push(crate::gates::cnot_spec(control, target)?)
    }

    fn cz(&mut self, q0: u32, q1: u32) -> PyResult<()> {
        self.push(crate::gates::cz_spec(q0, q1)?)
    }

    fn swap(&mut self, q0: u32, q1: u32) -> PyResult<()> {
        self.push(crate::gates::swap_spec(q0, q1)?)
    }

    fn rz(&mut self, theta: f64, qubit: u32) -> PyResult<()> {
        self.push(ChannelSpec::Rz { theta, qubit })
    }

    fn rx(&mut self, theta: f64, qubit: u32) -> PyResult<()> {
        self.push(ChannelSpec::Rx { theta, qubit })
    }

    fn ry(&mut self, theta: f64, qubit: u32) -> PyResult<()> {
        self.push(ChannelSpec::Ry { theta, qubit })
    }

    /// A rotation `exp(-i·θ·P/2)` about a Pauli string of any weight; see
    /// `gates.pauli_rotation` for the argument convention.
    fn pauli_rotation(&mut self, pauli: &str, qubits: Vec<u32>, theta: f64) -> PyResult<()> {
        let spec = crate::gates::pauli_rotation_spec(pauli, &qubits, theta)?;
        self.push(spec)
    }

    fn depolarize(&mut self, p: f64, qubits: Vec<u32>) -> PyResult<()> {
        for qubit in qubits {
            self.push(ChannelSpec::Depolarize { p, qubit })?;
        }
        Ok(())
    }

    fn dephase(&mut self, p: f64, qubits: Vec<u32>) -> PyResult<()> {
        for qubit in qubits {
            self.push(ChannelSpec::Dephase { p, qubit })?;
        }
        Ok(())
    }

    fn amplitude_damping(&mut self, gamma: f64, qubits: Vec<u32>) -> PyResult<()> {
        for qubit in qubits {
            self.push(ChannelSpec::AmplitudeDamping { gamma, qubit })?;
        }
        Ok(())
    }

    /// A general single-qubit Pauli channel on each of `qubits`; see
    /// `noise.pauli_channel` for the semantics.
    fn pauli_channel(&mut self, px: f64, py: f64, pz: f64, qubits: Vec<u32>) -> PyResult<()> {
        for qubit in qubits {
            let spec = crate::noise::pauli_channel_spec(px, py, pz, qubit)?;
            self.push(spec)?;
        }
        Ok(())
    }

    /// Uniform two-qubit depolarizing noise on each `(q0, q1)` pair; see
    /// `noise.depolarize2` for the semantics.
    fn depolarize2(&mut self, p: f64, pairs: Vec<(u32, u32)>) -> PyResult<()> {
        for (q0, q1) in pairs {
            let spec = crate::noise::depolarize2_spec(p, q0, q1)?;
            self.push(spec)?;
        }
        Ok(())
    }

    fn unitary_1q(
        &mut self,
        qubit: u32,
        matrix: numpy::PyReadonlyArray2<'_, num_complex::Complex64>,
    ) -> PyResult<()> {
        let ch = crate::gates::unitary_1q_spec(qubit, matrix)?;
        self.push(ch)
    }

    fn unitary_2q(
        &mut self,
        q0: u32,
        q1: u32,
        matrix: numpy::PyReadonlyArray2<'_, num_complex::Complex64>,
    ) -> PyResult<()> {
        let ch = crate::gates::unitary_2q_spec(q0, q1, matrix)?;
        self.push(ch)
    }
}
