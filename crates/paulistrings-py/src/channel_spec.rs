//! `PyChannel` — opaque, width-erased handle for the Python boundary.
//!
//! Free-function factories like `gates.h(qubit)` return a `PyChannel` whose
//! width is not yet bound. Width is fixed when the channel is appended to a
//! `Circuit` (via `Circuit.append(...)`), at which point the spec materialises
//! the appropriately monomorphized core channel and pushes it onto the
//! `Circuit<W>`.

use paulistrings::channel::{
    AmplitudeDamping, Clifford1Q, Clifford2Q, Dephasing, Depolarizing, PauliRotation,
};
use paulistrings::pauli_string::PauliString;
use paulistrings::Circuit as CoreCircuit;
use pyo3::prelude::*;

/// What kind of channel `PyChannel` represents. Stored as plain construction
/// parameters so the spec is `Send + Sync` and width-agnostic.
#[derive(Clone, Copy, Debug)]
pub enum ChannelSpec {
    H { qubit: u32 },
    S { qubit: u32 },
    X { qubit: u32 },
    Y { qubit: u32 },
    Z { qubit: u32 },
    Cnot { control: u32, target: u32 },
    Cz { q0: u32, q1: u32 },
    Swap { q0: u32, q1: u32 },
    Rz { theta: f64, qubit: u32 },
    Rx { theta: f64, qubit: u32 },
    Ry { theta: f64, qubit: u32 },
    Depolarize { p: f64, qubit: u32 },
    Dephase { p: f64, qubit: u32 },
    AmplitudeDamping { gamma: f64, qubit: u32 },
}

impl ChannelSpec {
    /// Materialize the spec at width `W` and push it onto `circuit`.
    pub fn push_into<const W: usize>(&self, circuit: &mut CoreCircuit<W>) {
        match *self {
            Self::H { qubit } => circuit.push(Clifford1Q::h(qubit)),
            Self::S { qubit } => circuit.push(Clifford1Q::s(qubit)),
            Self::X { qubit } => circuit.push(Clifford1Q::x(qubit)),
            Self::Y { qubit } => circuit.push(Clifford1Q::y(qubit)),
            Self::Z { qubit } => circuit.push(Clifford1Q::z(qubit)),
            Self::Cnot { control, target } => circuit.push(Clifford2Q::cnot(control, target)),
            Self::Cz { q0, q1 } => circuit.push(Clifford2Q::cz(q0, q1)),
            Self::Swap { q0, q1 } => circuit.push(Clifford2Q::swap(q0, q1)),
            Self::Rz { theta, qubit } => {
                let p = PauliString::<W>::z(qubit);
                circuit.push(PauliRotation::<W> {
                    support: vec![qubit],
                    gen_x: p.x,
                    gen_z: p.z,
                    theta,
                });
            }
            Self::Rx { theta, qubit } => {
                let p = PauliString::<W>::x(qubit);
                circuit.push(PauliRotation::<W> {
                    support: vec![qubit],
                    gen_x: p.x,
                    gen_z: p.z,
                    theta,
                });
            }
            Self::Ry { theta, qubit } => {
                let p = PauliString::<W>::y(qubit);
                circuit.push(PauliRotation::<W> {
                    support: vec![qubit],
                    gen_x: p.x,
                    gen_z: p.z,
                    theta,
                });
            }
            Self::Depolarize { p, qubit } => {
                circuit.push(Depolarizing {
                    support: [qubit],
                    p,
                });
            }
            Self::Dephase { p, qubit } => {
                circuit.push(Dephasing {
                    support: [qubit],
                    p,
                });
            }
            Self::AmplitudeDamping { gamma, qubit } => {
                circuit.push(AmplitudeDamping {
                    support: [qubit],
                    gamma,
                });
            }
        }
    }
}

/// Opaque, width-erased channel handle exposed to Python.
///
/// Construct via the free factories `gates.*` and `noise.*`. Instantiated at
/// the appropriate width when appended to a `Circuit`.
#[pyclass(module = "paulistrings._paulistrings", name = "Channel")]
#[derive(Clone)]
pub struct PyChannel {
    pub(crate) spec: ChannelSpec,
}

impl PyChannel {
    pub fn new(spec: ChannelSpec) -> Self {
        Self { spec }
    }
}

#[pymethods]
impl PyChannel {
    fn __repr__(&self) -> String {
        format!("Channel({:?})", self.spec)
    }
}
