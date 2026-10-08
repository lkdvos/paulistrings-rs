//! Classical simulation of quantum circuits by Pauli propagation: the library evolves operators in the Pauli basis under gates and noise channels, forward or Heisenberg, at the term counts (10⁶–10⁸) where state-vector or tensor-network simulators are infeasible. Not a state-vector, tensor-network, stabilizer, or MPS simulator — see ARCHITECTURE.md.
//!
//! # Quick example
//!
//! Heisenberg-evolve the observable `Z₀ + 0.5·X₁` through an `H` gate on qubit 0:
//!
//! ```
//! use paulistrings::{
//!     BuildAccumulator, Circuit, Clifford1Q, Direction, PauliString, Phase, TruncationPolicy,
//!     propagate,
//! };
//! use num_complex::Complex64;
//!
//! let mut acc = BuildAccumulator::<1>::new(2);
//! acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
//! acc.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(0.5, 0.0));
//! let observable = acc.finalize();
//!
//! let mut circuit = Circuit::<1>::new(2);
//! circuit.push(Clifford1Q::h(0));
//!
//! struct KeepAll;
//! impl<const W: usize> TruncationPolicy<W> for KeepAll {}
//!
//! let evolved = propagate(&circuit, observable, &KeepAll, Direction::Heisenberg);
//!
//! // H conjugates Z → X, so the Z₀ term becomes X₀; the X₁ term is untouched.
//! assert_eq!(evolved.len(), 2);
//! ```
//!
//! # Module map
//!
//! - [`PauliString`], [`PauliSum`], [`BuildAccumulator`], [`Phase`] — the data model (ARCHITECTURE.md §Data-Model).
//! - [`Circuit`], [`Channel`] (built-ins such as [`Clifford1Q`] and [`PauliRotation`]) — gates and noise.
//! - [`TruncationPolicy`] (built-ins such as [`CoefficientThreshold`] and [`ApproxTopN`]) — composable per-term and per-layer filters.
//! - [`propagate`] / [`Direction`] / [`propagate_with`] — the propagation entry point (ARCHITECTURE.md §Engine).
//! - [`ProductBasis`] / [`StabilizerState`] — read-out-only contraction states, never evolved.
//! - [`diagonal_echo`] — operator Loschmidt-echo read-outs ([`PauliSum::rotated_overlap`], [`PauliSum::anticommute_histogram`]).
//! - [`PartitionedSum`] / [`DistributedSum`] — the sum split across NUMA domains or ranks (ARCHITECTURE.md §Partitioning); [`propagate`] is the front door for almost all callers.
//! - [`examples`] — worked-example walkthroughs of full-scale simulations.
//!
//! # Choosing `W`
//!
//! [`PauliString`] is generic over a const `W: usize`, the number of 64-bit words per `x`/`z` part; `PauliString<W>` covers up to `64·W` qubits (ARCHITECTURE.md §Width). Pick the smallest `W` that fits; the Python bindings monomorphize at `W ∈ {1, 2, 4, 8, 16}` and dispatch on `num_qubits` at the boundary.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod channel;
mod circuit;
mod engine;
pub mod examples;
mod pauli_string;
mod pauli_sum;
mod phase;
mod readout;
mod rng;
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod test_support;
mod truncation;

pub use channel::{
    support_mask, AmplitudeDamping, Channel, Clifford1Q, Clifford2Q, Dephasing, Depolarizing,
    Depolarizing2Q, GeneralUnitary1Q, GeneralUnitary2Q, IdentityChannel, OutputBuffer,
    PauliChannel, PauliRotation,
};
pub use circuit::Circuit;
pub use engine::bucketed::{GateTrace, LayerScratch, TermTrace};
// The CUDA backend, behind the `cuda` feature.
#[cfg(feature = "cuda")]
pub use engine::gpu;
// The MPI transport and its distributed driver, behind the `mpi` feature.
#[cfg(feature = "mpi")]
pub use engine::partitioned::mpi;
#[cfg(feature = "phase-timing")]
pub use engine::partitioned::PartitionPhaseStats;
pub use engine::partitioned::{
    numa_nodes, propagate_partitioned, Collectives, CpuSet, DistributedSum, PartitionConfig,
    PartitionLayerRecord, PartitionRowPolicy, PartitionRuntime, PartitionSlot, PartitionTrace,
    PartitionedSum, PartitionedTruncation, Placement, TopologyError, Transport,
};
#[cfg(feature = "phase-timing")]
pub use engine::stats::PhaseStats;
pub use engine::{
    propagate, propagate_with, Direction, EngineSelection, PropagateOptions,
    DEFAULT_SMALL_SUM_THRESHOLD,
};
pub use pauli_string::PauliString;
pub use pauli_sum::accumulator::BuildAccumulator;
pub use pauli_sum::{Gf2Hash, PartitionRows, PauliSum, P_MAX_BITS};
pub use phase::Phase;
pub use readout::{
    diagonal_echo, PauliAxis, ProductBasis, ProductState, RotationAxis, StabilizerError,
    StabilizerState,
};
pub use truncation::{
    And, ApproxTopN, BuiltinTruncation, CoefficientThreshold, CollapseSample, Or, TopN,
    TruncationPolicy, WeightCutoff,
};
