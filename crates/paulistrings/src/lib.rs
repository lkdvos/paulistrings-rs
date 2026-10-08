//! Classical simulation of quantum circuits by Pauli propagation: operators evolve in the Pauli basis under gates and noise channels, forward or in the Heisenberg picture, with truncation keeping the sum tractable.
//!
//! The guide, showcases and benchmarks live at <https://lkdvos.github.io/paulistrings-rs/>.
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
//! let mut accumulator = BuildAccumulator::<1>::new(2);
//! accumulator.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
//! accumulator.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(0.5, 0.0));
//! let observable = accumulator.finalize();
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
#[cfg(feature = "cuda")]
pub use engine::gpu;
#[cfg(feature = "mpi")]
pub use engine::partitioned::mpi;
#[cfg(feature = "phase-timing")]
pub use engine::partitioned::PartitionPhaseStats;
pub use engine::partitioned::{
    numa_nodes, propagate_partitioned, Collectives, CpuSet, DistributedSum, PartitionConfig,
    PartitionLayerRecord, PartitionRowPolicy, PartitionRuntime, PartitionSlot, PartitionTrace,
    PartitionedSum, PartitionedTruncation, Placement, ScatterOptions, ScatterRows, TopologyError,
    Transport,
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
