//! The Rust core of `paulistrings-rs`: operators evolved in the Pauli basis under gates and noise channels, stored as a GF(2)-bucketed `PauliSum` and optionally partitioned over NUMA domains, MPI ranks or CUDA devices.
//!
//! These docs are a developer guide; the user-facing API is the Python package, documented at <https://lkdvos.github.io/paulistrings-rs/>.
//! [`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) is the design source of truth.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod channel;
mod circuit;
mod collectives;
mod engine;
mod pauli_string;
mod pauli_sum;
mod phase;
mod readout;
mod rng;
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod test_support;
mod truncation;

pub use channel::{support_mask, Channel, OutputBuffer, PauliRotation, PtmChannel};
pub use circuit::Circuit;
pub use collectives::Collectives;
pub use engine::bucketed::{LayerScratch, LayerTrace};
#[cfg(feature = "cuda")]
pub use engine::gpu;
#[cfg(feature = "mpi")]
pub use engine::partitioned::mpi;
#[cfg(feature = "phase-timing")]
pub use engine::partitioned::PartitionPhaseStats;
pub use engine::partitioned::{
    numa_nodes, propagate_partitioned, CpuSet, DistributedSum, PartitionConfig,
    PartitionLayerRecord, PartitionRowPolicy, PartitionRuntime, PartitionSlot, PartitionTrace,
    PartitionedSum, Placement, ScatterOptions, ScatterRows, TopologyError, Transport,
};
#[cfg(feature = "phase-timing")]
pub use engine::stats::PhaseStats;
pub use engine::{propagate, propagate_with, Direction, PropagateOptions};
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
