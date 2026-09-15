//! Classical simulation of quantum circuits by Pauli propagation: the library evolves operators in the Pauli basis under gates and noise channels, forward or Heisenberg, at the term counts (10⁶–10⁸) where state-vector or tensor-network simulators are infeasible. Not a state-vector, tensor-network, stabilizer, or MPS simulator — see ARCHITECTURE.md.
//!
//! # Quick example
//!
//! Heisenberg-evolve the observable `Z₀ + 0.5·X₁` through an `H` gate on qubit 0:
//!
//! ```
//! use paulistrings::{
//!     BuildAccumulator, Circuit, Direction, PauliString, Phase, TruncationPolicy,
//!     channel::Clifford1Q, propagate,
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
//! - [`Circuit`], [`Channel`] (built-ins in [`channel`]) — gates and noise.
//! - [`TruncationPolicy`] (built-ins in [`truncation`]) — composable per-term and per-layer filters.
//! - [`propagate`] / [`Direction`] / [`propagate_with_options`] — the propagation entry point (ARCHITECTURE.md §Engine).
//! - [`ProductBasis`] / [`StabilizerState`] — read-out-only contraction states, never evolved.
//! - [`engine`] / [`engine::partitioned`] — the bucketed engine and its NUMA/distributed partitioning (ARCHITECTURE.md §Engine, §Partitioning); [`propagate`] is the front door for almost all callers.
//! - [`examples`] — worked-example walkthroughs of full-scale simulations.
//!
//! # Choosing `W`
//!
//! [`PauliString`] is generic over a const `W: usize`, the number of 64-bit words per `x`/`z` part; `PauliString<W>` covers up to `64·W` qubits (ARCHITECTURE.md §Width). Pick the smallest `W` that fits; the Python bindings monomorphize at `W ∈ {1, 2, 4, 8, 16}` and dispatch on `num_qubits` at the boundary.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

pub mod accumulator;
pub mod bucket;
pub mod channel;
pub mod circuit;
pub mod engine;
pub mod examples;
pub mod pauli_string;
pub mod pauli_sum;
pub mod phase;
pub mod stabilizer;
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
pub mod test_support;
pub mod truncation;

pub use accumulator::BuildAccumulator;
pub use bucket::{Gf2Hash, PartitionRows};
pub use channel::{Channel, OutputBuffer};
pub use circuit::Circuit;
pub use engine::bucketed::{GateTrace, LayerScratch, TermTrace};
// The MPI transport and its distributed driver, behind the `mpi` feature.
#[cfg(feature = "mpi")]
pub use engine::partitioned::mpi;
#[cfg(feature = "phase-timing")]
pub use engine::partitioned::PartitionPhaseStats;
pub use engine::partitioned::{
    circuit_generators, count_remote_deltas, propagate_partitioned,
    propagate_partitioned_with_options, DistributedSum, GeneratorWeight, PartitionConfig,
    PartitionLayerRecord, PartitionRowPolicy, PartitionRuntime, PartitionTrace, PartitionedSum,
    PartitionedTruncation, Placement, TopologyError,
};
#[cfg(feature = "phase-timing")]
pub use engine::stats::PhaseStats;
pub use engine::{
    default_min_buckets, propagate, propagate_with_options, propagate_with_scratch,
    propagate_with_scratch_and_options, Direction, EngineSelection, PropagateOptions,
    DEFAULT_SMALL_SUM_THRESHOLD,
};
pub use pauli_string::PauliString;
pub use pauli_sum::{PauliAxis, PauliSum, ProductBasis, ProductState};
pub use phase::Phase;
pub use stabilizer::{StabilizerError, StabilizerState};
pub use truncation::TruncationPolicy;
