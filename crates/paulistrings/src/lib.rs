//! Classical simulation of quantum circuits by Pauli propagation.
//!
//! The library evolves operators in the Pauli basis under gates and noise
//! channels, in either the forward or Heisenberg picture. It targets
//! workloads where state-vector or tensor-network simulators are infeasible
//! (10⁶–10⁸ terms) but the operator stays sparse in the Pauli basis.
//!
//! It is **not** a state-vector, tensor-network, stabilizer, or MPS
//! simulator — those are explicit non-goals.
//!
//! # Design pillars
//!
//! In priority order:
//!
//! 1. **Correctness** of the Pauli algebra — symplectic encoding, exact
//!    phase tracking, sort-merge invariant restored after every layer.
//! 2. **Performance** at 10⁶–10⁸ terms — structure-of-arrays storage,
//!    sort-merge propagation engine, Rayon-parallel scan and merge.
//! 3. **Extensibility** for research — open [`Channel`] and
//!    [`TruncationPolicy`] traits with built-ins for Clifford gates,
//!    Pauli rotations, depolarizing / dephasing / amplitude-damping noise,
//!    coefficient and weight cutoffs, and `TopN` selection.
//! 4. **GPU readiness** — `#[repr(C)]` `Pod` data types, fixed-fanout
//!    output buffers, shared-nothing parallelism that maps onto CUB
//!    primitives without restructuring.
//!
//! # Quick example
//!
//! Heisenberg-evolve the observable `Z₀ + 0.5·X₁` through an `H` gate on
//! qubit 0:
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
//! - [`PauliString`] — symplectic-encoded Pauli operator on up to `64·W`
//!   qubits, `Copy + Pod`, `Ord`-comparable.
//! - [`PauliSum`] — weighted sum of Pauli strings in structure-of-arrays
//!   form. Sorted-and-deduplicated invariant maintained by every public
//!   operation.
//! - [`BuildAccumulator`] — hashmap-based ingestion path. Used for
//!   constructing a [`PauliSum`] from unsorted inputs (Hamiltonian
//!   parsing, dict construction). Not used during propagation.
//! - [`Phase`] — `i^k` factor (`k ∈ 0..=3`) returned by Pauli
//!   multiplication and folded into [`PauliSum`] coefficients at the
//!   boundary.
//! - [`Circuit`] — ordered, heterogeneous list of channels.
//! - [`Channel`] — extension trait shared by gates and noise. Built-ins
//!   in [`channel`]: [`channel::Clifford1Q`], [`channel::Clifford2Q`],
//!   [`channel::PauliRotation`], [`channel::Depolarizing`],
//!   [`channel::Dephasing`], [`channel::AmplitudeDamping`].
//! - [`TruncationPolicy`] — composable per-term and per-layer term
//!   filters. Built-ins in [`truncation`]: [`truncation::CoefficientThreshold`],
//!   [`truncation::WeightCutoff`], [`truncation::TopN`], plus the
//!   [`truncation::And`] / [`truncation::Or`] combinators.
//! - [`propagate`] / [`Direction`] — the propagation entry point.
//! - [`engine`] — the sort-merge pipeline (scan → sort → merge). Most
//!   users will not call this directly; [`propagate`] is the front door.
//! - [`examples`] — worked-example walkthroughs of full-scale simulations
//!   (currently: a 2D transverse-field Ising quench on 4×4 and 6×6
//!   lattices with embedded plot).
//!
//! # Choosing `W`
//!
//! [`PauliString`] is generic over a const `W: usize` — the number of
//! 64-bit words used to store each of the `x` and `z` parts. So
//! `PauliString<W>` covers up to `64·W` qubits, and the engine
//! monomorphizes the entire pipeline at that `W`.
//!
//! For direct Rust use, pick the smallest `W` that fits your problem:
//! one word per part is dramatically faster than two, and so on. The
//! Python bindings monomorphize at the fixed set `W ∈ {1, 2, 4, 8, 16}`
//! (≤ 64, 128, 256, 512, 1024 qubits) and dispatch on `num_qubits` at
//! the boundary.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![allow(unused)]

pub mod accumulator;
pub mod bucket;
pub mod channel;
pub mod circuit;
pub mod engine;
pub mod examples;
pub mod pauli_string;
pub mod pauli_sum;
pub mod phase;
pub mod truncation;

pub use accumulator::BuildAccumulator;
pub use bucket::Gf2Hash;
pub use channel::{Channel, OutputBuffer};
pub use circuit::Circuit;
pub use engine::{propagate, Direction};
pub use pauli_string::PauliString;
pub use pauli_sum::PauliSum;
pub use phase::Phase;
pub use truncation::TruncationPolicy;
