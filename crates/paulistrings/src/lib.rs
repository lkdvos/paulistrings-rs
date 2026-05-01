//! Pauli propagation in Rust.
//!
//! This crate is the pure-Rust core of `paulistrings`. It contains the
//! `PauliString<W>` / `PauliSum<W>` data types, the `Channel` and
//! `TruncationPolicy` extension traits, and the sort-merge propagation engine.
//!
//! Status: v0.1 scaffolding. Module surfaces match the design document at
//! `research/plans/2026-04-30-v0.1-scope.md`; algorithm bodies are stubs.

#![allow(unused)]

pub mod accumulator;
pub mod channel;
pub mod circuit;
pub mod engine;
pub mod pauli_string;
pub mod pauli_sum;
pub mod truncation;

pub use accumulator::BuildAccumulator;
pub use channel::{Channel, OutputBuffer};
pub use circuit::Circuit;
pub use engine::{propagate, Direction};
pub use pauli_string::PauliString;
pub use pauli_sum::PauliSum;
pub use truncation::TruncationPolicy;
