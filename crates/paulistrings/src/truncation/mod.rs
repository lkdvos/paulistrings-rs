//! `TruncationPolicy<W>` — composable per-term and per-layer term filters.
//!
//! See design doc §7. The split between `keep_term` (hot, per-output) and
//! `finalize_layer` (cold, once per layer) is performance-critical.

#![allow(unused)]

pub mod builtin;

pub use builtin::{And, CoefficientThreshold, Or, TopN, WeightCutoff};

use crate::pauli_sum::PauliSum;
use num_complex::Complex64;

/// A truncation strategy. Both methods have sensible defaults so users only
/// implement the one they need.
pub trait TruncationPolicy<const W: usize>: Send + Sync {
    /// Cheap per-term filter applied during the merge phase. Must inline.
    #[inline]
    fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], _c: Complex64) -> bool {
        true
    }

    /// Optional global pass after each circuit layer. May be non-local
    /// (e.g. partial sort for `TopN`).
    fn finalize_layer(&self, _sum: &mut PauliSum<W>) {}
}
