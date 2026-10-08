//! The [`TruncationPolicy`] trait and the built-in policies. See ARCHITECTURE.md §Truncation.

pub(crate) mod builtin;
mod tree;

pub use builtin::{And, ApproxTopN, CoefficientThreshold, CollapseSample, Or, TopN, WeightCutoff};
pub use tree::BuiltinTruncation;

use crate::pauli_sum::PauliSum;
use num_complex::Complex64;

/// A truncation strategy: a per-term filter, a per-layer pass, or both.
///
/// Built-ins: [`CoefficientThreshold`] and [`WeightCutoff`] filter terms, [`TopN`], [`ApproxTopN`] and [`CollapseSample`] run a layer pass, [`And`] and [`Or`] compose, and [`BuiltinTruncation`] holds any of them as one value.
///
/// ```
/// use paulistrings::TruncationPolicy;
/// use num_complex::Complex64;
///
/// /// Drop the imaginary half: keep only terms whose coefficient is real.
/// struct RealOnly;
/// impl<const W: usize> TruncationPolicy<W> for RealOnly {
///     #[inline]
///     fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], c: Complex64) -> bool {
///         c.im == 0.0
///     }
/// }
/// ```
pub trait TruncationPolicy<const W: usize>: Send + Sync {
    /// Per-term filter, run inside the merge on every output, so it must be cheap.
    ///
    /// `c` is the coefficient summed over every contribution to the key `(x, z)`.
    #[inline]
    fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], _c: Complex64) -> bool {
        true
    }

    /// Pass over the whole sum after each layer; [`PauliSum::retain`] is the easy way to write a filter here.
    fn finalize_layer(&self, _sum: &mut PauliSum<W>) {}

    /// Whether [`finalize_layer`](Self::finalize_layer) does anything, a hint that lets the engine skip it.
    ///
    /// Return `false` only if `finalize_layer` is a no-op: the engine may then skip it, which is a wrong answer otherwise.
    fn finalizes_layer(&self) -> bool {
        true
    }
}
