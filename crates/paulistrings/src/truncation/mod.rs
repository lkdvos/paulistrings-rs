//! [`TruncationPolicy<W>`] — composable per-term and per-layer term filters.
//!
//! Built-ins: [`CoefficientThreshold`] drops terms below a magnitude, [`WeightCutoff`] drops terms above a Pauli weight, [`TopN`] keeps at most `n` largest-magnitude terms without splitting a tie group, and [`ApproxTopN`] trades the exact count for a cheaper histogram threshold.
//! Compose with [`And`] (both must accept) or [`Or`] (either accepts).
//!
//! # Examples
//!
//! ```
//! use paulistrings::truncation::{And, CoefficientThreshold, WeightCutoff};
//! use paulistrings::TruncationPolicy;
//! use num_complex::Complex64;
//!
//! // Keep terms with |coeff| > 1e-9 AND weight ≤ 4.
//! let policy = And(CoefficientThreshold(1e-9), WeightCutoff(4));
//!
//! // A weight-1 term with coeff 0.5: passes both.
//! assert!(<_ as TruncationPolicy<1>>::keep_term(
//!     &policy, &[1], &[0], Complex64::new(0.5, 0.0),
//! ));
//! ```

pub mod builtin;

pub use builtin::{And, ApproxTopN, CoefficientThreshold, Or, TopN, WeightCutoff};

use crate::pauli_sum::PauliSum;
use num_complex::Complex64;

/// A truncation strategy. Both methods have sensible defaults so users only
/// implement the one they need.
///
/// # Implementing
///
/// Override [`keep_term`](Self::keep_term) for per-output decisions (runs inside the merge phase, must inline).
/// Override [`finalize_layer`](Self::finalize_layer) for global decisions that depend on the whole layer's output (e.g. partial sort for selecting the top `n` terms).
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
    /// Cheap per-term filter applied during the merge phase. Must inline.
    ///
    /// `c` is the summed coefficient at the key `(x, z)`: `keep_term` runs after all scratch entries with this key have been reduced, not on individual scratch entries.
    #[inline]
    fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], _c: Complex64) -> bool {
        true
    }

    /// Optional global pass after each circuit layer. May be non-local (e.g. partial sort for [`TopN`]).
    ///
    /// The sum arrives in its bucketed working form.
    /// A filter-shaped policy is easiest written with [`PauliSum::retain`] (per-bucket parallel, in place, preserves every invariant); a policy needing a global view first can read terms via [`PauliSum::iter`] or per bucket via [`PauliSum::bucket`].
    fn finalize_layer(&self, _sum: &mut PauliSum<W>) {}

    /// Whether [`finalize_layer`](Self::finalize_layer) does anything at all.
    ///
    /// Purely an optimization hint read only by the engine: it never changes what a policy means.
    /// The engine's small-sum direct path holds the sum in a hash map between layers, so honouring a layer pass there costs a full materialize → finalize → re-ingest round trip; this method lets a policy say that round trip is pointless.
    ///
    /// The default is `true`, the conservative answer.
    /// Override with `false` only if `finalize_layer` is genuinely a no-op: returning `false` while `finalize_layer` does something makes the direct path skip it, which is a wrong answer, not a slow one.
    ///
    /// ```
    /// use paulistrings::TruncationPolicy;
    /// use paulistrings::truncation::{CoefficientThreshold, TopN};
    ///
    /// assert!(!<_ as TruncationPolicy<1>>::finalizes_layer(&CoefficientThreshold(1e-9)));
    /// assert!(<_ as TruncationPolicy<1>>::finalizes_layer(&TopN(1000)));
    /// ```
    fn finalizes_layer(&self) -> bool {
        true
    }
}
