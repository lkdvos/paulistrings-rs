//! [`TruncationPolicy<W>`] — composable per-term and per-layer term filters.
//!
//! The split between [`TruncationPolicy::keep_term`] (hot, per-output, must
//! inline) and [`TruncationPolicy::finalize_layer`] (cold, once per layer,
//! may be non-local) is performance-critical: `keep_term` runs millions of
//! times per layer, `finalize_layer` runs once.
//!
//! Both methods have default no-op implementations, so a policy only needs
//! to override the one it uses. Compose two policies with [`And`] (both must
//! accept) or [`Or`] (either accepts).
//!
//! Built-ins: [`CoefficientThreshold`] drops terms below a magnitude;
//! [`WeightCutoff`] drops terms above a Pauli weight; [`TopN`] keeps the `n`
//! largest-magnitude terms via a layer-finalization partial sort.
//!
//! See design doc §7.
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

#![allow(unused)]

pub mod builtin;

pub use builtin::{And, CoefficientThreshold, Or, TopN, WeightCutoff};

use crate::bucket::sum::BucketedSum;
use crate::pauli_sum::PauliSum;
use num_complex::Complex64;

/// A truncation strategy. Both methods have sensible defaults so users only
/// implement the one they need.
///
/// # Implementing
///
/// Override [`keep_term`](Self::keep_term) for per-output decisions (runs
/// inside the merge phase, must inline). Override
/// [`finalize_layer`](Self::finalize_layer) for global decisions that
/// depend on the whole layer's output (e.g. partial sort for selecting the
/// top `n` terms).
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
    /// `c` is the **summed** coefficient at the key `(x, z)` — i.e.
    /// `keep_term` runs after all scratch entries with this key have been
    /// reduced, not on individual scratch entries.
    #[inline]
    fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], _c: Complex64) -> bool {
        true
    }

    /// Optional global pass after each circuit layer. May be non-local
    /// (e.g. partial sort for [`TopN`]).
    fn finalize_layer(&self, _sum: &mut PauliSum<W>) {}

    /// The bucketed counterpart of [`finalize_layer`](Self::finalize_layer),
    /// used by the v0.2 engine.
    ///
    /// The default is **correct for any policy**: it collapses the bucketed sum
    /// to a sorted [`PauliSum`], runs `finalize_layer`, and re-partitions. So a
    /// custom policy that only implements `finalize_layer` keeps working
    /// unchanged — no silent loss of truncation.
    ///
    /// It is also the slow path: a `B`-way merge plus a re-scatter, per layer.
    /// **If your policy has no layer-finalization step, override this to a
    /// no-op** — as [`CoefficientThreshold`], [`WeightCutoff`] and [`Or`] do —
    /// or you will pay a round trip per layer for nothing. Overriding it with a
    /// bucket-native implementation, as [`TopN`] does, is better still.
    fn finalize_layer_bucketed(&self, sum: &mut BucketedSum<W>) {
        let mut flat = sum.to_sum();
        self.finalize_layer(&mut flat);
        sum.refill_from_sum(&flat);
    }
}
