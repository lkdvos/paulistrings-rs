//! The [`TruncationPolicy`] trait and the built-in policies. See ARCHITECTURE.md §Truncation.

pub(crate) mod builtin;
mod tree;

pub use builtin::{And, ApproxTopN, CoefficientThreshold, CollapseSample, Or, TopN, WeightCutoff};
pub use tree::BuiltinTruncation;

use crate::collectives::Collectives;
use crate::pauli_sum::PauliSum;
use num_complex::Complex64;

/// A truncation strategy: a per-term filter, a per-layer pass, or both.
///
/// Built-ins: [`CoefficientThreshold`] and [`WeightCutoff`] filter terms, [`TopN`], [`ApproxTopN`] and [`CollapseSample`] run a layer pass, [`And`] and [`Or`] compose, and [`BuiltinTruncation`] holds any of them as one value.
///
/// Above one partition every partition sees a disjoint slice of the layer, so a layer pass must be collective: a policy whose `finalizes_layer()` is `true` runs partitioned only if it overrides [`finalize_layer_partitioned`](Self::finalize_layer_partitioned) and [`supports_partitioned`](Self::supports_partitioned).
/// Every built-in does except exact [`TopN`], which is not yet supported partitioned; [`ApproxTopN`] is the partitioned alternative.
///
/// ```
/// use paulistrings::TruncationPolicy;
/// use num_complex::Complex64;
///
/// /// Drop the imaginary half: keep only terms whose coefficient is real.
/// struct RealOnly;
/// impl<const W: usize> TruncationPolicy<W> for RealOnly {
///     fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], c: Complex64) -> bool {
///         c.im == 0.0
///     }
/// }
/// ```
pub trait TruncationPolicy<const W: usize>: Send + Sync {
    /// Per-term filter, run inside the merge on every output, so it must be cheap.
    ///
    /// `c` is the coefficient summed over every contribution to the key `(x, z)`.
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

    /// The layer pass over one partition's slice of the layer, run on every layer on every partition in lock-step.
    ///
    /// The default runs [`finalize_layer`](Self::finalize_layer) at one partition and nothing above it.
    /// An override must issue the same collective calls on every partition, never skipping one on a partition with nothing to do, and decide only from all-reduced values.
    ///
    /// # Panics
    ///
    /// Above one partition, if the policy reports [`finalizes_layer()`](Self::finalizes_layer) and has not overridden this method.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, collectives: &dyn Collectives) {
        if collectives.size() == 1 {
            self.finalize_layer(local);
            return;
        }
        assert!(
            !self.finalizes_layer(),
            "{} has a layer pass but no partitioned one: every partition sees only a disjoint \
             slice of the layer, so the pass has to be collective. Override \
             `finalize_layer_partitioned` and `supports_partitioned` (see `ApproxTopN`, which \
             all-reduces its octave histogram), or use `ApproxTopN` itself.",
            std::any::type_name_of_val(self),
        );
    }

    /// Whether [`finalize_layer_partitioned`](Self::finalize_layer_partitioned) runs above one partition; every partitioned driver checks it before the first layer.
    ///
    /// The default is `true` exactly when there is no layer pass; a policy that overrides `finalize_layer_partitioned` with a collective pass returns `true`.
    fn supports_partitioned(&self) -> bool {
        !self.finalizes_layer()
    }
}

/// Panics unless `policy` can run on `partitions` partitions.
pub(crate) fn assert_supports_partitions<const W: usize, T>(policy: &T, partitions: usize)
where
    T: TruncationPolicy<W> + ?Sized,
{
    assert!(
        partitions == 1 || policy.supports_partitioned(),
        "{} cannot run on {partitions} partitions: its layer pass has no collective form. \
         Partitioned exact `TopN` is not yet supported; `ApproxTopN` is the partitioned \
         alternative.",
        std::any::type_name_of_val(policy),
    );
}

#[cfg(test)]
mod tests;
