//! [`BuiltinTruncation`], every builtin policy and combinator as one runtime value.

use super::builtin::{
    And, ApproxTopN, CoefficientThreshold, CollapseSample, Or, TopN, WeightCutoff,
};
use super::TruncationPolicy;
use crate::pauli_sum::PauliSum;
use num_complex::Complex64;
use std::sync::Arc;

/// The builtin truncation policies as one value-level tree, the form a backend lowers.
///
/// Every variant delegates to the builtin type it names, so a `BuiltinTruncation` truncates exactly as the corresponding composition of [`CoefficientThreshold`], [`WeightCutoff`], [`TopN`], [`ApproxTopN`], [`CollapseSample`], [`And`](super::And) and [`Or`](super::Or) does, on the host and in partitioned mode.
/// Every builtin and combinator converts into one with [`From`], which is how the CUDA backend takes a policy; a custom [`TruncationPolicy`] has no conversion, so it cannot reach a device.
///
/// `Or` combines per-term filters only and runs neither side's layer pass, matching [`Or`](super::Or); `And` runs both, first then second.
/// In partitioned mode an exact `TopN` whose layer pass would run panics, since it has no collective form (see [`PartitionedTruncation`](crate::PartitionedTruncation)).
/// A [`CollapseSample`] is shared, not copied, by `Clone`: its pass counter is the trajectory's state, so every tree holding one handle continues the same sequence. No device backend runs it.
///
/// ```
/// use paulistrings::truncation::BuiltinTruncation as T;
/// use paulistrings::TruncationPolicy;
///
/// let policy = T::And(Box::new(T::Coeff(1e-9)), Box::new(T::ApproxTopN(1_000)));
/// assert!(<_ as TruncationPolicy<1>>::finalizes_layer(&policy));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum BuiltinTruncation {
    /// No filtering; exact zeros are still dropped by the merge.
    Keep,
    /// [`CoefficientThreshold`]`(eps)`.
    Coeff(f64),
    /// [`WeightCutoff`]`(k)`.
    Weight(u32),
    /// [`TopN`]`(n)`.
    TopN(usize),
    /// [`ApproxTopN`]`(n)`.
    ApproxTopN(usize),
    /// [`CollapseSample`], one trajectory shared by every clone of the tree.
    CollapseSample(Arc<CollapseSample>),
    /// [`And`](super::And) of two policies.
    And(Box<BuiltinTruncation>, Box<BuiltinTruncation>),
    /// [`Or`](super::Or) of two policies.
    Or(Box<BuiltinTruncation>, Box<BuiltinTruncation>),
}

impl BuiltinTruncation {
    /// Whether an exact [`TopN`] appears anywhere in the tree, `Or` branches included.
    pub fn contains_exact_top_n(&self) -> bool {
        match self {
            Self::TopN(_) => true,
            Self::And(a, b) | Self::Or(a, b) => {
                a.contains_exact_top_n() || b.contains_exact_top_n()
            }
            Self::Keep
            | Self::Coeff(_)
            | Self::Weight(_)
            | Self::ApproxTopN(_)
            | Self::CollapseSample(_) => false,
        }
    }

    /// Whether a [`CollapseSample`] appears anywhere in the tree, `Or` branches included; no device backend runs one.
    pub fn contains_collapse_sample(&self) -> bool {
        match self {
            Self::CollapseSample(_) => true,
            Self::And(a, b) | Self::Or(a, b) => {
                a.contains_collapse_sample() || b.contains_collapse_sample()
            }
            Self::Keep | Self::Coeff(_) | Self::Weight(_) | Self::TopN(_) | Self::ApproxTopN(_) => {
                false
            }
        }
    }
}

impl<const W: usize> TruncationPolicy<W> for BuiltinTruncation {
    #[inline]
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
        match self {
            Self::Keep => true,
            Self::Coeff(eps) => CoefficientThreshold(*eps).keep_term(x, z, c),
            Self::Weight(k) => WeightCutoff(*k).keep_term(x, z, c),
            Self::TopN(n) => <TopN as TruncationPolicy<W>>::keep_term(&TopN(*n), x, z, c),
            Self::ApproxTopN(n) => {
                <ApproxTopN as TruncationPolicy<W>>::keep_term(&ApproxTopN(*n), x, z, c)
            }
            Self::CollapseSample(s) => {
                <CollapseSample as TruncationPolicy<W>>::keep_term(s, x, z, c)
            }
            Self::And(a, b) => a.keep_term(x, z, c) && b.keep_term(x, z, c),
            Self::Or(a, b) => a.keep_term(x, z, c) || b.keep_term(x, z, c),
        }
    }

    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        match self {
            Self::TopN(n) => TopN(*n).finalize_layer(sum),
            Self::ApproxTopN(n) => ApproxTopN(*n).finalize_layer(sum),
            Self::CollapseSample(s) => {
                <CollapseSample as TruncationPolicy<W>>::finalize_layer(s, sum)
            }
            Self::And(a, b) => {
                a.finalize_layer(sum);
                b.finalize_layer(sum);
            }
            Self::Keep | Self::Coeff(_) | Self::Weight(_) | Self::Or(_, _) => {}
        }
    }

    fn finalizes_layer(&self) -> bool {
        match self {
            Self::TopN(_) | Self::ApproxTopN(_) | Self::CollapseSample(_) => true,
            Self::And(a, b) => {
                <Self as TruncationPolicy<W>>::finalizes_layer(a)
                    || <Self as TruncationPolicy<W>>::finalizes_layer(b)
            }
            Self::Keep | Self::Coeff(_) | Self::Weight(_) | Self::Or(_, _) => false,
        }
    }
}

impl From<CoefficientThreshold> for BuiltinTruncation {
    fn from(p: CoefficientThreshold) -> Self {
        Self::Coeff(p.0)
    }
}

impl From<WeightCutoff> for BuiltinTruncation {
    fn from(p: WeightCutoff) -> Self {
        Self::Weight(p.0)
    }
}

impl From<TopN> for BuiltinTruncation {
    fn from(p: TopN) -> Self {
        Self::TopN(p.0)
    }
}

impl From<ApproxTopN> for BuiltinTruncation {
    fn from(p: ApproxTopN) -> Self {
        Self::ApproxTopN(p.0)
    }
}

impl From<CollapseSample> for BuiltinTruncation {
    fn from(p: CollapseSample) -> Self {
        Self::CollapseSample(Arc::new(p))
    }
}

impl From<Arc<CollapseSample>> for BuiltinTruncation {
    fn from(p: Arc<CollapseSample>) -> Self {
        Self::CollapseSample(p)
    }
}

impl<A: Into<BuiltinTruncation>, B: Into<BuiltinTruncation>> From<And<A, B>> for BuiltinTruncation {
    fn from(p: And<A, B>) -> Self {
        Self::And(Box::new(p.0.into()), Box::new(p.1.into()))
    }
}

impl<A: Into<BuiltinTruncation>, B: Into<BuiltinTruncation>> From<Or<A, B>> for BuiltinTruncation {
    fn from(p: Or<A, B>) -> Self {
        Self::Or(Box::new(p.0.into()), Box::new(p.1.into()))
    }
}

/// A borrowed policy converts as its clone does, so a device driver takes `&policy` as the host engine does.
impl<P: Clone + Into<BuiltinTruncation>> From<&P> for BuiltinTruncation {
    fn from(p: &P) -> Self {
        p.clone().into()
    }
}

#[cfg(test)]
mod tests;
