//! [`BuiltinTruncation`], every builtin policy and combinator as one runtime value.

use super::builtin::{
    or_operand_error, And, ApproxTopN, CoefficientThreshold, CollapseSample, Or, TopN, WeightCutoff,
};
use super::TruncationPolicy;
use crate::collectives::Collectives;
use crate::pauli_sum::PauliSum;
use num_complex::Complex64;
use std::sync::Arc;

/// Any builtin policy or combinator as one runtime value, the form the device drivers take.
///
/// Each variant truncates exactly as the builtin it names, and every builtin converts into one with [`From`]; a custom [`TruncationPolicy`] does not.
/// A tree holding an exact `TopN` cannot run above one partition ([`supports_partitioned`](Self::supports_partitioned) is `false`).
/// `Or` combines per-term filters only: build it with [`or`](Self::or) or [`try_or`](Self::try_or), which reject an operand with a layer pass; a hand-built `Or` variant that holds one panics in [`finalizes_layer`](TruncationPolicy::finalizes_layer) and [`supports_partitioned`](Self::supports_partitioned).
/// `Clone` shares a [`CollapseSample`] rather than copying it, so every clone continues the same trajectory.
#[derive(Clone, Debug, PartialEq)]
pub enum BuiltinTruncation {
    /// No truncation.
    Keep,
    /// [`CoefficientThreshold`]`(eps)`.
    Coefficient(f64),
    /// [`WeightCutoff`]`(k)`.
    Weight(u32),
    /// [`TopN`]`(n)`.
    TopN(usize),
    /// [`ApproxTopN`]`(n)`.
    ApproxTopN(usize),
    /// [`CollapseSample`].
    CollapseSample(Arc<CollapseSample>),
    /// [`And`] of two policies.
    And(Box<BuiltinTruncation>, Box<BuiltinTruncation>),
    /// [`Or`] of two per-term filters.
    Or(Box<BuiltinTruncation>, Box<BuiltinTruncation>),
}

impl BuiltinTruncation {
    /// `first` or `second`, both per-term filters.
    ///
    /// # Panics
    ///
    /// If either operand has a layer pass (see [`try_or`](Self::try_or)).
    pub fn or(first: BuiltinTruncation, second: BuiltinTruncation) -> Self {
        Self::try_or(first, second).unwrap_or_else(|message| panic!("{message}"))
    }

    /// `first` or `second`, or the reason it is rejected: an operand with a layer pass (`TopN`, `ApproxTopN`, `CollapseSample`, or an `And` containing one).
    pub fn try_or(first: BuiltinTruncation, second: BuiltinTruncation) -> Result<Self, String> {
        match or_operand_error(first.has_layer_pass(), second.has_layer_pass()) {
            Some(message) => Err(message),
            None => Ok(Self::Or(Box::new(first), Box::new(second))),
        }
    }

    /// Whether the tree runs a layer pass; panics on an `Or` holding one.
    fn has_layer_pass(&self) -> bool {
        match self {
            Self::TopN(_) | Self::ApproxTopN(_) | Self::CollapseSample(_) => true,
            Self::And(a, b) => a.has_layer_pass() || b.has_layer_pass(),
            Self::Or(a, b) => {
                if let Some(message) = or_operand_error(a.has_layer_pass(), b.has_layer_pass()) {
                    panic!("{message}");
                }
                false
            }
            Self::Keep | Self::Coefficient(_) | Self::Weight(_) => false,
        }
    }

    /// Whether the tree runs above one partition: `false` exactly when it holds an exact `TopN`.
    ///
    /// # Panics
    ///
    /// On an `Or` variant holding a layer pass.
    pub fn supports_partitioned(&self) -> bool {
        match self {
            Self::TopN(_) => false,
            Self::And(a, b) => a.supports_partitioned() && b.supports_partitioned(),
            Self::Or(..) => !self.has_layer_pass(),
            Self::Keep
            | Self::Coefficient(_)
            | Self::Weight(_)
            | Self::ApproxTopN(_)
            | Self::CollapseSample(_) => true,
        }
    }

    /// Whether a [`CollapseSample`] appears anywhere in the tree, `Or` branches included.
    pub fn contains_collapse_sample(&self) -> bool {
        match self {
            Self::CollapseSample(_) => true,
            Self::And(a, b) | Self::Or(a, b) => {
                a.contains_collapse_sample() || b.contains_collapse_sample()
            }
            Self::Keep
            | Self::Coefficient(_)
            | Self::Weight(_)
            | Self::TopN(_)
            | Self::ApproxTopN(_) => false,
        }
    }
}

impl<const W: usize> TruncationPolicy<W> for BuiltinTruncation {
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
        match self {
            Self::Keep => true,
            Self::Coefficient(eps) => CoefficientThreshold(*eps).keep_term(x, z, c),
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
            Self::Keep | Self::Coefficient(_) | Self::Weight(_) | Self::Or(_, _) => {}
        }
    }

    fn finalizes_layer(&self) -> bool {
        self.has_layer_pass()
    }

    /// Arm for arm [`finalize_layer`](TruncationPolicy::finalize_layer), each leaf by its own collective form.
    fn finalize_layer_partitioned(&self, local: &mut PauliSum<W>, collectives: &dyn Collectives) {
        match self {
            Self::TopN(n) => <TopN as TruncationPolicy<W>>::finalize_layer_partitioned(
                &TopN(*n),
                local,
                collectives,
            ),
            Self::ApproxTopN(n) => <ApproxTopN as TruncationPolicy<W>>::finalize_layer_partitioned(
                &ApproxTopN(*n),
                local,
                collectives,
            ),
            Self::CollapseSample(s) => {
                <CollapseSample as TruncationPolicy<W>>::finalize_layer_partitioned(
                    s,
                    local,
                    collectives,
                )
            }
            Self::And(a, b) => {
                <Self as TruncationPolicy<W>>::finalize_layer_partitioned(a, local, collectives);
                <Self as TruncationPolicy<W>>::finalize_layer_partitioned(b, local, collectives);
            }
            Self::Keep | Self::Coefficient(_) | Self::Weight(_) | Self::Or(_, _) => {}
        }
    }

    fn supports_partitioned(&self) -> bool {
        BuiltinTruncation::supports_partitioned(self)
    }
}

impl From<CoefficientThreshold> for BuiltinTruncation {
    fn from(p: CoefficientThreshold) -> Self {
        Self::Coefficient(p.0)
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

/// # Panics
///
/// If either operand has a layer pass, as [`BuiltinTruncation::or`].
impl<A: Into<BuiltinTruncation>, B: Into<BuiltinTruncation>> From<Or<A, B>> for BuiltinTruncation {
    fn from(p: Or<A, B>) -> Self {
        Self::or(p.0.into(), p.1.into())
    }
}

/// A borrowed policy converts as its clone does.
impl<P: Clone + Into<BuiltinTruncation>> From<&P> for BuiltinTruncation {
    fn from(p: &P) -> Self {
        p.clone().into()
    }
}

#[cfg(test)]
mod tests;
