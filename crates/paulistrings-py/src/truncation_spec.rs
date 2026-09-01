//! `PyTruncation` — opaque, width-erased truncation policy handle.
//!
//! Free factories `truncation.coeff(...)`, `truncation.weight(...)`,
//! `truncation.topn(...)`, `truncation.approx_topn(...)` return a
//! `PyTruncation`. The `&` / `|` operators
//! compose them via `And` / `Or`. The composed spec is materialized at the
//! correct width inside `PauliSum.propagate` via the `SpecPolicy<'_, W>`
//! adapter, which implements `paulistrings::TruncationPolicy<W>` for any
//! const-generic `W`.

use num_complex::Complex64;
use paulistrings::pauli_sum::PauliSum;
use paulistrings::truncation::{
    And, ApproxTopN, CoefficientThreshold, Or, TopN, TruncationPolicy, WeightCutoff,
};
use pyo3::prelude::*;

#[derive(Clone, Debug, Default)]
pub enum PolicySpec {
    /// Drop terms with `|c| <= eps`.
    Coeff(f64),
    /// Drop terms with Pauli weight > k.
    Weight(u32),
    /// Keep **at most** n terms by magnitude after each layer, never splitting
    /// a group of equal magnitudes. See `paulistrings::truncation::TopN`.
    TopN(usize),
    /// Keep **approximately** n terms after each layer: at most n, and short of
    /// it by at most one octave's population. See
    /// `paulistrings::truncation::ApproxTopN`.
    ApproxTopN(usize),
    And(Box<PolicySpec>, Box<PolicySpec>),
    Or(Box<PolicySpec>, Box<PolicySpec>),
    /// Default — no filtering. Exact-zero coefficients are still dropped by
    /// the engine's merge phase regardless of policy.
    #[default]
    NoOp,
}

/// Borrow-only adapter implementing `TruncationPolicy<W>` for a `PolicySpec`.
///
/// The `'a` lifetime keeps the spec live for the duration of `propagate`,
/// avoiding the clone-per-layer that would otherwise be needed.
pub struct SpecPolicy<'a, const W: usize>(pub &'a PolicySpec);

impl<'a, const W: usize> TruncationPolicy<W> for SpecPolicy<'a, W> {
    #[inline]
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
        keep_spec::<W>(self.0, x, z, c)
    }

    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        finalize_spec::<W>(self.0, sum);
    }
}

/// Each arm delegates to the matching `paulistrings::truncation` builtin
/// rather than reimplementing its predicate — the delegation itself is free
/// (the newtype construction inlines away), and it keeps this file from
/// drifting out of sync with the core's actual semantics. See
/// `spec_keep_matches_core_builtins` for the cross-check.
#[inline]
fn keep_spec<const W: usize>(spec: &PolicySpec, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
    match spec {
        PolicySpec::Coeff(eps) => CoefficientThreshold(*eps).keep_term(x, z, c),
        PolicySpec::Weight(k) => WeightCutoff(*k).keep_term(x, z, c),
        // Both TopN flavours run in finalize_layer, not per-term.
        PolicySpec::TopN(_) | PolicySpec::ApproxTopN(_) => true,
        PolicySpec::And(a, b) => And(SpecPolicy::<W>(a), SpecPolicy::<W>(b)).keep_term(x, z, c),
        PolicySpec::Or(a, b) => Or(SpecPolicy::<W>(a), SpecPolicy::<W>(b)).keep_term(x, z, c),
        PolicySpec::NoOp => true,
    }
}

fn finalize_spec<const W: usize>(spec: &PolicySpec, sum: &mut PauliSum<W>) {
    match spec {
        PolicySpec::TopN(n) => TopN(*n).finalize_layer(sum),
        PolicySpec::ApproxTopN(n) => ApproxTopN(*n).finalize_layer(sum),
        // `And::finalize_layer` runs both sides' `finalize_layer` in order,
        // which recurses back into `finalize_spec` through `SpecPolicy`.
        PolicySpec::And(a, b) => And(SpecPolicy::<W>(a), SpecPolicy::<W>(b)).finalize_layer(sum),
        // Or has no finalize behavior in the core (matches builtin::Or).
        _ => {}
    }
}

/// Opaque truncation-policy handle exposed to Python.
#[pyclass(module = "paulistrings._paulistrings", name = "Truncation")]
#[derive(Clone)]
pub struct PyTruncation {
    pub(crate) spec: PolicySpec,
}

impl PyTruncation {
    pub fn new(spec: PolicySpec) -> Self {
        Self { spec }
    }
}

#[pymethods]
impl PyTruncation {
    fn __and__(&self, other: &PyTruncation) -> PyTruncation {
        PyTruncation::new(PolicySpec::And(
            Box::new(self.spec.clone()),
            Box::new(other.spec.clone()),
        ))
    }

    fn __or__(&self, other: &PyTruncation) -> PyTruncation {
        PyTruncation::new(PolicySpec::Or(
            Box::new(self.spec.clone()),
            Box::new(other.spec.clone()),
        ))
    }

    fn __repr__(&self) -> String {
        format!("Truncation({:?})", self.spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_W: usize = 1;

    /// Keys spanning weight 0..3 on a single `u64` word, for the grid below.
    fn keys() -> Vec<([u64; TEST_W], [u64; TEST_W])> {
        vec![
            ([0], [0]),         // I: weight 0
            ([1], [0]),         // X on q0: weight 1
            ([0b01], [0b10]),   // X on q0, Z on q1: weight 2
            ([0b011], [0b110]), // weight 3
        ]
    }

    fn coeffs() -> Vec<Complex64> {
        [0.0, 0.05, 0.1, 0.5, 1.0, 2.0]
            .into_iter()
            .map(|r| Complex64::new(r, 0.0))
            .collect()
    }

    /// `keep_spec` (the per-term predicate `SpecPolicy::keep_term` calls into)
    /// must agree with the corresponding `paulistrings::truncation` builtin
    /// on every (spec, key, coefficient) combination — this is the safety net
    /// for the delegation above, since no bench exercises the Python
    /// truncation path.
    #[test]
    fn spec_keep_matches_core_builtins() {
        for eps in [0.0, 0.1, 0.5, 1.0] {
            let spec = PolicySpec::Coeff(eps);
            let core = CoefficientThreshold(eps);
            for &(x, z) in &keys() {
                for c in coeffs() {
                    assert_eq!(
                        keep_spec::<TEST_W>(&spec, &x, &z, c),
                        <CoefficientThreshold as TruncationPolicy<TEST_W>>::keep_term(
                            &core, &x, &z, c
                        ),
                        "Coeff eps={eps} x={x:?} z={z:?} c={c}",
                    );
                }
            }
        }

        for k in [0u32, 1, 2, 3] {
            let spec = PolicySpec::Weight(k);
            let core = WeightCutoff(k);
            for &(x, z) in &keys() {
                for c in coeffs() {
                    assert_eq!(
                        keep_spec::<TEST_W>(&spec, &x, &z, c),
                        <WeightCutoff as TruncationPolicy<TEST_W>>::keep_term(&core, &x, &z, c),
                        "Weight k={k} x={x:?} z={z:?} c={c}",
                    );
                }
            }
        }

        // And / Or: pair a coeff threshold with a weight cutoff and check the
        // composed spec against the core combinators applied to the same pair.
        let and_spec = PolicySpec::And(
            Box::new(PolicySpec::Coeff(0.5)),
            Box::new(PolicySpec::Weight(1)),
        );
        let and_core = And(CoefficientThreshold(0.5), WeightCutoff(1));
        let or_spec = PolicySpec::Or(
            Box::new(PolicySpec::Coeff(0.5)),
            Box::new(PolicySpec::Weight(1)),
        );
        let or_core = Or(CoefficientThreshold(0.5), WeightCutoff(1));
        for &(x, z) in &keys() {
            for c in coeffs() {
                assert_eq!(
                    keep_spec::<TEST_W>(&and_spec, &x, &z, c),
                    <And<_, _> as TruncationPolicy<TEST_W>>::keep_term(&and_core, &x, &z, c),
                    "And x={x:?} z={z:?} c={c}",
                );
                assert_eq!(
                    keep_spec::<TEST_W>(&or_spec, &x, &z, c),
                    <Or<_, _> as TruncationPolicy<TEST_W>>::keep_term(&or_core, &x, &z, c),
                    "Or x={x:?} z={z:?} c={c}",
                );
            }
        }
    }
}
