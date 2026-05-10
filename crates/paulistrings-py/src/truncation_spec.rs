//! `PyTruncation` — opaque, width-erased truncation policy handle.
//!
//! Free factories `truncation.coeff(...)`, `truncation.weight(...)`,
//! `truncation.topn(...)` return a `PyTruncation`. The `&` / `|` operators
//! compose them via `And` / `Or`. The composed spec is materialized at the
//! correct width inside `PauliSum.propagate` via the `SpecPolicy<'_, W>`
//! adapter, which implements `paulistrings::TruncationPolicy<W>` for any
//! const-generic `W`.

use num_complex::Complex64;
use paulistrings::pauli_sum::PauliSum;
use paulistrings::truncation::{TopN, TruncationPolicy};
use pyo3::prelude::*;

#[derive(Clone, Debug, Default)]
pub enum PolicySpec {
    /// Drop terms with `|c| <= eps`.
    Coeff(f64),
    /// Drop terms with Pauli weight > k.
    Weight(u32),
    /// Keep only the n largest-magnitude terms after each layer.
    TopN(usize),
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

#[inline]
fn keep_spec<const W: usize>(
    spec: &PolicySpec,
    x: &[u64; W],
    z: &[u64; W],
    c: Complex64,
) -> bool {
    match spec {
        PolicySpec::Coeff(eps) => c.norm() > *eps,
        PolicySpec::Weight(k) => {
            let weight: u32 = (0..W).map(|i| (x[i] | z[i]).count_ones()).sum();
            weight <= *k
        }
        // TopN runs in finalize_layer, not per-term.
        PolicySpec::TopN(_) => true,
        PolicySpec::And(a, b) => keep_spec::<W>(a, x, z, c) && keep_spec::<W>(b, x, z, c),
        PolicySpec::Or(a, b) => keep_spec::<W>(a, x, z, c) || keep_spec::<W>(b, x, z, c),
        PolicySpec::NoOp => true,
    }
}

fn finalize_spec<const W: usize>(spec: &PolicySpec, sum: &mut PauliSum<W>) {
    match spec {
        PolicySpec::TopN(n) => TopN(*n).finalize_layer(sum),
        PolicySpec::And(a, b) => {
            finalize_spec::<W>(a, sum);
            finalize_spec::<W>(b, sum);
        }
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
