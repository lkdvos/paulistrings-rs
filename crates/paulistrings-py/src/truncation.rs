//! `paulistrings._paulistrings.truncation` submodule: policy factories. See
//! ARCHITECTURE.md §Truncation and ARCHITECTURE.md §Python-Bindings.
//!
//! Python composition is via the `&` and `|` operators on the returned objects.

use crate::truncation_spec::{PolicySpec, PyTruncation};
use pyo3::prelude::*;

#[pyfunction]
fn coeff(epsilon: f64) -> PyTruncation {
    PyTruncation::new(PolicySpec::Coeff(epsilon))
}

#[pyfunction]
fn weight(k: u32) -> PyTruncation {
    PyTruncation::new(PolicySpec::Weight(k))
}

/// Keep at most ``n`` terms by coefficient magnitude after each layer.
///
/// Terms are never split across a group of exactly equal magnitudes: let ``t``
/// be the n-th largest magnitude; everything above ``t`` is kept, and the tie
/// group at ``t`` is kept only if it fits within ``n`` in full, otherwise it is
/// dropped whole. So the result has at most ``n`` terms, and exactly ``n``
/// whenever the cut lands on a group boundary (in particular when all
/// magnitudes are distinct).
///
/// Equal magnitudes come from symmetry-related terms, and keeping an arbitrary
/// subset of such a multiplet breaks the symmetry of the truncated operator.
///
/// Note the degenerate case: if *every* candidate ties at the threshold, this
/// keeps nothing and the sum becomes empty. Combine with ``coeff`` via ``&``,
/// or raise ``n`` above the expected multiplet size, if that matters.
#[pyfunction]
fn topn(n: usize) -> PyTruncation {
    PyTruncation::new(PolicySpec::TopN(n))
}

/// Keep **approximately** ``n`` terms after each layer — the cheap sibling of
/// ``topn``, opt-in and never a default.
///
/// Terms are binned by the octave (binade) of ``|c|**2`` — a factor of 2 in
/// ``|c|**2``, i.e. ``sqrt(2)`` in ``|c|`` — and whole octaves are kept from the
/// top down while they still fit in ``n``. Writing ``S_k`` for the number of
/// terms in bin ``k`` and above, the kept set is ``S_k*`` for the lowest ``k*``
/// with ``S_k* <= n``. So:
///
/// 1. **At most ``n`` is kept, always** — the bound ``topn`` exists to provide
///    holds exactly, which is why the rounding goes this way.
/// 2. **The shortfall is bounded** by the population of the coarsest excluded
///    octave: more than ``n - p`` is kept, where ``p`` is the number of terms in
///    the next octave down (including it would have overshot ``n``).
/// 3. Every kept magnitude is at least every dropped one, and the kept set is a
///    union of whole octaves — so a **tie group is always kept whole**, with no
///    tie rule needed: equal magnitudes have equal squares and therefore share
///    an octave. (``topn`` keeps a tie group whole only if it fits, and drops it
///    whole otherwise.)
///
/// The shortfall is a small fraction of ``n`` on a magnitude distribution spread
/// over many octaves, and measured at 0.23–1.23% on the suite's circuits. It is
/// not small on a tightly clustered one — see the degenerate case below.
///
/// Like ``topn``, the retained set is a pure function of the magnitude multiset:
/// independent of the bucket partition, the hash seed and the thread count.
/// Reach for this when ``n`` is a *memory budget* and a few percent of slack in
/// the term count is cheaper than the exact selection; keep ``topn`` when the
/// retained count itself matters.
///
/// Note the degenerate case, the wider sibling of ``topn``'s: **if every
/// magnitude lands in a single octave and there are more than ``n`` terms, this
/// keeps nothing and the sum becomes empty.** ``S_k`` can then only be ``0`` or
/// ``len``, and ``len > n``. Keeping the octave anyway would let the policy
/// retain unboundedly more than ``n``, destroying the bound. Combine with
/// ``coeff`` via ``&``, or use ``topn``, if that matters.
#[pyfunction]
fn approx_topn(n: usize) -> PyTruncation {
    PyTruncation::new(PolicySpec::ApproxTopN(n))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coeff, m)?)?;
    m.add_function(wrap_pyfunction!(weight, m)?)?;
    m.add_function(wrap_pyfunction!(topn, m)?)?;
    m.add_function(wrap_pyfunction!(approx_topn, m)?)?;
    Ok(())
}
