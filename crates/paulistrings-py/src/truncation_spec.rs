//! `PyTruncation` — opaque, width-erased truncation policy handle.
//! Free factories `truncation.coeff/weight/topn/approx_topn(...)` return one; `&`/`|` compose via `And`/`Or`.
//! `SpecPolicy<'_, W>` materializes the spec at the correct width inside `PauliSum.propagate`.

use num_complex::Complex64;
use paulistrings::engine::partitioned::Collectives;
use paulistrings::pauli_sum::PauliSum;
use paulistrings::truncation::{
    And, ApproxTopN, CoefficientThreshold, Or, TopN, TruncationPolicy, WeightCutoff,
};
use paulistrings::PartitionedTruncation;
use pyo3::prelude::*;

#[derive(Clone, Debug, Default)]
pub enum PolicySpec {
    /// Drop terms with `|c| <= eps`.
    Coeff(f64),
    /// Drop terms with Pauli weight > k.
    Weight(u32),
    /// Keep at most n terms by magnitude, never splitting a tie group. See `paulistrings::truncation::TopN`.
    TopN(usize),
    /// Keep approximately n terms: at most n, short by at most one octave's population. See `paulistrings::truncation::ApproxTopN`.
    ApproxTopN(usize),
    And(Box<PolicySpec>, Box<PolicySpec>),
    Or(Box<PolicySpec>, Box<PolicySpec>),
    /// Default — no filtering; exact-zero coefficients are still dropped by the engine's merge.
    #[default]
    NoOp,
}

/// Borrow-only adapter implementing `TruncationPolicy<W>` for a `PolicySpec`. The `'a` lifetime keeps the spec live for `propagate`'s duration, avoiding a clone per layer.
pub struct SpecPolicy<'a, const W: usize>(pub &'a PolicySpec);

impl<'a, const W: usize> TruncationPolicy<W> for SpecPolicy<'a, W> {
    #[inline]
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
        keep_spec::<W>(self.0, x, z, c)
    }

    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        finalize_spec::<W>(self.0, sum);
    }

    /// The spec tree's own answer, rather than the trait's conservative `true` — without this override every Python policy would claim a layer pass and `EngineSelection::Auto` would never take the small-sum direct path (research/FINDINGS.md).
    /// Mirrors [`finalize_spec`]'s recursion exactly; both matches are exhaustive so a new `PolicySpec` variant cannot be added to one without the other.
    fn finalizes_layer(&self) -> bool {
        finalizes_spec(self.0)
    }
}

/// The collective form of [`finalize_layer`](TruncationPolicy::finalize_layer), run on every partition on every layer in lock-step.
/// Without this impl the trait's default body would fire its assertion on the first `topn`/`approx_topn` layer. The one spec with no collective form, exact [`TopN`], is rejected at the Python boundary first (see [`spec_has_exact_topn`]).
impl<'a, const W: usize> PartitionedTruncation<W> for SpecPolicy<'a, W> {
    fn finalize_layer_partitioned(&self, sum: &mut PauliSum<W>, coll: &dyn Collectives) {
        finalize_spec_partitioned::<W>(self.0, sum, coll);
    }
}

/// Each arm delegates to the matching `paulistrings::truncation` builtin rather than reimplementing its predicate, keeping this in sync with the core (cross-checked by `spec_keep_matches_core_builtins`).
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
        // `And::finalize_layer` runs both sides in order, recursing back into `finalize_spec` through `SpecPolicy`.
        PolicySpec::And(a, b) => And(SpecPolicy::<W>(a), SpecPolicy::<W>(b)).finalize_layer(sum),
        // Or has no finalize behavior (matches builtin::Or); Coeff/Weight/NoOp filter per term. Written out rather than `_` so a new variant has to answer here.
        PolicySpec::Coeff(_) | PolicySpec::Weight(_) | PolicySpec::Or(_, _) | PolicySpec::NoOp => {}
    }
}

/// Why a spec containing an exact `topn` cannot run partitioned, worded for the Python `NotImplementedError` and reused verbatim by [`finalize_spec_partitioned`]'s unreachable arm.
pub(crate) const TOPN_PARTITIONED_MSG: &str =
    "exact truncation.topn is not supported in partitioned mode; use \
     truncation.approx_topn or partitions=None";

/// The collective twin of [`finalize_spec`], arm for arm, delegating to the matching core `PartitionedTruncation` impl so the two layer passes cannot drift apart (`spec_finalize_partitioned_matches_core_builtins` cross-checks).
fn finalize_spec_partitioned<const W: usize>(
    spec: &PolicySpec,
    sum: &mut PauliSum<W>,
    coll: &dyn Collectives,
) {
    match spec {
        PolicySpec::ApproxTopN(n) => {
            <ApproxTopN as PartitionedTruncation<W>>::finalize_layer_partitioned(
                &ApproxTopN(*n),
                sum,
                coll,
            )
        }
        // `And` runs both sides in order on every partition, keeping the collectives in lock-step, same shape as `finalize_spec`.
        PolicySpec::And(a, b) => {
            <And<SpecPolicy<'_, W>, SpecPolicy<'_, W>> as PartitionedTruncation<W>>::
                finalize_layer_partitioned(
                    &And(SpecPolicy::<W>(a), SpecPolicy::<W>(b)),
                    sum,
                    coll,
                )
        }
        // Unreachable: `PauliSum.propagate` rejects an exact `topn` before the run starts (`spec_has_exact_topn`). Kept as a safety net so a future caller that skips the gate fails loudly instead of truncating per partition.
        PolicySpec::TopN(_) => panic!("{TOPN_PARTITIONED_MSG}"),
        // Per-term filters have no layer pass, and `Or` forwards to neither side. Written out rather than `_` so a new variant has to answer here too.
        PolicySpec::Coeff(_) | PolicySpec::Weight(_) | PolicySpec::Or(_, _) | PolicySpec::NoOp => {}
    }
}

/// Whether the spec tree mentions an exact [`TopN`] anywhere — the gate `PauliSum.propagate` checks before entering partitioned mode, since exact top-n has no collective form (see `PartitionedTruncation`'s docs).
/// `Or` votes too, even though its layer pass is a no-op: the contract is "an exact `topn` anywhere means no partitioned run", not "only where it would run".
pub(crate) fn spec_has_exact_topn(spec: &PolicySpec) -> bool {
    match spec {
        PolicySpec::TopN(_) => true,
        PolicySpec::And(a, b) | PolicySpec::Or(a, b) => {
            spec_has_exact_topn(a) || spec_has_exact_topn(b)
        }
        PolicySpec::ApproxTopN(_)
        | PolicySpec::Coeff(_)
        | PolicySpec::Weight(_)
        | PolicySpec::NoOp => false,
    }
}

/// Whether [`finalize_spec`] would do anything for this spec — reported to `TruncationPolicy::finalizes_layer`. One arm per [`finalize_spec`] arm, in the same order (`spec_finalizes_matches_core_builtins` cross-checks against the core builtins).
fn finalizes_spec(spec: &PolicySpec) -> bool {
    match spec {
        PolicySpec::TopN(_) | PolicySpec::ApproxTopN(_) => true,
        PolicySpec::And(a, b) => finalizes_spec(a) || finalizes_spec(b),
        PolicySpec::Coeff(_) | PolicySpec::Weight(_) | PolicySpec::Or(_, _) | PolicySpec::NoOp => {
            false
        }
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
    use paulistrings::accumulator::BuildAccumulator;
    use paulistrings::engine::partitioned::InProcessTransport;
    use paulistrings::pauli_string::PauliString;
    use paulistrings::phase::Phase;
    use std::panic::resume_unwind;

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

    /// `keep_spec` must agree with the corresponding core builtin on every (spec, key, coefficient) combination — the safety net for the delegation above.
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

    /// `SpecPolicy::finalizes_layer` must report the spec tree's own answer, not the trait's conservative `true` — this is the hint `EngineSelection::Auto` reads to decide whether the small-sum direct path is worth taking.
    /// Each expectation is stated against the core builtin the matching `finalize_spec` arm delegates to, so the two cannot drift apart.
    #[test]
    fn spec_finalizes_matches_core_builtins() {
        let finalizes =
            |spec: &PolicySpec| {
                <SpecPolicy<'_, TEST_W> as TruncationPolicy<TEST_W>>::finalizes_layer(
                    &SpecPolicy::<TEST_W>(spec),
                )
            };

        // Per-term filters: nothing to finalize, and the core builtins agree.
        assert!(
            !<CoefficientThreshold as TruncationPolicy<TEST_W>>::finalizes_layer(
                &CoefficientThreshold(0.5)
            )
        );
        assert!(!finalizes(&PolicySpec::Coeff(0.5)));
        assert!(!<WeightCutoff as TruncationPolicy<TEST_W>>::finalizes_layer(&WeightCutoff(2)));
        assert!(!finalizes(&PolicySpec::Weight(2)));

        // No policy at all is the case that matters most: `propagate(policy=None)`
        // builds a `NoOp` spec, and it must not claim a layer pass.
        assert!(!finalizes(&PolicySpec::NoOp));

        // Both TopN flavours have a real layer pass.
        assert!(<TopN as TruncationPolicy<TEST_W>>::finalizes_layer(&TopN(
            4
        )));
        assert!(finalizes(&PolicySpec::TopN(4)));
        assert!(<ApproxTopN as TruncationPolicy<TEST_W>>::finalizes_layer(
            &ApproxTopN(4)
        ));
        assert!(finalizes(&PolicySpec::ApproxTopN(4)));

        // And is the disjunction of its sides, at either position and nested.
        let cheap = PolicySpec::And(
            Box::new(PolicySpec::Coeff(0.5)),
            Box::new(PolicySpec::Weight(2)),
        );
        assert!(!finalizes(&cheap));
        assert!(!<And<_, _> as TruncationPolicy<TEST_W>>::finalizes_layer(
            &And(CoefficientThreshold(0.5), WeightCutoff(2))
        ));
        assert!(finalizes(&PolicySpec::And(
            Box::new(PolicySpec::Coeff(0.5)),
            Box::new(PolicySpec::TopN(4)),
        )));
        assert!(finalizes(&PolicySpec::And(
            Box::new(PolicySpec::ApproxTopN(4)),
            Box::new(PolicySpec::Weight(2)),
        )));
        assert!(finalizes(&PolicySpec::And(
            Box::new(cheap.clone()),
            Box::new(PolicySpec::And(
                Box::new(PolicySpec::NoOp),
                Box::new(PolicySpec::TopN(4)),
            )),
        )));

        // Or never finalizes, whatever its children are: `finalize_spec` leaves
        // it to the trait's no-op default, matching `builtin::Or`.
        let ored = PolicySpec::Or(
            Box::new(PolicySpec::TopN(4)),
            Box::new(PolicySpec::ApproxTopN(4)),
        );
        assert!(!finalizes(&ored));
        assert!(!<Or<_, _> as TruncationPolicy<TEST_W>>::finalizes_layer(
            &Or(TopN(4), TopN(4))
        ));
        // ... including inside an `And`, where only the non-`Or` side can vote.
        assert!(!finalizes(&PolicySpec::And(
            Box::new(ored),
            Box::new(PolicySpec::Coeff(0.5)),
        )));
    }

    // ---------------------------------------------------------------------
    // The collective layer pass

    /// A deterministic sum whose magnitudes span ~9 octaves, so an `approx_topn` cut lands strictly inside the histogram.
    /// Only every `stride`-th term starting at `offset` is taken; the partitioned fixtures below use this to build disjoint shares (any split will do).
    fn strided_sum(count: usize, offset: usize, stride: usize) -> PauliSum<TEST_W> {
        let mut acc = BuildAccumulator::<TEST_W>::with_capacity(32, count / stride + 1);
        for i in (offset..count).step_by(stride) {
            let key = PauliString {
                x: [i as u64],
                z: [(i as u64) << 7],
            };
            let magnitude = 2f64.powi(-((i % 9) as i32)) * (1.0 + i as f64 * 1e-3);
            acc.add_term(key, Phase::ONE, Complex64::new(magnitude, 0.0));
        }
        acc.finalize()
    }

    /// `{(x, z): coefficient}` over one or more sums, so a comparison does not
    /// depend on bucket order or on which partition held a term.
    fn terms_of(sums: &[PauliSum<TEST_W>]) -> Vec<([u64; TEST_W], [u64; TEST_W], Complex64)> {
        let mut out: Vec<_> = sums
            .iter()
            .flat_map(|s| s.iter().map(|(x, z, c)| (*x, *z, c)))
            .collect();
        out.sort_by_key(|(x, z, _)| (x[0], z[0]));
        out
    }

    /// Run `spec`'s collective layer pass on `parts`, one thread per part on an in-process transport group, since the blocking collectives need every rank to make progress independently.
    fn partitioned_finalize(
        spec: &PolicySpec,
        parts: Vec<PauliSum<TEST_W>>,
    ) -> Vec<PauliSum<TEST_W>> {
        let transports = InProcessTransport::group(parts.len() as u32);
        std::thread::scope(|scope| {
            let handles: Vec<_> = parts
                .into_iter()
                .zip(transports)
                .map(|(mut part, transport)| {
                    scope.spawn(move || {
                        <SpecPolicy<'_, TEST_W> as PartitionedTruncation<TEST_W>>::
                            finalize_layer_partitioned(
                                &SpecPolicy::<TEST_W>(spec),
                                &mut part,
                                &transport,
                            );
                        part
                    })
                })
                .collect();
            handles
                .into_iter()
                // Re-raise a partition's panic with its own payload, so a
                // `#[should_panic]` sees the message and not `Any { .. }`.
                .map(|h| h.join().unwrap_or_else(|payload| resume_unwind(payload)))
                .collect()
        })
    }

    /// The union of the per-partition results must be exactly what `SpecPolicy::finalize_layer` keeps on the whole sum. Bitwise equality is the right bar: finalization only retains terms, no arithmetic or summation order involved.
    #[test]
    fn spec_finalize_partitioned_matches_core_builtins() {
        const COUNT: usize = 400;
        let cheap = PolicySpec::And(
            Box::new(PolicySpec::Coeff(0.01)),
            Box::new(PolicySpec::Weight(64)),
        );
        let specs = vec![
            ("noop", PolicySpec::NoOp),
            ("coeff", PolicySpec::Coeff(0.5)),
            ("cheap and", cheap),
            ("approx_topn(0)", PolicySpec::ApproxTopN(0)),
            ("approx_topn(37)", PolicySpec::ApproxTopN(37)),
            ("approx_topn(200)", PolicySpec::ApproxTopN(200)),
            ("approx_topn(10_000)", PolicySpec::ApproxTopN(10_000)),
            (
                "coeff & approx_topn",
                PolicySpec::And(
                    Box::new(PolicySpec::Coeff(0.05)),
                    Box::new(PolicySpec::ApproxTopN(120)),
                ),
            ),
            (
                "approx_topn & approx_topn",
                PolicySpec::And(
                    Box::new(PolicySpec::ApproxTopN(300)),
                    Box::new(PolicySpec::ApproxTopN(80)),
                ),
            ),
            (
                // `Or` forwards to neither side, in both modes.
                "approx_topn | coeff",
                PolicySpec::Or(
                    Box::new(PolicySpec::ApproxTopN(5)),
                    Box::new(PolicySpec::Coeff(0.5)),
                ),
            ),
        ];

        for (name, spec) in &specs {
            for partitions in [1usize, 2, 4] {
                let parts: Vec<_> = (0..partitions)
                    .map(|rank| strided_sum(COUNT, rank, partitions))
                    .collect();
                let mut whole = strided_sum(COUNT, 0, 1);
                assert_eq!(
                    parts.iter().map(PauliSum::len).sum::<usize>(),
                    whole.len(),
                    "{name}: the fixture split must be disjoint and complete",
                );

                let got = partitioned_finalize(spec, parts);
                <SpecPolicy<'_, TEST_W> as TruncationPolicy<TEST_W>>::finalize_layer(
                    &SpecPolicy::<TEST_W>(spec),
                    &mut whole,
                );
                assert_eq!(
                    terms_of(&got),
                    terms_of(std::slice::from_ref(&whole)),
                    "{name}: P={partitions} partitioned result differs from the \
                     single-partition one",
                );
            }
        }
    }

    /// The unreachable arm is a safety net, not dead code: reaching it panics
    /// with the message the Python gate raises.
    #[test]
    #[should_panic(expected = "exact truncation.topn is not supported in partitioned mode")]
    fn finalize_partitioned_panics_on_an_exact_topn() {
        let spec = PolicySpec::And(
            Box::new(PolicySpec::Coeff(0.5)),
            Box::new(PolicySpec::TopN(4)),
        );
        partitioned_finalize(&spec, vec![strided_sum(16, 0, 1)]);
    }

    /// The gate `PauliSum.propagate` consults before entering partitioned
    /// mode: exact `topn` anywhere in the tree, `Or` included.
    #[test]
    fn spec_has_exact_topn_finds_it_anywhere() {
        assert!(spec_has_exact_topn(&PolicySpec::TopN(4)));
        assert!(!spec_has_exact_topn(&PolicySpec::ApproxTopN(4)));
        assert!(!spec_has_exact_topn(&PolicySpec::NoOp));
        assert!(!spec_has_exact_topn(&PolicySpec::Coeff(0.5)));
        assert!(!spec_has_exact_topn(&PolicySpec::Weight(2)));

        // Either side of an `And`, nested arbitrarily deep.
        assert!(spec_has_exact_topn(&PolicySpec::And(
            Box::new(PolicySpec::Coeff(0.5)),
            Box::new(PolicySpec::TopN(4)),
        )));
        assert!(spec_has_exact_topn(&PolicySpec::And(
            Box::new(PolicySpec::And(
                Box::new(PolicySpec::TopN(4)),
                Box::new(PolicySpec::NoOp),
            )),
            Box::new(PolicySpec::ApproxTopN(9)),
        )));
        assert!(!spec_has_exact_topn(&PolicySpec::And(
            Box::new(PolicySpec::ApproxTopN(4)),
            Box::new(PolicySpec::Weight(2)),
        )));

        // `Or` votes as well, unlike in `finalizes_spec`: the contract is
        // "no exact topn anywhere", not "no exact topn that would run".
        assert!(spec_has_exact_topn(&PolicySpec::Or(
            Box::new(PolicySpec::TopN(4)),
            Box::new(PolicySpec::Coeff(0.5)),
        )));
    }
}
