//! The opt-in per-layer term-count trace (`LayerScratch::enable_term_trace`).
//!
//! Cross-module: the trace lives on `engine::bucketed::LayerScratch` but is
//! written by `engine::propagate_with_scratch`'s per-layer epilogue, and its
//! counts are a property of the layer loop (fanout, then the policy's
//! per-term filter). Design: `research/notes/2026-09-01-python-api-extensions.md`
//! §A2. This trace is always compiled — unlike `PhaseStats`, which stays
//! behind the `phase-timing` feature.

use num_complex::Complex64;
use paulistrings::channel::{Clifford1Q, PauliRotation};
use paulistrings::test_support::assert_terms_close;
use paulistrings::truncation::CoefficientThreshold;
use paulistrings::{
    propagate, propagate_with_scratch, BuildAccumulator, Circuit, Direction, LayerScratch,
    PauliString, PauliSum, Phase, TruncationPolicy,
};

struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

const NUM_QUBITS: usize = 4;
/// cos 0.3 = 0.955336…, sin 0.3 = 0.295520… — far enough from each other and
/// from zero that no count below depends on a cancellation.
const THETA: f64 = 0.3;

/// `{X₀: 1}`.
fn x0_observable<const W: usize>() -> PauliSum<W> {
    let mut acc = BuildAccumulator::<W>::with_capacity(NUM_QUBITS, 1);
    acc.add_term(PauliString::<W>::x(0), Phase::ONE, Complex64::new(1.0, 0.0));
    acc.finalize()
}

/// Three layers whose forward term counts on `{X₀: 1}` are hand-computable:
///
/// 1. `exp(-iθZ₀/2)`: `X₀` anticommutes with the generator, so it fans out —
///    `cos θ·X₀ − sin θ·Y₀` (sign per the rotation's `i^k` fold; the counts
///    do not depend on it). 1 → 2 terms.
/// 2. `H₀`: `X₀ → Z₀`, `Y₀ → −Y₀`. A relabelling, 2 → 2 terms.
/// 3. `exp(-iθZ₀/2)` again: `Z₀` commutes with the generator (stays one
///    term), `Y₀` anticommutes (fans out to `Y₀` and `X₀`). The three keys
///    `Z₀`, `Y₀`, `X₀` are distinct, so 2 → 3 terms.
fn three_layer_circuit<const W: usize>() -> Circuit<W> {
    let mut c = Circuit::<W>::new(NUM_QUBITS);
    c.push(PauliRotation::new(PauliString::<W>::z(0), THETA));
    c.push(Clifford1Q::h(0));
    c.push(PauliRotation::new(PauliString::<W>::z(0), THETA));
    c
}

/// A fresh scratch traces nothing: `take_term_trace` is `None` after a
/// propagation that never enabled it, so `None` ⟺ "tracing off".
fn check_off_by_default<const W: usize>() {
    let mut scratch = LayerScratch::<W>::new();
    let out = propagate_with_scratch(
        &three_layer_circuit::<W>(),
        x0_observable::<W>(),
        &AlwaysKeep,
        Direction::Forward,
        &mut scratch,
    );
    assert_eq!(out.len(), 3);
    assert!(scratch.take_term_trace().is_none());
}

#[test]
fn term_trace_is_off_by_default_w1() {
    check_off_by_default::<1>();
}

#[test]
fn term_trace_is_off_by_default_w2() {
    check_off_by_default::<2>();
}

/// Hand-computed counts for the circuit above, and the "peak resident"
/// definition: `max(terms_in[0], terms_out…)`.
fn check_hand_computed_counts<const W: usize>() {
    let mut scratch = LayerScratch::<W>::new();
    scratch.enable_term_trace();
    let out = propagate_with_scratch(
        &three_layer_circuit::<W>(),
        x0_observable::<W>(),
        &AlwaysKeep,
        Direction::Forward,
        &mut scratch,
    );

    let trace = scratch.take_term_trace().expect("trace was enabled");
    assert_eq!(trace.terms_in, vec![1, 2, 2]);
    assert_eq!(trace.terms_out, vec![2, 2, 3]);
    assert_eq!(trace.peak_terms(), Some(3));
    assert_eq!(out.len(), 3);
}

#[test]
fn term_trace_records_hand_computed_counts_w1() {
    check_hand_computed_counts::<1>();
}

#[test]
fn term_trace_records_hand_computed_counts_w2() {
    check_hand_computed_counts::<2>();
}

/// The counts are read *after* `policy.finalize_layer`, so they are
/// post-truncation. Layer 3's third key is `X₀` with coefficient
/// `sin²θ·cos θ… = 0.0873…` (the two rotations' `sin θ` factors times the
/// intervening `cos θ`), which a threshold of 0.1 drops; the other two keys
/// carry 0.9553… (`Z₀`) and 0.2823… (`Y₀`) and survive. So the final layer
/// reports 2 out, not 3, and the peak drops with it.
fn check_counts_are_post_truncation<const W: usize>() {
    let mut scratch = LayerScratch::<W>::new();
    scratch.enable_term_trace();
    let out = propagate_with_scratch(
        &three_layer_circuit::<W>(),
        x0_observable::<W>(),
        &CoefficientThreshold(0.1),
        Direction::Forward,
        &mut scratch,
    );

    let trace = scratch.take_term_trace().expect("trace was enabled");
    assert_eq!(trace.terms_in, vec![1, 2, 2]);
    assert_eq!(trace.terms_out, vec![2, 2, 2]);
    assert_eq!(trace.peak_terms(), Some(2));
    assert_eq!(out.len(), 2);
}

#[test]
fn term_trace_counts_are_post_truncation_w1() {
    check_counts_are_post_truncation::<1>();
}

#[test]
fn term_trace_counts_are_post_truncation_w2() {
    check_counts_are_post_truncation::<2>();
}

/// A zero-layer circuit records nothing, and "peak" is undefined from the
/// trace alone (the caller knows the resident count — it never changed).
#[test]
fn term_trace_of_an_empty_circuit_is_empty() {
    let mut scratch = LayerScratch::<1>::new();
    scratch.enable_term_trace();
    let out = propagate_with_scratch(
        &Circuit::<1>::new(NUM_QUBITS),
        x0_observable::<1>(),
        &AlwaysKeep,
        Direction::Forward,
        &mut scratch,
    );
    assert_eq!(out.len(), 1);

    let trace = scratch.take_term_trace().expect("trace was enabled");
    assert!(trace.terms_in.is_empty());
    assert!(trace.terms_out.is_empty());
    assert_eq!(trace.peak_terms(), None);
}

/// Taking drains the counts but leaves tracing *on*, so a scratch reused
/// across `propagate_with_scratch` calls (a Trotter driver) reports each
/// call separately without re-enabling.
#[test]
fn taking_the_trace_drains_but_stays_enabled() {
    let circuit = three_layer_circuit::<1>();
    let mut scratch = LayerScratch::<1>::new();
    scratch.enable_term_trace();

    for _ in 0..2 {
        let _ = propagate_with_scratch(
            &circuit,
            x0_observable::<1>(),
            &AlwaysKeep,
            Direction::Forward,
            &mut scratch,
        );
        let trace = scratch.take_term_trace().expect("still enabled");
        assert_eq!(trace.terms_out, vec![2, 2, 3]);
    }
    // Nothing ran in between, so the third take sees an empty — but present —
    // trace.
    let trace = scratch.take_term_trace().expect("still enabled");
    assert!(trace.terms_out.is_empty());
}

/// Tracing is bookkeeping on the calling thread: the propagated sum is the
/// same one `propagate` (which never enables the trace) produces.
fn check_trace_does_not_change_the_result<const W: usize>() {
    let circuit = three_layer_circuit::<W>();
    let want = propagate(
        &circuit,
        x0_observable::<W>(),
        &AlwaysKeep,
        Direction::Forward,
    );

    let mut scratch = LayerScratch::<W>::new();
    scratch.enable_term_trace();
    let got = propagate_with_scratch(
        &circuit,
        x0_observable::<W>(),
        &AlwaysKeep,
        Direction::Forward,
        &mut scratch,
    );
    assert_terms_close(&got, &want, 1e-15, "traced vs untraced propagate");
}

#[test]
fn term_trace_does_not_change_the_result_w1() {
    check_trace_does_not_change_the_result::<1>();
}

#[test]
fn term_trace_does_not_change_the_result_w2() {
    check_trace_does_not_change_the_result::<2>();
}
