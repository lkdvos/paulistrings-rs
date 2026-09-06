//! Baseline 1 — the naive algorithm: one hash map for the whole sum.
//!
//! This is the engine's own direct-apply path (`engine/direct.rs`: a hashbrown
//! `HashMap<PauliString, Complex64, FxBuildHasher>`, one `Channel::apply` per
//! resident term, `keep_term` on summed coefficients) run with its size
//! threshold removed, so the entire propagation stays on the map. It is
//! production code with its own differential tests, not a strawman.

use paulistrings::truncation::CoefficientThreshold;
use paulistrings::{
    propagate_with_scratch_and_options, Circuit, Direction, EngineSelection, LayerScratch,
    PauliSum, PropagateOptions,
};
use std::time::Instant;

use crate::common::RunResult;
use crate::workload::W;

pub fn run(circuit: &Circuit<W>, observable: &PauliSum<W>, eps: f64) -> RunResult {
    let policy = CoefficientThreshold(eps);
    let options = PropagateOptions {
        engine: EngineSelection::SmallSumDirect,
        small_sum_threshold: usize::MAX,
        ..PropagateOptions::default()
    };
    let mut scratch = LayerScratch::<W>::new();
    scratch.enable_term_trace();
    let t0 = Instant::now();
    let sum = propagate_with_scratch_and_options(
        circuit,
        observable.clone(),
        &policy,
        Direction::Heisenberg,
        &mut scratch,
        options,
    );
    let wall_ns = t0.elapsed().as_nanos() as u64;
    let terms_out = scratch.take_term_trace().map(|t| t.terms_out).unwrap_or_default();
    let _ = scratch.take_stats();
    let buckets = Some(sum.num_buckets());
    RunResult { sum, wall_ns, terms_out, layer_wall_ns: None, phase: None, buckets }
}
