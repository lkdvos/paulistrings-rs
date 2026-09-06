//! The bucketed engine (`paulistrings::propagate`), with the talk's bucket-size
//! lever (`PropagateOptions { target_bucket_len, min_buckets }`).

use paulistrings::truncation::CoefficientThreshold;
use paulistrings::{
    propagate_with_scratch_and_options, Circuit, Direction, EngineSelection, LayerScratch,
    PauliSum, PropagateOptions,
};
use std::time::Instant;

use crate::common::RunResult;
use crate::workload::W;

pub fn options(target_bucket_len: usize, min_buckets: usize) -> PropagateOptions {
    PropagateOptions {
        engine: EngineSelection::SortedOnly,
        target_bucket_len,
        min_buckets,
        ..PropagateOptions::default()
    }
}

/// Whole-circuit propagation. `layers`, when given, is the same circuit as one
/// single-channel `Circuit` per layer (`workload::talk_layers`), driven through
/// one shared scratch with a clock per layer; the scratch retains its capacity
/// so this is the same computation, only with per-layer wall times.
pub fn run(
    circuit: &Circuit<W>,
    layers: Option<&[Circuit<W>]>,
    observable: &PauliSum<W>,
    eps: f64,
    options: PropagateOptions,
) -> RunResult {
    let policy = CoefficientThreshold(eps);
    let mut scratch = LayerScratch::<W>::new();
    scratch.enable_term_trace();
    if let Some(layers) = layers {
        let mut sum = observable.clone();
        let mut per_layer = Vec::with_capacity(layers.len());
        let mut terms_out = Vec::with_capacity(layers.len());
        let t0 = Instant::now();
        // Heisenberg: the last channel acts first.
        for one in layers.iter().rev() {
            let t = Instant::now();
            sum = propagate_with_scratch_and_options(
                one,
                sum,
                &policy,
                Direction::Heisenberg,
                &mut scratch,
                options,
            );
            per_layer.push(t.elapsed().as_nanos() as u64);
            terms_out.push(sum.len());
        }
        let wall_ns = t0.elapsed().as_nanos() as u64;
        let _ = scratch.take_term_trace();
        let phase = Some(scratch.take_stats());
        let buckets = Some(sum.num_buckets());
        return RunResult { sum, wall_ns, terms_out, layer_wall_ns: Some(per_layer), phase, buckets };
    }
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
    let phase = Some(scratch.take_stats());
    let buckets = Some(sum.num_buckets());
    RunResult { sum, wall_ns, terms_out, layer_wall_ns: None, phase, buckets }
}
