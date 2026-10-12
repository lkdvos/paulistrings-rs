//! The propagation front door [`propagate`], one partition of the partitioned layer loop; one layer is applied in `bucketed`.
//!
//! See ARCHITECTURE.md §Engine.

pub(crate) mod bucketed;
mod coset;
#[cfg(feature = "cuda")]
mod cuda_context;
#[cfg(feature = "cuda")]
pub mod gpu;
mod merge;
pub(crate) mod partitioned;
#[cfg(feature = "phase-timing")]
pub(crate) mod stats;
pub(crate) mod timing;

use crate::circuit::Circuit;
use crate::pauli_sum::hash::PartitionRows;
use crate::pauli_sum::storage::{DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;
use bucketed::{LayerScratch, LayerTrace};
use partitioned::backend::HostPartition;
use partitioned::driver::{run_layers, PartitionContext, PartitionWork};
use partitioned::trace::PartitionLayerRow;
use partitioned::transport::SoloTransport;

/// `log` target for the engine's progress events, partitioned or not, and the truncation policies' own.
pub(crate) const LOG_TARGET: &str = "paulistrings::propagate";

/// Propagation direction: channels in order, or in reverse with adjoints for backpropagating observables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Apply channels in the order they were pushed onto the [`Circuit`].
    Forward,
    /// Apply channels in reverse order, using each channel's [`Channel::apply_adjoint`](crate::Channel::apply_adjoint).
    Heisenberg,
}

/// Tuning knobs for [`propagate_with`]; [`Default`] is what [`propagate`] uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PropagateOptions {
    /// Terms per bucket the per-layer partition targets (ARCHITECTURE.md §Bucket-Policy).
    ///
    /// Above the `min_buckets` floor raising this alone does nothing; both fields must move together to get fewer buckets, and rebucketing is grow-only, so lowering either mid-run never coarsens a partition.
    /// At least 1 ([`validate`](Self::validate)).
    pub target_bucket_len: usize,
    /// Floor on the per-layer bucket count once the sum is worth splitting.
    ///
    /// At least 16, below which the "worth splitting" gate goes non-monotone ([`validate`](Self::validate)).
    pub min_buckets: usize,
}

/// The smallest accepted [`PropagateOptions::min_buckets`] (ARCHITECTURE.md §Bucket-Policy).
const SMALLEST_MIN_BUCKETS: usize = 16;

impl PropagateOptions {
    /// `Err` with the message every entry point panics with unless `target_bucket_len >= 1` and `min_buckets >= 16`.
    pub fn validate(&self) -> Result<(), String> {
        if self.target_bucket_len == 0 {
            return Err("target_bucket_len must be at least 1, got 0".to_string());
        }
        if self.min_buckets < SMALLEST_MIN_BUCKETS {
            return Err(format!(
                "min_buckets must be at least {SMALLEST_MIN_BUCKETS}, got {}",
                self.min_buckets
            ));
        }
        Ok(())
    }

    /// Panics with [`validate`](Self::validate)'s message.
    pub(crate) fn assert_valid(&self) {
        if let Err(message) = self.validate() {
            panic!("{message}");
        }
    }
}

impl Default for PropagateOptions {
    fn default() -> Self {
        Self {
            target_bucket_len: DEFAULT_TARGET_BUCKET_LEN,
            min_buckets: DEFAULT_MIN_BUCKETS,
        }
    }
}

/// Propagate `sum` through `circuit` under `policy`.
///
/// Under [`Direction::Heisenberg`] the channels run in reverse through [`Channel::apply_adjoint`](crate::Channel::apply_adjoint).
/// The sum stays bucketed throughout, so stepping one sum through repeated calls costs only the layers.
///
/// # Panics
///
/// If a channel's support exceeds `MAX_LOCAL_SUPPORT` (two qubits), which no built-in channel does; [`PauliRotation`](crate::PauliRotation) is exempt at any support.
///
/// # Progress logging
///
/// Through the [`log`] facade under the target `paulistrings::propagate`: `INFO` on entry and exit, `DEBUG` once per layer, named by [`Channel::debug_name`](crate::Channel::debug_name).
pub fn propagate<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
) -> PauliSum<W>
where
    T: TruncationPolicy<W> + ?Sized,
{
    let mut scratch = LayerScratch::<W>::new();
    propagate_with(
        circuit,
        sum,
        policy,
        direction,
        &mut scratch,
        PropagateOptions::default(),
    )
}

/// [`propagate`] with a caller-held [`LayerScratch`] and [`PropagateOptions`].
///
/// The scratch keeps its buffer capacity across calls and carries the opt-in [`LayerTrace`](crate::LayerTrace) (and, under `phase-timing`, the phase counters).
/// Log events are emitted on the calling thread, never inside a parallel layer.
///
/// # Panics
///
/// As [`propagate`], and if `options` fail [`PropagateOptions::validate`].
pub fn propagate_with<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    scratch: &mut LayerScratch<W>,
    options: PropagateOptions,
) -> PauliSum<W>
where
    T: TruncationPolicy<W> + ?Sized,
{
    options.assert_valid();
    let num_channels = circuit.channels.len();
    let terms_in = sum.len();
    let started = std::time::Instant::now();
    log::info!(
        target: LOG_TARGET,
        "propagate: {terms_in} terms through {num_channels} channels ({direction:?})",
    );

    // One partition of `run_layers` on the caller's pool (ARCHITECTURE.md §Partitioning).
    let tracing = scratch.trace.is_some();
    let rows = PartitionRows::none(sum.num_qubits());
    let mut work = PartitionWork {
        local: HostPartition::with_layer_scratch(sum, std::mem::take(scratch)),
        rows: Vec::with_capacity(if tracing { num_channels } else { 0 }),
    };
    let context = PartitionContext {
        rows: &rows,
        rank: 0,
        size: 1,
        tracing,
    };
    run_layers(
        circuit,
        policy,
        direction,
        options,
        context,
        &mut work,
        &SoloTransport,
    );
    let (sum, layer_scratch) = work.local.into_parts();
    *scratch = layer_scratch;
    if let Some(trace) = scratch.trace.as_mut() {
        record_trace(trace, &work.rows);
    }

    log::info!(
        target: LOG_TARGET,
        "propagate: {} layers applied, {} -> {} terms, {:.3} s",
        num_channels,
        terms_in,
        sum.len(),
        started.elapsed().as_secs_f64(),
    );

    sum
}

/// Append a solo run's per-layer rows to the scratch's trace.
fn record_trace(trace: &mut LayerTrace, rows: &[PartitionLayerRow]) {
    for row in rows {
        trace.circuit_index.push(row.circuit_index);
        trace.application_index.push(row.application_index);
        trace.gate_name.push(row.gate_name);
        trace.terms_in.push(row.terms_in);
        trace.terms_out.push(row.terms_out);
        trace.nanos.push(row.nanos);
    }
}

#[cfg(test)]
mod tests;
