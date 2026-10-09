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

use crate::circuit::Circuit;
use crate::pauli_sum::hash::PartitionRows;
use crate::pauli_sum::storage::{DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;
use bucketed::LayerScratch;
use partitioned::backend::HostPartition;
use partitioned::driver::{run_layers, PartitionContext, PartitionWork};
use partitioned::trace::PartitionLayerRow;
use partitioned::transport::SoloTransport;

/// `log` target for the engine's progress events, partitioned or not.
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
    pub target_bucket_len: usize,
    /// Floor on the per-layer bucket count once the sum is worth splitting.
    ///
    /// Must be `>= 16`, below which a sum of at most `target_bucket_len` terms no longer reliably gets one bucket.
    pub min_buckets: usize,
}

impl Default for PropagateOptions {
    fn default() -> Self {
        Self {
            target_bucket_len: DEFAULT_TARGET_BUCKET_LEN,
            min_buckets: DEFAULT_MIN_BUCKETS,
        }
    }
}

/// Propagate `initial` through `circuit` under `policy`.
///
/// Under [`Direction::Heisenberg`] the channels run in reverse through [`Channel::apply_adjoint`](crate::Channel::apply_adjoint).
/// The sum stays bucketed throughout, so stepping one sum through repeated calls costs only the layers.
///
/// # Panics
///
/// If a channel's [`Channel::prepare`](crate::Channel::prepare) declines, as for a user channel with support above two qubits; no built-in channel does.
///
/// # Progress logging
///
/// Through the [`log`] facade under the target `paulistrings::propagate`: `INFO` on entry and exit, `DEBUG` once per layer, named by [`Channel::debug_name`](crate::Channel::debug_name).
///
/// # Examples
///
/// ```
/// use paulistrings::{
///     BuildAccumulator, Circuit, Direction, PauliString, Phase, TruncationPolicy,
///     Clifford1Q, propagate,
/// };
/// use num_complex::Complex64;
///
/// let mut accumulator = BuildAccumulator::<1>::new(1);
/// accumulator.add_term(PauliString::<1>::z(0), Complex64::new(1.0, 0.0));
/// let observable = accumulator.finalize();
///
/// let mut circuit = Circuit::<1>::new(1);
/// circuit.push(Clifford1Q::h(0));
///
/// struct KeepAll;
/// impl<const W: usize> TruncationPolicy<W> for KeepAll {}
///
/// // H conjugates Z to X, so propagating Z₀ through H gives X₀.
/// let evolved = propagate(&circuit, observable, &KeepAll, Direction::Heisenberg);
/// assert_eq!(evolved.len(), 1);
/// assert_eq!(evolved.get(&[1], &[0]), Some(Complex64::new(1.0, 0.0)));
/// ```
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
/// The scratch keeps its buffer capacity across calls and carries the opt-in [`TermTrace`](crate::TermTrace) and [`GateTrace`](crate::GateTrace) (and, under `phase-timing`, the phase counters).
/// Log events are emitted on the calling thread, never inside a parallel layer.
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
    let num_channels = circuit.channels.len();
    let terms_in = sum.len();
    let started = std::time::Instant::now();
    log::info!(
        target: LOG_TARGET,
        "propagate: {terms_in} terms through {num_channels} channels ({direction:?})",
    );

    // One partition of `run_layers` on the caller's pool (ARCHITECTURE.md §Partitioning).
    let tracing = scratch.term_trace.is_some() || scratch.gate_trace.is_some();
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
    if tracing {
        record_traces(scratch, &work.rows);
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

/// Append a solo run's per-layer rows to the scratch's enabled traces.
fn record_traces<const W: usize>(scratch: &mut LayerScratch<W>, rows: &[PartitionLayerRow]) {
    if let Some(trace) = scratch.term_trace.as_mut() {
        trace.terms_in.extend(rows.iter().map(|row| row.terms_in));
        trace.terms_out.extend(rows.iter().map(|row| row.terms_out));
    }
    if let Some(trace) = scratch.gate_trace.as_mut() {
        for row in rows {
            trace.circuit_index.push(row.circuit_index);
            trace.application_index.push(row.application_index);
            trace.gate_name.push(row.gate_name);
            trace.terms_in.push(row.terms_in);
            trace.terms_out.push(row.terms_out);
            trace.nanos.push(row.nanos);
        }
    }
}

#[cfg(test)]
mod tests;
