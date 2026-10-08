//! The propagation front door [`propagate`] and its layer loop; one layer is applied in `bucketed`.
//!
//! See ARCHITECTURE.md §Engine.

pub(crate) mod bucketed;
mod coset;
#[cfg(feature = "cuda")]
mod cuda_context;
mod direct;
#[cfg(feature = "cuda")]
pub mod gpu;
mod merge;
pub(crate) mod partitioned;
#[cfg(feature = "phase-timing")]
pub(crate) mod stats;

use crate::channel::prepared::MAX_LOCAL_SUPPORT;
use crate::channel::Channel;
use crate::circuit::Circuit;
use crate::pauli_sum::storage::{DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;
use bucketed::{apply_layer_bucketed, LayerScratch};

/// `log` target for the engine's progress events.
const LOG_TARGET: &str = "paulistrings::propagate";

/// Propagation direction: channels in order, or in reverse with adjoints for backpropagating observables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Apply channels in the order they were pushed onto the [`Circuit`].
    Forward,
    /// Apply channels in reverse order, using each channel's [`Channel::apply_adjoint`].
    Heisenberg,
}

/// Which layer engine [`propagate_with`] uses.
///
/// The alternative to the bucketed sorting engine is a small-sum direct path that applies a layer through [`Channel::apply`] into a hash map, skipping [`Channel::prepare`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EngineSelection {
    /// The bucketed sorting engine for every layer.
    #[default]
    SortedOnly,
    /// The direct path while the sum is at most [`PropagateOptions::small_sum_threshold`] terms and the policy reports no [`TruncationPolicy::finalizes_layer`], then the sorting engine.
    ///
    /// The policy condition is for performance only: a finalizing policy would cost the direct path a materialize and re-ingest per layer.
    Auto,
    /// The direct path whenever the sum is at most [`PropagateOptions::small_sum_threshold`] terms, whatever the policy.
    ///
    /// A policy with a layer pass still gets it once per layer, on a materialized sum.
    SmallSumDirect,
}

/// Default for [`PropagateOptions::small_sum_threshold`]; research/FINDINGS.md §Direct-apply path for small sums.
pub const DEFAULT_SMALL_SUM_THRESHOLD: usize = 2048;

/// Tuning knobs for [`propagate_with`]; [`Default`] is what [`propagate`] uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PropagateOptions {
    /// Which layer engine to use.
    pub engine: EngineSelection,
    /// Resident term count up to which the small-sum direct path is used; ignored under [`EngineSelection::SortedOnly`].
    pub small_sum_threshold: usize,
    /// Terms per bucket the per-layer partition targets (ARCHITECTURE.md §Bucket-Policy).
    ///
    /// Above the `min_buckets` floor raising this alone does nothing; both fields must move together to get fewer buckets, and rebucketing is grow-only, so lowering either mid-run never coarsens a partition.
    pub target_bucket_len: usize,
    /// Floor on the per-layer bucket count once the sum is worth splitting.
    ///
    /// Must be `>= 16`, below which a sum of at most `target_bucket_len` terms no longer reliably gets one bucket.
    /// The small-sum direct path ignores it and sizes its materialized sum from the defaults.
    pub min_buckets: usize,
}

impl Default for PropagateOptions {
    fn default() -> Self {
        Self {
            engine: EngineSelection::SortedOnly,
            small_sum_threshold: DEFAULT_SMALL_SUM_THRESHOLD,
            target_bucket_len: DEFAULT_TARGET_BUCKET_LEN,
            min_buckets: DEFAULT_MIN_BUCKETS,
        }
    }
}

impl PropagateOptions {
    /// Whether a run starting at `len` terms starts on the direct path, the only place it is entered.
    fn starts_direct(&self, len: usize, policy_finalizes: bool) -> bool {
        match self.engine {
            EngineSelection::SortedOnly => false,
            EngineSelection::Auto => len <= self.small_sum_threshold && !policy_finalizes,
            EngineSelection::SmallSumDirect => len <= self.small_sum_threshold,
        }
    }
}

/// Propagate `initial` through `circuit` under `policy`.
///
/// Under [`Direction::Heisenberg`] the channels run in reverse through [`Channel::apply_adjoint`].
/// The sum stays bucketed throughout, so stepping one sum through repeated calls costs only the layers.
///
/// # Panics
///
/// If a channel's [`Channel::prepare`] declines, as for a user channel with support above two qubits; no built-in channel does.
///
/// # Progress logging
///
/// Through the [`log`] facade under the target `paulistrings::propagate`: `INFO` on entry and exit, `DEBUG` once per layer, named by [`Channel::debug_name`].
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
/// accumulator.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
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
/// let (x, z, _c) = evolved.bucket(0);
/// assert_eq!(x[0], [1]);
/// assert_eq!(z[0], [0]);
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
///
/// Under a direct-path [`EngineSelection`] the leading layers run on a hash map until a layer leaves the sum above the threshold; the switch to the sorting engine is one-way.
/// The direct path applies channels of any support width, so a circuit with a channel `prepare` declines panics only once the sum outgrows the threshold.
pub fn propagate_with<const W: usize, T>(
    circuit: &Circuit<W>,
    mut sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    scratch: &mut LayerScratch<W>,
    options: PropagateOptions,
) -> PauliSum<W>
where
    T: TruncationPolicy<W> + ?Sized,
{
    let num_channels = circuit.channels.len();
    let adjoint = matches!(direction, Direction::Heisenberg);
    let terms_in = sum.len();
    let started = std::time::Instant::now();
    log::info!(
        target: LOG_TARGET,
        "propagate: {terms_in} terms through {num_channels} channels ({direction:?})",
    );

    let tracing = scratch.term_trace.is_some();
    let gate_tracing = scratch.gate_trace.is_some();

    let mut start = 0usize;
    if num_channels > 0 && options.starts_direct(terms_in, policy.finalizes_layer()) {
        let (out, applied) =
            direct::run_direct_prefix(circuit, sum, policy, direction, scratch, options);
        sum = out;
        start = applied;
    }

    for application_index in start..num_channels {
        let circuit_index = match direction {
            Direction::Forward => application_index,
            Direction::Heisenberg => num_channels - 1 - application_index,
        };
        let channel: &dyn Channel<W> = circuit.channels[circuit_index].as_ref();

        // The clock is read only when a logger or the gate trace wants it.
        let debug_on = log::log_enabled!(target: LOG_TARGET, log::Level::Debug);
        let want_timer = gate_tracing || debug_on;
        let layer_started = want_timer.then(std::time::Instant::now);
        let terms_before = sum.len();

        #[cfg(feature = "phase-timing")]
        let mut stamp = stats::Stamp::now();
        #[cfg(feature = "phase-timing")]
        {
            scratch.stats.layers += 1;
            scratch.stats.terms_in += sum.len() as u64;
        }

        sum.rebucket(options.target_bucket_len, options.min_buckets);
        #[cfg(feature = "phase-timing")]
        stamp.lap(&mut scratch.stats.rebucket_ns);

        let prepared = channel.prepare(sum.hash(), adjoint);
        #[cfg(feature = "phase-timing")]
        stamp.lap(&mut scratch.stats.prepare_ns);

        match prepared {
            Some(prepared) => {
                apply_layer_bucketed(&mut sum, &prepared, policy, scratch);
                #[cfg(feature = "phase-timing")]
                stamp.rearm();
            }
            None => {
                // Reachable only from a user `Channel`: every built-in has support ≤ MAX_LOCAL_SUPPORT or, like `PauliRotation`, overrides `prepare`.
                let weight: u32 = channel.support().iter().map(|w| w.count_ones()).sum();
                panic!(
                    "layer {circuit_index}: Channel::prepare declined, so this channel cannot be \
                     propagated. The engine tabulates channels of support ≤ \
                     {MAX_LOCAL_SUPPORT} qubits (this one declares {weight}), and a \
                     channel must not write outside its declared support. See \
                     research/FINDINGS.md",
                );
            }
        }

        policy.finalize_layer(&mut sum);
        #[cfg(feature = "phase-timing")]
        {
            stamp.lap(&mut scratch.stats.finalize_ns);
            scratch.stats.terms_out += sum.len() as u64;
        }
        // Behind a hoisted flag and a `#[cold]` callee: this loop inlines the merge kernels, which are sensitive to code motion (CLAUDE.md §Performance discipline).
        if tracing {
            record_layer_terms(scratch, terms_before, sum.len());
        }

        if want_timer {
            let elapsed = layer_started
                .expect("want_timer implies layer_started is Some")
                .elapsed();
            if gate_tracing {
                record_gate_trace(
                    scratch,
                    circuit_index as u32,
                    application_index as u32,
                    channel.debug_name(),
                    terms_before,
                    sum.len(),
                    elapsed,
                );
            }
            if debug_on {
                log::debug!(
                    target: LOG_TARGET,
                    "layer {}/{} [{}]: {} -> {} terms, {:.1} ms",
                    application_index + 1,
                    num_channels,
                    channel.debug_name(),
                    terms_before,
                    sum.len(),
                    elapsed.as_secs_f64() * 1e3,
                );
            }
        }
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

/// Append one layer's term counts; cold and out of line to keep the trace off the inlined layer path.
#[cold]
#[inline(never)]
fn record_layer_terms<const W: usize>(
    scratch: &mut LayerScratch<W>,
    terms_in: usize,
    terms_out: usize,
) {
    if let Some(trace) = scratch.term_trace.as_mut() {
        trace.terms_in.push(terms_in);
        trace.terms_out.push(terms_out);
    }
}

/// Append one layer's [`GateTrace`](crate::GateTrace) record; cold for the same reason as [`record_layer_terms`].
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn record_gate_trace<const W: usize>(
    scratch: &mut LayerScratch<W>,
    circuit_index: u32,
    application_index: u32,
    gate_name: &'static str,
    terms_in: usize,
    terms_out: usize,
    elapsed: std::time::Duration,
) {
    if let Some(trace) = scratch.gate_trace.as_mut() {
        trace.circuit_index.push(circuit_index);
        trace.application_index.push(application_index);
        trace.gate_name.push(gate_name);
        trace.terms_in.push(terms_in);
        trace.terms_out.push(terms_out);
        trace.nanos.push(elapsed.as_nanos() as u64);
    }
}

#[cfg(test)]
mod tests;
