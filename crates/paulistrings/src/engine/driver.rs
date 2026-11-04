//! The layer loop `run_layers` every driver runs, one partition's share of a call at any partition count.
//!
//! Every partition issues the same transport calls in the same order, or the group falls out of step; per layer:
//! 1. `allreduce_max_u8` of the proposed bucket bits, on any layer whose plan has a remote delta (both sides index the exchange's blocks by the count) and otherwise on the schedule of `agrees_bucket_bits`;
//! 2. the exchange, only if the plan has a remote delta;
//! 3. the policy's layer pass, on every layer, if `finalizes_layer` is true.
//!
//! Between agreements a partition keeps its count even when its own proposal is higher, so the counts stay equal without communicating, and the schedule depends on the layer index and the plan alone.
//! At `P = 1` there is no group: the loop skips the reduction and refines every layer, as `propagate` does.

use crate::channel::Prepare;
use crate::coefficient::Coefficient;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use super::backend::PartitionBackend;
use super::bucketed::EmitTest;
use super::plan::PartitionPlan;
use super::transport::Transport;
use crate::channel::prepared::MAX_LOCAL_SUPPORT;
use crate::channel::Channel;
use crate::circuit::Circuit;
use crate::collectives::Collectives;
use crate::engine::timing::{lap, stamp, tally, timed};
use crate::engine::trace::{record_layer_row, LayerClock, LayerPhases, PartitionLayerRow};
use crate::engine::{Direction, PropagateOptions, LOG_TARGET};
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows};
use crate::truncation::TruncationPolicy;

/// Layers between two bucket-count agreements, and the length of the opening ramp that precedes them.
pub(crate) const BITS_AGREE_EVERY: usize = 16;

/// Whether the layer at `application_index` of a call agrees the bucket count; a pure function of it, so every partition answers alike.
fn agrees_bucket_bits(application_index: usize) -> bool {
    application_index < BITS_AGREE_EVERY || application_index.is_multiple_of(BITS_AGREE_EVERY)
}

/// Panic on every partition unless all hold `bits` bucket bits; two collectives, outside the module's schedule, so only a debug build issues them.
#[cfg(debug_assertions)]
fn assert_bucket_bits_agree(collectives: &impl Collectives, bits: u8, application_index: usize) {
    let highest = collectives.allreduce_max_u8(bits);
    let lowest = u8::MAX - collectives.allreduce_max_u8(u8::MAX - bits);
    assert_eq!(
        lowest,
        highest,
        "partition {}, layer {application_index}: the partitions hold between {lowest} and {highest} \
         bucket bits (this one {bits}), where the bucket-count agreement keeps them equal",
        collectives.rank(),
    );
}

/// A [`Collectives`] view that counts the policy finalization's calls for the trace and, when `nanos` is given, times them.
struct CountingCollectives<'a> {
    inner: &'a dyn Collectives,
    calls: &'a AtomicU32,
    nanos: Option<&'a AtomicU64>,
}

impl CountingCollectives<'_> {
    fn call<R>(&self, collective: impl FnOnce() -> R) -> R {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let start = self.nanos.map(|_| Instant::now());
        let result = collective();
        if let (Some(nanos), Some(start)) = (self.nanos, start) {
            nanos.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        result
    }
}

impl crate::collectives::sealed::Sealed for CountingCollectives<'_> {}

impl Collectives for CountingCollectives<'_> {
    fn rank(&self) -> u32 {
        self.inner.rank()
    }
    fn size(&self) -> u32 {
        self.inner.size()
    }
    fn allreduce_max_u8(&self, v: u8) -> u8 {
        self.call(|| self.inner.allreduce_max_u8(v))
    }
    fn allreduce_sum_u64(&self, buffer: &mut [u64]) {
        self.call(|| self.inner.allreduce_sum_u64(buffer))
    }
    fn allreduce_sum_f64(&self, buffer: &mut [f64]) {
        self.call(|| self.inner.allreduce_sum_f64(buffer))
    }
    fn barrier(&self) {
        self.call(|| self.inner.barrier())
    }
}

/// One partition's payload, moved into its thread for the duration of a call and handed back.
pub(crate) struct PartitionWork<B> {
    pub(crate) local: B,
    /// One row per layer, empty unless tracing is on.
    pub(crate) rows: Vec<PartitionLayerRow>,
}

/// What a partition knows about itself while it walks the layers.
pub(crate) struct PartitionContext<'a, const W: usize> {
    pub(crate) rows: &'a PartitionRows<W>,
    pub(crate) rank: usize,
    pub(crate) size: usize,
    pub(crate) tracing: bool,
}

/// Why a channel's preparation declines, shared by the host panic and the device's `GpuError::Unsupported`.
pub(crate) const DECLINED_PREPARATION: &str = "support above MAX_LOCAL_SUPPORT (2) qubits is \
     unsupported except for PauliRotation, and a channel must not write outside its declared support";
const _: () = assert!(MAX_LOCAL_SUPPORT == 2, "DECLINED_PREPARATION names the value");

/// The channel's preparation, or the unpartitioned engine's hard error naming the partition and layer.
fn prepare_or_panic<const W: usize, C: Coefficient>(
    channel: &dyn Channel<W>,
    hash: &Gf2Hash<W>,
    adjoint: bool,
    rank: usize,
    circuit_index: usize,
) -> crate::channel::prepared::Prepared<W, C> {
    let prepared = channel.prepare(hash, adjoint).unwrap_or_else(|| {
        let weight: u32 = channel.support().iter().map(|w| w.count_ones()).sum();
        panic!(
            "partition {rank}, layer {circuit_index}: channel preparation declined for a \
             channel of support {weight}: {DECLINED_PREPARATION}",
        )
    });
    prepared.convert().expect("assert_coefficient_type admitted the circuit")
}

/// Panic unless a sum of coefficient type `C` can carry `circuit`: a real type needs [`Circuit::preserves_hermiticity`].
pub(crate) fn assert_coefficient_type<const W: usize, C: Coefficient>(circuit: &Circuit<W>) {
    assert!(
        !C::IS_REAL || circuit.preserves_hermiticity(),
        "propagate: a {} sum needs every channel to preserve Hermiticity (a real transfer table); \
         use a complex coefficient type",
        C::NAME,
    );
}

/// One partition's layer loop, run in lock-step with its peers inside its own pool, in the module's collective order.
pub(crate) fn run_layers<const W: usize, C: Coefficient, T, X, B>(
    circuit: &Circuit<W>,
    policy: &T,
    direction: Direction,
    options: PropagateOptions,
    context: PartitionContext<'_, W>,
    work: &mut PartitionWork<B>,
    transport: &X,
) where
    T: TruncationPolicy<W, C> + ?Sized,
    X: Transport,
    B: PartitionBackend<W, C, T>,
{
    assert_coefficient_type::<W, C>(circuit);
    let PartitionContext { rows, rank, size, tracing } = context;
    let num_layers = circuit.channels.len();
    let adjoint = matches!(direction, Direction::Heisenberg);
    let group_size = transport.size();
    // `finalizes_layer` is a property of the policy type, so skipping the collective call is safe on every partition alike.
    let finalizes = policy.finalizes_layer();
    let policy_calls = AtomicU32::new(0);
    let policy_nanos = AtomicU64::new(0);
    // At `P = 1` the local answer is the agreed one, so the loop rebuckets every layer exactly as an unpartitioned run would.
    let solo = group_size == 1;
    let local = &mut work.local;
    let mut plan = PartitionPlan::default();
    let mut next_plan = PartitionPlan::default();
    // Overlap with the bucket rows only costs balance once the bucket count reaches it, so it is reported there, once, by partition 0.
    let mut overlap_onset = if rank == 0 { rows.overlap_onset(local.hash()) } else { None };

    for application_index in 0..num_layers {
        let circuit_index = match direction {
            Direction::Forward => application_index,
            Direction::Heisenberg => num_layers - 1 - application_index,
        };
        let channel: &dyn Channel<W> = circuit.channels[circuit_index].as_ref();

        let debug_enabled = log::log_enabled!(target: LOG_TARGET, log::Level::Debug);
        let timed_layer = tracing || debug_enabled;
        let layer_start = timed_layer.then(Instant::now);
        let terms_before = local.len();
        let bits_before = local.hash().bits();
        let mut phases = LayerPhases::default();
        let mut clock = LayerClock::new(if tracing { layer_start } else { None });
        #[cfg(feature = "phase-timing")]
        let stats_before = tracing.then(|| local.phase_snapshot());

        stamp!(stamp);
        tally!(local.stats().layers, 1);
        tally!(local.stats().terms_in, terms_before);

        // Alone, a backend that sizes its buckets without the prepared channel settles the count first, so a growing layer prepares once.
        let refined_early =
            solo && local.refine_unprepared(options.target_bucket_len, options.min_buckets);
        if refined_early {
            lap!(stamp, local.stats().rebucket_ns);
        }
        clock.lap(&mut phases.rebucket_ns);

        // Otherwise prepared before the bucket count is settled, since the plan decides whether this layer agrees it; only `bucket_delta` depends on the count, so a refining layer prepares again.
        let mut prepared = prepare_or_panic(channel, local.hash(), adjoint, rank, circuit_index);
        plan.rebuild(&prepared, rows, rank as u32);
        lap!(stamp, local.stats().prepare_ns);
        clock.lap(&mut phases.prepare_ns);

        let mut collectives = 0u32;
        if !refined_early && (solo || plan.has_remote() || agrees_bucket_bits(application_index)) {
            let mut wanted_bits =
                local.proposed_bits(&prepared, options.target_bucket_len, options.min_buckets);
            if !solo {
                wanted_bits = transport.allreduce_max_u8(wanted_bits);
                collectives += 1;
            }
            lap!(stamp, local.stats().collective_ns);
            clock.lap(&mut phases.collective_ns);
            if wanted_bits != local.hash().bits() {
                while local.hash().bits() < wanted_bits {
                    local.refine();
                }
                if wanted_bits < local.hash().bits() {
                    local.coarsen_to(wanted_bits);
                }
                lap!(stamp, local.stats().rebucket_ns);
                clock.lap(&mut phases.rebucket_ns);
                prepared = prepare_or_panic(channel, local.hash(), adjoint, rank, circuit_index);
                plan.rebuild(&prepared, rows, rank as u32);
                lap!(stamp, local.stats().prepare_ns);
                clock.lap(&mut phases.prepare_ns);
            }
        }

        if overlap_onset.is_some_and(|onset| local.hash().bits() >= onset) {
            overlap_onset = None;
            log::warn!(
                target: LOG_TARGET,
                "layer {}/{num_layers}: at {} buckets the partition rows overlap the span of the bucket hash's active rows, so partitions and buckets are correlated and some of their pairs stay empty",
                application_index + 1,
                local.hash().num_buckets(),
            );
        }

        // Gated on the build alone, never on data, so every partition issues these collectives alike.
        #[cfg(debug_assertions)]
        if !solo {
            assert_bucket_bits_agree(transport, local.hash().bits(), application_index);
        }
        clock.lap(&mut phases.collective_ns);

        // The next layer prepared under this layer's hash, so the coset loop can count the rows it exports; a rebucket in between drops the counts.
        let next_prepared = (!solo && application_index + 1 < num_layers).then(|| {
            let next_index = match direction {
                Direction::Forward => circuit_index + 1,
                Direction::Heisenberg => circuit_index - 1,
            };
            let next_channel: &dyn Channel<W> = circuit.channels[next_index].as_ref();
            let next = prepare_or_panic(next_channel, local.hash(), adjoint, rank, next_index);
            next_plan.rebuild(&next, rows, rank as u32);
            next
        });
        let ahead = match (&next_prepared, next_plan.remote.as_slice()) {
            (Some(next), [remote]) => Some(EmitTest::of(next, remote)),
            _ => None,
        };
        lap!(stamp, local.stats().prepare_ns);
        clock.lap(&mut phases.prepare_ns);

        let counts =
            local.apply_layer(&prepared, &plan, rows, policy, transport, tracing, ahead.as_ref());
        timed!(stamp.rearm());
        let mut applied_ns = 0;
        clock.lap(&mut applied_ns);
        phases.export_ns = counts.export_ns;
        phases.exchange_ns = counts.exchange_ns;
        phases.coset_loop_ns =
            applied_ns.saturating_sub(counts.export_ns).saturating_sub(counts.exchange_ns);

        if finalizes {
            policy_calls.store(0, Ordering::Relaxed);
            policy_nanos.store(0, Ordering::Relaxed);
            let counting = CountingCollectives {
                inner: transport,
                calls: &policy_calls,
                nanos: tracing.then_some(&policy_nanos),
            };
            local.finalize_layer(policy, &counting);
            collectives += policy_calls.load(Ordering::Relaxed);
        }
        lap!(stamp, local.stats().finalize_ns);
        clock.lap(&mut phases.finalize_ns);
        let policy_collective_ns = policy_nanos.load(Ordering::Relaxed).min(phases.finalize_ns);
        phases.finalize_ns -= policy_collective_ns;
        phases.collective_ns += policy_collective_ns;
        #[cfg(feature = "phase-timing")]
        {
            phases.fine = stats_before.map(|before| {
                crate::engine::trace::FineLayerPhases::between(&before, &local.phase_snapshot())
            });
        }
        tally!(local.stats().terms_out, local.len());

        // The trace sits behind a hoisted flag and a `#[cold]` callee: this loop inlines the merge kernels, which are sensitive to code motion.
        let remote_deltas = counts.remote_deltas;
        let rows_received = counts.rows_received;
        let elapsed = layer_start.map(|start| start.elapsed());
        if tracing {
            let mut row = PartitionLayerRow {
                bits: local.hash().bits(),
                remote_deltas: counts.remote_deltas as u32,
                collectives,
                circuit_index: circuit_index as u32,
                application_index: application_index as u32,
                gate_name: channel.debug_name(),
                terms_in: terms_before,
                terms_out: local.len(),
                rows_sent: counts.rows_sent,
                bytes_sent: counts.bytes_sent,
                rows_received: counts.rows_received,
                nanos: elapsed.unwrap_or_default().as_nanos() as u64,
                refine_passes: u32::from(local.hash().bits().saturating_sub(bits_before)),
                receive_chunks: counts.receive_chunks,
                device: None,
                phases,
            };
            local.trace_layer(&mut row);
            record_layer_row(&mut work.rows, row);
        }

        if debug_enabled {
            let Some(elapsed) = elapsed else { continue };
            log::debug!(
                target: LOG_TARGET,
                "layer {}/{} [{}]: {} -> {} terms, {:.1} ms (partition {}/{}, {} remote deltas, \
                 {} rows in)",
                application_index + 1,
                num_layers,
                channel.debug_name(),
                terms_before,
                local.len(),
                elapsed.as_secs_f64() * 1e3,
                rank,
                size,
                remote_deltas,
                rows_received,
            );
        }
    }
    local.settle();
}

#[cfg(test)]
mod tests;
