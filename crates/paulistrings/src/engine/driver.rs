//! The layer loop `run_layers` every driver runs, one partition's share of a call at any partition count.
//!
//! Every partition issues the same transport calls in the same order, or the group falls out of step; per layer:
//! 1. on a layer whose plan has a remote delta, an `exchange` of the proposed bucket bits within its exchange group ([`PartitionPlan::exchange_group`]), whose maximum every member then holds, since partners index an exchange's blocks by the count;
//! 2. the exchange, only if the plan has a remote delta;
//! 3. the policy's layer pass, on every layer, if `finalizes_layer` is true;
//!
//! and one `allreduce_max_u8` after the last layer, which refines every partition to the group's largest count, so a call returns with one count everywhere.
//! A local layer issues no call for the count: on the schedule of `rebuckets` a partition moves to its own proposal, so counts differ mid-call.
//! At `P = 1` there is no group: the loop refines every layer, as `propagate` does.
//!
//! Consecutive local layers may run as one pass ([`Runs`]); local layers issue no transport call, so a partition may take a run whatever its peers do.
//! A run defers the policy's layer pass to its end, so under a policy with one a group takes runs only over a backend that tracks [`PartitionBackend::agreed_len`]: one `allreduce_sum_u64` at the start of the call agrees it, and every partition admits a run against that bound and a span no bucket count refuses, so all take the same runs and the same passes.

use crate::channel::Prepare;
use crate::coefficient::Coefficient;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use super::backend::PartitionBackend;
use super::bucketed::{
    max_fanout, PermutationStep, RotationRunSpan, RowEmitter, Run, MAX_ROTATION_RUN_LAYERS,
};
use super::plan::PartitionPlan;
use super::transport::{ByteParts, Transport};
use crate::channel::prepared::{Prepared, MAX_LOCAL_SUPPORT};
use crate::channel::Channel;
use crate::circuit::Circuit;
use crate::collectives::Collectives;
use crate::engine::timing::{lap, stamp, tally, timed};
use crate::engine::trace::{record_layer_row, LayerClock, LayerPhases, PartitionLayerRow};
use crate::engine::{Direction, PropagateOptions, LOG_TARGET};
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows};
use crate::truncation::TruncationPolicy;

/// Layers between two rebuckets of a partition in a group, and the length of the opening ramp that precedes them.
const REBUCKET_EVERY: usize = 16;

/// Whether a partition of a group rebuckets at the layer at `application_index`, whatever the layer's plan.
fn rebuckets(application_index: usize) -> bool {
    application_index < REBUCKET_EVERY || application_index.is_multiple_of(REBUCKET_EVERY)
}

/// The largest of `bits` over this partition and `group`, one `exchange` among them.
fn exchange_bits<X: Transport>(transport: &X, group: &[u32], bits: u8) -> u8 {
    let mut send: Vec<Option<ByteParts>> = (0..transport.size()).map(|_| None).collect();
    for &member in group {
        send[member as usize] = Some(ByteParts(vec![vec![bits]]));
    }
    let received = transport.exchange(send, &mut Vec::new());
    received.iter().flatten().map(|payload| payload.0[0][0]).fold(bits, u8::max)
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
) -> Prepared<W, C> {
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

/// The layers after the current one, prepared on demand under the current hash.
struct Lookahead<'a, const W: usize> {
    circuit: &'a Circuit<W>,
    circuit_index_of: &'a dyn Fn(usize) -> usize,
    rows: &'a PartitionRows<W>,
    /// Scratch for [`exchanges`](Self::exchanges).
    plan: RefCell<PartitionPlan>,
    adjoint: bool,
    rank: usize,
}

impl<const W: usize> Lookahead<'_, W> {
    /// Whether `prepared` has a remote delta, which depends on the delta alone and so is alike on every partition.
    fn exchanges<C: Coefficient>(&self, prepared: &Prepared<W, C>) -> bool {
        let mut plan = self.plan.borrow_mut();
        plan.rebuild(prepared, self.rows, self.rank as u32);
        plan.has_remote()
    }

    /// Hand `accept` the layers from application index `start` on, prepared under `hash`, until it refuses one, one exchanges, or it took `limit`; returns how many it took.
    fn extend<C: Coefficient>(
        &self,
        start: usize,
        hash: &Gf2Hash<W>,
        limit: usize,
        mut accept: impl FnMut(Prepared<W, C>) -> bool,
    ) -> usize {
        let end = self.circuit.channels.len().min(start.saturating_add(limit));
        let mut taken = 0;
        for application_index in start..end {
            let index = (self.circuit_index_of)(application_index);
            let channel = self.circuit.channels[index].as_ref();
            let prepared = prepare_or_panic(channel, hash, self.adjoint, self.rank, index);
            if self.exchanges(&prepared) || !accept(prepared) {
                break;
            }
            taken += 1;
        }
        taken
    }
}

/// The runs a call takes, consecutive local layers applied in one pass by [`PartitionBackend::apply_run`], and the buffers kept between them.
struct Runs<const W: usize, C: Coefficient> {
    /// Runs of one-to-one channels (`engine/bucketed/permutation_run.rs`), tried first.
    permutations: bool,
    /// Rotation runs (`engine/bucketed/rotation_layer.rs`), up to this bucket-delta span dimension.
    rotations: Option<usize>,
    /// [`TruncationPolicy::run_length_limit`]: a run's input length times its layers' fanouts stays within it.
    limit: usize,
    steps: Vec<PermutationStep<W, C>>,
    layers: Vec<Prepared<W, C>>,
}

impl<const W: usize, C: Coefficient> Runs<W, C> {
    /// Apply the run that starts at application index `start` with `prepared`, admitted against `len` terms, and return its end, or `None` to apply layer `start` alone.
    // Never inlined: `run_layers` inlines the merge kernels, which are sensitive to code motion.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn take<T, B>(
        &mut self,
        start: usize,
        prepared: &Prepared<W, C>,
        len: usize,
        local: &mut B,
        policy: &T,
        lookahead: &Lookahead<'_, W>,
        logging: bool,
    ) -> Option<usize>
    where
        T: TruncationPolicy<W, C> + ?Sized,
        B: PartitionBackend<W, C, T>,
    {
        if lookahead.exchanges(prepared) {
            return None;
        }
        if self.permutations && len <= self.limit {
            if let Some(first) = PermutationStep::of(prepared) {
                self.steps.clear();
                self.steps.push(first);
                let steps = &mut self.steps;
                let joined = lookahead.extend(start + 1, local.hash(), usize::MAX, |next| {
                    PermutationStep::of(&next).map(|step| steps.push(step)).is_some()
                });
                if joined > 0 {
                    local.apply_run(Run::Permutations(&self.steps), policy);
                    return Some(start + 1 + joined);
                }
            }
        }
        let dimension = self.rotations.filter(|_| !logging)?;
        let limit = self.limit;
        let mut worst = len.saturating_mul(max_fanout(prepared));
        let mut span = RotationRunSpan::new(local.hash().bits(), dimension);
        if worst > limit || !span.join(prepared) {
            return None;
        }
        self.layers.clear();
        let layers = &mut self.layers;
        let joined =
            lookahead.extend(start + 1, local.hash(), MAX_ROTATION_RUN_LAYERS - 1, |next| {
                let grown = worst.saturating_mul(max_fanout(&next));
                grown <= limit && span.join(&next) && {
                    worst = grown;
                    layers.push(next);
                    true
                }
            });
        if joined == 0 {
            return None;
        }
        self.layers.insert(0, prepared.clone());
        local.apply_run(Run::Rotations(&self.layers), policy);
        Some(start + 1 + joined)
    }
}

/// Trace a run of the layers `layers` as one row per layer, its time and phases on the first and the backend's counters for the run on each, into `trace` when given, and log it at `DEBUG`.
#[cold]
#[allow(clippy::too_many_arguments)]
fn record_run<const W: usize, C: Coefficient, T: ?Sized, B: PartitionBackend<W, C, T>>(
    circuit: &Circuit<W>,
    layers: std::ops::Range<usize>,
    circuit_index_of: &dyn Fn(usize) -> usize,
    local: &B,
    (terms_before, bits_before, nanos): (usize, u8, u64),
    collectives: u32,
    mut phases: LayerPhases,
    (rank, size): (usize, usize),
    trace: Option<&mut Vec<PartitionLayerRow>>,
) {
    let first = layers.start;
    if let Some(trace) = trace {
        for layer in layers.clone() {
            let index = circuit_index_of(layer);
            let leads = layer == first;
            let mut row = PartitionLayerRow {
                bits: local.hash().bits(),
                remote_deltas: 0,
                collectives: if leads { collectives } else { 0 },
                circuit_index: index as u32,
                application_index: layer as u32,
                gate_name: circuit.channels[index].debug_name(),
                terms_in: if leads { terms_before } else { local.len() },
                terms_out: local.len(),
                rows_sent: vec![0; size],
                bytes_sent: vec![0; size],
                rows_received: 0,
                nanos: if leads { nanos } else { 0 },
                refine_passes: if leads {
                    u32::from(local.hash().bits().saturating_sub(bits_before))
                } else {
                    0
                },
                receive_chunks: 0,
                device: None,
                phases: if leads { std::mem::take(&mut phases) } else { LayerPhases::default() },
            };
            local.trace_layer(&mut row);
            if !leads {
                (row.refine_passes, row.receive_chunks) = (0, 0);
            }
            record_layer_row(trace, row);
        }
    }
    log::debug!(
        target: LOG_TARGET,
        "layers {}-{}/{} [run]: {terms_before} -> {} terms, {:.1} ms (partition {rank}/{size})",
        first + 1,
        layers.end,
        circuit.channels.len(),
        local.len(),
        nanos as f64 * 1e-6,
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

    let circuit_index_of = |application_index: usize| match direction {
        Direction::Forward => application_index,
        Direction::Heisenberg => num_layers - 1 - application_index,
    };
    let lookahead = Lookahead {
        circuit,
        circuit_index_of: &circuit_index_of,
        rows,
        plan: RefCell::default(),
        adjoint,
        rank,
    };
    // A run's layer pass follows it alone, so a group under a layer pass admits runs only against the agreed bound, and only spans no partition's bucket count refuses.
    let agrees = finalizes && !solo;
    let limit = if agrees && !B::TRACKS_AGREED_LEN { 0 } else { policy.run_length_limit() };
    let mut runs = Runs::<W, C> {
        permutations: limit > 0 && local.takes_runs(),
        rotations: local
            .rotation_run_dimension()
            .filter(|&dimension| limit > 0 && (!agrees || dimension >= MAX_ROTATION_RUN_LAYERS)),
        limit,
        steps: Vec::new(),
        layers: Vec::new(),
    };

    local.begin_call();
    // Collectives issued before the first layer, counted on its trace row.
    let mut opening_collectives = 0u32;
    if agrees && limit > 0 {
        let mut total = [local.len() as u64];
        transport.allreduce_sum_u64(&mut total);
        local.agree_len(total[0] as usize);
        opening_collectives = 1;
    }
    let mut application_index = 0;
    while application_index < num_layers {
        let circuit_index = circuit_index_of(application_index);
        let channel: &dyn Channel<W> = circuit.channels[circuit_index].as_ref();

        let debug_enabled = log::log_enabled!(target: LOG_TARGET, log::Level::Debug);
        let timed_layer = tracing || debug_enabled;
        let layer_start = timed_layer.then(Instant::now);
        // A traced layer reads exact counts, which on a device costs a synchronization; untraced, the length is the backend's bound.
        let terms_before = if tracing { local.exact_len() } else { local.len() };
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

        let mut collectives = std::mem::take(&mut opening_collectives);
        if !refined_early && (solo || plan.has_remote() || rebuckets(application_index)) {
            let mut wanted_bits =
                local.proposed_bits(&prepared, options.target_bucket_len, options.min_buckets);
            if !solo && plan.has_remote() {
                wanted_bits =
                    exchange_bits(transport, &plan.exchange_group(rank as u32), wanted_bits);
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

        // A group under a layer pass decides from what every partition holds alike, debug logging excepted.
        let admitted = if agrees { local.agreed_len() } else { Some(local.len()) };
        let logging = debug_enabled && !agrees;
        let taken = admitted.and_then(|len| {
            runs.take(application_index, &prepared, len, local, policy, &lookahead, logging)
        });
        if let Some(end) = taken {
            timed!(stamp.rearm());
            clock.lap(&mut phases.coset_loop_ns);
            if finalizes {
                policy_calls.store(0, Ordering::Relaxed);
                let counting =
                    CountingCollectives { inner: transport, calls: &policy_calls, nanos: None };
                local.finalize_layer(policy, &counting);
                collectives += policy_calls.load(Ordering::Relaxed);
                lap!(stamp, local.stats().finalize_ns);
                clock.lap(&mut phases.finalize_ns);
            }
            if tracing {
                local.exact_len();
            }
            // Per layer of the run, as the one-by-one path counts them.
            tally!(local.stats().layers, end - application_index - 1);
            tally!(local.stats().terms_in, terms_before * (end - application_index - 1));
            tally!(local.stats().terms_out, local.len() * (end - application_index));
            #[cfg(feature = "phase-timing")]
            {
                phases.fine = stats_before.map(|before| {
                    crate::engine::trace::FineLayerPhases::between(&before, &local.phase_snapshot())
                });
            }
            if let Some(start) = layer_start {
                let before = (terms_before, bits_before, start.elapsed().as_nanos() as u64);
                let trace = tracing.then_some(&mut work.rows);
                let layers = application_index..end;
                let order = &circuit_index_of;
                record_run::<W, C, T, B>(
                    circuit,
                    layers,
                    order,
                    &*local,
                    before,
                    collectives,
                    phases,
                    (rank, size),
                    trace,
                );
            }
            application_index = end;
            continue;
        }

        // The next layer prepared under this layer's hash, so the coset loop can count the rows it exports; a rebucket in between drops the counts.
        let next_prepared = (!solo && application_index + 1 < num_layers).then(|| {
            let next_index = circuit_index_of(application_index + 1);
            let next_channel = circuit.channels[next_index].as_ref();
            let next = prepare_or_panic(next_channel, local.hash(), adjoint, rank, next_index);
            next_plan.rebuild(&next, rows, rank as u32);
            next
        });
        let ahead = match (&next_prepared, next_plan.remote.as_slice()) {
            (Some(next), [remote]) => Some(RowEmitter::of(next, remote)),
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
        let terms_out = if tracing { local.exact_len() } else { local.len() };
        tally!(local.stats().terms_out, terms_out);

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
                terms_out,
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

        application_index += 1;
        if debug_enabled {
            let Some(elapsed) = elapsed else { continue };
            log::debug!(
                target: LOG_TARGET,
                "layer {}/{} [{}]: {} -> {} terms, {:.1} ms (partition {}/{}, {} remote deltas, \
                 {} rows in)",
                application_index,
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
    if !solo {
        let agreed = transport.allreduce_max_u8(local.hash().bits());
        while local.hash().bits() < agreed {
            local.refine();
        }
    }
    local.settle();
}

#[cfg(test)]
mod tests;
