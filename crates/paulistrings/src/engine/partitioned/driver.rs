//! The partitioned layer loop `run_layers`, shared by every partitioned driver, and the [`propagate_partitioned`] front door (ARCHITECTURE.md §Partitioning).

use crate::channel::Prepare;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use super::backend::{PartitionBackend, PartitionStorage};
use super::plan::PartitionPlan;
use super::topology::{PartitionConfig, TopologyError};
use super::trace::{record_layer_row, PartitionLayerRow};
use super::transport::Transport;
use crate::channel::prepared::MAX_LOCAL_SUPPORT;
use crate::channel::Channel;
use crate::circuit::Circuit;
use crate::collectives::Collectives;
use crate::engine::timing::{lap, stamp, tally, timed};
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows};
use crate::pauli_sum::storage::{desired_bits, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

use super::sum::PartitionedSum;

pub(super) use crate::engine::LOG_TARGET;

/// Layers between two bucket-count agreements, and the length of the opening ramp that precedes them (ARCHITECTURE.md §Partitioning).
pub(crate) const BITS_AGREE_EVERY: usize = 16;

/// Whether the layer at `application_index` of a call agrees the bucket count; a pure function of it, so every partition answers alike.
fn agrees_bucket_bits(application_index: usize) -> bool {
    application_index < BITS_AGREE_EVERY || application_index.is_multiple_of(BITS_AGREE_EVERY)
}

/// A [`Collectives`] view that counts the policy finalization's calls for the trace.
struct CountingCollectives<'a> {
    inner: &'a dyn Collectives,
    calls: &'a AtomicU32,
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
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.allreduce_max_u8(v)
    }
    fn allreduce_sum_u64(&self, buffer: &mut [u64]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.allreduce_sum_u64(buffer)
    }
    fn allreduce_sum_f64(&self, buffer: &mut [f64]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.allreduce_sum_f64(buffer)
    }
    fn barrier(&self) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.barrier()
    }
}

/// One partition's payload, moved into its thread for the duration of a call and handed back.
pub(crate) struct PartitionWork<B> {
    pub(crate) local: B,
    /// One row per layer, empty unless tracing is on.
    pub(crate) rows: Vec<PartitionLayerRow>,
}

impl<B> PartitionWork<B> {
    pub(super) fn take<const W: usize>(local: &mut B, layers: usize, tracing: bool) -> Self
    where
        B: PartitionStorage<W>,
    {
        Self {
            local: local.detach(),
            rows: Vec::with_capacity(if tracing { layers } else { 0 }),
        }
    }
}

/// One partition's share of `sum` at the agreed bucket count; collective, and run on the partition's own pool for first touch.
pub(crate) fn scatter_local<const W: usize>(
    sum: &PauliSum<W>,
    rows: &PartitionRows<W>,
    rank: u32,
    collectives: &dyn Collectives,
) -> PauliSum<W> {
    let mut local = sum.filter_partition(rows, rank);
    let want = desired_bits(local.len(), DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS);
    let want = collectives.allreduce_max_u8(want);
    local.coarsen_to(scatter_bits(local.hash().bits(), rows.bits(), want));
    local
}

/// The scatter bits rule of ARCHITECTURE.md §Partitioning, with `pbits = log2(P)` and `want` the all-reduced [`desired_bits`].
fn scatter_bits(bits: u8, pbits: u8, want: u8) -> u8 {
    want.max(bits.saturating_sub(pbits)).min(bits)
}

/// What a partition knows about itself while it walks the layers.
pub(crate) struct PartitionContext<'a, const W: usize> {
    pub(crate) rows: &'a PartitionRows<W>,
    pub(crate) rank: usize,
    pub(crate) size: usize,
    pub(crate) tracing: bool,
}

/// The channel's preparation, or the unpartitioned engine's hard error naming the partition and layer.
fn prepare_or_panic<const W: usize>(
    channel: &dyn Channel<W>,
    hash: &Gf2Hash<W>,
    adjoint: bool,
    rank: usize,
    circuit_index: usize,
) -> crate::channel::prepared::Prepared<W> {
    channel.prepare(hash, adjoint).unwrap_or_else(|| {
        let weight: u32 = channel.support().iter().map(|w| w.count_ones()).sum();
        panic!(
            "partition {rank}, layer {circuit_index}: channel preparation declined for a \
             channel of support {weight}: support above MAX_LOCAL_SUPPORT \
             ({MAX_LOCAL_SUPPORT}) qubits is unsupported except for PauliRotation, and a \
             channel must not write outside its declared support",
        )
    })
}

/// One partition's layer loop, run in lock-step with its peers inside its own pool (ARCHITECTURE.md §Partitioning).
pub(crate) fn run_layers<const W: usize, T, X, B>(
    circuit: &Circuit<W>,
    policy: &T,
    direction: Direction,
    options: PropagateOptions,
    context: PartitionContext<'_, W>,
    work: &mut PartitionWork<B>,
    transport: &X,
) where
    T: TruncationPolicy<W> + ?Sized,
    X: Transport,
    B: PartitionBackend<W, T>,
{
    let PartitionContext {
        rows,
        rank,
        size,
        tracing,
    } = context;
    let num_layers = circuit.channels.len();
    let adjoint = matches!(direction, Direction::Heisenberg);
    let group_size = transport.size();
    // `finalizes_layer` is a property of the policy type, so skipping the collective call is safe on every partition alike.
    let finalizes = policy.finalizes_layer();
    let policy_calls = AtomicU32::new(0);
    // At `P = 1` the local answer is the agreed one, so the loop rebuckets every layer exactly as an unpartitioned run would.
    let solo = group_size == 1;
    let local = &mut work.local;
    let mut plan = PartitionPlan::default();

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

        stamp!(stamp);
        tally!(local.stats().layers, 1);
        tally!(local.stats().terms_in, terms_before);

        // Alone, a backend that sizes its buckets without the prepared channel settles the count first, so a growing layer prepares once.
        let refined_early =
            solo && local.refine_unprepared(options.target_bucket_len, options.min_buckets);
        if refined_early {
            lap!(stamp, local.stats().rebucket_ns);
        }

        // Otherwise prepared before the bucket count is settled, since the plan decides whether this layer agrees it; only `bucket_delta` depends on the count, so a refining layer prepares again.
        let mut prepared = prepare_or_panic(channel, local.hash(), adjoint, rank, circuit_index);
        plan.rebuild(&prepared, rows, rank as u32);
        lap!(stamp, local.stats().prepare_ns);

        let mut collectives = 0u32;
        if !refined_early && (solo || plan.has_remote() || agrees_bucket_bits(application_index)) {
            let mut wanted_bits =
                local.proposed_bits(&prepared, options.target_bucket_len, options.min_buckets);
            if !solo {
                wanted_bits = transport.allreduce_max_u8(wanted_bits);
                collectives += 1;
            }
            lap!(stamp, local.stats().collective_ns);
            if wanted_bits > local.hash().bits() {
                while local.hash().bits() < wanted_bits {
                    local.refine();
                }
                lap!(stamp, local.stats().rebucket_ns);
                prepared = prepare_or_panic(channel, local.hash(), adjoint, rank, circuit_index);
                plan.rebuild(&prepared, rows, rank as u32);
                lap!(stamp, local.stats().prepare_ns);
            }
        }

        let counts = local.apply_layer(&prepared, &plan, rows, policy, transport);
        timed!(stamp.rearm());

        if finalizes {
            policy_calls.store(0, Ordering::Relaxed);
            let counting = CountingCollectives {
                inner: transport,
                calls: &policy_calls,
            };
            local.finalize_layer(policy, &counting);
            collectives += policy_calls.load(Ordering::Relaxed);
        }
        lap!(stamp, local.stats().finalize_ns);
        tally!(local.stats().terms_out, local.len());

        // The trace sits behind a hoisted flag and a `#[cold]` callee: this loop inlines the merge kernels, which are sensitive to code motion.
        let remote_deltas = counts.remote_deltas;
        let rows_received = counts.rows_received;
        let elapsed = layer_start.map(|start| start.elapsed());
        if tracing {
            record_layer_row(
                &mut work.rows,
                PartitionLayerRow {
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
                },
            );
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
}

/// Propagates `sum` through `circuit` on a partitioned engine built from `config`, and gathers the result.
///
/// A caller propagating repeatedly should hold a [`PartitionRuntime`](super::runtime::PartitionRuntime) and a [`PartitionedSum`] instead, so pools, split and scratch survive between calls.
///
/// # Errors
///
/// [`TopologyError`] if `config` cannot be resolved into slots or a pool cannot be built.
/// Everything else is a panic, as in [`propagate`](crate::propagate) and [`PartitionedSum::propagate_with`].
pub fn propagate_partitioned<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    config: &PartitionConfig,
    options: PropagateOptions,
) -> Result<PauliSum<W>, TopologyError>
where
    T: TruncationPolicy<W> + ?Sized,
{
    let mut split = PartitionedSum::scatter(sum, config)?;
    split.propagate_with(circuit, policy, direction, options);
    Ok(split.into_gathered())
}

#[cfg(test)]
mod tests;
