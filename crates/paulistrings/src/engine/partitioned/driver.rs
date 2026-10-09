//! The partitioned layer loop `run_layers`, shared by every partitioned driver, and the [`propagate_partitioned`] front door (ARCHITECTURE.md §Partitioning).

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use super::backend::{PartitionBackend, PartitionStorage};
use super::plan::PartitionPlan;
use super::runtime::PartitionRuntime;
use super::topology::{PartitionConfig, TopologyError};
use super::trace::{record_layer_row, PartitionLayerRow};
use super::transport::{Collectives, Transport};
use super::truncation::PartitionedTruncation;
use crate::channel::prepared::MAX_LOCAL_SUPPORT;
use crate::channel::Channel;
use crate::circuit::Circuit;
#[cfg(feature = "phase-timing")]
use crate::engine::stats::Stamp;
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows};
use crate::pauli_sum::storage::{desired_bits, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::pauli_sum::PauliSum;

use super::sum::PartitionedSum;

/// The unpartitioned engine's `log` target, so one filter covers both.
pub(super) const LOG_TARGET: &str = "paulistrings::propagate";

/// Layers between two bucket-count agreements, and the length of the opening ramp that precedes them (ARCHITECTURE.md §Partitioning).
pub const BITS_AGREE_EVERY: usize = 16;

/// Whether layer `k` of a call agrees the bucket count; a pure function of `k`, so every partition answers alike.
#[inline]
fn agrees_bucket_bits(k: usize) -> bool {
    k < BITS_AGREE_EVERY || k.is_multiple_of(BITS_AGREE_EVERY)
}

/// A [`Collectives`] view that counts the policy finalization's calls for the trace.
struct CountingCollectives<'a> {
    inner: &'a dyn Collectives,
    calls: &'a AtomicU32,
}

impl super::transport::sealed::Sealed for CountingCollectives<'_> {}

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
pub(super) struct PartitionWork<B> {
    pub(super) local: B,
    /// One row per layer, empty unless tracing is on.
    pub(super) rows: Vec<PartitionLayerRow>,
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
pub(super) struct PartitionContext<'a, const W: usize> {
    pub(super) rows: &'a PartitionRows<W>,
    pub(super) rank: usize,
    pub(super) size: usize,
    pub(super) tracing: bool,
}

/// [`Channel::prepare`], or the unpartitioned engine's hard error naming the partition and layer.
fn prepare_or_panic<const W: usize>(
    channel: &dyn Channel<W>,
    hash: &Gf2Hash<W>,
    adjoint: bool,
    rank: usize,
    index: usize,
) -> crate::channel::prepared::Prepared<W> {
    channel.prepare(hash, adjoint).unwrap_or_else(|| {
        let weight: u32 = channel.support().iter().map(|w| w.count_ones()).sum();
        panic!(
            "partition {rank}, layer {index}: Channel::prepare declined, so this channel \
             cannot be propagated. The engine tabulates channels of support ≤ \
             {MAX_LOCAL_SUPPORT} qubits (this one declares {weight}), and a channel must \
             not write outside its declared support. See \
             research/FINDINGS.md",
        )
    })
}

/// One partition's layer loop, run in lock-step with its peers inside its own pool (ARCHITECTURE.md §Partitioning).
pub(super) fn run_layers<const W: usize, T, X, B>(
    circuit: &Circuit<W>,
    policy: &T,
    direction: Direction,
    options: PropagateOptions,
    context: PartitionContext<'_, W>,
    work: &mut PartitionWork<B>,
    transport: &X,
) where
    T: PartitionedTruncation<W> + ?Sized,
    X: Transport,
    B: PartitionBackend<W, T>,
{
    let PartitionContext {
        rows,
        rank,
        size,
        tracing,
    } = context;
    let n = circuit.channels.len();
    let adjoint = matches!(direction, Direction::Heisenberg);
    let group_size = transport.size();
    // `finalizes_layer` is a property of the policy type, so skipping the collective call is safe on every partition alike.
    let finalizes = policy.finalizes_layer();
    let policy_calls = AtomicU32::new(0);
    let local = &mut work.local;

    for k in 0..n {
        let index = match direction {
            Direction::Forward => k,
            Direction::Heisenberg => n - 1 - k,
        };
        let channel: &dyn Channel<W> = circuit.channels[index].as_ref();

        let debug_on = log::log_enabled!(target: LOG_TARGET, log::Level::Debug);
        let want_timer = tracing || debug_on;
        let layer_start = want_timer.then(Instant::now);
        let terms_before = local.len();

        #[cfg(feature = "phase-timing")]
        let mut stamp = Stamp::now();
        #[cfg(feature = "phase-timing")]
        {
            let stats = local.stats();
            stats.layers += 1;
            stats.terms_in += terms_before as u64;
        }

        // Prepared before the bucket count is settled, since the plan decides whether this layer agrees it; only `bucket_delta` depends on the count, so a refining layer prepares again.
        let mut prepared = prepare_or_panic(channel, local.hash(), adjoint, rank, index);
        let mut plan = PartitionPlan::new(&prepared, rows, rank as u32);
        #[cfg(feature = "phase-timing")]
        stamp.lap(&mut local.stats().prepare_ns);

        let mut collectives = 0u32;
        // At `P = 1` the local answer is the agreed one, so the loop rebuckets every layer exactly as `propagate` does.
        let solo = group_size == 1;
        if solo || plan.has_remote() || agrees_bucket_bits(k) {
            let mut want =
                local.proposed_bits(&prepared, options.target_bucket_len, options.min_buckets);
            if !solo {
                want = transport.allreduce_max_u8(want);
                collectives += 1;
            }
            #[cfg(feature = "phase-timing")]
            stamp.lap(&mut local.stats().collective_ns);
            if want > local.hash().bits() {
                while local.hash().bits() < want {
                    local.refine();
                }
                #[cfg(feature = "phase-timing")]
                stamp.lap(&mut local.stats().rebucket_ns);
                prepared = prepare_or_panic(channel, local.hash(), adjoint, rank, index);
                plan = PartitionPlan::new(&prepared, rows, rank as u32);
                #[cfg(feature = "phase-timing")]
                stamp.lap(&mut local.stats().prepare_ns);
            }
        }

        let counts = local.apply_layer(&prepared, &plan, rows, policy, transport);
        #[cfg(feature = "phase-timing")]
        stamp.rearm();

        if finalizes {
            policy_calls.store(0, Ordering::Relaxed);
            let counting = CountingCollectives {
                inner: transport,
                calls: &policy_calls,
            };
            local.finalize_layer(policy, &counting);
            collectives += policy_calls.load(Ordering::Relaxed);
        }
        #[cfg(feature = "phase-timing")]
        {
            stamp.lap(&mut local.stats().finalize_ns);
            let terms_out = local.len() as u64;
            local.stats().terms_out += terms_out;
        }

        // The trace sits behind a hoisted flag and a `#[cold]` callee: this loop inlines the merge kernels, which are sensitive to code motion.
        let remote_deltas = counts.remote_deltas;
        let rows_received = counts.rows_received;
        let elapsed = layer_start.map(|start| start.elapsed());
        if tracing {
            record_layer_row(
                &mut work.rows,
                local.hash().bits(),
                collectives,
                index as u32,
                k as u32,
                channel.debug_name(),
                terms_before,
                local.len(),
                counts,
                elapsed.unwrap_or_default().as_nanos() as u64,
            );
        }

        if debug_on {
            if let Some(elapsed) = elapsed {
                log::debug!(
                    target: LOG_TARGET,
                    "partition {}/{} layer {}/{} [{}]: {} -> {} terms, {} remote deltas, \
                     {} rows in, {:.1} ms",
                    rank,
                    size,
                    k + 1,
                    n,
                    channel.debug_name(),
                    terms_before,
                    local.len(),
                    remote_deltas,
                    rows_received,
                    elapsed.as_secs_f64() * 1e3,
                );
            }
        }
    }
}

/// Propagates `sum` through `circuit` on a partitioned engine built from `config`, and gathers the result.
///
/// A caller propagating repeatedly should hold a [`PartitionRuntime`] and a [`PartitionedSum`] instead, so pools, split and scratch survive between calls.
/// [`EngineSelection`](crate::EngineSelection) in `options` is ignored.
///
/// # Errors
///
/// [`TopologyError`] if `config` cannot be resolved into slots or a pool cannot be built.
/// Everything else is a panic, as in [`propagate`](crate::propagate).
pub fn propagate_partitioned<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    config: &PartitionConfig,
    options: PropagateOptions,
) -> Result<PauliSum<W>, TopologyError>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    let runtime = PartitionRuntime::new(config)?;
    let mut split = PartitionedSum::scatter(sum, runtime, config);
    split.propagate_with(circuit, policy, direction, options);
    Ok(split.into_gathered())
}

#[cfg(test)]
mod tests;
