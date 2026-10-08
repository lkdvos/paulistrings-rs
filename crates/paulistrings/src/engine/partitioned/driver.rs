//! The partitioned propagation driver: [`PartitionedSum`] and the [`propagate_partitioned`] front doors.
//!
//! A [`PartitionedSum`] is one [`PauliSum`] split across the partitions of a [`PartitionRuntime`]; the layer loop mirrors the unpartitioned one in [`engine`](crate::engine) — rebucket → prepare → layer → finalize, once per channel, on every partition in lock-step (ARCHITECTURE.md §Partitioning).
//!
//! Three things are collective and differ from the unpartitioned loop: the bucket count is agreed on a schedule rather than computed every layer (see [`BITS_AGREE_EVERY`]), a layer may exchange rows when [`PartitionPlan::has_remote`](super::plan::PartitionPlan::has_remote), and layer finalization goes through [`PartitionedTruncation::finalize_layer_partitioned`] rather than the unpartitioned policy directly.
//!
//! [`PropagateOptions`] is reused unchanged except that [`EngineSelection`](crate::EngineSelection) is ignored: the partitioned path is always the bucketed layer.

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

/// `log` target for the partitioned engine's progress events — the same target
/// the unpartitioned [`propagate`](crate::propagate) uses, so one filter
/// covers both.
pub(super) const LOG_TARGET: &str = "paulistrings::propagate";

/// Layers between two bucket-count agreements, and the length of the opening ramp that precedes them (ARCHITECTURE.md §Partitioning).
///
/// A partitioned layer that exchanges nothing has no other reason to communicate, so the bucket-count all-reduce is the whole cost of a local layer; agreeing every `BITS_AGREE_EVERY`-th layer amortizes that.
///
/// Two regimes justify the value. In steady state under a truncation policy the term count moves by a few percent per layer, so a lag of 16 layers is far short of the factor of two that would cost a bucket bit at all.
/// A run starting from a small operator can double every layer for a while, so the first `BITS_AGREE_EVERY` layers of every call agree unconditionally rather than run the growth phase under-bucketed.
///
/// Between agreements a partition keeps the bucket count it has even if its own [`desired_bits`] is higher: nobody refines off-schedule, so the group's counts stay equal by construction and an exchange can always index a partner's blocks.
/// The lag is bounded by `BITS_AGREE_EVERY` layers.
pub const BITS_AGREE_EVERY: usize = 16;

/// Whether layer `k` of a call agrees the bucket count with the group.
///
/// A pure function of the layer index — the same answer on every partition, with nothing exchanged to reach it.
/// See [`BITS_AGREE_EVERY`] for the two terms.
#[inline]
fn agrees_bucket_bits(k: usize) -> bool {
    k < BITS_AGREE_EVERY || k.is_multiple_of(BITS_AGREE_EVERY)
}

/// A [`Collectives`] view that counts the calls made through it.
///
/// Wrapped around the transport for the *policy's* collective finalization only, so the trace's `collectives` figure counts what a composite policy does inside `finalize_layer_partitioned` rather than guessing.
/// The layer's own exchange never goes through here — it is point-to-point — so the extra indirection is one virtual call per collective.
struct CountingCollectives<'a> {
    inner: &'a dyn Collectives,
    calls: &'a AtomicU32,
}

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
    fn allreduce_sum_u64(&self, buf: &mut [u64]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.allreduce_sum_u64(buf)
    }
    fn allreduce_sum_f64(&self, buf: &mut [f64]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.allreduce_sum_f64(buf)
    }
    fn barrier(&self) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.barrier()
    }
}

/// One partition's payload, moved into its thread for the duration of a call
/// and handed back.
pub(crate) struct PartitionWork<B> {
    /// This partition's share of the sum and its retained scratch.
    pub(crate) local: B,
    /// One row per layer, empty unless tracing is on. Written on this
    /// partition's driving thread only, and transposed into the shared
    /// [`PartitionTrace`] after the join.
    pub(crate) rows: Vec<PartitionLayerRow>,
}

impl<B> PartitionWork<B> {
    /// Move one partition out of the driver for the duration of a call through [`PartitionStorage::detach`], which leaves a valid empty partition behind.
    pub(crate) fn take<const W: usize>(local: &mut B, layers: usize, tracing: bool) -> Self
    where
        B: PartitionStorage<W>,
    {
        Self {
            local: local.detach(),
            rows: Vec::with_capacity(if tracing { layers } else { 0 }),
        }
    }
}

/// One partition's share of `sum`, at the bucket count the group agrees on — **the** scatter body, shared by the in-process and distributed drivers.
///
/// Runs on the partition's own pool (the caller is inside `install`), so every column is first-touched in the domain that will read it.
/// One collective: the per-partition [`desired_bits`] maximum, which is what makes the group agree on a count before the first layer.
pub(crate) fn scatter_local<const W: usize>(
    sum: &PauliSum<W>,
    rows: &PartitionRows<W>,
    rank: u32,
    coll: &dyn Collectives,
) -> PauliSum<W> {
    let mut local = sum.filter_partition(rows, rank);
    let want = desired_bits(local.len(), DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS);
    let want = coll.allreduce_max_u8(want);
    local.coarsen_to(scatter_bits(local.hash().bits(), rows.bits(), want));
    local
}

/// The bucket bits a partition takes on at scatter: at most `pbits` shed from the count the unpartitioned sum arrived with, and never below what this partition's own share wants.
///
/// `bits` is the incoming count, `pbits = log2(P)`, `want` the all-reduced per-partition [`desired_bits`].
/// Shedding exactly `pbits` keeps the bucket count *summed over partitions* equal to the unpartitioned one; the `want` floor stops a sum that arrived under-bucketed from being coarsened at all.
/// At `P = 1`, `pbits = 0`, so this is `bits` — the scatter is the identity and the partitioned run is the unpartitioned one.
fn scatter_bits(bits: u8, pbits: u8, want: u8) -> u8 {
    want.max(bits.saturating_sub(pbits)).min(bits)
}

/// What a partition knows about itself while it walks the layers: which keys are its own, where it sits in the group, and whether it is recording.
///
/// The two drivers fill this differently — `rank` and `size` come from `map_partitions` in one and from the transport in the other — and nothing in the loop below cares which.
pub(crate) struct PartitionCtx<'a, const W: usize> {
    /// The rows deciding which partition a key belongs to.
    pub(crate) rows: &'a PartitionRows<W>,
    /// This partition's index, for the per-layer log line.
    pub(crate) rank: usize,
    /// Partitions in the group, likewise.
    pub(crate) size: usize,
    /// Whether to append a [`PartitionLayerRow`] per layer. Hoisted out of the
    /// loop: nothing inside one can turn tracing on or off.
    pub(crate) tracing: bool,
}

/// [`Channel::prepare`] or the engine's one hard error.
///
/// Called twice on a layer that refines (the bucket count is settled between the two), so the panic — which is the unpartitioned engine's, with the partition and layer named — lives here rather than inline.
fn prepare_or_panic<const W: usize>(
    ch: &dyn Channel<W>,
    hash: &Gf2Hash<W>,
    adjoint: bool,
    rank: usize,
    idx: usize,
) -> crate::channel::prepared::Prepared<W> {
    ch.prepare(hash, adjoint).unwrap_or_else(|| {
        // Same hard error as the unpartitioned engine: no whole-sum fallback
        // exists to absorb a channel the engine cannot tabulate.
        let weight: u32 = ch.support().iter().map(|w| w.count_ones()).sum();
        panic!(
            "partition {rank}, layer {idx}: Channel::prepare declined, so this channel \
             cannot be propagated. The engine tabulates channels of support ≤ \
             {MAX_LOCAL_SUPPORT} qubits (this one declares {weight}), and a channel must \
             not write outside its declared support. See \
             research/FINDINGS.md",
        )
    })
}

/// One partition's whole layer loop — **the** layer loop, shared by the in-process driver ([`PartitionedSum`]) and the distributed one ([`DistributedSum`](super::DistributedSum)).
///
/// Runs on the partition's driving thread inside its own pool, in lock-step with its peers: the same channels in the same order, the same collectives per layer.
/// The two drivers differ only in *what a partition is* — a NUMA domain and an in-process endpoint, or a whole process and an MPI rank — which is entirely the transport's business, so the body below is generic over it and there is exactly one copy of the per-layer sequence.
/// Every touch of the partition's storage goes through [`PartitionStorage`] and [`PartitionBackend`], so the loop is generic over where the partition lives as well.
pub(crate) fn run_layers<const W: usize, T, X, B>(
    circuit: &Circuit<W>,
    policy: &T,
    direction: Direction,
    options: PropagateOptions,
    ctx: PartitionCtx<'_, W>,
    work: &mut PartitionWork<B>,
    transport: &X,
) where
    T: PartitionedTruncation<W> + ?Sized,
    X: Transport,
    B: PartitionBackend<W, T>,
{
    let PartitionCtx {
        rows,
        rank,
        size,
        tracing,
    } = ctx;
    let n = circuit.channels.len();
    let adjoint = matches!(direction, Direction::Heisenberg);
    let size32 = transport.size();
    // Both hoisted out of the loop: neither can change inside one, and `finalizes_layer` is a property of the policy *type*, hence the same answer on every partition — which is what makes skipping the call collective-safe.
    let finalizes = policy.finalizes_layer();
    let policy_calls = AtomicU32::new(0);
    let local = &mut work.local;

    for k in 0..n {
        let idx = match direction {
            Direction::Forward => k,
            Direction::Heisenberg => n - 1 - k,
        };
        let ch: &dyn Channel<W> = circuit.channels[idx].as_ref();

        // As in the unpartitioned engine: the per-layer DEBUG log and the opt-in gate trace share one clock read.
        let debug_on = log::log_enabled!(target: LOG_TARGET, log::Level::Debug);
        let want_timer = tracing || debug_on;
        let layer_t0 = want_timer.then(Instant::now);
        let terms_before = local.len();

        #[cfg(feature = "phase-timing")]
        let mut st = Stamp::now();
        #[cfg(feature = "phase-timing")]
        {
            let stats = local.stats();
            stats.layers += 1;
            stats.terms_in += terms_before as u64;
        }

        // Prepared *before* the bucket count is settled, because the plan is what says whether this layer needs a collective at all: a delta's remoteness is a property of its mask and the partition rows, neither of which the bucket count touches.
        // Only `bucket_delta` depends on it, so the prepared form is re-derived below on the rare layer that actually refines.
        let mut prep = prepare_or_panic(ch, local.hash(), adjoint, rank, idx);
        let mut plan = PartitionPlan::new(&prep, rows, rank as u32);
        #[cfg(feature = "phase-timing")]
        st.lap(&mut local.stats().prepare_ns);

        // The bucket count, on the BITS_AGREE_EVERY schedule: on any layer that exchanges (both sides index the blocks by it), and otherwise every `BITS_AGREE_EVERY`-th.
        // At `P = 1` there is no group, so the local answer is the agreed one and the loop rebuckets every layer exactly as `propagate` does.
        let mut collectives = 0u32;
        let solo = size32 == 1;
        if solo || plan.has_remote() || agrees_bucket_bits(k) {
            let mut want =
                local.proposed_bits(&prep, options.target_bucket_len, options.min_buckets);
            if !solo {
                want = transport.allreduce_max_u8(want);
                collectives += 1;
            }
            #[cfg(feature = "phase-timing")]
            st.lap(&mut local.stats().collective_ns);
            if want > local.hash().bits() {
                while local.hash().bits() < want {
                    local.refine();
                }
                #[cfg(feature = "phase-timing")]
                st.lap(&mut local.stats().rebucket_ns);
                // The hash moved, so every `bucket_delta` in the prepared form did too.
                // Rare — the count is grow-only and settles — and this is the only reason a layer prepares twice.
                prep = prepare_or_panic(ch, local.hash(), adjoint, rank, idx);
                plan = PartitionPlan::new(&prep, rows, rank as u32);
                #[cfg(feature = "phase-timing")]
                st.lap(&mut local.stats().prepare_ns);
            }
        }

        // The layer times its own export, exchange and coset loop into the same `LayerScratch`.
        let counts = local.apply_layer(&prep, &plan, rows, policy, transport);
        #[cfg(feature = "phase-timing")]
        st.rearm();

        // Collective, so it runs on every partition or on none — and `finalizes_layer` is the same answer on all of them (see `PartitionedTruncation`).
        // A policy with no layer pass costs nothing.
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
            st.lap(&mut local.stats().finalize_ns);
            let terms_out = local.len() as u64;
            local.stats().terms_out += terms_out;
        }

        // Opt-in trace, behind a hoisted flag and a `#[cold]` callee for the same reason as the unpartitioned engine's term trace: this loop inlines the bucketed layer, whose merge kernels are sensitive to code motion (CLAUDE.md §Performance discipline).
        // The counts are moved, not copied — the exchange already allocated them.
        let remote_deltas = counts.remote_deltas;
        let rows_received = counts.rows_received;
        let dt = layer_t0.map(|t0| t0.elapsed());
        if tracing {
            record_layer_row(
                &mut work.rows,
                local.hash().bits(),
                collectives,
                idx as u32,
                k as u32,
                ch.debug_name(),
                terms_before,
                local.len(),
                counts,
                dt.unwrap_or_default().as_nanos() as u64,
            );
        }

        if debug_on {
            if let Some(dt) = dt {
                log::debug!(
                    target: LOG_TARGET,
                    "partition {}/{} layer {}/{} [{}]: {} -> {} terms, {} remote deltas, \
                     {} rows in, {:.1} ms",
                    rank,
                    size,
                    k + 1,
                    n,
                    ch.debug_name(),
                    terms_before,
                    local.len(),
                    remote_deltas,
                    rows_received,
                    dt.as_secs_f64() * 1e3,
                );
            }
        }
    }
}

/// Propagates `sum` through `circuit` on a partitioned engine built from `config`, and gathers the result.
///
/// One-shot convenience: it builds a [`PartitionRuntime`], scatters, runs and gathers.
/// A caller propagating repeatedly (a Trotter driver stepping an observable) should hold the runtime and a [`PartitionedSum`] instead, so the pools, the split and the scratch survive between calls.
///
/// # Errors
///
/// [`TopologyError`] if `config` cannot be resolved into slots or a pool cannot be built.
/// Everything else is a panic, as in [`propagate`](crate::propagate).
///
/// # Examples
///
/// ```
/// use paulistrings::channel::Clifford1Q;
/// use paulistrings::engine::partitioned::{propagate_partitioned, PartitionConfig, Placement};
/// use paulistrings::{
///     BuildAccumulator, Circuit, Direction, PartitionedTruncation, PauliString, Phase,
///     TruncationPolicy,
/// };
/// use num_complex::Complex64;
///
/// struct KeepAll;
/// impl<const W: usize> TruncationPolicy<W> for KeepAll {
///     // `finalizes_layer`'s default is the conservative `true`; the
///     // `PartitionedTruncation` default body rejects that, since it cannot
///     // know a layer pass it would be skipping is a no-op.
///     fn finalizes_layer(&self) -> bool { false }
/// }
/// impl<const W: usize> PartitionedTruncation<W> for KeepAll {}
///
/// let mut acc = BuildAccumulator::<1>::new(2);
/// acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
/// let mut circuit = Circuit::<1>::new(2);
/// circuit.push(Clifford1Q::h(0));
///
/// let config = PartitionConfig {
///     placement: Placement::Unpinned { partitions: 2, threads_per_partition: Some(1) },
///     bind_memory: false,
///     partition_row_seed: None,
/// };
/// let out = propagate_partitioned(
///     &circuit, acc.finalize(), &KeepAll, Direction::Heisenberg, &config,
/// ).expect("topology resolves");
/// assert_eq!(out.get(&[1], &[0]), Some(Complex64::new(1.0, 0.0)));
/// ```
pub fn propagate_partitioned<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    config: &PartitionConfig,
) -> Result<PauliSum<W>, TopologyError>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    propagate_partitioned_with_options(
        circuit,
        sum,
        policy,
        direction,
        config,
        PropagateOptions::default(),
    )
}

/// [`propagate_partitioned`] with explicit [`PropagateOptions`].
///
/// [`EngineSelection`](crate::EngineSelection) is ignored; see
/// [`PartitionedSum::propagate_with_options`].
///
/// # Errors
///
/// [`TopologyError`], as [`propagate_partitioned`].
pub fn propagate_partitioned_with_options<const W: usize, T>(
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
    split.propagate_with_options(circuit, policy, direction, options);
    Ok(split.into_gathered())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `P = 1` sheds nothing, so a scatter cannot change the bucket count and the partitioned run starts exactly where `propagate` would.
    #[test]
    fn scatter_bits_is_the_identity_at_one_partition() {
        for bits in 0u8..12 {
            for want in 0u8..12 {
                assert_eq!(scatter_bits(bits, 0, want), bits, "bits={bits} want={want}");
            }
        }
    }

    /// With `P` partitions the count sheds `log2(P)` bits, so the bucket count summed over partitions is the unpartitioned one — unless a partition's own share wants more.
    #[test]
    fn scatter_bits_sheds_at_most_log2_p() {
        // Incoming 10 bits, 4 partitions each wanting 8: shed exactly 2.
        assert_eq!(scatter_bits(10, 2, 8), 8);
        // Wanting more than the split leaves: the want wins, capped by what is there (the layer loop grows past it).
        assert_eq!(scatter_bits(10, 2, 9), 9);
        assert_eq!(scatter_bits(10, 2, 12), 10);
        // Wanting less than the split leaves: never shed more than log2(P).
        assert_eq!(scatter_bits(10, 2, 3), 8);
        // Fewer incoming bits than there are partitions: the floor saturates at a single bucket per partition, so the per-share want decides.
        assert_eq!(scatter_bits(1, 2, 0), 0);
        assert_eq!(scatter_bits(1, 2, 1), 1);
        assert_eq!(scatter_bits(0, 4, 0), 0);
    }
}
