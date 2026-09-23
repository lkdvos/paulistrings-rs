//! The partitioned propagation driver: [`PartitionedSum`] and the [`propagate_partitioned`] front doors.
//!
//! A [`PartitionedSum`] is one [`PauliSum`] split across the partitions of a [`PartitionRuntime`]; the layer loop mirrors the unpartitioned one in [`engine`](crate::engine) — rebucket → prepare → layer → finalize, once per channel, on every partition in lock-step (ARCHITECTURE.md §Partitioning).
//!
//! Three things are collective and differ from the unpartitioned loop: the bucket count is agreed on a schedule rather than computed every layer (see [`BITS_AGREE_EVERY`]), a layer may exchange rows when [`PartitionPlan::has_remote`](super::plan::PartitionPlan::has_remote), and layer finalization goes through [`PartitionedTruncation::finalize_layer_partitioned`] rather than the unpartitioned policy directly.
//!
//! [`PropagateOptions`] is reused unchanged except that [`EngineSelection`](crate::EngineSelection) is ignored: the partitioned path is always the bucketed layer.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;

use super::backend::{HostPartition, PartitionBackend, PartitionStorage};
use super::plan::PartitionPlan;
use super::runtime::PartitionRuntime;
use super::topology::{PartitionConfig, TopologyError};
use super::trace::{assemble, record_layer_row, PartitionLayerRow, PartitionTrace};
use super::transport::{Collectives, Transport};
use super::truncation::PartitionedTruncation;
use crate::bucket::hash::{Gf2Hash, PartitionRows};
use crate::bucket::sum::{desired_bits, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::channel::prepared::MAX_LOCAL_SUPPORT;
use crate::channel::Channel;
use crate::circuit::Circuit;
#[cfg(feature = "phase-timing")]
use crate::engine::stats::{PhaseStats, Stamp};
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::{PauliSum, ProductState};

/// `log` target for the partitioned engine's progress events — the same target
/// the unpartitioned [`propagate`](crate::propagate) uses, so one filter
/// covers both.
const LOG_TARGET: &str = "paulistrings::propagate";

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
    fn barrier(&self) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.barrier()
    }
}

/// One partition's payload, moved into its thread for the duration of a call
/// and handed back.
pub(super) struct PartitionWork<B> {
    /// This partition's share of the sum and its retained scratch.
    pub(super) local: B,
    /// One row per layer, empty unless tracing is on. Written on this
    /// partition's driving thread only, and transposed into the shared
    /// [`PartitionTrace`] after the join.
    pub(super) rows: Vec<PartitionLayerRow>,
}

impl<B> PartitionWork<B> {
    /// Move one partition out of the driver for the duration of a call through [`PartitionStorage::detach`], which leaves a valid empty partition behind.
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

/// A [`PauliSum`] split across the partitions of a [`PartitionRuntime`] (ARCHITECTURE.md §Partitioning).
///
/// Held across calls: the split, the partition rows, the pools and the per-partition scratch all persist, so a driver stepping an observable through many Trotter steps scatters once and gathers once ([`PartitionedSum::propagate`] per step).
///
/// # Invariants
///
/// Between calls — and asserted by [`PartitionedSum::assert_invariants`] — every partition's sum satisfies [`PauliSum`]'s own invariants, holds only keys of its own partition, and shares one hash family and bucket count with its peers.
///
/// # Examples
///
/// ```
/// use paulistrings::channel::Clifford1Q;
/// use paulistrings::engine::partitioned::{
///     PartitionConfig, PartitionRuntime, PartitionedSum, Placement,
/// };
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
/// let config = PartitionConfig {
///     placement: Placement::Unpinned { partitions: 2, threads_per_partition: Some(1) },
///     bind_memory: false,
///     partition_row_seed: None,
/// };
/// let runtime = PartitionRuntime::new(&config).expect("topology resolves");
///
/// let mut acc = BuildAccumulator::<1>::new(2);
/// acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
///
/// let mut circuit = Circuit::<1>::new(2);
/// circuit.push(Clifford1Q::h(0));
///
/// let mut split = PartitionedSum::scatter(acc.finalize(), runtime, &config);
/// split.propagate(&circuit, &KeepAll, Direction::Heisenberg);
/// // H conjugates Z to X, wherever the term happened to live.
/// assert_eq!(split.gather().get(&[1], &[0]), Some(Complex64::new(1.0, 0.0)));
/// ```
pub struct PartitionedSum<const W: usize> {
    /// One sum and its layer/export scratch per partition, in rank order.
    parts: Vec<HostPartition<W>>,
    /// The rows that decide which partition a key belongs to.
    rows: PartitionRows<W>,
    /// The placement and pools this sum runs on.
    runtime: Arc<PartitionRuntime>,
    /// The opt-in per-layer trace, `None` unless
    /// [`enable_trace`](Self::enable_trace) was called.
    trace: Option<PartitionTrace>,
    /// Driver-level scatter time (the layer phases live in each partition's
    /// own `LayerScratch`).
    #[cfg(feature = "phase-timing")]
    scatter_ns: u64,
    /// Driver-level gather time. An atomic because [`gather`](Self::gather)
    /// takes `&self`; nothing contends for it.
    #[cfg(feature = "phase-timing")]
    gather_ns: std::sync::atomic::AtomicU64,
    /// Layers driven, summed over calls since the counters were drained.
    #[cfg(feature = "phase-timing")]
    layers: u64,
}

impl<const W: usize> PartitionedSum<W> {
    /// Splits `sum` across `runtime`'s partitions, deriving the partition rows from `config`.
    ///
    /// The rows come from [`PartitionRows::from_seed`] with `config.partition_row_seed`, falling back to the sum's own hash seed — a different draw from the bucket hash's, so the two are independent with high probability.
    ///
    /// Each partition filters its own share **on its own pool**, so the columns are first-touched in the domain that will read them.
    ///
    /// # Panics
    ///
    /// If `runtime`'s partition count and the derived rows disagree (they cannot, both being `log2(P)` rows), and in debug builds if the partition rows are not independent of the sum's hash rows.
    pub fn scatter(
        sum: PauliSum<W>,
        runtime: Arc<PartitionRuntime>,
        config: &PartitionConfig,
    ) -> Self {
        let seed = config
            .partition_row_seed
            .unwrap_or_else(|| sum.hash().seed());
        let rows = PartitionRows::<W>::from_seed(sum.num_qubits(), runtime.partition_bits(), seed);
        Self::scatter_with_rows(sum, rows, runtime)
    }

    /// Splits `sum` across `runtime`'s partitions using caller-supplied rows.
    ///
    /// # Bucket counts
    ///
    /// Each partition takes the bucket count its *own* share wants ([`desired_bits`] on the default bucket policy, all-reduced to a maximum so the group agrees), but sheds at most `log2(P)` bits of the count the unpartitioned sum arrived with — so the bucket count summed over partitions is the one the sum already had, and at `P = 1` the scatter changes nothing at all.
    /// The layer loop then re-normalizes upward against the caller's own [`PropagateOptions`].
    ///
    /// # Panics
    ///
    /// If `rows.num_partitions()` is not the runtime's partition count.
    /// In debug builds, if the rows are not independent of the sum's hash rows — a partition row inside the hash's row space correlates partition with bucket, which costs load balance (not correctness).
    /// The check is made here only: the hash gains rows as the sum grows, and independence from *future* rows cannot be checked up front; random rows stay independent with high probability.
    pub fn scatter_with_rows(
        sum: PauliSum<W>,
        rows: PartitionRows<W>,
        runtime: Arc<PartitionRuntime>,
    ) -> Self {
        let size = runtime.num_partitions();
        assert_eq!(
            rows.num_partitions(),
            size,
            "partition rows name {} partitions but the runtime has {size}",
            rows.num_partitions(),
        );
        assert_eq!(
            rows.num_qubits(),
            sum.num_qubits(),
            "partition rows are for {} qubits, the sum for {}",
            rows.num_qubits(),
            sum.num_qubits(),
        );
        debug_assert!(
            rows.is_independent_of(sum.hash()),
            "partition rows are dependent on the bucket hash rows — the split \
             will correlate with the bucket partition and load-balance badly",
        );

        let started = Instant::now();
        let locals = {
            let sum = &sum;
            let rows = &rows;
            runtime.map_partitions((0..size).collect(), |rank, _, transport| {
                scatter_local(sum, rows, rank as u32, transport)
            })
        };
        log::info!(
            target: LOG_TARGET,
            "scatter: {} terms over {size} partitions, {} bucket bits, {:.3} s",
            sum.len(),
            locals[0].hash().bits(),
            started.elapsed().as_secs_f64(),
        );

        Self {
            parts: locals.into_iter().map(HostPartition::new).collect(),
            rows,
            runtime,
            trace: None,
            #[cfg(feature = "phase-timing")]
            scatter_ns: started.elapsed().as_nanos() as u64,
            #[cfg(feature = "phase-timing")]
            gather_ns: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "phase-timing")]
            layers: 0,
        }
    }

    /// Propagates through `circuit` under `policy`, in place, with [`PropagateOptions::default()`].
    ///
    /// See [`propagate_with_options`](Self::propagate_with_options) for the non-default knobs and for what the loop does per layer.
    pub fn propagate<T>(&mut self, circuit: &Circuit<W>, policy: &T, direction: Direction)
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        self.propagate_with_options(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagates through `circuit` under `policy` with explicit [`PropagateOptions`].
    ///
    /// `direction` means what it means in [`propagate`](crate::propagate): [`Direction::Forward`] applies the channels in order, [`Direction::Heisenberg`] in reverse through [`Channel::apply_adjoint`](crate::Channel::apply_adjoint).
    /// [`EngineSelection`](crate::EngineSelection) is ignored — the partitioned path is always the bucketed layer (see the module docs).
    ///
    /// # Progress logging
    ///
    /// Target `paulistrings::propagate`, as in the unpartitioned engine: one `INFO` line on entry and exit, on the calling thread, and one `DEBUG` line per layer **per partition**, tagged `partition r/P`.
    /// Unlike [`propagate_with_scratch`](crate::propagate_with_scratch) the per-layer lines come from each partition's own driving thread between layers rather than the calling thread; every site is behind `log_enabled!`, so a disabled logger reads no clock.
    ///
    /// # Panics
    ///
    /// If a channel's [`Channel::prepare`] declines (support wider than `MAX_LOCAL_SUPPORT`), exactly as the unpartitioned engine does — there is no fallback path.
    /// If `policy` reports [`finalizes_layer`](crate::TruncationPolicy::finalizes_layer) without overriding [`finalize_layer_partitioned`](PartitionedTruncation::finalize_layer_partitioned).
    /// A panic in any partition is re-raised on the calling thread.
    pub fn propagate_with_options<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
    ) where
        T: PartitionedTruncation<W> + ?Sized,
    {
        let n = circuit.channels.len();
        let size = self.num_partitions();
        let terms_in = self.len();
        let started = Instant::now();
        log::info!(
            target: LOG_TARGET,
            "propagate_partitioned: {terms_in} terms through {n} channels ({direction:?}) \
             on {size} partitions [{}]",
            self.runtime.placement_summary(),
        );

        if n > 0 {
            // Hoisted out of every partition's layer loop: nothing inside one
            // can turn tracing on or off.
            let tracing = self.trace.is_some();
            let items: Vec<PartitionWork<HostPartition<W>>> = self
                .parts
                .iter_mut()
                .map(|part| PartitionWork::take(part, n, tracing))
                .collect();

            let runtime = Arc::clone(&self.runtime);
            let rows = &self.rows;
            let done = runtime.map_partitions(items, |rank, mut work, transport| {
                let ctx = PartitionCtx {
                    rows,
                    rank,
                    size,
                    tracing,
                };
                run_layers(
                    circuit, policy, direction, options, ctx, &mut work, transport,
                );
                work
            });
            let mut traced = Vec::with_capacity(if tracing { size } else { 0 });
            for (rank, work) in done.into_iter().enumerate() {
                self.parts[rank] = work.local;
                if tracing {
                    traced.push(work.rows);
                }
            }
            if let Some(trace) = self.trace.as_mut() {
                assemble(trace, traced);
            }
            #[cfg(feature = "phase-timing")]
            {
                self.layers += n as u64;
            }
        }

        log::info!(
            target: LOG_TARGET,
            "propagate_partitioned: {n} layers applied, {terms_in} -> {} terms, {:.3} s",
            self.len(),
            started.elapsed().as_secs_f64(),
        );
    }

    /// Merges the partitions back into one sum, leaving `self` intact.
    ///
    /// Runs on the calling thread; merging is `log2(P)` passes over the payload.
    /// A caller that only needs a scalar should prefer [`len`](Self::len) or [`expectation_product_state`](Self::expectation_product_state), which read the partitions in place.
    pub fn gather(&self) -> PauliSum<W> {
        #[cfg(feature = "phase-timing")]
        let started = Instant::now();
        let out = PauliSum::merge_partitions(self.parts.iter().map(|p| p.sum.clone()).collect());
        #[cfg(feature = "phase-timing")]
        self.gather_ns.fetch_add(
            started.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        out
    }

    /// [`gather`](Self::gather) by value — no clone of the parts.
    pub fn into_gathered(self) -> PauliSum<W> {
        PauliSum::merge_partitions(self.parts.into_iter().map(|p| p.sum).collect())
    }

    /// Terms in the whole sum, summed over partitions.
    pub fn len(&self) -> usize {
        self.parts.iter().map(|p| p.sum.len()).sum()
    }

    /// Whether every partition is empty.
    pub fn is_empty(&self) -> bool {
        self.parts.iter().all(|p| p.sum.is_empty())
    }

    /// Partitions this sum is split across.
    pub fn num_partitions(&self) -> usize {
        self.parts.len()
    }

    /// The bucket bits every partition currently holds (they are equal by
    /// construction).
    pub fn bits(&self) -> u8 {
        self.parts[0].sum.hash().bits()
    }

    /// Qubits the sum is over.
    pub fn num_qubits(&self) -> usize {
        self.parts[0].sum.num_qubits()
    }

    /// Partition `r`'s share of the sum.
    ///
    /// # Panics
    ///
    /// If `r` is not a partition of this sum.
    pub fn partition(&self, r: usize) -> &PauliSum<W> {
        &self.parts[r].sum
    }

    /// The rows deciding which partition a key belongs to.
    pub fn rows(&self) -> &PartitionRows<W> {
        &self.rows
    }

    /// The runtime this sum runs on, for handing to another
    /// [`PartitionedSum`].
    pub fn runtime(&self) -> &Arc<PartitionRuntime> {
        &self.runtime
    }

    /// `⟨ψ|O|ψ⟩` in a uniform single-qubit product state — the sum of the partitions' own expectation values, since the partitions hold disjoint terms.
    ///
    /// Partitions are combined in rank order.
    /// As with [`PauliSum::expectation_product_state`], floating-point addition is not associative, so this need not agree bit for bit with the gathered sum's answer.
    pub fn expectation_product_state(&self, state: ProductState) -> Complex64 {
        self.parts
            .iter()
            .map(|part| part.sum.expectation_product_state(state))
            .sum()
    }

    /// Start recording a [`PartitionTrace`] on every subsequent [`propagate`](Self::propagate) call.
    /// Idempotent, and it never discards records already taken.
    ///
    /// Always compiled, unlike the `phase-timing` counters: everything recorded is already computed by the layer, so a traced layer costs a `Vec` push on the partition's driving thread and an untraced one costs a register test.
    pub fn enable_trace(&mut self) {
        self.trace.get_or_insert_with(PartitionTrace::default);
    }

    /// Drain and return the per-layer records, or `None` if tracing was never enabled (`Some` iff tracing is on).
    ///
    /// Draining leaves tracing *enabled* with no records, so a sum reused across calls reports each call separately without re-enabling; records accumulate across layers and calls until drained.
    pub fn take_trace(&mut self) -> Option<PartitionTrace> {
        self.trace.as_mut().map(std::mem::take)
    }

    /// Drain and return the per-phase timing counters: one [`PhaseStats`] per partition, plus the driver's own scatter/gather time and layer count.
    ///
    /// Every counter is zeroed afterwards, so a probe can read one measured region at a time.
    /// `gather_ns` covers [`Self::gather`] calls only — [`Self::into_gathered`] consumes the sum, and with it the counters.
    #[cfg(feature = "phase-timing")]
    pub fn take_stats(&mut self) -> PartitionPhaseStats {
        PartitionPhaseStats {
            per_partition: self
                .parts
                .iter_mut()
                .map(|part| part.state.layer.take_stats())
                .collect(),
            scatter_ns: std::mem::take(&mut self.scatter_ns),
            gather_ns: self.gather_ns.swap(0, std::sync::atomic::Ordering::Relaxed),
            layers: std::mem::take(&mut self.layers),
        }
    }

    /// Debug helper: checks every invariant a partitioned sum is supposed to
    /// hold between calls.
    ///
    /// `O(terms)` — a full scan per partition — so it belongs in a test or an
    /// assertion, not in a loop. The per-partition structural check is
    /// `PauliSum`'s own `assert_invariants`, which exists only in test and
    /// debug builds; in a release build this checks the cross-partition
    /// properties alone.
    ///
    /// # Panics
    ///
    /// If a partition's sum is internally inconsistent, holds a key belonging
    /// to another partition, or disagrees with partition 0 about the hash rows,
    /// the bucket count or the qubit count.
    pub fn assert_invariants(&self) {
        assert_eq!(
            self.parts.len(),
            self.rows.num_partitions(),
            "partition count disagrees with the rows",
        );
        let head = &self.parts[0].sum;
        for (rank, local) in self.parts.iter().map(|p| &p.sum).enumerate() {
            // `PauliSum::assert_invariants` itself exists only in test and
            // debug builds; the cross-partition checks below are always
            // compiled, so this method's signature does not change with the
            // profile.
            #[cfg(any(test, debug_assertions))]
            local.assert_invariants();
            assert_eq!(
                local.num_qubits(),
                head.num_qubits(),
                "partition {rank}: qubit count differs from partition 0",
            );
            assert!(
                local.hash().same_rows_as(head.hash()),
                "partition {rank}: hash rows differ from partition 0",
            );
            assert_eq!(
                local.hash().bits(),
                head.hash().bits(),
                "partition {rank}: bucket bits differ from partition 0",
            );
            let held = local.partition_rank_of_all(&self.rows);
            assert!(
                held.is_none() || held == Some(rank as u32),
                "partition {rank} holds keys of partition {held:?}",
            );
        }
    }
}

/// The partitioned engine's phase breakdown: one [`PhaseStats`] per partition plus the driver's own counters.
///
/// Drained by [`PartitionedSum::take_stats`].
/// Each partition's `PhaseStats` was measured on that partition's own driving thread and its own pool, so the wall-clock fields are **concurrent**, not additive: the partitions ran at the same time, and the spread between them is the group's load imbalance.
/// `exchange_ns` in particular absorbs the wait for a partner, so a partition that finishes its own share early pays for the imbalance there.
#[cfg(feature = "phase-timing")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartitionPhaseStats {
    /// One breakdown per partition, in rank order.
    pub per_partition: Vec<PhaseStats>,
    /// Wall time of the scatter that built this sum (filter + coarsen, on the
    /// partitions' own pools).
    pub scatter_ns: u64,
    /// Wall time summed over [`PartitionedSum::gather`] calls.
    pub gather_ns: u64,
    /// Layers driven since the counters were last drained.
    pub layers: u64,
}

/// One partition's share of `sum`, at the bucket count the group agrees on — **the** scatter body, shared by the in-process and distributed drivers.
///
/// Runs on the partition's own pool (the caller is inside `install`), so every column is first-touched in the domain that will read it.
/// One collective: the per-partition [`desired_bits`] maximum, which is what makes the group agree on a count before the first layer.
pub(super) fn scatter_local<const W: usize>(
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
pub(super) struct PartitionCtx<'a, const W: usize> {
    /// The rows deciding which partition a key belongs to.
    pub(super) rows: &'a PartitionRows<W>,
    /// This partition's index, for the per-layer log line.
    pub(super) rank: usize,
    /// Partitions in the group, likewise.
    pub(super) size: usize,
    /// Whether to append a [`PartitionLayerRow`] per layer. Hoisted out of the
    /// loop: nothing inside one can turn tracing on or off.
    pub(super) tracing: bool,
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
pub(super) fn run_layers<const W: usize, T, X, B>(
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
                desired_bits(local.len(), options.target_bucket_len, options.min_buckets)
                    .max(local.hash().bits());
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
