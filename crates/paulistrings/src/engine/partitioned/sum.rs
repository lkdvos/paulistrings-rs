//! [`PartitionedSum`], one [`PauliSum`] split across the partitions of a [`PartitionRuntime`].

use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;

use super::backend::{HostPartition, PartitionBackend, PartitionStorage};
use super::driver::{run_layers, scatter_local, PartitionCtx, PartitionWork, LOG_TARGET};
use super::runtime::PartitionRuntime;
use super::topology::PartitionConfig;
use super::trace::{assemble, PartitionTrace};
use super::truncation::PartitionedTruncation;
use crate::circuit::Circuit;
#[cfg(feature = "phase-timing")]
use crate::engine::stats::PhaseStats;
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::hash::PartitionRows;
use crate::pauli_sum::PauliSum;
use crate::readout::echo::RotationAxis;
use crate::readout::product_state::ProductState;

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
/// use paulistrings::Clifford1Q;
/// use paulistrings::{
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
///
/// `B` is where a partition lives: host memory by default, a CUDA device under the `cuda` feature's `gpu::GpuPartitionedSum`, which is this type over the device backend.
pub struct PartitionedSum<const W: usize, B = HostPartition<W>> {
    /// One partition and its layer/export scratch, in rank order.
    parts: Vec<B>,
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

impl<const W: usize, B> PartitionedSum<W, B> {
    /// A split around partitions another backend already scattered, with `scatter_ns` its scatter time.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_backend(
        parts: Vec<B>,
        rows: PartitionRows<W>,
        runtime: Arc<PartitionRuntime>,
        scatter_ns: u64,
    ) -> Self {
        #[cfg(not(feature = "phase-timing"))]
        let _ = scatter_ns;
        Self {
            parts,
            rows,
            runtime,
            trace: None,
            #[cfg(feature = "phase-timing")]
            scatter_ns,
            #[cfg(feature = "phase-timing")]
            gather_ns: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "phase-timing")]
            layers: 0,
        }
    }

    /// The partitions, in rank order.
    #[cfg(feature = "cuda")]
    pub(crate) fn backends(&self) -> &[B] {
        &self.parts
    }

    /// The partitions, mutably.
    #[cfg(feature = "cuda")]
    pub(crate) fn backends_mut(&mut self) -> &mut [B] {
        &mut self.parts
    }

    /// Drain the driver's own laps: scatter time, gather time and layers driven.
    #[cfg(all(feature = "cuda", feature = "phase-timing"))]
    pub(crate) fn take_driver_laps(&mut self) -> (u64, u64, u64) {
        (
            std::mem::take(&mut self.scatter_ns),
            self.gather_ns.swap(0, std::sync::atomic::Ordering::Relaxed),
            std::mem::take(&mut self.layers),
        )
    }

    /// Add one gather's wall time to the drained laps.
    #[cfg(all(feature = "cuda", feature = "phase-timing"))]
    pub(crate) fn lap_gather(&self, ns: u64) {
        self.gather_ns
            .fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    }

    /// Partitions this sum is split across.
    pub fn num_partitions(&self) -> usize {
        self.parts.len()
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

    /// Start recording a [`PartitionTrace`] on every subsequent propagation.
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

    /// The host driver's `propagate_with` on any backend.
    pub(crate) fn propagate_on_backend<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
    ) where
        T: PartitionedTruncation<W> + ?Sized,
        B: PartitionBackend<W, T>,
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
            let items: Vec<PartitionWork<B>> = self
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
}

impl<const W: usize, B: PartitionStorage<W>> PartitionedSum<W, B> {
    /// Terms in the whole sum, summed over partitions.
    pub fn len(&self) -> usize {
        self.parts.iter().map(|p| p.len()).sum()
    }

    /// Whether every partition is empty.
    pub fn is_empty(&self) -> bool {
        self.parts.iter().all(|p| p.len() == 0)
    }

    /// The bucket bits every partition currently holds (they are equal by
    /// construction).
    pub fn bits(&self) -> u8 {
        self.parts[0].hash().bits()
    }

    /// Qubits the sum is over.
    pub fn num_qubits(&self) -> usize {
        self.rows.num_qubits()
    }
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
    /// Each partition takes the bucket count its *own* share wants (`desired_bits` on the default bucket policy, all-reduced to a maximum so the group agrees), but sheds at most `log2(P)` bits of the count the unpartitioned sum arrived with — so the bucket count summed over partitions is the one the sum already had, and at `P = 1` the scatter changes nothing at all.
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
        rows.assert_splits(sum.hash(), sum.num_qubits(), size);

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
    /// See [`propagate_with`](Self::propagate_with) for the non-default knobs and for what the loop does per layer.
    pub fn propagate<T>(&mut self, circuit: &Circuit<W>, policy: &T, direction: Direction)
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        self.propagate_with(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagates through `circuit` under `policy` with explicit [`PropagateOptions`].
    ///
    /// `direction` means what it means in [`propagate`](crate::propagate): [`Direction::Forward`] applies the channels in order, [`Direction::Heisenberg`] in reverse through [`Channel::apply_adjoint`](crate::Channel::apply_adjoint).
    /// [`EngineSelection`](crate::EngineSelection) is ignored — the partitioned path is always the bucketed layer (see the module docs).
    ///
    /// # Progress logging
    ///
    /// Target `paulistrings::propagate`, as in the unpartitioned engine: one `INFO` line on entry and exit, on the calling thread, and one `DEBUG` line per layer **per partition**, tagged `partition r/P`.
    /// Unlike [`propagate_with`](crate::propagate_with) the per-layer lines come from each partition's own driving thread between layers rather than the calling thread; every site is behind `log_enabled!`, so a disabled logger reads no clock.
    ///
    /// # Panics
    ///
    /// If a channel's [`Channel::prepare`](crate::Channel::prepare) declines (support wider than `MAX_LOCAL_SUPPORT`), exactly as the unpartitioned engine does — there is no fallback path.
    /// If `policy` reports [`finalizes_layer`](crate::TruncationPolicy::finalizes_layer) without overriding [`finalize_layer_partitioned`](PartitionedTruncation::finalize_layer_partitioned).
    /// A panic in any partition is re-raised on the calling thread.
    pub fn propagate_with<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
    ) where
        T: PartitionedTruncation<W> + ?Sized,
    {
        self.propagate_on_backend(circuit, policy, direction, options);
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

    /// Partition `r`'s share of the sum.
    ///
    /// # Panics
    ///
    /// If `r` is not a partition of this sum.
    pub fn partition(&self, r: usize) -> &PauliSum<W> {
        &self.parts[r].sum
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

    /// [`PauliSum::anticommute_histogram`] of the whole sum, the partitions' histograms added in rank order.
    pub fn anticommute_histogram(&self, sites: &[usize], axis: RotationAxis) -> Vec<f64> {
        let mut hist = vec![0.0f64; sites.len() + 1];
        for part in &self.parts {
            for (h, v) in hist
                .iter_mut()
                .zip(part.sum.anticommute_histogram(sites, axis))
            {
                *h += v;
            }
        }
        hist
    }

    /// [`PauliSum::rotated_overlap`] of the whole sum.
    ///
    /// When the rows avoid [`RotationAxis::flip_mask`] of `sites` (see [`PartitionRows::from_seed_excluding`]) every class lies in one partition and the partitions' values are added in rank order; otherwise the partitions are gathered first.
    pub fn rotated_overlap(&self, sites: &[usize], delta: f64, axis: RotationAxis) -> f64 {
        if self.rows.keeps_flip_classes(sites, axis) {
            self.parts
                .iter()
                .map(|part| part.sum.rotated_overlap(sites, delta, axis))
                .sum()
        } else {
            self.gather().rotated_overlap(sites, delta, axis)
        }
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
