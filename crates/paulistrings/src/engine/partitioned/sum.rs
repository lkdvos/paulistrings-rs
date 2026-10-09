//! [`PartitionedSum`], one [`PauliSum`] split across the partitions of a [`PartitionRuntime`].

use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;

use super::backend::{HostPartition, PartitionBackend, PartitionStorage};
use super::driver::{run_layers, scatter_local, PartitionContext, PartitionWork, LOG_TARGET};
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

/// A [`PauliSum`] split across the partitions of a [`PartitionRuntime`], held across calls so a driver scatters and gathers once.
///
/// Between calls every partition's sum holds only keys of its own partition and shares one hash and bucket count with its peers ([`PartitionedSum::assert_invariants`]).
/// `B` is where a partition lives: host memory by default, a CUDA device for `gpu::GpuPartitionedSum`.
pub struct PartitionedSum<const W: usize, B = HostPartition<W>> {
    /// In rank order.
    parts: Vec<B>,
    rows: PartitionRows<W>,
    runtime: Arc<PartitionRuntime>,
    /// `None` unless [`enable_trace`](Self::enable_trace) was called.
    trace: Option<PartitionTrace>,
    #[cfg(feature = "phase-timing")]
    scatter_ns: u64,
    /// Atomic because [`gather`](Self::gather) takes `&self`.
    #[cfg(feature = "phase-timing")]
    gather_ns: std::sync::atomic::AtomicU64,
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

    #[cfg(feature = "cuda")]
    pub(crate) fn backends(&self) -> &[B] {
        &self.parts
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn backends_mut(&mut self) -> &mut [B] {
        &mut self.parts
    }

    /// Drain scatter time, gather time and layers driven.
    #[cfg(all(feature = "cuda", feature = "phase-timing"))]
    pub(crate) fn take_driver_laps(&mut self) -> (u64, u64, u64) {
        (
            std::mem::take(&mut self.scatter_ns),
            self.gather_ns.swap(0, std::sync::atomic::Ordering::Relaxed),
            std::mem::take(&mut self.layers),
        )
    }

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

    /// The runtime this sum runs on, for handing to another [`PartitionedSum`].
    pub fn runtime(&self) -> &Arc<PartitionRuntime> {
        &self.runtime
    }

    /// Start recording a [`PartitionTrace`] on every subsequent propagation; idempotent.
    pub fn enable_trace(&mut self) {
        self.trace.get_or_insert_with(PartitionTrace::default);
    }

    /// Drain the per-layer records, or `None` if tracing was never enabled; tracing stays enabled.
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
            let tracing = self.trace.is_some();
            let items: Vec<PartitionWork<B>> = self
                .parts
                .iter_mut()
                .map(|part| PartitionWork::take(part, n, tracing))
                .collect();

            let runtime = Arc::clone(&self.runtime);
            let rows = &self.rows;
            let done = runtime.map_partitions(items, |rank, mut work, transport| {
                let context = PartitionContext {
                    rows,
                    rank,
                    size,
                    tracing,
                };
                run_layers(
                    circuit, policy, direction, options, context, &mut work, transport,
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

    /// The bucket bits every partition currently holds.
    pub fn bits(&self) -> u8 {
        self.parts[0].hash().bits()
    }

    /// Qubits the sum is over.
    pub fn num_qubits(&self) -> usize {
        self.rows.num_qubits()
    }
}

impl<const W: usize> PartitionedSum<W> {
    /// Splits `sum` across `runtime`'s partitions, with rows from [`PartitionRows::from_seed`] at `config.partition_row_seed` or else the sum's hash seed.
    ///
    /// # Panics
    ///
    /// In debug builds, if the partition rows are not independent of the sum's hash rows.
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

    /// Splits `sum` across `runtime`'s partitions using caller-supplied rows; at `P = 1` the scatter changes nothing.
    ///
    /// # Panics
    ///
    /// If `rows.num_partitions()` is not the runtime's partition count.
    /// In debug builds, if the rows are not independent of the sum's hash rows, which costs load balance but not correctness.
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

    /// Propagates through `circuit` under `policy`, in place; see [`propagate_with`](Self::propagate_with).
    pub fn propagate<T>(&mut self, circuit: &Circuit<W>, policy: &T, direction: Direction)
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        self.propagate_with(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagates through `circuit` under `policy` with explicit [`PropagateOptions`], as [`propagate_with`](crate::propagate_with) does unpartitioned.
    ///
    /// [`EngineSelection`](crate::EngineSelection) is ignored: the partitioned path is always the bucketed layer.
    /// The per-layer `DEBUG` log line comes once per partition, tagged `partition r/P`, from that partition's own thread.
    ///
    /// # Panics
    ///
    /// If a channel's [`Channel::prepare`](crate::Channel::prepare) declines, as the unpartitioned engine does.
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

    /// Merges the partitions back into one sum on the calling thread, leaving `self` intact.
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

    /// `⟨ψ|O|ψ⟩` in a uniform single-qubit product state, the partitions' own values added in rank order.
    pub fn expectation_product_state(&self, state: ProductState) -> Complex64 {
        self.parts
            .iter()
            .map(|part| part.sum.expectation_product_state(state))
            .sum()
    }

    /// [`PauliSum::anticommute_histogram`] of the whole sum, the partitions' histograms added in rank order.
    pub fn anticommute_histogram(&self, sites: &[usize], axis: RotationAxis) -> Vec<f64> {
        let mut histogram = vec![0.0f64; sites.len() + 1];
        for part in &self.parts {
            for (h, v) in histogram
                .iter_mut()
                .zip(part.sum.anticommute_histogram(sites, axis))
            {
                *h += v;
            }
        }
        histogram
    }

    /// [`PauliSum::rotated_overlap`] of the whole sum; gathers first unless the rows avoid the flipped coordinates of `sites`.
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

    /// Drain the per-phase timing counters, zeroing them.
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

    /// Checks every invariant a partitioned sum holds between calls, in `O(terms)`; release builds skip the per-partition structural check.
    ///
    /// # Panics
    ///
    /// If a partition's sum is internally inconsistent, holds a key of another partition, or disagrees with partition 0 about the hash rows, the bucket count or the qubit count.
    pub fn assert_invariants(&self) {
        assert_eq!(
            self.parts.len(),
            self.rows.num_partitions(),
            "partition count disagrees with the rows",
        );
        let head = &self.parts[0].sum;
        for (rank, local) in self.parts.iter().map(|p| &p.sum).enumerate() {
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

/// The partitioned engine's phase breakdown, drained by [`PartitionedSum::take_stats`].
///
/// The per-partition wall-clock fields are concurrent, not additive; `exchange_ns` absorbs the wait for a partner, so the load imbalance shows there.
#[cfg(feature = "phase-timing")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartitionPhaseStats {
    /// One breakdown per partition, in rank order.
    pub per_partition: Vec<PhaseStats>,
    /// Wall time of the scatter that built this sum.
    pub scatter_ns: u64,
    /// Wall time summed over [`PartitionedSum::gather`] calls.
    pub gather_ns: u64,
    /// Layers driven since the counters were last drained.
    pub layers: u64,
}
