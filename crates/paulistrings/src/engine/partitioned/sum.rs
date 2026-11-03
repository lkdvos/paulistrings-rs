//! [`PartitionedSum`], one [`PauliSum`] split across the partitions of a [`PartitionRuntime`].

use std::sync::Arc;
use std::time::Instant;

use crate::coefficient::Coefficient;
use num_complex::Complex64;
use std::marker::PhantomData;

use super::distributed::{repartition_share, PartitionRowPolicy, ScatterOptions, ScatterRows};
#[cfg(feature = "phase-timing")]
use super::driver::DriverLaps;
use super::driver::{scatter_local, DriverFrame, LOG_TARGET};
use super::runtime::PartitionRuntime;
use super::topology::{PartitionConfig, TopologyError};
use crate::circuit::Circuit;
use crate::engine::backend::{HostPartition, PartitionBackend, PartitionStorage};
#[cfg(feature = "phase-timing")]
use crate::engine::stats::PartitionPhaseStats;
use crate::engine::timing::timed;
use crate::engine::trace::PartitionTrace;
use crate::engine::{progress_scope, Direction, PropagateOptions};
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows};
use crate::pauli_sum::PauliSum;
use crate::readout::product_state::ProductBasis;
use crate::truncation::TruncationPolicy;

/// A [`PauliSum`] split across the partitions of a [`PartitionRuntime`], held across calls so a driver scatters and gathers once.
///
/// Between calls every partition's sum holds only keys of its own partition and shares one hash and bucket count with its peers ([`PartitionedSum::assert_invariants`]).
/// `B` is where a partition lives: host memory by default, a CUDA device for `gpu::GpuPartitionedSum`.
pub struct PartitionedSum<const W: usize, C: Coefficient = Complex64, B = HostPartition<W, C>> {
    /// In rank order.
    parts: Vec<B>,
    coefficient: PhantomData<fn() -> C>,
    rows: PartitionRows<W>,
    runtime: Arc<PartitionRuntime>,
    /// `None` unless [`enable_trace`](Self::enable_trace) was called.
    trace: Option<PartitionTrace>,
    #[cfg(feature = "phase-timing")]
    laps: DriverLaps,
}

impl<const W: usize, C: Coefficient, B> PartitionedSum<W, C, B> {
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
            coefficient: PhantomData,
            rows,
            runtime,
            trace: None,
            #[cfg(feature = "phase-timing")]
            laps: DriverLaps::new(scatter_ns),
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
        self.laps.take()
    }

    #[cfg(all(feature = "cuda", feature = "phase-timing"))]
    pub(crate) fn lap_gather(&self, ns: u64) {
        self.laps.lap_gather(ns);
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

    /// Start recording a [`PartitionTrace`] on every later propagation; idempotent.
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
        T: TruncationPolicy<W, C> + ?Sized,
        B: PartitionBackend<W, C, T>,
    {
        let runtime = &*self.runtime;
        let size = self.parts.len();
        let frame = DriverFrame {
            parts: &mut self.parts,
            rows: &self.rows,
            runtime,
            trace: &mut self.trace,
            #[cfg(feature = "phase-timing")]
            laps: &mut self.laps,
            rank: None,
            size,
        };
        frame.propagate(
            circuit,
            policy,
            direction,
            options,
            || {},
            |items, layers| {
                runtime.map_partitions(items, |rank, mut work, transport| {
                    layers.run(rank, &mut work, transport);
                    work
                })
            },
        );
    }
}

impl<const W: usize, C: Coefficient, B: PartitionStorage<W, C>> PartitionedSum<W, C, B> {
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

impl<const W: usize, C: Coefficient> PartitionedSum<W, C> {
    /// Splits `sum` across the partitions `config` places, by the default rows ([`PartitionRowPolicy::Seeded`] at the sum's own hash seed); other rows go through [`scatter_with`](Self::scatter_with).
    ///
    /// # Errors
    ///
    /// [`TopologyError`] if `config` cannot be resolved or a pool cannot be built.
    ///
    /// # Panics
    ///
    /// In debug builds, if the partition rows are not independent of the sum's hash rows.
    pub fn scatter(sum: PauliSum<W, C>, config: &PartitionConfig) -> Result<Self, TopologyError> {
        let options = ScatterOptions {
            runtime: PartitionRuntime::new(config)?,
            rows: ScatterRows::Policy(PartitionRowPolicy::Seeded(None)),
        };
        Ok(Self::scatter_with(sum, options))
    }

    /// [`scatter`](Self::scatter) onto a caller-built runtime, split by the rows `options` names; at `P = 1` the scatter changes nothing.
    ///
    /// # Panics
    ///
    /// If the rows do not name the runtime's partition count, or as [`PartitionRows::cut`].
    /// In debug builds, if the rows are not independent of the sum's hash rows, which costs load balance but not correctness.
    pub fn scatter_with(sum: PauliSum<W, C>, options: ScatterOptions<W>) -> Self {
        let ScatterOptions { runtime, rows } = options;
        let rows = rows.resolve(&sum, runtime.partition_bits());
        let size = runtime.num_partitions();
        rows.assert_splits(sum.num_qubits(), size);

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
            "scatter: {} terms -> {} held, {} bucket bits, {:.3} s, {}",
            sum.len(),
            locals.iter().map(PauliSum::len).sum::<usize>(),
            locals[0].hash().bits(),
            started.elapsed().as_secs_f64(),
            progress_scope(None, size, Some(&runtime.placement_summary())),
        );

        Self {
            parts: locals.into_iter().map(HostPartition::new).collect(),
            coefficient: PhantomData,
            rows,
            runtime,
            trace: None,
            #[cfg(feature = "phase-timing")]
            laps: DriverLaps::new(started.elapsed().as_nanos() as u64),
        }
    }

    /// Move every term to the partition `rows` assigns it and split by `rows` from here on, keeping the bucket hash; each partition exchanges with every other once, with no gather.
    ///
    /// # Panics
    ///
    /// If `rows` does not name this split's partition count, or as [`repartition_share`].
    pub fn repartition(&mut self, rows: PartitionRows<W>) {
        let shares: Vec<PauliSum<W, C>> = self
            .parts
            .iter_mut()
            .map(|part| std::mem::replace(&mut part.sum, PauliSum::empty(rows.num_qubits())))
            .collect();
        let moved = {
            let rows = &rows;
            self.runtime.map_partitions(shares, |_, share, transport| {
                repartition_share(share, rows, transport)
            })
        };
        for (part, sum) in self.parts.iter_mut().zip(moved) {
            part.sum = sum;
        }
        self.rows = rows;
    }

    /// Rebucket every partition by `hash`, at its bucket count, each inside its own pool so its memory stays on its node; no term changes partition.
    ///
    /// # Panics
    ///
    /// As [`PauliSum::with_hash`].
    pub fn rehash_buckets(&mut self, hash: Gf2Hash<W>) {
        let shares: Vec<PauliSum<W, C>> = self
            .parts
            .iter_mut()
            .map(|part| std::mem::replace(&mut part.sum, PauliSum::empty(self.rows.num_qubits())))
            .collect();
        let rehashed =
            self.runtime.map_partitions(shares, |_, share, _| share.with_hash(hash.clone()));
        for (part, sum) in self.parts.iter_mut().zip(rehashed) {
            part.sum = sum;
        }
    }

    /// Propagates through `circuit` under `policy`, in place; see [`propagate_with`](Self::propagate_with).
    pub fn propagate<T>(&mut self, circuit: &Circuit<W>, policy: &T, direction: Direction)
    where
        T: TruncationPolicy<W, C> + ?Sized,
    {
        self.propagate_with(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagates through `circuit` under `policy` with explicit [`PropagateOptions`], as [`propagate_with`](crate::propagate_with) does unpartitioned.
    ///
    /// The per-layer `DEBUG` log line comes once per partition, tagged `partition r/P`, from that partition's own thread.
    ///
    /// # Panics
    ///
    /// If `options` fail [`PropagateOptions::validate`], or a channel's preparation declines, as the unpartitioned engine does.
    /// Before the first layer, above one partition, unless [`policy.supports_partitioned()`](TruncationPolicy::supports_partitioned) — exact [`TopN`](crate::TopN) is not yet supported partitioned.
    /// A panic in any partition is re-raised on the calling thread.
    pub fn propagate_with<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
    ) where
        T: TruncationPolicy<W, C> + ?Sized,
    {
        self.propagate_on_backend(circuit, policy, direction, options);
    }

    /// Merges the partitions back into one sum on the calling thread, leaving `self` intact.
    pub fn gather(&self) -> PauliSum<W, C> {
        timed!(let started = Instant::now());
        let out = PauliSum::merge_partitions(self.parts.iter().map(|p| p.sum.clone()).collect());
        timed!(self.laps.lap_gather(started.elapsed().as_nanos() as u64));
        out
    }

    /// [`gather`](Self::gather) by value — no clone of the parts.
    pub fn into_gathered(self) -> PauliSum<W, C> {
        PauliSum::merge_partitions(self.parts.into_iter().map(|p| p.sum).collect())
    }

    /// Partition `r`'s share of the sum.
    ///
    /// # Panics
    ///
    /// If `r` is not a partition of this sum.
    pub fn partition(&self, r: usize) -> &PauliSum<W, C> {
        &self.parts[r].sum
    }

    /// `⟨ψ|O|ψ⟩` in the product state `basis`, the partitions' own values added in rank order.
    pub fn expectation_product(&self, basis: &ProductBasis<W>) -> C {
        self.parts.iter().map(|part| part.sum.expectation_product(basis)).sum()
    }

    /// Drain the phase counters since the last drain, zeroing them; `per_partition` has one entry per partition.
    #[cfg(feature = "phase-timing")]
    pub fn take_stats(&mut self) -> PartitionPhaseStats {
        let (scatter_ns, gather_ns, layers) = self.laps.take();
        PartitionPhaseStats {
            per_partition: self
                .parts
                .iter_mut()
                .map(|part| part.state.layer.take_phase_stats())
                .collect(),
            scatter_ns,
            gather_ns,
            layers,
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
