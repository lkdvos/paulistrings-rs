//! The call frame around `run_layers` shared by every partitioned driver, the scatter, and the [`propagate_partitioned`] front door (ARCHITECTURE.md §Partitioning).

#[cfg(feature = "phase-timing")]
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::runtime::PartitionRuntime;
use super::topology::{PartitionConfig, TopologyError};
use crate::circuit::Circuit;
use crate::collectives::Collectives;
use crate::engine::backend::{PartitionBackend, PartitionStorage};
use crate::engine::driver::{run_layers, PartitionContext, PartitionWork};
use crate::engine::timing::tally;
use crate::engine::trace::{assemble, PartitionTrace};
use crate::engine::transport::Transport;
use crate::engine::{progress_scope, Direction, PropagateOptions};
use crate::pauli_sum::hash::PartitionRows;
use crate::pauli_sum::storage::{desired_bits, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::pauli_sum::PauliSum;
use crate::truncation::{assert_supports_partitions, TruncationPolicy};

use super::sum::PartitionedSum;

pub(super) use crate::engine::LOG_TARGET;

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

/// A driver's own laps since they were last drained.
#[cfg(feature = "phase-timing")]
pub(super) struct DriverLaps {
    scatter_ns: u64,
    /// Atomic because `gather` takes `&self`.
    gather_ns: std::sync::atomic::AtomicU64,
    layers: u64,
}

#[cfg(feature = "phase-timing")]
impl DriverLaps {
    pub(super) fn new(scatter_ns: u64) -> Self {
        Self {
            scatter_ns,
            gather_ns: std::sync::atomic::AtomicU64::new(0),
            layers: 0,
        }
    }

    pub(super) fn lap_gather(&self, ns: u64) {
        self.gather_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Drain scatter time, gather time and layers driven.
    pub(super) fn take(&mut self) -> (u64, u64, u64) {
        (
            std::mem::take(&mut self.scatter_ns),
            self.gather_ns.swap(0, Ordering::Relaxed),
            std::mem::take(&mut self.layers),
        )
    }
}

/// What one `propagate_on_backend` call reads besides a partition's own work, rank and endpoint.
pub(super) struct LayerRun<'a, const W: usize, T: ?Sized> {
    circuit: &'a Circuit<W>,
    policy: &'a T,
    direction: Direction,
    options: PropagateOptions,
    rows: &'a PartitionRows<W>,
    size: usize,
    tracing: bool,
}

impl<const W: usize, T: TruncationPolicy<W> + ?Sized> LayerRun<'_, W, T> {
    /// Partition `rank`'s [`run_layers`].
    pub(super) fn run<X: Transport, B: PartitionBackend<W, T>>(
        &self,
        rank: usize,
        work: &mut PartitionWork<B>,
        transport: &X,
    ) {
        let context = PartitionContext {
            rows: self.rows,
            rank,
            size: self.size,
            tracing: self.tracing,
        };
        run_layers(
            self.circuit,
            self.policy,
            self.direction,
            self.options,
            context,
            work,
            transport,
        );
    }
}

/// The partitions a driver holds and its records, lent to [`propagate`](Self::propagate) for one call.
pub(super) struct DriverFrame<'a, const W: usize, B> {
    /// In rank order: every partition in process, or this rank's one.
    pub(super) parts: &'a mut [B],
    pub(super) rows: &'a PartitionRows<W>,
    pub(super) runtime: &'a PartitionRuntime,
    pub(super) trace: &'a mut Option<PartitionTrace>,
    #[cfg(feature = "phase-timing")]
    pub(super) laps: &'a mut DriverLaps,
    /// This process's rank in a distributed group, `None` when it holds every partition.
    pub(super) rank: Option<usize>,
    /// Partitions in the group.
    pub(super) size: usize,
}

impl<const W: usize, B: PartitionStorage<W>> DriverFrame<'_, W, B> {
    /// The prologue and epilogue every partitioned driver shares around `run`, which drives the partitions' layers and returns their work in rank order; `agree` runs after the entry line, before any layer.
    pub(super) fn propagate<T>(
        self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
        agree: impl FnOnce(),
        run: impl FnOnce(Vec<PartitionWork<B>>, &LayerRun<'_, W, T>) -> Vec<PartitionWork<B>>,
    ) where
        T: TruncationPolicy<W> + ?Sized,
        B: PartitionBackend<W, T>,
    {
        options.assert_valid();
        let num_layers = circuit.channels.len();
        let size = self.size;
        assert_supports_partitions(policy, size);
        let terms_in: usize = self.parts.iter().map(PartitionStorage::len).sum();
        let started = Instant::now();
        log::info!(
            target: LOG_TARGET,
            "propagate: {terms_in} terms through {num_layers} channels ({direction:?}), {}",
            progress_scope(self.rank, size, Some(&self.runtime.placement_summary())),
        );

        agree();

        if num_layers > 0 {
            let tracing = self.trace.is_some();
            let items: Vec<PartitionWork<B>> = self
                .parts
                .iter_mut()
                .map(|part| PartitionWork::take(part, num_layers, tracing))
                .collect();
            let layers = LayerRun {
                circuit,
                policy,
                direction,
                options,
                rows: self.rows,
                size,
                tracing,
            };
            let done = run(items, &layers);
            assert_eq!(done.len(), self.parts.len(), "one work item per partition");
            let mut traced = Vec::with_capacity(if tracing { done.len() } else { 0 });
            for (part, work) in self.parts.iter_mut().zip(done) {
                *part = work.local;
                if tracing {
                    traced.push(work.rows);
                }
            }
            if let Some(trace) = self.trace.as_mut() {
                assemble(trace, traced, size);
            }
            tally!(self.laps.layers, num_layers);
        }

        log::info!(
            target: LOG_TARGET,
            "propagate: {num_layers} layers applied, {terms_in} -> {} terms, {:.3} s, {}",
            self.parts.iter().map(PartitionStorage::len).sum::<usize>(),
            started.elapsed().as_secs_f64(),
            progress_scope(self.rank, size, Some(&self.runtime.placement_summary())),
        );
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
