//! The storage seam under `run_layers`, [`PartitionStorage`] and [`PartitionBackend`], and its host-memory implementation [`HostPartition`] (ARCHITECTURE.md §Partitioning).

use super::layer::{apply_layer_partitioned_with_plan, LayerExchangeCounts, PartitionState};
use super::plan::PartitionPlan;
use super::transport::{Collectives, Transport};
use super::truncation::PartitionedTruncation;
use crate::channel::prepared::Prepared;
#[cfg(feature = "phase-timing")]
use crate::engine::stats::PhaseStats;
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows};
use crate::pauli_sum::storage::PauliSum;

/// The policy-independent half of a partition; `pub` only so the drivers' `pub` methods can bound on it.
pub trait PartitionStorage<const W: usize>: Send + Sized {
    fn len(&self) -> usize;
    fn hash(&self) -> &Gf2Hash<W>;
    fn refine(&mut self);
    /// The bucket bits this partition proposes for the layer; a backend may raise the host formula but never lower it.
    fn proposed_bits(
        &self,
        prep: &Prepared<W>,
        target_bucket_len: usize,
        min_buckets: usize,
    ) -> u8 {
        let _ = prep;
        crate::pauli_sum::storage::desired_bits(self.len(), target_bucket_len, min_buckets)
            .max(self.hash().bits())
    }
    /// Move the partition out, leaving an empty one under the same hash so the driver stays consistent if the work is never handed back.
    fn detach(&mut self) -> Self;
    #[cfg(feature = "phase-timing")]
    fn stats(&mut self) -> &mut PhaseStats;
}

/// One layer under policy `T`; an implementor must issue exactly the transport calls the host layer issues, in the same order, or the group falls out of step.
pub(crate) trait PartitionBackend<const W: usize, T: ?Sized>: PartitionStorage<W> {
    fn apply_layer<X: Transport>(
        &mut self,
        prep: &Prepared<W>,
        plan: &PartitionPlan,
        rows: &PartitionRows<W>,
        policy: &T,
        transport: &X,
    ) -> LayerExchangeCounts;
    fn finalize_layer(&mut self, policy: &T, coll: &dyn Collectives);
}

/// A partition in host memory with its retained scratch; `pub` only so it can be a `pub` type's default backend.
#[derive(Debug)]
pub struct HostPartition<const W: usize> {
    pub(crate) sum: PauliSum<W>,
    pub(super) state: PartitionState<W>,
}

impl<const W: usize> HostPartition<W> {
    pub(crate) fn new(sum: PauliSum<W>) -> Self {
        Self {
            sum,
            state: PartitionState::default(),
        }
    }
}

impl<const W: usize> PartitionStorage<W> for HostPartition<W> {
    #[inline]
    fn len(&self) -> usize {
        self.sum.len()
    }

    #[inline]
    fn hash(&self) -> &Gf2Hash<W> {
        self.sum.hash()
    }

    #[inline]
    fn refine(&mut self) {
        self.sum.refine();
    }

    fn detach(&mut self) -> Self {
        let placeholder = PauliSum::empty_with_hash(self.sum.num_qubits(), self.sum.hash().clone());
        Self {
            sum: std::mem::replace(&mut self.sum, placeholder),
            state: std::mem::take(&mut self.state),
        }
    }

    #[cfg(feature = "phase-timing")]
    #[inline]
    fn stats(&mut self) -> &mut PhaseStats {
        &mut self.state.layer.stats
    }
}

impl<const W: usize, T> PartitionBackend<W, T> for HostPartition<W>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    #[inline]
    fn apply_layer<X: Transport>(
        &mut self,
        prep: &Prepared<W>,
        plan: &PartitionPlan,
        rows: &PartitionRows<W>,
        policy: &T,
        transport: &X,
    ) -> LayerExchangeCounts {
        apply_layer_partitioned_with_plan(
            &mut self.sum,
            prep,
            plan,
            rows,
            policy,
            &mut self.state,
            transport,
        )
    }

    #[inline]
    fn finalize_layer(&mut self, policy: &T, coll: &dyn Collectives) {
        policy.finalize_layer_partitioned(&mut self.sum, coll);
    }
}

#[cfg(test)]
mod tests;
