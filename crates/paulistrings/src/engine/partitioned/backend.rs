//! The storage seam under `run_layers`: [`PartitionStorage`] and [`PartitionBackend`] are what one partition's share of the sum must do for the layer loop, and [`HostPartition`] is the host-memory implementation.
//! See ARCHITECTURE.md §Partitioning.

use super::layer::{apply_layer_partitioned_with_plan, LayerExchangeCounts, PartitionState};
use super::plan::PartitionPlan;
use super::transport::{Collectives, Transport};
use super::truncation::PartitionedTruncation;
use crate::bucket::hash::{Gf2Hash, PartitionRows};
use crate::bucket::sum::PauliSum;
use crate::channel::prepared::Prepared;
#[cfg(feature = "phase-timing")]
use crate::engine::stats::PhaseStats;

/// The policy-independent half of a partition: what the layer loop reads and reshapes between layers.
pub(crate) trait PartitionStorage<const W: usize>: Send + Sized {
    /// Terms this partition holds.
    fn len(&self) -> usize;
    /// The hash and bucket count every partition of the group shares.
    fn hash(&self) -> &Gf2Hash<W>;
    /// Add one bucket bit, as [`PauliSum::refine`].
    fn refine(&mut self);
    /// Move the partition out, leaving a valid empty partition under the same hash behind.
    ///
    /// The placeholder keeps a driver self-consistent (same rows, same hash, same bucket count on every partition) if the partition panics and the work is never handed back.
    fn detach(&mut self) -> Self;
    /// The phase counters the loop laps into.
    #[cfg(feature = "phase-timing")]
    fn stats(&mut self) -> &mut PhaseStats;
}

/// The layer itself under policy type `T`.
///
/// Everything collective stays in `run_layers`; an implementor must issue exactly the transport calls the host layer issues, in the same order, or the group falls out of step.
pub(crate) trait PartitionBackend<const W: usize, T: ?Sized>: PartitionStorage<W> {
    /// One layer's export, exchange and merge, as [`apply_layer_partitioned_with_plan`].
    fn apply_layer<X: Transport>(
        &mut self,
        prep: &Prepared<W>,
        plan: &PartitionPlan,
        rows: &PartitionRows<W>,
        policy: &T,
        transport: &X,
    ) -> LayerExchangeCounts;
    /// The policy's collective layer pass, as [`PartitionedTruncation::finalize_layer_partitioned`].
    fn finalize_layer(&mut self, policy: &T, coll: &dyn Collectives);
}

/// A partition held in host memory: its sum and its layer and export scratch, retained across calls.
///
/// Nominally `pub` only so it can be `DistributedSum`'s default backend; the module is crate-private, so it cannot be named outside the crate.
#[derive(Debug)]
pub struct HostPartition<const W: usize> {
    pub(super) sum: PauliSum<W>,
    pub(super) state: PartitionState<W>,
}

impl<const W: usize> HostPartition<W> {
    /// `sum` with fresh scratch.
    pub(super) fn new(sum: PauliSum<W>) -> Self {
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
mod tests {
    use super::*;
    use crate::test_support::{assert_same_terms, rand_sum};

    fn detach_leaves_an_empty_partition_under_the_same_hash<const W: usize>(num_qubits: usize) {
        let mut sum = rand_sum::<W>(3_000, num_qubits, 0xDE7AC4 + W as u64);
        sum.refine();
        sum.refine();
        let want = sum.clone();
        let mut part = HostPartition::new(sum);

        let moved = part.detach();

        assert_eq!(part.len(), 0);
        assert!(part.sum.is_empty());
        assert_eq!(part.hash().bits(), want.hash().bits());
        assert!(part.hash().same_rows_as(want.hash()));
        assert_eq!(part.sum.num_qubits(), want.num_qubits());
        assert_eq!(part.sum.num_buckets(), want.num_buckets());
        part.sum.assert_invariants();

        assert_eq!(moved.len(), want.len());
        assert_same_terms(&moved.sum, &want, "detached partition");
        moved.sum.assert_invariants();
    }

    #[test]
    fn detach_leaves_an_empty_partition_under_the_same_hash_w1() {
        detach_leaves_an_empty_partition_under_the_same_hash::<1>(40);
    }

    #[test]
    fn detach_leaves_an_empty_partition_under_the_same_hash_w2() {
        detach_leaves_an_empty_partition_under_the_same_hash::<2>(100);
    }
}
