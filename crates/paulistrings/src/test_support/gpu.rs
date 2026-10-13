//! Device hooks for tests, benches and the probe: fault injection on the device drivers.

use crate::engine::gpu::partition::DevicePartition;
use crate::engine::partitioned::{DistributedSum, PartitionedSum, Transport};

pub use crate::engine::gpu::{FaultPlan, WireFault};

/// A device driver whose partitions a test can reach by rank.
pub trait DeviceSplit<const W: usize> {
    #[doc(hidden)]
    fn device_partition_mut(&mut self, rank: usize) -> &mut DevicePartition<W>;
}

impl<const W: usize> DeviceSplit<W> for PartitionedSum<W, DevicePartition<W>> {
    fn device_partition_mut(&mut self, rank: usize) -> &mut DevicePartition<W> {
        &mut self.backends_mut()[rank]
    }
}

impl<const W: usize, X: Transport> DeviceSplit<W> for DistributedSum<W, X, DevicePartition<W>> {
    fn device_partition_mut(&mut self, rank: usize) -> &mut DevicePartition<W> {
        assert_eq!(
            rank,
            self.rank() as usize,
            "a distributed split arms faults on its own rank only"
        );
        self.backend_mut()
    }
}

/// Arm `plan` on partition `rank` of `split`, every fault firing once from the next call; a distributed split takes its own rank only.
pub fn set_fault_plan<const W: usize>(
    split: &mut impl DeviceSplit<W>,
    rank: usize,
    plan: FaultPlan,
) {
    split.device_partition_mut(rank).set_fault_plan(plan);
}
