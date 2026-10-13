//! Device hooks for tests, benches and the probe: fault injection on the device drivers and the crate-private `GpuSum` checks.

use crate::engine::gpu::partition::DevicePartition;
use crate::engine::gpu::{GpuError, GpuPartitionedSum, GpuSum};
use crate::engine::partitioned::{DistributedSum, PartitionedSum};
use crate::engine::transport::Transport;
use crate::pauli_sum::PauliSum;

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

/// `sum`'s device-side structural invariant, order within a bucket free: the first class of violation, or the device error that stopped the check.
pub fn assert_invariants_device<const W: usize>(sum: &GpuSum<W>) -> Result<(), String> {
    sum.assert_invariants_device()
}

/// Refine `sum` until its bucket count is `1 << bits`, up to four bits per counting pass.
pub fn refine_to<const W: usize>(sum: &mut GpuSum<W>, bits: u8) -> Result<(), GpuError> {
    sum.refine_to(bits)
}

/// [`GpuPartitionedSum::from_host`] with extra NVRTC options, such as the `-DFINGERPRINT_BITS=<b>` collision hook.
pub fn from_host_with_nvrtc_options<const W: usize>(
    sum: &PauliSum<W>,
    device: u32,
    extra_options: &[String],
) -> Result<GpuPartitionedSum<W>, GpuError> {
    GpuPartitionedSum::one_device(sum, device, extra_options)
}
