//! The CUDA backend (feature `cuda`): device probes, device-resident sums and their propagation drivers. See ARCHITECTURE.md §GPU-Readiness.
//!
//! [`GpuPartitionedSum`] holds a sum on one device or splits one across the devices of a process, and [`GpuDistributedSum`] holds one device's share per rank of a transport.

mod columns;
mod device;
mod driver;
mod error;
mod export;
mod finalize;
mod fingerprint;
mod kernel_cache;
mod layer;
mod module;
#[cfg(feature = "mpi")]
mod nccl;
pub(crate) mod partition;
mod payload;
mod prepared;
mod rank;
mod scan;
mod staging;
mod sum;
mod truncation;
mod wire;

/// `log` target of device setup: kernel compiles and the kernel cache, device choice, and the device wires (NCCL, peer access).
pub(crate) const GPU_LOG_TARGET: &str = "paulistrings::gpu";

#[cfg(feature = "mpi")]
pub use device::nccl_available;
pub use device::{cuda_available, device_count, devices, DeviceInfo};
pub use driver::{propagate_gpu, propagate_gpu_partitioned, GpuPartitionedSum, GpuScatterOptions};
pub use error::GpuError;
pub use layer::{GpuLayerOptions, DEFAULT_ARENA_BYTES, DEFAULT_RECORDS_PER_BLOCK};
#[cfg(any(test, feature = "test-utils"))]
pub use partition::FaultPlan;
pub use rank::GpuDistributedSum;
#[cfg(feature = "mpi")]
pub use rank::{local_device_for_comm, propagate_mpi_gpu, MpiGpuSum};
pub use sum::GpuSum;
#[cfg(any(test, feature = "test-utils"))]
pub use wire::WireFault;
