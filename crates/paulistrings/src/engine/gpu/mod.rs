//! The CUDA backend (feature `cuda`): device probes, device-resident sums and their propagation drivers. See ARCHITECTURE.md §GPU-Readiness.
//!
//! [`GpuPauliSum`] holds a sum on one device, [`GpuPartitionedSum`] splits one across the devices of a process, and [`GpuDistributedSum`] holds one device's share per rank of a transport.

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
mod partition;
mod payload;
mod prepared;
mod rank;
mod scan;
mod staging;
mod sum;
mod truncation;
mod wire;

#[cfg(feature = "mpi")]
pub use device::nccl_available;
pub use device::{cuda_available, device_count, devices, DeviceInfo};
pub use driver::{
    first_failure, propagate_gpu, propagate_gpu_partitioned, GpuPartitionedSum, GpuPauliSum,
};
pub use error::GpuError;
pub use layer::{
    GpuBucketPolicy, GpuLayerCounters, GpuLayerOptions, DEFAULT_ARENA_BYTES,
    DEFAULT_RECORDS_PER_BLOCK,
};
pub use rank::GpuDistributedSum;
#[cfg(feature = "mpi")]
pub use rank::{local_device_for_comm, propagate_mpi_gpu, MpiGpuSum};
pub use sum::GpuSum;
