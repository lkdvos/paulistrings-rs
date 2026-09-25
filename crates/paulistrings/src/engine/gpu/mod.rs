//! The CUDA backend. See ARCHITECTURE.md §GPU-Readiness.
//!
//! `device` is the runtime probe and device table, `module` the NVRTC kernel cache, [`GpuSum`] the device-resident sum over `columns`, [`GpuPauliSum`]/[`GpuPartitionedSum`] the propagation drivers over `layer`, `export` and `partition`, and [`GpuDistributedSum`] one device partition per process (`rank`).

mod columns;
pub(crate) mod device;
mod driver;
mod error;
mod export;
mod finalize;
mod fingerprint;
mod layer;
mod module;
#[cfg(feature = "nccl")]
mod nccl;
mod partition;
mod payload;
mod prepared;
mod rank;
mod scan;
mod staging;
mod sum;
mod truncation;

#[cfg(feature = "nccl")]
pub use device::nccl_available;
pub use device::{cuda_available, device_count, devices, DeviceInfo};
pub use driver::{propagate_gpu, propagate_gpu_partitioned, GpuPartitionedSum, GpuPauliSum};
pub use error::GpuError;
pub use layer::{
    GpuBucketPolicy, GpuKernelMs, GpuLayerCounters, GpuLayerOptions, DEFAULT_ARENA_BYTES,
    DEFAULT_RECORDS_PER_BLOCK,
};
#[cfg(all(feature = "nccl", any(test, feature = "test-utils")))]
pub use nccl::{LoopbackTally, LoopbackWire};
pub use payload::{peer_access, GpuExchange, PeerAccess};
pub use rank::{local_device_for_rank, GpuDistributedSum};
#[cfg(feature = "mpi")]
pub use rank::{propagate_mpi_gpu, MpiGpuSum};
pub use sum::GpuSum;
