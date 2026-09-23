//! The CUDA backend. See ARCHITECTURE.md §GPU-Readiness.
//!
//! `device` is the runtime probe and device table, `module` the NVRTC kernel cache, [`GpuSum`] the device-resident sum over `columns`, and [`GpuPauliSum`] the propagation driver over `layer` and `partition`.

mod columns;
mod device;
mod driver;
mod error;
mod finalize;
mod fingerprint;
mod layer;
mod module;
mod partition;
mod prepared;
mod scan;
mod staging;
mod sum;
mod truncation;

pub use device::{cuda_available, device_count, devices, DeviceInfo};
pub use driver::{propagate_gpu, GpuPauliSum};
pub use error::GpuError;
pub use layer::{
    GpuBucketPolicy, GpuKernelMs, GpuLayerCounters, GpuLayerOptions, DEFAULT_ARENA_BYTES,
    DEFAULT_RECORDS_PER_BLOCK,
};
pub use sum::GpuSum;
