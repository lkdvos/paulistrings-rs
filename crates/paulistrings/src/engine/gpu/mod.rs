//! The CUDA backend. See ARCHITECTURE.md §GPU-Readiness.
//!
//! `device` is the runtime probe and device table, `module` the NVRTC kernel cache, and [`GpuSum`] the device-resident sum over `columns`.

mod columns;
mod device;
mod error;
mod fingerprint;
mod module;
mod scan;
mod staging;
mod sum;

pub use device::{cuda_available, device_count, devices, DeviceInfo};
pub use error::GpuError;
pub use sum::GpuSum;
