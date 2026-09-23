//! The CUDA backend. See ARCHITECTURE.md §GPU-Readiness.
//!
//! `device` is the runtime probe and the process-wide device table; `module`
//! compiles kernels through NVRTC; `error` is the shared `Result` type.

mod device;
mod error;
mod module;

pub use device::{cuda_available, device_count, devices, DeviceInfo};
pub use error::GpuError;
