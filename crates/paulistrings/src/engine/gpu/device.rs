//! Runtime device probe and the process-wide context cache.
//!
//! `cudarc`'s own `culib()`/`device_count()` panic when the driver library is absent, so every function here checks `libcuda_present()` first.

use std::sync::Arc;

use cudarc::driver::CudaContext;

use super::error::GpuError;
#[cfg(feature = "mpi")]
use crate::engine::cuda_context::libnccl_present;
use crate::engine::cuda_context::{self, libcuda_present, libnvrtc_present};

/// Whether both `libcuda` and `libnvrtc` are present *and* at least one device answers; never panics, even with no CUDA installation at all.
pub fn cuda_available() -> bool {
    libcuda_present() && libnvrtc_present() && device_count() > 0
}

/// Whether `libnccl` can be loaded and a CUDA device is visible, per [`cuda_available`]; never panics, even with no NCCL installation.
#[cfg(feature = "mpi")]
pub fn nccl_available() -> bool {
    cuda_available() && libnccl_present()
}

/// Number of CUDA devices visible to this process, `0` when CUDA is unavailable.
pub fn device_count() -> usize {
    if !libcuda_present() {
        return 0;
    }
    CudaContext::device_count()
        .map(|count| count.max(0) as usize)
        .unwrap_or(0)
}

/// Static facts about one CUDA device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// The device's ordinal, as passed to `CudaContext::new`.
    pub ordinal: u32,
    /// The name the driver reports for the device.
    pub name: String,
    /// `(major, minor)` compute capability.
    pub compute_capability: (u32, u32),
    /// Total device memory in bytes.
    pub total_memory: u64,
}

/// [`DeviceInfo`] for every visible device, or [`GpuError::NoDevice`] rather than an empty `Vec`.
pub fn devices() -> Result<Vec<DeviceInfo>, GpuError> {
    if !libcuda_present() {
        return Err(GpuError::LibraryMissing("libcuda"));
    }
    let count = CudaContext::device_count().map_err(GpuError::from)?;
    if count <= 0 {
        return Err(GpuError::NoDevice);
    }
    let mut out = Vec::with_capacity(count as usize);
    for ordinal in 0..count as u32 {
        let context = context(ordinal)?;
        let (major, minor) = context.compute_capability().map_err(GpuError::from)?;
        out.push(DeviceInfo {
            ordinal,
            name: context.name().map_err(GpuError::from)?,
            compute_capability: (major as u32, minor as u32),
            total_memory: context.total_mem().map_err(GpuError::from)? as u64,
        });
    }
    Ok(out)
}

/// The context for `ordinal`, created on first use and cached for the process.
pub(crate) fn context(ordinal: u32) -> Result<Arc<CudaContext>, GpuError> {
    Ok(cuda_context::context(ordinal)?)
}

#[cfg(test)]
mod tests;
