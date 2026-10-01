//! Runtime device probe and the process-wide context cache.
//!
//! `cudarc`'s own `culib()`/`device_count()` panic when the driver library is absent, so every function here checks `is_culib_present()` first.

use std::sync::{Arc, Mutex, PoisonError};

use cudarc::driver::CudaContext;

use super::error::GpuError;

/// Whether both `libcuda` and `libnvrtc` are present *and* at least one device answers; never panics, even with no CUDA installation at all.
pub fn cuda_available() -> bool {
    // SAFETY: `is_culib_present` only probes `dlopen`-style candidates; it never touches a device.
    let libs_present = unsafe {
        cudarc::driver::sys::is_culib_present() && cudarc::nvrtc::sys::is_culib_present()
    };
    if !libs_present {
        return false;
    }
    device_count() > 0
}

/// Whether `libnccl` can be loaded and a CUDA device is visible, per [`cuda_available`]; never panics, even with no NCCL installation.
#[cfg(feature = "mpi")]
pub fn nccl_available() -> bool {
    if !cuda_available() {
        return false;
    }
    // SAFETY: `is_culib_present` only probes `dlopen`-style candidates; it never touches a device.
    unsafe { cudarc::nccl::sys::is_culib_present() }
}

/// Number of CUDA devices visible to this process, `0` when CUDA is unavailable.
pub fn device_count() -> usize {
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return 0;
    }
    CudaContext::device_count()
        .map(|n| n.max(0) as usize)
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
    pub total_mem: u64,
}

/// [`DeviceInfo`] for every visible device, or [`GpuError::NoDevice`] rather than an empty `Vec`.
pub fn devices() -> Result<Vec<DeviceInfo>, GpuError> {
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return Err(GpuError::LibraryMissing("libcuda"));
    }
    let n = CudaContext::device_count().map_err(GpuError::from)?;
    if n <= 0 {
        return Err(GpuError::NoDevice);
    }
    let mut out = Vec::with_capacity(n as usize);
    for ordinal in 0..n as u32 {
        let ctx = context(ordinal)?;
        let (major, minor) = ctx.compute_capability().map_err(GpuError::from)?;
        out.push(DeviceInfo {
            ordinal,
            name: ctx.name().map_err(GpuError::from)?,
            compute_capability: (major as u32, minor as u32),
            total_mem: ctx.total_mem().map_err(GpuError::from)? as u64,
        });
    }
    Ok(out)
}

/// The context for `ordinal`, created on first use and cached for the process.
pub(crate) fn context(ordinal: u32) -> Result<Arc<CudaContext>, GpuError> {
    static CONTEXTS: Mutex<Vec<(u32, Arc<CudaContext>)>> = Mutex::new(Vec::new());
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return Err(GpuError::LibraryMissing("libcuda"));
    }
    let mut cache = CONTEXTS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, ctx)) = cache.iter().find(|(o, _)| *o == ordinal) {
        return Ok(ctx.clone());
    }
    let ctx = CudaContext::new(ordinal as usize)?;
    cache.push((ordinal, ctx.clone()));
    Ok(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_available_never_panics() {
        let _ = cuda_available();
    }

    #[cfg(feature = "mpi")]
    #[test]
    fn nccl_available_never_panics() {
        let _ = nccl_available();
    }

    /// A `libnccl.so*` file on `LD_LIBRARY_PATH` means the module is loaded; without one the test returns early.
    /// A directory name containing "nccl" is not enough, since the `mpi` build script puts its `OUT_DIR` on the path.
    #[cfg(feature = "mpi")]
    #[test]
    fn nccl_available_true_with_module_on_path() {
        let module_on_path = std::env::var("LD_LIBRARY_PATH")
            .unwrap_or_default()
            .split(':')
            .any(|dir| {
                std::fs::read_dir(dir)
                    .map(|entries| {
                        entries
                            .flatten()
                            .any(|e| e.file_name().to_string_lossy().starts_with("libnccl.so"))
                    })
                    .unwrap_or(false)
            });
        if !module_on_path {
            return;
        }
        crate::require_cuda!();
        assert!(nccl_available());
    }

    #[test]
    fn devices_are_consistent() {
        crate::require_cuda!();
        let list = devices().expect("cuda_available() was true");
        assert_eq!(list.len(), device_count());
        for d in &list {
            assert!(d.compute_capability >= (5, 0));
            assert!(d.total_mem > 0);
        }
    }
}
