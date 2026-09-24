//! Runtime device probe and the process-wide device table.
//!
//! `cudarc`'s own `culib()`/`device_count()` panic when the driver library is absent, so every
//! function here checks `is_culib_present()` first and never reaches them on a library-less box.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::CudaContext;

use super::error::GpuError;

/// Whether both `libcuda` and `libnvrtc` are present *and* at least one device answers.
///
/// Safe to call on a box with no GPU and no CUDA installation at all: the presence checks run
/// before anything that would panic, and every step after them is itself a `Result`.
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

/// [`DeviceInfo`] for every visible device.
///
/// Returns [`GpuError::NoDevice`] rather than an empty `Vec` when there is nothing to report, so a
/// caller cannot mistake "no CUDA at all" for "zero devices, but CUDA is fine".
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
        let ctx = DEVICE_TABLE.get(ordinal)?;
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

/// Process-wide cache of one [`CudaContext`] per ordinal, so repeated calls do not re-bind.
static DEVICE_TABLE: DeviceTable = DeviceTable::new();

struct DeviceTable(OnceLock<Mutex<HashMap<u32, Arc<CudaContext>>>>);

impl DeviceTable {
    const fn new() -> Self {
        Self(OnceLock::new())
    }

    fn contexts(&self) -> &Mutex<HashMap<u32, Arc<CudaContext>>> {
        self.0.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// The [`Arc<CudaContext>`] for `ordinal`, creating and caching it on first use.
    fn get(&self, ordinal: u32) -> Result<Arc<CudaContext>, GpuError> {
        let mut map = self.contexts().lock().expect("device table mutex poisoned");
        if let Some(ctx) = map.get(&ordinal) {
            return Ok(ctx.clone());
        }
        let ctx = CudaContext::new(ordinal as usize).map_err(GpuError::from)?;
        map.insert(ordinal, ctx.clone());
        Ok(ctx)
    }
}

/// The cached context for `ordinal`, for the modules that bind one.
pub(crate) fn context(ordinal: u32) -> Result<Arc<CudaContext>, GpuError> {
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return Err(GpuError::LibraryMissing("libcuda"));
    }
    DEVICE_TABLE.get(ordinal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_available_never_panics() {
        let _ = cuda_available();
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
