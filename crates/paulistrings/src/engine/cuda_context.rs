//! The process-wide CUDA context cache, shared by the partitioned engine's device placement and the CUDA backend.

use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use cudarc::driver::{CudaContext, DriverError};

const LOG_TARGET: &str = "paulistrings::partitioned";

/// Why [`context`] could not produce a context.
#[derive(Debug)]
pub(crate) enum ContextError {
    /// `libcuda` could not be dynamically loaded.
    LibraryMissing,
    /// Creating the context failed.
    Driver(DriverError),
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextError::LibraryMissing => write!(f, "libcuda could not be loaded"),
            ContextError::Driver(error) => write!(f, "CUDA driver error: {error}"),
        }
    }
}

/// The context for `ordinal`, created on first use and cached for the process.
pub(crate) fn context(ordinal: u32) -> Result<Arc<CudaContext>, ContextError> {
    static CONTEXTS: Mutex<Vec<(u32, Arc<CudaContext>)>> = Mutex::new(Vec::new());
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return Err(ContextError::LibraryMissing);
    }
    let mut cache = CONTEXTS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, cuda_context)) = cache.iter().find(|(cached, _)| *cached == ordinal) {
        return Ok(cuda_context.clone());
    }
    let cuda_context = CudaContext::new(ordinal as usize).map_err(ContextError::Driver)?;
    cache.push((ordinal, cuda_context.clone()));
    Ok(cuda_context)
}

/// Make `device`'s CUDA context current on this thread, warning rather than failing like the pinning calls.
pub(crate) fn bind_device_context(device: u32) {
    match context(device) {
        Ok(cuda_context) => {
            if let Err(error) = cuda_context.bind_to_thread() {
                log::warn!(target: LOG_TARGET, "failed to bind device {device} to a partition thread: {error}");
            }
        }
        Err(error) => {
            log::warn!(target: LOG_TARGET, "device {device} for a partition thread: {error}")
        }
    }
}
