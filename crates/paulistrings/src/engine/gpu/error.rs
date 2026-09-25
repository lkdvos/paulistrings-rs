//! The CUDA backend's error type. Never a panic path — every device or library absence is a `Result`.

use std::fmt;

/// Everything that can go wrong talking to a CUDA device.
#[derive(Debug)]
pub enum GpuError {
    /// No CUDA device is visible to this process.
    NoDevice,
    /// `libcuda` or `libnvrtc` could not be dynamically loaded.
    LibraryMissing(&'static str),
    /// A CUDA driver call failed.
    Driver(cudarc::driver::DriverError),
    /// NVRTC compilation failed.
    Compile {
        /// The width the compilation was for.
        w: usize,
        /// NVRTC's compile log.
        log: String,
    },
    /// A device allocation ran out of memory.
    OutOfMemory {
        /// Which device ordinal.
        device: u32,
        /// The requested size, `0` when unknown.
        bytes: u64,
    },
    /// A feature this backend does not implement.
    Unsupported(&'static str),
    /// A device placement did not resolve.
    Topology(crate::engine::partitioned::TopologyError),
    /// A partitioned split whose partition `rank` failed at layer `layer` of an earlier call; its partitions no longer hold one consistent sum, so every later call is refused until the caller scatters again.
    Poisoned {
        /// The partition that failed.
        rank: usize,
        /// The layer index within that call, in application order.
        layer: usize,
    },
    /// An NCCL call reported a non-success result.
    #[cfg(feature = "nccl")]
    Nccl {
        /// The `ncclResult_t` value, as an `i32`.
        code: i32,
        /// What was being attempted.
        what: String,
    },
    /// A bounded wait on a device or communicator operation exceeded its deadline; see `PAULISTRINGS_NCCL_TIMEOUT_S`.
    #[cfg(feature = "nccl")]
    Timeout {
        /// What was being waited on.
        what: &'static str,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoDevice => write!(f, "no CUDA device is visible to this process"),
            GpuError::LibraryMissing(lib) => write!(f, "{lib} could not be loaded"),
            GpuError::Driver(e) => write!(f, "CUDA driver error: {e}"),
            GpuError::Compile { w, log } => write!(f, "NVRTC compilation failed at W={w}: {log}"),
            GpuError::OutOfMemory { device, bytes } => {
                write!(f, "device {device} out of memory (requested {bytes} bytes)")
            }
            GpuError::Unsupported(what) => write!(f, "unsupported on the CUDA backend: {what}"),
            GpuError::Topology(e) => write!(f, "device placement: {e}"),
            GpuError::Poisoned { rank, layer } => write!(
                f,
                "device partition {rank} failed at layer {layer} of an earlier propagate; scatter again"
            ),
            #[cfg(feature = "nccl")]
            GpuError::Nccl { code, what } => write!(f, "NCCL error {code} during {what}"),
            #[cfg(feature = "nccl")]
            GpuError::Timeout { what } => write!(f, "timed out waiting on {what}"),
        }
    }
}

impl std::error::Error for GpuError {}

impl GpuError {
    /// `e` with an out-of-memory result attributed to `device` and a request of `bytes`.
    pub(crate) fn from_alloc(e: cudarc::driver::DriverError, device: u32, bytes: u64) -> Self {
        match GpuError::from(e) {
            GpuError::OutOfMemory { .. } => GpuError::OutOfMemory { device, bytes },
            other => other,
        }
    }
}

impl From<cudarc::driver::DriverError> for GpuError {
    /// `CUDA_ERROR_OUT_OF_MEMORY` maps to [`GpuError::OutOfMemory`] with `device = 0` and `bytes = 0`
    /// (the driver error carries neither); a caller that knows better should build the variant itself.
    fn from(e: cudarc::driver::DriverError) -> Self {
        if e.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
            GpuError::OutOfMemory {
                device: 0,
                bytes: 0,
            }
        } else {
            GpuError::Driver(e)
        }
    }
}
