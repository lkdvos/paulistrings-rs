//! NVRTC kernel compilation, cached per `(ordinal, W)`.
//!
//! Every `.cu` family in [`KERNEL_SOURCES`] is concatenated behind `kernels/prelude.cuh` into one translation unit, and [`KernelSet`] holds the loaded functions.
// The scaffold's only caller is its own test; the allow goes when the device storage takes a `KernelSet`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::CudaFunction;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::device;
use super::error::GpuError;

const PRELUDE: &str = include_str!("kernels/prelude.cuh");
const PROBE: &str = include_str!("kernels/probe.cu");

/// Every kernel family, concatenated into one NVRTC translation unit.
const KERNEL_SOURCES: &[&str] = &[PRELUDE, PROBE];

/// One device's compiled kernels for one `W`. `CudaFunction` itself keeps its owning
/// `CudaModule` alive, so there is nothing else to hold here.
pub(crate) struct KernelSet {
    probe: CudaFunction,
}

impl KernelSet {
    pub(crate) fn probe(&self) -> &CudaFunction {
        &self.probe
    }
}

/// Compiled NVRTC PTX for `w`, at `arch` (`compute_<major><minor>`), with an optional extra
/// options hook (`extra_options`) — a future `FP_BITS` knob plugs in here without a signature
/// change. Needs only NVRTC, no device or context.
pub(crate) fn compile_ptx(
    w: usize,
    arch: &str,
    extra_options: &[String],
) -> Result<cudarc::nvrtc::Ptx, GpuError> {
    let src = KERNEL_SOURCES.concat();
    let mut options = vec![format!("-DW={w}"), "--std=c++17".to_string()];
    options.extend_from_slice(extra_options);
    compile_ptx_with_opts(
        src,
        CompileOptions {
            arch: Some(arch_static(arch)),
            fmad: Some(false),
            options,
            ..Default::default()
        },
    )
    .map_err(|e| GpuError::Compile {
        w,
        log: e.to_string(),
    })
}

/// `arch` strings are always one of a small fixed set (`compute_50`..`compute_90`), so this leaks
/// nothing unusual: `CompileOptions::arch` wants `&'static str`, and every caller here passes a
/// string built once per test/probe run, not a hot-path allocation.
fn arch_static(arch: &str) -> &'static str {
    match arch {
        "compute_80" => "compute_80",
        "compute_86" => "compute_86",
        other => Box::leak(other.to_string().into_boxed_str()),
    }
}

type KernelCache = Mutex<HashMap<(u32, usize), Arc<KernelSet>>>;
static KERNEL_CACHE: OnceLock<KernelCache> = OnceLock::new();

fn cache() -> &'static KernelCache {
    KERNEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The compiled [`KernelSet`] for `(ordinal, w)`, compiling and caching it on first use.
pub(crate) fn kernel_set(ordinal: u32, w: usize) -> Result<Arc<KernelSet>, GpuError> {
    if let Some(set) = cache()
        .lock()
        .expect("kernel cache mutex poisoned")
        .get(&(ordinal, w))
    {
        return Ok(set.clone());
    }
    let ctx = device::context(ordinal)?;
    let (major, minor) = ctx.compute_capability().map_err(GpuError::from)?;
    let arch = format!("compute_{major}{minor}");
    let ptx = compile_ptx(w, &arch, &[])?;
    let module = ctx.load_module(ptx).map_err(GpuError::from)?;
    let probe = module.load_function("k_probe").map_err(GpuError::from)?;
    let set = Arc::new(KernelSet { probe });
    cache()
        .lock()
        .expect("kernel cache mutex poisoned")
        .insert((ordinal, w), set.clone());
    Ok(set)
}

#[cfg(test)]
mod tests {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    use super::*;

    /// CI-runnable: needs NVRTC only, no device or driver.
    #[test]
    fn kernels_compile_for_every_width() {
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return;
        }
        for w in [1usize, 2, 4, 8, 16] {
            compile_ptx(w, "compute_80", &[]).unwrap_or_else(|e| panic!("W={w}: {e}"));
        }
    }

    #[test]
    fn probe_kernel_runs_for_every_width() {
        crate::require_cuda!();
        for w in [1usize, 2, 4, 8, 16] {
            let set = kernel_set(0, w).expect("compile+load");
            let ctx = device::context(0).expect("device context");
            let stream = ctx.default_stream();
            let n: usize = 32;
            let mut out = stream.alloc_zeros::<u64>(n).expect("alloc");
            unsafe {
                stream
                    .launch_builder(set.probe())
                    .arg(&mut out)
                    .arg(&(n as i32))
                    .launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (n as u32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .expect("launch");
            }
            let host = stream.clone_dtoh(&out).expect("d2h");
            for (i, &v) in host.iter().enumerate() {
                assert_eq!(v, (i as u64) * (w as u64), "W={w} i={i}");
            }
        }
    }
}
