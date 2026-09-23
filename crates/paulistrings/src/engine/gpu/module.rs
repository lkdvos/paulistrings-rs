//! NVRTC kernel compilation, cached per `(ordinal, W, extra options)`.
//!
//! Every `.cu` family in [`KERNEL_SOURCES`] is concatenated behind `kernels/prelude.cuh` into one translation unit, and [`KernelSet`] holds the loaded functions.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::CudaFunction;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::device;
use super::error::GpuError;

const PRELUDE: &str = include_str!("kernels/prelude.cuh");
const PROBE: &str = include_str!("kernels/probe.cu");
const HASH: &str = include_str!("kernels/hash.cu");
const FINGERPRINT: &str = include_str!("kernels/fingerprint.cu");
const SCAN: &str = include_str!("kernels/scan.cu");
const REFINE: &str = include_str!("kernels/refine.cu");
const INVARIANTS: &str = include_str!("kernels/invariants.cu");

/// Every kernel family, concatenated into one NVRTC translation unit; later families use earlier ones' device functions.
const KERNEL_SOURCES: &[&str] = &[PRELUDE, PROBE, HASH, FINGERPRINT, SCAN, REFINE, INVARIANTS];

/// One device's compiled kernels for one `W`; each `CudaFunction` keeps its `CudaModule` alive.
pub(crate) struct KernelSet {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) probe: CudaFunction,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) bucket_of: CudaFunction,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) partition_of: CudaFunction,
    pub(crate) fingerprint: CudaFunction,
    pub(crate) scan_block: CudaFunction,
    pub(crate) scan_single: CudaFunction,
    pub(crate) scan_add: CudaFunction,
    pub(crate) refine_count: CudaFunction,
    pub(crate) refine_scatter: CudaFunction,
    pub(crate) check_invariants: CudaFunction,
}

/// Compiled NVRTC PTX for `w` at `arch` (`compute_<major><minor>`), with `extra_options` appended (the `-DFP_BITS=<b>` hook). Needs only NVRTC, no device.
pub(crate) fn compile_ptx(
    w: usize,
    arch: &str,
    extra_options: &[String],
) -> Result<cudarc::nvrtc::Ptx, GpuError> {
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(GpuError::LibraryMissing("libnvrtc"));
    }
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

/// `CompileOptions::arch` wants `&'static str`; the arch set is small and fixed, so leaking an unlisted one is bounded.
fn arch_static(arch: &str) -> &'static str {
    match arch {
        "compute_80" => "compute_80",
        "compute_86" => "compute_86",
        other => Box::leak(other.to_string().into_boxed_str()),
    }
}

type CacheKey = (u32, usize, Vec<String>);
type KernelCache = Mutex<HashMap<CacheKey, Arc<KernelSet>>>;
static KERNEL_CACHE: OnceLock<KernelCache> = OnceLock::new();

fn cache() -> &'static KernelCache {
    KERNEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The compiled [`KernelSet`] for `(ordinal, w)` with no extra options.
pub(crate) fn kernel_set(ordinal: u32, w: usize) -> Result<Arc<KernelSet>, GpuError> {
    kernel_set_with_options(ordinal, w, &[])
}

/// The compiled [`KernelSet`] for `(ordinal, w, extra_options)`, compiling and caching it on first use.
pub(crate) fn kernel_set_with_options(
    ordinal: u32,
    w: usize,
    extra_options: &[String],
) -> Result<Arc<KernelSet>, GpuError> {
    let key = (ordinal, w, extra_options.to_vec());
    if let Some(set) = cache()
        .lock()
        .expect("kernel cache mutex poisoned")
        .get(&key)
    {
        return Ok(set.clone());
    }
    let ctx = device::context(ordinal)?;
    let (major, minor) = ctx.compute_capability().map_err(GpuError::from)?;
    let arch = format!("compute_{major}{minor}");
    let ptx = compile_ptx(w, &arch, extra_options)?;
    let module = ctx.load_module(ptx).map_err(GpuError::from)?;
    let f = |name: &str| module.load_function(name).map_err(GpuError::from);
    let set = Arc::new(KernelSet {
        probe: f("k_probe")?,
        bucket_of: f("k_bucket_of")?,
        partition_of: f("k_partition_of")?,
        fingerprint: f("k_fingerprint")?,
        scan_block: f("k_scan_block")?,
        scan_single: f("k_scan_single")?,
        scan_add: f("k_scan_add")?,
        refine_count: f("k_refine_count")?,
        refine_scatter: f("k_refine_scatter")?,
        check_invariants: f("k_check_invariants")?,
    });
    cache()
        .lock()
        .expect("kernel cache mutex poisoned")
        .insert(key, set.clone());
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
        compile_ptx(2, "compute_80", &["-DFP_BITS=8".to_string()])
            .unwrap_or_else(|e| panic!("FP_BITS=8: {e}"));
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
                    .launch_builder(&set.probe)
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
