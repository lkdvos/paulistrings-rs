//! NVRTC kernel compilation, cached per `(ordinal, W, extra options)`.
//!
//! Every `.cu` family in [`KERNEL_SOURCES`] is concatenated behind `kernels/prelude.cuh` into one translation unit, and [`KernelSet`] holds the loaded functions.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute};
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
const COUNT: &str = include_str!("kernels/count.cu");
const LAYER: &str = include_str!("kernels/layer.cu");
const EXPORT: &str = include_str!("kernels/export.cu");
const COMPACT: &str = include_str!("kernels/compact.cu");
const RESCALE: &str = include_str!("kernels/rescale.cu");
const TRUNCATE: &str = include_str!("kernels/truncate.cu");

/// Every kernel family, concatenated into one NVRTC translation unit; later families use earlier ones' device functions.
const KERNEL_SOURCES: &[&str] = &[
    PRELUDE,
    PROBE,
    HASH,
    FINGERPRINT,
    SCAN,
    REFINE,
    INVARIANTS,
    COUNT,
    EXPORT,
    LAYER,
    COMPACT,
    RESCALE,
    TRUNCATE,
];

/// Records per fused-layer block at the full opt-in shared memory; must match `CAP` in `kernels/prelude.cuh`.
/// A device with a smaller opt-in limit loads fewer variants and [`KernelSet::layer_cap`] is lower.
pub(crate) const LAYER_CAP: usize = 8192;

/// Test hook: an extra option `-DTEST_SHARED_LIMIT=<bytes>` caps the opt-in shared memory the loader assumes, which is inert to NVRTC.
const TEST_SHARED_LIMIT: &str = "-DTEST_SHARED_LIMIT=";

/// Source-bucket rows the 12-bit tag offset can address; must match `MAX_BUCKET_LEN` in `kernels/prelude.cuh`.
pub(crate) const MAX_BUCKET_LEN: usize = 1 << 12;

/// Fused-layer block width for `w`; must match `THREADS` in `kernels/prelude.cuh`.
pub(crate) fn layer_threads(w: usize) -> u32 {
    match w {
        1 | 2 => 1024,
        4 => 512,
        _ => 256,
    }
}

/// Dynamic shared bytes the fused layer needs for `n_cap` records, the `carve` layout in `kernels/layer.cu`.
pub(crate) fn layer_shared_bytes(n_cap: usize, w: usize) -> u32 {
    (11 * n_cap + 144 + 4096 + 256 * w + 128 + 320 + 72 + 16 + 640) as u32
}

/// One compiled fused-layer variant: `items` records per thread, so `items * layer_threads(W)` records per block.
pub(crate) struct LayerVariant {
    pub(crate) items: usize,
    pub(crate) serial: CudaFunction,
    pub(crate) segscan: CudaFunction,
}

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
    pub(crate) count: CudaFunction,
    pub(crate) rows: CudaFunction,
    pub(crate) export_counts: CudaFunction,
    pub(crate) export_fill: CudaFunction,
    pub(crate) premerge_counts: CudaFunction,
    pub(crate) premerge_split: CudaFunction,
    pub(crate) premerge_copy: CudaFunction,
    pub(crate) compact: CudaFunction,
    pub(crate) rescale: CudaFunction,
    pub(crate) octave_hist: CudaFunction,
    pub(crate) retain: CudaFunction,
    pub(crate) radix_hist: CudaFunction,
    pub(crate) radix_extract: CudaFunction,
    pub(crate) topn_counts: CudaFunction,
    pub(crate) retain_topn: CudaFunction,
    /// Ascending by `items`; the smallest whose capacity covers a layer's largest segment is launched.
    pub(crate) layer: Vec<LayerVariant>,
    threads: usize,
}

impl KernelSet {
    /// Records per fused block the largest loaded variant holds; `LAYER_CAP` unless the device's opt-in shared memory is smaller.
    pub(crate) fn layer_cap(&self) -> usize {
        self.layer.last().map_or(0, |v| v.items * self.threads)
    }
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
    let threads = layer_threads(w) as usize;
    let mut limit = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        .map_err(GpuError::from)?
        .max(0) as usize;
    if let Some(cap) = extra_options
        .iter()
        .find_map(|o| o.strip_prefix(TEST_SHARED_LIMIT))
        .and_then(|v| v.parse::<usize>().ok())
    {
        limit = limit.min(cap);
    }
    let mut layer = Vec::new();
    let mut items = 1usize;
    while items * threads <= LAYER_CAP && layer_shared_bytes(items * threads, w) as usize <= limit {
        let smem = layer_shared_bytes(items * threads, w) as i32;
        let serial = f(&format!("k_layer_serial_{items}"))?;
        let segscan = f(&format!("k_layer_segscan_{items}"))?;
        // Opt-in shared memory above the 48 KB default; the attribute is per function.
        for func in [&serial, &segscan] {
            func.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem,
            )
            .map_err(GpuError::from)?;
        }
        layer.push(LayerVariant {
            items,
            serial,
            segscan,
        });
        items *= 2;
    }
    if layer.is_empty() {
        return Err(GpuError::Unsupported(
            "device opt-in shared memory too small for the fused layer",
        ));
    }
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
        count: f("k_count")?,
        rows: f("k_rows")?,
        export_counts: f("k_export_counts")?,
        export_fill: f("k_export_fill")?,
        premerge_counts: f("k_premerge_counts")?,
        premerge_split: f("k_premerge_split")?,
        premerge_copy: f("k_premerge_copy")?,
        compact: f("k_compact")?,
        rescale: f("k_rescale")?,
        octave_hist: f("k_octave_hist")?,
        retain: f("k_retain")?,
        radix_hist: f("k_radix_hist")?,
        radix_extract: f("k_radix_extract")?,
        topn_counts: f("k_topn_counts")?,
        retain_topn: f("k_retain_topn")?,
        layer,
        threads,
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
    fn layer_variants_cover_the_record_cap_at_every_width() {
        crate::require_cuda!();
        for w in [1usize, 2, 4, 8, 16] {
            let set = kernel_set(0, w).expect("compile+load");
            let threads = layer_threads(w) as usize;
            let largest = set.layer.last().expect("at least one variant").items;
            assert_eq!(largest * threads, LAYER_CAP, "W={w}");
            assert!(set.layer.windows(2).all(|p| p[1].items == 2 * p[0].items));
        }
    }

    /// Every fused-layer variant honours its `__launch_bounds__` register budget and spills at most a few words; `--nocapture` prints the footprint.
    #[test]
    fn layer_variants_fit_their_register_budget() {
        crate::require_cuda!();
        use cudarc::driver::sys::CUfunction_attribute::{
            CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, CU_FUNC_ATTRIBUTE_NUM_REGS,
        };
        for w in [1usize, 2, 4, 8, 16] {
            let set = kernel_set(0, w).expect("compile+load");
            let budget = 65_536 / layer_threads(w) as i32;
            for v in &set.layer {
                for (name, f) in [("serial", &v.serial), ("segscan", &v.segscan)] {
                    let regs = f.get_attribute(CU_FUNC_ATTRIBUTE_NUM_REGS).unwrap();
                    let local = f.get_attribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES).unwrap();
                    println!(
                        "W={w} items={} {name}: {regs} regs, {local} B local",
                        v.items
                    );
                    assert!(
                        regs <= budget,
                        "W={w} items={} {name}: {regs} regs",
                        v.items
                    );
                    assert!(
                        local < 1024,
                        "W={w} items={} {name}: {local} B local",
                        v.items
                    );
                }
            }
        }
    }

    /// A device with less opt-in shared memory loads fewer variants, and the loader never fails on it.
    #[test]
    fn a_small_shared_memory_limit_loads_fewer_variants() {
        crate::require_cuda!();
        let set = kernel_set_with_options(0, 2, &["-DTEST_SHARED_LIMIT=40000".to_string()])
            .expect("load");
        assert_eq!(set.layer.len(), 2);
        assert_eq!(set.layer_cap(), 2048);
        assert!(matches!(
            kernel_set_with_options(0, 2, &["-DTEST_SHARED_LIMIT=1000".to_string()]),
            Err(GpuError::Unsupported(_))
        ));
        let full = kernel_set(0, 2).expect("load");
        assert_eq!(full.layer_cap(), LAYER_CAP);
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
