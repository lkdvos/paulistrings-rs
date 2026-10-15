//! NVRTC kernel compilation, cached per `(ordinal, W, extra options)`.
//!
//! Every `.cu` family in [`KERNEL_SOURCES`] is concatenated behind `kernels/prelude.cuh` into one translation unit, and [`KernelSet`] holds the loaded functions.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute};
use cudarc::driver::{CudaFunction, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use super::device;
use super::error::GpuError;
use super::kernel_cache;
use super::truncation::{KEEP_NODES, OP_AND, OP_COEFF, OP_KEEP, OP_OR, OP_WEIGHT};
use super::GPU_LOG_TARGET;
use crate::engine::cuda_context::libnvrtc_present;

const PRELUDE: &str = include_str!("kernels/prelude.cuh");
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
const PERMUTE: &str = include_str!("kernels/permute.cu");
const TRUNCATE: &str = include_str!("kernels/truncate.cu");

/// Every kernel family, concatenated into one NVRTC translation unit; later families use earlier ones' device functions.
const KERNEL_SOURCES: &[&str] = &[
    PRELUDE,
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
    PERMUTE,
    TRUNCATE,
];

/// Records per fused-layer block at the full opt-in shared memory; `-DLAYER_CAP` to the kernels.
const LAYER_CAP: usize = 8192;

/// Test hook: an extra option `-DTEST_SHARED_LIMIT=<bytes>` caps the opt-in shared memory the loader assumes, which is inert to NVRTC.
const TEST_SHARED_LIMIT: &str = "-DTEST_SHARED_LIMIT=";

/// Source-bucket rows the 12-bit tag offset of a fused-layer record can address; `-DMAX_BUCKET_LEN` to the kernels.
pub(crate) const MAX_BUCKET_LEN: usize = 1 << 12;

/// Fused-layer block width for `w`, register-bound (ARCHITECTURE.md §GPU); `-DTHREADS` to the kernels.
pub(crate) fn layer_threads(w: usize) -> u32 {
    match w {
        1 | 2 => 1024,
        4 => 512,
        _ => 256,
    }
}

/// Dynamic shared bytes the fused layer needs for `capacity` records, the `carve` layout in `kernels/layer.cu`.
pub(crate) fn layer_shared_bytes(capacity: usize, w: usize) -> u32 {
    (11 * capacity + 144 + 4096 + 256 * w + 128 + 320 + 72 + 16 + 640) as u32
}

/// One warp per bucket over `b` buckets, eight warps a block.
pub(crate) fn warp_per_bucket(b: usize) -> LaunchConfig {
    thread_per(b * 32, 256)
}

/// One thread per element over `n` elements, `threads` a block.
pub(crate) fn thread_per(n: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(threads).max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One compiled fused-layer variant: `items` records per thread, so `items * layer_threads(W)` records per block.
pub(crate) struct LayerVariant {
    pub(crate) items: usize,
    pub(crate) serial: CudaFunction,
    pub(crate) segscan: CudaFunction,
}

/// One device's compiled kernels for one `W`; each `CudaFunction` keeps its `CudaModule` alive.
pub(crate) struct KernelSet {
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
    pub(crate) permute_count: CudaFunction,
    pub(crate) permute_lengths: CudaFunction,
    pub(crate) permute_scatter: CudaFunction,
    pub(crate) octave_histogram: CudaFunction,
    pub(crate) radix_histogram: CudaFunction,
    pub(crate) radix_extract: CudaFunction,
    pub(crate) boundary_counts: CudaFunction,
    pub(crate) retain: CudaFunction,
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

#[cfg(any(test, feature = "test-utils"))]
thread_local! {
    /// Test hook: NVRTC compilations on this thread, so a cache-hit test is not perturbed by compiles in concurrent tests.
    pub(crate) static NVRTC_COMPILES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Compiled NVRTC PTX for `w` at `arch` (`compute_<major><minor>`), with `extra_options` appended (the `-DFINGERPRINT_BITS=<b>` hook), through the PTX cache in `cache`; needs only NVRTC, no device.
pub(crate) fn compile_ptx(
    w: usize,
    arch: &str,
    extra_options: &[String],
    cache: Option<&Path>,
) -> Result<cudarc::nvrtc::Ptx, GpuError> {
    if !libnvrtc_present() {
        return Err(GpuError::LibraryMissing("libnvrtc"));
    }
    let source = KERNEL_SOURCES.concat();
    let mut options = vec![
        format!("-DW={w}"),
        format!("-DTHREADS={}", layer_threads(w)),
        format!("-DLAYER_CAP={LAYER_CAP}"),
        format!("-DMAX_BUCKET_LEN={MAX_BUCKET_LEN}u"),
        format!("-DKEEP_NODES={KEEP_NODES}"),
        format!("-DOP_KEEP={OP_KEEP}"),
        format!("-DOP_COEFF={OP_COEFF}"),
        format!("-DOP_WEIGHT={OP_WEIGHT}"),
        format!("-DOP_AND={OP_AND}"),
        format!("-DOP_OR={OP_OR}"),
        "--std=c++17".to_string(),
    ];
    options.extend_from_slice(extra_options);
    let compile_options = CompileOptions {
        arch: Some(arch_static(arch)),
        fmad: Some(false),
        options,
        ..Default::default()
    };
    if let Some(ptx) = cache.and_then(|dir| kernel_cache::lookup(dir, &source, &compile_options)) {
        log::debug!(target: GPU_LOG_TARGET, "kernels W={w} {arch}: kernel cache hit");
        return Ok(ptx);
    }
    #[cfg(any(test, feature = "test-utils"))]
    NVRTC_COMPILES.with(|count| count.set(count.get() + 1));
    let start = std::time::Instant::now();
    let ptx = compile_ptx_with_opts(source.clone(), compile_options.clone()).map_err(|e| {
        GpuError::Compile {
            width: w,
            log: e.to_string(),
        }
    })?;
    log::debug!(
        target: GPU_LOG_TARGET,
        "kernels W={w} {arch}: NVRTC compile in {:.2} s",
        start.elapsed().as_secs_f64()
    );
    if let Some(dir) = cache {
        kernel_cache::store(dir, &source, &compile_options, &ptx.to_src());
    }
    Ok(ptx)
}

/// Virtual architectures from Volta to Blackwell, which [`arch_static`] hands out without allocating.
const KNOWN_ARCHES: [&str; 10] = [
    "compute_70",
    "compute_72",
    "compute_75",
    "compute_80",
    "compute_86",
    "compute_87",
    "compute_89",
    "compute_90",
    "compute_100",
    "compute_120",
];

/// `CompileOptions::arch` wants `&'static str`; an unlisted arch is leaked, once per compile, which the per-device kernel cache bounds.
fn arch_static(arch: &str) -> &'static str {
    KNOWN_ARCHES
        .iter()
        .find(|known| **known == arch)
        .copied()
        .unwrap_or_else(|| Box::leak(arch.to_string().into_boxed_str()))
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
    let context = device::context(ordinal)?;
    let (major, minor) = context.compute_capability().map_err(GpuError::from)?;
    let arch = format!("compute_{major}{minor}");
    let ptx = compile_ptx(w, &arch, extra_options, kernel_cache::directory())?;
    let module = context.load_module(ptx).map_err(GpuError::from)?;
    let load = |name: &str| module.load_function(name).map_err(GpuError::from);
    let threads = layer_threads(w) as usize;
    let mut limit = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        .map_err(GpuError::from)?
        .max(0) as usize;
    if let Some(cap) = extra_options
        .iter()
        .find_map(|option| option.strip_prefix(TEST_SHARED_LIMIT))
        .and_then(|value| value.parse::<usize>().ok())
    {
        limit = limit.min(cap);
    }
    let mut layer = Vec::new();
    let mut items = 1usize;
    while items * threads <= LAYER_CAP && layer_shared_bytes(items * threads, w) as usize <= limit {
        let shared_bytes = layer_shared_bytes(items * threads, w) as i32;
        let serial = load(&format!("k_layer_serial_{items}"))?;
        let segscan = load(&format!("k_layer_segscan_{items}"))?;
        // Opt-in shared memory above the 48 KB default; the attribute is per function.
        for func in [&serial, &segscan] {
            func.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes,
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
        fingerprint: load("k_fingerprint")?,
        scan_block: load("k_scan_block")?,
        scan_single: load("k_scan_single")?,
        scan_add: load("k_scan_add")?,
        refine_count: load("k_refine_count")?,
        refine_scatter: load("k_refine_scatter")?,
        check_invariants: load("k_check_invariants")?,
        count: load("k_count")?,
        rows: load("k_rows")?,
        export_counts: load("k_export_counts")?,
        export_fill: load("k_export_fill")?,
        premerge_counts: load("k_premerge_counts")?,
        premerge_split: load("k_premerge_split")?,
        premerge_copy: load("k_premerge_copy")?,
        compact: load("k_compact")?,
        rescale: load("k_rescale")?,
        permute_count: load("k_permute_count")?,
        permute_lengths: load("k_permute_lengths")?,
        permute_scatter: load("k_permute_scatter")?,
        octave_histogram: load("k_octave_histogram")?,
        radix_histogram: load("k_radix_histogram")?,
        radix_extract: load("k_radix_extract")?,
        boundary_counts: load("k_boundary_counts")?,
        retain: load("k_retain")?,
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
mod tests;
