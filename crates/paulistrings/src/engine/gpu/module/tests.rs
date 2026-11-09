use super::*;
use cudarc::driver::PushKernelArg;
use num_complex::{Complex32, Complex64};

fn c128() -> CoefficientLayout {
    CoefficientLayout::of::<Complex64>()
}

/// Every coefficient type's layout, `Coefficient::NAME` order.
fn every_layout() -> [CoefficientLayout; 4] {
    [
        CoefficientLayout::of::<f64>(),
        CoefficientLayout::of::<f32>(),
        CoefficientLayout::of::<Complex64>(),
        CoefficientLayout::of::<Complex32>(),
    ]
}

/// The widths a default run compiles every coefficient layout at; the exhaustive tier takes [`EVERY_WIDTH`], minutes on a cold kernel cache.
const DEFAULT_WIDTHS: [usize; 2] = [1, 2];
const EVERY_WIDTH: [usize; 5] = [1, 2, 4, 8, 16];

/// `f` at each of `widths` and every coefficient layout, each on its own thread since a cold one is seconds of NVRTC and module load; results in width-major order.
fn on_every_variant<T: Send>(
    widths: &[usize],
    f: impl Fn(usize, CoefficientLayout) -> T + Sync,
) -> Vec<(usize, CoefficientLayout, T)> {
    let variants: Vec<_> = widths.iter().flat_map(|&w| every_layout().map(|l| (w, l))).collect();
    let f = &f;
    std::thread::scope(|s| {
        let handles: Vec<_> = variants
            .into_iter()
            .map(|(w, layout)| s.spawn(move || (w, layout, f(w, layout))))
            .collect();
        handles.into_iter().map(|h| h.join().expect("variant thread")).collect()
    })
}

/// The visible device's architecture, so a compile shares its PTX-cache entry with [`kernel_set`]'s; `compute_80` without one.
fn test_arch() -> String {
    if !crate::gpu::cuda_available() {
        return "compute_80".to_string();
    }
    let (major, minor) =
        device::context(0).and_then(|c| Ok(c.compute_capability()?)).expect("capability");
    format!("compute_{major}{minor}")
}

#[test]
fn a_layout_has_its_types_bytes_and_name() {
    let layouts = every_layout();
    assert_eq!(layouts.map(CoefficientLayout::bytes), [8, 4, 16, 8]);
    assert_eq!(layouts.map(CoefficientLayout::name), ["f64", "f32", "c128", "c64"]);
}

/// CI-runnable: needs NVRTC only, no device or driver.
#[test]
fn warp_per_bucket_narrows_its_blocks_below_1024_buckets_a_warp() {
    let shape = |b: usize| {
        let c = warp_per_bucket(b);
        (c.grid_dim.0, c.block_dim.0)
    };
    assert_eq!(shape(1), (1, 32));
    assert_eq!(shape(1024), (1024, 32), "one warp a block");
    assert_eq!(shape(3000), (1500, 64), "two warps a block");
    assert_eq!(shape(5000), (1250, 128), "four warps a block");
    assert_eq!(shape(16384), (2048, 256), "eight warps a block, the widest");
    assert_eq!(shape(1 << 20), (131072, 256));
}

fn kernels_compile_at(widths: &[usize]) {
    if !libnvrtc_present() {
        return;
    }
    let arch = test_arch();
    for (w, layout, ptx) in on_every_variant(widths, |w, layout| {
        compile_ptx(w, layout, &arch, &[], kernel_cache::directory())
    }) {
        ptx.unwrap_or_else(|e| panic!("W={w} {}: {e}", layout.name()));
    }
    compile_ptx(2, c128(), &arch, &["-DFINGERPRINT_BITS=8".to_string()], kernel_cache::directory())
        .unwrap_or_else(|e| panic!("FINGERPRINT_BITS=8: {e}"));
}

#[test]
fn kernels_compile() {
    kernels_compile_at(&DEFAULT_WIDTHS);
}

#[test]
#[ignore = "exhaustive: NVRTC at every width and coefficient layout"]
fn kernels_compile_at_every_width() {
    kernels_compile_at(&EVERY_WIDTH);
}

/// Concurrent first requests for one key build it once and share it, so tests starting together on a cold cache do not each compile and load the same kernels.
#[test]
fn concurrent_first_requests_share_one_kernel_set() {
    crate::require_cuda!();
    let options = ["-DPAULISTRINGS_SINGLE_FLIGHT=1".to_string()];
    let barrier = std::sync::Barrier::new(4);
    let sets: Vec<Arc<KernelSet>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| {
                    barrier.wait();
                    kernel_set_with_options(0, 1, c128(), &options).expect("compile+load")
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("requester")).collect()
    });
    assert!(sets.iter().all(|set| Arc::ptr_eq(set, &sets[0])), "each requester built its own set");
}

fn layer_variants_cover_the_record_cap_at(widths: &[usize]) {
    crate::require_cuda!();
    for &w in widths {
        let set = kernel_set(0, w, c128()).expect("compile+load");
        let threads = layer_threads(w) as usize;
        let largest = set.layer.last().expect("at least one variant").items;
        assert_eq!(largest * threads, LAYER_CAP, "W={w}");
        assert!(set.layer.windows(2).all(|p| p[1].items == 2 * p[0].items));
    }
}

#[test]
fn layer_variants_cover_the_record_cap() {
    layer_variants_cover_the_record_cap_at(&DEFAULT_WIDTHS);
}

#[test]
#[ignore = "exhaustive: a kernel set at every width"]
fn layer_variants_cover_the_record_cap_at_every_width() {
    layer_variants_cover_the_record_cap_at(&EVERY_WIDTH);
}

/// Every fused-layer variant honours its `__launch_bounds__` register budget and spills at most a few words; `--nocapture` prints the footprint.
fn layer_variants_fit_their_register_budget_at(widths: &[usize]) {
    crate::require_cuda!();
    use cudarc::driver::sys::CUfunction_attribute::{
        CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, CU_FUNC_ATTRIBUTE_NUM_REGS,
    };
    for (w, layout, set) in on_every_variant(widths, |w, layout| kernel_set(0, w, layout)) {
        let set = set.expect("compile+load");
        let budget = 65_536 / layer_threads(w) as i32;
        let c = layout.name();
        for v in &set.layer {
            for (name, f) in [("serial", &v.serial), ("segscan", &v.segscan)] {
                let regs = f.get_attribute(CU_FUNC_ATTRIBUTE_NUM_REGS).unwrap();
                let local = f.get_attribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES).unwrap();
                println!("W={w} {c} items={} {name}: {regs} regs, {local} B local", v.items);
                assert!(regs <= budget, "W={w} {c} items={} {name}: {regs} regs", v.items);
                assert!(local < 1024, "W={w} {c} items={} {name}: {local} B local", v.items);
            }
        }
    }
}

#[test]
fn layer_variants_fit_their_register_budget() {
    layer_variants_fit_their_register_budget_at(&DEFAULT_WIDTHS);
}

#[test]
#[ignore = "exhaustive: a kernel set at every width and coefficient layout"]
fn layer_variants_fit_their_register_budget_at_every_width() {
    layer_variants_fit_their_register_budget_at(&EVERY_WIDTH);
}

/// A device with less opt-in shared memory loads fewer variants, and the loader never fails on it.
#[test]
fn a_small_shared_memory_limit_loads_fewer_variants() {
    crate::require_cuda!();
    let set = kernel_set_with_options(0, 2, c128(), &["-DTEST_SHARED_LIMIT=40000".to_string()])
        .expect("load");
    assert_eq!(set.layer.len(), 2);
    assert_eq!(set.layer_cap(), 2048);
    assert!(matches!(
        kernel_set_with_options(0, 2, c128(), &["-DTEST_SHARED_LIMIT=1000".to_string()]),
        Err(GpuError::Unsupported(_))
    ));
    let full = kernel_set(0, 2, c128()).expect("load");
    assert_eq!(full.layer_cap(), LAYER_CAP);
}

/// `layer_shared_bytes` is the byte count `carve` in `kernels/layer.cu` lays out, read back from the `-DTEST_CARVE_BYTES` hook.
fn layer_shared_bytes_matches_the_carve_layout_at(widths: &[usize]) {
    crate::require_cuda!();
    let context = device::context(0).expect("context");
    let (major, minor) = context.compute_capability().expect("capability");
    let stream = context.default_stream();
    let arch = format!("compute_{major}{minor}");
    let options = ["-DTEST_CARVE_BYTES".to_string()];
    let loaded = on_every_variant(widths, |w, layout| {
        let ptx =
            compile_ptx(w, layout, &arch, &options, kernel_cache::directory()).expect("compile");
        context.load_module(ptx).and_then(|module| module.load_function("k_carve_bytes"))
    });
    for (w, layout, carve_bytes) in loaded {
        let carve_bytes = carve_bytes.expect("load");
        let threads = layer_threads(w) as usize;
        for capacity in [2, 1000, threads, 2 * threads, 4096, LAYER_CAP] {
            let mut out = stream.alloc_zeros::<u64>(1).unwrap();
            let n_cap = capacity as u32;
            // SAFETY: arguments match `k_carve_bytes` in layer.cu; `out` holds one entry.
            unsafe {
                stream
                    .launch_builder(&carve_bytes)
                    .arg(&n_cap)
                    .arg(&mut out)
                    .launch(thread_per(1, 1))
            }
            .unwrap();
            let carved = stream.clone_dtoh(&out).unwrap()[0];
            assert_eq!(
                carved,
                u64::from(layer_shared_bytes(capacity, w, layout)),
                "W={w} {} capacity={capacity}",
                layout.name()
            );
        }
    }
}

#[test]
fn layer_shared_bytes_matches_the_carve_layout() {
    layer_shared_bytes_matches_the_carve_layout_at(&DEFAULT_WIDTHS);
}

#[test]
#[ignore = "exhaustive: an NVRTC compile and module load at every width and coefficient layout"]
fn layer_shared_bytes_matches_the_carve_layout_at_every_width() {
    layer_shared_bytes_matches_the_carve_layout_at(&EVERY_WIDTH);
}

/// The options key the kernel cache's entries, so whatever selects a different kernel must appear in them.
#[test]
fn compile_options_carry_the_coefficient_type_and_extra_options() {
    let base = compile_options(2, c128(), "compute_80", &[]).options;
    let f32_options = compile_options(2, CoefficientLayout::of::<f32>(), "compute_80", &[]).options;
    let extra = ["-DFINGERPRINT_BITS=8".to_string()];
    assert_ne!(f32_options, base, "coefficient type");
    assert_ne!(compile_options(2, c128(), "compute_80", &extra).options, base, "extra option");
    assert_ne!(compile_options(4, c128(), "compute_80", &[]).options, base, "width");
}

#[test]
fn any_arch_reaches_nvrtc_as_an_option() {
    let options = compile_options(1, CoefficientLayout::of::<Complex64>(), "compute_121", &[]);
    assert_eq!(options.arch, None);
    assert_eq!(options.options[0], "--gpu-architecture=compute_121");
}
