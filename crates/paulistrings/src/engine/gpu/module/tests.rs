use super::*;
use cudarc::driver::PushKernelArg;

/// CI-runnable: needs NVRTC only, no device or driver.
#[test]
fn kernels_compile_for_every_width() {
    if !libnvrtc_present() {
        return;
    }
    for w in [1usize, 2, 4, 8, 16] {
        compile_ptx(w, "compute_80", &[], kernel_cache::directory())
            .unwrap_or_else(|e| panic!("W={w}: {e}"));
    }
    compile_ptx(
        2,
        "compute_80",
        &["-DFINGERPRINT_BITS=8".to_string()],
        kernel_cache::directory(),
    )
    .unwrap_or_else(|e| panic!("FINGERPRINT_BITS=8: {e}"));
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
    let set =
        kernel_set_with_options(0, 2, &["-DTEST_SHARED_LIMIT=40000".to_string()]).expect("load");
    assert_eq!(set.layer.len(), 2);
    assert_eq!(set.layer_cap(), 2048);
    assert!(matches!(
        kernel_set_with_options(0, 2, &["-DTEST_SHARED_LIMIT=1000".to_string()]),
        Err(GpuError::Unsupported(_))
    ));
    let full = kernel_set(0, 2).expect("load");
    assert_eq!(full.layer_cap(), LAYER_CAP);
}

/// `layer_shared_bytes` is the byte count `carve` in `kernels/layer.cu` lays out, read back from the `-DTEST_CARVE_BYTES` hook.
#[test]
fn layer_shared_bytes_matches_the_carve_layout() {
    crate::require_cuda!();
    let context = device::context(0).expect("context");
    let (major, minor) = context.compute_capability().expect("capability");
    let stream = context.default_stream();
    for w in [1usize, 2, 4, 8, 16] {
        let ptx = compile_ptx(
            w,
            &format!("compute_{major}{minor}"),
            &["-DTEST_CARVE_BYTES".to_string()],
            kernel_cache::directory(),
        )
        .expect("compile");
        let carve_bytes = context
            .load_module(ptx)
            .and_then(|module| module.load_function("k_carve_bytes"))
            .expect("load");
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
                u64::from(layer_shared_bytes(capacity, w)),
                "W={w} capacity={capacity}"
            );
        }
    }
}

#[test]
fn every_arch_maps_to_itself() {
    for arch in KNOWN_ARCHES {
        assert_eq!(arch_static(arch), arch);
    }
    assert_eq!(arch_static("compute_121"), "compute_121");
}
