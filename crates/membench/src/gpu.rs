//! The device side of membench (feature `cuda`): `kernels/membench.cu` through NVRTC, timed with CUDA events.

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const SOURCE: &str = include_str!("../kernels/membench.cu");
const CFG: LaunchConfig = LaunchConfig {
    grid_dim: (8192, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 0,
};

fn text(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Best-of-`reps` and average GB/s per kernel on device `ordinal`, over three arrays of `mib` MiB of f64; STREAM nominal bytes.
pub fn run(
    ordinal: u32,
    mib: usize,
    reps: usize,
    kernels: &[String],
) -> Result<Vec<(String, f64, f64)>, String> {
    let ctx = CudaContext::new(ordinal as usize).map_err(|e| format!("device {ordinal}: {e}"))?;
    let (major, minor) = ctx.compute_capability().map_err(text)?;
    // `arch` wants a `&'static str`; one leaked string per run.
    let arch = Box::leak(format!("compute_{major}{minor}").into_boxed_str());
    let options = CompileOptions {
        arch: Some(arch),
        fmad: Some(false),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(SOURCE, options).map_err(|e| format!("nvrtc: {e}"))?;
    let module = ctx.load_module(ptx).map_err(text)?;
    let f = |name: &str| module.load_function(name).map_err(text);
    let (read, write, copy, triad) = (f("k_read")?, f("k_write")?, f("k_copy")?, f("k_triad")?);
    let stream = ctx.default_stream();
    let n = mib * (1 << 20) / std::mem::size_of::<f64>();
    let (n64, f8) = (n as u64, std::mem::size_of::<f64>());
    let mut a = stream.alloc_zeros::<f64>(n).map_err(text)?;
    let mut b = stream.alloc_zeros::<f64>(n).map_err(text)?;
    let mut c = stream.alloc_zeros::<f64>(n).map_err(text)?;
    let mut out = stream.alloc_zeros::<f64>(8192).map_err(text)?;
    // SAFETY: argument lists match the `extern "C"` signatures in membench.cu; every array holds `n` elements.
    unsafe {
        for (arr, v) in [(&mut a, 1.0f64), (&mut b, 2.0), (&mut c, 0.5)] {
            let launched = stream
                .launch_builder(&write)
                .arg(arr)
                .arg(&v)
                .arg(&n64)
                .launch(CFG);
            launched.map_err(text)?;
        }
    }
    stream.synchronize().map_err(text)?;
    let scalar = 3.0f64;
    let new_event = || {
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(text)
    };
    let (t0, t1) = (new_event()?, new_event()?);

    let mut results = Vec::new();
    for kernel in kernels {
        let bytes = match kernel.as_str() {
            "read" | "write" => n * f8,
            "copy" => 2 * n * f8,
            "triad" => 3 * n * f8,
            other => {
                eprintln!("membench: unknown kernel `{other}` (skipped)");
                continue;
            }
        };
        let (mut best, mut total) = (f64::INFINITY, 0.0);
        for _ in 0..reps {
            t0.record(&stream).map_err(text)?;
            // SAFETY: as above.
            let launched = unsafe {
                match kernel.as_str() {
                    "read" => {
                        let mut l = stream.launch_builder(&read);
                        l.arg(&a).arg(&mut out).arg(&n64).launch(CFG)
                    }
                    "write" => {
                        let mut l = stream.launch_builder(&write);
                        l.arg(&mut a).arg(&scalar).arg(&n64).launch(CFG)
                    }
                    "copy" => {
                        let mut l = stream.launch_builder(&copy);
                        l.arg(&mut a).arg(&b).arg(&n64).launch(CFG)
                    }
                    _ => {
                        let mut l = stream.launch_builder(&triad);
                        l.arg(&mut a)
                            .arg(&b)
                            .arg(&c)
                            .arg(&scalar)
                            .arg(&n64)
                            .launch(CFG)
                    }
                }
            };
            launched.map_err(text)?;
            t1.record(&stream).map_err(text)?;
            stream.synchronize().map_err(text)?;
            let dt = f64::from(t0.elapsed_ms(&t1).map_err(text)?) / 1e3;
            best = best.min(dt);
            total += dt;
        }
        let gbps = |dt: f64| bytes as f64 / dt / 1e9;
        results.push((kernel.clone(), gbps(best), gbps(total / reps as f64)));
    }
    Ok(results)
}
