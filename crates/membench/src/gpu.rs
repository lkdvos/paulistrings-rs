//! The device side of membench (feature `cuda`): `kernels/membench.cu` through NVRTC, timed with CUDA events.

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const SOURCE: &str = include_str!("../kernels/membench.cu");
const THREADS: u32 = 256;
const BLOCKS: u32 = 8192;

fn arch_static(arch: String) -> &'static str {
    match arch.as_str() {
        "compute_80" => "compute_80",
        "compute_86" => "compute_86",
        "compute_89" => "compute_89",
        "compute_90" => "compute_90",
        _ => Box::leak(arch.into_boxed_str()),
    }
}

/// Best-of-`reps` and average GB/s per kernel on device `ordinal`, over three arrays of `mib` MiB of f64; STREAM nominal bytes.
pub fn run(
    ordinal: u32,
    mib: usize,
    reps: usize,
    kernels: &[String],
) -> Result<Vec<(String, f64, f64)>, String> {
    let ctx = CudaContext::new(ordinal as usize).map_err(|e| format!("device {ordinal}: {e}"))?;
    let (major, minor) = ctx.compute_capability().map_err(|e| e.to_string())?;
    let ptx = compile_ptx_with_opts(
        SOURCE,
        CompileOptions {
            arch: Some(arch_static(format!("compute_{major}{minor}"))),
            fmad: Some(false),
            ..Default::default()
        },
    )
    .map_err(|e| format!("nvrtc: {e}"))?;
    let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;
    let f = |name: &str| -> Result<CudaFunction, String> {
        module.load_function(name).map_err(|e| e.to_string())
    };
    let (fill, read, write, copy, triad) = (
        f("k_fill")?,
        f("k_read")?,
        f("k_write")?,
        f("k_copy")?,
        f("k_triad")?,
    );
    let stream = ctx.default_stream();
    let n = mib * (1 << 20) / std::mem::size_of::<f64>();
    let n64 = n as u64;
    let f8 = std::mem::size_of::<f64>();
    let cfg = LaunchConfig {
        grid_dim: (BLOCKS, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut a = stream.alloc_zeros::<f64>(n).map_err(|e| e.to_string())?;
    let mut b = stream.alloc_zeros::<f64>(n).map_err(|e| e.to_string())?;
    let mut c = stream.alloc_zeros::<f64>(n).map_err(|e| e.to_string())?;
    let mut out = stream
        .alloc_zeros::<f64>(BLOCKS as usize)
        .map_err(|e| e.to_string())?;
    // SAFETY: argument lists match the `extern "C"` signatures in membench.cu; every array holds `n` elements.
    unsafe {
        for (arr, v) in [(&mut a, 1.0f64), (&mut b, 2.0), (&mut c, 0.5)] {
            stream
                .launch_builder(&fill)
                .arg(arr)
                .arg(&v)
                .arg(&n64)
                .launch(cfg)
                .map_err(|e| e.to_string())?;
        }
    }
    stream.synchronize().map_err(|e| e.to_string())?;
    let scalar = 3.0f64;
    let new_event = || {
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| e.to_string())
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
        let mut best = f64::INFINITY;
        let mut total = 0.0;
        for _ in 0..reps {
            t0.record(&stream).map_err(|e| e.to_string())?;
            // SAFETY: as above.
            unsafe {
                let r = match kernel.as_str() {
                    "read" => stream
                        .launch_builder(&read)
                        .arg(&a)
                        .arg(&mut out)
                        .arg(&n64)
                        .launch(cfg),
                    "write" => stream
                        .launch_builder(&write)
                        .arg(&mut a)
                        .arg(&scalar)
                        .arg(&n64)
                        .launch(cfg),
                    "copy" => stream
                        .launch_builder(&copy)
                        .arg(&mut a)
                        .arg(&b)
                        .arg(&n64)
                        .launch(cfg),
                    _ => stream
                        .launch_builder(&triad)
                        .arg(&mut a)
                        .arg(&b)
                        .arg(&c)
                        .arg(&scalar)
                        .arg(&n64)
                        .launch(cfg),
                };
                r.map_err(|e| e.to_string())?;
            }
            t1.record(&stream).map_err(|e| e.to_string())?;
            stream.synchronize().map_err(|e| e.to_string())?;
            let dt = f64::from(t0.elapsed_ms(&t1).map_err(|e| e.to_string())?) / 1e3;
            best = best.min(dt);
            total += dt;
        }
        let gbps = |dt: f64| bytes as f64 / dt / 1e9;
        results.push((kernel.clone(), gbps(best), gbps(total / reps as f64)));
    }
    Ok(results)
}
