//! STREAM-style memory-bandwidth ceiling probe.
//!
//! Measures sustained read / write / copy / triad bandwidth over arrays far
//! larger than the last-level cache, single- and multi-threaded, using the
//! same Rayon chunked idiom as the propagation engine. CPU and memory
//! *placement* is deliberately external — run under `numactl` / `taskset`
//! (see `scripts/bandwidth.sh`); the binary itself only controls the thread
//! count.
//!
//! Accounting follows the STREAM convention: nominal bytes only (copy =
//! 16 B/elem, triad = 24 B/elem). Plain stores incur read-for-ownership, so
//! the *actual* DRAM traffic of write/copy/triad is higher than the nominal
//! figure by one read stream — deliberately uncorrected, because the
//! propagation engine also uses plain (write-allocating) stores, making the
//! nominal figure the ceiling its phases should be compared against.
//! Cross-check against `perf stat -a -e uncore_imc/cas_count_*` if the raw
//! CAS traffic is wanted.

use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

const DEFAULT_MIB: usize = 512;
const DEFAULT_REPS: usize = 5;

struct Args {
    threads: usize,
    mib: usize,
    reps: usize,
    kernels: Vec<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        threads: 1,
        mib: DEFAULT_MIB,
        reps: DEFAULT_REPS,
        kernels: vec!["read", "write", "copy", "triad"]
            .into_iter()
            .map(String::from)
            .collect(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("missing value for {name}"));
        match flag.as_str() {
            "--threads" => {
                args.threads = value("--threads")?.parse().map_err(|e| format!("{e}"))?
            }
            "--mib" => args.mib = value("--mib")?.parse().map_err(|e| format!("{e}"))?,
            "--reps" => args.reps = value("--reps")?.parse().map_err(|e| format!("{e}"))?,
            "--kernels" => {
                args.kernels = value("--kernels")?.split(',').map(String::from).collect()
            }
            "--help" | "-h" => {
                println!(
                    "usage: membench [--threads N] [--mib SIZE] [--reps N] \
                     [--kernels read,write,copy,triad]\n\
                     Prints one `kernel=... threads=... best_gbps=...` line per kernel.\n\
                     Placement is external: run under numactl/taskset."
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if args.threads == 0 || args.mib == 0 || args.reps == 0 {
        return Err("--threads/--mib/--reps must be positive".into());
    }
    Ok(args)
}

/// Elements per array such that one array is `mib` MiB of f64.
fn elems(mib: usize) -> usize {
    mib * (1 << 20) / std::mem::size_of::<f64>()
}

/// Allocate and first-touch in parallel with the measuring pool, so page
/// placement matches the access pattern (and any external `--membind`).
fn alloc(n: usize, fill: f64) -> Vec<f64> {
    let mut v = vec![0.0f64; n];
    v.par_chunks_mut(1 << 16).for_each(|c| {
        for x in c {
            *x = fill;
        }
    });
    v
}

/// Multi-accumulator sum: a plain `iter().sum::<f64>()` is a serial float
/// fold the compiler may not reassociate, which measures FP latency rather
/// than bandwidth. Eight independent accumulators expose enough ILP that the
/// loads become the bottleneck.
fn sum_unrolled(ch: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    let mut it = ch.chunks_exact(8);
    for c in &mut it {
        for k in 0..8 {
            acc[k] += c[k];
        }
    }
    acc.iter().sum::<f64>() + it.remainder().iter().sum::<f64>()
}

/// Best-of-`reps` bandwidth in GB/s for `bytes` nominal bytes per pass.
fn measure<F: FnMut()>(reps: usize, bytes: usize, mut pass: F) -> (f64, f64) {
    let mut best = f64::INFINITY;
    let mut total = 0.0;
    for _ in 0..reps {
        let t0 = Instant::now();
        pass();
        let dt = t0.elapsed().as_secs_f64();
        best = best.min(dt);
        total += dt;
    }
    let gbps = |dt: f64| bytes as f64 / dt / 1e9;
    (gbps(best), gbps(total / reps as f64))
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("membench: {e}");
            std::process::exit(2);
        }
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .expect("failed to build thread pool");
    let n = elems(args.mib);
    let f8 = std::mem::size_of::<f64>();

    pool.install(|| {
        let mut a = alloc(n, 1.0);
        let b = alloc(n, 2.0);
        let c = alloc(n, 0.5);
        let scalar = black_box(3.0f64);

        for kernel in &args.kernels {
            let (best, avg) = match kernel.as_str() {
                "read" => measure(args.reps, n * f8, || {
                    let s: f64 = a.par_chunks(1 << 16).map(sum_unrolled).sum();
                    black_box(s);
                }),
                "write" => measure(args.reps, n * f8, || {
                    a.par_chunks_mut(1 << 16).for_each(|ch| {
                        for x in ch {
                            *x = scalar;
                        }
                    });
                    black_box(&a);
                }),
                "copy" => measure(args.reps, 2 * n * f8, || {
                    a.par_chunks_mut(1 << 16)
                        .zip(b.par_chunks(1 << 16))
                        .for_each(|(dst, src)| dst.copy_from_slice(src));
                    black_box(&a);
                }),
                "triad" => measure(args.reps, 3 * n * f8, || {
                    a.par_chunks_mut(1 << 16)
                        .zip(b.par_chunks(1 << 16).zip(c.par_chunks(1 << 16)))
                        .for_each(|(dst, (sb, sc))| {
                            for i in 0..dst.len() {
                                dst[i] = sb[i] + scalar * sc[i];
                            }
                        });
                    black_box(&a);
                }),
                other => {
                    eprintln!("membench: unknown kernel `{other}` (skipped)");
                    continue;
                }
            };
            println!(
                "kernel={kernel} threads={} mib={} reps={} best_gbps={best:.2} avg_gbps={avg:.2}",
                args.threads, args.mib, args.reps
            );
        }
    });
}
