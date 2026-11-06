//! Single-run throughput harness: one observable through the kicked-Ising or square-TFIM quench on the host or one CUDA device, timed per step.
//!
//! ```bash
//! cargo run --release --example sampling_bench -- --help
//! ```

mod cli;
mod models;
mod output;
mod run;

use paulistrings::{ApproxTopN, CoefficientThreshold, CollapseSample};

use cli::{parse_args, Backend, Config, Truncation, USAGE};
use models::IsingQuench;
use paulistrings::Coefficient;

fn collapse_sample(cache: usize, samples: usize, seed: u64) -> CollapseSample {
    CollapseSample::new(cache, samples, seed)
}

/// `--threshold`, `--approx-topn` or `--collapse` as the device drivers take it.
#[cfg(feature = "cuda")]
fn builtin_truncation(truncation: Truncation) -> paulistrings::BuiltinTruncation {
    match truncation {
        Truncation::Threshold(eps) => CoefficientThreshold(eps).into(),
        Truncation::ApproxTopN(n) => ApproxTopN(n).into(),
        Truncation::Collapse { cache, samples, seed } => {
            collapse_sample(cache, samples, seed).into()
        }
    }
}

fn run<const W: usize, C: Coefficient>(cfg: &Config, quench: &IsingQuench) {
    let report = match (cfg.backend, cfg.truncation) {
        (Backend::Cpu, Truncation::Threshold(eps)) => {
            run::run_cpu::<W, C, _>(cfg, quench, &CoefficientThreshold(eps))
        }
        (Backend::Cpu, Truncation::ApproxTopN(n)) => {
            run::run_cpu::<W, C, _>(cfg, quench, &ApproxTopN(n))
        }
        (Backend::Cpu, Truncation::Collapse { cache, samples, seed }) => {
            run::run_cpu::<W, C, _>(cfg, quench, &collapse_sample(cache, samples, seed))
        }
        #[cfg(feature = "cuda")]
        (Backend::Gpu, truncation) => {
            run::run_gpu::<W, C>(cfg, quench, &builtin_truncation(truncation))
        }
        #[cfg(not(feature = "cuda"))]
        (Backend::Gpu, _) => {
            eprintln!("sampling_bench: --backend gpu needs the `cuda` feature");
            std::process::exit(2);
        }
    };
    let (_, kinds) = quench.circuit::<W>(1);
    output::print_report(cfg, quench, W, &report, &kinds);
}

fn run_dtype<const W: usize>(cfg: &Config, quench: &IsingQuench) {
    match cfg.dtype {
        "f32" => run::<W, f32>(cfg, quench),
        _ => run::<W, f64>(cfg, quench),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return;
    }
    let cfg = parse_args(&args).unwrap_or_else(|msg| {
        eprintln!("sampling_bench: {msg}\n\n{USAGE}");
        std::process::exit(2);
    });
    let quench = cfg.quench();
    match quench.num_qubits.div_ceil(64) {
        1 => run_dtype::<1>(&cfg, &quench),
        2 => run_dtype::<2>(&cfg, &quench),
        3 | 4 => run_dtype::<4>(&cfg, &quench),
        _ => {
            eprintln!(
                "sampling_bench: {} qubits exceed the 256 this harness builds for",
                quench.num_qubits
            );
            std::process::exit(2);
        }
    }
}
