//! Single-run throughput harness: one observable through the kicked-Ising or square-TFIM quench on the host, one CUDA device, several devices of one process, or one device per MPI rank, timed per step.
//!
//! ```bash
//! cargo run --release --example sampling_bench -- --help
//! ```

mod cli;
mod models;
#[cfg(all(feature = "cuda", feature = "mpi"))]
mod mpi_run;
mod output;
mod run;

use paulistrings::{ApproxTopN, CoefficientThreshold, CollapseSample};

use cli::{parse_args, Backend, Config, Truncation, USAGE};
use models::IsingQuench;
use paulistrings::Coefficient;

/// `--threshold`, `--approx-topn` or `--collapse` as the device drivers take it.
#[cfg(feature = "cuda")]
fn builtin_truncation(truncation: Truncation) -> paulistrings::BuiltinTruncation {
    match truncation {
        Truncation::Threshold(eps) => CoefficientThreshold(eps).into(),
        Truncation::ApproxTopN(n) => ApproxTopN(n).into(),
        Truncation::Collapse { cache, samples, seed } => {
            CollapseSample::new(cache, samples, seed).into()
        }
    }
}

fn unavailable(what: &str) -> ! {
    eprintln!("sampling_bench: {what}");
    std::process::exit(2);
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
            run::run_cpu::<W, C, _>(cfg, quench, &CollapseSample::new(cache, samples, seed))
        }
        #[cfg(feature = "cuda")]
        (Backend::Gpu | Backend::Gpus, truncation) => {
            run::run_gpu::<W, C>(cfg, quench, &builtin_truncation(truncation))
        }
        #[cfg(not(feature = "cuda"))]
        (Backend::Gpu | Backend::Gpus, _) => {
            unavailable("a device backend needs the `cuda` feature")
        }
        #[cfg(all(feature = "cuda", feature = "mpi"))]
        (Backend::MpiGpu, truncation) => {
            mpi_run::run_mpi_gpu::<W, C>(cfg, quench, &builtin_truncation(truncation))
        }
        #[cfg(not(all(feature = "cuda", feature = "mpi")))]
        (Backend::MpiGpu, _) => {
            unavailable("--backend mpi-gpu needs the `cuda` and `mpi` features")
        }
    };
    if report.root {
        let (_, kinds) = quench.circuit::<W>(1);
        output::print_report(cfg, quench, W, &report, &kinds);
    }
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
    // The universe outlives the run; its drop is `MPI_Finalize`. SERIALIZED: the layer loop calls MPI off a pool worker.
    #[cfg(all(feature = "cuda", feature = "mpi"))]
    let _universe = (cfg.backend == Backend::MpiGpu).then(|| {
        use paulistrings::mpi::rsmpi;
        let (universe, threading) = rsmpi::initialize_with_threading(rsmpi::Threading::Serialized)
            .unwrap_or_else(|| unavailable("MPI is already initialized in this process"));
        if threading < rsmpi::Threading::Serialized {
            eprintln!("sampling_bench: warning: MPI provided {threading:?}, below SERIALIZED");
        }
        universe
    });
    let quench = cfg.quench();
    match quench.num_qubits.div_ceil(64) {
        1 => run_dtype::<1>(&cfg, &quench),
        2 => run_dtype::<2>(&cfg, &quench),
        3 | 4 => run_dtype::<4>(&cfg, &quench),
        _ => unavailable(&format!(
            "{} qubits exceed the 256 this harness builds for",
            quench.num_qubits
        )),
    }
}
