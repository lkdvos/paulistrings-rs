//! Per-phase timing / memory probe for the bucketed propagation engine.
//!
//! Drives [`propagate_with`] over a menu of layers (rotation, Clifford,
//! general-unitary, noise, Trotter, and partitioned/distributed variants) across a matrix of
//! thread and partition counts, and prints the [`PhaseStats`] breakdown the `phase-timing`
//! feature exposes.
//!
//! ```bash
//! cargo run --release --features phase-timing --example phase_breakdown -- --help
//! ```

mod cells;
mod cli;
#[cfg(feature = "cuda")]
mod device;
mod output;
mod rows;
mod workloads;

use paulistrings::test_support::KeepAll;
use paulistrings::{ApproxTopN, CoefficientThreshold, TopN, TruncationPolicy};

#[cfg(feature = "mpi")]
use cells::run_cell_mpi;
use cells::{run_cell, run_cell_partitioned, CellResult};
#[cfg(feature = "cuda")]
use cli::DEVICE_USAGE;
#[cfg(feature = "mpi")]
use cli::MPI_USAGE;
use cli::{parse_args, Config, Format, TruncSpec, USAGE};
#[cfg(all(feature = "cuda", feature = "mpi"))]
use device::run_cell_mpi_gpu;
#[cfg(feature = "cuda")]
use device::{builtin_truncation, resolve_devices, run_cell_gpu};
use output::{json_line, print_cell_line, print_json, print_table};

/// Dispatch `--truncation` into exactly one monomorphization of [`run_cells`], so the policy's
/// `keep_term` inlines into the merge the way a real caller's does. Each spec supplies two policy
/// values: the unpartitioned one and its partitioned form (`topn` has none, already rejected by
/// [`parse_args`]).
fn run<const W: usize>(cfg: &Config) {
    match cfg.truncation {
        TruncSpec::Keep => run_cells::<W, _, _>(cfg, &KeepAll, Some(&KeepAll)),
        TruncSpec::Coefficient(t) => run_cells::<W, _, _>(
            cfg,
            &CoefficientThreshold(t),
            Some(&CoefficientThreshold(t)),
        ),
        TruncSpec::TopN(n) => run_cells::<W, _, ApproxTopN>(cfg, &TopN(n), None),
        TruncSpec::ApproxTopN(n) => run_cells::<W, _, _>(cfg, &ApproxTopN(n), Some(&ApproxTopN(n))),
    }
}

fn run_cells<const W: usize, P, PP>(cfg: &Config, policy: &P, partitioned_policy: Option<&PP>)
where
    P: TruncationPolicy<W>,
    PP: TruncationPolicy<W>,
{
    let mut sidecar = cfg.json_out.as_ref().map(|path| {
        // Under `--mpi` every rank writes its own file, suffixed by rank, to avoid interleaving.
        let path = mpi_sidecar_path(path, cfg);
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|e| {
                eprintln!("phase_breakdown: cannot open --json-out '{path}': {e}");
                std::process::exit(2);
            })
    });

    // A device cell has no thread axis: one cell per layer, `--threads[0]` echoed into the row.
    #[cfg(feature = "cuda")]
    if let Some(spec) = &cfg.device {
        let policy = builtin_truncation(cfg.truncation);
        #[cfg(feature = "mpi")]
        if cfg.mpi {
            for &layer in &cfg.layers {
                let cell = run_cell_mpi_gpu::<W>(layer, cfg.threads[0], spec, cfg, &policy);
                emit_cell(cfg, &cell, sidecar.as_mut());
            }
            return;
        }
        let devices = resolve_devices(spec, cfg);
        for &layer in &cfg.layers {
            let cell = run_cell_gpu::<W>(layer, cfg.threads[0], &devices, cfg, &policy);
            emit_cell(cfg, &cell, sidecar.as_mut());
        }
        return;
    }

    let partitioned_policy = || {
        partitioned_policy.expect("parse_args rejects topn, the one spec with no partitioned form")
    };
    for &layer in &cfg.layers {
        for &partitions in &cfg.partitions {
            for &threads in &cfg.threads {
                #[cfg(feature = "mpi")]
                if cfg.mpi {
                    let cell = run_cell_mpi::<W, PP>(layer, threads, cfg, partitioned_policy());
                    emit_cell(cfg, &cell, sidecar.as_mut());
                    continue;
                }
                let cell = if partitions > 1 {
                    run_cell_partitioned::<W, PP>(
                        layer,
                        threads,
                        partitions,
                        cfg,
                        partitioned_policy(),
                    )
                } else {
                    run_cell::<W, P>(layer, threads, cfg, policy)
                };
                emit_cell(cfg, &cell, sidecar.as_mut());
            }
        }
    }
}

/// The cell line, the chosen format, and the sidecar line.
fn emit_cell(cfg: &Config, cell: &CellResult, sidecar: Option<&mut std::fs::File>) {
    print_cell_line(cell);
    match cfg.format {
        Format::Table => print_table(cell),
        Format::Json => print_json(cell),
    }
    if let Some(f) = sidecar {
        use std::io::Write;
        writeln!(f, "{}", json_line(cell)).unwrap_or_else(|e| {
            eprintln!("phase_breakdown: writing --json-out failed: {e}");
            std::process::exit(2);
        });
    }
}

/// The `--json-out` path this process writes: unchanged without `--mpi`, and
/// suffixed `.rank<N>` with it.
#[cfg(feature = "mpi")]
fn mpi_sidecar_path(path: &str, cfg: &Config) -> String {
    if !cfg.mpi {
        return path.to_string();
    }
    use paulistrings::mpi::rsmpi::topology::{Communicator, SimpleCommunicator};
    format!("{path}.rank{}", SimpleCommunicator::world().rank())
}

/// Without the `mpi` feature there is no rank, so the path is the path.
#[cfg(not(feature = "mpi"))]
fn mpi_sidecar_path(path: &str, _cfg: &Config) -> String {
    path.to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        #[cfg(feature = "mpi")]
        println!("{MPI_USAGE}");
        #[cfg(feature = "cuda")]
        println!("{DEVICE_USAGE}");
        return;
    }

    let cfg = match parse_args(&args) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("phase_breakdown: {msg}");
            eprintln!();
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    };

    // The universe outlives every cell; `Universe`'s drop is `MPI_Finalize`. The library never
    // creates one, so the probe does it here.
    #[cfg(feature = "mpi")]
    let _universe = cfg.mpi.then(|| {
        use paulistrings::mpi::rsmpi;
        // SERIALIZED, not FUNNELED: the layer loop runs inside a Rayon pool.
        let (universe, threading) = rsmpi::initialize_with_threading(rsmpi::Threading::Serialized)
            .unwrap_or_else(|| {
                eprintln!("phase_breakdown: MPI is already initialized in this process");
                std::process::exit(2);
            });
        if threading < rsmpi::Threading::Serialized {
            eprintln!(
                "phase_breakdown: warning: MPI provided {threading:?}, below the SERIALIZED the \
                 engine needs",
            );
        }
        universe
    });

    let words = cfg.qubits.div_ceil(64);
    match words {
        1 => run::<1>(&cfg),
        2 => run::<2>(&cfg),
        w => {
            eprintln!(
                "phase_breakdown: --qubits {} needs W={w} 64-bit words, but this probe only \
                 supports W in {{1, 2}} (qubits <= 128). Pass --qubits <= 128.",
                cfg.qubits,
            );
            std::process::exit(2);
        }
    }
}
