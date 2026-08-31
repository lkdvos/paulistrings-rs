//! Per-phase timing / memory probe for the bucketed propagation engine.
//!
//! Drives [`propagate_with_scratch`] over a small menu of single-channel
//! circuits (plus a multi-channel Trotter step) at a matrix of thread
//! counts, and prints the [`PhaseStats`] breakdown the `phase-timing`
//! feature exposes: per-layer wall-clock phases, per-coset worker busy
//! time, and derived throughput / overhead / memory figures.
//!
//! This binary only builds with `--features phase-timing` (see the
//! `required-features` entry in `Cargo.toml`); without the feature,
//! [`LayerScratch::take_stats`] does not exist, so there would be nothing
//! to read.
//!
//! Run with:
//! ```bash
//! cargo run --release --features phase-timing --example phase_breakdown -- \
//!     [--n 1000000] [--qubits 128] [--threads 1,8,16,32] \
//!     [--layers rotation_zz,cnot,gu2q,depolarizing,trotter] [--reps 8] \
//!     [--seed 0xC0FFEE] [--format table|json|tsv]
//! ```
//!
//! `--qubits` picks the const-generic width `W` by `ceil(qubits / 64)`;
//! this probe only supports `W ∈ {1, 2}` (qubits ≤ 128), which is the same
//! set the crate's Python bindings restrict to for the smallest two
//! widths. Wider requests fail with a clear message rather than silently
//! truncating.
//!
//! Every layer except `trotter` is a single-channel circuit repeated
//! `--reps` times; `trotter` is a fixed 64-channel TFIM Trotter step (32
//! `ZZ` bond rotations + 32 transverse-field `X` rotations, copied from
//! `benches/pauli_ops.rs::bench_propagate_trotter`) and ignores `--reps`.
//! It also ignores `--n` above [`TROTTER_MAX_N`]: 64 *distinct* generators
//! under `AlwaysKeep` (no truncation) grow combinatorially rather than
//! closing to a bounded key set, and this was measured driving RSS into the
//! tens of GB well before `--n`'s default of 1,000,000 — see
//! `TROTTER_MAX_N`'s doc comment for the measurement. A capped cell prints
//! a note to stderr.
//!
//! Each `(layer, thread count)` cell runs the circuit twice inside a
//! dedicated Rayon thread pool of that width: an untimed warm-up call
//! (which, for the fanout-bounded channels, drives the input to its closed
//! key set so the timed call measures steady-state cost, not first-layer
//! growth), then the timed call whose input is the warm-up's output. Its
//! `PhaseStats` are read via [`LayerScratch::take_stats`] and its
//! `/proc/self/status` `VmRSS` / `VmHWM` are sampled right after.
//!
//! The input generators (`Xs64`, `rand_sum`, `low_weight_sum`) come from
//! `paulistrings::test_support`, shared with `benches/pauli_ops.rs` and the
//! crate's own tests. The per-layer channel recipes below are still duplicated
//! from the bench, since they are bench-shaped fixtures rather than fixtures
//! the library's tests use.

use std::time::Instant;

use num_complex::Complex64;
use paulistrings::channel::{Clifford2Q, Depolarizing, GeneralUnitary2Q, PauliRotation};
use paulistrings::engine::stats::TIMER_READ_OVERHEAD_NS;
use paulistrings::test_support::{low_weight_sum, rand_sum};
use paulistrings::{
    propagate_with_scratch, Circuit, Direction, LayerScratch, PauliString, PhaseStats,
    TruncationPolicy,
};

// ---------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------

const USAGE: &str = "\
Usage: phase_breakdown [OPTIONS]

Measures the phase-timing breakdown of propagate_with_scratch across a
menu of layers and thread counts.

Options:
  --n <usize>              Target input term count (default: 1000000)
  --qubits <usize>         Qubit count; picks W = ceil(qubits/64), W in {1,2}
                            (default: 128)
  --threads <csv>          Comma-separated thread counts (default: 1,8,16,32;
                            16 = the reference host's physical-core count)
  --layers <csv>           Comma-separated layers, from:
                              rotation_zz, cnot, gu2q, depolarizing, trotter
                            (default: rotation_zz,cnot,gu2q,depolarizing,trotter)
  --reps <usize>           Channel repetitions per cell, ignored by trotter
                            (default: 8)
                            NOTE: trotter also ignores --n above 100 (see
                            TROTTER_MAX_N in the source) — it is 64 distinct
                            generators under no truncation, so growth is
                            combinatorial rather than bounded; an uncapped
                            large --n has been measured driving it to tens
                            of GB of RSS.
  --seed <u64|0xHEX>       RNG seed for the input sum (default: 0xC0FFEE)
  --format table|json|tsv  Output format (default: table)
  --json-out FILE          Also append one JSON line per cell to FILE,
                           regardless of --format (input for scripts/perf-viz.py)
  -h, --help               Print this message
";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerKind {
    RotationZz,
    Cnot,
    Gu2q,
    Depolarizing,
    Trotter,
}

impl LayerKind {
    fn name(self) -> &'static str {
        match self {
            LayerKind::RotationZz => "rotation_zz",
            LayerKind::Cnot => "cnot",
            LayerKind::Gu2q => "gu2q",
            LayerKind::Depolarizing => "depolarizing",
            LayerKind::Trotter => "trotter",
        }
    }

    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "rotation_zz" => Ok(LayerKind::RotationZz),
            "cnot" => Ok(LayerKind::Cnot),
            "gu2q" => Ok(LayerKind::Gu2q),
            "depolarizing" => Ok(LayerKind::Depolarizing),
            "trotter" => Ok(LayerKind::Trotter),
            other => Err(format!(
                "unknown layer '{other}' (expected one of: rotation_zz, cnot, gu2q, \
                 depolarizing, trotter)"
            )),
        }
    }
}

fn default_layers() -> Vec<LayerKind> {
    vec![
        LayerKind::RotationZz,
        LayerKind::Cnot,
        LayerKind::Gu2q,
        LayerKind::Depolarizing,
        LayerKind::Trotter,
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Table,
    Json,
    Tsv,
}

struct Config {
    n: usize,
    qubits: usize,
    threads: Vec<usize>,
    layers: Vec<LayerKind>,
    reps: usize,
    seed: u64,
    format: Format,
    /// Sidecar file that gets one JSON line appended per cell, regardless of
    /// the stdout `--format` — the input `scripts/perf-viz.py` renders.
    json_out: Option<String>,
}

fn parse_usize(s: &str, flag: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("{flag} expects a non-negative integer, got '{s}'"))
}

fn parse_seed(s: &str) -> Result<u64, String> {
    let t = s.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16)
            .map_err(|_| format!("--seed expects a hex (0x...) or decimal integer, got '{s}'"))
    } else {
        t.parse::<u64>()
            .map_err(|_| format!("--seed expects a hex (0x...) or decimal integer, got '{s}'"))
    }
}

fn parse_csv_usize(s: &str, flag: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(str::trim)
        .filter(|tok| !tok.is_empty())
        .map(|tok| parse_usize(tok, flag))
        .collect()
}

fn parse_csv_layers(s: &str) -> Result<Vec<LayerKind>, String> {
    s.split(',')
        .map(str::trim)
        .filter(|tok| !tok.is_empty())
        .map(LayerKind::parse)
        .collect()
}

fn parse_format(s: &str) -> Result<Format, String> {
    match s {
        "table" => Ok(Format::Table),
        "json" => Ok(Format::Json),
        "tsv" => Ok(Format::Tsv),
        other => Err(format!(
            "--format expects one of table|json|tsv, got '{other}'"
        )),
    }
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut n: usize = 1_000_000;
    let mut qubits: usize = 128;
    let mut threads: Vec<usize> = vec![1, 8, 16, 32];
    let mut layers: Vec<LayerKind> = default_layers();
    let mut reps: usize = 8;
    let mut seed: u64 = 0xC0FFEE;
    let mut format = Format::Table;
    let mut json_out: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag {
            "--n" => n = parse_usize(value, "--n")?,
            "--qubits" => qubits = parse_usize(value, "--qubits")?,
            "--threads" => threads = parse_csv_usize(value, "--threads")?,
            "--layers" => layers = parse_csv_layers(value)?,
            "--reps" => reps = parse_usize(value, "--reps")?,
            "--seed" => seed = parse_seed(value)?,
            "--format" => format = parse_format(value)?,
            "--json-out" => json_out = Some(value.clone()),
            other => return Err(format!("unknown flag '{other}' (see --help)")),
        }
        i += 2;
    }

    if qubits == 0 {
        return Err("--qubits must be at least 1".to_string());
    }
    if threads.is_empty() {
        return Err("--threads must list at least one thread count".to_string());
    }
    if threads.contains(&0) {
        return Err("--threads entries must be positive".to_string());
    }
    if layers.is_empty() {
        return Err("--layers must list at least one layer".to_string());
    }
    if reps == 0 {
        return Err("--reps must be at least 1".to_string());
    }

    Ok(Config {
        n,
        qubits,
        threads,
        layers,
        reps,
        seed,
        format,
        json_out,
    })
}

// ---------------------------------------------------------------------
// Per-layer channel recipes (duplicated from benches/pauli_ops.rs)
// ---------------------------------------------------------------------

/// A weight-2 `ZZ` rotation, verbatim from `benches/pauli_ops.rs::zz_rotation`.
fn zz_rotation<const W: usize>(q0: u32, q1: u32, theta: f64) -> PauliRotation<W> {
    let mut gen = PauliString::<W> {
        x: [0u64; W],
        z: [0u64; W],
    };
    gen.z[(q0 as usize) / 64] |= 1u64 << (q0 % 64);
    gen.z[(q1 as usize) / 64] |= 1u64 << (q1 % 64);
    PauliRotation::new(gen, theta)
}

/// sqrt(SWAP) on `(q0, q1)`, verbatim from `benches/pauli_ops.rs::sqrt_swap`.
fn sqrt_swap(q0: u32, q1: u32) -> GeneralUnitary2Q {
    let h = Complex64::new(0.5, 0.5);
    let hc = Complex64::new(0.5, -0.5);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    GeneralUnitary2Q::from_matrix(
        q0,
        q1,
        [
            [one, zero, zero, zero],
            [zero, h, hc, zero],
            [zero, hc, h, zero],
            [zero, zero, zero, one],
        ],
    )
}

/// Qubit count of the fixed [`trotter_circuit`] chain — also the qubit
/// count `run_cell` uses when building `trotter`'s low-weight input, so
/// every weight-3 excitation lands inside the circuit's actual support.
const TROTTER_QUBITS: usize = 32;

/// Safety cap on `trotter`'s own input size, overriding `--n`.
///
/// `trotter` chains two full 64-layer circuit applications (the warm-up and
/// the timed call — see the "chain them" step in `run_cell`) under
/// `AlwaysKeep`, i.e. no truncation. Unlike rotation_zz/cnot/gu2q (a single
/// generator repeated, which provably closes to a bounded key set — see
/// `run_cell`), trotter is 64 *distinct* generators, and per-pass growth is a
/// roughly n-independent multiplicative factor
/// (measured ~600-700x per pass at `weight = 3`, `TROTTER_QUBITS = 32`, so
/// ~4-5×10^5x over the two chained passes). Left uncapped, `--n`'s default
/// of 1_000_000 would try to materialize on the order of 10^11 terms.
/// Measured directly at this cap: ~1×10^7 output terms, ~1 GB peak RSS,
/// finishes in single-digit seconds — see `run_cell` for the warning this
/// triggers when `--n` requests more.
const TROTTER_MAX_N: usize = 100;

/// The 64-channel TFIM Trotter step from
/// `benches/pauli_ops.rs::bench_propagate_trotter`: 32 `ZZ` bond rotations
/// (periodic boundary conditions) followed by 32 transverse-field `X`
/// rotations. Fixed shape ([`TROTTER_QUBITS`] qubits), independent of
/// `--qubits`/`--reps` — `--qubits` only sizes the input sum for every other
/// layer; `trotter`'s own input is sized off `TROTTER_QUBITS` instead (see
/// `run_cell`).
fn trotter_circuit<const W: usize>() -> Circuit<W> {
    let num_qubits = TROTTER_QUBITS;
    let theta = 0.1;
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in 0..num_qubits {
        let q0 = q as u32;
        let q1 = ((q + 1) % num_qubits) as u32;
        circuit.push(zz_rotation::<W>(q0, q1, 2.0 * theta));
    }
    for q in 0..num_qubits {
        let qq = q as u32;
        let gen = PauliString::<W>::x(qq);
        circuit.push(PauliRotation::new(gen, 2.0 * theta));
    }
    circuit
}

/// A truncation policy that never drops anything — mirrors the `AlwaysKeep`
/// helper used throughout the engine's own tests and `benches/pauli_ops.rs`.
struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

fn build_circuit<const W: usize>(layer: LayerKind, qubits: usize, reps: usize) -> Circuit<W> {
    let theta = 0.1;
    match layer {
        LayerKind::RotationZz => {
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(zz_rotation::<W>(0, 1, theta));
            }
            c
        }
        LayerKind::Cnot => {
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(Clifford2Q::cnot(0, 1));
            }
            c
        }
        LayerKind::Gu2q => {
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(sqrt_swap(0, 1));
            }
            c
        }
        LayerKind::Depolarizing => {
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(Depolarizing {
                    support: [3],
                    p: 0.05,
                });
            }
            c
        }
        LayerKind::Trotter => trotter_circuit::<W>(),
    }
}

// ---------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------

struct CellResult {
    layer: &'static str,
    threads: usize,
    n: usize,
    reps: usize,
    qubits: usize,
    seed: u64,
    wall_ns: u64,
    stats: PhaseStats,
    vmrss_kb: u64,
    vmhwm_kb: u64,
}

/// Read `VmRSS`/`VmHWM` (kB) from `/proc/self/status`. Linux-only, like the
/// rest of this crate's deployment targets; returns `0` for either field it
/// cannot find (e.g. running the example on a non-Linux host).
fn read_proc_status_kb() -> (u64, u64) {
    let mut rss = 0u64;
    let mut hwm = 0u64;
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                rss = parse_kb_field(rest);
            } else if let Some(rest) = line.strip_prefix("VmHWM:") {
                hwm = parse_kb_field(rest);
            }
        }
    }
    (rss, hwm)
}

fn parse_kb_field(s: &str) -> u64 {
    s.split_whitespace()
        .next()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}

fn run_cell<const W: usize>(layer: LayerKind, threads: usize, cfg: &Config) -> CellResult {
    // `trotter` is 64 *distinct* generators applied once each, not one
    // generator repeated — the latter provably closes to a bounded key set,
    // which is what keeps rotation_zz/cnot/gu2q bounded here. A dense input
    // can anticommute with most of 64 distinct generators and blow up
    // combinatorially (`benches/pauli_ops.rs` puts it at "up to 2^64", and
    // benches only low-weight inputs for that reason): a dense `rand_sum`
    // input was measured driving this layer past 50 GB of RSS in well under a
    // minute at only n = 2000. So trotter alone gets a low-weight input sized
    // to its own fixed qubit count instead of `--qubits`, and its own `--n`
    // cap (see `TROTTER_MAX_N`).
    let base = match layer {
        LayerKind::Trotter => {
            if cfg.n > TROTTER_MAX_N {
                eprintln!(
                    "phase_breakdown: note: trotter ignores --n above {TROTTER_MAX_N} \
                     (requested {}) — two chained, untruncated 64-layer passes grow \
                     combinatorially, and an uncapped --n has been measured driving this \
                     layer past tens of GB of RSS. Capping this cell's input to {TROTTER_MAX_N}.",
                    cfg.n,
                );
            }
            low_weight_sum::<W>(cfg.n.min(TROTTER_MAX_N), TROTTER_QUBITS, 3, cfg.seed)
        }
        _ => rand_sum::<W>(cfg.n, cfg.qubits, cfg.seed),
    };
    let circuit = build_circuit::<W>(layer, cfg.qubits, cfg.reps);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build a rayon thread pool");

    let (steady_n, wall_ns, stats) = pool.install(|| {
        let mut scratch = LayerScratch::<W>::new();

        // Untimed warm-up: for rotation_zz/cnot/gu2q this drives the input
        // to its closed key set, so the timed call below measures
        // steady-state cost rather than first-layer growth; for
        // depolarizing/trotter it just warms scratch/buffer capacity.
        let warmed = propagate_with_scratch(
            &circuit,
            base.clone(),
            &AlwaysKeep,
            Direction::Forward,
            &mut scratch,
        );
        let _ = scratch.take_stats(); // discard warm-up counters

        let steady_n = warmed.len();
        let start = Instant::now();
        let output = propagate_with_scratch(
            &circuit,
            warmed,
            &AlwaysKeep,
            Direction::Forward,
            &mut scratch,
        );
        let wall_ns = start.elapsed().as_nanos() as u64;
        let stats = scratch.take_stats();
        std::hint::black_box(&output);

        (steady_n, wall_ns, stats)
    });

    let (vmrss_kb, vmhwm_kb) = read_proc_status_kb();

    CellResult {
        layer: layer.name(),
        threads,
        n: steady_n,
        reps: cfg.reps,
        qubits: cfg.qubits,
        seed: cfg.seed,
        wall_ns,
        stats,
        vmrss_kb,
        vmhwm_kb,
    }
}

// ---------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------

/// Always printed first for every cell, in every format: `n=` and
/// `layers=` are a contract other scripts grep for.
fn print_cell_line(cell: &CellResult) {
    println!(
        "cell layer={} threads={} n={} layers={} wall_ms={:.3}",
        cell.layer,
        cell.threads,
        cell.n,
        cell.stats.layers,
        cell.wall_ns as f64 / 1e6,
    );
}

type PhaseGetter = fn(&PhaseStats) -> u64;

const WALL_PHASES: [(&str, PhaseGetter); 9] = [
    ("rebucket", |s| s.rebucket_ns),
    ("prepare", |s| s.prepare_ns),
    ("rescale", |s| s.rescale_ns),
    ("span_plan", |s| s.span_plan_ns),
    ("permute", |s| s.permute_ns),
    ("coset_loop", |s| s.coset_loop_ns),
    ("unpermute", |s| s.unpermute_ns),
    ("recount", |s| s.recount_ns),
    ("finalize", |s| s.finalize_ns),
];

const BUSY_PHASES: [(&str, PhaseGetter); 6] = [
    ("gather", |s| s.gather_ns),
    ("sort", |s| s.sort_ns),
    ("merge", |s| s.merge_ns),
    ("swap", |s| s.swap_ns),
    ("size", |s| s.size_ns),
    ("clear", |s| s.clear_ns),
];

fn print_table(cell: &CellResult) {
    let s = &cell.stats;
    let wall_total = s.wall_total_ns();

    println!(
        "  {:<12} {:>10} {:>12} {:>8}",
        "phase", "ms", "ms/layer", "% wall"
    );
    for (name, get) in WALL_PHASES {
        let ns = get(s);
        if ns == 0 {
            continue;
        }
        let ms = ns as f64 / 1e6;
        let ms_per_layer = if s.layers > 0 {
            ms / s.layers as f64
        } else {
            0.0
        };
        let pct = if wall_total > 0 {
            ns as f64 / wall_total as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  {:<12} {:>10.3} {:>12.4} {:>7.1}%",
            name, ms, ms_per_layer, pct
        );
    }

    let busy_total = s.busy_total_ns();
    println!("  busy (summed over every coset task / worker):");
    for (name, get) in BUSY_PHASES {
        let ns = get(s);
        let ms = ns as f64 / 1e6;
        let pct = if busy_total > 0 {
            ns as f64 / busy_total as f64 * 100.0
        } else {
            0.0
        };
        println!("    {:<10} {:>10.3} ms {:>7.1}% of busy", name, ms, pct);
    }
    let efficiency = if s.coset_loop_ns > 0 && cell.threads > 0 {
        busy_total as f64 / (s.coset_loop_ns as f64 * cell.threads as f64)
    } else {
        0.0
    };
    println!(
        "    sum(busy) = {:.3} ms, parallel efficiency (busy / (coset_loop * threads)) = {:.2}",
        busy_total as f64 / 1e6,
        efficiency
    );

    let wall_s = cell.wall_ns as f64 / 1e9;
    let strings_per_s = if wall_s > 0.0 {
        s.terms_in as f64 / wall_s
    } else {
        0.0
    };
    let overhead_ns = s.timer_reads() * TIMER_READ_OVERHEAD_NS;
    let overhead_pct = if cell.wall_ns > 0 {
        overhead_ns as f64 / cell.wall_ns as f64 * 100.0
    } else {
        0.0
    };
    println!("  strings/s          = {strings_per_s:.3e}");
    println!(
        "  timer overhead est = {:.3} us ({} reads x {} ns) = {:.2}% of wall",
        overhead_ns as f64 / 1e3,
        s.timer_reads(),
        TIMER_READ_OVERHEAD_NS,
        overhead_pct
    );
    println!(
        "  VmRSS = {} kB   VmHWM = {} kB",
        cell.vmrss_kb, cell.vmhwm_kb
    );
    println!();
}

fn print_json(cell: &CellResult) {
    println!("{}", json_line(cell));
}

/// One cell as a single JSON line — shared by `--format json` (stdout) and
/// `--json-out` (sidecar file for `scripts/perf-viz.py`).
fn json_line(cell: &CellResult) -> String {
    let s = &cell.stats;
    format!(
        "{{\"layer\":\"{}\",\"threads\":{},\"n\":{},\"reps\":{},\"qubits\":{},\"seed\":{},\
         \"wall_ns\":{},\"rebucket_ns\":{},\"prepare_ns\":{},\"rescale_ns\":{},\
         \"span_plan_ns\":{},\"permute_ns\":{},\"coset_loop_ns\":{},\"unpermute_ns\":{},\
         \"recount_ns\":{},\"finalize_ns\":{},\"swap_ns\":{},\"size_ns\":{},\
         \"gather_ns\":{},\"sort_ns\":{},\"merge_ns\":{},\"clear_ns\":{},\"layers\":{},\
         \"cosets\":{},\"runs\":{},\"rows_gathered\":{},\"rows_sorted\":{},\"rows_id\":{},\"terms_in\":{},\"terms_out\":{},\"vmrss_kb\":{},\
         \"vmhwm_kb\":{}}}",
        cell.layer,
        cell.threads,
        cell.n,
        cell.reps,
        cell.qubits,
        cell.seed,
        cell.wall_ns,
        s.rebucket_ns,
        s.prepare_ns,
        s.rescale_ns,
        s.span_plan_ns,
        s.permute_ns,
        s.coset_loop_ns,
        s.unpermute_ns,
        s.recount_ns,
        s.finalize_ns,
        s.swap_ns,
        s.size_ns,
        s.gather_ns,
        s.sort_ns,
        s.merge_ns,
        s.clear_ns,
        s.layers,
        s.cosets,
        s.runs,
        s.rows_gathered,
        s.rows_sorted,
        s.rows_id,
        s.terms_in,
        s.terms_out,
        cell.vmrss_kb,
        cell.vmhwm_kb,
    )
}

const TSV_HEADER: &str =
    "layer\tthreads\tn\treps\tqubits\tseed\twall_ns\trebucket_ns\tprepare_ns\t\
rescale_ns\tspan_plan_ns\tpermute_ns\tcoset_loop_ns\tunpermute_ns\trecount_ns\tfinalize_ns\t\
swap_ns\tsize_ns\tgather_ns\tsort_ns\tmerge_ns\tclear_ns\tlayers\tcosets\truns\trows_gathered\trows_sorted\trows_id\t\
terms_in\tterms_out\tvmrss_kb\tvmhwm_kb";

fn print_tsv_row(cell: &CellResult) {
    let s = &cell.stats;
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\
         {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        cell.layer,
        cell.threads,
        cell.n,
        cell.reps,
        cell.qubits,
        cell.seed,
        cell.wall_ns,
        s.rebucket_ns,
        s.prepare_ns,
        s.rescale_ns,
        s.span_plan_ns,
        s.permute_ns,
        s.coset_loop_ns,
        s.unpermute_ns,
        s.recount_ns,
        s.finalize_ns,
        s.swap_ns,
        s.size_ns,
        s.gather_ns,
        s.sort_ns,
        s.merge_ns,
        s.clear_ns,
        s.layers,
        s.cosets,
        s.runs,
        s.rows_gathered,
        s.rows_sorted,
        s.rows_id,
        s.terms_in,
        s.terms_out,
        cell.vmrss_kb,
        cell.vmhwm_kb,
    );
}

fn run<const W: usize>(cfg: &Config) {
    if cfg.format == Format::Tsv {
        println!("{TSV_HEADER}");
    }

    let mut sidecar = cfg.json_out.as_ref().map(|path| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| {
                eprintln!("phase_breakdown: cannot open --json-out '{path}': {e}");
                std::process::exit(2);
            })
    });

    for &layer in &cfg.layers {
        for &threads in &cfg.threads {
            let cell = run_cell::<W>(layer, threads, cfg);

            print_cell_line(&cell);
            match cfg.format {
                Format::Table => print_table(&cell),
                Format::Json => print_json(&cell),
                Format::Tsv => print_tsv_row(&cell),
            }
            if let Some(f) = sidecar.as_mut() {
                use std::io::Write;
                writeln!(f, "{}", json_line(&cell)).unwrap_or_else(|e| {
                    eprintln!("phase_breakdown: writing --json-out failed: {e}");
                    std::process::exit(2);
                });
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
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
