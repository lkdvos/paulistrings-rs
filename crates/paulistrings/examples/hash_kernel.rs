//! The sorted-column engine against the hash engine, across bucket targets and thread counts.
//!
//! ```bash
//! source scripts/jcc-rustflags.sh
//! cargo run --release --features hash-kernel --example hash_kernel -- --layers su4,cnot --n 1000000 --threads 1,16
//! ```

use std::time::Instant;

use num_complex::Complex64;
use paulistrings::hash_kernel::{propagate_hashed, HashOptions, HashScratch, HashSum};
use paulistrings::test_support::{
    assert_terms_close, haar_su4_matrix, heavy_hex_127_edges, rand_sum,
    sqrt_swap_matrix, zz_rotation, KeepAll,
};
use paulistrings::{
    propagate_with, BuildAccumulator, Circuit, CoefficientThreshold, Direction, LayerScratch,
    PauliRotation, PauliString, PauliSum, PropagateOptions, PtmChannel, TruncationPolicy,
};

struct Config {
    layers: Vec<String>,
    n: usize,
    qubits: usize,
    reps: usize,
    threads: Vec<usize>,
    targets: Vec<usize>,
    engines: Vec<String>,
    threshold: Option<f64>,
    check: bool,
}

fn parse() -> Config {
    let mut cfg = Config {
        layers: vec!["rotation_zz".into(), "cnot".into(), "gu2q".into(), "su4".into()],
        n: 1_000_000,
        qubits: 128,
        reps: 10,
        threads: vec![1],
        targets: vec![256, 1024, 4096, 16384, 65536, usize::MAX],
        engines: vec!["sorted".into(), "hashed".into()],
        threshold: None,
        check: false,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let list = |s: &str| s.split(',').map(str::to_string).collect::<Vec<_>>();
    let sizes = |s: &str| {
        s.split(',')
            .map(|v| if v == "inf" { usize::MAX } else { v.parse().expect("a size or inf") })
            .collect::<Vec<usize>>()
    };
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1).map(String::as_str).unwrap_or("");
        match args[i].as_str() {
            "--layers" => cfg.layers = list(value),
            "--n" => cfg.n = value.parse().unwrap(),
            "--qubits" => cfg.qubits = value.parse().unwrap(),
            "--reps" => cfg.reps = value.parse().unwrap(),
            "--threads" => cfg.threads = sizes(value),
            "--targets" => cfg.targets = sizes(value),
            "--engines" => cfg.engines = list(value),
            "--threshold" => cfg.threshold = Some(value.parse().unwrap()),
            "--check" => {
                cfg.check = true;
                i += 1;
                continue;
            }
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    cfg
}

/// The growing Trotter workloads, whose timed call is the whole run from a single `Z`.
fn is_trotter(layer: &str) -> bool {
    matches!(layer, "tfim" | "heavyhex")
}

fn circuit<const W: usize>(layer: &str, cfg: &Config) -> Circuit<W> {
    let mut c = Circuit::<W>::new(cfg.qubits);
    let (h, zz) = (5.0 * std::f64::consts::PI / 16.0, -std::f64::consts::FRAC_PI_2);
    for _ in 0..cfg.reps {
        match layer {
            "rotation_zz" => c.push(zz_rotation::<W>(0, 1, 0.1)),
            "cnot" => c.push(PtmChannel::cnot(0, 1)),
            "rotation_wide" => {
                let mut generator = PauliString::<W>::x(0);
                generator.mul_assign(&PauliString::<W>::z(1));
                generator.mul_assign(&PauliString::<W>::y(2));
                generator.mul_assign(&PauliString::<W>::z(3));
                c.push(PauliRotation::<W>::new(generator, 0.1));
            }
            "clifford" => {
                let q = c.len() % 6;
                match c.len() % 4 {
                    0 => c.push(PtmChannel::hadamard(q)),
                    1 => c.push(PtmChannel::cnot(q, (q + 1) % 6)),
                    2 => c.push(PtmChannel::phase_gate(q)),
                    _ => c.push(PtmChannel::cz(q, (q + 2) % 6)),
                }
            }
            "gu2q" => c.push(PtmChannel::unitary_2q(0, 1, sqrt_swap_matrix())),
            "su4" => c.push(PtmChannel::unitary_2q(0, 1, haar_su4_matrix())),
            "depolarizing" => c.push(PtmChannel::depolarizing(3, 0.05)),
            "tfim" => {
                for a in 0..cfg.qubits - 1 {
                    c.push(zz_rotation::<W>(a, a + 1, zz));
                }
                for q in 0..cfg.qubits {
                    c.push(PauliRotation::<W>::new(PauliString::<W>::x(q), h));
                }
            }
            "heavyhex" => {
                for q in 0..127 {
                    c.push(PauliRotation::<W>::new(PauliString::<W>::x(q), h));
                }
                for (a, b) in heavy_hex_127_edges() {
                    c.push(zz_rotation::<W>(a, b, zz));
                }
            }
            other => panic!("unknown layer {other}"),
        }
    }
    c
}

fn input<const W: usize>(layer: &str, cfg: &Config) -> PauliSum<W> {
    if is_trotter(layer) {
        let mut acc = BuildAccumulator::<W>::with_capacity(cfg.qubits, 1);
        acc.add_term(PauliString::<W>::z(cfg.qubits / 2), Complex64::new(1.0, 0.0));
        acc.finalize()
    } else {
        rand_sum::<W>(cfg.n, cfg.qubits, 1)
    }
}

/// Process CPU time, which a loaded host's preemption does not inflate.
fn cpu_ns() -> u64 {
    let mut t = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: a valid clock id and an owned out-parameter.
    unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut t) };
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}

struct Cell {
    wall_ns: u64,
    cpu_ns: u64,
    terms: usize,
    buckets: usize,
    detail: String,
}

fn sorted_options(target: usize) -> PropagateOptions {
    if target == usize::MAX {
        PropagateOptions { target_bucket_len: usize::MAX / 2, min_buckets: 16 }
    } else {
        PropagateOptions { target_bucket_len: target, ..PropagateOptions::default() }
    }
}

fn hashed_options(target: usize) -> HashOptions {
    if target == usize::MAX {
        HashOptions { target_bucket_len: usize::MAX / 2, min_buckets: 1 }
    } else {
        HashOptions { target_bucket_len: target, min_buckets: PropagateOptions::default().min_buckets }
    }
}

/// Untimed warm-up to the steady state (not for the Trotter workloads), then one timed call; returns the cell and the output.
fn run_sorted<const W: usize, P: TruncationPolicy<W>>(
    layer: &str,
    cfg: &Config,
    policy: &P,
    target: usize,
) -> (Cell, PauliSum<W>) {
    let circuit = circuit::<W>(layer, cfg);
    let options = sorted_options(target);
    let mut scratch = LayerScratch::<W>::new();
    let mut sum = input::<W>(layer, cfg);
    if !is_trotter(layer) {
        sum = propagate_with(&circuit, sum, policy, Direction::Heisenberg, &mut scratch, options);
    }
    let terms_in = sum.len();
    let (started, cpu) = (Instant::now(), cpu_ns());
    let sum = propagate_with(&circuit, sum, policy, Direction::Heisenberg, &mut scratch, options);
    let wall_ns = started.elapsed().as_nanos() as u64;
    let cpu_ns = cpu_ns() - cpu;
    let cell = Cell {
        wall_ns,
        cpu_ns,
        terms: if is_trotter(layer) { sum.len() } else { terms_in },
        buckets: sum.num_buckets(),
        detail: String::new(),
    };
    (cell, sum)
}

fn run_hashed<const W: usize, P: TruncationPolicy<W>>(
    layer: &str,
    cfg: &Config,
    policy: &P,
    target: usize,
) -> (Cell, PauliSum<W>) {
    let circuit = circuit::<W>(layer, cfg);
    let options = hashed_options(target);
    let mut scratch = HashScratch::<W>::new();
    let mut sum = HashSum::from_pauli_sum(&input::<W>(layer, cfg));
    if !is_trotter(layer) {
        propagate_hashed(&circuit, &mut sum, policy, Direction::Heisenberg, &mut scratch, options);
        scratch.take_times();
    }
    let terms_in = sum.len();
    let (started, cpu) = (Instant::now(), cpu_ns());
    propagate_hashed(&circuit, &mut sum, policy, Direction::Heisenberg, &mut scratch, options);
    let wall_ns = started.elapsed().as_nanos() as u64;
    let cpu_ns = cpu_ns() - cpu;
    let times = scratch.take_times();
    let busy = (times.walk_ns + times.remove_ns + times.append_ns).max(1) as f64;
    let detail = format!(
        "walk {:.0}% remove {:.0}% append {:.0}% | appended/term-layer {:.2} | {:.0} B/term",
        100.0 * times.walk_ns as f64 / busy,
        100.0 * times.remove_ns as f64 / busy,
        100.0 * times.append_ns as f64 / busy,
        times.appended as f64 / (terms_in.max(1) * cfg.reps) as f64,
        sum.bytes() as f64 / sum.len().max(1) as f64,
    );
    let cell = Cell {
        wall_ns,
        cpu_ns,
        terms: if is_trotter(layer) { sum.len() } else { terms_in },
        buckets: sum.num_buckets(),
        detail,
    };
    (cell, sum.to_pauli_sum())
}

fn run<const W: usize, P: TruncationPolicy<W>>(cfg: &Config, policy: &P) {
    println!(
        "{:<12} {:<7} {:>3} {:>7} {:>8} {:>10} {:>10} {:>9} {:>9}  detail",
        "layer", "engine", "thr", "target", "buckets", "terms", "wall ms", "ns/term·l", "cpu ns/tl"
    );
    for layer in &cfg.layers {
        for &threads in &cfg.threads {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let mut reference: Option<PauliSum<W>> = None;
            for &target in &cfg.targets {
                for engine in &cfg.engines {
                    let (cell, output) = pool.install(|| match engine.as_str() {
                        "sorted" => run_sorted::<W, P>(layer, cfg, policy, target),
                        "hashed" => run_hashed::<W, P>(layer, cfg, policy, target),
                        other => panic!("unknown engine {other}"),
                    });
                    let units = if is_trotter(layer) {
                        cell.terms.max(1) as f64
                    } else {
                        (cell.terms.max(1) * cfg.reps) as f64
                    };
                    let per = cell.wall_ns as f64 / units;
                    let cpu_per = cell.cpu_ns as f64 / units;
                    let target_label =
                        if target == usize::MAX { "inf".to_string() } else { target.to_string() };
                    println!(
                        "{:<12} {:<7} {:>3} {:>7} {:>8} {:>10} {:>10.1} {:>9.2} {:>9.2}  {}",
                        layer,
                        engine,
                        threads,
                        target_label,
                        cell.buckets,
                        cell.terms,
                        cell.wall_ns as f64 / 1e6,
                        per,
                        cpu_per,
                        cell.detail,
                    );
                    if cfg.check {
                        match &reference {
                            None => reference = Some(output),
                            Some(want) if is_trotter(layer) => {
                                let cosine = output.overlap(want).re
                                    / (output.overlap(&output).re * want.overlap(want).re).sqrt();
                                println!(
                                    "  check: {} vs {} terms, 1 - cosine {:.2e}",
                                    output.len(),
                                    want.len(),
                                    1.0 - cosine
                                );
                            }
                            Some(want) => assert_terms_close(&output, want, 1e-9, "check"),
                        }
                    }
                }
            }
        }
    }
}

fn main() {
    let cfg = parse();
    match (cfg.qubits.div_ceil(64), cfg.threshold) {
        (1, None) => run::<1, _>(&cfg, &KeepAll),
        (2, None) => run::<2, _>(&cfg, &KeepAll),
        (1, Some(t)) => run::<1, _>(&cfg, &CoefficientThreshold(t)),
        (2, Some(t)) => run::<2, _>(&cfg, &CoefficientThreshold(t)),
        _ => panic!("--qubits must be at most 128"),
    }
}
