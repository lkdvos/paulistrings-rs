//! Ours-only A/B of the small-sum direct path, at the head-to-head study's
//! small-`m` configurations.
//!
//! This is gate (b) of experiment E3 (`research/notes/2026-09-01-small-m-path.md`):
//! the same circuit, the same observable, the same truncation, run with
//! `EngineSelection::SortedOnly` and with `EngineSelection::Auto`, interleaved
//! `abba` so a drift in machine state cannot masquerade as a win for whichever
//! selection always went second. Acceptance is `PROFILING.md`'s: **direction
//! consistency across every pair**, with the median Δ as the effect size.
//!
//! It is *not* a cross-engine benchmark — there is no Julia here. The
//! cross-engine numbers this is aimed at are
//! `benchmarks/python/jl_performance/README.md`'s kicked-Ising 2⁻⁴ (ratio 0.323)
//! and 2⁻⁶ (0.629) and XXZ 1e-2 (0.460) and 1e-3 (0.453), i.e. the four
//! configurations where PauliPropagation.jl is 1.6–3.1× faster than us.
//!
//! # Workloads
//!
//! Ported from `examples/common/circuits.py`, gate for gate, so the term-count
//! trajectories are the study's:
//!
//! - `kicked_ising` — 127-qubit heavy-hex (edge list read from
//!   `examples/data/heavy_hex_127.edges`), 5 Trotter steps, `theta_h = 5π/16`,
//!   `theta_zz = -π/2`, `x-then-zz` layer order, ZZ rotations emitted grouped by
//!   the same greedy edge coloring. Observable `Z₆₂`, Heisenberg. 1355 channels.
//! - `xxz` — open chain, `n = 100`, `Jz = 0.5`, `dt = 0.1`, 6 Trotter steps,
//!   even-odd bond order, three rotations per bond. Observable `Z₅₀`,
//!   Heisenberg. 1782 channels.
//!
//! Cutoffs: the four the study lost (kicked-Ising 2⁻⁴/2⁻⁶, XXZ 1e-2/1e-3), plus
//! one per workload from just *above* its crossover (kicked-Ising 2⁻⁸ at ratio
//! 1.126, XXZ 1e-4 at 0.895). The two extra ones are the control: they bound
//! what too high a threshold costs where the sorting engine already wins, and
//! their peaks cross the default threshold mid-run, so they exercise the
//! transition on a real workload rather than only in the tests.
//!
//! The port is self-checking: `--check` compares the final and peak term counts
//! against the study's committed values for every cutoff, so a divergence in the
//! circuit is a failure here rather than a silent difference in what was
//! measured.
//!
//! # Parity
//!
//! Every run records a `TermTrace`, and the two selections' full per-layer
//! `terms_in`/`terms_out` vectors must be **equal**, not close. That is the
//! same per-layer term-count parity the cross-engine driver gates on, checked
//! between our two paths instead of between two engines; a mismatch aborts
//! before any timing is reported.
//!
//! ```text
//! cargo build --release --example small_m_ab -p paulistrings
//! RAYON_NUM_THREADS=1 ./target/release/examples/small_m_ab --check
//! RAYON_NUM_THREADS=1 ./target/release/examples/small_m_ab --pairs 5
//! ```

use std::time::Instant;

use num_complex::Complex64;
use paulistrings::channel::PauliRotation;
use paulistrings::truncation::CoefficientThreshold;
use paulistrings::{
    propagate_with_scratch_and_options, BuildAccumulator, Circuit, Direction, EngineSelection,
    LayerScratch, PauliString, PauliSum, Phase, PropagateOptions, TermTrace,
    DEFAULT_SMALL_SUM_THRESHOLD,
};

/// Every workload here is at most 128 qubits.
const W: usize = 2;

// ---------------------------------------------------------------- circuits

/// Undirected edges, `lo hi` per line, `#` comments — the format of
/// `examples/data/heavy_hex_127.edges`.
fn read_edges(path: &str) -> Vec<(u32, u32)> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "cannot read the heavy-hex edge list at {path}: {e}\n\
             run from the repository root, or pass --edges <path>",
        )
    });
    let mut edges = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let a: u32 = it
            .next()
            .expect("edge line has no first index")
            .parse()
            .unwrap();
        let b: u32 = it
            .next()
            .expect("edge line has no second index")
            .parse()
            .unwrap();
        edges.push((a.min(b), a.max(b)));
    }
    edges.sort_unstable();
    edges
}

/// Greedy first-fit edge coloring in sorted edge order — the port of
/// `circuits.py::heavy_hex_edge_coloring`. Each class is a matching, i.e. a
/// hardware layer of disjoint-support ZZ rotations.
fn edge_coloring(edges: &[(u32, u32)]) -> Vec<Vec<(u32, u32)>> {
    let mut used: Vec<Vec<usize>> = Vec::new();
    let mut classes: Vec<Vec<(u32, u32)>> = Vec::new();
    let n = edges
        .iter()
        .map(|&(a, b)| a.max(b) as usize + 1)
        .max()
        .unwrap_or(0);
    used.resize(n, Vec::new());
    for &(a, b) in edges {
        let mut color = 0usize;
        while used[a as usize].contains(&color) || used[b as usize].contains(&color) {
            color += 1;
        }
        while classes.len() <= color {
            classes.push(Vec::new());
        }
        classes[color].push((a, b));
        used[a as usize].push(color);
        used[b as usize].push(color);
    }
    classes
}

/// Two-site generator on `(a, b)`, both factors `Z`.
fn zz(a: u32, b: u32) -> PauliString<W> {
    let mut g = PauliString::<W>::z(a);
    let (word, bit) = (b as usize / 64, 1u64 << (b % 64));
    g.z[word] |= bit;
    g
}

/// Two-site generator on `(a, b)` with both factors the same axis.
fn pair(a: u32, b: u32, axis: char) -> PauliString<W> {
    let mut g = PauliString::<W>::identity();
    for q in [a, b] {
        let (word, bit) = (q as usize / 64, 1u64 << (q % 64));
        match axis {
            'X' => g.x[word] |= bit,
            'Z' => g.z[word] |= bit,
            _ => {
                g.x[word] |= bit;
                g.z[word] |= bit;
            }
        }
    }
    g
}

/// `circuits.py::heavy_hex_kicked_ising(127, steps, theta_h, -pi/2)` with the
/// default `x-then-zz` order and colored ZZ layers.
fn kicked_ising(edges: &[(u32, u32)], n: usize, steps: usize, theta_h: f64) -> Circuit<W> {
    let theta_zz = -std::f64::consts::FRAC_PI_2;
    let zz_order: Vec<(u32, u32)> = edge_coloring(edges).into_iter().flatten().collect();
    let mut c = Circuit::<W>::new(n);
    for _ in 0..steps {
        for q in 0..n as u32 {
            c.push(PauliRotation::new(PauliString::<W>::x(q), theta_h));
        }
        for &(a, b) in &zz_order {
            c.push(PauliRotation::new(zz(a, b), theta_zz));
        }
    }
    c
}

/// `circuits.py::xxz_chain_trotter(n, steps, Jz, dt)` with the default even-odd
/// bond order.
fn xxz_chain(n: usize, steps: usize, jz: f64, dt: f64) -> Circuit<W> {
    let mut bonds: Vec<u32> = (0..n as u32 - 1).filter(|i| i % 2 == 0).collect();
    bonds.extend((0..n as u32 - 1).filter(|i| i % 2 == 1));
    let mut c = Circuit::<W>::new(n);
    for _ in 0..steps {
        for &i in &bonds {
            c.push(PauliRotation::new(pair(i, i + 1, 'X'), 2.0 * dt));
            c.push(PauliRotation::new(pair(i, i + 1, 'Y'), 2.0 * dt));
            c.push(PauliRotation::new(pair(i, i + 1, 'Z'), 2.0 * dt * jz));
        }
    }
    c
}

/// A single-term `Z_q` observable.
fn z_observable(num_qubits: usize, q: u32) -> PauliSum<W> {
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, 1);
    acc.add_term(PauliString::<W>::z(q), Phase::ONE, Complex64::new(1.0, 0.0));
    acc.finalize()
}

// ---------------------------------------------------------------- harness

struct Config {
    name: &'static str,
    /// `min_abs_coeff`, the study's swept knob.
    cutoff: f64,
    /// The study's committed `(final terms, peak terms)` for this cutoff, or
    /// `None` where the study did not report a recoverable peak.
    expect: Option<(usize, usize)>,
}

struct Run {
    /// Wall seconds per propagation.
    seconds: f64,
    /// Wall seconds of the whole timed region (`reps` propagations) — printed
    /// so the operator can see whether it cleared the governor's ramp.
    region_seconds: f64,
    trace: TermTrace,
    final_terms: usize,
}

/// One leg: `reps` propagations inside one timed region, reported per
/// propagation.
///
/// `reps > 1` is not optional at these sizes on the reference host — the
/// governor is `powersave`, and a timed region under ~50 ms is measured at
/// ~1200 MHz instead of ~3600 MHz (the phase-breakdown fact sheet's §0.1,
/// measured 3.62× at `m = 150`). The ratio this harness reports is between two
/// equally short legs and so survives that, but the absolute `us/L` columns do
/// not; pick `reps` so the region clears 200 ms and both are trustworthy. The
/// region's length is printed for exactly that check.
fn one_run(
    circuit: &Circuit<W>,
    observable: &PauliSum<W>,
    cutoff: f64,
    options: PropagateOptions,
    reps: usize,
) -> Run {
    let policy = CoefficientThreshold(cutoff);
    let mut scratch = LayerScratch::<W>::new();
    scratch.enable_term_trace();
    // Warm: propagate once untimed, exactly as the cross-engine protocol does,
    // so no reported number contains a cold cache or a first-touch page fault.
    let warm = propagate_with_scratch_and_options(
        circuit,
        observable.clone(),
        &policy,
        Direction::Heisenberg,
        &mut scratch,
        options,
    );
    let _ = scratch.take_term_trace();
    drop(warm);

    let mut final_terms = 0usize;
    let t0 = Instant::now();
    for _ in 0..reps {
        let out = propagate_with_scratch_and_options(
            circuit,
            observable.clone(),
            &policy,
            Direction::Heisenberg,
            &mut scratch,
            options,
        );
        final_terms = out.len();
    }
    let elapsed = t0.elapsed().as_secs_f64();
    // `reps` propagations appended `reps` identical copies of the trace (same
    // circuit, same input, same policy), so one call's worth is the head.
    let mut trace = scratch.take_term_trace().expect("tracing enabled");
    let layers = trace.terms_in.len() / reps;
    trace.terms_in.truncate(layers);
    trace.terms_out.truncate(layers);
    Run {
        seconds: elapsed / reps as f64,
        region_seconds: elapsed,
        trace,
        final_terms,
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let mut workload = String::from("all");
    let mut edges_path = String::from("examples/data/heavy_hex_127.edges");
    let mut pairs = 3usize;
    let mut threshold = DEFAULT_SMALL_SUM_THRESHOLD;
    let mut reps = 1usize;
    let mut check = false;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--workload" => {
                workload = args[i + 1].clone();
                i += 2;
            }
            "--edges" => {
                edges_path = args[i + 1].clone();
                i += 2;
            }
            "--pairs" => {
                pairs = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--reps" => {
                reps = args[i + 1].parse().unwrap();
                assert!(reps >= 1, "--reps must be at least 1");
                i += 2;
            }
            "--threshold" => {
                threshold = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--check" => {
                check = true;
                i += 1;
            }
            other => panic!(
                "unknown argument {other}\n\
                 usage: small_m_ab [--workload all|kicked_ising|xxz] [--edges PATH] \
                 [--pairs N] [--reps N] [--threshold N] [--check]",
            ),
        }
    }

    println!(
        "# small-sum direct path A/B (ours only, non-authoritative unless run \
         serialized on a quiet box)"
    );
    println!("# threads: {}", rayon::current_num_threads());
    println!("# small_sum_threshold: {threshold}");
    println!("# pairs per configuration: {pairs} (abba interleaved)");
    println!("# propagations per timed leg: {reps}");

    let sorted = PropagateOptions::default();
    let direct = PropagateOptions {
        engine: EngineSelection::Auto,
        small_sum_threshold: threshold,
    };

    let mut jobs: Vec<(&'static str, Circuit<W>, PauliSum<W>, Vec<Config>)> = Vec::new();
    if workload == "all" || workload == "kicked_ising" {
        let edges = read_edges(&edges_path);
        assert_eq!(edges.len(), 144, "expected the 144-edge Eagle map");
        let circuit = kicked_ising(&edges, 127, 5, 5.0 * std::f64::consts::PI / 16.0);
        assert_eq!(circuit.len(), 1355, "kicked-Ising channel count");
        jobs.push((
            "kicked_ising",
            circuit,
            z_observable(127, 62),
            vec![
                Config {
                    name: "2^-4",
                    cutoff: 0.0625,
                    expect: Some((7, 68)),
                },
                Config {
                    name: "2^-6",
                    cutoff: 0.015625,
                    expect: Some((408, 517)),
                },
                // Above the study's crossover (ratio 1.126, i.e. we already
                // win): here to bound what too high a threshold costs, and
                // because its peak crosses the default threshold mid-run.
                Config {
                    name: "2^-8",
                    cutoff: 0.00390625,
                    expect: Some((5038, 6311)),
                },
            ],
        ));
    }
    if workload == "all" || workload == "xxz" {
        let circuit = xxz_chain(100, 6, 0.5, 0.1);
        assert_eq!(circuit.len(), 1782, "XXZ channel count");
        jobs.push((
            "xxz",
            circuit,
            z_observable(100, 50),
            vec![
                Config {
                    name: "1e-2",
                    cutoff: 1e-2,
                    expect: Some((156, 164)),
                },
                Config {
                    name: "1e-3",
                    cutoff: 1e-3,
                    expect: Some((1625, 1625)),
                },
                // Above the study's crossover (0.895 — nearly a tie); same role
                // as kicked-Ising's 2^-8 above.
                Config {
                    name: "1e-4",
                    cutoff: 1e-4,
                    expect: Some((9918, 9918)),
                },
            ],
        ));
    }

    println!(
        "\n{:<13} {:>6} {:>8} {:>7} {:>8} {:>11} {:>11} {:>9} {:>9} {:>8} {:>6}",
        "workload",
        "cutoff",
        "final",
        "peak",
        "mean_m",
        "sorted_s",
        "direct_s",
        "sort_us/L",
        "dir_us/L",
        "speedup",
        "pairs",
    );
    println!(
        "# a `region_s` under 0.05 means the timed leg never left the governor's \
         idle clock -- raise --reps"
    );

    for (name, circuit, observable, configs) in &jobs {
        for cfg in configs {
            // One untimed run per selection first: it establishes term-count
            // parity, and validates the circuit port against the study.
            let a = one_run(circuit, observable, cfg.cutoff, sorted, reps);
            let b = one_run(circuit, observable, cfg.cutoff, direct, reps);
            assert_eq!(
                a.trace.terms_in, b.trace.terms_in,
                "{name} {}: per-layer terms_in differ between selections",
                cfg.name,
            );
            assert_eq!(
                a.trace.terms_out, b.trace.terms_out,
                "{name} {}: per-layer terms_out differ between selections",
                cfg.name,
            );
            let peak = a.trace.peak_terms().unwrap_or(a.final_terms);
            if let Some((want_final, want_peak)) = cfg.expect {
                let ok = a.final_terms == want_final && peak == want_peak;
                if check {
                    assert!(
                        ok,
                        "{name} {}: got final {} peak {peak}, study says final \
                         {want_final} peak {want_peak}",
                        cfg.name, a.final_terms,
                    );
                } else if !ok {
                    println!(
                        "# WARNING {name} {}: final {} peak {peak} vs study's \
                         {want_final}/{want_peak}",
                        cfg.name, a.final_terms,
                    );
                }
            }

            let mut region = 0.0f64;
            let mut sorted_s = Vec::with_capacity(pairs);
            let mut direct_s = Vec::with_capacity(pairs);
            let mut ratios = Vec::with_capacity(pairs);
            for p in 0..pairs {
                // abba: sorted first on even pairs, direct first on odd.
                let (s, d) = if p % 2 == 0 {
                    let s = one_run(circuit, observable, cfg.cutoff, sorted, reps);
                    let d = one_run(circuit, observable, cfg.cutoff, direct, reps);
                    (s, d)
                } else {
                    let d = one_run(circuit, observable, cfg.cutoff, direct, reps);
                    let s = one_run(circuit, observable, cfg.cutoff, sorted, reps);
                    (s, d)
                };
                region = s.region_seconds.min(d.region_seconds);
                let (s, d) = (s.seconds, d.seconds);
                sorted_s.push(s);
                direct_s.push(d);
                ratios.push(s / d);
            }
            let all_faster = ratios.iter().all(|&r| r > 1.0);
            let all_slower = ratios.iter().all(|&r| r < 1.0);
            let verdict = if all_faster || all_slower {
                format!("{}/{pairs}", pairs)
            } else {
                String::from("MIXED")
            };
            // Mean resident term count per layer — the `m` the fixed-cost
            // arithmetic in the phase-breakdown fact sheet is a function of, and
            // usually far below the peak because the sum starts at one term.
            let layers = a.trace.terms_in.len().max(1);
            let mean_m = a.trace.terms_in.iter().sum::<usize>() as f64 / layers as f64;
            let (med_sorted, med_direct) = (median(sorted_s.clone()), median(direct_s.clone()));
            println!(
                "{:<13} {:>6} {:>8} {:>7} {:>8.1} {:>11.5} {:>11.5} {:>9.3} {:>9.3} {:>8.3} {:>6}",
                name,
                cfg.name,
                a.final_terms,
                peak,
                mean_m,
                med_sorted,
                med_direct,
                med_sorted * 1e6 / layers as f64,
                med_direct * 1e6 / layers as f64,
                median(ratios.clone()),
                verdict,
            );
            print!("#   per-pair speedup:");
            for r in &ratios {
                print!(" {r:.3}");
            }
            println!("   region_s {region:.4}");
        }
    }
    println!(
        "\n# speedup = sorted / direct: > 1 means the direct path is faster. \
         A MIXED pairs column means no consistent change."
    );
}
