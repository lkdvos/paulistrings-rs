//! `presentation-bench <naive|threadmaps|mergesort|bucketed> [flags]`
//!
//! One JSON line per timed repetition (stdout and `--json-out`), preceded per
//! cell by the `cell layer=... threads=... n=... layers=... wall_ms=...` line
//! that `scripts/perf-stat.sh` greps. Threads come from a dedicated Rayon pool
//! installed around the whole timed region (warm-up included).

use paulistrings::PropagateOptions;
use presentation_bench::common::RunResult;
use presentation_bench::{bucketed, mergesort, naive, threadmaps, workload};
use std::fmt::Write as _;
use std::io::Write as _;
use std::time::Instant;

const USAGE: &str = "usage: presentation-bench <naive|threadmaps|mergesort|bucketed>
    [--eps F]               coefficient threshold (default 2^-13)
    [--steps N]             Trotter steps (default 5)
    [--edges PATH]          heavy-hex edge list
    [--threads csv]         e.g. 1,8,32 (default 1)
    [--target-bucket-len N] bucketed only (default 1024)
    [--min-buckets N]       bucketed only (default 128, must be >= 16)
    [--reps N]              timed repetitions after one warm-up (default 3)
    [--no-warmup]
    [--tag STR]             free label, e.g. default|native
    [--json-out FILE]       append JSON lines
    [--layer-times]         bucketed only: per-layer wall vector
    [--terms-only]          one propagation, print the term trajectory, no timing";

#[derive(Clone)]
struct Args {
    variant: String,
    eps: f64,
    steps: usize,
    edges: String,
    threads: Vec<usize>,
    target_bucket_len: usize,
    min_buckets: usize,
    reps: usize,
    warmup: bool,
    tag: String,
    json_out: Option<String>,
    layer_times: bool,
    terms_only: bool,
}

fn parse() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    let mut a = Args {
        variant: argv[0].clone(),
        eps: workload::DEFAULT_EPS,
        steps: workload::STEPS,
        edges: workload::default_edges_path(),
        threads: vec![1],
        target_bucket_len: 1024,
        min_buckets: 128,
        reps: 3,
        warmup: true,
        tag: "default".into(),
        json_out: None,
        layer_times: false,
        terms_only: false,
    };
    let mut i = 1;
    let val = |i: &mut usize| -> String {
        *i += 1;
        argv.get(*i).unwrap_or_else(|| { eprintln!("{USAGE}"); std::process::exit(2) }).clone()
    };
    while i < argv.len() {
        match argv[i].as_str() {
            "--eps" => a.eps = val(&mut i).parse().unwrap(),
            "--steps" => a.steps = val(&mut i).parse().unwrap(),
            "--edges" => a.edges = val(&mut i),
            "--threads" => a.threads = val(&mut i).split(',').map(|s| s.trim().parse().unwrap()).collect(),
            "--target-bucket-len" => a.target_bucket_len = val(&mut i).parse().unwrap(),
            "--min-buckets" => a.min_buckets = val(&mut i).parse().unwrap(),
            "--reps" => a.reps = val(&mut i).parse().unwrap(),
            "--no-warmup" => a.warmup = false,
            "--tag" => a.tag = val(&mut i),
            "--json-out" => a.json_out = Some(val(&mut i)),
            "--layer-times" => a.layer_times = true,
            "--terms-only" => a.terms_only = true,
            other => {
                eprintln!("unknown flag {other}\n{USAGE}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if !["naive", "threadmaps", "mergesort", "bucketed"].contains(&a.variant.as_str()) {
        eprintln!("unknown variant {}\n{USAGE}", a.variant);
        std::process::exit(2);
    }
    a
}

fn proc_status_kb(field: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    s.lines().find(|l| l.starts_with(field)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok())
}

fn sh(cmd: &str, args: &[&str]) -> String {
    std::process::Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

struct Provenance {
    commit: String,
    rustc: String,
    host: String,
    date: String,
    cpu_model: String,
    governor: String,
}

fn provenance() -> Provenance {
    let dirty = !sh("git", &["status", "--porcelain", "--", "crates", "presentation/bench", "Cargo.toml"]).is_empty();
    let commit = format!("{}{}", sh("git", &["rev-parse", "--short", "HEAD"]), if dirty { "-dirty" } else { "" });
    let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("model name")).map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string()))
        .unwrap_or_default();
    let governor = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    Provenance {
        commit,
        rustc: sh("rustc", &["-V"]),
        host: sh("hostname", &["-s"]),
        date: sh("date", &["+%F"]),
        cpu_model,
        governor,
    }
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn json_vec<T: std::fmt::Display>(v: &[T]) -> String {
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        write!(s, "{x}").unwrap();
    }
    s.push(']');
    s
}

fn run_once(a: &Args, circuit: &paulistrings::Circuit<{ workload::W }>, layers: Option<&[paulistrings::Circuit<{ workload::W }>]>, obs: &paulistrings::PauliSum<{ workload::W }>, threads: usize, options: PropagateOptions) -> RunResult {
    match a.variant.as_str() {
        "naive" => naive::run(circuit, obs, a.eps),
        "threadmaps" => threadmaps::run(circuit, obs, a.eps, threads),
        "mergesort" => mergesort::run(circuit, obs, a.eps, threads),
        _ => bucketed::run(circuit, layers, obs, a.eps, options),
    }
}

fn main() {
    let a = parse();
    if std::env::var_os("RUST_LOG").is_some() {
        eprintln!("warning: RUST_LOG is set; campaigns run with it unset (CLAUDE.md §Performance discipline)");
    }
    let prov = provenance();
    let circuit = workload::talk_circuit(&a.edges, a.steps);
    let layers = if a.layer_times && a.variant == "bucketed" { Some(workload::talk_layers(&a.edges, a.steps)) } else { None };
    let obs = workload::z_observable(workload::QUBITS, workload::OBSERVABLE_QUBIT);
    let options = bucketed::options(a.target_bucket_len, a.min_buckets);
    let floor_kb = proc_status_kb("VmRSS:").unwrap_or(0);

    if a.terms_only {
        let r = run_once(&a, &circuit, None, &obs, 1, options);
        println!(
            "{{\"variant\":{},\"eps\":{:e},\"steps\":{},\"layers\":{},\"peak_terms\":{},\"final_terms\":{},\"terms_out\":{}}}",
            json_str(&a.variant), a.eps, a.steps, circuit.len(), r.peak_terms(), r.sum.len(), json_vec(&r.terms_out)
        );
        return;
    }

    let mut out_file = a.json_out.as_ref().map(|p| {
        std::fs::OpenOptions::new().create(true).append(true).open(p).expect("open --json-out")
    });

    for &threads in &a.threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let (results, region_s): (Vec<RunResult>, f64) = pool.install(|| {
            if a.warmup {
                let _ = run_once(&a, &circuit, layers.as_deref(), &obs, threads, options);
            }
            let t0 = Instant::now();
            let mut v = Vec::with_capacity(a.reps);
            for _ in 0..a.reps {
                v.push(run_once(&a, &circuit, layers.as_deref(), &obs, threads, options));
            }
            (v, t0.elapsed().as_secs_f64())
        });
        let mut walls: Vec<u64> = results.iter().map(|r| r.wall_ns).collect();
        walls.sort_unstable();
        let median_ms = walls[walls.len() / 2] as f64 / 1e6;
        let peak = results[0].peak_terms();
        println!(
            "cell layer={} threads={} n={} layers={} wall_ms={:.3} trunc=coeff:{:e}",
            a.variant, threads, peak, circuit.len(), median_ms, a.eps
        );
        let vmhwm = proc_status_kb("VmHWM:").unwrap_or(0);
        let vmrss = proc_status_kb("VmRSS:").unwrap_or(0);
        for (rep, r) in results.iter().enumerate() {
            let mut line = String::from("{");
            write!(line, "\"layer\":{},\"tag\":{},\"threads\":{},\"target_bucket_len\":{},\"min_buckets\":{},",
                json_str(&a.variant), json_str(&a.tag), threads, a.target_bucket_len, a.min_buckets).unwrap();
            write!(line, "\"eps\":{:e},\"steps\":{},\"qubits\":{},\"layers\":{},\"n\":{},\"rep\":{},\"wall_ns\":{},\"region_s\":{:.3},",
                a.eps, a.steps, workload::QUBITS, circuit.len(), r.peak_terms(), rep, r.wall_ns, region_s).unwrap();
            write!(line, "\"peak_terms\":{},\"final_terms\":{},\"terms_out\":{},", r.peak_terms(), r.sum.len(), json_vec(&r.terms_out)).unwrap();
            match &r.layer_wall_ns {
                Some(v) => write!(line, "\"layer_wall_ns\":{},", json_vec(v)).unwrap(),
                None => line.push_str("\"layer_wall_ns\":null,"),
            }
            write!(line, "\"vmrss_kb\":{},\"vmhwm_kb\":{},\"floor_kb\":{},", vmrss, vmhwm, floor_kb).unwrap();
            match &r.buckets {
                Some(b) => write!(line, "\"buckets\":{},", b).unwrap(),
                None => line.push_str("\"buckets\":null,"),
            }
            match &r.phase {
                Some(p) => write!(line,
                    "\"rebucket_ns\":{},\"prepare_ns\":{},\"coset_loop_ns\":{},\"finalize_ns\":{},\"gather_ns\":{},\"sort_ns\":{},\"merge_ns\":{},\"busy_total_ns\":{},\"cosets\":{},\"runs\":{},\"rows_gathered\":{},\"rows_sorted\":{},\"rows_id\":{},\"terms_in\":{},",
                    p.rebucket_ns, p.prepare_ns, p.coset_loop_ns, p.finalize_ns, p.gather_ns, p.sort_ns, p.merge_ns,
                    p.busy_total_ns(), p.cosets, p.runs, p.rows_gathered, p.rows_sorted, p.rows_id, p.terms_in).unwrap(),
                None => line.push_str("\"rebucket_ns\":null,\"prepare_ns\":null,\"coset_loop_ns\":null,\"finalize_ns\":null,\"gather_ns\":null,\"sort_ns\":null,\"merge_ns\":null,\"busy_total_ns\":null,\"cosets\":null,\"runs\":null,\"rows_gathered\":null,\"rows_sorted\":null,\"rows_id\":null,\"terms_in\":null,"),
            }
            write!(line, "\"commit\":{},\"rustc\":{},\"host\":{},\"date\":{},\"cpu_model\":{},\"governor\":{}}}",
                json_str(&prov.commit), json_str(&prov.rustc), json_str(&prov.host), json_str(&prov.date),
                json_str(&prov.cpu_model), json_str(&prov.governor)).unwrap();
            println!("{line}");
            if let Some(f) = out_file.as_mut() {
                writeln!(f, "{line}").unwrap();
            }
        }
    }
}
