//! `--rotation-sparsity`: the structural diff of every gate of a cell's run and its summary.

use std::collections::BTreeMap;
use std::io::Write;

use paulistrings::test_support::{
    rebucket, single_channel_circuits, structural_diff, StructuralDiff,
};
use paulistrings::{propagate_with, Direction, LayerScratch, PropagateOptions, TruncationPolicy};
use serde_json::json;

use crate::cells::read_proc_status_kb;
use crate::cli::Config;
use crate::workloads::{build_base_sum, build_circuit, LayerKind};

/// One gate of the stepped run.
struct GateRow {
    gate: usize,
    step: usize,
    kind: String,
    /// Whether the gate's support misses every qubit any string of the sum acts on.
    disjoint: bool,
    diff: StructuralDiff,
}

/// Steps `layer`'s circuit gate by gate as `run_cell`'s single call would, diffs the sum across each gate of the `--sparsity-steps` (every step without it), appends one JSON line per diffed gate to `path`, and prints the summary.
pub(crate) fn run_rotation_sparsity<const W: usize, P>(
    layer: LayerKind,
    threads: usize,
    cfg: &Config,
    policy: &P,
    path: &str,
) where
    P: TruncationPolicy<W>,
{
    let options = PropagateOptions::default();
    let direction = Direction::Forward;
    let base = build_base_sum::<W>(layer, cfg);
    let circuit = build_circuit::<W>(layer, cfg.qubits, cfg.reps, (0, 1));
    let gates_per_step = if layer.is_trotter_step() { circuit.len() / cfg.reps } else { 1 };
    let (kinds, supports): (Vec<String>, Vec<[u64; W]>) = circuit
        .channels()
        .iter()
        .map(|channel| {
            let support = channel.support();
            let weight: u32 = support.iter().map(|w| w.count_ones()).sum();
            (format!("{}:{weight}q", channel.debug_name()), support)
        })
        .unzip();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build a rayon thread pool");
    let mut sidecar =
        std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap_or_else(|e| {
            eprintln!("phase_breakdown: cannot open --rotation-sparsity '{path}': {e}");
            std::process::exit(2);
        });

    let rows = pool.install(|| {
        let mut scratch = LayerScratch::<W>::new();
        let one_call =
            propagate_with(&circuit, base.clone(), policy, direction, &mut scratch, options).len();
        let mut sum = base;
        let mut rows = Vec::with_capacity(kinds.len());
        for (gate, ((single, kind), support)) in
            single_channel_circuits(circuit).into_iter().zip(kinds).zip(supports).enumerate()
        {
            let step = gate / gates_per_step;
            if !cfg.sparsity_steps.as_ref().is_none_or(|steps| steps.contains(&step)) {
                sum = propagate_with(&single, sum, policy, direction, &mut scratch, options);
                continue;
            }
            rebucket(&mut sum, options);
            let occupied = occupied_qubits(&sum);
            let disjoint = (0..W).all(|w| support[w] & occupied[w] == 0);
            let before = sum.clone();
            sum = propagate_with(&single, sum, policy, direction, &mut scratch, options);
            let diff = structural_diff(&before, &sum);
            drop(before);
            let row = GateRow { gate, step, kind, disjoint, diff };
            let (vmrss_kb, vmhwm_kb) = read_proc_status_kb();
            let line = json!({
                "layer": layer.name(),
                "truncation": cfg.truncation.label(),
                "words": W,
                "gate": row.gate,
                "step": row.step,
                "kind": row.kind,
                "disjoint": disjoint,
                "bits": sum.hash().bits(),
                "terms": diff.terms,
                "terms_after": diff.terms_after,
                "changed": diff.changed,
                "inserted": diff.inserted,
                "deleted": diff.deleted,
                "shift": diff.shift,
                "vmrss_kb": vmrss_kb,
            });
            writeln!(sidecar, "{line}").unwrap_or_else(|e| {
                eprintln!("phase_breakdown: writing --rotation-sparsity failed: {e}");
                std::process::exit(2);
            });
            if (gate + 1) % gates_per_step == 0 {
                eprintln!(
                    "phase_breakdown: {} step {} done: {} terms, VmHWM {:.1} GB",
                    layer.name(),
                    row.step,
                    diff.terms_after,
                    vmhwm_kb as f64 / 1e6,
                );
            }
            rows.push(row);
        }
        assert_eq!(
            sum.len(),
            one_call,
            "the gate-by-gate run must end on the one-call run's term count",
        );
        rows
    });
    print_summary::<W>(layer, cfg, &rows);
}

/// The qubits any string of `sum` acts on.
fn occupied_qubits<const W: usize>(sum: &paulistrings::PauliSum<W>) -> [u64; W] {
    sum.iter().fold([0; W], |mut occupied, (x, z, _)| {
        for w in 0..W {
            occupied[w] |= x[w] | z[w];
        }
        occupied
    })
}

/// Term-weighted totals of a group of gates.
#[derive(Default)]
struct Totals {
    gates: usize,
    /// Gates whose support misses the sum's occupied qubits.
    disjoint: usize,
    /// Terms in at those gates.
    disjoint_terms: usize,
    diff: StructuralDiff,
    /// Shift fractions of the gates that changed anything: a gate that commutes with every string shifts nothing and would only dilute the percentiles.
    shift_fractions: Vec<f64>,
}

impl Totals {
    fn add(&mut self, row: &GateRow) {
        let diff = row.diff;
        self.gates += 1;
        if row.disjoint {
            self.disjoint += 1;
            self.disjoint_terms += diff.terms;
        }
        self.diff = self.diff + diff;
        if diff.changed + diff.inserted + diff.deleted > 0 {
            self.shift_fractions.push(diff.shift as f64 / diff.terms_after as f64);
        }
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Nearest-rank percentile of `values`, sorted in place.
fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f64::total_cmp);
    let rank = ((p * values.len() as f64).ceil() as usize).clamp(1, values.len());
    values[rank - 1]
}

const COLUMNS: [&str; 13] = [
    "kind", "gates", "active", "disjoint", "terms", "anti", "ins", "del", "shift", "ins/anti",
    "p50", "p90", "skip",
];

fn print_group_header(label: &str) {
    let [kind, rest @ ..] = COLUMNS;
    let rest: Vec<String> = rest.iter().map(|c| format!("{c:>9}")).collect();
    println!("{label:>9} {kind:>16} {}", rest.join(" "));
}

fn print_group<const W: usize>(label: &str, kind: &str, totals: &mut Totals) {
    let d = totals.diff;
    let anticommuting = d.changed + d.deleted;
    let active = totals.shift_fractions.len();
    let p50 = percentile(&mut totals.shift_fractions, 0.5);
    let p90 = percentile(&mut totals.shift_fractions, 0.9);
    let fractions = [
        ratio(anticommuting, d.terms),
        ratio(d.inserted, d.terms),
        ratio(d.deleted, d.terms),
        ratio(d.shift, d.terms_after),
        ratio(d.inserted, anticommuting),
        p50,
        p90,
        ratio(totals.disjoint_terms, d.terms),
    ];
    let fractions: Vec<String> = fractions.iter().map(|f| format!("{f:>9.4}")).collect();
    println!(
        "{label:>9} {kind:>16} {:>9} {active:>9} {:>9} {:>9.3e} {}",
        totals.gates,
        totals.disjoint,
        d.terms as f64,
        fractions.join(" "),
    );
}

/// The per-kind and per-step tables: term-weighted fractions and per-gate shift percentiles.
fn print_summary<const W: usize>(layer: LayerKind, cfg: &Config, rows: &[GateRow]) {
    let mut by_kind: BTreeMap<&str, Totals> = BTreeMap::new();
    let mut by_step: BTreeMap<(usize, &str), Totals> = BTreeMap::new();
    for row in rows {
        by_kind.entry(&row.kind).or_default().add(row);
        by_step.entry((row.step, &row.kind)).or_default().add(row);
    }
    println!(
        "rotation-sparsity layer={} trunc={} reps={} W={W} gates={}",
        layer.name(),
        cfg.truncation.label(),
        cfg.reps,
        rows.len(),
    );
    println!(
        "anti, ins, del: over terms; shift: over terms after; active: gates that changed \
         anything; disjoint: gates off the occupied qubits; p50/p90: active per-gate shift; skip: \
         terms in at disjoint gates over terms"
    );
    print_group_header("steps");
    for (kind, totals) in &mut by_kind {
        print_group::<W>("all", kind, totals);
    }
    print_group_header("step");
    for ((step, kind), totals) in &mut by_step {
        print_group::<W>(&step.to_string(), kind, totals);
    }
}
