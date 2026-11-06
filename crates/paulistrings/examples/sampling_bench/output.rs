//! The printed report, the per-layer TSV and the JSON summary line.

use std::io::Write;

use crate::cli::{Backend, Config};
use crate::models::{GateKind, IsingQuench};
use crate::run::Report;

/// Totals of one gate kind over the trace.
#[derive(Default)]
struct KindTotals {
    layers: usize,
    nanos: u64,
    terms_in: u64,
}

/// What every output form shares, computed once from the trace.
struct Summary {
    total_ns: u64,
    layers: usize,
    /// Rows that carry a time: a run of fused layers is one pass.
    passes: usize,
    max_terms: usize,
    mean_terms: f64,
    peak_terms: usize,
    kinds: Vec<(GateKind, KindTotals)>,
    rebucket_ns: u64,
    prepare_ns: u64,
    coset_loop_ns: u64,
    finalize_ns: u64,
    /// Per step, the largest layer output and the summed layer times.
    step_max_terms: Vec<usize>,
    step_layer_ns: Vec<u64>,
}

fn summarize(report: &Report, kinds: &[GateKind], steps: usize) -> Summary {
    let rows = &report.trace.layers;
    let per_step = kinds.len();
    let mut by_kind: Vec<(GateKind, KindTotals)> =
        vec![(GateKind::Rx, KindTotals::default()), (GateKind::Rzz, KindTotals::default())];
    let mut summary = Summary {
        total_ns: report.step_ns.iter().sum(),
        layers: rows.len(),
        passes: rows.iter().filter(|row| row.nanos[0] > 0).count(),
        max_terms: rows.iter().map(|row| row.terms_out[0]).max().unwrap_or(0),
        mean_terms: rows.iter().map(|row| row.terms_out[0] as f64).sum::<f64>()
            / rows.len().max(1) as f64,
        peak_terms: report.trace.peak_terms().unwrap_or(0),
        kinds: Vec::new(),
        rebucket_ns: 0,
        prepare_ns: 0,
        coset_loop_ns: 0,
        finalize_ns: 0,
        step_max_terms: vec![0; steps],
        step_layer_ns: vec![0; steps],
    };
    for (i, row) in rows.iter().enumerate() {
        let kind = kinds[row.circuit_index as usize % per_step];
        let totals = &mut by_kind.iter_mut().find(|(k, _)| *k == kind).expect("both kinds").1;
        totals.layers += 1;
        totals.nanos += row.nanos[0];
        totals.terms_in += row.terms_in[0] as u64;
        let phases = &row.phases[0];
        summary.rebucket_ns += phases.rebucket_ns;
        summary.prepare_ns += phases.prepare_ns;
        summary.coset_loop_ns += phases.coset_loop_ns;
        summary.finalize_ns += phases.finalize_ns;
        let step = (i / per_step).min(steps - 1);
        summary.step_max_terms[step] = summary.step_max_terms[step].max(row.terms_out[0]);
        summary.step_layer_ns[step] += row.nanos[0];
    }
    summary.kinds = by_kind.into_iter().filter(|(_, totals)| totals.layers > 0).collect();
    summary
}

fn ms(ns: u64) -> f64 {
    ns as f64 * 1e-6
}

pub(crate) fn print_report(
    cfg: &Config,
    quench: &IsingQuench,
    words: usize,
    report: &Report,
    kinds: &[GateKind],
) {
    let summary = summarize(report, kinds, cfg.steps);
    let backend = match cfg.backend {
        Backend::Cpu => format!("cpu threads={}", cfg.threads),
        Backend::Gpu => format!("gpu device={}", cfg.device),
    };
    println!(
        "model={} qubits={} W={words} observable={} steps={} channels/step={} dtype={} {backend} truncation={}{}",
        quench.name,
        quench.num_qubits,
        quench.observable_label(),
        cfg.steps,
        kinds.len(),
        cfg.dtype,
        cfg.truncation.label(),
        if cfg.one_call { " one-call" } else { "" },
    );
    println!(
        "{:>5} {:>11} {:>13} {:>13} {:>14}",
        "step", "wall_ms", "terms_after", "max_layer", "readout"
    );
    for step in 0..cfg.steps {
        let wall = if cfg.one_call { summary.step_layer_ns[step] } else { report.step_ns[step] };
        let terms = if cfg.one_call {
            report.trace.layers[(step + 1) * kinds.len() - 1].terms_out[0]
        } else {
            report.step_terms[step]
        };
        let readout = report.readouts.get(step).map_or(String::new(), |r| format!("{r:+.9}"));
        println!(
            "{:>5} {:>11.3} {:>13} {:>13} {:>14}",
            step + 1,
            ms(wall),
            terms,
            summary.step_max_terms[step],
            readout
        );
    }
    println!(
        "total_s={:.4} mean_step_ms={:.3} layers={} passes={} terms/layer max={} mean={:.0} peak={}",
        summary.total_ns as f64 * 1e-9,
        ms(summary.total_ns) / cfg.steps as f64,
        summary.layers,
        summary.passes,
        summary.max_terms,
        summary.mean_terms,
        summary.peak_terms,
    );
    for (kind, totals) in &summary.kinds {
        println!(
            "  {:<4} layers={} total_ms={:.1} mean_us={:.1} mean_terms_in={:.0}",
            kind.name(),
            totals.layers,
            ms(totals.nanos),
            totals.nanos as f64 * 1e-3 / totals.layers as f64,
            totals.terms_in as f64 / totals.layers as f64,
        );
    }
    println!(
        "  layer phases ms: rebucket={:.1} prepare={:.1} coset_loop={:.1} finalize={:.1}",
        ms(summary.rebucket_ns),
        ms(summary.prepare_ns),
        ms(summary.coset_loop_ns),
        ms(summary.finalize_ns),
    );
    if cfg.backend == Backend::Gpu {
        println!(
            "  upload_ms={:.1} download_ms={:.1} clocks(sm,mem)={}",
            ms(report.upload_ns),
            ms(report.download_ns),
            report.clocks.as_deref().unwrap_or("unknown"),
        );
    }
    #[cfg(feature = "phase-timing")]
    print_phase_stats(&report.stats);
    println!("readout <0|O|0> = {:+.12}", report.final_readout);
    if let Some(path) = &cfg.layers_out {
        write_layers(path, report, kinds);
    }
    if let Some(path) = &cfg.json_out {
        append_json(path, cfg, quench, words, report, &summary);
    }
}

#[cfg(feature = "phase-timing")]
fn print_phase_stats(stats: &paulistrings::PhaseStats) {
    println!(
        "  phase-timing ms: prepare={:.1} rebucket={:.1} gather={:.1} sort={:.1} merge={:.1} compact={:.1} rescale={:.1} coset_loop={:.1} finalize={:.1} truncate={:.1} h2d={:.1} d2h={:.1}",
        ms(stats.prepare_ns),
        ms(stats.rebucket_ns),
        ms(stats.gather_ns),
        ms(stats.sort_ns),
        ms(stats.merge_ns),
        ms(stats.compact_ns),
        ms(stats.rescale_ns),
        ms(stats.coset_loop_ns),
        ms(stats.finalize_ns),
        ms(stats.truncate_ns),
        ms(stats.h2d_ns),
        ms(stats.d2h_ns),
    );
    println!(
        "  phase-timing counts: layers={} cosets={} runs={} rows_gathered={} rows_sorted={}",
        stats.layers, stats.cosets, stats.runs, stats.rows_gathered, stats.rows_sorted,
    );
}

fn write_layers(path: &str, report: &Report, kinds: &[GateKind]) {
    let mut file = std::fs::File::create(path).unwrap_or_else(|e| {
        eprintln!("sampling_bench: cannot create --layers-out '{path}': {e}");
        std::process::exit(2);
    });
    let mut text = String::from(
        "application\tcircuit_index\tkind\tbits\tterms_in\tterms_out\tnanos\trebucket_ns\tprepare_ns\tcoset_loop_ns\tfinalize_ns\trecords\tbatches\tpermuted\trescaled\n",
    );
    for row in &report.trace.layers {
        let phases = &row.phases[0];
        let device = row.device[0].unwrap_or_default();
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.application_index,
            row.circuit_index,
            kinds[row.circuit_index as usize % kinds.len()].name(),
            row.bits[0],
            row.terms_in[0],
            row.terms_out[0],
            row.nanos[0],
            phases.rebucket_ns,
            phases.prepare_ns,
            phases.coset_loop_ns,
            phases.finalize_ns,
            device.records,
            device.batches,
            u8::from(device.permuted),
            u8::from(device.rescaled),
        ));
    }
    file.write_all(text.as_bytes()).expect("writing --layers-out");
}

fn append_json(
    path: &str,
    cfg: &Config,
    quench: &IsingQuench,
    words: usize,
    report: &Report,
    summary: &Summary,
) {
    let kinds: serde_json::Map<String, serde_json::Value> = summary
        .kinds
        .iter()
        .map(|(kind, totals)| {
            let value = serde_json::json!({
                "layers": totals.layers,
                "nanos": totals.nanos,
                "mean_terms_in": totals.terms_in as f64 / totals.layers as f64,
            });
            (kind.name().to_string(), value)
        })
        .collect();
    let line = serde_json::json!({
        "model": quench.name,
        "qubits": quench.num_qubits,
        "words": words,
        "observable": quench.observable_label(),
        "steps": cfg.steps,
        "dtype": cfg.dtype,
        "backend": match cfg.backend { Backend::Cpu => "cpu", Backend::Gpu => "gpu" },
        "threads": cfg.threads,
        "device": cfg.device,
        "truncation": cfg.truncation.label(),
        "one_call": cfg.one_call,
        "total_ns": summary.total_ns,
        "step_ns": report.step_ns,
        "step_terms": report.step_terms,
        "layers": summary.layers,
        "passes": summary.passes,
        "max_terms": summary.max_terms,
        "mean_terms": summary.mean_terms,
        "peak_terms": summary.peak_terms,
        "kinds": kinds,
        "rebucket_ns": summary.rebucket_ns,
        "prepare_ns": summary.prepare_ns,
        "coset_loop_ns": summary.coset_loop_ns,
        "finalize_ns": summary.finalize_ns,
        "upload_ns": report.upload_ns,
        "download_ns": report.download_ns,
        "clocks": report.clocks,
        "readouts": report.readouts,
        "readout": report.final_readout,
    });
    let mut file =
        std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap_or_else(|e| {
            eprintln!("sampling_bench: cannot open --json-out '{path}': {e}");
            std::process::exit(2);
        });
    writeln!(file, "{line}").expect("writing --json-out");
}
