//! The printed report, the per-layer TSV and the JSON summary line.

use std::io::Write;

use paulistrings::{PartitionLayerRecord, PartitionTrace};

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

/// What every output form shares: the untraced run's figures, and the traced run's when `--trace` ran one.
struct Summary {
    total_ns: u64,
    /// The largest `terms_after`, exact after each `propagate` call.
    peak_step_terms: usize,
    traced: Option<TraceSummary>,
}

/// Computed once from the trace; a layer's terms are summed over partitions and its time is the slowest partition's.
struct TraceSummary {
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
    /// Per step, the largest layer output.
    step_max_terms: Vec<usize>,
    partitions: PartitionSummary,
}

/// Per-partition totals over the trace, in rank order.
struct PartitionSummary {
    terms_final: Vec<usize>,
    terms_max: Vec<usize>,
    layer_ns: Vec<u64>,
    /// Wall of the layers that exchanged.
    remote_layer_ns: Vec<u64>,
    /// Group calls: bucket-count agreement and the policy's collectives (a collapse's prefix all-reduces).
    collective_ns: Vec<u64>,
    finalize_ns: Vec<u64>,
    rows_sent: Vec<u64>,
    bytes_sent: Vec<u64>,
    remote_layers: usize,
}

fn layer_terms(row: &PartitionLayerRecord) -> usize {
    row.terms_out.iter().sum()
}

fn layer_nanos(row: &PartitionLayerRecord) -> u64 {
    row.nanos.iter().copied().max().unwrap_or(0)
}

fn summarize_partitions(rows: &[PartitionLayerRecord]) -> PartitionSummary {
    let size = rows.first().map_or(1, |row| row.terms_out.len());
    let mut out = PartitionSummary {
        terms_final: rows.last().map_or(vec![0; size], |row| row.terms_out.clone()),
        terms_max: vec![0; size],
        layer_ns: vec![0; size],
        remote_layer_ns: vec![0; size],
        collective_ns: vec![0; size],
        finalize_ns: vec![0; size],
        rows_sent: vec![0; size],
        bytes_sent: vec![0; size],
        remote_layers: rows.iter().filter(|row| row.remote_deltas > 0).count(),
    };
    for row in rows {
        for p in 0..size {
            out.terms_max[p] = out.terms_max[p].max(row.terms_out[p]);
            out.layer_ns[p] += row.nanos[p];
            if row.remote_deltas > 0 {
                out.remote_layer_ns[p] += row.nanos[p];
            }
            out.collective_ns[p] += row.phases[p].collective_ns;
            out.finalize_ns[p] += row.phases[p].finalize_ns;
            out.rows_sent[p] += row.rows_sent.get(p).map_or(0, |to| to.iter().sum());
            out.bytes_sent[p] += row.bytes_sent.get(p).map_or(0, |to| to.iter().sum());
        }
    }
    out
}

fn summarize(report: &Report, kinds: &[GateKind], steps: usize) -> Summary {
    Summary {
        total_ns: report.step_ns.iter().sum(),
        peak_step_terms: report.step_terms.iter().copied().max().unwrap_or(0),
        traced: report.trace.as_ref().map(|trace| summarize_trace(report, trace, kinds, steps)),
    }
}

fn summarize_trace(
    report: &Report,
    trace: &PartitionTrace,
    kinds: &[GateKind],
    steps: usize,
) -> TraceSummary {
    let rows = &trace.layers;
    let per_step = kinds.len();
    let mut by_kind: Vec<(GateKind, KindTotals)> =
        vec![(GateKind::Rx, KindTotals::default()), (GateKind::Rzz, KindTotals::default())];
    let mut summary = TraceSummary {
        total_ns: report.traced_ns.unwrap_or(0),
        layers: rows.len(),
        passes: rows.iter().filter(|row| layer_nanos(row) > 0).count(),
        max_terms: rows.iter().map(layer_terms).max().unwrap_or(0),
        mean_terms: rows.iter().map(|row| layer_terms(row) as f64).sum::<f64>()
            / rows.len().max(1) as f64,
        peak_terms: trace.peak_terms().unwrap_or(0),
        kinds: Vec::new(),
        rebucket_ns: 0,
        prepare_ns: 0,
        coset_loop_ns: 0,
        finalize_ns: 0,
        step_max_terms: vec![0; steps],
        partitions: summarize_partitions(rows),
    };
    for (i, row) in rows.iter().enumerate() {
        let kind = kinds[row.circuit_index as usize % per_step];
        let totals = &mut by_kind.iter_mut().find(|(k, _)| *k == kind).expect("both kinds").1;
        totals.layers += 1;
        totals.nanos += layer_nanos(row);
        totals.terms_in += row.terms_in.iter().sum::<usize>() as u64;
        // Phases are partition 0's; the per-partition section carries the rest.
        let phases = &row.phases[0];
        summary.rebucket_ns += phases.rebucket_ns;
        summary.prepare_ns += phases.prepare_ns;
        summary.coset_loop_ns += phases.coset_loop_ns;
        summary.finalize_ns += phases.finalize_ns;
        let step = (i / per_step).min(steps - 1);
        summary.step_max_terms[step] = summary.step_max_terms[step].max(layer_terms(row));
    }
    summary.kinds = by_kind.into_iter().filter(|(_, totals)| totals.layers > 0).collect();
    summary
}

fn ms(ns: u64) -> f64 {
    ns as f64 * 1e-6
}

fn rows_label(report: &Report) -> String {
    report.rows.as_ref().map_or("-".to_string(), |rows| rows.label.clone())
}

fn backend_label(cfg: &Config, report: &Report) -> String {
    match cfg.backend {
        Backend::Cpu => format!("cpu threads={}", cfg.threads),
        Backend::Cpus => format!(
            "cpus partitions={} threads={} rows={}",
            cfg.partitions,
            cfg.threads,
            rows_label(report)
        ),
        Backend::Gpu => format!("gpu device={}", cfg.device.unwrap_or(0)),
        Backend::Gpus | Backend::MpiGpu => format!(
            "{} partitions={} devices={:?} rows={}",
            cfg.backend.name(),
            report.devices.len(),
            report.devices,
            rows_label(report)
        ),
    }
}

pub(crate) fn print_report(
    cfg: &Config,
    quench: &IsingQuench,
    words: usize,
    report: &Report,
    kinds: &[GateKind],
) {
    let summary = summarize(report, kinds, cfg.steps);
    println!(
        "model={} qubits={} W={words} observable={} steps={} channels/step={} dtype={} {} truncation={}",
        quench.name,
        quench.num_qubits,
        quench.observable_label(),
        cfg.steps,
        kinds.len(),
        cfg.dtype,
        backend_label(cfg, report),
        cfg.truncation.label(),
    );
    println!("{:>5} {:>11} {:>13}", "step", "wall_ms", "terms_after");
    for step in 0..cfg.steps {
        println!(
            "{:>5} {:>11.3} {:>13}",
            step + 1,
            ms(report.step_ns[step]),
            report.step_terms[step]
        );
    }
    println!(
        "total_s={:.4} mean_step_ms={:.3} layers={} peak_terms_after={}",
        summary.total_ns as f64 * 1e-9,
        ms(summary.total_ns) / cfg.steps as f64,
        cfg.steps * kinds.len(),
        summary.peak_step_terms,
    );
    if cfg.backend != Backend::Cpu {
        println!(
            "  upload_ms={:.1} download_ms={:.1} clocks(sm,mem)={}",
            ms(report.upload_ns),
            ms(report.download_ns),
            report.clocks.as_deref().unwrap_or("unknown"),
        );
    }
    if let Some(traced) = &summary.traced {
        print_trace(report, traced);
    }
    #[cfg(feature = "phase-timing")]
    print_phase_stats(&report.stats);
    println!("readout <0|O|0> = {:+.12}", report.final_readout);
    if let Some(path) = &cfg.layers_out {
        write_layers(path, report, kinds);
    }
    if let Some(path) = &cfg.json_out {
        append_json(path, cfg, quench, words, report, &summary, cfg.steps * kinds.len());
    }
}

/// The figures of the traced run, which settles exact device term counts and so runs slower than the reported one.
fn print_trace(report: &Report, traced: &TraceSummary) {
    println!(
        "traced run: total_s={:.4} layers={} passes={} terms/layer max={} mean={:.0} peak={}",
        traced.total_ns as f64 * 1e-9,
        traced.layers,
        traced.passes,
        traced.max_terms,
        traced.mean_terms,
        traced.peak_terms,
    );
    println!("  max_layer per step: {:?}", traced.step_max_terms);
    for (kind, totals) in &traced.kinds {
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
        "  layer phases ms (partition 0): rebucket={:.1} prepare={:.1} coset_loop={:.1} finalize={:.1}",
        ms(traced.rebucket_ns),
        ms(traced.prepare_ns),
        ms(traced.coset_loop_ns),
        ms(traced.finalize_ns),
    );
    if traced.partitions.terms_final.len() > 1 {
        print_partitions(report, &traced.partitions, traced.layers);
    }
}

fn print_partitions(report: &Report, parts: &PartitionSummary, layers: usize) {
    let total_rows: u64 = parts.rows_sent.iter().sum();
    let total_bytes: u64 = parts.bytes_sent.iter().sum();
    println!(
        "  partitions: remote_layers={} of {layers} remote_channels/step={} rows_exchanged={total_rows} bytes_exchanged={total_bytes}",
        parts.remote_layers,
        report.rows.as_ref().and_then(|rows| rows.remote_channels).map_or("-".to_string(), |c| c.to_string()),
    );
    println!(
        "  {:>4} {:>6} {:>13} {:>13} {:>11} {:>15} {:>13} {:>11} {:>12}",
        "part",
        "device",
        "terms_final",
        "terms_max",
        "layer_ms",
        "remote_layer_ms",
        "collective_ms",
        "finalize_ms",
        "rows_sent"
    );
    for p in 0..parts.terms_final.len() {
        println!(
            "  {:>4} {:>6} {:>13} {:>13} {:>11.1} {:>15.1} {:>13.1} {:>11.1} {:>12}",
            p,
            report.devices.get(p).copied().unwrap_or(0),
            parts.terms_final[p],
            parts.terms_max[p],
            ms(parts.layer_ns[p]),
            ms(parts.remote_layer_ns[p]),
            ms(parts.collective_ns[p]),
            ms(parts.finalize_ns[p]),
            parts.rows_sent[p],
        );
    }
}

#[cfg(feature = "phase-timing")]
fn print_phase_stats(per_partition: &[paulistrings::PhaseStats]) {
    let stats = &per_partition[0];
    println!(
        "  phase-timing ms (partition 0): prepare={:.1} rebucket={:.1} gather={:.1} sort={:.1} merge={:.1} compact={:.1} rescale={:.1} coset_loop={:.1} finalize={:.1} truncate={:.1} h2d={:.1} d2h={:.1}",
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
    if per_partition.len() > 1 {
        for (p, stats) in per_partition.iter().enumerate() {
            println!(
                "  phase-timing partition {p}: export={:.1} exchange={:.1} chunk_wait={:.1} truncate={:.1} ms",
                ms(stats.export_ns),
                ms(stats.exchange_ns),
                ms(stats.chunk_wait_ns),
                ms(stats.truncate_ns),
            );
        }
    }
}

fn write_layers(path: &str, report: &Report, kinds: &[GateKind]) {
    let mut file = std::fs::File::create(path).unwrap_or_else(|e| {
        eprintln!("sampling_bench: cannot create --layers-out '{path}': {e}");
        std::process::exit(2);
    });
    let mut text = String::from(
        "application\tcircuit_index\tkind\tbits\tterms_in\tterms_out\tnanos\trebucket_ns\tprepare_ns\tcoset_loop_ns\tfinalize_ns\trecords\tbatches\tpermuted\trescaled\tremote_deltas\trows_exchanged\tterms_out_per_partition\n",
    );
    for row in &report.trace.as_ref().expect("--layers-out needs --trace").layers {
        let phases = &row.phases[0];
        let device = row.device[0].unwrap_or_default();
        let rows_exchanged: u64 = row.rows_sent.iter().flatten().sum();
        let per_partition =
            row.terms_out.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.application_index,
            row.circuit_index,
            kinds[row.circuit_index as usize % kinds.len()].name(),
            row.bits[0],
            row.terms_in.iter().sum::<usize>(),
            layer_terms(row),
            layer_nanos(row),
            phases.rebucket_ns,
            phases.prepare_ns,
            phases.coset_loop_ns,
            phases.finalize_ns,
            device.records,
            device.batches,
            u8::from(device.permuted),
            u8::from(device.rescaled),
            row.remote_deltas,
            rows_exchanged,
            per_partition,
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
    layers: usize,
) {
    let traced = summary.traced.as_ref();
    let kinds = traced.map(|t| {
        t.kinds
            .iter()
            .map(|(kind, totals)| {
                let value = serde_json::json!({
                    "layers": totals.layers,
                    "nanos": totals.nanos,
                    "mean_terms_in": totals.terms_in as f64 / totals.layers as f64,
                });
                (kind.name().to_string(), value)
            })
            .collect::<serde_json::Map<String, serde_json::Value>>()
    });
    let parts = traced.map(|t| &t.partitions);
    #[cfg(feature = "phase-timing")]
    let (export_ns, exchange_ns): (Vec<u64>, Vec<u64>) =
        report.stats.iter().map(|s| (s.export_ns, s.exchange_ns)).unzip();
    #[cfg(not(feature = "phase-timing"))]
    let (export_ns, exchange_ns): (Option<Vec<u64>>, Option<Vec<u64>>) = (None, None);
    let partitions = serde_json::json!({
        "count": if cfg.backend == Backend::Cpus { cfg.partitions } else { report.devices.len().max(1) },
        "devices": report.devices,
        "rows": report.rows.as_ref().map(|rows| &rows.label),
        "remote_channels_per_step": report.rows.as_ref().and_then(|rows| rows.remote_channels),
        "remote_layers": parts.map(|p| &p.remote_layers),
        "terms_final": parts.map(|p| &p.terms_final),
        "terms_max": parts.map(|p| &p.terms_max),
        "layer_ns": parts.map(|p| &p.layer_ns),
        "remote_layer_ns": parts.map(|p| &p.remote_layer_ns),
        "collective_ns": parts.map(|p| &p.collective_ns),
        "finalize_ns": parts.map(|p| &p.finalize_ns),
        "rows_sent": parts.map(|p| &p.rows_sent),
        "bytes_sent": parts.map(|p| &p.bytes_sent),
        "export_ns": export_ns,
        "exchange_ns": exchange_ns,
    });
    let line = serde_json::json!({
        "model": quench.name,
        "qubits": quench.num_qubits,
        "words": words,
        "observable": quench.observable_label(),
        "steps": cfg.steps,
        "dtype": cfg.dtype,
        "backend": cfg.backend.name(),
        "threads": cfg.threads,
        "device": report.devices.first(),
        "truncation": cfg.truncation.label(),
        "total_ns": summary.total_ns,
        "step_ns": report.step_ns,
        "step_terms": report.step_terms,
        "layers": layers,
        "peak_step_terms": summary.peak_step_terms,
        "traced_ns": report.traced_ns,
        "passes": traced.map(|t| t.passes),
        "max_terms": traced.map(|t| t.max_terms),
        "mean_terms": traced.map(|t| t.mean_terms),
        "peak_terms": traced.map(|t| t.peak_terms),
        "kinds": kinds,
        "rebucket_ns": traced.map(|t| t.rebucket_ns),
        "prepare_ns": traced.map(|t| t.prepare_ns),
        "coset_loop_ns": traced.map(|t| t.coset_loop_ns),
        "finalize_ns": traced.map(|t| t.finalize_ns),
        "upload_ns": report.upload_ns,
        "download_ns": report.download_ns,
        "clocks": report.clocks,
        "partitions": partitions,
        "readout": report.final_readout,
    });
    let mut file =
        std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap_or_else(|e| {
            eprintln!("sampling_bench: cannot open --json-out '{path}': {e}");
            std::process::exit(2);
        });
    writeln!(file, "{line}").expect("writing --json-out");
}
