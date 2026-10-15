//! What a cell prints: the `cell` line, the table, and the JSON row.

use paulistrings::test_support::{BITS_AGREE_EVERY, TIMER_READ_OVERHEAD_NS};
use paulistrings::PhaseStats;
use serde_json::{json, Value};

use crate::cells::{CellResult, DeviceCellStats};

/// Always printed first for every cell, in every format: `n=` and `layers=` are a contract other
/// scripts grep for (machine contract (b) in `benchmarks/PROFILING.md`).
pub(crate) fn print_cell_line(cell: &CellResult) {
    // Empty for every non-`--mpi` cell, so those lines keep their format.
    let mpi = match cell.mpi {
        Some((rank, ranks)) => format!(" rank={rank}/{ranks} vmhwm_kb={}", cell.vmhwm_kb),
        None => String::new(),
    };
    let device = match &cell.device {
        Some(d) => format!(" device={}", device_csv(d)),
        None => String::new(),
    };
    println!(
        "cell layer={} threads={} n={} layers={} wall_ms={:.3} trunc={} partitions={}{mpi}{device}",
        cell.layer,
        cell.threads,
        cell.n,
        cell.stats.layers,
        cell.wall_ns as f64 / 1e6,
        cell.truncation,
        cell.partitions,
    );
}

pub(crate) type PhaseGetter = fn(&PhaseStats) -> u64;

pub(crate) const WALL_PHASES: [(&str, PhaseGetter); 9] = [
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

pub(crate) const BUSY_PHASES: [(&str, PhaseGetter); 7] = [
    ("gather", |s| s.gather_ns),
    ("sort", |s| s.sort_ns),
    ("merge", |s| s.merge_ns),
    ("swap", |s| s.swap_ns),
    ("size", |s| s.size_ns),
    ("clear", |s| s.clear_ns),
    ("compact", |s| s.compact_ns),
];

pub(crate) fn print_table(cell: &CellResult) {
    let s = &cell.stats;
    let wall_total = s.wall_total_ns();

    println!("  {:<12} {:>10} {:>12} {:>8}", "phase", "ms", "ms/layer", "% wall");
    for (name, get) in WALL_PHASES {
        let ns = get(s);
        if ns == 0 {
            continue;
        }
        let ms = ns as f64 / 1e6;
        let ms_per_layer = if s.layers > 0 { ms / s.layers as f64 } else { 0.0 };
        let pct = if wall_total > 0 { ns as f64 / wall_total as f64 * 100.0 } else { 0.0 };
        println!("  {:<12} {:>10.3} {:>12.4} {:>7.1}%", name, ms, ms_per_layer, pct);
    }

    let busy_total = s.busy_total_ns();
    println!("  busy (summed over every coset task / worker):");
    for (name, get) in BUSY_PHASES {
        let ns = get(s);
        let ms = ns as f64 / 1e6;
        let pct = if busy_total > 0 { ns as f64 / busy_total as f64 * 100.0 } else { 0.0 };
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
    let strings_per_s = if wall_s > 0.0 { s.terms_in as f64 / wall_s } else { 0.0 };
    let overhead_ns = s.timer_reads() * TIMER_READ_OVERHEAD_NS;
    let overhead_pct =
        if cell.wall_ns > 0 { overhead_ns as f64 / cell.wall_ns as f64 * 100.0 } else { 0.0 };
    println!("  strings/s          = {strings_per_s:.3e}");
    println!(
        "  timer overhead est = {:.3} us ({} reads x {} ns) = {:.2}% of wall",
        overhead_ns as f64 / 1e3,
        s.timer_reads(),
        TIMER_READ_OVERHEAD_NS,
        overhead_pct
    );
    println!("  VmRSS = {} kB   VmHWM = {} kB", cell.vmrss_kb, cell.vmhwm_kb);
    print_partition_block(cell);
    if let Some(d) = &cell.device {
        println!(
            "  device             = {}   upload = {:.3} ms   download = {:.3} ms   [outside wall]",
            device_csv(d),
            d.upload_ns as f64 / 1e6,
            d.download_ns as f64 / 1e6,
        );
        println!(
            "    in-layer copies: h2d = {:.3} ms   d2h = {:.3} ms   [inside coset_loop / rescale]",
            s.h2d_ns as f64 / 1e6,
            s.d2h_ns as f64 / 1e6,
        );
        println!(
            "    kernels: truncate = {:.3} ms [inside finalize]   export = {:.3} ms [inside export]",
            s.truncate_ns as f64 / 1e6,
            s.export_kernel_ns as f64 / 1e6,
        );
    }
    println!();
}

/// The partition block of the table format, printed only for a cell that ran through the partitioned engine.
pub(crate) fn print_partition_block(cell: &CellResult) {
    let Some(p) = cell.partitioned.as_ref() else {
        return;
    };
    let s = &cell.stats;
    println!(
        "  partitions         = {} on {} (pin_memory = {}, gen_qubits = {},{})",
        cell.partitions,
        cell.partition_cpus,
        u8::from(cell.pin_memory),
        cell.gen_qubits.0,
        cell.gen_qubits.1,
    );
    println!(
        "    layers: {} local / {} remote   rows exported = {} ({:.3} MiB on the wire)",
        p.local_layers,
        p.remote_layers,
        p.rows_exported,
        p.bytes_exported as f64 / (1024.0 * 1024.0),
    );
    println!(
        "    collectives = {} over {} layers (bucket-count schedule every {} layers, \
         plus every remote layer and any the policy runs)",
        p.collectives,
        p.local_layers + p.remote_layers,
        BITS_AGREE_EVERY,
    );
    println!(
        "    export = {:.3} ms   exchange = {:.3} ms   barrier (bucket-count all-reduce) = \
         {:.3} ms   [max over partitions]",
        s.export_ns as f64 / 1e6,
        s.exchange_ns as f64 / 1e6,
        s.collective_ns as f64 / 1e6,
    );
    println!(
        "      received rows appended into the rest streams = {:.3} ms (of which {:.3} ms waiting \
         for a chunk to land) [worker busy time, part of gather]",
        s.append_ns as f64 / 1e6,
        s.chunk_wait_ns as f64 / 1e6,
    );
    println!(
        "    terms in per partition = {:?}   imbalance (max/mean) = {:.3}",
        p.terms_in, p.imbalance,
    );
    let coset_ms: Vec<String> =
        p.coset_loop_ns.iter().map(|ns| format!("{:.3}", *ns as f64 / 1e6)).collect();
    println!("    coset_loop ms per partition = [{}]", coset_ms.join(", "));
    // The human format prints the per-layer series' shape; the full arrays are in the JSON row.
    if let (Some(first), Some(last)) = (p.imbalance_by_layer.first(), p.imbalance_by_layer.last()) {
        let worst = p.imbalance_by_layer.iter().copied().fold(f64::MIN, f64::max);
        println!(
            "    imbalance per layer over {} layers: first {:.3}, worst {:.3}, last {:.3}   \
             (rows = {})",
            p.imbalance_by_layer.len(),
            first,
            worst,
            last,
            cell.partition_rows,
        );
    }
}

pub(crate) fn print_json(cell: &CellResult) {
    println!("{}", json_line(cell));
}

/// `x` at six decimals, the precision the sidecar's float fields carry.
fn six_decimals(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// One cell as a single JSON line, shared by `--format json` (stdout) and `--json-out` (sidecar).
/// The partition fields are written on every row so a campaign has one schema; `barrier_ns` is
/// the sidecar's name for the engine's `collective_ns`. See machine contract (a) in
/// `benchmarks/PROFILING.md`.
pub(crate) fn json_line(cell: &CellResult) -> String {
    let s = &cell.stats;
    let p = cell.partitioned.as_ref();
    let mut row = json!({
        "layer": cell.layer,
        "truncation": cell.truncation,
        "threads": cell.threads,
        "n": cell.n,
        "reps": cell.reps,
        "qubits": cell.qubits,
        "seed": cell.seed,
        "hash_seed": cell.hash_seed,
        "wall_ns": cell.wall_ns,
        "rebucket_ns": s.rebucket_ns,
        "prepare_ns": s.prepare_ns,
        "rescale_ns": s.rescale_ns,
        "span_plan_ns": s.span_plan_ns,
        "permute_ns": s.permute_ns,
        "coset_loop_ns": s.coset_loop_ns,
        "unpermute_ns": s.unpermute_ns,
        "recount_ns": s.recount_ns,
        "finalize_ns": s.finalize_ns,
        "swap_ns": s.swap_ns,
        "size_ns": s.size_ns,
        "gather_ns": s.gather_ns,
        "sort_ns": s.sort_ns,
        "merge_ns": s.merge_ns,
        "clear_ns": s.clear_ns,
        "layers": s.layers,
        "cosets": s.cosets,
        "runs": s.runs,
        "rows_gathered": s.rows_gathered,
        "rows_sorted": s.rows_sorted,
        "rows_id": s.rows_id,
        "terms_in": s.terms_in,
        "terms_out": s.terms_out,
        "vmrss_kb": cell.vmrss_kb,
        "vmhwm_kb": cell.vmhwm_kb,
        "compact_ns": s.compact_ns,
        "h2d_ns": s.h2d_ns,
        "d2h_ns": s.d2h_ns,
        "truncate_ns": s.truncate_ns,
        "export_kernel_ns": s.export_kernel_ns,
    });
    // Two literals: one this long overflows `json!`'s macro recursion limit.
    let partition_fields = json!({
        "partitions": cell.partitions,
        "partition_cpus": cell.partition_cpus,
        "pin_memory": u8::from(cell.pin_memory),
        "gen_qubits": [cell.gen_qubits.0, cell.gen_qubits.1],
        "local_layers": p.map_or(0, |p| p.local_layers),
        "remote_layers": p.map_or(0, |p| p.remote_layers),
        "collectives": p.map_or(0, |p| p.collectives),
        "rows_exported": p.map_or(0, |p| p.rows_exported),
        "bytes_exported": p.map_or(0, |p| p.bytes_exported),
        "partition_terms_in": p.map_or(&[][..], |p| &p.terms_in),
        "partition_imbalance": six_decimals(p.map_or(1.0, |p| p.imbalance)),
        "export_ns": s.export_ns,
        "exchange_ns": s.exchange_ns,
        "barrier_ns": s.collective_ns,
        "partition_coset_loop_ns": p.map_or(&[][..], |p| &p.coset_loop_ns),
        "append_ns": s.append_ns,
        "chunk_wait_ns": s.chunk_wait_ns,
        "initial": cell.initial,
        "partition_rows": cell.partition_rows,
        "partition_imbalance_by_layer": p.map_or(Vec::new(), |p| {
            p.imbalance_by_layer.iter().copied().map(six_decimals).collect()
        }),
        "terms_by_layer": p.map_or(&[][..], |p| &p.terms_by_layer),
        "rows_remote_gens": cell.row_stats.remote_gens,
        "rows_remote_weight": (cell.row_stats.remote_weight * 10.0).round() / 10.0,
    });
    if let (Value::Object(row), Value::Object(fields)) = (&mut row, partition_fields) {
        row.extend(fields);
    }
    // Absent on a host row and on a non-`--mpi` row, rather than null.
    if let Some((rank, ranks)) = cell.mpi {
        row["rank"] = json!(rank);
        row["ranks"] = json!(ranks);
    }
    if let Some(d) = &cell.device {
        row["device"] = json!(d.devices);
        row["upload_ns"] = json!(d.upload_ns);
        row["download_ns"] = json!(d.download_ns);
    }
    row.to_string()
}

/// A device row's ordinals, comma-joined.
pub(crate) fn device_csv(d: &DeviceCellStats) -> String {
    d.devices.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}
