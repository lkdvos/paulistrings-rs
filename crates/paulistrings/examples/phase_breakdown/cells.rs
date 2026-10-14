//! One measured cell per driver, host and partitioned, and the partition-axis summary.

use std::time::Instant;

use paulistrings::test_support::bucket_len;
use paulistrings::{
    propagate_with, Direction, LayerScratch, PartitionConfig, PartitionPhaseStats,
    PartitionRuntime, PartitionTrace, PartitionedSum, PauliSum, PhaseStats, PropagateOptions,
    ScatterOptions, ScatterRows, TruncationPolicy,
};

use crate::cli::{Config, Initial};
use crate::rows::{choose_generator, choose_partition_rows, RowChoiceStats};
use crate::workloads::{build_base_sum, build_circuit, LayerKind};

pub(crate) struct CellResult {
    pub(crate) layer: &'static str,
    pub(crate) truncation: String,
    pub(crate) threads: usize,
    pub(crate) n: usize,
    pub(crate) reps: usize,
    pub(crate) qubits: usize,
    pub(crate) seed: u64,
    pub(crate) hash_seed: u64,
    pub(crate) bucket_bits: u8,
    pub(crate) target_bucket_len: usize,
    pub(crate) min_buckets: usize,
    pub(crate) wall_ns: u64,
    /// The cell's phase breakdown; for a partitioned cell, [`fold_partition_stats`]'s cell-level view.
    pub(crate) stats: PhaseStats,
    pub(crate) vmrss_kb: u64,
    pub(crate) vmhwm_kb: u64,
    /// Partitions the cell ran on: `1` for an unpartitioned cell.
    pub(crate) partitions: usize,
    /// `--partition-cpus` echoed back (`"auto"`, `"unpinned"`, or the list).
    pub(crate) partition_cpus: String,
    /// `--bind-memory`, as the `pin_memory` field of the sidecar.
    pub(crate) pin_memory: bool,
    /// The `(q0, q1)` the `rotation_*` layers rotated about.
    pub(crate) gen_qubits: (usize, usize),
    /// `--initial` as it applied to this layer (each layer has its own default).
    pub(crate) initial: &'static str,
    /// `--partition-rows` echoed back; written on every row for a single schema.
    pub(crate) partition_rows: &'static str,
    /// What those rows cost this cell's circuit; all-zero on an unpartitioned row.
    pub(crate) row_stats: RowChoiceStats,
    /// Everything only a partitioned cell has, `None` at `P = 1` classic.
    pub(crate) partitioned: Option<PartitionCellStats>,
    /// `(rank, ranks)` for a `--mpi` cell; every field above is then this rank's own.
    pub(crate) mpi: Option<(u32, u32)>,
    /// Bucket occupancy sampled at `--occupancy-at`, `None` unless the flag was passed.
    pub(crate) occupancy: Option<Occupancy>,
    /// The CUDA device a `--device` cell ran on, with the untimed initial upload and final download it paid outside `wall_ns`.
    pub(crate) device: Option<DeviceCellStats>,
}

/// The device-axis numbers of one cell.
#[derive(Clone)]
pub(crate) struct DeviceCellStats {
    /// The ordinals the cell's partitions ran on, one entry per device (the rank's own under `--mpi`).
    pub(crate) devices: Vec<u32>,
    /// The scatter of the cell's input, including the first-use kernel compile.
    pub(crate) upload_ns: u64,
    /// The gather of the timed call's output.
    pub(crate) download_ns: u64,
}

/// A bucket-occupancy snapshot: how term counts spread across buckets at one rep.
pub(crate) struct Occupancy {
    /// The rep index this was sampled after (echoes `--occupancy-at`).
    pub(crate) step: usize,
    pub(crate) num_buckets: usize,
    pub(crate) empty_buckets: usize,
    /// Median/p95/max computed over non-empty buckets only: an empty bucket is idle capacity,
    /// not a small piece of work, and folding it in would understate the occupied buckets' skew.
    pub(crate) median: usize,
    pub(crate) p95: usize,
    pub(crate) max: usize,
}

/// Reads `sum`'s bucket-length histogram and reduces it to [`Occupancy`].
pub(crate) fn occupancy_histogram<const W: usize>(sum: &PauliSum<W>, step: usize) -> Occupancy {
    let num_buckets = sum.num_buckets();
    let mut nonempty: Vec<usize> = (0..num_buckets)
        .map(|b| bucket_len(sum, b))
        .filter(|&len| len > 0)
        .collect();
    nonempty.sort_unstable();
    let empty_buckets = num_buckets - nonempty.len();
    Occupancy {
        step,
        num_buckets,
        empty_buckets,
        median: percentile(&nonempty, 0.5),
        p95: percentile(&nonempty, 0.95),
        max: nonempty.last().copied().unwrap_or(0),
    }
}

/// Nearest-rank percentile of a sorted, non-empty-filtered slice; `0` when it's empty.
pub(crate) fn percentile(sorted: &[usize], p: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (((sorted.len() - 1) as f64) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// The partition-axis numbers of one cell, from the timed call's
/// [`PartitionTrace`] and its per-partition [`PhaseStats`].
pub(crate) struct PartitionCellStats {
    /// Layers with no remote delta, so no export and no transport call.
    pub(crate) local_layers: usize,
    /// Layers with at least one remote delta.
    pub(crate) remote_layers: usize,
    /// Collective calls over the timed run, summed over layers, excluding the exchanges.
    pub(crate) collectives: u64,
    /// Rows moved across partitions, summed over layers and senders.
    pub(crate) rows_exported: u64,
    /// Wire bytes for those rows.
    pub(crate) bytes_exported: u64,
    /// Sum over layers of each partition's input term count, by rank.
    pub(crate) terms_in: Vec<usize>,
    /// `max / mean` of [`Self::terms_in`]: 1.0 is a perfectly even split.
    pub(crate) imbalance: f64,
    /// The same ratio per layer, in application order: shows how fast terms mix between partitions.
    pub(crate) imbalance_by_layer: Vec<f64>,
    /// Total terms in, per layer, over the whole group.
    pub(crate) terms_by_layer: Vec<usize>,
    /// Each partition's `coset_loop_ns`, by rank.
    pub(crate) coset_loop_ns: Vec<u64>,
}

/// Read `VmRSS`/`VmHWM` (kB) from `/proc/self/status`; `0` for either field not found.
pub(crate) fn read_proc_status_kb() -> (u64, u64) {
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

pub(crate) fn parse_kb_field(s: &str) -> u64 {
    s.split_whitespace()
        .next()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}

pub(crate) fn run_cell<const W: usize, P>(
    layer: LayerKind,
    threads: usize,
    cfg: &Config,
    policy: &P,
) -> CellResult
where
    P: TruncationPolicy<W>,
{
    let base = build_base_sum::<W>(layer, cfg);
    // Nothing is remote without partitions, so `rotation_local`/`rotation_remote` are `rotation_zz` here.
    let gen_qubits = (0, 1);
    let circuit = build_circuit::<W>(layer, cfg.qubits, cfg.reps, gen_qubits);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build a rayon thread pool");

    // `rebucket` is grow-only, so this only means anything if the warm-up never grew past it.
    let options = PropagateOptions {
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
    };

    if cfg.occupancy_at.is_some() && layer == LayerKind::Trotter {
        eprintln!(
            "phase_breakdown: warning: --occupancy-at is not supported for `trotter` (its \
             circuit ignores --reps); no occupancy sample will be taken for this cell."
        );
    }
    let sample_by_step = cfg.occupancy_at.is_some() && layer != LayerKind::Trotter;

    let (steady_n, wall_ns, stats, occupancy) = pool.install(|| {
        let mut scratch = LayerScratch::<W>::new();

        if sample_by_step {
            // Sampling occupancy at rep `k` needs the sum as it actually stood after k reps of
            // a genuine `cfg.reps`-rep trajectory from `base`. The warm-up-then-repeat scheme
            // below measures a periodic layer's steady state by applying its circuit twice, which
            // is fine when repeating the same gate is idempotent-ish (a dense random input stays
            // in the same statistical shape) — but a Trotter-step layer's z0 input is a growing
            // light cone with no such periodicity, so "warm up over cfg.reps reps, then apply
            // another cfg.reps reps" runs the dynamics to depth 2 * cfg.reps, silently doubling
            // the step index's meaning. This branch runs the one real cfg.reps-deep trajectory
            // instead, so rep `cfg.occupancy_at` is actually rep `cfg.occupancy_at`.
            let one_rep = build_circuit::<W>(layer, cfg.qubits, 1, gen_qubits);
            let mut sum = base.clone();
            let mut wall_ns = 0u64;
            let mut stats = PhaseStats::default();
            let mut occupancy = None;
            for step in 0..cfg.reps {
                let start = Instant::now();
                sum = propagate_with(
                    &one_rep,
                    sum,
                    policy,
                    Direction::Forward,
                    &mut scratch,
                    options,
                );
                wall_ns += start.elapsed().as_nanos() as u64;
                stats.add(&scratch.take_stats().per_partition[0]);
                if Some(step) == cfg.occupancy_at {
                    occupancy = Some(occupancy_histogram(&sum, step));
                }
            }
            let steady_n = sum.len();
            std::hint::black_box(&sum);
            (steady_n, wall_ns, stats, occupancy)
        } else {
            // Untimed warm-up drives the input to its steady state, so the timed call measures that, not first-layer growth.
            let warmed = propagate_with(
                &circuit,
                base.clone(),
                policy,
                Direction::Forward,
                &mut scratch,
                options,
            );
            let _ = scratch.take_stats(); // discard warm-up counters

            let steady_n = warmed.len();
            let start = Instant::now();
            let output = propagate_with(
                &circuit,
                warmed,
                policy,
                Direction::Forward,
                &mut scratch,
                options,
            );
            let wall_ns = start.elapsed().as_nanos() as u64;
            let stats = scratch.take_stats().per_partition[0];
            std::hint::black_box(&output);

            (steady_n, wall_ns, stats, None)
        }
    });

    let (vmrss_kb, vmhwm_kb) = read_proc_status_kb();

    CellResult {
        layer: layer.name(),
        truncation: cfg.truncation.label(),
        threads,
        n: steady_n,
        reps: cfg.reps,
        qubits: cfg.qubits,
        seed: cfg.seed,
        hash_seed: cfg.hash_seed,
        bucket_bits: cfg.bucket_bits,
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
        wall_ns,
        stats,
        vmrss_kb,
        vmhwm_kb,
        partitions: 1,
        partition_cpus: cfg.partition_cpus.label(),
        pin_memory: cfg.bind_memory,
        gen_qubits,
        initial: cfg
            .initial
            .unwrap_or_else(|| Initial::default_for(layer))
            .label(),
        partition_rows: cfg.partition_rows.label(),
        row_stats: RowChoiceStats::default(),
        // No split, so no partition numbers on this row.
        partitioned: None,
        mpi: None,
        occupancy,
        device: None,
    }
}

/// One partitioned cell: scatter (untimed), warm up, drain, time one `propagate`, and read the
/// trace and the per-partition counters. Mirrors [`run_cell`]'s shape; the two differ only in
/// the engine underneath.
pub(crate) fn run_cell_partitioned<const W: usize, P>(
    layer: LayerKind,
    threads: usize,
    partitions: usize,
    cfg: &Config,
    policy: &P,
) -> CellResult
where
    P: TruncationPolicy<W>,
{
    let base = build_base_sum::<W>(layer, cfg);

    // `--threads` is the TOTAL, so P pools of threads/P compare against one pool of threads.
    let config = PartitionConfig {
        placement: cfg.partition_cpus.placement(partitions),
        bind_memory: cfg.bind_memory,
        threads_per_partition: Some(threads / partitions),
    };
    let runtime = PartitionRuntime::new(&config).unwrap_or_else(|err| {
        eprintln!("phase_breakdown: cannot resolve the partition placement: {err}");
        std::process::exit(2);
    });
    // `Auto` can silently resolve to fewer partitions than the host has NUMA nodes; refuse rather than mislabel every row.
    if runtime.num_partitions() != partitions {
        eprintln!(
            "phase_breakdown: --partitions {partitions} resolved to {} partitions: an `auto` \
             placement takes one partition per NUMA node in the affinity mask (rounded down to a \
             power of two). Pass --partition-cpus with {partitions} cpulists, or \
             --partition-cpus unpinned, to get {partitions} partitions on this host.",
            runtime.num_partitions(),
        );
        std::process::exit(2);
    }

    // Built here so the generator scan below sees exactly the split the run will use.
    let (base, rows, row_stats) = choose_partition_rows::<W>(layer, cfg, base, partitions);
    let gen_qubits = choose_generator::<W>(layer, cfg.qubits, &base, &rows);
    if layer.picks_generator() {
        eprintln!(
            "phase_breakdown: note: {} at P={partitions} acts on {}({}, {}) — the smallest \
             pair whose deltas are {} under these partition rows.",
            layer.name(),
            if layer == LayerKind::Su4Local {
                "SU4"
            } else {
                "ZZ"
            },
            gen_qubits.0,
            gen_qubits.1,
            if layer == LayerKind::RotationRemote {
                "remote"
            } else {
                "local"
            },
        );
    }
    let circuit = build_circuit::<W>(layer, cfg.qubits, cfg.reps, gen_qubits);

    let options = PropagateOptions {
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
    };

    let split_hash_seed = base.hash().seed();
    let mut split = PartitionedSum::scatter_with(
        base,
        ScatterOptions {
            runtime,
            rows: ScatterRows::Explicit(rows),
        },
    );
    split.enable_trace();

    // Untimed warm-up, then its counters discarded — same contract as the unpartitioned cell.
    split.propagate_with(&circuit, policy, Direction::Forward, options);
    let _ = split.take_trace();
    let _ = split.take_stats();

    let steady_n = split.len();
    let started = Instant::now();
    split.propagate_with(&circuit, policy, Direction::Forward, options);
    let wall_ns = started.elapsed().as_nanos() as u64;
    let trace = split.take_trace().expect("tracing was enabled");
    let per_partition = split.take_stats();
    std::hint::black_box(&split);

    let (vmrss_kb, vmhwm_kb) = read_proc_status_kb();
    let stats = fold_partition_stats(&per_partition);
    let summary = summarize_partitions(partitions, &trace, &per_partition, &stats);

    CellResult {
        layer: layer.name(),
        truncation: cfg.truncation.label(),
        threads,
        n: steady_n,
        reps: cfg.reps,
        qubits: cfg.qubits,
        seed: cfg.seed,
        // The seed actually used, which `choose_partition_rows` may have re-drawn.
        hash_seed: split_hash_seed,
        bucket_bits: cfg.bucket_bits,
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
        wall_ns,
        stats,
        vmrss_kb,
        vmhwm_kb,
        partitions,
        partition_cpus: cfg.partition_cpus.label(),
        pin_memory: cfg.bind_memory,
        gen_qubits,
        initial: cfg
            .initial
            .unwrap_or_else(|| Initial::default_for(layer))
            .label(),
        partition_rows: cfg.partition_rows.label(),
        row_stats,
        partitioned: Some(summary),
        mpi: None,
        // Occupancy sampling is unimplemented for the partitioned path; scoped to P = 1 for now.
        occupancy: None,
        device: None,
    }
}

/// One cell on the distributed driver: this process is one rank's partition, and the numbers it
/// reports are its own. The MPI counterpart of [`run_cell_partitioned`] — same shape, but the
/// runtime holds exactly one partition, placed by the launcher's affinity mask, over the world
/// communicator. Collective: every rank runs the identical cell matrix in the identical order.
/// The headline is `vmhwm_kb`, peak resident set per rank including the export/receive transients.
#[cfg(feature = "mpi")]
pub(crate) fn run_cell_mpi<const W: usize, P>(
    layer: LayerKind,
    threads: usize,
    cfg: &Config,
    policy: &P,
) -> CellResult
where
    P: TruncationPolicy<W>,
{
    use paulistrings::mpi::{rsmpi, MpiTransport};
    use paulistrings::{Collectives, DistributedSum};
    use rsmpi::topology::{Communicator, SimpleCommunicator};

    let world = SimpleCommunicator::world();
    let rank = world.rank() as u32;
    let ranks = world.size() as u32;
    if !ranks.is_power_of_two() {
        if rank == 0 {
            eprintln!(
                "phase_breakdown: --mpi needs a power-of-two rank count (a partition is named by \
                 log2(P) GF(2) rows), got {ranks}",
            );
        }
        std::process::exit(2);
    }

    let base = build_base_sum::<W>(layer, cfg);

    // D = 1: one partition over whatever CPUs the launcher left us.
    let config = PartitionConfig {
        placement: cfg.partition_cpus.placement(1),
        bind_memory: cfg.bind_memory,
        threads_per_partition: Some(threads),
    };
    let runtime = PartitionRuntime::new(&config).unwrap_or_else(|err| {
        eprintln!("phase_breakdown: rank {rank} cannot resolve its placement: {err}");
        std::process::exit(2);
    });

    // Same inputs on every rank, so every rank derives the same rows with nothing agreed at run time.
    let (base, rows, row_stats) = choose_partition_rows::<W>(layer, cfg, base, ranks as usize);
    let gen_qubits = choose_generator::<W>(layer, cfg.qubits, &base, &rows);
    if layer.picks_generator() && rank == 0 {
        eprintln!(
            "phase_breakdown: note: {} on {ranks} ranks acts on ({}, {}).",
            layer.name(),
            gen_qubits.0,
            gen_qubits.1,
        );
    }
    let circuit = build_circuit::<W>(layer, cfg.qubits, cfg.reps, gen_qubits);

    let options = PropagateOptions {
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
    };

    let transport = MpiTransport::from_communicator(&world);
    let split_hash_seed = base.hash().seed();
    let scatter_options = ScatterOptions {
        runtime,
        rows: ScatterRows::Explicit(rows),
    };
    let mut split = DistributedSum::scatter_with(base, transport, scatter_options);
    split.enable_trace();

    // Untimed warm-up, counters discarded; a barrier so the timed call starts together.
    split.propagate_with(&circuit, policy, Direction::Forward, options);
    let _ = split.take_trace();
    let _ = split.take_stats();
    split.transport().barrier();

    let steady_n = split.local().len();
    let started = Instant::now();
    split.propagate_with(&circuit, policy, Direction::Forward, options);
    let wall_ns = started.elapsed().as_nanos() as u64;
    let trace = split.take_trace().expect("tracing was enabled");
    let per_partition = split.take_stats();
    std::hint::black_box(&split);

    let (vmrss_kb, vmhwm_kb) = read_proc_status_kb();
    let stats = fold_partition_stats(&per_partition);
    // One "partition" here: this rank.
    let summary = summarize_partitions(1, &trace, &per_partition, &stats);

    CellResult {
        layer: layer.name(),
        truncation: cfg.truncation.label(),
        threads,
        n: steady_n,
        reps: cfg.reps,
        qubits: cfg.qubits,
        seed: cfg.seed,
        // The seed actually used, which `choose_partition_rows` may have re-drawn.
        hash_seed: split_hash_seed,
        bucket_bits: cfg.bucket_bits,
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
        wall_ns,
        stats,
        vmrss_kb,
        vmhwm_kb,
        partitions: 1,
        partition_cpus: cfg.partition_cpus.label(),
        pin_memory: cfg.bind_memory,
        gen_qubits,
        initial: cfg
            .initial
            .unwrap_or_else(|| Initial::default_for(layer))
            .label(),
        partition_rows: cfg.partition_rows.label(),
        row_stats,
        partitioned: Some(summary),
        mpi: Some((rank, ranks)),
        occupancy: None,
        device: None,
    }
}

/// The cell-level [`PhaseStats`] of a partitioned run: wall-clock fields are the maximum over
/// partitions (the critical path), busy-time and counter fields are sums. `layers` is the
/// driver's layer count, not a sum — every partition drove the same layers.
pub(crate) fn fold_partition_stats(stats: &PartitionPhaseStats) -> PhaseStats {
    let mut out = PhaseStats::default();
    for s in &stats.per_partition {
        // Wall-clock phases: max over partitions.
        out.rebucket_ns = out.rebucket_ns.max(s.rebucket_ns);
        out.prepare_ns = out.prepare_ns.max(s.prepare_ns);
        out.rescale_ns = out.rescale_ns.max(s.rescale_ns);
        out.span_plan_ns = out.span_plan_ns.max(s.span_plan_ns);
        out.permute_ns = out.permute_ns.max(s.permute_ns);
        out.coset_loop_ns = out.coset_loop_ns.max(s.coset_loop_ns);
        out.unpermute_ns = out.unpermute_ns.max(s.unpermute_ns);
        out.recount_ns = out.recount_ns.max(s.recount_ns);
        out.finalize_ns = out.finalize_ns.max(s.finalize_ns);
        out.collective_ns = out.collective_ns.max(s.collective_ns);
        out.export_ns = out.export_ns.max(s.export_ns);
        out.exchange_ns = out.exchange_ns.max(s.exchange_ns);
        // Device sub-phases of the two wall phases above, aggregated like them.
        out.truncate_ns = out.truncate_ns.max(s.truncate_ns);
        out.export_kernel_ns = out.export_kernel_ns.max(s.export_kernel_ns);
        // Worker busy time and counters: sums over the group.
        out.swap_ns += s.swap_ns;
        out.size_ns += s.size_ns;
        out.gather_ns += s.gather_ns;
        out.sort_ns += s.sort_ns;
        out.merge_ns += s.merge_ns;
        out.clear_ns += s.clear_ns;
        out.cosets += s.cosets;
        out.runs += s.runs;
        out.rows_gathered += s.rows_gathered;
        out.rows_sorted += s.rows_sorted;
        out.rows_id += s.rows_id;
        out.terms_in += s.terms_in;
        out.terms_out += s.terms_out;
        out.rows_exported += s.rows_exported;
        out.recv_rows += s.recv_rows;
        out.append_ns += s.append_ns;
        out.chunk_wait_ns += s.chunk_wait_ns;
        out.compact_ns += s.compact_ns;
        out.h2d_ns += s.h2d_ns;
        out.d2h_ns += s.d2h_ns;
    }
    out.layers = stats.layers;
    out
}

/// The partition-axis summary of one timed call. `rows_exported` and `bytes_exported` are the
/// same send traffic counted in rows and in wire bytes, respectively.
pub(crate) fn summarize_partitions(
    partitions: usize,
    trace: &PartitionTrace,
    per_partition: &PartitionPhaseStats,
    folded: &PhaseStats,
) -> PartitionCellStats {
    let mut terms_in = vec![0usize; partitions];
    for layer in &trace.layers {
        for (rank, &t) in layer.terms_in.iter().enumerate() {
            terms_in[rank] += t;
        }
    }
    let total: usize = terms_in.iter().sum();
    let imbalance = if total == 0 {
        1.0
    } else {
        let mean = total as f64 / partitions as f64;
        terms_in.iter().copied().max().unwrap_or(0) as f64 / mean
    };

    PartitionCellStats {
        local_layers: trace.local_layers(),
        remote_layers: trace.remote_layers(),
        collectives: trace.total_collectives(),
        rows_exported: folded.rows_exported,
        bytes_exported: trace
            .layers
            .iter()
            .flat_map(|layer| layer.bytes_sent.iter())
            .flat_map(|row| row.iter())
            .sum(),
        terms_in,
        imbalance,
        imbalance_by_layer: trace.imbalance(),
        terms_by_layer: trace
            .layers
            .iter()
            .map(|layer| layer.terms_in.iter().sum())
            .collect(),
        coset_loop_ns: per_partition
            .per_partition
            .iter()
            .map(|s| s.coset_loop_ns)
            .collect(),
    }
}
