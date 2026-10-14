//! Device cells: one process over CUDA devices, and one device per MPI rank.

use std::time::Instant;

use paulistrings::{
    BuiltinTruncation, Direction, PartitionConfig, PartitionPhaseStats, PartitionRuntime,
    PartitionTrace, Placement, PropagateOptions, ScatterRows,
};

use crate::cells::{
    fold_partition_stats, read_proc_status_kb, summarize_partitions, CellResult, DeviceCellStats,
};
use crate::cli::{check_device_partitions, Config, DeviceSpec, Initial, TruncSpec};
use crate::rows::{choose_generator, choose_partition_rows, RowChoiceStats};
use crate::workloads::{build_base_sum, build_circuit, LayerKind};

/// `--truncation` as a [`BuiltinTruncation`], the form the device drivers take.
pub(crate) fn builtin_truncation(spec: TruncSpec) -> BuiltinTruncation {
    match spec {
        TruncSpec::Keep => BuiltinTruncation::Keep,
        TruncSpec::Coefficient(t) => BuiltinTruncation::Coefficient(t),
        TruncSpec::TopN(n) => BuiltinTruncation::TopN(n),
        TruncSpec::ApproxTopN(n) => BuiltinTruncation::ApproxTopN(n),
    }
}

/// One device cell, `--gpu-partitions` partitions on each of `devices` (one device at one partition is `GpuPartitionedSum::from_host`'s shape): [`run_cell_partitioned`] with `GpuPartitionedSum`.
#[cfg(feature = "cuda")]
pub(crate) fn run_cell_gpu<const W: usize>(
    layer: LayerKind,
    threads: usize,
    devices: &[u32],
    cfg: &Config,
    policy: &BuiltinTruncation,
) -> CellResult {
    use paulistrings::gpu::{GpuLayerOptions, GpuPartitionedSum, GpuScatterOptions};

    let partitions = devices.len() * cfg.gpu_partitions;
    let config = PartitionConfig {
        placement: Placement::Devices {
            devices: devices.to_vec(),
            per_device: cfg.gpu_partitions,
        },
        bind_memory: false,
        threads_per_partition: None,
    };
    let runtime = PartitionRuntime::new(&config).unwrap_or_else(|err| {
        eprintln!("phase_breakdown: cannot resolve the device placement: {err}");
        std::process::exit(2);
    });
    let fail = |what: &str, e: paulistrings::gpu::GpuError| -> ! {
        eprintln!(
            "phase_breakdown: {} on devices {devices:?} x{}: {what}: {e}",
            layer.name(),
            cfg.gpu_partitions,
        );
        std::process::exit(2);
    };
    let base = build_base_sum::<W>(layer, cfg);
    let (base, rows, row_stats) = choose_partition_rows::<W>(layer, cfg, base, partitions);
    let gen_qubits = choose_generator::<W>(layer, cfg.qubits, &base, &rows);
    let circuit = build_circuit::<W>(layer, cfg.qubits, cfg.reps, gen_qubits);
    let hash_seed = base.hash().seed();

    let started = Instant::now();
    let mut split = GpuPartitionedSum::scatter_to_devices_with(
        &base,
        GpuScatterOptions {
            devices: runtime,
            rows: ScatterRows::Explicit(rows),
            layer: GpuLayerOptions::default(),
        },
    )
    .unwrap_or_else(|e| fail("scatter", e));
    drop(base);
    let upload_ns = started.elapsed().as_nanos() as u64;
    split.enable_trace();
    split
        .propagate_with(&circuit, policy, Direction::Forward, device_options(cfg))
        .unwrap_or_else(|e| fail("warm-up", e));
    let _ = (split.take_trace(), split.take_stats());
    let n = split.len();
    let started = Instant::now();
    split
        .propagate_with(&circuit, policy, Direction::Forward, device_options(cfg))
        .unwrap_or_else(|e| fail("timed call", e));
    let wall_ns = started.elapsed().as_nanos() as u64;
    let trace = split.take_trace().expect("tracing was enabled");
    let per_partition = split.take_stats();
    let started = Instant::now();
    std::hint::black_box(split.gather().unwrap_or_else(|e| fail("gather", e)));
    let download_ns = started.elapsed().as_nanos() as u64;
    let run = DeviceRun {
        n,
        hash_seed,
        wall_ns,
        partitions,
        gen_qubits,
        row_stats,
        mpi: None,
        device: DeviceCellStats {
            devices: devices.to_vec(),
            upload_ns,
            download_ns,
        },
    };
    device_cell(layer, threads, cfg, run, &trace, &per_partition)
}

/// What a device cell measured, beside its trace and phase counters.
#[cfg(feature = "cuda")]
pub(crate) struct DeviceRun {
    pub(crate) n: usize,
    pub(crate) hash_seed: u64,
    pub(crate) wall_ns: u64,
    pub(crate) partitions: usize,
    pub(crate) gen_qubits: (usize, usize),
    pub(crate) row_stats: RowChoiceStats,
    pub(crate) mpi: Option<(u32, u32)>,
    pub(crate) device: DeviceCellStats,
}

/// The bucket schedule a device cell propagates under.
#[cfg(feature = "cuda")]
pub(crate) fn device_options(cfg: &Config) -> PropagateOptions {
    PropagateOptions {
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
    }
}

/// A device cell's row.
#[cfg(feature = "cuda")]
pub(crate) fn device_cell(
    layer: LayerKind,
    threads: usize,
    cfg: &Config,
    run: DeviceRun,
    trace: &PartitionTrace,
    per_partition: &PartitionPhaseStats,
) -> CellResult {
    let (vmrss_kb, vmhwm_kb) = read_proc_status_kb();
    let stats = fold_partition_stats(per_partition);
    let summary = summarize_partitions(run.partitions, trace, per_partition, &stats);
    CellResult {
        layer: layer.name(),
        truncation: cfg.truncation.label(),
        threads,
        n: run.n,
        reps: cfg.reps,
        qubits: cfg.qubits,
        seed: cfg.seed,
        hash_seed: run.hash_seed,
        bucket_bits: cfg.bucket_bits,
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
        wall_ns: run.wall_ns,
        stats,
        vmrss_kb,
        vmhwm_kb,
        partitions: run.partitions,
        partition_cpus: "gpu".to_string(),
        pin_memory: false,
        gen_qubits: run.gen_qubits,
        initial: cfg
            .initial
            .unwrap_or_else(|| Initial::default_for(layer))
            .label(),
        partition_rows: cfg.partition_rows.label(),
        row_stats: run.row_stats,
        partitioned: Some(summary),
        mpi: run.mpi,
        occupancy: None,
        device: Some(run.device),
    }
}

/// The ordinals a single-process `--device` run places partitions on: the list, or every visible device for `auto`, which `parse_args` could not check.
#[cfg(feature = "cuda")]
pub(crate) fn resolve_devices(spec: &DeviceSpec, cfg: &Config) -> Vec<u32> {
    let DeviceSpec::Ordinals(devices) = spec else {
        let devices: Vec<u32> = (0..paulistrings::gpu::device_count() as u32).collect();
        if let Err(msg) = check_device_partitions(devices.len(), cfg.gpu_partitions, &cfg.layers) {
            eprintln!("phase_breakdown: --device auto: {msg}");
            std::process::exit(2);
        }
        return devices;
    };
    devices.clone()
}

/// One `--mpi --device` cell: [`run_cell_mpi`] with this rank's partition on one CUDA device (`MpiGpuSum`).
#[cfg(all(feature = "cuda", feature = "mpi"))]
pub(crate) fn run_cell_mpi_gpu<const W: usize>(
    layer: LayerKind,
    threads: usize,
    spec: &DeviceSpec,
    cfg: &Config,
    policy: &BuiltinTruncation,
) -> CellResult {
    use paulistrings::gpu::{local_device_for_comm, GpuScatterOptions, MpiGpuSum};
    use paulistrings::mpi::{rsmpi, MpiTransport};
    use paulistrings::Collectives;
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
    let fail = |what: &str, e: paulistrings::gpu::GpuError| -> ! {
        eprintln!(
            "phase_breakdown: rank {rank}: {} on its device: {what}: {e}",
            layer.name()
        );
        std::process::exit(2);
    };
    let device = match spec {
        DeviceSpec::Ordinals(v) => v[0],
        DeviceSpec::Auto => {
            local_device_for_comm(&world).unwrap_or_else(|e| fail("device pick", e))
        }
    };

    let base = build_base_sum::<W>(layer, cfg);
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
    let hash_seed = base.hash().seed();

    let transport = MpiTransport::from_communicator(&world);
    let started = Instant::now();
    let options = GpuScatterOptions {
        rows: ScatterRows::Explicit(rows),
        ..GpuScatterOptions::new(&[device]).unwrap_or_else(|e| fail("placement", e))
    };
    let mut split = MpiGpuSum::scatter_to_device_with(&base, transport, options)
        .unwrap_or_else(|e| fail("scatter", e));
    drop(base);
    let upload_ns = started.elapsed().as_nanos() as u64;
    split.enable_trace();
    split
        .propagate_with(&circuit, policy, Direction::Forward, device_options(cfg))
        .unwrap_or_else(|e| fail("warm-up", e));
    let _ = (split.take_trace(), split.take_stats());
    split.transport().barrier();
    let started = Instant::now();
    split
        .propagate_with(&circuit, policy, Direction::Forward, device_options(cfg))
        .unwrap_or_else(|e| fail("timed call", e));
    let wall_ns = started.elapsed().as_nanos() as u64;
    let trace = split.take_trace().expect("tracing was enabled");
    // The share stays on the device; the trace's first row holds the timed call's input count.
    let n = trace.layers.first().map_or(0, |layer| layer.terms_in[0]);
    let per_partition = split.take_stats();
    let started = Instant::now();
    std::hint::black_box(split.gather().unwrap_or_else(|e| fail("gather", e)));
    let download_ns = started.elapsed().as_nanos() as u64;
    let run = DeviceRun {
        n,
        hash_seed,
        wall_ns,
        partitions: 1,
        gen_qubits,
        row_stats,
        mpi: Some((rank, ranks)),
        device: DeviceCellStats {
            devices: vec![device],
            upload_ns,
            download_ns,
        },
    };
    device_cell(layer, threads, cfg, run, &trace, &per_partition)
}
