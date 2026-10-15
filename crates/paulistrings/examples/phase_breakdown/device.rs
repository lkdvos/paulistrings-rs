//! Device cells: one process over CUDA devices, and one device per MPI rank.

use std::time::Instant;

use paulistrings::gpu::GpuError;
use paulistrings::{
    BuiltinTruncation, Direction, PartitionConfig, PartitionRuntime, Placement, PropagateOptions,
    ScatterRows,
};

use crate::cells::{partition_stats, warm_up_then_time, CellResult, DeviceCellStats};
use crate::cli::{check_device_partitions, Config, DeviceSpec, TruncSpec};
use crate::rows::{choose_generator, choose_partition_rows};
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

/// One device cell, `--gpu-partitions` partitions on each of `devices`: [`run_cell_partitioned`](crate::cells::run_cell_partitioned) with `GpuPartitionedSum`.
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
    let fail = |what: &str, e: GpuError| -> ! {
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
    let (n, wall_ns) = warm_up_then_time(
        &mut split,
        |split| {
            split
                .propagate_with(
                    &circuit,
                    policy,
                    Direction::Forward,
                    PropagateOptions::default(),
                )
                .unwrap_or_else(|e| fail("propagate", e))
        },
        |split| {
            let _ = (split.take_trace(), split.take_stats());
            split.len()
        },
    );
    let trace = split.take_trace().expect("tracing was enabled");
    let per_partition = split.take_stats();
    let started = Instant::now();
    std::hint::black_box(split.gather().unwrap_or_else(|e| fail("gather", e)));
    let download_ns = started.elapsed().as_nanos() as u64;

    let (stats, summary) = partition_stats(partitions, &trace, &per_partition);
    CellResult {
        n,
        hash_seed,
        wall_ns,
        stats,
        partitions,
        partition_cpus: "gpu".to_string(),
        pin_memory: false,
        gen_qubits,
        row_stats,
        partitioned: Some(summary),
        device: Some(DeviceCellStats {
            devices: devices.to_vec(),
            upload_ns,
            download_ns,
        }),
        ..CellResult::new(layer, threads, cfg)
    }
}

/// The ordinals a single-process `--device` run places partitions on: the list, or every visible device for `auto`, which `parse_args` could not check.
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

/// One `--mpi --device` cell: [`run_cell_mpi`](crate::cells::run_cell_mpi) with this rank's partition on one CUDA device (`MpiGpuSum`).
#[cfg(feature = "mpi")]
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
    use rsmpi::topology::SimpleCommunicator;

    use crate::cells::{note_generator, world_rank};

    let (rank, ranks) = world_rank();
    let world = SimpleCommunicator::world();
    let fail = |what: &str, e: GpuError| -> ! {
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
    if rank == 0 {
        note_generator(layer, gen_qubits, &format!("on {ranks} ranks"));
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
    let ((), wall_ns) = warm_up_then_time(
        &mut split,
        |split| {
            split
                .propagate_with(
                    &circuit,
                    policy,
                    Direction::Forward,
                    PropagateOptions::default(),
                )
                .unwrap_or_else(|e| fail("propagate", e))
        },
        |split| {
            let _ = (split.take_trace(), split.take_stats());
            split.transport().barrier();
        },
    );
    let trace = split.take_trace().expect("tracing was enabled");
    // The share stays on the device; the trace's first row holds the timed call's input count.
    let n = trace.layers.first().map_or(0, |layer| layer.terms_in[0]);
    let per_partition = split.take_stats();
    let started = Instant::now();
    std::hint::black_box(split.gather().unwrap_or_else(|e| fail("gather", e)));
    let download_ns = started.elapsed().as_nanos() as u64;

    let (stats, summary) = partition_stats(1, &trace, &per_partition);
    CellResult {
        n,
        hash_seed,
        wall_ns,
        stats,
        partition_cpus: "gpu".to_string(),
        pin_memory: false,
        gen_qubits,
        row_stats,
        partitioned: Some(summary),
        mpi: Some((rank, ranks)),
        device: Some(DeviceCellStats {
            devices: vec![device],
            upload_ns,
            download_ns,
        }),
        ..CellResult::new(layer, threads, cfg)
    }
}
