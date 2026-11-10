//! The timed propagation on the host or on the devices of this process, and what it measured.

use std::time::Instant;

use paulistrings::{
    propagate_with, Coefficient, Direction, LayerScratch, PartitionTrace, PauliAxis, PauliSum,
    ProductBasis, PropagateOptions, TruncationPolicy,
};

use crate::cli::Config;
use crate::models::IsingQuench;
pub(crate) use crate::rows::scatter_rows;

/// One run's measurements; the step vectors have one entry per step.
pub(crate) struct Report {
    pub(crate) step_ns: Vec<u64>,
    pub(crate) step_terms: Vec<usize>,
    pub(crate) final_readout: f64,
    /// Every partition's rows, an MPI run's ranks included; `Some` only for the traced run of `--trace`.
    pub(crate) trace: Option<PartitionTrace>,
    /// The traced run's total wall time, `Some` exactly when `trace` is.
    pub(crate) traced_ns: Option<u64>,
    /// The device upload (the scatter, kernel loading included); zero on the host.
    pub(crate) upload_ns: u64,
    /// The device download before the final readout; zero on the host.
    pub(crate) download_ns: u64,
    /// `clocks.sm, clocks.mem` right after the timed steps, one entry per device, for a device run.
    pub(crate) clocks: Option<String>,
    /// Each partition's device ordinal in rank order; empty on the host.
    pub(crate) devices: Vec<u32>,
    /// The rows a split run scattered by, `None` unsplit.
    pub(crate) rows: Option<crate::rows::RowsSummary>,
    /// Whether this process prints and writes the report: false only on a non-zero MPI rank.
    pub(crate) root: bool,
    /// One entry per partition; an MPI run carries rank 0's in full and the exchange fields of the others.
    #[cfg(feature = "phase-timing")]
    pub(crate) stats: Vec<paulistrings::PhaseStats>,
}

pub(crate) fn readout<const W: usize, C: Coefficient>(sum: &PauliSum<W, C>) -> f64 {
    sum.expectation_product(&ProductBasis::uniform(PauliAxis::Z)).to_complex64().re
}

pub(crate) fn run_cpu<const W: usize, C: Coefficient, P: TruncationPolicy<W, C>>(
    cfg: &Config,
    quench: &IsingQuench,
    policy: &P,
    traced: bool,
) -> Report {
    let (circuit, _) = quench.circuit::<W>(1);
    let observable = quench.observable_sum::<W, C>();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.threads)
        .build()
        .expect("failed to build a rayon thread pool");
    pool.install(|| {
        let mut scratch = LayerScratch::<W, C>::new();
        if traced {
            scratch.enable_trace();
        }
        let (mut step_ns, mut step_terms) = (Vec::new(), Vec::new());
        let mut sum = observable;
        for _ in 0..cfg.steps {
            let started = Instant::now();
            sum = propagate_with(
                &circuit,
                sum,
                policy,
                Direction::Heisenberg,
                &mut scratch,
                PropagateOptions::default(),
            );
            step_ns.push(started.elapsed().as_nanos() as u64);
            step_terms.push(sum.len());
        }
        Report {
            step_ns,
            step_terms,
            final_readout: readout(&sum),
            trace: scratch.take_trace(),
            traced_ns: None,
            upload_ns: 0,
            download_ns: 0,
            clocks: None,
            devices: Vec::new(),
            rows: None,
            root: true,
            #[cfg(feature = "phase-timing")]
            stats: scratch.take_stats().per_partition,
        }
    })
}

/// One sum split over `--partitions` host partitions of this process (`--backend cpus`), `--threads` shared evenly.
pub(crate) fn run_cpus<const W: usize, C: Coefficient, P: TruncationPolicy<W, C>>(
    cfg: &Config,
    quench: &IsingQuench,
    policy: &P,
    traced: bool,
) -> Report {
    use paulistrings::{
        PartitionConfig, PartitionRuntime, PartitionedSum, Placement, ScatterOptions,
    };

    let partitions = cfg.partitions;
    let (circuit, _) = quench.circuit::<W>(1);
    let observable = quench.observable_sum::<W, C>();
    let config = PartitionConfig {
        placement: Placement::Unpinned { partitions },
        bind_memory: false,
        threads_per_partition: Some((cfg.threads / partitions).max(1)),
    };
    let runtime = PartitionRuntime::new(&config).unwrap_or_else(|e| {
        eprintln!("sampling_bench: partitions: {e}");
        std::process::exit(2);
    });
    let (rows, rows_summary) =
        scatter_rows::<W, C>(cfg.rows, quench, &observable, &circuit, partitions);
    let mut split =
        PartitionedSum::<W, C>::scatter_with(observable, ScatterOptions { runtime, rows });
    if traced {
        split.enable_trace();
    }
    let (mut step_ns, mut step_terms) = (Vec::new(), Vec::new());
    for _ in 0..cfg.steps {
        let started = Instant::now();
        split.propagate_with(&circuit, policy, Direction::Heisenberg, PropagateOptions::default());
        step_ns.push(started.elapsed().as_nanos() as u64);
        step_terms.push(split.len());
    }
    Report {
        step_ns,
        step_terms,
        final_readout: readout(&split.gather()),
        trace: split.take_trace(),
        traced_ns: None,
        upload_ns: 0,
        download_ns: 0,
        clocks: None,
        devices: Vec::new(),
        rows: Some(rows_summary),
        root: true,
        #[cfg(feature = "phase-timing")]
        stats: split.take_stats().per_partition,
    }
}

/// One device (`--backend gpu`) or several partitions of this process (`--backend gpus`): a [`GpuPartitionedSum`](paulistrings::gpu::GpuPartitionedSum) either way.
#[cfg(feature = "cuda")]
pub(crate) fn run_gpu<const W: usize, C: Coefficient>(
    cfg: &Config,
    quench: &IsingQuench,
    policy: &paulistrings::BuiltinTruncation,
    traced: bool,
) -> Report {
    use paulistrings::gpu::{GpuError, GpuPartitionedSum, GpuScatterOptions};

    let fail = |what: &str, e: GpuError| -> ! {
        eprintln!("sampling_bench: {what}: {e}");
        std::process::exit(2);
    };
    let devices = match cfg.backend {
        crate::cli::Backend::Gpu => vec![cfg.device.unwrap_or(0)],
        _ => cfg.devices.clone(),
    };
    let (circuit, _) = quench.circuit::<W>(1);
    let observable = quench.observable_sum::<W, C>();
    let started = Instant::now();
    let options = GpuScatterOptions::new(&devices).unwrap_or_else(|e| fail("placement", e));
    let (rows, rows_summary) =
        scatter_rows::<W, C>(cfg.rows, quench, &observable, &circuit, devices.len());
    let mut split = GpuPartitionedSum::<W, C>::scatter_to_devices_with(
        &observable,
        GpuScatterOptions { rows, ..options },
    )
    .unwrap_or_else(|e| fail("scatter", e));
    let upload_ns = started.elapsed().as_nanos() as u64;
    drop(observable);
    if traced {
        split.enable_trace();
    }
    let (mut step_ns, mut step_terms) = (Vec::new(), Vec::new());
    for _ in 0..cfg.steps {
        let started = Instant::now();
        split
            .propagate_with(&circuit, policy, Direction::Heisenberg, PropagateOptions::default())
            .unwrap_or_else(|e| fail("propagate", e));
        step_ns.push(started.elapsed().as_nanos() as u64);
        step_terms.push(split.len());
    }
    let devices = split.devices();
    let clocks = query_clocks(&devices);
    let started = Instant::now();
    let host = split.gather().unwrap_or_else(|e| fail("gather", e));
    let download_ns = started.elapsed().as_nanos() as u64;
    Report {
        step_ns,
        step_terms,
        final_readout: readout(&host),
        trace: split.take_trace(),
        traced_ns: None,
        upload_ns,
        download_ns,
        clocks,
        devices,
        rows: Some(rows_summary),
        root: true,
        #[cfg(feature = "phase-timing")]
        stats: split.take_stats().per_partition,
    }
}

/// `nvidia-smi`'s current SM and memory clocks of each distinct device in `devices`, `;`-separated, or `None` without it.
#[cfg(feature = "cuda")]
pub(crate) fn query_clocks(devices: &[u32]) -> Option<String> {
    let mut distinct = devices.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let ids = distinct.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm,clocks.mem", "--format=csv,noheader", &format!("--id={ids}")])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().lines().collect::<Vec<_>>().join("; "))
}
