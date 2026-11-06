//! The timed propagation on the host or on one device, and what it measured.

use std::time::Instant;

use paulistrings::{
    propagate_with, Coefficient, Direction, LayerScratch, PartitionTrace, PauliAxis, PauliSum,
    ProductBasis, PropagateOptions, TruncationPolicy,
};

use crate::cli::Config;
use crate::models::IsingQuench;

/// One run's measurements; the step vectors have one entry per propagate call.
pub(crate) struct Report {
    pub(crate) step_ns: Vec<u64>,
    pub(crate) step_terms: Vec<usize>,
    pub(crate) readouts: Vec<f64>,
    pub(crate) final_readout: f64,
    pub(crate) trace: PartitionTrace,
    /// The device upload, kernel loading included; zero on the host.
    pub(crate) upload_ns: u64,
    /// The device download before the final readout; zero on the host.
    pub(crate) download_ns: u64,
    /// `clocks.sm, clocks.mem` right after the timed steps, for a device run.
    pub(crate) clocks: Option<String>,
    #[cfg(feature = "phase-timing")]
    pub(crate) stats: paulistrings::PhaseStats,
}

fn readout<const W: usize, C: Coefficient>(sum: &PauliSum<W, C>) -> f64 {
    sum.expectation_product(&ProductBasis::uniform(PauliAxis::Z)).to_complex64().re
}

/// Propagate calls a run makes, and the steps each one covers.
fn calls(cfg: &Config) -> (usize, usize) {
    if cfg.one_call {
        (1, cfg.steps)
    } else {
        (cfg.steps, 1)
    }
}

pub(crate) fn run_cpu<const W: usize, C: Coefficient, P: TruncationPolicy<W, C>>(
    cfg: &Config,
    quench: &IsingQuench,
    policy: &P,
) -> Report {
    let (calls, steps_per_call) = calls(cfg);
    let (circuit, _) = quench.circuit::<W>(steps_per_call);
    let observable = quench.observable_sum::<W, C>();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.threads)
        .build()
        .expect("failed to build a rayon thread pool");
    pool.install(|| {
        let mut scratch = LayerScratch::<W, C>::new();
        scratch.enable_trace();
        let (mut step_ns, mut step_terms, mut readouts) = (Vec::new(), Vec::new(), Vec::new());
        let mut sum = observable;
        for _ in 0..calls {
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
            if cfg.readout_every_step {
                readouts.push(readout(&sum));
            }
        }
        Report {
            step_ns,
            step_terms,
            readouts,
            final_readout: readout(&sum),
            trace: scratch.take_trace().expect("tracing was enabled"),
            upload_ns: 0,
            download_ns: 0,
            clocks: None,
            #[cfg(feature = "phase-timing")]
            stats: scratch.take_stats().per_partition[0],
        }
    })
}

#[cfg(feature = "cuda")]
pub(crate) fn run_gpu<const W: usize, C: Coefficient>(
    cfg: &Config,
    quench: &IsingQuench,
    policy: &paulistrings::BuiltinTruncation,
) -> Report {
    use paulistrings::gpu::{GpuError, GpuPartitionedSum};

    let fail = |what: &str, e: GpuError| -> ! {
        eprintln!("sampling_bench: device {}: {what}: {e}", cfg.device);
        std::process::exit(2);
    };
    let (calls, steps_per_call) = calls(cfg);
    let (circuit, _) = quench.circuit::<W>(steps_per_call);
    let observable = quench.observable_sum::<W, C>();
    let started = Instant::now();
    let mut split = GpuPartitionedSum::<W, C>::from_host(&observable, cfg.device)
        .unwrap_or_else(|e| fail("upload", e));
    let upload_ns = started.elapsed().as_nanos() as u64;
    split.enable_trace();
    let (mut step_ns, mut step_terms, mut readouts) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..calls {
        let started = Instant::now();
        split
            .propagate_with(&circuit, policy, Direction::Heisenberg, PropagateOptions::default())
            .unwrap_or_else(|e| fail("propagate", e));
        step_ns.push(started.elapsed().as_nanos() as u64);
        step_terms.push(split.len());
        if cfg.readout_every_step {
            readouts.push(readout(&split.gather().unwrap_or_else(|e| fail("gather", e))));
        }
    }
    let clocks = query_clocks(cfg.device);
    let started = Instant::now();
    let host = split.gather().unwrap_or_else(|e| fail("gather", e));
    let download_ns = started.elapsed().as_nanos() as u64;
    Report {
        step_ns,
        step_terms,
        readouts,
        final_readout: readout(&host),
        trace: split.take_trace().expect("tracing was enabled"),
        upload_ns,
        download_ns,
        clocks,
        #[cfg(feature = "phase-timing")]
        stats: split.take_stats().per_partition[0],
    }
}

/// `nvidia-smi`'s current SM and memory clocks of `device`, or `None` without it.
#[cfg(feature = "cuda")]
fn query_clocks(device: u32) -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=clocks.sm,clocks.mem",
            "--format=csv,noheader",
            &format!("--id={device}"),
        ])
        .output()
        .ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}
