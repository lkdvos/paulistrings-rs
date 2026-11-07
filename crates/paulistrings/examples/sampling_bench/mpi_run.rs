//! `--backend mpi-gpu`: one partition per MPI rank on its own device (`MpiGpuSum`), and the per-rank trace gathered into one.

use std::time::Instant;

use paulistrings::gpu::{local_device_for_comm, GpuError, GpuScatterOptions, MpiGpuSum};
use paulistrings::mpi::rsmpi::topology::{Communicator, SimpleCommunicator};
use paulistrings::mpi::MpiTransport;
use paulistrings::{
    BuiltinTruncation, Coefficient, Collectives, Direction, LayerPhases, PartitionLayerRecord,
    PartitionTrace, PropagateOptions,
};

use crate::cli::Config;
use crate::models::IsingQuench;
use crate::run::{query_clocks, readout, scatter_rows, Report};

/// Every rank's slot of `local` filled in, summed over the group so each rank holds all of them. **Collective.**
fn allgather_u64(collectives: &dyn Collectives, local: &[u64]) -> Vec<u64> {
    let (rank, size) = (collectives.rank() as usize, collectives.size() as usize);
    let mut all = vec![0u64; local.len() * size];
    all[rank * local.len()..(rank + 1) * local.len()].copy_from_slice(local);
    collectives.allreduce_sum_u64(&mut all);
    all
}

/// Per-layer `u64` fields of one rank's row, in [`pack_row`] order, before the two per-destination vectors.
const SCALAR_FIELDS: usize = 13;

fn pack_row(layer: &PartitionLayerRecord, out: &mut Vec<u64>) {
    let p = &layer.phases[0];
    out.extend([
        u64::from(layer.bits[0]),
        layer.terms_in[0] as u64,
        layer.terms_out[0] as u64,
        layer.rows_received[0],
        layer.nanos[0],
        u64::from(layer.receive_chunks[0]),
        p.rebucket_ns,
        p.prepare_ns,
        p.collective_ns,
        p.export_ns,
        p.exchange_ns,
        p.coset_loop_ns,
        p.finalize_ns,
    ]);
    out.extend(&layer.rows_sent[0]);
    out.extend(&layer.bytes_sent[0]);
}

/// This rank's trace, one partition per row, widened to every rank's partition: the shape an in-process split traces. **Collective.**
fn gather_trace(collectives: &dyn Collectives, local: PartitionTrace) -> PartitionTrace {
    let (rank, size) = (collectives.rank() as usize, collectives.size() as usize);
    let per_layer = SCALAR_FIELDS + 2 * size;
    let mut packed = Vec::with_capacity(local.layers.len() * per_layer);
    for layer in &local.layers {
        pack_row(layer, &mut packed);
    }
    let all = allgather_u64(collectives, &packed);
    let per_rank = packed.len();
    let mut layers = local.layers;
    for (i, layer) in layers.iter_mut().enumerate() {
        let own_device = layer.device[0];
        *layer = PartitionLayerRecord {
            device: (0..size).map(|r| if r == rank { own_device } else { None }).collect(),
            ..std::mem::take(layer)
        };
        let (mut bits, mut terms_in, mut terms_out, mut received) =
            (vec![], vec![], vec![], vec![]);
        let (mut nanos, mut chunks, mut phases) = (vec![], vec![], vec![]);
        let (mut rows_sent, mut bytes_sent) = (vec![], vec![]);
        for r in 0..size {
            let f = &all[r * per_rank + i * per_layer..][..per_layer];
            bits.push(f[0] as u8);
            terms_in.push(f[1] as usize);
            terms_out.push(f[2] as usize);
            received.push(f[3]);
            nanos.push(f[4]);
            chunks.push(f[5] as u32);
            phases.push(LayerPhases {
                rebucket_ns: f[6],
                prepare_ns: f[7],
                collective_ns: f[8],
                export_ns: f[9],
                exchange_ns: f[10],
                coset_loop_ns: f[11],
                finalize_ns: f[12],
                fine: None,
            });
            rows_sent.push(f[SCALAR_FIELDS..][..size].to_vec());
            bytes_sent.push(f[SCALAR_FIELDS + size..][..size].to_vec());
        }
        layer.bits = bits;
        layer.terms_in = terms_in;
        layer.terms_out = terms_out;
        layer.rows_received = received;
        layer.nanos = nanos;
        layer.receive_chunks = chunks;
        layer.phases = phases;
        layer.rows_sent = rows_sent;
        layer.bytes_sent = bytes_sent;
        layer.refine_passes = vec![0; size];
    }
    PartitionTrace { layers }
}

/// Rank 0's phase counters in full, and the exchange counters of every rank. **Collective.**
#[cfg(feature = "phase-timing")]
fn gather_stats(
    collectives: &dyn Collectives,
    own: paulistrings::PhaseStats,
) -> Vec<paulistrings::PhaseStats> {
    let fields = [own.export_ns, own.exchange_ns, own.chunk_wait_ns, own.truncate_ns];
    let all = allgather_u64(collectives, &fields);
    let rank = collectives.rank() as usize;
    all.chunks(fields.len())
        .enumerate()
        .map(|(r, f)| {
            let mut stats = if r == rank { own } else { paulistrings::PhaseStats::default() };
            (stats.export_ns, stats.exchange_ns, stats.chunk_wait_ns, stats.truncate_ns) =
                (f[0], f[1], f[2], f[3]);
            stats
        })
        .collect()
}

/// Each rank's `clocks.sm, clocks.mem`, as `r<rank>: <sm>, <mem>` joined on rank 0. **Collective.**
fn gather_clocks(collectives: &dyn Collectives, own: Option<String>) -> String {
    let mhz = |field: Option<&str>| -> u64 {
        field.and_then(|f| f.trim().trim_end_matches("MHz").trim().parse().ok()).unwrap_or(0)
    };
    let own = own.unwrap_or_default();
    let mut fields = own.split(',');
    let all = allgather_u64(collectives, &[mhz(fields.next()), mhz(fields.next())]);
    all.chunks(2)
        .enumerate()
        .map(|(r, c)| format!("r{r}: {} MHz, {} MHz", c[0], c[1]))
        .collect::<Vec<_>>()
        .join("; ")
}

pub(crate) fn run_mpi_gpu<const W: usize, C: Coefficient>(
    cfg: &Config,
    quench: &IsingQuench,
    policy: &BuiltinTruncation,
) -> Report {
    let world = SimpleCommunicator::world();
    let (rank, size) = (world.rank() as u32, world.size() as u32);
    let fail = |what: &str, e: GpuError| -> ! {
        eprintln!("sampling_bench: rank {rank}: {what}: {e}");
        std::process::exit(2);
    };
    let device = match cfg.device {
        Some(device) => device,
        None => local_device_for_comm(&world).unwrap_or_else(|e| fail("device pick", e)),
    };
    let (circuit, _) = quench.circuit::<W>(1);
    let observable = quench.observable_sum::<W, C>();
    let (rows, crossed_bonds) = scatter_rows::<W>(cfg, quench, size as usize);
    let options = GpuScatterOptions {
        rows,
        ..GpuScatterOptions::new(&[device]).unwrap_or_else(|e| fail("placement", e))
    };
    let transport = MpiTransport::from_communicator(&world);
    let started = Instant::now();
    let mut split = MpiGpuSum::<W, C>::scatter_to_device_with(&observable, transport, options)
        .unwrap_or_else(|e| fail("scatter", e));
    split.transport().barrier();
    let upload_ns = started.elapsed().as_nanos() as u64;
    drop(observable);
    split.enable_trace();
    let (mut step_ns, mut step_terms) = (Vec::new(), Vec::new());
    for _ in 0..cfg.steps {
        let started = Instant::now();
        split
            .propagate_with(&circuit, policy, Direction::Heisenberg, PropagateOptions::default())
            .unwrap_or_else(|e| fail("propagate", e));
        // The step ends when the slowest rank does.
        split.transport().barrier();
        step_ns.push(started.elapsed().as_nanos() as u64);
        step_terms.push(split.len());
    }
    let clocks = gather_clocks(split.transport(), query_clocks(&[device]));
    let started = Instant::now();
    // Each rank reads its own share.
    let share = split.local_to_host().unwrap_or_else(|e| fail("download", e));
    let mut final_readout = [readout(&share)];
    split.transport().allreduce_sum_f64(&mut final_readout);
    let download_ns = started.elapsed().as_nanos() as u64;
    let local = split.take_trace().expect("tracing was enabled");
    let trace = gather_trace(split.transport(), local);
    let devices = allgather_u64(split.transport(), &[u64::from(device)]);
    #[cfg(feature = "phase-timing")]
    let stats = {
        let own = split.take_stats().per_partition.remove(0);
        gather_stats(split.transport(), own)
    };
    Report {
        step_ns,
        step_terms,
        final_readout: final_readout[0],
        trace,
        upload_ns,
        download_ns,
        clocks: Some(clocks),
        devices: devices.into_iter().map(|d| d as u32).collect(),
        crossed_bonds,
        root: rank == 0,
        #[cfg(feature = "phase-timing")]
        stats,
    }
}
