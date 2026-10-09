//! One layer of the partitioned engine, export → exchange → coset loop (ARCHITECTURE.md §Partitioning).
//! It neither rebuckets nor finalizes the layer; the driver owns both.

use super::export::{export_layer, ExportScratch};
use super::plan::PartitionPlan;
use super::transport::{ChunkMap, ChunkWait, ExchangeBlock, PartnerPayload, Transport};
use crate::channel::prepared::Prepared;
use crate::engine::bucketed::{
    apply_layer_bucketed, apply_layer_bucketed_with, rest_rows_per_key, ExtraRows, LayerKnobs,
    LayerScratch,
};
use crate::engine::coset::Gf2Span;
use crate::pauli_sum::hash::PartitionRows;
use crate::pauli_sum::storage::PauliSum;
use crate::truncation::TruncationPolicy;
use num_complex::Complex64;

/// Chunks a layer's bulk transfer is cut into by default; not derived from the thread count, since both sides must cut a block alike.
const DEFAULT_EXCHANGE_CHUNKS: usize = 8;

/// [`DEFAULT_EXCHANGE_CHUNKS`] unless `PAULISTRINGS_EXCHANGE_CHUNKS` names another; every rank must see the same environment.
fn exchange_chunks() -> usize {
    static CHUNKS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CHUNKS.get_or_init(|| {
        std::env::var("PAULISTRINGS_EXCHANGE_CHUNKS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&k| k > 0)
            .unwrap_or(DEFAULT_EXCHANGE_CHUNKS)
    })
}

/// The rows this partition received, one block per remote delta, as an [`ExtraRows`] source for the coset loop.
struct RecvRows<'a, const W: usize> {
    blocks: Vec<Option<&'a ExchangeBlock<W>>>,
    map: &'a ChunkMap,
    wait: &'a dyn ChunkWait,
    /// Summed across coset tasks.
    #[cfg(feature = "phase-timing")]
    append_ns: std::sync::atomic::AtomicU64,
    /// The part of `append_ns` spent blocked in [`ChunkWait::wait_chunk`].
    #[cfg(feature = "phase-timing")]
    chunk_wait_ns: std::sync::atomic::AtomicU64,
}

impl<'a, const W: usize> RecvRows<'a, W> {
    /// Pair each remote delta with its block: the `j`-th of `remote_for_partner(q)` is the `j`-th block of `recv[q]`, since remoteness is symmetric.
    fn new(
        plan: &PartitionPlan,
        recv: &'a [Option<PartnerPayload<W>>],
        map: &'a ChunkMap,
        wait: &'a dyn ChunkWait,
    ) -> Self {
        let mut blocks = Vec::with_capacity(plan.remote.len());
        for r in &plan.remote {
            let j = plan
                .remote_for_partner(r.partner)
                .position(|other| other.entry == r.entry)
                .expect("a remote delta is in its own partner's list");
            let block = recv[r.partner as usize]
                .as_ref()
                .and_then(|payload| payload.blocks.get(j));
            debug_assert!(
                block.is_some(),
                "partition {} sent no block for remote delta {} (entry {}) — the partitions \
                 disagree about the layer's delta set",
                r.partner,
                j,
                r.entry,
            );
            debug_assert!(
                block.is_none_or(|b| b.header.entry == r.entry as u32),
                "partition {} sent entry {:?} where entry {} was expected",
                r.partner,
                block.map(|b| b.header.entry),
                r.entry,
            );
            blocks.push(block);
        }
        Self {
            blocks,
            map,
            wait,
            #[cfg(feature = "phase-timing")]
            append_ns: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "phase-timing")]
            chunk_wait_ns: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl<const W: usize> ExtraRows<W> for RecvRows<'_, W> {
    const NEEDS_BETA: bool = true;

    fn count(&self, beta: u32) -> usize {
        let p = self.map.position_of(beta);
        let mut n = 0usize;
        for block in self.blocks.iter().flatten() {
            n += block.segment(p).2.len();
        }
        n
    }

    fn append_into(
        &self,
        beta: u32,
        x: &mut Vec<[u64; W]>,
        z: &mut Vec<[u64; W]>,
        c: &mut Vec<Complex64>,
    ) {
        #[cfg(feature = "phase-timing")]
        let start = std::time::Instant::now();
        let p = self.map.position_of(beta);
        // A whole coset is inside one chunk, so this is one wait per task.
        self.wait.wait_chunk(self.map.chunk_of_position(p));
        #[cfg(feature = "phase-timing")]
        self.chunk_wait_ns.fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        for block in &self.blocks {
            let Some(block) = block else { continue };
            let (sx, sz, sc) = block.segment(p);
            x.extend_from_slice(sx);
            z.extend_from_slice(sz);
            c.extend_from_slice(sc);
        }
        #[cfg(feature = "phase-timing")]
        self.append_ns.fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// One partition's reusable per-layer scratch.
#[derive(Debug, Default)]
pub(super) struct PartitionState<const W: usize> {
    pub layer: LayerScratch<W>,
    pub export: ExportScratch<W>,
    pub chunks: ChunkMap,
}

/// What one layer's exchange moved, from this partition's point of view.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LayerExchangeCounts {
    pub remote_deltas: usize,
    /// Indexed by partner rank.
    pub rows_sent: Vec<u64>,
    /// Indexed by partner rank.
    pub bytes_sent: Vec<u64>,
    /// Summed over partners.
    pub rows_received: u64,
}

impl LayerExchangeCounts {
    pub(crate) fn none(size: u32) -> Self {
        Self {
            remote_deltas: 0,
            rows_sent: vec![0; size as usize],
            bytes_sent: vec![0; size as usize],
            rows_received: 0,
        }
    }
}

/// One layer on this partition's share; `plan` must be `PartitionPlan::new(prepared, rows, transport.rank())`.
pub(super) fn apply_layer_partitioned_with_plan<const W: usize, T, X>(
    local: &mut PauliSum<W>,
    prepared: &Prepared<W>,
    plan: &PartitionPlan,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] rows: &PartitionRows<W>,
    policy: &T,
    state: &mut PartitionState<W>,
    transport: &X,
) -> LayerExchangeCounts
where
    T: TruncationPolicy<W> + ?Sized,
    X: Transport,
{
    let size = transport.size();

    // No transport call at all: every partition takes this branch, since `part(d)` depends on the delta alone.
    if !plan.has_remote() {
        apply_layer_bucketed(local, prepared, policy, &mut state.layer);
        return LayerExchangeCounts::none(size);
    }

    #[cfg(feature = "phase-timing")]
    let mut stamp = crate::engine::stats::Stamp::now();
    // The receiver's coset order, derived identically on both sides from the local bucket deltas and the agreed bucket count.
    let span = Gf2Span::new(&plan.local_bucket_deltas, local.hash().bits());
    state
        .chunks
        .rebuild(&span, local.num_buckets(), exchange_chunks());
    let (send, export) = export_layer(
        local,
        prepared,
        plan,
        size,
        &state.chunks,
        &mut state.export,
    );
    #[cfg(feature = "phase-timing")]
    {
        stamp.lap(&mut state.layer.stats.export_ns);
        state.layer.stats.rows_exported += export.rows_to.iter().sum::<u64>();
    }
    #[cfg(debug_assertions)]
    {
        super::export::debug_assert_exported_partitions(&send, rows);
        for r in &plan.remote {
            debug_assert!(
                send[r.partner as usize].is_some(),
                "no payload for partner {}, which the plan names",
                r.partner,
            );
        }
    }
    let retained;
    let local_prepared: &Prepared<W> = match prepared {
        Prepared::Local(ptm) => {
            retained = Prepared::Local(ptm.retain_entries(&plan.local_entries));
            &retained
        }
        // A rotation's identity entry is always local, so the generator crosses; `generator_local` switches its pass off.
        Prepared::Rotation(_) => prepared,
    };
    let knobs = LayerKnobs {
        bucket_deltas: Some(&plan.local_bucket_deltas),
        rest_streams: Some(plan.rest_streams_total),
        // From `prepared`, not `local_prepared`, so every partition picks the unpartitioned run's sort kernel.
        rows_per_key: match prepared {
            Prepared::Local(ptm) => Some(rest_rows_per_key(ptm)),
            Prepared::Rotation(_) => None,
        },
        generator_local: match prepared {
            Prepared::Rotation(_) => plan.local_entries[1],
            Prepared::Local(_) => true,
        },
    };

    let PartitionState {
        layer: layer_scratch,
        export: export_scratch,
        chunks: map,
    } = state;
    #[cfg(feature = "phase-timing")]
    let body_ns = std::cell::Cell::new(0u64);
    #[cfg(feature = "phase-timing")]
    let exchange_start = std::time::Instant::now();

    // The rows may still be in flight when the body runs; each coset task waits for its own chunk in `append_into`.
    let (recv, rows_received) =
        transport.exchange_layer(send, &mut export_scratch.pool, map, |recv, wait| {
            #[cfg(feature = "phase-timing")]
            let body_start = std::time::Instant::now();
            #[cfg(debug_assertions)]
            for block in recv.iter().flatten().flat_map(|payload| &payload.blocks) {
                debug_assert_eq!(
                    block.num_buckets() as usize,
                    local.num_buckets(),
                    "a partner sent a block indexed by {} buckets where this partition has {}: \
                     the partitions disagree about the bucket count",
                    block.num_buckets(),
                    local.num_buckets(),
                );
            }
            let rows_received: u64 = recv
                .iter()
                .flatten()
                .flat_map(|payload| &payload.blocks)
                .map(|block| block.rows() as u64)
                .sum();
            let recv_rows = RecvRows::new(plan, recv, map, wait);
            apply_layer_bucketed_with(
                local,
                local_prepared,
                policy,
                layer_scratch,
                &recv_rows,
                knobs,
            );
            #[cfg(feature = "phase-timing")]
            {
                layer_scratch.stats.recv_rows += rows_received;
                layer_scratch.stats.append_ns += recv_rows
                    .append_ns
                    .load(std::sync::atomic::Ordering::Relaxed);
                layer_scratch.stats.chunk_wait_ns += recv_rows
                    .chunk_wait_ns
                    .load(std::sync::atomic::Ordering::Relaxed);
                body_ns.set(body_start.elapsed().as_nanos() as u64);
            }
            rows_received
        });
    #[cfg(feature = "phase-timing")]
    {
        // Excludes the body it wraps: the transfer the coset loop did not hide.
        layer_scratch.stats.exchange_ns +=
            exchange_start.elapsed().as_nanos() as u64 - body_ns.get();
        stamp.rearm();
    }
    export_scratch.pool.extend(recv.into_iter().flatten());

    LayerExchangeCounts {
        remote_deltas: plan.remote.len(),
        rows_sent: export.rows_to,
        bytes_sent: export.bytes_to,
        rows_received,
    }
}

#[cfg(test)]
mod tests;
