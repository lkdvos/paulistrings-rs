//! One layer of the partitioned engine, export → exchange → coset loop (ARCHITECTURE.md §Partitioning).
//! It neither rebuckets nor finalizes the layer; the driver owns both.

use super::export::{export_layer, ExportScratch};
use super::plan::PartitionPlan;
use super::transport::{ChunkMap, ChunkWait, ExchangeBlock, PartnerPayload, Transport};
use super::PLACEMENT_LOG_TARGET;
use crate::channel::prepared::Prepared;
use crate::engine::bucketed::{
    apply_layer_bucketed, apply_layer_bucketed_with, ExtraRows, LayerKnobs, LayerScratch,
};
use crate::engine::coset::{CosetOrder, Gf2Span};
use crate::engine::knob::{parse_positive, Knob};
use crate::engine::merge::rest_rows_per_key;
use crate::engine::timing::{lap, stamp, tally, timed};
use crate::pauli_sum::hash::PartitionRows;
use crate::pauli_sum::storage::{BucketColumns, PauliSum};
use crate::truncation::TruncationPolicy;

/// Chunks a layer's bulk transfer is cut into by default; not derived from the thread count, since both sides must cut a block alike.
const DEFAULT_EXCHANGE_CHUNKS: usize = 8;

/// `PAULISTRINGS_EXCHANGE_CHUNKS`, else [`DEFAULT_EXCHANGE_CHUNKS`].
static EXCHANGE_CHUNKS: Knob<usize> = Knob::new(
    "PAULISTRINGS_EXCHANGE_CHUNKS",
    PLACEMENT_LOG_TARGET,
    parse_positive,
    || DEFAULT_EXCHANGE_CHUNKS,
);

/// The rows this partition received, one block per remote delta, as an [`ExtraRows`] source for the coset loop.
struct ReceivedRows<'a, const W: usize> {
    blocks: Vec<Option<&'a ExchangeBlock<W>>>,
    order: &'a CosetOrder,
    map: &'a ChunkMap,
    wait: &'a dyn ChunkWait,
    /// Summed across coset tasks.
    #[cfg(feature = "phase-timing")]
    append_ns: std::sync::atomic::AtomicU64,
    /// The part of `append_ns` spent blocked in [`ChunkWait::wait_chunk`].
    #[cfg(feature = "phase-timing")]
    chunk_wait_ns: std::sync::atomic::AtomicU64,
}

impl<'a, const W: usize> ReceivedRows<'a, W> {
    /// Pair each remote delta with its block by [`PartitionPlan::received_block_index`].
    fn new(
        plan: &PartitionPlan,
        received: &'a [Option<PartnerPayload<W>>],
        order: &'a CosetOrder,
        map: &'a ChunkMap,
        wait: &'a dyn ChunkWait,
    ) -> Self {
        let mut blocks = Vec::with_capacity(plan.remote.len());
        for (k, remote) in plan.remote.iter().enumerate() {
            let block_index = plan.received_block_index(k);
            let block = received[remote.partner as usize]
                .as_ref()
                .and_then(|payload| payload.blocks.get(block_index));
            debug_assert!(
                block.is_some(),
                "partition {} sent no block for remote delta {} (entry {}) — the partitions \
                 disagree about the layer's delta set",
                remote.partner,
                block_index,
                remote.entry,
            );
            debug_assert!(
                block.is_none_or(|block| block.header.entry == remote.entry as u32),
                "partition {} sent entry {:?} where entry {} was expected",
                remote.partner,
                block.map(|block| block.header.entry),
                remote.entry,
            );
            blocks.push(block);
        }
        Self {
            blocks,
            order,
            map,
            wait,
            #[cfg(feature = "phase-timing")]
            append_ns: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "phase-timing")]
            chunk_wait_ns: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl<const W: usize> ExtraRows<W> for ReceivedRows<'_, W> {
    const NEEDS_BETA: bool = true;

    fn count(&self, beta: u32) -> usize {
        let position = self.order.position_of(beta);
        let mut rows = 0usize;
        for block in self.blocks.iter().flatten() {
            rows += block.segment(position).2.len();
        }
        rows
    }

    fn append_into(&self, beta: u32, rest: &mut BucketColumns<W>) {
        timed!(let start = std::time::Instant::now());
        let position = self.order.position_of(beta);
        // A whole coset is inside one chunk, so this is one wait per task.
        self.wait.wait_chunk(self.map.chunk_of_position(position));
        timed!(self.chunk_wait_ns.fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        ));
        for block in &self.blocks {
            let Some(block) = block else { continue };
            let (segment_x, segment_z, segment_coeff) = block.segment(position);
            rest.x.extend_from_slice(segment_x);
            rest.z.extend_from_slice(segment_z);
            rest.coeff.extend_from_slice(segment_coeff);
        }
        timed!(self.append_ns.fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        ));
    }
}

/// One partition's reusable per-layer scratch.
#[derive(Debug, Default)]
pub(crate) struct PartitionState<const W: usize> {
    pub layer: LayerScratch<W>,
    pub export: ExportScratch<W>,
    pub chunks: ChunkMap,
}

/// What one layer's exchange moved, from this partition's point of view; the per-partner vectors are empty after a layer without an exchange.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LayerExchangeCounts {
    pub remote_deltas: usize,
    /// Indexed by partner rank.
    pub rows_sent: Vec<u64>,
    /// Indexed by partner rank.
    pub bytes_sent: Vec<u64>,
    /// Summed over partners.
    pub rows_received: u64,
    /// Chunks the received rows moved in; zero after a layer without an exchange.
    pub receive_chunks: u32,
}

impl LayerExchangeCounts {
    /// Zero counts for `size` partners.
    pub(crate) fn none(size: u32) -> Self {
        Self {
            remote_deltas: 0,
            rows_sent: vec![0; size as usize],
            bytes_sent: vec![0; size as usize],
            rows_received: 0,
            receive_chunks: 0,
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
        return LayerExchangeCounts::default();
    }

    stamp!(stamp);
    // The receiver's coset order, derived identically on both sides from the local bucket deltas and the agreed bucket count; the coset loop reuses it.
    let span = Gf2Span::new(&plan.local_bucket_deltas, local.hash().bits());
    let mut order = std::mem::take(&mut state.layer.order);
    order.rebuild(&span, true);
    state.chunks = ChunkMap::of_order(&order, *EXCHANGE_CHUNKS.get());
    let (send, mut counts) = export_layer(local, prepared, plan, size, &order, &mut state.export);
    lap!(stamp, state.layer.stats.export_ns);
    tally!(
        state.layer.stats.rows_exported,
        counts.rows_sent.iter().sum::<u64>()
    );
    #[cfg(debug_assertions)]
    {
        super::export::debug_assert_exported_partitions(&send, rows);
        for remote in &plan.remote {
            debug_assert!(
                send[remote.partner as usize].is_some(),
                "no payload for partner {}, which the plan names",
                remote.partner,
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
        order: Some(&order),
    };

    let PartitionState {
        layer: layer_scratch,
        export: export_scratch,
        chunks: map,
    } = state;
    timed!(let body_ns = std::cell::Cell::new(0u64));
    timed!(let exchange_start = std::time::Instant::now());

    // The rows may still be in flight when the body runs; each coset task waits for its own chunk in `append_into`.
    let (received, rows_received) =
        transport.exchange_layer(send, &mut export_scratch.pool, map, |received, wait| {
            timed!(let body_start = std::time::Instant::now());
            #[cfg(debug_assertions)]
            for block in received
                .iter()
                .flatten()
                .flat_map(|payload| &payload.blocks)
            {
                debug_assert_eq!(
                    block.num_buckets() as usize,
                    local.num_buckets(),
                    "a partner sent a block indexed by {} buckets where this partition has {}: \
                     the partitions disagree about the bucket count",
                    block.num_buckets(),
                    local.num_buckets(),
                );
            }
            let rows_received: u64 = received
                .iter()
                .flatten()
                .flat_map(|payload| &payload.blocks)
                .map(|block| block.rows() as u64)
                .sum();
            let received_rows = ReceivedRows::new(plan, received, &order, map, wait);
            apply_layer_bucketed_with(
                local,
                local_prepared,
                policy,
                layer_scratch,
                &received_rows,
                knobs,
            );
            timed! {
                layer_scratch.stats.recv_rows += rows_received;
                layer_scratch.stats.append_ns += received_rows
                    .append_ns
                    .load(std::sync::atomic::Ordering::Relaxed);
                layer_scratch.stats.chunk_wait_ns += received_rows
                    .chunk_wait_ns
                    .load(std::sync::atomic::Ordering::Relaxed);
                body_ns.set(body_start.elapsed().as_nanos() as u64);
            }
            rows_received
        });
    timed! {
        // Excludes the body it wraps: the transfer the coset loop did not hide.
        layer_scratch.stats.exchange_ns +=
            exchange_start.elapsed().as_nanos() as u64 - body_ns.get();
        stamp.rearm();
    }
    export_scratch.pool.extend(received.into_iter().flatten());
    layer_scratch.order = order;

    counts.remote_deltas = plan.remote.len();
    counts.rows_received = rows_received;
    counts.receive_chunks = map.chunks() as u32;
    counts
}

#[cfg(test)]
mod tests;
