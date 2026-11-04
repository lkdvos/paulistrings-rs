//! One layer of the partitioned engine, export → exchange → coset loop.
//! It neither rebuckets nor finalizes the layer; the driver owns both.

use super::exchange_block::{ExchangeBlock, PartnerPayload};
use super::export::{export_layer, ExportScratch};
use super::PLACEMENT_LOG_TARGET;
use crate::channel::prepared::Prepared;
use crate::coefficient::Coefficient;
use crate::engine::bucketed::{
    apply_layer_bucketed_with, is_rotation_table, EmitTest, ExtraRows, LayerKnobs, LayerScratch,
    NoExtra,
};
use crate::engine::coset::{CosetOrder, Gf2Span};
use crate::engine::knob::{parse_positive, Knob};
use crate::engine::merge::rest_rows_per_key;
use crate::engine::plan::PartitionPlan;
use crate::engine::timing::{tally, timed};
use crate::engine::transport::{ChunkMap, ChunkWait, Transport};
use crate::pauli_sum::hash::PartitionRows;
use crate::pauli_sum::storage::{BucketColumns, PauliSum};
use crate::truncation::TruncationPolicy;
use std::time::Instant;

/// Chunks a layer's bulk transfer is cut into by default; not derived from the thread count, since both sides must cut a block alike.
const DEFAULT_EXCHANGE_CHUNKS: usize = 8;

/// `PAULISTRINGS_EXCHANGE_CHUNKS`, else [`DEFAULT_EXCHANGE_CHUNKS`].
static EXCHANGE_CHUNKS: Knob<usize> =
    Knob::new("PAULISTRINGS_EXCHANGE_CHUNKS", PLACEMENT_LOG_TARGET, parse_positive, || {
        DEFAULT_EXCHANGE_CHUNKS
    });

/// The rows this partition received, one block per remote delta, as an [`ExtraRows`] source for the coset loop.
struct ReceivedRows<'a, const W: usize, C: Coefficient = num_complex::Complex64> {
    blocks: Vec<Option<&'a ExchangeBlock<W, C>>>,
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

impl<'a, const W: usize, C: Coefficient> ReceivedRows<'a, W, C> {
    /// Pair each remote delta with its block by [`PartitionPlan::received_block_index`].
    fn new(
        plan: &PartitionPlan,
        received: &'a [Option<PartnerPayload<W, C>>],
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

impl<const W: usize, C: Coefficient> ExtraRows<W, C> for ReceivedRows<'_, W, C> {
    const NEEDS_BETA: bool = true;

    fn count(&self, beta: u32) -> usize {
        let position = self.order.position_of(beta);
        let mut rows = 0usize;
        for block in self.blocks.iter().flatten() {
            rows += block.segment(position).2.len();
        }
        rows
    }

    fn append_into(&self, beta: u32, rest: &mut BucketColumns<W, C>) {
        timed!(let start = std::time::Instant::now());
        let position = self.order.position_of(beta);
        // A whole coset is inside one chunk, so this is one wait per task.
        self.wait.wait_chunk(self.map.chunk_of_position(position));
        timed!(self
            .chunk_wait_ns
            .fetch_add(start.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed,));
        for block in &self.blocks {
            let Some(block) = block else { continue };
            let (segment_x, segment_z, segment_coeff) = block.segment(position);
            rest.x.extend_from_slice(segment_x);
            rest.z.extend_from_slice(segment_z);
            rest.coeff.extend_from_slice(segment_coeff);
        }
        timed!(self
            .append_ns
            .fetch_add(start.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed,));
    }
}

/// One partition's reusable per-layer scratch.
#[derive(Debug, Default)]
pub(crate) struct PartitionState<const W: usize, C: Coefficient = num_complex::Complex64> {
    pub layer: LayerScratch<W, C>,
    pub export: ExportScratch<W, C>,
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
    /// [`LayerPhases::export_ns`](crate::LayerPhases::export_ns), zero unless the layer was timed.
    pub export_ns: u64,
    /// [`LayerPhases::exchange_ns`](crate::LayerPhases::exchange_ns), zero unless the layer was timed.
    pub exchange_ns: u64,
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
            export_ns: 0,
            exchange_ns: 0,
        }
    }
}

/// One layer on this partition's share; `plan` must be `PartitionPlan::new(prepared, rows, transport.rank())`, `clocked` fills the counts' phase times, and `ahead` is the next layer's one remote delta, if it has exactly one, whose rows the coset loop counts for it.
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn apply_layer_partitioned_with_plan<const W: usize, C: Coefficient, T, X>(
    local: &mut PauliSum<W, C>,
    prepared: &Prepared<W, C>,
    plan: &PartitionPlan,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] rows: &PartitionRows<W>,
    policy: &T,
    state: &mut PartitionState<W, C>,
    transport: &X,
    clocked: bool,
    ahead: Option<&EmitTest<'_, W, C>>,
) -> LayerExchangeCounts
where
    T: TruncationPolicy<W, C> + ?Sized,
    X: Transport,
{
    let size = transport.size();

    // No transport call at all: every partition takes this branch, since `part(d)` depends on the delta alone.
    if !plan.has_remote() {
        apply_layer_bucketed_with(
            local,
            prepared,
            policy,
            &mut state.layer,
            &NoExtra,
            LayerKnobs::default(),
            ahead,
        );
        return LayerExchangeCounts::default();
    }

    let measured = clocked || cfg!(feature = "phase-timing");
    let export_start = measured.then(Instant::now);
    // The receiver's coset order, derived identically on both sides from the local bucket deltas and the agreed bucket count; the coset loop reuses it.
    let span = Gf2Span::new(&plan.local_bucket_deltas, local.hash().bits());
    let mut order = std::mem::take(&mut state.layer.order);
    order.rebuild(&span, true);
    state.chunks = ChunkMap::of_order(&order, *EXCHANGE_CHUNKS.get());
    let (send, mut counts) =
        export_layer(local, &state.layer.octaves, prepared, plan, size, &order, &mut state.export);
    let export_ns = export_start.map_or(0, |start| start.elapsed().as_nanos() as u64);
    tally!(state.layer.stats.export_ns, export_ns);
    tally!(state.layer.stats.rows_exported, counts.rows_sent.iter().sum::<u64>());
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
    // A rotation's identity entry is always local, so on a remote layer its generator crosses: it stays whole and `generator_local` switches that pass off.
    let rotation_table = matches!(prepared, Prepared::Local(ptm) if is_rotation_table(ptm));
    let retained;
    let local_prepared: &Prepared<W, C> = match prepared {
        Prepared::Local(ptm) if !rotation_table => {
            retained = Prepared::Local(ptm.retain_entries(&plan.local_entries));
            &retained
        }
        Prepared::Local(_) | Prepared::Rotation(_) => prepared,
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
            Prepared::Local(_) => !rotation_table,
        },
        order: Some(&order),
    };

    let PartitionState { layer: layer_scratch, export: export_scratch, chunks: map } = state;
    let body_ns = std::cell::Cell::new(0u64);
    let exchange_start = measured.then(Instant::now);

    // The rows may still be in flight when the body runs; each coset task waits for its own chunk in `append_into`.
    let (received, rows_received) =
        transport.exchange_layer(send, &mut export_scratch.pool, map, |received, wait| {
            let body_start = measured.then(Instant::now);
            #[cfg(debug_assertions)]
            for block in received.iter().flatten().flat_map(|payload| &payload.blocks) {
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
                ahead,
            );
            timed! {
                layer_scratch.stats.recv_rows += rows_received;
                layer_scratch.stats.append_ns += received_rows
                    .append_ns
                    .load(std::sync::atomic::Ordering::Relaxed);
                layer_scratch.stats.chunk_wait_ns += received_rows
                    .chunk_wait_ns
                    .load(std::sync::atomic::Ordering::Relaxed);
            }
            if let Some(start) = body_start {
                body_ns.set(start.elapsed().as_nanos() as u64);
            }
            rows_received
        });
    // Excludes the body it wraps: the transfer the coset loop did not hide.
    let exchange_ns = exchange_start
        .map_or(0, |start| (start.elapsed().as_nanos() as u64).saturating_sub(body_ns.get()));
    tally!(layer_scratch.stats.exchange_ns, exchange_ns);
    export_scratch.pool.extend(received.into_iter().flatten());
    layer_scratch.order = order;

    counts.remote_deltas = plan.remote.len();
    counts.rows_received = rows_received;
    counts.receive_chunks = map.chunks() as u32;
    if clocked {
        (counts.export_ns, counts.exchange_ns) = (export_ns, exchange_ns);
    }
    counts
}

#[cfg(test)]
mod tests;
