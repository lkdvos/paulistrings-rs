//! One layer of the partitioned engine: export -> exchange -> coset loop.
//!
//! A partition holds only the terms with `rows.partition_of(v) == rank`.
//! [`export_layer`] builds one exchange block per remote delta, [`Transport::exchange_layer`] runs the all-to-all, and the bucketed coset loop merges local deltas with received rows via [`ExtraRows`] before `keep_term` runs (ARCHITECTURE.md §Truncation).
//! Whether a delta is remote depends only on its mask, so every partition reaches the same verdict on whether to exchange.
//! This function does not rebucket and does not call [`TruncationPolicy::finalize_layer`]; the driver owns both.

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

/// Chunks a layer's bulk transfer is cut into, by default.
///
/// The receiver consumes chunk `k` as soon as it lands, so the pipeline depth is the chunk count, traded off against the per-message MPI overhead of a small chunk.
/// Not derived from the thread count: both sides of an exchange must cut the same block the same way, and two ranks need not have equal-sized pools.
pub(crate) const DEFAULT_EXCHANGE_CHUNKS: usize = 8;

/// The chunk count every partition cuts this layer's transfer into.
///
/// [`DEFAULT_EXCHANGE_CHUNKS`], unless `PAULISTRINGS_EXCHANGE_CHUNKS` names another (`1` is the un-pipelined layout).
/// Read once per process, so every rank in a group launched with the same environment agrees, which it must since both sides cut the same block the same way.
pub(crate) fn exchange_chunks() -> usize {
    static CHUNKS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CHUNKS.get_or_init(|| {
        std::env::var("PAULISTRINGS_EXCHANGE_CHUNKS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&k| k > 0)
            .unwrap_or(DEFAULT_EXCHANGE_CHUNKS)
    })
}

/// The rows this partition received, as an [`ExtraRows`] source for the coset loop.
///
/// One entry per remote delta, in ascending remote-delta index.
/// Output bucket `β′` reads `segment(map.position_of(β′))` of each, per the receive rule in [`transport`](super::transport)'s module docs.
/// Received rows always join the **rest** stream: [`NEEDS_BETA`](ExtraRows::NEEDS_BETA) is `true`.
pub(crate) struct RecvRows<'a, const W: usize> {
    /// The block per remote delta, ascending by entry.
    blocks: Vec<Option<&'a ExchangeBlock<W>>>,
    /// The destination-coset order the blocks are laid out in.
    map: &'a ChunkMap,
    /// What a coset task blocks on before it reads a chunk's rows; a no-op under a blocking transport.
    wait: &'a dyn ChunkWait,
    /// Nanoseconds spent in [`append_into`](ExtraRows::append_into), summed across coset tasks. Measurement only.
    #[cfg(feature = "phase-timing")]
    append_ns: std::sync::atomic::AtomicU64,
    /// The part of [`append_ns`](Self::append_ns) spent blocked in [`ChunkWait::wait_chunk`]. Measurement only.
    #[cfg(feature = "phase-timing")]
    chunk_wait_ns: std::sync::atomic::AtomicU64,
}

impl<'a, const W: usize> RecvRows<'a, W> {
    /// Pair each of `plan`'s remote deltas with the block that carries it.
    ///
    /// A delta is classified from its mask, so partner `q`'s remote deltas addressed here are the same entries, in the same order, as this partition's remote deltas addressed to `q`: the `j`-th of `remote_for_partner(q)` is the `j`-th block of `recv[q]`.
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
        let t0 = std::time::Instant::now();
        let p = self.map.position_of(beta);
        // Every member of a coset is in one chunk (`ChunkMap`), so this is one wait per task, not one per member.
        self.wait.wait_chunk(self.map.chunk_of_position(p));
        #[cfg(feature = "phase-timing")]
        self.chunk_wait_ns.fetch_add(
            t0.elapsed().as_nanos() as u64,
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
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// One partition's reusable per-layer scratch: the coset loop's and the export pass's, held together so the driver carries one value per partition.
#[derive(Debug, Default)]
pub(crate) struct PartitionState<const W: usize> {
    /// The bucketed engine's layer scratch.
    pub layer: LayerScratch<W>,
    /// The export pass's count buffers and its pool of exchange payloads.
    pub export: ExportScratch<W>,
    /// The destination-coset order this layer's blocks are laid out in, and the chunks its bulk transfer is cut into.
    pub chunks: ChunkMap,
}

/// What one layer's exchange moved, from this partition's point of view.
///
/// `rows_sent` and `bytes_sent` are indexed by partner rank; `rows_received` is a total, since a received row's provenance stops mattering once merged.
/// A layer with no remote delta reports all zeros and issues no transport call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LayerExchangeCounts {
    /// Remote deltas the layer had, i.e. blocks sent per partner-delta pair.
    pub remote_deltas: usize,
    /// Rows sent to each partner rank.
    pub rows_sent: Vec<u64>,
    /// Wire bytes sent to each partner rank.
    pub bytes_sent: Vec<u64>,
    /// Rows received from all partners together.
    pub rows_received: u64,
}

impl LayerExchangeCounts {
    /// The counts of a layer that exchanged nothing.
    pub(crate) fn none(size: u32) -> Self {
        Self {
            remote_deltas: 0,
            rows_sent: vec![0; size as usize],
            bytes_sent: vec![0; size as usize],
            rows_received: 0,
        }
    }
}

/// Apply one prepared channel to this partition's share of a sum.
///
/// `local` must hold exactly the terms with `rows.partition_of(v) == transport.rank()`, under a hash and bucket count every partition agrees on; it comes back holding this partition's share of the layer's output, merged, deduplicated and filtered through `policy`'s `keep_term`.
/// Neither rebuckets nor calls `finalize_layer`: both are collective decisions the driver makes with the counts this returns.
/// Classifies `prep`'s deltas itself; the driver needs that classification before it decides whether the layer takes a collective, so it holds the plan and calls [`apply_layer_partitioned_with_plan`] instead.
#[cfg(test)]
pub(crate) fn apply_layer_partitioned<const W: usize, T, X>(
    local: &mut PauliSum<W>,
    prep: &Prepared<W>,
    rows: &PartitionRows<W>,
    policy: &T,
    state: &mut PartitionState<W>,
    transport: &X,
) -> LayerExchangeCounts
where
    T: TruncationPolicy<W> + ?Sized,
    X: Transport,
{
    let plan = PartitionPlan::new(prep, rows, transport.rank());
    apply_layer_partitioned_with_plan(local, prep, &plan, rows, policy, state, transport)
}

/// [`apply_layer_partitioned`] with the delta classification already made.
///
/// `plan` must be `PartitionPlan::new(prep, rows, transport.rank())`; the driver builds it a step earlier, because `has_remote()` is what decides whether the layer agrees the bucket count with the group (ARCHITECTURE.md §Partitioning).
pub(crate) fn apply_layer_partitioned_with_plan<const W: usize, T, X>(
    local: &mut PauliSum<W>,
    prep: &Prepared<W>,
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

    // Nothing crosses: the ordinary engine, and — crucially — *no* transport
    // call. Every partition took this branch, because `part(d)` is a function
    // of the delta alone.
    if !plan.has_remote() {
        apply_layer_bucketed(local, prep, policy, &mut state.layer);
        return LayerExchangeCounts::none(size);
    }

    #[cfg(feature = "phase-timing")]
    let mut st = crate::engine::stats::Stamp::now();
    // Both sides lay the blocks out in the *receiver's* coset order, and every
    // partition derives it from the same local bucket deltas and the same
    // collectively agreed bucket count — so the sender can permute without a
    // word of negotiation. `apply_layer_bucketed_with` rebuilds the identical
    // span below from the same two inputs.
    let span = Gf2Span::new(&plan.local_bucket_deltas, local.hash().bits());
    state
        .chunks
        .rebuild(&span, local.num_buckets(), exchange_chunks());
    let (send, export) = export_layer(local, prep, plan, size, &state.chunks, &mut state.export);
    #[cfg(feature = "phase-timing")]
    {
        st.lap(&mut state.layer.stats.export_ns);
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
    // The local delta table, the local bucket deltas, the channel's total stream count, and — for a wide rotation — whether the generator pass emits here at all.
    let retained;
    let local_prep: &Prepared<W> = match prep {
        Prepared::Local(ptm) => {
            retained = Prepared::Local(ptm.retain_entries(&plan.local_entries));
            &retained
        }
        // The identity entry of a rotation is always local, so `has_remote()` means the generator crosses; `gen_local` switches its pass off instead.
        Prepared::Rotation(_) => prep,
    };
    let knobs = LayerKnobs {
        bucket_deltas: Some(&plan.local_bucket_deltas),
        rest_streams: Some(plan.rest_streams_total),
        // From `prep`, not `local_prep`: asking the restricted PTM could put one partition on a different sort kernel than the unpartitioned run — see `LayerKnobs::rows_per_key`.
        rows_per_key: match prep {
            Prepared::Local(ptm) => Some(rest_rows_per_key(ptm)),
            Prepared::Rotation(_) => None,
        },
        gen_local: match prep {
            // Entry 1 is the generator pass (`plan`'s numbering).
            Prepared::Rotation(_) => plan.local_entries[1],
            // Meaningless for a tabulated channel.
            Prepared::Local(_) => true,
        },
    };

    // Split the scratch so the exchange can borrow the payload pool while the coset loop inside it borrows the layer scratch and the chunk map.
    let PartitionState {
        layer: layer_scratch,
        export: export_scratch,
        chunks: map,
    } = state;
    #[cfg(feature = "phase-timing")]
    let body_ns = std::cell::Cell::new(0u64);
    #[cfg(feature = "phase-timing")]
    let exchange_start = std::time::Instant::now();

    // `exchange_layer` returns once the block headers and CSR offsets are here; the rows themselves may still be in flight, and each coset task waits for its own chunk at the top of `append_into`.
    let (recv, rows_received) =
        transport.exchange_layer(send, &mut export_scratch.pool, map, |recv, wait| {
            #[cfg(feature = "phase-timing")]
            let body_start = std::time::Instant::now();
            #[cfg(debug_assertions)]
            for block in recv.iter().flatten().flat_map(|payload| &payload.blocks) {
                // The bucket count is a collective decision the driver makes before the layer; a block indexed by a different one would be read at the wrong offsets.
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
            apply_layer_bucketed_with(local, local_prep, policy, layer_scratch, &recv_rows, knobs);
            #[cfg(feature = "phase-timing")]
            {
                // The coset loop has joined, so the counters are quiescent.
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
        // `exchange_ns` excludes the layer work it wraps, so it isolates whatever transfer the coset loop did not manage to hide before the closing wait.
        layer_scratch.stats.exchange_ns +=
            exchange_start.elapsed().as_nanos() as u64 - body_ns.get();
        st.rearm();
    }
    // The payloads that carried the received rows go back into the pool with their columns intact, for the next export or receive to reuse.
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
