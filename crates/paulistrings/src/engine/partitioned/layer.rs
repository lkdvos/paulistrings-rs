//! One layer of the partitioned engine: export → exchange → coset loop.
//!
//! A partition holds the terms with `rows.partition_of(v) == rank` and nothing
//! else. A prepared channel's deltas split into *local* ones, whose rows stay
//! in this partition, and *remote* ones, whose rows all belong to one partner
//! ([`PartitionPlan`]). So a layer is:
//!
//! 1. [`export_layer`] — build one exchange block per remote delta.
//! 2. [`Transport::exchange_layer`] — one all-to-all; every partition issues
//!    it, or none does (see below).
//! 3. The ordinary bucketed coset loop over the *local* deltas only, with the
//!    received rows fed into each output bucket's **rest** stream through
//!    [`ExtraRows`] — so the merge sees a key's complete sum, local plus
//!    received, before `keep_term` runs (ARCHITECTURE.md §Truncation).
//!
//! Whether a delta is remote depends on its mask alone, never on local data,
//! so every partition reaches the same verdict: either all of them exchange or
//! none of them does, and the collective-order invariant in
//! [`transport`](super::transport) holds without a vote.
//!
//! This function does not rebucket and does not call
//! [`TruncationPolicy::finalize_layer`] — the driver owns both, as it does in
//! the unpartitioned engine.

use super::export::{export_layer, ExportScratch};
use super::plan::PartitionPlan;
use super::transport::{ChunkMap, ChunkWait, ExchangeBlock, PartnerPayload, Transport};
use crate::bucket::hash::PartitionRows;
use crate::bucket::sum::PauliSum;
use crate::channel::prepared::Prepared;
use crate::engine::bucketed::{
    apply_layer_bucketed, apply_layer_bucketed_with, ExtraRows, LayerKnobs, LayerScratch,
};
use crate::engine::coset::Gf2Span;
use crate::truncation::TruncationPolicy;
use num_complex::Complex64;

/// Chunks a layer's bulk transfer is cut into, by default.
///
/// The receiver consumes chunk `k` as soon as it lands, so the pipeline is
/// only as deep as the chunk count — but every chunk is three MPI messages per
/// block per partner, and a chunk under a few megabytes stops amortizing the
/// per-message cost. Eight is deep enough that the last chunk's coset work is a
/// small tail behind the transfer and coarse enough that a 384 MB layer still
/// moves 48 MB per message.
///
/// It is **not** derived from the thread count: both sides of an exchange must
/// cut the same block the same way, and two ranks need not have equal-sized
/// pools.
pub(crate) const DEFAULT_EXCHANGE_CHUNKS: usize = 8;

/// The chunk count every partition cuts this layer's transfer into.
///
/// [`DEFAULT_EXCHANGE_CHUNKS`], unless the environment names another.
///
/// `PAULISTRINGS_EXCHANGE_CHUNKS` is **the** knob, and this is the only place
/// that reads it: the sweep that chose the default turned it, and
/// `scripts/mpi-test.sh` runs the whole MPI net a second time at 64 to force
/// many small batches. `1` is the un-pipelined layout. Read once per process,
/// so a group launched by one `mpirun -x PAULISTRINGS_EXCHANGE_CHUNKS=...`
/// agrees — which it must, both sides cutting the same block the same way.
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

/// The rows this partition received, as an [`ExtraRows`] source for the coset
/// loop.
///
/// One entry per remote delta, in ascending remote-delta index. Output bucket
/// `β′` reads `segment(map.position_of(β′))` of each — the receive rule from
/// [`transport`](super::transport)'s module docs. The sender laid its CSR out
/// in this partition's destination-coset order, so the lookup is a table read
/// and a coset's rows are contiguous in the block.
///
/// Received rows always join the **rest** stream:
/// [`NEEDS_BETA`](ExtraRows::NEEDS_BETA) is `true`, so the engine names each
/// coset member by its original bucket index.
pub(crate) struct RecvRows<'a, const W: usize> {
    /// The block per remote delta, ascending by entry.
    blocks: Vec<Option<&'a ExchangeBlock<W>>>,
    /// The destination-coset order the blocks are laid out in.
    map: &'a ChunkMap,
    /// What a coset task blocks on before it reads a chunk's rows. A blocking
    /// transport's is a no-op; a pipelined one's completes the receive.
    wait: &'a dyn ChunkWait,
    /// Nanoseconds spent in [`append_into`](ExtraRows::append_into), summed
    /// over every coset task that ran one. Measurement only; the coset loop is
    /// parallel, so the counter is an atomic, and it is read once after the
    /// loop has joined.
    #[cfg(feature = "phase-timing")]
    append_ns: std::sync::atomic::AtomicU64,
    /// The part of [`append_ns`](Self::append_ns) spent blocked in
    /// [`ChunkWait::wait_chunk`] — how much of the transfer the coset loop
    /// failed to hide. Measurement only.
    #[cfg(feature = "phase-timing")]
    chunk_wait_ns: std::sync::atomic::AtomicU64,
}

impl<'a, const W: usize> RecvRows<'a, W> {
    /// Pair each of `plan`'s remote deltas with the block that carries it.
    ///
    /// A delta is classified from its mask, which does not depend on the rank,
    /// so partner `q`'s remote deltas addressed here are the same entries, in
    /// the same order, as this partition's remote deltas addressed to `q`: the
    /// `j`-th of `remote_for_partner(q)` is the `j`-th block of `recv[q]`.
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
        // The rows for this coset may still be on the wire. Every member of a
        // coset is in one chunk (`ChunkMap`), so this is one wait per task, not
        // one per member — and under a blocking transport it is a no-op call
        // the optimizer keeps but the branch predictor never notices.
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

/// One partition's reusable per-layer scratch: the coset loop's and the export
/// pass's, held together so the driver carries one value per partition.
#[derive(Debug, Default)]
pub(crate) struct PartitionState<const W: usize> {
    /// The bucketed engine's layer scratch.
    pub layer: LayerScratch<W>,
    /// The export pass's count buffers and its pool of exchange payloads.
    pub export: ExportScratch<W>,
    /// The destination-coset order this layer's blocks are laid out in, and
    /// the chunks its bulk transfer is cut into. Rebuilt per remote layer,
    /// keeping its permutation buffers.
    pub chunks: ChunkMap,
}

/// What one layer's exchange moved, from this partition's point of view.
///
/// `rows_sent` and `bytes_sent` are indexed by partner rank (length is the
/// group size); `rows_received` is a total, since a received row's provenance
/// stops mattering the moment it is merged. A layer with no remote delta
/// reports all zeros and issues no transport call at all.
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
    fn none(size: u32) -> Self {
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
/// `local` must hold exactly the terms with `rows.partition_of(v) ==
/// transport.rank()`, under a hash and bucket count every partition agrees on;
/// it comes back holding this partition's share of the layer's output, merged,
/// deduplicated and filtered through `policy`'s `keep_term`.
///
/// Neither rebuckets nor calls `finalize_layer`: both are collective decisions
/// the driver makes with the counts this returns.
///
/// Classifies `prep`'s deltas itself. The driver needs the classification
/// *before* it decides whether the layer takes a collective, so it holds the
/// plan and calls [`apply_layer_partitioned_with_plan`] instead.
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
/// `plan` must be `PartitionPlan::new(prep, rows, transport.rank())` — the
/// driver builds it a step earlier, because `has_remote()` is what decides
/// whether the layer agrees the bucket count with the group (ARCHITECTURE.md
/// §Partitioning).
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
        state.layer.stats.export_count_ns += std::mem::take(&mut state.export.count_ns);
        state.layer.stats.export_fill_ns += std::mem::take(&mut state.export.fill_ns);
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
    // The layer as this partition sees it: the local delta table (a tabulated
    // channel keeps only its local entries, so the gather loops stay a plain
    // `deltas()` walk with no per-entry predicate), the local bucket deltas
    // for the coset span, the channel's *total* stream count for the sort
    // kernel, and — for a wide rotation — whether the generator pass emits
    // here at all.
    let retained;
    let local_prep: &Prepared<W> = match prep {
        Prepared::Local(ptm) => {
            retained = Prepared::Local(ptm.retain_entries(&plan.local_entries));
            &retained
        }
        // A rotation has two entries and the identity one is always local, so
        // `has_remote()` means the generator crosses and the prepared form is
        // used unchanged, with `gen_local` switching its pass off.
        Prepared::Rotation(_) => prep,
    };
    let knobs = LayerKnobs {
        bucket_deltas: Some(&plan.local_bucket_deltas),
        rest_streams: Some(plan.rest_streams_total),
        gen_local: match prep {
            // Entry 1 is the generator pass (`plan`'s numbering).
            Prepared::Rotation(_) => plan.local_entries[1],
            // Meaningless for a tabulated channel; the default keeps it out
            // of the way.
            Prepared::Local(_) => true,
        },
    };

    // Split the scratch so the exchange can borrow the payload pool while the
    // coset loop inside it borrows the layer scratch and the chunk map.
    let PartitionState {
        layer: layer_scratch,
        export: export_scratch,
        chunks: map,
    } = state;
    #[cfg(feature = "phase-timing")]
    let body_ns = std::cell::Cell::new(0u64);
    #[cfg(feature = "phase-timing")]
    let exchange_start = std::time::Instant::now();

    // Two phases. `exchange_layer` returns once the block headers and CSR
    // offsets are here — everything `RecvRows::count` needs to size a run —
    // and the rows themselves may still be in flight; each coset task waits for
    // its own chunk at the top of `append_into`.
    let (recv, rows_received) =
        transport.exchange_layer(send, &mut export_scratch.pool, map, |recv, wait| {
            #[cfg(feature = "phase-timing")]
            let body_start = std::time::Instant::now();
            #[cfg(debug_assertions)]
            for block in recv.iter().flatten().flat_map(|payload| &payload.blocks) {
                // The bucket count is a *collective* decision the driver makes
                // before the layer; a block indexed by a different one would be
                // read at the wrong offsets. Without this the failure mode is a
                // garbage segment length — a wild allocation or a silently
                // wrong answer — rather than an assertion naming the cause.
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
        // `exchange_ns` is the exchange minus the layer work it now wraps: the
        // send posting, the early receives, and whatever transfer the coset
        // loop did not manage to hide before the closing wait.
        layer_scratch.stats.exchange_ns +=
            exchange_start.elapsed().as_nanos() as u64 - body_ns.get();
        transport.drain_timings(&mut layer_scratch.stats);
        st.rearm();
    }
    // The received rows are merged; the payloads that carried them go back into
    // the pool with their columns intact, and the next layer's export — or the
    // next receive — takes them from there rather than from the allocator.
    export_scratch.pool.extend(recv.into_iter().flatten());

    LayerExchangeCounts {
        remote_deltas: plan.remote.len(),
        rows_sent: export.rows_to,
        bytes_sent: export.bytes_to,
        rows_received,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulator::BuildAccumulator;
    use crate::bucket::hash::Gf2Hash;
    use crate::channel::clifford::Clifford1Q;
    use crate::channel::rotation::PauliRotation;
    use crate::channel::Channel;
    use crate::engine::partitioned::transport::{Collectives, InProcessTransport, Payload};
    use crate::pauli_string::PauliString;
    use crate::phase::Phase;
    use crate::test_support::{
        assert_terms_close, differential_channels_w1, differential_channels_w2, naive_apply_layer,
        rand_sum, rand_sum_real,
    };
    use crate::truncation::builtin::CoefficientThreshold;
    use num_complex::Complex64;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const TOL: f64 = 1e-11;
    const ZERO: Complex64 = Complex64::new(0.0, 0.0);

    struct AlwaysKeep;
    impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

    /// A [`ChunkWait`] that records what it was asked for.
    struct WaitSpy(std::sync::Mutex<Vec<usize>>);

    impl super::ChunkWait for WaitSpy {
        fn wait_chunk(&self, k: usize) {
            self.0.lock().expect("spy").push(k);
        }
    }

    /// `count` and `append_into` read the block at the *destination position*,
    /// and each waits for exactly the chunk that position falls in.
    ///
    /// This is the whole receive-side contract of the pipelined exchange: get
    /// the position wrong and rows land in the wrong bucket; get the chunk
    /// wrong and a task reads a column MPI is still writing. The oracle is
    /// hand-built — one row per position, carrying that position's index — so a
    /// misread is visible as a wrong number rather than as a wrong sum.
    #[test]
    fn received_rows_are_read_by_position_and_wait_for_their_own_chunk() {
        const BITS: u8 = 5;
        let n = 1u32 << BITS;
        for deltas in [vec![0u32], vec![0, 4], vec![0, 1, 2, 3]] {
            let span = crate::engine::coset::Gf2Span::new(&deltas, BITS);
            for chunks in [1usize, 2, 4, 8, 64] {
                let mut map = ChunkMap::default();
                map.rebuild(&span, n as usize, chunks);

                // One row per position, coefficient = the position index.
                let counts: Vec<u32> = vec![1; n as usize];
                let mut block = ExchangeBlock::<1>::with_counts(1, &counts);
                block.x = (0..n).map(|p| [u64::from(p)]).collect();
                block.z = (0..n).map(|_| [0u64]).collect();
                block.coeff = (0..n).map(|p| Complex64::new(f64::from(p), 0.0)).collect();

                let payload = PartnerPayload::<1> {
                    blocks: vec![block],
                };
                let recv = vec![None, Some(payload)];
                let mut plan_deltas = deltas.clone();
                plan_deltas.sort_unstable();
                let plan = PartitionPlan {
                    local_entries: vec![true, false],
                    local_bucket_deltas: plan_deltas,
                    remote: vec![super::super::plan::RemoteDelta {
                        entry: 1,
                        partner: 1,
                        bucket_delta: 0,
                        partition_delta: 1,
                    }],
                    rest_streams_total: 1,
                };

                let spy = WaitSpy(std::sync::Mutex::new(Vec::new()));
                let rows = RecvRows::new(&plan, &recv, &map, &spy);
                for beta in 0..n {
                    let p = map.position_of(beta);
                    assert_eq!(rows.count(beta), 1, "one row per bucket");
                    let (mut x, mut z, mut c) = (Vec::new(), Vec::new(), Vec::new());
                    rows.append_into(beta, &mut x, &mut z, &mut c);
                    assert_eq!(x, vec![[u64::from(p)]], "bucket {beta} read position {p}");
                    assert_eq!(z, vec![[0u64]]);
                    assert_eq!(c, vec![Complex64::new(f64::from(p), 0.0)]);
                }
                let waited = spy.0.into_inner().expect("spy");
                assert_eq!(waited.len(), n as usize, "one wait per append_into");
                for (beta, &k) in waited.iter().enumerate() {
                    assert_eq!(
                        k,
                        map.chunk_of_position(map.position_of(beta as u32)),
                        "bucket {beta} waited for the wrong chunk at {chunks} chunks",
                    );
                    assert!(k < map.chunks(), "chunk {k} is outside the map");
                }
            }
        }
    }

    /// Run one layer on every partition of `rows`, each on its own thread with
    /// its own transport, and give back the parts in rank order with their
    /// exchange counts.
    ///
    /// `threads`, when given, runs each partition inside a Rayon pool of that
    /// size — the knob the byte-identity test turns.
    ///
    /// Every part is checked on the way out: the bucketed invariants hold, and
    /// it holds keys of its own partition only (`None` for an empty part,
    /// which is legal and is what the `P = 4` fixtures produce).
    fn run_parts<const W: usize, T>(
        whole: &PauliSum<W>,
        prep: &Prepared<W>,
        rows: &PartitionRows<W>,
        policy: &T,
        threads: Option<usize>,
    ) -> (Vec<PauliSum<W>>, Vec<LayerExchangeCounts>)
    where
        T: TruncationPolicy<W> + ?Sized,
    {
        let size = rows.num_partitions() as u32;
        let transports = InProcessTransport::group(size);
        let parts: Vec<PauliSum<W>> = (0..size).map(|r| whole.filter_partition(rows, r)).collect();
        let results: Vec<(PauliSum<W>, LayerExchangeCounts)> = std::thread::scope(|scope| {
            let handles: Vec<_> = parts
                .into_iter()
                .zip(transports)
                .map(|(mut local, transport)| {
                    scope.spawn(move || {
                        let mut state = PartitionState::<W>::default();
                        let mut run = || {
                            apply_layer_partitioned(
                                &mut local, prep, rows, policy, &mut state, &transport,
                            )
                        };
                        let counts = match threads {
                            Some(n) => rayon::ThreadPoolBuilder::new()
                                .num_threads(n)
                                .build()
                                .expect("pool")
                                .install(run),
                            None => run(),
                        };
                        (local, counts)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("a partition panicked"))
                .collect()
        });

        let mut sums = Vec::with_capacity(results.len());
        let mut counts = Vec::with_capacity(results.len());
        for (rank, (part, count)) in results.into_iter().enumerate() {
            part.assert_invariants();
            let held = part.partition_rank_of_all(rows);
            assert!(
                held.is_none() || held == Some(rank as u32),
                "part {rank} holds keys of partition {held:?}",
            );
            sums.push(part);
            counts.push(count);
        }
        (sums, counts)
    }

    /// The unpartitioned layer on the whole sum, for the same hash.
    fn unsplit_layer<const W: usize, T>(
        whole: &PauliSum<W>,
        prep: &Prepared<W>,
        policy: &T,
    ) -> PauliSum<W>
    where
        T: TruncationPolicy<W> + ?Sized,
    {
        let mut sum = whole.clone();
        let mut scratch = LayerScratch::<W>::new();
        apply_layer_bucketed(&mut sum, prep, policy, &mut scratch);
        sum
    }

    /// `Z₀X₂X₄X₆` — weight 4, so `prepare` gives the `Prepared::Rotation` arm.
    fn wide_gen() -> PauliString<1> {
        let mut g = PauliString::<1>::z(0);
        for q in [2u32, 4, 6] {
            g.mul_assign(&PauliString::<1>::x(q));
        }
        g
    }

    /// A partitioning that sees exactly the `x` bit of qubit 2, which
    /// [`wide_gen`] carries: `part(gen) = 1`, so the generator pass is remote
    /// at every rank.
    fn rows_seeing_qubit_2_x() -> PartitionRows<1> {
        PartitionRows::<1>::from_rows(8, vec![[1u64 << 2]], vec![[0u64]])
    }

    /// The whole differential net, split every way: merged output against the
    /// naive oracle *and* against the unpartitioned engine on the same hash.
    ///
    /// This is the primary correctness net for the partitioned layer. It
    /// covers both prepared arms, both directions, four bucket counts, one/two/
    /// four partitions and two independent partition-row draws — so a delta
    /// that crosses in one draw stays local in another, and the export,
    /// exchange and receive paths are exercised against the same oracle the
    /// unpartitioned engine answers to.
    #[test]
    fn partitioned_layer_matches_naive_oracle_w1() {
        let input = rand_sum::<1>(600, 8, 0x9D0);
        for (name, ch) in &differential_channels_w1() {
            let cr: &dyn Channel<1> = ch.as_ref();
            for &adjoint in &[false, true] {
                let want = naive_apply_layer(&input, cr, &AlwaysKeep, adjoint);
                for &bits in &[0u8, 1, 3, 6] {
                    let hash = Gf2Hash::<1>::new(8, bits, 0xABCD);
                    let whole = input.clone().with_hash(hash);
                    let prep = cr.prepare(whole.hash(), adjoint).expect("prepare");
                    let unsplit = unsplit_layer(&whole, &prep, &AlwaysKeep);
                    for &pbits in &[0u8, 1, 2] {
                        for &pseed in &[0x1357u64, 0x2468] {
                            let rows = PartitionRows::<1>::from_seed(8, pbits, pseed);
                            let (parts, _) = run_parts(&whole, &prep, &rows, &AlwaysKeep, None);
                            let got = PauliSum::merge_partitions(parts);
                            let what = format!(
                                "{name} adjoint={adjoint} bits={bits} P={} pseed={pseed:x}",
                                rows.num_partitions(),
                            );
                            assert_terms_close(&got, &want, TOL, &what);
                            assert_terms_close(&got, &unsplit, TOL, &format!("{what} vs unsplit"));
                        }
                    }
                }
            }
        }
    }

    /// The same at `W = 2`: wide keys, supports straddling the word boundary.
    #[test]
    fn partitioned_layer_matches_naive_oracle_w2() {
        let input = rand_sum_real::<2>(700, 128, 0x9D1);
        for (name, ch) in &differential_channels_w2() {
            let cr: &dyn Channel<2> = ch.as_ref();
            for &adjoint in &[false, true] {
                let want = naive_apply_layer(&input, cr, &AlwaysKeep, adjoint);
                for &bits in &[2u8, 5] {
                    let hash = Gf2Hash::<2>::new(128, bits, 0xABCD);
                    let whole = input.clone().with_hash(hash);
                    let prep = cr.prepare(whole.hash(), adjoint).expect("prepare");
                    let unsplit = unsplit_layer(&whole, &prep, &AlwaysKeep);
                    for &pbits in &[0u8, 1, 2] {
                        for &pseed in &[0x1357u64, 0x2468] {
                            let rows = PartitionRows::<2>::from_seed(128, pbits, pseed);
                            let (parts, _) = run_parts(&whole, &prep, &rows, &AlwaysKeep, None);
                            let got = PauliSum::merge_partitions(parts);
                            let what = format!(
                                "{name} adjoint={adjoint} bits={bits} P={} pseed={pseed:x}",
                                rows.num_partitions(),
                            );
                            assert_terms_close(&got, &want, TOL, &what);
                            assert_terms_close(&got, &unsplit, TOL, &format!("{what} vs unsplit"));
                        }
                    }
                }
            }
        }
    }

    /// A transport that counts the exchanges it is asked for.
    struct CountingTransport {
        inner: InProcessTransport,
        exchanges: AtomicUsize,
    }

    impl Collectives for CountingTransport {
        fn rank(&self) -> u32 {
            self.inner.rank()
        }
        fn size(&self) -> u32 {
            self.inner.size()
        }
        fn allreduce_max_u8(&self, v: u8) -> u8 {
            self.inner.allreduce_max_u8(v)
        }
        fn allreduce_sum_u64(&self, buf: &mut [u64]) {
            self.inner.allreduce_sum_u64(buf)
        }
        fn barrier(&self) {
            self.inner.barrier()
        }
    }

    impl Transport for CountingTransport {
        fn exchange_layer<P, F, R>(
            &self,
            send: Vec<Option<P>>,
            spare: &mut Vec<P>,
            map: &ChunkMap,
            body: F,
        ) -> (Vec<Option<P>>, R)
        where
            P: Payload,
            F: FnOnce(&[Option<P>], &dyn ChunkWait) -> R,
        {
            self.exchanges.fetch_add(1, Ordering::Relaxed);
            self.inner.exchange_layer(send, spare, map, body)
        }
    }

    /// A layer whose every delta stays inside its partition issues **no**
    /// transport call — not an empty one.
    ///
    /// That is what lets a partitioned run skip the collective entirely on the
    /// many layers that do not cross: the verdict comes from the plan, which
    /// every partition computes identically, so no vote is needed.
    #[test]
    fn no_remote_deltas_means_no_exchange() {
        let input = rand_sum::<1>(300, 8, 0x9D2);
        // `h(3)`'s only non-identity delta is the mask `x₃ z₃`; a partition
        // row reading qubit 0 alone cannot see it.
        let rows = PartitionRows::<1>::from_rows(8, vec![[1u64]], vec![[0u64]]);
        let h = Clifford1Q::h(3);
        let hash = Gf2Hash::<1>::new(8, 4, 0x9D);
        let whole = input.clone().with_hash(hash);
        let prep = Channel::<1>::prepare(&h, whole.hash(), false).expect("prepare");
        assert!(
            !PartitionPlan::new(&prep, &rows, 0).has_remote(),
            "fixture must have no remote delta",
        );

        let transports: Vec<CountingTransport> = InProcessTransport::group(2)
            .into_iter()
            .map(|inner| CountingTransport {
                inner,
                exchanges: AtomicUsize::new(0),
            })
            .collect();
        let parts: Vec<PauliSum<1>> = (0..2).map(|r| whole.filter_partition(&rows, r)).collect();
        let prep = &prep;
        let rows = &rows;
        let results: Vec<(PauliSum<1>, LayerExchangeCounts, usize)> = std::thread::scope(|scope| {
            let handles: Vec<_> = parts
                .into_iter()
                .zip(transports)
                .map(|(mut local, transport)| {
                    scope.spawn(move || {
                        let mut state = PartitionState::<1>::default();
                        let counts = apply_layer_partitioned(
                            &mut local,
                            prep,
                            rows,
                            &AlwaysKeep,
                            &mut state,
                            &transport,
                        );
                        let seen = transport.exchanges.load(Ordering::Relaxed);
                        (local, counts, seen)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut got_parts = Vec::new();
        for (part, counts, exchanges) in results {
            assert_eq!(exchanges, 0, "a local-only layer called the transport");
            assert_eq!(counts, LayerExchangeCounts::none(2));
            got_parts.push(part);
        }
        let got = PauliSum::merge_partitions(got_parts);
        let want = naive_apply_layer(&input, &h, &AlwaysKeep, false);
        assert_terms_close(&got, &want, TOL, "local-only layer");
    }

    /// A wide rotation whose generator crosses: every generator row is
    /// exported, and the exported total is exactly the number of terms that
    /// anticommute with the generator.
    #[test]
    fn all_remote_rotation_generator() {
        let gen = wide_gen();
        let rot = PauliRotation::new(gen, 0.41);
        let rows = rows_seeing_qubit_2_x();
        assert_eq!(rows.partition_of(&gen.x, &gen.z), 1);
        let input = rand_sum::<1>(500, 8, 0x9D3);
        let hash = Gf2Hash::<1>::new(8, 3, 0x9E);
        let whole = input.clone().with_hash(hash);
        let prep = Channel::<1>::prepare(&rot, whole.hash(), false).expect("prepare");
        let plan = PartitionPlan::new(&prep, &rows, 0);
        assert!(plan.has_remote(), "the generator must be remote");
        assert!(!plan.local_entries[1], "entry 1 is the generator pass");
        assert_eq!(plan.local_bucket_deltas, vec![0], "only the identity stays");

        let anticommuting = whole
            .iter()
            .filter(|(x, z, _)| !PauliString::<1> { x: **x, z: **z }.commutes_with(&gen))
            .count() as u64;
        assert!(anticommuting > 0, "fixture must have anticommuting terms");

        let (parts, counts) = run_parts(&whole, &prep, &rows, &AlwaysKeep, None);
        let sent: u64 = counts.iter().map(|c| c.rows_sent.iter().sum::<u64>()).sum();
        let received: u64 = counts.iter().map(|c| c.rows_received).sum();
        assert_eq!(
            sent, anticommuting,
            "one exported row per anticommuting term"
        );
        assert_eq!(received, anticommuting, "everything sent is received");
        for c in &counts {
            assert_eq!(c.remote_deltas, 1);
        }

        let got = PauliSum::merge_partitions(parts);
        let want = naive_apply_layer(&input, &rot, &AlwaysKeep, false);
        assert_terms_close(&got, &want, TOL, "all-remote generator");
    }

    /// A key whose contributions live on two partitions, and the amplitudes
    /// that bring each of them to it.
    ///
    /// `w` anticommutes with the generator, so it contributes to itself
    /// through the identity pass with amplitude `alpha = cos θ`, and its
    /// generator image `u = w · gen` contributes through the generator pass
    /// with amplitude `beta = i^k sin θ`. `part(u) = part(w) ^ part(gen)`
    /// differs from `part(w)`, so the two contributions *must* meet across the
    /// exchange or not at all. Both amplitudes are measured off the oracle
    /// rather than assumed, and returning them lets the caller pick
    /// coefficients whose sum at `w` is exactly `beta·alpha − alpha·beta`.
    fn cancelling_pair(
        rot: &PauliRotation<1>,
        rows: &PartitionRows<1>,
    ) -> (PauliString<1>, PauliString<1>, Complex64, Complex64) {
        let gen = wide_gen();
        let w = PauliString::<1>::x(0);
        assert!(!w.commutes_with(&gen));
        let mut u = w;
        u.mul_assign(&gen);
        assert_ne!(
            rows.partition_of(&w.x, &w.z),
            rows.partition_of(&u.x, &u.z),
            "the pair must straddle the partition boundary",
        );

        let probe = |term: PauliString<1>| -> Complex64 {
            let mut acc = BuildAccumulator::<1>::with_capacity(8, 1);
            acc.add_term(term, Phase::ONE, Complex64::new(1.0, 0.0));
            naive_apply_layer(&acc.finalize(), rot, &AlwaysKeep, false)
                .get(&w.x, &w.z)
                .unwrap_or(ZERO)
        };
        let alpha = probe(w);
        let beta = probe(u);
        assert!(alpha.norm() > 0.1 && beta.norm() > 0.1, "degenerate probe");
        (w, u, alpha, beta)
    }

    /// The layer's two-term fixture: `w` with coefficient `beta` and `u` with
    /// coefficient `-alpha · scale`, so `w`'s output is `beta·alpha` plus
    /// `-scale · alpha·beta`.
    fn cancelling_input(
        w: PauliString<1>,
        u: PauliString<1>,
        alpha: Complex64,
        beta: Complex64,
        scale: f64,
    ) -> PauliSum<1> {
        let mut acc = BuildAccumulator::<1>::with_capacity(8, 2);
        acc.add_term(w, Phase::ONE, beta);
        acc.add_term(u, Phase::ONE, -alpha * scale);
        acc.finalize()
    }

    /// `keep_term` sees the sum across the partition boundary: two
    /// contributions that each clear the threshold by five orders of magnitude
    /// cancel to below it, and the term is dropped — exactly as the oracle,
    /// which sums before it filters, drops it.
    ///
    /// Were the received rows merged *after* the policy ran (or into a
    /// separate bucket), the term would survive with the wrong coefficient.
    #[test]
    fn keep_term_sees_the_sum_across_partitions() {
        let rot = PauliRotation::new(wide_gen(), 0.41);
        let rows = rows_seeing_qubit_2_x();
        let (w, u, alpha, beta) = cancelling_pair(&rot, &rows);
        let input = cancelling_input(w, u, alpha, beta, 1.0 - 1e-7);
        let policy = CoefficientThreshold(1e-6);

        for bits in [0u8, 3] {
            let hash = Gf2Hash::<1>::new(8, bits, 0x9F);
            let whole = input.clone().with_hash(hash);
            let prep = Channel::<1>::prepare(&rot, whole.hash(), false).expect("prepare");
            let (parts, _) = run_parts(&whole, &prep, &rows, &policy, None);
            let got = PauliSum::merge_partitions(parts);
            assert!(
                got.get(&w.x, &w.z).is_none(),
                "bits={bits}: a term that only survives unsummed was kept ({:?})",
                got.get(&w.x, &w.z),
            );
            let want = naive_apply_layer(&input, &rot, &policy, false);
            assert_terms_close(&got, &want, TOL, &format!("threshold bits={bits}"));

            // Without the threshold the residue is there, and it is the sum.
            let (parts, _) = run_parts(&whole, &prep, &rows, &AlwaysKeep, None);
            let kept = PauliSum::merge_partitions(parts);
            let residue = kept.get(&w.x, &w.z).expect("the residue survives");
            assert!(
                residue.norm() < 1e-7 && residue.norm() > 1e-9,
                "bits={bits}: residue {residue} is not the near-cancellation",
            );
            let want = naive_apply_layer(&input, &rot, &AlwaysKeep, false);
            assert_terms_close(&kept, &want, TOL, &format!("no threshold bits={bits}"));
        }
    }

    /// Contributions from two partitions that cancel **exactly** leave no
    /// term: the merge drops exact zeros, and it can only see the zero because
    /// the received row was summed with the local one first.
    ///
    /// The cancellation is exact by construction, not by luck:
    /// `beta·alpha` and `(−alpha)·beta` round to the same magnitude with
    /// opposite signs in every component.
    #[test]
    fn exact_zero_sum_across_partitions_is_dropped() {
        let rot = PauliRotation::new(wide_gen(), 0.41);
        let rows = rows_seeing_qubit_2_x();
        let (w, u, alpha, beta) = cancelling_pair(&rot, &rows);
        let input = cancelling_input(w, u, alpha, beta, 1.0);

        for bits in [0u8, 3] {
            let hash = Gf2Hash::<1>::new(8, bits, 0x9F);
            let whole = input.clone().with_hash(hash);
            let prep = Channel::<1>::prepare(&rot, whole.hash(), false).expect("prepare");
            let (parts, _) = run_parts(&whole, &prep, &rows, &AlwaysKeep, None);
            let got = PauliSum::merge_partitions(parts);
            assert!(
                got.get(&w.x, &w.z).is_none(),
                "bits={bits}: an exactly cancelled term survived ({:?})",
                got.get(&w.x, &w.z),
            );
            let want = naive_apply_layer(&input, &rot, &AlwaysKeep, false);
            assert!(
                want.get(&w.x, &w.z).is_none(),
                "bits={bits}: the fixture does not cancel exactly in the oracle either",
            );
            assert_terms_close(&got, &want, TOL, &format!("exact zero bits={bits}"));
        }
    }

    /// Partitions with nothing in them still take part: they export empty
    /// blocks, receive real ones, and produce their share of the output.
    #[test]
    fn empty_partition_participates() {
        let channels = differential_channels_w1();
        let (_, su4) = channels
            .iter()
            .find(|(n, _)| *n == "haar_su4")
            .expect("the dense SU(4) cell");
        // Two terms over four partitions: at least two parts are empty.
        let mut acc = BuildAccumulator::<1>::with_capacity(8, 2);
        acc.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(1.0, 0.0));
        acc.add_term(
            PauliString::<1>::z(5),
            Phase::ONE,
            Complex64::new(-0.5, 0.25),
        );
        let input = acc.finalize();
        let rows = PartitionRows::<1>::from_seed(8, 2, 0x9D4);
        assert_eq!(rows.num_partitions(), 4);

        let hash = Gf2Hash::<1>::new(8, 2, 0x9D5);
        let whole = input.clone().with_hash(hash);
        let prep = su4.prepare(whole.hash(), false).expect("prepare");
        assert!(PartitionPlan::new(&prep, &rows, 0).has_remote());

        // The fixture property is an empty *input* share — which two terms
        // over four partitions guarantee by pigeonhole, whatever the row draw.
        // (It used to be asserted on the outputs, where it depended on the
        // draw, and `f39341a`'s reseeding made every output share non-empty.)
        assert!(
            (0..rows.num_partitions() as u32).any(|r| whole.filter_partition(&rows, r).is_empty()),
            "the fixture must leave a partition's input empty",
        );
        let (parts, counts) = run_parts(&whole, &prep, &rows, &AlwaysKeep, None);
        assert_eq!(counts.len(), 4);
        let got = PauliSum::merge_partitions(parts);
        let want = naive_apply_layer(&input, su4.as_ref(), &AlwaysKeep, false);
        assert_terms_close(&got, &want, TOL, "empty partition");
    }

    /// Each partition's output is byte-identical across Rayon pool sizes.
    ///
    /// The determinism argument of ARCHITECTURE.md §Determinism survives
    /// partitioning: cosets stay write-disjoint, and the received rows are
    /// appended to a run in a fixed order (the plan's delta order, then CSR
    /// order inside a block) that no thread count can perturb.
    #[test]
    fn partitioned_output_is_byte_identical_across_pool_sizes() {
        let input = rand_sum::<1>(1500, 8, 0x9D6);
        let channels = differential_channels_w1();
        let rot = PauliRotation::new(wide_gen(), 0.41);
        let (_, su4) = channels
            .iter()
            .find(|(n, _)| *n == "haar_su4")
            .expect("the dense SU(4) cell");
        // The dense PTM crosses under a random draw; the wide rotation needs
        // the row that sees its generator, or its one non-identity delta
        // stays local and there is nothing to exchange.
        let cases: [(&str, &dyn Channel<1>, PartitionRows<1>); 2] = [
            (
                "su4",
                su4.as_ref(),
                PartitionRows::<1>::from_seed(8, 1, 0x9D8),
            ),
            ("rot_wide", &rot, rows_seeing_qubit_2_x()),
        ];

        for (name, ch, rows) in cases {
            // 64 buckets: comfortably above MIN_COSETS_FOR_PARALLEL.
            let hash = Gf2Hash::<1>::new(8, 6, 0x9D7);
            let whole = input.clone().with_hash(hash);
            let prep = ch.prepare(whole.hash(), false).expect("prepare");
            assert!(PartitionPlan::new(&prep, &rows, 0).has_remote(), "{name}");

            let (one, counts_one) = run_parts(&whole, &prep, &rows, &AlwaysKeep, Some(1));
            let (four, counts_four) = run_parts(&whole, &prep, &rows, &AlwaysKeep, Some(4));
            assert_eq!(counts_one, counts_four, "{name}: counts");
            for (rank, (a, b)) in one.iter().zip(&four).enumerate() {
                assert_eq!(
                    a.to_arrays(),
                    b.to_arrays(),
                    "{name}: partition {rank} is not byte-identical across pool sizes",
                );
            }
        }
    }
}
