//! Per-phase timing counters for the propagation engine (feature `phase-timing`).
//!
//! Compiled only under `--features phase-timing`; the default build carries no timing code and no stats fields, so it is byte- and performance-identical to an uninstrumented build.
//!
//! Read the counters through [`LayerScratch::take_stats`](crate::engine::bucketed::LayerScratch::take_stats) after driving layers with [`propagate_with_scratch`](crate::engine::propagate_with_scratch), or — per partition, plus the driver's own scatter/gather — through `PartitionedSum::take_stats`.
//!
//! The three `*_ns` fields the partitioned engine adds (`collective_ns`, `export_ns`, `exchange_ns`), its two row counters, and the two worker sub-phases (`append_ns`, `chunk_wait_ns`) are all zero in the unpartitioned engine, which has no exchange.

use std::time::Instant;

/// Rough estimate of the cost of one `Instant::now()` read, in nanoseconds, for this hardware class; used by the `phase_breakdown` probe's overhead line (`timer_reads() * TIMER_READ_OVERHEAD_NS`).
pub const TIMER_READ_OVERHEAD_NS: u64 = 25;

/// Cumulative per-phase breakdown of one or more propagation layers.
///
/// All `*_ns` fields are nanoseconds, summed over every layer since the counters were last drained.
/// **Two clock domains are deliberately mixed**:
///
/// - **Wall-clock phases** (`rebucket_ns` through `exchange_ns`) are measured once per layer on the calling thread — in partitioned mode, on the partition's own driving thread, one `PhaseStats` per partition; per layer they sum to approximately the layer's wall time.
/// - **Sub-phases** (`append_ns`, `chunk_wait_ns`) break a phase above down further and are *contained in* it — `append_ns` is part of `gather_ns`, and `chunk_wait_ns` part of `append_ns`. They are excluded from [`wall_total_ns`](Self::wall_total_ns) for exactly that reason.
/// - **Worker busy-time phases** (`swap_ns` through `clear_ns`) are summed across every coset task on every Rayon worker, so under a `t`-thread pool they sum to `coset_loop_ns × t × efficiency`, **not** to `coset_loop_ns`; the mismatch between the two domains is the load-balance signal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseStats {
    // -- wall-clock, once per layer, on the calling thread --
    /// `PauliSum::rebucket` before each layer (grow-only; refine parallelizes above the worth-splitting threshold — ARCHITECTURE.md §Bucket-Policy).
    pub rebucket_ns: u64,
    /// `Channel::prepare` (PTM derivation; O(1) in the term count).
    pub prepare_ns: u64,
    /// Key-preserving fast path (`rescale_in_place`, whole call), taken by identity / depolarizing / dephasing / Pauli layers instead of the coset machinery.
    pub rescale_ns: u64,
    /// `Gf2Span::new` + `DeltaPlan::new` (per-layer coset planning).
    pub span_plan_ns: u64,
    /// Serial bucket-handle permutation into coset-contiguous order.
    pub permute_ns: u64,
    /// Wall time of the whole coset loop (serial or parallel branch).
    pub coset_loop_ns: u64,
    /// Serial bucket-handle un-permutation.
    pub unpermute_ns: u64,
    /// `PauliSum::recount` at the end of the bucketed layer (serial).
    pub recount_ns: u64,
    /// `TruncationPolicy::finalize_layer` after each layer — in partitioned mode `PartitionedTruncation::finalize_layer_partitioned`, including the collectives it issues.
    pub finalize_ns: u64,
    /// **Partitioned only.** The driver's own per-layer collective (bucket-count all-reduce). Zero in the unpartitioned engine, which computes the count locally.
    pub collective_ns: u64,
    /// **Partitioned only.** The export pass building this layer's per-partner exchange blocks. Zero on a layer with no remote delta.
    pub export_ns: u64,
    /// **Partitioned only.** The exchange itself, minus the coset loop it now wraps, including the wait for a partner — read it with `chunk_wait_ns`, where the hidden transfer shows up.
    pub exchange_ns: u64,
    // -- worker busy time, summed over all coset tasks (see type docs) --
    /// Scratch resize + column swap-out at the top of each coset task.
    pub swap_ns: u64,
    /// Exact per-run capacity sizing.
    pub size_ns: u64,
    /// Gather (input-major, output-major, or inline rotation — all variants).
    pub gather_ns: u64,
    /// `sort_rows_with_scratch` over each run's rest stream; the pre-sorted id stream skips it.
    pub sort_ns: u64,
    /// `merge2_into` — the fused id/rest two-stream merge + reduction into the live bucket column, including interleaving the id rows.
    pub merge_ns: u64,
    /// Clearing the swapped-out columns at the end of each coset task.
    pub clear_ns: u64,
    // -- counters --
    /// Layers driven through `propagate_with_scratch`.
    pub layers: u64,
    /// Coset tasks executed (feeds the timer-overhead estimate).
    pub cosets: u64,
    /// Sort/merge runs executed (= Σ coset sizes).
    pub runs: u64,
    /// Rows pushed into gather runs (= Σ run lengths entering sort/merge) — the roofline model's traffic multiplier.
    pub rows_gathered: u64,
    /// The subset of `rows_gathered` that went through the per-run sort — the rest streams only.
    /// Identity-delta rows arrive pre-sorted and skip it, so `rows_gathered - rows_sorted` is the sorted-volume saving the split buys.
    pub rows_sorted: u64,
    /// The subset of the identity rows whose **keys** were never materialized: a dense identity plan borrows the source bucket's key columns in place, so these rows cost `2×16` bytes of run traffic instead of `2×T`.
    /// Zero for sparse (Clifford) identity plans, which keep the full key+coeff materialization.
    pub rows_id: u64,
    /// Σ over layers of the term count *before* the layer.
    pub terms_in: u64,
    /// Σ over layers of the term count *after* the layer (post-truncation).
    pub terms_out: u64,
    /// **Partitioned only.** Rows this partition exported to its partners, summed over layers.
    pub rows_exported: u64,
    /// **Partitioned only.** Rows this partition received, summed over layers — the exchange's contribution to `rows_sorted`.
    pub recv_rows: u64,
    // -- sub-phases, contained in the phases above (see the type docs) --
    /// **Partitioned only, worker busy time.** Appending received rows into each output bucket's rest stream. A part of `gather_ns`.
    pub append_ns: u64,
    /// **Distributed only, worker busy time.** Blocking for a chunk of the layer's rows to land. A part of `append_ns`; zero means the rows were always there when a task reached them.
    pub chunk_wait_ns: u64,
}

impl PhaseStats {
    /// Accumulate another drained snapshot into `self` (e.g. summing repetitions in a probe).
    pub fn add(&mut self, o: &PhaseStats) {
        self.rebucket_ns += o.rebucket_ns;
        self.prepare_ns += o.prepare_ns;
        self.rescale_ns += o.rescale_ns;
        self.span_plan_ns += o.span_plan_ns;
        self.permute_ns += o.permute_ns;
        self.coset_loop_ns += o.coset_loop_ns;
        self.unpermute_ns += o.unpermute_ns;
        self.recount_ns += o.recount_ns;
        self.finalize_ns += o.finalize_ns;
        self.collective_ns += o.collective_ns;
        self.export_ns += o.export_ns;
        self.exchange_ns += o.exchange_ns;
        self.swap_ns += o.swap_ns;
        self.size_ns += o.size_ns;
        self.gather_ns += o.gather_ns;
        self.sort_ns += o.sort_ns;
        self.merge_ns += o.merge_ns;
        self.clear_ns += o.clear_ns;
        self.layers += o.layers;
        self.cosets += o.cosets;
        self.runs += o.runs;
        self.rows_gathered += o.rows_gathered;
        self.rows_sorted += o.rows_sorted;
        self.rows_id += o.rows_id;
        self.terms_in += o.terms_in;
        self.terms_out += o.terms_out;
        self.rows_exported += o.rows_exported;
        self.recv_rows += o.recv_rows;
        self.append_ns += o.append_ns;
        self.chunk_wait_ns += o.chunk_wait_ns;
    }

    /// Fold one coset task's busy-time counters into the totals.
    pub(crate) fn absorb_coset(&mut self, c: &CosetStats) {
        self.swap_ns += c.swap_ns;
        self.size_ns += c.size_ns;
        self.gather_ns += c.gather_ns;
        self.sort_ns += c.sort_ns;
        self.merge_ns += c.merge_ns;
        self.clear_ns += c.clear_ns;
        self.cosets += c.cosets;
        self.runs += c.runs;
        self.rows_gathered += c.rows_gathered;
        self.rows_sorted += c.rows_sorted;
        self.rows_id += c.rows_id;
    }

    /// Sum of the wall-clock phase fields — approximately the total wall time spent inside the instrumented region across all layers.
    pub fn wall_total_ns(&self) -> u64 {
        self.rebucket_ns
            + self.prepare_ns
            + self.rescale_ns
            + self.span_plan_ns
            + self.permute_ns
            + self.coset_loop_ns
            + self.unpermute_ns
            + self.recount_ns
            + self.finalize_ns
            + self.collective_ns
            + self.export_ns
            + self.exchange_ns
    }

    /// Sum of the worker busy-time phase fields (see the type docs for how this relates to `coset_loop_ns`).
    pub fn busy_total_ns(&self) -> u64 {
        self.swap_ns + self.size_ns + self.gather_ns + self.sort_ns + self.merge_ns + self.clear_ns
    }

    /// Upper-bound estimate of the number of `Instant::now()` reads behind these counters: ~11 per layer, ~5 per coset task, 2 per run.
    /// At [`TIMER_READ_OVERHEAD_NS`] ns per read, `timer_reads() × TIMER_READ_OVERHEAD_NS` ns is the self-inflicted overhead ceiling a probe should print next to the breakdown.
    pub fn timer_reads(&self) -> u64 {
        11 * self.layers + 5 * self.cosets + 2 * self.runs
    }
}

/// One coset task's busy-time counters, embedded in each `CosetScratch` so a worker only ever touches its own slot — same disjointness argument as the scratch itself, no synchronization added.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CosetStats {
    pub(crate) swap_ns: u64,
    pub(crate) size_ns: u64,
    pub(crate) gather_ns: u64,
    pub(crate) sort_ns: u64,
    pub(crate) merge_ns: u64,
    pub(crate) clear_ns: u64,
    pub(crate) cosets: u64,
    pub(crate) runs: u64,
    pub(crate) rows_gathered: u64,
    pub(crate) rows_sorted: u64,
    pub(crate) rows_id: u64,
}

/// Chained timestamp: `lap` records elapsed-since-last into a slot and re-arms, so N sequential phases cost N+1 clock reads instead of 2N.
pub(crate) struct Stamp(Instant);

impl Stamp {
    #[inline]
    pub(crate) fn now() -> Self {
        Stamp(Instant::now())
    }

    /// Add the time since the last stamp to `slot` and re-arm.
    #[inline]
    pub(crate) fn lap(&mut self, slot: &mut u64) {
        let t = Instant::now();
        *slot += t.duration_since(self.0).as_nanos() as u64;
        self.0 = t;
    }

    /// Re-arm without recording — used to skip over a region that does its own internal timing (e.g. the bucketed layer between the `prepare` and `finalize` laps).
    #[inline]
    pub(crate) fn rearm(&mut self) {
        self.0 = Instant::now();
    }
}
