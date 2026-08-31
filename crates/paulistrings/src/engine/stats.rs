//! Per-phase timing counters for the propagation engine (feature
//! `phase-timing`).
//!
//! This whole module — and every field and statement that feeds it — is
//! compiled only under `--features phase-timing`. The default build carries
//! no timing code and no stats fields, so it is byte- and
//! performance-identical to an uninstrumented build; the acceptance test for
//! that claim is the engine's own output-stability nets (the fingerprint
//! table, the thread-count/bucket-count/seed bitwise-identity tests, and
//! `capacity_stabilizes_across_repeated_layers`) passing *with the feature
//! enabled*: timers read the clock and add to plain integers, and touch no
//! term data, no ordering, and no capacities.
//!
//! Read the counters through
//! [`LayerScratch::take_stats`](crate::engine::bucketed::LayerScratch::take_stats)
//! after driving layers with
//! [`propagate_with_scratch`](crate::engine::propagate_with_scratch).

use std::time::Instant;

/// Rough estimate of the cost of one `Instant::now()` read, in nanoseconds,
/// for this hardware class; used by the `phase_breakdown` probe's overhead
/// line (`timer_reads() * TIMER_READ_OVERHEAD_NS`).
pub const TIMER_READ_OVERHEAD_NS: u64 = 25;

/// Cumulative per-phase breakdown of one or more propagation layers.
///
/// All `*_ns` fields are nanoseconds, summed over every layer since the
/// counters were last drained. **Two clock domains are deliberately mixed**:
///
/// - **Wall-clock phases** (`rebucket_ns` through `finalize_ns`) are measured
///   once per layer on the calling thread; per layer they sum to approximately
///   the layer's wall time.
/// - **Worker busy-time phases** (`swap_ns` through `clear_ns`) are summed
///   across every coset task on every Rayon worker. Under a `t`-thread pool
///   they sum to `coset_loop_ns × t × efficiency`, **not** to
///   `coset_loop_ns`; the ratio `Σbusy / (coset_loop_ns × t)` is the coset
///   loop's parallel efficiency, and the mismatch between the two domains is
///   itself the load-balance signal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseStats {
    // -- wall-clock, once per layer, on the calling thread --
    /// `PauliSum::rebucket` before each layer (grow-only since v0.5 §R1;
    /// refine parallelizes above the worth-splitting threshold, §R2).
    pub rebucket_ns: u64,
    /// `Channel::prepare` (PTM derivation; O(1) in the term count).
    pub prepare_ns: u64,
    /// Key-preserving fast path (`rescale_in_place`, whole call), taken by
    /// identity / depolarizing / dephasing / Pauli layers instead of the
    /// coset machinery.
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
    /// `TruncationPolicy::finalize_layer` after each layer.
    pub finalize_ns: u64,
    // -- worker busy time, summed over all coset tasks (see type docs) --
    /// Scratch resize + column swap-out at the top of each coset task.
    pub swap_ns: u64,
    /// Exact per-run capacity sizing.
    pub size_ns: u64,
    /// Gather (input-major, output-major, or inline rotation — all variants).
    pub gather_ns: u64,
    /// `sort_rows_with_scratch` over each run's rest stream (v0.5 S1/S2:
    /// key-only adaptive sort on worker-persistent scratch, no per-run
    /// allocation in the steady state; the pre-sorted id stream skips it).
    pub sort_ns: u64,
    /// `merge2_into` — the fused id/rest two-stream merge + reduction into
    /// the live bucket column (v0.5 S2: includes interleaving the id rows,
    /// which the single-stream pipeline used to pay for inside `sort_ns`).
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
    /// Rows pushed into gather runs (= Σ run lengths entering sort/merge).
    /// The traffic multiplier for the roofline model: each gathered row is
    /// one key+coeff written by gather (no tag column since v0.5 S1) and read
    /// once more by merge; the `rows_sorted` subset is additionally read and
    /// rewritten by the sort.
    pub rows_gathered: u64,
    /// The subset of `rows_gathered` that went through the per-run sort —
    /// the rest streams only. Identity-delta rows arrive pre-sorted and skip
    /// the sort entirely (v0.5 S2), so `rows_gathered - rows_sorted` is the
    /// sorted-volume saving the split buys.
    pub rows_sorted: u64,
    /// The subset of the identity rows whose **keys** were never
    /// materialized (v0.6 G1d): under a dense identity plan the merge
    /// borrows the source bucket's key columns in place and only the
    /// 16-byte coefficient moves through the run, so these rows cost
    /// `2×16` bytes of run traffic instead of `2×T`. Zero for sparse
    /// (Clifford) identity plans, which keep the full v0.5 key+coeff
    /// materialization.
    pub rows_id: u64,
    /// Σ over layers of the term count *before* the layer.
    pub terms_in: u64,
    /// Σ over layers of the term count *after* the layer (post-truncation).
    pub terms_out: u64,
}

impl PhaseStats {
    /// Accumulate another drained snapshot into `self` (e.g. summing
    /// repetitions in a probe).
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

    /// Sum of the wall-clock phase fields — approximately the total wall
    /// time spent inside the instrumented region across all layers.
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
    }

    /// Sum of the worker busy-time phase fields (see the type docs for how
    /// this relates to `coset_loop_ns`).
    pub fn busy_total_ns(&self) -> u64 {
        self.swap_ns + self.size_ns + self.gather_ns + self.sort_ns + self.merge_ns + self.clear_ns
    }

    /// Upper-bound estimate of the number of `Instant::now()` reads behind
    /// these counters: ~11 per layer, ~5 per coset task, 2 per run. At
    /// [`TIMER_READ_OVERHEAD_NS`] ns per read on this class of hardware,
    /// `timer_reads() × TIMER_READ_OVERHEAD_NS` ns is the self-inflicted
    /// overhead ceiling — a probe should print it next to the breakdown so
    /// the reader can see when the measurement pollutes itself (tiny
    /// cosets, many runs).
    pub fn timer_reads(&self) -> u64 {
        11 * self.layers + 5 * self.cosets + 2 * self.runs
    }
}

/// One coset task's busy-time counters, embedded in each `CosetScratch` so a
/// worker only ever touches its own slot — same disjointness argument as the
/// scratch itself, no synchronization added.
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

/// Chained timestamp: `lap` records elapsed-since-last into a slot and
/// re-arms, so N sequential phases cost N+1 clock reads instead of 2N.
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

    /// Re-arm without recording — used to skip over a region that does its
    /// own internal timing (e.g. the bucketed layer between the `prepare`
    /// and `finalize` laps).
    #[inline]
    pub(crate) fn rearm(&mut self) {
        self.0 = Instant::now();
    }
}
