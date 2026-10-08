//! Per-phase timing counters, compiled only under the `phase-timing` feature.

use std::time::Instant;

/// Cumulative per-phase breakdown of the layers driven since the counters were last drained, in nanoseconds.
///
/// Read through `LayerScratch::take_stats` or the partitioned sums' `take_stats`.
/// Two clock domains are mixed: the wall-clock phases (`rebucket_ns` through `exchange_ns`) are measured once per layer on the driving thread, while the busy-time phases (`swap_ns` through `clear_ns`) are summed over every Rayon worker and so exceed `coset_loop_ns` under parallelism.
/// The sub-phases `append_ns` and `chunk_wait_ns` are contained in `gather_ns` and `append_ns` respectively, never additional to them.
/// Fields marked partitioned- or device-only are zero elsewhere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseStats {
    // -- wall-clock, once per layer, on the calling thread --
    /// `PauliSum::rebucket` before each layer.
    pub rebucket_ns: u64,
    /// `Channel::prepare`.
    pub prepare_ns: u64,
    /// The key-preserving rescale path, taken instead of the coset loop.
    pub rescale_ns: u64,
    /// Per-layer coset planning.
    pub span_plan_ns: u64,
    /// Serial bucket-handle permutation into coset-contiguous order.
    pub permute_ns: u64,
    /// Wall time of the whole coset loop (serial or parallel branch).
    pub coset_loop_ns: u64,
    /// Serial bucket-handle un-permutation.
    pub unpermute_ns: u64,
    /// The length recount at the end of the layer.
    pub recount_ns: u64,
    /// `finalize_layer` after each layer, including any collectives it issues.
    pub finalize_ns: u64,
    /// Partitioned only: the driver's per-layer bucket-count all-reduce.
    pub collective_ns: u64,
    /// Partitioned only: the export pass building the per-partner exchange blocks.
    pub export_ns: u64,
    /// Partitioned only: the exchange minus the coset loop it wraps, including the wait for a partner.
    pub exchange_ns: u64,
    // -- worker busy time, summed over all coset tasks --
    /// Column swap-out at the top of each coset task.
    pub swap_ns: u64,
    /// Exact per-run capacity sizing.
    pub size_ns: u64,
    /// Gather, every variant.
    pub gather_ns: u64,
    /// Sorting each run's rest stream.
    pub sort_ns: u64,
    /// The fused two-stream merge and reduction.
    pub merge_ns: u64,
    /// Clearing the swapped-out columns at the end of each coset task.
    pub clear_ns: u64,
    // -- counters --
    /// Layers driven.
    pub layers: u64,
    /// Coset tasks executed.
    pub cosets: u64,
    /// Sort/merge runs executed.
    pub runs: u64,
    /// Rows pushed into gather runs.
    pub rows_gathered: u64,
    /// The subset of `rows_gathered` that went through the sort, the rest streams only.
    pub rows_sorted: u64,
    /// Identity rows whose keys were borrowed from the source bucket rather than materialized.
    pub rows_id: u64,
    /// Term counts before each layer, summed over layers.
    pub terms_in: u64,
    /// Term counts after each layer, summed over layers.
    pub terms_out: u64,
    /// Partitioned only: rows exported to partners.
    pub rows_exported: u64,
    /// Partitioned only: rows received, part of `rows_sorted`.
    pub recv_rows: u64,
    // -- sub-phases, contained in the phases above --
    /// Partitioned only, busy time: appending received rows, part of `gather_ns`.
    pub append_ns: u64,
    /// Distributed only, busy time: waiting for a chunk of rows to land, part of `append_ns`.
    pub chunk_wait_ns: u64,
    // -- device only (feature `cuda`); zero on a host partition --
    /// Device only, kernel time: compaction into the output columns, part of `coset_loop_ns`.
    pub compact_ns: u64,
    /// Device only: host-to-device copies inside a layer, part of `coset_loop_ns` or `rescale_ns`.
    pub h2d_ns: u64,
    /// Device only: device-to-host copies inside a layer, part of `coset_loop_ns` or `rescale_ns`.
    pub d2h_ns: u64,
}

impl PhaseStats {
    /// Accumulate another drained snapshot into `self`.
    pub fn add(&mut self, other: &PhaseStats) {
        self.rebucket_ns += other.rebucket_ns;
        self.prepare_ns += other.prepare_ns;
        self.rescale_ns += other.rescale_ns;
        self.span_plan_ns += other.span_plan_ns;
        self.permute_ns += other.permute_ns;
        self.coset_loop_ns += other.coset_loop_ns;
        self.unpermute_ns += other.unpermute_ns;
        self.recount_ns += other.recount_ns;
        self.finalize_ns += other.finalize_ns;
        self.collective_ns += other.collective_ns;
        self.export_ns += other.export_ns;
        self.exchange_ns += other.exchange_ns;
        self.swap_ns += other.swap_ns;
        self.size_ns += other.size_ns;
        self.gather_ns += other.gather_ns;
        self.sort_ns += other.sort_ns;
        self.merge_ns += other.merge_ns;
        self.clear_ns += other.clear_ns;
        self.layers += other.layers;
        self.cosets += other.cosets;
        self.runs += other.runs;
        self.rows_gathered += other.rows_gathered;
        self.rows_sorted += other.rows_sorted;
        self.rows_id += other.rows_id;
        self.terms_in += other.terms_in;
        self.terms_out += other.terms_out;
        self.rows_exported += other.rows_exported;
        self.recv_rows += other.recv_rows;
        self.append_ns += other.append_ns;
        self.chunk_wait_ns += other.chunk_wait_ns;
        self.compact_ns += other.compact_ns;
        self.h2d_ns += other.h2d_ns;
        self.d2h_ns += other.d2h_ns;
    }

    /// Fold one coset task's busy-time counters into the totals.
    pub(crate) fn absorb_coset(&mut self, coset: &CosetStats) {
        self.swap_ns += coset.swap_ns;
        self.size_ns += coset.size_ns;
        self.gather_ns += coset.gather_ns;
        self.sort_ns += coset.sort_ns;
        self.merge_ns += coset.merge_ns;
        self.clear_ns += coset.clear_ns;
        self.cosets += coset.cosets;
        self.runs += coset.runs;
        self.rows_gathered += coset.rows_gathered;
        self.rows_sorted += coset.rows_sorted;
        self.rows_id += coset.rows_id;
    }

    /// Sum of the wall-clock phase fields.
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

    /// Sum of the busy-time phase fields; on a device row, the kernel time inside the coset loop.
    pub fn busy_total_ns(&self) -> u64 {
        self.swap_ns
            + self.size_ns
            + self.gather_ns
            + self.sort_ns
            + self.merge_ns
            + self.clear_ns
            + self.compact_ns
    }

    /// Upper bound on the clock reads behind these counters, for estimating the instrumentation's own overhead.
    pub fn timer_reads(&self) -> u64 {
        11 * self.layers + 5 * self.cosets + 2 * self.runs
    }
}

/// One coset task's busy-time counters, held per worker slot.
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

/// Chained timestamp: `lap` records the time since the last stamp and re-arms.
pub(crate) struct Stamp(Instant);

impl Stamp {
    #[inline]
    pub(crate) fn now() -> Self {
        Stamp(Instant::now())
    }

    #[inline]
    pub(crate) fn lap(&mut self, slot: &mut u64) {
        let now = Instant::now();
        *slot += now.duration_since(self.0).as_nanos() as u64;
        self.0 = now;
    }

    /// Re-arm without recording, skipping a region that times itself.
    #[inline]
    pub(crate) fn rearm(&mut self) {
        self.0 = Instant::now();
    }
}
