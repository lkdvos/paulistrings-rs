//! The bucketed layer engine. See ARCHITECTURE.md §Engine.
//!
//! Applies one prepared channel to a `PauliSum` one coset of `span(h(D))` at a time: cosets are write-disjoint, so each is gathered, sorted and merged in place with no cross-task synchronization. `LayerScratch` holds the reusable per-layer working set.

use std::sync::Mutex;

use num_complex::Complex64;
use rayon::prelude::*;

use super::coset::Gf2Span;
use super::merge::{
    RADIX_MAX_REST_ROWS_PER_KEY, RADIX_MIN_DISJOINT_STREAMS, RADIX_MIN_REST_STREAMS,
};
use crate::channel::prepared::{LocalPtm, Prepared, RotationPrep};
use crate::pauli_sum::storage::{BucketCols, PauliSum};
use crate::truncation::TruncationPolicy;

#[cfg(feature = "phase-timing")]
use super::stats::{PhaseStats, Stamp};

mod coset_fill;

pub(super) use coset_fill::{fill_coset, CosetScratch, MIN_COSETS_FOR_PARALLEL};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Reusable per-layer scratch, held by the caller across layers so a layer allocates nothing after the first call: every field retains its high-water capacity across cosets and layers.
/// The serial path uses this instance directly; the parallel path gives each Rayon worker its own slot in `workers`, so capacity is bounded by `threads × coset working set`.
///
/// A task's output cannot depend on which scratch slot it drew: the swap site clears every write destination before use, so worker-to-slot assignment varying run to run is unobservable — which is what keeps output byte-identical across thread counts.
#[derive(Debug, Default)]
pub struct LayerScratch<const W: usize> {
    /// The per-coset working set (serial path).
    pub(super) task: CosetScratch<W>,
    /// The layer's handle permutation, `perm[β] = span.perm_index(β)`.
    pub(super) perm: Vec<u32>,
    /// The inverse of [`Self::perm`], `inv_perm[perm[β]] = β`, so a coset member's *original* bucket index is recoverable from its permuted position. Filled only when the layer's [`ExtraRows`] source asks for it ([`ExtraRows::NEEDS_BETA`]); left empty otherwise, which [`fill_coset`] reads as "the permutation is the identity".
    pub(super) inv_perm: Vec<u32>,
    /// Staging area the bucket handles are permuted into. Holds handles only while a layer runs; its elements carry no capacity of their own.
    pub(super) staging: Vec<BucketCols<W>>,
    /// Worker-persistent coset working sets for the parallel path, one slot per Rayon worker, indexed by `rayon::current_thread_index()`. Each worker locks only its own slot, so the mutexes are uncontended.
    pub(super) workers: Vec<Mutex<CosetScratch<W>>>,
    /// Layer-level (wall-clock) phase counters; the per-coset busy-time counters live in each `CosetScratch`.
    #[cfg(feature = "phase-timing")]
    pub(crate) stats: PhaseStats,
    /// The opt-in per-layer term-count trace, `None` unless [`Self::enable_term_trace`] was called.
    pub(crate) term_trace: Option<TermTrace>,
    /// The opt-in per-layer gate trace, `None` unless [`Self::enable_gate_trace`] was called.
    pub(crate) gate_trace: Option<GateTrace>,
}

impl<const W: usize> LayerScratch<W> {
    /// An empty scratch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drain and return the accumulated phase counters (layer-level wall-clock fields plus every worker's busy-time counters), zeroing them.
    /// Counters accumulate across layers and `propagate_with` calls until drained.
    #[cfg(feature = "phase-timing")]
    pub fn take_stats(&mut self) -> PhaseStats {
        let mut total = std::mem::take(&mut self.stats);
        total.absorb_coset(&std::mem::take(&mut self.task.stats));
        for slot in &self.workers {
            let mut ws = slot.lock().unwrap();
            total.absorb_coset(&std::mem::take(&mut ws.stats));
        }
        total
    }

    /// Start recording a [`TermTrace`] on every subsequent [`propagate_with`](crate::propagate_with) call driven by this scratch. Idempotent, and it never discards counts already recorded. Always compiled: the counts come from `sum.len()` reads the layer loop already performs.
    pub fn enable_term_trace(&mut self) {
        self.term_trace.get_or_insert_with(TermTrace::default);
    }

    /// Drain and return the per-layer term counts, or `None` if tracing was never enabled.
    /// Draining leaves tracing enabled with empty vectors, so a reused scratch reports each call separately without re-enabling.
    pub fn take_term_trace(&mut self) -> Option<TermTrace> {
        self.term_trace.as_mut().map(std::mem::take)
    }

    /// Start recording a [`GateTrace`] on every subsequent [`propagate_with`](crate::propagate_with) call driven by this scratch. Idempotent, and it never discards records already taken.
    /// Always compiled: enabling it costs one extra `Instant::now()` pair per traced layer, gated behind the same hoisted flag as the per-layer `DEBUG` log.
    pub fn enable_gate_trace(&mut self) {
        self.gate_trace.get_or_insert_with(GateTrace::default);
    }

    /// Drain and return the per-layer gate trace, or `None` if tracing was never enabled.
    /// Draining leaves tracing enabled with empty vectors, so a reused scratch reports each call separately without re-enabling.
    pub fn take_gate_trace(&mut self) -> Option<GateTrace> {
        self.gate_trace.as_mut().map(std::mem::take)
    }
}

/// Per-layer resident term counts, recorded by [`propagate_with`](crate::propagate_with) when the driving [`LayerScratch`] has [`enable_term_trace`](LayerScratch::enable_term_trace) set.
/// Both vectors have one entry per layer applied, in application order (so *reverse* circuit order under [`Direction::Heisenberg`](crate::Direction)). Always compiled — the `phase-timing` feature gates only the timing counters.
/// These are counts of the sum as it rests between layers, post-truncation; the transient in-layer expansion is not captured, since observing it would mean instrumenting the coset loop.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TermTrace {
    /// Resident term count before each layer. `terms_in[k + 1]` equals `terms_out[k]`.
    pub terms_in: Vec<usize>,
    /// Resident term count after each layer, i.e. after the truncation policy's `finalize_layer`.
    pub terms_out: Vec<usize>,
}

impl TermTrace {
    /// Peak resident term count between layers, or `None` for a zero-layer trace.
    pub fn peak_terms(&self) -> Option<usize> {
        self.terms_in
            .first()
            .copied()
            .into_iter()
            .chain(self.terms_out.iter().copied())
            .max()
    }
}

/// Per-layer structured gate trace: application index, original circuit index, gate name, term counts, and the complete gate's elapsed wall time.
///
/// Recorded by [`propagate_with`](crate::propagate_with) when the driving [`LayerScratch`] has [`enable_gate_trace`](LayerScratch::enable_gate_trace) set.
/// Always compiled, like [`TermTrace`]; unlike it, a traced layer pays one extra `Instant::now()` pair, gated behind the same hoisted flag that already guards the per-layer `DEBUG` log, so an untraced layer's cost is unchanged (CLAUDE.md §Performance discipline).
/// `nanos[k]` covers the same window `propagate`'s per-layer `DEBUG` line reports: before `rebucket`/`prepare`, through the coset loop, `finalize_layer`, and any completion sync that phase needs — a complete gate application.
///
/// Every field has one entry per layer applied, in *application* order (so reverse circuit order under [`Direction::Heisenberg`](crate::Direction::Heisenberg)); `circuit_index` recovers the original position regardless of direction, and `application_index` is the loop counter, so a consumer needs neither the direction nor the circuit length to pair the two.
/// There is no per-gate Trotter-step index here: [`Circuit`](crate::Circuit) has no notion of steps, so a step boundary is `circuit_index / channels_per_step` for a caller who knows that constant, not something the engine can compute.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GateTrace {
    /// This layer's position in [`Circuit::channels`](crate::Circuit), independent of propagation direction.
    pub circuit_index: Vec<u32>,
    /// This layer's position in the propagation loop, i.e. `k` in `propagate`'s `for k in 0..n`.
    pub application_index: Vec<u32>,
    /// [`Channel::debug_name`](crate::Channel::debug_name) of the applied channel.
    pub gate_name: Vec<&'static str>,
    /// Resident term count before the layer.
    pub terms_in: Vec<usize>,
    /// Resident term count after the layer, i.e. after `finalize_layer`.
    pub terms_out: Vec<usize>,
    /// Elapsed wall-clock nanoseconds for the complete gate application.
    pub nanos: Vec<u64>,
}

/// A prepared channel's delta set, annotated with each entry's coset coordinate
/// (`span.coord_of(bucket_delta)`), computed once per layer.
pub(super) enum DeltaPlan<'p, const W: usize> {
    /// Tabulated deltas; `coords[e]` pairs with `ptm.deltas()[e]`.
    Local {
        ptm: &'p LocalPtm<W>,
        coords: Vec<u32>,
        /// Whether `deltas()[0]` is the identity delta (entry 0 by construction order), whose stream the gather routes into the run's pre-sorted `id` columns. True for every built-in channel.
        has_identity: bool,
        /// Whether the identity entry's amplitude is nonzero for every active support pattern. Dense means each source row emits exactly one id row with its key untouched, so the gather materializes only the coefficient and the merge borrows the keys from `old[j]` in place. True for `GeneralUnitary1Q/2Q` and weight-≤2 rotations; false for Cliffords (e.g. CNOT), which keep the materialized key+coeff form (see `research/FINDINGS.md`).
        dense_identity: bool,
        /// Whether this layer's gather runs go to `merge::sort_rows_radix_with_scratch` instead of the comparison kernel: at least `merge::RADIX_MIN_REST_STREAMS` rest streams, or at least `merge::RADIX_MIN_DISJOINT_STREAMS` streams whose [`rest_rows_per_key`] is below `merge::RADIX_MAX_REST_ROWS_PER_KEY`. Decided once per layer; see `RADIX_MIN_REST_STREAMS`.
        radix_sort: bool,
    },
    /// Wide rotation: two implicit entries, the identity pass and the generator pass.
    Rotation {
        prep: &'p RotationPrep<W>,
        coord_identity: u32,
        coord_gen: u32,
        /// Whether the generator pass emits here. False only under a partitioning whose partition rows see the generator, where every generator row belongs to a partner instead ([`LayerKnobs::gen_local`]); `coord_gen` is then meaningless and set to 0.
        gen_local: bool,
    },
}

/// Per-layer overrides the partitioned engine hands the coset loop.
/// [`Default`] is the non-partitioned answer to all three, so `apply_layer_bucketed` passes it and nothing about the single-partition path changes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerKnobs<'k> {
    /// Bucket deltas to build the coset span from. `None` (default) means the prepared channel's own [`Prepared::bucket_deltas`]; the partitioned layer passes its plan's local bucket deltas.
    pub bucket_deltas: Option<&'k [u32]>,
    /// Rest-stream count driving the sort-kernel choice. `None` (default) derives it from the plan. The partitioned layer passes the channel's total stream count (local + remote), so every partition picks the kernel the unpartitioned run would.
    pub rest_streams: Option<usize>,
    /// Rest-stream overlap driving the sort-kernel choice, as [`rest_streams`](Self::rest_streams) does for the count. `None` (default) derives it from the plan's own PTM. The partitioned layer passes the unrestricted channel's [`rest_rows_per_key`]: a PTM cut down to local entries can only look more disjoint than the channel is, which could otherwise switch one partition onto the radix kernel that the unpartitioned run keeps on the comparison kernel.
    pub rows_per_key: Option<f64>,
    /// Whether a wide rotation's generator pass emits here (see [`DeltaPlan::Rotation::gen_local`]). `true` by default.
    pub gen_local: bool,
}

impl Default for LayerKnobs<'_> {
    fn default() -> Self {
        Self {
            bucket_deltas: None,
            rest_streams: None,
            rows_per_key: None,
            gen_local: true,
        }
    }
}

/// Rest rows landing on one output key, averaged over the output keys that get at least one —
/// the plan-time estimate of `rows_sorted / distinct keys`, and the second arm of the radix
/// gate (`merge::RADIX_MAX_REST_ROWS_PER_KEY`). `1.0` means the rest streams are pairwise
/// disjoint, so the per-run merge's branch is a coin flip; higher means the streams overlap
/// and the merge steps through them in lock-step, which the branch predictor learns.
/// Must be asked of the channel's own PTM, never a restricted one: a partitioned layer hands
/// [`DeltaPlan::new`] a PTM cut down to its local entries, and a cut-down delta set can only
/// look more disjoint than the channel is — see [`LayerKnobs::rows_per_key`].
pub(super) fn rest_rows_per_key<const W: usize>(ptm: &LocalPtm<W>) -> f64 {
    // Entry 0 is the identity delta when present, and its stream is the pre-sorted `id` columns — never part of the sorted rest stream.
    let rest_start = usize::from(ptm.deltas().first().is_some_and(|d| d.local_delta == 0));
    let dim = 1usize << (2 * ptm.k());
    let (mut rows, mut keys) = (0u32, 0u32);
    for o in 0..dim {
        let n = ptm.deltas()[rest_start..]
            .iter()
            .filter(|d| d.amp[o ^ d.local_delta as usize] != ZERO)
            .count() as u32;
        rows += n;
        keys += u32::from(n > 0);
    }
    if keys == 0 {
        0.0
    } else {
        f64::from(rows) / f64::from(keys)
    }
}

impl<'p, const W: usize> DeltaPlan<'p, W> {
    pub(super) fn new(prep: &'p Prepared<W>, span: &Gf2Span, knobs: LayerKnobs<'_>) -> Self {
        match prep {
            Prepared::Local(ptm) => {
                let coords: Vec<u32> = ptm
                    .deltas()
                    .iter()
                    .map(|d| span.coord_of(d.bucket_delta))
                    .collect();
                let has_identity = ptm.deltas().first().is_some_and(|d| d.local_delta == 0);
                // The identity delta hashes to bucket delta 0, so the id stream stays in its own member.
                debug_assert!(!has_identity || coords[0] == 0);
                // Dense over the active patterns only: `amp` is sized LOCAL_DIM but the channel populates `4^k` entries.
                let dim = 1usize << (2 * ptm.k());
                let dense_identity =
                    has_identity && ptm.deltas()[0].amp[..dim].iter().all(|a| *a != ZERO);
                // `deltas()` is the realized delta set (§Bucketing), so its length minus the identity entry is the stream count the sort-kernel crossover turns on.
                let rest_streams = knobs
                    .rest_streams
                    .unwrap_or(ptm.deltas().len() - has_identity as usize);
                // Both arms read channel-wide quantities, never this partition's view of them, so every partition picks the kernel the unpartitioned run would.
                let radix_sort = rest_streams >= RADIX_MIN_REST_STREAMS
                    || (rest_streams >= RADIX_MIN_DISJOINT_STREAMS
                        && knobs.rows_per_key.unwrap_or_else(|| rest_rows_per_key(ptm))
                            < RADIX_MAX_REST_ROWS_PER_KEY);
                DeltaPlan::Local {
                    ptm,
                    coords,
                    has_identity,
                    dense_identity,
                    radix_sort,
                }
            }
            Prepared::Rotation(r) => DeltaPlan::Rotation {
                prep: r,
                coord_identity: span.coord_of(r.bucket_delta_identity),
                // `coord_of` demands its argument be in the span, and a
                // remote generator's bucket delta is not.
                coord_gen: if knobs.gen_local {
                    span.coord_of(r.bucket_delta_gen)
                } else {
                    0
                },
                gen_local: knobs.gen_local,
            },
        }
    }
}

/// Extra rest-stream rows for an output bucket, supplied by the partitioned engine.
/// A partition generates rows whose output bucket lives on another partition; they arrive here
/// and are appended to that bucket's gather run after the local gather and before the per-run
/// sort. Going into the rest stream (never the pre-sorted id stream) is what makes them safe:
/// it is sorted anyway, so `merge2_into` sees the complete sum before `keep_term` runs
/// (ARCHITECTURE.md §Truncation).
/// Zero-cost when [`NoExtra`]: [`NEEDS_BETA`](Self::NEEDS_BETA) is a `const false` that deletes
/// every call site, along with the inverse-permutation pass that exists only to answer them.
///
/// # Contract
///
/// An implementation that can return rows must set `NEEDS_BETA = true`.
pub(crate) trait ExtraRows<const W: usize> {
    /// Whether the engine must recover each coset member's original bucket index before calling this source. `false` (default) also means the engine never calls [`count`](Self::count) or [`append_into`](Self::append_into).
    const NEEDS_BETA: bool = false;

    /// How many rows are destined for output bucket `beta` (its original index). Used to size the gather run exactly, so it must agree with what [`append_into`](Self::append_into) pushes.
    #[inline]
    fn count(&self, beta: u32) -> usize {
        let _ = beta;
        0
    }

    /// Append bucket `beta`'s rows onto the run's rest columns, all three in step. `beta` is the bucket's original index.
    #[inline]
    fn append_into(
        &self,
        beta: u32,
        x: &mut Vec<[u64; W]>,
        z: &mut Vec<[u64; W]>,
        c: &mut Vec<Complex64>,
    ) {
        let _ = (beta, x, z, c);
    }
}

/// The non-partitioned engine's [`ExtraRows`]: no rows, ever. Every method is
/// the trait default, so the hook compiles away entirely.
pub(crate) struct NoExtra;

impl<const W: usize> ExtraRows<W> for NoExtra {}

/// Apply one prepared channel to a bucketed sum.
/// `policy`'s `keep_term` is folded into the per-bucket merge, so it sees fully summed coefficients (ARCHITECTURE.md §Truncation).
/// `finalize_layer` is not called here; `propagate` owns that.
pub fn apply_layer_bucketed<const W: usize, T>(
    sum: &mut PauliSum<W>,
    prep: &Prepared<W>,
    policy: &T,
    scratch: &mut LayerScratch<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    apply_layer_bucketed_with(sum, prep, policy, scratch, &NoExtra, LayerKnobs::default())
}

/// [`apply_layer_bucketed`] with an [`ExtraRows`] source feeding each output bucket's rest stream and the partitioned engine's per-layer [`LayerKnobs`] — the entry point the partitioned engine drives.
/// Under [`NoExtra`] and default knobs it is [`apply_layer_bucketed`], with the hook's call sites and the inverse-permutation pass behind a `const false`.
pub(crate) fn apply_layer_bucketed_with<const W: usize, T, X>(
    sum: &mut PauliSum<W>,
    prep: &Prepared<W>,
    policy: &T,
    scratch: &mut LayerScratch<W>,
    extra: &X,
    knobs: LayerKnobs<'_>,
) where
    T: TruncationPolicy<W> + ?Sized,
    X: ExtraRows<W> + Sync,
{
    #[cfg(feature = "phase-timing")]
    let mut st = Stamp::now();

    // Key-preserving channels leave every key bitwise unchanged, so the output is already
    // sorted and duplicate-free: rescaling each coefficient is an in-place filter, no sort needed.
    // `!X::NEEDS_BETA` must still be asked: a partitioned layer's local delta table can look
    // key-preserving while received rows still have to be merged in.
    if let Prepared::Local(ptm) = prep {
        if !X::NEEDS_BETA && ptm.is_key_preserving() {
            rescale_in_place(sum, ptm, policy);
            #[cfg(feature = "phase-timing")]
            st.lap(&mut scratch.stats.rescale_ns);
            return;
        }
    }

    // The coset structure of this layer's bucket-delta set: `span(h(D))` rather than `h(D)`
    // itself, since an open-trait channel's delta set need not be XOR-closed.
    let own_deltas;
    let deltas: &[u32] = match knobs.bucket_deltas {
        Some(d) => d,
        None => {
            own_deltas = prep.bucket_deltas();
            &own_deltas
        }
    };
    let span = Gf2Span::new(deltas, sum.hash().bits());
    let plan = DeltaPlan::new(prep, &span, knobs);
    let m = span.coset_size();
    let num_cosets = span.num_cosets();
    #[cfg(feature = "phase-timing")]
    st.lap(&mut scratch.stats.span_plan_ns);

    // Permute the bucket handles into coset-contiguous order: coset `c` owns
    // `staging[c·2^r .. (c+1)·2^r]`, members ascending by basis coordinate. Handles are three
    // `Vec` headers; the term data never moves. At `r = 0` `perm_index` is the identity, so
    // both handle passes are skipped and the chunk loop runs on the buckets directly.
    let identity_perm = span.r() == 0;
    if !identity_perm {
        let buckets = sum.buckets_mut();
        scratch.perm.clear();
        scratch
            .perm
            .extend((0..buckets.len() as u32).map(|beta| span.perm_index(beta)));
        scratch
            .staging
            .resize_with(buckets.len(), BucketCols::default);
        for (beta, cols) in buckets.iter_mut().enumerate() {
            scratch.staging[scratch.perm[beta] as usize] = std::mem::take(cols);
        }
    }
    // The inverse handle permutation, so `fill_coset` can name a member's original bucket.
    // Under `NoExtra` this is a `const false` branch the compiler deletes and `inv_perm` stays
    // empty, which `fill_coset` reads as "the permutation is the identity".
    if X::NEEDS_BETA {
        scratch.inv_perm.clear();
        if !identity_perm {
            scratch.inv_perm.resize(scratch.perm.len(), 0);
            for (beta, &p) in scratch.perm.iter().enumerate() {
                scratch.inv_perm[p as usize] = beta as u32;
            }
        }
    }
    #[cfg(feature = "phase-timing")]
    st.lap(&mut scratch.stats.permute_ns);

    // Each coset is a closed task: it reads and writes only its own chunk, so the chunk loop
    // needs no atomics, no cross-task locks, and no reconciliation pass, and output is
    // byte-identical across thread counts (ARCHITECTURE.md §Determinism).
    {
        // Size the worker pool before `staging` is borrowed below; keeping existing slots
        // preserves their high-water capacity.
        if num_cosets >= MIN_COSETS_FOR_PARALLEL {
            let pool = rayon::current_num_threads().max(1);
            if scratch.workers.len() < pool {
                scratch.workers.resize_with(pool, Mutex::default);
            }
        }
        let workers = &scratch.workers;
        // Empty unless this layer's `ExtraRows` asked for it; `fill_coset` takes empty to
        // mean the identity permutation.
        let inv_perm: &[u32] = &scratch.inv_perm;
        let chunks: &mut [BucketCols<W>] = if identity_perm {
            sum.buckets_mut()
        } else {
            scratch.staging.as_mut_slice()
        };
        if num_cosets < MIN_COSETS_FOR_PARALLEL {
            for (ci, chunk) in chunks.chunks_mut(m).enumerate() {
                let base = ci * m;
                fill_coset::<W, T, X>(
                    chunk,
                    &plan,
                    policy,
                    &mut scratch.task,
                    extra,
                    base,
                    inv_perm,
                );
            }
        } else {
            chunks
                .par_chunks_mut(m)
                .enumerate()
                .for_each(|(ci, chunk)| {
                    let base = ci * m;
                    // Inside `par_chunks_mut` the body always runs on a pool worker, so the
                    // index is present and below the pool size; the fresh-scratch arm is a
                    // defensive fallback only.
                    match rayon::current_thread_index() {
                        Some(i) if i < workers.len() => {
                            let mut ws = workers[i].lock().unwrap();
                            fill_coset::<W, T, X>(
                                chunk, &plan, policy, &mut ws, extra, base, inv_perm,
                            );
                        }
                        _ => {
                            let mut ws = CosetScratch::<W>::default();
                            fill_coset::<W, T, X>(
                                chunk, &plan, policy, &mut ws, extra, base, inv_perm,
                            );
                        }
                    }
                });
        }
    }
    #[cfg(feature = "phase-timing")]
    st.lap(&mut scratch.stats.coset_loop_ns);

    // Un-permute: every handle goes back to its bucket index, leaving the staging slots as
    // empty, capacity-free defaults.
    if !identity_perm {
        let buckets = sum.buckets_mut();
        for (beta, cols) in buckets.iter_mut().enumerate() {
            *cols = std::mem::take(&mut scratch.staging[scratch.perm[beta] as usize]);
        }
    }
    #[cfg(feature = "phase-timing")]
    st.lap(&mut scratch.stats.unpermute_ns);
    sum.recount();
    #[cfg(feature = "phase-timing")]
    st.lap(&mut scratch.stats.recount_ns);

    #[cfg(debug_assertions)]
    sum.assert_invariants();
}

/// In-place coefficient rescale for a key-preserving channel.
///
/// Keys are untouched, so each bucket stays sorted and duplicate-free and no
/// gather, sort or merge is needed. `keep_term` still applies, on the rescaled
/// coefficient, and exact zeros are still dropped — matching the general path.
fn rescale_in_place<const W: usize, T>(sum: &mut PauliSum<W>, ptm: &LocalPtm<W>, policy: &T)
where
    T: TruncationPolicy<W> + ?Sized,
{
    let amp = &ptm.deltas()[0].amp;
    sum.buckets_mut().par_iter_mut().for_each(|cols| {
        let n = cols.len();
        let mut keep = 0usize;
        for i in 0..n {
            let s = ptm.support_bits(&cols.x[i], &cols.z[i]);
            let c = cols.coeff[i] * amp[s];
            if c == ZERO || !policy.keep_term(&cols.x[i], &cols.z[i], c) {
                continue;
            }
            // `keep <= i` always, so this never overwrites an unread slot.
            cols.x[keep] = cols.x[i];
            cols.z[keep] = cols.z[i];
            cols.coeff[keep] = c;
            keep += 1;
        }
        cols.x.truncate(keep);
        cols.z.truncate(keep);
        cols.coeff.truncate(keep);
    });
    sum.recount();

    #[cfg(debug_assertions)]
    sum.assert_invariants();
}

#[cfg(test)]
mod tests;
