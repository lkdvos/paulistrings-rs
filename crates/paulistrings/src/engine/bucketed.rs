//! The bucketed layer engine. See ARCHITECTURE.md §Engine.
//!
//! The unit of work is one **coset** of `span(h(D))` in the bucket-index space
//! (`Gf2Span`): every output bucket in a coset reads only input buckets in
//! that same coset, so a coset is a closed task that can work **in place** —
//! its `2^r` bucket columns are swapped into thread scratch, and the emptied
//! (capacity-retaining) slots become the write destinations. One layer:
//!
//! 1. **Permute** the bucket *handles* into coset-contiguous order
//!    (`Gf2Span::perm_index`); two `O(B)` handle moves bracket the layer.
//! 2. Per coset: **swap** the member columns into scratch, **size** each
//!    per-member gather run exactly from the swapped-out lengths, **gather**
//!    input-member-major — each term is loaded once and its whole fanout is
//!    scattered to runs by the O(1) index identity
//!    `member(i) ⊕ δ = member(i ⊕ coord(δ))` — then per run **sort** by key
//!    alone and **merge** straight into the member's live slot. When the
//!    identity delta's amplitude never vanishes (dense: every rotation and
//!    general unitary) the id stream's keys are the source bucket's keys row
//!    for row, so the gather materializes only the 16-byte coefficients and
//!    the merge borrows the key columns in place.
//! 3. Un-permute the handles, recount, assert invariants.
//!
//! The gather visits each input term exactly once, and there is no second
//! full-size buffer: peak memory is `n` plus per-worker scratch of one
//! coset's working set.
//!
//! Determinism (ARCHITECTURE.md §Determinism): cosets are write-disjoint and
//! work within one is sequential, so output is bitwise identical across
//! thread counts *and* across repeat runs at a fixed bucket count and hash
//! seed — `sort_rows_with_scratch`'s key-only sort is a deterministic
//! function of its input, even though equal-key order is unspecified.
//! Across bucket counts or hash seeds, output agrees only to
//! floating-point tolerance: a different partition can gather equal-key
//! contributions in a different order, and `f64` addition is not associative.

use std::sync::Mutex;

use num_complex::Complex64;
use rayon::prelude::*;

use super::coset::Gf2Span;
use super::merge::{merge2_into, sort_rows_with_scratch, SortScratch};
use crate::bucket::sum::{BucketCols, PauliSum};
use crate::channel::prepared::{LocalPtm, Prepared, RotationPrep};
use crate::pauli_string::PauliString;
use crate::phase::Phase;
use crate::truncation::TruncationPolicy;

#[cfg(feature = "phase-timing")]
use super::stats::{CosetStats, PhaseStats, Stamp};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Reusable per-layer scratch.
///
/// Held by the caller across layers because a layer must allocate nothing
/// after the first: every field retains its high-water capacity
/// across cosets and layers. The serial path uses the caller's instance
/// directly; the parallel path takes one slot of `workers` per Rayon worker
/// thread, so scratch capacity is bounded by `threads × coset working set`
/// and survives across cosets, layers, and `propagate` calls. (Rayon's
/// `for_each_init` would instead construct its init value once per *split* —
/// many times per layer — which reallocated these MB-scale buffers over and
/// over; that churn measured as a 20–50% per-layer regression.)
///
/// A task's output cannot depend on which scratch slot it drew: the swap
/// site clears every write destination before use, and gather runs reset on
/// take — so worker→slot assignment varying run to run is unobservable,
/// which is what keeps output byte-identical across thread counts.
#[derive(Debug, Default)]
pub struct LayerScratch<const W: usize> {
    /// The per-coset working set (serial path).
    task: CosetScratch<W>,
    /// The layer's handle permutation, `perm[β] = span.perm_index(β)`.
    perm: Vec<u32>,
    /// Staging area the bucket handles are permuted into. Holds handles only
    /// while a layer runs; its elements carry no capacity of their own.
    staging: Vec<BucketCols<W>>,
    /// Worker-persistent coset working sets for the parallel path, one slot
    /// per Rayon worker, indexed by `rayon::current_thread_index()`. Each
    /// worker locks only its own slot, so the mutexes are uncontended; they
    /// exist to make the shared borrow safe, not to arbitrate.
    workers: Vec<Mutex<CosetScratch<W>>>,
    /// Layer-level (wall-clock) phase counters; the per-coset busy-time
    /// counters live in each `CosetScratch`.
    #[cfg(feature = "phase-timing")]
    pub(crate) stats: PhaseStats,
    /// The opt-in per-layer term-count trace, `None` unless
    /// [`Self::enable_term_trace`] was called. Written only by
    /// `propagate_with_scratch`'s per-layer epilogue — nothing in this module
    /// touches it, so it costs the layer nothing.
    pub(crate) term_trace: Option<TermTrace>,
}

impl<const W: usize> LayerScratch<W> {
    /// An empty scratch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drain and return the accumulated phase counters: the layer-level
    /// wall-clock fields plus every worker's busy-time counters, all zeroed
    /// afterwards.
    ///
    /// Call between measured regions; counters accumulate across layers and
    /// `propagate_with_scratch` calls until drained. Caveat: the defensive
    /// fresh-scratch arm of the parallel coset loop (a worker with no pool
    /// index) drops its counters — that arm is unreachable in practice.
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

    /// Start recording a [`TermTrace`] on every subsequent
    /// [`propagate_with_scratch`](crate::propagate_with_scratch) call driven
    /// by this scratch. Idempotent, and it never discards counts already
    /// recorded.
    ///
    /// Unlike `take_stats` this is always compiled: the
    /// counts come from the `sum.len()` reads the layer loop already performs,
    /// so recording them is two `usize` pushes per *layer* on the calling
    /// thread — no clock, no per-term work, nothing inside the coset loop.
    pub fn enable_term_trace(&mut self) {
        self.term_trace.get_or_insert_with(TermTrace::default);
    }

    /// Drain and return the per-layer term counts, or `None` if tracing was
    /// never enabled (`Some` ⟺ tracing is on).
    ///
    /// Draining leaves tracing *enabled* with empty vectors, so a scratch
    /// reused across calls (a Trotter driver stepping an observable) reports
    /// each call separately without re-enabling; counts accumulate across
    /// layers and calls until drained.
    pub fn take_term_trace(&mut self) -> Option<TermTrace> {
        self.term_trace.as_mut().map(std::mem::take)
    }
}

/// Per-layer resident term counts, recorded by
/// [`propagate_with_scratch`](crate::propagate_with_scratch) when the
/// driving [`LayerScratch`] has [`enable_term_trace`](LayerScratch::enable_term_trace)
/// set. Both vectors have one entry per layer applied, in application order
/// (so *reverse* circuit order under [`Direction::Heisenberg`](crate::Direction)).
///
/// Always compiled — the `phase-timing` feature gates the *timing* counters
/// (`PhaseStats`), not these counts.
///
/// # What is *not* here
///
/// These are the counts of the sum as it rests between layers: `terms_in[k]`
/// is read before layer `k` starts, `terms_out[k]` after that layer's
/// `finalize_layer`, i.e. **post-truncation**. The transient in-layer
/// expansion — the sum after a channel's fanout but before the merge
/// deduplicates and the policy filters — is deliberately not captured:
/// observing it would mean instrumenting the coset loop, which is where the
/// engine's time goes. Peak *memory* is a harness-level measurement
/// (`/proc/self/status`), not this struct's job.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TermTrace {
    /// Resident term count before each layer. `terms_in[k + 1]` equals
    /// `terms_out[k]`, so the whole trace is the sequence
    /// `terms_in[0], terms_out[0], terms_out[1], …`.
    pub terms_in: Vec<usize>,
    /// Resident term count after each layer, i.e. after the truncation
    /// policy's `finalize_layer`.
    pub terms_out: Vec<usize>,
}

impl TermTrace {
    /// Peak *resident* term count: `max(terms_in[0], terms_out…)` — the
    /// largest the sum ever was between layers (see the type's "What is not
    /// here"). `None` for a zero-layer trace, where the resident count never
    /// changed and only the caller knows it.
    pub fn peak_terms(&self) -> Option<usize> {
        self.terms_in
            .first()
            .copied()
            .into_iter()
            .chain(self.terms_out.iter().copied())
            .max()
    }
}

/// One coset task's working set: the swapped-out input columns and the
/// per-output-member gather runs.
#[derive(Clone, Debug, Default)]
struct CosetScratch<const W: usize> {
    /// The coset's input columns, one slot per member, `mem::swap`ped with the
    /// live bucket slots. After the swap the live slots hold these slots'
    /// previous — cleared, capacity-retaining — columns, which is what makes
    /// the layer in-place: bucket capacity circulates through here instead of
    /// through a second full-sum copy.
    old: Vec<BucketCols<W>>,
    /// Per-output-member gather runs.
    runs: Vec<GatherRun<W>>,
    /// Scratch for `sort_rows_with_scratch`'s per-run sort, reused across
    /// every run in every coset this scratch instance handles.
    sort: SortScratch<W>,
    /// This slot's busy-time phase counters, drained by
    /// `LayerScratch::take_stats`.
    #[cfg(feature = "phase-timing")]
    stats: CosetStats,
}

/// One output member's gather run: key columns and coefficients.
///
/// Equal-key summation order is not pinned by a sort tiebreak — see the
/// module doc and `merge::sort_rows_with_scratch`.
#[derive(Clone, Debug, Default)]
struct GatherRun<const W: usize> {
    /// The identity-delta stream: keys untouched, so it inherits the
    /// source bucket's strictly-ascending, duplicate-free order and is never
    /// sorted. `H·0 = 0` puts this stream in the member's own run, in source
    /// position order. Under a **dense** identity plan only
    /// `id_coeff` is populated — one coefficient per source row, aligned 1:1
    /// with `old[j]`, whose key columns the merge borrows in place — and
    /// `id_x`/`id_z` stay empty. Under a sparse plan all three columns are
    /// filled with the zero-amplitude rows filtered out.
    id_x: Vec<[u64; W]>,
    id_z: Vec<[u64; W]>,
    id_coeff: Vec<Complex64>,
    /// Every other delta's rows — keys XOR'd by a constant mask, so generally
    /// unsorted; canonicalized per run by `sort_rows_with_scratch`.
    x: Vec<[u64; W]>,
    z: Vec<[u64; W]>,
    coeff: Vec<Complex64>,
}

impl<const W: usize> GatherRun<W> {
    #[inline]
    fn reset(&mut self, cap_id_keys: usize, cap_id_coeff: usize, cap_rest: usize) {
        self.id_x.clear();
        self.id_z.clear();
        self.id_coeff.clear();
        self.x.clear();
        self.z.clear();
        self.coeff.clear();
        if self.id_x.capacity() < cap_id_keys {
            let extra = cap_id_keys - self.id_x.capacity();
            self.id_x.reserve(extra);
            self.id_z.reserve(extra);
        }
        if self.id_coeff.capacity() < cap_id_coeff {
            self.id_coeff
                .reserve(cap_id_coeff - self.id_coeff.capacity());
        }
        if self.x.capacity() < cap_rest {
            let extra = cap_rest - self.x.capacity();
            self.x.reserve(extra);
            self.z.reserve(extra);
            self.coeff.reserve(extra);
        }
    }

    #[inline]
    fn push_id(&mut self, x: [u64; W], z: [u64; W], c: Complex64) {
        self.id_x.push(x);
        self.id_z.push(z);
        self.id_coeff.push(c);
    }

    #[inline]
    fn push(&mut self, x: [u64; W], z: [u64; W], c: Complex64) {
        self.x.push(x);
        self.z.push(z);
        self.coeff.push(c);
    }

    #[cfg(any(test, feature = "phase-timing"))]
    #[inline]
    fn len(&self) -> usize {
        self.id_coeff.len() + self.coeff.len()
    }
}

/// A prepared channel's delta set, annotated with each entry's coset
/// coordinate (`span.coord_of(bucket_delta)`), computed once per layer.
enum DeltaPlan<'p, const W: usize> {
    /// Tabulated deltas; `coords[e]` pairs with `ptm.deltas()[e]`.
    Local {
        ptm: &'p LocalPtm<W>,
        coords: Vec<u32>,
        /// Whether `deltas()[0]` is the identity delta (`local_delta == 0` —
        /// entry 0 by the ascending construction order), whose stream the
        /// gather routes into the run's pre-sorted `id` columns.
        /// True for every built-in channel; a custom channel without it
        /// gathers everything into the sorted rest stream.
        has_identity: bool,
        /// Whether the identity entry's amplitude is nonzero for **every**
        /// active support pattern. Dense means each source row
        /// emits exactly one id row with its key untouched — the id stream's
        /// keys *are* the source bucket's key columns, row for row — so the
        /// gather materializes only the 16-byte coefficient into `id_coeff`
        /// and the merge borrows the keys from `old[j]` in place, saving the
        /// 32-byte-per-row key write + re-read. True for
        /// `GeneralUnitary1Q/2Q` and weight-≤2 rotations; false for
        /// Cliffords (e.g. CNOT's id amp is nonzero on 4 of 16 patterns),
        /// which keep the pre-filtered key+coeff materialization —
        /// borrowing there would make the merge scan mostly-skipped rows,
        /// measured +15–30% on cnot/h (`research/notes/2026-08-31-v0.6-results.md`).
        dense_identity: bool,
    },
    /// Wide rotation: two implicit entries, the identity pass and the
    /// generator pass.
    Rotation {
        prep: &'p RotationPrep<W>,
        coord_identity: u32,
        coord_gen: u32,
    },
}

impl<'p, const W: usize> DeltaPlan<'p, W> {
    fn new(prep: &'p Prepared<W>, span: &Gf2Span) -> Self {
        match prep {
            Prepared::Local(ptm) => {
                let coords: Vec<u32> = ptm
                    .deltas()
                    .iter()
                    .map(|d| span.coord_of(d.bucket_delta))
                    .collect();
                let has_identity = ptm.deltas().first().is_some_and(|d| d.local_delta == 0);
                // The identity delta hashes to bucket delta 0, whose coset
                // coordinate is 0 — the id stream stays in its own member.
                debug_assert!(!has_identity || coords[0] == 0);
                // Dense over the *active* patterns only: `amp` is sized
                // LOCAL_DIM but the channel populates `4^k` entries.
                let dim = 1usize << (2 * ptm.k());
                let dense_identity =
                    has_identity && ptm.deltas()[0].amp[..dim].iter().all(|a| *a != ZERO);
                DeltaPlan::Local {
                    ptm,
                    coords,
                    has_identity,
                    dense_identity,
                }
            }
            Prepared::Rotation(r) => DeltaPlan::Rotation {
                prep: r,
                coord_identity: span.coord_of(r.bucket_delta_identity),
                coord_gen: span.coord_of(r.bucket_delta_gen),
            },
        }
    }
}

/// Apply one prepared channel to a bucketed sum.
///
/// `policy`'s `keep_term` is folded into the per-bucket merge, so it sees fully
/// **summed** coefficients (ARCHITECTURE.md §Truncation).
/// `finalize_layer` is *not* called here; `propagate` owns that.
pub fn apply_layer_bucketed<const W: usize, T>(
    sum: &mut PauliSum<W>,
    prep: &Prepared<W>,
    policy: &T,
    scratch: &mut LayerScratch<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    #[cfg(feature = "phase-timing")]
    let mut st = Stamp::now();

    // Key-preserving channels (identity, depolarizing, dephasing, Pauli gates)
    // leave every key bitwise unchanged, so the output is already sorted and
    // duplicate-free: multiplying each coefficient by a scalar is an
    // in-place filter, with no sort needed.
    if let Prepared::Local(ptm) = prep {
        if ptm.is_key_preserving() {
            rescale_in_place(sum, ptm, policy);
            #[cfg(feature = "phase-timing")]
            st.lap(&mut scratch.stats.rescale_ns);
            return;
        }
    }

    // The coset structure of this layer's bucket-delta set. `span(h(D))`
    // rather than `h(D)` itself: an open-trait channel's delta set need not be
    // XOR-closed, and only the span's cosets are guaranteed to partition.
    let span = Gf2Span::new(&prep.bucket_deltas(), sum.hash().bits());
    let plan = DeltaPlan::new(prep, &span);
    let m = span.coset_size();
    let num_cosets = span.num_cosets();
    #[cfg(feature = "phase-timing")]
    st.lap(&mut scratch.stats.span_plan_ns);

    // Permute the bucket *handles* into coset-contiguous order: coset `c`
    // owns `staging[c·2^r .. (c+1)·2^r]`, members ascending by basis
    // coordinate. Handles are three `Vec` headers; the term data never moves.
    // At `r = 0` every coset is a single bucket and `perm_index` is the
    // identity (`rank_of_rep` compresses over every bit), so the two handle
    // passes are skipped and the chunk loop runs on the buckets directly.
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
    #[cfg(feature = "phase-timing")]
    st.lap(&mut scratch.stats.permute_ns);

    // Each coset is a closed task: it reads and writes only its own chunk, so
    // the chunk loop needs no atomics, no cross-task locks, and no
    // reconciliation pass. Work within a task is sequential and deterministic
    // (the per-run key-only sort is a deterministic function of its input),
    // so output is byte-identical across thread counts
    // (ARCHITECTURE.md §Determinism).
    {
        // Size the worker pool before `staging` is borrowed below; keeping
        // existing slots preserves their high-water capacity.
        if num_cosets >= MIN_COSETS_FOR_PARALLEL {
            let pool = rayon::current_num_threads().max(1);
            if scratch.workers.len() < pool {
                scratch.workers.resize_with(pool, Mutex::default);
            }
        }
        let workers = &scratch.workers;
        let chunks: &mut [BucketCols<W>] = if identity_perm {
            sum.buckets_mut()
        } else {
            scratch.staging.as_mut_slice()
        };
        if num_cosets < MIN_COSETS_FOR_PARALLEL {
            for chunk in chunks.chunks_mut(m) {
                fill_coset::<W, T>(chunk, &plan, policy, &mut scratch.task);
            }
        } else {
            chunks.par_chunks_mut(m).for_each(|chunk| {
                // Inside `par_chunks_mut` the body always runs on a pool
                // worker, so the index is present and below the pool size;
                // the fresh-scratch arm is a defensive fallback only.
                match rayon::current_thread_index() {
                    Some(i) if i < workers.len() => {
                        let mut ws = workers[i].lock().unwrap();
                        fill_coset::<W, T>(chunk, &plan, policy, &mut ws);
                    }
                    _ => {
                        let mut ws = CosetScratch::<W>::default();
                        fill_coset::<W, T>(chunk, &plan, policy, &mut ws);
                    }
                }
            });
        }
    }
    #[cfg(feature = "phase-timing")]
    st.lap(&mut scratch.stats.coset_loop_ns);

    // Un-permute: every handle goes back to its bucket index, leaving the
    // staging slots as empty, capacity-free defaults.
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

/// Below this many cosets there is nothing to spread, so skip Rayon entirely.
///
/// `desired_bits` already gives a small sum few buckets, so this mostly catches
/// the `bits = 0` case (or `r = bits`), where one coset spans every bucket and
/// the layer degenerates to a single whole-sum task on the same code path.
const MIN_COSETS_FOR_PARALLEL: usize = 2;

/// Gather, sort and merge one coset, in place. The unit of parallel work.
///
/// `chunk` holds the coset's `2^r` bucket columns, members ascending by basis
/// coordinate, serving as both input source and output destination.
fn fill_coset<const W: usize, T>(
    chunk: &mut [BucketCols<W>],
    plan: &DeltaPlan<'_, W>,
    policy: &T,
    ws: &mut CosetScratch<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    let m = chunk.len();
    #[cfg(feature = "phase-timing")]
    let CosetScratch {
        old,
        runs,
        sort,
        stats,
    } = ws;
    #[cfg(not(feature = "phase-timing"))]
    let CosetScratch { old, runs, sort } = ws;
    #[cfg(feature = "phase-timing")]
    let mut st = Stamp::now();
    old.resize_with(m, BucketCols::default);
    runs.resize_with(m, GatherRun::default);

    // Swap the coset's columns out. The chunk slots inherit this scratch's
    // cleared, capacity-retaining columns and become the write destinations —
    // capacities circulate between buckets across cosets, which holds the
    // steady state allocation-free in aggregate.
    for (slot, cols) in chunk.iter_mut().zip(old.iter_mut()) {
        std::mem::swap(slot, cols);
        slot.clear();
    }
    #[cfg(feature = "phase-timing")]
    st.lap(&mut stats.swap_ns);

    // Exact per-run capacity, counted once per delta entry — two entries
    // colliding on one bucket delta count twice, matching the rows they can
    // emit. Split by destination stream: the identity entry feeds the
    // pre-sorted `id` columns, everything else the sorted rest.
    // Under a dense identity the id *key* columns stay empty —
    // the merge borrows the source bucket's keys — so only `id_coeff` needs
    // capacity; a rotation's id stream is dense by construction (every row
    // emits exactly one id row).
    for (j, run) in runs.iter_mut().enumerate() {
        let (cap_id_keys, cap_id_coeff, cap_rest): (usize, usize, usize) = match plan {
            DeltaPlan::Local {
                coords,
                has_identity,
                dense_identity,
                ..
            } => {
                let mut id = 0usize;
                let mut rest = 0usize;
                for (e, &c) in coords.iter().enumerate() {
                    let l = old[j ^ c as usize].len();
                    if *has_identity && e == 0 {
                        id += l;
                    } else {
                        rest += l;
                    }
                }
                (if *dense_identity { 0 } else { id }, id, rest)
            }
            DeltaPlan::Rotation {
                coord_identity,
                coord_gen,
                ..
            } => (
                0,
                old[j ^ *coord_identity as usize].len(),
                old[j ^ *coord_gen as usize].len(),
            ),
        };
        run.reset(cap_id_keys, cap_id_coeff, cap_rest);
    }
    #[cfg(feature = "phase-timing")]
    st.lap(&mut stats.size_ns);

    // Gather. Two visit orders produce the same multiset of rows per run —
    // only their arrival order differs, which the key-only sort below erases
    // up to floating-point tolerance on any equal-key summation (see
    // `local_gather_orders_agree_to_fp_tolerance`). Which order is *faster*
    // depends on `r`: input-major loads each term once but keeps `2^r` write
    // streams open per task, and at `r = 4` those streams plus the swapped
    // coset no longer fit L2 — measured +48% on a 32-thread
    // `GeneralUnitary2Q` layer at 10⁶ terms. Output-major re-reads each input
    // bucket `2^r` times but the reads stay coset-local, with a single write
    // stream. Only `Local` plans can reach `r ≥ 3` (a wide rotation has at
    // most two bucket deltas).
    match plan {
        DeltaPlan::Local {
            ptm,
            coords,
            has_identity,
            dense_identity,
        } => {
            if m >= 1 << GATHER_OUTPUT_MAJOR_MIN_R {
                gather_local_output_major(old, runs, ptm, coords, *has_identity, *dense_identity);
            } else {
                gather_local_input_major(old, runs, ptm, coords, *has_identity, *dense_identity);
            }
        }
        DeltaPlan::Rotation {
            prep,
            coord_identity,
            coord_gen,
        } => {
            // The identity pass and the generator pass. `cos`/`sin` stay
            // hoisted; the `i^k` phase depends on 2w support bits and is
            // computed per anticommuting term, exactly as before. Every term
            // emits exactly one identity-pass row (full coefficient when it
            // commutes, `cos`-scaled when it doesn't — kept even when
            // `cos == 0`, see `merge2_into` on signed zeros), so the id
            // stream is the whole source bucket in order: sorted, unique —
            // and its keys are the source keys row for row, so only the
            // coefficient is materialized; the merge borrows the keys from
            // the source bucket in place.
            for (i, src) in old.iter().enumerate() {
                for t in 0..src.len() {
                    let v = PauliString::<W> {
                        x: src.x[t],
                        z: src.z[t],
                    };
                    if v.commutes_with(&prep.gen) {
                        runs[i ^ *coord_identity as usize]
                            .id_coeff
                            .push(src.coeff[t]);
                    } else {
                        runs[i ^ *coord_identity as usize]
                            .id_coeff
                            .push(src.coeff[t] * prep.cos);
                        let mut prod = v;
                        let phase = prod.mul_assign(&prep.gen);
                        let total = Phase::I + phase;
                        runs[i ^ *coord_gen as usize].push(
                            prod.x,
                            prod.z,
                            total.apply(src.coeff[t]) * prep.sin,
                        );
                    }
                }
            }
        }
    }
    #[cfg(feature = "phase-timing")]
    st.lap(&mut stats.gather_ns);

    // Sort each run's rest stream by key alone, then fuse the two-stream
    // merge with the segmented reduction into the member's live slot: the id
    // stream never moves through the sort at all. Under a dense
    // identity plan the id stream's *keys* were never materialized either —
    // they are the source bucket's own key columns, borrowed here, with the
    // gathered `id_coeff` aligned to them row for row; `H·0 = 0`
    // means member `j`'s id source is `old[j]`.
    for (j, (run, dst)) in runs.iter_mut().zip(chunk.iter_mut()).enumerate() {
        #[cfg(feature = "phase-timing")]
        {
            stats.rows_gathered += run.len() as u64;
            stats.rows_sorted += run.coeff.len() as u64;
        }
        sort_rows_with_scratch(&mut run.x, &mut run.z, &mut run.coeff, sort);
        #[cfg(feature = "phase-timing")]
        st.lap(&mut stats.sort_ns);
        let (a_x, a_z): (&[[u64; W]], &[[u64; W]]) = match plan {
            DeltaPlan::Local {
                dense_identity: true,
                ..
            } => {
                let src = &old[j];
                debug_assert_eq!(src.len(), run.id_coeff.len());
                #[cfg(feature = "phase-timing")]
                {
                    stats.rows_id += src.len() as u64;
                }
                (&src.x, &src.z)
            }
            DeltaPlan::Local { .. } => (&run.id_x, &run.id_z),
            DeltaPlan::Rotation { coord_identity, .. } => {
                let src = &old[j ^ *coord_identity as usize];
                debug_assert_eq!(src.len(), run.id_coeff.len());
                #[cfg(feature = "phase-timing")]
                {
                    stats.rows_id += src.len() as u64;
                }
                (&src.x, &src.z)
            }
        };
        merge2_into::<W, T>(
            a_x,
            a_z,
            &run.id_coeff,
            &run.x,
            &run.z,
            &run.coeff,
            &mut dst.x,
            &mut dst.z,
            &mut dst.coeff,
            policy,
        );
        #[cfg(feature = "phase-timing")]
        st.lap(&mut stats.merge_ns);
    }

    // Leave `old` cleared so the next coset's swap hands its chunk clean,
    // capacity-retaining columns. Runs are cleared by their own `reset`.
    for cols in old.iter_mut() {
        cols.clear();
    }
    #[cfg(feature = "phase-timing")]
    {
        st.lap(&mut stats.clear_ns);
        stats.cosets += 1;
        stats.runs += m as u64;
    }
}

/// Coset dimension at or above which the gather switches to output-major.
///
/// Measured at `r = 2` (both `Clifford2Q` and `GeneralUnitary2Q` — a 2Q
/// channel whose delta masks have Pauli structure, like sqrt-SWAP's
/// `{XX, ZZ, YY}`, spans only rank 2): output-major *loses* 14–22% at 10⁶
/// terms, because re-reading each input bucket `2^r` times costs more than
/// input-major's `2^r` open write streams. So every built-in channel takes
/// the input-major path. The output-major branch survives, unmeasured, as a
/// guard for a custom full-rank channel (`r = 4`: sixteen live gather runs
/// per task), where the scatter working set doubles twice more. Both paths
/// gather the identical multiset of rows in different orders; the key-only
/// sort does not canonicalize that to a bitwise-identical sequence
/// (equal-key order can differ between the two), so the two orders
/// agree only to floating-point tolerance — pinned by
/// `local_gather_orders_agree_to_fp_tolerance` — and the threshold remains a
/// pure performance knob, not a correctness one.
const GATHER_OUTPUT_MAJOR_MIN_R: u8 = 3;

/// Input-major gather for a tabulated (`Local`) plan: each term is loaded
/// once and its whole fanout is scattered by
/// `member(i) ⊕ δ = member(i ⊕ coord(δ))`. Rows land in the runs in
/// (input member, input position, delta) order.
fn gather_local_input_major<const W: usize>(
    old: &[BucketCols<W>],
    runs: &mut [GatherRun<W>],
    ptm: &LocalPtm<W>,
    coords: &[u32],
    has_identity: bool,
    dense_identity: bool,
) {
    let rest_start = has_identity as usize;
    for (i, src) in old.iter().enumerate() {
        for t in 0..src.len() {
            let s = ptm.support_bits(&src.x[t], &src.z[t]);
            if has_identity {
                // Entry 0 is the identity delta: masks are zero and
                // `coords[0] == 0`, so the row lands in this member's own run
                // with its key untouched — the pre-sorted id stream.
                let a = ptm.deltas()[0].amp[s];
                if dense_identity {
                    // Dense: `a` never vanishes and the stream is 1:1 with
                    // the source rows, so only the coefficient is stored —
                    // the merge borrows the keys from `old[i]`.
                    debug_assert!(a != ZERO);
                    runs[i].id_coeff.push(src.coeff[t] * a);
                } else if a != ZERO {
                    runs[i].push_id(src.x[t], src.z[t], src.coeff[t] * a);
                }
            }
            for (e, d) in ptm.deltas().iter().enumerate().skip(rest_start) {
                let a = d.amp[s];
                if a == ZERO {
                    continue;
                }
                let mut kx = src.x[t];
                let mut kz = src.z[t];
                for w in 0..W {
                    kx[w] ^= d.mask_x[w];
                    kz[w] ^= d.mask_z[w];
                }
                runs[i ^ coords[e] as usize].push(kx, kz, src.coeff[t] * a);
            }
        }
    }
}

/// Output-major gather for a tabulated (`Local`) plan: for each output member,
/// stream the one input bucket per delta entry and append to that member's run
/// only. Rows land in (delta, input position) order — the same *multiset* as
/// [`gather_local_input_major`] in a different order. The per-run sort is
/// key-only, so it does not canonicalize the two orders to an identical
/// sequence; they agree only up to floating-point tolerance on
/// any equal-key summation (see `local_gather_orders_agree_to_fp_tolerance`).
fn gather_local_output_major<const W: usize>(
    old: &[BucketCols<W>],
    runs: &mut [GatherRun<W>],
    ptm: &LocalPtm<W>,
    coords: &[u32],
    has_identity: bool,
    dense_identity: bool,
) {
    let rest_start = has_identity as usize;
    for (j, run) in runs.iter_mut().enumerate() {
        if has_identity {
            // Entry 0: masks zero, `coords[0] == 0` — the member's own bucket
            // streams into the pre-sorted id columns; coefficient
            // only when the identity is dense (keys borrowed).
            let d = &ptm.deltas()[0];
            let src = &old[j];
            for t in 0..src.len() {
                let s = ptm.support_bits(&src.x[t], &src.z[t]);
                let a = d.amp[s];
                if dense_identity {
                    debug_assert!(a != ZERO);
                    run.id_coeff.push(src.coeff[t] * a);
                } else if a != ZERO {
                    run.push_id(src.x[t], src.z[t], src.coeff[t] * a);
                }
            }
        }
        for (e, d) in ptm.deltas().iter().enumerate().skip(rest_start) {
            let src = &old[j ^ coords[e] as usize];
            for t in 0..src.len() {
                let s = ptm.support_bits(&src.x[t], &src.z[t]);
                let a = d.amp[s];
                if a == ZERO {
                    continue;
                }
                let mut kx = src.x[t];
                let mut kz = src.z[t];
                for w in 0..W {
                    kx[w] ^= d.mask_x[w];
                    kz[w] ^= d.mask_z[w];
                }
                run.push(kx, kz, src.coeff[t] * a);
            }
        }
    }
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
mod tests {
    use super::*;
    use crate::accumulator::BuildAccumulator;
    use crate::bucket::hash::Gf2Hash;
    use crate::channel::clifford::{Clifford1Q, Clifford2Q};
    use crate::channel::identity::IdentityChannel;
    use crate::channel::noise::{AmplitudeDamping, Dephasing, Depolarizing};
    use crate::channel::rotation::PauliRotation;
    use crate::channel::Channel;
    use crate::pauli_sum::PauliSum;
    use crate::truncation::builtin::{And, CoefficientThreshold, WeightCutoff};

    // The differential oracle and the shared fixtures live in
    // `crate::test_support`; re-exported here so the sibling test modules keep
    // reaching them through `super::tests::…`.
    pub(super) use crate::test_support::{
        assert_same_terms, assert_terms_close, canonical_triples, naive_apply_layer, rand_sum,
    };

    const TOL: f64 = 1e-11;

    pub(super) struct AlwaysKeep;
    impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

    /// The term trace's state machine, independent of any propagation:
    /// `None` ⟺ off, `enable` is idempotent and non-destructive, `take`
    /// drains but stays on. What the counts *mean* is pinned by
    /// `tests/term_trace.rs`, which drives the layer loop that writes them.
    #[test]
    fn term_trace_is_opt_in_and_drains_on_take() {
        let mut scratch = LayerScratch::<1>::new();
        assert!(scratch.take_term_trace().is_none(), "off by default");

        scratch.enable_term_trace();
        scratch.term_trace.as_mut().unwrap().terms_in.push(7);
        scratch.enable_term_trace(); // idempotent: must not clear the 7
        assert_eq!(
            scratch.take_term_trace(),
            Some(TermTrace {
                terms_in: vec![7],
                terms_out: vec![],
            })
        );
        assert_eq!(scratch.take_term_trace(), Some(TermTrace::default()));
    }

    /// `peak_terms` is the between-layer resident maximum, which lives in
    /// `terms_out` except for a first layer that only ever shrinks.
    #[test]
    fn peak_terms_spans_the_first_input_and_every_output() {
        assert_eq!(TermTrace::default().peak_terms(), None);
        assert_eq!(
            TermTrace {
                terms_in: vec![9, 4],
                terms_out: vec![4, 6],
            }
            .peak_terms(),
            Some(9),
            "a shrinking first layer keeps the input as the peak"
        );
        assert_eq!(
            TermTrace {
                terms_in: vec![1, 5],
                terms_out: vec![5, 3],
            }
            .peak_terms(),
            Some(5)
        );
    }

    /// Run one layer through the bucketed engine, converting in and out.
    pub(super) fn bucketed_layer<const W: usize, C, T>(
        input: &PauliSum<W>,
        ch: &C,
        policy: &T,
        adjoint: bool,
        bits: u8,
        seed: u64,
    ) -> PauliSum<W>
    where
        C: Channel<W> + ?Sized,
        T: TruncationPolicy<W> + ?Sized,
    {
        let hash = Gf2Hash::<W>::new(input.num_qubits(), bits, seed);
        let mut b = input.clone().with_hash(hash);
        let prep = ch
            .prepare(b.hash(), adjoint)
            .expect("channel could not be prepared");
        let mut scratch = LayerScratch::<W>::new();
        apply_layer_bucketed(&mut b, &prep, policy, &mut scratch);
        b
    }

    /// Keys must match exactly; coefficients only to tolerance — see
    /// [`assert_terms_close`].
    pub(super) fn assert_sums_close<const W: usize>(
        got: &PauliSum<W>,
        want: &PauliSum<W>,
        what: &str,
    ) {
        assert_terms_close(got, want, TOL, what);
    }

    // ---- hand-checked behaviour ----

    #[test]
    fn h_conjugates_z_to_x() {
        let mut acc = BuildAccumulator::<1>::with_capacity(4, 1);
        acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
        let input = acc.finalize();
        let out = bucketed_layer(&input, &Clifford1Q::h(0), &AlwaysKeep, false, 4, 0x1);
        assert_eq!(out.len(), 1);
        let (x, z, c) = out.iter().next().unwrap();
        assert_eq!(*x, [1]);
        assert_eq!(*z, [0]);
        assert!((c - Complex64::new(1.0, 0.0)).norm() < TOL);
    }

    #[test]
    fn cnot_propagates_z_on_the_control() {
        let mut acc = BuildAccumulator::<1>::with_capacity(4, 1);
        acc.add_term(PauliString::<1>::z(1), Phase::ONE, Complex64::new(1.0, 0.0));
        let input = acc.finalize();
        // I⊗Z under CNOT(0 -> 1) becomes Z⊗Z.
        let out = bucketed_layer(&input, &Clifford2Q::cnot(0, 1), &AlwaysKeep, false, 4, 0x1);
        assert_eq!(out.len(), 1);
        let (x, z, _) = out.iter().next().unwrap();
        assert_eq!(*z, [0b11]);
        assert_eq!(*x, [0]);
    }

    #[test]
    fn a_rotation_fans_out_to_two_terms() {
        let mut acc = BuildAccumulator::<1>::with_capacity(4, 1);
        acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(1.0, 0.0));
        let input = acc.finalize();
        let rot = PauliRotation::new(PauliString::<1>::z(0), std::f64::consts::FRAC_PI_3);
        let out = bucketed_layer(&input, &rot, &AlwaysKeep, false, 4, 0x1);
        // cos(pi/3)*X + sin(pi/3)*(i * X * Z) = 0.5*X - 0.866*Y
        assert_eq!(out.len(), 2);
        let want = naive_apply_layer(&input, &rot, &AlwaysKeep, false);
        assert_sums_close(&out, &want, "rotation fanout");
    }

    // ---- the differential test against the naive oracle ----

    /// Every built-in channel, over both occupancy regimes, several bucket
    /// counts, forward and adjoint, against three policies.
    ///
    /// This is the primary correctness net for the engine: `naive_apply_layer`
    /// (`crate::test_support`) is the oracle. A disagreement is a bug in the
    /// bucketed engine until proven otherwise.
    #[test]
    fn differential_against_the_naive_oracle_w1_dense_collisions() {
        // Only 8 qubits, so 2000 random terms collide heavily under a rotation
        // (both `v` and `v ^ gen` are usually present) and the merge phase has
        // real duplicate runs to combine. This is the case that matters.
        let input = rand_sum::<1>(2000, 8, 0xC0FFEE);
        let channels: Vec<(&str, Box<dyn Channel<1>>)> = vec![
            ("identity", Box::new(IdentityChannel::new())),
            ("h", Box::new(Clifford1Q::h(3))),
            ("s", Box::new(Clifford1Q::s(3))),
            ("x", Box::new(Clifford1Q::x(3))),
            ("y", Box::new(Clifford1Q::y(3))),
            ("z", Box::new(Clifford1Q::z(3))),
            ("cnot", Box::new(Clifford2Q::cnot(1, 5))),
            ("cz", Box::new(Clifford2Q::cz(1, 5))),
            ("swap", Box::new(Clifford2Q::swap(1, 5))),
            (
                "depolarizing",
                Box::new(Depolarizing {
                    support: [2],
                    p: 0.07,
                }),
            ),
            (
                "dephasing",
                Box::new(Dephasing {
                    support: [2],
                    p: 0.07,
                }),
            ),
            (
                "amp_damping",
                Box::new(AmplitudeDamping {
                    support: [2],
                    gamma: 0.3,
                }),
            ),
            (
                "rot_z",
                Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.41)),
            ),
            (
                "rot_zz",
                Box::new(PauliRotation::new(
                    {
                        let mut g = PauliString::<1>::z(1);
                        g.mul_assign(&PauliString::<1>::z(6));
                        g
                    },
                    0.41,
                )),
            ),
            (
                // General unitaries: a non-Clifford T gate (fanout 2) and a
                // dense 2Q unitary (fanout up to 16), both as local PTMs.
                "t_gate",
                Box::new(crate::channel::GeneralUnitary1Q::from_matrix(
                    2,
                    [
                        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                        [
                            Complex64::new(0.0, 0.0),
                            Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
                        ],
                    ],
                )),
            ),
            (
                "general_2q",
                Box::new({
                    // sqrt(SWAP): dense enough to exercise a wide delta set.
                    let h = Complex64::new(0.5, 0.5);
                    let hc = Complex64::new(0.5, -0.5);
                    let one = Complex64::new(1.0, 0.0);
                    let zero = Complex64::new(0.0, 0.0);
                    crate::channel::GeneralUnitary2Q::from_matrix(
                        1,
                        5,
                        [
                            [one, zero, zero, zero],
                            [zero, h, hc, zero],
                            [zero, hc, h, zero],
                            [zero, zero, zero, one],
                        ],
                    )
                }),
            ),
            (
                // Weight 4 > MAX_LOCAL_SUPPORT: exercises the Rotation variant.
                "rot_wide",
                Box::new(PauliRotation::new(
                    {
                        let mut g = PauliString::<1>::z(0);
                        for q in [2u32, 4, 6] {
                            g.mul_assign(&PauliString::<1>::x(q));
                        }
                        g
                    },
                    0.41,
                )),
            ),
        ];

        for (name, ch) in &channels {
            let cr: &dyn Channel<1> = ch.as_ref();
            for &adjoint in &[false, true] {
                for &bits in &[0u8, 1, 3, 6, 11] {
                    let want = naive_apply_layer(&input, cr, &AlwaysKeep, adjoint);
                    let got = bucketed_layer(&input, cr, &AlwaysKeep, adjoint, bits, 0xABCD);
                    assert_terms_close(
                        &got,
                        &want,
                        TOL,
                        &format!("{name} adjoint={adjoint} bits={bits}"),
                    );
                }
            }
        }
    }

    #[test]
    fn differential_against_the_naive_oracle_w2_sparse() {
        // The other regime: wide keys, few collisions, word-boundary supports.
        let input = rand_sum::<2>(3000, 128, 0xBEEF);
        let channels: Vec<(&str, Box<dyn Channel<2>>)> = vec![
            ("h@70", Box::new(Clifford1Q::h(70))),
            ("s@64", Box::new(Clifford1Q::s(64))),
            ("cnot@60,70", Box::new(Clifford2Q::cnot(60, 70))),
            ("swap@0,127", Box::new(Clifford2Q::swap(0, 127))),
            (
                "amp_damping@70",
                Box::new(AmplitudeDamping {
                    support: [70],
                    gamma: 0.25,
                }),
            ),
            (
                "rot_y@70",
                Box::new(PauliRotation::new(PauliString::<2>::y(70), 0.33)),
            ),
            (
                "rot_zz_cross_word",
                Box::new(PauliRotation::new(
                    {
                        let mut g = PauliString::<2>::z(9);
                        g.mul_assign(&PauliString::<2>::z(70));
                        g
                    },
                    0.33,
                )),
            ),
        ];
        for (name, ch) in &channels {
            let cr: &dyn Channel<2> = ch.as_ref();
            for &adjoint in &[false, true] {
                for &bits in &[2u8, 5, 9] {
                    let want = naive_apply_layer(&input, cr, &AlwaysKeep, adjoint);
                    let got = bucketed_layer(&input, cr, &AlwaysKeep, adjoint, bits, 0xABCD);
                    assert_terms_close(
                        &got,
                        &want,
                        TOL,
                        &format!("{name} adjoint={adjoint} bits={bits}"),
                    );
                }
            }
        }
    }

    #[test]
    fn differential_with_truncation_policies() {
        let input = rand_sum::<1>(1500, 8, 0xF00D);
        // Thresholds are chosen far from the coefficient scale so the two
        // engines cannot disagree merely by rounding across a cutoff.
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
        let cnot = Clifford2Q::cnot(1, 5);

        for bits in [0u8, 4, 9] {
            let got = bucketed_layer(&input, &rot, &CoefficientThreshold(1e-9), false, bits, 0x11);
            let want = naive_apply_layer(&input, &rot, &CoefficientThreshold(1e-9), false);
            assert_terms_close(&got, &want, TOL, &format!("threshold bits={bits}"));

            let got = bucketed_layer(&input, &rot, &WeightCutoff(4), false, bits, 0x11);
            let want = naive_apply_layer(&input, &rot, &WeightCutoff(4), false);
            assert_terms_close(&got, &want, TOL, &format!("weight bits={bits}"));

            let policy = And(CoefficientThreshold(1e-9), WeightCutoff(5));
            let got = bucketed_layer(&input, &cnot, &policy, false, bits, 0x11);
            let want = naive_apply_layer(&input, &cnot, &policy, false);
            assert_terms_close(&got, &want, TOL, &format!("and bits={bits}"));
        }
    }

    #[test]
    fn keep_term_sees_the_summed_coefficient() {
        // Two terms that nearly cancel must be dropped by a threshold their
        // individual magnitudes would pass. A rotation at theta = pi/2 sends
        // X and Y to the same key with opposite-ish weights.
        let mut acc = BuildAccumulator::<1>::with_capacity(4, 2);
        acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(0.5, 0.0));
        acc.add_term(
            PauliString::<1>::y(0),
            Phase::ONE,
            Complex64::new(-0.4999999, 0.0),
        );
        let input = acc.finalize();
        // theta = 0 keeps keys fixed but the sum has no duplicates, so use the
        // oracle for the general statement instead of hand-computing.
        let rot = PauliRotation::new(PauliString::<1>::z(0), std::f64::consts::FRAC_PI_2);
        for bits in [0u8, 3, 7] {
            let policy = CoefficientThreshold(1e-6);
            let got = bucketed_layer(&input, &rot, &policy, false, bits, 0x21);
            let want = naive_apply_layer(&input, &rot, &policy, false);
            assert_terms_close(&got, &want, TOL, &format!("post-sum threshold bits={bits}"));
        }
    }

    // ---- the key-preserving fast path ----

    #[test]
    fn rescale_fast_path_agrees_with_the_general_path() {
        // Depolarizing/Dephasing/Pauli gates take `rescale_in_place`. Compare
        // against the naive oracle, which has no such special case.
        let input = rand_sum::<1>(1500, 8, 0x5A5A);
        let chans: Vec<(&str, Box<dyn Channel<1>>)> = vec![
            ("identity", Box::new(IdentityChannel::new())),
            (
                "depolarizing",
                Box::new(Depolarizing {
                    support: [3],
                    p: 0.11,
                }),
            ),
            (
                "dephasing",
                Box::new(Dephasing {
                    support: [3],
                    p: 0.11,
                }),
            ),
            ("pauli_z", Box::new(Clifford1Q::z(3))),
        ];
        for (name, ch) in &chans {
            let cr: &dyn Channel<1> = ch.as_ref();
            for bits in [0u8, 4, 8] {
                let got = bucketed_layer(&input, cr, &AlwaysKeep, false, bits, 0x31);
                let want = naive_apply_layer(&input, cr, &AlwaysKeep, false);
                assert_terms_close(&got, &want, TOL, &format!("{name} bits={bits}"));
            }
        }
    }

    #[test]
    fn rescale_fast_path_still_applies_truncation() {
        let input = rand_sum::<1>(1500, 8, 0x5A5B);
        let depol = Depolarizing {
            support: [3],
            p: 0.11,
        };
        for bits in [0u8, 5] {
            let policy = And(CoefficientThreshold(0.3), WeightCutoff(4));
            let got = bucketed_layer(&input, &depol, &policy, false, bits, 0x41);
            let want = naive_apply_layer(&input, &depol, &policy, false);
            assert_terms_close(&got, &want, TOL, &format!("truncated rescale bits={bits}"));
            assert!(got.len() < input.len(), "truncation dropped nothing");
        }
    }

    // ---- determinism ----

    /// sqrt(SWAP) on two qubits: a wide delta set whose outputs can merge
    /// three or more contributions into one key. That is the only regime
    /// where the accumulation *order* is observable at all — with at most
    /// two summands, float addition is commutative and any order gives the
    /// same bits — so a determinism test without a channel like this cannot
    /// see the delta-index tiebreak in the per-bucket sort.
    fn sqrt_swap_w1(a: u32, b: u32) -> crate::channel::GeneralUnitary2Q {
        let h = Complex64::new(0.5, 0.5);
        let hc = Complex64::new(0.5, -0.5);
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        crate::channel::GeneralUnitary2Q::from_matrix(
            a,
            b,
            [
                [one, zero, zero, zero],
                [zero, h, hc, zero],
                [zero, hc, h, zero],
                [zero, zero, zero, one],
            ],
        )
    }

    #[test]
    fn output_agrees_across_bucket_counts_to_fp_tolerance() {
        // A different bucket count can gather a duplicate key's
        // contributions in a different order, and `f64` addition is not
        // associative, so only floating-point-tolerance agreement is
        // expected (ARCHITECTURE.md §Determinism). The GeneralUnitary2Q
        // case is load-bearing: rotations and Cliffords merge at most two
        // contributions per key, where any order is bitwise-equal by
        // commutativity, so only a wide-delta channel can exercise the
        // relaxed axis at all (see `sqrt_swap_w1`).
        let input = rand_sum::<1>(2000, 8, 0x9001);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
        let cnot = Clifford2Q::cnot(1, 5);
        let gu2q = sqrt_swap_w1(1, 5);
        for ch in [
            &rot as &dyn Channel<1>,
            &cnot as &dyn Channel<1>,
            &gu2q as &dyn Channel<1>,
        ] {
            let reference = bucketed_layer(&input, ch, &AlwaysKeep, false, 0, 0x51);
            for bits in [1u8, 2, 3, 5, 8, 11] {
                let got = bucketed_layer(&input, ch, &AlwaysKeep, false, bits, 0x51);
                assert_terms_close(&got, &reference, TOL, &format!("bits={bits}"));
            }
        }
    }

    #[test]
    fn output_agrees_across_hash_seeds_to_fp_tolerance() {
        // A different `H` permutes which terms share a bucket but must not
        // change the arithmetic beyond floating-point tolerance (see
        // `output_agrees_across_bucket_counts_to_fp_tolerance` above). The
        // GeneralUnitary2Q case is load-bearing for the same reason as in
        // the bucket-count test above.
        let input = rand_sum::<1>(2000, 8, 0x9002);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
        let gu2q = sqrt_swap_w1(1, 5);
        for ch in [&rot as &dyn Channel<1>, &gu2q as &dyn Channel<1>] {
            let reference = bucketed_layer(&input, ch, &AlwaysKeep, false, 6, 1);
            for seed in [2u64, 3, 5, 8, 13, 21] {
                let got = bucketed_layer(&input, ch, &AlwaysKeep, false, 6, seed);
                assert_terms_close(&got, &reference, TOL, &format!("seed={seed}"));
            }
        }
    }

    #[test]
    fn local_gather_orders_agree_to_fp_tolerance() {
        // The r-threshold hybrid ships output-major gathering for wide spans
        // (GeneralUnitary2Q) and input-major below the threshold. The two
        // visit orders emit the same *multiset* of rows per run in different
        // sequences. The key-only sort makes no promise about a duplicate
        // key's relative row order, so the two orders can gather a duplicate
        // key's contributions in different orders and their *unmerged* rows
        // need not line up element-wise. What must still hold: merging each
        // run's rows (summing duplicate keys) gives the same keys with the
        // same totals, to floating-point tolerance. Note sqrt-SWAP's nonzero
        // delta masks are {XX, ZZ, YY}-shaped, so its span has rank exactly
        // 2 under any hash —
        // the property pinned here is rank-independent (the argument never
        // mentions r), so a rank-2 coset of four members exercises it fully.
        let input = rand_sum::<1>(2000, 8, 0xAB12);
        let gu2q = sqrt_swap_w1(1, 5);
        let hash = Gf2Hash::<1>::new(8, 5, 0x77);
        let sum = input.clone().with_hash(hash);
        let prep = gu2q.prepare(sum.hash(), false).unwrap();
        let Prepared::Local(ptm) = &prep else {
            panic!("gu2q prepares to a Local plan");
        };
        let span = Gf2Span::new(&prep.bucket_deltas(), sum.hash().bits());
        assert!(
            span.r() >= 2,
            "want a multi-member coset so the two visit orders actually differ; got r={}",
            span.r()
        );
        let coords: Vec<u32> = ptm
            .deltas()
            .iter()
            .map(|d| span.coord_of(d.bucket_delta))
            .collect();
        let m = span.coset_size();

        // Assemble the rank-0 coset's member columns, ascending by coordinate.
        let mut old: Vec<BucketCols<1>> = (0..m).map(|_| BucketCols::default()).collect();
        for beta in 0..sum.num_buckets() as u32 {
            let p = span.perm_index(beta) as usize;
            if p < m {
                let (bx, bz, bc) = sum.bucket(beta as usize);
                old[p] = BucketCols {
                    x: bx.to_vec(),
                    z: bz.to_vec(),
                    coeff: bc.to_vec(),
                };
            }
        }

        let has_identity = ptm.deltas().first().is_some_and(|d| d.local_delta == 0);
        // gu2q's identity amplitude is dense, so the gather materializes only
        // the id coefficients and the merge borrows the keys from the source
        // bucket — mirrored below in `merge_run`.
        let dim = 1usize << (2 * ptm.k());
        let dense_identity = has_identity && ptm.deltas()[0].amp[..dim].iter().all(|a| *a != ZERO);
        assert!(dense_identity, "gu2q's identity amplitude must be dense");
        let gather = |output_major: bool| {
            let mut runs: Vec<GatherRun<1>> = (0..m).map(|_| GatherRun::default()).collect();
            for (j, run) in runs.iter_mut().enumerate() {
                let mut cap_id = 0usize;
                let mut cap_rest = 0usize;
                for (e, &c) in coords.iter().enumerate() {
                    let l = old[j ^ c as usize].len();
                    if has_identity && e == 0 {
                        cap_id += l;
                    } else {
                        cap_rest += l;
                    }
                }
                run.reset(0, cap_id, cap_rest);
            }
            if output_major {
                gather_local_output_major(&old, &mut runs, ptm, &coords, has_identity, true);
            } else {
                gather_local_input_major(&old, &mut runs, ptm, &coords, has_identity, true);
            }
            let mut scratch = SortScratch::<1>::default();
            for run in runs.iter_mut() {
                sort_rows_with_scratch(&mut run.x, &mut run.z, &mut run.coeff, &mut scratch);
            }
            runs
        };
        let a = gather(false);
        let b = gather(true);
        assert!(
            a.iter().map(GatherRun::len).sum::<usize>() > 0,
            "gather produced nothing — the coset assembly is wrong"
        );
        // Merge (sum) each run's rows into unique-key triples before
        // comparing, rather than comparing the sorted-but-unmerged rows
        // element-wise: a duplicate key's rows can land in either relative
        // order under the key-only sort, independent of visit order, so a
        // raw element-wise comparison could see a spurious mismatch at a tie
        // that has nothing to do with which order gathered it.
        let merge_run =
            |j: usize, run: &GatherRun<1>| -> (Vec<[u64; 1]>, Vec<[u64; 1]>, Vec<Complex64>) {
                let src = &old[j];
                assert_eq!(src.len(), run.id_coeff.len(), "dense id must be 1:1");
                let mut mx = Vec::new();
                let mut mz = Vec::new();
                let mut mc = Vec::new();
                merge2_into::<1, AlwaysKeep>(
                    &src.x,
                    &src.z,
                    &run.id_coeff,
                    &run.x,
                    &run.z,
                    &run.coeff,
                    &mut mx,
                    &mut mz,
                    &mut mc,
                    &AlwaysKeep,
                );
                (mx, mz, mc)
            };
        for (j, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
            let (max, maz, mac) = merge_run(j, ra);
            let (mbx, mbz, mbc) = merge_run(j, rb);
            assert_eq!(max, mbx, "run {j}: merged keys (x) diverge");
            assert_eq!(maz, mbz, "run {j}: merged keys (z) diverge");
            assert_eq!(mac.len(), mbc.len(), "run {j}: merged term count diverges");
            for (i, (ca, cb)) in mac.iter().zip(mbc.iter()).enumerate() {
                let d = (ca - cb).norm();
                assert!(
                    d < TOL,
                    "run {j} term {i}: merged coefficients {ca} vs {cb} (delta {d:e})"
                );
            }
        }
    }

    /// The dense/sparse identity classification that decides
    /// whether the merge borrows the id stream's keys from the source
    /// bucket. Dense = the identity amplitude never vanishes over the
    /// active support patterns; pinned per built-in so a PTM change that
    /// silently flips a channel's path shows up here.
    #[test]
    fn identity_density_classification() {
        let hash = Gf2Hash::<1>::new(12, 6, 0xD1CE);
        let check = |ch: &dyn Channel<1>, want: bool, label: &str| {
            let prep = ch.prepare(&hash, false).unwrap();
            let span = Gf2Span::new(&prep.bucket_deltas(), 6);
            match DeltaPlan::new(&prep, &span) {
                DeltaPlan::Local { dense_identity, .. } => {
                    assert_eq!(dense_identity, want, "{label}")
                }
                DeltaPlan::Rotation { .. } => panic!("{label}: expected a Local plan"),
            }
        };
        // Dense: every source row emits an id row.
        check(&sqrt_swap_w1(1, 5), true, "gu2q");
        check(
            &PauliRotation::new(PauliString::<1>::z(2), 0.3),
            true,
            "rot_z",
        );
        check(
            &AmplitudeDamping {
                support: [5],
                gamma: 0.3,
            },
            true,
            "amplitude_damping",
        );
        // Sparse: the id amplitude vanishes on some patterns (CNOT: 12 of
        // 16; H: 2 of 4) — these keep the materialized id stream.
        check(&Clifford2Q::cnot(1, 4), false, "cnot");
        check(&Clifford1Q::h(3), false, "h");
    }

    // ---- multi-layer, staying bucketed ----

    #[test]
    fn many_layers_without_converting_out() {
        // The point of the bucketed form: convert in once, run many layers,
        // convert out once. Compare against the same sequence through the
        // naive oracle.
        let input = rand_sum::<1>(800, 8, 0x7001);
        let chans: Vec<Box<dyn Channel<1>>> = vec![
            Box::new(Clifford1Q::h(0)),
            Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.3)),
            Box::new(Clifford2Q::cnot(1, 5)),
            Box::new(Depolarizing {
                support: [3],
                p: 0.05,
            }),
            Box::new(Clifford1Q::s(6)),
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<1>::z(1);
                    g.mul_assign(&PauliString::<1>::z(4));
                    g
                },
                0.2,
            )),
        ];

        let mut want = input.clone();
        for ch in &chans {
            want = naive_apply_layer(&want, ch.as_ref(), &AlwaysKeep, false);
        }

        let hash = Gf2Hash::<1>::new(8, 5, 0x77);
        let mut b = input.clone().with_hash(hash);
        let mut scratch = LayerScratch::<1>::new();
        for ch in &chans {
            let prep = ch.prepare(b.hash(), false).unwrap();
            apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);
        }
        let got = b;
        assert_terms_close(&got, &want, TOL, "six layers");
    }

    #[test]
    fn layers_survive_a_rebucket_in_between() {
        let input = rand_sum::<1>(800, 8, 0x7002);
        let h = Clifford1Q::h(0);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.3);

        let want = naive_apply_layer(
            &naive_apply_layer(&input, &h, &AlwaysKeep, false),
            &rot,
            &AlwaysKeep,
            false,
        );

        let hash = Gf2Hash::<1>::new(8, 2, 0x77);
        let mut b = input.clone().with_hash(hash);
        let mut scratch = LayerScratch::<1>::new();

        let prep = h.prepare(b.hash(), false).unwrap();
        apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);
        b.rebucket(32, 1);
        let prep = rot.prepare(b.hash(), false).unwrap();
        apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);

        assert_terms_close(&b, &want, TOL, "layer, rebucket, layer");
    }

    // ---- the fingerprint net ----

    /// FNV-1a over the eight little-endian bytes of one `u64`.
    ///
    /// Written out rather than pulled from a crate so the constant stays part
    /// of the test: the hardcoded fingerprints below are only meaningful next
    /// to the exact mix that produced them.
    fn fnv_fold(h: u64, v: u64) -> u64 {
        let mut h = h;
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// A u64 digest of the sum's *exact bits*, in canonical key order.
    ///
    /// Goes through [`canonical_triples`], hence only through the public
    /// `iter()`, so it is blind to how the sum is partitioned or stored: the
    /// digest depends on the term set and the coefficient bit patterns, and on
    /// nothing else. Coefficients are folded as `f64::to_bits`, so a change of
    /// one ULP — a different summation order for duplicate keys, say — moves
    /// the digest.
    fn layer_fingerprint<const W: usize>(s: &PauliSum<W>) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        h = fnv_fold(h, s.len() as u64);
        for (x, z, c) in canonical_triples(s) {
            for &w in x.iter().chain(z.iter()) {
                h = fnv_fold(h, w);
            }
            h = fnv_fold(h, c.re.to_bits());
            h = fnv_fold(h, c.im.to_bits());
        }
        h
    }

    /// The channels in the fingerprint net: one per prepared-path shape.
    ///
    /// `Clifford1Q::h` (2 deltas), `Clifford2Q::cnot` and `::swap` (4 deltas,
    /// different tables), `GeneralUnitary2Q` (up to 16 deltas — the case the
    /// coset walk affects most), a weight-2 `PauliRotation` (local PTM path), a
    /// weight-4 `PauliRotation` (the `RotationPrep` path), `Depolarizing` (the
    /// key-preserving `rescale_in_place` path) and `AmplitudeDamping`.
    fn fingerprint_channels() -> Vec<(&'static str, Box<dyn Channel<2>>)> {
        vec![
            ("clifford1q_h", Box::new(Clifford1Q::h(3))),
            ("clifford2q_cnot", Box::new(Clifford2Q::cnot(1, 5))),
            ("clifford2q_swap", Box::new(Clifford2Q::swap(1, 5))),
            (
                // sqrt(SWAP): non-Clifford, and dense enough to realize a wide
                // delta set rather than collapsing to a permutation.
                "general_unitary2q",
                Box::new({
                    let h = Complex64::new(0.5, 0.5);
                    let hc = Complex64::new(0.5, -0.5);
                    let one = Complex64::new(1.0, 0.0);
                    let zero = Complex64::new(0.0, 0.0);
                    crate::channel::GeneralUnitary2Q::from_matrix(
                        1,
                        5,
                        [
                            [one, zero, zero, zero],
                            [zero, h, hc, zero],
                            [zero, hc, h, zero],
                            [zero, zero, zero, one],
                        ],
                    )
                }),
            ),
            (
                "rotation_zz",
                Box::new(PauliRotation::new(
                    {
                        let mut g = PauliString::<2>::z(1);
                        g.mul_assign(&PauliString::<2>::z(6));
                        g
                    },
                    0.41,
                )),
            ),
            (
                // Weight 4 > MAX_LOCAL_SUPPORT, so this takes `gather_rotation`.
                "rotation_w4",
                Box::new(PauliRotation::new(
                    {
                        let mut g = PauliString::<2>::z(0);
                        for q in [2u32, 4, 7] {
                            g.mul_assign(&PauliString::<2>::x(q));
                        }
                        g
                    },
                    0.41,
                )),
            ),
            (
                "depolarizing",
                Box::new(Depolarizing {
                    support: [2],
                    p: 0.07,
                }),
            ),
            (
                "amp_damping",
                Box::new(AmplitudeDamping {
                    support: [2],
                    gamma: 0.3,
                }),
            ),
        ]
    }

    /// Every `(channel, direction, bits)` fingerprint the current engine
    /// produces, pinned to a literal. Order matches `fingerprint_channels`.
    const LAYER_FINGERPRINTS: &[(&str, bool, u8, u64)] = &[
        ("clifford1q_h", false, 2, 0x8a01_7283_1dac_9905),
        ("clifford1q_h", false, 5, 0x8a01_7283_1dac_9905),
        ("clifford1q_h", true, 2, 0x8a01_7283_1dac_9905),
        ("clifford1q_h", true, 5, 0x8a01_7283_1dac_9905),
        ("clifford2q_cnot", false, 2, 0x8d22_5efb_4856_044f),
        ("clifford2q_cnot", false, 5, 0x8d22_5efb_4856_044f),
        ("clifford2q_cnot", true, 2, 0x8d22_5efb_4856_044f),
        ("clifford2q_cnot", true, 5, 0x8d22_5efb_4856_044f),
        ("clifford2q_swap", false, 2, 0x5fe9_a80d_62af_1da9),
        ("clifford2q_swap", false, 5, 0x5fe9_a80d_62af_1da9),
        ("clifford2q_swap", true, 2, 0x5fe9_a80d_62af_1da9),
        ("clifford2q_swap", true, 5, 0x5fe9_a80d_62af_1da9),
        ("general_unitary2q", false, 2, 0x6a89_211e_1337_0d4b),
        ("general_unitary2q", false, 5, 0x6a89_211e_1337_0d4b),
        ("general_unitary2q", true, 2, 0x54b3_481c_3682_b7db),
        ("general_unitary2q", true, 5, 0x54b3_481c_3682_b7db),
        ("rotation_zz", false, 2, 0x79b5_287d_69fe_3049),
        ("rotation_zz", false, 5, 0x79b5_287d_69fe_3049),
        ("rotation_zz", true, 2, 0x0888_9337_8137_9549),
        ("rotation_zz", true, 5, 0x0888_9337_8137_9549),
        ("rotation_w4", false, 2, 0xd22c_2678_5d1a_6ec7),
        ("rotation_w4", false, 5, 0xd22c_2678_5d1a_6ec7),
        ("rotation_w4", true, 2, 0xda87_ea29_d292_f0c7),
        ("rotation_w4", true, 5, 0xda87_ea29_d292_f0c7),
        ("depolarizing", false, 2, 0x0c2d_0f88_a7cb_3051),
        ("depolarizing", false, 5, 0x0c2d_0f88_a7cb_3051),
        ("depolarizing", true, 2, 0x0c2d_0f88_a7cb_3051),
        ("depolarizing", true, 5, 0x0c2d_0f88_a7cb_3051),
        ("amp_damping", false, 2, 0xd3cf_d844_cd3d_2be8),
        ("amp_damping", false, 5, 0xd3cf_d844_cd3d_2be8),
        ("amp_damping", true, 2, 0x8b0f_59fb_c452_c0bf),
        ("amp_damping", true, 5, 0x8b0f_59fb_c452_c0bf),
    ];

    /// Exact-bit characterization of one bucketed layer, across every
    /// prepared-path shape, both directions and two bucket counts.
    ///
    /// A convenience tripwire, not a correctness requirement
    /// (ARCHITECTURE.md §Determinism): a red fingerprint means the engine's
    /// output bits moved and should be looked at, but when the change is
    /// correct to floating-point tolerance the fix is to regenerate the
    /// literals below in the same commit, not to preserve the old bits.
    ///
    /// The differential tests above compare against the naive oracle to a
    /// tolerance, which is the right net for "is the answer correct". This one
    /// is the complementary net: it says nothing about correctness and
    /// everything about *stability*, catching a reordered gather that stays
    /// within tolerance but silently changes what users get.
    #[test]
    fn layer_fingerprints_are_stable() {
        let input = rand_sum::<2>(2000, 10, 0xC05E7);
        let channels = fingerprint_channels();
        let mut got: Vec<(&str, bool, u8, u64)> = Vec::new();
        for (name, ch) in &channels {
            let cr: &dyn Channel<2> = ch.as_ref();
            for &adjoint in &[false, true] {
                for &bits in &[2u8, 5] {
                    let out = bucketed_layer(&input, cr, &AlwaysKeep, adjoint, bits, 0xF17E);
                    got.push((name, adjoint, bits, layer_fingerprint(&out)));
                }
            }
        }

        // Printed so a deliberate re-pin is a copy-paste, never a guess. Run
        // with `--nocapture` to see it.
        for &(name, adjoint, bits, fp) in &got {
            println!("(\"{name}\", {adjoint}, {bits}, {fp:#018x}),");
        }

        assert_eq!(
            got.len(),
            LAYER_FINGERPRINTS.len(),
            "the net and the pinned table cover different cases"
        );
        for (g, w) in got.iter().zip(LAYER_FINGERPRINTS.iter()) {
            assert_eq!(
                (g.0, g.1, g.2),
                (w.0, w.1, w.2),
                "the net and the pinned table are out of order"
            );
            assert_eq!(
                g.3, w.3,
                "fingerprint changed for {} adjoint={} bits={}: {:#018x} != {:#018x}",
                g.0, g.1, g.2, g.3, w.3,
            );
        }
    }

    #[test]
    fn an_empty_sum_survives_a_layer() {
        let input = PauliSum::<1>::empty(8);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.3);
        let out = bucketed_layer(&input, &rot, &AlwaysKeep, false, 4, 0x1);
        assert!(out.is_empty());
    }

    /// `bits = 0` means every bucket delta is 0, the span is trivial, and the
    /// whole sum is one coset processed as a single serial task — the small-sum
    /// degenerate case runs on the same code path, not a special one.
    #[test]
    fn single_bucket_sum_is_one_serial_coset() {
        let input = rand_sum::<1>(600, 8, 0xB1);
        for ch in [
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<1>::z(1);
                    g.mul_assign(&PauliString::<1>::z(5));
                    g
                },
                0.37,
            )) as Box<dyn Channel<1>>,
            Box::new(Clifford2Q::cnot(2, 6)),
        ] {
            let got = bucketed_layer(&input, ch.as_ref(), &AlwaysKeep, false, 0, 0xEE);
            assert_eq!(got.num_buckets(), 1);
            let want = naive_apply_layer(&input, ch.as_ref(), &AlwaysKeep, false);
            // Tolerance, not bitwise: the oracle sums equal keys in hashmap
            // iteration order.
            assert_terms_close(&got, &want, TOL, "bits=0 single coset");
        }
    }

    /// A wide rotation whose generator hashes to bucket delta 0: the span is
    /// trivial (`r = 0`), each coset is a single bucket, and both passes gather
    /// the same swapped-out bucket. A rotation merges at most two
    /// contributions per output key (see the doc on
    /// `output_agrees_across_bucket_counts_to_fp_tolerance`), so float
    /// addition is commutative here regardless of gather or sort order — this
    /// stays a bitwise check even under the relaxed determinism policy
    /// (ARCHITECTURE.md §Determinism).
    #[test]
    fn wide_rotation_with_colliding_bucket_delta() {
        // Weight-4 generator, wider than MAX_LOCAL_SUPPORT, so it prepares as
        // Prepared::Rotation.
        let mut gen = PauliString::<1>::z(0);
        for q in [2u32, 4, 6] {
            gen.mul_assign(&PauliString::<1>::x(q));
        }
        let rot = PauliRotation::new(gen, 0.53);
        let input = rand_sum::<1>(800, 8, 0xC0111);

        // Find a seed whose 3-bit hash sends the generator's key delta to
        // bucket 0, which is exactly the H·P = 0 collision.
        let bits = 3u8;
        let mut chosen = None;
        for seed in 0u64..4096 {
            let hash = Gf2Hash::<1>::new(8, bits, seed);
            if hash.bucket_of(&gen.x, &gen.z) == 0 {
                chosen = Some(seed);
                break;
            }
        }
        let seed = chosen.expect("no seed with H·P = 0 in 4096 tries");

        let hash = Gf2Hash::<1>::new(8, bits, seed);
        let mut b = input.clone().with_hash(hash);
        let prep = rot.prepare(b.hash(), false).unwrap();
        match &prep {
            Prepared::Rotation(r) => {
                assert_eq!(
                    r.bucket_delta_gen, r.bucket_delta_identity,
                    "seed search failed to produce the collision"
                );
            }
            _ => panic!("weight-4 rotation must prepare as Rotation"),
        }
        let mut scratch = LayerScratch::<1>::new();
        apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);

        let want = naive_apply_layer(&input, &rot, &AlwaysKeep, false);
        // Tolerance, not bitwise: the oracle sums equal keys in hashmap order.
        assert_terms_close(&b, &want, TOL, "H·P = 0 collision");
    }

    /// One `LayerScratch` serves layers of every prepared shape back to back —
    /// the coset scratch is shape-agnostic and only ever grows.
    #[test]
    fn in_place_layers_share_one_scratch_across_channel_types() {
        let input = rand_sum::<2>(1500, 10, 0x5CA7C4);
        let rot = PauliRotation::new(
            {
                let mut g = PauliString::<2>::z(1);
                g.mul_assign(&PauliString::<2>::x(7));
                g
            },
            0.29,
        );
        let cnot = Clifford2Q::cnot(3, 8);
        let h = Complex64::new(0.5, 0.5);
        let hc = Complex64::new(0.5, -0.5);
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let gu2q = crate::channel::GeneralUnitary2Q::from_matrix(
            2,
            6,
            [
                [one, zero, zero, zero],
                [zero, h, hc, zero],
                [zero, hc, h, zero],
                [zero, zero, zero, one],
            ],
        );
        let channels: [&dyn Channel<2>; 3] = [&rot, &cnot, &gu2q];

        let hash = Gf2Hash::<2>::new(10, 5, 0xD00D);
        let mut b = input.clone().with_hash(hash);
        let mut scratch = LayerScratch::<2>::new();
        let mut want = input;
        for ch in channels {
            let prep = ch.prepare(b.hash(), false).unwrap();
            apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);
            want = naive_apply_layer(&want, ch, &AlwaysKeep, false);
        }
        // Tolerance, not bitwise: the oracle sums equal keys in hashmap order.
        assert_terms_close(&b, &want, TOL, "rot → cnot → gu2q through one scratch");
    }

    /// After the working set stops growing, repeated layers allocate nothing:
    /// the total capacity held by the buckets and the scratch is identical
    /// after layer `k` and layer `k + 1`.
    #[test]
    fn capacity_stabilizes_across_repeated_layers() {
        let input = rand_sum::<1>(2000, 10, 0xCAFE);
        let hash = Gf2Hash::<1>::new(10, 4, 0xF00);
        let mut b = input.with_hash(hash);
        let hgate = Clifford1Q::h(3);
        let prep = hgate.prepare(b.hash(), false).unwrap();
        let mut scratch = LayerScratch::<1>::new();

        let total_capacity = |s: &PauliSum<1>, sc: &LayerScratch<1>| -> usize {
            let bucket_cap: usize = (0..s.num_buckets())
                .map(|i| {
                    let (x, _, _) = s.bucket(i);
                    // Capacity is not observable through the slice view; go
                    // through len as a proxy for the data, and measure the
                    // scratch's real capacities, which are where growth lands.
                    x.len()
                })
                .sum();
            let old_cap: usize = sc.task.old.iter().map(|c| c.x.capacity()).sum();
            let run_cap: usize = sc.task.runs.iter().map(|r| r.x.capacity()).sum();
            let sort_cap = sc.task.sort.total_capacity();
            bucket_cap + old_cap + run_cap + sort_cap + sc.perm.capacity() + sc.staging.capacity()
        };

        let mut snapshots = Vec::new();
        for _ in 0..4 {
            apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);
            snapshots.push(total_capacity(&b, &scratch));
        }
        assert_eq!(
            snapshots[2], snapshots[3],
            "scratch/bucket footprint still growing at layer 4: {snapshots:?}"
        );
    }

    /// A channel whose delta set is **not** XOR-closed: `{0, a, b}` with
    /// `a ⊕ b` absent. `h(D)` is then not a subspace, and only the *span*'s
    /// cosets partition the bucket space — a legal `Channel` impl that would
    /// silently lose terms if the engine grouped by `h(D)` directly.
    #[test]
    fn coset_path_is_correct_for_a_non_subspace_delta_set() {
        struct ThreeDeltas;
        impl<const W: usize> Channel<W> for ThreeDeltas {
            fn max_fanout(&self) -> usize {
                3
            }
            fn support(&self) -> [u64; W] {
                crate::channel::support_mask(&[0, 1])
            }
            fn apply(
                &self,
                input_x: &[u64; W],
                input_z: &[u64; W],
                coeff: Complex64,
                out: &mut crate::channel::OutputBuffer<'_, W>,
            ) {
                // v (0.5) + v⊕x₀ (0.3) + v⊕x₁ (0.2): key deltas {0, a, b}
                // with a ⊕ b = x₀x₁ never emitted.
                out.push(*input_x, *input_z, coeff * 0.5);
                let mut xa = *input_x;
                xa[0] ^= 1;
                out.push(xa, *input_z, coeff * 0.3);
                let mut xb = *input_x;
                xb[0] ^= 2;
                out.push(xb, *input_z, coeff * 0.2);
            }
        }

        let ch = ThreeDeltas;
        let input = rand_sum::<1>(1200, 8, 0xAB5EA7);
        let want = naive_apply_layer(&input, &ch, &AlwaysKeep, false);
        for bits in [0u8, 2, 5] {
            let got = bucketed_layer(&input, &ch, &AlwaysKeep, false, bits, 0x7EA);
            // Tolerance, not bitwise: three deltas can merge three
            // contributions into one key, and the oracle sums them in hashmap
            // iteration order.
            assert_terms_close(
                &got,
                &want,
                TOL,
                &format!("non-subspace deltas, bits={bits}"),
            );
        }
    }
}

#[cfg(test)]
mod finalize_tests {
    use super::tests::{assert_same_terms, assert_terms_close, naive_apply_layer, rand_sum};
    use super::*;
    use crate::bucket::hash::Gf2Hash;
    use crate::channel::clifford::Clifford1Q;
    use crate::channel::rotation::PauliRotation;
    use crate::channel::Channel;
    use crate::pauli_sum::PauliSum;
    use crate::truncation::builtin::{And, CoefficientThreshold, Or, TopN, WeightCutoff};

    /// `TopN` bucketed must keep exactly `n` terms, and the same *set* as the
    /// flat implementation when there are no ties in magnitude.
    #[test]
    fn top_n_bucketed_matches_the_flat_implementation() {
        let input = rand_sum::<1>(2000, 8, 0x1234);
        for n in [1usize, 7, 100, 999, 1999, 5000] {
            let policy = TopN(n);
            let mut flat = input.clone();
            policy.finalize_layer(&mut flat);

            for bits in [0u8, 3, 6, 10] {
                let hash = Gf2Hash::<1>::new(8, bits, 0x99);
                let mut b = input.clone().with_hash(hash);
                policy.finalize_layer(&mut b);
                b.assert_invariants();
                let got = b;
                assert_same_terms(&got, &flat, &format!("n={n} bits={bits}"));
            }
        }
    }

    #[test]
    fn top_n_bucketed_keeps_exactly_n_and_the_largest() {
        let input = rand_sum::<1>(1000, 8, 0x4321);
        let hash = Gf2Hash::<1>::new(8, 5, 0x99);
        let mut b = input.clone().with_hash(hash);
        TopN(50).finalize_layer(&mut b);
        assert_eq!(b.len(), 50);
        let got = b;

        // Every retained magnitude must be >= every dropped one.
        let mut all: Vec<f64> = input.iter().map(|(_, _, c)| c.norm()).collect();
        all.sort_by(|a, c| c.partial_cmp(a).unwrap());
        let cutoff = all[49];
        for (_, _, c) in got.iter() {
            assert!(c.norm() >= cutoff - 1e-15, "kept a below-cutoff term");
        }
    }

    #[test]
    fn top_n_zero_clears_and_preserves_the_invariant() {
        let input = rand_sum::<1>(500, 8, 0x5555);
        let hash = Gf2Hash::<1>::new(8, 4, 0x99);
        let mut b = input.clone().with_hash(hash);
        TopN(0).finalize_layer(&mut b);
        b.assert_invariants();
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
    }

    #[test]
    fn top_n_above_the_length_is_a_no_op() {
        // Note `rand_sum` dedups, so the realized length is below the request
        // at only 8 qubits; compare against it rather than the literal.
        let input = rand_sum::<1>(300, 8, 0x6666);
        let hash = Gf2Hash::<1>::new(8, 4, 0x99);
        let mut b = input.clone().with_hash(hash);
        TopN(10_000).finalize_layer(&mut b);
        assert_eq!(b.len(), input.len());
        let got = b;
        assert_same_terms(&got, &input, "top_n above length");
    }

    #[test]
    fn and_runs_both_finalizers_bucketed() {
        // TopN(n) twice with different n must behave like the tighter one.
        let input = rand_sum::<1>(1000, 8, 0x7777);
        let policy = And(TopN(400), TopN(120));
        let mut flat = input.clone();
        policy.finalize_layer(&mut flat);

        let hash = Gf2Hash::<1>::new(8, 5, 0x99);
        let mut b = input.clone().with_hash(hash);
        policy.finalize_layer(&mut b);
        b.assert_invariants();
        let got = b;
        assert_eq!(got.len(), 120);
        assert_same_terms(&got, &flat, "and of two top_n");
    }

    #[test]
    fn threshold_and_weight_and_or_finalizers_are_no_ops() {
        // These three have no layer-finalization step; the bucketed override
        // must leave the sum untouched rather than round-trip it.
        let input = rand_sum::<1>(500, 8, 0x8888);
        let hash = Gf2Hash::<1>::new(8, 4, 0x99);
        for tag in 0..3 {
            let mut b = input.clone().with_hash(hash.clone());
            match tag {
                0 => CoefficientThreshold(0.5).finalize_layer(&mut b),
                1 => WeightCutoff(2).finalize_layer(&mut b),
                _ => Or(CoefficientThreshold(0.5), WeightCutoff(2)).finalize_layer(&mut b),
            }
            assert_eq!(b.len(), input.len(), "tag {tag} changed the sum");
        }
    }

    /// A custom `finalize_layer` written against the public surface (`retain`)
    /// must act on the bucketed sum directly, and its result must not depend on
    /// the partition.
    #[test]
    fn a_custom_finalizer_runs_on_the_bucketed_sum() {
        /// Drops every term whose coefficient has negative real part — a global
        /// pass expressed only as `finalize_layer`, via `retain`.
        struct DropNegativeReal;
        impl<const W: usize> TruncationPolicy<W> for DropNegativeReal {
            fn finalize_layer(&self, sum: &mut PauliSum<W>) {
                sum.retain(|_x, _z, c| c.re >= 0.0);
            }
        }

        let input = rand_sum::<1>(800, 8, 0x9999);
        let mut flat = input.clone();
        DropNegativeReal.finalize_layer(&mut flat);
        assert!(
            flat.len() < input.len(),
            "the custom policy dropped nothing"
        );

        for bits in [0u8, 3, 7] {
            let hash = Gf2Hash::<1>::new(8, bits, 0x99);
            let mut b = input.clone().with_hash(hash);
            DropNegativeReal.finalize_layer(&mut b);
            b.assert_invariants();
            let got = b;
            assert_same_terms(&got, &flat, &format!("bits={bits}"));
        }
    }

    /// Layer then finalize, repeatedly — the shape `propagate` will use.
    #[test]
    fn interleaved_layers_and_finalizers_match_the_naive_sequence() {
        let input = rand_sum::<1>(1200, 8, 0xAAAA);
        let policy = And(CoefficientThreshold(1e-9), TopN(300));
        let chans: Vec<Box<dyn Channel<1>>> = vec![
            Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.37)),
            Box::new(Clifford1Q::h(0)),
            Box::new(PauliRotation::new(PauliString::<1>::x(5), 0.21)),
        ];

        let mut want = input.clone();
        for ch in &chans {
            want = naive_apply_layer(&want, ch.as_ref(), &policy, false);
            policy.finalize_layer(&mut want);
        }

        let hash = Gf2Hash::<1>::new(8, 5, 0xBB);
        let mut b = input.clone().with_hash(hash);
        let mut scratch = LayerScratch::<1>::new();
        for ch in &chans {
            let prep = ch.prepare(b.hash(), false).unwrap();
            apply_layer_bucketed(&mut b, &prep, &policy, &mut scratch);
            policy.finalize_layer(&mut b);
        }
        let got = b;

        assert_terms_close(&got, &want, 1e-11, "3 truncated layers");
    }
}

#[cfg(test)]
mod tie_tests {
    /// The C.1 determinism contract: byte-identical output across thread counts,
    /// with the *engine* parallel. `apply_layer_bucketed` fixes the bucket count
    /// here, so this isolates thread count from partition (the propagate-level
    /// test in tests/propagate_bucketed.rs exercises the public entry point).
    #[test]
    fn parallel_output_is_byte_identical_across_thread_counts() {
        use crate::channel::rotation::PauliRotation;
        use crate::channel::Channel;

        let input = rand_sum::<1>(4000, 10, 0xC1C1);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.37);
        let cnot = crate::channel::clifford::Clifford2Q::cnot(1, 5);

        for ch in [&rot as &dyn Channel<1>, &cnot as &dyn Channel<1>] {
            // 64 buckets: comfortably above MIN_COSETS_FOR_PARALLEL, so the
            // parallel path is genuinely exercised.
            let run = |threads: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("pool")
                    .install(|| {
                        let hash = Gf2Hash::<1>::new(10, 6, 0xC1);
                        let mut b = input.clone().with_hash(hash);
                        let prep = ch.prepare(b.hash(), false).unwrap();
                        let mut scratch = LayerScratch::<1>::new();
                        apply_layer_bucketed(
                            &mut b,
                            &prep,
                            &super::tests::AlwaysKeep,
                            &mut scratch,
                        );
                        b
                    })
            };
            let reference = run(1);
            for threads in [2usize, 4, 8, 16, 32] {
                let got = run(threads);
                assert_eq!(got.len(), reference.len(), "threads={threads}");
                // Identical fixed hash on both sides, so canonical order is
                // shared and whole-column equality is the bitwise statement.
                assert_eq!(
                    got.to_arrays(),
                    reference.to_arrays(),
                    "threads={threads}: output is not byte-identical",
                );
            }
        }
    }

    /// The in-place rescale path is parallel too, and must give the same answer.
    #[test]
    fn parallel_rescale_is_byte_identical_across_thread_counts() {
        use crate::channel::noise::Depolarizing;
        use crate::channel::Channel;

        let input = rand_sum::<1>(4000, 10, 0xC1C2);
        let depol = Depolarizing {
            support: [3],
            p: 0.11,
        };
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| {
                    let hash = Gf2Hash::<1>::new(10, 6, 0xC2);
                    let mut b = input.clone().with_hash(hash);
                    let prep = Channel::<1>::prepare(&depol, b.hash(), false).unwrap();
                    let mut scratch = LayerScratch::<1>::new();
                    apply_layer_bucketed(&mut b, &prep, &super::tests::AlwaysKeep, &mut scratch);
                    b
                })
        };
        let reference = run(1);
        for threads in [2usize, 8, 32] {
            let got = run(threads);
            // Identical fixed hash on both sides: canonical order is shared.
            assert_eq!(
                got.to_arrays().2,
                reference.to_arrays().2,
                "threads={threads}"
            );
        }
    }

    use super::tests::{assert_same_terms, canonical_triples};
    use super::*;
    use crate::bucket::hash::Gf2Hash;
    use crate::pauli_sum::PauliSum;
    use crate::test_support::{rand_sum, tie_heavy_sum};
    use crate::truncation::builtin::TopN;

    /// A direct, deliberately naive transcription of the `TopN` tie-group
    /// rule (ARCHITECTURE.md §Truncation), used as an oracle for the
    /// production `finalize_layer`.
    ///
    /// Full sort instead of a selection, `retain` instead of a per-bucket
    /// compaction, no parallelism: nothing here shares code with the thing it
    /// checks. Returns the surviving terms as canonical triples.
    fn top_n_reference<const W: usize>(
        sum: &PauliSum<W>,
        n: usize,
    ) -> Vec<([u64; W], [u64; W], Complex64)> {
        let mut triples = canonical_triples(sum);
        if triples.len() <= n {
            return triples;
        }
        if n == 0 {
            return Vec::new();
        }
        let mut mags: Vec<f64> = triples.iter().map(|t| t.2.norm()).collect();
        mags.sort_by(|a, b| b.partial_cmp(a).expect("no NaN magnitudes"));
        // `t` = the n-th largest magnitude.
        let t = mags[n - 1];
        let count_gt = mags.iter().filter(|&&m| m > t).count();
        let count_eq = mags.iter().filter(|&&m| m == t).count();
        // Keep the tie group iff it fits entirely.
        let keep_tied = count_gt + count_eq <= n;
        triples.retain(|(_, _, c)| {
            let m = c.norm();
            m > t || (keep_tied && m == t)
        });
        triples
    }

    /// The retained set is a pure function of the magnitude multiset, so it
    /// cannot depend on the bucket partition. Checked on tie-dense data, where
    /// a partition-sensitive rule (flat position, or the old key tiebreak read
    /// through a bucket index) would show up.
    #[test]
    fn top_n_is_bucket_count_independent_on_tied_magnitudes() {
        let input = tie_heavy_sum::<1>(2000, 8, 0x7135);
        let n = 700; // cuts inside the group of magnitude-0.5 terms
        let reference = {
            let hash = Gf2Hash::<1>::new(8, 0, 0x99);
            let mut b = input.clone().with_hash(hash);
            TopN(n).finalize_layer(&mut b);
            b
        };
        for bits in [1u8, 2, 4, 6, 9] {
            let hash = Gf2Hash::<1>::new(8, bits, 0x99);
            let mut b = input.clone().with_hash(hash);
            TopN(n).finalize_layer(&mut b);
            let got = b;
            assert_same_terms(
                &got,
                &reference,
                &format!("bits={bits}: TopN kept a different set of tied terms"),
            );
        }
    }

    /// `finalize_layer` must agree with the `TopN` tie-group rule computed
    /// the obvious way, on tie-dense data and across partitions. This is the
    /// semantics test: it
    /// pins *which* terms survive, not merely that all partitions agree.
    ///
    /// The `n` sweep mixes arbitrary cut points (which straddle a group, since
    /// there are only four magnitudes) with the *exact* group boundaries read
    /// off the fixture (which fit). The assertions at the end fail the test if
    /// it ever stops exercising one of the two branches.
    #[test]
    fn top_n_matches_the_reference_rule_on_tied_magnitudes() {
        let input = tie_heavy_sum::<1>(2000, 8, 0x7136);
        let len = input.len();

        // Cumulative sizes of the magnitude groups, descending: an `n` equal to
        // one of these is a cut that lands exactly on a group boundary.
        let mut mags: Vec<f64> = input.iter().map(|(_, _, c)| c.norm()).collect();
        mags.sort_by(|a, b| b.partial_cmp(a).expect("no NaN magnitudes"));
        let boundaries: Vec<usize> = (1..mags.len())
            .filter(|&i| mags[i] != mags[i - 1])
            .collect();
        assert!(
            boundaries.len() >= 2,
            "fixture must have several magnitude groups, got {}",
            boundaries.len() + 1
        );

        let mut sweep = vec![3usize, 250, 700, 1200, 1900];
        sweep.extend_from_slice(&boundaries);
        let mut saw_straddle = false;
        let mut saw_fit = false;

        for n in sweep {
            let want = top_n_reference(&input, n);
            // A straddling group is discarded whole, so fewer than `n` survive;
            // a group that fits leaves exactly `n`.
            assert!(want.len() <= n, "n={n}: rule must retain at most n");
            if want.len() < n {
                saw_straddle = true;
            } else {
                saw_fit = true;
            }

            let policy = TopN(n);
            for bits in [0u8, 2, 5, 9] {
                let hash = Gf2Hash::<1>::new(8, bits, 0x99);
                let mut b = input.clone().with_hash(hash);
                policy.finalize_layer(&mut b);
                b.assert_invariants();
                assert_eq!(
                    canonical_triples(&b),
                    want,
                    "n={n} bits={bits} (len={len}): retained set differs from the \
                     reference rule",
                );
            }
        }

        assert!(
            saw_straddle,
            "the n sweep no longer covers a straddling tie group"
        );
        assert!(
            saw_fit,
            "the n sweep no longer covers a tie group that fits"
        );
    }

    /// The same, across hash seeds: a different `H` permutes bucket membership
    /// without changing anything about the magnitudes.
    #[test]
    fn top_n_is_hash_seed_independent_on_tied_magnitudes() {
        let input = tie_heavy_sum::<1>(2000, 8, 0x7123);
        let n = 700;
        let reference = {
            let hash = Gf2Hash::<1>::new(8, 5, 1);
            let mut b = input.clone().with_hash(hash);
            TopN(n).finalize_layer(&mut b);
            b
        };
        for seed in [2u64, 3, 5, 8, 13] {
            let hash = Gf2Hash::<1>::new(8, 5, seed);
            let mut b = input.clone().with_hash(hash);
            TopN(n).finalize_layer(&mut b);
            let got = b;
            assert_same_terms(
                &got,
                &reference,
                &format!("seed={seed}: different set kept"),
            );
        }
    }

    /// Tie-dense data multi-bucket: a straddling group leaves no member behind
    /// in *any* bucket. The per-bucket compaction is where a partial drop would
    /// hide, so this is checked on the real partition rather than at `B = 1`.
    #[test]
    fn top_n_drops_a_straddling_group_from_every_bucket() {
        let input = tie_heavy_sum::<1>(2000, 8, 0x71C0);
        let n = 700;
        // t is the 700th largest magnitude; the group at t straddles the cut.
        let mut mags: Vec<f64> = input.iter().map(|(_, _, c)| c.norm()).collect();
        mags.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let t = mags[n - 1];
        let count_gt = mags.iter().filter(|&&m| m > t).count();
        let count_eq = mags.iter().filter(|&&m| m == t).count();
        assert!(
            count_gt + count_eq > n,
            "fixture no longer straddles: gt={count_gt} eq={count_eq} n={n}"
        );

        for bits in [0u8, 3, 7] {
            let hash = Gf2Hash::<1>::new(8, bits, 0x99);
            let mut b = input.clone().with_hash(hash);
            TopN(n).finalize_layer(&mut b);
            b.assert_invariants();
            assert_eq!(
                b.len(),
                count_gt,
                "bits={bits}: retained count must be exactly count(|c| > t)"
            );
            for nb in 0..b.num_buckets() {
                let (_, _, coeff) = b.bucket(nb);
                assert!(
                    coeff.iter().all(|c| c.norm() > t),
                    "bits={bits} bucket={nb}: a member of the discarded group survived"
                );
            }
        }
    }
}
