//! The bucketed layer engine. See v0.2 design doc §6 and v0.3 §2.
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
//!    `member(i) ⊕ δ = member(i ⊕ coord(δ))` — then per run **sort** by
//!    `(key, tag)` and **merge** straight into the member's live slot.
//! 3. Un-permute the handles, recount, assert invariants.
//!
//! The gather visits each input term exactly once (v0.2 §6.1's `|D| · n` read
//! amplification is gone), and there is no second full-size buffer: peak memory
//! is `n` plus per-worker scratch of one coset's working set.
//!
//! Determinism: cosets are write-disjoint, work within one is sequential, and
//! duplicate keys are summed in ascending delta order — enforced by the tag
//! column in `sort_merge::sort_phase_tagged`, not by gather order — so output
//! is bitwise identical across thread counts *and* bucket counts, exactly as
//! before (v0.2 §9.1).

use std::sync::Mutex;

use num_complex::Complex64;
use rayon::prelude::*;

use super::coset::Gf2Span;
use super::sort_merge::{merge_into, sort_phase_tagged};
use crate::bucket::sum::{BucketCols, PauliSum};
use crate::channel::prepared::{LocalPtm, Prepared, RotationPrep};
use crate::pauli_string::PauliString;
use crate::phase::Phase;
use crate::truncation::TruncationPolicy;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Reusable per-layer scratch.
///
/// Held by the caller across layers because a layer must allocate nothing
/// after the first (v0.2 §4.2): every field retains its high-water capacity
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
}

impl<const W: usize> LayerScratch<W> {
    /// An empty scratch.
    pub fn new() -> Self {
        Self::default()
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
}

/// One output member's gather run: key columns, coefficients, and the delta
/// tag that `sort_phase_tagged` breaks equal-key ties on.
#[derive(Clone, Debug, Default)]
struct GatherRun<const W: usize> {
    x: Vec<[u64; W]>,
    z: Vec<[u64; W]>,
    coeff: Vec<Complex64>,
    tag: Vec<u8>,
}

impl<const W: usize> GatherRun<W> {
    #[inline]
    fn reset(&mut self, cap: usize) {
        self.x.clear();
        self.z.clear();
        self.coeff.clear();
        self.tag.clear();
        if self.x.capacity() < cap {
            let extra = cap - self.x.capacity();
            self.x.reserve(extra);
            self.z.reserve(extra);
            self.coeff.reserve(extra);
            self.tag.reserve(extra);
        }
    }

    #[inline]
    fn push(&mut self, x: [u64; W], z: [u64; W], c: Complex64, tag: u8) {
        self.x.push(x);
        self.z.push(z);
        self.coeff.push(c);
        self.tag.push(tag);
    }

    #[inline]
    fn len(&self) -> usize {
        self.coeff.len()
    }
}

/// A prepared channel's delta set, annotated with each entry's coset
/// coordinate (`span.coord_of(bucket_delta)`), computed once per layer.
enum DeltaPlan<'p, const W: usize> {
    /// Tabulated deltas; `coords[e]` pairs with `ptm.deltas()[e]`, whose index
    /// `e` is also the entry's tag (ascending `local_delta` order).
    Local {
        ptm: &'p LocalPtm<W>,
        coords: Vec<u32>,
    },
    /// Wide rotation: two implicit entries, tag 0 = identity pass,
    /// tag 1 = generator pass.
    Rotation {
        prep: &'p RotationPrep<W>,
        coord_identity: u32,
        coord_gen: u32,
    },
}

impl<'p, const W: usize> DeltaPlan<'p, W> {
    fn new(prep: &'p Prepared<W>, span: &Gf2Span) -> Self {
        match prep {
            Prepared::Local(ptm) => DeltaPlan::Local {
                ptm,
                coords: ptm
                    .deltas()
                    .iter()
                    .map(|d| span.coord_of(d.bucket_delta))
                    .collect(),
            },
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
/// **summed** coefficients — the same contract as the v0.1 engine.
/// `finalize_layer` is *not* called here; `propagate` owns that.
pub fn apply_layer_bucketed<const W: usize, T>(
    sum: &mut PauliSum<W>,
    prep: &Prepared<W>,
    policy: &T,
    scratch: &mut LayerScratch<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    // Key-preserving channels (identity, depolarizing, dephasing, Pauli gates)
    // leave every key bitwise unchanged, so the output is already sorted and
    // duplicate-free. v0.1 paid a full O(n log n) sort to multiply each
    // coefficient by a scalar; here it is an in-place filter.
    if let Prepared::Local(ptm) = prep {
        if ptm.is_key_preserving() {
            rescale_in_place(sum, ptm, policy);
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

    // Each coset is a closed task: it reads and writes only its own chunk, so
    // the chunk loop needs no atomics, no cross-task locks, and no
    // reconciliation pass. Work within a task is sequential and its summation
    // order is fixed by the tag sort, so output is byte-identical across
    // thread counts.
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

    // Un-permute: every handle goes back to its bucket index, leaving the
    // staging slots as empty, capacity-free defaults.
    if !identity_perm {
        let buckets = sum.buckets_mut();
        for (beta, cols) in buckets.iter_mut().enumerate() {
            *cols = std::mem::take(&mut scratch.staging[scratch.perm[beta] as usize]);
        }
    }
    sum.recount();

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
    let CosetScratch { old, runs } = ws;
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

    // Exact per-run capacity, counted once per delta entry — two entries
    // colliding on one bucket delta count twice, matching the rows they can
    // emit.
    for (j, run) in runs.iter_mut().enumerate() {
        let cap: usize = match plan {
            DeltaPlan::Local { coords, .. } => {
                coords.iter().map(|&c| old[j ^ c as usize].len()).sum()
            }
            DeltaPlan::Rotation {
                coord_identity,
                coord_gen,
                ..
            } => old[j ^ *coord_identity as usize].len() + old[j ^ *coord_gen as usize].len(),
        };
        run.reset(cap);
    }

    // Gather. Two visit orders produce the same rows per run — `(key, tag)` is
    // unique within a run, so the tag sort canonicalizes either order to the
    // same sequence and the output is bitwise-identical (v0.2 §9.1). Which
    // order is *faster* depends on `r`: input-major loads each term once but
    // keeps `2^r` write streams open per task, and at `r = 4` those streams
    // plus the swapped coset no longer fit L2 — measured +48% on a 32-thread
    // `GeneralUnitary2Q` layer at 10⁶ terms. Output-major re-reads each input
    // bucket `2^r` times but the reads stay coset-local and there is a single
    // write stream, which is the cache behavior the per-output-bucket v0.2
    // engine had. Only `Local` plans can reach `r ≥ 3` (a wide rotation has at
    // most two bucket deltas).
    match plan {
        DeltaPlan::Local { ptm, coords } => {
            if m >= 1 << GATHER_OUTPUT_MAJOR_MIN_R {
                gather_local_output_major(old, runs, ptm, coords);
            } else {
                gather_local_input_major(old, runs, ptm, coords);
            }
        }
        DeltaPlan::Rotation {
            prep,
            coord_identity,
            coord_gen,
        } => {
            // Tag 0 = the identity pass, tag 1 = the generator pass, matching
            // the canonical `local_delta` order (0 before P). `cos`/`sin` stay
            // hoisted; the `i^k` phase depends on 2w support bits and is
            // computed per anticommuting term, exactly as before.
            for (i, src) in old.iter().enumerate() {
                for t in 0..src.len() {
                    let v = PauliString::<W> {
                        x: src.x[t],
                        z: src.z[t],
                    };
                    if v.commutes_with(&prep.gen) {
                        runs[i ^ *coord_identity as usize].push(
                            src.x[t],
                            src.z[t],
                            src.coeff[t],
                            0,
                        );
                    } else {
                        runs[i ^ *coord_identity as usize].push(
                            src.x[t],
                            src.z[t],
                            src.coeff[t] * prep.cos,
                            0,
                        );
                        let mut prod = v;
                        let phase = prod.mul_assign(&prep.gen);
                        let total = Phase::I + phase;
                        runs[i ^ *coord_gen as usize].push(
                            prod.x,
                            prod.z,
                            total.apply(src.coeff[t]) * prep.sin,
                            1,
                        );
                    }
                }
            }
        }
    }

    // Sort each run by (key, tag) and merge into the member's live slot.
    for (run, dst) in runs.iter_mut().zip(chunk.iter_mut()) {
        let len = run.len();
        sort_phase_tagged(&mut run.x, &mut run.z, &mut run.coeff, &run.tag, len);
        merge_into::<W, T>(
            &run.x,
            &run.z,
            &run.coeff,
            0,
            len,
            &mut dst.x,
            &mut dst.z,
            &mut dst.coeff,
            policy,
        );
    }

    // Leave `old` cleared so the next coset's swap hands its chunk clean,
    // capacity-retaining columns. Runs are cleared by their own `reset`.
    for cols in old.iter_mut() {
        cols.clear();
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
/// per task), where the scatter working set doubles twice more; both paths
/// produce bitwise-identical output (the `(key, tag)` sort canonicalizes
/// either visit order — pinned by `local_gather_orders_are_bitwise_equivalent`),
/// so the threshold is a pure performance knob.
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
) {
    for (i, src) in old.iter().enumerate() {
        for t in 0..src.len() {
            let s = ptm.support_bits(&src.x[t], &src.z[t]);
            for (e, d) in ptm.deltas().iter().enumerate() {
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
                runs[i ^ coords[e] as usize].push(kx, kz, src.coeff[t] * a, e as u8);
            }
        }
    }
}

/// Output-major gather for a tabulated (`Local`) plan: for each output member,
/// stream the one input bucket per delta entry and append to that member's run
/// only. Rows land in (delta, input position) order — the same multiset as
/// [`gather_local_input_major`] in a different order, which the `(key, tag)`
/// sort erases: equal-key rows never share a tag, so both orders canonicalize
/// to the identical sequence and the merged output is bitwise-equal.
fn gather_local_output_major<const W: usize>(
    old: &[BucketCols<W>],
    runs: &mut [GatherRun<W>],
    ptm: &LocalPtm<W>,
    coords: &[u32],
) {
    for (j, run) in runs.iter_mut().enumerate() {
        for (e, d) in ptm.deltas().iter().enumerate() {
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
                run.push(kx, kz, src.coeff[t] * a, e as u8);
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

// Gated on `debug_assertions` because these tests call `assert_invariants`,
// which is itself debug-only. Matches the convention in `pauli_sum.rs` and
// `sort_merge.rs`; without it `cargo bench` and `cargo test --release`, which
// compile the lib tests in release mode, fail to build.
#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;
    use crate::accumulator::BuildAccumulator;
    use crate::bucket::hash::Gf2Hash;
    use crate::channel::clifford::{Clifford1Q, Clifford2Q};
    use crate::channel::identity::IdentityChannel;
    use crate::channel::noise::{AmplitudeDamping, Dephasing, Depolarizing};
    use crate::channel::rotation::PauliRotation;
    use crate::channel::Channel;
    use crate::engine::sort_merge::{apply_layer, apply_layer_adjoint};
    use crate::pauli_sum::PauliSum;
    use crate::truncation::builtin::{And, CoefficientThreshold, WeightCutoff};

    const TOL: f64 = 1e-11;

    pub(super) struct AlwaysKeep;
    impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

    pub(super) struct Xs64(u64);
    impl Xs64 {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    pub(super) fn word_mask(num_qubits: usize, word: usize) -> u64 {
        let lo = 64 * word;
        if num_qubits >= lo + 64 {
            !0u64
        } else if num_qubits <= lo {
            0
        } else {
            (1u64 << (num_qubits - lo)) - 1
        }
    }

    pub(super) fn rand_sum<const W: usize>(n: usize, num_qubits: usize, seed: u64) -> PauliSum<W> {
        let mut rng = Xs64::new(seed);
        let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n);
        for _ in 0..n {
            let mut p = PauliString::<W> {
                x: [0u64; W],
                z: [0u64; W],
            };
            for w in 0..W {
                let m = word_mask(num_qubits, w);
                p.x[w] = rng.next_u64() & m;
                p.z[w] = rng.next_u64() & m;
            }
            let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
            let im = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
            acc.add_term(p, Phase::ONE, Complex64::new(re, im));
        }
        acc.finalize()
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

    /// `(x, z, coeff)` triples sorted by the `(x, z)` key.
    ///
    /// Keys are globally unique (the `PauliSum` invariant forbids duplicates),
    /// so this is a canonical, storage-order-independent view: two sums with
    /// the same terms produce the same triples regardless of which order their
    /// backing engine happened to store them in.
    pub(super) fn canonical_triples<const W: usize>(
        s: &PauliSum<W>,
    ) -> Vec<([u64; W], [u64; W], Complex64)> {
        let mut v: Vec<([u64; W], [u64; W], Complex64)> =
            s.iter().map(|(x, z, c)| (*x, *z, c)).collect();
        v.sort_unstable_by_key(|&(x, z, _)| (x, z));
        v
    }

    /// Same keys, same coefficients bitwise (`Complex64` `==`) — order-agnostic.
    pub(super) fn assert_same_terms<const W: usize>(
        got: &PauliSum<W>,
        want: &PauliSum<W>,
        what: &str,
    ) {
        assert_eq!(got.len(), want.len(), "{what}: term count");
        let got = canonical_triples(got);
        let want = canonical_triples(want);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!((g.0, g.1), (w.0, w.1), "{what}: term {i} key mismatch");
            assert_eq!(
                g.2, w.2,
                "{what}: term {i} key {:?}/{:?} coeff {} vs {} (not bitwise equal)",
                g.0, g.1, g.2, w.2,
            );
        }
    }

    /// Same keys; coefficients within `tol`, because the two engines can sum
    /// duplicate keys in different orders and floating-point addition is not
    /// associative.
    pub(super) fn assert_terms_close<const W: usize>(
        got: &PauliSum<W>,
        want: &PauliSum<W>,
        tol: f64,
        what: &str,
    ) {
        assert_eq!(got.len(), want.len(), "{what}: term count");
        let got = canonical_triples(got);
        let want = canonical_triples(want);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!((g.0, g.1), (w.0, w.1), "{what}: term {i} key mismatch");
            let d = (g.2 - w.2).norm();
            assert!(
                d < tol,
                "{what}: term {i} key {:?}/{:?} coeff {} vs {} (delta {d:e})",
                g.0,
                g.1,
                g.2,
                w.2,
            );
        }
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
        let want = apply_layer(&input, &rot, &AlwaysKeep);
        assert_sums_close(&out, &want, "rotation fanout");
    }

    // ---- the differential test against the v0.1 engine ----

    /// Every built-in channel, over both occupancy regimes, several bucket
    /// counts, forward and adjoint, against three policies.
    ///
    /// This is the primary correctness net for the rewrite: the v0.1 engine is
    /// the oracle (v0.2 §9.2). A disagreement is a bug in the new engine until
    /// proven otherwise.
    #[test]
    fn differential_against_sort_merge_w1_dense_collisions() {
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
                    let want = if adjoint {
                        apply_layer_adjoint(&input, cr, &AlwaysKeep)
                    } else {
                        apply_layer(&input, cr, &AlwaysKeep)
                    };
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
    fn differential_against_sort_merge_w2_sparse() {
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
                    let want = if adjoint {
                        apply_layer_adjoint(&input, cr, &AlwaysKeep)
                    } else {
                        apply_layer(&input, cr, &AlwaysKeep)
                    };
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
            let want = apply_layer(&input, &rot, &CoefficientThreshold(1e-9));
            assert_terms_close(&got, &want, TOL, &format!("threshold bits={bits}"));

            let got = bucketed_layer(&input, &rot, &WeightCutoff(4), false, bits, 0x11);
            let want = apply_layer(&input, &rot, &WeightCutoff(4));
            assert_terms_close(&got, &want, TOL, &format!("weight bits={bits}"));

            let policy = And(CoefficientThreshold(1e-9), WeightCutoff(5));
            let got = bucketed_layer(&input, &cnot, &policy, false, bits, 0x11);
            let want = apply_layer(&input, &cnot, &policy);
            assert_terms_close(&got, &want, TOL, &format!("and bits={bits}"));
        }
    }

    #[test]
    fn keep_term_sees_the_summed_coefficient() {
        // Port of sort_merge's `threshold_applied_after_summation`: two terms
        // that nearly cancel must be dropped by a threshold their individual
        // magnitudes would pass. A rotation at theta = pi/2 sends X and Y to the
        // same key with opposite-ish weights.
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
            let want = apply_layer(&input, &rot, &policy);
            assert_terms_close(&got, &want, TOL, &format!("post-sum threshold bits={bits}"));
        }
    }

    // ---- the key-preserving fast path ----

    #[test]
    fn rescale_fast_path_agrees_with_the_general_path() {
        // Depolarizing/Dephasing/Pauli gates take `rescale_in_place`. Compare
        // against the v0.1 engine, which has no such special case.
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
                let want = apply_layer(&input, cr, &AlwaysKeep);
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
            let want = apply_layer(&input, &depol, &policy);
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
    fn output_is_bitwise_identical_across_bucket_counts() {
        // The strong form of v0.2 §9.1: not merely close, but *bitwise* equal,
        // which is what the canonical `local_delta` accumulation order buys.
        // The GeneralUnitary2Q case is load-bearing: rotations and Cliffords
        // merge at most two contributions per key, where any order is
        // bitwise-equal by commutativity, so only a wide-delta channel can
        // catch an ordering regression (see `sqrt_swap_w1`).
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
                assert_same_terms(&got, &reference, &format!("bits={bits}"));
            }
        }
    }

    #[test]
    fn output_is_bitwise_identical_across_hash_seeds() {
        // A different `H` permutes which terms share a bucket but must not
        // change the arithmetic. The GeneralUnitary2Q case is load-bearing
        // for the same reason as in the bucket-count test above.
        let input = rand_sum::<1>(2000, 8, 0x9002);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
        let gu2q = sqrt_swap_w1(1, 5);
        for ch in [&rot as &dyn Channel<1>, &gu2q as &dyn Channel<1>] {
            let reference = bucketed_layer(&input, ch, &AlwaysKeep, false, 6, 1);
            for seed in [2u64, 3, 5, 8, 13, 21] {
                let got = bucketed_layer(&input, ch, &AlwaysKeep, false, 6, seed);
                assert_same_terms(&got, &reference, &format!("seed={seed}"));
            }
        }
    }

    #[test]
    fn local_gather_orders_are_bitwise_equivalent() {
        // The r-threshold hybrid ships output-major gathering for wide spans
        // (GeneralUnitary2Q) and input-major below the threshold. The two
        // visit orders emit the same rows per run in different sequences, and
        // the (key, tag) sort — under which equal-key rows never share a tag —
        // must canonicalize both to the identical, bitwise-equal sequence.
        // Note sqrt-SWAP's nonzero delta masks are {XX, ZZ, YY}-shaped, so its
        // span has rank exactly 2 under any hash — the equivalence being
        // pinned here is rank-independent (the sort argument never mentions
        // r), so a rank-2 coset of four members exercises it fully.
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

        let gather = |output_major: bool| {
            let mut runs: Vec<GatherRun<1>> = (0..m).map(|_| GatherRun::default()).collect();
            for (j, run) in runs.iter_mut().enumerate() {
                let cap: usize = coords.iter().map(|&c| old[j ^ c as usize].len()).sum();
                run.reset(cap);
            }
            if output_major {
                gather_local_output_major(&old, &mut runs, ptm, &coords);
            } else {
                gather_local_input_major(&old, &mut runs, ptm, &coords);
            }
            for run in runs.iter_mut() {
                let len = run.len();
                sort_phase_tagged(&mut run.x, &mut run.z, &mut run.coeff, &run.tag, len);
            }
            runs
        };
        let a = gather(false);
        let b = gather(true);
        assert!(
            a.iter().map(GatherRun::len).sum::<usize>() > 0,
            "gather produced nothing — the coset assembly is wrong"
        );
        // The tag column is comparator-only (never permuted), so the
        // comparison is over the sorted keys and coefficients.
        for (j, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(ra.x, rb.x, "run {j}: keys (x) diverge");
            assert_eq!(ra.z, rb.z, "run {j}: keys (z) diverge");
            assert_eq!(
                ra.coeff, rb.coeff,
                "run {j}: coefficients not bitwise equal"
            );
        }
    }

    // ---- multi-layer, staying bucketed ----

    #[test]
    fn many_layers_without_converting_out() {
        // The point of the bucketed form: convert in once, run many layers,
        // convert out once. Compare against the same sequence through v0.1.
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
            want = apply_layer(&want, ch.as_ref(), &AlwaysKeep);
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

        let want = apply_layer(&apply_layer(&input, &h, &AlwaysKeep), &rot, &AlwaysKeep);

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

    // ---- the fingerprint net (v0.3 §2, slice C0) ----

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
    /// different tables), `GeneralUnitary2Q` (up to 16 deltas — the case §2's
    /// coset walk changes most), a weight-2 `PauliRotation` (local PTM path), a
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
    /// The §2 coset rewrite must reproduce these EXACT u64s — a later red
    /// fingerprint means the rewrite changed observable bits and must be
    /// analyzed, never regenerated.
    ///
    /// The differential tests above compare against the v0.1 oracle to a
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
            let want = apply_layer(&input, ch.as_ref(), &AlwaysKeep);
            assert_same_terms(&got, &want, "bits=0 single coset");
        }
    }

    /// A wide rotation whose generator hashes to bucket delta 0: the span is
    /// trivial (`r = 0`), each coset is a single bucket, and both passes gather
    /// the same swapped-out bucket — the tag sort alone restores the canonical
    /// identity-before-generator order.
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

        let want = apply_layer(&input, &rot, &AlwaysKeep);
        assert_same_terms(&b, &want, "H·P = 0 collision");
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
            want = apply_layer(&want, ch, &AlwaysKeep);
        }
        assert_same_terms(&b, &want, "rot → cnot → gu2q through one scratch");
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
            bucket_cap + old_cap + run_cap + sc.perm.capacity() + sc.staging.capacity()
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
        let want = apply_layer(&input, &ch, &AlwaysKeep);
        for bits in [0u8, 2, 5] {
            let got = bucketed_layer(&input, &ch, &AlwaysKeep, false, bits, 0x7EA);
            assert_same_terms(&got, &want, &format!("non-subspace deltas, bits={bits}"));
        }
    }
}

// Gated on `debug_assertions` because these tests call `assert_invariants`,
// which is itself debug-only. Matches the convention in `pauli_sum.rs` and
// `sort_merge.rs`; without it `cargo bench` and `cargo test --release`, which
// compile the lib tests in release mode, fail to build.
#[cfg(all(test, debug_assertions))]
mod finalize_tests {
    use super::tests::{assert_same_terms, assert_terms_close, rand_sum};
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
    fn interleaved_layers_and_finalizers_match_the_v0_1_sequence() {
        use crate::engine::sort_merge::apply_layer;

        let input = rand_sum::<1>(1200, 8, 0xAAAA);
        let policy = And(CoefficientThreshold(1e-9), TopN(300));
        let chans: Vec<Box<dyn Channel<1>>> = vec![
            Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.37)),
            Box::new(Clifford1Q::h(0)),
            Box::new(PauliRotation::new(PauliString::<1>::x(5), 0.21)),
        ];

        let mut want = input.clone();
        for ch in &chans {
            want = apply_layer(&want, ch.as_ref(), &policy);
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

// Gated on `debug_assertions` because these tests call `assert_invariants`,
// which is itself debug-only. Matches the convention in `pauli_sum.rs` and
// `sort_merge.rs`; without it `cargo bench` and `cargo test --release`, which
// compile the lib tests in release mode, fail to build.
#[cfg(all(test, debug_assertions))]
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

    use super::tests::{assert_same_terms, rand_sum};
    use super::*;
    use crate::bucket::hash::Gf2Hash;
    use crate::pauli_sum::PauliSum;
    use crate::truncation::builtin::TopN;

    /// A sum whose coefficients take only a handful of distinct magnitudes, so
    /// `TopN` is guaranteed to cut through a large tie group.
    ///
    /// This is not a contrived case: a symmetric Hamiltonian on a periodic
    /// lattice produces many terms related by lattice symmetry with *exactly*
    /// equal coefficients, which is why the 2D Ising example hits it.
    fn tie_heavy_sum(n: usize, num_qubits: usize, seed: u64) -> PauliSum<1> {
        let base = rand_sum::<1>(n, num_qubits, seed);
        let mut acc = crate::accumulator::BuildAccumulator::<1>::with_capacity(num_qubits, n);
        for (i, (x, z, _)) in base.iter().enumerate() {
            // Only 4 distinct magnitudes across the whole sum.
            let mag = [1.0f64, 0.5, 0.25, 0.125][i % 4];
            acc.add_term(
                PauliString::<1> { x: *x, z: *z },
                Phase::ONE,
                Complex64::new(mag, 0.0),
            );
        }
        acc.finalize()
    }

    /// `TopN` must keep the same set regardless of the bucket partition, even
    /// when the cut falls inside a tie group.
    ///
    /// Tie-breaking on flat position would fail this: flat position depends on
    /// which bucket a term landed in, hence on the bucket count.
    #[test]
    fn top_n_is_bucket_count_independent_on_tied_magnitudes() {
        let input = tie_heavy_sum(2000, 8, 0x7135);
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

    /// The bucketed and flat implementations must agree **on ties too**, not
    /// just on distinct magnitudes. This is what lets the two engines produce
    /// identical output on a symmetric Hamiltonian.
    #[test]
    fn top_n_bucketed_matches_flat_on_tied_magnitudes() {
        let input = tie_heavy_sum(2000, 8, 0x7136);
        for n in [3usize, 250, 700, 1200, 1900] {
            let policy = TopN(n);
            let mut flat = input.clone();
            policy.finalize_layer(&mut flat);
            for bits in [0u8, 2, 5, 9] {
                let hash = Gf2Hash::<1>::new(8, bits, 0x99);
                let mut b = input.clone().with_hash(hash);
                policy.finalize_layer(&mut b);
                let got = b;
                assert_same_terms(
                    &got,
                    &flat,
                    &format!("n={n} bits={bits}: keys or coeffs differ on ties"),
                );
            }
        }
    }

    /// The same, across hash seeds: a different `H` permutes bucket membership
    /// without changing anything about the magnitudes.
    #[test]
    fn top_n_is_hash_seed_independent_on_tied_magnitudes() {
        let input = tie_heavy_sum(2000, 8, 0x7123);
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
}
