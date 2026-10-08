//! The per-coset kernel of the bucketed layer: gather, sort and merge one coset in place.

use num_complex::Complex64;

use super::{DeltaPlan, ExtraRows, ZERO};
use crate::channel::prepared::LocalPtm;
use crate::engine::merge::{
    merge2_into, sort_rows_radix_with_scratch, sort_rows_with_scratch, SortScratch,
};
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::BucketCols;
use crate::phase::Phase;
use crate::truncation::TruncationPolicy;

#[cfg(feature = "phase-timing")]
use crate::engine::stats::{CosetStats, Stamp};

/// One coset task's working set: the swapped-out input columns and the per-output-member gather runs.
#[derive(Clone, Debug, Default)]
pub(in crate::engine) struct CosetScratch<const W: usize> {
    /// The coset's input columns, `mem::swap`ped with the live bucket slots so the layer runs in place: bucket capacity circulates through here instead of through a second full-sum copy.
    pub(super) old: Vec<BucketCols<W>>,
    /// Per-output-member gather runs.
    pub(super) runs: Vec<GatherRun<W>>,
    /// Scratch for `sort_rows_with_scratch`'s per-run sort, reused across every run in every coset this scratch instance handles.
    pub(super) sort: SortScratch<W>,
    /// This slot's busy-time phase counters, drained by `LayerScratch::take_stats`.
    #[cfg(feature = "phase-timing")]
    pub(super) stats: CosetStats,
}

/// One output member's gather run: key columns and coefficients.
/// Equal-key summation order is not pinned by a sort tiebreak — see the module doc and `merge::sort_rows_with_scratch`.
#[derive(Clone, Debug, Default)]
pub(in crate::engine) struct GatherRun<const W: usize> {
    /// The identity-delta stream: keys untouched, so it inherits the source bucket's strictly-ascending, duplicate-free order and is never sorted. Under a dense identity plan only `id_coeff` is populated, aligned 1:1 with `old[j]` whose keys the merge borrows in place; under a sparse plan all three columns are filled.
    pub(super) id_x: Vec<[u64; W]>,
    pub(super) id_z: Vec<[u64; W]>,
    pub(super) id_coeff: Vec<Complex64>,
    /// Every other delta's rows — keys XOR'd by a constant mask, so generally unsorted; canonicalized per run by `sort_rows_with_scratch`.
    pub(super) x: Vec<[u64; W]>,
    pub(super) z: Vec<[u64; W]>,
    pub(super) coeff: Vec<Complex64>,
}

/// `a != ZERO`, without the `||` short-circuit — one flag, no branch.
#[inline(always)]
pub(super) fn nonzero(a: Complex64) -> bool {
    (a.re != 0.0) | (a.im != 0.0)
}

impl<const W: usize> GatherRun<W> {
    #[inline]
    pub(super) fn reset(&mut self, cap_id_keys: usize, cap_id_coeff: usize, cap_rest: usize) {
        self.id_x.clear();
        self.id_z.clear();
        self.id_coeff.clear();
        self.x.clear();
        self.z.clear();
        self.coeff.clear();
        // One slot past the exact capacity: `push_if` writes a discarded row at `len` before
        // deciding not to publish it via `set_len`, so the write must stay in bounds even when
        // every countable row is kept.
        self.id_x.reserve(cap_id_keys + 1);
        self.id_z.reserve(cap_id_keys + 1);
        self.id_coeff.reserve(cap_id_coeff + 1);
        self.x.reserve(cap_rest + 1);
        self.z.reserve(cap_rest + 1);
        self.coeff.reserve(cap_rest + 1);
    }

    /// Write one row at `len` in a three-column stream and publish it iff `keep`.
    /// The single unsafe block behind every gather append, so there is one invariant to audit
    /// rather than several copies of it that drift apart.
    ///
    /// # Safety
    ///
    /// `xs.len() < xs.capacity()` and likewise for `zs` and `cs`. The caller gets this from
    /// [`GatherRun::reset`], which reserves one slot past the plan's exact per-run capacity.
    #[inline]
    unsafe fn append_if(
        xs: &mut Vec<[u64; W]>,
        zs: &mut Vec<[u64; W]>,
        cs: &mut Vec<Complex64>,
        keep: bool,
        x: [u64; W],
        z: [u64; W],
        c: Complex64,
    ) {
        let n = xs.len();
        debug_assert!(n < xs.capacity());
        debug_assert!(n < zs.capacity());
        debug_assert!(n < cs.capacity());
        unsafe {
            xs.as_mut_ptr().add(n).write(x);
            zs.as_mut_ptr().add(n).write(z);
            cs.as_mut_ptr().add(n).write(c);
            let m = n + keep as usize;
            xs.set_len(m);
            zs.set_len(m);
            cs.set_len(m);
        }
    }

    /// Branchless filtered append to the rest stream.
    #[inline]
    pub(super) fn push_if(&mut self, keep: bool, x: [u64; W], z: [u64; W], c: Complex64) {
        // SAFETY: `reset` reserved `cap_rest + 1` on all three columns, and `cap_rest` bounds
        // every row the plan's deltas can emit into this run.
        unsafe { Self::append_if(&mut self.x, &mut self.z, &mut self.coeff, keep, x, z, c) }
    }

    /// Unconditional append to the rest stream — [`Self::push_if`] with the predicate known true, which is what the rotation generator pass wants.
    #[inline]
    pub(super) fn push_row(&mut self, x: [u64; W], z: [u64; W], c: Complex64) {
        self.push_if(true, x, z, c);
    }

    /// Branchless filtered append to the identity stream.
    #[inline]
    pub(super) fn push_id_if(&mut self, keep: bool, x: [u64; W], z: [u64; W], c: Complex64) {
        // SAFETY: as `push_if`, with `cap_id_keys` / `cap_id_coeff` bounding the identity stream's rows.
        unsafe {
            Self::append_if(
                &mut self.id_x,
                &mut self.id_z,
                &mut self.id_coeff,
                keep,
                x,
                z,
                c,
            )
        }
    }

    /// Unchecked unconditional append to the identity *coefficient* column.
    /// The dense-identity plans emit exactly one identity row per source row and borrow the
    /// keys from the source bucket, so this is the only column materialized — hence its own
    /// helper rather than [`Self::push_id_if`].
    #[inline]
    pub(super) fn push_id_coeff(&mut self, c: Complex64) {
        let n = self.id_coeff.len();
        debug_assert!(n < self.id_coeff.capacity());
        // SAFETY: `reset` reserved `cap_id_coeff + 1`, and `cap_id_coeff` is the source
        // bucket's length — exactly the number of calls this loop makes.
        unsafe {
            self.id_coeff.as_mut_ptr().add(n).write(c);
            self.id_coeff.set_len(n + 1);
        }
    }

    #[cfg(any(test, feature = "phase-timing"))]
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.id_coeff.len() + self.coeff.len()
    }
}

/// Below this many cosets there is nothing to spread, so skip Rayon entirely — mostly the
/// `bits = 0` / `r = bits` case, where one coset spans every bucket and the layer degenerates
/// to a single whole-sum task on the same code path.
pub(in crate::engine) const MIN_COSETS_FOR_PARALLEL: usize = 2;

/// Gather, sort and merge one coset, in place. The unit of parallel work.
/// `chunk` holds the coset's `2^r` bucket columns, members ascending by basis coordinate,
/// serving as both input source and output destination.
/// `extra` supplies rows generated elsewhere (the partitioned engine); they join each member's
/// rest stream between the gather and the sort, so the merge sees the complete sum. Naming
/// their destination needs the member's original bucket index, recovered from `chunk_base` and
/// `inv_perm`. Under [`NoExtra`] all of that is behind `X::NEEDS_BETA == false` and compiles away.
pub(in crate::engine) fn fill_coset<const W: usize, T, X>(
    chunk: &mut [BucketCols<W>],
    plan: &DeltaPlan<'_, W>,
    policy: &T,
    ws: &mut CosetScratch<W>,
    extra: &X,
    chunk_base: usize,
    inv_perm: &[u32],
) where
    T: TruncationPolicy<W> + ?Sized,
    X: ExtraRows<W>,
{
    let m = chunk.len();
    // Member `j`'s original bucket index. An empty `inv_perm` means the layer skipped the
    // handle permutation (`r = 0`), where the permuted slot is the bucket.
    let beta_of = |j: usize| -> u32 {
        if inv_perm.is_empty() {
            (chunk_base + j) as u32
        } else {
            inv_perm[chunk_base + j]
        }
    };
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

    // Swap the coset's columns out. The chunk slots inherit this scratch's cleared,
    // capacity-retaining columns and become the write destinations — capacities circulate
    // between buckets across cosets, which holds the steady state allocation-free in aggregate.
    for (slot, cols) in chunk.iter_mut().zip(old.iter_mut()) {
        std::mem::swap(slot, cols);
        slot.clear();
    }
    #[cfg(feature = "phase-timing")]
    st.lap(&mut stats.swap_ns);

    // Exact per-run capacity, counted once per delta entry, split by destination stream: the
    // identity entry feeds the pre-sorted `id` columns, everything else the sorted rest. Under
    // a dense identity the id key columns stay empty since the merge borrows the source
    // bucket's keys, so only `id_coeff` needs capacity.
    for (j, run) in runs.iter_mut().enumerate() {
        let (cap_id_keys, cap_id_coeff, mut cap_rest): (usize, usize, usize) = match plan {
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
                gen_local,
                ..
            } => (
                0,
                old[j ^ *coord_identity as usize].len(),
                // A remote generator emits nothing here, so the rest stream is whatever
                // `extra` adds below and nothing else.
                if *gen_local {
                    old[j ^ *coord_gen as usize].len()
                } else {
                    0
                },
            ),
        };
        // Received rows land in the rest stream, so they belong to its exact capacity.
        // `const false` under `NoExtra`.
        if X::NEEDS_BETA {
            cap_rest += extra.count(beta_of(j));
        }
        run.reset(cap_id_keys, cap_id_coeff, cap_rest);
    }
    #[cfg(feature = "phase-timing")]
    st.lap(&mut stats.size_ns);

    // Gather. Two visit orders produce the same multiset of rows per run, differing only in
    // arrival order, which the key-only sort erases up to floating-point tolerance (see
    // `local_gather_orders_agree_to_fp_tolerance`). Which order is faster depends on `r`; see
    // `GATHER_OUTPUT_MAJOR_MIN_R`.
    match plan {
        DeltaPlan::Local {
            ptm,
            coords,
            has_identity,
            dense_identity,
            ..
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
            gen_local,
        } => {
            // The identity pass and the generator pass. Every term emits exactly one
            // identity-pass row (full coefficient when it commutes, `cos`-scaled when it
            // doesn't, kept even when `cos == 0` — see `merge2_into` on signed zeros), so the
            // id stream is the whole source bucket in order and only the coefficient is
            // materialized; the merge borrows the keys from the source bucket in place.
            for (i, src) in old.iter().enumerate() {
                for t in 0..src.len() {
                    let v = PauliString::<W> {
                        x: src.x[t],
                        z: src.z[t],
                    };
                    if v.commutes_with(&prep.gen) {
                        runs[i ^ *coord_identity as usize].push_id_coeff(src.coeff[t]);
                    } else {
                        runs[i ^ *coord_identity as usize].push_id_coeff(src.coeff[t] * prep.cos);
                        // `true` on every unpartitioned layer; under a partitioning that sees
                        // the generator, these rows belong to a partner and the export pass
                        // has already shipped them.
                        if *gen_local {
                            let mut prod = v;
                            let phase = prod.mul_assign(&prep.gen);
                            let total = Phase::I + phase;
                            runs[i ^ *coord_gen as usize].push_row(
                                prod.x,
                                prod.z,
                                total.apply(src.coeff[t]) * prep.sin,
                            );
                        }
                    }
                }
            }
        }
    }
    // Rows generated on other partitions whose output bucket lives here. The rest stream only:
    // it is sorted below, so a received row may duplicate a local key and `merge2_into` still
    // sees the complete sum before `keep_term` runs. `const false` under `NoExtra`.
    if X::NEEDS_BETA {
        for (j, run) in runs.iter_mut().enumerate() {
            extra.append_into(beta_of(j), &mut run.x, &mut run.z, &mut run.coeff);
        }
    }
    #[cfg(feature = "phase-timing")]
    st.lap(&mut stats.gather_ns);

    // Sort each run's rest stream by key alone, then fuse the two-stream merge with the
    // segmented reduction into the member's live slot: the id stream never moves through the
    // sort. Under a dense identity plan the id stream's keys were never materialized either —
    // they are the source bucket's own key columns, borrowed here.
    // Which sort kernel this layer uses, hoisted out of the run loop: a per-layer property of
    // the plan, never a per-run one.
    let radix = matches!(
        plan,
        DeltaPlan::Local {
            radix_sort: true,
            ..
        }
    );
    for (j, (run, dst)) in runs.iter_mut().zip(chunk.iter_mut()).enumerate() {
        #[cfg(feature = "phase-timing")]
        {
            stats.rows_gathered += run.len() as u64;
            stats.rows_sorted += run.coeff.len() as u64;
        }
        if radix {
            sort_rows_radix_with_scratch(&mut run.x, &mut run.z, &mut run.coeff, sort);
        } else {
            sort_rows_with_scratch(&mut run.x, &mut run.z, &mut run.coeff, sort);
        }
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

    // Leave `old` cleared so the next coset's swap hands its chunk clean, capacity-retaining
    // columns. Runs are cleared by their own `reset`.
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
/// Input-major loads each term once but keeps `2^r` write streams open per task; at large `r`
/// those streams plus the swapped coset no longer fit L2. Output-major re-reads each input
/// bucket `2^r` times but keeps a single write stream, trading re-reads (coset-local, cheap)
/// for write-stream pressure. Below this threshold the extra re-reads cost more than the write
/// streams save; above it, the reverse. Both orders gather the same multiset of rows, so the
/// choice is a pure performance knob, never a correctness one — pinned by
/// `local_gather_orders_agree_to_fp_tolerance`. See `research/FINDINGS.md` for the measurements
/// behind the current value.
const GATHER_OUTPUT_MAJOR_MIN_R: u8 = 3;

/// Input-major gather for a tabulated (`Local`) plan: each term is loaded once and its whole
/// fanout is scattered by `member(i) ⊕ δ = member(i ⊕ coord(δ))`. Rows land in the runs in
/// (input member, input position, delta) order.
///
/// The zero-amplitude filter is branchless: the row is always materialized and
/// [`GatherRun::push_if`] publishes it only when the amplitude is nonzero, avoiding a
/// data-dependent branch on which PTM entries vanish for a given support pattern (see
/// `research/FINDINGS.md`).
pub(super) fn gather_local_input_major<const W: usize>(
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
                // Entry 0 is the identity delta: masks are zero, so the row lands in this
                // member's own run with its key untouched — the pre-sorted id stream.
                let a = ptm.deltas()[0].amp[s];
                if dense_identity {
                    // Dense: `a` never vanishes and the stream is 1:1 with the source rows,
                    // so only the coefficient is stored — the merge borrows the keys from `old[i]`.
                    debug_assert!(a != ZERO);
                    runs[i].push_id_coeff(src.coeff[t] * a);
                } else {
                    runs[i].push_id_if(nonzero(a), src.x[t], src.z[t], src.coeff[t] * a);
                }
            }
            for (e, d) in ptm.deltas().iter().enumerate().skip(rest_start) {
                let a = d.amp[s];
                let mut kx = src.x[t];
                let mut kz = src.z[t];
                for w in 0..W {
                    kx[w] ^= d.mask_x[w];
                    kz[w] ^= d.mask_z[w];
                }
                runs[i ^ coords[e] as usize].push_if(nonzero(a), kx, kz, src.coeff[t] * a);
            }
        }
    }
}

/// Output-major gather for a tabulated (`Local`) plan: for each output member, stream the one
/// input bucket per delta entry and append to that member's run only. Rows land in (delta,
/// input position) order — the same multiset as [`gather_local_input_major`] in a different
/// order; they agree only up to floating-point tolerance (see
/// `local_gather_orders_agree_to_fp_tolerance`).
///
/// The zero-amplitude filter is branchless, as in [`gather_local_input_major`], and helps here
/// even though output-major is only reached on a dense PTM with nothing to filter: what
/// [`GatherRun::push_if`] removes is the `Vec::push` capacity check and length increment per row.
pub(super) fn gather_local_output_major<const W: usize>(
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
            // Entry 0: masks zero — the member's own bucket streams into the pre-sorted id
            // columns; coefficient only when the identity is dense (keys borrowed).
            let d = &ptm.deltas()[0];
            let src = &old[j];
            for t in 0..src.len() {
                let s = ptm.support_bits(&src.x[t], &src.z[t]);
                let a = d.amp[s];
                if dense_identity {
                    debug_assert!(a != ZERO);
                    run.push_id_coeff(src.coeff[t] * a);
                } else {
                    run.push_id_if(nonzero(a), src.x[t], src.z[t], src.coeff[t] * a);
                }
            }
        }
        for (e, d) in ptm.deltas().iter().enumerate().skip(rest_start) {
            let src = &old[j ^ coords[e] as usize];
            for t in 0..src.len() {
                let s = ptm.support_bits(&src.x[t], &src.z[t]);
                let a = d.amp[s];
                let mut kx = src.x[t];
                let mut kz = src.z[t];
                for w in 0..W {
                    kx[w] ^= d.mask_x[w];
                    kz[w] ^= d.mask_z[w];
                }
                run.push_if(nonzero(a), kx, kz, src.coeff[t] * a);
            }
        }
    }
}
