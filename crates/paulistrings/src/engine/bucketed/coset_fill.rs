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
    /// The coset's input columns, swapped with the live bucket slots so the layer runs in place.
    pub(super) old: Vec<BucketCols<W>>,
    /// Per-output-member gather runs.
    pub(super) runs: Vec<GatherRun<W>>,
    /// The per-run sort's scratch.
    pub(super) sort: SortScratch<W>,
    /// This slot's busy-time counters.
    #[cfg(feature = "phase-timing")]
    pub(super) stats: CosetStats,
}

/// One output member's gather run, split into the identity stream and the rest.
#[derive(Clone, Debug, Default)]
pub(in crate::engine) struct GatherRun<const W: usize> {
    /// The identity-delta stream, already sorted; under a dense identity only `id_coeff` is filled.
    pub(super) id_x: Vec<[u64; W]>,
    pub(super) id_z: Vec<[u64; W]>,
    pub(super) id_coeff: Vec<Complex64>,
    /// Every other delta's rows, unsorted until the per-run sort.
    pub(super) x: Vec<[u64; W]>,
    pub(super) z: Vec<[u64; W]>,
    pub(super) coeff: Vec<Complex64>,
}

/// `value != ZERO`, without the `||` short-circuit — one flag, no branch.
#[inline(always)]
pub(super) fn nonzero(value: Complex64) -> bool {
    (value.re != 0.0) | (value.im != 0.0)
}

impl<const W: usize> GatherRun<W> {
    #[inline]
    pub(super) fn reset(
        &mut self,
        id_key_capacity: usize,
        id_coeff_capacity: usize,
        rest_capacity: usize,
    ) {
        self.id_x.clear();
        self.id_z.clear();
        self.id_coeff.clear();
        self.x.clear();
        self.z.clear();
        self.coeff.clear();
        // One slot past the exact capacity: `append_if` writes a discarded row at `len` before deciding not to publish it.
        self.id_x.reserve(id_key_capacity + 1);
        self.id_z.reserve(id_key_capacity + 1);
        self.id_coeff.reserve(id_coeff_capacity + 1);
        self.x.reserve(rest_capacity + 1);
        self.z.reserve(rest_capacity + 1);
        self.coeff.reserve(rest_capacity + 1);
    }

    /// Write one row at `len` and publish it iff `keep`.
    ///
    /// # Safety
    ///
    /// `xs.len() < xs.capacity()` and likewise for `zs` and `cs`, which [`GatherRun::reset`] guarantees.
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
        let len = xs.len();
        debug_assert!(len < xs.capacity());
        debug_assert!(len < zs.capacity());
        debug_assert!(len < cs.capacity());
        unsafe {
            xs.as_mut_ptr().add(len).write(x);
            zs.as_mut_ptr().add(len).write(z);
            cs.as_mut_ptr().add(len).write(c);
            let published = len + keep as usize;
            xs.set_len(published);
            zs.set_len(published);
            cs.set_len(published);
        }
    }

    /// Branchless filtered append to the rest stream.
    #[inline]
    pub(super) fn push_if(&mut self, keep: bool, x: [u64; W], z: [u64; W], c: Complex64) {
        // SAFETY: `reset` reserved `rest_capacity + 1` on all three columns, and `rest_capacity` bounds every row the plan's deltas can emit into this run.
        unsafe { Self::append_if(&mut self.x, &mut self.z, &mut self.coeff, keep, x, z, c) }
    }

    /// Unconditional append to the rest stream.
    #[inline]
    pub(super) fn push_row(&mut self, x: [u64; W], z: [u64; W], c: Complex64) {
        self.push_if(true, x, z, c);
    }

    /// Branchless filtered append to the identity stream.
    #[inline]
    pub(super) fn push_id_if(&mut self, keep: bool, x: [u64; W], z: [u64; W], c: Complex64) {
        // SAFETY: as `push_if`, with `id_key_capacity` / `id_coeff_capacity` bounding the identity stream's rows.
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

    /// Unchecked append to the identity coefficient column, the only one a dense identity materializes.
    #[inline]
    pub(super) fn push_id_coeff(&mut self, c: Complex64) {
        let len = self.id_coeff.len();
        debug_assert!(len < self.id_coeff.capacity());
        // SAFETY: `reset` reserved `id_coeff_capacity + 1`, and `id_coeff_capacity` is the source bucket's length, exactly the number of calls made.
        unsafe {
            self.id_coeff.as_mut_ptr().add(len).write(c);
            self.id_coeff.set_len(len + 1);
        }
    }

    #[cfg(any(test, feature = "phase-timing"))]
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.id_coeff.len() + self.coeff.len()
    }
}

/// Below this many cosets the layer runs serially.
pub(in crate::engine) const MIN_COSETS_FOR_PARALLEL: usize = 2;

/// Gather, sort and merge one coset's `2^r` bucket columns in place; `chunk_base` and `inverse_permutation` name each member's original bucket for `extra`.
pub(in crate::engine) fn fill_coset<const W: usize, T, X>(
    chunk: &mut [BucketCols<W>],
    plan: &DeltaPlan<'_, W>,
    policy: &T,
    scratch: &mut CosetScratch<W>,
    extra: &X,
    chunk_base: usize,
    inverse_permutation: &[u32],
) where
    T: TruncationPolicy<W> + ?Sized,
    X: ExtraRows<W>,
{
    let members = chunk.len();
    let beta_of = |j: usize| -> u32 {
        if inverse_permutation.is_empty() {
            (chunk_base + j) as u32
        } else {
            inverse_permutation[chunk_base + j]
        }
    };
    #[cfg(feature = "phase-timing")]
    let CosetScratch {
        old,
        runs,
        sort,
        stats,
    } = scratch;
    #[cfg(not(feature = "phase-timing"))]
    let CosetScratch { old, runs, sort } = scratch;
    #[cfg(feature = "phase-timing")]
    let mut stamp = Stamp::now();
    old.resize_with(members, BucketCols::default);
    runs.resize_with(members, GatherRun::default);

    for (slot, cols) in chunk.iter_mut().zip(old.iter_mut()) {
        std::mem::swap(slot, cols);
        slot.clear();
    }
    #[cfg(feature = "phase-timing")]
    stamp.lap(&mut stats.swap_ns);

    // Exact per-run capacity per stream; it is also the bound `append_if`'s safety relies on.
    for (j, run) in runs.iter_mut().enumerate() {
        let (id_key_capacity, id_coeff_capacity, mut rest_capacity): (usize, usize, usize) =
            match plan {
                DeltaPlan::Local {
                    coords,
                    has_identity,
                    dense_identity,
                    ..
                } => {
                    let mut id = 0usize;
                    let mut rest = 0usize;
                    for (entry, &coord) in coords.iter().enumerate() {
                        let len = old[j ^ coord as usize].len();
                        if *has_identity && entry == 0 {
                            id += len;
                        } else {
                            rest += len;
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
                    if *gen_local {
                        old[j ^ *coord_gen as usize].len()
                    } else {
                        0
                    },
                ),
            };
        if X::NEEDS_BETA {
            rest_capacity += extra.count(beta_of(j));
        }
        run.reset(id_key_capacity, id_coeff_capacity, rest_capacity);
    }
    #[cfg(feature = "phase-timing")]
    stamp.lap(&mut stats.size_ns);

    match plan {
        DeltaPlan::Local {
            ptm,
            coords,
            has_identity,
            dense_identity,
            ..
        } => {
            if members >= 1 << GATHER_OUTPUT_MAJOR_MIN_R {
                gather_local_output_major(old, runs, ptm, coords, *has_identity, *dense_identity);
            } else {
                gather_local_input_major(old, runs, ptm, coords, *has_identity, *dense_identity);
            }
        }
        DeltaPlan::Rotation {
            rotation,
            coord_identity,
            coord_gen,
            gen_local,
        } => {
            // Every term emits one id row, kept even when `cos == 0` (the signed-zero contract on `merge2_into`), so the merge borrows the source keys.
            for (i, source) in old.iter().enumerate() {
                for row in 0..source.len() {
                    let key = PauliString::<W> {
                        x: source.x[row],
                        z: source.z[row],
                    };
                    if key.commutes_with(&rotation.gen) {
                        runs[i ^ *coord_identity as usize].push_id_coeff(source.coeff[row]);
                    } else {
                        runs[i ^ *coord_identity as usize]
                            .push_id_coeff(source.coeff[row] * rotation.cos);
                        if *gen_local {
                            let mut product = key;
                            let phase = product.mul_assign(&rotation.gen);
                            let total = Phase::I + phase;
                            runs[i ^ *coord_gen as usize].push_row(
                                product.x,
                                product.z,
                                total.apply(source.coeff[row]) * rotation.sin,
                            );
                        }
                    }
                }
            }
        }
    }
    if X::NEEDS_BETA {
        for (j, run) in runs.iter_mut().enumerate() {
            extra.append_into(beta_of(j), &mut run.x, &mut run.z, &mut run.coeff);
        }
    }
    #[cfg(feature = "phase-timing")]
    stamp.lap(&mut stats.gather_ns);

    let radix = matches!(
        plan,
        DeltaPlan::Local {
            radix_sort: true,
            ..
        }
    );
    for (j, (run, destination)) in runs.iter_mut().zip(chunk.iter_mut()).enumerate() {
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
        stamp.lap(&mut stats.sort_ns);
        let (a_x, a_z): (&[[u64; W]], &[[u64; W]]) = match plan {
            DeltaPlan::Local {
                dense_identity: true,
                ..
            } => {
                let source = &old[j];
                debug_assert_eq!(source.len(), run.id_coeff.len());
                #[cfg(feature = "phase-timing")]
                {
                    stats.rows_id += source.len() as u64;
                }
                (&source.x, &source.z)
            }
            DeltaPlan::Local { .. } => (&run.id_x, &run.id_z),
            DeltaPlan::Rotation { coord_identity, .. } => {
                let source = &old[j ^ *coord_identity as usize];
                debug_assert_eq!(source.len(), run.id_coeff.len());
                #[cfg(feature = "phase-timing")]
                {
                    stats.rows_id += source.len() as u64;
                }
                (&source.x, &source.z)
            }
        };
        merge2_into::<W, T>(
            a_x,
            a_z,
            &run.id_coeff,
            &run.x,
            &run.z,
            &run.coeff,
            &mut destination.x,
            &mut destination.z,
            &mut destination.coeff,
            policy,
        );
        #[cfg(feature = "phase-timing")]
        stamp.lap(&mut stats.merge_ns);
    }

    for cols in old.iter_mut() {
        cols.clear();
    }
    #[cfg(feature = "phase-timing")]
    {
        stamp.lap(&mut stats.clear_ns);
        stats.cosets += 1;
        stats.runs += members as u64;
    }
}

/// Coset dimension at or above which the gather switches to output-major, trading re-reads for fewer open write streams.
/// Both orders gather the same rows, so the choice is performance only; no built-in channel reaches it.
const GATHER_OUTPUT_MAJOR_MIN_R: u8 = 3;

/// Input-major gather for a `Local` plan: each term is loaded once and scattered by `member(i) ⊕ δ = member(i ⊕ coord(δ))`.
// Not a branch on zero amplitudes: research/FINDINGS.md §Branch misprediction: merge loop split and branchless gather filter
pub(super) fn gather_local_input_major<const W: usize>(
    old: &[BucketCols<W>],
    runs: &mut [GatherRun<W>],
    ptm: &LocalPtm<W>,
    coords: &[u32],
    has_identity: bool,
    dense_identity: bool,
) {
    let rest_start = has_identity as usize;
    for (i, source) in old.iter().enumerate() {
        for row in 0..source.len() {
            let pattern = ptm.support_bits(&source.x[row], &source.z[row]);
            if has_identity {
                let amplitude = ptm.deltas()[0].amp[pattern];
                if dense_identity {
                    debug_assert!(amplitude != ZERO);
                    runs[i].push_id_coeff(source.coeff[row] * amplitude);
                } else {
                    runs[i].push_id_if(
                        nonzero(amplitude),
                        source.x[row],
                        source.z[row],
                        source.coeff[row] * amplitude,
                    );
                }
            }
            for (entry, delta) in ptm.deltas().iter().enumerate().skip(rest_start) {
                let amplitude = delta.amp[pattern];
                let mut key_x = source.x[row];
                let mut key_z = source.z[row];
                for w in 0..W {
                    key_x[w] ^= delta.mask_x[w];
                    key_z[w] ^= delta.mask_z[w];
                }
                runs[i ^ coords[entry] as usize].push_if(
                    nonzero(amplitude),
                    key_x,
                    key_z,
                    source.coeff[row] * amplitude,
                );
            }
        }
    }
}

/// Output-major gather for a `Local` plan: each output member streams one input bucket per delta.
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
            let delta = &ptm.deltas()[0];
            let source = &old[j];
            for row in 0..source.len() {
                let pattern = ptm.support_bits(&source.x[row], &source.z[row]);
                let amplitude = delta.amp[pattern];
                if dense_identity {
                    debug_assert!(amplitude != ZERO);
                    run.push_id_coeff(source.coeff[row] * amplitude);
                } else {
                    run.push_id_if(
                        nonzero(amplitude),
                        source.x[row],
                        source.z[row],
                        source.coeff[row] * amplitude,
                    );
                }
            }
        }
        for (entry, delta) in ptm.deltas().iter().enumerate().skip(rest_start) {
            let source = &old[j ^ coords[entry] as usize];
            for row in 0..source.len() {
                let pattern = ptm.support_bits(&source.x[row], &source.z[row]);
                let amplitude = delta.amp[pattern];
                let mut key_x = source.x[row];
                let mut key_z = source.z[row];
                for w in 0..W {
                    key_x[w] ^= delta.mask_x[w];
                    key_z[w] ^= delta.mask_z[w];
                }
                run.push_if(
                    nonzero(amplitude),
                    key_x,
                    key_z,
                    source.coeff[row] * amplitude,
                );
            }
        }
    }
}
