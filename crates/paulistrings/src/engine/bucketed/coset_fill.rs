//! The per-coset kernel of the bucketed layer: gather, sort and merge one coset in place.

use num_complex::Complex64;

use super::{DeltaPlan, ExtraRows, ZERO};
use crate::channel::prepared::LocalPtm;
use crate::engine::coset::CosetOrder;
use crate::engine::merge::{merge_identity_and_rest, SortKernel, SortScratch};
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::{BucketColumns, ColumnsRef};
use crate::phase::Phase;
use crate::truncation::TruncationPolicy;

#[cfg(feature = "phase-timing")]
use crate::engine::stats::CosetStats;
use crate::engine::timing::{lap, stamp, tally};

/// One coset task's working set: the swapped-out input columns and the per-output-member gather runs.
#[derive(Clone, Debug, Default)]
pub(in crate::engine) struct CosetScratch<const W: usize> {
    /// The coset's input columns, swapped with the live bucket slots so the layer runs in place.
    pub(super) old: Vec<BucketColumns<W>>,
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
    /// The identity-delta stream, already sorted; under a dense identity only `coeff` is filled.
    pub(super) identity: BucketColumns<W>,
    /// Every other delta's rows, unsorted until the per-run sort.
    pub(super) rest: BucketColumns<W>,
}

/// `value != ZERO`, without the `||` short-circuit — one flag, no branch.
pub(super) fn nonzero(value: Complex64) -> bool {
    (value.re != 0.0) | (value.im != 0.0)
}

impl<const W: usize> GatherRun<W> {
    pub(super) fn reset(
        &mut self,
        identity_key_capacity: usize,
        identity_coefficient_capacity: usize,
        rest_capacity: usize,
    ) {
        self.identity.clear();
        self.rest.clear();
        // One slot past the exact capacity: `append_if` writes a discarded row at `len` before deciding not to publish it.
        self.identity.x.reserve(identity_key_capacity + 1);
        self.identity.z.reserve(identity_key_capacity + 1);
        self.identity.coeff.reserve(identity_coefficient_capacity + 1);
        self.rest.x.reserve(rest_capacity + 1);
        self.rest.z.reserve(rest_capacity + 1);
        self.rest.coeff.reserve(rest_capacity + 1);
    }

    /// Write one row at `len` and publish it iff `keep`.
    ///
    /// # Safety
    ///
    /// `len < capacity` on each of the three columns, which [`GatherRun::reset`] guarantees.
    unsafe fn append_if(
        columns: &mut BucketColumns<W>,
        keep: bool,
        x: [u64; W],
        z: [u64; W],
        c: Complex64,
    ) {
        let BucketColumns { x: xs, z: zs, coeff: cs } = columns;
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
    pub(super) fn push_if(&mut self, keep: bool, x: [u64; W], z: [u64; W], c: Complex64) {
        // SAFETY: `reset` reserved `rest_capacity + 1` on all three columns, and `rest_capacity` bounds every row the plan's deltas can emit into this run.
        unsafe { Self::append_if(&mut self.rest, keep, x, z, c) }
    }

    /// Branchless filtered append to the identity stream.
    pub(super) fn push_identity_if(&mut self, keep: bool, x: [u64; W], z: [u64; W], c: Complex64) {
        // SAFETY: as `push_if`, with `identity_key_capacity` / `identity_coefficient_capacity` bounding the identity stream's rows.
        unsafe { Self::append_if(&mut self.identity, keep, x, z, c) }
    }

    /// Unchecked append to the identity coefficient column, the only one a dense identity materializes.
    pub(super) fn push_identity_coefficient(&mut self, c: Complex64) {
        let coeff = &mut self.identity.coeff;
        let len = coeff.len();
        debug_assert!(len < coeff.capacity());
        // SAFETY: `reset` reserved `identity_coefficient_capacity + 1`, and `identity_coefficient_capacity` is the source bucket's length, exactly the number of calls made.
        unsafe {
            coeff.as_mut_ptr().add(len).write(c);
            coeff.set_len(len + 1);
        }
    }

    #[cfg(any(test, feature = "phase-timing"))]
    pub(super) fn len(&self) -> usize {
        self.identity.len() + self.rest.len()
    }
}

/// Panic unless `append_into` left the run's identity stream as it was and grew its rest stream by exactly the rows `count` promised: received rows must never join the identity stream, which the merge requires strictly ascending.
#[cfg(debug_assertions)]
fn debug_assert_received_rows_joined_the_rest<const W: usize>(
    run: &GatherRun<W>,
    (identity_keys, identity_coefficients, rest): (usize, usize, usize),
    promised: usize,
) {
    assert_eq!(
        (run.identity.x.len(), run.identity.z.len(), run.identity.coeff.len()),
        (identity_keys, identity_keys, identity_coefficients),
        "received rows joined the identity stream",
    );
    assert_eq!(
        run.rest.x.len() - rest,
        promised,
        "the rest stream took {} received rows where ExtraRows::count promised {promised}",
        run.rest.x.len() - rest,
    );
    debug_assert_eq!(run.rest.x.len(), run.rest.z.len());
    debug_assert_eq!(run.rest.x.len(), run.rest.coeff.len());
}

/// Below this many cosets the layer runs serially.
pub(in crate::engine) const MIN_COSETS_FOR_PARALLEL: usize = 2;

/// Gather, sort and merge one coset's `2^r` bucket columns in place; `chunk_base` and `order` name each member's original bucket for `extra`.
pub(in crate::engine) fn fill_coset<const W: usize, T, X>(
    chunk: &mut [BucketColumns<W>],
    plan: &DeltaPlan<'_, W>,
    policy: &T,
    scratch: &mut CosetScratch<W>,
    extra: &X,
    chunk_base: usize,
    order: &CosetOrder,
) where
    T: TruncationPolicy<W> + ?Sized,
    X: ExtraRows<W>,
{
    let members = chunk.len();
    let beta_of = |j: usize| -> u32 { order.bucket_at((chunk_base + j) as u32) };
    let CosetScratch { old, runs, sort, .. } = scratch;
    stamp!(stamp);
    old.resize_with(members, BucketColumns::default);
    runs.resize_with(members, GatherRun::default);

    for (slot, columns) in chunk.iter_mut().zip(old.iter_mut()) {
        std::mem::swap(slot, columns);
        slot.clear();
    }
    lap!(stamp, scratch.stats.swap_ns);

    // Exact per-run capacity per stream; it is also the bound `append_if`'s safety relies on.
    for (j, run) in runs.iter_mut().enumerate() {
        let (identity_key_capacity, identity_coefficient_capacity, mut rest_capacity) = match plan {
            DeltaPlan::Local { coordinates, has_identity, dense_identity, .. } => {
                let mut identity = 0usize;
                let mut rest = 0usize;
                for (entry, &coordinate) in coordinates.iter().enumerate() {
                    let len = old[j ^ coordinate as usize].len();
                    if *has_identity && entry == 0 {
                        identity += len;
                    } else {
                        rest += len;
                    }
                }
                (if *dense_identity { 0 } else { identity }, identity, rest)
            }
            DeltaPlan::Rotation {
                coordinate_identity,
                generator_coordinate,
                generator_local,
                ..
            } => (
                0,
                old[j ^ *coordinate_identity as usize].len(),
                if *generator_local { old[j ^ *generator_coordinate as usize].len() } else { 0 },
            ),
        };
        if X::NEEDS_BETA {
            rest_capacity += extra.count(beta_of(j));
        }
        run.reset(identity_key_capacity, identity_coefficient_capacity, rest_capacity);
    }
    lap!(stamp, scratch.stats.size_ns);

    match plan {
        DeltaPlan::Local { ptm, coordinates, has_identity, dense_identity, .. } => {
            if members >= 1 << GATHER_OUTPUT_MAJOR_MIN_R {
                gather_local_output_major(
                    old,
                    runs,
                    ptm,
                    coordinates,
                    *has_identity,
                    *dense_identity,
                );
            } else {
                gather_local_input_major(
                    old,
                    runs,
                    ptm,
                    coordinates,
                    *has_identity,
                    *dense_identity,
                );
            }
        }
        DeltaPlan::Rotation {
            rotation,
            coordinate_identity,
            generator_coordinate,
            generator_local,
        } => {
            // Every term emits one identity row, kept even when `cos == 0` (the signed-zero contract on `merge_identity_and_rest`), so the merge borrows the source keys.
            for (i, source) in old.iter().enumerate() {
                for row in 0..source.len() {
                    let key = PauliString::<W> { x: source.x[row], z: source.z[row] };
                    if key.commutes_with(&rotation.generator) {
                        runs[i ^ *coordinate_identity as usize]
                            .push_identity_coefficient(source.coeff[row]);
                    } else {
                        runs[i ^ *coordinate_identity as usize]
                            .push_identity_coefficient(source.coeff[row] * rotation.cos);
                        if *generator_local {
                            let mut product = key;
                            let phase = product.mul_assign(&rotation.generator);
                            let total = Phase::I + phase;
                            runs[i ^ *generator_coordinate as usize].push_if(
                                true,
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
            #[cfg(debug_assertions)]
            let lens_before = (run.identity.x.len(), run.identity.coeff.len(), run.rest.len());
            extra.append_into(beta_of(j), &mut run.rest);
            #[cfg(debug_assertions)]
            debug_assert_received_rows_joined_the_rest(run, lens_before, extra.count(beta_of(j)));
        }
    }
    lap!(stamp, scratch.stats.gather_ns);

    let sort_kernel = match plan {
        DeltaPlan::Local { sort_kernel, .. } => *sort_kernel,
        DeltaPlan::Rotation { .. } => SortKernel::Comparison,
    };
    for (j, (run, destination)) in runs.iter_mut().zip(chunk.iter_mut()).enumerate() {
        tally!(scratch.stats.rows_gathered, run.len());
        tally!(scratch.stats.rows_sorted, run.rest.len());
        sort_kernel.sort(&mut run.rest, sort);
        lap!(stamp, scratch.stats.sort_ns);
        let (a_x, a_z): (&[[u64; W]], &[[u64; W]]) = match plan {
            DeltaPlan::Local { dense_identity: true, .. } => {
                let source = &old[j];
                debug_assert_eq!(source.len(), run.identity.len());
                tally!(scratch.stats.rows_id, source.len());
                (&source.x, &source.z)
            }
            DeltaPlan::Local { .. } => (&run.identity.x, &run.identity.z),
            DeltaPlan::Rotation { coordinate_identity, .. } => {
                let source = &old[j ^ *coordinate_identity as usize];
                debug_assert_eq!(source.len(), run.identity.len());
                tally!(scratch.stats.rows_id, source.len());
                (&source.x, &source.z)
            }
        };
        let identity = ColumnsRef { x: a_x, z: a_z, coeff: &run.identity.coeff };
        merge_identity_and_rest::<W, T>(identity, run.rest.view(), destination, policy);
        lap!(stamp, scratch.stats.merge_ns);
    }

    for columns in old.iter_mut() {
        columns.clear();
    }
    lap!(stamp, scratch.stats.clear_ns);
    tally!(scratch.stats.cosets, 1);
    tally!(scratch.stats.runs, members);
}

/// Coset dimension at or above which the gather switches to output-major, trading re-reads for fewer open write streams.
/// Both orders gather the same rows, so the choice is performance only; a dense two-qubit gate takes output-major at `r = 4`.
const GATHER_OUTPUT_MAJOR_MIN_R: u8 = 3;

/// Input-major gather for a `Local` plan: each term is loaded once and scattered by `member(i) ⊕ δ = member(i ⊕ coord(δ))`.
// Not a branch on zero amplitudes: research/FINDINGS.md §Branch misprediction: merge loop split and branchless gather filter
pub(super) fn gather_local_input_major<const W: usize>(
    old: &[BucketColumns<W>],
    runs: &mut [GatherRun<W>],
    ptm: &LocalPtm<W>,
    coordinates: &[u32],
    has_identity: bool,
    dense_identity: bool,
) {
    let rest_start = has_identity as usize;
    for (i, source) in old.iter().enumerate() {
        for row in 0..source.len() {
            let pattern = ptm.support_bits(&source.x[row], &source.z[row]);
            if has_identity {
                let amplitude = ptm.deltas()[0].amplitude[pattern];
                if dense_identity {
                    debug_assert!(amplitude != ZERO);
                    runs[i].push_identity_coefficient(source.coeff[row] * amplitude);
                } else {
                    runs[i].push_identity_if(
                        nonzero(amplitude),
                        source.x[row],
                        source.z[row],
                        source.coeff[row] * amplitude,
                    );
                }
            }
            for (entry, delta) in ptm.deltas().iter().enumerate().skip(rest_start) {
                let amplitude = delta.amplitude[pattern];
                let mut key_x = source.x[row];
                let mut key_z = source.z[row];
                for w in 0..W {
                    key_x[w] ^= delta.mask_x[w];
                    key_z[w] ^= delta.mask_z[w];
                }
                runs[i ^ coordinates[entry] as usize].push_if(
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
    old: &[BucketColumns<W>],
    runs: &mut [GatherRun<W>],
    ptm: &LocalPtm<W>,
    coordinates: &[u32],
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
                let amplitude = delta.amplitude[pattern];
                if dense_identity {
                    debug_assert!(amplitude != ZERO);
                    run.push_identity_coefficient(source.coeff[row] * amplitude);
                } else {
                    run.push_identity_if(
                        nonzero(amplitude),
                        source.x[row],
                        source.z[row],
                        source.coeff[row] * amplitude,
                    );
                }
            }
        }
        for (entry, delta) in ptm.deltas().iter().enumerate().skip(rest_start) {
            let source = &old[j ^ coordinates[entry] as usize];
            for row in 0..source.len() {
                let pattern = ptm.support_bits(&source.x[row], &source.z[row]);
                let amplitude = delta.amplitude[pattern];
                let mut key_x = source.x[row];
                let mut key_z = source.z[row];
                for w in 0..W {
                    key_x[w] ^= delta.mask_x[w];
                    key_z[w] ^= delta.mask_z[w];
                }
                run.push_if(nonzero(amplitude), key_x, key_z, source.coeff[row] * amplitude);
            }
        }
    }
}
