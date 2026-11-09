//! Runs of one-to-one channels applied as one pass: every string walks the steps that touch its qubits, then the sum is rehashed and each bucket sorted once.
//!
//! Contracts: a [`PermutationStep`] maps distinct strings to distinct strings, so a run never sums two strings and agrees with the per-layer path to floating-point tolerance (its coefficients see the same products in the same order); every intermediate string is dropped exactly when the per-layer merge would drop it (an exact zero, or `keep_term` refusing it), with `keep_term` taken as a function of its arguments, so a step that leaves a string unchanged does not ask it again; the run skips the layer passes between its steps, so it runs only within the policy's `TruncationPolicy::run_length_limit`.

use rayon::prelude::*;

use super::RotationLayer;
use crate::channel::prepared::{Prepared, LOCAL_DIM, MAX_LOCAL_SUPPORT};
use crate::coefficient::Coefficient;
use crate::engine::merge::{SortKernel, SortScratch};
use crate::pauli_sum::storage::{BucketColumns, PauliSum};
use crate::truncation::TruncationPolicy;

/// Steps per segment of a walk, the width of a [`StepSet`].
const SEGMENT_STEPS: usize = 256;

/// A set of a segment's steps, one bit per step.
type StepSet = [u64; SEGMENT_STEPS / 64];

/// Below this many terms a run walks, scatters and sorts on the calling thread.
const PARALLEL_MIN_TERMS: usize = 1 << 13;

/// A rehashed bucket holds `1 / RUN_BUCKET_SLACK_DIVISOR` spare capacity, so the next rotation layer can merge it in place (research/FINDINGS.md §Clifford runs as one pass).
const RUN_BUCKET_SLACK_DIVISOR: usize = 2;

/// Source-bucket chunks per Rayon thread in a parallel run, each with its own row of destination counts.
const CHUNKS_PER_THREAD: usize = 4;

/// A prepared channel as a permutation of strings: support pattern `s` moves to `s ⊕ delta[s]` with its coefficient times `amplitude[s]`, and an exact-zero amplitude drops it.
#[derive(Clone, Debug)]
pub(crate) struct PermutationStep<const W: usize, C: Coefficient> {
    /// Word and bit of support qubit `j`; a one-qubit step repeats qubit 0 and masks its second pattern pair off.
    word: [usize; MAX_LOCAL_SUPPORT],
    bit: [u32; MAX_LOCAL_SUPPORT],
    /// `0b11` for a two-qubit step, `0` for a one-qubit step: the pattern bits of the second qubit.
    second: usize,
    /// The support qubits, the first repeated for a one-qubit step.
    qubit: [usize; MAX_LOCAL_SUPPORT],
    /// Bit `s` when pattern `s` keeps its key and coefficient exactly.
    inert: u16,
    /// Per pattern, bit `j` when support qubit `j` is the identity before the step and not after.
    grows: [u8; LOCAL_DIM],
    delta: [u8; LOCAL_DIM],
    /// `h(lift(delta[s]))` under the hash the channel was prepared for.
    bucket_delta: [u32; LOCAL_DIM],
    amplitude: [C; LOCAL_DIM],
}

impl<const W: usize, C: Coefficient> PermutationStep<W, C> {
    /// The permutation `prepared` applies, as the per-layer path applies it, or `None` when it can split or merge strings.
    ///
    /// A quarter-turn rotation table counts as its Clifford arm, which drops the identity residue.
    pub(crate) fn of(prepared: &Prepared<W, C>) -> Option<Self> {
        let Prepared::Local(ptm) = prepared else { return None };
        let k = ptm.k();
        if k == 0 {
            return None;
        }
        let dim = 1usize << (2 * k);
        let deltas = ptm.deltas();
        // Per pattern, the one entry that emits for it.
        let mut emitting = [None::<usize>; LOCAL_DIM];
        match RotationLayer::of(ptm) {
            Some(layer) if !layer.is_clifford() => return None,
            Some(_) => {
                for (s, slot) in emitting.iter_mut().enumerate().take(dim) {
                    *slot = Some(usize::from(deltas[1].amplitude[s] != C::ZERO));
                }
            }
            None => {
                for (s, slot) in emitting.iter_mut().enumerate().take(dim) {
                    let mut entries =
                        (0..deltas.len()).filter(|&e| deltas[e].amplitude[s] != C::ZERO);
                    *slot = entries.next();
                    if entries.next().is_some() {
                        return None;
                    }
                }
            }
        }
        let mut reached = [false; LOCAL_DIM];
        let mut delta = [0u8; LOCAL_DIM];
        let mut bucket_delta = [0u32; LOCAL_DIM];
        let mut amplitude = [C::ZERO; LOCAL_DIM];
        for s in 0..dim {
            let Some(entry) = emitting[s] else { continue };
            let entry = &deltas[entry];
            let target = s ^ entry.local_delta as usize;
            if std::mem::replace(&mut reached[target], true) {
                return None;
            }
            delta[s] = entry.local_delta;
            bucket_delta[s] = entry.bucket_delta;
            amplitude[s] = entry.amplitude[s];
        }
        let qubits: Vec<usize> = ptm.qubits().collect();
        let last = qubits[k - 1];
        let mut inert = 0u16;
        let mut grows = [0u8; LOCAL_DIM];
        for s in 0..dim {
            inert |= u16::from(delta[s] == 0 && amplitude[s] == C::ONE) << s;
            let t = s ^ delta[s] as usize;
            for j in 0..k {
                let occupied = |pattern: usize| (pattern >> (2 * j)) & 0b11 != 0;
                grows[s] |= u8::from(!occupied(s) && occupied(t)) << j;
            }
        }
        let word = [qubits[0] / 64, last / 64];
        let bit = [(qubits[0] % 64) as u32, (last % 64) as u32];
        Some(Self {
            word,
            bit,
            second: if k == 2 { 0b11 } else { 0 },
            qubit: [qubits[0], last],
            inert,
            grows,
            delta,
            bucket_delta,
            amplitude,
        })
    }

    /// The step as the device walks it: its two support qubits (the first repeated for a one-qubit step), whether the second is one, the inert patterns, and per pattern the local delta and amplitude.
    #[cfg(feature = "cuda")]
    pub(crate) fn walk_table(&self) -> StepWalk<'_, C> {
        StepWalk {
            qubits: self.qubit,
            two_qubit: self.second != 0,
            inert: self.inert,
            delta: &self.delta,
            amplitude: &self.amplitude,
        }
    }

    /// The support pattern of a key, packed as the prepared table's `local_delta`.
    #[inline(always)]
    fn pattern(&self, x: &[u64; W], z: &[u64; W]) -> usize {
        let [w0, w1] = self.word;
        let [b0, b1] = self.bit;
        let first = ((x[w0] >> b0) & 1) | (((z[w0] >> b0) & 1) << 1);
        let second = ((x[w1] >> b1) & 1) | (((z[w1] >> b1) & 1) << 1);
        first as usize | ((second as usize & self.second) << 2)
    }

    /// `key ⊕ lift(delta[s])`.
    #[inline(always)]
    fn move_key(&self, x: &mut [u64; W], z: &mut [u64; W], s: usize) {
        let d = self.delta[s] as u64;
        let [w0, w1] = self.word;
        let [b0, b1] = self.bit;
        x[w0] ^= (d & 1) << b0;
        z[w0] ^= ((d >> 1) & 1) << b0;
        x[w1] ^= ((d >> 2) & 1) << b1;
        z[w1] ^= ((d >> 3) & 1) << b1;
    }
}

/// [`PermutationStep::walk_table`].
#[cfg(feature = "cuda")]
pub(crate) struct StepWalk<'s, C> {
    pub(crate) qubits: [usize; MAX_LOCAL_SUPPORT],
    pub(crate) two_qubit: bool,
    pub(crate) inert: u16,
    pub(crate) delta: &'s [u8; LOCAL_DIM],
    pub(crate) amplitude: &'s [C; LOCAL_DIM],
}

/// The working set of [`apply_permutation_run`], kept across runs for its allocations.
#[derive(Debug, Default)]
pub(crate) struct PermutationScratch {
    /// Per surviving row, in source order, the bucket it moves to.
    destinations: Vec<u32>,
    /// Per source chunk, a count and then a write cursor per destination bucket.
    cursors: Vec<u32>,
}

/// Up to [`SEGMENT_STEPS`] consecutive steps of a run, indexed by the qubits they read.
struct Segment<'s, const W: usize, C: Coefficient> {
    /// Index of `steps[0]` in the run.
    first: usize,
    steps: &'s [PermutationStep<W, C>],
    /// Per qubit, the steps whose support holds it.
    by_qubit: Vec<StepSet>,
    /// The steps that change a string which is the identity on their support.
    always: StepSet,
}

impl<'s, const W: usize, C: Coefficient> Segment<'s, W, C> {
    fn new(first: usize, steps: &'s [PermutationStep<W, C>], num_qubits: usize) -> Self {
        debug_assert!(steps.len() <= SEGMENT_STEPS);
        let mut by_qubit = vec![[0u64; SEGMENT_STEPS / 64]; num_qubits];
        let mut always = [0u64; SEGMENT_STEPS / 64];
        for (i, step) in steps.iter().enumerate() {
            for &q in &step.qubit {
                by_qubit[q][i / 64] |= 1 << (i % 64);
            }
            if step.inert & 1 == 0 {
                always[i / 64] |= 1 << (i % 64);
            }
        }
        Self { first, steps, by_qubit, always }
    }

    /// Add the steps after `i` that read qubit `q`.
    #[inline(always)]
    fn add_after(&self, relevant: &mut StepSet, q: usize, i: usize) {
        let (word, bit) = (i / 64, i % 64);
        let mask = &self.by_qubit[q];
        relevant[word] |= mask[word] & (!0u64 << bit << 1);
        for w in word + 1..relevant.len() {
            relevant[w] |= mask[w];
        }
    }
}

/// Walk one bucket's rows through the run in place, keeping the survivors in order, and record where each moves.
fn walk_bucket<const W: usize, C: Coefficient, T>(
    source: u32,
    bucket: &mut BucketColumns<W, C>,
    segments: &[Segment<'_, W, C>],
    policy: &T,
    destinations: &mut [u32],
    counts: &mut [u32],
) where
    T: TruncationPolicy<W, C> + ?Sized,
{
    let survives = |x: &[u64; W], z: &[u64; W], c: C| c.is_nonzero() && policy.keep_term(x, z, c);
    let BucketColumns { x, z, coeff } = bucket;
    let len = coeff.len();
    let mut kept = 0usize;
    'rows: for row in 0..len {
        let (mut key_x, mut key_z, mut c, mut destination) = (x[row], z[row], coeff[row], source);
        // Whether the current string passed `survives`, which the per-layer path asks after every step.
        let mut tested = false;
        for segment in segments {
            let mut relevant = segment.always;
            for w in 0..W {
                let mut occupied = key_x[w] | key_z[w];
                while occupied != 0 {
                    let q = 64 * w + occupied.trailing_zeros() as usize;
                    for (slot, &mask) in relevant.iter_mut().zip(&segment.by_qubit[q]) {
                        *slot |= mask;
                    }
                    occupied &= occupied - 1;
                }
            }
            let mut word = 0usize;
            while word < relevant.len() {
                let bits = relevant[word];
                if bits == 0 {
                    word += 1;
                    continue;
                }
                relevant[word] = bits & (bits - 1);
                let i = 64 * word + bits.trailing_zeros() as usize;
                let step = &segment.steps[i];
                let s = step.pattern(&key_x, &key_z);
                if (step.inert >> s) & 1 == 1 {
                    continue;
                }
                // A step skipped or inert before this one was asked about the unchanged input.
                if !tested && segment.first + i > 0 && !survives(&key_x, &key_z, c) {
                    continue 'rows;
                }
                step.move_key(&mut key_x, &mut key_z, s);
                destination ^= step.bucket_delta[s];
                c *= step.amplitude[s];
                if !survives(&key_x, &key_z, c) {
                    continue 'rows;
                }
                tested = true;
                let grows = step.grows[s];
                if grows != 0 {
                    for j in 0..MAX_LOCAL_SUPPORT {
                        if (grows >> j) & 1 == 1 {
                            segment.add_after(&mut relevant, step.qubit[j], i);
                        }
                    }
                }
            }
        }
        if !tested && !segments.is_empty() && !survives(&key_x, &key_z, c) {
            continue;
        }
        x[kept] = key_x;
        z[kept] = key_z;
        coeff[kept] = c;
        destinations[kept] = destination;
        counts[destination as usize] += 1;
        kept += 1;
    }
    x.truncate(kept);
    z.truncate(kept);
    coeff.truncate(kept);
}

/// Raw write positions into the new buckets' columns, shared by the scatter's tasks.
struct ScatterTargets<const W: usize, C> {
    x: Vec<*mut [u64; W]>,
    z: Vec<*mut [u64; W]>,
    coeff: Vec<*mut C>,
}

// SAFETY: each task writes only the slots its own cursors span, disjoint from every other task's, and none reads.
unsafe impl<const W: usize, C: Send> Sync for ScatterTargets<W, C> {}

/// Apply `steps` in order to every term of `sum` as one pass, then move each term to its bucket and sort each bucket once.
pub(crate) fn apply_permutation_run<const W: usize, C: Coefficient, T>(
    sum: &mut PauliSum<W, C>,
    steps: &[PermutationStep<W, C>],
    policy: &T,
    scratch: &mut PermutationScratch,
) where
    T: TruncationPolicy<W, C> + ?Sized,
{
    let num_buckets = sum.num_buckets();
    let num_qubits = sum.num_qubits();
    let segments: Vec<Segment<'_, W, C>> = steps
        .chunks(SEGMENT_STEPS)
        .enumerate()
        .map(|(j, chunk)| Segment::new(j * SEGMENT_STEPS, chunk, num_qubits))
        .collect();
    let segments = segments.as_slice();
    let parallel = sum.len() >= PARALLEL_MIN_TERMS;
    let wanted_chunks = if parallel { rayon::current_num_threads() * CHUNKS_PER_THREAD } else { 1 };
    let buckets_per_chunk = num_buckets.div_ceil(wanted_chunks.clamp(1, num_buckets));
    let chunks = num_buckets.div_ceil(buckets_per_chunk);

    let PermutationScratch { destinations, cursors } = scratch;
    destinations.clear();
    destinations.resize(sum.len(), 0);
    cursors.clear();
    cursors.resize(chunks * num_buckets, 0);

    // Each source bucket's destinations start at its row offset before the walk, and its survivors fill them from the front.
    let mut origins = Vec::with_capacity(num_buckets + 1);
    origins.push(0usize);
    for bucket in sum.buckets() {
        origins.push(origins.last().copied().unwrap_or(0) + bucket.len());
    }

    {
        let mut tasks = Vec::with_capacity(chunks);
        let mut rest_destinations = destinations.as_mut_slice();
        let mut rest_counts = cursors.as_mut_slice();
        for (chunk, buckets) in sum.buckets_mut().chunks_mut(buckets_per_chunk).enumerate() {
            let first = chunk * buckets_per_chunk;
            let rows = origins[first + buckets.len()] - origins[first];
            let (mine, later) = std::mem::take(&mut rest_destinations).split_at_mut(rows);
            rest_destinations = later;
            let (counts, later) = std::mem::take(&mut rest_counts).split_at_mut(num_buckets);
            rest_counts = later;
            tasks.push((first, buckets, mine, counts));
        }
        let walk = |(first, buckets, destinations, counts): (
            usize,
            &mut [BucketColumns<W, C>],
            &mut [u32],
            &mut [u32],
        )| {
            let base = origins[first];
            for (j, bucket) in buckets.iter_mut().enumerate() {
                let (start, end) = (origins[first + j] - base, origins[first + j + 1] - base);
                let source = (first + j) as u32;
                walk_bucket(
                    source,
                    bucket,
                    segments,
                    policy,
                    &mut destinations[start..end],
                    counts,
                );
            }
        };
        if parallel {
            tasks.into_par_iter().for_each(walk);
        } else {
            tasks.into_iter().for_each(walk);
        }
    }

    // Counts to cursors: chunk `c` writes destination `b` after the rows earlier chunks send there.
    let mut totals = vec![0u32; num_buckets];
    for row in cursors.chunks_mut(num_buckets) {
        for (total, cursor) in totals.iter_mut().zip(row.iter_mut()) {
            let count = *cursor;
            *cursor = *total;
            *total += count;
        }
    }

    let allocate = |&total: &u32| {
        BucketColumns::with_capacity(total as usize + total as usize / RUN_BUCKET_SLACK_DIVISOR)
    };
    let mut moved: Vec<BucketColumns<W, C>> = if parallel {
        totals.par_iter().map(allocate).collect()
    } else {
        totals.iter().map(allocate).collect()
    };
    let targets = ScatterTargets {
        x: moved.iter_mut().map(|bucket| bucket.x.as_mut_ptr()).collect(),
        z: moved.iter_mut().map(|bucket| bucket.z.as_mut_ptr()).collect(),
        coeff: moved.iter_mut().map(|bucket| bucket.coeff.as_mut_ptr()).collect(),
    };

    {
        let buckets = sum.buckets();
        let destinations = destinations.as_slice();
        let targets = &targets;
        let scatter = |(chunk, cursors): (usize, &mut [u32])| {
            let first = chunk * buckets_per_chunk;
            let last = (first + buckets_per_chunk).min(num_buckets);
            for source in first..last {
                let bucket = &buckets[source];
                let moves = &destinations[origins[source]..origins[source] + bucket.len()];
                for (row, &destination) in moves.iter().enumerate() {
                    let destination = destination as usize;
                    let slot = cursors[destination] as usize;
                    cursors[destination] += 1;
                    // SAFETY: `slot` lies in this chunk's span of `destination`'s cursors, below its `totals` capacity and written once.
                    unsafe {
                        targets.x[destination].add(slot).write(bucket.x[row]);
                        targets.z[destination].add(slot).write(bucket.z[row]);
                        targets.coeff[destination].add(slot).write(bucket.coeff[row]);
                    }
                }
            }
        };
        if parallel {
            cursors.par_chunks_mut(num_buckets).enumerate().for_each(scatter);
        } else {
            cursors.chunks_mut(num_buckets).enumerate().for_each(scatter);
        }
    }
    drop(targets);

    let finish = |sort: &mut SortScratch<W, C>,
                  (bucket, &total): (&mut BucketColumns<W, C>, &u32)| {
        // SAFETY: the scatter wrote every slot below `total`, the sum of every chunk's count for this bucket.
        unsafe {
            bucket.x.set_len(total as usize);
            bucket.z.set_len(total as usize);
            bucket.coeff.set_len(total as usize);
        }
        SortKernel::Comparison.sort(bucket, sort);
    };
    if parallel {
        moved.par_iter_mut().zip(totals.par_iter()).for_each_init(SortScratch::default, finish);
    } else {
        let mut sort = SortScratch::default();
        moved.iter_mut().zip(totals.iter()).for_each(|item| finish(&mut sort, item));
    }
    sum.replace_buckets(moved);
}

#[cfg(test)]
mod tests;
