//! Storage and partition maintenance for [`PauliSum`] — per-bucket structure-of-arrays columns under a [`Gf2Hash`] partition. See ARCHITECTURE.md §Data-Model.
//!
//! Re-exported as [`crate::pauli_sum::PauliSum`]; this module owns the column storage, the bucket-count policy ([`desired_bits`] and the sizing constants), and the merge helpers.

use num_complex::Complex64;
use rayon::prelude::*;

use super::hash::Gf2Hash;
use crate::pauli_string::PauliString;

/// Default seed for the partitioning hash. Fixed so a `propagate` run is reproducible across processes.
/// Exposed as a constant rather than hidden so a caller who needs a different partition can build their own [`Gf2Hash`].
pub const DEFAULT_HASH_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Bucket bits for a sum of `len` terms: the smallest `b` with `len <= target << b`, clamped below by the parallelism floor.
/// Used to size the partition once at ingestion and at the start of `propagate`. [`PauliSum::rebucket`] tracks it afterwards, but only upward.
pub fn desired_bits(len: usize, target: usize, min_buckets: usize) -> u8 {
    debug_assert!(target > 0);
    let worth_splitting = len >= min_buckets.saturating_mul(MIN_TERMS_PER_TASK);
    let mut floor = 0u8;
    if worth_splitting {
        while (1usize << floor) < min_buckets && floor < super::hash::B_MAX_BITS {
            floor += 1;
        }
    }
    let mut b = floor;
    while b < super::hash::B_MAX_BITS && len > target.saturating_mul(1usize << b) {
        b += 1;
    }
    b
}

/// Minimum terms per bucket for the parallelism floor to apply.
/// The floor in [`PauliSum::rebucket`] exists to give Rayon enough independent tasks, but a task carrying almost nothing is pure overhead; below `min_buckets × MIN_TERMS_PER_TASK` total terms we would rather have few buckets and let the small-`n` fallback handle it. See ARCHITECTURE.md §Bucket-Policy for the sweep that set this value.
///
/// This gate is not only about parallelism: the bucket count caps the engine's coset dimension, and the per-run sort's comparison count reaches its floor only at full delta rank — so it also controls how much sort work a dense-PTM layer pays. Do not lower or drop the gate without checking a sparse-PTM layer too; the two regimes pull in opposite directions (see `research/FINDINGS.md`).
pub const MIN_TERMS_PER_TASK: usize = 64;

/// Default target terms per bucket.
/// Chosen so a bucket plus its gather scratch stays L2-resident on the reference host. See ARCHITECTURE.md §Bucket-Policy for the sweep that set this value, and [`MIN_TERMS_PER_TASK`] for the dense-PTM caveat.
pub const DEFAULT_TARGET_BUCKET_LEN: usize = 1024;

/// Default floor on the bucket count.
/// Fixed, not thread-derived, so the bucket count `B` stays a deterministic function of the sum's history alone (ARCHITECTURE.md §Determinism) rather than of how many threads happen to be available.
/// Must be `>= 16`: [`desired_bits`]'s "worth splitting" floor is non-monotone below that, and we want "a sum of `<= 1024` terms gets a single bucket" to hold.
pub const DEFAULT_MIN_BUCKETS: usize = 128;

/// One bucket's structure-of-arrays columns.
/// Capacity is retained across layers, which is the point of owning per-bucket columns rather than slicing one flat array: the steady state of a propagation loop allocates nothing (ARCHITECTURE.md §Data-Model).
#[derive(Clone, Debug, Default)]
pub(crate) struct BucketCols<const W: usize> {
    pub(crate) x: Vec<[u64; W]>,
    pub(crate) z: Vec<[u64; W]>,
    pub(crate) coeff: Vec<Complex64>,
}

impl<const W: usize> BucketCols<W> {
    pub(super) fn new() -> Self {
        Self {
            x: Vec::new(),
            z: Vec::new(),
            coeff: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.coeff.len()
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.x.clear();
        self.z.clear();
        self.coeff.clear();
    }

    #[inline]
    pub(crate) fn push(&mut self, x: [u64; W], z: [u64; W], c: Complex64) {
        self.x.push(x);
        self.z.push(z);
        self.coeff.push(c);
    }
}

/// Split one input bucket into its "low" (kept in place) and "high" (new bucket at `b + old_nb`) halves under a hash that has just gained `new_bit` as its top bit.
/// Shared by [`PauliSum::refine`]'s serial and parallel branches. `b` is the old bucket index, used only by the debug-only low-bits invariant check.
fn refine_bucket<const W: usize>(
    cols: &mut BucketCols<W>,
    up: &mut BucketCols<W>,
    hash: &Gf2Hash<W>,
    new_bit: u8,
    b: u32,
) {
    let _ = b; // referenced only inside the `cfg(debug_assertions)` block below
    let n = cols.len();
    let mut keep = 0usize;
    for i in 0..n {
        let bit = hash.row_parity(&cols.x[i], &cols.z[i], new_bit);
        #[cfg(debug_assertions)]
        {
            let full = hash.bucket_of(&cols.x[i], &cols.z[i]);
            debug_assert_eq!(
                full & ((1u32 << new_bit) - 1),
                b,
                "refine: low bits must be preserved",
            );
        }
        if bit == 1 {
            up.push(cols.x[i], cols.z[i], cols.coeff[i]);
        } else {
            // Compact in place: `keep <= i` always, so this never overwrites an unread slot.
            cols.x[keep] = cols.x[i];
            cols.z[keep] = cols.z[i];
            cols.coeff[keep] = cols.coeff[i];
            keep += 1;
        }
    }
    cols.x.truncate(keep);
    cols.z.truncate(keep);
    cols.coeff.truncate(keep);
}

/// Merge two sorted runs. No coefficient combining: keys are globally unique.
pub(super) fn merge_two<const W: usize>(a: &BucketCols<W>, b: &BucketCols<W>) -> BucketCols<W> {
    let mut out = BucketCols::<W>::new();
    let total = a.len() + b.len();
    out.x.reserve_exact(total);
    out.z.reserve_exact(total);
    out.coeff.reserve_exact(total);
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        if (&a.x[i], &a.z[i]) <= (&b.x[j], &b.z[j]) {
            out.push(a.x[i], a.z[i], a.coeff[i]);
            i += 1;
        } else {
            out.push(b.x[j], b.z[j], b.coeff[j]);
            j += 1;
        }
    }
    while i < a.len() {
        out.push(a.x[i], a.z[i], a.coeff[i]);
        i += 1;
    }
    while j < b.len() {
        out.push(b.x[j], b.z[j], b.coeff[j]);
        j += 1;
    }
    out
}

/// Merge two sorted runs, summing equal keys and dropping exact-zero sums.
/// The counterpart of [`merge_two`] for operands that may share keys: within a partition, equal keys are always in the same bucket pair, so a two-pointer pass over one bucket of each operand sees every collision there is.
fn merge_two_adding<const W: usize>(a: &BucketCols<W>, b: &BucketCols<W>) -> BucketCols<W> {
    let mut out = BucketCols::<W>::new();
    let total = a.len() + b.len();
    out.x.reserve_exact(total);
    out.z.reserve_exact(total);
    out.coeff.reserve_exact(total);
    let zero = Complex64::new(0.0, 0.0);
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match (&a.x[i], &a.z[i]).cmp(&(&b.x[j], &b.z[j])) {
            std::cmp::Ordering::Less => {
                out.push(a.x[i], a.z[i], a.coeff[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b.x[j], b.z[j], b.coeff[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                let c = a.coeff[i] + b.coeff[j];
                if c != zero {
                    out.push(a.x[i], a.z[i], c);
                }
                i += 1;
                j += 1;
            }
        }
    }
    while i < a.len() {
        out.push(a.x[i], a.z[i], a.coeff[i]);
        i += 1;
    }
    while j < b.len() {
        out.push(b.x[j], b.z[j], b.coeff[j]);
        j += 1;
    }
    out
}

/// Merge `B` sorted runs into one, by `log2(B)` rounds of pairwise merges.
/// Faster than a `BinaryHeap`-based `B`-way merge, whose pops need `log2(B)` comparisons against keys scattered across `B` runs and which is inherently sequential. The tree does the same `O(n log B)` comparisons but reads two sequential streams at a time, and every pair within a round is independent, so the rounds parallelize — sequential bandwidth traded for `log B` passes over the payload instead of one.
fn merge_runs<const W: usize>(mut runs: Vec<BucketCols<W>>) -> BucketCols<W> {
    if runs.is_empty() {
        return BucketCols::new();
    }
    while runs.len() > 1 {
        runs = runs
            .par_chunks(2)
            .map(|pair| match pair {
                [a, b] => merge_two(a, b),
                [a] => a.clone(),
                _ => unreachable!("par_chunks(2) yields 1 or 2 elements"),
            })
            .collect();
    }
    runs.pop().expect("non-empty by the check above")
}

/// Weighted sum of Pauli operators, stored as structure-of-arrays columns partitioned by a GF(2)-linear hash.
///
/// # Canonical order
///
/// Terms are ordered by bucket index ascending, then lexicographic `(x, z)` within a bucket — the order [`Self::iter`] and [`Self::to_arrays`] produce. This is a public promise, not an implementation detail.
///
/// # Invariant
///
/// Every term lies in `buckets[hash.bucket_of(term)]`, and each bucket is sorted by the lexicographic `(x, z)` key with no duplicate keys. Because `h` is a function, equal keys always share a bucket, so per-bucket dedup implies global dedup and no global sort is ever needed.
///
/// # Partition scatter/gather
///
/// The crate-internal `filter_partition` / `merge_partitions` pair are the scatter and gather primitives of the partitioned engine (ARCHITECTURE.md §Partitioning). Neither direction sorts or combines coefficients, so a round trip is bitwise.
///
/// [`propagate`]: crate::propagate
#[derive(Clone, Debug)]
pub struct PauliSum<const W: usize> {
    pub(super) buckets: Vec<BucketCols<W>>,
    pub(super) hash: Gf2Hash<W>,
    pub(super) num_qubits: usize,
    pub(super) len: usize,
}

impl<const W: usize> PauliSum<W> {
    /// Partition a globally key-sorted stream of terms. `O(n)`: one hash evaluation and one scatter per term, and each bucket comes out sorted for free since order within it is inherited from the input.
    /// The caller owes the sortedness: `x`, `z`, `coeff` must be parallel columns ascending in `(x, z)` with no duplicate keys, or the per-bucket sort invariant silently breaks.
    pub(crate) fn from_key_sorted(
        x: &[[u64; W]],
        z: &[[u64; W]],
        coeff: &[Complex64],
        hash: Gf2Hash<W>,
        num_qubits: usize,
    ) -> Self {
        let n = coeff.len();
        let nb = hash.num_buckets();

        // Hashing is the expensive part, so it runs in parallel; the counts come from the resulting indices rather than a second hashing pass.
        // The scatter below stays sequential: buckets are separate allocations, so a parallel scatter would need every thread to write into every bucket.
        let idx: Vec<u32> = (0..n)
            .into_par_iter()
            .map(|i| hash.bucket_of(&x[i], &z[i]))
            .collect();
        let mut counts: Vec<usize> = vec![0; nb];
        for &b in idx.iter() {
            counts[b as usize] += 1;
        }

        let mut buckets: Vec<BucketCols<W>> = Vec::with_capacity(nb);
        for &c in counts.iter() {
            let mut cols = BucketCols::<W>::new();
            cols.x.reserve_exact(c);
            cols.z.reserve_exact(c);
            cols.coeff.reserve_exact(c);
            buckets.push(cols);
        }

        for i in 0..n {
            buckets[idx[i] as usize].push(x[i], z[i], coeff[i]);
        }

        Self {
            buckets,
            hash,
            num_qubits,
            len: n,
        }
    }

    /// Empty sum on `num_qubits` qubits, in a single bucket.
    /// The hash is the zero-bit prefix of the default seed's matrix, so the canonical order is plain lexicographic `(x, z)` until the sum grows past the [`desired_bits`] threshold and something refines it.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `num_qubits > 64 · W`.
    pub fn empty(num_qubits: usize) -> Self {
        debug_assert!(num_qubits <= 64 * W);
        Self::empty_with_hash(num_qubits, Gf2Hash::new(num_qubits, 0, DEFAULT_HASH_SEED))
    }

    /// An empty sum over `num_qubits`, partitioned by `hash`.
    pub fn empty_with_hash(num_qubits: usize, hash: Gf2Hash<W>) -> Self {
        let nb = hash.num_buckets();
        Self {
            buckets: (0..nb).map(|_| BucketCols::new()).collect(),
            hash,
            num_qubits,
            len: 0,
        }
    }

    /// Test/oracle constructor: wrap globally key-sorted columns as a single-bucket sum (zero hash bits, default seed), whose canonical order is therefore exactly the given column order.
    #[cfg(test)]
    pub(crate) fn from_sorted_columns(
        x: Vec<[u64; W]>,
        z: Vec<[u64; W]>,
        coeff: Vec<Complex64>,
        num_qubits: usize,
    ) -> Self {
        let n = coeff.len();
        let hash = Gf2Hash::new(num_qubits, 0, DEFAULT_HASH_SEED);
        Self {
            buckets: vec![BucketCols { x, z, coeff }],
            hash,
            num_qubits,
            len: n,
        }
    }

    /// Repartition under `hash`, keeping every term. Flattens to a globally key-sorted stream and rescatters.
    /// Prefer [`Self::refine`] / [`Self::coarsen`] when only the bucket count changes and the hash rows are the same; those are `O(n)` and never merge.
    ///
    /// # Panics
    ///
    /// Panics if `hash` was built for a different qubit count.
    pub fn with_hash(self, hash: Gf2Hash<W>) -> Self {
        assert_eq!(
            self.num_qubits,
            hash.num_qubits(),
            "PauliSum::with_hash: num_qubits mismatch",
        );
        let num_qubits = self.num_qubits;
        let merged = merge_runs(self.buckets);
        Self::from_key_sorted(&merged.x, &merged.z, &merged.coeff, hash, num_qubits)
    }

    /// A copy of `self` partitioned exactly as `target` partitions.
    /// Three cases, cheapest first: identical partition is a clone; same hash rows at a different bucket count is a clone plus `O(n)` refine/coarsen; different rows falls back to [`Self::with_hash`]'s `O(n log B)` flatten.
    pub(crate) fn align_to(&self, target: &Gf2Hash<W>) -> Self {
        if !self.hash.same_rows_as(target) {
            return self.clone().with_hash(target.clone());
        }
        let mut out = self.clone();
        while out.hash.bits() < target.bits() {
            out.refine();
        }
        while out.hash.bits() > target.bits() {
            out.coarsen();
        }
        out
    }

    /// Drop every term, keeping the hash and the bucket storage.
    pub fn clear(&mut self) {
        for cols in self.buckets.iter_mut() {
            cols.clear();
        }
        self.len = 0;
    }

    /// Total number of terms.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the sum has no terms.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of qubits this sum acts on.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of buckets, `1 << hash().bits()`.
    #[inline]
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// The partitioning hash.
    #[inline]
    pub fn hash(&self) -> &Gf2Hash<W> {
        &self.hash
    }

    /// Borrow bucket `b`'s columns as `(x, z, coeff)`.
    #[inline]
    pub fn bucket(&self, b: usize) -> (&[[u64; W]], &[[u64; W]], &[Complex64]) {
        let cols = &self.buckets[b];
        (&cols.x, &cols.z, &cols.coeff)
    }

    /// Number of terms in bucket `b`.
    #[inline]
    pub fn bucket_len(&self, b: usize) -> usize {
        self.buckets[b].len()
    }

    /// Double the bucket count, splitting each bucket in two.
    /// One `Gf2Hash::row_parity` evaluation per term against just the new high bit — `O(n)` total — since only whether the new bit is set decides which half a term lands in; both halves inherit the source bucket's order, so nothing is re-sorted.
    /// Bucket pairs are independent, so above [`MIN_TERMS_PER_TASK`] × [`DEFAULT_MIN_BUCKETS`] total terms the per-bucket work runs across Rayon; below it the sequential loop avoids per-task overhead.
    pub fn refine(&mut self) {
        let old_nb = self.buckets.len();
        self.hash.refine();
        let new_bit = self.hash.bits() - 1;
        let hash = &self.hash;

        // Take the old buckets out so the upper halves can reuse their storage.
        let mut old = std::mem::take(&mut self.buckets);
        let mut upper: Vec<BucketCols<W>> = (0..old_nb).map(|_| BucketCols::new()).collect();

        if self.len < DEFAULT_MIN_BUCKETS * MIN_TERMS_PER_TASK {
            for (b, (cols, up)) in old.iter_mut().zip(upper.iter_mut()).enumerate() {
                refine_bucket(cols, up, hash, new_bit, b as u32);
            }
        } else {
            old.par_iter_mut()
                .zip(upper.par_iter_mut())
                .enumerate()
                .for_each(|(b, (cols, up))| {
                    refine_bucket(cols, up, hash, new_bit, b as u32);
                });
        }

        old.extend(upper);
        self.buckets = old;
    }

    /// Halve the bucket count, merging bucket pairs `(i, i + B/2)`.
    /// A 2-way merge per pair via `merge_two`, no coefficient combining, since equal keys were already in the same source bucket. Same threshold and rationale as [`Self::refine`].
    pub fn coarsen(&mut self) {
        self.hash.coarsen();
        let new_nb = self.buckets.len() / 2;

        let old = std::mem::take(&mut self.buckets);
        let (lower, upper) = old.split_at(new_nb);

        let merged: Vec<BucketCols<W>> = if self.len < DEFAULT_MIN_BUCKETS * MIN_TERMS_PER_TASK {
            lower
                .iter()
                .zip(upper.iter())
                .map(|(lo, hi)| merge_two(lo, hi))
                .collect()
        } else {
            lower
                .par_iter()
                .zip(upper.par_iter())
                .map(|(lo, hi)| merge_two(lo, hi))
                .collect()
        };

        self.buckets = merged;
    }

    /// Bring the bucket count up to what [`desired_bits`] would choose for the current length — but never down.
    ///
    /// # Grow-only policy
    ///
    /// A sum's bucket count is monotone non-decreasing over its lifetime: this clamps the target to `self.hash.bits()`, so `rebucket` only ever refines, never coarsens. Term counts oscillate every layer (fanout grows them, truncation cuts them back), so tracking `desired_bits` exactly in both directions would refine and coarsen on alternate layers near a power-of-two boundary, each an `O(n · bits)` serial pass — see `research/FINDINGS.md`.
    /// Keeping the larger partition is otherwise free: the cost of not coarsening back down is three empty `Vec` headers per surplus bucket, not a term-proportional cost. A caller that wants to shrink a sum can still do so explicitly via [`Self::with_hash`] or [`Self::coarsen`].
    /// Also keeps at least `min_buckets` buckets once there is enough work to spread, so the bucket-parallel decomposition has slack to load-balance.
    pub fn rebucket(&mut self, target: usize, min_buckets: usize) {
        debug_assert!(target > 0);
        let want = desired_bits(self.len, target, min_buckets).max(self.hash.bits());
        while self.hash.bits() < want {
            self.refine();
        }
    }

    /// The buckets' columns, for read-outs that scan them.
    pub(crate) fn buckets(&self) -> &[BucketCols<W>] {
        &self.buckets
    }

    /// Mutable access to the buckets, for layers that are applied in place.
    pub(crate) fn buckets_mut(&mut self) -> &mut [BucketCols<W>] {
        &mut self.buckets
    }

    /// Recompute the cached total after an in-place layer.
    pub(crate) fn recount(&mut self) {
        self.len = self.buckets.iter().map(|c| c.len()).sum();
    }

    /// Iterate every term in canonical order: buckets by ascending index, and within a bucket by ascending `(x, z)` key.
    /// This is not globally sorted — a bucket is a hash class, and the classes interleave arbitrarily in key order — but it is a total, deterministic order fixed by the partition, and the one [`Self::to_arrays`] concatenates in.
    pub fn iter(&self) -> impl Iterator<Item = (&[u64; W], &[u64; W], Complex64)> + '_ {
        self.buckets.iter().flat_map(|cols| {
            cols.x
                .iter()
                .zip(cols.z.iter())
                .zip(cols.coeff.iter())
                .map(|((x, z), c)| (x, z, *c))
        })
    }

    /// Copy every term out as three parallel columns, in the canonical order of [`Self::iter`].
    /// The columns are not globally key-sorted unless there is a single bucket; sort the triples yourself for a canonical, partition-independent view.
    pub fn to_arrays(&self) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>) {
        let mut x = Vec::with_capacity(self.len);
        let mut z = Vec::with_capacity(self.len);
        let mut coeff = Vec::with_capacity(self.len);
        for cols in self.buckets.iter() {
            x.extend_from_slice(&cols.x);
            z.extend_from_slice(&cols.z);
            coeff.extend_from_slice(&cols.coeff);
        }
        (x, z, coeff)
    }

    /// Coefficient of the term with key `(x, z)`, or `None` if absent.
    /// `O(b·W + log m)`: one hash evaluation to find the bucket, then a binary search of that bucket's `m` terms — one bucket is the whole search space, which is the lookup that per-bucket dedup buys.
    pub fn get(&self, x: &[u64; W], z: &[u64; W]) -> Option<Complex64> {
        let cols = &self.buckets[self.hash.bucket_of(x, z) as usize];
        let mut lo = 0usize;
        let mut hi = cols.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match (&cols.x[mid], &cols.z[mid]).cmp(&(x, z)) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(cols.coeff[mid]),
            }
        }
        None
    }

    /// Coefficient of the identity term, i.e. `tr(O) / 2^n`.
    /// The Pauli basis is orthogonal under the trace, so every non-identity term is traceless and only this one contributes.
    pub fn identity_coefficient(&self) -> Complex64 {
        let zero_key = [0u64; W];
        self.get(&zero_key, &zero_key)
            .unwrap_or(Complex64::new(0.0, 0.0))
    }

    /// Multiply every coefficient by `c` in place.
    /// Elementwise, so the partition is irrelevant to the result. Parallel across buckets, which are separate allocations.
    pub fn scale(&mut self, c: Complex64) {
        self.buckets.par_iter_mut().for_each(|cols| {
            for coeff in cols.coeff.iter_mut() {
                *coeff *= c;
            }
        });
    }

    /// Keep only the terms for which `f(x, z, coeff)` is `true`.
    /// Order-preserving and in place, so the per-bucket sort and no-duplicates invariant both survive; a term never changes bucket, so the hash invariant does too. `f` may run on several threads at once, hence the `Sync` bound.
    pub fn retain(&mut self, f: impl Fn(&[u64; W], &[u64; W], Complex64) -> bool + Sync) {
        self.buckets.par_iter_mut().for_each(|cols| {
            let n = cols.len();
            let mut w = 0usize;
            for r in 0..n {
                if f(&cols.x[r], &cols.z[r], cols.coeff[r]) {
                    if w != r {
                        cols.x[w] = cols.x[r];
                        cols.z[w] = cols.z[r];
                        cols.coeff[w] = cols.coeff[r];
                    }
                    w += 1;
                }
            }
            cols.x.truncate(w);
            cols.z.truncate(w);
            cols.coeff.truncate(w);
        });
        self.recount();
    }

    /// Hilbert-Schmidt overlap `tr(self† · other) / 2ⁿ`, i.e. `Σ conj(aᵢ)·bᵢ` over the keys the two sums share.
    /// Equal keys always land in the same bucket under a shared hash, so this is `B` independent two-pointer merges, one per bucket, and no term is ever compared across buckets.
    ///
    /// # Summation order
    ///
    /// Each bucket's partial is accumulated in key order, and the partials then combined in ascending bucket index; different partitions of the same operands agree to within rounding, not bit for bit.
    ///
    /// # Panics
    ///
    /// Panics unless the two sums share a partition: same hash rows and the same bucket count. Combining sums under different partitions is [`Self::add`]'s job; overlap does not realign. Under the grow-only [`Self::rebucket`] policy, two sums can have equal `len()` but different bucket counts, so align them with [`Self::with_hash`] first if that is a possibility.
    pub fn overlap(&self, other: &Self) -> Complex64 {
        assert!(
            self.hash.same_rows_as(&other.hash),
            "PauliSum::overlap: hash mismatch (seed or num_qubits differs)",
        );
        assert_eq!(
            self.hash.bits(),
            other.hash.bits(),
            "PauliSum::overlap: bucket count mismatch",
        );
        self.buckets
            .par_iter()
            .zip(other.buckets.par_iter())
            .map(|(a, b)| {
                let mut acc = Complex64::new(0.0, 0.0);
                let (mut i, mut j) = (0usize, 0usize);
                while i < a.len() && j < b.len() {
                    match (&a.x[i], &a.z[i]).cmp(&(&b.x[j], &b.z[j])) {
                        std::cmp::Ordering::Less => i += 1,
                        std::cmp::Ordering::Greater => j += 1,
                        std::cmp::Ordering::Equal => {
                            acc += a.coeff[i].conj() * b.coeff[j];
                            i += 1;
                            j += 1;
                        }
                    }
                }
                acc
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    }

    /// Sum of two bucketed sums.
    /// The left partition wins: the result is partitioned exactly as `self` is, and `other` is realigned onto that partition first. `self` and `other` are both left untouched.
    /// Once aligned, equal keys sit in the same bucket index on both sides, so this is `B` independent two-pointer merges — no global sort, no cross-bucket comparison. Terms whose coefficients sum to exactly `0+0i` are dropped.
    ///
    /// # Summation order
    ///
    /// Each surviving coefficient is a single `self + other` addition, bit-identical to what the flat merge would produce. Only derived quantities that accumulate across terms (e.g. [`Self::overlap`]) see the bucket-order effect.
    ///
    /// # Panics
    ///
    /// Panics if the two sums disagree about `num_qubits`.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(
            self.num_qubits, other.num_qubits,
            "PauliSum::add: num_qubits mismatch ({} vs {})",
            self.num_qubits, other.num_qubits,
        );
        let rhs = other.align_to(&self.hash);
        let buckets: Vec<BucketCols<W>> = self
            .buckets
            .par_iter()
            .zip(rhs.buckets.par_iter())
            .map(|(a, b)| merge_two_adding(a, b))
            .collect();
        let len = buckets.iter().map(|c| c.len()).sum();
        Self {
            buckets,
            hash: self.hash.clone(),
            num_qubits: self.num_qubits,
            len,
        }
    }

    /// Assert the structural invariant (debug and test builds only): every term is in its hash bucket, each bucket is strictly ascending in `(x, z)`, every key is within `num_qubits`, and the cached length agrees.
    #[cfg(any(test, debug_assertions))]
    pub fn assert_invariants(&self) {
        assert_eq!(
            self.buckets.len(),
            self.hash.num_buckets(),
            "PauliSum: bucket count disagrees with hash",
        );
        let mut total = 0usize;
        for (b, cols) in self.buckets.iter().enumerate() {
            assert_eq!(cols.x.len(), cols.z.len());
            assert_eq!(cols.x.len(), cols.coeff.len());
            total += cols.len();
            for i in 0..cols.len() {
                let got = self.hash.bucket_of(&cols.x[i], &cols.z[i]);
                assert_eq!(
                    got as usize, b,
                    "PauliSum: term {i} of bucket {b} hashes to {got}",
                );
                let term = PauliString::<W> {
                    x: cols.x[i],
                    z: cols.z[i],
                };
                assert!(
                    term.is_within(self.num_qubits),
                    "PauliSum: term {i} of bucket {b} exceeds num_qubits",
                );
            }
            for i in 1..cols.len() {
                let prev = (&cols.x[i - 1], &cols.z[i - 1]);
                let cur = (&cols.x[i], &cols.z[i]);
                assert!(prev < cur, "PauliSum: bucket {b} out of order at {i}");
            }
        }
        assert_eq!(total, self.len, "PauliSum: cached len disagrees");
    }
}

#[cfg(test)]
mod tests;
