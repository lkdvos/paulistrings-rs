//! [`PauliSum`]'s per-bucket column storage, the bucket-count policy and the merge helpers (ARCHITECTURE.md §Data-Model, §Bucket-Policy).

use std::borrow::Cow;

use num_complex::Complex64;
use rayon::prelude::*;

use super::hash::Gf2Hash;

/// Default seed for the partitioning hash.
pub const DEFAULT_HASH_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Bucket bits for `len` terms: the smallest `b` with `len <= target << b`, clamped below by the parallelism floor.
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

/// Minimum terms per bucket for the parallelism floor to apply (ARCHITECTURE.md §Bucket-Policy).
// Not lower without checking a dense and a sparse PTM layer: the bucket count also caps the coset dimension a dense layer's sort needs (research/FINDINGS.md §The dense-PTM bucket cliff is a delta-span rank effect).
pub(crate) const MIN_TERMS_PER_TASK: usize = 64;

/// Below this many terms the bucket-parallel maintenance passes run serially.
pub(super) const PARALLEL_MIN_TERMS: usize = DEFAULT_MIN_BUCKETS * MIN_TERMS_PER_TASK;

/// Default target terms per bucket (ARCHITECTURE.md §Bucket-Policy).
pub const DEFAULT_TARGET_BUCKET_LEN: usize = 1024;

/// Default floor on the bucket count, fixed rather than thread-derived (ARCHITECTURE.md §Determinism).
/// Must be `>= 16`, or `desired_bits` stops giving a sum of `<= 1024` terms a single bucket.
pub const DEFAULT_MIN_BUCKETS: usize = 128;

/// One bucket's columns, whose capacity is retained across layers (ARCHITECTURE.md §Data-Model).
#[derive(Clone, Debug, Default)]
pub(crate) struct BucketColumns<const W: usize> {
    pub(crate) x: Vec<[u64; W]>,
    pub(crate) z: Vec<[u64; W]>,
    pub(crate) coeff: Vec<Complex64>,
}

impl<const W: usize> BucketColumns<W> {
    pub(super) fn new() -> Self {
        Self {
            x: Vec::new(),
            z: Vec::new(),
            coeff: Vec::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.coeff.len()
    }

    pub(crate) fn clear(&mut self) {
        self.x.clear();
        self.z.clear();
        self.coeff.clear();
    }

    pub(crate) fn push(&mut self, x: [u64; W], z: [u64; W], c: Complex64) {
        self.x.push(x);
        self.z.push(z);
        self.coeff.push(c);
    }
}

/// Split bucket `b` into the half kept in place and the half moving to `upper`, by the hash's new top bit `new_bit`.
fn refine_bucket<const W: usize>(
    columns: &mut BucketColumns<W>,
    upper_half: &mut BucketColumns<W>,
    hash: &Gf2Hash<W>,
    new_bit: u8,
    b: u32,
) {
    let _ = b;
    let n = columns.len();
    let mut keep = 0usize;
    for i in 0..n {
        let bit = hash.row_parity(&columns.x[i], &columns.z[i], new_bit);
        #[cfg(debug_assertions)]
        {
            let full = hash.bucket_of(&columns.x[i], &columns.z[i]);
            debug_assert_eq!(
                full & ((1u32 << new_bit) - 1),
                b,
                "refine: low bits must be preserved",
            );
        }
        if bit == 1 {
            upper_half.push(columns.x[i], columns.z[i], columns.coeff[i]);
        } else {
            columns.x[keep] = columns.x[i];
            columns.z[keep] = columns.z[i];
            columns.coeff[keep] = columns.coeff[i];
            keep += 1;
        }
    }
    columns.x.truncate(keep);
    columns.z.truncate(keep);
    columns.coeff.truncate(keep);
}

/// Merge two sorted runs. No coefficient combining: keys are globally unique.
pub(super) fn merge_two<const W: usize>(
    a: &BucketColumns<W>,
    b: &BucketColumns<W>,
) -> BucketColumns<W> {
    let mut out = BucketColumns::<W>::new();
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
fn merge_two_adding<const W: usize>(
    a: &BucketColumns<W>,
    b: &BucketColumns<W>,
) -> BucketColumns<W> {
    let mut out = BucketColumns::<W>::new();
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

/// Merge `B` sorted runs into one, by `log2(B)` parallel rounds of pairwise merges rather than a sequential heap merge.
fn merge_runs<const W: usize>(mut runs: Vec<BucketColumns<W>>) -> BucketColumns<W> {
    if runs.is_empty() {
        return BucketColumns::new();
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

/// Weighted sum of Pauli strings, stored as structure-of-arrays columns partitioned into buckets by a GF(2)-linear hash.
///
/// Terms come in canonical order: bucket index ascending, then lexicographic `(x, z)` within a bucket, with no repeated key.
/// A sum of at most 1024 terms built by [`crate::BuildAccumulator`] has one bucket and so is plain lex-sorted; a larger one interleaves buckets, so compare sums by key ([`Self::get`]), not by position.
///
/// ```
/// use paulistrings::{BuildAccumulator, PauliString, Phase};
/// use num_complex::Complex64;
///
/// let mut accumulator = BuildAccumulator::<1>::new(2);
/// accumulator.add_term(PauliString::<1>::z(0), Complex64::new(1.0, 0.0));
/// accumulator.add_term(PauliString::<1>::x(1), Complex64::new(0.5, 0.0));
/// let a = accumulator.finalize();
///
/// let mut accumulator = BuildAccumulator::<1>::new(2);
/// accumulator.add_term(PauliString::<1>::x(1), Complex64::new(-0.25, 0.0));
/// let b = accumulator.finalize();
///
/// let merged = a.add(&b);
/// assert_eq!(merged.len(), 2);
/// assert_eq!(merged.get(&[0], &[1]), Some(Complex64::new(1.0, 0.0)));
/// assert_eq!(merged.get(&[0b10], &[0]), Some(Complex64::new(0.25, 0.0)));
/// ```
#[derive(Clone, Debug)]
pub struct PauliSum<const W: usize> {
    pub(super) buckets: Vec<BucketColumns<W>>,
    pub(super) hash: Gf2Hash<W>,
    pub(super) num_qubits: usize,
    pub(super) len: usize,
}

impl<const W: usize> PauliSum<W> {
    /// Partition terms the caller has sorted ascending in `(x, z)` with no repeated key; unsorted input silently breaks the bucket invariant.
    pub(crate) fn from_key_sorted(
        x: &[[u64; W]],
        z: &[[u64; W]],
        coeff: &[Complex64],
        hash: Gf2Hash<W>,
        num_qubits: usize,
    ) -> Self {
        let n = coeff.len();
        let num_buckets = hash.num_buckets();

        let bucket_indices: Vec<u32> = (0..n)
            .into_par_iter()
            .map(|i| hash.bucket_of(&x[i], &z[i]))
            .collect();
        let mut counts: Vec<usize> = vec![0; num_buckets];
        for &b in bucket_indices.iter() {
            counts[b as usize] += 1;
        }

        let mut buckets: Vec<BucketColumns<W>> = Vec::with_capacity(num_buckets);
        for &c in counts.iter() {
            let mut columns = BucketColumns::<W>::new();
            columns.x.reserve_exact(c);
            columns.z.reserve_exact(c);
            columns.coeff.reserve_exact(c);
            buckets.push(columns);
        }

        for i in 0..n {
            buckets[bucket_indices[i] as usize].push(x[i], z[i], coeff[i]);
        }

        Self {
            buckets,
            hash,
            num_qubits,
            len: n,
        }
    }

    /// Empty sum on `num_qubits` qubits, in a single bucket.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `num_qubits > 64 · W`.
    pub fn empty(num_qubits: usize) -> Self {
        debug_assert!(num_qubits <= 64 * W);
        Self::empty_with_hash(num_qubits, Gf2Hash::new(num_qubits, 0, DEFAULT_HASH_SEED))
    }

    /// An empty sum over `num_qubits`, partitioned by `hash`.
    pub(crate) fn empty_with_hash(num_qubits: usize, hash: Gf2Hash<W>) -> Self {
        let num_buckets = hash.num_buckets();
        Self {
            buckets: (0..num_buckets).map(|_| BucketColumns::new()).collect(),
            hash,
            num_qubits,
            len: 0,
        }
    }

    /// Repartition under `hash`, keeping every term; panics if `hash` was built for a different qubit count.
    pub(crate) fn with_hash(self, hash: Gf2Hash<W>) -> Self {
        assert_eq!(
            self.num_qubits,
            hash.num_qubits(),
            "PauliSum::with_hash: num_qubits mismatch",
        );
        let num_qubits = self.num_qubits;
        let merged = merge_runs(self.buckets);
        Self::from_key_sorted(&merged.x, &merged.z, &merged.coeff, hash, num_qubits)
    }

    /// `self` partitioned exactly as `target` partitions, borrowed when it already is.
    fn align_to(&self, target: &Gf2Hash<W>) -> Cow<'_, Self> {
        if !self.hash.same_rows_as(target) {
            return Cow::Owned(self.clone().with_hash(target.clone()));
        }
        if self.hash.bits() == target.bits() {
            return Cow::Borrowed(self);
        }
        let mut out = self.clone();
        while out.hash.bits() < target.bits() {
            out.refine();
        }
        while out.hash.bits() > target.bits() {
            out.coarsen();
        }
        Cow::Owned(out)
    }

    /// Drop every term, keeping the hash and the bucket storage.
    pub fn clear(&mut self) {
        for columns in self.buckets.iter_mut() {
            columns.clear();
        }
        self.len = 0;
    }

    /// Total number of terms.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the sum has no terms.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of qubits this sum acts on.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of buckets, `1 << hash().bits()`.
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// The partitioning hash.
    pub fn hash(&self) -> &Gf2Hash<W> {
        &self.hash
    }

    /// Borrow bucket `b`'s columns as `(x, z, coeff)`.
    pub(crate) fn bucket(&self, b: usize) -> (&[[u64; W]], &[[u64; W]], &[Complex64]) {
        let columns = &self.buckets[b];
        (&columns.x, &columns.z, &columns.coeff)
    }

    /// Number of terms in bucket `b`.
    pub(crate) fn bucket_len(&self, b: usize) -> usize {
        self.buckets[b].len()
    }

    /// Double the bucket count, splitting each bucket in two by the hash's new row.
    pub(crate) fn refine(&mut self) {
        let old_num_buckets = self.buckets.len();
        self.hash.refine();
        let new_bit = self.hash.bits() - 1;
        let hash = &self.hash;

        let mut old = std::mem::take(&mut self.buckets);
        let mut upper: Vec<BucketColumns<W>> =
            (0..old_num_buckets).map(|_| BucketColumns::new()).collect();

        if self.len < PARALLEL_MIN_TERMS {
            for (b, (columns, upper_half)) in old.iter_mut().zip(upper.iter_mut()).enumerate() {
                refine_bucket(columns, upper_half, hash, new_bit, b as u32);
            }
        } else {
            old.par_iter_mut()
                .zip(upper.par_iter_mut())
                .enumerate()
                .for_each(|(b, (columns, upper_half))| {
                    refine_bucket(columns, upper_half, hash, new_bit, b as u32);
                });
        }

        old.extend(upper);
        self.buckets = old;
    }

    /// Halve the bucket count, merging bucket pairs `(i, i + B/2)`.
    pub(crate) fn coarsen(&mut self) {
        self.hash.coarsen();
        let new_num_buckets = self.buckets.len() / 2;

        let old = std::mem::take(&mut self.buckets);
        let (lower, upper) = old.split_at(new_num_buckets);

        let merged: Vec<BucketColumns<W>> = if self.len < PARALLEL_MIN_TERMS {
            lower
                .iter()
                .zip(upper.iter())
                .map(|(low, high)| merge_two(low, high))
                .collect()
        } else {
            lower
                .par_iter()
                .zip(upper.par_iter())
                .map(|(low, high)| merge_two(low, high))
                .collect()
        };

        self.buckets = merged;
    }

    /// Refine until the bucket count suits `len()` terms at `target` per bucket, with the `min_buckets` floor once the sum is large enough; never coarsens.
    pub(crate) fn rebucket(&mut self, target: usize, min_buckets: usize) {
        debug_assert!(target > 0);
        let want = desired_bits(self.len, target, min_buckets).max(self.hash.bits());
        while self.hash.bits() < want {
            self.refine();
        }
    }

    /// The buckets' columns, for read-outs that scan them.
    pub(crate) fn buckets(&self) -> &[BucketColumns<W>] {
        &self.buckets
    }

    /// Mutable access to the buckets, for layers that are applied in place.
    pub(crate) fn buckets_mut(&mut self) -> &mut [BucketColumns<W>] {
        &mut self.buckets
    }

    /// Recompute the cached total after an in-place layer.
    pub(crate) fn recount(&mut self) {
        self.len = self.buckets.iter().map(|c| c.len()).sum();
    }

    /// Iterate every term in canonical order, which is not globally key-sorted.
    pub fn iter(&self) -> impl Iterator<Item = (&[u64; W], &[u64; W], Complex64)> + '_ {
        self.buckets.iter().flat_map(|columns| {
            columns
                .x
                .iter()
                .zip(columns.z.iter())
                .zip(columns.coeff.iter())
                .map(|((x, z), c)| (x, z, *c))
        })
    }

    /// Copy every term out as three parallel columns, in canonical order.
    pub fn to_arrays(&self) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>) {
        let mut x = Vec::with_capacity(self.len);
        let mut z = Vec::with_capacity(self.len);
        let mut coeff = Vec::with_capacity(self.len);
        for columns in self.buckets.iter() {
            x.extend_from_slice(&columns.x);
            z.extend_from_slice(&columns.z);
            coeff.extend_from_slice(&columns.coeff);
        }
        (x, z, coeff)
    }

    /// Coefficient of the term with key `(x, z)`, or `None` if absent.
    pub fn get(&self, x: &[u64; W], z: &[u64; W]) -> Option<Complex64> {
        let columns = &self.buckets[self.hash.bucket_of(x, z) as usize];
        let mut low = 0usize;
        let mut high = columns.len();
        while low < high {
            let mid = low + (high - low) / 2;
            match (&columns.x[mid], &columns.z[mid]).cmp(&(x, z)) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => return Some(columns.coeff[mid]),
            }
        }
        None
    }

    /// Coefficient of the identity term, i.e. `tr(O) / 2^n`.
    pub fn identity_coefficient(&self) -> Complex64 {
        let zero_key = [0u64; W];
        self.get(&zero_key, &zero_key)
            .unwrap_or(Complex64::new(0.0, 0.0))
    }

    /// Multiply every coefficient by `c` in place.
    pub fn scale(&mut self, c: Complex64) {
        self.buckets.par_iter_mut().for_each(|columns| {
            for coeff in columns.coeff.iter_mut() {
                *coeff *= c;
            }
        });
    }

    /// Keep only the terms for which `f(x, z, coeff)` is `true`; `f` may run on several threads at once.
    pub fn retain(&mut self, f: impl Fn(&[u64; W], &[u64; W], Complex64) -> bool + Sync) {
        self.buckets.par_iter_mut().for_each(|columns| {
            let n = columns.len();
            let mut w = 0usize;
            for r in 0..n {
                if f(&columns.x[r], &columns.z[r], columns.coeff[r]) {
                    if w != r {
                        columns.x[w] = columns.x[r];
                        columns.z[w] = columns.z[r];
                        columns.coeff[w] = columns.coeff[r];
                    }
                    w += 1;
                }
            }
            columns.x.truncate(w);
            columns.z.truncate(w);
            columns.coeff.truncate(w);
        });
        self.recount();
    }

    /// Hilbert-Schmidt overlap `tr(self† · other) / 2ⁿ`, i.e. `Σ conj(aᵢ)·bᵢ` over the keys the two sums share.
    ///
    /// # Panics
    ///
    /// Panics if the two sums disagree about `num_qubits`.
    pub fn overlap(&self, other: &Self) -> Complex64 {
        assert_eq!(
            self.num_qubits, other.num_qubits,
            "PauliSum::overlap: num_qubits mismatch ({} vs {})",
            self.num_qubits, other.num_qubits,
        );
        let rhs = other.align_to(&self.hash);
        self.buckets
            .par_iter()
            .zip(rhs.buckets.par_iter())
            .map(|(a, b)| {
                let mut partial = Complex64::new(0.0, 0.0);
                let (mut i, mut j) = (0usize, 0usize);
                while i < a.len() && j < b.len() {
                    match (&a.x[i], &a.z[i]).cmp(&(&b.x[j], &b.z[j])) {
                        std::cmp::Ordering::Less => i += 1,
                        std::cmp::Ordering::Greater => j += 1,
                        std::cmp::Ordering::Equal => {
                            partial += a.coeff[i].conj() * b.coeff[j];
                            i += 1;
                            j += 1;
                        }
                    }
                }
                partial
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    }

    /// Sum of two sums, partitioned as `self`; terms summing to exactly zero are dropped.
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
        let buckets: Vec<BucketColumns<W>> = self
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

    /// Assert the structural invariant: every term in its hash bucket, each bucket strictly ascending in `(x, z)`, every key within `num_qubits`.
    #[cfg(any(test, debug_assertions, feature = "test-utils"))]
    pub(crate) fn assert_invariants(&self) {
        assert_eq!(
            self.buckets.len(),
            self.hash.num_buckets(),
            "PauliSum: bucket count disagrees with hash",
        );
        let mut total = 0usize;
        for (b, columns) in self.buckets.iter().enumerate() {
            assert_eq!(columns.x.len(), columns.z.len());
            assert_eq!(columns.x.len(), columns.coeff.len());
            total += columns.len();
            for i in 0..columns.len() {
                let got = self.hash.bucket_of(&columns.x[i], &columns.z[i]);
                assert_eq!(
                    got as usize, b,
                    "PauliSum: term {i} of bucket {b} hashes to {got}",
                );
                let term = crate::pauli_string::PauliString::<W> {
                    x: columns.x[i],
                    z: columns.z[i],
                };
                assert!(
                    term.is_within(self.num_qubits),
                    "PauliSum: term {i} of bucket {b} exceeds num_qubits",
                );
            }
            for i in 1..columns.len() {
                let prev = (&columns.x[i - 1], &columns.z[i - 1]);
                let cur = (&columns.x[i], &columns.z[i]);
                assert!(prev < cur, "PauliSum: bucket {b} out of order at {i}");
            }
        }
        assert_eq!(total, self.len, "PauliSum: cached len disagrees");
    }
}

#[cfg(test)]
mod tests;
