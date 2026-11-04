//! [`PauliSum`]'s per-bucket column storage, the bucket-count policy and the merge helpers.

use std::borrow::Cow;

use crate::coefficient::Coefficient;
use rayon::prelude::*;

use super::hash::{Gf2Hash, INDEX_BITS};
use crate::pauli_string::PauliString;

/// Default seed for the partitioning hash.
pub const DEFAULT_HASH_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Bucket bits for `len` terms: the smallest `b` with `len <= target << b`, clamped below by the parallelism floor and above by the index width; a sum's own hash caps it further at [`Gf2Hash::max_bits`].
pub fn desired_bits(len: usize, target: usize, min_buckets: usize) -> u8 {
    debug_assert!(target > 0);
    let worth_splitting = len >= min_buckets.saturating_mul(MIN_TERMS_PER_TASK);
    let mut floor = 0u8;
    if worth_splitting {
        while (1usize << floor) < min_buckets && floor < INDEX_BITS {
            floor += 1;
        }
    }
    let mut b = floor;
    while b < INDEX_BITS && len > target.saturating_mul(1usize << b) {
        b += 1;
    }
    b
}

/// Bucket bits a sum at `bits` moves to before a layer: [`desired_bits`] once buckets average above `target` or below `target / COARSEN_BELOW`, otherwise `bits`.
// After a move buckets average `(target / 2, target]`, so a refine and a coarsen are at least a 4× change in length apart.
pub fn rebucketed_bits(len: usize, bits: u8, target: usize, min_buckets: usize) -> u8 {
    let want = desired_bits(len, target, min_buckets);
    let sparse = len.saturating_mul(COARSEN_BELOW) < target.saturating_mul(1usize << bits);
    if want > bits || sparse {
        want
    } else {
        bits
    }
}

/// How far below `target` the mean bucket length falls before [`rebucketed_bits`] coarsens.
pub const COARSEN_BELOW: usize = 8;

/// Minimum terms per bucket for the parallelism floor to apply.
// Not lower without checking a dense and a sparse PTM layer: the bucket count also caps the coset dimension a dense layer's sort needs (research/FINDINGS.md §The dense-PTM bucket cliff is a delta-span rank effect).
pub(crate) const MIN_TERMS_PER_TASK: usize = 64;

/// Below this many terms the bucket-parallel maintenance passes run serially.
const PARALLEL_MIN_TERMS: usize = DEFAULT_MIN_BUCKETS * MIN_TERMS_PER_TASK;

/// Default target terms per bucket.
pub const DEFAULT_TARGET_BUCKET_LEN: usize = 1024;

/// Default floor on the bucket count, fixed rather than thread-derived, so the partition is a function of the sum alone.
/// Must be `>= 16`, or `desired_bits` stops giving a sum of `<= 1024` terms a single bucket.
pub const DEFAULT_MIN_BUCKETS: usize = 128;

/// One bucket's columns, whose capacity is retained across layers.
#[derive(Clone, Debug, Default)]
pub(crate) struct BucketColumns<const W: usize, C: Coefficient = num_complex::Complex64> {
    pub(crate) x: Vec<[u64; W]>,
    pub(crate) z: Vec<[u64; W]>,
    pub(crate) coeff: Vec<C>,
}

impl<const W: usize, C: Coefficient> BucketColumns<W, C> {
    pub(super) fn new() -> Self {
        Self::with_capacity(0)
    }

    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            x: Vec::with_capacity(capacity),
            z: Vec::with_capacity(capacity),
            coeff: Vec::with_capacity(capacity),
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

    pub(crate) fn push(&mut self, x: [u64; W], z: [u64; W], c: C) {
        self.x.push(x);
        self.z.push(z);
        self.coeff.push(c);
    }

    /// Keep, in order, the terms `update(x, z, coeff)` maps to a coefficient, rewritten to it.
    pub(crate) fn update(&mut self, mut update: impl FnMut(&[u64; W], &[u64; W], C) -> Option<C>) {
        let n = self.len();
        let mut w = 0usize;
        for r in 0..n {
            if let Some(coeff) = update(&self.x[r], &self.z[r], self.coeff[r]) {
                if w != r {
                    self.x[w] = self.x[r];
                    self.z[w] = self.z[r];
                }
                self.coeff[w] = coeff;
                w += 1;
            }
        }
        self.x.truncate(w);
        self.z.truncate(w);
        self.coeff.truncate(w);
    }

    /// Keep, in order, the terms `keep(x, z, coeff)` accepts.
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&[u64; W], &[u64; W], C) -> bool) {
        self.update(|x, z, c| keep(x, z, c).then_some(c));
    }

    fn extend_from(&mut self, other: &Self, start: usize) {
        self.x.extend_from_slice(&other.x[start..]);
        self.z.extend_from_slice(&other.z[start..]);
        self.coeff.extend_from_slice(&other.coeff[start..]);
    }

    pub(crate) fn view(&self) -> ColumnsRef<'_, W, C> {
        ColumnsRef { x: &self.x, z: &self.z, coeff: &self.coeff }
    }
}

/// Borrowed `(x, z, coeff)` columns of equal length, which need not come from one [`BucketColumns`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ColumnsRef<'a, const W: usize, C: Coefficient = num_complex::Complex64> {
    pub(crate) x: &'a [[u64; W]],
    pub(crate) z: &'a [[u64; W]],
    pub(crate) coeff: &'a [C],
}

/// Mutable `(x, z, coeff)` columns of equal length, which need not come from one [`BucketColumns`].
#[derive(Debug)]
pub(crate) struct ColumnsMut<'a, const W: usize, C: Coefficient = num_complex::Complex64> {
    pub(crate) x: &'a mut [[u64; W]],
    pub(crate) z: &'a mut [[u64; W]],
    pub(crate) coeff: &'a mut [C],
}

impl<'a, const W: usize, C: Coefficient> ColumnsMut<'a, W, C> {
    pub(crate) fn len(&self) -> usize {
        self.coeff.len()
    }

    pub(crate) fn split_at(self, at: usize) -> (ColumnsMut<'a, W, C>, ColumnsMut<'a, W, C>) {
        let (x_head, x_tail) = self.x.split_at_mut(at);
        let (z_head, z_tail) = self.z.split_at_mut(at);
        let (coeff_head, coeff_tail) = self.coeff.split_at_mut(at);
        (
            ColumnsMut { x: x_head, z: z_head, coeff: coeff_head },
            ColumnsMut { x: x_tail, z: z_tail, coeff: coeff_tail },
        )
    }
}

/// `f` over every item, on the Rayon pool once the sum holds `PARALLEL_MIN_TERMS` terms.
pub(super) fn map_buckets<Items, T>(
    num_terms: usize,
    items: Items,
    f: impl Fn(<Items as IntoParallelIterator>::Item) -> T + Sync + Send,
) -> Vec<T>
where
    Items: IntoParallelIterator + IntoIterator<Item = <Items as IntoParallelIterator>::Item>,
    T: Send,
{
    if num_terms < PARALLEL_MIN_TERMS {
        items.into_iter().map(f).collect()
    } else {
        items.into_par_iter().map(f).collect()
    }
}

/// Split off the terms whose hash row `new_bit` is set, returning them in order and keeping the rest in place.
fn refine_bucket<const W: usize, C: Coefficient>(
    columns: &mut BucketColumns<W, C>,
    hash: &Gf2Hash<W>,
    new_bit: u8,
) -> BucketColumns<W, C> {
    let mut upper_half = BucketColumns::new();
    let mut keep = 0usize;
    for i in 0..columns.len() {
        if hash.row_parity(&columns.x[i], &columns.z[i], new_bit) == 1 {
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
    upper_half
}

/// Merge two sorted runs; `on_equal` combines a shared key's coefficients, `None` dropping the key.
pub(super) fn merge_sorted<const W: usize, C: Coefficient>(
    a: &BucketColumns<W, C>,
    b: &BucketColumns<W, C>,
    mut on_equal: impl FnMut(C, C) -> Option<C>,
) -> BucketColumns<W, C> {
    let mut out = BucketColumns::with_capacity(a.len() + b.len());
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
                if let Some(c) = on_equal(a.coeff[i], b.coeff[j]) {
                    out.push(a.x[i], a.z[i], c);
                }
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from(a, i);
    out.extend_from(b, j);
    out
}

/// The `on_equal` of [`merge_sorted`] for runs whose keys are disjoint.
pub(super) fn keys_are_disjoint<C: Coefficient>(_: C, _: C) -> Option<C> {
    debug_assert!(false, "PauliSum: duplicate key across merged runs");
    None
}

/// Weighted sum of Pauli strings, stored as structure-of-arrays columns partitioned into buckets by a GF(2)-linear hash.
///
/// Terms come in canonical order: bucket index ascending, then lexicographic `(x, z)` within a bucket, with no repeated key.
/// A sum of at most 1024 terms built by [`crate::BuildAccumulator`] has one bucket and so is plain lex-sorted; a larger one interleaves buckets, so compare sums by key ([`Self::get`]), not by position.
#[derive(Debug)]
pub struct PauliSum<const W: usize, C: Coefficient = num_complex::Complex64> {
    pub(super) buckets: Vec<BucketColumns<W, C>>,
    pub(super) hash: Gf2Hash<W>,
    pub(super) num_qubits: usize,
    pub(super) len: usize,
}

/// Bucket by bucket on the Rayon pool, so the copy's pages are first-touched by the workers that will stream them rather than all on the caller's NUMA node.
impl<const W: usize, C: Coefficient> Clone for PauliSum<W, C> {
    fn clone(&self) -> Self {
        Self {
            buckets: map_buckets(self.len, &self.buckets, BucketColumns::clone),
            hash: self.hash.clone(),
            num_qubits: self.num_qubits,
            len: self.len,
        }
    }
}

impl<const W: usize, C: Coefficient> PauliSum<W, C> {
    /// Partition terms the caller has sorted ascending in `(x, z)` with no repeated key; unsorted input silently breaks the bucket invariant.
    pub(crate) fn from_key_sorted(
        x: &[[u64; W]],
        z: &[[u64; W]],
        coeff: &[C],
        hash: Gf2Hash<W>,
        num_qubits: usize,
    ) -> Self {
        let n = coeff.len();
        let num_buckets = hash.num_buckets();

        let bucket_indices: Vec<u32> =
            (0..n).into_par_iter().map(|i| hash.bucket_of(&x[i], &z[i])).collect();
        let mut counts: Vec<usize> = vec![0; num_buckets];
        for &b in bucket_indices.iter() {
            counts[b as usize] += 1;
        }

        let mut buckets: Vec<BucketColumns<W, C>> =
            counts.iter().map(|&c| BucketColumns::with_capacity(c)).collect();

        for i in 0..n {
            buckets[bucket_indices[i] as usize].push(x[i], z[i], coeff[i]);
        }

        Self { buckets, hash, num_qubits, len: n }
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

    /// Repartition under `hash`, keeping every term, at `hash`'s bucket count; propagation keeps the hash a sum carries.
    ///
    /// # Panics
    ///
    /// If `hash` was built for a different qubit count.
    pub fn with_hash(self, hash: Gf2Hash<W>) -> Self {
        assert_eq!(self.num_qubits, hash.num_qubits(), "PauliSum::with_hash: num_qubits mismatch",);
        let targets: Vec<Vec<u32>> = map_buckets(self.len, &self.buckets, |columns| {
            (0..columns.len()).map(|i| hash.bucket_of(&columns.x[i], &columns.z[i])).collect()
        });
        let mut scattered: Vec<Vec<(PauliString<W>, C)>> =
            (0..hash.num_buckets()).map(|_| Vec::new()).collect();
        for (columns, targets) in self.buckets.iter().zip(&targets) {
            for (i, &b) in targets.iter().enumerate() {
                let key = PauliString { x: columns.x[i], z: columns.z[i] };
                scattered[b as usize].push((key, columns.coeff[i]));
            }
        }
        let buckets = map_buckets(self.len, scattered, |mut terms| {
            terms.sort_unstable_by(|p, q| p.0.cmp(&q.0));
            let mut columns = BucketColumns::with_capacity(terms.len());
            for (key, c) in terms {
                columns.push(key.x, key.z, c);
            }
            columns
        });
        Self { buckets, hash, num_qubits: self.num_qubits, len: self.len }
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

    /// Terms per bucket, in bucket order: the occupancy the hash produces.
    pub fn bucket_lengths(&self) -> Vec<usize> {
        self.buckets.iter().map(BucketColumns::len).collect()
    }

    /// Borrow bucket `b`'s columns as `(x, z, coeff)`.
    pub(crate) fn bucket(&self, b: usize) -> (&[[u64; W]], &[[u64; W]], &[C]) {
        let columns = &self.buckets[b];
        (&columns.x, &columns.z, &columns.coeff)
    }

    /// Number of terms in bucket `b`.
    pub(crate) fn bucket_len(&self, b: usize) -> usize {
        self.buckets[b].len()
    }

    /// Double the bucket count, splitting each bucket in two by the hash's new row.
    pub(crate) fn refine(&mut self) {
        self.hash.refine();
        let new_bit = self.hash.bits() - 1;
        let hash = &self.hash;
        let mut buckets = std::mem::take(&mut self.buckets);
        let upper =
            map_buckets(self.len, &mut buckets, |columns| refine_bucket(columns, hash, new_bit));
        buckets.extend(upper);
        self.buckets = buckets;
        #[cfg(debug_assertions)]
        self.assert_invariants();
    }

    /// Halve the bucket count, merging bucket pairs `(i, i + B/2)`.
    pub(crate) fn coarsen(&mut self) {
        self.hash.coarsen();
        let (lower, upper) = self.buckets.split_at(self.buckets.len() / 2);
        self.buckets = map_buckets(self.len, 0..lower.len(), |b| {
            merge_sorted(&lower[b], &upper[b], keys_are_disjoint)
        });
    }

    /// Move to [`rebucketed_bits`]: refine or coarsen until the count suits `len()` terms at `target` per bucket, with the `min_buckets` floor once the sum is large enough.
    pub(crate) fn rebucket(&mut self, target: usize, min_buckets: usize) {
        debug_assert!(target > 0);
        let want = rebucketed_bits(self.len, self.hash.bits(), target, min_buckets)
            .min(self.hash.max_bits());
        while self.hash.bits() < want {
            self.refine();
        }
        self.coarsen_to(want.min(self.hash.bits()));
    }

    /// The buckets' columns, for read-outs that scan them.
    pub(crate) fn buckets(&self) -> &[BucketColumns<W, C>] {
        &self.buckets
    }

    /// Mutable access to the buckets, for layers that are applied in place.
    pub(crate) fn buckets_mut(&mut self) -> &mut [BucketColumns<W, C>] {
        &mut self.buckets
    }

    /// Recompute the cached total after an in-place layer.
    pub(crate) fn recount(&mut self) {
        self.len = self.buckets.iter().map(BucketColumns::len).sum();
    }

    /// Set the cached total to `len`, which an in-place layer counted while it wrote the buckets.
    pub(crate) fn set_len_after_layer(&mut self, len: usize) {
        debug_assert_eq!(len, self.buckets.iter().map(BucketColumns::len).sum::<usize>());
        self.len = len;
    }

    /// Iterate every term in canonical order, which is not globally key-sorted.
    pub fn iter(&self) -> impl Iterator<Item = (&[u64; W], &[u64; W], C)> + '_ {
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
    pub fn to_arrays(&self) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<C>) {
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
    pub fn get(&self, x: &[u64; W], z: &[u64; W]) -> Option<C> {
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
    pub fn identity_coefficient(&self) -> C {
        let zero_key = [0u64; W];
        self.get(&zero_key, &zero_key).unwrap_or(C::ZERO)
    }

    /// A copy with every coefficient converted to `T`, partitioned as `self`; `None` when a real `T` meets a coefficient whose imaginary part exceeds `tolerance`.
    pub fn cast<T: Coefficient>(&self, tolerance: f64) -> Option<PauliSum<W, T>> {
        let buckets = map_buckets(self.len, &self.buckets, |columns| {
            let coeff = columns
                .coeff
                .iter()
                .map(|&c| T::from_complex64(c.to_complex64(), tolerance))
                .collect::<Option<Vec<T>>>()?;
            Some(BucketColumns { x: columns.x.clone(), z: columns.z.clone(), coeff })
        })
        .into_iter()
        .collect::<Option<Vec<_>>>()?;
        Some(PauliSum {
            buckets,
            hash: self.hash.clone(),
            num_qubits: self.num_qubits,
            len: self.len,
        })
    }

    /// Multiply every coefficient by `c` in place.
    pub fn scale(&mut self, c: C) {
        self.buckets.par_iter_mut().for_each(|columns| {
            for coeff in columns.coeff.iter_mut() {
                *coeff *= c;
            }
        });
    }

    /// Keep only the terms for which `f(x, z, coeff)` is `true`; `f` may run on several threads at once.
    pub fn retain(&mut self, f: impl Fn(&[u64; W], &[u64; W], C) -> bool + Sync) {
        self.retain_per_bucket(|_| &f);
    }

    /// [`retain`](Self::retain) with one filter per bucket, `filter_for(b)`, run over bucket `b` in order.
    pub(crate) fn retain_per_bucket<F>(&mut self, filter_for: impl Fn(usize) -> F + Sync)
    where
        F: FnMut(&[u64; W], &[u64; W], C) -> bool,
    {
        self.update_per_bucket(|b| {
            let mut keep = filter_for(b);
            move |x: &[u64; W], z: &[u64; W], c: C| keep(x, z, c).then_some(c)
        });
    }

    /// [`retain_per_bucket`](Self::retain_per_bucket) that also rewrites each kept coefficient: `update_for(b)` maps a term to its new coefficient, or to `None` to drop it.
    pub(crate) fn update_per_bucket<F>(&mut self, update_for: impl Fn(usize) -> F + Sync)
    where
        F: FnMut(&[u64; W], &[u64; W], C) -> Option<C>,
    {
        self.buckets
            .par_iter_mut()
            .enumerate()
            .for_each(|(b, columns)| columns.update(update_for(b)));
        self.recount();
    }

    /// Hilbert-Schmidt overlap `tr(self† · other) / 2ⁿ`, i.e. `Σ conj(aᵢ)·bᵢ` over the keys the two sums share.
    ///
    /// # Panics
    ///
    /// Panics if the two sums disagree about `num_qubits`.
    pub fn overlap(&self, other: &Self) -> C {
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
                let mut partial = C::ZERO;
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
            .fold(C::ZERO, |a, b| a + b)
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
        let zero = C::ZERO;
        let buckets: Vec<BucketColumns<W, C>> = self
            .buckets
            .par_iter()
            .zip(rhs.buckets.par_iter())
            .map(|(a, b)| merge_sorted(a, b, |p, q| Some(p + q).filter(|c| *c != zero)))
            .collect();
        let len = buckets.iter().map(|c| c.len()).sum();
        Self { buckets, hash: self.hash.clone(), num_qubits: self.num_qubits, len }
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
                assert_eq!(got as usize, b, "PauliSum: term {i} of bucket {b} hashes to {got}",);
                let term = PauliString::<W> { x: columns.x[i], z: columns.z[i] };
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
