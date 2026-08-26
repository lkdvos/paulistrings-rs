//! [`BucketedSum`] — a Pauli sum partitioned by [`Gf2Hash`]. See v0.2 design
//! doc §4.

use num_complex::Complex64;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::hash::Gf2Hash;
use crate::pauli_string::PauliString;
use crate::pauli_sum::PauliSum;

/// Lexicographic `(x, z)` merge key. `[u64; W]`'s derived `Ord` is lex over
/// elements, so this matches `PauliString::cmp` and `sort_phase` exactly.
type MergeKey<const W: usize> = ([u64; W], [u64; W]);

/// Min-heap over bucket heads for the `B`-way merge in
/// [`BucketedSum::into_sum`], carrying the source bucket index alongside the key.
type MergeHeap<const W: usize> = BinaryHeap<Reverse<(MergeKey<W>, usize)>>;

/// Minimum terms per bucket for the parallelism floor to apply.
///
/// The floor in [`BucketedSum::rebucket`] exists to give Rayon enough
/// independent tasks, but a task carrying almost nothing is pure overhead. Below
/// `min_buckets × MIN_TERMS_PER_TASK` total terms we would rather have few
/// buckets and let the small-`n` fallback to the whole-sum path handle it
/// (v0.2 §6). Provisional; v0.2 §7.4 measures it.
pub const MIN_TERMS_PER_TASK: usize = 64;

/// Default target terms per bucket.
///
/// Chosen so a bucket plus its gather scratch stays L2-resident: a term at
/// `W = 2` is `2·8·2 + 16 = 48` bytes, so 1024 terms is ~48 KB against 1 MiB of
/// L2 per core on the reference host. Provisional — v0.2 §7.4 measures it.
pub const DEFAULT_TARGET_BUCKET_LEN: usize = 1024;

/// One bucket's structure-of-arrays columns.
///
/// Capacity is retained across layers, which is the point of owning per-bucket
/// columns rather than slicing one flat array: v0.1 allocated ~11 + 3k buffers
/// per layer and reused none of them (v0.2 §4.2).
#[derive(Clone, Debug, Default)]
pub(crate) struct BucketCols<const W: usize> {
    pub(crate) x: Vec<[u64; W]>,
    pub(crate) z: Vec<[u64; W]>,
    pub(crate) coeff: Vec<Complex64>,
}

impl<const W: usize> BucketCols<W> {
    fn new() -> Self {
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
    fn clear(&mut self) {
        self.x.clear();
        self.z.clear();
        self.coeff.clear();
    }

    #[inline]
    fn push(&mut self, x: [u64; W], z: [u64; W], c: Complex64) {
        self.x.push(x);
        self.z.push(z);
        self.coeff.push(c);
    }
}

/// A Pauli sum partitioned by a GF(2)-linear hash — the propagation working
/// form.
///
/// [`PauliSum`] remains the public, canonical, globally-sorted type;
/// `BucketedSum` is what a layer actually operates on. [`propagate`] converts in
/// once, runs every layer bucketed, and converts out once — amortizing both
/// conversions over the whole circuit (4320 layers for a single 6x6 Ising
/// quench).
///
/// # Invariant
///
/// Every term lies in `buckets[hash.bucket_of(term)]`, and each bucket is sorted
/// by the lexicographic `(x, z)` key with no duplicate keys. Because `h` is a
/// function, equal keys always share a bucket — so per-bucket dedup *implies*
/// global dedup, and no global sort is ever needed (v0.2 §2.5).
///
/// [`propagate`]: crate::propagate
#[derive(Clone, Debug)]
pub struct BucketedSum<const W: usize> {
    buckets: Vec<BucketCols<W>>,
    /// Retired bucket storage, kept for its capacity so a layer does not
    /// allocate. See [`BucketedSum::begin_layer`].
    spare: Vec<BucketCols<W>>,
    hash: Gf2Hash<W>,
    num_qubits: usize,
    len: usize,
}

impl<const W: usize> BucketedSum<W> {
    /// Partition a sorted [`PauliSum`] by `hash`.
    ///
    /// `O(n)`: one hash evaluation and one scatter per term. Because the input
    /// is globally sorted and terms are appended in input order, each bucket
    /// comes out sorted for free — order outside a bucket is irrelevant, and
    /// order within one is inherited.
    pub fn from_sum(sum: &PauliSum<W>, hash: Gf2Hash<W>) -> Self {
        let n = sum.len();
        let nb = hash.num_buckets();

        // Hashing is the expensive part, so index once and count from the
        // indices rather than hashing twice.
        let mut idx: Vec<u32> = Vec::with_capacity(n);
        let mut counts: Vec<usize> = vec![0; nb];
        for i in 0..n {
            let b = hash.bucket_of(&sum.x()[i], &sum.z()[i]);
            counts[b as usize] += 1;
            idx.push(b);
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
            buckets[idx[i] as usize].push(sum.x()[i], sum.z()[i], sum.coeff()[i]);
        }

        Self {
            buckets,
            spare: Vec::new(),
            hash,
            num_qubits: sum.num_qubits(),
            len: n,
        }
    }

    /// An empty bucketed sum over `num_qubits`, partitioned by `hash`.
    pub fn empty(num_qubits: usize, hash: Gf2Hash<W>) -> Self {
        let nb = hash.num_buckets();
        Self {
            buckets: (0..nb).map(|_| BucketCols::new()).collect(),
            spare: Vec::new(),
            hash,
            num_qubits,
            len: 0,
        }
    }

    /// Collapse back to a globally-sorted [`PauliSum`].
    ///
    /// A `B`-way merge over the already-sorted buckets, `O(n log B)`. No
    /// coefficient combining is needed: equal keys share a bucket and buckets
    /// are already deduplicated, so every key in the merge is distinct.
    pub fn into_sum(self) -> PauliSum<W> {
        self.to_sum()
    }

    /// Collapse to a globally-sorted [`PauliSum`] without consuming `self`.
    ///
    /// Same `B`-way merge as [`Self::into_sum`]; used by the default
    /// `finalize_layer_bucketed`, which has to hand a `&mut PauliSum` to a
    /// policy that only knows how to work on the whole sum.
    pub fn to_sum(&self) -> PauliSum<W> {
        let mut x: Vec<[u64; W]> = Vec::with_capacity(self.len);
        let mut z: Vec<[u64; W]> = Vec::with_capacity(self.len);
        let mut coeff: Vec<Complex64> = Vec::with_capacity(self.len);

        // Min-heap over bucket heads, keyed by the same lexicographic
        // `(x, z)` order `PauliString::cmp` and `sort_phase` use.
        let mut heap: MergeHeap<W> = BinaryHeap::with_capacity(self.buckets.len());
        let mut cursor: Vec<usize> = vec![0; self.buckets.len()];

        for (b, cols) in self.buckets.iter().enumerate() {
            if cols.len() > 0 {
                heap.push(Reverse(((cols.x[0], cols.z[0]), b)));
            }
        }

        while let Some(Reverse(((kx, kz), b))) = heap.pop() {
            let i = cursor[b];
            x.push(kx);
            z.push(kz);
            coeff.push(self.buckets[b].coeff[i]);
            cursor[b] = i + 1;
            if cursor[b] < self.buckets[b].len() {
                let j = cursor[b];
                heap.push(Reverse(((self.buckets[b].x[j], self.buckets[b].z[j]), b)));
            }
        }

        PauliSum::<W> {
            x,
            z,
            coeff,
            num_qubits: self.num_qubits,
        }
    }

    /// Replace the contents from a globally-sorted [`PauliSum`], keeping the
    /// current hash and reusing bucket storage.
    ///
    /// The counterpart of [`Self::to_sum`]. `sum` must be sorted and
    /// deduplicated, i.e. satisfy `PauliSum`'s invariant.
    pub fn refill_from_sum(&mut self, sum: &PauliSum<W>) {
        for cols in self.buckets.iter_mut() {
            cols.clear();
        }
        for i in 0..sum.len() {
            let b = self.hash.bucket_of(&sum.x()[i], &sum.z()[i]) as usize;
            self.buckets[b].push(sum.x()[i], sum.z()[i], sum.coeff()[i]);
        }
        self.len = sum.len();
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
    ///
    /// One parity evaluation per term. The new index is `i` or `i + B` (the new
    /// hash bit is the *high* bit), and both halves inherit the source bucket's
    /// order, so nothing is re-sorted (v0.2 §2.7).
    pub fn refine(&mut self) {
        let old_nb = self.buckets.len();
        self.hash.refine();
        let new_bit = self.hash.bits() - 1;

        // Take the old buckets out so the upper halves can reuse their storage.
        let mut old = std::mem::take(&mut self.buckets);
        let mut upper: Vec<BucketCols<W>> = (0..old_nb).map(|_| BucketCols::new()).collect();

        for (b, cols) in old.iter_mut().enumerate() {
            let mut keep = 0usize;
            let n = cols.len();
            for i in 0..n {
                let full = self.hash.bucket_of(&cols.x[i], &cols.z[i]);
                debug_assert_eq!(
                    full & ((1u32 << new_bit) - 1),
                    b as u32,
                    "refine: low bits must be preserved",
                );
                if (full >> new_bit) & 1 == 1 {
                    upper[b].push(cols.x[i], cols.z[i], cols.coeff[i]);
                } else {
                    // Compact in place: `keep <= i` always, so this never
                    // overwrites an unread slot.
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

        old.extend(upper);
        self.buckets = old;
    }

    /// Halve the bucket count, merging bucket pairs `(i, i + B/2)`.
    ///
    /// A 2-way merge per pair. No coefficient combining: equal keys agree at
    /// every prefix length, so they were already in the same source bucket.
    pub fn coarsen(&mut self) {
        self.hash.coarsen();
        let new_nb = self.buckets.len() / 2;

        let old = std::mem::take(&mut self.buckets);
        let (lower, upper) = old.split_at(new_nb);
        let mut merged: Vec<BucketCols<W>> = Vec::with_capacity(new_nb);

        for (lo, hi) in lower.iter().zip(upper.iter()) {
            let mut out = BucketCols::<W>::new();
            out.x.reserve_exact(lo.len() + hi.len());
            out.z.reserve_exact(lo.len() + hi.len());
            out.coeff.reserve_exact(lo.len() + hi.len());
            let (mut i, mut j) = (0usize, 0usize);
            while i < lo.len() && j < hi.len() {
                if (&lo.x[i], &lo.z[i]) <= (&hi.x[j], &hi.z[j]) {
                    out.push(lo.x[i], lo.z[i], lo.coeff[i]);
                    i += 1;
                } else {
                    out.push(hi.x[j], hi.z[j], hi.coeff[j]);
                    j += 1;
                }
            }
            while i < lo.len() {
                out.push(lo.x[i], lo.z[i], lo.coeff[i]);
                i += 1;
            }
            while j < hi.len() {
                out.push(hi.x[j], hi.z[j], hi.coeff[j]);
                j += 1;
            }
            merged.push(out);
        }

        self.buckets = merged;
    }

    /// Bring the mean bucket occupancy back toward `target`.
    ///
    /// Hysteresis: refine only above `4 × target`, coarsen only below
    /// `target / 4`. `n` swings by orders of magnitude every layer — fanout
    /// grows it, truncation cuts it back — so acting on every small deviation
    /// would thrash. Also keeps at least `4 × threads` buckets once there is
    /// enough work to spread, so the bucket-parallel decomposition has slack to
    /// load-balance (v0.2 §4.3).
    ///
    /// All three constants are provisional; v0.2 §7.4 measures them.
    pub fn rebucket(&mut self, target: usize, min_buckets: usize) {
        debug_assert!(target > 0);
        let min_bits = {
            let mut b = 0u8;
            while (1usize << b) < min_buckets && b < super::hash::B_MAX_BITS {
                b += 1;
            }
            b
        };

        // The parallelism floor applies only when there is enough work to be
        // worth spreading. Note this is deliberately independent of `target`:
        // a caller asking for huge buckets still wants its cores fed, but a
        // 10-term sum wants neither.
        let worth_splitting = self.len >= min_buckets.saturating_mul(MIN_TERMS_PER_TASK);
        let floor_bits = if worth_splitting { min_bits } else { 0 };

        // Grow while the mean is more than 4x the target.
        while self.hash.bits() < super::hash::B_MAX_BITS
            && self.len > 4 * target * self.num_buckets()
        {
            self.refine();
        }
        // Grow to the parallelism floor.
        while self.hash.bits() < floor_bits {
            self.refine();
        }
        // Shrink while the mean is under a quarter of the target, never below
        // the floor.
        while self.hash.bits() > floor_bits && 4 * self.len < target * self.num_buckets() {
            self.coarsen();
        }
    }

    /// Start a layer: take the current buckets out as the layer's read-only
    /// input, and hand back a cleared output set reusing the spare's capacity.
    ///
    /// The two must be separate allocations because a layer reads several input
    /// buckets while writing one output bucket, so they cannot alias. Recycling
    /// the spare is what keeps a layer allocation-free after the first
    /// (v0.2 §4.2).
    pub(crate) fn begin_layer(&mut self) -> (Vec<BucketCols<W>>, Vec<BucketCols<W>>) {
        let input = std::mem::take(&mut self.buckets);
        let mut out = std::mem::take(&mut self.spare);
        out.truncate(input.len());
        out.resize_with(input.len(), BucketCols::new);
        for cols in out.iter_mut() {
            cols.clear();
        }
        (input, out)
    }

    /// Finish a layer: install `output` and retire `input` as the new spare.
    pub(crate) fn end_layer(&mut self, output: Vec<BucketCols<W>>, mut spare: Vec<BucketCols<W>>) {
        self.len = output.iter().map(|c| c.len()).sum();
        for cols in spare.iter_mut() {
            cols.clear();
        }
        self.buckets = output;
        self.spare = spare;
    }

    /// Mutable access to the buckets, for layers that can be applied in place.
    pub(crate) fn buckets_mut(&mut self) -> &mut [BucketCols<W>] {
        &mut self.buckets
    }

    /// Recompute the cached total after an in-place layer.
    pub(crate) fn recount(&mut self) {
        self.len = self.buckets.iter().map(|c| c.len()).sum();
    }

    /// Assert the structural invariant. Debug builds only.
    ///
    /// Checks that every term is in its hash bucket, that each bucket is
    /// strictly ascending in `(x, z)` (so sorted *and* duplicate-free), that
    /// every key is within `num_qubits`, and that the cached length agrees.
    #[cfg(debug_assertions)]
    pub fn assert_invariants(&self) {
        assert_eq!(
            self.buckets.len(),
            self.hash.num_buckets(),
            "BucketedSum: bucket count disagrees with hash",
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
                    "BucketedSum: term {i} of bucket {b} hashes to {got}",
                );
                let term = PauliString::<W> {
                    x: cols.x[i],
                    z: cols.z[i],
                };
                assert!(
                    term.is_within(self.num_qubits),
                    "BucketedSum: term {i} of bucket {b} exceeds num_qubits",
                );
            }
            for i in 1..cols.len() {
                let prev = (&cols.x[i - 1], &cols.z[i - 1]);
                let cur = (&cols.x[i], &cols.z[i]);
                assert!(prev < cur, "BucketedSum: bucket {b} out of order at {i}");
            }
        }
        assert_eq!(total, self.len, "BucketedSum: cached len disagrees");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulator::BuildAccumulator;
    use crate::phase::Phase;

    struct Xs64(u64);
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

    fn word_mask(num_qubits: usize, word: usize) -> u64 {
        let lo = 64 * word;
        if num_qubits >= lo + 64 {
            !0u64
        } else if num_qubits <= lo {
            0
        } else {
            (1u64 << (num_qubits - lo)) - 1
        }
    }

    /// A valid (sorted, deduplicated) `PauliSum` of at most `n` terms.
    fn rand_sum<const W: usize>(n: usize, num_qubits: usize, seed: u64) -> PauliSum<W> {
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

    /// Low-weight keys — the physically relevant regime, and the one where a
    /// badly chosen hash would collapse into one bucket.
    fn rand_low_weight_sum<const W: usize>(
        n: usize,
        num_qubits: usize,
        weight: usize,
        seed: u64,
    ) -> PauliSum<W> {
        let mut rng = Xs64::new(seed);
        let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n);
        for _ in 0..n {
            let mut p = PauliString::<W> {
                x: [0u64; W],
                z: [0u64; W],
            };
            for _ in 0..weight {
                let q = (rng.next_u64() as usize) % num_qubits;
                let bit = 1u64 << (q % 64);
                match rng.next_u64() % 3 {
                    0 => p.x[q / 64] |= bit,
                    1 => p.z[q / 64] |= bit,
                    _ => {
                        p.x[q / 64] |= bit;
                        p.z[q / 64] |= bit;
                    }
                }
            }
            let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
            acc.add_term(p, Phase::ONE, Complex64::new(re, 0.0));
        }
        acc.finalize()
    }

    fn assert_same_sum<const W: usize>(a: &PauliSum<W>, b: &PauliSum<W>) {
        assert_eq!(a.len(), b.len(), "length");
        assert_eq!(a.num_qubits(), b.num_qubits(), "num_qubits");
        assert_eq!(a.x(), b.x(), "x column");
        assert_eq!(a.z(), b.z(), "z column");
        assert_eq!(a.coeff(), b.coeff(), "coeff column");
    }

    // ---- round trip ----

    #[test]
    fn round_trip_is_bitwise_identical_w1() {
        let sum = rand_sum::<1>(5000, 64, 0xA1);
        let h = Gf2Hash::<1>::new(64, 7, 0xBEEF);
        let bucketed = BucketedSum::from_sum(&sum, h);
        bucketed.assert_invariants();
        let back = bucketed.into_sum();
        back.assert_invariants();
        assert_same_sum(&sum, &back);
    }

    #[test]
    fn round_trip_is_bitwise_identical_w2() {
        let sum = rand_sum::<2>(5000, 128, 0xA2);
        let h = Gf2Hash::<2>::new(128, 9, 0xBEEF);
        let bucketed = BucketedSum::from_sum(&sum, h);
        bucketed.assert_invariants();
        let back = bucketed.into_sum();
        back.assert_invariants();
        assert_same_sum(&sum, &back);
    }

    #[test]
    fn round_trip_on_low_weight_input() {
        let sum = rand_low_weight_sum::<2>(4000, 100, 4, 0xA3);
        let h = Gf2Hash::<2>::new(100, 8, 0xBEEF);
        let bucketed = BucketedSum::from_sum(&sum, h);
        bucketed.assert_invariants();
        assert_same_sum(&sum, &bucketed.into_sum());
    }

    #[test]
    fn round_trip_at_a_mid_word_qubit_count() {
        // 70 qubits in W=2: word 1 is only partly live.
        let sum = rand_sum::<2>(2000, 70, 0xA4);
        let h = Gf2Hash::<2>::new(70, 6, 0xBEEF);
        let bucketed = BucketedSum::from_sum(&sum, h);
        bucketed.assert_invariants();
        assert_same_sum(&sum, &bucketed.into_sum());
    }

    #[test]
    fn round_trip_empty_sum() {
        let sum = PauliSum::<1>::empty(64);
        let h = Gf2Hash::<1>::new(64, 5, 0x1);
        let bucketed = BucketedSum::from_sum(&sum, h);
        bucketed.assert_invariants();
        assert!(bucketed.is_empty());
        assert_eq!(bucketed.len(), 0);
        assert_same_sum(&sum, &bucketed.into_sum());
    }

    #[test]
    fn round_trip_single_term() {
        let sum = rand_sum::<1>(1, 64, 0xA5);
        assert_eq!(sum.len(), 1);
        let h = Gf2Hash::<1>::new(64, 8, 0x2);
        let bucketed = BucketedSum::from_sum(&sum, h);
        bucketed.assert_invariants();
        assert_same_sum(&sum, &bucketed.into_sum());
    }

    #[test]
    fn round_trip_with_a_single_bucket() {
        // bits = 0: everything in bucket 0, so `into_sum` degenerates to a copy
        // and `from_sum` must already produce a sorted bucket.
        let sum = rand_sum::<1>(2000, 64, 0xA6);
        let h = Gf2Hash::<1>::new(64, 0, 0x3);
        let bucketed = BucketedSum::from_sum(&sum, h);
        assert_eq!(bucketed.num_buckets(), 1);
        bucketed.assert_invariants();
        assert_same_sum(&sum, &bucketed.into_sum());
    }

    #[test]
    fn round_trip_with_more_buckets_than_terms() {
        let sum = rand_sum::<1>(50, 64, 0xA7);
        let h = Gf2Hash::<1>::new(64, 12, 0x4);
        let bucketed = BucketedSum::from_sum(&sum, h);
        assert_eq!(bucketed.num_buckets(), 4096);
        bucketed.assert_invariants();
        assert_same_sum(&sum, &bucketed.into_sum());
    }

    #[test]
    fn empty_constructor_matches_empty_sum() {
        let h = Gf2Hash::<2>::new(128, 6, 0x5);
        let b = BucketedSum::<2>::empty(128, h);
        b.assert_invariants();
        assert_eq!(b.num_buckets(), 64);
        assert_eq!(b.into_sum().len(), 0);
    }

    // ---- refine / coarsen ----

    #[test]
    fn refine_doubles_buckets_and_preserves_content() {
        let sum = rand_sum::<2>(3000, 128, 0xB1);
        let h = Gf2Hash::<2>::new(128, 6, 0xC0DE);
        let mut b = BucketedSum::from_sum(&sum, h);
        let before_len = b.len();

        b.refine();
        assert_eq!(b.num_buckets(), 128);
        assert_eq!(b.len(), before_len);
        // The invariant check is the real assertion: it verifies every term is
        // in its *new* hash bucket and that each bucket is still sorted.
        b.assert_invariants();
        assert_same_sum(&sum, &b.into_sum());
    }

    #[test]
    fn refine_splits_each_bucket_into_the_pair_i_and_i_plus_b() {
        let sum = rand_sum::<1>(3000, 64, 0xB2);
        let h = Gf2Hash::<1>::new(64, 5, 0xC0DE);
        let mut b = BucketedSum::from_sum(&sum, h);
        let old_nb = b.num_buckets();
        let old_lens: Vec<usize> = (0..old_nb).map(|i| b.bucket_len(i)).collect();

        b.refine();
        for (i, &old_len) in old_lens.iter().enumerate() {
            assert_eq!(
                b.bucket_len(i) + b.bucket_len(i + old_nb),
                old_len,
                "bucket {i} did not split into (i, i + B)",
            );
        }
    }

    #[test]
    fn coarsen_halves_buckets_and_preserves_content() {
        let sum = rand_sum::<2>(3000, 128, 0xB3);
        let h = Gf2Hash::<2>::new(128, 7, 0xC0DE);
        let mut b = BucketedSum::from_sum(&sum, h);

        b.coarsen();
        assert_eq!(b.num_buckets(), 64);
        b.assert_invariants();
        assert_same_sum(&sum, &b.into_sum());
    }

    #[test]
    fn coarsen_merges_the_pair_i_and_i_plus_new_b() {
        let sum = rand_sum::<1>(3000, 64, 0xB4);
        let h = Gf2Hash::<1>::new(64, 6, 0xC0DE);
        let mut b = BucketedSum::from_sum(&sum, h);
        let new_nb = b.num_buckets() / 2;
        let expect: Vec<usize> = (0..new_nb)
            .map(|i| b.bucket_len(i) + b.bucket_len(i + new_nb))
            .collect();

        b.coarsen();
        for (i, &want) in expect.iter().enumerate() {
            assert_eq!(b.bucket_len(i), want, "bucket {i} merge size");
        }
    }

    #[test]
    fn refine_then_coarsen_round_trips() {
        let sum = rand_sum::<2>(2500, 128, 0xB5);
        let h = Gf2Hash::<2>::new(128, 6, 0xC0DE);
        let mut b = BucketedSum::from_sum(&sum, h);
        let lens: Vec<usize> = (0..b.num_buckets()).map(|i| b.bucket_len(i)).collect();

        b.refine();
        b.coarsen();

        assert_eq!(b.num_buckets(), 64);
        let after: Vec<usize> = (0..b.num_buckets()).map(|i| b.bucket_len(i)).collect();
        assert_eq!(lens, after);
        b.assert_invariants();
        assert_same_sum(&sum, &b.into_sum());
    }

    #[test]
    fn repeated_refine_stays_consistent() {
        let sum = rand_sum::<2>(4000, 128, 0xB6);
        let h = Gf2Hash::<2>::new(128, 2, 0xC0DE);
        let mut b = BucketedSum::from_sum(&sum, h);
        for _ in 0..8 {
            b.refine();
            b.assert_invariants();
        }
        assert_eq!(b.num_buckets(), 1024);
        assert_same_sum(&sum, &b.into_sum());
    }

    #[test]
    fn refine_and_coarsen_on_an_empty_sum() {
        let h = Gf2Hash::<1>::new(64, 3, 0x7);
        let mut b = BucketedSum::<1>::empty(64, h);
        b.refine();
        b.assert_invariants();
        b.coarsen();
        b.assert_invariants();
        assert_eq!(b.num_buckets(), 8);
        assert_eq!(b.len(), 0);
    }

    // ---- rebucket policy ----

    #[test]
    fn rebucket_grows_toward_the_target() {
        // 8000 terms at target 64 wants ~125 buckets, i.e. 128.
        let sum = rand_sum::<2>(8000, 128, 0xD1);
        let h = Gf2Hash::<2>::new(128, 0, 0xC0DE);
        let mut b = BucketedSum::from_sum(&sum, h);
        b.rebucket(64, 1);
        b.assert_invariants();
        let mean = b.len() / b.num_buckets();
        assert!(
            mean <= 4 * 64,
            "mean {mean} still above the hysteresis band with {} buckets",
            b.num_buckets(),
        );
        assert_same_sum(&sum, &b.into_sum());
    }

    #[test]
    fn rebucket_shrinks_toward_the_target() {
        let sum = rand_sum::<1>(200, 64, 0xD2);
        let h = Gf2Hash::<1>::new(64, 10, 0xC0DE);
        let mut b = BucketedSum::from_sum(&sum, h);
        assert_eq!(b.num_buckets(), 1024);
        b.rebucket(256, 1);
        b.assert_invariants();
        assert!(
            b.num_buckets() < 1024,
            "expected coarsening, still {} buckets",
            b.num_buckets(),
        );
        assert_same_sum(&sum, &b.into_sum());
    }

    #[test]
    fn rebucket_is_hysteretic_not_twitchy() {
        // Inside the band [target/4, 4*target] nothing should move, so a layer
        // that nudges `n` up and down does not thrash rebucketing.
        let sum = rand_sum::<2>(6400, 128, 0xD3);
        let h = Gf2Hash::<2>::new(128, 6, 0xC0DE); // 64 buckets, mean 100
        let mut b = BucketedSum::from_sum(&sum, h);
        let before = b.num_buckets();
        b.rebucket(100, 1);
        assert_eq!(b.num_buckets(), before, "rebucket moved inside the band");
    }

    #[test]
    fn rebucket_respects_the_parallelism_floor() {
        let sum = rand_sum::<2>(4096, 128, 0xD4);
        let h = Gf2Hash::<2>::new(128, 0, 0xC0DE);
        let mut b = BucketedSum::from_sum(&sum, h);
        // Target far above n, but the floor still demands 32 buckets.
        b.rebucket(1 << 20, 32);
        b.assert_invariants();
        assert!(
            b.num_buckets() >= 32,
            "floor not respected: {} buckets",
            b.num_buckets(),
        );
    }

    #[test]
    fn rebucket_does_not_split_a_tiny_sum_to_hit_the_floor() {
        // With only 10 terms, splitting to 32 buckets is pure overhead; the
        // floor is gated on there being enough work to spread.
        let sum = rand_sum::<1>(10, 64, 0xD5);
        let h = Gf2Hash::<1>::new(64, 0, 0xC0DE);
        let mut b = BucketedSum::from_sum(&sum, h);
        b.rebucket(1024, 32);
        assert_eq!(b.num_buckets(), 1, "tiny sum was split anyway");
    }
}
