//! Storage and partition maintenance for [`PauliSum`] — per-bucket
//! structure-of-arrays columns under a [`Gf2Hash`] partition. There is one
//! bucketed representation; see ARCHITECTURE.md §Data-Model.
//!
//! The type itself is re-exported as [`crate::pauli_sum::PauliSum`], which is
//! its public home; this module owns the per-bucket column storage, the
//! bucket-count policy ([`desired_bits`] and the sizing constants), and the
//! merge helpers.

use num_complex::Complex64;
use rayon::prelude::*;

use super::hash::Gf2Hash;
use crate::pauli_string::PauliString;
#[cfg(test)]
use crate::pauli_sum::PauliAxis;
use crate::pauli_sum::{ProductBasis, ProductState};
use crate::stabilizer::StabilizerState;

/// Default seed for the partitioning hash.
///
/// Fixed so a `propagate` run is reproducible across processes. Exposed as a
/// constant rather than hidden so a caller who needs a different partition (to
/// rule out a pathological interaction with their Hamiltonian's structure) can
/// build their own [`Gf2Hash`].
pub const DEFAULT_HASH_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Bucket bits for a sum of `len` terms: the smallest `b` with
/// `len <= target << b`, clamped below by the parallelism floor.
///
/// Used to size the partition once at ingestion and at the start of
/// `propagate`, so the initial scatter hashes in a single pass rather than
/// being refined bit by bit.
/// [`PauliSum::rebucket`] tracks it afterwards, but only upward: it grows the
/// partition to `desired_bits` when that exceeds the current count, and
/// otherwise leaves the (larger) current count alone.
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
///
/// The floor in [`PauliSum::rebucket`] exists to give Rayon enough
/// independent tasks, but a task carrying almost nothing is pure overhead. Below
/// `min_buckets × MIN_TERMS_PER_TASK` total terms we would rather have few
/// buckets and let the small-`n` fallback to the whole-sum path handle it.
/// See ARCHITECTURE.md §Bucket-Policy for the sweep that set this value.
pub const MIN_TERMS_PER_TASK: usize = 64;

/// Default target terms per bucket.
///
/// Chosen so a bucket plus its gather scratch stays L2-resident: a term at
/// `W = 2` is `2·8·2 + 16 = 48` bytes, so 1024 terms is ~48 KB against 1 MiB of
/// L2 per core on the reference host. See ARCHITECTURE.md §Bucket-Policy for
/// the sweep that set this value.
pub const DEFAULT_TARGET_BUCKET_LEN: usize = 1024;

/// Default floor on the bucket count.
///
/// Fixed, not thread-derived: `128 = 4×32` is the value the 32-thread
/// reference host (ccqlin038) has always used, so this constant reproduces
/// every committed baseline and the committed `examples/output/*.csv`
/// trajectories exactly. Deriving the floor from `rayon::current_num_threads`
/// instead would make the bucket count `B` depend on how many threads happen
/// to be available; a fixed floor instead keeps `B` a deterministic function
/// of the sum's history alone (see ARCHITECTURE.md §Determinism), and 128
/// gives Rayon slack to load-balance at any realistic core count.
/// [`PauliSum::rebucket`] is grow-only, so `B` is the running max of
/// [`desired_bits`] over every layer seen so far, not the instantaneous term
/// count `n` alone; that history is fully determined by the circuit and the
/// starting sum, so `B` can no longer be recovered from `len()` at a single
/// point in time.
///
/// Must be `>= 16`: [`desired_bits`]'s "worth splitting" floor is non-monotone
/// below that (e.g. `min_buckets = 2` would split at 128 terms), and we want
/// "a sum of `<= 1024` terms gets a single bucket" to hold.
pub const DEFAULT_MIN_BUCKETS: usize = 128;

/// One bucket's structure-of-arrays columns.
///
/// Capacity is retained across layers, which is the point of owning per-bucket
/// columns rather than slicing one flat array: the steady state of a
/// propagation loop allocates nothing (ARCHITECTURE.md §Data-Model).
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
    pub(crate) fn clear(&mut self) {
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

/// Split one input bucket into its "low" (kept in place) and "high" (new
/// bucket at `b + old_nb`) halves under a hash that has just gained
/// `new_bit` as its top bit.
///
/// Shared by [`PauliSum::refine`]'s serial and parallel branches, so there is
/// exactly one implementation of the split. `b` is the *old* bucket index,
/// used only by the debug-only low-bits invariant check.
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
            // Compact in place: `keep <= i` always, so this never overwrites
            // an unread slot.
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
fn merge_two<const W: usize>(a: &BucketCols<W>, b: &BucketCols<W>) -> BucketCols<W> {
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
///
/// The counterpart of [`merge_two`] for operands that may share keys: within a
/// partition, equal keys are always in the same bucket pair, so a two-pointer
/// pass over one bucket of each operand sees every collision there is.
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
///
/// Replaces a `BinaryHeap`-based `B`-way merge, which measured 147 ms at
/// 10⁶ terms and 1024 buckets — 147 ns per element, because every pop does
/// `log2(B)` comparisons against 32-byte keys scattered across 1024 runs, and
/// the heap is inherently sequential.
///
/// The tree does the same `O(n log B)` comparisons but reads two sequential
/// streams at a time, and every pair within a round is independent, so the rounds
/// parallelize. It costs `log B` passes over the payload instead of one, which is
/// the trade: sequential bandwidth for random access, and it wins.
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

/// Weighted sum of Pauli operators, stored as structure-of-arrays columns
/// partitioned by a GF(2)-linear hash.
///
/// # Canonical order
///
/// Terms are ordered by **(bucket index `h(x, z)` ascending, then
/// lexicographic `(x, z)` within a bucket)** — the order [`Self::iter`] and
/// [`Self::to_arrays`] produce. This is a public promise, not an
/// implementation detail. A single-bucket sum has `h ≡ 0`, so its canonical
/// order *is* plain lexicographic `(x, z)` — and [`desired_bits`] gives every
/// sum of at most [`DEFAULT_TARGET_BUCKET_LEN`] terms a single bucket, so
/// small sums always come out lex-sorted.
///
/// # Invariant
///
/// Every term lies in `buckets[hash.bucket_of(term)]`, and each bucket is sorted
/// by the lexicographic `(x, z)` key with no duplicate keys. Because `h` is a
/// function, equal keys always share a bucket — so per-bucket dedup *implies*
/// global dedup, and no global sort is ever needed. The engine
/// ([`propagate`]) operates on the buckets directly; there is no separate
/// "flat" representation and nothing to convert in or out of.
///
/// [`propagate`]: crate::propagate
#[derive(Clone, Debug)]
pub struct PauliSum<const W: usize> {
    buckets: Vec<BucketCols<W>>,
    hash: Gf2Hash<W>,
    num_qubits: usize,
    len: usize,
}

impl<const W: usize> PauliSum<W> {
    /// Partition a globally key-sorted stream of terms.
    ///
    /// `O(n)`: one hash evaluation and one scatter per term. Because the input
    /// is globally key-sorted and terms are appended in input order, each
    /// bucket comes out sorted for free — order outside a bucket is
    /// irrelevant, and order within one is inherited.
    ///
    /// The caller owes the sortedness: `x`, `z`, `coeff` must be parallel
    /// columns ascending in `(x, z)` with no duplicate keys. Feeding it an
    /// unsorted stream silently breaks the per-bucket sort invariant.
    pub(crate) fn from_key_sorted(
        x: &[[u64; W]],
        z: &[[u64; W]],
        coeff: &[Complex64],
        hash: Gf2Hash<W>,
        num_qubits: usize,
    ) -> Self {
        let n = coeff.len();
        let nb = hash.num_buckets();

        // Hashing is the expensive part -- `b × 2W` AND+popcount-parity ops per
        // term, and the only place in the whole design where it happens at all
        // -- so it runs in parallel, and the counts come from the resulting
        // indices rather than from a second hashing pass.
        //
        // The scatter below stays sequential: buckets are separate allocations,
        // so a parallel scatter would need every thread to write into every
        // bucket. Measured, hashing dominates.
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
    ///
    /// The hash is the zero-bit prefix of the default seed's matrix, so the
    /// canonical order of anything built on top of this is plain lexicographic
    /// `(x, z)` until the sum grows past the [`desired_bits`] threshold and a
    /// caller (or the engine's `rebucket`) refines it.
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

    /// Test/oracle constructor: wrap globally key-sorted columns as a
    /// single-bucket sum (zero hash bits, default seed), whose canonical order
    /// is therefore exactly the given column order.
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

    /// Repartition under `hash`, keeping every term.
    ///
    /// Flattens to a globally key-sorted stream and rescatters. The scatter is
    /// what needs the global sort: a bucket inherits its order from the input
    /// stream, so a key-sorted stream produces key-sorted buckets — the same
    /// argument the crate-internal key-sorted constructor rests on.
    ///
    /// Prefer [`Self::refine`] / [`Self::coarsen`] when only the bucket *count*
    /// changes and the hash rows are the same; those are `O(n)` and never merge.
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
    ///
    /// Three cases, cheapest first: identical partition is a clone; the same
    /// hash rows at a different bucket count is a clone plus `O(n)` refine or
    /// coarsen steps; different rows falls back to [`Self::with_hash`], which
    /// pays the `O(n log B)` flatten.
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

    /// Expectation value `⟨ψ|O|ψ⟩` in a uniform single-qubit product state.
    ///
    /// For each [`ProductState`] there is exactly one single-qubit Pauli with
    /// expectation `1`; the others have expectation `0`. A Pauli string
    /// therefore contributes iff every one of its factors is either `I` or that
    /// Pauli, in which case it contributes its full coefficient — a masked scan
    /// over the key columns, run as a per-bucket parallel reduction. This is
    /// what lets a driver hold one sum across many [`propagate`](crate::propagate)
    /// calls and still read its observable each step.
    ///
    /// The uniform states are the special cases of [`ProductBasis`] with sign
    /// `+1` everywhere, so this is a thin wrapper over
    /// [`Self::expectation_product_basis`] — there is exactly one scan.
    ///
    /// Returns `Complex64` rather than `f64` because `self` need not be
    /// Hermitian; take `.re` when it is.
    ///
    /// # Summation order
    ///
    /// Partial sums are combined in bucket order — the canonical order — which
    /// is deterministic given the partition. Floating-point addition is not
    /// associative, so two partitions of the same terms can differ in the last
    /// bits — far below any physically meaningful tolerance, but do not expect
    /// bitwise equality across different hashes or bucket counts.
    pub fn expectation_product_state(&self, state: ProductState) -> Complex64 {
        self.expectation_product_basis(&ProductBasis::<W>::uniform(state))
    }

    /// Expectation value `⟨ψ|O|ψ⟩` in an arbitrary single-qubit product state.
    ///
    /// `basis` gives each qubit its own axis and sign; see [`ProductBasis`] for
    /// the per-word match condition and the sign rule this evaluates. A term
    /// contributes its coefficient — negated when an odd number of the sites
    /// in its support are `-1` eigenstates — iff every non-identity factor
    /// equals that qubit's axis exactly. Cost is one pass over the key columns
    /// as a per-bucket parallel reduction, `O(terms · W)`, with no expansion
    /// over basis states.
    ///
    /// Mask bits at qubit indices `>= num_qubits()` are irrelevant: a stored
    /// key never has one set there.
    ///
    /// Returns `Complex64` rather than `f64` because `self` need not be
    /// Hermitian; take `.re` when it is.
    ///
    /// # Summation order
    ///
    /// As in [`Self::expectation_product_state`] — partials are combined in
    /// bucket order, so two partitions of the same terms can differ in the
    /// last bits.
    pub fn expectation_product_basis(&self, basis: &ProductBasis<W>) -> Complex64 {
        self.buckets
            .par_iter()
            .map(|cols| {
                let mut acc = Complex64::new(0.0, 0.0);
                for i in 0..cols.len() {
                    // `mismatch` stays zero iff every word's non-identity
                    // sites carry exactly the local axis Pauli; `sign_bits`
                    // counts the `-1` eigenstates inside the support.
                    let mut mismatch = 0u64;
                    let mut sign_bits = 0u32;
                    for w in 0..W {
                        let x = cols.x[i][w];
                        let z = cols.z[i][w];
                        let sup = x | z;
                        mismatch |= (x ^ (sup & basis.ax_x[w])) | (z ^ (sup & basis.ax_z[w]));
                        sign_bits += (sup & basis.neg[w]).count_ones();
                    }
                    if mismatch == 0 {
                        if sign_bits & 1 == 0 {
                            acc += cols.coeff[i];
                        } else {
                            acc -= cols.coeff[i];
                        }
                    }
                }
                acc
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    }

    /// Expectation value `⟨ψ|O|ψ⟩` in a stabilizer state.
    ///
    /// `⟨ψ|P|ψ⟩` is `±1` when `±P` lies in the state's stabilizer group and `0`
    /// otherwise, so this is a *filter with a sign*, exactly like
    /// [`Self::expectation_product_basis`] — a term either contributes its
    /// coefficient, contributes its negation, or drops out. What widens is the
    /// admissible state: any stabilizer state (Bell, GHZ, cluster, a Clifford
    /// circuit's output) rather than only a product state. See
    /// [`StabilizerState`] for the membership test and the sign bookkeeping.
    ///
    /// Cost is `O(terms · n²/64)` word operations — `O(n)` pivot tests per
    /// term plus a `W`-word Pauli multiply per pivot hit — after the state's
    /// one-time `O(n³/64)` reduction, and never an expansion over basis
    /// states. That is `n` (= `num_qubits`) times more work per term than the
    /// product-state scan, so keep using
    /// [`Self::expectation_product_basis`] for states that factorize.
    ///
    /// Run as the same per-bucket parallel reduction
    /// [`Self::expectation_product_basis`] uses. Returns `Complex64` rather
    /// than `f64` because `self` need not be Hermitian; take `.re` when it is.
    ///
    /// # Summation order
    ///
    /// As in [`Self::expectation_product_state`] — partials are combined in
    /// bucket order, so two partitions of the same terms can differ in the
    /// last bits.
    ///
    /// # Panics
    ///
    /// Panics if `state.num_qubits()` differs from [`Self::num_qubits`]: the
    /// membership test would silently report `0` for every term supported
    /// outside the state's qubits.
    pub fn expectation_stabilizer(&self, state: &StabilizerState<W>) -> Complex64 {
        assert_eq!(
            self.num_qubits,
            state.num_qubits(),
            "PauliSum::expectation_stabilizer: num_qubits mismatch ({} vs {})",
            self.num_qubits,
            state.num_qubits(),
        );
        self.buckets
            .par_iter()
            .map(|cols| {
                let mut acc = Complex64::new(0.0, 0.0);
                for i in 0..cols.len() {
                    let key = PauliString::<W> {
                        x: cols.x[i],
                        z: cols.z[i],
                    };
                    match state.sign_of(&key) {
                        None => {}
                        Some(false) => acc += cols.coeff[i],
                        Some(true) => acc -= cols.coeff[i],
                    }
                }
                acc
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
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
    /// One `Gf2Hash::row_parity` evaluation per term against just the new
    /// high bit — `O(n)` total, not `O(n · bits)` — since a term's bucket
    /// under the old (shorter) hash prefix is exactly the bucket it is already
    /// in; only whether the new bit is set decides which half it lands in.
    /// The new index is `i` or `i + B` (the new hash bit is the *high* bit),
    /// and both halves inherit the source bucket's order, so nothing is
    /// re-sorted.
    ///
    /// Bucket pairs are independent — output bucket `b` and `b + old_nb`
    /// depend only on input bucket `b` — so above [`MIN_TERMS_PER_TASK`] ×
    /// [`DEFAULT_MIN_BUCKETS`] total terms (the same "worth splitting"
    /// threshold [`desired_bits`] uses) the per-bucket work runs across Rayon;
    /// below it the sequential loop avoids per-task overhead on a sum too
    /// small to benefit.
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
    ///
    /// A 2-way merge per pair via `merge_two` — no coefficient combining,
    /// since equal keys agree at every prefix length and were already in the
    /// same source bucket. Pairs are independent, so — same threshold and
    /// rationale as [`Self::refine`] — this is a parallel map above the
    /// worth-splitting threshold and a serial one below it.
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

    /// Bring the bucket count up to what [`desired_bits`] would choose for the
    /// current length — but never down.
    ///
    /// # Grow-only policy
    ///
    /// A sum's bucket count is monotone non-decreasing over its lifetime: this
    /// clamps the target to `self.hash.bits()`, so `rebucket` only ever refines,
    /// never coarsens. Measured: the alternative "track `desired_bits` exactly,
    /// up or down" policy made `rebucket` 74% of wall time on a sqrt-SWAP
    /// `GeneralUnitary2Q` workload and 45–46% at ≥8 threads on the TFIM Trotter
    /// workload: term counts oscillate every layer (fanout grows them,
    /// cancellation/truncation cuts them back), so a sum sitting near a
    /// power-of-two boundary in `desired_bits` would refine and coarsen on
    /// alternate layers, each an `O(n · bits)` serial pass.
    ///
    /// Keeping the larger partition instead is physically the right default:
    /// operator support only grows under propagation until truncation cuts it,
    /// and the cost of *not* coarsening back down is bounded — three empty `Vec`
    /// headers per surplus bucket, not a term-proportional cost. A caller that
    /// actually wants to shrink a sum (to reclaim memory between independent
    /// circuits, say) still can, explicitly, via [`Self::with_hash`] or
    /// [`Self::coarsen`]; `rebucket` itself no longer does it implicitly.
    ///
    /// Also keeps at least `min_buckets` buckets once there is enough work to
    /// spread, so the bucket-parallel decomposition has slack to load-balance.
    pub fn rebucket(&mut self, target: usize, min_buckets: usize) {
        debug_assert!(target > 0);
        let want = desired_bits(self.len, target, min_buckets).max(self.hash.bits());
        while self.hash.bits() < want {
            self.refine();
        }
    }

    /// Mutable access to the buckets, for layers that are applied in place.
    pub(crate) fn buckets_mut(&mut self) -> &mut [BucketCols<W>] {
        &mut self.buckets
    }

    /// Recompute the cached total after an in-place layer.
    pub(crate) fn recount(&mut self) {
        self.len = self.buckets.iter().map(|c| c.len()).sum();
    }

    /// Iterate every term in canonical order: buckets by ascending index, and
    /// within a bucket by ascending `(x, z)` key.
    ///
    /// This is *not* globally sorted — a bucket is a hash class, and the classes
    /// interleave arbitrarily in key order. It is nonetheless a total,
    /// deterministic order fixed by the partition, and it is the order
    /// [`Self::to_arrays`] concatenates in. Setup is `O(1)`: the iterator is
    /// lazy and borrows the bucket columns in place.
    pub fn iter(&self) -> impl Iterator<Item = (&[u64; W], &[u64; W], Complex64)> + '_ {
        self.buckets.iter().flat_map(|cols| {
            cols.x
                .iter()
                .zip(cols.z.iter())
                .zip(cols.coeff.iter())
                .map(|((x, z), c)| (x, z, *c))
        })
    }

    /// Copy every term out as three parallel columns, in the canonical order of
    /// [`Self::iter`].
    ///
    /// The columns are *not* globally key-sorted unless there is a single
    /// bucket. Sort the triples yourself if you need a canonical,
    /// partition-independent view.
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
    ///
    /// `O(b·W + log m)`: one hash evaluation to find the bucket, then a binary
    /// search of that bucket's `m` terms. Because `h` is a function, a key can
    /// only ever live in `h(x, z)`, so one bucket is the whole search space —
    /// this is the lookup that per-bucket dedup buys.
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
    ///
    /// The Pauli basis is orthogonal under the trace, so every non-identity
    /// term is traceless and only this one contributes.
    pub fn identity_coefficient(&self) -> Complex64 {
        let zero_key = [0u64; W];
        self.get(&zero_key, &zero_key)
            .unwrap_or(Complex64::new(0.0, 0.0))
    }

    /// Multiply every coefficient by `c` in place.
    ///
    /// Elementwise, so the partition is irrelevant to the result: the same
    /// terms scale to bitwise-identical coefficients whatever the bucket
    /// count. Parallel across buckets, which are separate allocations.
    pub fn scale(&mut self, c: Complex64) {
        self.buckets.par_iter_mut().for_each(|cols| {
            for coeff in cols.coeff.iter_mut() {
                *coeff *= c;
            }
        });
    }

    /// Keep only the terms for which `f(x, z, coeff)` is `true`.
    ///
    /// Order-preserving and in place, so the per-bucket sort and the
    /// no-duplicates invariant both survive; a term never changes bucket, so the
    /// hash invariant does too. `f` runs on every term, possibly on several
    /// threads at once, hence the `Sync` bound.
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

    /// Hilbert-Schmidt overlap `tr(self† · other) / 2ⁿ`, i.e. `Σ conj(aᵢ)·bᵢ`
    /// over the keys the two sums share.
    ///
    /// Equal keys always land in the same bucket under a shared hash, so a
    /// shared key can only be found by comparing bucket `i` against bucket `i`:
    /// this is `B` independent two-pointer merges, one per bucket, and no term
    /// is ever compared across buckets.
    ///
    /// # Summation order
    ///
    /// Each bucket's partial is accumulated in that bucket's key order, and the
    /// partials are then combined in ascending bucket index. That is
    /// deterministic given the partition; floating-point addition is not
    /// associative, so different partitions of the same operands agree to
    /// within rounding, not bit for bit. At a single bucket the accumulation
    /// is in plain key order.
    ///
    /// # Panics
    ///
    /// Panics unless the two sums share a partition: same hash rows (same seed
    /// and qubit count) *and* the same bucket count. Combining sums under
    /// different partitions is [`Self::add`]'s job; overlap does not realign.
    ///
    /// Note that under the grow-only [`Self::rebucket`] policy, two sums can
    /// have equal `len()` but different bucket counts if they were
    /// grown to different high-water marks along the way (e.g. one was built
    /// directly at its final size, the other passed through a larger
    /// intermediate sum and never coarsened back down). Align them with
    /// [`Self::with_hash`] before calling if that is a possibility.
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
    ///
    /// **The left partition wins**: the result is partitioned exactly as `self`
    /// is, and `other` is realigned onto that partition first (a clone plus
    /// refine/coarsen when the hash rows match, a full [`Self::with_hash`] when
    /// they do not). `self` and `other` are both left untouched.
    ///
    /// Once aligned, equal keys are guaranteed to sit in the same bucket index
    /// on both sides, so this is `B` independent two-pointer merges — no global
    /// sort, and no cross-bucket comparison. Terms whose coefficients sum to
    /// exactly `0+0i` are dropped.
    ///
    /// # Summation order
    ///
    /// Each surviving coefficient is a single `self + other` addition, so it is
    /// bit-identical to what the flat merge would produce. Only *derived*
    /// quantities that accumulate across terms — [`Self::overlap`],
    /// [`Self::expectation_product_state`] — see the bucket-order effect, since
    /// their partials are combined in bucket order rather than key order.
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

    /// Assert the structural invariant. Debug builds and test builds only.
    ///
    /// Checks that every term is in its hash bucket, that each bucket is
    /// strictly ascending in `(x, z)` (so sorted *and* duplicate-free), that
    /// every key is within `num_qubits`, and that the cached length agrees.
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
mod tests {
    #[test]
    fn bucketed_expectation_agrees_with_the_flat_version() {
        // Not bitwise: partials are combined in bucket order, not global sorted
        // order, and float addition is not associative. The tolerance below is
        // ~1e5 times looser than the observed difference and ~1e5 times tighter
        // than anything physically meaningful.
        for &weight in &[2usize, 4] {
            let sum = rand_low_weight_sum::<2>(20_000, 100, weight, 0xE1 + weight as u64);
            let want = sum.expectation_product_state(ProductState::XPlus);
            for bits in [0u8, 3, 7, 11] {
                let h = Gf2Hash::<2>::new(100, bits, 0xE2);
                let b = sum.clone().with_hash(h);
                let got = b.expectation_product_state(ProductState::XPlus);
                assert!(
                    (got - want).norm() < 1e-9,
                    "weight={weight} bits={bits}: {got} vs {want}",
                );
            }
        }
    }

    #[test]
    fn bucketed_expectation_covers_all_three_states() {
        let sum = rand_sum::<1>(5000, 20, 0xE3);
        let h = Gf2Hash::<1>::new(20, 5, 0xE4);
        let b = sum.clone().with_hash(h);
        for state in [
            ProductState::XPlus,
            ProductState::YPlus,
            ProductState::ZPlus,
        ] {
            let got = b.expectation_product_state(state);
            let want = sum.expectation_product_state(state);
            assert!((got - want).norm() < 1e-9, "{state:?}: {got} vs {want}");
        }
    }

    #[test]
    fn bucketed_expectation_of_an_empty_sum_is_zero() {
        let h = Gf2Hash::<1>::new(8, 3, 0xE5);
        let b = PauliSum::<1>::empty_with_hash(8, h);
        assert!(
            b.expectation_product_state(ProductState::XPlus)
                .norm()
                .abs()
                < 1e-15
        );
    }

    use super::*;
    use crate::accumulator::BuildAccumulator;
    use crate::phase::Phase;
    // `Xs64` and `rand_sum` are the canonical fixtures from
    // `crate::test_support` — this module's copies were byte-identical.
    use crate::test_support::{low_weight_sum, rand_sum, Xs64};

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

    /// Same multiset of terms, coefficients bitwise — partition forgotten.
    fn assert_same_sum<const W: usize>(a: &PauliSum<W>, b: &PauliSum<W>) {
        assert_eq!(a.len(), b.len(), "length");
        assert_eq!(a.num_qubits(), b.num_qubits(), "num_qubits");
        let ta = {
            let mut v: Vec<([u64; W], [u64; W], Complex64)> =
                a.iter().map(|(x, z, c)| (*x, *z, c)).collect();
            v.sort_unstable_by_key(|&(x, z, _)| (x, z));
            v
        };
        let tb = {
            let mut v: Vec<([u64; W], [u64; W], Complex64)> =
                b.iter().map(|(x, z, c)| (*x, *z, c)).collect();
            v.sort_unstable_by_key(|&(x, z, _)| (x, z));
            v
        };
        assert_eq!(ta, tb, "terms");
    }

    // ---- round trip ----

    #[test]
    fn round_trip_is_bitwise_identical_w1() {
        let sum = rand_sum::<1>(5000, 64, 0xA1);
        let h = Gf2Hash::<1>::new(64, 7, 0xBEEF);
        let bucketed = sum.clone().with_hash(h);
        bucketed.assert_invariants();
        let back = bucketed;
        back.assert_invariants();
        assert_same_sum(&sum, &back);
    }

    #[test]
    fn round_trip_is_bitwise_identical_w2() {
        let sum = rand_sum::<2>(5000, 128, 0xA2);
        let h = Gf2Hash::<2>::new(128, 9, 0xBEEF);
        let bucketed = sum.clone().with_hash(h);
        bucketed.assert_invariants();
        let back = bucketed;
        back.assert_invariants();
        assert_same_sum(&sum, &back);
    }

    #[test]
    fn round_trip_on_low_weight_input() {
        let sum = rand_low_weight_sum::<2>(4000, 100, 4, 0xA3);
        let h = Gf2Hash::<2>::new(100, 8, 0xBEEF);
        let bucketed = sum.clone().with_hash(h);
        bucketed.assert_invariants();
        assert_same_sum(&sum, &bucketed);
    }

    #[test]
    fn round_trip_at_a_mid_word_qubit_count() {
        // 70 qubits in W=2: word 1 is only partly live.
        let sum = rand_sum::<2>(2000, 70, 0xA4);
        let h = Gf2Hash::<2>::new(70, 6, 0xBEEF);
        let bucketed = sum.clone().with_hash(h);
        bucketed.assert_invariants();
        assert_same_sum(&sum, &bucketed);
    }

    #[test]
    fn round_trip_empty_sum() {
        let sum = PauliSum::<1>::empty(64);
        let h = Gf2Hash::<1>::new(64, 5, 0x1);
        let bucketed = sum.clone().with_hash(h);
        bucketed.assert_invariants();
        assert!(bucketed.is_empty());
        assert_eq!(bucketed.len(), 0);
        assert_same_sum(&sum, &bucketed);
    }

    #[test]
    fn round_trip_single_term() {
        let sum = rand_sum::<1>(1, 64, 0xA5);
        assert_eq!(sum.len(), 1);
        let h = Gf2Hash::<1>::new(64, 8, 0x2);
        let bucketed = sum.clone().with_hash(h);
        bucketed.assert_invariants();
        assert_same_sum(&sum, &bucketed);
    }

    #[test]
    fn round_trip_with_a_single_bucket() {
        // bits = 0: everything in bucket 0, so the canonical order is plain
        // lex and the scatter must already have produced a sorted bucket.
        let sum = rand_sum::<1>(2000, 64, 0xA6);
        let h = Gf2Hash::<1>::new(64, 0, 0x3);
        let bucketed = sum.clone().with_hash(h);
        assert_eq!(bucketed.num_buckets(), 1);
        bucketed.assert_invariants();
        assert_same_sum(&sum, &bucketed);
    }

    #[test]
    fn round_trip_with_more_buckets_than_terms() {
        let sum = rand_sum::<1>(50, 64, 0xA7);
        let h = Gf2Hash::<1>::new(64, 12, 0x4);
        let bucketed = sum.clone().with_hash(h);
        assert_eq!(bucketed.num_buckets(), 4096);
        bucketed.assert_invariants();
        assert_same_sum(&sum, &bucketed);
    }

    #[test]
    fn empty_constructor_matches_empty_sum() {
        let h = Gf2Hash::<2>::new(128, 6, 0x5);
        let b = PauliSum::<2>::empty_with_hash(128, h);
        b.assert_invariants();
        assert_eq!(b.num_buckets(), 64);
        assert_eq!(b.len(), 0);
    }

    // ---- refine / coarsen ----

    #[test]
    fn refine_doubles_buckets_and_preserves_content() {
        let sum = rand_sum::<2>(3000, 128, 0xB1);
        let h = Gf2Hash::<2>::new(128, 6, 0xC0DE);
        let mut b = sum.clone().with_hash(h);
        let before_len = b.len();

        b.refine();
        assert_eq!(b.num_buckets(), 128);
        assert_eq!(b.len(), before_len);
        // The invariant check is the real assertion: it verifies every term is
        // in its *new* hash bucket and that each bucket is still sorted.
        b.assert_invariants();
        assert_same_sum(&sum, &b);
    }

    #[test]
    fn refine_splits_each_bucket_into_the_pair_i_and_i_plus_b() {
        let sum = rand_sum::<1>(3000, 64, 0xB2);
        let h = Gf2Hash::<1>::new(64, 5, 0xC0DE);
        let mut b = sum.clone().with_hash(h);
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
        let mut b = sum.clone().with_hash(h);

        b.coarsen();
        assert_eq!(b.num_buckets(), 64);
        b.assert_invariants();
        assert_same_sum(&sum, &b);
    }

    #[test]
    fn coarsen_merges_the_pair_i_and_i_plus_new_b() {
        let sum = rand_sum::<1>(3000, 64, 0xB4);
        let h = Gf2Hash::<1>::new(64, 6, 0xC0DE);
        let mut b = sum.clone().with_hash(h);
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
        let mut b = sum.clone().with_hash(h);
        let lens: Vec<usize> = (0..b.num_buckets()).map(|i| b.bucket_len(i)).collect();

        b.refine();
        b.coarsen();

        assert_eq!(b.num_buckets(), 64);
        let after: Vec<usize> = (0..b.num_buckets()).map(|i| b.bucket_len(i)).collect();
        assert_eq!(lens, after);
        b.assert_invariants();
        assert_same_sum(&sum, &b);
    }

    #[test]
    fn repeated_refine_stays_consistent() {
        let sum = rand_sum::<2>(4000, 128, 0xB6);
        let h = Gf2Hash::<2>::new(128, 2, 0xC0DE);
        let mut b = sum.clone().with_hash(h);
        for _ in 0..8 {
            b.refine();
            b.assert_invariants();
        }
        assert_eq!(b.num_buckets(), 1024);
        assert_same_sum(&sum, &b);
    }

    #[test]
    fn refine_and_coarsen_take_the_parallel_path_above_the_threshold() {
        // Above DEFAULT_MIN_BUCKETS * MIN_TERMS_PER_TASK (8192) terms,
        // refine/coarsen run their per-bucket work across Rayon instead of
        // serially. Exercise that branch directly and check it produces the
        // same invariants and content as the serial path.
        let n = 10_000;
        assert!(n >= DEFAULT_MIN_BUCKETS * MIN_TERMS_PER_TASK);
        let sum = rand_sum::<2>(n, 128, 0xB7);
        let h = Gf2Hash::<2>::new(128, 2, 0xC0DE);
        let mut b = sum.clone().with_hash(h);

        b.refine();
        b.assert_invariants();
        b.refine();
        b.assert_invariants();
        assert_eq!(b.num_buckets(), 16);
        assert_same_sum(&sum, &b);

        b.coarsen();
        b.assert_invariants();
        b.coarsen();
        b.assert_invariants();
        assert_eq!(b.num_buckets(), 4);
        assert_same_sum(&sum, &b);
    }

    #[test]
    fn refine_and_coarsen_on_an_empty_sum() {
        let h = Gf2Hash::<1>::new(64, 3, 0x7);
        let mut b = PauliSum::<1>::empty_with_hash(64, h);
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
        let mut b = sum.clone().with_hash(h);
        b.rebucket(64, 1);
        b.assert_invariants();
        let mean = b.len() / b.num_buckets();
        assert!(
            mean <= 4 * 64,
            "mean {mean} still above the hysteresis band with {} buckets",
            b.num_buckets(),
        );
        assert_same_sum(&sum, &b);
    }

    #[test]
    fn rebucket_never_shrinks() {
        // rebucket only ever grows. 200 terms at target 256 wants far fewer
        // than 1024 buckets, but starting at 1024 must stay at 1024.
        let sum = rand_sum::<1>(200, 64, 0xD2);
        let h = Gf2Hash::<1>::new(64, 10, 0xC0DE);
        let mut b = sum.clone().with_hash(h);
        assert_eq!(b.num_buckets(), 1024);
        b.rebucket(256, 1);
        b.assert_invariants();
        assert_eq!(
            b.num_buckets(),
            1024,
            "rebucket shrank from 1024 to {} buckets",
            b.num_buckets(),
        );
        assert_same_sum(&sum, &b);
    }

    #[test]
    fn rebucket_is_a_no_op_when_already_at_the_target() {
        // 6400 terms at target 100 wants exactly 64 buckets, so nothing moves.
        // A hysteresis band that parks the steady state up to 4x above target
        // was tried and measured ~10% slower on the Ising quench (see
        // ARCHITECTURE.md §Bucket-Policy) — this test pins the no-hysteresis
        // behavior.
        let sum = rand_sum::<2>(6400, 128, 0xD3);
        let h = Gf2Hash::<2>::new(128, 6, 0xC0DE); // 64 buckets, mean 100
        let mut b = sum.clone().with_hash(h);
        let before = b.num_buckets();
        b.rebucket(100, 1);
        assert_eq!(
            b.num_buckets(),
            before,
            "rebucket moved when already on target"
        );
    }

    #[test]
    fn rebucket_lands_on_desired_bits_or_stays_at_the_high_water_mark() {
        // `rebucket` only grows, so it converges on exactly what
        // `desired_bits` would have chosen only when the starting partition is
        // at or below that — otherwise (e.g. `start = 12`, above every `want`
        // in this table) the starting bit count is the high-water mark and
        // survives unchanged. Both are `want.max(start)`.
        for &n in &[500usize, 6400, 60_000] {
            let sum = rand_sum::<2>(n, 128, 0xD9 + n as u64);
            let want = desired_bits(sum.len(), 256, 8);
            for start in [0u8, 3, 12] {
                let h = Gf2Hash::<2>::new(128, start, 0xC0DE);
                let mut b = sum.clone().with_hash(h);
                b.rebucket(256, 8);
                assert_eq!(
                    b.hash().bits(),
                    want.max(start),
                    "n={n} start={start}: expected max(want={want}, start={start})",
                );
                b.assert_invariants();
            }
        }
    }

    #[test]
    fn rebucket_keeps_the_high_water_mark_after_len_shrinks() {
        // Grow to a high bucket count from a large sum, then shrink the term
        // count sharply (as truncation/cancellation would between layers) and
        // rebucket again: the grow-only policy says the bucket count is a
        // high-water mark, so it must not follow the length back down.
        let sum = rand_sum::<2>(60_000, 128, 0xDA1);
        let h = Gf2Hash::<2>::new(128, 0, 0xC0DE);
        let mut b = sum.with_hash(h);
        b.rebucket(256, 8);
        let grown_bits = b.hash().bits();
        assert!(
            grown_bits > 0,
            "sanity: rebucket should have grown from 0 bits"
        );

        // Shrink the term count sharply, well below what would justify
        // `grown_bits` under `desired_bits`.
        b.retain(|_, _, c| c.re > 0.995);
        b.assert_invariants();
        assert!(
            b.len() < 512,
            "sanity: shrink did not reduce len enough ({})",
            b.len(),
        );

        b.rebucket(256, 8);
        assert_eq!(
            b.hash().bits(),
            grown_bits,
            "rebucket shrank the high-water mark from {grown_bits} to {}",
            b.hash().bits(),
        );
        b.assert_invariants();
    }

    #[test]
    fn rebucket_respects_the_parallelism_floor() {
        let sum = rand_sum::<2>(4096, 128, 0xD4);
        let h = Gf2Hash::<2>::new(128, 0, 0xC0DE);
        let mut b = sum.clone().with_hash(h);
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
        let mut b = sum.clone().with_hash(h);
        b.rebucket(1024, 32);
        assert_eq!(b.num_buckets(), 1, "tiny sum was split anyway");
    }

    // ---- the canonical-order contract ----

    #[test]
    fn canonical_order_is_bucket_then_key() {
        let sum = rand_sum::<1>(3000, 64, 0xC0);
        for bits in [1u8, 4, 8] {
            let b = sum.clone().with_hash(Gf2Hash::<1>::new(64, bits, 0xC1));
            let h = b.hash().clone();
            let mut prev: Option<(u32, [u64; 1], [u64; 1])> = None;
            for (x, z, _) in b.iter() {
                let bucket = h.bucket_of(x, z);
                if let Some((pb, px, pz)) = prev {
                    assert!(
                        (pb, (px, pz)) < (bucket, (*x, *z)),
                        "bits={bits}: (bucket, key) not strictly ascending",
                    );
                }
                prev = Some((bucket, *x, *z));
            }
        }
    }

    #[test]
    fn single_bucket_sum_is_plain_lex_sorted() {
        // Below the split threshold the canonical order IS lex order — the
        // property every small-sum positional expectation in the crate rests on.
        let sum = rand_sum::<1>(1000, 64, 0xC2);
        assert_eq!(sum.num_buckets(), 1);
        let (x, z, _) = sum.to_arrays();
        for i in 1..x.len() {
            assert!(
                (x[i - 1], z[i - 1]) < (x[i], z[i]),
                "single-bucket sum not lex-sorted at {i}",
            );
        }
    }

    // ---- S1: canonical iteration / export ----

    /// The canonical order as a plain vector, read out of `bucket()` alone.
    fn canonical_triples<const W: usize>(b: &PauliSum<W>) -> Vec<([u64; W], [u64; W], Complex64)> {
        let mut out = Vec::with_capacity(b.len());
        for i in 0..b.num_buckets() {
            let (x, z, c) = b.bucket(i);
            for k in 0..c.len() {
                out.push((x[k], z[k], c[k]));
            }
        }
        out
    }

    #[test]
    fn iter_yields_bucket_then_key_order() {
        let sum = rand_sum::<2>(3000, 128, 0xF1);
        let h = Gf2Hash::<2>::new(128, 6, 0xF2);
        let b = sum.clone().with_hash(h);

        let got: Vec<([u64; 2], [u64; 2], Complex64)> =
            b.iter().map(|(x, z, c)| (*x, *z, c)).collect();
        assert_eq!(got.len(), b.len());
        assert_eq!(got, canonical_triples(&b));

        // Within a bucket the keys ascend; the boundaries are exactly the
        // bucket lengths, so the concatenation is not globally sorted.
        let mut start = 0usize;
        for i in 0..b.num_buckets() {
            let n = b.bucket_len(i);
            for k in start + 1..start + n {
                assert!(
                    (got[k - 1].0, got[k - 1].1) < (got[k].0, got[k].1),
                    "bucket {i} not ascending at {k}",
                );
            }
            start += n;
        }
    }

    #[test]
    fn to_arrays_concatenates_buckets_in_index_order() {
        let sum = rand_sum::<1>(2500, 64, 0xF3);
        let h = Gf2Hash::<1>::new(64, 5, 0xF4);
        let b = sum.clone().with_hash(h);

        let (x, z, c) = b.to_arrays();
        let want = canonical_triples(&b);
        assert_eq!(x.len(), want.len());
        for (k, (wx, wz, wc)) in want.iter().enumerate() {
            assert_eq!(x[k], *wx, "x at {k}");
            assert_eq!(z[k], *wz, "z at {k}");
            assert_eq!(c[k], *wc, "coeff at {k}");
        }
    }

    #[test]
    fn single_bucket_to_arrays_is_the_key_sorted_order() {
        // bits = 0 collapses the bucket order onto the global sorted order, so
        // the export and the merge must agree bit for bit.
        let sum = rand_sum::<2>(2000, 128, 0xF5);
        let h = Gf2Hash::<2>::new(128, 0, 0xF6);
        let b = sum.clone().with_hash(h);
        let (x, z, c) = b.to_arrays();
        let want = sorted_triples(&b);
        for (i, &(wx, wz, wc)) in want.iter().enumerate() {
            assert_eq!(x[i], wx, "x column at {i}");
            assert_eq!(z[i], wz, "z column at {i}");
            assert_eq!(c[i], wc, "coeff column at {i}");
        }
    }

    // ---- S2: keyed lookup ----

    #[test]
    fn get_hits_and_misses_across_bucket_counts() {
        let sum = rand_sum::<1>(2000, 64, 0xF7);
        // Keys that are definitely absent: the sum has 2000 of 2^128 keys, but
        // rather than gamble, take misses from a disjoint second draw and skip
        // any that happen to collide.
        let other = rand_sum::<1>(2000, 64, 0xF8);

        for bits in [0u8, 3, 7] {
            let h = Gf2Hash::<1>::new(64, bits, 0xF9);
            let b = sum.clone().with_hash(h);
            for (i, (x, z, c)) in sum.iter().enumerate() {
                assert_eq!(
                    b.get(x, z),
                    Some(c),
                    "bits={bits}: miss on present term {i}"
                );
            }
            let mut misses = 0usize;
            for (i, (x, z, _)) in other.iter().enumerate() {
                if sum.get(x, z).is_some() {
                    continue;
                }
                misses += 1;
                assert_eq!(b.get(x, z), None, "bits={bits}: hit on absent term {i}",);
            }
            assert!(misses > 1000, "bits={bits}: only {misses} absent probes");
        }
    }

    #[test]
    fn get_w2_word_boundary() {
        // Keys live entirely in word 1, so a lookup that only compared word 0
        // would confuse them.
        let mut acc = BuildAccumulator::<2>::new(128);
        for q in [64u32, 65, 100, 127] {
            acc.add_term(
                PauliString::<2>::x(q),
                Phase::ONE,
                Complex64::new(q as f64, 0.0),
            );
            acc.add_term(
                PauliString::<2>::z(q),
                Phase::ONE,
                Complex64::new(0.0, q as f64),
            );
        }
        let sum = acc.finalize();
        for bits in [0u8, 4] {
            let h = Gf2Hash::<2>::new(128, bits, 0xFA);
            let b = sum.clone().with_hash(h);
            for q in [64u32, 65, 100, 127] {
                let px = PauliString::<2>::x(q);
                let pz = PauliString::<2>::z(q);
                assert_eq!(
                    b.get(&px.x, &px.z),
                    Some(Complex64::new(q as f64, 0.0)),
                    "bits={bits} X{q}",
                );
                assert_eq!(
                    b.get(&pz.x, &pz.z),
                    Some(Complex64::new(0.0, q as f64)),
                    "bits={bits} Z{q}",
                );
            }
            // X on qubit 63 is a distinct key in word 0 and is absent.
            let absent = PauliString::<2>::x(63);
            assert_eq!(b.get(&absent.x, &absent.z), None);
        }
    }

    #[test]
    fn get_agrees_with_a_map_model() {
        use std::collections::BTreeMap;
        let sum = rand_low_weight_sum::<2>(3000, 100, 3, 0xFB);
        let probes = rand_low_weight_sum::<2>(3000, 100, 3, 0xFC);
        let h = Gf2Hash::<2>::new(100, 6, 0xFD);
        let b = sum.clone().with_hash(h);
        let model: BTreeMap<([u64; 2], [u64; 2]), Complex64> =
            sum.iter().map(|(x, z, c)| ((*x, *z), c)).collect();
        for (i, (x, z, _)) in probes.iter().enumerate() {
            assert_eq!(b.get(x, z), model.get(&(*x, *z)).copied(), "probe {i}");
        }
    }

    // ---- S3: per-bucket mutators ----

    #[test]
    fn scale_matches_flat_bitwise() {
        // Scaling is elementwise, so per-bucket and flat orders cannot diverge:
        // the comparison is exact, not toleranced.
        for bits in [0u8, 5] {
            let sum = rand_sum::<2>(3000, 128, 0x101);
            let h = Gf2Hash::<2>::new(128, bits, 0x102);
            let mut b = sum.clone().with_hash(h);
            let mut flat = sum.clone();
            let c = Complex64::new(-0.75, 1.25);
            b.scale(c);
            flat.scale(c);
            b.assert_invariants();
            assert_eq!(b.len(), flat.len());
            assert_same_sum(&flat, &b);
        }
    }

    #[test]
    fn retain_filters_in_place_and_keeps_invariants() {
        for bits in [0u8, 6] {
            let sum = rand_sum::<2>(4000, 128, 0x106);
            let h = Gf2Hash::<2>::new(128, bits, 0x107);
            let mut b = sum.clone().with_hash(h);
            // A predicate that reads the key as well as the coefficient, so a
            // key/coefficient column desync would show up.
            let keep = |x: &[u64; 2], _z: &[u64; 2], c: Complex64| x[0] & 1 == 0 && c.re > 0.0;
            b.retain(keep);
            b.assert_invariants();

            let mut want_x = Vec::new();
            let mut want_z = Vec::new();
            let mut want_c = Vec::new();
            for (x, z, c) in sorted_triples(&sum) {
                if keep(&x, &z, c) {
                    want_x.push(x);
                    want_z.push(z);
                    want_c.push(c);
                }
            }
            assert!(
                !want_x.is_empty(),
                "predicate kept nothing; test is vacuous"
            );
            assert_eq!(b.len(), want_x.len(), "bits={bits} length");
            let got = sorted_triples(&b);
            for (i, t) in got.iter().enumerate() {
                assert_eq!(t.0, want_x[i], "bits={bits} x at {i}");
                assert_eq!(t.1, want_z[i], "bits={bits} z at {i}");
                assert_eq!(t.2, want_c[i], "bits={bits} coeff at {i}");
            }
        }
    }

    // ---- S4: overlap ----

    /// Reference overlap: two-pointer over globally key-sorted triples — the
    /// accumulation order a single-bucket sum uses.
    fn flat_overlap<const W: usize>(a: &PauliSum<W>, b: &PauliSum<W>) -> Complex64 {
        let ta = sorted_triples(a);
        let tb = sorted_triples(b);
        let mut acc = Complex64::new(0.0, 0.0);
        let (mut i, mut j) = (0usize, 0usize);
        while i < ta.len() && j < tb.len() {
            match (ta[i].0, ta[i].1).cmp(&(tb[j].0, tb[j].1)) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    acc += ta[i].2.conj() * tb[j].2;
                    i += 1;
                    j += 1;
                }
            }
        }
        acc
    }

    #[test]
    fn overlap_single_bucket_is_bitwise_flat() {
        // One bucket means one two-pointer pass over globally sorted columns —
        // the same additions in the same order as the flat version, so equality
        // is exact.
        let a = rand_low_weight_sum::<2>(3000, 100, 3, 0x201);
        let b = rand_low_weight_sum::<2>(3000, 100, 3, 0x202);
        let h = Gf2Hash::<2>::new(100, 0, 0x203);
        let ba = a.clone().with_hash(h.clone());
        let bb = b.clone().with_hash(h);
        assert_eq!(ba.overlap(&bb), flat_overlap(&a, &b));
        // Self-overlap is the squared norm.
        assert_eq!(ba.overlap(&ba), flat_overlap(&a, &a));
    }

    #[test]
    fn overlap_matches_flat_within_tolerance_across_bits() {
        // Not bitwise past one bucket: partials are combined in bucket order.
        let a = rand_low_weight_sum::<2>(4000, 100, 3, 0x204);
        let b = rand_low_weight_sum::<2>(4000, 100, 3, 0x205);
        let want = flat_overlap(&a, &b);
        for bits in [0u8, 3, 6] {
            let h = Gf2Hash::<2>::new(100, bits, 0x206);
            let ba = a.clone().with_hash(h.clone());
            let bb = b.clone().with_hash(h);
            let got = ba.overlap(&bb);
            if bits == 0 {
                assert_eq!(got, want, "bits=0 must be bitwise");
            }
            // Relative, not absolute: the overlap here is ~1.4e3, so a handful
            // of ulps is ~4e-12 in absolute terms. The reordering error is
            // bounded by the accumulated magnitude, which is what a relative
            // bound tracks.
            assert!(
                (got - want).norm() <= 1e-12 * want.norm(),
                "bits={bits}: {got} vs {want}",
            );
        }
        assert!(want.norm() > 0.0, "operands share no keys; test is vacuous");
    }

    #[test]
    fn overlap_with_an_empty_operand_is_zero() {
        let a = rand_sum::<1>(500, 64, 0x207);
        let h = Gf2Hash::<1>::new(64, 4, 0x208);
        let ba = a.clone().with_hash(h.clone());
        let empty = PauliSum::<1>::empty_with_hash(64, h);
        assert_eq!(ba.overlap(&empty), Complex64::new(0.0, 0.0));
        assert_eq!(empty.overlap(&ba), Complex64::new(0.0, 0.0));
    }

    #[test]
    #[should_panic(expected = "bucket count mismatch")]
    fn overlap_rejects_a_different_bucket_count() {
        let a = rand_sum::<1>(200, 64, 0x209);
        let ba = a.clone().with_hash(Gf2Hash::<1>::new(64, 3, 0x20A));
        let bb = a.clone().with_hash(Gf2Hash::<1>::new(64, 4, 0x20A));
        let _ = ba.overlap(&bb);
    }

    #[test]
    #[should_panic(expected = "hash mismatch")]
    fn overlap_rejects_a_different_hash() {
        let a = rand_sum::<1>(200, 64, 0x20B);
        let ba = a.clone().with_hash(Gf2Hash::<1>::new(64, 3, 0x20C));
        let bb = a.clone().with_hash(Gf2Hash::<1>::new(64, 3, 0x20D));
        let _ = ba.overlap(&bb);
    }

    // ---- S5: flatten / repartition ----

    /// The canonical order, sorted — i.e. the multiset of terms, partition
    /// forgotten.
    fn sorted_triples<const W: usize>(b: &PauliSum<W>) -> Vec<([u64; W], [u64; W], Complex64)> {
        let mut v = canonical_triples(b);
        v.sort_by(|p, q| (p.0, p.1).cmp(&(q.0, q.1)));
        v
    }

    #[test]
    fn with_hash_round_trips_terms_bitwise() {
        let sum = rand_low_weight_sum::<2>(4000, 100, 3, 0x303);
        let ha = Gf2Hash::<2>::new(100, 5, 0x304);
        let hb = Gf2Hash::<2>::new(100, 8, 0x305);
        let a = sum.clone().with_hash(ha.clone());
        let before = canonical_triples(&a);

        let moved = a.with_hash(hb);
        moved.assert_invariants();
        let back = moved.with_hash(ha);
        back.assert_invariants();

        assert_eq!(canonical_triples(&back), before);
    }

    #[test]
    fn with_hash_only_changes_partition() {
        let sum = rand_sum::<1>(3000, 64, 0x306);
        let a = sum.clone().with_hash(Gf2Hash::<1>::new(64, 4, 0x307));
        let want = sorted_triples(&a);

        for (bits, seed) in [(0u8, 0x308u64), (9, 0x309), (4, 0x30A)] {
            let moved = a.clone().with_hash(Gf2Hash::<1>::new(64, bits, seed));
            moved.assert_invariants();
            assert_eq!(moved.len(), a.len(), "bits={bits} seed={seed} length");
            assert_eq!(moved.num_buckets(), 1usize << bits);
            assert_eq!(
                sorted_triples(&moved),
                want,
                "bits={bits} seed={seed}: term multiset changed",
            );
        }
    }

    #[test]
    fn with_hash_rejects_a_qubit_count_mismatch() {
        let sum = rand_sum::<2>(100, 100, 0x30B);
        let a = sum.clone().with_hash(Gf2Hash::<2>::new(100, 3, 0x30C));
        let err = std::panic::catch_unwind(move || a.with_hash(Gf2Hash::<2>::new(128, 3, 0x30C)));
        assert!(err.is_err(), "expected a num_qubits mismatch panic");
    }

    #[test]
    fn align_same_rows_via_refine_coarsen() {
        // Same rows: aligning must land on exactly the partition a scatter
        // would have built under the target hash, both up and down.
        let sum = rand_low_weight_sum::<2>(4000, 100, 4, 0x30D);
        let seed = 0x30E;
        for &(from, to) in &[(5u8, 9u8), (9, 5), (6, 6), (0, 7), (7, 0)] {
            let a = sum.clone().with_hash(Gf2Hash::<2>::new(100, from, seed));
            let target = Gf2Hash::<2>::new(100, to, seed);
            let got = a.align_to(&target);
            got.assert_invariants();
            assert_eq!(got.hash().bits(), to, "from={from} to={to}");
            let want = sum.clone().with_hash(target);
            assert_eq!(
                canonical_triples(&got),
                canonical_triples(&want),
                "from={from} to={to}: partition differs from a direct build",
            );
        }
    }

    #[test]
    fn align_different_rows_goes_through_with_hash() {
        let sum = rand_low_weight_sum::<2>(3000, 100, 3, 0x30F);
        let a = sum.clone().with_hash(Gf2Hash::<2>::new(100, 6, 0x310));
        let target = Gf2Hash::<2>::new(100, 4, 0x311);
        let got = a.align_to(&target);
        got.assert_invariants();
        assert_eq!(got.hash().seed(), 0x311);
        assert_eq!(got.hash().bits(), 4);
        let want = sum.clone().with_hash(target);
        assert_eq!(canonical_triples(&got), canonical_triples(&want));
    }

    // ---- S6: add ----

    /// Two sums over the same keyspace with heavy key overlap, so `add` sees
    /// merges and not just interleaving.
    fn overlapping_pair<const W: usize>(
        n: usize,
        num_qubits: usize,
        weight: usize,
        seed: u64,
    ) -> (PauliSum<W>, PauliSum<W>) {
        (
            rand_low_weight_sum::<W>(n, num_qubits, weight, seed),
            rand_low_weight_sum::<W>(n, num_qubits, weight, seed ^ 0xFFFF),
        )
    }

    #[test]
    fn add_same_hash_is_bitwise_flat_add() {
        // Every surviving coefficient is one `a + b`, computed in the same
        // operand order as the flat merge, so equality is exact even though the
        // partial *sums* live in different buckets.
        let (a, b) = overlapping_pair::<2>(4000, 100, 3, 0x401);
        let want = a.add(&b);
        for bits in [0u8, 4, 9] {
            let h = Gf2Hash::<2>::new(100, bits, 0x402);
            let ba = a.clone().with_hash(h.clone());
            let bb = b.clone().with_hash(h);
            let got = ba.add(&bb);
            got.assert_invariants();
            assert_eq!(got.len(), want.len(), "bits={bits} length");
            assert_same_sum(&want, &got);
        }
    }

    #[test]
    fn add_mixed_bits_matches_flat_bitwise() {
        let (a, b) = overlapping_pair::<2>(3000, 100, 3, 0x403);
        let want = a.add(&b);
        for &(bits_a, bits_b) in &[(4u8, 8u8), (8, 4), (0, 7), (7, 0), (6, 6)] {
            let ba = a.clone().with_hash(Gf2Hash::<2>::new(100, bits_a, 0x404));
            let bb = b.clone().with_hash(Gf2Hash::<2>::new(100, bits_b, 0x404));
            let got = ba.add(&bb);
            got.assert_invariants();
            assert_same_sum(&want, &got);
        }
    }

    #[test]
    fn add_mixed_seeds_matches_flat_bitwise() {
        let (a, b) = overlapping_pair::<2>(3000, 100, 3, 0x405);
        let want = a.add(&b);
        for &(bits_a, bits_b) in &[(5u8, 5u8), (3, 8)] {
            let ba = a.clone().with_hash(Gf2Hash::<2>::new(100, bits_a, 0x406));
            let bb = b.clone().with_hash(Gf2Hash::<2>::new(100, bits_b, 0x407));
            let got = ba.add(&bb);
            got.assert_invariants();
            assert_same_sum(&want, &got);
        }
    }

    #[test]
    fn add_result_carries_left_hash() {
        let (a, b) = overlapping_pair::<1>(500, 40, 2, 0x408);
        let ba = a.clone().with_hash(Gf2Hash::<1>::new(40, 3, 0x409));
        let bb = b.clone().with_hash(Gf2Hash::<1>::new(40, 7, 0x40A));
        let got = ba.add(&bb);
        assert_eq!(got.hash().bits(), 3, "bits");
        assert_eq!(got.hash().seed(), 0x409, "seed");
        assert_eq!(got.num_buckets(), 8);
        // The operands are untouched.
        assert_eq!(ba.hash().bits(), 3);
        assert_eq!(bb.hash().bits(), 7);
        assert_eq!(bb.hash().seed(), 0x40A);
    }

    #[test]
    fn add_cancels_to_nothing() {
        let a = rand_low_weight_sum::<1>(300, 40, 2, 0x40B);
        let mut neg = a.clone();
        neg.scale(Complex64::new(-1.0, 0.0));
        let ba = a.clone().with_hash(Gf2Hash::<1>::new(40, 5, 0x40C));
        let bn = neg.clone().with_hash(Gf2Hash::<1>::new(40, 2, 0x40D));
        let got = ba.add(&bn);
        got.assert_invariants();
        assert!(
            got.is_empty(),
            "{} terms survived exact cancellation",
            got.len()
        );
    }

    #[test]
    #[should_panic(expected = "num_qubits mismatch")]
    fn add_rejects_a_qubit_count_mismatch() {
        let a = rand_sum::<2>(100, 100, 0x40E);
        let b = rand_sum::<2>(100, 128, 0x40F);
        let ba = a.clone().with_hash(Gf2Hash::<2>::new(100, 3, 0x410));
        let bb = b.clone().with_hash(Gf2Hash::<2>::new(128, 3, 0x410));
        let _ = ba.add(&bb);
    }

    // =====================================================================
    // Hand-computed small-sum semantics, merged here from `pauli_sum.rs`'s
    // own test module (the shim that re-exports this type).
    //
    // Everything above is differential or a bucket-count sweep: it pins that
    // the bucketed path agrees with the single-bucket path, not what either
    // one computes. These pin the values themselves, on inputs small enough
    // to work out by hand — which is why they mostly survived the merge
    // rather than collapsing into the tests above.
    // =====================================================================

    // ---- overlap / expectation ----
    //
    // Before this there was no way to get a *number* out of a propagated sum:
    // examples/ising_2d_quench.rs hand-rolled its own observable against the raw
    // SoA columns, a gap CLAUDE.md flagged.

    fn b10_build<const W: usize>(
        num_qubits: usize,
        terms: &[(PauliString<W>, Complex64)],
    ) -> PauliSum<W> {
        let mut acc =
            crate::accumulator::BuildAccumulator::<W>::with_capacity(num_qubits, terms.len());
        for &(pp, c) in terms {
            acc.add_term(pp, crate::phase::Phase::ONE, c);
        }
        acc.finalize()
    }

    #[test]
    fn overlap_with_self_is_the_squared_norm() {
        let a = b10_build::<1>(
            8,
            &[
                (PauliString::<1>::x(0), Complex64::new(2.0, 0.0)),
                (PauliString::<1>::z(3), Complex64::new(0.0, 3.0)),
            ],
        );
        assert!((a.overlap(&a) - Complex64::new(13.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn overlap_is_conjugate_symmetric() {
        let a = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(1.0, 2.0))]);
        let b = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(3.0, -1.0))]);
        let ab = a.overlap(&b);
        let ba = b.overlap(&a);
        assert!((ab - ba.conj()).norm() < 1e-12, "{ab} vs conj({ba})");
    }

    #[test]
    fn overlap_of_disjoint_supports_is_zero() {
        let a = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(1.0, 0.0))]);
        let b = b10_build::<1>(8, &[(PauliString::<1>::z(5), Complex64::new(1.0, 0.0))]);
        assert!(a.overlap(&b).norm() < 1e-12);
    }

    #[test]
    fn overlap_only_counts_shared_keys() {
        let a = b10_build::<1>(
            8,
            &[
                (PauliString::<1>::x(0), Complex64::new(2.0, 0.0)),
                (PauliString::<1>::y(1), Complex64::new(5.0, 0.0)),
            ],
        );
        let b = b10_build::<1>(
            8,
            &[
                (PauliString::<1>::x(0), Complex64::new(3.0, 0.0)),
                (PauliString::<1>::z(2), Complex64::new(7.0, 0.0)),
            ],
        );
        assert!((a.overlap(&b) - Complex64::new(6.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn overlap_across_a_word_boundary_w2() {
        let a = b10_build::<2>(
            128,
            &[
                (PauliString::<2>::x(3), Complex64::new(1.0, 0.0)),
                (PauliString::<2>::z(70), Complex64::new(2.0, 0.0)),
            ],
        );
        let b = b10_build::<2>(128, &[(PauliString::<2>::z(70), Complex64::new(4.0, 0.0))]);
        assert!((a.overlap(&b) - Complex64::new(8.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn identity_coefficient_picks_out_the_trace() {
        let a = b10_build::<1>(
            8,
            &[
                (PauliString::<1>::identity(), Complex64::new(1.5, 0.0)),
                (PauliString::<1>::x(0), Complex64::new(9.0, 0.0)),
            ],
        );
        assert!((a.identity_coefficient() - Complex64::new(1.5, 0.0)).norm() < 1e-12);
        let b = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(9.0, 0.0))]);
        assert!(b.identity_coefficient().norm() < 1e-12);
    }

    #[test]
    fn expectation_of_single_paulis_in_each_product_state() {
        let cases = [
            (PauliString::<1>::identity(), 1.0, 1.0, 1.0),
            (PauliString::<1>::x(0), 1.0, 0.0, 0.0),
            (PauliString::<1>::y(0), 0.0, 1.0, 0.0),
            (PauliString::<1>::z(0), 0.0, 0.0, 1.0),
        ];
        for (pp, ex, ey, ez) in cases {
            let s = b10_build::<1>(8, &[(pp, Complex64::new(1.0, 0.0))]);
            assert!(
                (s.expectation_product_state(ProductState::XPlus).re - ex).abs() < 1e-12,
                "XPlus for {pp:?}",
            );
            assert!(
                (s.expectation_product_state(ProductState::YPlus).re - ey).abs() < 1e-12,
                "YPlus for {pp:?}",
            );
            assert!(
                (s.expectation_product_state(ProductState::ZPlus).re - ez).abs() < 1e-12,
                "ZPlus for {pp:?}",
            );
        }
    }

    #[test]
    fn expectation_of_multi_qubit_products() {
        let mut xx = PauliString::<1>::x(0);
        xx.mul_assign(&PauliString::<1>::x(1));
        let mut xz = PauliString::<1>::x(0);
        xz.mul_assign(&PauliString::<1>::z(1));
        let mut yy = PauliString::<1>::y(0);
        yy.mul_assign(&PauliString::<1>::y(1));

        let s = b10_build::<1>(
            8,
            &[
                (xx, Complex64::new(1.0, 0.0)),
                (xz, Complex64::new(10.0, 0.0)),
                (yy, Complex64::new(100.0, 0.0)),
            ],
        );
        assert!((s.expectation_product_state(ProductState::XPlus).re - 1.0).abs() < 1e-12);
        assert!((s.expectation_product_state(ProductState::YPlus).re - 100.0).abs() < 1e-12);
        assert!(s.expectation_product_state(ProductState::ZPlus).re.abs() < 1e-12);
    }

    #[test]
    fn expectation_is_linear_and_keeps_the_imaginary_part() {
        let s = b10_build::<1>(
            8,
            &[
                (PauliString::<1>::x(0), Complex64::new(1.0, 2.0)),
                (PauliString::<1>::x(1), Complex64::new(3.0, -5.0)),
            ],
        );
        let e = s.expectation_product_state(ProductState::XPlus);
        assert!((e - Complex64::new(4.0, -3.0)).norm() < 1e-12);
    }

    #[test]
    fn expectation_across_a_word_boundary_w2() {
        let s = b10_build::<2>(
            128,
            &[
                (PauliString::<2>::x(70), Complex64::new(2.0, 0.0)),
                (PauliString::<2>::z(70), Complex64::new(9.0, 0.0)),
            ],
        );
        assert!((s.expectation_product_state(ProductState::XPlus).re - 2.0).abs() < 1e-12);
        assert!((s.expectation_product_state(ProductState::ZPlus).re - 9.0).abs() < 1e-12);
    }

    /// The new API must reproduce the observable
    /// `examples/ising_2d_quench.rs` hand-rolled, which is why it exists.
    #[test]
    fn expectation_xplus_matches_the_hand_rolled_reference() {
        let mut rng = 0x2468u64 | 1;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut acc = crate::accumulator::BuildAccumulator::<1>::with_capacity(16, 500);
        for _ in 0..500 {
            let pp = PauliString::<1> {
                x: [next() & 0xFFFF],
                z: [next() & 0xFFFF],
            };
            let c = Complex64::new((next() as i64 as f64) / (i64::MAX as f64), 0.0);
            acc.add_term(pp, crate::phase::Phase::ONE, c);
        }
        let sum = acc.finalize();

        let mut want = 0.0f64;
        for i in 0..sum.len() {
            if sum.bucket(0).1[i] == [0u64] {
                want += sum.bucket(0).2[i].re;
            }
        }
        let got = sum.expectation_product_state(ProductState::XPlus).re;
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    // --- non-uniform product states (ProductBasis) ---------------------
    //
    // Every expected value below is the product of single-qubit Bloch-vector
    // components, hand-derived once here:
    //
    //   |0⟩ = Z+:  ⟨Z⟩ = +1,  ⟨X⟩ = ⟨Y⟩ = 0
    //   |1⟩ = Z-:  ⟨Z⟩ = -1,  ⟨X⟩ = ⟨Y⟩ = 0
    //   |+⟩ = X+:  ⟨X⟩ = +1,  ⟨Y⟩ = ⟨Z⟩ = 0
    //   |-⟩ = X-:  ⟨X⟩ = -1,  ⟨Y⟩ = ⟨Z⟩ = 0
    //   |r⟩ = Y+ = (|0⟩ + i|1⟩)/√2:  ⟨Y⟩ = +1,  ⟨X⟩ = ⟨Z⟩ = 0
    //   |l⟩ = Y- = (|0⟩ - i|1⟩)/√2:  ⟨Y⟩ = -1,  ⟨X⟩ = ⟨Z⟩ = 0
    //
    // and ⟨I⟩ = 1 in every state. Off-axis components vanish because two
    // distinct single-qubit Paulis anticommute.

    /// Per-qubit label string → [`ProductBasis`], in the alphabet the Python
    /// binding accepts (qiskit `Statevector.from_label`): `0`/`1` = Z±,
    /// `+`/`-` = X±, `r`/`l` = Y±. Character `i` addresses qubit `i`.
    ///
    /// Deliberately spelled out here rather than shared with the binding: the
    /// test's job is to encode the convention independently.
    fn basis_from_labels<const W: usize>(labels: &str) -> ProductBasis<W> {
        ProductBasis::<W>::from_axes(labels.chars().map(|ch| match ch {
            '0' => (PauliAxis::Z, false),
            '1' => (PauliAxis::Z, true),
            '+' => (PauliAxis::X, false),
            '-' => (PauliAxis::X, true),
            'r' => (PauliAxis::Y, false),
            'l' => (PauliAxis::Y, true),
            other => panic!("unexpected label {other:?}"),
        }))
    }

    /// Differential oracle: `⟨ψ|O|ψ⟩` evaluated one qubit at a time straight
    /// from the Bloch table above, sharing no code with the masked scan.
    fn naive_labelled_expectation<const W: usize>(sum: &PauliSum<W>, labels: &str) -> Complex64 {
        let mut total = Complex64::new(0.0, 0.0);
        for (x, z, c) in sum.iter() {
            let mut factor = 1.0f64;
            for (q, label) in labels.chars().enumerate() {
                let bx = (x[q / 64] >> (q % 64)) & 1 == 1;
                let bz = (z[q / 64] >> (q % 64)) & 1 == 1;
                factor *= match (bx, bz, label) {
                    (false, false, _) => 1.0, // identity factor: no constraint
                    (true, false, '+') => 1.0,
                    (true, false, '-') => -1.0,
                    (false, true, '0') => 1.0,
                    (false, true, '1') => -1.0,
                    (true, true, 'r') => 1.0,
                    (true, true, 'l') => -1.0,
                    _ => 0.0, // off-axis Pauli: zero overlap
                };
            }
            total += c * factor;
        }
        total
    }

    fn expect_close<const W: usize>(sum: &PauliSum<W>, labels: &str, want: f64) {
        let got = sum.expectation_product_basis(&basis_from_labels::<W>(labels));
        assert!(
            (got - Complex64::new(want, 0.0)).norm() < 1e-12,
            "state {labels:?}: got {got}, want {want}",
        );
    }

    fn single_qubit_labels_against_every_pauli<const W: usize>() {
        // (label, ⟨I⟩, ⟨X⟩, ⟨Y⟩, ⟨Z⟩) — the Bloch table above, transposed.
        let cases = [
            ('0', 1.0, 0.0, 0.0, 1.0),
            ('1', 1.0, 0.0, 0.0, -1.0),
            ('+', 1.0, 1.0, 0.0, 0.0),
            ('-', 1.0, -1.0, 0.0, 0.0),
            ('r', 1.0, 0.0, 1.0, 0.0),
            ('l', 1.0, 0.0, -1.0, 0.0),
        ];
        for (label, ei, ex, ey, ez) in cases {
            let labels = label.to_string();
            for (pauli, want) in [("I", ei), ("X", ex), ("Y", ey), ("Z", ez)] {
                let s = PauliSum::<W>::from_strings(&[(pauli, Complex64::new(1.0, 0.0))]);
                let got = s.expectation_product_basis(&basis_from_labels::<W>(&labels));
                assert!(
                    (got - Complex64::new(want, 0.0)).norm() < 1e-12,
                    "⟨{label}|{pauli}|{label}⟩ = {got}, want {want} (W={W})",
                );
            }
        }
    }

    #[test]
    fn single_qubit_labels_against_every_pauli_w1() {
        single_qubit_labels_against_every_pauli::<1>();
    }

    #[test]
    fn single_qubit_labels_against_every_pauli_w2() {
        single_qubit_labels_against_every_pauli::<2>();
    }

    fn multi_qubit_products_compose_per_qubit_signs<const W: usize>() {
        // ⟨01|Z⊗Z|01⟩ = ⟨0|Z|0⟩·⟨1|Z|1⟩ = (+1)(-1) = -1.
        let zz = PauliSum::<W>::from_strings(&[("ZZ", Complex64::new(1.0, 0.0))]);
        expect_close(&zz, "01", -1.0);
        expect_close(&zz, "10", -1.0);
        expect_close(&zz, "00", 1.0);
        expect_close(&zz, "11", 1.0); // (-1)(-1)

        // State |0⟩|+⟩|r⟩: axes Z, X, Y, every sign +1.
        let zxy = PauliSum::<W>::from_strings(&[("ZXY", Complex64::new(1.0, 0.0))]);
        expect_close(&zxy, "0+r", 1.0);
        // X on the Y-axis qubit is off-axis → the whole term drops.
        let zxx = PauliSum::<W>::from_strings(&[("ZXX", Complex64::new(1.0, 0.0))]);
        expect_close(&zxx, "0+r", 0.0);
        // Identity factors are ignored, whatever that qubit's label is.
        let ziy = PauliSum::<W>::from_strings(&[("ZIY", Complex64::new(1.0, 0.0))]);
        expect_close(&ziy, "0+r", 1.0);
        expect_close(&ziy, "0-r", 1.0);
        expect_close(&ziy, "01r", 1.0);

        // State |1⟩|-⟩|l⟩: the same axes with all three signs flipped, so a
        // weight-3 term picks up (-1)^3 = -1.
        expect_close(&zxy, "1-l", -1.0);
        // Only the two non-identity sites' signs count: (-1)·(-1) = +1.
        expect_close(&ziy, "1-l", 1.0);
        // A single flipped site: -1.
        expect_close(&zxy, "1+r", -1.0);
        expect_close(&zxy, "0-r", -1.0);
        expect_close(&zxy, "0+l", -1.0);
        // Two flipped sites: +1.
        expect_close(&zxy, "1-r", 1.0);
    }

    #[test]
    fn multi_qubit_products_compose_per_qubit_signs_w1() {
        multi_qubit_products_compose_per_qubit_signs::<1>();
    }

    #[test]
    fn multi_qubit_products_compose_per_qubit_signs_w2() {
        multi_qubit_products_compose_per_qubit_signs::<2>();
    }

    fn an_off_axis_pauli_never_matches<const W: usize>() {
        // The subset-match trap: `X` on a Y-axis qubit must NOT contribute,
        // even though the Y axis has its x-bit set — the match is an equality
        // on both halves of the key, not `x & !ax_x == 0`. ⟨r|X|r⟩ = 0.
        let off_axis = [
            ('r', "X"),
            ('r', "Z"),
            ('l', "X"),
            ('l', "Z"),
            ('+', "Y"),
            ('+', "Z"),
            ('-', "Y"),
            ('0', "X"),
            ('0', "Y"),
            ('1', "X"),
            ('1', "Y"),
        ];
        for (label, pauli) in off_axis {
            let s = PauliSum::<W>::from_strings(&[(pauli, Complex64::new(3.0, -4.0))]);
            let got = s.expectation_product_basis(&basis_from_labels::<W>(&label.to_string()));
            assert!(
                got.norm() < 1e-12,
                "⟨{label}|{pauli}|{label}⟩ = {got}, want 0 (W={W})",
            );
        }
        // Mixed: one off-axis factor kills a term whose other factors match.
        let s = PauliSum::<W>::from_strings(&[("XXX", Complex64::new(1.0, 0.0))]);
        expect_close(&s, "++r", 0.0);
        expect_close(&s, "+++", 1.0);
    }

    #[test]
    fn an_off_axis_pauli_never_matches_w1() {
        an_off_axis_pauli_never_matches::<1>();
    }

    #[test]
    fn an_off_axis_pauli_never_matches_w2() {
        an_off_axis_pauli_never_matches::<2>();
    }

    #[test]
    fn labelled_expectation_is_linear_and_keeps_the_imaginary_part() {
        // ⟨1|Z|1⟩ = -1 and ⟨1|I|1⟩ = +1, so this is -(1+2i) + (3-5i).
        let s = PauliSum::<1>::from_strings(&[
            ("Z", Complex64::new(1.0, 2.0)),
            ("I", Complex64::new(3.0, -5.0)),
        ]);
        let got = s.expectation_product_basis(&basis_from_labels::<1>("1"));
        assert!((got - Complex64::new(2.0, -7.0)).norm() < 1e-12, "{got}");
    }

    #[test]
    fn labels_across_the_word_boundary_are_independent_w2() {
        // 128 qubits, |0…0⟩ except qubit 64, which is |1⟩ — its sign bit lives
        // in word 1 of `neg`, so a word-0-only implementation would miss it.
        let mut labels: String = "0".repeat(128);
        labels.replace_range(64..65, "1");
        let basis = basis_from_labels::<2>(&labels);
        let cases = [
            (PauliString::<2>::z(0), 1.0),   // qubit 0 is |0⟩
            (PauliString::<2>::z(64), -1.0), // qubit 64 is |1⟩
            (PauliString::<2>::x(64), 0.0),  // off-axis on a Z qubit
        ];
        for (p, want) in cases {
            let s = b10_build::<2>(128, &[(p, Complex64::new(1.0, 0.0))]);
            let got = s.expectation_product_basis(&basis);
            assert!(
                (got - Complex64::new(want, 0.0)).norm() < 1e-12,
                "{p:?}: got {got}, want {want}",
            );
        }
        // Z on both sides of the boundary: (+1)·(-1) = -1.
        let mut z0z64 = PauliString::<2>::z(0);
        z0z64.mul_assign(&PauliString::<2>::z(64));
        let s = b10_build::<2>(128, &[(z0z64, Complex64::new(1.0, 0.0))]);
        let got = s.expectation_product_basis(&basis);
        assert!((got - Complex64::new(-1.0, 0.0)).norm() < 1e-12, "{got}");
    }

    #[test]
    fn labelled_expectation_of_an_empty_sum_is_zero() {
        let h = Gf2Hash::<1>::new(8, 3, 0xE6);
        let b = PauliSum::<1>::empty_with_hash(8, h);
        let got = b.expectation_product_basis(&basis_from_labels::<1>("01+-rl01"));
        assert!(got.norm() < 1e-15, "{got}");
    }

    fn uniform_states_agree_with_their_label_spellings<const W: usize>() {
        let num_qubits = 50 * W;
        let sum = rand_sum::<W>(4000, num_qubits, 0xA40 + W as u64);
        for (state, label) in [
            (ProductState::XPlus, '+'),
            (ProductState::YPlus, 'r'),
            (ProductState::ZPlus, '0'),
        ] {
            let want = sum.expectation_product_state(state);
            let labels: String = std::iter::repeat_n(label, num_qubits).collect();
            let got = sum.expectation_product_basis(&basis_from_labels::<W>(&labels));
            assert!(
                (got - want).norm() < 1e-12,
                "{state:?} vs {label:?}: {got} vs {want} (W={W})",
            );
        }
    }

    #[test]
    fn uniform_states_agree_with_their_label_spellings_w1() {
        uniform_states_agree_with_their_label_spellings::<1>();
    }

    #[test]
    fn uniform_states_agree_with_their_label_spellings_w2() {
        uniform_states_agree_with_their_label_spellings::<2>();
    }

    fn labelled_expectation_agrees_with_the_naive_reference<const W: usize>() {
        // 33 qubits per word so W=2 straddles the boundary at 64.
        let num_qubits = 33 * W;
        let alphabet: Vec<char> = "01+-rl".chars().collect();
        let mut rng = Xs64::new(0xB40 + W as u64);
        let labels: String = (0..num_qubits)
            .map(|_| alphabet[(rng.next_u64() % 6) as usize])
            .collect();
        for &weight in &[1usize, 2, 3] {
            let sum = low_weight_sum::<W>(3000, num_qubits, weight, 0xB50 + weight as u64);
            let want = naive_labelled_expectation(&sum, &labels);
            let got = sum.expectation_product_basis(&basis_from_labels::<W>(&labels));
            assert!(
                (got - want).norm() < 1e-9,
                "W={W} weight={weight}: {got} vs {want}",
            );
        }
    }

    #[test]
    fn labelled_expectation_agrees_with_the_naive_reference_w1() {
        labelled_expectation_agrees_with_the_naive_reference::<1>();
    }

    #[test]
    fn labelled_expectation_agrees_with_the_naive_reference_w2() {
        labelled_expectation_agrees_with_the_naive_reference::<2>();
    }

    #[test]
    fn bucketed_labelled_expectation_agrees_across_partitions() {
        // The sign parity is accumulated inside a bucket, so a partition
        // change must not move the value (beyond float re-association).
        let alphabet: Vec<char> = "01+-rl".chars().collect();
        let mut rng = Xs64::new(0xB60);
        let labels: String = (0..100)
            .map(|_| alphabet[(rng.next_u64() % 6) as usize])
            .collect();
        let basis = basis_from_labels::<2>(&labels);
        let sum = low_weight_sum::<2>(20_000, 100, 3, 0xB61);
        let want = sum.expectation_product_basis(&basis);
        for bits in [0u8, 3, 7, 11] {
            let h = Gf2Hash::<2>::new(100, bits, 0xB62);
            let b = sum.clone().with_hash(h);
            let got = b.expectation_product_basis(&basis);
            assert!((got - want).norm() < 1e-9, "bits={bits}: {got} vs {want}");
        }
    }

    #[test]
    fn assert_invariants_accepts_bits_within_num_qubits() {
        // num_qubits=50, single term with X on qubit 49 (in range).
        let sum = PauliSum::<1>::from_sorted_columns(
            vec![[1u64 << 49]],
            vec![[0u64; 1]],
            vec![Complex64::new(1.0, 0.0)],
            50,
        );
        sum.assert_invariants();
    }

    #[test]
    #[should_panic(expected = "exceeds num_qubits")]
    fn assert_invariants_rejects_bit_beyond_num_qubits() {
        // num_qubits=50, but X bit set at qubit 50 — must panic.
        let sum = PauliSum::<1>::from_sorted_columns(
            vec![[1u64 << 50]],
            vec![[0u64; 1]],
            vec![Complex64::new(1.0, 0.0)],
            50,
        );
        sum.assert_invariants();
    }

    #[test]
    #[should_panic(expected = "exceeds num_qubits")]
    fn assert_invariants_rejects_z_bit_beyond_num_qubits() {
        // Same as above but on the Z-part: invariant must check both parts.
        let sum = PauliSum::<1>::from_sorted_columns(
            vec![[0u64; 1]],
            vec![[1u64 << 60]],
            vec![Complex64::new(1.0, 0.0)],
            50,
        );
        sum.assert_invariants();
    }

    #[test]
    #[should_panic(expected = "exceeds num_qubits")]
    fn assert_invariants_rejects_bit_in_unused_word() {
        // num_qubits=64 (one full word), W=2. Bit on qubit 64 lives in word 1
        // and is therefore out of range.
        let sum = PauliSum::<2>::from_sorted_columns(
            vec![[0u64, 1u64]],
            vec![[0u64; 2]],
            vec![Complex64::new(1.0, 0.0)],
            64,
        );
        sum.assert_invariants();
    }

    // --- keyed lookup (get) -----------------------------------------------

    /// Three-term `PauliSum<1>` with sorted, distinct keys `K0 < K1 < K2`.
    fn three_term_sum_w1() -> PauliSum<1> {
        // K0 = (x=0, z=1), K1 = (x=1, z=0), K2 = (x=1, z=2). Sorted by lex
        // on (x, z): K0 has smallest x; K1, K2 share x but K1 has smaller z.
        PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64], [1u64]],
            vec![[1u64], [0u64], [2u64]],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            4,
        )
    }

    #[test]
    fn get_on_empty_is_none() {
        let s = PauliSum::<1>::empty(4);
        assert_eq!(s.get(&[0u64], &[0u64]), None);
    }

    #[test]
    fn get_hits_every_key_and_misses_between() {
        let s = three_term_sum_w1();
        assert_eq!(s.get(&[0u64], &[1u64]), Some(Complex64::new(1.0, 0.0)));
        assert_eq!(s.get(&[1u64], &[0u64]), Some(Complex64::new(2.0, 0.0)));
        assert_eq!(s.get(&[1u64], &[2u64]), Some(Complex64::new(3.0, 0.0)));
        // Below the smallest, in a gap, and above the largest key.
        assert_eq!(s.get(&[0u64], &[0u64]), None);
        assert_eq!(s.get(&[1u64], &[1u64]), None);
        assert_eq!(s.get(&[2u64], &[0u64]), None);
    }

    #[test]
    fn canonical_order_is_lex_x_before_z_on_a_single_bucket() {
        // Two terms with K_a=(x=0, z=5) and K_b=(x=1, z=0). Despite z_a > z_b,
        // x_a < x_b, so K_a < K_b in the canonical (lex) order of a
        // single-bucket sum. A lex-on-x-only order would invert this.
        //
        // `single_bucket_sum_is_plain_lex_sorted` above does not cover this:
        // its keys are random, so two of them essentially never share an `x`
        // and the `z` tiebreak is never exercised.
        let s = PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64]],
            vec![[5u64], [0u64]],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            4,
        );
        s.assert_invariants();
        let (x, z, _) = s.to_arrays();
        assert_eq!((x[0], z[0]), ([0u64], [5u64]));
        assert_eq!((x[1], z[1]), ([1u64], [0u64]));
    }

    // --- scale() ----------------------------------------------------------

    #[test]
    fn scale_by_zero_zeros_all_coeffs() {
        let mut s = three_term_sum_w1();
        s.scale(Complex64::new(0.0, 0.0));
        assert_eq!(s.len(), 3);
        for (_, _, c) in s.iter() {
            assert_eq!(c, Complex64::new(0.0, 0.0));
        }
        s.assert_invariants();
    }

    #[test]
    fn scale_by_one_is_identity() {
        let mut s = three_term_sum_w1();
        let (_, _, before) = s.to_arrays();
        s.scale(Complex64::new(1.0, 0.0));
        assert_eq!(s.to_arrays().2, before);
    }

    #[test]
    fn scale_by_i_rotates_phases() {
        let mut s = PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64]],
            vec![[1u64], [0u64]],
            vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, -3.0)],
            4,
        );
        s.scale(Complex64::new(0.0, 1.0));
        // (2 + 0i) * i = 0 + 2i; (0 - 3i) * i = 3 + 0i.
        assert_eq!(s.bucket(0).2[0], Complex64::new(0.0, 2.0));
        assert_eq!(s.bucket(0).2[1], Complex64::new(3.0, 0.0));
    }

    // --- add() ------------------------------------------------------------

    #[test]
    fn add_empty_left_is_other() {
        let a = PauliSum::<1>::empty(4);
        let b = three_term_sum_w1();
        let r = a.add(&b);
        assert_eq!(r.len(), 3);
        assert_eq!(r.to_arrays(), b.to_arrays());
        r.assert_invariants();
    }

    #[test]
    fn add_empty_right_is_self() {
        let a = three_term_sum_w1();
        let b = PauliSum::<1>::empty(4);
        let r = a.add(&b);
        assert_eq!(r.len(), 3);
        assert_eq!(r.to_arrays(), a.to_arrays());
        r.assert_invariants();
    }

    #[test]
    fn add_disjoint_keys_interleaves_in_sort_order() {
        // a has K0=(0,1), K2=(1,2); b has K1=(1,0), K3=(2,0).
        // Lex sort across the union: (0,1) < (1,0) < (1,2) < (2,0).
        let a = PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64]],
            vec![[1u64], [2u64]],
            vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
            4,
        );
        let b = PauliSum::<1>::from_sorted_columns(
            vec![[1u64], [2u64]],
            vec![[0u64], [0u64]],
            vec![Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)],
            4,
        );
        let r = a.add(&b);
        assert_eq!(r.len(), 4);
        let (rx, rz, rc) = r.to_arrays();
        assert_eq!(rx, vec![[0u64], [1u64], [1u64], [2u64]]);
        assert_eq!(rz, vec![[1u64], [0u64], [2u64], [0u64]]);
        assert_eq!(
            rc,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ]
        );
        r.assert_invariants();
    }

    #[test]
    fn add_equal_keys_sum_coeffs() {
        let a = three_term_sum_w1();
        let r = a.add(&a);
        assert_eq!(r.len(), 3);
        assert_eq!(r.to_arrays().0, a.to_arrays().0);
        assert_eq!(r.to_arrays().1, a.to_arrays().1);
        for k in 0..3 {
            assert_eq!(
                r.bucket(0).2[k],
                a.bucket(0).2[k] * Complex64::new(2.0, 0.0)
            );
        }
        r.assert_invariants();
    }

    #[test]
    fn add_mixed_cancellation_and_merge() {
        // a = {K1: 1, K2: 2, K3: 3}, b = {K1: -1, K2: 0.5, K4: 4}
        // K1 cancels, K2 sums to 2.5, K3 unique to a, K4 unique to b.
        let a = PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64], [2u64]],
            vec![[0u64], [0u64], [0u64]],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            4,
        );
        let b = PauliSum::<1>::from_sorted_columns(
            vec![[0u64], [1u64], [3u64]],
            vec![[0u64], [0u64], [0u64]],
            vec![
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(4.0, 0.0),
            ],
            4,
        );
        let r = a.add(&b);
        assert_eq!(r.len(), 3);
        let (rx, rz, rc) = r.to_arrays();
        assert_eq!(rx, vec![[1u64], [2u64], [3u64]]);
        assert_eq!(rz, vec![[0u64], [0u64], [0u64]]);
        assert_eq!(
            rc,
            vec![
                Complex64::new(2.5, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ]
        );
        r.assert_invariants();
    }

    #[test]
    fn add_w2_across_word_boundary() {
        let a = PauliSum::<2>::from_sorted_columns(
            vec![[0u64, 1u64], [0u64, 2u64]],
            vec![[0u64, 0u64], [0u64, 0u64]],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            128,
        );
        let b = PauliSum::<2>::from_sorted_columns(
            vec![[0u64, 1u64], [0u64, 4u64]],
            vec![[0u64, 0u64], [0u64, 0u64]],
            vec![Complex64::new(0.5, 0.0), Complex64::new(7.0, 0.0)],
            128,
        );
        let r = a.add(&b);
        assert_eq!(r.len(), 3);
        assert_eq!(
            r.to_arrays().0,
            vec![[0u64, 1u64], [0u64, 2u64], [0u64, 4u64]]
        );
        assert_eq!(r.bucket(0).2[0], Complex64::new(1.5, 0.0));
        assert_eq!(r.bucket(0).2[1], Complex64::new(2.0, 0.0));
        assert_eq!(r.bucket(0).2[2], Complex64::new(7.0, 0.0));
        r.assert_invariants();
    }

    // --- PauliSum::from_strings test helper ----------------------------
    //
    // `from_strings` itself is a `#[cfg(test)]` inherent impl over in
    // `pauli_sum.rs`; only its tests moved here.

    #[test]
    fn from_strings_single_x_term() {
        let s = PauliSum::<1>::from_strings(&[("XII", Complex64::new(1.0, 0.0))]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.num_qubits(), 3);
        assert_eq!(s.bucket(0).0[0], [0b001u64]);
        assert_eq!(s.bucket(0).1[0], [0u64]);
        assert_eq!(s.bucket(0).2[0], Complex64::new(1.0, 0.0));
        s.assert_invariants();
    }

    #[test]
    fn from_strings_x_z_combined() {
        // "XZI": X on qubit 0, Z on qubit 1, I on qubit 2.
        let s = PauliSum::<1>::from_strings(&[("XZI", Complex64::new(1.0, 0.0))]);
        assert_eq!(s.bucket(0).0[0], [0b001u64]);
        assert_eq!(s.bucket(0).1[0], [0b010u64]);
        s.assert_invariants();
    }

    #[test]
    fn from_strings_y_is_hermitian() {
        // Coefficients multiply the literal Hermitian Pauli string: "Y" maps
        // to the symplectic key (x=1, z=1) with no phase factor, matching
        // PauliString::y and expectation_product_state.
        let s = PauliSum::<1>::from_strings(&[("Y", Complex64::new(1.0, 0.0))]);
        assert_eq!(s.bucket(0).0[0], [1u64]);
        assert_eq!(s.bucket(0).1[0], [1u64]);
        assert_eq!(s.bucket(0).2[0], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn from_strings_real_coeffs_stay_real_for_any_y_count() {
        // A Hermitian observable keeps real coefficients regardless of how
        // many Y characters a term contains — no per-Y phase is folded.
        for s in ["Y", "YY", "YYY", "YYYY"] {
            let padded: String = format!("{s:I<4}");
            let sum = PauliSum::<1>::from_strings(&[(&padded, Complex64::new(2.5, 0.0))]);
            assert_eq!(sum.bucket(0).2[0], Complex64::new(2.5, 0.0), "{s}");
        }
    }

    #[test]
    fn from_strings_dedup_sums_coeffs() {
        let s = PauliSum::<1>::from_strings(&[
            ("XI", Complex64::new(1.0, 0.0)),
            ("XI", Complex64::new(0.5, -0.25)),
        ]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.bucket(0).2[0], Complex64::new(1.5, -0.25));
        s.assert_invariants();
    }

    #[test]
    fn from_strings_cancellation_drops_term() {
        let s = PauliSum::<1>::from_strings(&[
            ("XI", Complex64::new(1.0, 0.0)),
            ("XI", Complex64::new(-1.0, 0.0)),
            ("ZI", Complex64::new(2.0, 0.0)),
        ]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.bucket(0).0[0], [0u64]);
        assert_eq!(s.bucket(0).1[0], [1u64]);
        assert_eq!(s.bucket(0).2[0], Complex64::new(2.0, 0.0));
        s.assert_invariants();
    }

    #[test]
    fn from_strings_sorts_lex_keys() {
        // Insert out of order: ZI=(0,1), XI=(1,0), YI=(1,1) — lex sorted is
        // ZI < XI < YI.
        let s = PauliSum::<1>::from_strings(&[
            ("YI", Complex64::new(1.0, 0.0)),
            ("ZI", Complex64::new(2.0, 0.0)),
            ("XI", Complex64::new(3.0, 0.0)),
        ]);
        assert_eq!(s.len(), 3);
        assert_eq!((s.bucket(0).0[0], s.bucket(0).1[0]), ([0u64], [1u64])); // ZI
        assert_eq!((s.bucket(0).0[1], s.bucket(0).1[1]), ([1u64], [0u64])); // XI
        assert_eq!((s.bucket(0).0[2], s.bucket(0).1[2]), ([1u64], [1u64])); // YI
        assert_eq!(s.bucket(0).2[0], Complex64::new(2.0, 0.0));
        assert_eq!(s.bucket(0).2[1], Complex64::new(3.0, 0.0));
        assert_eq!(s.bucket(0).2[2], Complex64::new(1.0, 0.0));
        s.assert_invariants();
    }

    #[test]
    fn from_strings_w2_qubit_64() {
        // 65-character string: X at index 64 lands in word 1.
        let mut s_chars: String = "I".repeat(65);
        // Replace index 64 with 'X'.
        unsafe {
            let bytes = s_chars.as_bytes_mut();
            bytes[64] = b'X';
        }
        let s = PauliSum::<2>::from_strings(&[(s_chars.as_str(), Complex64::new(1.0, 0.0))]);
        assert_eq!(s.num_qubits(), 65);
        assert_eq!(s.bucket(0).0[0], [0u64, 1u64]);
        assert_eq!(s.bucket(0).1[0], [0u64, 0u64]);
        s.assert_invariants();
    }

    #[test]
    #[should_panic(expected = "unexpected Pauli char")]
    fn from_strings_panics_on_invalid_char() {
        let _ = PauliSum::<1>::from_strings(&[("AB", Complex64::new(1.0, 0.0))]);
    }

    #[test]
    #[should_panic(expected = "all pauli strings must have the same length")]
    fn from_strings_panics_on_length_mismatch() {
        let _ = PauliSum::<1>::from_strings(&[
            ("XI", Complex64::new(1.0, 0.0)),
            ("XII", Complex64::new(1.0, 0.0)),
        ]);
    }

    mod props {
        use super::*;
        use proptest::prelude::*;
        use std::collections::BTreeMap;

        const NQ: usize = 6;

        fn build(terms: &[(u64, u64, i32, i32)]) -> PauliSum<1> {
            let mut acc = BuildAccumulator::<1>::new(NQ);
            for &(x, z, re, im) in terms {
                acc.add_term(
                    PauliString::<1> { x: [x], z: [z] },
                    Phase::ONE,
                    Complex64::new(re as f64, im as f64),
                );
            }
            acc.finalize()
        }

        proptest! {
            /// `add` against an independent model: a `BTreeMap` keyed by
            /// `(x, z)`, summed then stripped of exact zeros.
            ///
            /// Coefficients are small integers so exact cancellation actually
            /// happens, and the keyspace is 6 qubits so the two operands share
            /// keys often. Every surviving coefficient is a single `a + b`, so
            /// the comparison is bitwise rather than toleranced.
            #[test]
            fn bucketed_add_matches_btreemap_model(
                terms_a in prop::collection::vec(
                    (0u64..64, 0u64..64, -4i32..=4, -4i32..=4), 0..60),
                terms_b in prop::collection::vec(
                    (0u64..64, 0u64..64, -4i32..=4, -4i32..=4), 0..60),
                bits_a in 0u8..=6,
                bits_b in 0u8..=6,
                seed_shift in 0u64..=1,
            ) {
                let a = build(&terms_a);
                let b = build(&terms_b);

                let seed_a = 0x5EEDu64;
                let ba = a.clone().with_hash(Gf2Hash::<1>::new(NQ, bits_a, seed_a));
                let bb = b
                    .clone()
                    .with_hash(Gf2Hash::<1>::new(NQ, bits_b, seed_a + seed_shift));

                let got = ba.add(&bb);
                got.assert_invariants();
                prop_assert_eq!(got.hash().bits(), bits_a, "left partition must win");
                prop_assert_eq!(got.hash().seed(), seed_a, "left partition must win");

                let mut model: BTreeMap<([u64; 1], [u64; 1]), Complex64> = BTreeMap::new();
                for (x, z, c) in a.iter() {
                    model.insert((*x, *z), c);
                }
                for (x, z, c) in b.iter() {
                    model
                        .entry((*x, *z))
                        .and_modify(|acc| *acc += c)
                        .or_insert(c);
                }
                let zero = Complex64::new(0.0, 0.0);
                model.retain(|_, c| *c != zero);

                prop_assert_eq!(got.len(), model.len());
                let triples = sorted_triples(&got);
                for (i, (&(mx, mz), &mc)) in model.iter().enumerate() {
                    prop_assert_eq!(triples[i].0, mx);
                    prop_assert_eq!(triples[i].1, mz);
                    prop_assert_eq!(triples[i].2, mc);
                }
            }
        }

        // ---- merged here from `pauli_sum.rs`'s own `props` module ----

        /// Build a sorted, deduplicated `PauliSum<2>` from random `(x, z, coeff)`
        /// triples. Uses `BTreeMap` keyed on `(x, z)` to enforce the sorted /
        /// unique invariant before SoA materialization. Coefficients are kept
        /// small (`re, im ∈ [-4, 4]`) so the property assertions don't run into
        /// FP cancellation noise. Length capped at 8 — sufficient to exercise
        /// merge interleaving without blowing up shrinking time.
        fn arb_pauli_sum_w2() -> impl Strategy<Value = PauliSum<2>> {
            prop::collection::vec(
                (
                    any::<u64>(),
                    any::<u64>(),
                    any::<u64>(),
                    any::<u64>(),
                    -4.0f64..4.0,
                    -4.0f64..4.0,
                ),
                0..8,
            )
            .prop_map(|entries| {
                let mut map: BTreeMap<([u64; 2], [u64; 2]), Complex64> = BTreeMap::new();
                for (x0, x1, z0, z1, re, im) in entries {
                    map.insert(([x0, x1], [z0, z1]), Complex64::new(re, im));
                }
                let mut x = Vec::with_capacity(map.len());
                let mut z = Vec::with_capacity(map.len());
                let mut coeff = Vec::with_capacity(map.len());
                for ((kx, kz), c) in map {
                    x.push(kx);
                    z.push(kz);
                    coeff.push(c);
                }
                PauliSum::<2>::from_sorted_columns(x, z, coeff, 128)
            })
        }

        proptest! {
            #[test]
            fn add_is_associative(
                a in arb_pauli_sum_w2(),
                b in arb_pauli_sum_w2(),
                c in arb_pauli_sum_w2(),
            ) {
                let left = a.add(&b).add(&c);
                let right = a.add(&b.add(&c));
                left.assert_invariants();
                right.assert_invariants();
                let (lx, lz, lc) = left.to_arrays();
                let (rx, rz, rc) = right.to_arrays();
                prop_assert_eq!(lx, rx);
                prop_assert_eq!(lz, rz);
                prop_assert_eq!(lc.len(), rc.len());
                for k in 0..lc.len() {
                    let diff = lc[k] - rc[k];
                    prop_assert!(
                        diff.norm() <= 1e-12,
                        "coeff mismatch at idx {}: lhs={:?} rhs={:?}",
                        k, lc[k], rc[k]
                    );
                }
            }
        }
    }
}
