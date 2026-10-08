//! The GF(2)-linear bucket function `h(v) = H·v`. See ARCHITECTURE.md §Hash.

use crate::pauli_string::PauliString;
use crate::rng::{mix64, SPLITMIX_GAMMA};

/// Maximum number of bucket bits, i.e. `B ≤ 2^22 = 4_194_304` buckets.
/// Rows for all `B_MAX_BITS` bits are generated up front so that [`Gf2Hash::refine`] is free: the active hash is always a prefix of the same fixed matrix, so refinement is a single parity pass rather than a re-hash.
pub const B_MAX_BITS: u8 = 22;

/// Maximum number of partition bits, i.e. `P ≤ 2^6 = 64` partitions.
/// Partitions are the coarse split of a sum across independent workers (see [`PartitionRows`]); the bucket bits of [`Gf2Hash`] refine within one partition. The cap is deliberately small: `P` tracks hardware parallelism (NUMA domains in-process, nodes under a distributed run), not term count.
pub const P_MAX_BITS: u8 = 6;

/// Salt mixed into a [`PartitionRows`] seed before row generation.
/// Without it, `PartitionRows::from_seed(n, p, s)` and `Gf2Hash::new(n, b, s)` would draw from the same stream and the partition rows would be the hash's first `p` rows — dependent by construction, and the global bucket `(part(v), loc(v))` would only have `max(p, b)` bits of entropy instead of `p + b`.
/// The seed is mixed through a splitmix64 finalizer before the salt so no particular seed value can cancel it (see `research/FINDINGS.md`).
const PARTITION_ROW_SALT: u64 = 0xD1B5_4A32_D192_ED03;

/// Word `word` of the `half` (0 = x, 1 = z) of row `row`, draw `attempt`: splitmix64's output at the stream position encoding that tuple.
/// Row words must not be successive outputs of a GF(2)-linear generator such as xorshift (ARCHITECTURE.md §Hash).
#[inline]
fn row_word(seed: u64, row: usize, attempt: u32, word: usize, half: u64) -> u64 {
    debug_assert!(row < 1 << 16 && word < 1 << 15);
    let position = ((attempt as u64) << 32) | ((row as u64) << 16) | ((word as u64) << 1) | half;
    mix64(seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(position.wrapping_add(1))))
}

/// Mask of the live qubit bits in word `word`, given `num_qubits` total.
/// Same construction as [`PauliString::is_within`]; kept separate because that method folds the words together and we need them individually.
#[inline]
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

/// `n_rows` rows of `(x-mask, z-mask)` drawn from `seed` and masked to the live qubit columns.
/// A row that masks to all-zero would waste a bit; at `num_qubits = 1` the chance is 1/4 per row, so it is redrawn, except at `num_qubits == 0` where every row is legitimately zero.
fn draw_rows<const W: usize>(
    num_qubits: usize,
    n_rows: usize,
    seed: u64,
    exclude_x: &[u64; W],
    exclude_z: &[u64; W],
) -> (Vec<[u64; W]>, Vec<[u64; W]>) {
    let mut rows_x: Vec<[u64; W]> = Vec::with_capacity(n_rows);
    let mut rows_z: Vec<[u64; W]> = Vec::with_capacity(n_rows);
    let has_live_columns = num_qubits > 0;
    for row in 0..n_rows {
        let mut attempt = 0u32;
        let (rx, rz) = loop {
            let mut rx = [0u64; W];
            let mut rz = [0u64; W];
            let mut any = false;
            for w in 0..W {
                let mask = word_mask(num_qubits, w);
                rx[w] = row_word(seed, row, attempt, w, 0) & mask & !exclude_x[w];
                rz[w] = row_word(seed, row, attempt, w, 1) & mask & !exclude_z[w];
                any |= (rx[w] | rz[w]) != 0;
            }
            if any || !has_live_columns {
                break (rx, rz);
            }
            attempt += 1;
        };
        rows_x.push(rx);
        rows_z.push(rz);
    }
    (rows_x, rows_z)
}

/// A GF(2)-linear hash from Pauli keys to bucket indices.
///
/// `h(v) = H·v` for a fixed dense random `H ∈ GF(2)^{b × 2n}`, where `v = (x, z)` is the symplectic key of a [`PauliString`]. Bit `i` of the result is the parity of `(x & rows_x[i]) ^ (z & rows_z[i])`.
///
/// # Why dense and random
///
/// A coordinate projection (bucket = some chosen key bits) is GF(2)-linear too, but wrong here: `WeightCutoff` truncation keeps sums low-weight, so the chosen coordinates are almost always zero and everything lands in bucket 0. A dense random `H` is a universal hash family instead — bucket load stays balanced independent of the input's structure. See ARCHITECTURE.md §Hash, and the `occupancy_*` tests below, which pin the property.
///
/// Cost is `b × 2W` AND + popcount-parity operations, evaluated only at ingestion and at rehash — never in the propagation loop, where buckets are tracked structurally by XOR instead.
///
/// # Examples
///
/// ```
/// use paulistrings::Gf2Hash;
/// use paulistrings::PauliString;
///
/// let h = Gf2Hash::<1>::new(64, 6, 0xC0FFEE);
/// assert_eq!(h.num_buckets(), 64);
///
/// // Linearity: h(v ^ w) == h(v) ^ h(w).
/// let v = PauliString::<1>::x(3);
/// let w = PauliString::<1>::z(11);
/// let xor = PauliString::<1> { x: [v.x[0] ^ w.x[0]], z: [v.z[0] ^ w.z[0]] };
/// assert_eq!(h.bucket_of_pauli(&xor), h.bucket_of_pauli(&v) ^ h.bucket_of_pauli(&w));
/// ```
#[derive(Clone, Debug)]
pub struct Gf2Hash<const W: usize> {
    /// X-part of each row of `H`, masked to the live qubit columns.
    rows_x: Vec<[u64; W]>,
    /// Z-part of each row of `H`, masked to the live qubit columns.
    rows_z: Vec<[u64; W]>,
    /// Active prefix length: `B = 1 << bits` buckets. `0 ≤ bits ≤ B_MAX_BITS`.
    bits: u8,
    /// Seed the rows were generated from. Kept so the hash is reproducible and so two sums can be checked for compatibility.
    seed: u64,
    /// Qubit count the rows were masked against.
    num_qubits: usize,
}

impl<const W: usize> Gf2Hash<W> {
    /// Build a hash over `num_qubits` qubits with `bits` active bucket bits.
    /// Rows are generated deterministically from `seed`, so two `Gf2Hash` values with the same `(num_qubits, seed)` are identical and their sums are combinable; the rows do not depend on `W`.
    ///
    /// # Panics
    ///
    /// Panics if `bits > B_MAX_BITS`, or in debug builds if `num_qubits > 64 · W`.
    pub fn new(num_qubits: usize, bits: u8, seed: u64) -> Self {
        assert!(
            bits <= B_MAX_BITS,
            "Gf2Hash: bits {bits} exceeds B_MAX_BITS {B_MAX_BITS}",
        );
        debug_assert!(num_qubits <= 64 * W);

        let (rows_x, rows_z) =
            draw_rows::<W>(num_qubits, B_MAX_BITS as usize, seed, &[0; W], &[0; W]);

        Self {
            rows_x,
            rows_z,
            bits,
            seed,
            num_qubits,
        }
    }

    /// Number of active bucket bits.
    #[inline]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Number of buckets, `1 << bits()`.
    #[inline]
    pub fn num_buckets(&self) -> usize {
        1usize << self.bits
    }

    /// The seed the rows were generated from.
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The qubit count the rows were masked against.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// `h(v)` for a key given as separate `x` and `z` words.
    /// The SoA-friendly entry point: the engine and the bucketed sum hold parallel `x`/`z` columns, not [`PauliString`] values.
    #[inline]
    pub fn bucket_of(&self, x: &[u64; W], z: &[u64; W]) -> u32 {
        let mut acc: u32 = 0;
        for i in 0..self.bits as usize {
            acc |= self.row_parity(x, z, i as u8) << i;
        }
        acc
    }

    /// One row of `H·v`: the parity of `(x & rows_x[row]) ^ (z & rows_z[row])`, as `0` or `1`.
    /// [`Self::bucket_of`] is this evaluated for every row `0..bits` and assembled into one `u32`; a caller that needs only the new bit [`Self::refine`] just introduced can get it in `O(2W)` instead of paying `O(bits · 2W)` for the whole prefix.
    /// Popcount parity is GF(2)-linear, so the masked words are XOR-folded first and reduced by a single `count_ones` rather than one per word: `2W` popcounts become 1.
    #[inline]
    pub(crate) fn row_parity(&self, x: &[u64; W], z: &[u64; W], row: u8) -> u32 {
        let rx = &self.rows_x[row as usize];
        let rz = &self.rows_z[row as usize];
        let mut acc: u64 = 0;
        for w in 0..W {
            acc ^= (x[w] & rx[w]) ^ (z[w] & rz[w]);
        }
        acc.count_ones() & 1
    }

    /// `h(v)` for a [`PauliString`]. Convenience wrapper over [`Self::bucket_of`].
    ///
    /// Also the form used for a channel's delta vectors at prepare time, where
    /// `h(d)` is computed once per layer rather than per term.
    #[inline]
    pub fn bucket_of_pauli(&self, p: &PauliString<W>) -> u32 {
        self.bucket_of(&p.x, &p.z)
    }

    /// Double the bucket count: `B → 2B`.
    /// Because the active hash is a prefix of a fixed matrix, refining splits each existing bucket in two and the within-bucket order is inherited by both halves — an `O(n)` parity pass with no re-sorting.
    ///
    /// # Panics
    ///
    /// Panics if already at `B_MAX_BITS`.
    #[inline]
    pub fn refine(&mut self) {
        assert!(
            self.bits < B_MAX_BITS,
            "Gf2Hash::refine: already at B_MAX_BITS {B_MAX_BITS}",
        );
        self.bits += 1;
    }

    /// Halve the bucket count: `B → B/2`. Merges bucket pairs `(2i, 2i+1)`.
    ///
    /// # Panics
    ///
    /// Panics if already at a single bucket.
    #[inline]
    pub fn coarsen(&mut self) {
        assert!(
            self.bits > 0,
            "Gf2Hash::coarsen: already at a single bucket"
        );
        self.bits -= 1;
    }

    /// `true` if `other` was generated with the same rows, so sums partitioned by the two can be combined (after matching `bits`).
    #[inline]
    pub(crate) fn same_rows_as(&self, other: &Self) -> bool {
        self.seed == other.seed && self.num_qubits == other.num_qubits
    }

    /// Row `i` of `H` as `(x-mask, z-mask)`, already masked to the live columns.
    /// Rows for all [`B_MAX_BITS`] bits exist regardless of the active prefix length, so `i` may exceed [`Self::bits`]; a caller that means "the rows currently in use" must restrict itself to `0..bits()` — [`PartitionRows::is_independent_of`] is the one such caller.
    ///
    /// # Panics
    ///
    /// Panics if `i >= B_MAX_BITS`.
    #[inline]
    pub(crate) fn row(&self, i: usize) -> ([u64; W], [u64; W]) {
        (self.rows_x[i], self.rows_z[i])
    }
}

/// The `p` designated partition rows that split a sum across `P = 2^p` independent partitions.
///
/// A global bucket is the pair `(part(v), loc(v))`: `part(v) = P·v` from these rows, and `loc(v) = H·v` from an unchanged [`Gf2Hash`]. Both maps are GF(2)-linear, so `part(v ⊕ d) = part(v) ⊕ part(d)` exactly as in ARCHITECTURE.md §Bucketing — a channel's partition deltas are as statically predictable as its bucket deltas.
///
/// # Why a separate matrix
///
/// `Gf2Hash`'s active rows are a prefix of one fixed matrix that grows and shrinks with the term count; partition rows must not move when it does, or a term would change owner mid-run. Drawing them from a salted seed ([`Self::from_seed`]) keeps them independent of the refinement-row stream at every bucket count; [`Self::is_independent_of`] checks the resulting matrix actually has full rank.
///
/// # Examples
///
/// ```
/// use paulistrings::PartitionRows;
/// use paulistrings::PauliString;
///
/// let p = PartitionRows::<1>::from_seed(64, 2, 0xC0FFEE);
/// assert_eq!(p.num_partitions(), 4);
///
/// // Linearity: part(v ^ w) == part(v) ^ part(w).
/// let v = PauliString::<1>::x(3);
/// let w = PauliString::<1>::z(11);
/// let xor = PauliString::<1> { x: [v.x[0] ^ w.x[0]], z: [v.z[0] ^ w.z[0]] };
/// assert_eq!(p.partition_of_pauli(&xor), p.partition_of_pauli(&v) ^ p.partition_of_pauli(&w));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionRows<const W: usize> {
    /// X-part of each partition row, masked to the live qubit columns.
    rows_x: Vec<[u64; W]>,
    /// Z-part of each partition row, masked to the live qubit columns.
    rows_z: Vec<[u64; W]>,
    /// Number of rows: `P = 1 << bits` partitions. `0 ≤ bits ≤ P_MAX_BITS`.
    bits: u8,
    /// Qubit count the rows were masked against.
    num_qubits: usize,
}

impl<const W: usize> PartitionRows<W> {
    /// Panics unless the rows split a sum over `num_qubits` qubits into `partitions` partitions; debug builds also check them independent of `hash`, whose correlation with the split costs load balance.
    pub(crate) fn assert_splits(&self, hash: &Gf2Hash<W>, num_qubits: usize, partitions: usize) {
        assert_eq!(
            self.num_partitions(),
            partitions,
            "partition rows name {} partitions for a split into {partitions}",
            self.num_partitions(),
        );
        assert_eq!(
            self.num_qubits(),
            num_qubits,
            "partition rows are for {} qubits, the sum for {num_qubits}",
            self.num_qubits(),
        );
        debug_assert!(
            self.is_independent_of(hash),
            "partition rows are dependent on the bucket hash rows — the split will correlate \
             with the bucket partition and load-balance badly",
        );
    }

    /// Draw `bits` partition rows deterministically from `seed`.
    /// The seed is salted, so these rows are unrelated to `Gf2Hash::new(num_qubits, _, seed)`'s rows at any bucket count. Column masking and the all-zero-row retry match [`Gf2Hash::new`].
    ///
    /// # Panics
    ///
    /// Panics if `bits > P_MAX_BITS`, or in debug builds if `num_qubits > 64 · W`.
    pub fn from_seed(num_qubits: usize, bits: u8, seed: u64) -> Self {
        Self::from_seed_excluding(num_qubits, bits, seed, &[0; W], &[0; W])
    }

    /// [`Self::from_seed`] with the key coordinates in `(exclude_x, exclude_z)` cleared from every row, so no partition label reads them.
    /// Two keys differing only there always share a partition, which is what keeps [`PauliSum::rotated_overlap`](crate::PauliSum::rotated_overlap)'s classes on one rank: pass [`RotationAxis::flip_mask`](crate::RotationAxis::flip_mask).
    /// With nothing excluded the rows are exactly [`Self::from_seed`]'s.
    ///
    /// # Panics
    ///
    /// As [`Self::from_seed`], and if `bits > 0` while the exclusion covers every live column.
    pub fn from_seed_excluding(
        num_qubits: usize,
        bits: u8,
        seed: u64,
        exclude_x: &[u64; W],
        exclude_z: &[u64; W],
    ) -> Self {
        assert!(
            bits <= P_MAX_BITS,
            "PartitionRows: bits {bits} exceeds P_MAX_BITS {P_MAX_BITS}",
        );
        debug_assert!(num_qubits <= 64 * W);

        // As in `Gf2Hash::new`: `num_qubits == 0` has a single key, so every row is legitimately zero and the retry must not spin.
        let keeps_a_column = (0..W).any(|w| {
            let mask = word_mask(num_qubits, w);
            (mask & !exclude_x[w]) | (mask & !exclude_z[w]) != 0
        });
        assert!(
            bits == 0 || num_qubits == 0 || keeps_a_column,
            "PartitionRows::from_seed_excluding: the exclusion covers every column",
        );
        let (rows_x, rows_z) = draw_rows::<W>(
            num_qubits,
            bits as usize,
            mix64(seed) ^ PARTITION_ROW_SALT,
            exclude_x,
            exclude_z,
        );

        Self {
            rows_x,
            rows_z,
            bits,
            num_qubits,
        }
    }

    /// Build from explicit rows — the hash-tuning hook.
    ///
    /// Rows are masked to the live qubit columns on the way in, so
    /// [`Self::rows`] returns the masked form, not the argument verbatim.
    ///
    /// # Panics
    ///
    /// Panics if `rows_x` and `rows_z` differ in length, if there are more than [`P_MAX_BITS`] rows, or if any row masks to all-zero while `num_qubits > 0` (wasting half the partitions). Panics in debug builds if `num_qubits > 64 · W`.
    pub fn from_rows(num_qubits: usize, rows_x: Vec<[u64; W]>, rows_z: Vec<[u64; W]>) -> Self {
        assert_eq!(
            rows_x.len(),
            rows_z.len(),
            "PartitionRows::from_rows: row count mismatch, {} x-rows vs {} z-rows",
            rows_x.len(),
            rows_z.len(),
        );
        assert!(
            rows_x.len() <= P_MAX_BITS as usize,
            "PartitionRows: bits {} exceeds P_MAX_BITS {P_MAX_BITS}",
            rows_x.len(),
        );
        debug_assert!(num_qubits <= 64 * W);

        let bits = rows_x.len() as u8;
        let mut rows_x = rows_x;
        let mut rows_z = rows_z;
        let has_live_columns = num_qubits > 0;
        for i in 0..bits as usize {
            let mut any = false;
            for w in 0..W {
                let mask = word_mask(num_qubits, w);
                rows_x[i][w] &= mask;
                rows_z[i][w] &= mask;
                any |= (rows_x[i][w] | rows_z[i][w]) != 0;
            }
            assert!(
                any || !has_live_columns,
                "PartitionRows::from_rows: row {i} masks to zero",
            );
        }

        Self {
            rows_x,
            rows_z,
            bits,
            num_qubits,
        }
    }

    /// Rows that label a qubit cut: `log2(blocks.len())` z-only rows giving block `b` the label `b`.
    ///
    /// Row `i` has its z-bits set on exactly the qubits of the blocks whose index has bit `i` set, and no x-bits at all. Since `part` is GF(2) linear and reads the z-half only, a term's partition label is the XOR of the labels of the blocks it has odd z-weight in. Qubits in no block behave as if they were in block 0; blocks need not cover every qubit, but they must be disjoint.
    ///
    /// # Why this shape
    ///
    /// This is the geometric row set for 1- and 2-local Pauli generators (ARCHITECTURE.md §Partitioning): a generator with no z-bits has `part = 0` and is local under any cut, and a `ZZ(i, j)` bond is remote exactly when the edge crosses between blocks with different labels.
    ///
    /// # Examples
    ///
    /// ```
    /// use paulistrings::PartitionRows;
    /// use paulistrings::PauliString;
    ///
    /// // A chain of four qubits bisected: {0,1} | {2,3}.
    /// let rows = PartitionRows::<1>::cut(4, &[vec![0, 1], vec![2, 3]]);
    /// assert_eq!(rows.num_partitions(), 2);
    ///
    /// // The transverse-field generator is x-only: local.
    /// assert_eq!(rows.partition_of_pauli(&PauliString::<1>::x(2)), 0);
    /// // The bond ZZ(1,2) crosses the cut; ZZ(0,1) does not.
    /// assert_eq!(rows.partition_of(&[0], &[0b0110]), 1);
    /// assert_eq!(rows.partition_of(&[0], &[0b0011]), 0);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `blocks.len()` is not a power of two, if it exceeds `2^P_MAX_BITS`, if a qubit is `>= num_qubits` or appears in two blocks, or if some row would be all-zero (no qubit lies in any block whose label has that bit set — the same condition [`Self::from_rows`] rejects).
    pub fn cut(num_qubits: usize, blocks: &[Vec<u32>]) -> Self {
        assert!(
            blocks.len().is_power_of_two(),
            "PartitionRows::cut: block count {} is not a power of two",
            blocks.len(),
        );
        let bits = blocks.len().trailing_zeros() as u8;
        assert!(
            bits <= P_MAX_BITS,
            "PartitionRows: bits {bits} exceeds P_MAX_BITS {P_MAX_BITS}",
        );
        debug_assert!(num_qubits <= 64 * W);

        let mut seen = vec![false; num_qubits];
        let mut rows_z = vec![[0u64; W]; bits as usize];
        for (b, qubits) in blocks.iter().enumerate() {
            for &q in qubits {
                let qi = q as usize;
                assert!(
                    qi < num_qubits,
                    "PartitionRows::cut: qubit {q} in block {b} is outside 0..{num_qubits}",
                );
                assert!(
                    !seen[qi],
                    "PartitionRows::cut: blocks must be disjoint, qubit {q} appears twice",
                );
                seen[qi] = true;
                for (i, row) in rows_z.iter_mut().enumerate() {
                    if (b >> i) & 1 == 1 {
                        row[qi / 64] |= 1u64 << (qi % 64);
                    }
                }
            }
        }
        for (i, row) in rows_z.iter().enumerate() {
            assert!(
                row.iter().any(|w| *w != 0),
                "PartitionRows::cut: row {i} is empty — no qubit lies in a block \
                 whose label has bit {i} set",
            );
        }

        Self::from_rows(num_qubits, vec![[0u64; W]; bits as usize], rows_z)
    }

    /// The trivial partitioning: one partition, no rows.
    #[inline]
    pub fn none(num_qubits: usize) -> Self {
        Self {
            rows_x: Vec::new(),
            rows_z: Vec::new(),
            bits: 0,
            num_qubits,
        }
    }

    /// Number of partition bits.
    #[inline]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Number of partitions, `1 << bits()`.
    #[inline]
    pub fn num_partitions(&self) -> usize {
        1usize << self.bits
    }

    /// The qubit count the rows were masked against.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// `part(v)` for a key given as separate `x` and `z` words.
    /// Bit `i` is the parity of `(x & rows_x[i]) ^ (z & rows_z[i])`. The identity key maps to partition 0, the same wart `h` has.
    #[inline]
    pub fn partition_of(&self, x: &[u64; W], z: &[u64; W]) -> u32 {
        let mut acc: u32 = 0;
        for i in 0..self.bits as usize {
            let rx = &self.rows_x[i];
            let rz = &self.rows_z[i];
            // XOR-fold then one popcount; see `Gf2Hash::row_parity`.
            let mut fold: u64 = 0;
            for w in 0..W {
                fold ^= (x[w] & rx[w]) ^ (z[w] & rz[w]);
            }
            acc |= (fold.count_ones() & 1) << i;
        }
        acc
    }

    /// `part(v)` for a [`PauliString`]. Convenience wrapper over
    /// [`Self::partition_of`].
    #[inline]
    pub fn partition_of_pauli(&self, p: &PauliString<W>) -> u32 {
        self.partition_of(&p.x, &p.z)
    }

    /// The rows as `(x-masks, z-masks)`, already masked to the live columns.
    #[inline]
    pub fn rows(&self) -> (&[[u64; W]], &[[u64; W]]) {
        (&self.rows_x, &self.rows_z)
    }

    /// `true` if no row reads a coordinate in `(mask_x, mask_z)`, i.e. keys differing only there always share a partition.
    pub fn avoids(&self, mask_x: &[u64; W], mask_z: &[u64; W]) -> bool {
        self.rows_x
            .iter()
            .zip(&self.rows_z)
            .all(|(rx, rz)| (0..W).all(|w| rx[w] & mask_x[w] == 0 && rz[w] & mask_z[w] == 0))
    }

    /// `true` if the partition rows and `hash`'s active rows are jointly GF(2)-independent over the `2·num_qubits` key columns.
    /// Equivalent to: the global bucket `(part(v), loc(v))` really has `bits() + hash.bits()` bits of entropy, so no partition is a function of the local bucket index or vice versa. Only rows `0..hash.bits()` are considered, so at `hash.bits() == 0` this reduces to the partition rows being independent among themselves.
    pub fn is_independent_of(&self, hash: &Gf2Hash<W>) -> bool {
        let n = self.bits as usize + hash.bits() as usize;
        let mut rows: Vec<KeyRow<W>> = Vec::with_capacity(n);
        for i in 0..self.bits as usize {
            rows.push((self.rows_x[i], self.rows_z[i]));
        }
        for i in 0..hash.bits() as usize {
            rows.push(hash.row(i));
        }
        gf2_rank_wide(&rows) == n
    }
}

/// One row of the key space as `(x-masks, z-masks)` — a vector over the
/// `2·W·64` symplectic columns.
type KeyRow<const W: usize> = ([u64; W], [u64; W]);

/// GF(2) rank of rows over the `2·W·64`-column key space.
/// Plain Gaussian elimination on a handful of rows (at most `P_MAX_BITS + B_MAX_BITS`), used only at construction/validation time, never in a loop that sees terms.
fn gf2_rank_wide<const W: usize>(rows: &[KeyRow<W>]) -> usize {
    // (leading column, reduced row), one entry per pivot found so far.
    let mut pivots: Vec<(usize, KeyRow<W>)> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut cur = *row;
        'reduce: loop {
            let Some(lead) = leading_column(&cur) else {
                break; // reduced to zero: dependent on the pivots so far.
            };
            for (pl, prow) in pivots.iter() {
                if *pl == lead {
                    for w in 0..W {
                        cur.0[w] ^= prow.0[w];
                        cur.1[w] ^= prow.1[w];
                    }
                    continue 'reduce;
                }
            }
            pivots.push((lead, cur));
            break;
        }
    }
    pivots.len()
}

/// Index of the highest set column of a key-space row, or `None` if it is zero.
/// Columns `0..W·64` are the `x` half and `W·64..2·W·64` the `z` half; only the ordering matters, not the particular convention.
#[inline]
fn leading_column<const W: usize>(row: &KeyRow<W>) -> Option<usize> {
    for w in (0..W).rev() {
        if row.1[w] != 0 {
            return Some(W * 64 + w * 64 + (63 - row.1[w].leading_zeros() as usize));
        }
    }
    for w in (0..W).rev() {
        if row.0[w] != 0 {
            return Some(w * 64 + (63 - row.0[w].leading_zeros() as usize));
        }
    }
    None
}

#[cfg(test)]
mod tests;
