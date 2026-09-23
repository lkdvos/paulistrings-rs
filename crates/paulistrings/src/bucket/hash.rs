//! The GF(2)-linear bucket function `h(v) = H·v`. See ARCHITECTURE.md §Hash.

use crate::pauli_string::PauliString;

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

/// splitmix64's output finalizer: a bijection on `u64` with full avalanche.
#[inline]
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// splitmix64's increment, the odd constant `⌊2^64/φ⌋`.
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

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
                rx[w] = row_word(seed, row, attempt, w, 0) & mask;
                rz[w] = row_word(seed, row, attempt, w, 1) & mask;
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
/// use paulistrings::bucket::Gf2Hash;
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

        let (rows_x, rows_z) = draw_rows::<W>(num_qubits, B_MAX_BITS as usize, seed);

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
    /// Panics if already at [`B_MAX_BITS`].
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
    pub fn same_rows_as(&self, other: &Self) -> bool {
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
/// use paulistrings::bucket::PartitionRows;
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
    /// Draw `bits` partition rows deterministically from `seed`.
    /// The seed is salted, so these rows are unrelated to `Gf2Hash::new(num_qubits, _, seed)`'s rows at any bucket count. Column masking and the all-zero-row retry match [`Gf2Hash::new`].
    ///
    /// # Panics
    ///
    /// Panics if `bits > P_MAX_BITS`, or in debug builds if `num_qubits > 64 · W`.
    pub fn from_seed(num_qubits: usize, bits: u8, seed: u64) -> Self {
        assert!(
            bits <= P_MAX_BITS,
            "PartitionRows: bits {bits} exceeds P_MAX_BITS {P_MAX_BITS}",
        );
        debug_assert!(num_qubits <= 64 * W);

        let (rows_x, rows_z) =
            draw_rows::<W>(num_qubits, bits as usize, mix64(seed) ^ PARTITION_ROW_SALT);

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
    /// use paulistrings::bucket::PartitionRows;
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
mod tests {
    use super::*;
    use crate::test_support::Xs64;

    /// XOR two Pauli keys — the group operation on the key space.
    fn xor<const W: usize>(a: &PauliString<W>, b: &PauliString<W>) -> PauliString<W> {
        let mut out = *a;
        for w in 0..W {
            out.x[w] ^= b.x[w];
            out.z[w] ^= b.z[w];
        }
        out
    }

    fn rand_key<const W: usize>(rng: &mut Xs64, num_qubits: usize) -> PauliString<W> {
        let mut p = PauliString::<W> {
            x: [0u64; W],
            z: [0u64; W],
        };
        for w in 0..W {
            let mask = word_mask(num_qubits, w);
            p.x[w] = rng.next_u64() & mask;
            p.z[w] = rng.next_u64() & mask;
        }
        p
    }

    fn low_weight_key<const W: usize>(
        rng: &mut Xs64,
        num_qubits: usize,
        weight: usize,
    ) -> PauliString<W> {
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
        p
    }

    // ---- range and the identity key ----

    #[test]
    fn bucket_is_within_range_w1() {
        let h = Gf2Hash::<1>::new(64, 7, 0xABCDEF);
        let mut rng = Xs64::new(1);
        for _ in 0..2000 {
            let p = rand_key::<1>(&mut rng, 64);
            assert!((h.bucket_of_pauli(&p) as usize) < h.num_buckets());
        }
    }

    #[test]
    fn bucket_is_within_range_w2() {
        let h = Gf2Hash::<2>::new(128, 11, 0xABCDEF);
        let mut rng = Xs64::new(2);
        for _ in 0..2000 {
            let p = rand_key::<2>(&mut rng, 128);
            assert!((h.bucket_of_pauli(&p) as usize) < h.num_buckets());
        }
    }

    #[test]
    fn identity_key_maps_to_bucket_zero() {
        // h(0) = 0 for any linear h. Documented wart: the identity string always lands in bucket 0.
        let h = Gf2Hash::<2>::new(128, 10, 0x1234);
        assert_eq!(h.bucket_of(&[0, 0], &[0, 0]), 0);
    }

    #[test]
    fn zero_bits_is_a_single_bucket() {
        let h = Gf2Hash::<1>::new(64, 0, 0x55);
        assert_eq!(h.num_buckets(), 1);
        let mut rng = Xs64::new(3);
        for _ in 0..100 {
            assert_eq!(h.bucket_of_pauli(&rand_key::<1>(&mut rng, 64)), 0);
        }
    }

    // ---- linearity: the property everything else rests on ----

    #[test]
    fn linearity_hand_checked_w1() {
        let h = Gf2Hash::<1>::new(64, 8, 0xFEED);
        let v = PauliString::<1>::x(3);
        let w = PauliString::<1>::z(11);
        assert_eq!(
            h.bucket_of_pauli(&xor(&v, &w)),
            h.bucket_of_pauli(&v) ^ h.bucket_of_pauli(&w)
        );
    }

    #[test]
    fn linearity_random_w1() {
        let h = Gf2Hash::<1>::new(64, 9, 0xFEED);
        let mut rng = Xs64::new(11);
        for _ in 0..2000 {
            let v = rand_key::<1>(&mut rng, 64);
            let w = rand_key::<1>(&mut rng, 64);
            assert_eq!(
                h.bucket_of_pauli(&xor(&v, &w)),
                h.bucket_of_pauli(&v) ^ h.bucket_of_pauli(&w),
            );
        }
    }

    #[test]
    fn linearity_random_w2_crosses_word_boundary() {
        let h = Gf2Hash::<2>::new(128, 12, 0xFEED);
        let mut rng = Xs64::new(12);
        for _ in 0..2000 {
            let v = rand_key::<2>(&mut rng, 128);
            let w = rand_key::<2>(&mut rng, 128);
            assert_eq!(
                h.bucket_of_pauli(&xor(&v, &w)),
                h.bucket_of_pauli(&v) ^ h.bucket_of_pauli(&w),
            );
        }
    }

    // ---- column masking ----

    #[test]
    fn bits_beyond_num_qubits_do_not_affect_the_bucket() {
        // 100 qubits in W=2: bits 100..128 are dead. Setting them must not move a term, or `PauliSum`'s `is_within` contract and the hash would disagree about which keys are distinguishable.
        let h = Gf2Hash::<2>::new(100, 10, 0x99);
        let mut rng = Xs64::new(21);
        for _ in 0..500 {
            let p = rand_key::<2>(&mut rng, 100);
            let mut polluted = p;
            // Set every dead bit in word 1 (qubits 100..128).
            let dead = !((1u64 << (100 - 64)) - 1);
            polluted.x[1] |= dead;
            polluted.z[1] |= dead;
            assert_eq!(h.bucket_of_pauli(&p), h.bucket_of_pauli(&polluted));
        }
    }

    #[test]
    fn rows_are_masked_at_a_mid_word_boundary() {
        // Directly: a key that is *only* out-of-range bits hashes to 0.
        let h = Gf2Hash::<2>::new(70, 12, 0x7A);
        let dead = !((1u64 << (70 - 64)) - 1);
        assert_eq!(h.bucket_of(&[0, dead], &[0, dead]), 0);
    }

    // ---- row_parity ----

    /// The XOR-fold in `row_parity` / `partition_of` must agree bitwise with the naive one-popcount-per-word form it replaced.
    /// This is an independent oracle: `row_parity` and `bucket_of` share the same fold, so checking them against each other would not catch a fold that is wrong in the same way twice.
    #[test]
    fn xor_fold_parity_matches_per_word_popcount() {
        fn naive<const W: usize>(x: &[u64; W], z: &[u64; W], rx: &[u64; W], rz: &[u64; W]) -> u32 {
            let mut parity: u32 = 0;
            for w in 0..W {
                parity ^= (x[w] & rx[w]).count_ones();
                parity ^= (z[w] & rz[w]).count_ones();
            }
            parity & 1
        }

        fn check<const W: usize>(num_qubits: usize, seed: u64) {
            let h = Gf2Hash::<W>::new(num_qubits, B_MAX_BITS, seed);
            let mut rng = Xs64::new(seed ^ 0x5EED);
            for _ in 0..500 {
                let p = rand_key::<W>(&mut rng, num_qubits);
                for row in 0..B_MAX_BITS {
                    assert_eq!(
                        h.row_parity(&p.x, &p.z, row),
                        naive(&p.x, &p.z, &h.rows_x[row as usize], &h.rows_z[row as usize]),
                        "W={W} row={row}: XOR-fold disagrees with per-word popcount"
                    );
                }
            }
        }

        check::<1>(64, 0xA11CE);
        check::<2>(128, 0xB0B);
        check::<4>(256, 0xC0FFEE);
        check::<8>(512, 0xD00D);
    }

    #[test]
    fn row_parity_matches_bucket_of_bit_extraction() {
        let h = Gf2Hash::<2>::new(128, B_MAX_BITS, 0xF00D);
        let mut rng = Xs64::new(81);
        for _ in 0..500 {
            let p = rand_key::<2>(&mut rng, 128);
            let full = h.bucket_of_pauli(&p);
            for row in 0..B_MAX_BITS {
                let bit = h.row_parity(&p.x, &p.z, row);
                assert!(bit == 0 || bit == 1, "row_parity must return 0 or 1");
                assert_eq!(
                    bit,
                    (full >> row) & 1,
                    "row {row} disagrees with bucket_of's bit extraction",
                );
            }
        }
    }

    // ---- refine / coarsen prefix consistency ----

    #[test]
    fn refine_preserves_the_low_bits() {
        let mut h = Gf2Hash::<2>::new(128, 6, 0xB0B);
        let mut rng = Xs64::new(31);
        let keys: Vec<PauliString<2>> = (0..500).map(|_| rand_key::<2>(&mut rng, 128)).collect();
        let before: Vec<u32> = keys.iter().map(|k| h.bucket_of_pauli(k)).collect();

        h.refine();
        assert_eq!(h.num_buckets(), 128);
        let mask = (1u32 << 6) - 1;
        for (k, &b) in keys.iter().zip(before.iter()) {
            // Refining splits each bucket in two: the new index agrees with the old one on the low `bits` bits, so within-bucket order is inherited by both halves.
            assert_eq!(h.bucket_of_pauli(k) & mask, b);
        }
    }

    #[test]
    fn coarsen_inverts_refine() {
        let mut h = Gf2Hash::<1>::new(64, 8, 0xCAFE);
        let mut rng = Xs64::new(41);
        let keys: Vec<PauliString<1>> = (0..500).map(|_| rand_key::<1>(&mut rng, 64)).collect();
        let before: Vec<u32> = keys.iter().map(|k| h.bucket_of_pauli(k)).collect();

        h.refine();
        h.coarsen();
        assert_eq!(h.bits(), 8);
        let after: Vec<u32> = keys.iter().map(|k| h.bucket_of_pauli(k)).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn coarsen_merges_bucket_pairs() {
        let mut h = Gf2Hash::<1>::new(64, 8, 0xDEAD);
        let mut rng = Xs64::new(51);
        let keys: Vec<PauliString<1>> = (0..500).map(|_| rand_key::<1>(&mut rng, 64)).collect();
        let fine: Vec<u32> = keys.iter().map(|k| h.bucket_of_pauli(k)).collect();

        h.coarsen();
        for (k, &f) in keys.iter().zip(fine.iter()) {
            // Dropping the top bit merges (b, b + B/2) — the pair that differs only in the bit being dropped.
            assert_eq!(h.bucket_of_pauli(k), f & ((1 << 7) - 1));
        }
    }

    #[test]
    #[should_panic(expected = "already at B_MAX_BITS")]
    fn refine_past_the_maximum_panics() {
        let mut h = Gf2Hash::<1>::new(64, B_MAX_BITS, 0x1);
        h.refine();
    }

    #[test]
    #[should_panic(expected = "already at a single bucket")]
    fn coarsen_below_one_bucket_panics() {
        let mut h = Gf2Hash::<1>::new(64, 0, 0x1);
        h.coarsen();
    }

    #[test]
    #[should_panic(expected = "exceeds B_MAX_BITS")]
    fn constructing_past_the_maximum_panics() {
        let _ = Gf2Hash::<1>::new(64, B_MAX_BITS + 1, 0x1);
    }

    // ---- reproducibility ----

    #[test]
    fn same_seed_gives_the_same_hash() {
        let a = Gf2Hash::<2>::new(128, 10, 0x5EED);
        let b = Gf2Hash::<2>::new(128, 10, 0x5EED);
        assert!(a.same_rows_as(&b));
        let mut rng = Xs64::new(61);
        for _ in 0..500 {
            let p = rand_key::<2>(&mut rng, 128);
            assert_eq!(a.bucket_of_pauli(&p), b.bucket_of_pauli(&p));
        }
    }

    #[test]
    fn different_seeds_give_different_hashes() {
        let a = Gf2Hash::<2>::new(128, 10, 0x5EED);
        let b = Gf2Hash::<2>::new(128, 10, 0x5EEE);
        assert!(!a.same_rows_as(&b));
        let mut rng = Xs64::new(71);
        let differs = (0..500)
            .map(|_| rand_key::<2>(&mut rng, 128))
            .filter(|p| a.bucket_of_pauli(p) != b.bucket_of_pauli(p))
            .count();
        // Two independent hashes agree on a given key with probability 2^-10.
        assert!(
            differs > 400,
            "expected most keys to hash differently, got {differs}/500"
        );
    }

    #[test]
    fn no_row_masks_to_zero_even_at_one_qubit() {
        // At num_qubits = 1 there are only 2 live columns, so a naive generator produces an all-zero (and therefore useless) row 1/4 of the time.
        let h = Gf2Hash::<1>::new(1, 2, 0x1);
        for i in 0..B_MAX_BITS as usize {
            assert!(
                (h.rows_x[i][0] | h.rows_z[i][0]) != 0,
                "row {i} masked to zero",
            );
        }
    }

    #[test]
    fn zero_qubits_is_degenerate_but_terminates() {
        // Every row is legitimately zero; construction must not spin forever.
        let h = Gf2Hash::<1>::new(0, 3, 0x1);
        assert_eq!(h.bucket_of(&[0], &[0]), 0);
    }

    // ---- row generation: no GF(2)-linear relation between row words ----

    /// One step of xorshift64 (13, 7, 17), a GF(2)-linear map `M` on `u64`.
    fn xorshift64_step(mut x: u64) -> u64 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }

    /// Row `j` of `M` as a mask: bit `j` of `M·v` is the parity of `xorshift64_row(j) & v`.
    fn xorshift64_row(j: u32) -> u64 {
        (0..64).fold(0u64, |row, k| {
            row | (((xorshift64_step(1u64 << k) >> j) & 1) << k)
        })
    }

    /// Consecutive outputs of a linear generator satisfy `rows_z = M·rows_x` word for word under every seed.
    #[test]
    fn row_words_are_not_one_xorshift_step_apart() {
        for seed in [0x1u64, 0x5EED, crate::bucket::sum::DEFAULT_HASH_SEED] {
            let h = Gf2Hash::<2>::new(128, B_MAX_BITS, seed);
            let p = PartitionRows::<2>::from_seed(128, P_MAX_BITS, seed);
            let (px, pz) = p.rows();
            let linked = h
                .rows_x
                .iter()
                .zip(&h.rows_z)
                .chain(px.iter().zip(pz))
                .flat_map(|(rx, rz)| (0..2).map(move |w| xorshift64_step(rx[w]) == rz[w]))
                .filter(|&linked| linked)
                .count();
            assert_eq!(linked, 0, "seed {seed:#x}: {linked} row words are M·x");
        }
    }

    /// `rows_z = M·rows_x` puts `d_j = (x = row_j(M), z = e_j)`, Pauli weight ≈ 6, in the kernel of every row, so `u` and `u ⊕ d_j` always share a bucket.
    /// A dense random `H` sends each to 0 with probability `2^-20` at 20 bits.
    #[test]
    fn xorshift_kernel_deltas_do_not_share_bucket_zero() {
        fn zeros<const W: usize>(seed: u64) -> usize {
            let h = Gf2Hash::<W>::new(64 * W, 20, seed);
            let mut n = 0;
            for w in 0..W {
                for j in 0..64u32 {
                    let mut x = [0u64; W];
                    let mut z = [0u64; W];
                    x[w] = xorshift64_row(j);
                    z[w] = 1u64 << j;
                    n += (h.bucket_of(&x, &z) == 0) as usize;
                }
            }
            n
        }
        let seeds = [0x1u64, 0x5EED, crate::bucket::sum::DEFAULT_HASH_SEED];
        let w1: usize = seeds.iter().map(|&s| zeros::<1>(s)).sum();
        let w2: usize = seeds.iter().map(|&s| zeros::<2>(s)).sum();
        assert!(w1 <= 1, "{w1}/192 kernel deltas hash to 0 at W=1");
        assert!(w2 <= 1, "{w2}/384 kernel deltas hash to 0 at W=2");
    }

    /// 64 rows over 64 qubits as a fingerprint must separate the 18 337 keys of weight ≤ 2; a random linear map collides on some pair with probability ~2^-37.
    #[test]
    fn a_64_row_fingerprint_is_injective_on_weight_two_keys() {
        let (rx, rz) = draw_rows::<1>(64, 64, crate::bucket::sum::DEFAULT_HASH_SEED);
        let image = |x: u64, z: u64| {
            (0..64).fold(0u64, |acc, i| {
                acc | ((((x & rx[i][0]) ^ (z & rz[i][0])).count_ones() as u64 & 1) << i)
            })
        };
        // (x, z) bits of X, Z and Y on one qubit.
        let paulis = [(1u64, 0u64), (0, 1), (1, 1)];
        let mut keys = vec![(0u64, 0u64)];
        for q in 0..64 {
            for (a, b) in paulis {
                keys.push((a << q, b << q));
            }
        }
        for q in 0..64 {
            for r in (q + 1)..64 {
                for (a, b) in paulis {
                    for (c, d) in paulis {
                        keys.push(((a << q) | (c << r), (b << q) | (d << r)));
                    }
                }
            }
        }
        assert_eq!(keys.len(), 1 + 64 * 3 + 2016 * 9);
        let images: std::collections::HashSet<u64> =
            keys.iter().map(|&(x, z)| image(x, z)).collect();
        assert_eq!(images.len(), keys.len(), "fingerprint collisions");
    }

    /// Row word `w` depends on `(seed, row, w, x-or-z)` alone, so the same `(num_qubits, seed)` gives the same rows at every width.
    #[test]
    fn rows_do_not_depend_on_the_width() {
        let seed = crate::bucket::sum::DEFAULT_HASH_SEED;
        for n in [1usize, 5, 64] {
            let h1 = Gf2Hash::<1>::new(n, 8, seed);
            let h2 = Gf2Hash::<2>::new(n, 8, seed);
            for i in 0..B_MAX_BITS as usize {
                let (x1, z1) = h1.row(i);
                assert_eq!(h2.row(i), ([x1[0], 0], [z1[0], 0]), "n={n} row {i}");
            }
            let p1 = PartitionRows::<1>::from_seed(n, P_MAX_BITS, seed);
            let p2 = PartitionRows::<2>::from_seed(n, P_MAX_BITS, seed);
            let widened = |rows: &[[u64; 1]]| rows.iter().map(|r| [r[0], 0]).collect::<Vec<_>>();
            assert_eq!(p2.rows().0, widened(p1.rows().0), "n={n} partition x-rows");
            assert_eq!(p2.rows().1, widened(p1.rows().1), "n={n} partition z-rows");
        }
        let h2 = Gf2Hash::<2>::new(128, 8, seed);
        let h4 = Gf2Hash::<4>::new(128, 8, seed);
        for i in 0..B_MAX_BITS as usize {
            let (x2, z2) = h2.row(i);
            let (x4, z4) = h4.row(i);
            assert_eq!(x4, [x2[0], x2[1], 0, 0], "row {i}");
            assert_eq!(z4, [z2[0], z2[1], 0, 0], "row {i}");
        }
    }

    // ---- occupancy: what guards the choice of a dense random H ----

    /// Bucket occupancy on low-weight keys, the physically relevant regime.
    /// This is the test that fails for a coordinate-projection `H`: weight-4 strings over 64 qubits leave any fixed handful of key coordinates zero almost always, so projection dumps nearly everything into bucket 0, whereas a dense random `H` spreads them.
    #[test]
    fn occupancy_is_balanced_on_low_weight_keys() {
        let num_qubits = 64;
        let h = Gf2Hash::<1>::new(num_qubits, 6, 0x0CC1);
        let b = h.num_buckets();

        let mut rng = Xs64::new(0xBA1);
        let mut seen = std::collections::HashSet::new();
        let mut counts = vec![0usize; b];
        let target = 8192usize;
        while seen.len() < target {
            let p = low_weight_key::<1>(&mut rng, num_qubits, 4);
            if seen.insert((p.x, p.z)) {
                counts[h.bucket_of_pauli(&p) as usize] += 1;
            }
        }

        let mean = target / b; // 128
        let max = *counts.iter().max().unwrap();
        let min = *counts.iter().min().unwrap();
        // Deterministic given the seeds, so these bounds are not flaky. A projection hash would put >99% of the mass in one bucket and blow the upper bound by two orders of magnitude.
        assert!(max < 2 * mean, "max load {max} vs mean {mean}");
        assert!(min > mean / 2, "min load {min} vs mean {mean}");
    }

    // ---- rank of `h` on a channel's delta space ----
    //
    // A channel supported on qubits `{i, j}` has a 4-dimensional key-delta space `span{X_i, Z_i, X_j, Z_j}`; the engine's coset dimension is `r = rank(h(D))` (`engine::coset::Gf2Span::r`), and the per-run sort's comparison count collapses to its floor exactly when `r` is full (4).
    // `r` is not a property of the channel alone: it depends on which rows `H` happens to have, so it moves with the hash seed and the support, but not with `W`. See `research/FINDINGS.md`.

    /// Occupancy balance is not the whole story: a dense random `H` can still fail to separate a two-qubit channel's four delta generators, so two distinct local deltas share one bucket delta.
    #[test]
    fn support_delta_rank_is_usually_full_but_not_always() {
        // Deterministic given the seed, so these counts are not flaky.
        let h = Gf2Hash::<2>::new(128, 7, crate::bucket::sum::DEFAULT_HASH_SEED);
        let mut deficient = 0usize;
        let mut total = 0usize;
        for i in 0..128u32 {
            for j in (i + 1)..128u32 {
                total += 1;
                if crate::test_support::support_delta_rank(&h, &[i, j]) < 4 {
                    deficient += 1;
                }
            }
        }
        // About one placement in six at the default seed and bucket-count floor (B = 128), 11% in expectation over seeds. The bound is loose on purpose: it pins the order of magnitude, the load-bearing fact, not the exact draw.
        assert_eq!(total, 8128);
        assert!(
            (200..2000).contains(&deficient),
            "expected O(10%) rank-deficient support pairs at 7 bucket bits, got {deficient}/{total}"
        );
    }

    /// Rank is monotone in the number of active bucket bits, since the active hash is a prefix of one fixed matrix: refining can only separate deltas that were colliding, never merge separated ones.
    #[test]
    fn support_delta_rank_is_monotone_in_bits() {
        for seed in [0x1u64, 0xBEEF, crate::bucket::sum::DEFAULT_HASH_SEED] {
            let mut last = 0usize;
            for bits in 0..=12u8 {
                let h = Gf2Hash::<2>::new(128, bits, seed);
                let r = crate::test_support::support_delta_rank(&h, &[0, 1]);
                assert!(
                    r >= last && r <= 4,
                    "seed {seed:#x}: rank went {last} -> {r} at bits {bits}"
                );
                assert!(r <= bits as usize, "rank {r} exceeds bits {bits}");
                last = r;
            }
        }
    }

    /// Rows do not depend on `W` (`rows_do_not_depend_on_the_width`), so neither does a support's delta-span rank.
    /// At the default seed the su4 probe's support `(0, 1)` is full-rank at every bucket count the engine's own policy reaches, and `(0, 7)` is one rank short at the floor, at both widths.
    /// Regenerated literals: the splitmix64 row draw replaced a pinned `(0, 1)` rank of 3 at `W = 1`.
    #[test]
    fn support_delta_rank_is_width_independent_at_the_default_seed() {
        use crate::test_support::support_delta_rank as rank;
        let seed = crate::bucket::sum::DEFAULT_HASH_SEED;
        for bits in 7..=9u8 {
            let w1 = Gf2Hash::<1>::new(64, bits, seed);
            let w2 = Gf2Hash::<2>::new(65, bits, seed);
            let w2_wide = Gf2Hash::<2>::new(128, bits, seed);
            assert_eq!(rank(&w1, &[0, 1]), 4, "W=1/q=64 at {bits} bits");
            assert_eq!(rank(&w2, &[0, 1]), 4, "W=2/q=65 at {bits} bits");
            assert_eq!(rank(&w2_wide, &[0, 1]), 4, "W=2/q=128 at {bits} bits");
        }
        assert_eq!(rank(&Gf2Hash::<1>::new(64, 7, seed), &[0, 7]), 3);
        assert_eq!(rank(&Gf2Hash::<2>::new(128, 7, seed), &[0, 7]), 3);
    }

    /// The mechanism: a support delta cannot reorder a bucket's key column exactly when `h` separates the support's delta space.
    /// The engine's per-run "rest" stream concatenates blocks `{v ⊕ d : v ∈ bucket}`, one per non-identity delta `d`. At full delta rank each bucket holds at most one of the `2^(2k)` local variants of any off-support pattern, so XOR-by-`d` preserves the column's order and every block arrives already ascending. One rank short and each bucket holds two such variants, adjacent in key order, and half the deltas invert every such pair, shattering the block into runs of ~2.
    #[test]
    fn support_delta_preserves_bucket_order_iff_the_delta_span_is_full_rank() {
        // Regenerated for the splitmix64 row draw: the deficient support at the default seed is now `(0, 7)`, not `(0, 1)`.
        assert!(!order_broken_by_some_delta::<2>(128, 7, &[0, 1]));
        assert!(order_broken_by_some_delta::<1>(64, 7, &[0, 7]));
    }

    /// Partition a closed key set under `h`, then check every non-identity support delta against every bucket's ascending key column; returns `true` if any delta reorders any bucket.
    /// The key set has to be closed (every off-support pattern paired with all `2^(2k)` local patterns), since that is the fixed point a repeated dense-PTM layer drives the sum to and the structure that puts local variants of one pattern in the same bucket when the rank is short. Random keys would essentially never contain such a pair.
    fn order_broken_by_some_delta<const W: usize>(
        num_qubits: usize,
        bits: u8,
        support: &[u32],
    ) -> bool {
        let h = Gf2Hash::<W>::new(num_qubits, bits, crate::bucket::sum::DEFAULT_HASH_SEED);
        // Enumerate the support's delta space: one bit per (qubit, x-or-z).
        let gens: Vec<PauliString<W>> = support
            .iter()
            .flat_map(|&q| [PauliString::<W>::x(q), PauliString::<W>::z(q)])
            .collect();
        let local = |combo: usize| -> PauliString<W> {
            let mut d = PauliString::<W> {
                x: [0u64; W],
                z: [0u64; W],
            };
            for (g, gen) in gens.iter().enumerate() {
                if combo >> g & 1 == 1 {
                    d = xor(&d, gen);
                }
            }
            d
        };
        let mut rng = Xs64::new(0xD17A);
        let mut buckets: Vec<Vec<([u64; W], [u64; W])>> = vec![Vec::new(); h.num_buckets()];
        for _ in 0..2_000 {
            // A random off-support pattern, then its whole local orbit.
            let mut rest = rand_key::<W>(&mut rng, num_qubits);
            for &q in support {
                let (w, bit) = ((q / 64) as usize, 1u64 << (q % 64));
                rest.x[w] &= !bit;
                rest.z[w] &= !bit;
            }
            for combo in 0..(1usize << gens.len()) {
                let p = xor(&rest, &local(combo));
                buckets[h.bucket_of_pauli(&p) as usize].push((p.x, p.z));
            }
        }
        for cols in buckets.iter_mut() {
            cols.sort_unstable();
            cols.dedup();
            for combo in 1..(1usize << gens.len()) {
                let d = local(combo);
                let translated: Vec<([u64; W], [u64; W])> = cols
                    .iter()
                    .map(|(x, z)| {
                        let mut kx = *x;
                        let mut kz = *z;
                        for w in 0..W {
                            kx[w] ^= d.x[w];
                            kz[w] ^= d.z[w];
                        }
                        (kx, kz)
                    })
                    .collect();
                if translated.windows(2).any(|w| w[1] < w[0]) {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn occupancy_is_balanced_on_dense_keys() {
        let h = Gf2Hash::<2>::new(128, 8, 0x0CC2);
        let b = h.num_buckets();
        let mut rng = Xs64::new(0xBA2);
        let mut counts = vec![0usize; b];
        let target = 32768usize;
        for _ in 0..target {
            counts[h.bucket_of_pauli(&rand_key::<2>(&mut rng, 128)) as usize] += 1;
        }
        let mean = target / b; // 128
        let max = *counts.iter().max().unwrap();
        let min = *counts.iter().min().unwrap();
        assert!(max < 2 * mean, "max load {max} vs mean {mean}");
        assert!(min > mean / 2, "min load {min} vs mean {mean}");
    }

    // ---- partition rows: the coarse prefix, independent of `H`'s refinement rows ----

    #[test]
    fn partition_of_hand_checked_w1() {
        // Two explicit rows over 8 qubits:
        //   row 0: x-mask 0b011, z-mask 0
        //   row 1: x-mask 0,     z-mask 0b101
        let p = PartitionRows::<1>::from_rows(8, vec![[0b11], [0]], vec![[0], [0b101]]);
        assert_eq!(p.bits(), 2);
        assert_eq!(p.num_partitions(), 4);
        assert_eq!(p.num_qubits(), 8);
        // X_0: bit 0 = parity(0b001 & 0b011) = 1, bit 1 = 0.
        assert_eq!(p.partition_of_pauli(&PauliString::<1>::x(0)), 1);
        // Z_0: bit 0 = 0, bit 1 = parity(0b001 & 0b101) = 1.
        assert_eq!(p.partition_of_pauli(&PauliString::<1>::z(0)), 2);
        // Y_0 = X_0 ⊕ Z_0.
        assert_eq!(p.partition_of(&[1], &[1]), 3);
        // X_0 X_1: parity(0b011 & 0b011) = 0.
        assert_eq!(p.partition_of(&[0b11], &[0]), 0);
        // Z_1: parity(0b010 & 0b101) = 0.
        assert_eq!(p.partition_of(&[0], &[0b10]), 0);
        // Z_2: parity(0b100 & 0b101) = 1.
        assert_eq!(p.partition_of(&[0], &[0b100]), 2);
    }

    #[test]
    fn partition_is_within_range_and_the_identity_key_is_partition_zero() {
        let p = PartitionRows::<2>::from_seed(128, P_MAX_BITS, 0xABCDEF);
        assert_eq!(p.num_partitions(), 1usize << P_MAX_BITS);
        // p(0) = 0 for any linear map — the same documented wart as `h`.
        assert_eq!(p.partition_of(&[0, 0], &[0, 0]), 0);
        let mut rng = Xs64::new(101);
        for _ in 0..2000 {
            let k = rand_key::<2>(&mut rng, 128);
            assert!((p.partition_of_pauli(&k) as usize) < p.num_partitions());
        }
    }

    #[test]
    fn zero_partition_bits_is_a_single_partition() {
        let p = PartitionRows::<1>::none(64);
        assert_eq!(p.bits(), 0);
        assert_eq!(p.num_partitions(), 1);
        assert_eq!(p.num_qubits(), 64);
        let (rx, rz) = p.rows();
        assert!(rx.is_empty() && rz.is_empty());
        let mut rng = Xs64::new(102);
        for _ in 0..200 {
            assert_eq!(p.partition_of_pauli(&rand_key::<1>(&mut rng, 64)), 0);
        }
        // `from_seed` at zero bits is the same object.
        assert_eq!(PartitionRows::<1>::from_seed(64, 0, 0x1234), p);
    }

    #[test]
    fn partition_bits_beyond_num_qubits_do_not_affect_the_partition() {
        let p = PartitionRows::<2>::from_seed(100, 3, 0x99);
        let mut rng = Xs64::new(103);
        let dead = !((1u64 << (100 - 64)) - 1);
        for _ in 0..500 {
            let k = rand_key::<2>(&mut rng, 100);
            let mut polluted = k;
            polluted.x[1] |= dead;
            polluted.z[1] |= dead;
            assert_eq!(p.partition_of_pauli(&k), p.partition_of_pauli(&polluted));
        }
    }

    // ---- construction ----

    #[test]
    fn from_seed_is_reproducible_and_seed_dependent() {
        let a = PartitionRows::<2>::from_seed(128, 4, 0x5EED);
        let b = PartitionRows::<2>::from_seed(128, 4, 0x5EED);
        assert_eq!(a, b);
        assert_ne!(a, PartitionRows::<2>::from_seed(128, 4, 0x5EEE));
    }

    #[test]
    fn partition_rows_are_salted_away_from_the_hash_rows() {
        // Drawn from the same seed, the partition rows must not simply be the hash's first rows — otherwise they would be dependent on `h` at every bucket count and `is_independent_of` could never hold.
        for seed in [0x1u64, 0x5EED, crate::bucket::sum::DEFAULT_HASH_SEED] {
            let p = PartitionRows::<2>::from_seed(128, P_MAX_BITS, seed);
            let h = Gf2Hash::<2>::new(128, P_MAX_BITS, seed);
            let (px, pz) = p.rows();
            for i in 0..P_MAX_BITS as usize {
                let (hx, hz) = h.row(i);
                assert!(
                    px[i] != hx || pz[i] != hz,
                    "seed {seed:#x}: partition row {i} equals hash row {i}"
                );
            }
        }
    }

    #[test]
    fn from_rows_round_trips_after_masking() {
        let rows_x = vec![[!0u64, !0u64], [0x1, 0x0]];
        let rows_z = vec![[0x0u64, 0x3], [0xF, 0x0]];
        let p = PartitionRows::<2>::from_rows(70, rows_x, rows_z);
        let live = (1u64 << (70 - 64)) - 1;
        let (rx, rz) = p.rows();
        assert_eq!(rx, [[!0u64, live], [0x1, 0x0]]);
        assert_eq!(rz, [[0x0u64, 0x3], [0xF, 0x0]]);
        assert_eq!(p.bits(), 2);
    }

    /// A distributed run needs one partition per rank, and one rank per node (rather than per
    /// NUMA domain) means fewer, bigger partitions for the same rank count is not the point —
    /// more nodes at a fixed granularity is. 64 partitions covers that without making the row
    /// count term-count-dependent, the thing `P_MAX_BITS` exists to avoid.
    #[test]
    fn partition_row_ceiling_covers_64_ranks() {
        // `from_seed` panics with "exceeds P_MAX_BITS" if the constant is still below 6.
        let p = PartitionRows::<2>::from_seed(127, 6, 0xFEED_1234);
        assert_eq!(p.num_partitions(), 64);
    }

    #[test]
    #[should_panic(expected = "exceeds P_MAX_BITS")]
    fn partition_from_seed_past_the_maximum_panics() {
        let _ = PartitionRows::<1>::from_seed(64, P_MAX_BITS + 1, 0x1);
    }

    #[test]
    #[should_panic(expected = "exceeds P_MAX_BITS")]
    fn partition_from_rows_past_the_maximum_panics() {
        let rows: Vec<[u64; 1]> = (0..=P_MAX_BITS as u64).map(|i| [i + 1]).collect();
        let _ = PartitionRows::<1>::from_rows(64, rows.clone(), rows);
    }

    #[test]
    #[should_panic(expected = "row count mismatch")]
    fn partition_from_rows_length_mismatch_panics() {
        let _ = PartitionRows::<1>::from_rows(64, vec![[0x1], [0x2]], vec![[0x1]]);
    }

    #[test]
    #[should_panic(expected = "row 1 masks to zero")]
    fn partition_from_rows_all_zero_row_panics() {
        // Row 1 is nonzero only outside the 8 live qubit columns.
        let _ = PartitionRows::<1>::from_rows(8, vec![[0x1], [1 << 20]], vec![[0x0], [1 << 30]]);
    }

    #[test]
    fn partition_from_rows_all_zero_row_is_fine_at_zero_qubits() {
        // Degenerate but legal: with no live columns every row is zero.
        let p = PartitionRows::<1>::from_rows(0, vec![[0x0]], vec![[0x0]]);
        assert_eq!(p.partition_of(&[0], &[0]), 0);
    }

    // ---- `cut`: z-only rows labelling blocks of qubits ----

    #[test]
    fn cut_two_blocks_is_one_z_row_over_the_second_block() {
        // 4 qubits, blocks {0,1} | {2,3}. One row, z-only, set on block 1.
        let p = PartitionRows::<1>::cut(4, &[vec![0, 1], vec![2, 3]]);
        assert_eq!(p.bits(), 1);
        assert_eq!(p.num_partitions(), 2);
        let (rx, rz) = p.rows();
        assert_eq!(rx, [[0u64]]);
        assert_eq!(rz, [[0b1100u64]]);

        // A term's label is the XOR of the labels of the blocks it has odd z-weight in. Z0 sits in block 0, label 0; Z2 in block 1, label 1.
        assert_eq!(p.partition_of_pauli(&PauliString::<1>::z(0)), 0);
        assert_eq!(p.partition_of_pauli(&PauliString::<1>::z(2)), 1);
        // X rotation generators are x-only, so a cut row never reads them.
        assert_eq!(p.partition_of_pauli(&PauliString::<1>::x(2)), 0);
        // Bond generators: ZZ(0,1) is inside block 0, ZZ(1,2) crosses the cut,
        // ZZ(2,3) is inside block 1 and so has even z-weight there.
        assert_eq!(p.partition_of(&[0], &[0b0011]), 0);
        assert_eq!(p.partition_of(&[0], &[0b0110]), 1);
        assert_eq!(p.partition_of(&[0], &[0b1100]), 0);
    }

    #[test]
    fn cut_four_blocks_labels_each_block_by_its_index() {
        // 8 qubits in four pairs; row `i` is set on the blocks whose index has bit `i` set, so a single-Z term lands on its own block's label.
        let p = PartitionRows::<1>::cut(8, &[vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]);
        assert_eq!(p.bits(), 2);
        let (rx, rz) = p.rows();
        assert_eq!(rx, [[0u64], [0u64]]);
        assert_eq!(rz, [[0b1100_1100u64], [0b1111_0000u64]]);
        for (q, want) in [
            (0u32, 0u32),
            (1, 0),
            (2, 1),
            (3, 1),
            (4, 2),
            (5, 2),
            (6, 3),
            (7, 3),
        ] {
            assert_eq!(
                p.partition_of_pauli(&PauliString::<1>::z(q)),
                want,
                "qubit {q}",
            );
        }
        // Two odd blocks XOR their labels: Z2·Z4 -> 1 ^ 2 = 3.
        assert_eq!(p.partition_of(&[0], &[0b0001_0100]), 3);
        // Even z-weight inside one block contributes nothing.
        assert_eq!(p.partition_of(&[0], &[0b0000_1100]), 0);
    }

    #[test]
    fn cut_leaves_uncovered_qubits_in_the_zero_label() {
        let p = PartitionRows::<1>::cut(4, &[vec![0], vec![1]]);
        assert_eq!(p.rows().1, [[0b0010u64]]);
        assert_eq!(p.partition_of_pauli(&PauliString::<1>::z(2)), 0);
        assert_eq!(p.partition_of_pauli(&PauliString::<1>::z(3)), 0);
    }

    #[test]
    fn cut_of_one_block_is_the_trivial_partitioning() {
        assert_eq!(
            PartitionRows::<1>::cut(4, &[vec![0, 1, 2, 3]]),
            PartitionRows::<1>::none(4),
        );
    }

    #[test]
    fn cut_rows_round_trip_across_the_word_boundary() {
        let lo: Vec<u32> = (0..64).collect();
        let hi: Vec<u32> = (64..70).collect();
        let p = PartitionRows::<2>::cut(70, &[lo, hi]);
        assert_eq!(p.rows().0, [[0u64, 0]]);
        assert_eq!(p.rows().1, [[0u64, 0b11_1111]]);
        assert_eq!(p.partition_of_pauli(&PauliString::<2>::z(63)), 0);
        assert_eq!(p.partition_of_pauli(&PauliString::<2>::z(64)), 1);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn cut_with_three_blocks_panics() {
        let _ = PartitionRows::<1>::cut(4, &[vec![0], vec![1], vec![2]]);
    }

    #[test]
    #[should_panic(expected = "blocks must be disjoint")]
    fn cut_with_overlapping_blocks_panics() {
        let _ = PartitionRows::<1>::cut(4, &[vec![0, 1], vec![1, 2]]);
    }

    #[test]
    #[should_panic(expected = "outside 0..4")]
    fn cut_with_an_out_of_range_qubit_panics() {
        let _ = PartitionRows::<1>::cut(4, &[vec![0], vec![9]]);
    }

    #[test]
    #[should_panic(expected = "no qubit")]
    fn cut_with_an_empty_labelled_block_panics() {
        let _ = PartitionRows::<1>::cut(4, &[vec![0, 1, 2, 3], vec![]]);
    }

    // ---- occupancy ----

    #[test]
    fn partition_occupancy_is_balanced_on_low_weight_keys() {
        // The same guard as `occupancy_is_balanced_on_low_weight_keys`, one level up: a partition carries a whole worker's share of the sum, so a structured (projection-like) row choice would be fatal here.
        let num_qubits = 128;
        let p = PartitionRows::<2>::from_seed(num_qubits, 2, 0x0CC3);
        let mut rng = Xs64::new(0xBA3);
        let mut counts = vec![0usize; p.num_partitions()];
        let target = 4000usize;
        for _ in 0..target {
            let weight = 1 + (rng.next_u64() % 3) as usize;
            let k = low_weight_key::<2>(&mut rng, num_qubits, weight);
            counts[p.partition_of_pauli(&k) as usize] += 1;
        }
        let mean = target / p.num_partitions(); // 1000
        let max = *counts.iter().max().unwrap();
        let min = *counts.iter().min().unwrap();
        assert!(
            max < 2 * mean,
            "max load {max} vs mean {mean}, counts {counts:?}"
        );
        assert!(
            min > mean / 2,
            "min load {min} vs mean {mean}, counts {counts:?}"
        );
    }

    // ---- independence from the refinement rows ----

    #[test]
    fn seeded_partition_rows_are_independent_of_the_hash() {
        for seed in [
            0x1u64,
            0x5EED,
            0xBEEF,
            crate::bucket::sum::DEFAULT_HASH_SEED,
        ] {
            let h = Gf2Hash::<2>::new(128, 7, seed);
            let p = PartitionRows::<2>::from_seed(128, 3, seed);
            assert!(p.is_independent_of(&h), "seed {seed:#x}");
        }
    }

    #[test]
    fn a_partition_row_copied_from_the_hash_is_not_independent() {
        let h = Gf2Hash::<2>::new(128, 7, 0x5EED);
        let (hx, hz) = h.row(0);
        let seeded = PartitionRows::<2>::from_seed(128, 2, 0x5EED);
        let (sx, sz) = seeded.rows();
        let p = PartitionRows::<2>::from_rows(128, vec![sx[0], hx], vec![sz[0], hz]);
        assert!(!p.is_independent_of(&h));
        // Only the active rows count: at zero bucket bits there is nothing to be dependent on, and the two partition rows are independent among themselves.
        let h0 = Gf2Hash::<2>::new(128, 0, 0x5EED);
        assert!(p.is_independent_of(&h0));
    }
}

#[cfg(test)]
mod props {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Linearity over arbitrary keys — the load-bearing algebraic property.
        /// Everything about bucket prediction follows from it.
        #[test]
        fn hash_is_gf2_linear_w2(
            ax in any::<[u64; 2]>(), az in any::<[u64; 2]>(),
            bx in any::<[u64; 2]>(), bz in any::<[u64; 2]>(),
            bits in 0u8..=13u8,
            seed in any::<u64>(),
        ) {
            let h = Gf2Hash::<2>::new(128, bits, seed);
            let cx = [ax[0] ^ bx[0], ax[1] ^ bx[1]];
            let cz = [az[0] ^ bz[0], az[1] ^ bz[1]];
            prop_assert_eq!(
                h.bucket_of(&cx, &cz),
                h.bucket_of(&ax, &az) ^ h.bucket_of(&bx, &bz)
            );
        }

        /// Refining keeps the low bits, so a bucket only ever splits.
        #[test]
        fn refine_is_a_prefix_extension_w1(
            x in any::<[u64; 1]>(), z in any::<[u64; 1]>(),
            bits in 0u8..=12u8,
            seed in any::<u64>(),
        ) {
            let mut h = Gf2Hash::<1>::new(64, bits, seed);
            let before = h.bucket_of(&x, &z);
            h.refine();
            let after = h.bucket_of(&x, &z);
            let mask = (1u32 << bits) - 1;
            prop_assert_eq!(after & mask, before);
        }

        /// The partition map is GF(2)-linear for the same reason `h` is — which is what makes the global bucket `(part(v), loc(v))` predictable under a channel's delta set.
        #[test]
        fn partition_of_is_gf2_linear_w1(
            ax in any::<[u64; 1]>(), az in any::<[u64; 1]>(),
            bx in any::<[u64; 1]>(), bz in any::<[u64; 1]>(),
            bits in 0u8..=P_MAX_BITS,
            seed in any::<u64>(),
        ) {
            let p = PartitionRows::<1>::from_seed(64, bits, seed);
            let cx = [ax[0] ^ bx[0]];
            let cz = [az[0] ^ bz[0]];
            prop_assert_eq!(
                p.partition_of(&cx, &cz),
                p.partition_of(&ax, &az) ^ p.partition_of(&bx, &bz)
            );
        }

        #[test]
        fn partition_of_is_gf2_linear_w2(
            ax in any::<[u64; 2]>(), az in any::<[u64; 2]>(),
            bx in any::<[u64; 2]>(), bz in any::<[u64; 2]>(),
            bits in 0u8..=P_MAX_BITS,
            seed in any::<u64>(),
        ) {
            let p = PartitionRows::<2>::from_seed(128, bits, seed);
            let cx = [ax[0] ^ bx[0], ax[1] ^ bx[1]];
            let cz = [az[0] ^ bz[0], az[1] ^ bz[1]];
            prop_assert_eq!(
                p.partition_of(&cx, &cz),
                p.partition_of(&ax, &az) ^ p.partition_of(&bx, &bz)
            );
        }
    }
}
