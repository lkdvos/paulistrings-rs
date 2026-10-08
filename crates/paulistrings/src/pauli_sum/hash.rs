//! The GF(2)-linear bucket hash [`Gf2Hash`] and the partition rows [`PartitionRows`] (ARCHITECTURE.md §Hash, §Partitioning).

use crate::pauli_string::PauliString;
use crate::rng::{mix64, SPLITMIX_GAMMA};

/// Maximum number of bucket bits; all rows are drawn up front so the active hash is a prefix of one fixed matrix.
pub const B_MAX_BITS: u8 = 22;

/// Maximum number of partition bits, i.e. `P ≤ 64` partitions.
pub const P_MAX_BITS: u8 = 6;

/// Salt keeping [`PartitionRows::from_seed`]'s rows independent of [`Gf2Hash::new`]'s under the same seed.
const PARTITION_ROW_SALT: u64 = 0xD1B5_4A32_D192_ED03;

/// Word `word` of the `half` (0 = x, 1 = z) of row `row`, draw `attempt`.
// Not a GF(2)-linear generator such as xorshift: research/FINDINGS.md §`Gf2Hash` rows are splitmix64, not xorshift successors.
#[inline]
fn row_word(seed: u64, row: usize, attempt: u32, word: usize, half: u64) -> u64 {
    debug_assert!(row < 1 << 16 && word < 1 << 15);
    let position = ((attempt as u64) << 32) | ((row as u64) << 16) | ((word as u64) << 1) | half;
    mix64(seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(position.wrapping_add(1))))
}

/// Mask of the live qubit bits in word `word`, given `num_qubits` total.
#[inline]
fn word_mask(num_qubits: usize, word: usize) -> u64 {
    let first_qubit = 64 * word;
    if num_qubits >= first_qubit + 64 {
        !0u64
    } else if num_qubits <= first_qubit {
        0
    } else {
        (1u64 << (num_qubits - first_qubit)) - 1
    }
}

/// `n_rows` rows drawn from `seed` and masked to the live columns, an all-zero row redrawn unless `num_qubits == 0`.
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
        let (row_x, row_z) = loop {
            let mut row_x = [0u64; W];
            let mut row_z = [0u64; W];
            let mut any = false;
            for w in 0..W {
                let mask = word_mask(num_qubits, w);
                row_x[w] = row_word(seed, row, attempt, w, 0) & mask & !exclude_x[w];
                row_z[w] = row_word(seed, row, attempt, w, 1) & mask & !exclude_z[w];
                any |= (row_x[w] | row_z[w]) != 0;
            }
            if any || !has_live_columns {
                break (row_x, row_z);
            }
            attempt += 1;
        };
        rows_x.push(row_x);
        rows_z.push(row_z);
    }
    (rows_x, rows_z)
}

/// The GF(2)-linear hash `h(v) = H·v` from a Pauli key `v = (x, z)` to a bucket index, for a dense random `H` drawn from a seed.
///
/// Linearity gives `h(v ⊕ d) = h(v) ⊕ h(d)`, so a channel's output buckets are predictable from its input buckets.
#[derive(Clone, Debug)]
pub struct Gf2Hash<const W: usize> {
    /// X-part of each row of `H`, masked to the live qubit columns.
    rows_x: Vec<[u64; W]>,
    /// Z-part of each row of `H`, masked to the live qubit columns.
    rows_z: Vec<[u64; W]>,
    /// Active prefix length: `B = 1 << bits` buckets.
    bits: u8,
    /// Seed the rows were generated from.
    seed: u64,
    /// Qubit count the rows were masked against.
    num_qubits: usize,
}

impl<const W: usize> Gf2Hash<W> {
    /// Build a hash over `num_qubits` qubits with `bits` active bucket bits.
    ///
    /// The rows are a function of `(num_qubits, seed)` alone, independent of `W`.
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
    #[inline]
    pub fn bucket_of(&self, x: &[u64; W], z: &[u64; W]) -> u32 {
        let mut index: u32 = 0;
        for i in 0..self.bits as usize {
            index |= self.row_parity(x, z, i as u8) << i;
        }
        index
    }

    /// Bit `row` of `H·v`, as `0` or `1`.
    #[inline]
    pub(crate) fn row_parity(&self, x: &[u64; W], z: &[u64; W], row: u8) -> u32 {
        let row_x = &self.rows_x[row as usize];
        let row_z = &self.rows_z[row as usize];
        let mut folded: u64 = 0;
        for w in 0..W {
            folded ^= (x[w] & row_x[w]) ^ (z[w] & row_z[w]);
        }
        folded.count_ones() & 1
    }

    /// `h(v)` for a [`PauliString`].
    #[inline]
    pub fn bucket_of_pauli(&self, p: &PauliString<W>) -> u32 {
        self.bucket_of(&p.x, &p.z)
    }

    /// Double the bucket count: `B → 2B`.
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

    /// Halve the bucket count: `B → B/2`, merging buckets `b` and `b + B/2`.
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

    /// Row `i < B_MAX_BITS` of `H`, which may lie beyond the active prefix.
    #[inline]
    pub(crate) fn row(&self, i: usize) -> ([u64; W], [u64; W]) {
        (self.rows_x[i], self.rows_z[i])
    }
}

/// The `p` GF(2) rows `part(v) = P·v` that split a sum across `2^p` partitions, fixed while the bucket hash refines (ARCHITECTURE.md §Partitioning).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionRows<const W: usize> {
    /// X-part of each partition row, masked to the live qubit columns.
    rows_x: Vec<[u64; W]>,
    /// Z-part of each partition row, masked to the live qubit columns.
    rows_z: Vec<[u64; W]>,
    /// Number of rows: `P = 1 << bits` partitions.
    bits: u8,
    /// Qubit count the rows were masked against.
    num_qubits: usize,
}

impl<const W: usize> PartitionRows<W> {
    /// Panics unless the rows split a sum over `num_qubits` qubits into `partitions` partitions; debug builds also check them independent of `hash`.
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

    /// Draw `bits` partition rows from `seed`, unrelated to [`Gf2Hash::new`]'s rows under the same seed.
    ///
    /// # Panics
    ///
    /// Panics if `bits > P_MAX_BITS`, or in debug builds if `num_qubits > 64 · W`.
    pub fn from_seed(num_qubits: usize, bits: u8, seed: u64) -> Self {
        Self::from_seed_excluding(num_qubits, bits, seed, &[0; W], &[0; W])
    }

    /// [`Self::from_seed`] with the coordinates in `(exclude_x, exclude_z)` cleared from every row, so keys differing only there share a partition.
    ///
    /// [`crate::DistributedSum::rotated_overlap`] needs this with [`crate::RotationAxis::flip_mask`].
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

    /// Build from explicit rows, masked to the live qubit columns.
    ///
    /// # Panics
    ///
    /// Panics if `rows_x` and `rows_z` differ in length, if there are more than [`P_MAX_BITS`] rows, or if any row masks to all-zero while `num_qubits > 0`.
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
    /// A term's label is the XOR of the labels of the blocks it has odd z-weight in, so an x-only generator is local under any cut and a `ZZ(i, j)` bond is remote exactly when it crosses blocks (ARCHITECTURE.md §Partitioning).
    /// Blocks must be disjoint; qubits in no block behave as if in block 0.
    ///
    /// # Panics
    ///
    /// Panics if `blocks.len()` is not a power of two or exceeds `2^P_MAX_BITS`, if a qubit is `>= num_qubits` or appears in two blocks, or if some row would be all-zero.
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
                let qubit_index = q as usize;
                assert!(
                    qubit_index < num_qubits,
                    "PartitionRows::cut: qubit {q} in block {b} is outside 0..{num_qubits}",
                );
                assert!(
                    !seen[qubit_index],
                    "PartitionRows::cut: blocks must be disjoint, qubit {q} appears twice",
                );
                seen[qubit_index] = true;
                for (i, row) in rows_z.iter_mut().enumerate() {
                    if (b >> i) & 1 == 1 {
                        row[qubit_index / 64] |= 1u64 << (qubit_index % 64);
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
    #[inline]
    pub fn partition_of(&self, x: &[u64; W], z: &[u64; W]) -> u32 {
        let mut index: u32 = 0;
        for i in 0..self.bits as usize {
            let row_x = &self.rows_x[i];
            let row_z = &self.rows_z[i];
            let mut fold: u64 = 0;
            for w in 0..W {
                fold ^= (x[w] & row_x[w]) ^ (z[w] & row_z[w]);
            }
            index |= (fold.count_ones() & 1) << i;
        }
        index
    }

    /// `part(v)` for a [`PauliString`].
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
        self.rows_x.iter().zip(&self.rows_z).all(|(row_x, row_z)| {
            (0..W).all(|w| row_x[w] & mask_x[w] == 0 && row_z[w] & mask_z[w] == 0)
        })
    }

    /// `true` if the partition rows and `hash`'s active rows are jointly GF(2)-independent.
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

/// One row of the key space as `(x-masks, z-masks)`.
type KeyRow<const W: usize> = ([u64; W], [u64; W]);

/// GF(2) rank of rows over the key space, by Gaussian elimination.
fn gf2_rank_wide<const W: usize>(rows: &[KeyRow<W>]) -> usize {
    let mut pivots: Vec<(usize, KeyRow<W>)> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut reduced = *row;
        'reduce: loop {
            let Some(lead) = leading_column(&reduced) else {
                break;
            };
            for (pivot_column, pivot_row) in pivots.iter() {
                if *pivot_column == lead {
                    for w in 0..W {
                        reduced.0[w] ^= pivot_row.0[w];
                        reduced.1[w] ^= pivot_row.1[w];
                    }
                    continue 'reduce;
                }
            }
            pivots.push((lead, reduced));
            break;
        }
    }
    pivots.len()
}

/// Index of the highest set column of a key-space row, or `None` if it is zero.
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
