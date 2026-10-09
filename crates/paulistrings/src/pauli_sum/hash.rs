//! The GF(2)-linear bucket hash [`Gf2Hash`] and the partition rows [`PartitionRows`] (ARCHITECTURE.md §Hash, §Partitioning).

use crate::pauli_string::word_mask;
use crate::rng::{mix64, SPLITMIX_GAMMA};

/// Maximum number of bucket bits; all rows are drawn up front so the active hash is a prefix of one fixed matrix.
pub const B_MAX_BITS: u8 = 22;

/// Maximum number of partition bits, i.e. `P ≤ 64` partitions.
pub const P_MAX_BITS: u8 = 6;

/// Salt keeping [`PartitionRows::from_seed`]'s rows independent of [`Gf2Hash::new`]'s under the same seed.
const PARTITION_ROW_SALT: u64 = 0xD1B5_4A32_D192_ED03;

/// Word `word` of the `half` (0 = x, 1 = z) of row `row`, draw `attempt`.
// Not a GF(2)-linear generator such as xorshift: research/FINDINGS.md §`Gf2Hash` rows are splitmix64, not xorshift successors.
fn row_word(seed: u64, row: usize, attempt: u32, word: usize, half: u64) -> u64 {
    debug_assert!(row < 1 << 16 && word < 1 << 15);
    let position = ((attempt as u64) << 32) | ((row as u64) << 16) | ((word as u64) << 1) | half;
    mix64(seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(position.wrapping_add(1))))
}

/// A GF(2) matrix over the key space `(x, z)`, each row masked to the live qubit columns; `Gf2Hash` and `PartitionRows` are its two uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Gf2Matrix<const W: usize> {
    rows_x: Vec<[u64; W]>,
    rows_z: Vec<[u64; W]>,
    num_qubits: usize,
}

impl<const W: usize> Gf2Matrix<W> {
    /// `num_rows` rows drawn from `seed` with the `exclude` coordinates cleared, an all-zero row redrawn unless `num_qubits == 0`.
    pub(crate) fn draw(
        num_qubits: usize,
        num_rows: usize,
        seed: u64,
        exclude_x: &[u64; W],
        exclude_z: &[u64; W],
    ) -> Self {
        let mut rows_x: Vec<[u64; W]> = Vec::with_capacity(num_rows);
        let mut rows_z: Vec<[u64; W]> = Vec::with_capacity(num_rows);
        let has_live_columns = num_qubits > 0;
        for row in 0..num_rows {
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
        Self {
            rows_x,
            rows_z,
            num_qubits,
        }
    }

    /// Explicit rows, masked to the live columns.
    pub(crate) fn from_rows(
        num_qubits: usize,
        mut rows_x: Vec<[u64; W]>,
        mut rows_z: Vec<[u64; W]>,
    ) -> Self {
        assert_eq!(rows_x.len(), rows_z.len(), "Gf2Matrix: row count mismatch");
        for (row_x, row_z) in rows_x.iter_mut().zip(rows_z.iter_mut()) {
            for w in 0..W {
                let mask = word_mask(num_qubits, w);
                row_x[w] &= mask;
                row_z[w] &= mask;
            }
        }
        Self {
            rows_x,
            rows_z,
            num_qubits,
        }
    }

    pub(crate) fn num_rows(&self) -> usize {
        self.rows_x.len()
    }

    pub(crate) fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub(crate) fn row(&self, i: usize) -> KeyRow<W> {
        (self.rows_x[i], self.rows_z[i])
    }

    pub(crate) fn rows(&self) -> (&[[u64; W]], &[[u64; W]]) {
        (&self.rows_x, &self.rows_z)
    }

    /// Bit `row` of `M·v`, as `0` or `1`.
    pub(crate) fn row_parity(&self, x: &[u64; W], z: &[u64; W], row: usize) -> u32 {
        let row_x = &self.rows_x[row];
        let row_z = &self.rows_z[row];
        let mut folded: u64 = 0;
        for w in 0..W {
            folded ^= (x[w] & row_x[w]) ^ (z[w] & row_z[w]);
        }
        folded.count_ones() & 1
    }

    /// The first `rows` bits of `M·v`, row `i` at bit `i`.
    pub(crate) fn apply(&self, x: &[u64; W], z: &[u64; W], rows: usize) -> u32 {
        let mut index: u32 = 0;
        for i in 0..rows {
            index |= self.row_parity(x, z, i) << i;
        }
        index
    }

    /// Whether this matrix's first `rows` rows and `other`'s first `other_rows` are jointly GF(2)-independent.
    pub(crate) fn independent_with(&self, rows: usize, other: &Self, other_rows: usize) -> bool {
        let stacked: Vec<KeyRow<W>> = (0..rows)
            .map(|i| self.row(i))
            .chain((0..other_rows).map(|i| other.row(i)))
            .collect();
        gf2_rank_wide(&stacked) == rows + other_rows
    }
}

/// The GF(2)-linear hash `h(v) = H·v` from a Pauli key `v = (x, z)` to a bucket index, for a dense random `H` drawn from a seed.
///
/// Linearity gives `h(v ⊕ d) = h(v) ⊕ h(d)`, so a channel's output buckets are predictable from its input buckets.
#[derive(Clone, Debug)]
pub struct Gf2Hash<const W: usize> {
    /// All [`B_MAX_BITS`] rows of `H`.
    matrix: Gf2Matrix<W>,
    /// Active prefix length: `B = 1 << bits` buckets.
    bits: u8,
    /// Seed the rows were generated from.
    seed: u64,
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

        Self {
            matrix: Gf2Matrix::draw(num_qubits, B_MAX_BITS as usize, seed, &[0; W], &[0; W]),
            bits,
            seed,
        }
    }

    /// Number of active bucket bits.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Number of buckets, `1 << bits()`.
    pub fn num_buckets(&self) -> usize {
        1usize << self.bits
    }

    /// The seed the rows were generated from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The qubit count the rows were masked against.
    pub fn num_qubits(&self) -> usize {
        self.matrix.num_qubits()
    }

    /// `h(v)` for a key given as separate `x` and `z` words.
    pub fn bucket_of(&self, x: &[u64; W], z: &[u64; W]) -> u32 {
        self.matrix.apply(x, z, self.bits as usize)
    }

    /// Bit `row` of `H·v`, as `0` or `1`.
    pub(super) fn row_parity(&self, x: &[u64; W], z: &[u64; W], row: u8) -> u32 {
        self.matrix.row_parity(x, z, row as usize)
    }

    /// Double the bucket count: `B → 2B`.
    ///
    /// # Panics
    ///
    /// Panics if already at `B_MAX_BITS`.
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
    pub fn coarsen(&mut self) {
        assert!(
            self.bits > 0,
            "Gf2Hash::coarsen: already at a single bucket"
        );
        self.bits -= 1;
    }

    /// `true` if `other` was generated with the same rows, so sums partitioned by the two can be combined (after matching `bits`).
    pub(crate) fn same_rows_as(&self, other: &Self) -> bool {
        self.seed == other.seed && self.num_qubits() == other.num_qubits()
    }

    /// Row `i < B_MAX_BITS` of `H`, which may lie beyond the active prefix.
    #[cfg(any(test, feature = "cuda"))]
    pub(crate) fn row(&self, i: usize) -> ([u64; W], [u64; W]) {
        self.matrix.row(i)
    }
}

/// The `p` GF(2) rows `part(v) = P·v` that split a sum across `2^p` partitions, fixed while the bucket hash refines (ARCHITECTURE.md §Partitioning).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionRows<const W: usize> {
    /// One row per partition bit: `P = 1 << rows` partitions.
    matrix: Gf2Matrix<W>,
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
        Self {
            matrix: Gf2Matrix::draw(
                num_qubits,
                bits as usize,
                mix64(seed) ^ PARTITION_ROW_SALT,
                exclude_x,
                exclude_z,
            ),
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

        let matrix = Gf2Matrix::from_rows(num_qubits, rows_x, rows_z);
        for i in 0..matrix.num_rows() {
            let (row_x, row_z) = matrix.row(i);
            assert!(
                num_qubits == 0 || row_x.iter().chain(&row_z).any(|&w| w != 0),
                "PartitionRows::from_rows: row {i} masks to zero",
            );
        }
        Self { matrix }
    }

    /// Rows that label a qubit cut: `log2(blocks.len())` z-only rows giving block `b` the label `b`.
    ///
    /// A term's label is the XOR of the labels of the blocks it has odd z-weight in, so an x-only generator is local under any cut and a `ZZ(i, j)` bond is remote exactly when it crosses blocks (ARCHITECTURE.md §Partitioning).
    /// Blocks must be disjoint; qubits in no block behave as if in block 0.
    ///
    /// # Panics
    ///
    /// Panics if `blocks.len()` is not a power of two or exceeds `2^P_MAX_BITS`, if a qubit is `>= num_qubits` or appears in two blocks, or if some row would be all-zero.
    pub fn cut(num_qubits: usize, blocks: &[Vec<usize>]) -> Self {
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
                assert!(
                    q < num_qubits,
                    "PartitionRows::cut: qubit {q} in block {b} is outside 0..{num_qubits}",
                );
                assert!(
                    !seen[q],
                    "PartitionRows::cut: blocks must be disjoint, qubit {q} appears twice",
                );
                seen[q] = true;
                for (i, row) in rows_z.iter_mut().enumerate() {
                    if (b >> i) & 1 == 1 {
                        row[q / 64] |= 1u64 << (q % 64);
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
    pub fn none(num_qubits: usize) -> Self {
        Self {
            matrix: Gf2Matrix::from_rows(num_qubits, Vec::new(), Vec::new()),
        }
    }

    /// Number of partition bits.
    pub fn bits(&self) -> u8 {
        self.matrix.num_rows() as u8
    }

    /// Number of partitions, `1 << bits()`.
    pub fn num_partitions(&self) -> usize {
        1usize << self.bits()
    }

    /// The qubit count the rows were masked against.
    pub fn num_qubits(&self) -> usize {
        self.matrix.num_qubits()
    }

    /// `part(v)` for a key given as separate `x` and `z` words.
    pub fn partition_of(&self, x: &[u64; W], z: &[u64; W]) -> u32 {
        self.matrix.apply(x, z, self.matrix.num_rows())
    }

    /// The rows as `(x-masks, z-masks)`, already masked to the live columns.
    pub fn rows(&self) -> (&[[u64; W]], &[[u64; W]]) {
        self.matrix.rows()
    }

    /// `true` if no row reads a coordinate in `(mask_x, mask_z)`, i.e. keys differing only there always share a partition.
    pub fn avoids(&self, mask_x: &[u64; W], mask_z: &[u64; W]) -> bool {
        let (rows_x, rows_z) = self.matrix.rows();
        rows_x.iter().zip(rows_z).all(|(row_x, row_z)| {
            (0..W).all(|w| row_x[w] & mask_x[w] == 0 && row_z[w] & mask_z[w] == 0)
        })
    }

    /// `true` if the partition rows and `hash`'s active rows are jointly GF(2)-independent.
    pub fn is_independent_of(&self, hash: &Gf2Hash<W>) -> bool {
        self.matrix
            .independent_with(self.matrix.num_rows(), &hash.matrix, hash.bits() as usize)
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
