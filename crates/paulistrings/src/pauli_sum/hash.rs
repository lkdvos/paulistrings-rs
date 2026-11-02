//! The GF(2)-linear bucket hash [`Gf2Hash`] and the partition rows [`PartitionRows`], two matrices over the `2 · num_qubits` key coordinates.

use crate::pauli_string::word_mask;
use crate::rng::{mix64, SPLITMIX_GAMMA};

/// The ceiling on the rows either matrix can use: bucket and partition counts are `u32` (`ChunkMap`), so `2^31` is the largest.
pub(crate) const INDEX_BITS: u8 = u32::BITS as u8 - 1;

/// Rows a GF(2) hash over `num_qubits` qubits can hold: one per key coordinate, at most 31 since bucket and partition counts are `u32`; the bound on bucket bits and on partition bits alike.
pub fn max_hash_bits(num_qubits: usize) -> u8 {
    (2 * num_qubits).min(INDEX_BITS as usize) as u8
}

/// Salt keeping [`PartitionRows::from_seed`]'s rows independent of [`Gf2Hash::new`]'s under the same seed.
const PARTITION_ROW_SALT: u64 = 0xD1B5_4A32_D192_ED03;

/// Candidates tried for one row before the last is taken as drawn, so a draw with nothing acceptable to find still terminates.
const REDRAW_ATTEMPTS: u32 = 1 << 12;

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
    /// `num_rows` dense rows drawn from `seed` reading no coordinate in `(exclude_x, exclude_z)`: linearly independent until the readable coordinates are spanned and nonzero after, or all zero when nothing is readable.
    /// The rows depend on `(num_qubits, seed, exclusion)` alone, not on `W`.
    pub(crate) fn draw(
        num_qubits: usize,
        num_rows: usize,
        seed: u64,
        exclude_x: &[u64; W],
        exclude_z: &[u64; W],
    ) -> Self {
        let readable: usize = (0..W)
            .map(|w| {
                let mask = word_mask(num_qubits, w);
                ((mask & !exclude_x[w]).count_ones() + (mask & !exclude_z[w]).count_ones()) as usize
            })
            .sum();
        if readable == 0 {
            return Self::from_rows(num_qubits, vec![[0; W]; num_rows], vec![[0; W]; num_rows]);
        }
        let candidate = |row: usize, attempt: u32| -> KeyRow<W> {
            let mut key = ([0u64; W], [0u64; W]);
            for w in 0..W {
                let mask = word_mask(num_qubits, w);
                key.0[w] = row_word(seed, row, attempt, w, 0) & mask & !exclude_x[w];
                key.1[w] = row_word(seed, row, attempt, w, 1) & mask & !exclude_z[w];
            }
            key
        };
        let mut basis = Gf2Basis::<W>::default();
        Self::accept_rows(num_qubits, num_rows, num_rows.min(readable), &mut basis, 0, candidate)
    }

    /// `given` rows followed by dense rows from `seed` up to `num_rows`, each independent of everything before it until the `2 · num_qubits` coordinates are spanned and nonzero after.
    pub(crate) fn complete(
        num_qubits: usize,
        num_rows: usize,
        seed: u64,
        given: Vec<KeyRow<W>>,
    ) -> Self {
        let first = given.len();
        let (rows_x, rows_z): (Vec<_>, Vec<_>) = given.into_iter().unzip();
        let Self { mut rows_x, mut rows_z, .. } = Self::from_rows(num_qubits, rows_x, rows_z);
        if first < num_rows {
            let mut basis = Gf2Basis::<W>::default();
            for i in 0..first {
                basis.insert(&(rows_x[i], rows_z[i]));
            }
            let candidate = |row: usize, attempt: u32| -> KeyRow<W> {
                let mut key = ([0u64; W], [0u64; W]);
                for w in 0..W {
                    let mask = word_mask(num_qubits, w);
                    key.0[w] = row_word(seed, row, attempt, w, 0) & mask;
                    key.1[w] = row_word(seed, row, attempt, w, 1) & mask;
                }
                key
            };
            let independent = num_rows.min(2 * num_qubits);
            let tail =
                Self::accept_rows(num_qubits, num_rows, independent, &mut basis, first, candidate);
            rows_x.extend_from_slice(&tail.rows_x[first..]);
            rows_z.extend_from_slice(&tail.rows_z[first..]);
        }
        Self::from_rows(num_qubits, rows_x, rows_z)
    }

    /// Rows `first..num_rows` from `candidate(row, attempt)`: while `basis` holds fewer than `independent` rows a candidate is accepted only if it extends it, after that only if it is nonzero, and after [`REDRAW_ATTEMPTS`] as it is; rows before `first` are zero placeholders.
    fn accept_rows(
        num_qubits: usize,
        num_rows: usize,
        independent: usize,
        basis: &mut Gf2Basis<W>,
        first: usize,
        candidate: impl Fn(usize, u32) -> KeyRow<W>,
    ) -> Self {
        let mut rows_x = vec![[0u64; W]; first];
        let mut rows_z = vec![[0u64; W]; first];
        for row in first..num_rows {
            let mut attempt = 0u32;
            let key = loop {
                let key = candidate(row, attempt);
                let accepted = if basis.len() < independent {
                    basis.insert(&key)
                } else {
                    key != ([0; W], [0; W])
                };
                if accepted || attempt == REDRAW_ATTEMPTS {
                    break key;
                }
                attempt += 1;
            };
            rows_x.push(key.0);
            rows_z.push(key.1);
        }
        Self::from_rows(num_qubits, rows_x, rows_z)
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
        Self { rows_x, rows_z, num_qubits }
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

    /// GF(2) rank of the first `rows` rows.
    pub(crate) fn rank(&self, rows: usize) -> usize {
        gf2_rank_wide(&(0..rows).map(|i| self.row(i)).collect::<Vec<_>>())
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
        let stacked: Vec<KeyRow<W>> =
            (0..rows).map(|i| self.row(i)).chain((0..other_rows).map(|i| other.row(i))).collect();
        gf2_rank_wide(&stacked) == rows + other_rows
    }
}

/// The GF(2)-linear hash `h(v) = H·v` from a Pauli key `v = (x, z)` to a bucket index, for an `H` of [`max_hash_bits`] rows drawn from a seed or given by the caller.
///
/// Linearity gives `h(v ⊕ d) = h(v) ⊕ h(d)`, so a channel's output buckets are predictable from its input buckets.
/// The active rows are a prefix of one fixed matrix, so a refine splits every bucket in two by one row parity with the within-bucket order inherited, and a coarsen merges bucket pairs; neither needs a sort to keep the [`PauliSum`](crate::PauliSum) invariant.
#[derive(Clone, Debug)]
pub struct Gf2Hash<const W: usize> {
    /// All [`max_hash_bits`] rows of `H`.
    matrix: Gf2Matrix<W>,
    /// Active prefix length: `B = 1 << bits` buckets.
    bits: u8,
    /// Seed the drawn rows came from.
    seed: u64,
}

impl<const W: usize> Gf2Hash<W> {
    /// Build a hash over `num_qubits` qubits with `bits` active bucket bits, every row drawn from `seed` and the rows linearly independent.
    ///
    /// The rows are a function of `(num_qubits, seed)` alone, independent of `W`.
    ///
    /// # Panics
    ///
    /// Panics if `bits > max_hash_bits(num_qubits)`, or in debug builds if `num_qubits > 64 · W`.
    pub fn new(num_qubits: usize, bits: u8, seed: u64) -> Self {
        Self::from_rows(num_qubits, bits, seed, Vec::new(), Vec::new())
    }

    /// A hash whose first rows are the caller's, masked to the live columns, completed to [`max_hash_bits`] rows by dense rows drawn from `seed` that are independent of everything before them.
    ///
    /// Dependent or zero rows are accepted, and [`Gf2Hash::rank_deficiency`] reports them.
    ///
    /// # Panics
    ///
    /// As [`Gf2Hash::new`], if `rows_x` and `rows_z` differ in length, or if more than `max_hash_bits(num_qubits)` rows are given.
    pub fn from_rows(
        num_qubits: usize,
        bits: u8,
        seed: u64,
        rows_x: Vec<[u64; W]>,
        rows_z: Vec<[u64; W]>,
    ) -> Self {
        let max_bits = max_hash_bits(num_qubits);
        assert!(
            bits <= max_bits,
            "Gf2Hash: bits {bits} exceeds the {max_bits} rows of {num_qubits} qubits"
        );
        assert_eq!(
            rows_x.len(),
            rows_z.len(),
            "Gf2Hash::from_rows: row count mismatch, {} x-rows vs {} z-rows",
            rows_x.len(),
            rows_z.len(),
        );
        assert!(
            rows_x.len() <= max_bits as usize,
            "Gf2Hash::from_rows: {} rows exceed the {max_bits} rows of {num_qubits} qubits",
            rows_x.len()
        );
        debug_assert!(num_qubits <= 64 * W);
        let given = rows_x.into_iter().zip(rows_z).collect();
        Self { matrix: Gf2Matrix::complete(num_qubits, max_bits as usize, seed, given), bits, seed }
    }

    /// [`Gf2Hash::from_rows`] from a flag matrix of `2 · num_qubits` columns, the x-bits of the qubits first and then their z-bits.
    ///
    /// # Panics
    ///
    /// As [`Gf2Hash::from_rows`], and if a row has not `2 · num_qubits` entries.
    pub fn from_matrix(num_qubits: usize, bits: u8, seed: u64, rows: &[Vec<bool>]) -> Self {
        let (rows_x, rows_z) = key_rows_from_matrix(num_qubits, rows);
        Self::from_rows(num_qubits, bits, seed, rows_x, rows_z)
    }

    /// Number of active bucket bits.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// The most bucket bits this hash can refine to: its row count, [`max_hash_bits`] of its qubit count.
    pub fn max_bits(&self) -> u8 {
        self.matrix.num_rows() as u8
    }

    /// Number of buckets, `1 << bits()`.
    pub fn num_buckets(&self) -> usize {
        1usize << self.bits
    }

    /// The seed the drawn rows came from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// All rows as `(x-masks, z-masks)`, the active ones first.
    pub fn rows(&self) -> (&[[u64; W]], &[[u64; W]]) {
        self.matrix.rows()
    }

    /// The rank of the rows: how many bucket bits can be used before some bucket can never hold a term.
    pub fn independent_rows(&self) -> usize {
        self.matrix.rank(self.matrix.num_rows())
    }

    /// `Some(reason)` when the rows are not full rank.
    pub fn rank_deficiency(&self) -> Option<String> {
        let rows = self.matrix.num_rows();
        let rank = self.independent_rows();
        (rank < rows).then(|| {
            format!(
                "the bucket hash's {rows} rows have rank {rank}: beyond {rank} bucket bits, some buckets can never hold a term"
            )
        })
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
    /// Panics if already at [`Gf2Hash::max_bits`].
    pub fn refine(&mut self) {
        assert!(
            self.bits < self.max_bits(),
            "Gf2Hash::refine: already at the {} rows of {} qubits",
            self.max_bits(),
            self.num_qubits()
        );
        self.bits += 1;
    }

    /// Halve the bucket count: `B → B/2`, merging buckets `b` and `b + B/2`.
    ///
    /// # Panics
    ///
    /// Panics if already at a single bucket.
    pub fn coarsen(&mut self) {
        assert!(self.bits > 0, "Gf2Hash::coarsen: already at a single bucket");
        self.bits -= 1;
    }

    /// `true` if `other` has the same rows over the same qubits, so sums partitioned by the two can be combined (after matching `bits`).
    pub(crate) fn same_rows_as(&self, other: &Self) -> bool {
        self.matrix == other.matrix
    }

    /// Row `i < max_bits()` of `H`, which may lie beyond the active prefix.
    #[cfg(any(test, feature = "cuda"))]
    pub(crate) fn row(&self, i: usize) -> ([u64; W], [u64; W]) {
        self.matrix.row(i)
    }
}

/// The `p` GF(2) rows `part(v) = P·v` that split a sum across `2^p` partitions: a partition holds exactly the terms with `part(v)` equal to its rank.
///
/// A matrix separate from [`Gf2Hash`]'s and fixed for a run, since a row that moved as the bucket hash refines would change a term's owner mid-run; it holds exactly the rows the partition count needs.
/// Dependence on the bucket hash's rows costs load balance, never correctness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionRows<const W: usize> {
    /// One row per partition bit: `P = 1 << rows` partitions.
    matrix: Gf2Matrix<W>,
}

impl<const W: usize> PartitionRows<W> {
    /// Panics unless the rows split a sum over `num_qubits` qubits into `partitions` partitions; rank shortfalls are [`PartitionRows::rank_deficiency`]'s, never a panic.
    pub(crate) fn assert_splits(&self, num_qubits: usize, partitions: usize) {
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
    }

    /// Draw `bits` linearly independent partition rows from `seed`, unrelated to [`Gf2Hash::new`]'s rows under the same seed.
    ///
    /// # Panics
    ///
    /// Panics if `bits > max_hash_bits(num_qubits)`, or in debug builds if `num_qubits > 64 · W`.
    pub fn from_seed(num_qubits: usize, bits: u8, seed: u64) -> Self {
        Self::from_seed_excluding(num_qubits, bits, seed, &[0; W], &[0; W])
    }

    /// [`Self::from_seed`] with the coordinates in `(exclude_x, exclude_z)` cleared from every row, so keys differing only there share a partition.
    ///
    /// The rows are independent as far as the readable coordinates allow; an exclusion that leaves none gives zero rows, which [`Self::rank_deficiency`] reports.
    ///
    /// # Panics
    ///
    /// As [`Self::from_seed`].
    pub fn from_seed_excluding(
        num_qubits: usize,
        bits: u8,
        seed: u64,
        exclude_x: &[u64; W],
        exclude_z: &[u64; W],
    ) -> Self {
        assert_bits(num_qubits, bits as usize);
        debug_assert!(num_qubits <= 64 * W);
        let seed = mix64(seed) ^ PARTITION_ROW_SALT;
        Self { matrix: Gf2Matrix::draw(num_qubits, bits as usize, seed, exclude_x, exclude_z) }
    }

    /// Explicit rows, masked to the live qubit columns; dependent or zero rows are accepted and reported by [`Self::rank_deficiency`].
    ///
    /// # Panics
    ///
    /// Panics if `rows_x` and `rows_z` differ in length, or if there are more than `max_hash_bits(num_qubits)` rows.
    pub fn from_rows(num_qubits: usize, rows_x: Vec<[u64; W]>, rows_z: Vec<[u64; W]>) -> Self {
        assert_eq!(
            rows_x.len(),
            rows_z.len(),
            "PartitionRows::from_rows: row count mismatch, {} x-rows vs {} z-rows",
            rows_x.len(),
            rows_z.len(),
        );
        assert_bits(num_qubits, rows_x.len());
        debug_assert!(num_qubits <= 64 * W);
        Self { matrix: Gf2Matrix::from_rows(num_qubits, rows_x, rows_z) }
    }

    /// [`Self::from_rows`] from a flag matrix laid out as [`Gf2Hash::from_matrix`] takes it.
    ///
    /// # Panics
    ///
    /// As [`Self::from_rows`], and if a row has not `2 · num_qubits` entries.
    pub fn from_matrix(num_qubits: usize, rows: &[Vec<bool>]) -> Self {
        let (rows_x, rows_z) = key_rows_from_matrix(num_qubits, rows);
        Self::from_rows(num_qubits, rows_x, rows_z)
    }

    /// Rows that label a qubit cut: `log2(blocks.len())` z-only rows giving block `b` the label `b`.
    ///
    /// A term's label is the XOR of the labels of the blocks it has odd z-weight in, so an x-only generator is local under any cut and a `ZZ(i, j)` bond is remote exactly when it crosses blocks.
    /// Blocks must be disjoint; qubits in no block behave as if in block 0.
    ///
    /// # Panics
    ///
    /// Panics if `blocks.len()` is not a power of two or exceeds `2^max_hash_bits(num_qubits)`, if a qubit is `>= num_qubits` or appears in two blocks, or if some row would be all-zero.
    pub fn cut(num_qubits: usize, blocks: &[Vec<usize>]) -> Self {
        assert!(
            blocks.len().is_power_of_two(),
            "PartitionRows::cut: block count {} is not a power of two",
            blocks.len(),
        );
        let bits = blocks.len().trailing_zeros() as usize;
        assert_bits(num_qubits, bits);
        debug_assert!(num_qubits <= 64 * W);

        let mut seen = vec![false; num_qubits];
        let mut rows_z = vec![[0u64; W]; bits];
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

        Self::from_rows(num_qubits, vec![[0u64; W]; bits], rows_z)
    }

    /// The trivial partitioning: one partition, no rows.
    pub fn none(num_qubits: usize) -> Self {
        Self { matrix: Gf2Matrix::from_rows(num_qubits, Vec::new(), Vec::new()) }
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

    /// The rank of the rows: `2^independent_rows()` of the partitions can hold terms.
    pub fn independent_rows(&self) -> usize {
        self.matrix.rank(self.matrix.num_rows())
    }

    /// `Some(reason)` when the rows are linearly dependent, which leaves partitions that can never hold a term.
    pub fn rank_deficiency(&self) -> Option<String> {
        let bits = self.matrix.num_rows();
        let rank = self.independent_rows();
        (rank < bits).then(|| {
            format!(
                "the {bits} partition rows have rank {rank}: only {} of the {} partitions can hold a term",
                1usize << rank,
                1usize << bits
            )
        })
    }

    /// The fewest active bits of `hash` at which its rows stop being independent of these rows, correlating partitions with buckets so that some of their pairs stay empty.
    /// `None` when that never happens before the two together span every coordinate, where overlap is forced and a partition holds fewer strings than it has buckets.
    pub fn overlap_onset(&self, hash: &Gf2Hash<W>) -> Option<u8> {
        let mut joint = Gf2Basis::<W>::default();
        for i in 0..self.matrix.num_rows() {
            joint.insert(&self.matrix.row(i));
        }
        let mut alone = Gf2Basis::<W>::default();
        let coordinates = 2 * self.num_qubits();
        for i in 0..hash.matrix.num_rows() {
            if joint.len() == coordinates {
                return None;
            }
            let row = hash.matrix.row(i);
            if alone.insert(&row) && !joint.insert(&row) {
                return Some(i as u8 + 1);
            }
        }
        None
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
        self.matrix.independent_with(self.matrix.num_rows(), &hash.matrix, hash.bits() as usize)
    }
}

/// Panics unless `bits` partition rows fit `num_qubits` qubits.
fn assert_bits(num_qubits: usize, bits: usize) {
    let max_bits = max_hash_bits(num_qubits);
    assert!(
        bits <= max_bits as usize,
        "PartitionRows: {bits} rows exceed the {max_bits} rows of {num_qubits} qubits"
    );
}

/// One row of the key space as `(x-masks, z-masks)`.
type KeyRow<const W: usize> = ([u64; W], [u64; W]);

/// A flag matrix of `2 · num_qubits` columns as `(x-rows, z-rows)`: column `c < num_qubits` is the x-bit of qubit `c`, column `num_qubits + c` its z-bit.
fn key_rows_from_matrix<const W: usize>(
    num_qubits: usize,
    rows: &[Vec<bool>],
) -> (Vec<[u64; W]>, Vec<[u64; W]>) {
    rows.iter()
        .enumerate()
        .map(|(i, flags)| {
            assert_eq!(
                flags.len(),
                2 * num_qubits,
                "row {i} has {} entries, not the {} coordinates of {num_qubits} qubits",
                flags.len(),
                2 * num_qubits
            );
            let mut key = ([0u64; W], [0u64; W]);
            for c in flags.iter().enumerate().filter(|(_, &set)| set).map(|(c, _)| c) {
                let (half, q) =
                    if c < num_qubits { (&mut key.0, c) } else { (&mut key.1, c - num_qubits) };
                half[q / 64] |= 1u64 << (q % 64);
            }
            key
        })
        .unzip()
}

/// GF(2) rank of rows over the key space, by Gaussian elimination.
fn gf2_rank_wide<const W: usize>(rows: &[KeyRow<W>]) -> usize {
    let mut basis = Gf2Basis::<W>::default();
    for row in rows {
        basis.insert(row);
    }
    basis.len()
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

/// An incrementally grown row-echelon basis of key-space rows.
struct Gf2Basis<const W: usize> {
    pivots: Vec<(usize, KeyRow<W>)>,
}

impl<const W: usize> Default for Gf2Basis<W> {
    fn default() -> Self {
        Self { pivots: Vec::new() }
    }
}

impl<const W: usize> Gf2Basis<W> {
    fn len(&self) -> usize {
        self.pivots.len()
    }

    /// Add `row` and return `true` if it is independent of the rows so far; leave the basis unchanged and return `false` otherwise.
    fn insert(&mut self, row: &KeyRow<W>) -> bool {
        let mut reduced = *row;
        while let Some(lead) = leading_column(&reduced) {
            match self.pivots.iter().find(|(column, _)| *column == lead) {
                Some((_, pivot)) => {
                    for w in 0..W {
                        reduced.0[w] ^= pivot.0[w];
                        reduced.1[w] ^= pivot.1[w];
                    }
                }
                None => {
                    self.pivots.push((lead, reduced));
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests;
