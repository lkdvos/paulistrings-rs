//! The GF(2)-linear bucket hash [`Gf2Hash`] and the partition rows [`PartitionRows`].

use crate::pauli_string::word_mask;
use crate::rng::{mix64, Rng, SPLITMIX_GAMMA};

/// Maximum number of bucket bits; all rows are drawn up front so the active hash is a prefix of one fixed matrix.
pub const B_MAX_BITS: u8 = 22;

/// Maximum number of partition bits, i.e. `P ≤ 64` partitions.
pub const P_MAX_BITS: u8 = 6;

/// Salt keeping [`PartitionRows::from_seed`]'s rows independent of [`Gf2Hash::new`]'s under the same seed.
const PARTITION_ROW_SALT: u64 = 0xD1B5_4A32_D192_ED03;

/// Salts keeping the sparse and projection families' coordinate draws independent of each other and of the dense rows.
const SPARSE_SALT: u64 = 0x2545_F491_4F6C_DD1D;
const PROJECTION_SALT: u64 = 0x94D0_49BB_1331_11EB;

/// How a family draws the rows of a [`Gf2Hash`] or of [`PartitionRows`].
///
/// Every family is GF(2)-linear, so every family partitions correctly; they differ in how evenly they spread a sum.
/// Draws aim for full rank, the first `min(rows, 2 · num_qubits)` rows linearly independent with a dependent candidate redrawn, as far as the family can reach; [`Gf2Hash::rank_deficiency`] and [`PartitionRows::rank_deficiency`] report any shortfall.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HashFamily {
    /// Every live coordinate set with probability 1/2, a universal hash.
    #[default]
    Dense,
    /// Exactly `weight` live coordinates per row, drawn without replacement.
    Sparse {
        /// Coordinates set per row, in `1..=` the coordinates a row may read.
        weight: u16,
    },
    /// One live coordinate per row, distinct until the readable coordinates run out and cycling after: a coordinate projection.
    Projection,
}

impl HashFamily {
    /// `Err` with the reason unless this family can draw rows over `coordinates` readable coordinates: a sparse weight must lie in `1..=coordinates`.
    pub fn validate(self, coordinates: usize) -> Result<(), String> {
        match self {
            HashFamily::Sparse { weight }
                if coordinates > 0 && !(1..=coordinates).contains(&(weight as usize)) =>
            {
                Err(format!(
                    "HashFamily::Sparse: weight {weight} must lie in 1..={coordinates}, the coordinates a row may read"
                ))
            }
            _ => Ok(()),
        }
    }

    /// The largest rank this family's rows reach over `coordinates` readable coordinates.
    ///
    /// Weight-`w` rows over `m` coordinates span the whole space for odd `w < m`, the even-weight subspace of dimension `m - 1` for even `w < m`, and one vector for `w = m`.
    pub(crate) fn reachable_rank(self, coordinates: usize) -> usize {
        match self {
            HashFamily::Sparse { weight } => {
                let weight = weight as usize;
                if coordinates == 0 {
                    0
                } else if weight >= coordinates {
                    1
                } else if weight % 2 == 1 {
                    coordinates
                } else {
                    coordinates - 1
                }
            }
            HashFamily::Dense | HashFamily::Projection => coordinates,
        }
    }
}

/// Where a [`Gf2Hash`]'s rows came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HashOrigin {
    /// Drawn by a family from the hash's seed.
    Drawn(HashFamily),
    /// The first `rows` rows given by the caller, the rest dense rows drawn from the hash's seed.
    Explicit {
        /// Rows the caller gave.
        rows: usize,
    },
}

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
        Self { rows_x, rows_z, num_qubits }
    }

    /// `num_rows` rows of `family` drawn from `seed`, reading no coordinate in `(exclude_x, exclude_z)`.
    /// The first rows are kept linearly independent up to the rank the family reaches over the readable coordinates, a dependent candidate redrawn; every row is nonzero while any coordinate is readable.
    /// The rows depend on `(num_qubits, seed, family, exclusion)` alone, not on `W`.
    pub(crate) fn draw_family(
        num_qubits: usize,
        num_rows: usize,
        seed: u64,
        family: HashFamily,
        exclude_x: &[u64; W],
        exclude_z: &[u64; W],
    ) -> Self {
        let excluded = |c: usize| {
            let (half, q) =
                if c < num_qubits { (exclude_x, c) } else { (exclude_z, c - num_qubits) };
            (half[q / 64] >> (q % 64)) & 1 == 1
        };
        let readable: Vec<usize> = (0..2 * num_qubits).filter(|&c| !excluded(c)).collect();
        if readable.is_empty() {
            return Self::from_rows(num_qubits, vec![[0; W]; num_rows], vec![[0; W]; num_rows]);
        }
        let projection_order = matches!(family, HashFamily::Projection).then(|| {
            partial_shuffle(
                readable.len(),
                readable.len(),
                &mut Rng::from_key(&[PROJECTION_SALT, seed]),
            )
        });
        let candidate = |row: usize, attempt: u32| -> KeyRow<W> {
            match family {
                HashFamily::Dense => {
                    let mut key = ([0u64; W], [0u64; W]);
                    for w in 0..W {
                        let mask = word_mask(num_qubits, w);
                        key.0[w] = row_word(seed, row, attempt, w, 0) & mask & !exclude_x[w];
                        key.1[w] = row_word(seed, row, attempt, w, 1) & mask & !exclude_z[w];
                    }
                    key
                }
                HashFamily::Sparse { weight } => {
                    let mut stream =
                        Rng::from_key(&[SPARSE_SALT, seed, row as u64, attempt as u64]);
                    let count = (weight as usize).min(readable.len());
                    let picked = partial_shuffle(readable.len(), count, &mut stream);
                    let coordinates: Vec<usize> = picked.iter().map(|&i| readable[i]).collect();
                    coordinate_row(num_qubits, &coordinates)
                }
                HashFamily::Projection => {
                    let order =
                        projection_order.as_ref().expect("drawn above for the projection family");
                    coordinate_row(num_qubits, &[readable[order[row % readable.len()]]])
                }
            }
        };
        let independent = num_rows.min(family.reachable_rank(readable.len()));
        let mut basis = Gf2Basis::<W>::default();
        Self::accept_rows(num_qubits, num_rows, independent, &mut basis, 0, candidate)
    }

    /// `given` rows followed by dense rows from `seed` up to `num_rows`, the completion kept independent of everything before it up to rank `min(num_rows, 2 · num_qubits)`.
    pub(crate) fn complete(
        num_qubits: usize,
        num_rows: usize,
        seed: u64,
        given: Vec<KeyRow<W>>,
    ) -> Self {
        let mut basis = Gf2Basis::<W>::default();
        for row in &given {
            basis.insert(row);
        }
        let first = given.len();
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
        let completed =
            Self::accept_rows(num_qubits, num_rows, independent, &mut basis, first, candidate);
        let (mut rows_x, mut rows_z): (Vec<_>, Vec<_>) = given.into_iter().unzip();
        let (tail_x, tail_z) = completed.rows();
        rows_x.extend_from_slice(&tail_x[first..]);
        rows_z.extend_from_slice(&tail_z[first..]);
        Self::from_rows(num_qubits, rows_x, rows_z)
    }

    /// Rows `first..num_rows` from `candidate(row, attempt)`: while `basis` holds fewer than `independent` rows a candidate is accepted only if it extends it, after that only if it is nonzero.
    /// A row that no candidate within a bounded number of attempts satisfies is taken as drawn, which keeps an unreachable request finite; rows before `first` are zero placeholders.
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
                if accepted || attempt == 1 << 12 {
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

/// The GF(2)-linear hash `h(v) = H·v` from a Pauli key `v = (x, z)` to a bucket index, for an `H` drawn by a [`HashFamily`] (dense random by default) or given explicitly.
///
/// Linearity gives `h(v ⊕ d) = h(v) ⊕ h(d)`, so a channel's output buckets are predictable from its input buckets.
/// The active rows are a prefix of one fixed matrix, so a refine splits every bucket in two by one row parity with the within-bucket order inherited, and a coarsen merges bucket pairs; neither needs a sort to keep the [`PauliSum`](crate::PauliSum) invariant.
#[derive(Clone, Debug)]
pub struct Gf2Hash<const W: usize> {
    /// All [`B_MAX_BITS`] rows of `H`.
    matrix: Gf2Matrix<W>,
    /// Active prefix length: `B = 1 << bits` buckets.
    bits: u8,
    /// Seed the drawn rows came from.
    seed: u64,
    /// Where the rows came from.
    origin: HashOrigin,
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
        Self::with_family(num_qubits, bits, seed, HashFamily::Dense)
    }

    /// As [`Gf2Hash::new`], with the rows drawn by `family`; a family that cannot reach full rank still builds, and [`Gf2Hash::rank_deficiency`] says so.
    ///
    /// # Panics
    ///
    /// As [`Gf2Hash::new`], and if [`HashFamily::validate`] rejects `family` over the `2 · num_qubits` coordinates.
    pub fn with_family(num_qubits: usize, bits: u8, seed: u64, family: HashFamily) -> Self {
        assert!(bits <= B_MAX_BITS, "Gf2Hash: bits {bits} exceeds B_MAX_BITS {B_MAX_BITS}",);
        debug_assert!(num_qubits <= 64 * W);
        if let Err(reason) = family.validate(2 * num_qubits) {
            panic!("{reason}");
        }
        let rows = B_MAX_BITS as usize;
        Self {
            matrix: Gf2Matrix::draw_family(num_qubits, rows, seed, family, &[0; W], &[0; W]),
            bits,
            seed,
            origin: HashOrigin::Drawn(family),
        }
    }

    /// A hash whose first rows are the caller's, completed to [`B_MAX_BITS`] rows by dense rows drawn from `seed`.
    ///
    /// Each given row is the list of coordinates it reads: `c < num_qubits` is the x-bit of qubit `c`, anything above the z-bit of qubit `c - num_qubits`.
    /// Dependent or zero rows are accepted, and [`Gf2Hash::rank_deficiency`] reports them.
    ///
    /// # Panics
    ///
    /// As [`Gf2Hash::new`], if more than [`B_MAX_BITS`] rows are given, or if a coordinate is `>= 2 · num_qubits`.
    pub fn from_coordinates(num_qubits: usize, bits: u8, seed: u64, rows: &[Vec<usize>]) -> Self {
        assert!(bits <= B_MAX_BITS, "Gf2Hash: bits {bits} exceeds B_MAX_BITS {B_MAX_BITS}",);
        assert!(
            rows.len() <= B_MAX_BITS as usize,
            "Gf2Hash::from_coordinates: {} rows exceed B_MAX_BITS {B_MAX_BITS}",
            rows.len()
        );
        debug_assert!(num_qubits <= 64 * W);
        let given = rows.iter().map(|row| checked_coordinate_row(num_qubits, row)).collect();
        Self {
            matrix: Gf2Matrix::complete(num_qubits, B_MAX_BITS as usize, seed, given),
            bits,
            seed,
            origin: HashOrigin::Explicit { rows: rows.len() },
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

    /// Where the rows came from.
    pub fn origin(&self) -> HashOrigin {
        self.origin
    }

    /// All [`B_MAX_BITS`] rows as `(x-masks, z-masks)`, the active ones first.
    pub fn rows(&self) -> (&[[u64; W]], &[[u64; W]]) {
        self.matrix.rows()
    }

    /// The rank of the first `min(B_MAX_BITS, 2 · num_qubits)` rows: how many bucket bits can be used before some bucket can never hold a term.
    pub fn independent_rows(&self) -> usize {
        let needed = (B_MAX_BITS as usize).min(2 * self.num_qubits());
        gf2_rank_wide(&(0..needed).map(|i| self.matrix.row(i)).collect::<Vec<_>>())
    }

    /// `Some(reason)` when the hash is not full rank, i.e. when [`Gf2Hash::independent_rows`] falls short of `min(B_MAX_BITS, 2 · num_qubits)`.
    pub fn rank_deficiency(&self) -> Option<String> {
        let needed = (B_MAX_BITS as usize).min(2 * self.num_qubits());
        let rank = self.independent_rows();
        (rank < needed).then(|| {
            format!(
                "the bucket hash's first {needed} rows have rank {rank}: beyond {rank} bucket bits, some buckets can never hold a term"
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
    /// Panics if already at `B_MAX_BITS`.
    pub fn refine(&mut self) {
        assert!(self.bits < B_MAX_BITS, "Gf2Hash::refine: already at B_MAX_BITS {B_MAX_BITS}",);
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

    /// Row `i < B_MAX_BITS` of `H`, which may lie beyond the active prefix.
    #[cfg(any(test, feature = "cuda"))]
    pub(crate) fn row(&self, i: usize) -> ([u64; W], [u64; W]) {
        self.matrix.row(i)
    }
}

/// The `p` GF(2) rows `part(v) = P·v` that split a sum across `2^p` partitions: a partition holds exactly the terms with `part(v)` equal to its rank.
///
/// A matrix separate from [`Gf2Hash`]'s and fixed for a run, since a row that moved as the bucket hash refines would change a term's owner mid-run.
/// Dependence on the bucket hash's rows costs load balance, never correctness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionRows<const W: usize> {
    /// One row per partition bit: `P = 1 << rows` partitions.
    matrix: Gf2Matrix<W>,
}

impl<const W: usize> PartitionRows<W> {
    /// Panics unless the rows split a sum over `num_qubits` qubits into `partitions` partitions; dependence on `hash` is reported by [`PartitionRows::rank_deficiency`], never a panic.
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
        let _ = hash;
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
        assert!(bits <= P_MAX_BITS, "PartitionRows: bits {bits} exceeds P_MAX_BITS {P_MAX_BITS}",);
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

    /// `bits` rows drawn by `family` from `seed`, reading no coordinate in `(exclude_x, exclude_z)`; the dense family gives exactly [`Self::from_seed_excluding`]'s rows.
    /// A family that cannot reach full rank still builds, and [`Self::rank_deficiency`] says so.
    ///
    /// # Panics
    ///
    /// As [`Self::from_seed_excluding`], and if [`HashFamily::validate`] rejects `family` over the readable coordinates.
    pub fn from_family(
        num_qubits: usize,
        bits: u8,
        seed: u64,
        family: HashFamily,
        exclude_x: &[u64; W],
        exclude_z: &[u64; W],
    ) -> Self {
        if family == HashFamily::Dense {
            return Self::from_seed_excluding(num_qubits, bits, seed, exclude_x, exclude_z);
        }
        assert!(bits <= P_MAX_BITS, "PartitionRows: bits {bits} exceeds P_MAX_BITS {P_MAX_BITS}",);
        let readable: usize = (0..W)
            .map(|w| {
                let mask = word_mask(num_qubits, w);
                ((mask & !exclude_x[w]).count_ones() + (mask & !exclude_z[w]).count_ones()) as usize
            })
            .sum();
        if let Err(reason) = family.validate(readable) {
            panic!("{reason}");
        }
        let seed = mix64(seed) ^ PARTITION_ROW_SALT;
        Self {
            matrix: Gf2Matrix::draw_family(
                num_qubits,
                bits as usize,
                seed,
                family,
                exclude_x,
                exclude_z,
            ),
        }
    }

    /// Explicit rows, each the list of coordinates it reads as in [`Gf2Hash::from_coordinates`]; dependent or zero rows are accepted and reported by [`Self::rank_deficiency`].
    ///
    /// # Panics
    ///
    /// If there are more than [`P_MAX_BITS`] rows, or if a coordinate is `>= 2 · num_qubits`.
    pub fn from_coordinates(num_qubits: usize, rows: &[Vec<usize>]) -> Self {
        let (rows_x, rows_z) =
            rows.iter().map(|row| checked_coordinate_row::<W>(num_qubits, row)).unzip();
        Self::from_rows(num_qubits, rows_x, rows_z)
    }

    /// Build from explicit rows, masked to the live qubit columns; dependent or zero rows are accepted and reported by [`Self::rank_deficiency`].
    ///
    /// # Panics
    ///
    /// Panics if `rows_x` and `rows_z` differ in length, or if there are more than [`P_MAX_BITS`] rows.
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

        Self { matrix: Gf2Matrix::from_rows(num_qubits, rows_x, rows_z) }
    }

    /// Rows that label a qubit cut: `log2(blocks.len())` z-only rows giving block `b` the label `b`.
    ///
    /// A term's label is the XOR of the labels of the blocks it has odd z-weight in, so an x-only generator is local under any cut and a `ZZ(i, j)` bond is remote exactly when it crosses blocks.
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
        assert!(bits <= P_MAX_BITS, "PartitionRows: bits {bits} exceeds P_MAX_BITS {P_MAX_BITS}",);
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
        gf2_rank_wide(&(0..self.matrix.num_rows()).map(|i| self.matrix.row(i)).collect::<Vec<_>>())
    }

    /// Every way these rows fall short, as reasons: dependent rows, which leave partitions that can never hold a term, and overlap with the span of `hash`'s rows, which correlates partitions with buckets so that some pairs of the two stay empty.
    pub fn rank_deficiency(&self, hash: &Gf2Hash<W>) -> Vec<String> {
        let mut reasons = Vec::new();
        let bits = self.matrix.num_rows();
        let rank = self.independent_rows();
        if rank < bits {
            reasons.push(format!(
                "the {bits} partition rows have rank {rank}: only {} of the {} partitions can hold a term",
                1usize << rank,
                1usize << bits
            ));
        }
        let bucket = hash.independent_rows();
        let needed = (B_MAX_BITS as usize).min(2 * hash.num_qubits());
        let stacked: Vec<KeyRow<W>> = (0..bits)
            .map(|i| self.matrix.row(i))
            .chain((0..needed).map(|i| hash.matrix.row(i)))
            .collect();
        let joint = gf2_rank_wide(&stacked);
        let attainable = (rank + bucket).min(2 * self.num_qubits());
        if joint < attainable {
            reasons.push(format!(
                "the partition rows overlap the span of the bucket hash's rows ({} of {attainable} dimensions lost): partitions and buckets are correlated, so some of their pairs stay empty",
                attainable - joint
            ));
        }
        reasons
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

/// The first `count` entries of a uniformly random permutation of `0..len`, by a partial Fisher–Yates shuffle.
fn partial_shuffle(len: usize, count: usize, stream: &mut Rng) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    for i in 0..count {
        let j = i + (stream.next_u64() % (len - i) as u64) as usize;
        order.swap(i, j);
    }
    order.truncate(count);
    order
}

/// The row setting each of `coordinates`: `c` is the x-bit of qubit `c` below `num_qubits` and the z-bit of qubit `c - num_qubits` above.
fn coordinate_row<const W: usize>(num_qubits: usize, coordinates: &[usize]) -> KeyRow<W> {
    let mut key = ([0u64; W], [0u64; W]);
    for &c in coordinates {
        let (half, q) = if c < num_qubits { (&mut key.0, c) } else { (&mut key.1, c - num_qubits) };
        half[q / 64] |= 1u64 << (q % 64);
    }
    key
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

/// The row reading `coordinates`, each checked to lie below `2 · num_qubits`.
fn checked_coordinate_row<const W: usize>(num_qubits: usize, coordinates: &[usize]) -> KeyRow<W> {
    for &c in coordinates {
        assert!(
            c < 2 * num_qubits,
            "coordinate {c} is outside 0..{} for {num_qubits} qubits",
            2 * num_qubits
        );
    }
    coordinate_row(num_qubits, coordinates)
}
