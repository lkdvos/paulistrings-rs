//! [`HashSum`], a bucketed sum whose buckets are open-addressing hash tables, and its rebucketing.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::pauli_sum::storage::rebucketed_bits;
use crate::pauli_sum::{Gf2Hash, PauliSum};

/// One term, stored inline in its bucket's table.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry<const W: usize> {
    pub(crate) x: [u64; W],
    pub(crate) z: [u64; W],
    pub(crate) c: Complex64,
}

impl<const W: usize> Entry<W> {
    pub(crate) fn same_key(&self, other: &Self) -> bool {
        self.x == other.x && self.z == other.z
    }
}

pub(crate) use super::open_table::Table;

/// The 64×64→128 multiply folded to 64 bits.
#[inline(always)]
fn folded_multiply(a: u64, b: u64) -> u64 {
    let product = (a as u128) * (b as u128);
    (product as u64) ^ ((product >> 64) as u64)
}

const KEY_SEEDS: [u64; 4] =
    [0x243F_6A88_85A3_08D3, 0x1319_8A2E_0370_7344, 0xA409_3822_299F_31D0, 0x082E_FA98_EC4E_6C89];

/// The in-bucket key hash; the bucket index is a GF(2)-linear function of the key, so this one must be nonlinear in every bit.
#[inline(always)]
pub(crate) fn key_hash<const W: usize>(x: &[u64; W], z: &[u64; W]) -> u64 {
    let mut h = 0u64;
    for w in 0..W {
        h = h.rotate_left(29)
            ^ folded_multiply(x[w] ^ KEY_SEEDS[(2 * w) % 4], z[w] ^ KEY_SEEDS[(2 * w + 1) % 4]);
    }
    h
}

#[inline(always)]
pub(crate) fn entry_hash<const W: usize>(e: &Entry<W>) -> u64 {
    key_hash(&e.x, &e.z)
}

/// `c` into `table` under `e`'s key, accumulating onto an existing entry.
#[inline(always)]
pub(crate) fn accumulate<const W: usize>(table: &mut Table<W>, e: Entry<W>) {
    table.accumulate(e);
}

/// A weighted sum of Pauli strings partitioned by a GF(2)-linear hash into hash-table buckets, updated in place by each layer.
pub struct HashSum<const W: usize> {
    pub(crate) buckets: Vec<Table<W>>,
    pub(crate) hash: Gf2Hash<W>,
    pub(crate) num_qubits: usize,
    pub(crate) len: usize,
}

impl<const W: usize> HashSum<W> {
    /// The terms of `sum`, under its hash and bucket count.
    pub fn from_pauli_sum(sum: &PauliSum<W>) -> Self {
        let hash = sum.hash().clone();
        let buckets = sum
            .buckets()
            .par_iter()
            .map(|columns| {
                let mut table = Table::with_capacity(columns.len());
                for i in 0..columns.len() {
                    let e = Entry { x: columns.x[i], z: columns.z[i], c: columns.coeff[i] };
                    table.insert_unique(entry_hash(&e), e);
                }
                table
            })
            .collect();
        Self { buckets, hash, num_qubits: sum.num_qubits(), len: sum.len() }
    }

    /// The terms as a canonical [`PauliSum`] under the same hash.
    pub fn to_pauli_sum(&self) -> PauliSum<W> {
        let mut terms: Vec<Entry<W>> =
            self.buckets.iter().flat_map(|table| table.iter().copied()).collect();
        terms.par_sort_unstable_by(|a, b| (a.x, a.z).cmp(&(b.x, b.z)));
        let x: Vec<[u64; W]> = terms.iter().map(|e| e.x).collect();
        let z: Vec<[u64; W]> = terms.iter().map(|e| e.z).collect();
        let c: Vec<Complex64> = terms.iter().map(|e| e.c).collect();
        PauliSum::from_key_sorted(&x, &z, &c, self.hash.clone(), self.num_qubits)
    }

    /// Number of terms.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the sum has no terms.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of buckets.
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Summed table capacity in entries, the resident footprint once divided into `len`.
    pub fn capacity(&self) -> usize {
        self.buckets.iter().map(Table::capacity).sum()
    }

    /// The partitioning hash.
    pub fn hash(&self) -> &Gf2Hash<W> {
        &self.hash
    }

    /// Keep the terms `keep` accepts.
    pub(crate) fn retain(&mut self, keep: impl Fn(&[u64; W], &[u64; W], Complex64) -> bool + Sync) {
        self.buckets.par_iter_mut().for_each(|table| table.retain(|e| keep(&e.x, &e.z, e.c)));
        self.recount();
    }

    pub(crate) fn recount(&mut self) {
        self.len = self.buckets.iter().map(Table::len).sum();
    }

    /// Double the bucket count by the hash's next row.
    pub(crate) fn refine(&mut self) {
        self.hash.refine();
        let new_bit = self.hash.bits() - 1;
        let hash = &self.hash;
        let upper: Vec<Table<W>> = self
            .buckets
            .par_iter_mut()
            .map(|table| {
                let moving = table.extract(|e| (hash.bucket_of(&e.x, &e.z) >> new_bit) & 1 == 1);
                let mut upper = Table::with_capacity(moving.len());
                for e in moving {
                    upper.insert_unique(entry_hash(&e), e);
                }
                table.shrink_to_fit();
                upper
            })
            .collect();
        self.buckets.extend(upper);
    }

    /// Halve the bucket count, merging bucket pairs `(i, i + B/2)`.
    pub(crate) fn coarsen(&mut self) {
        self.hash.coarsen();
        let half = self.buckets.len() / 2;
        let upper = self.buckets.split_off(half);
        self.buckets.par_iter_mut().zip(upper).for_each(|(lower, upper)| {
            lower.reserve(upper.len());
            for e in upper.iter() {
                lower.insert_unique(entry_hash(e), *e);
            }
        });
    }

    /// Move to [`rebucketed_bits`] for `target` terms per bucket.
    pub(crate) fn rebucket(&mut self, target: usize, min_buckets: usize) {
        let want = rebucketed_bits(self.len, self.hash.bits(), target, min_buckets)
            .min(self.hash.max_bits());
        while self.hash.bits() < want {
            self.refine();
        }
        while self.hash.bits() > want {
            self.coarsen();
        }
    }
}
