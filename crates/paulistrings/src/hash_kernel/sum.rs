//! [`HashSum`], a bucketed sum whose buckets are indexed dense tables, and its rebucketing.

use num_complex::Complex64;
use rayon::prelude::*;

use super::table::{Entry, Table};
use crate::pauli_sum::storage::rebucketed_bits;
use crate::pauli_sum::{Gf2Hash, PauliSum};

/// A weighted sum of Pauli strings partitioned by a GF(2)-linear hash into dense indexed buckets, updated in place by each layer.
pub struct HashSum<const W: usize> {
    pub(crate) buckets: Vec<Table<W>>,
    pub(crate) hash: Gf2Hash<W>,
    pub(crate) num_qubits: usize,
    pub(crate) len: usize,
}

impl<const W: usize> HashSum<W> {
    /// The terms of `sum`, under its hash and bucket count.
    pub fn from_pauli_sum(sum: &PauliSum<W>) -> Self {
        let buckets = sum
            .buckets()
            .par_iter()
            .map(|columns| {
                Table::from_entries(
                    (0..columns.len())
                        .map(|i| Entry { x: columns.x[i], z: columns.z[i], c: columns.coeff[i] })
                        .collect(),
                )
            })
            .collect();
        Self { buckets, hash: sum.hash().clone(), num_qubits: sum.num_qubits(), len: sum.len() }
    }

    /// The terms as a canonical [`PauliSum`] under the same hash.
    pub fn to_pauli_sum(&self) -> PauliSum<W> {
        let mut terms: Vec<Entry<W>> =
            self.buckets.iter().flat_map(|table| table.entries().iter().copied()).collect();
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

    /// Resident bytes of entries and indices.
    pub fn bytes(&self) -> usize {
        self.buckets.iter().map(Table::bytes).sum()
    }

    /// The partitioning hash.
    pub fn hash(&self) -> &Gf2Hash<W> {
        &self.hash
    }

    pub(crate) fn retain(&mut self, keep: impl Fn(&[u64; W], &[u64; W], Complex64) -> bool + Sync) {
        self.buckets.par_iter_mut().for_each(|table| table.retain(|e| keep(&e.x, &e.z, e.c)));
        self.recount();
    }

    pub(crate) fn recount(&mut self) {
        self.len = self.buckets.iter().map(Table::len).sum();
    }

    /// Double the bucket count by the hash's next row.
    fn refine(&mut self) {
        self.hash.refine();
        let new_bit = self.hash.bits() - 1;
        let hash = &self.hash;
        let upper: Vec<Table<W>> = self
            .buckets
            .par_iter_mut()
            .map(|table| table.split_off_where(|e| (hash.bucket_of(&e.x, &e.z) >> new_bit) & 1 == 1))
            .collect();
        self.buckets.extend(upper);
    }

    /// Halve the bucket count, merging bucket pairs `(i, i + B/2)`.
    fn coarsen(&mut self) {
        self.hash.coarsen();
        let half = self.buckets.len() / 2;
        let upper = self.buckets.split_off(half);
        self.buckets.par_iter_mut().zip(upper).for_each(|(lower, upper)| lower.absorb(upper));
    }

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
