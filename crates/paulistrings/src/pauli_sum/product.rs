//! Operator products of two [`PauliSum`]s: [`PauliSum::product`], [`PauliSum::liouvillian`] and [`PauliSum::anticommutator`].
//!
//! A string product XORs keys and the bucket hash is GF(2)-linear, so output bucket `β′` reads only the larger operand's buckets `β′ ⊕ h(s)` over the smaller operand's terms `s`.
//! Each output bucket is therefore one task that accumulates into its own [`Table`], and no two tasks write the same key.

use rayon::prelude::*;

use super::storage::{
    desired_bits, BucketColumns, PauliSum, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN,
};
use super::table::{Entry, Table};
use crate::coefficient::Coefficient;
use crate::pauli_string::PauliString;
use crate::phase::Phase;

/// Which pairs a product keeps and the factor it puts on them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pairing {
    /// `A·B`: every pair, at `i^k`.
    Product,
    /// `i[A, B]`: anticommuting pairs, at `2·i^(k+1)`, which is real.
    Liouvillian,
    /// `{A, B}`: commuting pairs, at `2·i^k`, which is real.
    Anticommutator,
}

impl Pairing {
    /// The term `(a·P_a)·(b·P_b)` contributes, or `None` when the pairing drops the pair.
    #[inline(always)]
    fn term<const W: usize, C: Coefficient>(
        self,
        left: &Entry<W, C>,
        right: &Entry<W, C>,
    ) -> Option<Entry<W, C>> {
        let mut key = PauliString { x: left.x, z: left.z };
        let other = PauliString { x: right.x, z: right.z };
        let (keep, extra) = match self {
            Pairing::Product => (true, Phase::ONE),
            Pairing::Liouvillian => (!key.commutes_with(&other), Phase::I),
            Pairing::Anticommutator => (key.commutes_with(&other), Phase::ONE),
        };
        if !keep {
            return None;
        }
        let phase = key.mul_assign(&other) + extra;
        let mut c = left.c * right.c;
        if self != Pairing::Product {
            c = c * C::real(2.0);
        }
        Some(Entry { x: key.x, z: key.z, c: c.mul_phase(phase) })
    }
}

/// The smaller operand's terms sorted by their bucket under the output hash, with the run boundaries.
struct Groups<const W: usize, C: Coefficient> {
    terms: Vec<Entry<W, C>>,
    /// The bucket of each run, ascending.
    buckets: Vec<u32>,
    /// Run `g` is `terms[starts[g]..starts[g + 1]]`.
    starts: Vec<usize>,
}

impl<const W: usize, C: Coefficient> Groups<W, C> {
    fn new(sum: &PauliSum<W, C>, hash: &super::Gf2Hash<W>) -> Self {
        let mut keyed: Vec<(u32, Entry<W, C>)> =
            sum.iter().map(|(x, z, c)| (hash.bucket_of(x, z), Entry { x: *x, z: *z, c })).collect();
        keyed.sort_unstable_by_key(|(bucket, _)| *bucket);
        let mut buckets = Vec::new();
        let mut starts = Vec::new();
        for (i, (bucket, _)) in keyed.iter().enumerate() {
            if buckets.last() != Some(bucket) {
                buckets.push(*bucket);
                starts.push(i);
            }
        }
        starts.push(keyed.len());
        Self { terms: keyed.into_iter().map(|(_, e)| e).collect(), buckets, starts }
    }
}

/// The table's nonzero terms as sorted bucket columns.
fn sorted_columns<const W: usize, C: Coefficient>(table: &Table<W, C>) -> BucketColumns<W, C> {
    let mut entries: Vec<&Entry<W, C>> =
        table.entries().iter().filter(|e| e.c.is_nonzero()).collect();
    entries.sort_unstable_by(|a, b| (&a.x, &a.z).cmp(&(&b.x, &b.z)));
    let mut columns = BucketColumns::with_capacity(entries.len());
    for e in entries {
        columns.push(e.x, e.z, e.c);
    }
    columns
}

impl<const W: usize, C: Coefficient> PauliSum<W, C> {
    fn pair(&self, other: &Self, pairing: Pairing) -> Self {
        assert_eq!(
            self.num_qubits, other.num_qubits,
            "PauliSum::{pairing:?}: num_qubits mismatch ({} vs {})",
            self.num_qubits, other.num_qubits,
        );
        let self_is_large = self.len >= other.len;
        let (large, small) = if self_is_large { (self, other) } else { (other, self) };

        // The large operand's own bucket count, raised to the parallelism floor when the pair count warrants it; only a sum below that floor is refined, so the copy is small.
        let pairs = large.len.saturating_mul(small.len);
        let floor = desired_bits(pairs, DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS)
            .min(DEFAULT_MIN_BUCKETS.trailing_zeros() as u8)
            .min(large.hash.max_bits());
        let refined;
        let large = if floor > large.hash.bits() {
            let mut copy = large.clone();
            while copy.hash.bits() < floor {
                copy.refine();
            }
            refined = copy;
            &refined
        } else {
            large
        };

        let groups = Groups::new(small, &large.hash);
        let buckets: Vec<BucketColumns<W, C>> = (0..large.buckets.len())
            .into_par_iter()
            .map_init(Table::<W, C>::default, |table, out| {
                table.clear();
                for (g, &delta) in groups.buckets.iter().enumerate() {
                    let input = &large.buckets[out ^ delta as usize];
                    let run = &groups.terms[groups.starts[g]..groups.starts[g + 1]];
                    table.reserve(input.len());
                    for i in 0..input.len() {
                        let l = Entry { x: input.x[i], z: input.z[i], c: input.coeff[i] };
                        for s in run {
                            let term = if self_is_large {
                                pairing.term(&l, s)
                            } else {
                                pairing.term(s, &l)
                            };
                            if let Some(t) = term {
                                table.accumulate(t.x, t.z, t.c);
                            }
                        }
                    }
                }
                sorted_columns(table)
            })
            .collect();

        let len = buckets.iter().map(BucketColumns::len).sum();
        let mut out = Self { buckets, hash: large.hash.clone(), num_qubits: self.num_qubits, len };
        out.rebucket(DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS);
        out
    }

    /// The operator product `self · other`; terms summing to exactly zero are dropped.
    ///
    /// The result is partitioned under the larger operand's hash rows, at the bucket count its size asks for.
    ///
    /// # Panics
    ///
    /// If the two sums disagree about `num_qubits`, or if `C` is real: a product of anticommuting Hermitian strings carries an odd phase that no real coefficient holds.
    pub fn product(&self, other: &Self) -> Self {
        assert!(
            !C::IS_REAL,
            "PauliSum::product: a product is not Hermitian, so a real coefficient type ({}) cannot hold it; cast to a complex type first",
            C::NAME,
        );
        self.pair(other, Pairing::Product)
    }

    /// The Liouvillian `i[self, other] = i(self · other − other · self)`, Hermitian whenever both operands are, so a real coefficient type stays exact.
    ///
    /// Only anticommuting string pairs contribute; terms summing to exactly zero are dropped, and the result is partitioned as [`Self::product`]'s.
    ///
    /// # Panics
    ///
    /// If the two sums disagree about `num_qubits`.
    pub fn liouvillian(&self, other: &Self) -> Self {
        self.pair(other, Pairing::Liouvillian)
    }

    /// The anticommutator `{self, other} = self · other + other · self`, Hermitian whenever both operands are.
    ///
    /// Only commuting string pairs contribute; terms summing to exactly zero are dropped, and the result is partitioned as [`Self::product`]'s.
    ///
    /// # Panics
    ///
    /// If the two sums disagree about `num_qubits`.
    pub fn anticommutator(&self, other: &Self) -> Self {
        self.pair(other, Pairing::Anticommutator)
    }
}

#[cfg(test)]
mod tests;
