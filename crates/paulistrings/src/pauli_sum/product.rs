//! Operator products of two [`PauliSum`]s: [`PauliSum::product`], [`PauliSum::liouvillian`] and [`PauliSum::anticommutator`].
//!
//! A string product XORs keys and the bucket hash is GF(2)-linear, so output bucket `β′` reads only the larger operand's buckets `β′ ⊕ h(s)` over the smaller operand's terms `s`.
//! Each output bucket is therefore one task that accumulates into its own [`Table`], and no two tasks write the same key.
//! A Liouvillian visits only the pairs whose supports meet, through a [`support::SupportIndex`] of the larger operand.

use rayon::prelude::*;

use super::hash::Gf2Hash;
use super::storage::{
    desired_bits, merge_summed, scaled_terms, BucketColumns, PauliSum, DEFAULT_MIN_BUCKETS,
    DEFAULT_TARGET_BUCKET_LEN,
};
pub(crate) use super::table::Entry;
use super::table::Table;
use crate::coefficient::Coefficient;
use crate::pauli_string::PauliString;
use crate::phase::Phase;
use support::{SupportIndex, SCAN_ABOVE_WEIGHT};

/// Which pairs a product keeps and the factor it puts on them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pairing {
    /// `A·B`: every pair, at `i^k`.
    Product,
    /// `i[A, B]`: anticommuting pairs, at `2·i^(k+1)`, which is real.
    Liouvillian,
    /// `{A, B}`: commuting pairs, at `2·i^k`, which is real.
    Anticommutator,
}

impl Pairing {
    /// The name a trace records the call under.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Pairing::Product => "product",
            Pairing::Liouvillian => "liouvillian",
            Pairing::Anticommutator => "anticommutator",
        }
    }

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

/// A generator's terms sorted by their bucket under the output hash, with the run boundaries.
struct Groups<const W: usize, C: Coefficient> {
    terms: Vec<Entry<W, C>>,
    /// The bucket of each run, ascending.
    buckets: Vec<u32>,
    /// Run `g` is `terms[starts[g]..starts[g + 1]]`.
    starts: Vec<usize>,
}

impl<const W: usize, C: Coefficient> Groups<W, C> {
    fn new(terms: impl Iterator<Item = Entry<W, C>>, hash: &Gf2Hash<W>) -> Self {
        let mut keyed: Vec<(u32, Entry<W, C>)> =
            terms.map(|e| (hash.bucket_of(&e.x, &e.z), e)).collect();
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

/// One generator against one operand, as an output pass reads them.
pub(crate) struct Job<'a, const W: usize, C: Coefficient> {
    groups: Groups<W, C>,
    operand: &'a PauliSum<W, C>,
    /// Built for a Liouvillian only, since disjoint supports always commute.
    index: Option<SupportIndex>,
    /// Whether the generator is the left factor of each pair.
    generator_left: bool,
}

impl<'a, const W: usize, C: Coefficient> Job<'a, W, C> {
    pub(crate) fn new(
        generator: impl Iterator<Item = Entry<W, C>>,
        operand: &'a PauliSum<W, C>,
        pairing: Pairing,
        generator_left: bool,
    ) -> Self {
        let index = (pairing == Pairing::Liouvillian).then(|| SupportIndex::new(operand));
        Self::with_index(generator, operand, index, generator_left)
    }

    /// [`Job::new`] with `operand`'s support index already built, which must be `Some` exactly for a Liouvillian.
    pub(crate) fn with_index(
        generator: impl Iterator<Item = Entry<W, C>>,
        operand: &'a PauliSum<W, C>,
        index: Option<SupportIndex>,
        generator_left: bool,
    ) -> Self {
        Self { groups: Groups::new(generator, &operand.hash), operand, index, generator_left }
    }

    /// Every pair of this job landing in output bucket `out`, accumulated into `table`.
    #[inline]
    fn accumulate_into(
        &self,
        out: usize,
        pairing: Pairing,
        table: &mut Table<W, C>,
        candidates: &mut Vec<u64>,
    ) {
        for (g, &delta) in self.groups.buckets.iter().enumerate() {
            let b = out ^ delta as usize;
            let input = &self.operand.buckets[b];
            let mut emit = |i: usize, s: &Entry<W, C>| {
                let l = Entry { x: input.x[i], z: input.z[i], c: input.coeff[i] };
                let term =
                    if self.generator_left { pairing.term(s, &l) } else { pairing.term(&l, s) };
                if let Some(t) = term {
                    table.accumulate(t.x, t.z, t.c);
                }
            };
            for s in &self.groups.terms[self.groups.starts[g]..self.groups.starts[g + 1]] {
                let support: [u64; W] = std::array::from_fn(|w| s.x[w] | s.z[w]);
                let weight: u32 = support.iter().map(|w| w.count_ones()).sum();
                match &self.index {
                    Some(index) if weight <= SCAN_ABOVE_WEIGHT => {
                        index.candidates(b, &support, candidates);
                        for (w, &word) in candidates.iter().enumerate() {
                            let mut bits = word;
                            while bits != 0 {
                                emit(w * 64 + bits.trailing_zeros() as usize, s);
                                bits &= bits - 1;
                            }
                        }
                    }
                    _ => (0..input.len()).for_each(|i| emit(i, s)),
                }
            }
        }
    }
}

/// The factors of `dest ← alpha · pairs + beta · dest`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Weights<C> {
    pub(crate) alpha: C,
    pub(crate) beta: C,
}

impl<C: Coefficient> Weights<C> {
    /// A fresh result: `alpha = 1`, `beta = 0`.
    pub(crate) fn fresh() -> Self {
        Self { alpha: C::ONE, beta: C::ZERO }
    }
}

/// The table's terms merged with `beta · dest`, key-sorted, keeping the nonzero sums `keep` accepts.
fn drained_into<const W: usize, C: Coefficient>(
    table: &Table<W, C>,
    dest: &BucketColumns<W, C>,
    beta: C,
    keep: &impl Fn(&[u64; W], &[u64; W], C) -> bool,
) -> BucketColumns<W, C> {
    let mut entries: Vec<&Entry<W, C>> = table.entries().iter().collect();
    entries.sort_unstable_by(|a, b| (&a.x, &a.z).cmp(&(&b.x, &b.z)));
    let held = if beta == C::ZERO { 0 } else { dest.len() };
    merge_summed(
        entries.iter().map(|e| (e.x, e.z, e.c)),
        scaled_terms(dest, beta).take(held),
        entries.len() + held,
        keep,
    )
}

/// `dest ← alpha · Σ pairs + beta · dest` over every pair of every job, where the jobs' generators already carry `alpha` and `dest` is partitioned exactly as the operands; one bucket of `dest` is replaced at a time, keeping the summed terms `keep` accepts.
///
/// # Panics
///
/// If `dest` or an operand is partitioned differently from `hash`, or the qubit counts differ.
pub(crate) fn accumulate_into<const W: usize, C: Coefficient>(
    dest: &mut PauliSum<W, C>,
    beta: C,
    jobs: &[Job<'_, W, C>],
    pairing: Pairing,
    keep: &(impl Fn(&[u64; W], &[u64; W], C) -> bool + Sync),
) {
    for job in jobs {
        assert!(
            job.operand.hash.same_rows_as(&dest.hash) && job.operand.hash.bits() == dest.hash.bits(),
            "accumulate_into: operands partitioned differently from the destination",
        );
        assert_eq!(job.operand.num_qubits, dest.num_qubits, "accumulate_into: num_qubits mismatch");
    }
    dest.buckets.par_iter_mut().enumerate().for_each_init(
        || (Table::<W, C>::default(), Vec::new()),
        |(table, candidates), (out, columns)| {
            table.clear();
            for job in jobs {
                job.accumulate_into(out, pairing, table, candidates);
            }
            *columns = drained_into(table, columns, beta, keep);
        },
    );
    dest.recount();
}

impl<const W: usize, C: Coefficient> PauliSum<W, C> {
    /// `self ← weights.alpha · pairing(generator, operand) + weights.beta · self`, keeping the summed terms `keep` accepts; `self` ends partitioned under the larger operand's hash rows.
    pub(crate) fn pair_into(
        &mut self,
        generator: &Self,
        operand: &Self,
        pairing: Pairing,
        weights: Weights<C>,
        keep: &(impl Fn(&[u64; W], &[u64; W], C) -> bool + Sync),
    ) {
        assert!(
            generator.num_qubits == operand.num_qubits && self.num_qubits == operand.num_qubits,
            "PauliSum::{pairing:?}: num_qubits mismatch (generator {}, operand {}, destination {})",
            generator.num_qubits,
            operand.num_qubits,
            self.num_qubits,
        );
        let generator_is_large = generator.len > operand.len;
        let (large, small) =
            if generator_is_large { (generator, operand) } else { (operand, generator) };

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

        self.align_destination(&large.hash, weights.beta);
        let alpha = weights.alpha;
        let scaled = small.iter().map(|(x, z, c)| Entry { x: *x, z: *z, c: c * alpha });
        let job = Job::new(scaled, large, pairing, !generator_is_large);
        accumulate_into(self, weights.beta, &[job], pairing, keep);
        self.rebucket(DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS);
    }

    /// `pairing(self, other)` as a new sum, keeping every summed term.
    fn paired(&self, other: &Self, pairing: Pairing) -> Self {
        let mut out = Self::empty_with_hash(self.num_qubits, other.hash.clone());
        out.pair_into(self, other, pairing, Weights::fresh(), &|_, _, _| true);
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
        self.paired(other, Pairing::Product)
    }

    /// The Liouvillian `i[self, other] = i(self · other − other · self)`, Hermitian whenever both operands are, so a real coefficient type stays exact.
    ///
    /// Only anticommuting string pairs contribute; terms summing to exactly zero are dropped, and the result is partitioned as [`Self::product`]'s.
    ///
    /// # Panics
    ///
    /// If the two sums disagree about `num_qubits`.
    pub fn liouvillian(&self, other: &Self) -> Self {
        self.paired(other, Pairing::Liouvillian)
    }

    /// The anticommutator `{self, other} = self · other + other · self`, Hermitian whenever both operands are.
    ///
    /// Only commuting string pairs contribute; terms summing to exactly zero are dropped, and the result is partitioned as [`Self::product`]'s.
    ///
    /// # Panics
    ///
    /// If the two sums disagree about `num_qubits`.
    pub fn anticommutator(&self, other: &Self) -> Self {
        self.paired(other, Pairing::Anticommutator)
    }
}

pub(crate) mod support;
#[cfg(test)]
mod tests;
