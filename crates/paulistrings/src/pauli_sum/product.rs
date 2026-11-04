//! Operator products of two [`PauliSum`]s: [`PauliSum::product`], [`PauliSum::liouvillian`] and [`PauliSum::anticommutator`].
//!
//! A string product XORs keys and the bucket hash is GF(2)-linear, so output bucket `β′` reads only the larger operand's buckets `β′ ⊕ h(s)` over the smaller operand's terms `s`.
//! Each output bucket is therefore one task that accumulates into its own [`Table`], and no two tasks write the same key.
//! A Liouvillian visits only the pairs whose supports meet, through a [`support::SupportIndex`] of the larger operand.

use rayon::prelude::*;

use super::hash::Gf2Hash;
use super::storage::{
    desired_bits, BucketColumns, PauliSum, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN,
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
    /// The term `(a·P_a)·(b·P_b)` contributes, or `None` when the pairing drops the pair.
    #[inline(always)]
    fn term<const W: usize, C: Coefficient>(
        self,
        left: &Entry<W, C>,
        right: &Entry<W, C>,
    ) -> Option<Entry<W, C>> {
        let key = PauliString { x: left.x, z: left.z };
        let other = PauliString { x: right.x, z: right.z };
        let keep = match self {
            Pairing::Product => true,
            Pairing::Liouvillian => !key.commutes_with(&other),
            Pairing::Anticommutator => key.commutes_with(&other),
        };
        keep.then(|| self.kept_term(left, right))
    }

    /// The term of a kept pair whose product phase is `i^k`, the key being `left ⊕ right`.
    #[inline(always)]
    fn term_with_phase<const W: usize, C: Coefficient>(
        self,
        left: &Entry<W, C>,
        right: &Entry<W, C>,
        k: u8,
    ) -> Entry<W, C> {
        let c = left.c * right.c;
        let (c, phase) = match self {
            Pairing::Product => (c, Phase::new(k)),
            Pairing::Liouvillian => (c * C::real(2.0), Phase::new(k + 1)),
            Pairing::Anticommutator => (c * C::real(2.0), Phase::new(k)),
        };
        Entry {
            x: std::array::from_fn(|w| left.x[w] ^ right.x[w]),
            z: std::array::from_fn(|w| left.z[w] ^ right.z[w]),
            c: c.mul_phase(phase),
        }
    }

    /// [`Self::term`] for a pair the caller already knows the pairing keeps.
    #[inline(always)]
    fn kept_term<const W: usize, C: Coefficient>(
        self,
        left: &Entry<W, C>,
        right: &Entry<W, C>,
    ) -> Entry<W, C> {
        let mut key = PauliString { x: left.x, z: left.z };
        let phase = key.mul_assign(&PauliString { x: right.x, z: right.z });
        let c = left.c * right.c;
        let (c, phase) = match self {
            Pairing::Product => (c, phase),
            Pairing::Liouvillian => (c * C::real(2.0), phase + Phase::I),
            Pairing::Anticommutator => (c * C::real(2.0), phase),
        };
        Entry { x: key.x, z: key.z, c: c.mul_phase(phase) }
    }
}

/// Runs at least this long are bit-sliced into [`Planes`]; shorter ones visit through the [`SupportIndex`].
const SLICE_MIN_RUN: usize = 4;

/// A generator run transposed by qubit and Pauli: plane `3·q + p` is the bitset of members carrying Pauli `p` (`X = 0`, `Y = 1`, `Z = 2`) on qubit `q`.
///
/// A product of Hermitian strings carries `i^k` with `k` the sum over qubits of `+1` where the right factor's Pauli follows the left's in the cycle `X → Y → Z` and `−1` where it precedes it, so [`Planes::phases`] counts `k mod 4` for every member at once in two bit planes.
struct Planes {
    words: usize,
    rows: Vec<u64>,
}

/// The packed Pauli index `0 = X, 1 = Y, 2 = Z` of a qubit with bits `(x, z)`, not the identity.
#[inline(always)]
fn pauli_index(x: bool, z: bool) -> usize {
    match (x, z) {
        (true, false) => 0,
        (true, true) => 1,
        _ => 2,
    }
}

impl Planes {
    fn new<const W: usize, C: Coefficient>(run: &[Entry<W, C>]) -> Self {
        let words = run.len().div_ceil(64);
        let mut rows = vec![0u64; 3 * 64 * W * words];
        for (j, s) in run.iter().enumerate() {
            for w in 0..W {
                let mut support = s.x[w] | s.z[w];
                while support != 0 {
                    let bit = support.trailing_zeros() as usize;
                    support &= support - 1;
                    let p = pauli_index(s.x[w] >> bit & 1 == 1, s.z[w] >> bit & 1 == 1);
                    let q = w * 64 + bit;
                    rows[(3 * q + p) * words + j / 64] |= 1 << (j % 64);
                }
            }
        }
        Self { words, rows }
    }

    /// Overwrite `(low, high)` with the bits of `k mod 4` for the product of every member with the key `(x, z)`, the member on the left when `member_left`.
    #[inline(always)]
    fn phases<const W: usize>(
        &self,
        x: &[u64; W],
        z: &[u64; W],
        member_left: bool,
        low: &mut [u64],
        high: &mut [u64],
    ) {
        low.fill(0);
        high.fill(0);
        for w in 0..W {
            let mut support = x[w] | z[w];
            while support != 0 {
                let bit = support.trailing_zeros() as usize;
                support &= support - 1;
                let p = pauli_index(x[w] >> bit & 1 == 1, z[w] >> bit & 1 == 1);
                let q = w * 64 + bit;
                // With the member on the left, a member Pauli preceding the key's gives +1 and one following it −1.
                let (before, after) = ((p + 2) % 3, (p + 1) % 3);
                let (up, down) = if member_left { (before, after) } else { (after, before) };
                let up = &self.rows[(3 * q + up) * self.words..][..self.words];
                let down = &self.rows[(3 * q + down) * self.words..][..self.words];
                for i in 0..self.words {
                    high[i] ^= low[i] & up[i];
                    low[i] ^= up[i];
                    high[i] ^= !low[i] & down[i];
                    low[i] ^= down[i];
                }
            }
        }
    }
}

/// A generator's terms sorted by their bucket under the output hash, with the run boundaries.
struct Groups<const W: usize, C: Coefficient> {
    terms: Vec<Entry<W, C>>,
    /// The bucket of each run, ascending.
    buckets: Vec<u32>,
    /// Run `g` is `terms[starts[g]..starts[g + 1]]`.
    starts: Vec<usize>,
    /// Run `g`'s planes, for a run of at least [`SLICE_MIN_RUN`] members.
    planes: Vec<Option<Planes>>,
}

impl<const W: usize, C: Coefficient> Groups<W, C> {
    fn new(terms: impl Iterator<Item = Entry<W, C>>, hash: &Gf2Hash<W>, pairing: Pairing) -> Self {
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
        let terms: Vec<Entry<W, C>> = keyed.into_iter().map(|(_, e)| e).collect();
        let planes = starts
            .windows(2)
            .map(|run| {
                let run = &terms[run[0]..run[1]];
                (pairing != Pairing::Product && run.len() >= SLICE_MIN_RUN)
                    .then(|| Planes::new(run))
            })
            .collect();
        Self { terms, buckets, starts, planes }
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
        Self::with_index(generator, operand, pairing, index, generator_left)
    }

    /// [`Job::new`] with `operand`'s support index already built, which must be `Some` exactly for a Liouvillian.
    pub(crate) fn with_index(
        generator: impl Iterator<Item = Entry<W, C>>,
        operand: &'a PauliSum<W, C>,
        pairing: Pairing,
        index: Option<SupportIndex>,
        generator_left: bool,
    ) -> Self {
        Self {
            groups: Groups::new(generator, &operand.hash, pairing),
            operand,
            index,
            generator_left,
        }
    }

    /// Every pair of this job landing in output bucket `out`, accumulated into `table`.
    #[inline(always)]
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
            let run = &self.groups.terms[self.groups.starts[g]..self.groups.starts[g + 1]];
            if let Some(planes) = &self.groups.planes[g] {
                // The micro-kernel: each operand term against the whole run at once, every member's phase counted in two bit planes.
                candidates.resize(2 * planes.words, 0);
                let (low, high) = candidates.split_at_mut(planes.words);
                let tail = match run.len() % 64 {
                    0 => u64::MAX,
                    r => (1u64 << r) - 1,
                };
                for i in 0..input.len() {
                    planes.phases(&input.x[i], &input.z[i], self.generator_left, low, high);
                    let l = Entry { x: input.x[i], z: input.z[i], c: input.coeff[i] };
                    for w in 0..planes.words {
                        let valid = if w + 1 == planes.words { tail } else { u64::MAX };
                        let mut bits = valid
                            & match pairing {
                                Pairing::Product => u64::MAX,
                                Pairing::Liouvillian => low[w],
                                Pairing::Anticommutator => !low[w],
                            };
                        while bits != 0 {
                            let j = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            let k = ((low[w] >> j & 1) | (high[w] >> j & 1) << 1) as u8;
                            let s = &run[w * 64 + j];
                            let t = if self.generator_left {
                                pairing.term_with_phase(s, &l, k)
                            } else {
                                pairing.term_with_phase(&l, s, k)
                            };
                            table.accumulate(t.x, t.z, t.c);
                        }
                    }
                }
                continue;
            }
            for s in run {
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

/// The table's nonzero terms that `keep` accepts, as sorted bucket columns.
fn sorted_columns<const W: usize, C: Coefficient>(
    table: &Table<W, C>,
    keep: &impl Fn(&[u64; W], &[u64; W], C) -> bool,
) -> BucketColumns<W, C> {
    let mut entries: Vec<&Entry<W, C>> =
        table.entries().iter().filter(|e| e.c.is_nonzero() && keep(&e.x, &e.z, e.c)).collect();
    entries.sort_unstable_by(|a, b| (&a.x, &a.z).cmp(&(&b.x, &b.z)));
    let mut columns = BucketColumns::with_capacity(entries.len());
    for e in entries {
        columns.push(e.x, e.z, e.c);
    }
    columns
}

/// Every pair of every job landing in output bucket `out`, accumulated into `table`.
#[inline(always)]
fn output_bucket<const W: usize, C: Coefficient>(
    jobs: &[Job<'_, W, C>],
    out: usize,
    pairing: Pairing,
    table: &mut Table<W, C>,
    candidates: &mut Vec<u64>,
) {
    for job in jobs {
        job.accumulate_into(out, pairing, table, candidates);
    }
}

/// [`output_bucket`] compiled with the `POPCNT` instruction, which the baseline target lacks and the commutation tests and phases count with.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
fn output_bucket_popcnt<const W: usize, C: Coefficient>(
    jobs: &[Job<'_, W, C>],
    out: usize,
    pairing: Pairing,
    table: &mut Table<W, C>,
    candidates: &mut Vec<u64>,
) {
    output_bucket(jobs, out, pairing, table, candidates);
}

/// Every pair of every job, summed per key into one sum partitioned as the operands are, keeping the summed terms `keep` accepts.
///
/// # Panics
///
/// If the operands disagree about their hash rows, bucket count or `num_qubits`.
pub(crate) fn accumulate<const W: usize, C: Coefficient>(
    jobs: &[Job<'_, W, C>],
    pairing: Pairing,
    keep: &(impl Fn(&[u64; W], &[u64; W], C) -> bool + Sync),
) -> PauliSum<W, C> {
    let first = jobs.first().expect("accumulate: no jobs").operand;
    for job in jobs {
        assert!(
            job.operand.hash.same_rows_as(&first.hash)
                && job.operand.hash.bits() == first.hash.bits(),
            "accumulate: operands partitioned differently",
        );
        assert_eq!(job.operand.num_qubits, first.num_qubits, "accumulate: num_qubits mismatch");
    }
    #[cfg(target_arch = "x86_64")]
    let popcnt = std::is_x86_feature_detected!("popcnt");
    let buckets: Vec<BucketColumns<W, C>> = (0..first.buckets.len())
        .into_par_iter()
        .map_init(
            || (Table::<W, C>::default(), Vec::new()),
            |(table, candidates), out| {
                table.clear();
                #[cfg(target_arch = "x86_64")]
                if popcnt {
                    // SAFETY: the CPU reports POPCNT.
                    unsafe { output_bucket_popcnt(jobs, out, pairing, table, candidates) };
                    return sorted_columns(table, keep);
                }
                output_bucket(jobs, out, pairing, table, candidates);
                sorted_columns(table, keep)
            },
        )
        .collect();
    let len = buckets.iter().map(BucketColumns::len).sum();
    PauliSum { buckets, hash: first.hash.clone(), num_qubits: first.num_qubits, len }
}

impl<const W: usize, C: Coefficient> PauliSum<W, C> {
    /// The operator product `pairing` names, keeping the summed terms `keep` accepts.
    pub(crate) fn pair(
        &self,
        other: &Self,
        pairing: Pairing,
        keep: &(impl Fn(&[u64; W], &[u64; W], C) -> bool + Sync),
    ) -> Self {
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

        let generator = small.iter().map(|(x, z, c)| Entry { x: *x, z: *z, c });
        let job = Job::new(generator, large, pairing, !self_is_large);
        let mut out = accumulate(&[job], pairing, keep);
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
        self.pair(other, Pairing::Product, &|_, _, _| true)
    }

    /// The Liouvillian `i[self, other] = i(self · other − other · self)`, Hermitian whenever both operands are, so a real coefficient type stays exact.
    ///
    /// Only anticommuting string pairs contribute; terms summing to exactly zero are dropped, and the result is partitioned as [`Self::product`]'s.
    ///
    /// # Panics
    ///
    /// If the two sums disagree about `num_qubits`.
    pub fn liouvillian(&self, other: &Self) -> Self {
        self.pair(other, Pairing::Liouvillian, &|_, _, _| true)
    }

    /// The anticommutator `{self, other} = self · other + other · self`, Hermitian whenever both operands are.
    ///
    /// Only commuting string pairs contribute; terms summing to exactly zero are dropped, and the result is partitioned as [`Self::product`]'s.
    ///
    /// # Panics
    ///
    /// If the two sums disagree about `num_qubits`.
    pub fn anticommutator(&self, other: &Self) -> Self {
        self.pair(other, Pairing::Anticommutator, &|_, _, _| true)
    }
}

pub(crate) mod support;
#[cfg(test)]
mod tests;
