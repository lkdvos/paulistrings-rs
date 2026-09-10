//! Choosing the partition rows: the circuit's generator masks, a weighted
//! MAX-XOR-SAT selector over them, and the per-layer locality it produces.
//!
//! # The algebra
//!
//! A partitioning splits keys by `part(v) = R·v` over GF(2), one bit per row
//! `r = (rx, rz)` of `R` (ARCHITECTURE.md §Partitioning). A prepared channel
//! moves a term by a key delta `d`, so the layer is local for that delta iff
//!
//! ```text
//! ⟨r, d⟩ = parity(rx & dx) ^ parity(rz & dz) = 0     for every row r,
//! ```
//!
//! which is exactly [`PartitionRows::partition_of`] returning 0. Remoteness is
//! a property of the mask alone, so picking the rows is picking which of the
//! circuit's delta masks are orthogonal to them: a **weighted MAX-XOR-SAT**
//! problem over the `2n` symplectic coordinates, one homogeneous constraint
//! `⟨r, m⟩ = 0` per generator mask `m`, weighted by how much of the circuit
//! carries it.
//!
//! # The balance side condition
//!
//! Satisfying *every* constraint is the wrong answer. A row with `⟨r, m⟩ = 0`
//! for every delta of every layer is a **conserved quantity** of the dynamics:
//! `part` is linear and no reachable term ever flips that bit, so every term
//! stays in the partition the input put it in and the other half of the
//! partitions are empty for the whole run. The selector therefore maximizes
//! satisfied weight *subject to* leaving at least one generator unsatisfied per
//! row — the cost of one remote generator buys the run its mixing.
//!
//! # Geometry
//!
//! For 1- and 2-local generators the optimum is a graph cut: a z-only row makes
//! every single-qubit `X` rotation local (its mask has no z-bits), and a bond
//! `ZZ(i, j)` is remote exactly when the edge crosses the cut. That row set has
//! its own constructor, [`PartitionRows::cut`]; [`select_rows`] is what to use
//! when the geometry is not known in advance.

use std::collections::HashMap;

use crate::bucket::hash::{Gf2Hash, PartitionRows};
use crate::channel::prepared::Prepared;
use crate::circuit::Circuit;
use crate::pauli_sum::PauliSum;

use super::plan::count_remote_deltas;

/// Seed for the fallback rows [`select_rows`] returns when the circuit
/// constrains nothing (no generators at all).
///
/// Any row is as good as any other there, and a fixed seed keeps the function
/// deterministic.
const FALLBACK_ROW_SEED: u64 = 0x5041_5254_5F52_4F57; // "PART_ROW"

/// How many probe terms [`select_rows`] evaluates balance on.
///
/// The probe is read at setup time, once per candidate row, so it is bounded:
/// a longer sum is sampled by a fixed stride over its canonical order, which
/// keeps the cost independent of the term count and the choice deterministic.
const PROBE_SAMPLES: usize = 1024;

/// One key delta mask the circuit produces, with how much of the circuit
/// carries it.
///
/// The unit of the MAX-XOR-SAT problem [`select_rows`] solves: the constraint
/// is `⟨r, mask⟩ = 0` and `weight` is what satisfying it is worth.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratorWeight<const W: usize> {
    /// X-half of the key delta.
    pub mask_x: [u64; W],
    /// Z-half of the key delta.
    pub mask_z: [u64; W],
    /// Number of layers carrying this mask, as produced by
    /// [`circuit_generators`]. A caller building the list itself may weight it
    /// any way it likes — only the ordering and the sums are read.
    pub weight: f64,
    /// [`Channel::debug_name`](crate::Channel::debug_name) of the first layer
    /// that produced this mask. Diagnostics only.
    pub name: &'static str,
}

/// The rows [`select_rows`] chose, and what they cost.
#[derive(Clone, Debug)]
pub struct RowSelection<const W: usize> {
    /// The chosen rows, ready for a [`PartitionConfig`](super::PartitionConfig)
    /// or a [`PartitionedSum`](super::PartitionedSum).
    pub rows: PartitionRows<W>,
    /// Total weight of the generators that came out local (`part(mask) == 0`).
    pub local_weight: f64,
    /// Total weight of the generators that came out remote.
    pub remote_weight: f64,
    /// The generators left remote, in the order they were given.
    pub remote: Vec<GeneratorWeight<W>>,
    /// How many generators were rejected *because* satisfying them would have
    /// made every candidate row conserved.
    ///
    /// Nonzero is the normal outcome — it is the price of a mixing
    /// partitioning, and those generators appear in [`Self::remote`]. It is
    /// counted separately from the generators dropped for want of solution-space
    /// dimension, which are the ones a larger `bits` would not have saved
    /// either.
    pub conserved_rejected: usize,
    /// Share of the probe's terms held by the least-loaded partition, or
    /// `None` when no (non-empty) probe was given.
    ///
    /// The balance score the selector maximized, reported so a caller can see
    /// what it got: `1 / num_partitions` is a perfect split and `0.0` says some
    /// partition is empty on the probe — the rows are *conserved in practice*
    /// even if no single generator says so, and the run will be serial until
    /// something mixes. Measured on the same bounded sample the selector uses
    /// (1024 terms of the probe's canonical order), so a longer probe makes it
    /// an estimate.
    pub probe_min_share: Option<f64>,
}

/// Every non-identity key delta mask `circuit` produces, with the number of
/// layers carrying it.
///
/// Layers are walked in **application order** — circuit order for
/// `adjoint == false`, reverse order for `adjoint == true`, matching
/// [`Direction::Heisenberg`](crate::Direction) — and each layer is prepared
/// against `hash` exactly as `propagate` prepares it, so the masks are the
/// engine's own. A [`Prepared::Rotation`] contributes its generator mask; a
/// [`Prepared::Local`] contributes each delta entry's mask. Duplicates across
/// layers are merged and their weights added, so a Trotter circuit of `k`
/// identical steps reports each mask once with weight `k`.
///
/// The identity delta (mask 0) is dropped: `part(0) = 0` unconditionally, so it
/// constrains nothing.
///
/// The result is in first-appearance order, which makes it deterministic but
/// carries no other meaning — [`select_rows`] sorts it itself.
///
/// # Panics
///
/// Panics if any channel declines
/// [`Channel::prepare`](crate::Channel::prepare), the same condition on which
/// `propagate` panics.
pub fn circuit_generators<const W: usize>(
    circuit: &Circuit<W>,
    hash: &Gf2Hash<W>,
    adjoint: bool,
) -> Vec<GeneratorWeight<W>> {
    let n = circuit.channels.len();
    let mut out: Vec<GeneratorWeight<W>> = Vec::new();
    let mut seen: HashMap<Mask<W>, usize> = HashMap::new();

    for k in 0..n {
        let idx = if adjoint { n - 1 - k } else { k };
        let ch = &circuit.channels[idx];
        let prep = ch
            .prepare(hash, adjoint)
            .unwrap_or_else(|| panic!("circuit_generators: layer {idx} declined Channel::prepare"));
        // One layer's masks are distinct by construction (a `LocalPtm`'s
        // entries are keyed by distinct `local_delta`s), so a mask seen twice
        // here is a mask seen in two layers.
        let masks: Vec<Mask<W>> = match &prep {
            Prepared::Local(ptm) => ptm.deltas().iter().map(|d| d.mask()).collect(),
            Prepared::Rotation(r) => vec![r.gen_mask()],
        };
        for m in masks {
            if mask_is_zero(&m) {
                continue;
            }
            match seen.get(&m) {
                Some(&i) => out[i].weight += 1.0,
                None => {
                    seen.insert(m, out.len());
                    out.push(GeneratorWeight {
                        mask_x: m.0,
                        mask_z: m.1,
                        weight: 1.0,
                        name: ch.debug_name(),
                    });
                }
            }
        }
    }
    out
}

/// Choose `bits` partition rows that leave as much of `gens`' weight local as
/// possible without making any row a conserved quantity.
///
/// # The algorithm
///
/// Greedy, deterministic, in two stages.
///
/// 1. **Accept constraints.** Generators are ordered by weight descending,
///    then by mask support size ascending, then by the mask words
///    lexicographically (x-half first). A generator is *accepted* — its
///    constraint `⟨r, m⟩ = 0` imposed on every row — when the accepted set's
///    GF(2) rank stays at most `2n − bits`, so the solution space keeps
///    dimension at least `bits`, **and** the accepted span stays strictly
///    inside the span of all generators, so some generator is left to break
///    conservation. A generator already implied by the accepted set is free and
///    is always taken.
///
///    Ordering support-ascending within a weight tie is what makes cuts fall
///    out of chain- and heavy-hex-shaped circuits: the single-qubit generators
///    go in first, and satisfying every `X` rotation is exactly the statement
///    that the rows have no x-bits, which leaves the whole z-half for the bond
///    constraints to cut up.
///
/// 2. **Pick rows out of the solution space.** A GF(2) nullspace basis of the
///    accepted masks is computed, and `bits` independent rows are taken from it
///    greedily, each preferred by, in order:
///    * **not conserved** — required, never traded away;
///    * **balance on `probe`**, if given: the row maximizing the smallest
///      partition's share of the probe's terms under the rows chosen so far
///      (for the first row that is simply the bit closest to 50/50). A long
///      probe is sampled by a fixed stride, so the cost does not grow with the
///      term count;
///    * **low row weight** — few set bits, which is what a cut looks like;
///    * the mask words lexicographically, so the choice is total.
///
/// # What it guarantees
///
/// Stage 1 is the standard matroid greedy on a weighted set of GF(2) vectors,
/// so it is **exact whenever the constraint set it must choose from is
/// consistent** — every subset of GF(2) vectors is realizable as a homogeneous
/// system, so the only obstruction is the rank budget, and the greedy on a
/// linear matroid maximizes weight over independent sets exactly. That covers
/// the graph-cut case (1- and 2-local generators, `bits = 1`) outright. What it
/// is a *heuristic* for is the interaction with the two side conditions: the
/// conserved-row veto is applied greedily in weight order rather than searched
/// over, and balance is optimized only within the solution space stage 1
/// already fixed, never by trading a satisfied constraint for a better split.
/// Weighted MAX-XOR-SAT is NP-hard in general; this does not solve it.
///
/// # Edge cases
///
/// * `bits == 0` gives [`PartitionRows::none`] and reports everything local.
/// * No generators (an empty circuit, or one whose every delta is the identity)
///   leaves nothing to optimize, so the rows come from a fixed seed.
///
/// # Panics
///
/// Panics if `bits` exceeds [`P_MAX_BITS`](crate::bucket::hash::P_MAX_BITS), or
/// if `2 · num_qubits < bits` — there are not that many independent rows to be
/// had.
pub fn select_rows<const W: usize>(
    gens: &[GeneratorWeight<W>],
    num_qubits: usize,
    bits: u8,
    probe: Option<&PauliSum<W>>,
) -> RowSelection<W> {
    let samples = probe_samples(probe);
    if bits == 0 {
        return classify(PartitionRows::none(num_qubits), gens, 0, &samples);
    }
    assert!(
        2 * num_qubits >= bits as usize,
        "select_rows: {bits} rows need at least {bits} of the 2·{num_qubits} key columns",
    );

    let cols = 2 * num_qubits;
    let masks: Vec<Mask<W>> = gens
        .iter()
        .map(|g| (g.mask_x, g.mask_z))
        .filter(|m| !mask_is_zero(m))
        .collect();
    if masks.is_empty() {
        // Nothing constrains the rows; a fixed seed keeps the answer stable.
        return classify(
            PartitionRows::from_seed(num_qubits, bits, FALLBACK_ROW_SEED),
            gens,
            0,
            &samples,
        );
    }

    // The span of *all* generators. Reaching it with the accepted set would
    // make every solution conserved, so it is the ceiling stage 1 stops below.
    let mut all = Rref::<W>::new(cols);
    for m in &masks {
        all.insert(m);
    }
    let full_rank = all.rank();

    // ---- stage 1: accept constraints, weight-greedy ----

    let mut order: Vec<usize> = (0..gens.len()).collect();
    order.sort_by(|&a, &b| {
        let (ga, gb) = (&gens[a], &gens[b]);
        gb.weight
            .total_cmp(&ga.weight)
            .then(mask_weight(&(ga.mask_x, ga.mask_z)).cmp(&mask_weight(&(gb.mask_x, gb.mask_z))))
            .then((ga.mask_x, ga.mask_z).cmp(&(gb.mask_x, gb.mask_z)))
    });

    let rank_cap = cols.saturating_sub(bits as usize);
    let mut accepted = Rref::<W>::new(cols);
    let mut conserved_rejected = 0usize;
    for i in order {
        let m = (gens[i].mask_x, gens[i].mask_z);
        if mask_is_zero(&m) || accepted.is_implied(&m) {
            continue; // free: constrains nothing new.
        }
        if accepted.rank() + 1 >= full_rank {
            // Would span every generator: all solutions conserved.
            conserved_rejected += 1;
            continue;
        }
        if accepted.rank() + 1 > rank_cap {
            continue; // no room left in the solution space.
        }
        accepted.insert(&m);
    }

    // ---- stage 2: pick rows out of the solution space ----

    let basis = accepted.nullspace_basis();
    debug_assert!(basis.len() >= bits as usize);
    let conserved = |r: &Mask<W>| masks.iter().all(|m| dot(r, m) == 0);

    // A non-conserved basis vector exists because stage 1 kept the accepted
    // rank strictly below the generators' rank, so `null(accepted)` is strictly
    // larger than the conserved subspace `null(all generators)`.
    let pivot = basis.iter().find(|b| !conserved(b)).copied();
    let Some(pivot) = pivot else {
        return classify(
            PartitionRows::from_seed(num_qubits, bits, FALLBACK_ROW_SEED),
            gens,
            conserved_rejected,
            &samples,
        );
    };

    // Candidates: the non-conserved basis vectors, plus each conserved one
    // pushed off the conserved subspace by the pivot. Every basis vector is
    // therefore represented by a non-conserved candidate that spans the same
    // direction modulo the pivot, which is what keeps an independent
    // non-conserved candidate available at every step below.
    let mut pool: Vec<Mask<W>> = Vec::with_capacity(basis.len());
    for b in &basis {
        if conserved(b) {
            let mut c = *b;
            mask_xor(&mut c, &pivot);
            if !mask_is_zero(&c) && !conserved(&c) {
                pool.push(c);
            }
        } else {
            pool.push(*b);
        }
    }
    pool.sort_by(|a, b| mask_weight(a).cmp(&mask_weight(b)).then(a.cmp(b)));
    pool.dedup();

    let mut labels: Vec<u32> = vec![0; samples.len()];
    let mut chosen = Rref::<W>::new(cols);
    let mut rows_x: Vec<[u64; W]> = Vec::with_capacity(bits as usize);
    let mut rows_z: Vec<[u64; W]> = Vec::with_capacity(bits as usize);

    for k in 0..bits as usize {
        // The pool is already in (support ascending, mask ascending) order, so
        // keeping the *first* candidate of maximal score applies the two
        // tiebreaks for free.
        let mut best: Option<(usize, Mask<W>)> = None;
        for c in &pool {
            if chosen.is_implied(c) {
                continue; // dependent on the rows already taken.
            }
            let score = balance_score(c, &samples, &labels, k);
            if best.as_ref().is_none_or(|(bs, _)| score > *bs) {
                best = Some((score, *c));
            }
        }
        let (_, row) = best.expect(
            "select_rows: the candidate pool always holds a non-conserved row \
             independent of the rows chosen so far",
        );
        for (s, lab) in samples.iter().zip(labels.iter_mut()) {
            *lab |= dot(&row, s) << k;
        }
        chosen.insert(&row);
        rows_x.push(row.0);
        rows_z.push(row.1);
    }

    classify(
        PartitionRows::from_rows(num_qubits, rows_x, rows_z),
        gens,
        conserved_rejected,
        &samples,
    )
}

/// Per layer, `true` if the layer moves nothing across a partition boundary
/// under `rows`.
///
/// The boolean form of [`count_remote_deltas`]: layers in application order
/// (circuit order forward, reverse order adjoint), one flag each, `true` when
/// the layer's every delta has `part(mask) == 0` and the layer therefore runs
/// with no exchange at all.
///
/// # Panics
///
/// Panics if any channel declines
/// [`Channel::prepare`](crate::Channel::prepare).
pub fn layer_locality<const W: usize>(
    circuit: &Circuit<W>,
    rows: &PartitionRows<W>,
    hash: &Gf2Hash<W>,
    adjoint: bool,
) -> Vec<bool> {
    count_remote_deltas(circuit, hash, rows, adjoint)
        .into_iter()
        .map(|(_, remote)| remote == 0)
        .collect()
}

// ---------------------------------------------------------------------------
// GF(2) helpers over the 2n key columns
// ---------------------------------------------------------------------------

/// A vector over the `2·num_qubits` symplectic key columns, as `(x, z)` words —
/// the same shape as a key delta mask and as one [`PartitionRows`] row.
///
/// Columns are numbered `0..n` for the x-half and `n..2n` for the z-half, and
/// `⟨a, b⟩` is the ordinary GF(2) dot product in that numbering (not the
/// symplectic form): remoteness reads `parity(rx & mx) ^ parity(rz & mz)`.
type Mask<const W: usize> = ([u64; W], [u64; W]);

#[inline]
fn mask_is_zero<const W: usize>(m: &Mask<W>) -> bool {
    m.0.iter().chain(m.1.iter()).all(|w| *w == 0)
}

#[inline]
fn mask_weight<const W: usize>(m: &Mask<W>) -> u32 {
    m.0.iter().chain(m.1.iter()).map(|w| w.count_ones()).sum()
}

#[inline]
fn mask_xor<const W: usize>(a: &mut Mask<W>, b: &Mask<W>) {
    for w in 0..W {
        a.0[w] ^= b.0[w];
        a.1[w] ^= b.1[w];
    }
}

/// `⟨a, b⟩` over GF(2).
#[inline]
fn dot<const W: usize>(a: &Mask<W>, b: &Mask<W>) -> u32 {
    let mut parity = 0u32;
    for w in 0..W {
        parity ^= (a.0[w] & b.0[w]).count_ones();
        parity ^= (a.1[w] & b.1[w]).count_ones();
    }
    parity & 1
}

/// Lowest set column of `v`, x-half first, or `None` if `v` is zero.
#[inline]
fn first_column<const W: usize>(v: &Mask<W>, num_qubits: usize) -> Option<usize> {
    for (w, word) in v.0.iter().enumerate() {
        if *word != 0 {
            return Some(w * 64 + word.trailing_zeros() as usize);
        }
    }
    for (w, word) in v.1.iter().enumerate() {
        if *word != 0 {
            return Some(num_qubits + w * 64 + word.trailing_zeros() as usize);
        }
    }
    None
}

#[inline]
fn column_bit<const W: usize>(v: &Mask<W>, col: usize, num_qubits: usize) -> bool {
    let (words, q) = if col < num_qubits {
        (&v.0, col)
    } else {
        (&v.1, col - num_qubits)
    };
    (words[q / 64] >> (q % 64)) & 1 == 1
}

#[inline]
fn set_column<const W: usize>(v: &mut Mask<W>, col: usize, num_qubits: usize) {
    let (words, q) = if col < num_qubits {
        (&mut v.0, col)
    } else {
        (&mut v.1, col - num_qubits)
    };
    words[q / 64] |= 1u64 << (q % 64);
}

/// A reduced row-echelon basis over the key columns, grown one vector at a
/// time.
///
/// Kept in full RREF (each pivot column is set in exactly one basis row) so
/// that [`Self::nullspace_basis`] is one pass. Sizes here are tiny — at most
/// `2n ≤ 2048` rows of `2W` words, built once at setup.
struct Rref<const W: usize> {
    /// `(pivot column, reduced row)`, ascending by pivot column.
    pivots: Vec<(usize, Mask<W>)>,
    /// Half the column count: columns `0..num_qubits` are x, the rest z.
    num_qubits: usize,
    /// Total column count, `2 · num_qubits`.
    cols: usize,
}

impl<const W: usize> Rref<W> {
    fn new(cols: usize) -> Self {
        Self {
            pivots: Vec::new(),
            num_qubits: cols / 2,
            cols,
        }
    }

    fn rank(&self) -> usize {
        self.pivots.len()
    }

    /// `v` reduced against the current pivots.
    fn reduce(&self, v: &Mask<W>) -> Mask<W> {
        let mut cur = *v;
        for (pc, prow) in &self.pivots {
            if column_bit(&cur, *pc, self.num_qubits) {
                mask_xor(&mut cur, prow);
            }
        }
        cur
    }

    /// `true` if `v` is in the span already.
    fn is_implied(&self, v: &Mask<W>) -> bool {
        mask_is_zero(&self.reduce(v))
    }

    /// Add `v`; returns `true` if the rank grew.
    fn insert(&mut self, v: &Mask<W>) -> bool {
        let cur = self.reduce(v);
        let Some(pc) = first_column(&cur, self.num_qubits) else {
            return false;
        };
        // Re-reduce the existing rows so the basis stays in full RREF.
        for (_, prow) in self.pivots.iter_mut() {
            if column_bit(prow, pc, self.num_qubits) {
                mask_xor(prow, &cur);
            }
        }
        self.pivots.push((pc, cur));
        self.pivots.sort_by_key(|(pc, _)| *pc);
        true
    }

    /// A basis of `{ r : ⟨r, m⟩ = 0 for every inserted m }`, one vector per
    /// free column — `cols − rank()` of them.
    fn nullspace_basis(&self) -> Vec<Mask<W>> {
        let mut is_pivot = vec![false; self.cols];
        for (pc, _) in &self.pivots {
            is_pivot[*pc] = true;
        }
        let mut out = Vec::with_capacity(self.cols - self.pivots.len());
        for (free, pivot) in is_pivot.iter().enumerate() {
            if *pivot {
                continue;
            }
            let mut v: Mask<W> = ([0u64; W], [0u64; W]);
            set_column(&mut v, free, self.num_qubits);
            for (pc, prow) in &self.pivots {
                if column_bit(prow, free, self.num_qubits) {
                    set_column(&mut v, *pc, self.num_qubits);
                }
            }
            out.push(v);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// selection helpers
// ---------------------------------------------------------------------------

/// Up to [`PROBE_SAMPLES`] keys of `probe`, taken by a fixed stride over its
/// canonical order so the sample is deterministic and the cost bounded.
fn probe_samples<const W: usize>(probe: Option<&PauliSum<W>>) -> Vec<Mask<W>> {
    let Some(p) = probe else {
        return Vec::new();
    };
    if p.is_empty() {
        return Vec::new();
    }
    let stride = p.len().div_ceil(PROBE_SAMPLES).max(1);
    p.iter()
        .step_by(stride)
        .take(PROBE_SAMPLES)
        .map(|(x, z, _)| (*x, *z))
        .collect()
}

/// Occupancy of the least-loaded partition if `row` became row `k`, given the
/// labels the first `k` rows already assign to the probe samples.
///
/// Maximizing it is "closest to 50/50" for `k == 0` and "balance the `2^(k+1)`
/// blocks" after that. With no probe every candidate scores 0 and the tiebreaks
/// decide.
fn balance_score<const W: usize>(
    row: &Mask<W>,
    samples: &[Mask<W>],
    labels: &[u32],
    k: usize,
) -> usize {
    if samples.is_empty() {
        return 0;
    }
    let mut counts = vec![0usize; 1usize << (k + 1)];
    for (s, lab) in samples.iter().zip(labels.iter()) {
        counts[(*lab | (dot(row, s) << k)) as usize] += 1;
    }
    counts.into_iter().min().unwrap_or(0)
}

/// Wrap `rows` with the local/remote split of `gens` under them, and with the
/// split of `samples` they produce.
fn classify<const W: usize>(
    rows: PartitionRows<W>,
    gens: &[GeneratorWeight<W>],
    conserved_rejected: usize,
    samples: &[Mask<W>],
) -> RowSelection<W> {
    let mut local_weight = 0.0;
    let mut remote_weight = 0.0;
    let mut remote = Vec::new();
    for g in gens {
        if rows.partition_of(&g.mask_x, &g.mask_z) == 0 {
            local_weight += g.weight;
        } else {
            remote_weight += g.weight;
            remote.push(g.clone());
        }
    }
    let probe_min_share = (!samples.is_empty()).then(|| {
        let mut counts = vec![0usize; rows.num_partitions()];
        for (x, z) in samples {
            counts[rows.partition_of(x, z) as usize] += 1;
        }
        counts.into_iter().min().unwrap_or(0) as f64 / samples.len() as f64
    });
    RowSelection {
        rows,
        local_weight,
        remote_weight,
        remote,
        conserved_rejected,
        probe_min_share,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::channel::clifford::Clifford1Q;
    use crate::channel::rotation::PauliRotation;
    use crate::pauli_string::PauliString;
    use crate::test_support::{low_weight_sum, zz_rotation};

    fn hash(num_qubits: usize) -> Gf2Hash<1> {
        Gf2Hash::<1>::new(num_qubits, 4, 0xB00C)
    }

    /// One TFIM Trotter step on an **open** chain: an `X` rotation on every
    /// qubit, then a `ZZ` bond rotation on every edge. The open chain has one
    /// fewer bond than qubits, which is what makes a single cut able to leave
    /// exactly one bond remote.
    fn tfim_open_chain(num_qubits: usize) -> Circuit<1> {
        let mut c = Circuit::<1>::new(num_qubits);
        for q in 0..num_qubits {
            c.push(PauliRotation::new(PauliString::<1>::x(q as u32), 0.3));
        }
        for q in 0..num_qubits - 1 {
            c.push(zz_rotation::<1>(q as u32, q as u32 + 1, 0.2));
        }
        c
    }

    /// A single heavy-hex plaquette: six data qubits with a flag qubit on every
    /// bond, i.e. a 12-cycle, with an `X` rotation on every qubit and a `ZZ` on
    /// every bond — the kicked-Ising step restricted to one plaquette.
    fn heavy_hex_plaquette() -> Circuit<1> {
        let mut c = Circuit::<1>::new(12);
        for q in 0..12u32 {
            c.push(PauliRotation::new(PauliString::<1>::x(q), 0.3));
        }
        for q in 0..12u32 {
            c.push(zz_rotation::<1>(q, (q + 1) % 12, 0.2));
        }
        c
    }

    fn z_support(rows: &PartitionRows<1>, i: usize) -> Vec<u32> {
        let (_, rz) = rows.rows();
        (0..rows.num_qubits() as u32)
            .filter(|q| (rz[i][0] >> q) & 1 == 1)
            .collect()
    }

    /// `true` if `qubits` (ascending) is a run of consecutive indices, or its
    /// complement in `0..n` is — the two ways a row can name one side of a
    /// single cut of a chain.
    fn is_one_side_of_a_cut(qubits: &[u32], n: usize) -> bool {
        let run = |v: &[u32]| !v.is_empty() && v.windows(2).all(|w| w[1] == w[0] + 1);
        let complement: Vec<u32> = (0..n as u32).filter(|q| !qubits.contains(q)).collect();
        run(qubits) || run(&complement)
    }

    // ---- circuit_generators ----

    #[test]
    fn circuit_generators_merges_duplicate_masks_and_counts_layers() {
        let mut c = Circuit::<1>::new(4);
        c.push(PauliRotation::new(PauliString::<1>::x(0), 0.3));
        c.push(zz_rotation::<1>(0, 1, 0.2));
        c.push(PauliRotation::new(PauliString::<1>::x(0), 0.7));
        let gens = circuit_generators(&c, &hash(4), false);

        // Two distinct masks: X_0 (twice) and Z_0·Z_1 (once). The identity
        // delta of every layer is dropped.
        assert_eq!(gens.len(), 2);
        assert_eq!(gens[0].mask_x, [0b0001]);
        assert_eq!(gens[0].mask_z, [0]);
        assert_eq!(gens[0].weight, 2.0);
        assert_eq!(gens[0].name, "PauliRotation");
        assert_eq!(gens[1].mask_x, [0]);
        assert_eq!(gens[1].mask_z, [0b0011]);
        assert_eq!(gens[1].weight, 1.0);
    }

    #[test]
    fn circuit_generators_walks_layers_in_application_order() {
        let mut c = Circuit::<1>::new(4);
        c.push(Clifford1Q::h(0)); // delta X_0 Z_0
        c.push(Clifford1Q::s(1)); // delta Z_1
        let forward = circuit_generators(&c, &hash(4), false);
        let adjoint = circuit_generators(&c, &hash(4), true);
        assert_eq!(forward.len(), 2);
        assert_eq!((forward[0].mask_x, forward[0].mask_z), ([0b1], [0b1]));
        assert_eq!((forward[1].mask_x, forward[1].mask_z), ([0], [0b10]));
        // Reverse order, same masks (an adjoint's delta set is the same
        // subspace).
        assert_eq!((adjoint[0].mask_x, adjoint[0].mask_z), ([0], [0b10]));
        assert_eq!((adjoint[1].mask_x, adjoint[1].mask_z), ([0b1], [0b1]));
    }

    #[test]
    fn circuit_generators_of_an_empty_circuit_is_empty() {
        assert!(circuit_generators(&Circuit::<1>::new(4), &hash(4), false).is_empty());
    }

    // ---- select_rows on a chain: the cut ----

    #[test]
    fn a_chain_step_selects_a_cut_that_leaves_one_bond_remote() {
        const N: usize = 8;
        let c = tfim_open_chain(N);
        let gens = circuit_generators(&c, &hash(N), false);
        assert_eq!(gens.len(), N + (N - 1)); // 8 X masks, 7 bond masks

        let probe = low_weight_sum::<1>(400, N, 2, 0xC0FFEE);
        let sel = select_rows(&gens, N, 1, Some(&probe));

        // A cut: one z-only row naming one contiguous side of the chain.
        assert_eq!(sel.rows.bits(), 1);
        assert_eq!(sel.rows.rows().0, [[0u64]], "the row must have no x-bits");
        let support = z_support(&sel.rows, 0);
        assert!(
            is_one_side_of_a_cut(&support, N),
            "row z-support {support:?} is not one side of a single cut",
        );

        // Every transverse-field generator is local; exactly one bond crosses.
        assert_eq!(sel.remote.len(), 1, "remote: {:?}", sel.remote);
        assert_eq!(sel.remote[0].mask_x, [0], "the remote generator is a bond");
        assert_eq!(
            sel.remote[0]
                .mask_z
                .iter()
                .map(|w| w.count_ones())
                .sum::<u32>(),
            2
        );
        for g in &gens {
            let want_local = g.mask_z == [0] || g != &sel.remote[0];
            assert_eq!(
                sel.rows.partition_of(&g.mask_x, &g.mask_z) == 0,
                want_local,
                "{:?}",
                g,
            );
        }
        assert_eq!(sel.remote_weight, 1.0);
        assert_eq!(sel.local_weight, (gens.len() - 1) as f64);
        // The rejected bond is rejected for conservation, not for room.
        assert_eq!(sel.conserved_rejected, 1);
    }

    #[test]
    fn a_chain_step_at_two_bits_leaves_at_most_three_bonds_remote() {
        const N: usize = 8;
        let c = tfim_open_chain(N);
        let gens = circuit_generators(&c, &hash(N), false);
        let probe = low_weight_sum::<1>(400, N, 2, 0xC0FFEE);
        let sel = select_rows(&gens, N, 2, Some(&probe));

        assert_eq!(sel.rows.bits(), 2);
        assert_eq!(sel.rows.rows().0, [[0u64], [0u64]]);
        assert!(sel.remote.len() <= 3, "remote: {:?}", sel.remote);
        for g in &sel.remote {
            assert_eq!(g.mask_x, [0], "only bonds should cross a z-only cut");
        }
    }

    // ---- conserved-row rejection ----

    #[test]
    fn a_circuit_of_only_x_rotations_still_gets_a_non_conserved_row() {
        // Every mask is x-only, so *every* z-only row satisfies every
        // constraint — and is a conserved quantity: no term ever changes its
        // partition. The selector must refuse full satisfaction and leave one
        // generator remote.
        const N: usize = 8;
        let mut c = Circuit::<1>::new(N);
        for q in 0..N as u32 {
            c.push(PauliRotation::new(PauliString::<1>::x(q), 0.3));
        }
        let gens = circuit_generators(&c, &hash(N), false);
        assert_eq!(gens.len(), N);

        for bits in 1u8..=2 {
            let sel = select_rows(&gens, N, bits, None);
            assert!(
                !sel.remote.is_empty(),
                "bits {bits}: every row is conserved",
            );
            assert!(sel.conserved_rejected >= 1, "bits {bits}");
            // Each row on its own must break conservation, or its partition
            // bit is frozen for the whole run.
            let (rx, rz) = sel.rows.rows();
            for i in 0..bits as usize {
                let row = (rx[i], rz[i]);
                assert!(
                    gens.iter().any(|g| dot(&row, &(g.mask_x, g.mask_z)) == 1),
                    "bits {bits}: row {i} is conserved",
                );
            }
        }
    }

    // ---- heavy-hex toy ----

    #[test]
    fn a_heavy_hex_plaquette_beats_random_rows() {
        let c = heavy_hex_plaquette();
        let gens = circuit_generators(&c, &hash(12), false);
        assert_eq!(gens.len(), 24);
        let probe = low_weight_sum::<1>(400, 12, 2, 0x5EED);
        let sel = select_rows(&gens, 12, 1, Some(&probe));

        // A ring needs two crossings, and that is what the selector pays.
        assert_eq!(sel.remote.len(), 2, "remote: {:?}", sel.remote);
        for seed in 0..5u64 {
            let random = PartitionRows::<1>::from_seed(12, 1, 0xA11CE + seed);
            let rw: f64 = gens
                .iter()
                .filter(|g| random.partition_of(&g.mask_x, &g.mask_z) != 0)
                .map(|g| g.weight)
                .sum();
            assert!(
                sel.remote_weight <= rw,
                "seed {seed}: selected {} > random {rw}",
                sel.remote_weight,
            );
        }
    }

    // ---- degenerate inputs ----

    #[test]
    fn zero_bits_is_the_trivial_partitioning_and_all_local() {
        let c = tfim_open_chain(8);
        let gens = circuit_generators(&c, &hash(8), false);
        let sel = select_rows(&gens, 8, 0, None);
        assert_eq!(sel.rows, PartitionRows::<1>::none(8));
        assert!(sel.remote.is_empty());
        assert_eq!(sel.local_weight, gens.len() as f64);
        assert_eq!(sel.remote_weight, 0.0);
    }

    #[test]
    fn no_generators_falls_back_to_seeded_rows() {
        let sel = select_rows::<1>(&[], 8, 2, None);
        assert_eq!(sel.rows.bits(), 2);
        assert_eq!(
            sel.rows,
            PartitionRows::<1>::from_seed(8, 2, FALLBACK_ROW_SEED)
        );
        assert!(sel.remote.is_empty());
    }

    #[test]
    fn the_selection_reports_how_it_split_the_probe() {
        const N: usize = 8;
        let c = tfim_open_chain(N);
        let gens = circuit_generators(&c, &hash(N), false);
        let probe = low_weight_sum::<1>(400, N, 2, 0xC0FFEE);

        // Hand-checked against the rows: the same count, over every term
        // (the probe is shorter than the sample cap, so nothing is skipped).
        let sel = select_rows(&gens, N, 1, Some(&probe));
        assert!(probe.len() <= PROBE_SAMPLES);
        let mut counts = [0usize; 2];
        for (x, z, _) in probe.iter() {
            counts[sel.rows.partition_of(x, z) as usize] += 1;
        }
        let want = counts.iter().min().copied().unwrap() as f64 / probe.len() as f64;
        assert_eq!(sel.probe_min_share, Some(want));

        // No probe, nothing to report.
        assert_eq!(select_rows(&gens, N, 1, None).probe_min_share, None);
    }

    #[test]
    fn selection_is_deterministic() {
        let c = tfim_open_chain(8);
        let gens = circuit_generators(&c, &hash(8), false);
        let probe = low_weight_sum::<1>(400, 8, 2, 0xC0FFEE);
        let a = select_rows(&gens, 8, 1, Some(&probe));
        let b = select_rows(&gens, 8, 1, Some(&probe));
        assert_eq!(a.rows, b.rows);
    }

    // ---- layer_locality ----

    #[test]
    fn layer_locality_is_count_remote_deltas_as_a_predicate() {
        const N: usize = 8;
        let c = tfim_open_chain(N);
        let h = hash(N);
        let gens = circuit_generators(&c, &h, false);
        let probe = low_weight_sum::<1>(400, N, 2, 0xC0FFEE);
        let rows = select_rows(&gens, N, 1, Some(&probe)).rows;

        for adjoint in [false, true] {
            let want: Vec<bool> = count_remote_deltas(&c, &h, &rows, adjoint)
                .into_iter()
                .map(|(_, remote)| remote == 0)
                .collect();
            let got = layer_locality(&c, &rows, &h, adjoint);
            assert_eq!(got, want, "adjoint {adjoint}");
            assert_eq!(got.len(), c.channels.len());
            // Exactly one layer — the bond crossing the cut — is remote.
            assert_eq!(got.iter().filter(|b| !**b).count(), 1, "adjoint {adjoint}");
        }
    }

    #[test]
    fn layer_locality_without_rows_is_all_local() {
        let c = tfim_open_chain(6);
        let rows = PartitionRows::<1>::none(6);
        assert!(layer_locality(&c, &rows, &hash(6), false)
            .into_iter()
            .all(|b| b));
    }

    // ---- property: the reported split is the rows' own verdict ----

    mod props {
        use super::*;
        use crate::test_support::{low_weight_pauli, Xs64};
        use proptest::prelude::*;

        /// A random circuit of single-qubit `X` rotations and weight-≤2 Pauli
        /// rotations on `num_qubits` qubits.
        fn random_low_weight_circuit(num_qubits: usize, layers: usize, seed: u64) -> Circuit<1> {
            let mut rng = Xs64::new(seed);
            let mut c = Circuit::<1>::new(num_qubits);
            for _ in 0..layers {
                let weight = 1 + (rng.next_u64() % 2) as usize;
                let gen = low_weight_pauli::<1>(&mut rng, num_qubits, weight);
                if gen.x == [0] && gen.z == [0] {
                    continue; // the draw collided onto the identity
                }
                c.push(PauliRotation::new(gen, 0.31));
            }
            c
        }

        proptest! {
            #[test]
            fn reported_locality_matches_partition_of(
                seed in any::<u64>(),
                layers in 1usize..14,
                bits in 1u8..3,
            ) {
                let n = 10usize;
                let c = random_low_weight_circuit(n, layers, seed);
                let h = Gf2Hash::<1>::new(n, 4, 0xB00C);
                let gens = circuit_generators(&c, &h, false);
                let probe = low_weight_sum::<1>(200, n, 2, seed ^ 0x1234);
                let sel = select_rows(&gens, n, bits, Some(&probe));

                let mut remote_seen = 0usize;
                for g in &gens {
                    let part = sel.rows.partition_of(&g.mask_x, &g.mask_z);
                    let listed = sel.remote.iter().any(|r| {
                        (r.mask_x, r.mask_z) == (g.mask_x, g.mask_z)
                    });
                    prop_assert_eq!(part != 0, listed, "{:?}", g);
                    if listed {
                        remote_seen += 1;
                    }
                }
                prop_assert_eq!(remote_seen, sel.remote.len());
                let total: f64 = gens.iter().map(|g| g.weight).sum();
                prop_assert_eq!(sel.local_weight + sel.remote_weight, total);

                // No row is a conserved quantity unless the circuit gave the
                // selector nothing to work with.
                if !gens.is_empty() {
                    let (rx, rz) = sel.rows.rows();
                    for i in 0..bits as usize {
                        let row = (rx[i], rz[i]);
                        prop_assert!(
                            gens.iter().any(|g| dot(&row, &(g.mask_x, g.mask_z)) == 1),
                            "row {} is conserved", i,
                        );
                    }
                }
            }
        }
    }
}
