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
//!
//! # Why balance has to reach into stage 1
//!
//! *Which* cut is a free choice the constraint count cannot see. On an open
//! 64-qubit chain every cut leaves exactly one bond remote, so the greedy is
//! indifferent between the 63 of them and its tie order picks the one at the
//! end of the chain — a `{63}`-versus-rest split under which a sum grown from a
//! `Z` in the middle never leaves partition 0. The remote *count* is right and
//! the run is serial. So the greedy is run several times under different tie
//! orders ([`SelectOptions`]) and the candidates are scored on a probe: the
//! same optimum on paper, chosen by the split it actually produces.

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

/// Tie orders [`select_rows`] scores by default: the greedy's own, then the
/// explicit rejections, then seeded permutations up to this many runs in all.
pub const DEFAULT_RESTARTS: usize = 16;

/// How many "reject this constraint instead" variants [`select_rows`]
/// enumerates by default, on top of the [`DEFAULT_RESTARTS`] permutations.
pub const DEFAULT_ENUMERATE_REJECTIONS: usize = 64;

/// How finely [`select_rows_with`] compares balance: eighths of the ideal
/// share `1 / 2^bits`.
///
/// Balance outranks remote weight in the candidate ranking, so the comparison
/// has to be coarse or the search would buy a percent of load with a whole
/// extra remote layer. Bands of an eighth mean it pays only for a
/// *qualitatively* better split. Measured on the 127-qubit heavy-hex step: the
/// two best row sets differ by 3% of the ideal share and by one remote
/// generator, land in the same band, and the cheaper one wins.
const BALANCE_BANDS: usize = 8;

/// Seed the default [`SelectOptions`] permutes tie groups with.
const DEFAULT_SELECT_SEED: u64 = 0x524F_575F_5449_4553; // "ROW_TIES"

/// How hard [`select_rows_with`] looks for a *balanced* row set.
///
/// Every field only controls how many tie orders the greedy is run under; the
/// answer is deterministic given them, and `restarts: 1` with
/// `enumerate_rejections: 0` is the plain single-run greedy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectOptions {
    /// Total number of tie orders to score, the greedy's own included.
    ///
    /// Run 0 is always today's order — weight, then support size, then mask
    /// bytes — so the plain greedy's answer is always among the candidates.
    /// Runs `1..restarts` shuffle the generators *within* each
    /// `(weight, support size)` tie group by a [`Self::seed`]ed key, which is
    /// the only freedom that does not destroy the property that makes cuts fall
    /// out of chain-like circuits (single-qubit generators first).
    pub restarts: usize,
    /// Cap on the explicit enumeration of single rejections.
    ///
    /// Each variant moves one generator the plain run *accepted* to the end of
    /// the order, so the greedy reaches it last and — when the budget is tight,
    /// which is the interesting case — rejects it instead. On a graph that is
    /// "cut this edge rather than that one": the 63 cuts of a 64-qubit chain
    /// are 63 variants, each scored by a pass over the probe sample. The
    /// generators are taken from the end of the plain order first (the
    /// least-preferred, i.e. the two-local ones), so a cap smaller than the
    /// accepted count still covers the bonds. `0` disables it.
    pub enumerate_rejections: usize,
    /// Seed for the per-restart tie keys. Fixed input plus fixed seed gives a
    /// fixed answer.
    pub seed: u64,
}

impl Default for SelectOptions {
    fn default() -> Self {
        Self {
            restarts: DEFAULT_RESTARTS,
            enumerate_rejections: DEFAULT_ENUMERATE_REJECTIONS,
            seed: DEFAULT_SELECT_SEED,
        }
    }
}

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
/// possible without making any row a conserved quantity, and split `probe`
/// evenly.
///
/// [`SelectOptions::default`] applied to [`select_rows_with`], which is the
/// entry point to read.
pub fn select_rows<const W: usize>(
    gens: &[GeneratorWeight<W>],
    num_qubits: usize,
    bits: u8,
    probe: Option<&PauliSum<W>>,
) -> RowSelection<W> {
    select_rows_with(gens, num_qubits, bits, probe, &SelectOptions::default())
}

/// [`select_rows`] with the search effort spelled out.
///
/// # The algorithm
///
/// A two-stage greedy, run once per tie order and scored on the probe. Every
/// step is deterministic given `gens`, `probe` and `opts`.
///
/// 1. **Accept constraints.** Generators are ordered by weight descending,
///    then by mask support size ascending, then by the tie order this run
///    carries (run 0: the mask words lexicographically). A generator is
///    *accepted* — its constraint `⟨r, m⟩ = 0` imposed on every row — when the
///    accepted set's GF(2) rank stays at most `2n − bits`, so the solution
///    space keeps dimension at least `bits`, **and** the accepted span stays
///    strictly inside the span of all generators, so some generator is left to
///    break conservation. A generator already implied by the accepted set is
///    free and is always taken.
///
///    Ordering support-ascending within a weight tie is what makes cuts fall
///    out of chain- and heavy-hex-shaped circuits: the single-qubit generators
///    go in first, and satisfying every `X` rotation is exactly the statement
///    that the rows have no x-bits, which leaves the whole z-half for the bond
///    constraints to cut up. Only the *last* key varies between runs, so every
///    candidate keeps that shape.
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
/// 3. **Score the run and keep the best.** Candidates are compared by, in
///    order: every row non-conserved (required); every partition **populated**
///    on the probe — rows that empty one are conserved in practice whatever the
///    generators say, and rank below every candidate that fills them all;
///    the least-loaded partition's share of the probe, in eighths of the ideal
///    (maximize — coarse on purpose, so a percent of load never costs a remote
///    layer); total remote weight (minimize); total row weight (minimize); and
///    the run index, so run 0 — the plain greedy — wins every remaining tie.
///
///    Which tie orders are tried is [`SelectOptions`]: run 0 is the plain
///    greedy, then one variant per generator moved to the end of the order
///    (`enumerate_rejections`), then seeded shuffles of each tie group up to
///    `restarts` runs in all. **Without a probe there is nothing to choose on,
///    so only run 0 is scored** and the result is exactly the single-run
///    greedy's.
///
/// # What it guarantees
///
/// Stage 1 is the standard matroid greedy on a weighted set of GF(2) vectors,
/// so it is **exact whenever the constraint set it must choose from is
/// consistent** — every subset of GF(2) vectors is realizable as a homogeneous
/// system, so the only obstruction is the rank budget, and the greedy on a
/// linear matroid maximizes weight over independent sets exactly. That covers
/// the graph-cut case (1- and 2-local generators, `bits = 1`) outright, and
/// because the matroid greedy's *weight* is the same for every tie order, the
/// restarts trade nothing away there: they choose among optima. What this is a
/// *heuristic* for is the interaction with the two side conditions: the
/// conserved-row veto is applied greedily in weight order rather than searched
/// over, and balance is searched only over the tie orders enumerated here, not
/// over the constraint set. Weighted MAX-XOR-SAT is NP-hard in general; this
/// does not solve it.
///
/// # Cost
///
/// `restarts + enumerate_rejections` runs of a greedy that is `O(g · n)` word
/// operations on `g` generators over `n` qubits, plus one pass over the probe
/// sample per candidate row. Setup-time work, once per partitioned run, but it
/// is not free: the default options are ~80 runs, which measures (release, one
/// core) at 7 ms on a 64-qubit TFIM step and 14–31 ms on the 127-qubit
/// heavy-hex step, against 0.4–1.1 ms for the single run.
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
pub fn select_rows_with<const W: usize>(
    gens: &[GeneratorWeight<W>],
    num_qubits: usize,
    bits: u8,
    probe: Option<&PauliSum<W>>,
    opts: &SelectOptions,
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

    let greedy = Greedy {
        gens,
        masks: &masks,
        samples: &samples,
        cols,
        bits,
        full_rank: all.rank(),
    };

    // Run 0 is the plain greedy, and its rejects are the answer whenever there
    // is no probe to prefer anything else on.
    let base = base_order(gens);
    let (conserved_rejected, mut best) = greedy.run(&base);
    if !samples.is_empty() {
        // The explicit rejections: one run per generator the plain run
        // accepted, moved to the end of the order so the greedy meets it last.
        // Least-preferred first — on a graph those are the edges.
        let accepted = best
            .as_ref()
            .map(|c| c.accepted.clone())
            .unwrap_or_default();
        for &g in accepted.iter().rev().take(opts.enumerate_rejections) {
            let (_, cand) = greedy.run(&order_rejecting(&base, g));
            keep_best(&mut best, cand);
        }
        for restart in 1..opts.restarts {
            let (_, cand) = greedy.run(&permuted_order(gens, opts.seed, restart));
            keep_best(&mut best, cand);
        }
    }

    let Some(best) = best else {
        // Every solution of the accepted system is conserved — stage 1 could
        // not leave a generator unsatisfied. Nothing to choose; fall back.
        return classify(
            PartitionRows::from_seed(num_qubits, bits, FALLBACK_ROW_SEED),
            gens,
            conserved_rejected,
            &samples,
        );
    };
    classify(
        PartitionRows::from_rows(num_qubits, best.rows_x, best.rows_z),
        gens,
        best.conserved_rejected,
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
// the greedy, one run per tie order
// ---------------------------------------------------------------------------

/// One tie order's answer, with everything [`keep_best`] ranks it by.
struct Candidate<const W: usize> {
    /// X-half of the chosen rows.
    rows_x: Vec<[u64; W]>,
    /// Z-half of the chosen rows.
    rows_z: Vec<[u64; W]>,
    /// Indices into `gens` of the constraints stage 1 accepted, in acceptance
    /// order. The tail of this list is what the rejection enumeration varies.
    accepted: Vec<usize>,
    /// Generators rejected because satisfying them would conserve every row.
    conserved_rejected: usize,
    /// Every row breaks conservation. Stage 2 only ever picks such rows, so
    /// this is a check, not a knob.
    non_conserved: bool,
    /// Every one of the `2^bits` partitions holds at least one probe sample.
    ///
    /// The practical form of [`Self::non_conserved`]: rows that leave a
    /// partition empty on the probe freeze it for the whole run whatever the
    /// generators say, so such a candidate ranks below every one that fills
    /// them all. `false` with no probe, where every candidate ties anyway.
    populated: bool,
    /// The least-loaded partition's share of the probe, in
    /// [`BALANCE_BANDS`]ths of the ideal `1 / 2^bits` — `BALANCE_BANDS` is a
    /// perfect split. `0` with no probe.
    balance_band: usize,
    /// Weight of the generators these rows leave remote.
    remote_weight: f64,
    /// Total set bits over all rows.
    row_weight: u32,
}

/// The parts of a [`select_rows_with`] problem that do not change between runs.
struct Greedy<'a, const W: usize> {
    gens: &'a [GeneratorWeight<W>],
    /// The generators' non-zero masks, for the conservation test.
    masks: &'a [Mask<W>],
    /// The bounded probe sample, empty when there is no probe.
    samples: &'a [Mask<W>],
    cols: usize,
    bits: u8,
    /// Rank of the span of every generator: the ceiling stage 1 stops below.
    full_rank: usize,
}

impl<const W: usize> Greedy<'_, W> {
    /// The two stages under one tie order.
    ///
    /// The candidate is `None` only when every solution of the accepted system
    /// is conserved; the `usize` is the conserved-rejection count either way,
    /// since the caller reports it even then.
    fn run(&self, order: &[usize]) -> (usize, Option<Candidate<W>>) {
        let (accepted_span, accepted, conserved_rejected) = self.accept(order);
        let mut cand = self.pick_rows(&accepted_span, accepted);
        if let Some(c) = cand.as_mut() {
            c.conserved_rejected = conserved_rejected;
        }
        (conserved_rejected, cand)
    }

    /// Stage 1: the weight-greedy over the constraints, under `order`.
    fn accept(&self, order: &[usize]) -> (Rref<W>, Vec<usize>, usize) {
        let rank_cap = self.cols.saturating_sub(self.bits as usize);
        let mut span = Rref::<W>::new(self.cols);
        let mut accepted = Vec::new();
        let mut conserved_rejected = 0usize;
        for &i in order {
            let m = (self.gens[i].mask_x, self.gens[i].mask_z);
            if mask_is_zero(&m) || span.is_implied(&m) {
                continue; // free: constrains nothing new.
            }
            if span.rank() + 1 >= self.full_rank {
                // Would span every generator: all solutions conserved.
                conserved_rejected += 1;
                continue;
            }
            if span.rank() + 1 > rank_cap {
                continue; // no room left in the solution space.
            }
            span.insert(&m);
            accepted.push(i);
        }
        (span, accepted, conserved_rejected)
    }

    /// Stage 2: `bits` independent, non-conserved rows out of the solution
    /// space, preferring the balanced ones.
    fn pick_rows(&self, accepted_span: &Rref<W>, accepted: Vec<usize>) -> Option<Candidate<W>> {
        let basis = accepted_span.nullspace_basis();
        debug_assert!(basis.len() >= self.bits as usize);
        let conserved = |r: &Mask<W>| self.masks.iter().all(|m| dot(r, m) == 0);

        // A non-conserved basis vector exists because stage 1 kept the accepted
        // rank strictly below the generators' rank, so `null(accepted)` is
        // strictly larger than the conserved subspace `null(all generators)`.
        let pivot = basis.iter().find(|b| !conserved(b)).copied()?;

        // Candidates: the non-conserved basis vectors, plus each conserved one
        // pushed off the conserved subspace by the pivot. Every basis vector is
        // therefore represented by a non-conserved candidate that spans the
        // same direction modulo the pivot, which is what keeps an independent
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

        let mut labels: Vec<u32> = vec![0; self.samples.len()];
        let mut chosen = Rref::<W>::new(self.cols);
        let mut rows_x: Vec<[u64; W]> = Vec::with_capacity(self.bits as usize);
        let mut rows_z: Vec<[u64; W]> = Vec::with_capacity(self.bits as usize);

        for k in 0..self.bits as usize {
            // The pool is already in (support ascending, mask ascending) order,
            // so keeping the *first* candidate of maximal score applies the two
            // tiebreaks for free.
            let mut best: Option<(usize, Mask<W>)> = None;
            for c in &pool {
                if chosen.is_implied(c) {
                    continue; // dependent on the rows already taken.
                }
                let score = balance_score(c, self.samples, &labels, k);
                if best.as_ref().is_none_or(|(bs, _)| score > *bs) {
                    best = Some((score, *c));
                }
            }
            let (_, row) = best.expect(
                "select_rows: the candidate pool always holds a non-conserved row \
                 independent of the rows chosen so far",
            );
            for (s, lab) in self.samples.iter().zip(labels.iter_mut()) {
                *lab |= dot(&row, s) << k;
            }
            chosen.insert(&row);
            rows_x.push(row.0);
            rows_z.push(row.1);
        }

        // The whole row set's verdict on the probe and on the generators.
        let parts = 1usize << self.bits;
        let min_count = if self.samples.is_empty() {
            0
        } else {
            let mut counts = vec![0usize; parts];
            for lab in &labels {
                counts[*lab as usize] += 1;
            }
            counts.into_iter().min().unwrap_or(0)
        };
        // `min_count / len` in units of `(1 / parts) / BALANCE_BANDS`, which is
        // at most `BALANCE_BANDS` since the least-loaded partition cannot hold
        // more than its share.
        let balance_band = if self.samples.is_empty() {
            0
        } else {
            min_count * BALANCE_BANDS * parts / self.samples.len()
        };
        let rows: Vec<Mask<W>> = rows_x.iter().zip(&rows_z).map(|(x, z)| (*x, *z)).collect();
        let remote_weight = self
            .gens
            .iter()
            .filter(|g| {
                let m = (g.mask_x, g.mask_z);
                rows.iter().any(|r| dot(r, &m) == 1)
            })
            .map(|g| g.weight)
            .sum();

        Some(Candidate {
            non_conserved: rows.iter().all(|r| !conserved(r)),
            row_weight: rows.iter().map(mask_weight).sum(),
            rows_x,
            rows_z,
            accepted,
            conserved_rejected: 0, // filled in by `run`
            populated: !self.samples.is_empty() && min_count > 0,
            balance_band,
            remote_weight,
        })
    }
}

/// Replace `best` with `cand` if `cand` is strictly better.
///
/// The comparison, in order: every row non-conserved; every partition
/// populated on the probe; the least-loaded partition's share, in
/// [`BALANCE_BANDS`]ths, descending; remote weight ascending; row weight
/// ascending. Anything still tied keeps the incumbent, which is why run 0's
/// answer — the plain greedy's — survives unless something beats it outright.
fn keep_best<const W: usize>(best: &mut Option<Candidate<W>>, cand: Option<Candidate<W>>) {
    let Some(cand) = cand else { return };
    if best
        .as_ref()
        .is_none_or(|cur| cand.rank_key() > cur.rank_key())
    {
        *best = Some(cand);
    }
}

/// Today's tie order: weight descending, support size ascending, mask words
/// ascending.
fn base_order<const W: usize>(gens: &[GeneratorWeight<W>]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..gens.len()).collect();
    order.sort_by(|&a, &b| {
        let (ga, gb) = (&gens[a], &gens[b]);
        tie_class(ga)
            .cmp(&tie_class(gb))
            .then((ga.mask_x, ga.mask_z).cmp(&(gb.mask_x, gb.mask_z)))
    });
    order
}

/// [`base_order`] with each `(weight, support size)` tie group shuffled by a
/// seeded key.
///
/// The two leading keys are what makes a cut fall out of a chain-like circuit
/// at all (see [`select_rows_with`]), so only the order *within* a tie group is
/// free — and on a chain that is exactly the choice of which bond to reject.
fn permuted_order<const W: usize>(
    gens: &[GeneratorWeight<W>],
    seed: u64,
    restart: usize,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..gens.len()).collect();
    order.sort_by(|&a, &b| {
        let (ga, gb) = (&gens[a], &gens[b]);
        tie_class(ga)
            .cmp(&tie_class(gb))
            .then(tie_key(seed, restart, a).cmp(&tie_key(seed, restart, b)))
            .then((ga.mask_x, ga.mask_z).cmp(&(gb.mask_x, gb.mask_z)))
    });
    order
}

/// `base` with generator `last` moved to the end, so the greedy reaches it only
/// after every other constraint — on a graph, "cut this edge".
fn order_rejecting(base: &[usize], last: usize) -> Vec<usize> {
    let mut order: Vec<usize> = base.iter().copied().filter(|&i| i != last).collect();
    order.push(last);
    order
}

/// The part of the greedy's order no restart may permute: heavier first, then
/// smaller support. `f64` weight is compared by its bits, descending, which is
/// a total order agreeing with `total_cmp` on the non-negative weights a
/// generator list carries.
fn tie_class<const W: usize>(g: &GeneratorWeight<W>) -> (std::cmp::Reverse<u64>, u32) {
    (
        std::cmp::Reverse(g.weight.to_bits()),
        mask_weight(&(g.mask_x, g.mask_z)),
    )
}

impl<const W: usize> Candidate<W> {
    /// The whole ranking as one tuple, greater is better.
    ///
    /// Generator weights are sums of non-negative layer counts, so their bit
    /// pattern is monotone in the value and `Reverse` turns the two
    /// "smaller is better" keys around without a float comparison.
    fn rank_key(
        &self,
    ) -> (
        bool,
        bool,
        usize,
        std::cmp::Reverse<u64>,
        std::cmp::Reverse<u32>,
    ) {
        (
            self.non_conserved,
            self.populated,
            self.balance_band,
            std::cmp::Reverse(self.remote_weight.to_bits()),
            std::cmp::Reverse(self.row_weight),
        )
    }
}

/// SplitMix64's finalizer: a cheap, deterministic bit mixer.
fn mix64(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The per-restart sort key of generator `idx`.
fn tie_key(seed: u64, restart: usize, idx: usize) -> u64 {
    mix64(seed ^ mix64((restart as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ idx as u64))
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

    /// A probe shaped like a single-site `Z` observable propagated a few steps
    /// on the chain: every term's z-support is an **odd**-size subset of
    /// `lo..hi`, with arbitrary x-bits there.
    ///
    /// The odd parity is not decoration, it is the reachability law that makes
    /// this case hard. A `ZZ` bond flips two z-bits and an `X` rotation flips
    /// none, so the total z-weight parity of a term is conserved by every
    /// generator of the chain; a sum grown from one `Z` therefore has odd
    /// z-weight in *every* term. A cut that falls outside the light cone
    /// splits nothing — the whole sum sits on one side of it and stays there —
    /// which a probe of freely random terms would never reveal.
    fn light_cone_sum(n: usize, num_qubits: usize, lo: u32, hi: u32, seed: u64) -> PauliSum<1> {
        use crate::accumulator::BuildAccumulator;
        use crate::phase::Phase;
        use num_complex::Complex64;

        let mut rng = crate::test_support::Xs64::new(seed);
        let mut acc = BuildAccumulator::<1>::with_capacity(num_qubits, n);
        let draw = |rng: &mut crate::test_support::Xs64| lo + (rng.next_u64() as u32) % (hi - lo);
        for _ in 0..n {
            let mut p = PauliString::<1> { x: [0], z: [0] };
            // An odd number of z-toggles keeps the parity odd whatever collides.
            for _ in 0..(1 + 2 * (rng.next_u64() % 3)) {
                p.z[0] ^= 1u64 << draw(&mut rng);
            }
            for _ in 0..(rng.next_u64() % 3) {
                p.x[0] ^= 1u64 << draw(&mut rng);
            }
            acc.add_term(p, Phase::ONE, Complex64::new(1.0, 0.0));
        }
        acc.finalize()
    }

    /// The share of `probe`'s terms held by the least-loaded partition, over
    /// every term (not the sample), computed straight from the rows.
    fn probe_min_share_of(rows: &PartitionRows<1>, probe: &PauliSum<1>) -> f64 {
        let mut counts = vec![0usize; rows.num_partitions()];
        let mut total = 0usize;
        for (x, z, _) in probe.iter() {
            counts[rows.partition_of(x, z) as usize] += 1;
            total += 1;
        }
        counts.into_iter().min().unwrap() as f64 / total as f64
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

    #[test]
    fn a_long_chain_cuts_where_the_probe_lives() {
        const N: usize = 64;
        let c = tfim_open_chain(N);
        let gens = circuit_generators(&c, &hash(N), false);
        assert_eq!(gens.len(), N + (N - 1)); // 64 X masks, 63 bond masks

        // The observable is spread around the middle of the chain, so a cut at
        // either end is correct on the remote *count* and useless in practice:
        // every reachable term lands in one partition.
        let probe = light_cone_sum(600, N, 20, 44, 0xC0FFEE);
        let sel = select_rows(&gens, N, 1, Some(&probe));

        // Still a cut with exactly one bond remote.
        assert_eq!(sel.rows.rows().0, [[0u64]], "the row must have no x-bits");
        let support = z_support(&sel.rows, 0);
        assert!(
            is_one_side_of_a_cut(&support, N),
            "row z-support {support:?} is not one side of a single cut",
        );
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

        // ... and the cut falls where the probe lives, so both partitions carry
        // real work.
        let share = probe_min_share_of(&sel.rows, &probe);
        assert!(
            share >= 0.3,
            "cut at z-support {support:?} leaves the smaller partition {share} of the probe",
        );
        // The probe is shorter than the sample cap, so the reported share is
        // the exact one, not an estimate.
        assert!(probe.len() <= PROBE_SAMPLES);
        assert_eq!(sel.probe_min_share, Some(share));
    }

    #[test]
    fn the_single_run_greedy_is_the_one_that_cuts_at_the_end() {
        // The same case with the search turned off: this is what `select_rows`
        // did before the restarts, and why they exist. One bond remote either
        // way — the remote *count* never saw the difference.
        const N: usize = 64;
        let c = tfim_open_chain(N);
        let gens = circuit_generators(&c, &hash(N), false);
        let probe = light_cone_sum(600, N, 20, 44, 0xC0FFEE);
        let plain = SelectOptions {
            restarts: 1,
            enumerate_rejections: 0,
            ..SelectOptions::default()
        };
        let sel = select_rows_with(&gens, N, 1, Some(&probe), &plain);

        assert_eq!(sel.remote.len(), 1);
        assert_eq!(z_support(&sel.rows, 0), vec![63]);
        assert_eq!(sel.probe_min_share, Some(0.0));
    }

    #[test]
    fn without_a_probe_the_restarts_change_nothing() {
        // No probe, no balance signal: the answer must be the plain greedy's,
        // so nothing can trade remote weight away for a split it cannot see.
        for bits in 1u8..=2 {
            for n in [8usize, 12] {
                let c = tfim_open_chain(n);
                let gens = circuit_generators(&c, &hash(n), false);
                let plain = SelectOptions {
                    restarts: 1,
                    enumerate_rejections: 0,
                    ..SelectOptions::default()
                };
                let want = select_rows_with(&gens, n, bits, None, &plain);
                let got = select_rows(&gens, n, bits, None);
                assert_eq!(got.rows, want.rows, "n {n}, bits {bits}");
                assert_eq!(got.remote_weight, want.remote_weight);
                assert_eq!(got.conserved_rejected, want.conserved_rejected);
                assert_eq!(got.probe_min_share, None);
            }
        }
    }

    #[test]
    fn a_restarted_selection_never_costs_remote_weight_it_need_not() {
        // With a probe the search may pay remote weight for balance, but only
        // when it buys balance: here every candidate cut leaves exactly one
        // bond remote, so the winner must too.
        const N: usize = 64;
        let c = tfim_open_chain(N);
        let gens = circuit_generators(&c, &hash(N), false);
        let probe = light_cone_sum(600, N, 20, 44, 0xC0FFEE);
        let plain = SelectOptions {
            restarts: 1,
            enumerate_rejections: 0,
            ..SelectOptions::default()
        };
        let base = select_rows_with(&gens, N, 1, Some(&probe), &plain);
        let searched = select_rows(&gens, N, 1, Some(&probe));
        assert_eq!(searched.remote_weight, base.remote_weight);
        assert!(searched.probe_min_share > base.probe_min_share);
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

    #[test]
    fn the_heavy_hex_step_does_not_buy_balance_inside_a_band() {
        // The banded balance comparison, pinned. On the 127-qubit heavy-hex
        // kicked-Ising step the plain greedy already splits a spread probe
        // 0.477/0.523 at two crossings, so there is nothing qualitative left to
        // buy — and with an ungraded balance score the search bought 0.492
        // anyway, for a third remote generator. Inside one band it takes the
        // cheap end instead, and finds a row set with *one* remote generator at
        // the same split (0.478).
        let mut c = Circuit::<2>::new(127);
        for q in 0..127u32 {
            c.push(PauliRotation::new(PauliString::<2>::x(q), 0.3));
        }
        for (a, b) in crate::test_support::heavy_hex_127_edges() {
            c.push(zz_rotation::<2>(a, b, 0.2));
        }
        let h = Gf2Hash::<2>::new(127, 4, 0xB00C);
        let gens = circuit_generators(&c, &h, false);
        let probe = low_weight_sum::<2>(600, 127, 3, 0xC0FFEE);

        let plain = SelectOptions {
            restarts: 1,
            enumerate_rejections: 0,
            ..SelectOptions::default()
        };
        let base = select_rows_with(&gens, 127, 1, Some(&probe), &plain);
        let sel = select_rows(&gens, 127, 1, Some(&probe));

        assert_eq!(base.remote_weight, 2.0, "remote: {:?}", base.remote);
        assert!(
            sel.remote_weight <= base.remote_weight,
            "the search paid remote weight ({} vs {}) for balance inside one \
             band ({:?} vs {:?})",
            sel.remote_weight,
            base.remote_weight,
            sel.probe_min_share,
            base.probe_min_share,
        );
        assert!(sel.probe_min_share.unwrap() >= 0.4);
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

        // The searched path too: same seed, same permutations, same winner.
        let c = tfim_open_chain(64);
        let gens = circuit_generators(&c, &hash(64), false);
        let probe = light_cone_sum(600, 64, 20, 44, 0xC0FFEE);
        let a = select_rows(&gens, 64, 1, Some(&probe));
        let b = select_rows(&gens, 64, 1, Some(&probe));
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.probe_min_share, b.probe_min_share);

        // And a different seed is allowed to disagree, but not to be
        // non-deterministic.
        let opts = SelectOptions {
            seed: 0x1234_5678,
            ..SelectOptions::default()
        };
        let c1 = select_rows_with(&gens, 64, 1, Some(&probe), &opts);
        let c2 = select_rows_with(&gens, 64, 1, Some(&probe), &opts);
        assert_eq!(c1.rows, c2.rows);
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
