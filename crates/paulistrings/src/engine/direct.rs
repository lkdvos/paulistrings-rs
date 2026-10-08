//! The direct-apply small-sum layer path: a hash map instead of the bucketed sort-merge pipeline, for sums small enough that the pipeline's per-layer fixed cost dominates.
//!
//! This is **not** the canonical engine. [`bucketed`](super::bucketed) is, at every term count, and it is what [`propagate`](crate::propagate) uses unless the caller opts into [`EngineSelection::Auto`](super::EngineSelection).
//! See `research/FINDINGS.md` for the threshold's justification.
//!
//! # Why a second path exists at all
//!
//! The bucketed layer has a per-layer fixed cost that does not shrink with the term count — dominated by `Channel::prepare`'s PTM derivation — so at small enough resident term counts that cost is most of the layer.
//! This path removes it: no `prepare` (no PTM derivation, no delta plan), no rebucket, no coset span, no permute/unpermute, no sort.
//! One [`Channel::apply`] call per resident term, straight into an [`FxBuildHasher`] map keyed by the Pauli string, exactly as `test_support::naive_apply_layer` does — that oracle *is* this algorithm, which is why the differential tests here are a real check and not a tautology.
//!
//! # The representation is the map, between layers too
//!
//! The win requires that consecutive small layers do **not** round-trip through a [`PauliSum`]: materializing one costs a sort of the whole sum, which is more than the fixed cost being saved.
//! So a [`DirectSum`] holds the terms in its map across layers and materializes exactly once — when the sum outgrows the threshold, or when the propagation ends.
//!
//! Two consequences, both deliberate:
//!
//! - A [`TruncationPolicy`]'s `keep_term` is applied per layer, on summed coefficients, in the same place the merge applies it.
//!   Its `finalize_layer` needs a real `PauliSum`, so it costs a materialize → finalize → re-ingest round trip; [`TruncationPolicy::finalizes_layer`] is how a policy says that round trip is pointless.
//! - This path needs only [`Channel::apply`], never `Channel::prepare`, so it applies channels of **any** support width — including the > 2-qubit channels that make the bucketed path panic.
//!   It is a strictly wider fallback, not a narrower fast path (see `super::propagate_with_options` for what that means for a circuit that mixes the two).
//!
//! GPU-readiness (ARCHITECTURE.md §GPU-Readiness) is the bucketed path's story and this path makes no claim on it: a hash map is not a device buffer, and at these term counts there is nothing to offload.

use hashbrown::HashMap;
use num_complex::Complex64;
use rustc_hash::FxBuildHasher;

use crate::channel::{Channel, OutputBuffer};
use crate::pauli_string::PauliString;
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::storage::{desired_bits, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// A Pauli sum held as a hash map, for the direct-apply path.
///
/// Owns the map, one double-buffer for the layer's output, and the [`OutputBuffer`] columns a `Channel::apply` writes into.
/// Capacity in all of them is retained across layers, so the steady state of a run of small layers allocates nothing.
///
/// The [`Gf2Hash`] is carried, not used: the map needs no partition, but [`Self::to_sum`] must hand back a `PauliSum` under the *same* hash rows and seed the caller entered with, and with at least as many bucket bits (the engine's partition is grow-only — see [`PauliSum::rebucket`]).
pub(crate) struct DirectSum<const W: usize> {
    /// The resident terms. Never holds an exact-zero coefficient: every layer filters them out, and an entering `PauliSum` cannot contain one.
    live: HashMap<PauliString<W>, Complex64, FxBuildHasher>,
    /// The layer's output, swapped into `live` at the end of the layer.
    /// Output keys can collide with input keys not yet visited, so the accumulation cannot be done in place.
    next: HashMap<PauliString<W>, Complex64, FxBuildHasher>,
    /// `Channel::apply`'s output columns, sized to the widest `max_fanout` seen.
    buf_x: Vec<[u64; W]>,
    buf_z: Vec<[u64; W]>,
    buf_c: Vec<Complex64>,
    /// The partition to hand back under, at the bit count entered with.
    hash: Gf2Hash<W>,
    num_qubits: usize,
}

impl<const W: usize> DirectSum<W> {
    /// Ingest a bucketed sum: one map insert per term, `O(n)`.
    ///
    /// Consumes the sum — its bucket columns are dead once the terms are in the map, and freeing them here keeps the direct path's footprint to the map.
    pub(crate) fn from_sum(sum: PauliSum<W>) -> Self {
        let n = sum.len();
        let mut live = HashMap::with_capacity_and_hasher(n, FxBuildHasher);
        for (x, z, c) in sum.iter() {
            live.insert(PauliString::<W> { x: *x, z: *z }, c);
        }
        Self {
            live,
            next: HashMap::with_capacity_and_hasher(n, FxBuildHasher),
            buf_x: Vec::new(),
            buf_z: Vec::new(),
            buf_c: Vec::new(),
            hash: sum.hash().clone(),
            num_qubits: sum.num_qubits(),
        }
    }

    /// Resident term count — the same quantity [`PauliSum::len`] reports, so the engine's per-layer term counts and `TermTrace` are unaffected by which path produced them.
    pub(crate) fn len(&self) -> usize {
        self.live.len()
    }

    /// Apply one channel layer in place.
    ///
    /// `Channel::apply` (or `apply_adjoint`) once per resident term into the fanout-sized buffer, accumulate every emitted row into the output map, then one filtering pass: drop exact-zero sums and apply [`TruncationPolicy::keep_term`] to the *summed* coefficient.
    /// That order is the merge phase's order (`engine::merge`), not a variant of it — a row whose coefficient is an exact zero still participates in its key's sum, since pre-filtering it would flip the sign of a zero sum.
    ///
    /// Equal-key contributions are summed in map iteration order, which is unspecified, so results agree with the bucketed path to floating-point tolerance and not bitwise (ARCHITECTURE.md §Determinism).
    pub(crate) fn apply_layer<T>(&mut self, ch: &dyn Channel<W>, policy: &T, adjoint: bool)
    where
        T: TruncationPolicy<W> + ?Sized,
    {
        let Self {
            live,
            next,
            buf_x,
            buf_z,
            buf_c,
            ..
        } = self;

        let fanout = ch.max_fanout().max(1);
        if buf_x.len() < fanout {
            buf_x.resize(fanout, [0u64; W]);
            buf_z.resize(fanout, [0u64; W]);
            buf_c.resize(fanout, ZERO);
        }

        next.clear();
        next.reserve(live.len());

        for (p, &c) in live.iter() {
            let mut len = 0usize;
            {
                let mut out = OutputBuffer::<W> {
                    x: buf_x,
                    z: buf_z,
                    coeff: buf_c,
                    len: &mut len,
                };
                if adjoint {
                    ch.apply_adjoint(&p.x, &p.z, c, &mut out);
                } else {
                    ch.apply(&p.x, &p.z, c, &mut out);
                }
            }
            for i in 0..len {
                let key = PauliString::<W> {
                    x: buf_x[i],
                    z: buf_z[i],
                };
                *next.entry(key).or_insert(ZERO) += buf_c[i];
            }
        }

        next.retain(|p, c| *c != ZERO && policy.keep_term(&p.x, &p.z, *c));
        std::mem::swap(live, next);
    }

    /// Materialize a bucketed [`PauliSum`], leaving the map intact.
    ///
    /// One sort by key plus the key-sorted scatter.
    /// The partition is the entering hash's rows and seed at `max(desired_bits(len), entering bits)` — the same clamp [`PauliSum::rebucket`] applies, so a sum handed back to the bucketed path is partitioned exactly as that path would have partitioned it, and the grow-only invariant on the bucket count survives the detour.
    ///
    /// Borrowing rather than consuming because the `finalize_layer` round trip needs the map back afterwards ([`Self::reload`]).
    pub(crate) fn to_sum(&self) -> PauliSum<W> {
        let mut entries: Vec<(PauliString<W>, Complex64)> =
            self.live.iter().map(|(p, c)| (*p, *c)).collect();
        entries.sort_unstable_by(|a, b| (&a.0.x, &a.0.z).cmp(&(&b.0.x, &b.0.z)));
        let n = entries.len();
        let mut x = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut coeff = Vec::with_capacity(n);
        for (p, c) in entries {
            x.push(p.x);
            z.push(p.z);
            coeff.push(c);
        }
        let bits =
            desired_bits(n, DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS).max(self.hash.bits());
        let hash = if bits == self.hash.bits() {
            self.hash.clone()
        } else {
            Gf2Hash::new(self.num_qubits, bits, self.hash.seed())
        };
        PauliSum::from_key_sorted(&x, &z, &coeff, hash, self.num_qubits)
    }

    /// Re-ingest a sum that left this path for one operation and came back — the materialize → [`TruncationPolicy::finalize_layer`] → re-ingest round trip.
    /// Retains the map's capacity; the carried hash is left as it was (`to_sum` only ever grows it, and `rebucket` would too).
    pub(crate) fn reload(&mut self, sum: &PauliSum<W>) {
        self.live.clear();
        self.live.reserve(sum.len());
        for (x, z, c) in sum.iter() {
            self.live.insert(PauliString::<W> { x: *x, z: *z }, c);
        }
    }
}

/// Run the leading layers of `circuit` on the direct path, returning the materialized sum and **how many layers were applied** — the `k` the caller's sorting loop resumes at.
///
/// Stops after the layer that leaves the sum above [`PropagateOptions::small_sum_threshold`], or when the circuit runs out.
/// The caller has already decided this path applies ([`PropagateOptions::starts_direct`]); nothing here re-decides, and nothing here can hand control back mid-circuit.
///
/// `#[inline(never)]` on purpose: it must not land inside `propagate_with_scratch_and_options`'s body, whose layer loop inlines `apply_layer_bucketed` and its merge kernels, which are sensitive to a few bytes of code motion (CLAUDE.md §Performance discipline).
/// The default `SortedOnly` path must be able to reach this function's call site and not its code.
///
/// # Per-layer records
///
/// The `TermTrace`/`GateTrace` pushes and the `DEBUG` progress line are emitted here in the same order, with the same fields and the same format string as the sorting loop's epilogue, because downstream tooling parses them: the cross-engine head-to-head driver reads per-layer `terms_in -> terms_out` counts out of exactly these `DEBUG` records to gate term-count parity against PauliPropagation.jl.
#[inline(never)]
pub(crate) fn run_direct_prefix<const W: usize, T>(
    circuit: &crate::circuit::Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: super::Direction,
    scratch: &mut super::LayerScratch<W>,
    options: super::PropagateOptions,
) -> (PauliSum<W>, usize)
where
    T: TruncationPolicy<W> + ?Sized,
{
    let n = circuit.channels.len();
    let adjoint = matches!(direction, super::Direction::Heisenberg);
    let tracing = scratch.term_trace.is_some();
    let gate_tracing = scratch.gate_trace.is_some();
    // Read once: a policy cannot change its answer mid-circuit, and the branch it guards costs a materialize plus a re-ingest.
    let finalizes = policy.finalizes_layer();

    let mut direct = DirectSum::from_sum(sum);
    let mut applied = 0usize;

    while applied < n {
        let idx = match direction {
            super::Direction::Forward => applied,
            super::Direction::Heisenberg => n - 1 - applied,
        };
        let ch: &dyn Channel<W> = circuit.channels[idx].as_ref();
        let application_index = applied;

        let debug_on = log::log_enabled!(target: super::LOG_TARGET, log::Level::Debug);
        let want_timer = gate_tracing || debug_on;
        let layer_t0 = want_timer.then(std::time::Instant::now);
        let terms_before = direct.len();

        direct.apply_layer(ch, policy, adjoint);

        // A layer pass needs a real `PauliSum`, so it costs a round trip.
        // The policy machinery itself is untouched — this is the same `finalize_layer` call the sorting loop makes, on the same type, at the same point in the layer.
        if finalizes {
            let mut materialized = direct.to_sum();
            policy.finalize_layer(&mut materialized);
            direct.reload(&materialized);
        }

        let terms_after = direct.len();
        applied += 1;

        if tracing {
            super::record_layer_terms(scratch, terms_before, terms_after);
        }
        if want_timer {
            let dt = layer_t0
                .expect("want_timer implies layer_t0 is Some")
                .elapsed();
            if gate_tracing {
                super::record_gate_trace(
                    scratch,
                    idx as u32,
                    application_index as u32,
                    ch.debug_name(),
                    terms_before,
                    terms_after,
                    dt,
                );
            }
            if debug_on {
                log::debug!(
                    target: super::LOG_TARGET,
                    "layer {}/{} [{}]: {} -> {} terms, {:.1} ms",
                    applied,
                    n,
                    ch.debug_name(),
                    terms_before,
                    terms_after,
                    dt.as_secs_f64() * 1e3,
                );
            }
        }

        if terms_after > options.small_sum_threshold {
            break;
        }
    }

    (direct.to_sum(), applied)
}

#[cfg(test)]
mod tests;
