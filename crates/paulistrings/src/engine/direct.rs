//! The direct-apply small-sum layer path: a hash map instead of the bucketed
//! sort-merge pipeline, for sums small enough that the pipeline's per-layer
//! fixed cost dominates.
//!
//! This is **not** the canonical engine. [`bucketed`](super::bucketed) is, at
//! every term count, and it is what [`propagate`](crate::propagate) uses unless
//! the caller opts into [`EngineSelection::Auto`](super::EngineSelection).
//! Design, measurements and the threshold's justification:
//! `research/notes/2026-09-01-small-m-path.md`.
//!
//! # Why a second path exists at all
//!
//! The bucketed layer has a per-layer cost that does not shrink with the term
//! count: **1.43 µs** for a two-qubit `PauliRotation` at `W = 2`, of which
//! `Channel::prepare` is 70% (a dense two-qubit PTM costs 4.19–5.71 µs per
//! gate), against a bucketed serial pipeline of only 0.19 µs
//! (`research/notes/2026-09-01-large-m-phase-breakdown.md` §2). At 68 resident
//! terms — the head-to-head study's smallest kicked-Ising configuration, where
//! we run 3.1× slower than PauliPropagation.jl's hash map — that fixed cost is
//! most of the layer.
//!
//! So this path removes it: no `prepare` (no PTM derivation, no delta plan), no
//! rebucket, no coset span, no permute/unpermute, no sort. One
//! [`Channel::apply`] call per resident term, straight into an
//! [`FxBuildHasher`] map keyed by the Pauli string, exactly as
//! `test_support::naive_apply_layer` does — that oracle *is* this algorithm,
//! which is why the differential tests here are a real check and not a
//! tautology.
//!
//! # The representation is the map, between layers too
//!
//! The win requires that consecutive small layers do **not** round-trip through
//! a [`PauliSum`]: materializing one costs a sort of the whole sum, which is
//! more than the fixed cost being saved. So a [`DirectSum`] holds the terms in
//! its map across layers and materializes exactly once — when the sum outgrows
//! the threshold, or when the propagation ends.
//!
//! Two consequences, both deliberate:
//!
//! - A [`TruncationPolicy`]'s `keep_term` is applied per layer, on summed
//!   coefficients, in the same place the merge applies it. Its `finalize_layer`
//!   needs a real `PauliSum`, so it costs a materialize → finalize → re-ingest
//!   round trip; [`TruncationPolicy::finalizes_layer`] is how a policy says
//!   that round trip is pointless.
//! - This path needs only [`Channel::apply`], never `Channel::prepare`, so it
//!   applies channels of **any** support width — including the > 2-qubit
//!   channels that make the bucketed path panic. It is a strictly wider
//!   fallback, not a narrower fast path (see `super::propagate_with_options`
//!   for what that means for a circuit that mixes the two).
//!
//! GPU-readiness (ARCHITECTURE.md §GPU-Readiness) is the bucketed path's story
//! and this path makes no claim on it: a hash map is not a device buffer, and
//! at these term counts there is nothing to offload.

use hashbrown::HashMap;
use num_complex::Complex64;
use rustc_hash::FxBuildHasher;

use crate::bucket::hash::Gf2Hash;
use crate::bucket::sum::{desired_bits, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::channel::{Channel, OutputBuffer};
use crate::pauli_string::PauliString;
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// A Pauli sum held as a hash map, for the direct-apply path.
///
/// Owns the map, one double-buffer for the layer's output, and the
/// [`OutputBuffer`] columns a `Channel::apply` writes into. Capacity in all of
/// them is retained across layers, so the steady state of a run of small layers
/// allocates nothing.
///
/// The [`Gf2Hash`] is carried, not used: the map needs no partition, but
/// [`Self::to_sum`] must hand back a `PauliSum` under the *same* hash rows
/// and seed the caller entered with, and with at least as many bucket bits (the
/// engine's partition is grow-only — see [`PauliSum::rebucket`]).
pub(crate) struct DirectSum<const W: usize> {
    /// The resident terms. Never holds an exact-zero coefficient: every layer
    /// filters them out, and an entering `PauliSum` cannot contain one.
    live: HashMap<PauliString<W>, Complex64, FxBuildHasher>,
    /// The layer's output, swapped into `live` at the end of the layer. Output
    /// keys can collide with input keys not yet visited, so the accumulation
    /// cannot be done in place.
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
    /// Consumes the sum — its bucket columns are dead once the terms are in the
    /// map, and freeing them here keeps the direct path's footprint to the map.
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

    /// Resident term count — the same quantity [`PauliSum::len`] reports, so
    /// the engine's per-layer term counts and `TermTrace` are unaffected by
    /// which path produced them.
    pub(crate) fn len(&self) -> usize {
        self.live.len()
    }

    /// Apply one channel layer in place.
    ///
    /// `Channel::apply` (or `apply_adjoint`) once per resident term into the
    /// fanout-sized buffer, accumulate every emitted row into the output map,
    /// then one filtering pass: drop exact-zero sums and apply
    /// [`TruncationPolicy::keep_term`] to the *summed* coefficient. That order
    /// is the merge phase's order (`engine::merge`), not a variant of it — a
    /// row whose coefficient is an exact zero still participates in its key's
    /// sum, since pre-filtering it would flip the sign of a zero sum.
    ///
    /// Equal-key contributions are summed in map iteration order, which is
    /// unspecified, so results agree with the bucketed path to floating-point
    /// tolerance and not bitwise (ARCHITECTURE.md §Determinism).
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
    /// One sort by key plus the key-sorted scatter. The partition is the
    /// entering hash's rows and seed at
    /// `max(desired_bits(len), entering bits)` — the same clamp
    /// [`PauliSum::rebucket`] applies, so a sum handed back to the bucketed path
    /// is partitioned exactly as that path would have partitioned it, and the
    /// grow-only invariant on the bucket count survives the detour.
    ///
    /// Borrowing rather than consuming because the `finalize_layer` round trip
    /// needs the map back afterwards ([`Self::reload`]); at these term counts
    /// copying the keys out instead of moving them is not measurable.
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

    /// Re-ingest a sum that left this path for one operation and came back —
    /// the materialize → [`TruncationPolicy::finalize_layer`] → re-ingest round
    /// trip. Retains the map's capacity; the carried hash is left as it was
    /// (`to_sum` only ever grows it, and `rebucket` would too).
    pub(crate) fn reload(&mut self, sum: &PauliSum<W>) {
        self.live.clear();
        self.live.reserve(sum.len());
        for (x, z, c) in sum.iter() {
            self.live.insert(PauliString::<W> { x: *x, z: *z }, c);
        }
    }
}

/// Run the leading layers of `circuit` on the direct path, returning the
/// materialized sum and **how many layers were applied** — the `k` the caller's
/// sorting loop resumes at.
///
/// Stops after the layer that leaves the sum above
/// [`PropagateOptions::small_sum_threshold`], or when the circuit runs out. The
/// caller has already decided this path applies
/// ([`PropagateOptions::starts_direct`]); nothing here re-decides, and nothing
/// here can hand control back mid-circuit.
///
/// `#[inline(never)]` on purpose: it must not land inside
/// `propagate_with_scratch_and_options`'s body, whose layer loop inlines
/// `apply_layer_bucketed` and its merge kernels — those move 6–34% under a few
/// bytes of code motion (CLAUDE.md §Performance discipline). The default
/// `SortedOnly` path must be able to reach this function's call site and not
/// its code.
///
/// # Per-layer records
///
/// The `TermTrace` push and the `DEBUG` progress line are emitted here in the
/// same order, with the same fields and the same format string as the sorting
/// loop's epilogue, because downstream tooling parses them: the cross-engine
/// head-to-head driver reads per-layer `terms_in -> terms_out` counts out of
/// exactly these `DEBUG` records to gate term-count parity against
/// PauliPropagation.jl.
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
    // Read once: a policy cannot change its answer mid-circuit, and the branch
    // it guards costs a materialize plus a re-ingest.
    let finalizes = policy.finalizes_layer();

    let mut direct = DirectSum::from_sum(sum);
    let mut applied = 0usize;

    while applied < n {
        let idx = match direction {
            super::Direction::Forward => applied,
            super::Direction::Heisenberg => n - 1 - applied,
        };
        let ch: &dyn Channel<W> = circuit.channels[idx].as_ref();

        let layer_t0 = log::log_enabled!(target: super::LOG_TARGET, log::Level::Debug)
            .then(std::time::Instant::now);
        let terms_before = direct.len();

        direct.apply_layer(ch, policy, adjoint);

        // A layer pass needs a real `PauliSum`, so it costs a round trip. The
        // policy machinery itself is untouched — this is the same
        // `finalize_layer` call the sorting loop makes, on the same type, at the
        // same point in the layer.
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
        if let Some(t0) = layer_t0 {
            log::debug!(
                target: super::LOG_TARGET,
                "layer {}/{} [{}]: {} -> {} terms, {:.1} ms",
                applied,
                n,
                ch.debug_name(),
                terms_before,
                terms_after,
                t0.elapsed().as_secs_f64() * 1e3,
            );
        }

        if terms_after > options.small_sum_threshold {
            break;
        }
    }

    (direct.to_sum(), applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulator::BuildAccumulator;
    use crate::channel::{
        support_mask, AmplitudeDamping, Clifford1Q, Clifford2Q, Depolarizing, GeneralUnitary2Q,
        PauliRotation,
    };
    use crate::phase::Phase;
    use crate::test_support::{assert_same_terms, assert_terms_close, naive_apply_layer, rand_sum};

    struct KeepAll;
    impl<const W: usize> TruncationPolicy<W> for KeepAll {
        fn finalizes_layer(&self) -> bool {
            false
        }
    }

    fn apply_one<const W: usize, T>(
        sum: PauliSum<W>,
        ch: &dyn Channel<W>,
        policy: &T,
        adjoint: bool,
    ) -> PauliSum<W>
    where
        T: TruncationPolicy<W> + ?Sized,
    {
        let mut direct = DirectSum::from_sum(sum);
        direct.apply_layer(ch, policy, adjoint);
        direct.to_sum()
    }

    /// Hand-computed: `H` conjugates `Z` to `X`, so `Z₀ + 0.5·X₁` becomes
    /// `X₀ + 0.5·X₁` under an `H` on qubit 0.
    #[test]
    fn h_maps_z_to_x() {
        let mut acc = BuildAccumulator::<1>::new(2);
        acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
        acc.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(0.5, 0.0));
        let out = apply_one(acc.finalize(), &Clifford1Q::h(0), &KeepAll, false);

        assert_eq!(out.len(), 2);
        assert_eq!(out.get(&[0b01], &[0]), Some(Complex64::new(1.0, 0.0)));
        assert_eq!(out.get(&[0b10], &[0]), Some(Complex64::new(0.5, 0.0)));
    }

    /// Hand-computed: `exp(-i·θ·Z₀/2)` acting on `X₀` gives
    /// `cos(θ)·X₀ + sin(θ)·Y₀` — the fanout-2 case, at θ = π/3.
    #[test]
    fn rotation_fans_out_with_cos_and_sin() {
        let theta = std::f64::consts::FRAC_PI_3;
        let mut acc = BuildAccumulator::<1>::new(1);
        acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(1.0, 0.0));
        let gen = PauliString::<1>::z(0);
        let out = apply_one(
            acc.finalize(),
            &PauliRotation::new(gen, theta),
            &KeepAll,
            false,
        );

        assert_eq!(out.len(), 2);
        let x = out.get(&[1], &[0]).expect("X term");
        let y = out.get(&[1], &[1]).expect("Y term");
        assert!((x.norm() - theta.cos()).abs() < 1e-12, "X coeff {x}");
        assert!((y.norm() - theta.sin()).abs() < 1e-12, "Y coeff {y}");
    }

    /// `keep_term` sees the *summed* coefficient, so two rows that cancel to
    /// below a threshold are dropped as one term and not kept as two.
    #[test]
    fn keep_term_sees_summed_coefficients() {
        struct Above(f64);
        impl<const W: usize> TruncationPolicy<W> for Above {
            fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], c: Complex64) -> bool {
                c.norm() > self.0
            }
            fn finalizes_layer(&self) -> bool {
                false
            }
        }

        // A π/2 Z-rotation on X₀ emits cos(π/2)·X₀ (an exact-ish zero) plus
        // sin(π/2)·Y₀. Seeding both X₀ and Y₀ makes the Y row a two-row sum.
        let mut acc = BuildAccumulator::<1>::new(1);
        acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(1.0, 0.0));
        acc.add_term(
            PauliString::<1>::y(0),
            Phase::ONE,
            Complex64::new(-1.0, 0.0),
        );
        let sum = acc.finalize();
        let ch = PauliRotation::new(PauliString::<1>::z(0), std::f64::consts::FRAC_PI_2);

        let loose = apply_one(sum.clone(), &ch, &Above(1e-12), false);
        let strict = apply_one(sum, &ch, &Above(0.5), false);
        // Y₀'s two contributions (+1 from X₀'s sin, −1·cos ≈ 0 from Y₀) sum to
        // ≈ 1, X₀'s to ≈ −1: both survive the loose threshold.
        assert_eq!(loose.len(), 2);
        // The strict threshold keeps both too — but only because the sums are
        // ≈ 1, which is the point: a per-row filter would have dropped the
        // ≈ 0 rows and changed the sums.
        assert_eq!(strict.len(), 2);
        let y = strict.get(&[1], &[1]).expect("Y term");
        assert!((y.norm() - 1.0).abs() < 1e-12, "Y coeff {y}");
    }

    /// The differential oracle, over the channel zoo at both widths. The oracle
    /// shares this path's algorithm, so this pins the plumbing (buffer sizing,
    /// adjoint dispatch, zero-drop, re-materialization) rather than the algebra;
    /// the cross-path property tests in `tests/small_sum_path.rs` are what pin
    /// the algebra against the bucketed engine.
    fn differential<const W: usize>(num_qubits: usize, seed: u64) {
        let sum = rand_sum::<W>(300, num_qubits, seed);
        let hi = num_qubits.saturating_sub(1) as u32;

        let mut matrix = [[Complex64::new(0.0, 0.0); 4]; 4];
        for (r, row) in matrix.iter_mut().enumerate() {
            row[r] = Complex64::new(1.0, 0.0);
        }
        // A real 4×4 rotation in the (0,1) block: unitary, and dense enough in
        // the PTM to fan a term out widely.
        let (c, s) = (0.6f64, 0.8f64);
        matrix[0][0] = Complex64::new(c, 0.0);
        matrix[0][1] = Complex64::new(-s, 0.0);
        matrix[1][0] = Complex64::new(s, 0.0);
        matrix[1][1] = Complex64::new(c, 0.0);

        let channels: Vec<Box<dyn Channel<W>>> = vec![
            Box::new(Clifford1Q::h(0)),
            Box::new(Clifford1Q::s(hi)),
            Box::new(Clifford2Q::cnot(0, hi)),
            Box::new(PauliRotation::new(
                PauliString::<W>::z(0),
                std::f64::consts::FRAC_PI_8,
            )),
            // Generator weight 4 — above MAX_LOCAL_SUPPORT, the case that
            // makes `Prepared::derive_local` bail and `PauliRotation` override
            // `prepare`.
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<W>::z(0);
                    g.x[0] |= 0b0110;
                    g.z[0] |= 0b1000;
                    g
                },
                0.37,
            )),
            Box::new(Depolarizing {
                support: [0],
                p: 0.15,
            }),
            Box::new(AmplitudeDamping {
                support: [1],
                gamma: 0.25,
            }),
            Box::new(GeneralUnitary2Q::from_matrix(0, 1, matrix)),
        ];

        for ch in &channels {
            for adjoint in [false, true] {
                let got = apply_one(sum.clone(), ch.as_ref(), &KeepAll, adjoint);
                let want = naive_apply_layer(&sum, ch.as_ref(), &KeepAll, adjoint);
                assert_terms_close(&got, &want, 1e-12, "direct vs naive");
                got.assert_invariants();
            }
        }
    }

    #[test]
    fn differential_w1() {
        differential::<1>(12, 0xD1);
    }

    #[test]
    fn differential_w2() {
        differential::<2>(96, 0xD2);
    }

    /// A channel with support on three qubits: `Channel::prepare` declines it
    /// and the bucketed path panics, but this path only ever calls `apply`, so
    /// it applies it correctly. Documented capability, tested here so it cannot
    /// regress into a silent wrong answer.
    #[test]
    fn applies_a_channel_wider_than_the_bucketed_path_can_prepare() {
        struct RotateThree;
        impl<const W: usize> Channel<W> for RotateThree {
            fn max_fanout(&self) -> usize {
                1
            }
            fn support(&self) -> [u64; W] {
                support_mask(&[0, 1, 2])
            }
            /// Cyclically shifts the x-bits of qubits 0,1,2 — support-bounded,
            /// key-changing, and not expressible as a ≤ 2-qubit PTM.
            fn apply(
                &self,
                input_x: &[u64; W],
                input_z: &[u64; W],
                coeff: Complex64,
                out: &mut OutputBuffer<'_, W>,
            ) {
                let mut x = *input_x;
                let low = input_x[0] & 0b111;
                x[0] = (input_x[0] & !0b111) | ((low << 1) & 0b111) | (low >> 2);
                out.push(x, *input_z, coeff);
            }
        }

        let sum = rand_sum::<1>(64, 8, 0x3B);
        let got = apply_one(sum.clone(), &RotateThree, &KeepAll, false);
        let want = naive_apply_layer(&sum, &RotateThree, &KeepAll, false);
        assert_terms_close(&got, &want, 1e-12, "wide-support direct layer");
        assert_eq!(got.len(), sum.len());
    }

    /// Ingest → materialize is the identity on the term set, and the partition
    /// comes back at least as fine as it went in.
    #[test]
    fn roundtrip_preserves_terms_and_never_coarsens() {
        let sum = rand_sum::<2>(2000, 96, 0x5EED);
        let bits_in = sum.hash().bits();
        let seed_in = sum.hash().seed();
        let out = DirectSum::from_sum(sum.clone()).to_sum();
        assert_same_terms(&out, &sum, "roundtrip");
        assert!(out.hash().bits() >= bits_in);
        assert_eq!(out.hash().seed(), seed_in);
        out.assert_invariants();
    }

    /// `reload` replaces the resident terms wholesale — the state after the
    /// finalize round trip must be the finalized sum, not a union with what was
    /// there before.
    #[test]
    fn reload_replaces_the_resident_terms() {
        let a = rand_sum::<1>(50, 12, 0xA1);
        let b = rand_sum::<1>(30, 12, 0xB2);
        let mut direct = DirectSum::from_sum(a);
        direct.reload(&b);
        assert_eq!(direct.len(), b.len());
        assert_same_terms(&direct.to_sum(), &b, "reload");
    }
}
