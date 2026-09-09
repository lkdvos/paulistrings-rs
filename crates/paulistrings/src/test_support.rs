//! Shared fixtures for the crate's own tests, benches and examples.
//!
//! Compiled only under `cfg(test)` or the `test-utils` feature, which the
//! crate turns on for its own dev builds via a self-dev-dependency. Nothing
//! here is part of the public API — the module is `#[doc(hidden)]` and may
//! change without notice.
//!
//! It exists so the differential oracle and the random-sum fixtures have one
//! canonical implementation instead of a copy per test file. In particular
//! [`naive_apply_layer`] is the **oracle** the bucketed engine is tested
//! against, and deliberately shares no code with the thing it checks — one
//! `Channel::apply` call per input term, a hashmap accumulation, and a sort.

use hashbrown::HashMap;
use num_complex::Complex64;
use rustc_hash::FxBuildHasher;

use crate::accumulator::BuildAccumulator;
use crate::channel::{Channel, OutputBuffer};
use crate::pauli_string::PauliString;
use crate::pauli_sum::PauliSum;
use crate::phase::Phase;
use crate::truncation::TruncationPolicy;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Apply one channel layer the obvious way, as a differential oracle.
///
/// For every input term, call [`Channel::apply`] (or
/// [`Channel::apply_adjoint`] when `adjoint`) into a `max_fanout`-sized
/// [`OutputBuffer`], accumulate the emitted rows into a hashmap keyed by
/// `(x, z)`, then filter the *summed* coefficients through
/// [`TruncationPolicy::keep_term`], drop exact zeros, sort by key, and rebuild
/// a [`PauliSum`] under the input's own hash.
///
/// Deliberately naive: no bucketing, no coset structure, no parallelism, and
/// no shared code with `engine::bucketed` beyond the `Channel` trait itself.
/// Equal keys are summed in hashmap iteration order, which is unspecified, so
/// results agree with the engine only to floating-point tolerance — compare
/// with [`assert_terms_close`], never bitwise. Dropping exact zeros matches the
/// engine's own zero-drop, and near-zero terms present on one side only are
/// tolerated by `assert_terms_close`.
pub fn naive_apply_layer<const W: usize>(
    input: &PauliSum<W>,
    ch: &dyn Channel<W>,
    policy: &dyn TruncationPolicy<W>,
    adjoint: bool,
) -> PauliSum<W> {
    let mf = ch.max_fanout().max(1);
    let mut buf_x = vec![[0u64; W]; mf];
    let mut buf_z = vec![[0u64; W]; mf];
    let mut buf_c = vec![ZERO; mf];
    let mut acc: HashMap<([u64; W], [u64; W]), Complex64, FxBuildHasher> =
        HashMap::with_capacity_and_hasher(input.len(), FxBuildHasher);

    for (x, z, c) in input.iter() {
        let mut len = 0usize;
        {
            let mut out = OutputBuffer::<W> {
                x: &mut buf_x,
                z: &mut buf_z,
                coeff: &mut buf_c,
                len: &mut len,
            };
            if adjoint {
                ch.apply_adjoint(x, z, c, &mut out);
            } else {
                ch.apply(x, z, c, &mut out);
            }
        }
        for i in 0..len {
            *acc.entry((buf_x[i], buf_z[i])).or_insert(ZERO) += buf_c[i];
        }
    }

    let mut kept: Vec<([u64; W], [u64; W], Complex64)> = acc
        .into_iter()
        .filter(|((x, z), v)| *v != ZERO && policy.keep_term(x, z, *v))
        .map(|((x, z), v)| (x, z, v))
        .collect();
    kept.sort_unstable_by_key(|&(x, z, _)| (x, z));

    let xs: Vec<[u64; W]> = kept.iter().map(|t| t.0).collect();
    let zs: Vec<[u64; W]> = kept.iter().map(|t| t.1).collect();
    let cs: Vec<Complex64> = kept.iter().map(|t| t.2).collect();
    PauliSum::from_key_sorted(&xs, &zs, &cs, input.hash().clone(), input.num_qubits())
}

/// Xorshift64 — small, deterministic, no dev-dependency.
pub struct Xs64(u64);

impl Xs64 {
    /// Seed the generator, avoiding the degenerate all-zero state.
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    /// Next 64 random bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// `W` consecutive draws, word 0 first.
    #[inline]
    pub fn next_array<const W: usize>(&mut self) -> [u64; W] {
        let mut a = [0u64; W];
        for slot in a.iter_mut() {
            *slot = self.next_u64();
        }
        a
    }
}

/// Mask of the bits of word `word` that belong to a `num_qubits`-qubit key.
pub fn word_mask(num_qubits: usize, word: usize) -> u64 {
    let lo = 64 * word;
    if num_qubits >= lo + 64 {
        !0u64
    } else if num_qubits <= lo {
        0
    } else {
        (1u64 << (num_qubits - lo)) - 1
    }
}

/// `n` random dense terms on `num_qubits` qubits, deduplicated by
/// [`BuildAccumulator`] — so the realized length can be below `n` at small
/// qubit counts.
///
/// The RNG draw order is `(x[0], z[0], x[1], z[1], …, re, im)` per term; keep
/// it stable, because every fixture seed in the test suite encodes it.
pub fn rand_sum<const W: usize>(n: usize, num_qubits: usize, seed: u64) -> PauliSum<W> {
    let mut rng = Xs64::new(seed);
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n);
    for _ in 0..n {
        let mut p = PauliString::<W> {
            x: [0u64; W],
            z: [0u64; W],
        };
        for w in 0..W {
            let m = word_mask(num_qubits, w);
            p.x[w] = rng.next_u64() & m;
            p.z[w] = rng.next_u64() & m;
        }
        let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        let im = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        acc.add_term(p, Phase::ONE, Complex64::new(re, im));
    }
    acc.finalize()
}

/// As [`rand_sum`], but coefficients are **real** — one draw per term instead
/// of two.
///
/// A separate generator rather than a flag because the draw order differs
/// (`…, re` versus `…, re, im`), so the two produce completely different
/// streams from the same seed. Fixtures seeded through this one are pinned to
/// it; do not "unify" the two.
pub fn rand_sum_real<const W: usize>(n: usize, num_qubits: usize, seed: u64) -> PauliSum<W> {
    let mut rng = Xs64::new(seed);
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n);
    for _ in 0..n {
        let mut p = PauliString::<W> {
            x: [0u64; W],
            z: [0u64; W],
        };
        for w in 0..W {
            let m = word_mask(num_qubits, w);
            p.x[w] = rng.next_u64() & m;
            p.z[w] = rng.next_u64() & m;
        }
        let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        acc.add_term(p, Phase::ONE, Complex64::new(re, 0.0));
    }
    acc.finalize()
}

/// A dense random key with **no** masking: all `W` `x` words first, then all
/// `W` `z` words.
///
/// Valid only when `num_qubits` is a multiple of 64 — every bit it sets must be
/// a live qubit or `BuildAccumulator` will reject the term. The draw order is
/// word-major (`x[0..W]` then `z[0..W]`), *not* the per-word interleave
/// [`rand_sum`] uses, so the two disagree from `W = 2` up.
pub fn rand_pauli<const W: usize>(rng: &mut Xs64) -> PauliString<W> {
    PauliString::<W> {
        x: rng.next_array::<W>(),
        z: rng.next_array::<W>(),
    }
}

/// `n` dense random terms built from [`rand_pauli`] — the benchmark input
/// recipe, whose seeds are pinned by the committed criterion baselines.
///
/// Distinct stream from [`rand_sum`]; see [`rand_pauli`] for why.
pub fn rand_sum_unmasked<const W: usize>(n: usize, num_qubits: usize, seed: u64) -> PauliSum<W> {
    let mut rng = Xs64::new(seed);
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n);
    for _ in 0..n {
        let p = rand_pauli::<W>(&mut rng);
        let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        let im = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        acc.add_term(p, Phase::ONE, Complex64::new(re, im));
    }
    acc.finalize()
}

/// A Pauli string of Hamming weight `weight` over `num_qubits` qubits.
///
/// This is the *realistic* occupancy regime: physical Hamiltonians are
/// low-weight, and `WeightCutoff` truncation keeps them that way. The dense
/// [`rand_pauli`] is the opposite extreme. Any bucketing scheme derived from
/// key bits behaves very differently on the two, so both are benched.
///
/// Index-bounded by construction (`q` is drawn `mod num_qubits`), so — unlike
/// [`rand_pauli`] — it needs no masking pass and is safe at any qubit count.
pub fn low_weight_pauli<const W: usize>(
    rng: &mut Xs64,
    num_qubits: usize,
    weight: usize,
) -> PauliString<W> {
    let mut p = PauliString::<W> {
        x: [0u64; W],
        z: [0u64; W],
    };
    for _ in 0..weight {
        let q = (rng.next_u64() as usize) % num_qubits;
        let word = q / 64;
        let bit = 1u64 << (q % 64);
        // Pick one of X, Z, Y (never I, or the weight would not be `weight`).
        match rng.next_u64() % 3 {
            0 => p.x[word] |= bit,
            1 => p.z[word] |= bit,
            _ => {
                p.x[word] |= bit;
                p.z[word] |= bit;
            }
        }
    }
    p
}

/// As [`rand_sum_unmasked`], but with low-weight keys from
/// [`low_weight_pauli`]. Collisions are far more likely here, so the realized
/// length can be noticeably below `n`.
pub fn low_weight_sum<const W: usize>(
    n: usize,
    num_qubits: usize,
    weight: usize,
    seed: u64,
) -> PauliSum<W> {
    let mut rng = Xs64::new(seed);
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n);
    for _ in 0..n {
        let p = low_weight_pauli::<W>(&mut rng, num_qubits, weight);
        let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        let im = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        acc.add_term(p, Phase::ONE, Complex64::new(re, im));
    }
    acc.finalize()
}

/// [`rand_sum`]'s keys with only four distinct coefficient magnitudes, so any
/// cut through the sum lands inside a tie group spanning a quarter of it.
///
/// Not a contrived case: a symmetric Hamiltonian on a periodic lattice
/// produces many terms related by lattice symmetry with *exactly* equal
/// coefficients, which is why the 2D Ising example hits it — and why `TopN`
/// has a tie rule at all (ARCHITECTURE.md §Truncation).
pub fn tie_heavy_sum<const W: usize>(n: usize, num_qubits: usize, seed: u64) -> PauliSum<W> {
    let base = rand_sum::<W>(n, num_qubits, seed);
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n);
    for (i, (x, z, _)) in base.iter().enumerate() {
        let mag = [1.0f64, 0.5, 0.25, 0.125][i % 4];
        acc.add_term(
            PauliString::<W> { x: *x, z: *z },
            Phase::ONE,
            Complex64::new(mag, 0.0),
        );
    }
    acc.finalize()
}

/// [`tie_heavy_sum`] over [`rand_pauli`] keys instead of [`rand_sum`] ones —
/// the benchmark variant, whose seeds are pinned by the criterion baselines.
///
/// Same tie structure, different key stream (and no coefficient draws at all,
/// so it is not merely a re-magnituded [`rand_sum_unmasked`]).
pub fn tie_heavy_sum_unmasked<const W: usize>(
    n: usize,
    num_qubits: usize,
    seed: u64,
) -> PauliSum<W> {
    let mut rng = Xs64::new(seed);
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n);
    for i in 0..n {
        let p = rand_pauli::<W>(&mut rng);
        let mag = [1.0f64, 0.5, 0.25, 0.125][i % 4];
        acc.add_term(p, Phase::ONE, Complex64::new(mag, 0.0));
    }
    acc.finalize()
}

/// Output-buffer columns plus a zeroed cursor, sized for `n` rows.
///
/// Returned as a tuple rather than an [`OutputBuffer`] because the buffer
/// borrows its columns: the caller has to own them, then build the borrow in a
/// narrower scope.
#[allow(clippy::type_complexity)]
pub fn alloc_bufs<const W: usize>(
    n: usize,
) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>, usize) {
    (
        vec![[0u64; W]; n],
        vec![[0u64; W]; n],
        vec![ZERO; n],
        0usize,
    )
}

/// Complex numbers equal to within `tol`.
pub fn approx_eq(a: Complex64, b: Complex64, tol: f64) -> bool {
    (a - b).norm() <= tol
}

/// One channel application, normalized the way the merge phase would leave it:
/// exact-ish zeros dropped and rows sorted by key.
///
/// Use this to compare two channels' *mathematical* action. When the emission
/// order or the presence of a zero row is itself the thing under test, use
/// [`raw_outputs`] instead.
pub fn outputs<const W: usize, C: Channel<W> + ?Sized>(
    ch: &C,
    adjoint: bool,
    p: PauliString<W>,
    coeff: Complex64,
) -> Vec<([u64; W], [u64; W], Complex64)> {
    let mut v = raw_outputs(ch, adjoint, p, coeff)
        .into_iter()
        .map(|(q, c)| (q.x, q.z, c))
        .collect::<Vec<_>>();
    v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    v.retain(|t| t.2.norm() > 1e-15);
    v
}

/// One channel application, exactly as emitted: buffer order, zeros included.
pub fn raw_outputs<const W: usize, C: Channel<W> + ?Sized>(
    ch: &C,
    adjoint: bool,
    p: PauliString<W>,
    coeff: Complex64,
) -> Vec<(PauliString<W>, Complex64)> {
    let f = ch.max_fanout().max(1);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<W>(f);
    {
        let mut out = OutputBuffer::<W> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        if adjoint {
            ch.apply_adjoint(&p.x, &p.z, coeff, &mut out);
        } else {
            ch.apply(&p.x, &p.z, coeff, &mut out);
        }
    }
    (0..len)
        .map(|i| (PauliString::<W> { x: bx[i], z: bz[i] }, bc[i]))
        .collect()
}

/// `(x, z, coeff)` triples sorted by the `(x, z)` key.
///
/// Keys are globally unique (the `PauliSum` invariant forbids duplicates), so
/// this is a canonical, storage-order-independent view: two sums with the same
/// terms produce the same triples regardless of which order their backing
/// engine happened to store them in.
pub fn canonical_triples<const W: usize>(s: &PauliSum<W>) -> Vec<([u64; W], [u64; W], Complex64)> {
    let mut v: Vec<([u64; W], [u64; W], Complex64)> =
        s.iter().map(|(x, z, c)| (*x, *z, c)).collect();
    v.sort_unstable_by_key(|&(x, z, _)| (x, z));
    v
}

/// Same keys, same coefficients bitwise (`Complex64` `==`) — order-agnostic.
///
/// Only appropriate between two computations that sum equal keys in the same
/// order. Anything compared against [`naive_apply_layer`] wants
/// [`assert_terms_close`] instead.
pub fn assert_same_terms<const W: usize>(got: &PauliSum<W>, want: &PauliSum<W>, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: term count");
    let got = canonical_triples(got);
    let want = canonical_triples(want);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!((g.0, g.1), (w.0, w.1), "{what}: term {i} key mismatch");
        assert_eq!(
            g.2, w.2,
            "{what}: term {i} key {:?}/{:?} coeff {} vs {} (not bitwise equal)",
            g.0, g.1, g.2, w.2,
        );
    }
}

/// Same keys; coefficients within `tol`, because two implementations can sum
/// duplicate keys in different orders and floating-point addition is not
/// associative. This is the correctness bar per the crate's determinism
/// policy.
pub fn assert_terms_close<const W: usize>(
    got: &PauliSum<W>,
    want: &PauliSum<W>,
    tol: f64,
    what: &str,
) {
    assert_eq!(got.len(), want.len(), "{what}: term count");
    let got = canonical_triples(got);
    let want = canonical_triples(want);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!((g.0, g.1), (w.0, w.1), "{what}: term {i} key mismatch");
        let d = (g.2 - w.2).norm();
        assert!(
            d < tol,
            "{what}: term {i} key {:?}/{:?} coeff {} vs {} (delta {d:e})",
            g.0,
            g.1,
            g.2,
            w.2,
        );
    }
}

/// The `sqrt(SWAP)` 4×4 unitary on `(a, b)`, as a matrix for
/// [`GeneralUnitary2Q::from_matrix`](crate::channel::GeneralUnitary2Q::from_matrix).
///
/// The canonical **sparse but wide** two-qubit fixture: its delta set spans
/// more than one bucket bit, yet its PTM is far from dense (steady-state
/// fanout 3.65 against a dense PTM's 14.94), so it exercises multi-delta
/// behaviour without the all-sixteen-entries cost of
/// [`haar_su4_matrix`].
pub fn sqrt_swap_matrix() -> [[Complex64; 4]; 4] {
    let h = Complex64::new(0.5, 0.5);
    let hc = Complex64::new(0.5, -0.5);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    [
        [one, zero, zero, zero],
        [zero, h, hc, zero],
        [zero, hc, h, zero],
        [zero, zero, zero, one],
    ]
}

/// One draw of Haar-random SU(4), as a 4×4 unitary in the computational basis.
///
/// The entries come from `examples/common/circuits.py::haar_su4` (Mezzadri
/// phase-fixed QR of a complex Ginibre matrix, then divided by `det^(1/4)`)
/// under `numpy.random.default_rng(0xC0FFEE)`, transcribed via Python `repr`
/// (shortest round-tripping `f64` literals) — i.e. one draw of exactly the
/// distribution `benchmarks/python/bench_jl_performance.py::su4_gates` and
/// benchmark E's `random_su4_staircase` sample. Unitary to 2.5e-16;
/// `GeneralUnitary2Q::from_matrix` does not check, and a non-unitary matrix
/// would silently give a non-physical PTM.
///
/// Shared because it is the canonical **dense-PTM** fixture: a generic SU(4)
/// gives all sixteen local delta entries a nonzero amplitude, which is what
/// makes the sort dominate the layer and what
/// `research/notes/2026-09-01-bucket-cliff.md` is about. `sqrt(SWAP)` and the
/// Cliffords are not substitutes — their PTMs are sparse (steady-state fanout
/// 3.65 and 1.0 against a dense PTM's 14.94).
pub fn haar_su4_matrix() -> [[Complex64; 4]; 4] {
    [
        [
            Complex64::new(0.44535882417102446, 0.1243885298575445),
            Complex64::new(-0.09402453034947537, -0.14670085591185988),
            Complex64::new(0.7459177705812382, -0.3801992439705379),
            Complex64::new(0.052557524520682804, 0.22828530169893588),
        ],
        [
            Complex64::new(-0.04863200501298571, -0.40347772310563557),
            Complex64::new(0.7069517563162028, 0.008408200837924597),
            Complex64::new(0.26012555224671347, -0.12053357328338017),
            Complex64::new(-0.3528728311960538, -0.3581567969892209),
        ],
        [
            Complex64::new(-0.35880447773297086, 0.11743595956162649),
            Complex64::new(-0.3097428484619983, 0.594366207605036),
            Complex64::new(0.1610278707748687, -0.25258937630123157),
            Complex64::new(-0.5597163461470217, 0.07240555858329784),
        ],
        [
            Complex64::new(-0.578592686173378, -0.3791072567045837),
            Complex64::new(0.05738813758483608, 0.13145928539206422),
            Complex64::new(0.3453330441780492, 0.08874282848443517),
            Complex64::new(0.5033919610984813, 0.34698642893070086),
        ],
    ]
}

/// GF(2) rank of a set of bucket indices, by Gaussian elimination.
///
/// One pivot slot per bit position, so each vector is either absorbed by an
/// existing pivot or becomes a new one. (The naive "reduce against a `Vec` of
/// vectors" version overcounts unless that `Vec` is kept *reduced* — the same
/// subtlety `engine::coset::Gf2Span::new` handles by back-substituting.)
///
/// Used to reason about the **coset dimension** the engine gets for a layer:
/// `Gf2Span::r()` is exactly this rank applied to the prepared channel's
/// `bucket_deltas()`, and it is the quantity the dense-PTM sort's cost turns
/// on (`research/notes/2026-09-01-bucket-cliff.md`).
pub fn gf2_rank(vs: &[u32]) -> usize {
    let mut pivot = [0u32; 32];
    let mut r = 0usize;
    for &v in vs {
        let mut v = v;
        while v != 0 {
            let hb = (31 - v.leading_zeros()) as usize;
            if pivot[hb] == 0 {
                pivot[hb] = v;
                r += 1;
                break;
            }
            v ^= pivot[hb];
        }
    }
    r
}

/// Rank of `h` restricted to the key-delta space of a support.
///
/// A channel supported on `qubits` can only change those qubits' `x` and `z`
/// bits, so its key-delta set lies in `span{X_q, Z_q : q ∈ qubits}` — dimension
/// `2·|qubits|`. Its *bucket*-delta span is the image of that space under `h`,
/// and this returns that image's dimension. Full rank (`2·|qubits|`) means `h`
/// separates every local delta; anything less means two distinct local deltas
/// share one bucket delta, which is the rank deficiency the note above
/// diagnoses.
pub fn support_delta_rank<const W: usize>(h: &crate::bucket::Gf2Hash<W>, qubits: &[u32]) -> usize {
    let mut imgs: Vec<u32> = Vec::with_capacity(2 * qubits.len());
    for &q in qubits {
        imgs.push(h.bucket_of_pauli(&PauliString::<W>::x(q)));
        imgs.push(h.bucket_of_pauli(&PauliString::<W>::z(q)));
    }
    gf2_rank(&imgs)
}

/// The engine's differential channel net at `W = 1`: every built-in channel
/// class, on an 8-qubit key space.
///
/// One list, shared by `engine::bucketed`'s differential net and the
/// partitioned engine's, so the two cover exactly the same channels and a
/// channel added here is exercised by both. Supports are chosen to spread over
/// the key space (`h` at 3, two-qubit gates at 1 and 5, noise at 2) and the
/// list ends with the three shapes that are structurally distinct for the
/// gather: a fanout-2 non-Clifford, a sparse-but-wide 2Q PTM, a dense one, and
/// a rotation wider than `MAX_LOCAL_SUPPORT` (the `Prepared::Rotation` arm).
pub fn differential_channels_w1() -> Vec<(&'static str, Box<dyn Channel<1>>)> {
    use crate::channel::clifford::{Clifford1Q, Clifford2Q};
    use crate::channel::identity::IdentityChannel;
    use crate::channel::noise::{AmplitudeDamping, Dephasing, Depolarizing};
    use crate::channel::rotation::PauliRotation;
    use crate::channel::{GeneralUnitary1Q, GeneralUnitary2Q};

    vec![
        ("identity", Box::new(IdentityChannel::new())),
        ("h", Box::new(Clifford1Q::h(3))),
        ("s", Box::new(Clifford1Q::s(3))),
        ("x", Box::new(Clifford1Q::x(3))),
        ("y", Box::new(Clifford1Q::y(3))),
        ("z", Box::new(Clifford1Q::z(3))),
        ("cnot", Box::new(Clifford2Q::cnot(1, 5))),
        ("cz", Box::new(Clifford2Q::cz(1, 5))),
        ("swap", Box::new(Clifford2Q::swap(1, 5))),
        (
            "depolarizing",
            Box::new(Depolarizing {
                support: [2],
                p: 0.07,
            }),
        ),
        (
            "dephasing",
            Box::new(Dephasing {
                support: [2],
                p: 0.07,
            }),
        ),
        (
            "amp_damping",
            Box::new(AmplitudeDamping {
                support: [2],
                gamma: 0.3,
            }),
        ),
        (
            "rot_z",
            Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.41)),
        ),
        (
            "rot_zz",
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<1>::z(1);
                    g.mul_assign(&PauliString::<1>::z(6));
                    g
                },
                0.41,
            )),
        ),
        (
            // General unitaries: a non-Clifford T gate (fanout 2) and a
            // dense 2Q unitary (fanout up to 16), both as local PTMs.
            "t_gate",
            Box::new(GeneralUnitary1Q::from_matrix(
                2,
                [
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [
                        Complex64::new(0.0, 0.0),
                        Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
                    ],
                ],
            )),
        ),
        (
            // sqrt(SWAP): a wide delta set with a sparse PTM.
            "general_2q",
            Box::new(GeneralUnitary2Q::from_matrix(1, 5, sqrt_swap_matrix())),
        ),
        (
            // A *dense* SU(4): every PTM entry nonzero, so all 16 bucket
            // deltas are realized (fanout ~15) — the shape the per-run sort
            // kernel is selected on (see `merge::sort_rows_radix_with_scratch`).
            "haar_su4",
            Box::new(GeneralUnitary2Q::from_matrix(1, 5, haar_su4_matrix())),
        ),
        (
            // Weight 4 > MAX_LOCAL_SUPPORT: exercises the Rotation variant.
            "rot_wide",
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<1>::z(0);
                    for q in [2u32, 4, 6] {
                        g.mul_assign(&PauliString::<1>::x(q));
                    }
                    g
                },
                0.41,
            )),
        ),
    ]
}

/// The engine's differential channel net at `W = 2`: the other occupancy
/// regime — 128 qubits, wide keys, supports straddling the 64-bit word
/// boundary.
///
/// Shared by `engine::bucketed`'s differential net and the partitioned
/// engine's, as [`differential_channels_w1`] is.
pub fn differential_channels_w2() -> Vec<(&'static str, Box<dyn Channel<2>>)> {
    use crate::channel::clifford::{Clifford1Q, Clifford2Q};
    use crate::channel::noise::AmplitudeDamping;
    use crate::channel::rotation::PauliRotation;
    use crate::channel::GeneralUnitary2Q;

    vec![
        ("h@70", Box::new(Clifford1Q::h(70))),
        ("s@64", Box::new(Clifford1Q::s(64))),
        ("cnot@60,70", Box::new(Clifford2Q::cnot(60, 70))),
        ("swap@0,127", Box::new(Clifford2Q::swap(0, 127))),
        (
            "amp_damping@70",
            Box::new(AmplitudeDamping {
                support: [70],
                gamma: 0.25,
            }),
        ),
        (
            "rot_y@70",
            Box::new(PauliRotation::new(PauliString::<2>::y(70), 0.33)),
        ),
        (
            "rot_zz_cross_word",
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<2>::z(9);
                    g.mul_assign(&PauliString::<2>::z(70));
                    g
                },
                0.33,
            )),
        ),
        // Dense SU(4), support straddling the word boundary — the dense-PTM
        // run shape at `W = 2`.
        (
            "haar_su4_cross_word",
            Box::new(GeneralUnitary2Q::from_matrix(60, 70, haar_su4_matrix())),
        ),
    ]
}

// ---- partitioned-engine fixtures -------------------------------------------
//
// The partitioned nets (`tests/propagate_partitioned.rs`,
// `tests/propagate_distributed.rs`, `tests/mpi_ranks.rs`,
// `tests/partitioned_*.rs`, `tests/phase_timing.rs`) all need the same three
// things: a policy with no layer pass, a `ZZ` rotation, and a placement with no
// placement. They live here so a change to any of them is one edit.

/// Keep every term, with no layer finalization at all.
///
/// [`TruncationPolicy::finalizes_layer`]'s default is the conservative `true`,
/// which [`PartitionedTruncation`]'s default body rejects — a policy with no
/// layer pass has to say so, since the trait cannot know that skipping a
/// collective is safe.
///
/// [`PartitionedTruncation`]: crate::PartitionedTruncation
pub struct KeepAll;

impl<const W: usize> TruncationPolicy<W> for KeepAll {
    fn finalizes_layer(&self) -> bool {
        false
    }
}

impl<const W: usize> crate::PartitionedTruncation<W> for KeepAll {}

/// A weight-2 `ZZ` rotation — the TFIM bond term, and the smallest layer whose
/// generator can cross a partition boundary.
pub fn zz_rotation<const W: usize>(
    q0: u32,
    q1: u32,
    theta: f64,
) -> crate::channel::rotation::PauliRotation<W> {
    let mut gen = PauliString::<W> {
        x: [0u64; W],
        z: [0u64; W],
    };
    for q in [q0, q1] {
        gen.z[q as usize / 64] |= 1u64 << (q % 64);
    }
    crate::channel::rotation::PauliRotation::new(gen, theta)
}

/// One TFIM Trotter step: `num_qubits` periodic `ZZ` bond rotations, then that
/// many transverse-field `X` rotations, all at angle `2 · theta`.
///
/// `2 · num_qubits` layers, enough that the term count — and with it the bucket
/// count the group agrees on every layer — grows across the run.
pub fn trotter_circuit<const W: usize>(num_qubits: usize, theta: f64) -> crate::Circuit<W> {
    let mut circuit = crate::Circuit::<W>::new(num_qubits);
    for q in 0..num_qubits {
        let q1 = ((q + 1) % num_qubits) as u32;
        circuit.push(zz_rotation::<W>(q as u32, q1, 2.0 * theta));
    }
    for q in 0..num_qubits {
        circuit.push(crate::channel::rotation::PauliRotation::new(
            PauliString::<W>::x(q as u32),
            2.0 * theta,
        ));
    }
    circuit
}

/// A partitioned placement with no placement: `partitions` unpinned pools of
/// `threads` workers each, drawing partition rows from `row_seed`.
///
/// Every partitioned test uses this rather than `Placement::Auto`, so the suite
/// runs on a one-node box or a `taskset`ed CI container; placement itself is
/// covered by `engine::partitioned::topology`'s own tests.
pub fn unpinned_partitions(
    partitions: usize,
    threads: usize,
    row_seed: u64,
) -> crate::engine::partitioned::PartitionConfig {
    crate::engine::partitioned::PartitionConfig {
        placement: crate::engine::partitioned::Placement::Unpinned {
            partitions,
            threads_per_partition: Some(threads),
        },
        bind_memory: false,
        partition_row_seed: Some(row_seed),
    }
}
