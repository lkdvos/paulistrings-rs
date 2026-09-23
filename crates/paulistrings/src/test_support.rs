//! Shared fixtures for the crate's own tests, benches and examples.
//!
//! Compiled only under `cfg(test)` or the `test-utils` feature (the crate's self-dev-dependency); `#[doc(hidden)]`, not public API.
//! [`naive_apply_layer`] is the oracle the bucketed engine is tested against — one `Channel::apply` call per term, a hashmap accumulation, and a sort — and shares no code with the thing it checks.

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

/// Returns early from the calling test when no CUDA device is available, so a device test is
/// never `#[ignore]`d — it just does nothing on a box without a GPU.
#[cfg(feature = "cuda")]
#[macro_export]
macro_rules! require_cuda {
    () => {
        if !$crate::engine::gpu::cuda_available() {
            return;
        }
    };
}

/// Apply one channel layer the obvious way, as a differential oracle.
///
/// For every input term: [`Channel::apply`] (or `apply_adjoint` when `adjoint`) into a `max_fanout`-sized [`OutputBuffer`], accumulate into a hashmap keyed by `(x, z)`, filter the summed coefficients through [`TruncationPolicy::keep_term`], drop exact zeros, sort by key, rebuild a [`PauliSum`].
/// Deliberately naive: no bucketing, no coset structure, no parallelism, no code shared with `engine::bucketed` beyond the `Channel` trait itself.
/// Hashmap summation order is unspecified, so compare with [`assert_terms_close`], never bitwise.
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

/// `n` random dense terms on `num_qubits` qubits, deduplicated by [`BuildAccumulator`] (realized length can be below `n` at small qubit counts).
/// Draw order per term is `(x[0], z[0], x[1], z[1], …, re, im)` — keep it stable, every fixture seed in the suite encodes it.
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

/// As [`rand_sum`], but coefficients are real — one draw per term instead of two.
/// A separate function rather than a flag: the draw order differs (`…, re` vs `…, re, im`), so the two streams disagree from the same seed. Do not unify them.
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

/// A dense random key with no masking: all `W` `x` words first, then all `W` `z` words.
/// Valid only when `num_qubits` is a multiple of 64 (every bit set must be a live qubit or `BuildAccumulator` rejects the term); word-major draw order disagrees with [`rand_sum`]'s per-word interleave from `W = 2` up.
pub fn rand_pauli<const W: usize>(rng: &mut Xs64) -> PauliString<W> {
    PauliString::<W> {
        x: rng.next_array::<W>(),
        z: rng.next_array::<W>(),
    }
}

/// `n` dense random terms built from [`rand_pauli`] — the benchmark input recipe, seeds pinned by the committed criterion baselines. Distinct stream from [`rand_sum`]; see [`rand_pauli`] for why.
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
/// The realistic occupancy regime (physical Hamiltonians are low-weight); [`rand_pauli`] is the dense opposite extreme, and both get benched since bucketing behaves very differently on the two.
/// Index-bounded by construction (`q` drawn `mod num_qubits`), so unlike [`rand_pauli`] it needs no masking pass.
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

/// As [`rand_sum_unmasked`], but with low-weight keys from [`low_weight_pauli`]; collisions are far more likely, so realized length can be noticeably below `n`.
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

/// [`rand_sum`]'s keys with only four distinct coefficient magnitudes, so any cut through the sum lands inside a tie group spanning a quarter of it.
/// Not contrived: lattice-symmetric Hamiltonians produce exactly-equal-coefficient terms this way, which is why `TopN` has a tie rule at all (ARCHITECTURE.md §Truncation).
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

/// [`tie_heavy_sum`] over [`rand_pauli`] keys instead of [`rand_sum`] ones — the benchmark variant, seeds pinned by the criterion baselines. Same tie structure, different key stream, no coefficient draws.
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
/// A tuple, not an [`OutputBuffer`], because the buffer borrows its columns — the caller owns them, then builds the borrow in a narrower scope.
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

/// One channel application, normalized the way the merge phase would leave it: exact-ish zeros dropped, rows sorted by key.
/// Use this to compare two channels' mathematical action; when emission order or a zero row is itself under test, use [`raw_outputs`] instead.
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
/// Keys are globally unique (the `PauliSum` invariant forbids duplicates), so this is a canonical, storage-order-independent view.
pub fn canonical_triples<const W: usize>(s: &PauliSum<W>) -> Vec<([u64; W], [u64; W], Complex64)> {
    let mut v: Vec<([u64; W], [u64; W], Complex64)> =
        s.iter().map(|(x, z, c)| (*x, *z, c)).collect();
    v.sort_unstable_by_key(|&(x, z, _)| (x, z));
    v
}

/// Same keys, same coefficients bitwise (`Complex64` `==`) — order-agnostic.
/// Only appropriate between two computations that sum equal keys in the same order; anything compared against [`naive_apply_layer`] wants [`assert_terms_close`] instead.
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

/// Same keys; coefficients within `tol`, since two implementations can sum duplicate keys in different orders and floating-point addition is not associative — the correctness bar per the crate's determinism policy.
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

/// The `sqrt(SWAP)` 4×4 unitary on `(a, b)`, as a matrix for [`GeneralUnitary2Q::from_matrix`](crate::channel::GeneralUnitary2Q::from_matrix).
/// The canonical sparse-but-wide two-qubit fixture: its delta set spans more than one bucket bit, yet its PTM is far from dense (steady-state fanout 3.65 vs. a dense PTM's 14.94), exercising multi-delta behaviour without [`haar_su4_matrix`]'s all-sixteen-entries cost.
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
/// Transcribed from `examples/common/circuits.py::haar_su4` under `numpy.random.default_rng(0xC0FFEE)` — the same distribution `bench_jl_performance.py::su4_gates` and benchmark E's `random_su4_staircase` draw from. Unitary to 2.5e-16; `GeneralUnitary2Q::from_matrix` does not check.
///
/// The canonical dense-PTM fixture: a generic SU(4) gives all sixteen local delta entries a nonzero amplitude, which is what makes the sort dominate the layer (research/FINDINGS.md). `sqrt(SWAP)` and the Cliffords are not substitutes — their PTMs are sparse.
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
/// One pivot slot per bit position, so each vector is either absorbed by an existing pivot or becomes a new one — the same reduced-basis subtlety `engine::coset::Gf2Span::new` handles by back-substituting.
/// `Gf2Span::r()` is exactly this rank applied to a prepared channel's `bucket_deltas()`, the coset dimension the engine gets for a layer (research/FINDINGS.md).
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
/// A channel on `qubits` can only change those qubits' `x`/`z` bits, so its key-delta set lies in `span{X_q, Z_q : q ∈ qubits}` (dimension `2·|qubits|`); this returns the dimension of that space's image under `h`.
/// Full rank means `h` separates every local delta; anything less means two distinct local deltas share one bucket delta.
pub fn support_delta_rank<const W: usize>(h: &crate::bucket::Gf2Hash<W>, qubits: &[u32]) -> usize {
    let mut imgs: Vec<u32> = Vec::with_capacity(2 * qubits.len());
    for &q in qubits {
        imgs.push(h.bucket_of_pauli(&PauliString::<W>::x(q)));
        imgs.push(h.bucket_of_pauli(&PauliString::<W>::z(q)));
    }
    gf2_rank(&imgs)
}

/// The engine's differential channel net at `W = 1`: every built-in channel class, on an 8-qubit key space.
///
/// Shared by `engine::bucketed`'s differential net and the partitioned engine's, so both exercise exactly the same channels.
/// Ends with three structurally distinct gather shapes: a fanout-2 non-Clifford, a sparse-but-wide 2Q PTM, a dense one, and a rotation wider than `MAX_LOCAL_SUPPORT` (the `Prepared::Rotation` arm).
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
            // A non-Clifford T gate (fanout 2), as a local PTM.
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
            // A dense SU(4): every PTM entry nonzero (fanout ~15) — the shape the per-run sort kernel is selected on (`merge::sort_rows_radix_with_scratch`).
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

/// The engine's differential channel net at `W = 2`: the other occupancy regime — 128 qubits, wide keys, supports straddling the 64-bit word boundary.
/// Shared by `engine::bucketed`'s differential net and the partitioned engine's, as [`differential_channels_w1`] is.
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
        // Dense SU(4), support straddling the word boundary — the dense-PTM run shape at `W = 2`.
        (
            "haar_su4_cross_word",
            Box::new(GeneralUnitary2Q::from_matrix(60, 70, haar_su4_matrix())),
        ),
    ]
}

// ---- partitioned-engine fixtures -------------------------------------------
//
// Shared by the partitioned/distributed/MPI test nets, which all need the same three things: a policy with no layer pass, a `ZZ` rotation, and a placement with no placement.

/// Keep every term, with no layer finalization at all.
///
/// [`TruncationPolicy::finalizes_layer`] defaults to `true`, which [`PartitionedTruncation`]'s default body rejects — a policy with no layer pass has to say so explicitly.
///
/// [`PartitionedTruncation`]: crate::PartitionedTruncation
pub struct KeepAll;

impl<const W: usize> TruncationPolicy<W> for KeepAll {
    fn finalizes_layer(&self) -> bool {
        false
    }

    fn device_policy(&self) -> Option<crate::truncation::BuiltinTruncation> {
        Some(crate::truncation::BuiltinTruncation::Keep)
    }
}

impl<const W: usize> crate::PartitionedTruncation<W> for KeepAll {}

fn set_x<const W: usize>(p: &mut PauliString<W>, q: u32) {
    p.x[q as usize / 64] |= 1u64 << (q % 64);
}

fn set_z<const W: usize>(p: &mut PauliString<W>, q: u32) {
    p.z[q as usize / 64] |= 1u64 << (q % 64);
}

/// A seeded circuit drawing from every built-in channel class.
///
/// `dense` adds the wide-fanout classes (a dense 1Q PTM, sqrt-SWAP, a Haar SU(4) block); without it every layer has fanout at most 2, which is what keeps an untruncated run bounded.
/// Kind 8 is a weight-4 rotation, so `prepare` takes the `Prepared::Rotation` arm.
pub fn random_circuit<const W: usize>(
    num_qubits: usize,
    layers: usize,
    seed: u64,
    dense: bool,
) -> crate::Circuit<W> {
    use crate::channel::clifford::{Clifford1Q, Clifford2Q};
    use crate::channel::noise::{
        AmplitudeDamping, Dephasing, Depolarizing, Depolarizing2Q, PauliChannel,
    };
    use crate::channel::rotation::PauliRotation;
    use crate::channel::{GeneralUnitary1Q, GeneralUnitary2Q};

    let mut rng = Xs64::new(seed);
    let mut circuit = crate::Circuit::<W>::new(num_qubits);
    let kinds: u64 = if dense { 17 } else { 14 };
    let n = num_qubits as u64;
    for _ in 0..layers {
        let q0 = (rng.next_u64() % n) as u32;
        let q1 = ((q0 as u64 + 1 + rng.next_u64() % (n - 1)) % n) as u32;
        let wrap = |q: u32, d: u32| (q + d) % num_qubits as u32;
        match rng.next_u64() % kinds {
            0 => circuit.push(Clifford1Q::h(q0)),
            1 => circuit.push(Clifford1Q::s(q0)),
            2 => circuit.push(Clifford1Q::y(q0)),
            3 => circuit.push(Clifford2Q::cnot(q0, q1)),
            4 => circuit.push(Clifford2Q::cz(q0, q1)),
            5 => circuit.push(Clifford2Q::swap(q0, q1)),
            6 => circuit.push(PauliRotation::new(PauliString::<W>::z(q0), 0.37)),
            7 => circuit.push(zz_rotation::<W>(q0, q1, 0.21)),
            8 => {
                let mut gen = PauliString::<W> {
                    x: [0u64; W],
                    z: [0u64; W],
                };
                set_x(&mut gen, q0);
                set_z(&mut gen, wrap(q0, 1));
                set_x(&mut gen, wrap(q0, 2));
                set_z(&mut gen, wrap(q0, 3));
                circuit.push(PauliRotation::new(gen, 0.29));
            }
            9 => circuit.push(Depolarizing {
                support: [q0],
                p: 0.05,
            }),
            10 => circuit.push(Dephasing {
                support: [q0],
                p: 0.11,
            }),
            11 => circuit.push(PauliChannel {
                support: [q0],
                px: 0.03,
                py: 0.04,
                pz: 0.05,
            }),
            12 => circuit.push(Depolarizing2Q {
                support: [q0, q1],
                p: 0.07,
            }),
            13 => circuit.push(AmplitudeDamping {
                support: [q0],
                gamma: 0.09,
            }),
            14 => circuit.push(GeneralUnitary1Q::from_matrix(
                q0,
                [
                    [Complex64::new(0.6, 0.0), Complex64::new(0.0, -0.8)],
                    [Complex64::new(0.0, -0.8), Complex64::new(0.6, 0.0)],
                ],
            )),
            15 => circuit.push(GeneralUnitary2Q::from_matrix(q0, q1, sqrt_swap_matrix())),
            16 => circuit.push(GeneralUnitary2Q::from_matrix(q0, q1, haar_su4_matrix())),
            _ => unreachable!(),
        }
    }
    circuit
}

/// A sum on which `cancellation_channel` produces an exact `±0` coefficient: `-0.5·I + 1.0·Z₀` under amplitude damping at `γ = 0.5` sums `-0.5 + 0.5·1.0` onto the identity key.
/// The other terms keep the layer from being trivial.
pub fn cancellation_sum<const W: usize>(num_qubits: usize) -> PauliSum<W> {
    let mut acc = BuildAccumulator::<W>::new(num_qubits);
    let identity = PauliString::<W> {
        x: [0u64; W],
        z: [0u64; W],
    };
    acc.add_term(identity, Phase::ONE, Complex64::new(-0.5, 0.0));
    acc.add_term(PauliString::<W>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
    acc.add_term(
        PauliString::<W>::x(0),
        Phase::ONE,
        Complex64::new(0.25, 0.25),
    );
    acc.add_term(
        PauliString::<W>::y(1),
        Phase::ONE,
        Complex64::new(0.75, -0.5),
    );
    acc.add_term(
        PauliString::<W>::z(2),
        Phase::ONE,
        Complex64::new(0.125, 0.0),
    );
    acc.finalize()
}

/// A channel with no identity delta: every term moves to `v ⊕ x₀` with its coefficient unchanged.
/// Not physical; it exercises a prepared table whose entry 0 is not the identity.
pub struct ShiftX;

impl<const W: usize> Channel<W> for ShiftX {
    fn max_fanout(&self) -> usize {
        1
    }

    fn support(&self) -> [u64; W] {
        crate::channel::support_mask(&[0])
    }

    fn apply(&self, x: &[u64; W], z: &[u64; W], c: Complex64, out: &mut OutputBuffer<'_, W>) {
        let mut kx = *x;
        kx[0] ^= 1;
        out.push(kx, *z, c);
    }
}

/// The channel [`cancellation_sum`] is built for.
pub fn cancellation_channel() -> crate::channel::noise::AmplitudeDamping {
    crate::channel::noise::AmplitudeDamping {
        support: [0],
        gamma: 0.5,
    }
}

/// A weight-2 `ZZ` rotation — the TFIM bond term, the smallest layer whose generator can cross a partition boundary.
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

/// One TFIM Trotter step: `num_qubits` periodic `ZZ` bond rotations, then that many transverse-field `X` rotations, all at angle `2 · theta`.
/// `2 · num_qubits` layers, enough that the term count grows across the run.
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

/// A partitioned placement with no placement: `partitions` unpinned pools of `threads` workers each, drawing partition rows from `row_seed`.
/// Every partitioned test uses this rather than `Placement::Auto`, so the suite runs on a one-node box or a `taskset`ed CI container.
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

/// The 127-qubit heavy-hex coupling map (IBM Eagle r3), 144 undirected edges as `(lo, hi)` pairs in sorted order.
///
/// A verbatim copy of `examples/data/heavy_hex_127.edges`, kept here as a constant so Rust probes need no file I/O; `heavy_hex_127_edges_match_the_source_lattice` pins the transcription.
/// Degree histogram: 2 qubits of degree 1, 89 of degree 2, 36 of degree 3.
#[rustfmt::skip]
pub const HEAVY_HEX_127_EDGES: [(u32, u32); 144] = [
    (0, 1), (0, 14), (1, 2), (2, 3), (3, 4), (4, 5),
    (4, 15), (5, 6), (6, 7), (7, 8), (8, 9), (8, 16),
    (9, 10), (10, 11), (11, 12), (12, 13), (12, 17), (14, 18),
    (15, 22), (16, 26), (17, 30), (18, 19), (19, 20), (20, 21),
    (20, 33), (21, 22), (22, 23), (23, 24), (24, 25), (24, 34),
    (25, 26), (26, 27), (27, 28), (28, 29), (28, 35), (29, 30),
    (30, 31), (31, 32), (32, 36), (33, 39), (34, 43), (35, 47),
    (36, 51), (37, 38), (37, 52), (38, 39), (39, 40), (40, 41),
    (41, 42), (41, 53), (42, 43), (43, 44), (44, 45), (45, 46),
    (45, 54), (46, 47), (47, 48), (48, 49), (49, 50), (49, 55),
    (50, 51), (52, 56), (53, 60), (54, 64), (55, 68), (56, 57),
    (57, 58), (58, 59), (58, 71), (59, 60), (60, 61), (61, 62),
    (62, 63), (62, 72), (63, 64), (64, 65), (65, 66), (66, 67),
    (66, 73), (67, 68), (68, 69), (69, 70), (70, 74), (71, 77),
    (72, 81), (73, 85), (74, 89), (75, 76), (75, 90), (76, 77),
    (77, 78), (78, 79), (79, 80), (79, 91), (80, 81), (81, 82),
    (82, 83), (83, 84), (83, 92), (84, 85), (85, 86), (86, 87),
    (87, 88), (87, 93), (88, 89), (90, 94), (91, 98), (92, 102),
    (93, 106), (94, 95), (95, 96), (96, 97), (96, 109), (97, 98),
    (98, 99), (99, 100), (100, 101), (100, 110), (101, 102), (102, 103),
    (103, 104), (104, 105), (104, 111), (105, 106), (106, 107), (107, 108),
    (108, 112), (109, 114), (110, 118), (111, 122), (112, 126), (113, 114),
    (114, 115), (115, 116), (116, 117), (117, 118), (118, 119), (119, 120),
    (120, 121), (121, 122), (122, 123), (123, 124), (124, 125), (125, 126),
];

/// [`HEAVY_HEX_127_EDGES`] as a `Vec`, for callers that want to own it.
pub fn heavy_hex_127_edges() -> Vec<(u32, u32)> {
    HEAVY_HEX_127_EDGES.to_vec()
}

#[cfg(test)]
mod tests {
    use super::HEAVY_HEX_127_EDGES;

    /// Pins the transcribed copy: 144 undirected edges over qubits `0..126`, sorted and unique as `(lo, hi)`, degree histogram 2 × 1, 89 × 2, 36 × 3.
    #[test]
    fn heavy_hex_127_edges_match_the_source_lattice() {
        assert_eq!(HEAVY_HEX_127_EDGES.len(), 144);
        let mut degree = [0usize; 127];
        let mut prev = (0u32, 0u32);
        for (i, &(a, b)) in HEAVY_HEX_127_EDGES.iter().enumerate() {
            assert!(a < b, "edge {i} is not (lo, hi): ({a}, {b})");
            assert!(b < 127, "edge {i} names qubit {b} outside 0..126");
            if i > 0 {
                assert!(prev < (a, b), "edge {i} breaks the sorted-unique order");
            }
            prev = (a, b);
            degree[a as usize] += 1;
            degree[b as usize] += 1;
        }
        let mut histogram = [0usize; 4];
        for d in degree {
            histogram[d] += 1;
        }
        assert_eq!(histogram, [0, 2, 89, 36]);
    }
}
