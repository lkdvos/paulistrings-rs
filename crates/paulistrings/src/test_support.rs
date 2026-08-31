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
//! against: it replaced the retained v0.1 sort-merge pipeline in that role, and
//! deliberately shares no code with the thing it checks — one `Channel::apply`
//! call per input term, a hashmap accumulation, and a sort.

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
