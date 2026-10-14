//! Seeded random sums: the [`Xs64`] generator and every sum builder drawn from it.

use num_complex::Complex64;

#[cfg(feature = "cuda")]
use super::ZERO;
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::PauliSum;

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
        acc.add_term(p, Complex64::new(re, im));
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
        acc.add_term(p, Complex64::new(re, 0.0));
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
        acc.add_term(p, Complex64::new(re, im));
    }
    acc.finalize()
}

/// A Pauli string of Hamming weight `weight` over `num_qubits` qubits.
///
/// The realistic occupancy regime (physical Hamiltonians are low-weight); [`rand_pauli`] is the dense opposite extreme, and both get benched since bucketing behaves very differently on the two.
/// Index-bounded by construction (`q` drawn `mod num_qubits`), so unlike [`rand_pauli`] it needs no masking pass.
fn low_weight_pauli<const W: usize>(
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

/// As [`rand_sum_unmasked`], but with low-weight keys from `low_weight_pauli`; collisions are far more likely, so realized length can be noticeably below `n`.
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
        acc.add_term(p, Complex64::new(re, im));
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
        acc.add_term(PauliString::<W> { x: *x, z: *z }, Complex64::new(mag, 0.0));
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
        acc.add_term(p, Complex64::new(mag, 0.0));
    }
    acc.finalize()
}

/// `sum` with each term scaled by `2^-k`, `k` drawn per term in `0..octaves`: a cut through the low octaves drops little weight, as [`assert_runs_agree`](super::assert_runs_agree) needs.
pub fn spread_magnitudes<const W: usize>(
    sum: &PauliSum<W>,
    octaves: u32,
    seed: u64,
) -> PauliSum<W> {
    let mut rng = Xs64::new(seed);
    let mut acc = BuildAccumulator::<W>::with_capacity(sum.num_qubits(), sum.len());
    for (x, z, c) in sum.iter() {
        let k = (rng.next_u64() % u64::from(octaves)) as i32;
        acc.add_term(PauliString::<W> { x: *x, z: *z }, c * 2f64.powi(-k));
    }
    acc.finalize()
}

/// `sum` with every `every`-th term's coefficient replaced by an exact `0 + 0i`, which no accumulator would keep: the input a layer must drop rather than carry.
#[cfg(feature = "cuda")]
pub fn with_zero_coefficients<const W: usize>(sum: &PauliSum<W>, every: usize) -> PauliSum<W> {
    let mut i = 0usize;
    let buckets = (0..sum.num_buckets())
        .map(|b| {
            let (x, z, c) = sum.bucket(b);
            let mut columns = crate::pauli_sum::storage::BucketColumns::<W>::default();
            columns.x.extend_from_slice(x);
            columns.z.extend_from_slice(z);
            for &c in c {
                columns
                    .coeff
                    .push(if i.is_multiple_of(every) { ZERO } else { c });
                i += 1;
            }
            columns
        })
        .collect();
    PauliSum::from_buckets(buckets, sum.hash().clone(), sum.num_qubits())
}

/// A sum on which `cancellation_channel` produces an exact `±0` coefficient: `-0.5·I + 1.0·Z₀` under amplitude damping at `γ = 0.5` sums `-0.5 + 0.5·1.0` onto the identity key.
/// The other terms keep the layer from being trivial.
pub fn cancellation_sum<const W: usize>(num_qubits: usize) -> PauliSum<W> {
    let mut acc = BuildAccumulator::<W>::new(num_qubits);
    let identity = PauliString::<W> {
        x: [0u64; W],
        z: [0u64; W],
    };
    acc.add_term(identity, Complex64::new(-0.5, 0.0));
    acc.add_term(PauliString::<W>::z(0), Complex64::new(1.0, 0.0));
    acc.add_term(PauliString::<W>::x(0), Complex64::new(0.25, 0.25));
    acc.add_term(PauliString::<W>::y(1), Complex64::new(0.75, -0.5));
    acc.add_term(PauliString::<W>::z(2), Complex64::new(0.125, 0.0));
    acc.finalize()
}

/// `Z0` on eight qubits with coefficient 1.
pub fn z0_sum() -> PauliSum<1> {
    let mut acc = BuildAccumulator::<1>::new(8);
    acc.add_term(PauliString::<1>::z(0), Complex64::new(1.0, 0.0));
    acc.finalize()
}
