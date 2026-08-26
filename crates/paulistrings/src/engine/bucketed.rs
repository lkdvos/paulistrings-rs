//! The bucketed layer engine. See v0.2 design doc §6.
//!
//! One layer, for each output bucket `β'` independently:
//!
//! 1. **Size** the scratch exactly, from the lengths of the input buckets this
//!    output bucket reads. No `n × MAX_FANOUT` over-allocation and no zero-fill.
//! 2. **Gather** — for each key delta `d`, in canonical order, scan input bucket
//!    `β' ^ h(d)` and emit `v ⊕ d` with amplitude `amp[support_bits(v)]`.
//! 3. **Sort** the scratch, reusing [`sort_phase`].
//! 4. **Merge** — segmented reduction with the zero-drop and `keep_term`,
//!    reusing [`merge_into`], writing straight into the destination bucket.
//!
//! Output buckets are write-disjoint (v0.2 §2.4), so steps 2-4 for different `β'`
//! never interact: the bucket loop is a plain `par_iter_mut` with no atomics, no
//! locks, and no reconciliation pass. That was the whole point of gathering
//! rather than scattering (§6.1).
//!
//! Phase B established correctness with this loop sequential; C.1 changed exactly
//! the two iterators and nothing else, which is why the differential and
//! determinism tests carried over unchanged.

use num_complex::Complex64;
use rayon::prelude::*;

use super::sort_merge::{merge_into, sort_phase};
use crate::bucket::sum::{BucketCols, BucketedSum};
use crate::channel::prepared::{LocalPtm, Prepared, RotationPrep};
use crate::pauli_string::PauliString;
use crate::phase::Phase;
use crate::truncation::TruncationPolicy;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Reusable per-layer gather scratch.
///
/// One instance per thread. Held by the caller rather than allocated per layer,
/// because the whole point of the bucketed layout is that a layer allocates
/// nothing after the first (v0.2 §4.2).
#[derive(Clone, Debug, Default)]
pub struct LayerScratch<const W: usize> {
    x: Vec<[u64; W]>,
    z: Vec<[u64; W]>,
    coeff: Vec<Complex64>,
}

impl<const W: usize> LayerScratch<W> {
    /// An empty scratch.
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            z: Vec::new(),
            coeff: Vec::new(),
        }
    }

    #[inline]
    fn reset(&mut self, cap: usize) {
        self.x.clear();
        self.z.clear();
        self.coeff.clear();
        if self.x.capacity() < cap {
            let extra = cap - self.x.capacity();
            self.x.reserve(extra);
            self.z.reserve(extra);
            self.coeff.reserve(extra);
        }
    }

    #[inline]
    fn push(&mut self, x: [u64; W], z: [u64; W], c: Complex64) {
        self.x.push(x);
        self.z.push(z);
        self.coeff.push(c);
    }

    #[inline]
    fn len(&self) -> usize {
        self.coeff.len()
    }
}

/// Apply one prepared channel to a bucketed sum.
///
/// `policy`'s `keep_term` is folded into the per-bucket merge, so it sees fully
/// **summed** coefficients — the same contract as the v0.1 engine.
/// `finalize_layer` is *not* called here; `propagate` owns that.
pub fn apply_layer_bucketed<const W: usize, T>(
    sum: &mut BucketedSum<W>,
    prep: &Prepared<W>,
    policy: &T,
    scratch: &mut LayerScratch<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    // Key-preserving channels (identity, depolarizing, dephasing, Pauli gates)
    // leave every key bitwise unchanged, so the output is already sorted and
    // duplicate-free. v0.1 paid a full O(n log n) sort to multiply each
    // coefficient by a scalar; here it is an in-place filter.
    if let Prepared::Local(ptm) = prep {
        if ptm.is_key_preserving() {
            rescale_in_place(sum, ptm, policy);
            return;
        }
    }

    let (input, mut output) = sum.begin_layer();

    // Iterating the *output* buckets is the parallel decomposition: each task owns
    // its destination and only reads `input`. `scratch` is used for the
    // single-threaded path; parallel tasks get their own via `for_each_init`.
    let input_ref = &input;
    if output.len() < MIN_BUCKETS_FOR_PARALLEL {
        for (out_b, dst) in output.iter_mut().enumerate() {
            fill_bucket::<W, T>(input_ref, out_b, dst, prep, policy, scratch);
        }
    } else {
        output.par_iter_mut().enumerate().for_each_init(
            LayerScratch::<W>::new,
            |local, (out_b, dst)| {
                fill_bucket::<W, T>(input_ref, out_b, dst, prep, policy, local);
            },
        );
    }

    sum.end_layer(output, input);

    #[cfg(debug_assertions)]
    sum.assert_invariants();
}

/// Below this many buckets there is nothing to spread, so skip Rayon entirely.
///
/// `desired_bits` already gives a small sum few buckets, so this mostly catches
/// the `bits = 0` case where the bucketed path degenerates to a single
/// whole-sum gather.
const MIN_BUCKETS_FOR_PARALLEL: usize = 2;

/// Gather, sort and merge one output bucket. The unit of parallel work.
fn fill_bucket<const W: usize, T>(
    input: &[BucketCols<W>],
    out_b: usize,
    dst: &mut BucketCols<W>,
    prep: &Prepared<W>,
    policy: &T,
    scratch: &mut LayerScratch<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    let cap = gather_capacity(input, out_b, prep);
    scratch.reset(cap);
    match prep {
        Prepared::Local(ptm) => gather_local(input, out_b, ptm, scratch),
        Prepared::Rotation(r) => gather_rotation(input, out_b, r, scratch),
    }
    debug_assert!(scratch.len() <= cap, "gather exceeded its computed bound");

    let len = scratch.len();
    sort_phase(&mut scratch.x, &mut scratch.z, &mut scratch.coeff, len);

    merge_into::<W, T>(
        &scratch.x,
        &scratch.z,
        &scratch.coeff,
        0,
        len,
        &mut dst.x,
        &mut dst.z,
        &mut dst.coeff,
        policy,
    );
}

/// Exact upper bound on the gather for one output bucket: the total size of the
/// input buckets it reads, counted once per delta.
fn gather_capacity<const W: usize>(
    input: &[BucketCols<W>],
    out_b: usize,
    prep: &Prepared<W>,
) -> usize {
    match prep {
        Prepared::Local(ptm) => ptm
            .deltas()
            .iter()
            .map(|d| input[out_b ^ d.bucket_delta as usize].len())
            .sum(),
        Prepared::Rotation(r) => {
            input[out_b ^ r.bucket_delta_identity as usize].len()
                + input[out_b ^ r.bucket_delta_gen as usize].len()
        }
    }
}

/// The tabulated inner loop (v0.2 §2.6): one lookup on ≤ 4 extracted bits, one
/// XOR with a precomputed mask, one complex multiply.
///
/// Deltas are visited in `local_delta` order, never grouped by input bucket.
/// Grouping would give better locality, but the group order depends on `H·d` and
/// hence on the bucket count, while duplicate-key summation order is observable
/// through floating-point non-associativity. This order is bucket-count
/// independent (v0.2 §9.1).
fn gather_local<const W: usize>(
    input: &[BucketCols<W>],
    out_b: usize,
    ptm: &LocalPtm<W>,
    scratch: &mut LayerScratch<W>,
) {
    for d in ptm.deltas() {
        let src = &input[out_b ^ d.bucket_delta as usize];
        for i in 0..src.len() {
            let s = ptm.support_bits(&src.x[i], &src.z[i]);
            let a = d.amp[s];
            if a == ZERO {
                continue;
            }
            let mut kx = src.x[i];
            let mut kz = src.z[i];
            for w in 0..W {
                kx[w] ^= d.mask_x[w];
                kz[w] ^= d.mask_z[w];
            }
            scratch.push(kx, kz, src.coeff[i] * a);
        }
    }
}

/// Gather for a rotation whose generator is too wide to tabulate.
///
/// The delta set is still `{0, P}`, so only two buckets are read at any
/// generator weight — but the `i^k` phase depends on `2w` support bits, so it is
/// computed per term. `cos`/`sin` are hoisted (v0.1 recomputed them per term).
/// Delta `0` is visited before delta `P`, matching the canonical `local_delta`
/// order used by [`gather_local`].
fn gather_rotation<const W: usize>(
    input: &[BucketCols<W>],
    out_b: usize,
    r: &RotationPrep<W>,
    scratch: &mut LayerScratch<W>,
) {
    // Delta 0: the input key survives, scaled by 1 (commuting) or cos (not).
    {
        let src = &input[out_b ^ r.bucket_delta_identity as usize];
        for i in 0..src.len() {
            let v = PauliString::<W> {
                x: src.x[i],
                z: src.z[i],
            };
            let c = if v.commutes_with(&r.gen) {
                src.coeff[i]
            } else {
                src.coeff[i] * r.cos
            };
            scratch.push(src.x[i], src.z[i], c);
        }
    }
    // Delta P: only anticommuting inputs contribute.
    {
        let src = &input[out_b ^ r.bucket_delta_gen as usize];
        for i in 0..src.len() {
            let v = PauliString::<W> {
                x: src.x[i],
                z: src.z[i],
            };
            if v.commutes_with(&r.gen) {
                continue;
            }
            let mut prod = v;
            let phase = prod.mul_assign(&r.gen);
            let total = Phase::I + phase;
            scratch.push(prod.x, prod.z, total.apply(src.coeff[i]) * r.sin);
        }
    }
}

/// In-place coefficient rescale for a key-preserving channel.
///
/// Keys are untouched, so each bucket stays sorted and duplicate-free and no
/// gather, sort or merge is needed. `keep_term` still applies, on the rescaled
/// coefficient, and exact zeros are still dropped — matching the general path.
fn rescale_in_place<const W: usize, T>(sum: &mut BucketedSum<W>, ptm: &LocalPtm<W>, policy: &T)
where
    T: TruncationPolicy<W> + ?Sized,
{
    let amp = &ptm.deltas()[0].amp;
    sum.buckets_mut().par_iter_mut().for_each(|cols| {
        let n = cols.len();
        let mut keep = 0usize;
        for i in 0..n {
            let s = ptm.support_bits(&cols.x[i], &cols.z[i]);
            let c = cols.coeff[i] * amp[s];
            if c == ZERO || !policy.keep_term(&cols.x[i], &cols.z[i], c) {
                continue;
            }
            // `keep <= i` always, so this never overwrites an unread slot.
            cols.x[keep] = cols.x[i];
            cols.z[keep] = cols.z[i];
            cols.coeff[keep] = c;
            keep += 1;
        }
        cols.x.truncate(keep);
        cols.z.truncate(keep);
        cols.coeff.truncate(keep);
    });
    sum.recount();

    #[cfg(debug_assertions)]
    sum.assert_invariants();
}

// Gated on `debug_assertions` because these tests call `assert_invariants`,
// which is itself debug-only. Matches the convention in `pauli_sum.rs` and
// `sort_merge.rs`; without it `cargo bench` and `cargo test --release`, which
// compile the lib tests in release mode, fail to build.
#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;
    use crate::accumulator::BuildAccumulator;
    use crate::bucket::hash::Gf2Hash;
    use crate::channel::clifford::{Clifford1Q, Clifford2Q};
    use crate::channel::identity::IdentityChannel;
    use crate::channel::noise::{AmplitudeDamping, Dephasing, Depolarizing};
    use crate::channel::rotation::PauliRotation;
    use crate::channel::Channel;
    use crate::engine::sort_merge::{apply_layer, apply_layer_adjoint};
    use crate::pauli_sum::PauliSum;
    use crate::truncation::builtin::{And, CoefficientThreshold, WeightCutoff};

    const TOL: f64 = 1e-11;

    pub(super) struct AlwaysKeep;
    impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

    pub(super) struct Xs64(u64);
    impl Xs64 {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    pub(super) fn word_mask(num_qubits: usize, word: usize) -> u64 {
        let lo = 64 * word;
        if num_qubits >= lo + 64 {
            !0u64
        } else if num_qubits <= lo {
            0
        } else {
            (1u64 << (num_qubits - lo)) - 1
        }
    }

    pub(super) fn rand_sum<const W: usize>(n: usize, num_qubits: usize, seed: u64) -> PauliSum<W> {
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

    /// Run one layer through the bucketed engine, converting in and out.
    pub(super) fn bucketed_layer<const W: usize, C, T>(
        input: &PauliSum<W>,
        ch: &C,
        policy: &T,
        adjoint: bool,
        bits: u8,
        seed: u64,
    ) -> PauliSum<W>
    where
        C: Channel<W> + ?Sized,
        T: TruncationPolicy<W> + ?Sized,
    {
        let hash = Gf2Hash::<W>::new(input.num_qubits(), bits, seed);
        let mut b = BucketedSum::from_sum(input, hash);
        let prep = ch
            .prepare(b.hash(), adjoint)
            .expect("channel could not be prepared");
        let mut scratch = LayerScratch::<W>::new();
        apply_layer_bucketed(&mut b, &prep, policy, &mut scratch);
        b.into_sum()
    }

    /// Keys must match exactly; coefficients only to tolerance, because the two
    /// engines sum duplicate keys in different orders and floating-point
    /// addition is not associative.
    pub(super) fn assert_sums_close<const W: usize>(
        got: &PauliSum<W>,
        want: &PauliSum<W>,
        what: &str,
    ) {
        assert_eq!(got.len(), want.len(), "{what}: term count");
        assert_eq!(got.x(), want.x(), "{what}: x keys");
        assert_eq!(got.z(), want.z(), "{what}: z keys");
        for i in 0..got.len() {
            let d = (got.coeff()[i] - want.coeff()[i]).norm();
            assert!(
                d < TOL,
                "{what}: coeff[{i}] {} vs {} (delta {d:e})",
                got.coeff()[i],
                want.coeff()[i],
            );
        }
    }

    // ---- hand-checked behaviour ----

    #[test]
    fn h_conjugates_z_to_x() {
        let mut acc = BuildAccumulator::<1>::with_capacity(4, 1);
        acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
        let input = acc.finalize();
        let out = bucketed_layer(&input, &Clifford1Q::h(0), &AlwaysKeep, false, 4, 0x1);
        assert_eq!(out.len(), 1);
        assert_eq!(out.x()[0], [1]);
        assert_eq!(out.z()[0], [0]);
        assert!((out.coeff()[0] - Complex64::new(1.0, 0.0)).norm() < TOL);
    }

    #[test]
    fn cnot_propagates_z_on_the_control() {
        let mut acc = BuildAccumulator::<1>::with_capacity(4, 1);
        acc.add_term(PauliString::<1>::z(1), Phase::ONE, Complex64::new(1.0, 0.0));
        let input = acc.finalize();
        // I⊗Z under CNOT(0 -> 1) becomes Z⊗Z.
        let out = bucketed_layer(&input, &Clifford2Q::cnot(0, 1), &AlwaysKeep, false, 4, 0x1);
        assert_eq!(out.len(), 1);
        assert_eq!(out.z()[0], [0b11]);
        assert_eq!(out.x()[0], [0]);
    }

    #[test]
    fn a_rotation_fans_out_to_two_terms() {
        let mut acc = BuildAccumulator::<1>::with_capacity(4, 1);
        acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(1.0, 0.0));
        let input = acc.finalize();
        let rot = PauliRotation::new(PauliString::<1>::z(0), std::f64::consts::FRAC_PI_3);
        let out = bucketed_layer(&input, &rot, &AlwaysKeep, false, 4, 0x1);
        // cos(pi/3)*X + sin(pi/3)*(i * X * Z) = 0.5*X - 0.866*Y
        assert_eq!(out.len(), 2);
        let want = apply_layer(&input, &rot, &AlwaysKeep);
        assert_sums_close(&out, &want, "rotation fanout");
    }

    // ---- the differential test against the v0.1 engine ----

    /// Every built-in channel, over both occupancy regimes, several bucket
    /// counts, forward and adjoint, against three policies.
    ///
    /// This is the primary correctness net for the rewrite: the v0.1 engine is
    /// the oracle (v0.2 §9.2). A disagreement is a bug in the new engine until
    /// proven otherwise.
    #[test]
    fn differential_against_sort_merge_w1_dense_collisions() {
        // Only 8 qubits, so 2000 random terms collide heavily under a rotation
        // (both `v` and `v ^ gen` are usually present) and the merge phase has
        // real duplicate runs to combine. This is the case that matters.
        let input = rand_sum::<1>(2000, 8, 0xC0FFEE);
        let channels: Vec<(&str, Box<dyn Channel<1>>)> = vec![
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
                Box::new(crate::channel::GeneralUnitary1Q::from_matrix(
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
                "general_2q",
                Box::new({
                    // sqrt(SWAP): dense enough to exercise a wide delta set.
                    let h = Complex64::new(0.5, 0.5);
                    let hc = Complex64::new(0.5, -0.5);
                    let one = Complex64::new(1.0, 0.0);
                    let zero = Complex64::new(0.0, 0.0);
                    crate::channel::GeneralUnitary2Q::from_matrix(
                        1,
                        5,
                        [
                            [one, zero, zero, zero],
                            [zero, h, hc, zero],
                            [zero, hc, h, zero],
                            [zero, zero, zero, one],
                        ],
                    )
                }),
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
        ];

        for (name, ch) in &channels {
            let cr: &dyn Channel<1> = ch.as_ref();
            for &adjoint in &[false, true] {
                for &bits in &[0u8, 1, 3, 6, 11] {
                    let want = if adjoint {
                        apply_layer_adjoint(&input, cr, &AlwaysKeep)
                    } else {
                        apply_layer(&input, cr, &AlwaysKeep)
                    };
                    let got = bucketed_layer(&input, cr, &AlwaysKeep, adjoint, bits, 0xABCD);
                    assert_sums_close(
                        &got,
                        &want,
                        &format!("{name} adjoint={adjoint} bits={bits}"),
                    );
                }
            }
        }
    }

    #[test]
    fn differential_against_sort_merge_w2_sparse() {
        // The other regime: wide keys, few collisions, word-boundary supports.
        let input = rand_sum::<2>(3000, 128, 0xBEEF);
        let channels: Vec<(&str, Box<dyn Channel<2>>)> = vec![
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
        ];
        for (name, ch) in &channels {
            let cr: &dyn Channel<2> = ch.as_ref();
            for &adjoint in &[false, true] {
                for &bits in &[2u8, 5, 9] {
                    let want = if adjoint {
                        apply_layer_adjoint(&input, cr, &AlwaysKeep)
                    } else {
                        apply_layer(&input, cr, &AlwaysKeep)
                    };
                    let got = bucketed_layer(&input, cr, &AlwaysKeep, adjoint, bits, 0xABCD);
                    assert_sums_close(
                        &got,
                        &want,
                        &format!("{name} adjoint={adjoint} bits={bits}"),
                    );
                }
            }
        }
    }

    #[test]
    fn differential_with_truncation_policies() {
        let input = rand_sum::<1>(1500, 8, 0xF00D);
        // Thresholds are chosen far from the coefficient scale so the two
        // engines cannot disagree merely by rounding across a cutoff.
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
        let cnot = Clifford2Q::cnot(1, 5);

        for bits in [0u8, 4, 9] {
            let got = bucketed_layer(&input, &rot, &CoefficientThreshold(1e-9), false, bits, 0x11);
            let want = apply_layer(&input, &rot, &CoefficientThreshold(1e-9));
            assert_sums_close(&got, &want, &format!("threshold bits={bits}"));

            let got = bucketed_layer(&input, &rot, &WeightCutoff(4), false, bits, 0x11);
            let want = apply_layer(&input, &rot, &WeightCutoff(4));
            assert_sums_close(&got, &want, &format!("weight bits={bits}"));

            let policy = And(CoefficientThreshold(1e-9), WeightCutoff(5));
            let got = bucketed_layer(&input, &cnot, &policy, false, bits, 0x11);
            let want = apply_layer(&input, &cnot, &policy);
            assert_sums_close(&got, &want, &format!("and bits={bits}"));
        }
    }

    #[test]
    fn keep_term_sees_the_summed_coefficient() {
        // Port of sort_merge's `threshold_applied_after_summation`: two terms
        // that nearly cancel must be dropped by a threshold their individual
        // magnitudes would pass. A rotation at theta = pi/2 sends X and Y to the
        // same key with opposite-ish weights.
        let mut acc = BuildAccumulator::<1>::with_capacity(4, 2);
        acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(0.5, 0.0));
        acc.add_term(
            PauliString::<1>::y(0),
            Phase::ONE,
            Complex64::new(-0.4999999, 0.0),
        );
        let input = acc.finalize();
        // theta = 0 keeps keys fixed but the sum has no duplicates, so use the
        // oracle for the general statement instead of hand-computing.
        let rot = PauliRotation::new(PauliString::<1>::z(0), std::f64::consts::FRAC_PI_2);
        for bits in [0u8, 3, 7] {
            let policy = CoefficientThreshold(1e-6);
            let got = bucketed_layer(&input, &rot, &policy, false, bits, 0x21);
            let want = apply_layer(&input, &rot, &policy);
            assert_sums_close(&got, &want, &format!("post-sum threshold bits={bits}"));
        }
    }

    // ---- the key-preserving fast path ----

    #[test]
    fn rescale_fast_path_agrees_with_the_general_path() {
        // Depolarizing/Dephasing/Pauli gates take `rescale_in_place`. Compare
        // against the v0.1 engine, which has no such special case.
        let input = rand_sum::<1>(1500, 8, 0x5A5A);
        let chans: Vec<(&str, Box<dyn Channel<1>>)> = vec![
            ("identity", Box::new(IdentityChannel::new())),
            (
                "depolarizing",
                Box::new(Depolarizing {
                    support: [3],
                    p: 0.11,
                }),
            ),
            (
                "dephasing",
                Box::new(Dephasing {
                    support: [3],
                    p: 0.11,
                }),
            ),
            ("pauli_z", Box::new(Clifford1Q::z(3))),
        ];
        for (name, ch) in &chans {
            let cr: &dyn Channel<1> = ch.as_ref();
            for bits in [0u8, 4, 8] {
                let got = bucketed_layer(&input, cr, &AlwaysKeep, false, bits, 0x31);
                let want = apply_layer(&input, cr, &AlwaysKeep);
                assert_sums_close(&got, &want, &format!("{name} bits={bits}"));
            }
        }
    }

    #[test]
    fn rescale_fast_path_still_applies_truncation() {
        let input = rand_sum::<1>(1500, 8, 0x5A5B);
        let depol = Depolarizing {
            support: [3],
            p: 0.11,
        };
        for bits in [0u8, 5] {
            let policy = And(CoefficientThreshold(0.3), WeightCutoff(4));
            let got = bucketed_layer(&input, &depol, &policy, false, bits, 0x41);
            let want = apply_layer(&input, &depol, &policy);
            assert_sums_close(&got, &want, &format!("truncated rescale bits={bits}"));
            assert!(got.len() < input.len(), "truncation dropped nothing");
        }
    }

    // ---- determinism ----

    #[test]
    fn output_is_bitwise_identical_across_bucket_counts() {
        // The strong form of v0.2 §9.1: not merely close, but *bitwise* equal,
        // which is what the canonical `local_delta` gather order buys. A
        // group-by-bucket gather order would break this.
        let input = rand_sum::<1>(2000, 8, 0x9001);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
        let cnot = Clifford2Q::cnot(1, 5);
        for ch in [&rot as &dyn Channel<1>, &cnot as &dyn Channel<1>] {
            let reference = bucketed_layer(&input, ch, &AlwaysKeep, false, 0, 0x51);
            for bits in [1u8, 2, 3, 5, 8, 11] {
                let got = bucketed_layer(&input, ch, &AlwaysKeep, false, bits, 0x51);
                assert_eq!(got.len(), reference.len(), "bits={bits}: length");
                assert_eq!(got.x(), reference.x(), "bits={bits}: x keys");
                assert_eq!(got.z(), reference.z(), "bits={bits}: z keys");
                assert_eq!(
                    got.coeff(),
                    reference.coeff(),
                    "bits={bits}: coefficients are not bitwise identical",
                );
            }
        }
    }

    #[test]
    fn output_is_bitwise_identical_across_hash_seeds() {
        // A different `H` permutes which terms share a bucket but must not
        // change the arithmetic.
        let input = rand_sum::<1>(2000, 8, 0x9002);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.41);
        let reference = bucketed_layer(&input, &rot, &AlwaysKeep, false, 6, 1);
        for seed in [2u64, 3, 5, 8, 13, 21] {
            let got = bucketed_layer(&input, &rot, &AlwaysKeep, false, 6, seed);
            assert_eq!(got.coeff(), reference.coeff(), "seed={seed}");
            assert_eq!(got.x(), reference.x(), "seed={seed}");
        }
    }

    // ---- multi-layer, staying bucketed ----

    #[test]
    fn many_layers_without_converting_out() {
        // The point of the bucketed form: convert in once, run many layers,
        // convert out once. Compare against the same sequence through v0.1.
        let input = rand_sum::<1>(800, 8, 0x7001);
        let chans: Vec<Box<dyn Channel<1>>> = vec![
            Box::new(Clifford1Q::h(0)),
            Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.3)),
            Box::new(Clifford2Q::cnot(1, 5)),
            Box::new(Depolarizing {
                support: [3],
                p: 0.05,
            }),
            Box::new(Clifford1Q::s(6)),
            Box::new(PauliRotation::new(
                {
                    let mut g = PauliString::<1>::z(1);
                    g.mul_assign(&PauliString::<1>::z(4));
                    g
                },
                0.2,
            )),
        ];

        let mut want = input.clone();
        for ch in &chans {
            want = apply_layer(&want, ch.as_ref(), &AlwaysKeep);
        }

        let hash = Gf2Hash::<1>::new(8, 5, 0x77);
        let mut b = BucketedSum::from_sum(&input, hash);
        let mut scratch = LayerScratch::<1>::new();
        for ch in &chans {
            let prep = ch.prepare(b.hash(), false).unwrap();
            apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);
        }
        let got = b.into_sum();
        assert_sums_close(&got, &want, "six layers");
    }

    #[test]
    fn layers_survive_a_rebucket_in_between() {
        let input = rand_sum::<1>(800, 8, 0x7002);
        let h = Clifford1Q::h(0);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.3);

        let want = apply_layer(&apply_layer(&input, &h, &AlwaysKeep), &rot, &AlwaysKeep);

        let hash = Gf2Hash::<1>::new(8, 2, 0x77);
        let mut b = BucketedSum::from_sum(&input, hash);
        let mut scratch = LayerScratch::<1>::new();

        let prep = h.prepare(b.hash(), false).unwrap();
        apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);
        b.rebucket(32, 1);
        let prep = rot.prepare(b.hash(), false).unwrap();
        apply_layer_bucketed(&mut b, &prep, &AlwaysKeep, &mut scratch);

        assert_sums_close(&b.into_sum(), &want, "layer, rebucket, layer");
    }

    #[test]
    fn an_empty_sum_survives_a_layer() {
        let input = PauliSum::<1>::empty(8);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.3);
        let out = bucketed_layer(&input, &rot, &AlwaysKeep, false, 4, 0x1);
        assert!(out.is_empty());
    }
}

// Gated on `debug_assertions` because these tests call `assert_invariants`,
// which is itself debug-only. Matches the convention in `pauli_sum.rs` and
// `sort_merge.rs`; without it `cargo bench` and `cargo test --release`, which
// compile the lib tests in release mode, fail to build.
#[cfg(all(test, debug_assertions))]
mod finalize_tests {
    use super::tests::rand_sum;
    use super::*;
    use crate::bucket::hash::Gf2Hash;
    use crate::channel::clifford::Clifford1Q;
    use crate::channel::rotation::PauliRotation;
    use crate::channel::Channel;
    use crate::pauli_sum::PauliSum;
    use crate::truncation::builtin::{And, CoefficientThreshold, Or, TopN, WeightCutoff};

    /// `TopN` bucketed must keep exactly `n` terms, and the same *set* as the
    /// flat implementation when there are no ties in magnitude.
    #[test]
    fn top_n_bucketed_matches_the_flat_implementation() {
        let input = rand_sum::<1>(2000, 8, 0x1234);
        for n in [1usize, 7, 100, 999, 1999, 5000] {
            let policy = TopN(n);
            let mut flat = input.clone();
            policy.finalize_layer(&mut flat);

            for bits in [0u8, 3, 6, 10] {
                let hash = Gf2Hash::<1>::new(8, bits, 0x99);
                let mut b = BucketedSum::from_sum(&input, hash);
                policy.finalize_layer_bucketed(&mut b);
                b.assert_invariants();
                let got = b.into_sum();
                assert_eq!(got.len(), flat.len(), "n={n} bits={bits}: length");
                assert_eq!(got.x(), flat.x(), "n={n} bits={bits}: x keys");
                assert_eq!(got.z(), flat.z(), "n={n} bits={bits}: z keys");
                assert_eq!(got.coeff(), flat.coeff(), "n={n} bits={bits}: coeffs");
            }
        }
    }

    #[test]
    fn top_n_bucketed_keeps_exactly_n_and_the_largest() {
        let input = rand_sum::<1>(1000, 8, 0x4321);
        let hash = Gf2Hash::<1>::new(8, 5, 0x99);
        let mut b = BucketedSum::from_sum(&input, hash);
        TopN(50).finalize_layer_bucketed(&mut b);
        assert_eq!(b.len(), 50);
        let got = b.into_sum();

        // Every retained magnitude must be >= every dropped one.
        let mut all: Vec<f64> = input.coeff().iter().map(|c| c.norm()).collect();
        all.sort_by(|a, c| c.partial_cmp(a).unwrap());
        let cutoff = all[49];
        for c in got.coeff() {
            assert!(c.norm() >= cutoff - 1e-15, "kept a below-cutoff term");
        }
    }

    #[test]
    fn top_n_zero_clears_and_preserves_the_invariant() {
        let input = rand_sum::<1>(500, 8, 0x5555);
        let hash = Gf2Hash::<1>::new(8, 4, 0x99);
        let mut b = BucketedSum::from_sum(&input, hash);
        TopN(0).finalize_layer_bucketed(&mut b);
        b.assert_invariants();
        assert_eq!(b.len(), 0);
        assert!(b.into_sum().is_empty());
    }

    #[test]
    fn top_n_above_the_length_is_a_no_op() {
        // Note `rand_sum` dedups, so the realized length is below the request
        // at only 8 qubits; compare against it rather than the literal.
        let input = rand_sum::<1>(300, 8, 0x6666);
        let hash = Gf2Hash::<1>::new(8, 4, 0x99);
        let mut b = BucketedSum::from_sum(&input, hash);
        TopN(10_000).finalize_layer_bucketed(&mut b);
        assert_eq!(b.len(), input.len());
        let got = b.into_sum();
        assert_eq!(got.x(), input.x());
        assert_eq!(got.coeff(), input.coeff());
    }

    #[test]
    fn and_runs_both_finalizers_bucketed() {
        // TopN(n) twice with different n must behave like the tighter one.
        let input = rand_sum::<1>(1000, 8, 0x7777);
        let policy = And(TopN(400), TopN(120));
        let mut flat = input.clone();
        policy.finalize_layer(&mut flat);

        let hash = Gf2Hash::<1>::new(8, 5, 0x99);
        let mut b = BucketedSum::from_sum(&input, hash);
        policy.finalize_layer_bucketed(&mut b);
        b.assert_invariants();
        let got = b.into_sum();
        assert_eq!(got.len(), 120);
        assert_eq!(got.x(), flat.x());
        assert_eq!(got.coeff(), flat.coeff());
    }

    #[test]
    fn threshold_and_weight_and_or_finalizers_are_no_ops() {
        // These three have no layer-finalization step; the bucketed override
        // must leave the sum untouched rather than round-trip it.
        let input = rand_sum::<1>(500, 8, 0x8888);
        let hash = Gf2Hash::<1>::new(8, 4, 0x99);
        for tag in 0..3 {
            let mut b = BucketedSum::from_sum(&input, hash.clone());
            match tag {
                0 => CoefficientThreshold(0.5).finalize_layer_bucketed(&mut b),
                1 => WeightCutoff(2).finalize_layer_bucketed(&mut b),
                _ => Or(CoefficientThreshold(0.5), WeightCutoff(2)).finalize_layer_bucketed(&mut b),
            }
            assert_eq!(b.len(), input.len(), "tag {tag} changed the sum");
        }
    }

    /// The default trait implementation must keep a custom `finalize_layer`
    /// working on the bucketed path, without the policy knowing about buckets.
    #[test]
    fn the_default_round_trip_preserves_a_custom_finalizer() {
        /// Drops every term whose coefficient has negative real part — a global
        /// pass expressed only as `finalize_layer`.
        struct DropNegativeReal;
        impl<const W: usize> TruncationPolicy<W> for DropNegativeReal {
            fn finalize_layer(&self, sum: &mut PauliSum<W>) {
                let keep: Vec<bool> = sum.coeff().iter().map(|c| c.re >= 0.0).collect();
                let mut w = 0usize;
                for (i, &k) in keep.iter().enumerate() {
                    if k {
                        sum.x[w] = sum.x[i];
                        sum.z[w] = sum.z[i];
                        sum.coeff[w] = sum.coeff[i];
                        w += 1;
                    }
                }
                sum.x.truncate(w);
                sum.z.truncate(w);
                sum.coeff.truncate(w);
            }
        }

        let input = rand_sum::<1>(800, 8, 0x9999);
        let mut flat = input.clone();
        DropNegativeReal.finalize_layer(&mut flat);
        assert!(
            flat.len() < input.len(),
            "the custom policy dropped nothing"
        );

        for bits in [0u8, 3, 7] {
            let hash = Gf2Hash::<1>::new(8, bits, 0x99);
            let mut b = BucketedSum::from_sum(&input, hash);
            DropNegativeReal.finalize_layer_bucketed(&mut b);
            b.assert_invariants();
            let got = b.into_sum();
            assert_eq!(got.len(), flat.len(), "bits={bits}");
            assert_eq!(got.x(), flat.x(), "bits={bits}");
            assert_eq!(got.coeff(), flat.coeff(), "bits={bits}");
        }
    }

    /// Layer then finalize, repeatedly — the shape `propagate` will use.
    #[test]
    fn interleaved_layers_and_finalizers_match_the_v0_1_sequence() {
        use crate::engine::sort_merge::apply_layer;

        let input = rand_sum::<1>(1200, 8, 0xAAAA);
        let policy = And(CoefficientThreshold(1e-9), TopN(300));
        let chans: Vec<Box<dyn Channel<1>>> = vec![
            Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.37)),
            Box::new(Clifford1Q::h(0)),
            Box::new(PauliRotation::new(PauliString::<1>::x(5), 0.21)),
        ];

        let mut want = input.clone();
        for ch in &chans {
            want = apply_layer(&want, ch.as_ref(), &policy);
            policy.finalize_layer(&mut want);
        }

        let hash = Gf2Hash::<1>::new(8, 5, 0xBB);
        let mut b = BucketedSum::from_sum(&input, hash);
        let mut scratch = LayerScratch::<1>::new();
        for ch in &chans {
            let prep = ch.prepare(b.hash(), false).unwrap();
            apply_layer_bucketed(&mut b, &prep, &policy, &mut scratch);
            policy.finalize_layer_bucketed(&mut b);
        }
        let got = b.into_sum();

        assert_eq!(got.len(), want.len(), "term count after 3 truncated layers");
        assert_eq!(got.x(), want.x(), "keys after 3 truncated layers");
        for i in 0..got.len() {
            assert!(
                (got.coeff()[i] - want.coeff()[i]).norm() < 1e-11,
                "coeff[{i}]",
            );
        }
    }
}

// Gated on `debug_assertions` because these tests call `assert_invariants`,
// which is itself debug-only. Matches the convention in `pauli_sum.rs` and
// `sort_merge.rs`; without it `cargo bench` and `cargo test --release`, which
// compile the lib tests in release mode, fail to build.
#[cfg(all(test, debug_assertions))]
mod tie_tests {
    /// The C.1 determinism contract: byte-identical output across thread counts,
    /// with the *engine* parallel. `apply_layer_bucketed` fixes the bucket count
    /// here, so this isolates thread count from partition (the propagate-level
    /// test in tests/propagate_bucketed.rs varies both together).
    #[test]
    fn parallel_output_is_byte_identical_across_thread_counts() {
        use crate::channel::rotation::PauliRotation;
        use crate::channel::Channel;

        let input = rand_sum::<1>(4000, 10, 0xC1C1);
        let rot = PauliRotation::new(PauliString::<1>::z(2), 0.37);
        let cnot = crate::channel::clifford::Clifford2Q::cnot(1, 5);

        for ch in [&rot as &dyn Channel<1>, &cnot as &dyn Channel<1>] {
            // 64 buckets: comfortably above MIN_BUCKETS_FOR_PARALLEL, so the
            // parallel path is genuinely exercised.
            let run = |threads: usize| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("pool")
                    .install(|| {
                        let hash = Gf2Hash::<1>::new(10, 6, 0xC1);
                        let mut b = BucketedSum::from_sum(&input, hash);
                        let prep = ch.prepare(b.hash(), false).unwrap();
                        let mut scratch = LayerScratch::<1>::new();
                        apply_layer_bucketed(
                            &mut b,
                            &prep,
                            &super::tests::AlwaysKeep,
                            &mut scratch,
                        );
                        b.into_sum()
                    })
            };
            let reference = run(1);
            for threads in [2usize, 4, 8, 16, 32] {
                let got = run(threads);
                assert_eq!(got.len(), reference.len(), "threads={threads}");
                assert_eq!(got.x(), reference.x(), "threads={threads}: x keys");
                assert_eq!(got.z(), reference.z(), "threads={threads}: z keys");
                assert_eq!(
                    got.coeff(),
                    reference.coeff(),
                    "threads={threads}: coefficients are not byte-identical",
                );
            }
        }
    }

    /// The in-place rescale path is parallel too, and must give the same answer.
    #[test]
    fn parallel_rescale_is_byte_identical_across_thread_counts() {
        use crate::channel::noise::Depolarizing;
        use crate::channel::Channel;

        let input = rand_sum::<1>(4000, 10, 0xC1C2);
        let depol = Depolarizing {
            support: [3],
            p: 0.11,
        };
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| {
                    let hash = Gf2Hash::<1>::new(10, 6, 0xC2);
                    let mut b = BucketedSum::from_sum(&input, hash);
                    let prep = Channel::<1>::prepare(&depol, b.hash(), false).unwrap();
                    let mut scratch = LayerScratch::<1>::new();
                    apply_layer_bucketed(&mut b, &prep, &super::tests::AlwaysKeep, &mut scratch);
                    b.into_sum()
                })
        };
        let reference = run(1);
        for threads in [2usize, 8, 32] {
            let got = run(threads);
            assert_eq!(got.coeff(), reference.coeff(), "threads={threads}");
        }
    }

    use super::tests::rand_sum;
    use super::*;
    use crate::bucket::hash::Gf2Hash;
    use crate::pauli_sum::PauliSum;
    use crate::truncation::builtin::TopN;

    /// A sum whose coefficients take only a handful of distinct magnitudes, so
    /// `TopN` is guaranteed to cut through a large tie group.
    ///
    /// This is not a contrived case: a symmetric Hamiltonian on a periodic
    /// lattice produces many terms related by lattice symmetry with *exactly*
    /// equal coefficients, which is why the 2D Ising example hits it.
    fn tie_heavy_sum(n: usize, num_qubits: usize, seed: u64) -> PauliSum<1> {
        let base = rand_sum::<1>(n, num_qubits, seed);
        let mut acc = crate::accumulator::BuildAccumulator::<1>::with_capacity(num_qubits, n);
        for i in 0..base.len() {
            // Only 4 distinct magnitudes across the whole sum.
            let mag = [1.0f64, 0.5, 0.25, 0.125][i % 4];
            acc.add_term(
                PauliString::<1> {
                    x: base.x()[i],
                    z: base.z()[i],
                },
                Phase::ONE,
                Complex64::new(mag, 0.0),
            );
        }
        acc.finalize()
    }

    /// `TopN` must keep the same set regardless of the bucket partition, even
    /// when the cut falls inside a tie group.
    ///
    /// Tie-breaking on flat position would fail this: flat position depends on
    /// which bucket a term landed in, hence on the bucket count.
    #[test]
    fn top_n_is_bucket_count_independent_on_tied_magnitudes() {
        let input = tie_heavy_sum(2000, 8, 0x7135);
        let n = 700; // cuts inside the group of magnitude-0.5 terms
        let reference = {
            let hash = Gf2Hash::<1>::new(8, 0, 0x99);
            let mut b = BucketedSum::from_sum(&input, hash);
            TopN(n).finalize_layer_bucketed(&mut b);
            b.into_sum()
        };
        for bits in [1u8, 2, 4, 6, 9] {
            let hash = Gf2Hash::<1>::new(8, bits, 0x99);
            let mut b = BucketedSum::from_sum(&input, hash);
            TopN(n).finalize_layer_bucketed(&mut b);
            let got = b.into_sum();
            assert_eq!(got.len(), reference.len(), "bits={bits}: length");
            assert_eq!(
                got.x(),
                reference.x(),
                "bits={bits}: TopN kept a different set of tied terms",
            );
            assert_eq!(got.coeff(), reference.coeff(), "bits={bits}: coefficients");
        }
    }

    /// The bucketed and flat implementations must agree **on ties too**, not
    /// just on distinct magnitudes. This is what lets the two engines produce
    /// identical output on a symmetric Hamiltonian.
    #[test]
    fn top_n_bucketed_matches_flat_on_tied_magnitudes() {
        let input = tie_heavy_sum(2000, 8, 0x7136);
        for n in [3usize, 250, 700, 1200, 1900] {
            let policy = TopN(n);
            let mut flat = input.clone();
            policy.finalize_layer(&mut flat);
            for bits in [0u8, 2, 5, 9] {
                let hash = Gf2Hash::<1>::new(8, bits, 0x99);
                let mut b = BucketedSum::from_sum(&input, hash);
                policy.finalize_layer_bucketed(&mut b);
                let got = b.into_sum();
                assert_eq!(got.len(), flat.len(), "n={n} bits={bits}: length");
                assert_eq!(got.x(), flat.x(), "n={n} bits={bits}: keys differ on ties");
                assert_eq!(got.coeff(), flat.coeff(), "n={n} bits={bits}: coeffs");
            }
        }
    }

    /// The same, across hash seeds: a different `H` permutes bucket membership
    /// without changing anything about the magnitudes.
    #[test]
    fn top_n_is_hash_seed_independent_on_tied_magnitudes() {
        let input = tie_heavy_sum(2000, 8, 0x7123);
        let n = 700;
        let reference = {
            let hash = Gf2Hash::<1>::new(8, 5, 1);
            let mut b = BucketedSum::from_sum(&input, hash);
            TopN(n).finalize_layer_bucketed(&mut b);
            b.into_sum()
        };
        for seed in [2u64, 3, 5, 8, 13] {
            let hash = Gf2Hash::<1>::new(8, 5, seed);
            let mut b = BucketedSum::from_sum(&input, hash);
            TopN(n).finalize_layer_bucketed(&mut b);
            let got = b.into_sum();
            assert_eq!(got.x(), reference.x(), "seed={seed}: different set kept");
        }
    }
}
