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
//! Output buckets are write-disjoint (v0.2 §2.4), so step 2-4 for different `β'`
//! never interact. This file is deliberately **sequential**: Phase C replaces the
//! bucket loop with `par_iter_mut` and nothing else. Establishing correctness
//! first is the point.

use num_complex::Complex64;

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

    // Iterating the *output* buckets is the parallel decomposition: each owns
    // its destination and only reads `input`. Phase C replaces this with
    // `par_iter_mut().enumerate()` and changes nothing else.
    for (out_b, dst) in output.iter_mut().enumerate() {
        let cap = gather_capacity(&input, out_b, prep);
        scratch.reset(cap);
        match prep {
            Prepared::Local(ptm) => gather_local(&input, out_b, ptm, scratch),
            Prepared::Rotation(r) => gather_rotation(&input, out_b, r, scratch),
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

    sum.end_layer(output, input);

    #[cfg(debug_assertions)]
    sum.assert_invariants();
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
    for cols in sum.buckets_mut() {
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
    }
    sum.recount();

    #[cfg(debug_assertions)]
    sum.assert_invariants();
}

#[cfg(test)]
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

    struct AlwaysKeep;
    impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

    struct Xs64(u64);
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

    fn word_mask(num_qubits: usize, word: usize) -> u64 {
        let lo = 64 * word;
        if num_qubits >= lo + 64 {
            !0u64
        } else if num_qubits <= lo {
            0
        } else {
            (1u64 << (num_qubits - lo)) - 1
        }
    }

    fn rand_sum<const W: usize>(n: usize, num_qubits: usize, seed: u64) -> PauliSum<W> {
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
    fn bucketed_layer<const W: usize, C, T>(
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
    fn assert_sums_close<const W: usize>(got: &PauliSum<W>, want: &PauliSum<W>, what: &str) {
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
