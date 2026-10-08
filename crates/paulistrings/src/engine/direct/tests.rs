use super::*;
use crate::channel::{
    support_mask, AmplitudeDamping, Clifford1Q, Clifford2Q, Depolarizing, GeneralUnitary2Q,
    PauliRotation,
};
use crate::pauli_sum::accumulator::BuildAccumulator;
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

/// `H` conjugates `Z` to `X`, so `Z₀ + 0.5·X₁` becomes `X₀ + 0.5·X₁`.
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

/// `exp(-i·θ·Z₀/2)` on `X₀` gives `cos(θ)·X₀ + sin(θ)·Y₀`, the fanout-2 case, at θ = π/3.
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

/// `keep_term` sees the summed coefficient.
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

    // A π/2 Z-rotation on X₀ emits cos(π/2)·X₀ (an exact-ish zero) plus sin(π/2)·Y₀.
    // Seeding both X₀ and Y₀ makes the Y row a two-row sum.
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
    // Y₀'s two contributions sum to ≈ 1 and X₀'s to ≈ −1, so both survive the loose threshold.
    assert_eq!(loose.len(), 2);
    // The strict threshold keeps both too; a per-row filter would have dropped the ≈ 0 rows and changed the sums.
    assert_eq!(strict.len(), 2);
    let y = strict.get(&[1], &[1]).expect("Y term");
    assert!((y.norm() - 1.0).abs() < 1e-12, "Y coeff {y}");
}

/// The differential oracle over the channel zoo at both widths, pinning the plumbing; `tests/small_sum_path.rs` pins the algebra.
fn differential<const W: usize>(num_qubits: usize, seed: u64) {
    let sum = rand_sum::<W>(300, num_qubits, seed);
    let hi = num_qubits.saturating_sub(1) as u32;

    let mut matrix = [[Complex64::new(0.0, 0.0); 4]; 4];
    for (r, row) in matrix.iter_mut().enumerate() {
        row[r] = Complex64::new(1.0, 0.0);
    }
    // A real 4×4 rotation in the (0,1) block, dense in the PTM.
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
        // Generator weight 4, above MAX_LOCAL_SUPPORT.
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

/// A three-qubit channel that `prepare` declines still applies correctly on this path.
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
        /// Cyclically shifts the x-bits of qubits 0, 1, 2.
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

/// Ingest then materialize keeps the term set, at a partition no coarser.
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

/// `reload` replaces the resident terms rather than merging into them.
#[test]
fn reload_replaces_the_resident_terms() {
    let a = rand_sum::<1>(50, 12, 0xA1);
    let b = rand_sum::<1>(30, 12, 0xB2);
    let mut direct = DirectSum::from_sum(a);
    direct.reload(&b);
    assert_eq!(direct.len(), b.len());
    assert_same_terms(&direct.to_sum(), &b, "reload");
}
