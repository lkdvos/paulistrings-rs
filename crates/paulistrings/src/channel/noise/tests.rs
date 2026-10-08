use super::*;
use crate::channel::prepared::Prepared;
use crate::pauli_string::PauliString;
use crate::pauli_sum::hash::Gf2Hash;
use crate::phase::Phase;
use crate::test_support::{alloc_bufs, approx_eq};

/// Collect the outputs of `apply` or `apply_adjoint` on one input.
///
/// In emission order, zeros included, since these tests assert on row positions.
fn outputs<const W: usize>(
    ch: &AmplitudeDamping,
    adjoint: bool,
    p: PauliString<W>,
) -> Vec<(PauliString<W>, Complex64)> {
    crate::test_support::raw_outputs::<W, AmplitudeDamping>(
        ch,
        adjoint,
        p,
        Complex64::new(1.0, 0.0),
    )
}

/// Build the 4x4 single-qubit PTM on the support, `t[out][in]`, using the `I=0, X=1, Z=2, Y=3` packing.
fn ptm4(ch: &AmplitudeDamping, adjoint: bool, q: u32) -> [[f64; 4]; 4] {
    let basis = |idx: usize| -> PauliString<1> {
        match idx {
            0 => PauliString::<1>::identity(),
            1 => PauliString::<1>::x(q),
            2 => PauliString::<1>::z(q),
            _ => PauliString::<1>::y(q),
        }
    };
    let index_of = |p: &PauliString<1>| -> usize {
        let bit = q % 64;
        let xb = ((p.x[0] >> bit) & 1) as usize;
        let zb = ((p.z[0] >> bit) & 1) as usize;
        xb | (zb << 1)
    };
    let mut t = [[0.0f64; 4]; 4];
    #[allow(clippy::needless_range_loop)]
    for j in 0..4 {
        for (out_p, c) in outputs::<1>(ch, adjoint, basis(j)) {
            t[index_of(&out_p)][j] += c.re;
        }
    }
    t
}

/// Both maps of a damping channel on qubit `q`, row by row.
fn check_both_maps<const W: usize>(q: u32, g: f64) {
    let ch = AmplitudeDamping {
        support: [q],
        gamma: g,
    };
    let s = (1.0f64 - g).sqrt();
    let id = PauliString::<W>::identity();
    let zq = PauliString::<W>::z(q);

    // Φ = `apply`: the fan-out is on I. `Φ(I) = I + γ Z`.
    let got = outputs::<W>(&ch, false, id);
    assert_eq!(got.len(), 2, "W={W}: Φ(I) has two terms");
    assert_eq!(got[0].0, id);
    assert!((got[0].1 - Complex64::new(1.0, 0.0)).norm() < 1e-15);
    assert_eq!(got[1].0, zq);
    assert!((got[1].1 - Complex64::new(g, 0.0)).norm() < 1e-15);

    // `Φ(Z) = (1-γ) Z`, with no identity component: Φ preserves trace and `tr Z = 0`, so the I coefficient must vanish.
    let got = outputs::<W>(&ch, false, zq);
    assert_eq!(got.len(), 1, "W={W}: Φ(Z) has no I component");
    assert_eq!(got[0].0, zq);
    assert!((got[0].1 - Complex64::new(1.0 - g, 0.0)).norm() < 1e-15);

    // Φ† = `apply_adjoint`: unital, so `Φ†(I) = I` exactly, fan-out 1.
    let got = outputs::<W>(&ch, true, id);
    assert_eq!(got.len(), 1, "W={W}: Φ†(I) must be I alone (unitality)");
    assert_eq!(got[0].0, id);
    assert!((got[0].1 - Complex64::new(1.0, 0.0)).norm() < 1e-15);

    // `Φ†(Z) = (1-γ) Z + γ I` — the adjoint's only fan-out row.
    let got = outputs::<W>(&ch, true, zq);
    assert_eq!(got.len(), 2, "W={W}: Φ†(Z) has two terms");
    assert_eq!(got[0].0, zq);
    assert!((got[0].1 - Complex64::new(1.0 - g, 0.0)).norm() < 1e-15);
    assert_eq!(got[1].0, id);
    assert!((got[1].1 - Complex64::new(g, 0.0)).norm() < 1e-15);

    // X and Y: the one row where Φ and Φ† agree, because `⟨1|X|1⟩` and `⟨0|X|0⟩` both vanish and the K₁ term drops out either way.
    for p in [PauliString::<W>::x(q), PauliString::<W>::y(q)] {
        for adjoint in [false, true] {
            let got = outputs::<W>(&ch, adjoint, p);
            assert_eq!(got.len(), 1, "W={W}: X/Y stay fan-out 1");
            assert_eq!(got[0].0, p);
            assert!((got[0].1 - Complex64::new(s, 0.0)).norm() < 1e-15);
        }
    }
}

/// `apply` is Φ (Schrödinger, fan-out on I) and `apply_adjoint` is Φ† (Heisenberg, unital, fan-out on Z) — W=1.
#[test]
fn forward_is_phi_and_adjoint_is_phi_dagger_w1() {
    check_both_maps::<1>(0, 0.3);
    check_both_maps::<1>(5, 0.75);
}

/// Same, with the support qubit in word 1 at W=2.
#[test]
fn forward_is_phi_and_adjoint_is_phi_dagger_w2() {
    check_both_maps::<2>(70, 0.3);
    check_both_maps::<2>(64, 0.4);
}

/// `⟨Z⟩_after = tr[Φ†(Z) ρ]` with `Φ†(Z) = γ I + (1-γ) Z` gives `⟨Z⟩ = 1` for `ρ = |0⟩⟨0|` (fixed point) and `2γ - 1` for `ρ = |1⟩⟨1|` (decay toward `|0⟩`).
#[test]
fn heisenberg_z_reproduces_the_damped_qubit_expectation() {
    // `⟨b|P|b⟩` for a computational state whose qubits-in-|1⟩ are `ones`:
    // any X or Y factor gives 0, each Z factor contributes `(-1)^bit`.
    let expect = |p: &PauliString<1>, ones: u64| -> f64 {
        if p.x[0] != 0 {
            return 0.0;
        }
        if (p.z[0] & ones).count_ones() % 2 == 1 {
            -1.0
        } else {
            1.0
        }
    };
    for &g in &[0.0, 0.3, 0.5, 1.0] {
        let ch = AmplitudeDamping {
            support: [0],
            gamma: g,
        };
        let terms = outputs::<1>(&ch, true, PauliString::<1>::z(0));
        let ev = |ones: u64| -> f64 {
            terms
                .iter()
                .map(|(p, c)| {
                    assert!(c.im.abs() < 1e-15, "damping keeps coefficients real");
                    c.re * expect(p, ones)
                })
                .sum()
        };
        assert!(
            (ev(0) - 1.0).abs() < 1e-15,
            "gamma={g}: |0⟩ must stay at ⟨Z⟩ = 1, got {}",
            ev(0),
        );
        assert!(
            (ev(1) - (2.0 * g - 1.0)).abs() < 1e-15,
            "gamma={g}: |1⟩ must give ⟨Z⟩ = 2γ-1, got {}",
            ev(1),
        );
    }
}

/// The structural statement: the adjoint's PTM is the forward PTM transposed.
#[test]
fn adjoint_ptm_is_the_transpose_of_the_forward_ptm() {
    for &g in &[0.0, 0.15, 0.5, 0.99, 1.0] {
        let ch = AmplitudeDamping {
            support: [0],
            gamma: g,
        };
        let fwd = ptm4(&ch, false, 0);
        let adj = ptm4(&ch, true, 0);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (adj[i][j] - fwd[j][i]).abs() < 1e-15,
                    "gamma={g}: adj[{i}][{j}]={} vs fwd[{j}][{i}]={}",
                    adj[i][j],
                    fwd[j][i],
                );
            }
        }
    }
}

/// `Φ†` is unital and `Φ` is trace-preserving — transposed statements of each other, both physical requirements.
#[test]
fn adjoint_is_unital_and_forward_is_trace_preserving() {
    for &g in &[0.0, 0.3, 1.0] {
        let ch = AmplitudeDamping {
            support: [0],
            gamma: g,
        };
        // Unitality of the adjoint (Heisenberg) map: I -> I exactly.
        let adj_i = outputs::<1>(&ch, true, PauliString::<1>::identity());
        assert_eq!(adj_i.len(), 1, "gamma={g}");
        assert_eq!(adj_i[0].0, PauliString::<1>::identity());
        assert!((adj_i[0].1 - Complex64::new(1.0, 0.0)).norm() < 1e-15);

        // Trace preservation of the forward map: the I row of its PTM is [1, 0, 0, 0].
        let fwd = ptm4(&ch, false, 0);
        assert!((fwd[0][0] - 1.0).abs() < 1e-15, "gamma={g}");
        for (j, &v) in fwd[0].iter().enumerate().skip(1) {
            assert!(v.abs() < 1e-15, "gamma={g}: fwd[0][{j}] nonzero");
        }
    }
}

#[test]
fn both_directions_at_gamma_zero_are_the_identity_channel() {
    let ch = AmplitudeDamping {
        support: [0],
        gamma: 0.0,
    };
    for adjoint in [false, true] {
        for p in [
            PauliString::<1>::identity(),
            PauliString::<1>::x(0),
            PauliString::<1>::y(0),
            PauliString::<1>::z(0),
        ] {
            let got: Vec<_> = outputs::<1>(&ch, adjoint, p)
                .into_iter()
                .filter(|(_, c)| c.norm() > 1e-15)
                .collect();
            assert_eq!(got.len(), 1, "gamma=0 should not fan out");
            assert_eq!(got[0].0, p);
            assert!((got[0].1 - Complex64::new(1.0, 0.0)).norm() < 1e-15);
        }
    }
}

#[test]
fn forward_respects_a_word_boundary_w2() {
    let g = 0.4;
    let ch = AmplitudeDamping {
        support: [70],
        gamma: g,
    };
    // Φ's fan-out is on the identity, and the new Z lands in word 1.
    let got = outputs::<2>(&ch, false, PauliString::<2>::identity());
    assert_eq!(got.len(), 2);
    assert_eq!(got[1].0, PauliString::<2>::z(70));
    assert!((got[1].1 - Complex64::new(g, 0.0)).norm() < 1e-15);
    // A term on the other side of the boundary is untouched, but qubit 70 is still in the identity sector, so Φ fans it out.
    let other = PauliString::<2>::x(3);
    let got = outputs::<2>(&ch, false, other);
    assert_eq!(got.len(), 2, "q=3 is I on the support, so Φ fans out");
    assert_eq!(got[0].0, other);
    let mut with_z = other;
    assert_eq!(with_z.mul_assign(&PauliString::<2>::z(70)), Phase::ONE);
    assert_eq!(got[1].0, with_z);
}

const TOL: f64 = 1e-12;

/// Identity on the support qubit is preserved exactly — coefficient rescaling does not touch the I sector.
#[test]
fn depolarizing_passes_identity_through() {
    let ch = Depolarizing {
        support: [0],
        p: 0.1,
    };
    let p = PauliString::<1>::identity();
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(1);
    let mut buf = OutputBuffer::<1> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    <Depolarizing as Channel<1>>::apply(&ch, &p.x, &p.z, Complex64::new(2.0, 0.0), &mut buf);
    assert_eq!(len, 1);
    assert_eq!(bx[0], p.x);
    assert_eq!(bz[0], p.z);
    assert!(approx_eq(bc[0], Complex64::new(2.0, 0.0), TOL));
}

/// X, Y, Z on the support qubit each get scaled by `1 - 4p/3`.
#[test]
fn depolarizing_scales_xyz() {
    let p = 0.15;
    let ch = Depolarizing { support: [0], p };
    let scale = 1.0 - 4.0 * p / 3.0;
    for pauli in [
        PauliString::<1>::x(0),
        PauliString::<1>::y(0),
        PauliString::<1>::z(0),
    ] {
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(1);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        <Depolarizing as Channel<1>>::apply(
            &ch,
            &pauli.x,
            &pauli.z,
            Complex64::new(1.0, 0.0),
            &mut buf,
        );
        assert_eq!(len, 1);
        assert_eq!(bx[0], pauli.x);
        assert_eq!(bz[0], pauli.z);
        assert!(approx_eq(bc[0], Complex64::new(scale, 0.0), TOL));
    }
}

/// Off-support qubits are ignored: a Z on qubit 1 with the channel supported on qubit 0 leaves the coefficient untouched (the support qubit is in I-state).
#[test]
fn depolarizing_off_support_is_no_op() {
    let ch = Depolarizing {
        support: [0],
        p: 0.2,
    };
    let p = PauliString::<1>::z(1);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(1);
    let mut buf = OutputBuffer::<1> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    <Depolarizing as Channel<1>>::apply(&ch, &p.x, &p.z, Complex64::new(3.0, 0.0), &mut buf);
    assert_eq!(len, 1);
    assert!(approx_eq(bc[0], Complex64::new(3.0, 0.0), TOL));
}

/// W=2: the scale triggers on a non-identity Pauli at qubit 64, in word 1.
#[test]
fn depolarizing_w2_word_boundary() {
    let p = 0.25;
    let ch = Depolarizing { support: [64], p };
    let scale = 1.0 - 4.0 * p / 3.0;
    // X on qubit 64 → word-1 x-bit set.
    let pauli = PauliString::<2>::x(64);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<2>(1);
    let mut buf = OutputBuffer::<2> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    <Depolarizing as Channel<2>>::apply(
        &ch,
        &pauli.x,
        &pauli.z,
        Complex64::new(1.0, 0.0),
        &mut buf,
    );
    assert_eq!(len, 1);
    assert_eq!(bx[0], pauli.x);
    assert_eq!(bz[0], pauli.z);
    assert!(approx_eq(bc[0], Complex64::new(scale, 0.0), TOL));
}

/// I and Z commute with Z, so dephasing leaves their coefficients alone.
#[test]
fn dephasing_preserves_i_and_z() {
    let ch = Dephasing {
        support: [0],
        p: 0.3,
    };
    for pauli in [PauliString::<1>::identity(), PauliString::<1>::z(0)] {
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(1);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        <Dephasing as Channel<1>>::apply(
            &ch,
            &pauli.x,
            &pauli.z,
            Complex64::new(2.5, 0.0),
            &mut buf,
        );
        assert_eq!(len, 1);
        assert_eq!(bx[0], pauli.x);
        assert_eq!(bz[0], pauli.z);
        assert!(approx_eq(bc[0], Complex64::new(2.5, 0.0), TOL));
    }
}

/// X and Y both anticommute with Z, so dephasing scales them by 1 - 2p.
#[test]
fn dephasing_scales_x_and_y() {
    let p = 0.2;
    let ch = Dephasing { support: [0], p };
    let scale = 1.0 - 2.0 * p;
    for pauli in [PauliString::<1>::x(0), PauliString::<1>::y(0)] {
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(1);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        <Dephasing as Channel<1>>::apply(
            &ch,
            &pauli.x,
            &pauli.z,
            Complex64::new(1.0, 0.0),
            &mut buf,
        );
        assert_eq!(len, 1);
        assert_eq!(bx[0], pauli.x);
        assert_eq!(bz[0], pauli.z);
        assert!(approx_eq(bc[0], Complex64::new(scale, 0.0), TOL));
    }
}

/// The four scale factors at `(px, py, pz) = (0.1, 0.2, 0.3)`: `I → 1`, `X → 0`, `Y → 0.2`, `Z → 0.4`.
#[test]
fn pauli_channel_scales_are_hand_computed_w1() {
    let ch = PauliChannel {
        support: [0],
        px: 0.1,
        py: 0.2,
        pz: 0.3,
    };
    let cases = [
        (PauliString::<1>::identity(), 1.0),
        (PauliString::<1>::x(0), 0.0),
        (PauliString::<1>::y(0), 0.2),
        (PauliString::<1>::z(0), 0.4),
    ];
    for (pauli, want) in cases {
        let got = crate::test_support::raw_outputs::<1, PauliChannel>(
            &ch,
            false,
            pauli,
            Complex64::new(1.0, 0.0),
        );
        assert_eq!(got.len(), 1, "fanout must stay 1");
        assert_eq!(got[0].0, pauli, "the key must be preserved");
        assert!(
            approx_eq(got[0].1, Complex64::new(want, 0.0), TOL),
            "scale for {pauli:?}: got {}, want {want}",
            got[0].1,
        );
    }
}

/// Same four factors with the support in word 1 (qubit 70 at W=2).
#[test]
fn pauli_channel_scales_are_hand_computed_w2() {
    let ch = PauliChannel {
        support: [70],
        px: 0.1,
        py: 0.2,
        pz: 0.3,
    };
    let cases = [
        (PauliString::<2>::identity(), 1.0),
        (PauliString::<2>::x(70), 0.0),
        (PauliString::<2>::y(70), 0.2),
        (PauliString::<2>::z(70), 0.4),
    ];
    for (pauli, want) in cases {
        let got = crate::test_support::raw_outputs::<2, PauliChannel>(
            &ch,
            false,
            pauli,
            Complex64::new(1.0, 0.0),
        );
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, pauli);
        assert!(approx_eq(got[0].1, Complex64::new(want, 0.0), TOL));
    }
}

/// `pauli_channel(p/3, p/3, p/3) ≡ depolarize(p)`: uniform Pauli error is exactly depolarizing noise, so the two channels must agree Pauli for Pauli.
#[test]
fn pauli_channel_at_uniform_probabilities_is_depolarizing() {
    let p = 0.42;
    let pc = PauliChannel {
        support: [0],
        px: p / 3.0,
        py: p / 3.0,
        pz: p / 3.0,
    };
    let dep = Depolarizing { support: [0], p };
    for pauli in [
        PauliString::<1>::identity(),
        PauliString::<1>::x(0),
        PauliString::<1>::y(0),
        PauliString::<1>::z(0),
    ] {
        let c = Complex64::new(1.5, -0.25);
        let a = crate::test_support::outputs::<1, PauliChannel>(&pc, false, pauli, c);
        let b = crate::test_support::outputs::<1, Depolarizing>(&dep, false, pauli, c);
        assert_eq!(a.len(), b.len(), "{pauli:?}");
        for (ta, tb) in a.iter().zip(b.iter()) {
            assert_eq!((ta.0, ta.1), (tb.0, tb.1), "{pauli:?}: keys differ");
            assert!(
                approx_eq(ta.2, tb.2, TOL),
                "{pauli:?}: {} vs {}",
                ta.2,
                tb.2
            );
        }
    }

    let pc2 = PauliChannel {
        support: [64],
        px: p / 3.0,
        py: p / 3.0,
        pz: p / 3.0,
    };
    let dep2 = Depolarizing { support: [64], p };
    for pauli in [
        PauliString::<2>::identity(),
        PauliString::<2>::x(64),
        PauliString::<2>::y(64),
        PauliString::<2>::z(64),
    ] {
        let c = Complex64::new(1.0, 0.0);
        let a = crate::test_support::outputs::<2, PauliChannel>(&pc2, false, pauli, c);
        let b = crate::test_support::outputs::<2, Depolarizing>(&dep2, false, pauli, c);
        assert_eq!(a.len(), b.len(), "{pauli:?}");
        for (ta, tb) in a.iter().zip(b.iter()) {
            assert_eq!((ta.0, ta.1), (tb.0, tb.1));
            assert!(approx_eq(ta.2, tb.2, TOL));
        }
    }
}

/// `pauli_channel(0, 0, p) ≡ dephase(p)`: a pure Z error is dephasing.
#[test]
fn pauli_channel_with_only_pz_is_dephasing() {
    let p = 0.37;
    let pc = PauliChannel {
        support: [1],
        px: 0.0,
        py: 0.0,
        pz: p,
    };
    let deph = Dephasing { support: [1], p };
    for pauli in [
        PauliString::<1>::identity(),
        PauliString::<1>::x(1),
        PauliString::<1>::y(1),
        PauliString::<1>::z(1),
    ] {
        let c = Complex64::new(0.5, 2.0);
        let a = crate::test_support::outputs::<1, PauliChannel>(&pc, false, pauli, c);
        let b = crate::test_support::outputs::<1, Dephasing>(&deph, false, pauli, c);
        assert_eq!(a.len(), b.len(), "{pauli:?}");
        for (ta, tb) in a.iter().zip(b.iter()) {
            assert_eq!((ta.0, ta.1), (tb.0, tb.1));
            assert!(
                approx_eq(ta.2, tb.2, TOL),
                "{pauli:?}: {} vs {}",
                ta.2,
                tb.2
            );
        }
    }
}

/// Off-support qubits are invisible: the support qubit sits in the identity sector, so the coefficient passes through untouched.
#[test]
fn pauli_channel_off_support_is_a_no_op() {
    let ch = PauliChannel {
        support: [0],
        px: 0.1,
        py: 0.2,
        pz: 0.3,
    };
    let pauli = PauliString::<1>::y(5);
    let got = crate::test_support::raw_outputs::<1, PauliChannel>(
        &ch,
        false,
        pauli,
        Complex64::new(3.0, 0.0),
    );
    assert_eq!(got.len(), 1);
    assert!(approx_eq(got[0].1, Complex64::new(3.0, 0.0), TOL));
}

/// A diagonal rescaling is its own adjoint, so the default `apply_adjoint` is correct.
#[test]
fn pauli_channel_is_self_adjoint() {
    let ch = PauliChannel {
        support: [0],
        px: 0.05,
        py: 0.15,
        pz: 0.25,
    };
    for pauli in [
        PauliString::<1>::identity(),
        PauliString::<1>::x(0),
        PauliString::<1>::y(0),
        PauliString::<1>::z(0),
    ] {
        let c = Complex64::new(1.0, -1.0);
        let fwd = crate::test_support::raw_outputs::<1, PauliChannel>(&ch, false, pauli, c);
        let adj = crate::test_support::raw_outputs::<1, PauliChannel>(&ch, true, pauli, c);
        assert_eq!(fwd.len(), adj.len());
        assert_eq!(fwd[0].0, adj[0].0);
        assert!(approx_eq(fwd[0].1, adj[0].1, TOL));
    }
}

/// Key-preserving, so the engine takes the in-place rescale.
#[test]
fn pauli_channel_prepares_as_a_key_preserving_local_ptm() {
    let ch = PauliChannel {
        support: [3],
        px: 0.1,
        py: 0.2,
        pz: 0.3,
    };
    let hash = Gf2Hash::<1>::new(16, 4, 0xC0FFEE);
    let prepared = <PauliChannel as Channel<1>>::prepare(&ch, &hash, false)
        .expect("a weight-1 support must prepare");
    match prepared {
        Prepared::Local(ptm) => assert!(ptm.is_key_preserving()),
        Prepared::Rotation(_) => panic!("expected a local PTM"),
    }
}

/// `p = 0.3`: the scale is `0.68` for all 15 non-identity restrictions, `1` for `I⊗I`.
/// Weight-1 restrictions take the same factor.
#[test]
fn depolarize2_scale_is_hand_computed_w1() {
    let ch = Depolarizing2Q {
        support: [0, 1],
        p: 0.3,
    };
    let local = |q: u32, idx: usize| -> PauliString<1> {
        match idx {
            0 => PauliString::<1>::identity(),
            1 => PauliString::<1>::x(q),
            2 => PauliString::<1>::z(q),
            _ => PauliString::<1>::y(q),
        }
    };
    for a in 0..4 {
        for b in 0..4 {
            let mut pauli = local(0, a);
            let phase = pauli.mul_assign(&local(1, b));
            // Distinct qubits, so the product carries no phase.
            assert_eq!(phase, Phase::ONE);
            let want = if a == 0 && b == 0 { 1.0 } else { 0.68 };
            let got = crate::test_support::raw_outputs::<1, Depolarizing2Q>(
                &ch,
                false,
                pauli,
                Complex64::new(1.0, 0.0),
            );
            assert_eq!(got.len(), 1, "fanout must stay 1");
            assert_eq!(got[0].0, pauli, "the key must be preserved");
            assert!(
                approx_eq(got[0].1, Complex64::new(want, 0.0), TOL),
                "a={a} b={b}: got {}, want {want}",
                got[0].1,
            );
        }
    }
}

/// At `p = 15/16` the scale is exactly zero, so any Pauli touching the pair is annihilated while `I⊗I` is untouched.
#[test]
fn depolarize2_at_fifteen_sixteenths_annihilates_the_pair() {
    let ch = Depolarizing2Q {
        support: [0, 1],
        p: 15.0 / 16.0,
    };
    let mut xz = PauliString::<1>::x(0);
    xz.mul_assign(&PauliString::<1>::z(1));
    let got = crate::test_support::raw_outputs::<1, Depolarizing2Q>(
        &ch,
        false,
        xz,
        Complex64::new(1.0, 0.0),
    );
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].1, Complex64::new(0.0, 0.0), "must be exactly zero");

    let id = PauliString::<1>::identity();
    let got = crate::test_support::raw_outputs::<1, Depolarizing2Q>(
        &ch,
        false,
        id,
        Complex64::new(2.0, 0.0),
    );
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].1, Complex64::new(2.0, 0.0));
}

/// Qubits outside the pair do not trigger the scale.
#[test]
fn depolarize2_ignores_off_support_qubits_w1() {
    let ch = Depolarizing2Q {
        support: [0, 1],
        p: 0.3,
    };
    let pauli = PauliString::<1>::y(9);
    let got = crate::test_support::raw_outputs::<1, Depolarizing2Q>(
        &ch,
        false,
        pauli,
        Complex64::new(3.0, 0.0),
    );
    assert_eq!(got.len(), 1);
    assert!(approx_eq(got[0].1, Complex64::new(3.0, 0.0), TOL));
}

/// W=2 with the pair straddling the 64-bit word boundary (qubits 63 and 64):
/// a Pauli on either half of the pair takes the scale.
#[test]
fn depolarize2_w2_across_a_word_boundary() {
    let ch = Depolarizing2Q {
        support: [63, 64],
        p: 0.3,
    };
    for pauli in [
        PauliString::<2>::x(63),
        PauliString::<2>::z(64),
        PauliString::<2>::y(63),
    ] {
        let got = crate::test_support::raw_outputs::<2, Depolarizing2Q>(
            &ch,
            false,
            pauli,
            Complex64::new(1.0, 0.0),
        );
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, pauli);
        assert!(approx_eq(got[0].1, Complex64::new(0.68, 0.0), TOL));
    }
    // A qubit just outside the pair (62) is not in the support.
    let outside = PauliString::<2>::x(62);
    let got = crate::test_support::raw_outputs::<2, Depolarizing2Q>(
        &ch,
        false,
        outside,
        Complex64::new(1.0, 0.0),
    );
    assert_eq!(got.len(), 1);
    assert!(approx_eq(got[0].1, Complex64::new(1.0, 0.0), TOL));
}

#[test]
fn depolarize2_is_self_adjoint() {
    let ch = Depolarizing2Q {
        support: [2, 5],
        p: 0.2,
    };
    let mut yx = PauliString::<1>::y(2);
    yx.mul_assign(&PauliString::<1>::x(5));
    for pauli in [PauliString::<1>::identity(), yx] {
        let c = Complex64::new(1.0, -1.0);
        let fwd = crate::test_support::raw_outputs::<1, Depolarizing2Q>(&ch, false, pauli, c);
        let adj = crate::test_support::raw_outputs::<1, Depolarizing2Q>(&ch, true, pauli, c);
        assert_eq!(fwd.len(), adj.len());
        assert_eq!(fwd[0].0, adj[0].0);
        assert!(approx_eq(fwd[0].1, adj[0].1, TOL));
    }
}

/// Support weight 2 fits `MAX_LOCAL_SUPPORT`, and the channel is key-preserving, so the engine takes `rescale_in_place` here too.
#[test]
fn depolarize2_prepares_as_a_key_preserving_local_ptm() {
    let ch = Depolarizing2Q {
        support: [1, 4],
        p: 0.3,
    };
    let hash = Gf2Hash::<1>::new(16, 4, 0xC0FFEE);
    let prepared = <Depolarizing2Q as Channel<1>>::prepare(&ch, &hash, false)
        .expect("a weight-2 support must prepare");
    match prepared {
        Prepared::Local(ptm) => {
            assert_eq!(ptm.k(), 2);
            assert!(ptm.is_key_preserving());
        }
        Prepared::Rotation(_) => panic!("expected a local PTM"),
    }
}

/// W=2: dephasing on qubit 64 scales an X@64 by 1 - 2p.
#[test]
fn dephasing_w2_word_boundary() {
    let p = 0.4;
    let ch = Dephasing { support: [64], p };
    let scale = 1.0 - 2.0 * p;
    let pauli = PauliString::<2>::x(64);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<2>(1);
    let mut buf = OutputBuffer::<2> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    <Dephasing as Channel<2>>::apply(&ch, &pauli.x, &pauli.z, Complex64::new(1.0, 0.0), &mut buf);
    assert_eq!(len, 1);
    assert!(approx_eq(bc[0], Complex64::new(scale, 0.0), TOL));
}

/// The adjoint map fixes I on the support with fanout 1 (unitality); `apply` instead fans I out to `I + γ Z`.
#[test]
fn amplitude_damping_adjoint_passes_identity_through() {
    let ch = AmplitudeDamping {
        support: [0],
        gamma: 0.3,
    };
    let p = PauliString::<1>::identity();
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
    let mut buf = OutputBuffer::<1> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    <AmplitudeDamping as Channel<1>>::apply_adjoint(
        &ch,
        &p.x,
        &p.z,
        Complex64::new(2.0, 0.0),
        &mut buf,
    );
    assert_eq!(len, 1);
    assert_eq!(bx[0], p.x);
    assert_eq!(bz[0], p.z);
    assert!(approx_eq(bc[0], Complex64::new(2.0, 0.0), TOL));
}

/// X and Y on the support each get scaled by √(1-γ), fanout 1.
#[test]
fn amplitude_damping_scales_x_and_y_by_sqrt() {
    let gamma = 0.2;
    let ch = AmplitudeDamping {
        support: [0],
        gamma,
    };
    let scale = (1.0 - gamma).sqrt();
    for pauli in [PauliString::<1>::x(0), PauliString::<1>::y(0)] {
        let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
        let mut buf = OutputBuffer::<1> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        <AmplitudeDamping as Channel<1>>::apply(
            &ch,
            &pauli.x,
            &pauli.z,
            Complex64::new(1.0, 0.0),
            &mut buf,
        );
        assert_eq!(len, 1);
        assert_eq!(bx[0], pauli.x);
        assert_eq!(bz[0], pauli.z);
        assert!(approx_eq(bc[0], Complex64::new(scale, 0.0), TOL));
    }
}

/// The adjoint map fans Z on the support out to `(1-γ)·Z` first, then `γ·I` with the z-bit cleared.
#[test]
fn amplitude_damping_adjoint_z_fans_out_to_z_plus_i() {
    let gamma = 0.25;
    let ch = AmplitudeDamping {
        support: [0],
        gamma,
    };
    let p = PauliString::<1>::z(0);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<1>(2);
    let mut buf = OutputBuffer::<1> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    <AmplitudeDamping as Channel<1>>::apply_adjoint(
        &ch,
        &p.x,
        &p.z,
        Complex64::new(1.0, 0.0),
        &mut buf,
    );
    assert_eq!(len, 2);
    // First: (1-γ)·Z (z-bit kept).
    assert_eq!(bx[0], p.x);
    assert_eq!(bz[0], p.z);
    assert!(approx_eq(bc[0], Complex64::new(1.0 - gamma, 0.0), TOL));
    // Second: γ·I (z-bit cleared).
    let id = PauliString::<1>::identity();
    assert_eq!(bx[1], id.x);
    assert_eq!(bz[1], id.z);
    assert!(approx_eq(bc[1], Complex64::new(gamma, 0.0), TOL));
}

/// W=2, adjoint map: Z on qubit 64 fans out to (1-γ)·Z@64 + γ·I, with the I term's z-bit cleared in word 1 only.
#[test]
fn amplitude_damping_adjoint_w2_word_boundary() {
    let gamma = 0.4;
    let ch = AmplitudeDamping {
        support: [64],
        gamma,
    };
    let p = PauliString::<2>::z(64);
    let (mut bx, mut bz, mut bc, mut len) = alloc_bufs::<2>(2);
    let mut buf = OutputBuffer::<2> {
        x: &mut bx,
        z: &mut bz,
        coeff: &mut bc,
        len: &mut len,
    };
    <AmplitudeDamping as Channel<2>>::apply_adjoint(
        &ch,
        &p.x,
        &p.z,
        Complex64::new(1.0, 0.0),
        &mut buf,
    );
    assert_eq!(len, 2);
    assert_eq!(bx[0], p.x);
    assert_eq!(bz[0], p.z);
    assert!(approx_eq(bc[0], Complex64::new(1.0 - gamma, 0.0), TOL));
    // I term: word 1 z-bit cleared.
    assert_eq!(bx[1], [0u64, 0u64]);
    assert_eq!(bz[1], [0u64, 0u64]);
    assert!(approx_eq(bc[1], Complex64::new(gamma, 0.0), TOL));
}
