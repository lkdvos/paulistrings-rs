use super::*;
use crate::pauli_string::PauliString;
use crate::phase::Phase;
use crate::test_support::{approx_eq, raw_outputs};

/// Collect the outputs of `apply` or `apply_adjoint` on one input.
///
/// Sorted by key, so these tests can assert on row positions.
fn outputs<const W: usize>(
    channel: &PtmChannel,
    adjoint: bool,
    p: PauliString<W>,
) -> Vec<(PauliString<W>, Complex64)> {
    let mut rows = raw_outputs::<W, PtmChannel>(channel, adjoint, p, Complex64::new(1.0, 0.0));
    rows.sort_by_key(|(key, _)| *key);
    rows
}

/// Both maps of a damping channel on qubit `q`, row by row.
fn check_both_maps<const W: usize>(q: usize, g: f64) {
    let channel = PtmChannel::amplitude_damping(q, g);
    let s = (1.0f64 - g).sqrt();
    let id = PauliString::<W>::identity();
    let zq = PauliString::<W>::z(q);

    // Φ = `apply`: the fan-out is on I. `Φ(I) = I + γ Z`.
    let got = outputs::<W>(&channel, false, id);
    assert_eq!(got.len(), 2, "W={W}: Φ(I) has two terms");
    assert_eq!(got[0].0, id);
    assert!((got[0].1 - Complex64::new(1.0, 0.0)).norm() < 1e-15);
    assert_eq!(got[1].0, zq);
    assert!((got[1].1 - Complex64::new(g, 0.0)).norm() < 1e-15);

    // `Φ(Z) = (1-γ) Z`, with no identity component: Φ preserves trace and `tr Z = 0`, so the I coefficient must vanish.
    let got = outputs::<W>(&channel, false, zq);
    assert_eq!(got.len(), 1, "W={W}: Φ(Z) has no I component");
    assert_eq!(got[0].0, zq);
    assert!((got[0].1 - Complex64::new(1.0 - g, 0.0)).norm() < 1e-15);

    // Φ† = `apply_adjoint`: unital, so `Φ†(I) = I` exactly, fan-out 1.
    let got = outputs::<W>(&channel, true, id);
    assert_eq!(got.len(), 1, "W={W}: Φ†(I) must be I alone (unitality)");
    assert_eq!(got[0].0, id);
    assert!((got[0].1 - Complex64::new(1.0, 0.0)).norm() < 1e-15);

    // `Φ†(Z) = (1-γ) Z + γ I` — the adjoint's only fan-out row.
    let got = outputs::<W>(&channel, true, zq);
    assert_eq!(got.len(), 2, "W={W}: Φ†(Z) has two terms");
    assert_eq!(got[0].0, id);
    assert!((got[0].1 - Complex64::new(g, 0.0)).norm() < 1e-15);
    assert_eq!(got[1].0, zq);
    assert!((got[1].1 - Complex64::new(1.0 - g, 0.0)).norm() < 1e-15);

    // X and Y: the one row where Φ and Φ† agree, because `⟨1|X|1⟩` and `⟨0|X|0⟩` both vanish and the K₁ term drops out either way.
    for p in [PauliString::<W>::x(q), PauliString::<W>::y(q)] {
        for adjoint in [false, true] {
            let got = outputs::<W>(&channel, adjoint, p);
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

/// The diagonal constructors keep the key and scale by `1 - 4p/3` (depolarizing), `1 - 2p` on X and Y (dephasing), `(1, 0, 0.2, 0.4)` at `(px, py, pz) = (0.1, 0.2, 0.3)` (Pauli channel) and `0.68` off `I⊗I` at `p = 0.3` (two-qubit depolarizing).
#[test]
fn diagonal_channels_scale_by_the_hand_computed_factor() {
    let (i, x, y, z) = (
        PauliString::<1>::identity(),
        PauliString::<1>::x(0),
        PauliString::<1>::y(0),
        PauliString::<1>::z(0),
    );
    let mut cases = vec![
        (PtmChannel::depolarizing(0, 0.15), x, 0.8),
        (PtmChannel::depolarizing(0, 0.15), y, 0.8),
        (PtmChannel::depolarizing(0, 0.15), z, 0.8),
        (PtmChannel::dephasing(0, 0.2), i, 1.0),
        (PtmChannel::dephasing(0, 0.2), x, 0.6),
        (PtmChannel::dephasing(0, 0.2), y, 0.6),
        (PtmChannel::dephasing(0, 0.2), z, 1.0),
        (PtmChannel::pauli_channel(0, 0.1, 0.2, 0.3), i, 1.0),
        (PtmChannel::pauli_channel(0, 0.1, 0.2, 0.3), x, 0.0),
        (PtmChannel::pauli_channel(0, 0.1, 0.2, 0.3), y, 0.2),
        (PtmChannel::pauli_channel(0, 0.1, 0.2, 0.3), z, 0.4),
    ];
    let local = |q: usize, index: usize| match index {
        0 => PauliString::<1>::identity(),
        1 => PauliString::<1>::x(q),
        2 => PauliString::<1>::z(q),
        _ => PauliString::<1>::y(q),
    };
    for a in 0..4 {
        for b in 0..4 {
            let mut pauli = local(0, a);
            assert_eq!(pauli.mul_assign(&local(1, b)), Phase::ONE);
            let want = if a == 0 && b == 0 { 1.0 } else { 0.68 };
            cases.push((PtmChannel::depolarizing_2q(0, 1, 0.3), pauli, want));
        }
    }
    for (channel, pauli, want) in cases {
        let got = raw_outputs::<1, PtmChannel>(&channel, false, pauli, Complex64::new(1.0, 0.0));
        assert_eq!(got.len(), 1, "{channel:?} on {pauli:?}: fanout must stay 1");
        assert_eq!(got[0].0, pauli, "{channel:?}: the key must be preserved");
        assert!(
            approx_eq(got[0].1, Complex64::new(want, 0.0), 1e-12),
            "{channel:?} on {pauli:?}: got {}, want {want}",
            got[0].1,
        );
    }
}
