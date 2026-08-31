//! Noise channels: Depolarizing, Dephasing, AmplitudeDamping. See
//! ARCHITECTURE.md §Channels.

use super::{qubit_loc, read_pauli, set_bit, support_mask, Channel, OutputBuffer};
use num_complex::Complex64;

/// Shared body of `Depolarizing::apply` and `Dephasing::apply`: both are pure
/// coefficient rescalings on the support qubit that leave the key unchanged.
/// They differ only in which local Pauli indices (`I=0, X=1, Z=2, Y=3`) are
/// `affected` and in the `scale` applied to those — read the support qubit's
/// packed Pauli index once and push the (possibly) rescaled coefficient.
#[inline]
fn rescale_on_support<const W: usize>(
    support: u32,
    scale: f64,
    affected: impl FnOnce(usize) -> bool,
    input_x: &[u64; W],
    input_z: &[u64; W],
    coeff: Complex64,
    out: &mut OutputBuffer<'_, W>,
) {
    let q = support as usize;
    debug_assert!(q < 64 * W);
    let (word, bit, _mask) = qubit_loc(q);
    let idx = read_pauli(input_x, input_z, word, bit);
    let s = if affected(idx) { scale } else { 1.0 };
    out.push(*input_x, *input_z, coeff * s);
}

/// Single-qubit depolarizing noise with error probability `p`.
///
/// In the Heisenberg picture this is just a coefficient rescaling: the
/// identity on the support qubit is preserved unchanged; every non-identity
/// Pauli on the support is multiplied by `1 - 4p/3`. Self-adjoint, so the
/// default `apply_adjoint` from the trait is correct.
///
/// # Examples
///
/// ```
/// use paulistrings::channel::Depolarizing;
/// let ch = Depolarizing { support: [3], p: 0.05 };
/// # let _ = ch;
/// ```
pub struct Depolarizing {
    /// The single qubit this channel acts on.
    pub support: [u32; 1],
    /// Error probability `p ∈ [0, 1]`.
    pub p: f64,
}

impl<const W: usize> Channel<W> for Depolarizing {
    #[inline]
    fn max_fanout(&self) -> usize {
        1
    }

    #[inline]
    fn support(&self) -> [u64; W] {
        support_mask(&self.support)
    }

    #[inline]
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        rescale_on_support(
            self.support[0],
            1.0 - 4.0 * self.p / 3.0,
            |idx| idx != 0,
            input_x,
            input_z,
            coeff,
            out,
        );
    }
}

/// Single-qubit dephasing noise with error probability `p`.
///
/// Heisenberg dual: `E*(P) = (1-p) P + p Z P Z`. So I and Z are preserved;
/// X and Y are scaled by `1 - 2p` (`Z·X·Z = -X`, `Z·Y·Z = -Y`). Equivalently,
/// the scale fires iff the support qubit's `x_bit` is set. Self-adjoint.
pub struct Dephasing {
    /// The single qubit this channel acts on.
    pub support: [u32; 1],
    /// Error probability `p ∈ [0, 1]`.
    pub p: f64,
}

impl<const W: usize> Channel<W> for Dephasing {
    #[inline]
    fn max_fanout(&self) -> usize {
        1
    }

    #[inline]
    fn support(&self) -> [u64; W] {
        support_mask(&self.support)
    }

    #[inline]
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        rescale_on_support(
            self.support[0],
            1.0 - 2.0 * self.p,
            |idx| idx & 1 == 1,
            input_x,
            input_z,
            coeff,
            out,
        );
    }
}

/// Single-qubit amplitude damping with parameter `gamma`.
///
/// The only noise in the built-in set with genuine fan-out > 1. Heisenberg
/// dual via Kraus operators `E_0 = |0⟩⟨0| + √(1-γ)|1⟩⟨1|` and
/// `E_1 = √γ |0⟩⟨1|`:
///
/// - `I → I`
/// - `X → √(1-γ) X`
/// - `Y → √(1-γ) Y`
/// - `Z → (1-γ) Z + γ I`   (the only fanout-2 case)
///
/// Not self-adjoint. `apply` above is the Heisenberg map `Φ†`; the adjoint is
/// `Φ` itself, obtained by transposing the Pauli-transfer matrix (all four
/// Paulis share a norm, so the Gram matrix is a multiple of the identity and the
/// adjoint is the plain transpose):
///
/// - `I → I + γ Z`   (now the only fanout-2 case)
/// - `X → √(1-γ) X`
/// - `Y → √(1-γ) Y`
/// - `Z → (1-γ) Z`
///
/// Note the fan-out moves from `Z` to `I`. Structurally: `Φ†` is unital
/// (`Φ†(I) = I`, because `Φ` is trace-preserving), and `Φ` is trace-preserving
/// (the `I` coefficient of `Φ(P)` depends only on the `I` coefficient of `P`),
/// which are transposed statements of each other.
pub struct AmplitudeDamping {
    /// The single qubit this channel acts on.
    pub support: [u32; 1],
    /// Damping parameter `γ ∈ [0, 1]`. The amplitude that escapes to the
    /// `|0⟩` state per application.
    pub gamma: f64,
}

impl<const W: usize> Channel<W> for AmplitudeDamping {
    #[inline]
    fn max_fanout(&self) -> usize {
        2
    }

    #[inline]
    fn support(&self) -> [u64; W] {
        support_mask(&self.support)
    }

    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        let q = self.support[0] as usize;
        debug_assert!(q < 64 * W);
        let (word, bit, mask) = qubit_loc(q);
        let idx = read_pauli(input_x, input_z, word, bit);
        match idx {
            0 => {
                // I → I.
                out.push(*input_x, *input_z, coeff);
            }
            1 | 3 => {
                // X or Y → √(1-γ) · same.
                let scale = (1.0 - self.gamma).sqrt();
                out.push(*input_x, *input_z, coeff * scale);
            }
            2 => {
                // Z → (1-γ) Z + γ I. Emit Z first (matches the order in the
                // doc-comment), then I (with the support's z-bit cleared).
                out.push(*input_x, *input_z, coeff * (1.0 - self.gamma));
                let mut nz = *input_z;
                set_bit(&mut nz, word, mask, false);
                out.push(*input_x, nz, coeff * self.gamma);
            }
            _ => unreachable!(),
        }
    }

    /// The Hilbert-Schmidt adjoint of [`Self::apply`] — the transpose of its
    /// Pauli-transfer matrix. See the type's documentation for the derivation.
    fn apply_adjoint(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        let q = self.support[0] as usize;
        debug_assert!(q < 64 * W);
        let (word, bit, mask) = qubit_loc(q);
        let idx = read_pauli(input_x, input_z, word, bit);
        match idx {
            0 => {
                // I → I + γ Z. The fan-out sits here in the adjoint, where the
                // forward map had it on Z.
                out.push(*input_x, *input_z, coeff);
                let mut nz = *input_z;
                set_bit(&mut nz, word, mask, true);
                out.push(*input_x, nz, coeff * self.gamma);
            }
            1 | 3 => {
                // X or Y → √(1-γ) · same, as in the forward map.
                let scale = (1.0 - self.gamma).sqrt();
                out.push(*input_x, *input_z, coeff * scale);
            }
            2 => {
                // Z → (1-γ) Z, with no I component.
                out.push(*input_x, *input_z, coeff * (1.0 - self.gamma));
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli_string::PauliString;
    use crate::test_support::{alloc_bufs, approx_eq};

    // ---- AmplitudeDamping adjoint ----
    //
    // These pin the real adjoint: the transpose of the forward
    // Pauli-transfer matrix. `apply_adjoint = apply` would be a silent trap
    // for Heisenberg backpropagation.

    /// Collect the outputs of `apply` or `apply_adjoint` on one input.
    ///
    /// Emission order, zeros included — these tests assert on row *positions*,
    /// so the normalizing `test_support::outputs` would not do.
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

    /// Build the 4x4 single-qubit PTM on the support, `t[out][in]`, using the
    /// `I=0, X=1, Z=2, Y=3` packing.
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
        // `j` is the *column* (input basis element) while the row is computed
        // from each output, so there is nothing to iterate over here.
        #[allow(clippy::needless_range_loop)]
        for j in 0..4 {
            for (out_p, c) in outputs::<1>(ch, adjoint, basis(j)) {
                t[index_of(&out_p)][j] += c.re;
            }
        }
        t
    }

    #[test]
    fn adjoint_maps_identity_to_i_plus_gamma_z() {
        let g = 0.3;
        let ch = AmplitudeDamping {
            support: [0],
            gamma: g,
        };
        let got = outputs::<1>(&ch, true, PauliString::<1>::identity());
        assert_eq!(got.len(), 2, "I should fan out to two terms in the adjoint");
        assert_eq!(got[0].0, PauliString::<1>::identity());
        assert!((got[0].1 - Complex64::new(1.0, 0.0)).norm() < 1e-15);
        assert_eq!(got[1].0, PauliString::<1>::z(0));
        assert!((got[1].1 - Complex64::new(g, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn adjoint_scales_x_and_y_by_sqrt_one_minus_gamma() {
        let g = 0.3;
        let ch = AmplitudeDamping {
            support: [0],
            gamma: g,
        };
        let s = (1.0f64 - g).sqrt();
        for p in [PauliString::<1>::x(0), PauliString::<1>::y(0)] {
            let got = outputs::<1>(&ch, true, p);
            assert_eq!(got.len(), 1);
            assert_eq!(got[0].0, p);
            assert!((got[0].1 - Complex64::new(s, 0.0)).norm() < 1e-15);
        }
    }

    #[test]
    fn adjoint_maps_z_to_one_minus_gamma_z_with_no_identity_part() {
        let g = 0.3;
        let ch = AmplitudeDamping {
            support: [0],
            gamma: g,
        };
        let got = outputs::<1>(&ch, true, PauliString::<1>::z(0));
        assert_eq!(got.len(), 1, "the adjoint's Z row has no I component");
        assert_eq!(got[0].0, PauliString::<1>::z(0));
        assert!((got[0].1 - Complex64::new(1.0 - g, 0.0)).norm() < 1e-15);
    }

    /// The structural statement: the adjoint's PTM is the forward PTM
    /// transposed. This is the property the fix is *for*, and it fails outright
    /// under the old `apply_adjoint = apply` default.
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

    /// `Φ†` is unital and `Φ` is trace-preserving — transposed statements of
    /// each other, and both are physical requirements.
    #[test]
    fn forward_is_unital_and_adjoint_is_trace_preserving() {
        for &g in &[0.0, 0.3, 1.0] {
            let ch = AmplitudeDamping {
                support: [0],
                gamma: g,
            };
            // Unitality of the forward (Heisenberg) map: I -> I exactly.
            let fwd_i = outputs::<1>(&ch, false, PauliString::<1>::identity());
            assert_eq!(fwd_i.len(), 1, "gamma={g}");
            assert_eq!(fwd_i[0].0, PauliString::<1>::identity());
            assert!((fwd_i[0].1 - Complex64::new(1.0, 0.0)).norm() < 1e-15);

            // Trace preservation of the adjoint: the I row of its PTM is
            // [1, 0, 0, 0], i.e. the I component of the output depends only on
            // the I component of the input, with unit weight.
            let adj = ptm4(&ch, true, 0);
            assert!((adj[0][0] - 1.0).abs() < 1e-15, "gamma={g}");
            for (j, &v) in adj[0].iter().enumerate().skip(1) {
                assert!(v.abs() < 1e-15, "gamma={g}: adj[0][{j}] nonzero");
            }
        }
    }

    #[test]
    fn adjoint_at_gamma_zero_is_the_identity_channel() {
        let ch = AmplitudeDamping {
            support: [0],
            gamma: 0.0,
        };
        for p in [
            PauliString::<1>::identity(),
            PauliString::<1>::x(0),
            PauliString::<1>::y(0),
            PauliString::<1>::z(0),
        ] {
            let got: Vec<_> = outputs::<1>(&ch, true, p)
                .into_iter()
                .filter(|(_, c)| c.norm() > 1e-15)
                .collect();
            assert_eq!(got.len(), 1, "gamma=0 should not fan out");
            assert_eq!(got[0].0, p);
            assert!((got[0].1 - Complex64::new(1.0, 0.0)).norm() < 1e-15);
        }
    }

    #[test]
    fn adjoint_respects_a_word_boundary_w2() {
        let g = 0.4;
        let ch = AmplitudeDamping {
            support: [70],
            gamma: g,
        };
        let got = outputs::<2>(&ch, true, PauliString::<2>::identity());
        assert_eq!(got.len(), 2);
        assert_eq!(got[1].0, PauliString::<2>::z(70));
        assert!((got[1].1 - Complex64::new(g, 0.0)).norm() < 1e-15);
        // A term on the other side of the boundary is untouched.
        let other = PauliString::<2>::x(3);
        let got = outputs::<2>(&ch, true, other);
        assert_eq!(got.len(), 2, "q=3 is I on the support, so it fans out");
        assert_eq!(got[0].0, other);
    }

    const TOL: f64 = 1e-12;

    /// Identity on the support qubit is preserved exactly — coefficient
    /// rescaling does not touch the I sector.
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

    /// Off-support qubits are ignored: a Z on qubit 1 with the channel
    /// supported on qubit 0 leaves the coefficient untouched (the support
    /// qubit is in I-state).
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

    /// W=2: support qubit lives in word 1 (qubit 64+). The scale must
    /// trigger when bits at qubit 64 are non-identity.
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
        <Dephasing as Channel<2>>::apply(
            &ch,
            &pauli.x,
            &pauli.z,
            Complex64::new(1.0, 0.0),
            &mut buf,
        );
        assert_eq!(len, 1);
        assert!(approx_eq(bc[0], Complex64::new(scale, 0.0), TOL));
    }

    /// I on the support is fixed by amplitude damping (fanout 1, coeff
    /// preserved).
    #[test]
    fn amplitude_damping_passes_identity_through() {
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
        <AmplitudeDamping as Channel<1>>::apply(
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

    /// Z on the support fans out to `(1-γ)·Z + γ·I`. The first emit is the
    /// Z term, the second is the I term (z-bit cleared on the support
    /// qubit).
    #[test]
    fn amplitude_damping_z_fans_out_to_z_plus_i() {
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
        <AmplitudeDamping as Channel<1>>::apply(
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

    /// W=2: Z on qubit 64 fans out to (1-γ)·Z@64 + γ·I, with the I term's
    /// z-bit cleared in word 1 only.
    #[test]
    fn amplitude_damping_w2_word_boundary() {
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
        <AmplitudeDamping as Channel<2>>::apply(
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
}
