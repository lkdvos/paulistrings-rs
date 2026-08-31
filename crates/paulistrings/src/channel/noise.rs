//! Noise channels: Depolarizing, Dephasing, PauliChannel, Depolarizing2Q,
//! AmplitudeDamping. See ARCHITECTURE.md §Channels.

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

/// A general single-qubit Pauli channel with independent error probabilities.
///
/// `E(ρ) = (1-px-py-pz)ρ + px·XρX + py·YρY + pz·ZρZ`. Like [`Depolarizing`] and
/// [`Dephasing`] this is a pure coefficient rescaling in the Heisenberg picture
/// — fanout 1, key-preserving, self-adjoint — so the engine takes the in-place
/// rescale path (`engine/bucketed.rs::rescale_in_place`) rather than a
/// gather/sort/merge.
///
/// # Dual scales
///
/// Each Pauli anticommutes with exactly the other two, so `P_k Q P_k = ±Q` with
/// the sign negative for the two terms that anticommute with `Q`:
///
/// - `I → 1` (the four probabilities sum to one, so `E†` is unital)
/// - `X → 1 - 2(py + pz)`
/// - `Y → 1 - 2(px + pz)`
/// - `Z → 1 - 2(px + py)`
///
/// The two consistency checks worth remembering: `(p/3, p/3, p/3)` reproduces
/// `Depolarizing { p }` (`1 - 4p/3` on every non-identity Pauli) and `(0, 0, p)`
/// reproduces `Dephasing { p }` (`1 - 2p` on X and Y, `1` on Z).
///
/// # Examples
///
/// ```
/// use paulistrings::channel::PauliChannel;
/// let ch = PauliChannel { support: [3], px: 0.01, py: 0.02, pz: 0.03 };
/// # let _ = ch;
/// ```
pub struct PauliChannel {
    /// The single qubit this channel acts on.
    pub support: [u32; 1],
    /// Probability of an `X` error.
    pub px: f64,
    /// Probability of a `Y` error.
    pub py: f64,
    /// Probability of a `Z` error.
    pub pz: f64,
}

impl<const W: usize> Channel<W> for PauliChannel {
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
        let q = self.support[0] as usize;
        debug_assert!(q < 64 * W);
        let (word, bit, _mask) = qubit_loc(q);
        // Packed local Pauli index: `I=0, X=1, Z=2, Y=3` — note Z and Y are not
        // in alphabet order, so the table is written out rather than computed.
        let idx = read_pauli(input_x, input_z, word, bit);
        let s = match idx {
            0 => 1.0,
            1 => 1.0 - 2.0 * (self.py + self.pz),
            2 => 1.0 - 2.0 * (self.px + self.py),
            3 => 1.0 - 2.0 * (self.px + self.pz),
            _ => unreachable!(),
        };
        out.push(*input_x, *input_z, coeff * s);
    }
}

/// Uniform two-qubit depolarizing noise: probability `p` spread evenly over the
/// 15 non-identity two-qubit Paulis.
///
/// `E(ρ) = (1-p)ρ + (p/15)·Σ_k P_k ρ P_k`. Fanout 1, key-preserving,
/// self-adjoint, like its single-qubit siblings; the support weight of 2 is
/// exactly [`MAX_LOCAL_SUPPORT`](super::prepared::MAX_LOCAL_SUPPORT), so the
/// default `prepare` derivation applies.
///
/// # Dual scales
///
/// For a `Q` that is the identity on both support qubits, every `P_k Q P_k = Q`
/// and the scale is 1. Otherwise exactly 8 of the 16 two-qubit Paulis
/// anticommute with `Q`, so among the 15 error terms 7 commute and 8
/// anticommute:
///
/// `(1-p)Q + (p/15)(7 - 8)Q = (1 - 16p/15)·Q`.
///
/// Note this is the same factor whether `Q` is non-identity on one support
/// qubit or on both — the count of anticommuting two-qubit Paulis does not
/// depend on `Q`'s weight.
///
/// # Examples
///
/// ```
/// use paulistrings::channel::Depolarizing2Q;
/// let ch = Depolarizing2Q { support: [3, 4], p: 0.01 };
/// # let _ = ch;
/// ```
pub struct Depolarizing2Q {
    /// The two qubits this channel acts on. They must differ — an overlapping
    /// pair would declare a two-qubit support over one qubit, which the engine's
    /// local-PTM derivation would then mis-tabulate.
    pub support: [u32; 2],
    /// Error probability `p ∈ [0, 1]`.
    pub p: f64,
}

impl<const W: usize> Channel<W> for Depolarizing2Q {
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
        debug_assert!(
            self.support[0] != self.support[1],
            "Depolarizing2Q support qubits must differ (both are {})",
            self.support[0]
        );
        let mut touches_pair = false;
        for &q in &self.support {
            let q = q as usize;
            debug_assert!(q < 64 * W);
            let (word, bit, _mask) = qubit_loc(q);
            touches_pair |= read_pauli(input_x, input_z, word, bit) != 0;
        }
        let s = if touches_pair {
            1.0 - 16.0 * self.p / 15.0
        } else {
            1.0
        };
        out.push(*input_x, *input_z, coeff * s);
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
    use crate::bucket::hash::Gf2Hash;
    use crate::channel::prepared::Prepared;
    use crate::pauli_string::PauliString;
    use crate::phase::Phase;
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

    // ---- PauliChannel ----
    //
    // Dual scales, hand-derived: each Pauli anticommutes with exactly the other
    // two, so `X → 1 - 2(py + pz)`, `Y → 1 - 2(px + pz)`, `Z → 1 - 2(px + py)`,
    // and `I → 1` because the four Kraus probabilities sum to one.

    /// The four scale factors at `(px, py, pz) = (0.1, 0.2, 0.3)`, each computed
    /// by hand: `I → 1`, `X → 1 - 2(0.5) = 0`, `Y → 1 - 2(0.4) = 0.2`,
    /// `Z → 1 - 2(0.3) = 0.4`.
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

    /// `pauli_channel(p/3, p/3, p/3) ≡ depolarize(p)`: uniform Pauli error is
    /// exactly depolarizing noise, so the two channels must agree Pauli for
    /// Pauli. Checked at both widths.
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

    /// Off-support qubits are invisible: the support qubit sits in the identity
    /// sector, so the coefficient passes through untouched.
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

    /// A diagonal rescaling is its own Hilbert-Schmidt adjoint, so the default
    /// `apply_adjoint` is correct — pinned rather than assumed.
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

    /// Key-preserving fanout-1, so the engine takes `rescale_in_place`: no
    /// gather, no sort, no merge. Losing this would be a silent slowdown.
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

    // ---- Depolarizing2Q ----
    //
    // Uniform two-qubit depolarizing, probability `p` spread over the 15
    // non-identity two-qubit Paulis. Dual scale, hand-derived: for a Pauli `Q`
    // that is non-identity on the pair, exactly 8 of the 16 two-qubit Paulis
    // anticommute with it, so 7 of the 15 error terms commute and 8 anticommute:
    // `(1-p)Q + (p/15)(7 - 8)Q = (1 - 16p/15) Q`. `I⊗I` is fixed.

    /// `p = 0.3`: the scale is `1 - 16(0.3)/15 = 1 - 0.32 = 0.68` for all 15
    /// non-identity restrictions, and exactly 1 for `I⊗I`. Enumerated over the
    /// full 4x4 local basis so the weight-1 restrictions are covered too — they
    /// take the *same* factor, which is the easy thing to get wrong.
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

    /// At `p = 15/16` the scale is exactly zero (`1 - 16·(15/16)/15 = 0`, and
    /// every step is exact in binary floating point), so any Pauli touching the
    /// pair is annihilated while `I⊗I` is untouched.
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

    /// Support weight 2 fits `MAX_LOCAL_SUPPORT`, and the channel is
    /// key-preserving, so the engine takes `rescale_in_place` here too.
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
