//! Noise channels: Depolarizing, Dephasing, PauliChannel, Depolarizing2Q,
//! AmplitudeDamping. See ARCHITECTURE.md §Channels.

use super::{qubit_loc, read_pauli, set_bit, support_mask, Channel, OutputBuffer};
use num_complex::Complex64;

/// Shared body of `Depolarizing::apply` and `Dephasing::apply`: both are pure coefficient rescalings on the support qubit that leave the key unchanged.
/// They differ only in which local Pauli indices (`I=0, X=1, Z=2, Y=3`) are `affected` and in the `scale` applied to those.
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
/// In the Heisenberg picture this is just a coefficient rescaling: the identity on the support qubit is preserved unchanged, every non-identity Pauli on the support is multiplied by `1 - 4p/3`.
/// Self-adjoint, so the default `apply_adjoint` from the trait is correct.
///
/// # Examples
///
/// ```
/// use paulistrings::Depolarizing;
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
/// Heisenberg dual: `E*(P) = (1-p) P + p Z P Z`, so I and Z are preserved and X and Y are scaled by `1 - 2p` (`Z·X·Z = -X`, `Z·Y·Z = -Y`); equivalently, the scale fires iff the support qubit's `x_bit` is set. Self-adjoint.
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
/// `E(ρ) = (1-px-py-pz)ρ + px·XρX + py·YρY + pz·ZρZ`.
/// Like [`Depolarizing`] and [`Dephasing`] this is a pure coefficient rescaling in the Heisenberg picture (fanout 1, key-preserving, self-adjoint), so the engine takes the in-place rescale path (`engine/bucketed.rs::rescale_in_place`) rather than a gather/sort/merge.
///
/// # Dual scales
///
/// Each Pauli anticommutes with exactly the other two, so `P_k Q P_k = ±Q` with the sign negative for the two terms that anticommute with `Q`: `I → 1` (the probabilities sum to one, so `E†` is unital), `X → 1 - 2(py + pz)`, `Y → 1 - 2(px + pz)`, `Z → 1 - 2(px + py)`.
/// `(p/3, p/3, p/3)` reproduces `Depolarizing { p }` and `(0, 0, p)` reproduces `Dephasing { p }`.
///
/// # Examples
///
/// ```
/// use paulistrings::PauliChannel;
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

/// Uniform two-qubit depolarizing noise: probability `p` spread evenly over the 15 non-identity two-qubit Paulis.
///
/// `E(ρ) = (1-p)ρ + (p/15)·Σ_k P_k ρ P_k`. Fanout 1, key-preserving, self-adjoint, like its single-qubit siblings; the support weight of 2 is exactly `MAX_LOCAL_SUPPORT`, so the default `prepare` derivation applies.
///
/// # Dual scales
///
/// For `Q` the identity on both support qubits, every `P_k Q P_k = Q` and the scale is 1.
/// Otherwise exactly 8 of the 16 two-qubit Paulis anticommute with `Q`, so among the 15 error terms 7 commute and 8 anticommute: `(1-p)Q + (p/15)(7 - 8)Q = (1 - 16p/15)·Q` — the same factor whether `Q` is non-identity on one support qubit or on both.
///
/// # Examples
///
/// ```
/// use paulistrings::Depolarizing2Q;
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
/// The only built-in noise with fan-out > 1, and the only one that is not self-adjoint, so it is the one built-in where the `apply` / `apply_adjoint` orientation is observable.
/// Kraus operators `K_0 = |0⟩⟨0| + √(1-γ)|1⟩⟨1|`, `K_1 = √γ |0⟩⟨1|`, trace-preserving (`K_0†K_0 + K_1†K_1 = I`).
///
/// [`Self::apply`] is the Schrödinger map `Φ(ρ) = K_0 ρ K_0† + K_1 ρ K_1†` (`direction = "forward"`, the same orientation as every other channel): `I → I + γ Z` (the only fanout-2 case), `X → √(1-γ) X`, `Y → √(1-γ) Y`, `Z → (1-γ) Z`.
/// [`Self::apply_adjoint`] is the Heisenberg dual `Φ†(O) = K_0† O K_0 + K_1† O K_1` (`direction = "heisenberg"`), the transpose of `Φ`'s Pauli-transfer matrix: `I → I`, `X → √(1-γ) X`, `Y → √(1-γ) Y`, `Z → (1-γ) Z + γ I` (now the fanout-2 case).
/// The fan-out moves from `I` to `Z` because `Φ†` is unital and `Φ` is trace-preserving — transposed statements of each other. A non-unital Heisenberg map would be unphysical: `⟨Z⟩` for a qubit already in `|0⟩` would decay instead of staying at 1.
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

    /// The Schrödinger map `Φ`, run by `direction = "forward"`.
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
                // I → I + γ Z. The fan-out sits on the identity in Φ: the
                // channel is non-unital, which is the same statement as its
                // dual being trace-preserving.
                out.push(*input_x, *input_z, coeff);
                let mut nz = *input_z;
                set_bit(&mut nz, word, mask, true);
                out.push(*input_x, nz, coeff * self.gamma);
            }
            1 | 3 => {
                // X or Y → √(1-γ) · same.
                let scale = (1.0 - self.gamma).sqrt();
                out.push(*input_x, *input_z, coeff * scale);
            }
            2 => {
                // Z → (1-γ) Z, with no I component (Φ preserves trace and
                // `tr Z = 0`).
                out.push(*input_x, *input_z, coeff * (1.0 - self.gamma));
            }
            _ => unreachable!(),
        }
    }

    /// The Heisenberg dual `Φ†` — the Hilbert-Schmidt adjoint of
    /// [`Self::apply`], i.e. the transpose of its Pauli-transfer matrix. Run by
    /// `direction = "heisenberg"`. See the type's documentation for the
    /// derivation.
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
                // I → I. `Φ†` is unital because `Φ` is trace-preserving.
                out.push(*input_x, *input_z, coeff);
            }
            1 | 3 => {
                // X or Y → √(1-γ) · same, as in the forward map.
                let scale = (1.0 - self.gamma).sqrt();
                out.push(*input_x, *input_z, coeff * scale);
            }
            2 => {
                // Z → (1-γ) Z + γ I. The fan-out moves to Z in the dual. Emit
                // Z first (matches the order in the doc-comment), then I (with
                // the support's z-bit cleared).
                out.push(*input_x, *input_z, coeff * (1.0 - self.gamma));
                let mut nz = *input_z;
                set_bit(&mut nz, word, mask, false);
                out.push(*input_x, nz, coeff * self.gamma);
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests;
