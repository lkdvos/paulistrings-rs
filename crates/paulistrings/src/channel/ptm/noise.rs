//! The noise constructors of [`PtmChannel`]: Pauli-diagonal rescalings and amplitude damping.

use super::{assert_distinct, assert_probability, sparse_storage, PtmChannel, PtmStorage};
use num_complex::Complex64;

impl PtmChannel {
    /// Single-qubit depolarizing noise with error probability `p`: `X`, `Y` and `Z` are scaled by `1 - 4p/3`.
    ///
    /// # Panics
    ///
    /// Unless `p ∈ [0, 1]`.
    pub fn depolarizing(qubit: usize, p: f64) -> Self {
        assert_probability("depolarizing", "p", p);
        let scale = 1.0 - 4.0 * p / 3.0;
        Self::diagonal(&[qubit], vec![1.0, scale, scale, scale], "Depolarizing")
    }

    /// Single-qubit dephasing noise with error probability `p`: `X` and `Y` are scaled by `1 - 2p`.
    ///
    /// # Panics
    ///
    /// Unless `p ∈ [0, 1]`.
    pub fn dephasing(qubit: usize, p: f64) -> Self {
        assert_probability("dephasing", "p", p);
        let scale = 1.0 - 2.0 * p;
        Self::diagonal(&[qubit], vec![1.0, scale, 1.0, scale], "Dephasing")
    }

    /// Single-qubit Pauli channel `E(ρ) = (1-px-py-pz)ρ + px·XρX + py·YρY + pz·ZρZ`.
    ///
    /// It scales `X` by `1 - 2(py + pz)`, `Y` by `1 - 2(px + pz)` and `Z` by `1 - 2(px + py)`, so `(p/3, p/3, p/3)` is [`depolarizing`](Self::depolarizing) and `(0, 0, p)` is [`dephasing`](Self::dephasing).
    ///
    /// # Panics
    ///
    /// Unless each probability lies in `[0, 1]` and `px + py + pz ≤ 1`.
    pub fn pauli_channel(qubit: usize, px: f64, py: f64, pz: f64) -> Self {
        for (name, value) in [("px", px), ("py", py), ("pz", pz)] {
            assert_probability("pauli_channel", name, value);
        }
        let total = px + py + pz;
        assert!(
            total <= 1.0,
            "PtmChannel::pauli_channel: px + py + pz must be at most 1 (got {total})"
        );
        Self::diagonal(
            &[qubit],
            vec![
                1.0,
                1.0 - 2.0 * (py + pz),
                1.0 - 2.0 * (px + py),
                1.0 - 2.0 * (px + pz),
            ],
            "PauliChannel",
        )
    }

    /// Two-qubit depolarizing noise `E(ρ) = (1-p)ρ + (p/15)·Σ_k P_k ρ P_k` over the 15 non-identity two-qubit Paulis.
    ///
    /// Every Pauli that is non-identity on the pair is scaled by `1 - 16p/15`.
    ///
    /// # Panics
    ///
    /// If `q0 == q1`, or unless `p ∈ [0, 1]`.
    pub fn depolarizing_2q(q0: usize, q1: usize, p: f64) -> Self {
        assert_distinct("depolarizing_2q", q0, q1);
        assert_probability("depolarizing_2q", "p", p);
        let mut scale = vec![1.0 - 16.0 * p / 15.0; 16];
        scale[0] = 1.0;
        Self::diagonal(&[q0, q1], scale, "Depolarizing2Q")
    }

    /// Single-qubit amplitude damping with Kraus operators `K_0 = |0⟩⟨0| + √(1-γ)|1⟩⟨1|`, `K_1 = √γ |0⟩⟨1|`.
    ///
    /// Not self-adjoint: `apply` is the Schrödinger map (`I → I + γZ`, `Z → (1-γ)Z`) and `apply_adjoint` its Heisenberg dual (`I → I`, `Z → (1-γ)Z + γI`); both scale `X` and `Y` by `√(1-γ)`.
    ///
    /// # Panics
    ///
    /// Unless `gamma ∈ [0, 1]`.
    pub fn amplitude_damping(qubit: usize, gamma: f64) -> Self {
        assert_probability("amplitude_damping", "gamma", gamma);
        let transverse = Complex64::new((1.0 - gamma).sqrt(), 0.0);
        let storage = sparse_storage(vec![
            ((0, 0), Complex64::new(1.0, 0.0)),
            ((0, 2), Complex64::new(gamma, 0.0)),
            ((1, 1), transverse),
            ((2, 2), Complex64::new(1.0 - gamma, 0.0)),
            ((3, 3), transverse),
        ]);
        Self::new(&[qubit], storage, "AmplitudeDamping")
    }

    fn diagonal(qubits: &[usize], scale: Vec<f64>, name: &'static str) -> Self {
        Self::new(
            qubits,
            PtmStorage::Diagonal {
                scale: scale.into_boxed_slice(),
            },
            name,
        )
    }
}

#[cfg(test)]
mod tests;
