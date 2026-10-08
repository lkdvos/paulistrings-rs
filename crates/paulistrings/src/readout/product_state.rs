//! Product-state read-outs: [`ProductState`], [`ProductBasis`] and the [`PauliSum`] expectation values in them.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::pauli_sum::PauliSum;

/// A uniform single-qubit product state, for [`PauliSum::expectation_product_state`].
///
/// Each variant names the single-qubit Pauli whose `+1` eigenstate is taken on every qubit — the uniform special case of [`ProductBasis`], which allows a different axis and sign per qubit ([`PauliSum::expectation_product_basis`] is the one scan underneath both).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductState {
    /// `|+…+⟩`, the `+1` eigenstate of `X` on every qubit.
    XPlus,
    /// `|+i…+i⟩`, the `+1` eigenstate of `Y` on every qubit.
    YPlus,
    /// `|0…0⟩`, the `+1` eigenstate of `Z` on every qubit.
    ZPlus,
}

/// One of the three single-qubit Pauli axes: the axis a [`ProductBasis`] qubit is an eigenstate of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauliAxis {
    /// `X`, with eigenstates `|+⟩` (`+1`) and `|-⟩` (`-1`).
    X,
    /// `Y`, with eigenstates `|+i⟩` (`+1`) and `|-i⟩` (`-1`).
    Y,
    /// `Z`, with eigenstates `|0⟩` (`+1`) and `|1⟩` (`-1`).
    Z,
}

impl PauliAxis {
    /// The axis Pauli's symplectic bits `(x, z)`: `X = (1, 0)`, `Z = (0, 1)`, `Y = (1, 1)` (Hermitian convention, no phase).
    #[inline]
    const fn bits(self) -> (bool, bool) {
        match self {
            PauliAxis::X => (true, false),
            PauliAxis::Y => (true, true),
            PauliAxis::Z => (false, true),
        }
    }
}

/// A product of single-qubit stabilizer states, one per qubit: on qubit `q` an axis `A_q ∈ {X, Y, Z}` with a sign `s_q ∈ {+1, -1}`, i.e. `|ψ⟩ = ⊗_q |A_q, s_q⟩`.
///
/// Stored as per-word bit masks in the same symplectic layout as a [`PauliString`] key, so [`PauliSum::expectation_product_basis`] stays a masked scan over the key columns, never an expansion over `2ⁿ` basis states.
/// Build one with [`ProductBasis::uniform`] or [`ProductBasis::from_axes`].
///
/// # Semantics
///
/// For a term `P` with key `(x, z)`, `⟨ψ|P|ψ⟩ = Π_q ⟨P_q⟩`, where `⟨P_q⟩ = 1` if `P_q = I`, `s_q` if `P_q = A_q`, and `0` otherwise (distinct single-qubit Paulis anticommute, so every off-axis Bloch component vanishes).
/// With `sup = x | z` the term's support mask, the term contributes iff `x == sup & ax_x && z == sup & ax_z` — an equality on both halves of the key, not a subset test, so e.g. an `X` term on a `Y`-axis qubit contributes `0` (`⟨+i|X|+i⟩ = 0`). When it does contribute, its sign is `(-1)^popcount(sup & neg)`.
///
/// # Examples
///
/// `⟨01|Z⊗Z|01⟩ = ⟨0|Z|0⟩·⟨1|Z|1⟩ = (+1)(-1) = -1`.
///
/// ```
/// use paulistrings::{BuildAccumulator, PauliAxis, PauliString, Phase, ProductBasis};
/// use num_complex::Complex64;
///
/// let mut acc = BuildAccumulator::<1>::new(2);
/// let mut zz = PauliString::<1>::z(0);
/// zz.mul_assign(&PauliString::<1>::z(1));
/// acc.add_term(zz, Phase::ONE, Complex64::new(1.0, 0.0));
/// let sum = acc.finalize();
///
/// // Qubit 0 in |0⟩, qubit 1 in |1⟩ — both Z-axis, the second one negative.
/// let basis = ProductBasis::<1>::from_axes([(PauliAxis::Z, false), (PauliAxis::Z, true)]);
/// assert!((sum.expectation_product_basis(&basis).re + 1.0).abs() < 1e-12);
/// ```
///
/// [`PauliString`]: crate::PauliString
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductBasis<const W: usize> {
    /// `x`-bit of each qubit's axis Pauli.
    pub ax_x: [u64; W],
    /// `z`-bit of each qubit's axis Pauli.
    pub ax_z: [u64; W],
    /// Sign bit per qubit: `1` selects the `-1` eigenstate.
    pub neg: [u64; W],
}

impl<const W: usize> ProductBasis<W> {
    /// The uniform basis of a [`ProductState`]: the same axis, sign `+1`, on every qubit.
    ///
    /// The axis masks are all-ones rather than trimmed to a qubit count — safe at any width, since a stored key never has a bit set beyond `num_qubits`.
    pub fn uniform(state: ProductState) -> Self {
        let axis = match state {
            ProductState::XPlus => PauliAxis::X,
            ProductState::YPlus => PauliAxis::Y,
            ProductState::ZPlus => PauliAxis::Z,
        };
        let (bx, bz) = axis.bits();
        let all = |b: bool| if b { [!0u64; W] } else { [0u64; W] };
        Self {
            ax_x: all(bx),
            ax_z: all(bz),
            neg: [0u64; W],
        }
    }

    /// A basis from per-qubit `(axis, minus)` pairs: item `i` describes qubit `i`, and `minus = true` selects that axis's `-1` eigenstate.
    ///
    /// Qubits past the end of the iterator keep an all-zero (identity) axis, matching only an identity factor — supply one pair per qubit the sum actually uses.
    ///
    /// # Panics
    ///
    /// Panics if more than `64 * W` pairs are supplied.
    pub fn from_axes<I>(axes: I) -> Self
    where
        I: IntoIterator<Item = (PauliAxis, bool)>,
    {
        let mut out = Self {
            ax_x: [0u64; W],
            ax_z: [0u64; W],
            neg: [0u64; W],
        };
        for (q, (axis, minus)) in axes.into_iter().enumerate() {
            assert!(
                q < 64 * W,
                "ProductBasis::from_axes: qubit {q} exceeds the {W}-word width",
            );
            let word = q / 64;
            let bit = 1u64 << (q % 64);
            let (bx, bz) = axis.bits();
            if bx {
                out.ax_x[word] |= bit;
            }
            if bz {
                out.ax_z[word] |= bit;
            }
            if minus {
                out.neg[word] |= bit;
            }
        }
        out
    }
}

impl<const W: usize> PauliSum<W> {
    /// Expectation value `⟨ψ|O|ψ⟩` in a uniform single-qubit product state.
    /// For each [`ProductState`] there is exactly one single-qubit Pauli with expectation `1`; a term contributes its full coefficient iff every factor is `I` or that Pauli — a masked scan over the key columns, run as a per-bucket parallel reduction.
    /// The uniform states are the special case of [`ProductBasis`] with sign `+1` everywhere, so this is a thin wrapper over [`Self::expectation_product_basis`].
    /// Returns `Complex64` rather than `f64` because `self` need not be Hermitian; take `.re` when it is.
    ///
    /// # Summation order
    ///
    /// Partial sums are combined in bucket order, which is deterministic given the partition; two partitions of the same terms can differ in the last bits, since `f64` addition is not associative.
    pub fn expectation_product_state(&self, state: ProductState) -> Complex64 {
        self.expectation_product_basis(&ProductBasis::<W>::uniform(state))
    }

    /// Expectation value `⟨ψ|O|ψ⟩` in an arbitrary single-qubit product state.
    /// `basis` gives each qubit its own axis and sign; see [`ProductBasis`] for the per-word match condition and the sign rule this evaluates. A term contributes its coefficient (negated when an odd number of support sites are `-1` eigenstates) iff every non-identity factor equals that qubit's axis exactly. Cost is one pass over the key columns as a per-bucket parallel reduction, with no expansion over basis states.
    /// Returns `Complex64` rather than `f64` because `self` need not be Hermitian; take `.re` when it is.
    ///
    /// # Summation order
    ///
    /// As in [`Self::expectation_product_state`] — partials are combined in bucket order, so two partitions of the same terms can differ in the last bits.
    pub fn expectation_product_basis(&self, basis: &ProductBasis<W>) -> Complex64 {
        self.buckets()
            .par_iter()
            .map(|cols| {
                let mut acc = Complex64::new(0.0, 0.0);
                for i in 0..cols.len() {
                    // `mismatch` stays zero iff every word's non-identity sites carry exactly the local axis Pauli; `sign_bits` counts the `-1` eigenstates inside the support.
                    let mut mismatch = 0u64;
                    let mut sign_bits = 0u32;
                    for w in 0..W {
                        let x = cols.x[i][w];
                        let z = cols.z[i][w];
                        let sup = x | z;
                        mismatch |= (x ^ (sup & basis.ax_x[w])) | (z ^ (sup & basis.ax_z[w]));
                        sign_bits += (sup & basis.neg[w]).count_ones();
                    }
                    if mismatch == 0 {
                        if sign_bits & 1 == 0 {
                            acc += cols.coeff[i];
                        } else {
                            acc -= cols.coeff[i];
                        }
                    }
                }
                acc
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    }
}
