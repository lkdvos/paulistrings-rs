//! Product-state read-outs: [`ProductState`], [`ProductBasis`] and the [`PauliSum`] expectation values in them.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::pauli_sum::PauliSum;

/// A uniform single-qubit product state, for [`PauliSum::expectation_product_state`]; [`ProductBasis`] is the per-qubit generalization.
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
    /// The axis Pauli's symplectic bits `(x, z)`.
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
/// Stored as bit masks in the symplectic key layout; build one with [`ProductBasis::uniform`] or [`ProductBasis::from_axes`].
/// A term `P` has `⟨ψ|P|ψ⟩ = Π_q ⟨P_q⟩`, with `⟨P_q⟩ = 1` for `I`, `s_q` for `A_q`, and `0` for any other Pauli.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductBasis<const W: usize> {
    /// `x`-bit of each qubit's axis Pauli.
    pub axis_x: [u64; W],
    /// `z`-bit of each qubit's axis Pauli.
    pub axis_z: [u64; W],
    /// Sign bit per qubit: `1` selects the `-1` eigenstate.
    pub negative: [u64; W],
}

impl<const W: usize> ProductBasis<W> {
    /// The uniform basis of a [`ProductState`]: the same axis, sign `+1`, on every qubit.
    pub fn uniform(state: ProductState) -> Self {
        let axis = match state {
            ProductState::XPlus => PauliAxis::X,
            ProductState::YPlus => PauliAxis::Y,
            ProductState::ZPlus => PauliAxis::Z,
        };
        let (axis_x, axis_z) = axis.bits();
        let all = |b: bool| if b { [!0u64; W] } else { [0u64; W] };
        Self {
            axis_x: all(axis_x),
            axis_z: all(axis_z),
            negative: [0u64; W],
        }
    }

    /// A basis from per-qubit `(axis, minus)` pairs: item `i` describes qubit `i`, and `minus = true` selects that axis's `-1` eigenstate.
    ///
    /// Qubits past the end of the iterator match only an identity factor, so supply one pair per qubit the sum uses.
    ///
    /// # Panics
    ///
    /// Panics if more than `64 * W` pairs are supplied.
    pub fn from_axes<I>(axes: I) -> Self
    where
        I: IntoIterator<Item = (PauliAxis, bool)>,
    {
        let mut out = Self {
            axis_x: [0u64; W],
            axis_z: [0u64; W],
            negative: [0u64; W],
        };
        for (q, (axis, minus)) in axes.into_iter().enumerate() {
            assert!(
                q < 64 * W,
                "ProductBasis::from_axes: qubit {q} exceeds the {W}-word width",
            );
            let word = q / 64;
            let bit = 1u64 << (q % 64);
            let (axis_x, axis_z) = axis.bits();
            if axis_x {
                out.axis_x[word] |= bit;
            }
            if axis_z {
                out.axis_z[word] |= bit;
            }
            if minus {
                out.negative[word] |= bit;
            }
        }
        out
    }
}

impl<const W: usize> PauliSum<W> {
    /// Expectation value `⟨ψ|O|ψ⟩` in a uniform single-qubit product state; complex since `O` need not be Hermitian.
    pub fn expectation_product_state(&self, state: ProductState) -> Complex64 {
        self.expectation_product_basis(&ProductBasis::<W>::uniform(state))
    }

    /// Expectation value `⟨ψ|O|ψ⟩` in a single-qubit product state with a per-qubit axis and sign; complex since `O` need not be Hermitian.
    pub fn expectation_product_basis(&self, basis: &ProductBasis<W>) -> Complex64 {
        self.buckets()
            .par_iter()
            .map(|columns| {
                let mut partial = Complex64::new(0.0, 0.0);
                for i in 0..columns.len() {
                    // An equality on both key halves, not a subset test: an X factor on a Y-axis qubit must not match.
                    let mut mismatch = 0u64;
                    let mut sign_bits = 0u32;
                    for w in 0..W {
                        let x = columns.x[i][w];
                        let z = columns.z[i][w];
                        let support = x | z;
                        mismatch |=
                            (x ^ (support & basis.axis_x[w])) | (z ^ (support & basis.axis_z[w]));
                        sign_bits += (support & basis.negative[w]).count_ones();
                    }
                    if mismatch == 0 {
                        if sign_bits & 1 == 0 {
                            partial += columns.coeff[i];
                        } else {
                            partial -= columns.coeff[i];
                        }
                    }
                }
                partial
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    }
}
