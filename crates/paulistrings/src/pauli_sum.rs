//! [`PauliSum<W>`] — weighted sum of Pauli strings in structure-of-arrays form.
//!
//! Storage is per-bucket parallel `Vec<[u64; W]>` columns for the `x` and `z`
//! parts plus a `Vec<Complex64>` for coefficients, partitioned by a
//! GF(2)-linear hash ([`Gf2Hash`](crate::bucket::Gf2Hash)). SoA is chosen so
//! coefficient-only and key-only scans get full cache utilization, and so each
//! `Vec` maps directly to a GPU device buffer.
//!
//! # Canonical order
//!
//! **Terms are ordered by (bucket index `h(x, z)` ascending, then
//! lexicographic `(x, z)` key within a bucket).** [`PauliSum::iter`] and
//! [`PauliSum::to_arrays`] produce exactly this order, and no two entries
//! share a key. Every public operation preserves the invariant or returns a
//! fresh [`PauliSum`] that does.
//!
//! A single-bucket sum's canonical order is plain lexicographic `(x, z)` —
//! `h ≡ 0` — and every sum of at most 1024 terms
//! ([`DEFAULT_TARGET_BUCKET_LEN`](crate::bucket::DEFAULT_TARGET_BUCKET_LEN))
//! is single-bucket when built through [`BuildAccumulator`], so small sums
//! always come out lex-sorted. Larger sums interleave their buckets in an
//! `H`-dependent order; compare them by key ([`PauliSum::get`],
//! [`PauliSum::iter`]) rather than by position.
//!
//! Build a [`PauliSum`] from unsorted inputs via [`BuildAccumulator`]; once
//! built, combine sums with [`PauliSum::add`] or scale coefficients with
//! [`PauliSum::scale`].
//!
//! See ARCHITECTURE.md §Data-Model for the storage design rationale.
//!
//! # Examples
//!
//! Construct the observable `Z₀ + 0.5·X₁` on two qubits via
//! [`BuildAccumulator`], then merge in a second sum.
//!
//! ```
//! use paulistrings::{BuildAccumulator, PauliString, PauliSum, Phase};
//! use num_complex::Complex64;
//!
//! let mut acc = BuildAccumulator::<1>::new(2);
//! acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
//! acc.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(0.5, 0.0));
//! let a = acc.finalize();
//! assert_eq!(a.len(), 2);
//!
//! let mut acc2 = BuildAccumulator::<1>::new(2);
//! acc2.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(-0.25, 0.0));
//! let b = acc2.finalize();
//!
//! let merged = a.add(&b);
//! assert_eq!(merged.len(), 2); // Z₀ + 0.25·X₁
//! ```
//!
//! [`BuildAccumulator`]: crate::BuildAccumulator

#[cfg(test)]
use num_complex::Complex64;

#[cfg(test)]
use crate::pauli_string::PauliString;

pub use crate::bucket::sum::PauliSum;

/// A uniform single-qubit product state, for
/// [`PauliSum::expectation_product_state`].
///
/// Each variant names the single-qubit Pauli whose `+1` eigenstate is taken on
/// every qubit. These are the states quench experiments actually start from, and
/// each one makes the expectation a masked scan rather than a simulation.
///
/// They are the uniform special cases of [`ProductBasis`], which allows a
/// different axis and sign per qubit; there is exactly one scan underneath
/// ([`PauliSum::expectation_product_basis`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductState {
    /// `|+…+⟩`, the `+1` eigenstate of `X` on every qubit.
    XPlus,
    /// `|+i…+i⟩`, the `+1` eigenstate of `Y` on every qubit.
    YPlus,
    /// `|0…0⟩`, the `+1` eigenstate of `Z` on every qubit.
    ZPlus,
}

/// One of the three single-qubit Pauli axes: the axis a [`ProductBasis`] qubit
/// is an eigenstate of.
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
    /// The axis Pauli's symplectic bits `(x, z)`: `X = (1, 0)`, `Z = (0, 1)`,
    /// `Y = (1, 1)` — the crate's Hermitian-`Y` convention, with no phase.
    #[inline]
    const fn bits(self) -> (bool, bool) {
        match self {
            PauliAxis::X => (true, false),
            PauliAxis::Y => (true, true),
            PauliAxis::Z => (false, true),
        }
    }
}

/// A product of single-qubit stabilizer states, one per qubit: on qubit `q` an
/// axis `A_q ∈ {X, Y, Z}` with a sign `s_q ∈ {+1, -1}`, i.e.
/// `|ψ⟩ = ⊗_q |A_q, s_q⟩`.
///
/// Stored as per-word bit masks in the same symplectic layout as a
/// [`PauliString`] key, so [`PauliSum::expectation_product_basis`] stays a
/// masked scan over the key columns — never an expansion over `2ⁿ` basis
/// states. Build one with [`ProductBasis::uniform`] or
/// [`ProductBasis::from_axes`].
///
/// # Semantics
///
/// For a term `P` with key `(x, z)`, `⟨ψ|P|ψ⟩ = Π_q ⟨P_q⟩`, where
/// `⟨P_q⟩ = 1` if `P_q = I`, `s_q` if `P_q = A_q`, and `0` otherwise (two
/// distinct single-qubit Paulis anticommute, so every off-axis component of
/// the Bloch vector vanishes). Written bit-parallel per word, with
/// `sup = x | z` the term's support mask, the term contributes iff
///
/// ```text
/// x == sup & ax_x   &&   z == sup & ax_z
/// ```
///
/// — every non-identity site's Pauli equals that qubit's axis **exactly**.
/// This is an equality on both halves of the key, not a subset test: an `X`
/// term on a `Y`-axis qubit has `(x, z) = (1, 0)` against
/// `(ax_x, ax_z) = (1, 1)`, so the `z` half fails and the term contributes
/// `0` — which is right, since `⟨+i|X|+i⟩ = 0`. When it does contribute, its
/// sign is `(-1)^popcount(sup & neg)`: identity sites carry no sign, because
/// they are not in `sup`.
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
    /// The uniform basis of a [`ProductState`]: the same axis, sign `+1`, on
    /// every qubit.
    ///
    /// The axis masks are all-ones rather than trimmed to a qubit count, which
    /// is safe at any width because a stored key never has a bit set beyond
    /// `num_qubits` — those mask bits are simply never read. The match
    /// condition then reduces to `x == 0` for `ZPlus`, `z == 0` for `XPlus`
    /// and `x == z` for `YPlus`.
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

    /// A basis from per-qubit `(axis, minus)` pairs: item `i` describes qubit
    /// `i`, and `minus = true` selects that axis's `-1` eigenstate.
    ///
    /// Qubits past the end of the iterator keep an all-zero (identity) axis,
    /// which matches only an identity factor — so supply one pair per qubit
    /// the sum actually uses.
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

#[cfg(test)]
impl<const W: usize> PauliSum<W> {
    /// Test-only helper: build a `PauliSum<W>` from `(pauli_str, coeff)`
    /// pairs. Each `pauli_str` is a sequence of `I/X/Y/Z` characters where
    /// index `i` of the string corresponds to qubit `i`. Coefficients
    /// multiply the literal Hermitian Pauli string: `Y` maps to the
    /// symplectic key `(x=1, z=1)` with no phase factor, matching
    /// [`PauliString::y`] and `expectation_product_state`.
    ///
    /// `num_qubits` is taken from the length of the first string; all
    /// other strings must match. Routes through `BuildAccumulator`, so
    /// duplicate keys sum and exact-zero coefficients are dropped.
    pub(crate) fn from_strings(terms: &[(&str, Complex64)]) -> Self {
        use crate::phase::Phase;
        assert!(!terms.is_empty(), "from_strings requires at least one term");
        let num_qubits = terms[0].0.len();
        assert!(num_qubits <= 64 * W, "num_qubits must fit in W*64 bits");
        let mut acc = crate::accumulator::BuildAccumulator::<W>::new(num_qubits);
        for (s, c) in terms {
            assert_eq!(
                s.len(),
                num_qubits,
                "all pauli strings must have the same length",
            );
            let mut x = [0u64; W];
            let mut z = [0u64; W];
            for (i, ch) in s.chars().enumerate() {
                let word = i / 64;
                let bit = 1u64 << (i % 64);
                match ch {
                    'I' => {}
                    'X' => x[word] |= bit,
                    'Z' => z[word] |= bit,
                    'Y' => {
                        x[word] |= bit;
                        z[word] |= bit;
                    }
                    other => panic!("unexpected Pauli char {:?} (expected I/X/Y/Z)", other),
                }
            }
            let p = PauliString::<W> { x, z };
            acc.add_term(p, Phase::ONE, *c);
        }
        acc.finalize()
    }
}
