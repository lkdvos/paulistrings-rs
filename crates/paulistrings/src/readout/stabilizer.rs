//! [`StabilizerState<W>`] — expectation values `⟨ψ|P|ψ⟩` in a stabilizer state.
//!
//! This is an expectation feature, not stabilizer simulation: the state is a fixed contraction target, never evolved under gates (`lib.rs`'s non-goals cover simulation, not this).
//! Circuits still propagate on the operator side via [`propagate`](crate::propagate); a [`StabilizerState`] only replaces [`ProductBasis`](crate::ProductBasis) at the final read-out, widening readable initial states from product states to any stabilizer state (Bell, GHZ, cluster states, any Clifford circuit's output).
//!
//! # Math
//!
//! A stabilizer state on `n` qubits is fixed by `n` independent, pairwise commuting signed Pauli generators `s_i·G_i`; the group `S` they generate has `2ⁿ` elements and `|ψ⟩` is the unique joint `+1` eigenvector.
//! For a Hermitian Pauli string `P`, `⟨ψ|P|ψ⟩ = σ` if `σ·P ∈ S` for some `σ = ±1`, and `0` otherwise: `E|ψ⟩ = |ψ⟩` for `E = σ·P ∈ S` gives `P|ψ⟩ = σ|ψ⟩`, while anticommuting with some group element forces `⟨ψ|P|ψ⟩ = -⟨ψ|P|ψ⟩ = 0`.
//!
//! # Algorithm and cost
//!
//! Setup row-reduces the generators to echelon form over GF(2) in symplectic `(x, z)` coordinates (`O(n³/64)` word operations), carrying each row's sign so every stored row is a signed element of `S`.
//! Per-term membership is `O(n)` pivot tests plus up to `n` row multiplications (`O(n²/64)` word ops), so contracting an `m`-term [`PauliSum`] is `O(m·n²/64)` — see [`PauliSum::expectation_stabilizer`](crate::PauliSum::expectation_stabilizer) — never a `2ⁿ` basis expansion.
//!
//! # Sign bookkeeping
//!
//! Signs are tracked by composing group elements themselves rather than a separate phase table: a row is `(key, neg)` for the operator `(-1)^neg · key`, and a row multiplication folds [`PauliString::mul_assign`]'s `i^k` into `neg`.
//! That `i^k` is always real because the two factors are commuting Hermitian Paulis, whose product is again Hermitian.
//! On the query side, intermediate phases can go imaginary (`Y·Z = iX`) since `P` need not commute with the group, but the total is real again once the reduction reaches the identity key, by the same Hermitian-product argument applied to `P·∏K_j`; [`StabilizerState::sign_of`] debug-asserts exactly that.
//!
//! # Example
//!
//! The Bell state `(|00⟩ + |11⟩)/√2` is stabilized by `+XX` and `+ZZ`; the third non-identity group element is `XX·ZZ = -YY`, so `⟨YY⟩ = -1`.
//!
//! ```
//! use paulistrings::{PauliString, StabilizerState};
//!
//! let mut xx = PauliString::<1>::x(0);
//! xx.mul_assign(&PauliString::<1>::x(1));
//! let mut zz = PauliString::<1>::z(0);
//! zz.mul_assign(&PauliString::<1>::z(1));
//! let mut yy = PauliString::<1>::y(0);
//! yy.mul_assign(&PauliString::<1>::y(1));
//!
//! let bell = StabilizerState::<1>::from_generators(2, &[(xx, false), (zz, false)]).unwrap();
//! assert_eq!(bell.expectation_of(&xx), 1.0);
//! assert_eq!(bell.expectation_of(&zz), 1.0);
//! assert_eq!(bell.expectation_of(&yy), -1.0);
//! assert_eq!(bell.expectation_of(&PauliString::<1>::z(0)), 0.0);
//! ```

use std::fmt;

use num_complex::Complex64;
use rayon::prelude::*;

use crate::pauli_string::PauliString;
use crate::pauli_sum::PauliSum;
use crate::phase::Phase;

/// Why a set of generators does not define a stabilizer state.
///
/// Returned by [`StabilizerState::from_generators`]; every variant is a caller-visible input problem except [`Self::InternalPhase`], which is an invariant violation and cannot arise from validated input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StabilizerError {
    /// The generator count is not `num_qubits`. A stabilizer *state* (rather than a stabilizer code space) needs exactly one generator per qubit.
    GeneratorCount {
        /// The required count, i.e. `num_qubits`.
        expected: usize,
        /// The number of generators supplied.
        found: usize,
    },
    /// Generator `generator` acts on a qubit index at or beyond `num_qubits`.
    QubitOutOfRange {
        /// Index of the offending generator in the input slice.
        generator: usize,
        /// The state's qubit count.
        num_qubits: usize,
    },
    /// Generators `first` and `second` anticommute, so they have no common eigenvector.
    NotCommuting {
        /// Index of the first of the two anticommuting generators.
        first: usize,
        /// Index of the second.
        second: usize,
    },
    /// Generator `generator` is a product of the others (up to sign), so the generators span fewer than `num_qubits` GF(2) dimensions.
    ///
    /// Also covers `-I ∈ S`: two generators with the same key and opposite signs are GF(2)-dependent, and their product is `-I`, which stabilizes nothing.
    Dependent {
        /// Index of a generator that reduced to the identity key.
        generator: usize,
    },
    /// Internal invariant violation: a product of two commuting Hermitian group elements came out with an imaginary `i^k` factor.
    /// Unreachable once the commutation check has passed.
    InternalPhase {
        /// Index of the generator whose row carried the imaginary phase.
        generator: usize,
    },
}

impl fmt::Display for StabilizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GeneratorCount { expected, found } => write!(
                f,
                "a stabilizer state on {expected} qubits needs exactly {expected} generators, \
                 got {found}",
            ),
            Self::QubitOutOfRange {
                generator,
                num_qubits,
            } => write!(
                f,
                "generator {generator} acts on a qubit at or beyond index {num_qubits}",
            ),
            Self::NotCommuting { first, second } => write!(
                f,
                "generators {first} and {second} anticommute; stabilizer generators must \
                 pairwise commute",
            ),
            Self::Dependent { generator } => write!(
                f,
                "generator {generator} is a product of the others (or the generators imply \
                 -I ∈ S); stabilizer generators must be independent over GF(2)",
            ),
            Self::InternalPhase { generator } => write!(
                f,
                "internal: row {generator} of the stabilizer tableau acquired an imaginary \
                 phase from a product of commuting Hermitian Paulis",
            ),
        }
    }
}

impl std::error::Error for StabilizerError {}

/// One GF(2) coordinate of a symplectic key: the `x`- or `z`-bit of one qubit.
///
/// Column order for the elimination is "all `x` bits, qubit 0 first, then all `z` bits" — any fixed order gives a valid echelon form; this one keeps the per-row pivot test a single mask against one word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Column {
    /// Which `[u64; W]` word the bit lives in.
    word: usize,
    /// Single-bit mask within that word.
    mask: u64,
    /// `true` for the `z` half of the key, `false` for the `x` half.
    is_z: bool,
}

impl Column {
    #[inline]
    fn is_set<const W: usize>(self, p: &PauliString<W>) -> bool {
        let half = if self.is_z { &p.z } else { &p.x };
        half[self.word] & self.mask != 0
    }
}

/// Every symplectic coordinate of an `num_qubits`-qubit key, in elimination order.
fn columns(num_qubits: usize) -> Vec<Column> {
    let mut cols = Vec::with_capacity(2 * num_qubits);
    for is_z in [false, true] {
        for q in 0..num_qubits {
            cols.push(Column {
                word: q / 64,
                mask: 1u64 << (q % 64),
                is_z,
            });
        }
    }
    cols
}

/// One row of the reduced tableau: the signed group element `(-1)^neg · key`, plus the column it pivots on.
#[derive(Clone, Copy, Debug)]
struct Row<const W: usize> {
    key: PauliString<W>,
    neg: bool,
    pivot: Column,
}

/// A stabilizer state on `n = num_qubits` qubits, held as a reduced tableau of signed generators.
///
/// Build one with [`Self::from_generators`], then read expectation values term-by-term with [`Self::sign_of`] / [`Self::expectation_of`], or over a whole sum with [`PauliSum::expectation_stabilizer`](crate::PauliSum::expectation_stabilizer).
/// Product states are the special case of `n` single-qubit generators; those stay faster through [`ProductBasis`](crate::ProductBasis) (one masked word scan per term versus `O(n)` pivot tests here), so prefer [`PauliSum::expectation_product_basis`](crate::PauliSum::expectation_product_basis) when the state factorizes.
///
/// See the module documentation for the algorithm, its cost, and the sign bookkeeping.
#[derive(Clone, Debug)]
pub struct StabilizerState<const W: usize> {
    num_qubits: usize,
    /// Echelon rows, ascending in pivot column; exactly `num_qubits` of them (construction fails otherwise), each a signed element of `S`.
    rows: Vec<Row<W>>,
}

impl<const W: usize> StabilizerState<W> {
    /// Build the state stabilized by `generators`, where entry `i` is `(key, minus)` standing for the signed Pauli `(-1)^minus · key`.
    ///
    /// Keys are Hermitian in the crate's convention (`Y = (x=1, z=1)`, no phase factor — CLAUDE.md §Known gaps), so a generator is exactly the operator its key spells out, times `±1`.
    /// The `minus` flag matches [`ProductBasis`](crate::ProductBasis)'s `neg`: `true` selects the `-1` eigenstate.
    ///
    /// Validation, in order: exactly `num_qubits` generators; every key within `num_qubits`; pairwise commuting; independent over GF(2). Each failure is a [`StabilizerError`], never a panic.
    ///
    /// # Panics
    ///
    /// Panics if `num_qubits > 64 · W` — a width-selection bug on the caller's side, not an input-validation case.
    pub fn from_generators(
        num_qubits: usize,
        generators: &[(PauliString<W>, bool)],
    ) -> Result<Self, StabilizerError> {
        assert!(
            num_qubits <= 64 * W,
            "StabilizerState::from_generators: num_qubits {num_qubits} exceeds the {W}-word width",
        );
        if generators.len() != num_qubits {
            return Err(StabilizerError::GeneratorCount {
                expected: num_qubits,
                found: generators.len(),
            });
        }
        for (i, (key, _)) in generators.iter().enumerate() {
            if !key.is_within(num_qubits) {
                return Err(StabilizerError::QubitOutOfRange {
                    generator: i,
                    num_qubits,
                });
            }
        }
        for i in 0..generators.len() {
            for j in (i + 1)..generators.len() {
                if !generators[i].0.commutes_with(&generators[j].0) {
                    return Err(StabilizerError::NotCommuting {
                        first: i,
                        second: j,
                    });
                }
            }
        }

        // Row-reduce the signed generators. `work[r] = (key, neg, origin)` always denotes a genuine element `(-1)^neg · key` of `S`; `origin` is the input index a row started as, for error reporting.
        let mut work: Vec<(PauliString<W>, bool, usize)> = generators
            .iter()
            .enumerate()
            .map(|(i, (key, neg))| (*key, *neg, i))
            .collect();
        let mut rows: Vec<Row<W>> = Vec::with_capacity(num_qubits);

        for col in columns(num_qubits) {
            let rank = rows.len();
            let Some(p) = (rank..work.len()).find(|&r| col.is_set(&work[r].0)) else {
                continue;
            };
            work.swap(rank, p);
            let (key, neg, _) = work[rank];
            // Full reduction: clear this column from every other row. Rows already placed keep their own pivots, because `work[rank]` was itself cleared of those columns when they were processed.
            for (r, row) in work.iter_mut().enumerate() {
                if r == rank || !col.is_set(&row.0) {
                    continue;
                }
                let phase = row.0.mul_assign(&key);
                if phase.exponent() & 1 != 0 {
                    return Err(StabilizerError::InternalPhase { generator: row.2 });
                }
                row.1 ^= neg ^ (phase == Phase::MINUS_ONE);
            }
            rows.push(Row {
                key,
                neg,
                pivot: col,
            });
            if rows.len() == num_qubits {
                break;
            }
        }

        if rows.len() < num_qubits {
            // Every unplaced row has reduced to the identity key: it was a product of placed rows all along.
            return Err(StabilizerError::Dependent {
                generator: work[rows.len()].2,
            });
        }
        Ok(Self { num_qubits, rows })
    }

    /// Number of qubits the state is defined on.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The stabilizer sign of `key`: `None` when `±key ∉ S` (expectation `0`), otherwise `Some(negative)` with `negative == true` iff `⟨ψ|key|ψ⟩ = -1`.
    ///
    /// Returning the sign as a `bool` rather than a float keeps the caller's accumulation a branch between `+=` and `-=`, matching [`PauliSum::expectation_product_basis`](crate::PauliSum::expectation_product_basis).
    /// Cost is one pivot test per row plus one `W`-word Pauli multiply per hit: `O(n²/64)` word operations.
    #[inline]
    pub fn sign_of(&self, key: &PauliString<W>) -> Option<bool> {
        let mut acc = *key;
        let mut phase = Phase::ONE;
        let mut neg = false;
        for row in &self.rows {
            if row.pivot.is_set(&acc) {
                phase += acc.mul_assign(&row.key);
                neg ^= row.neg;
            }
        }
        // Pivot columns are now clear in `acc`, and a nonzero row-space vector cannot have all of them clear — so surviving support means `key` is outside the span.
        if acc != PauliString::<W>::identity() {
            return None;
        }
        // `key · ∏K_j = i^phase · I` with both factors Hermitian forces `i^phase = ±1`; see the module's sign-bookkeeping section.
        debug_assert_eq!(
            phase.exponent() & 1,
            0,
            "stabilizer membership produced an imaginary phase i^{}",
            phase.exponent(),
        );
        Some(neg ^ (phase == Phase::MINUS_ONE))
    }

    /// `⟨ψ|key|ψ⟩` as a float: `0.0` when `±key ∉ S`, else `±1.0`.
    ///
    /// A convenience wrapper over [`Self::sign_of`] for single-term reads and doctests; the sum-level contraction uses `sign_of` directly.
    #[inline]
    pub fn expectation_of(&self, key: &PauliString<W>) -> f64 {
        match self.sign_of(key) {
            None => 0.0,
            Some(false) => 1.0,
            Some(true) => -1.0,
        }
    }
}

impl<const W: usize> PauliSum<W> {
    /// Expectation value `⟨ψ|O|ψ⟩` in a stabilizer state.
    /// `⟨ψ|P|ψ⟩` is `±1` when `±P` lies in the state's stabilizer group and `0` otherwise, so this is a filter with a sign, exactly like [`Self::expectation_product_basis`], but the admissible state widens to any stabilizer state (Bell, GHZ, cluster, a Clifford circuit's output). See [`StabilizerState`] for the membership test and the sign bookkeeping.
    /// Cost is `O(terms · n²/64)` word operations after the state's one-time `O(n³/64)` reduction — `n` times more work per term than the product-state scan, so prefer [`Self::expectation_product_basis`] for states that factorize.
    /// Returns `Complex64` rather than `f64` because `self` need not be Hermitian; take `.re` when it is.
    ///
    /// # Summation order
    ///
    /// As in [`Self::expectation_product_state`] — partials are combined in bucket order, so two partitions of the same terms can differ in the last bits.
    ///
    /// # Panics
    ///
    /// Panics if `state.num_qubits()` differs from [`Self::num_qubits`].
    pub fn expectation_stabilizer(&self, state: &StabilizerState<W>) -> Complex64 {
        assert_eq!(
            self.num_qubits(),
            state.num_qubits(),
            "PauliSum::expectation_stabilizer: num_qubits mismatch ({} vs {})",
            self.num_qubits(),
            state.num_qubits(),
        );
        self.buckets()
            .par_iter()
            .map(|cols| {
                let mut acc = Complex64::new(0.0, 0.0);
                for i in 0..cols.len() {
                    let key = PauliString::<W> {
                        x: cols.x[i],
                        z: cols.z[i],
                    };
                    match state.sign_of(&key) {
                        None => {}
                        Some(false) => acc += cols.coeff[i],
                        Some(true) => acc -= cols.coeff[i],
                    }
                }
                acc
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    }
}

#[cfg(test)]
mod tests;
