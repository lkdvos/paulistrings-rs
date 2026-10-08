//! [`StabilizerState<W>`], a reduced tableau for reading expectation values `⟨ψ|P|ψ⟩` in a stabilizer state; it is a read-out target, never evolved.

use std::fmt;

use num_complex::Complex64;
use rayon::prelude::*;

use crate::pauli_string::PauliString;
use crate::pauli_sum::PauliSum;
use crate::phase::Phase;

/// Why a set of generators does not define a stabilizer state; [`Self::InternalPhase`] is an invariant violation, the rest are input errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StabilizerError {
    /// The generator count is not `num_qubits`.
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
    /// Generator `generator` is a product of the others up to sign, which also covers `-I ∈ S`.
    Dependent {
        /// Index of a generator that reduced to the identity key.
        generator: usize,
    },
    /// A product of two commuting Hermitian group elements came out imaginary; unreachable once commutation is checked.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Column {
    word: usize,
    mask: u64,
    is_z: bool,
}

impl Column {
    #[inline]
    fn is_set<const W: usize>(self, p: &PauliString<W>) -> bool {
        let half = if self.is_z { &p.z } else { &p.x };
        half[self.word] & self.mask != 0
    }
}

/// Every symplectic coordinate of a `num_qubits`-qubit key, in elimination order.
fn columns(num_qubits: usize) -> Vec<Column> {
    let mut columns = Vec::with_capacity(2 * num_qubits);
    for is_z in [false, true] {
        for q in 0..num_qubits {
            columns.push(Column {
                word: q / 64,
                mask: 1u64 << (q % 64),
                is_z,
            });
        }
    }
    columns
}

/// One row of the reduced tableau: the signed group element `(-1)^neg · key`, plus the column it pivots on.
#[derive(Clone, Copy, Debug)]
struct Row<const W: usize> {
    key: PauliString<W>,
    neg: bool,
    pivot: Column,
}

/// A stabilizer state on `num_qubits` qubits, held as a reduced tableau of signed generators.
///
/// Read it out with [`crate::PauliSum::expectation_stabilizer`]; a product state reads faster through [`crate::ProductBasis`].
#[derive(Clone, Debug)]
pub struct StabilizerState<const W: usize> {
    num_qubits: usize,
    /// Echelon rows ascending in pivot column, each a signed element of the stabilizer group.
    rows: Vec<Row<W>>,
}

impl<const W: usize> StabilizerState<W> {
    /// Build the state stabilized by `generators`, each `(key, minus)` standing for `(-1)^minus · key`.
    ///
    /// Generators must number `num_qubits`, lie within `num_qubits`, pairwise commute and be independent, or a [`StabilizerError`] says which failed.
    ///
    /// # Panics
    ///
    /// Panics if `num_qubits > 64 · W`.
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

        // `work[r] = (key, neg, origin)` is always the group element `(-1)^neg · key`; `origin` indexes the input for errors.
        let mut work: Vec<(PauliString<W>, bool, usize)> = generators
            .iter()
            .enumerate()
            .map(|(i, (key, neg))| (*key, *neg, i))
            .collect();
        let mut rows: Vec<Row<W>> = Vec::with_capacity(num_qubits);

        for column in columns(num_qubits) {
            let rank = rows.len();
            let Some(p) = (rank..work.len()).find(|&r| column.is_set(&work[r].0)) else {
                continue;
            };
            work.swap(rank, p);
            let (key, neg, _) = work[rank];
            for (r, row) in work.iter_mut().enumerate() {
                if r == rank || !column.is_set(&row.0) {
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
                pivot: column,
            });
            if rows.len() == num_qubits {
                break;
            }
        }

        if rows.len() < num_qubits {
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

    /// The stabilizer sign of `key`: `None` when `±key` is outside the group (expectation `0`), else `Some(true)` iff `⟨ψ|key|ψ⟩ = -1`.
    #[inline]
    pub fn sign_of(&self, key: &PauliString<W>) -> Option<bool> {
        let mut reduced = *key;
        let mut phase = Phase::ONE;
        let mut neg = false;
        for row in &self.rows {
            if row.pivot.is_set(&reduced) {
                phase += reduced.mul_assign(&row.key);
                neg ^= row.neg;
            }
        }
        if reduced != PauliString::<W>::identity() {
            return None;
        }
        // Intermediate phases can be imaginary, but `key · ∏K_j = i^phase · I` with both factors Hermitian forces `i^phase = ±1`.
        debug_assert_eq!(
            phase.exponent() & 1,
            0,
            "stabilizer membership produced an imaginary phase i^{}",
            phase.exponent(),
        );
        Some(neg ^ (phase == Phase::MINUS_ONE))
    }

    /// `⟨ψ|key|ψ⟩`: `0.0`, `1.0` or `-1.0`.
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
    /// Expectation value `⟨ψ|O|ψ⟩` in a stabilizer state; complex since `O` need not be Hermitian.
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
            .map(|columns| {
                let mut partial = Complex64::new(0.0, 0.0);
                for i in 0..columns.len() {
                    let key = PauliString::<W> {
                        x: columns.x[i],
                        z: columns.z[i],
                    };
                    match state.sign_of(&key) {
                        None => {}
                        Some(false) => partial += columns.coeff[i],
                        Some(true) => partial -= columns.coeff[i],
                    }
                }
                partial
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Complex64::new(0.0, 0.0), |a, b| a + b)
    }
}

#[cfg(test)]
mod tests;
