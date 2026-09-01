//! [`StabilizerState<W>`] — expectation values `⟨ψ|P|ψ⟩` in a stabilizer state.
//!
//! This is an **expectation** feature, not stabilizer simulation: the state is
//! a fixed contraction target, never updated under gates. The crate's non-goal
//! (`lib.rs`: "not a state-vector, tensor-network, stabilizer, or MPS
//! simulator") is about *simulation* — evolving a state through a circuit —
//! and nothing here does that. Circuits are still propagated on the operator
//! side by [`propagate`](crate::propagate); a [`StabilizerState`] only replaces
//! [`ProductBasis`](crate::ProductBasis) at the final read-out, widening the
//! set of readable initial states from product states to all stabilizer states
//! (Bell, GHZ, cluster states, any Clifford circuit's output).
//!
//! # Math
//!
//! A stabilizer state on `n` qubits is fixed by `n` independent, pairwise
//! commuting signed Pauli generators `s_i·G_i` (`s_i = ±1`); the group `S` they
//! generate has `2ⁿ` elements and `|ψ⟩` is the unique joint `+1` eigenvector.
//! For a Hermitian Pauli string `P`,
//!
//! ```text
//! ⟨ψ|P|ψ⟩ = σ  if  σ·P ∈ S for some σ = ±1,   and  0 otherwise
//! ```
//!
//! because `E|ψ⟩ = |ψ⟩` for `E = σ·P ∈ S` gives `P|ψ⟩ = σ|ψ⟩`, while a `P`
//! anticommuting with some group element has `⟨ψ|P|ψ⟩ = ⟨ψ|E P E|ψ⟩ =
//! -⟨ψ|P|ψ⟩ = 0`.
//!
//! # Algorithm and cost
//!
//! Setup reduces the generators to row echelon form over GF(2) in the
//! symplectic `(x, z)` coordinates — `O(n²)` Pauli row multiplications of `W`
//! words each, i.e. `O(n³/64)` word operations — carrying each row's sign
//! along, so every stored row *is* a signed element of `S`.
//!
//! Per term, membership is `O(n)` one-bit pivot tests and at most `n` row
//! multiplications of `W` words: `O(n²/64)` word operations. Contracting an
//! `m`-term [`PauliSum`](crate::PauliSum) is therefore `O(m·n²/64)` — see
//! [`PauliSum::expectation_stabilizer`](crate::PauliSum::expectation_stabilizer)
//! — and never a `2ⁿ` expansion over basis states.
//!
//! # Sign bookkeeping
//!
//! Signs are tracked by *composing the group elements themselves* rather than
//! by a separate Aaronson–Gottesman phase table: a row is a pair
//! `(key, neg)` standing for the operator `(-1)^neg · key`, and a row
//! multiplication folds [`PauliString::mul_assign`]'s `i^k` into `neg`. That
//! `i^k` is always real (`k` even) because the two factors are commuting
//! Hermitian Paulis: `AB = BA` and `(AB)† = B†A† = BA = AB`, so `AB` is
//! Hermitian and `i^k·(A⊕B)` can only be `±(A⊕B)`.
//!
//! On the query side the reduction multiplies the term `P` by the rows whose
//! pivots it hits. Intermediate phases there *can* be imaginary (`Y·Z = iX`),
//! since `P` need not commute with the group — but on the path that reaches
//! the identity key the accumulated total is again real, by the same argument
//! applied to `P·∏K_j` (both factors Hermitian, product proportional to `I`).
//! [`StabilizerState::sign_of`] debug-asserts exactly that.
//!
//! # Examples
//!
//! The Bell state `(|00⟩ + |11⟩)/√2` is stabilized by `+XX` and `+ZZ`; the
//! third non-identity group element is `XX·ZZ = (-i Y)(-i Y) = -YY`, so
//! `⟨YY⟩ = -1`.
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

use crate::pauli_string::PauliString;
use crate::phase::Phase;

/// Why a set of generators does not define a stabilizer state.
///
/// Returned by [`StabilizerState::from_generators`]; every variant is a
/// caller-visible input problem except [`Self::InternalPhase`], which is an
/// invariant violation and cannot arise from validated input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StabilizerError {
    /// The generator count is not `num_qubits`. A stabilizer *state* (rather
    /// than a stabilizer code space) needs exactly one generator per qubit.
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
    /// Generators `first` and `second` anticommute, so they have no common
    /// eigenvector.
    NotCommuting {
        /// Index of the first of the two anticommuting generators.
        first: usize,
        /// Index of the second.
        second: usize,
    },
    /// Generator `generator` is a product of the others (up to sign), so the
    /// generators span fewer than `num_qubits` GF(2) dimensions.
    ///
    /// This also covers the `-I ∈ S` case: two generators with the same key
    /// and opposite signs are GF(2)-dependent, and their product is `-I`,
    /// which stabilizes nothing.
    Dependent {
        /// Index of a generator that reduced to the identity key.
        generator: usize,
    },
    /// Internal invariant violation: a product of two commuting Hermitian
    /// group elements came out with an imaginary `i^k` factor. Unreachable
    /// once the commutation check has passed.
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
/// Column order for the elimination is "all `x` bits, qubit 0 first, then all
/// `z` bits" — any fixed order gives a valid echelon form; this one keeps the
/// per-row pivot test a single mask against one word.
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

/// Every symplectic coordinate of an `num_qubits`-qubit key, in elimination
/// order.
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

/// One row of the reduced tableau: the signed group element
/// `(-1)^neg · key`, plus the column it pivots on.
#[derive(Clone, Copy, Debug)]
struct Row<const W: usize> {
    key: PauliString<W>,
    neg: bool,
    pivot: Column,
}

/// A stabilizer state on `n = num_qubits` qubits, held as a reduced tableau of
/// signed generators.
///
/// Build one with [`Self::from_generators`], then read expectation values
/// term-by-term with [`Self::sign_of`] / [`Self::expectation_of`], or over a
/// whole sum with
/// [`PauliSum::expectation_stabilizer`](crate::PauliSum::expectation_stabilizer).
///
/// Product states are the special case of `n` single-qubit generators; those
/// stay faster through
/// [`ProductBasis`](crate::ProductBasis) (one masked word scan per term versus
/// `O(n)` pivot tests here), so prefer
/// [`PauliSum::expectation_product_basis`](crate::PauliSum::expectation_product_basis)
/// when the state factorizes.
///
/// See the module documentation for the algorithm, its cost, and the sign
/// bookkeeping.
#[derive(Clone, Debug)]
pub struct StabilizerState<const W: usize> {
    num_qubits: usize,
    /// Echelon rows, ascending in pivot column; exactly `num_qubits` of them
    /// (construction fails otherwise), each a signed element of `S`.
    rows: Vec<Row<W>>,
}

impl<const W: usize> StabilizerState<W> {
    /// Build the state stabilized by `generators`, where entry `i` is
    /// `(key, minus)` standing for the signed Pauli `(-1)^minus · key`.
    ///
    /// Keys are Hermitian in the crate's convention (`Y = (x=1, z=1)`, no
    /// phase factor — CLAUDE.md §Known gaps), so a generator is exactly the
    /// operator its key spells out, times `±1`. The `minus` flag matches
    /// [`ProductBasis`](crate::ProductBasis)'s `neg`: `true` selects the `-1`
    /// eigenstate.
    ///
    /// Validation, in order: exactly `num_qubits` generators; every key within
    /// `num_qubits`; pairwise commuting; independent over GF(2). Each failure
    /// is a [`StabilizerError`], never a panic.
    ///
    /// # Panics
    ///
    /// Panics if `num_qubits > 64 · W` — a width-selection bug on the caller's
    /// side, not an input-validation case.
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

        // Row-reduce the signed generators. `work[r] = (key, neg, origin)`
        // always denotes a genuine element `(-1)^neg · key` of `S`; `origin`
        // is the input index a row started as, for error reporting.
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
            // Full reduction: clear this column from *every* other row. Rows
            // already placed keep their own pivots, because `work[rank]` was
            // itself cleared of those columns when they were processed.
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
            // Every unplaced row has reduced to the identity key: it was a
            // product of placed rows all along.
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

    /// The stabilizer sign of `key`: `None` when `±key ∉ S` (expectation `0`),
    /// otherwise `Some(negative)` with `negative == true` iff
    /// `⟨ψ|key|ψ⟩ = -1`.
    ///
    /// Returning the sign as a `bool` rather than a float keeps the caller's
    /// accumulation a branch between `+=` and `-=`, matching
    /// [`PauliSum::expectation_product_basis`](crate::PauliSum::expectation_product_basis).
    ///
    /// Cost is one pivot test per row plus one `W`-word Pauli multiply per hit:
    /// `O(n²/64)` word operations.
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
        // Pivot columns are now clear in `acc`, and a nonzero row-space vector
        // cannot have all of them clear — so surviving support means `key` is
        // outside the span.
        if acc != PauliString::<W>::identity() {
            return None;
        }
        // `key · ∏K_j = i^phase · I` with both factors Hermitian forces
        // `i^phase = ±1`; see the module's sign-bookkeeping section.
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
    /// A convenience wrapper over [`Self::sign_of`] for single-term reads and
    /// doctests; the sum-level contraction uses `sign_of` directly.
    #[inline]
    pub fn expectation_of(&self, key: &PauliString<W>) -> f64 {
        match self.sign_of(key) {
            None => 0.0,
            Some(false) => 1.0,
            Some(true) => -1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli_sum::{PauliSum, ProductState};
    use crate::test_support::rand_sum;
    use crate::Gf2Hash;
    use num_complex::Complex64;

    /// Parse `"+XZY"` / `"-XZY"` / `"XZY"` into `(key, minus)`; character `i`
    /// is qubit `i`, Hermitian convention (`Y = (1, 1)`, no phase).
    fn gen_of<const W: usize>(s: &str) -> (PauliString<W>, bool) {
        let (minus, body) = match s.as_bytes().first() {
            Some(b'-') => (true, &s[1..]),
            Some(b'+') => (false, &s[1..]),
            _ => (false, s),
        };
        let mut p = PauliString::<W>::identity();
        for (i, ch) in body.chars().enumerate() {
            let w = i / 64;
            let bit = 1u64 << (i % 64);
            match ch {
                'I' => {}
                'X' => p.x[w] |= bit,
                'Z' => p.z[w] |= bit,
                'Y' => {
                    p.x[w] |= bit;
                    p.z[w] |= bit;
                }
                other => panic!("unexpected Pauli char {other:?}"),
            }
        }
        (p, minus)
    }

    fn state<const W: usize>(num_qubits: usize, gens: &[&str]) -> StabilizerState<W> {
        let g: Vec<_> = gens.iter().map(|s| gen_of::<W>(s)).collect();
        StabilizerState::<W>::from_generators(num_qubits, &g).expect("valid generators")
    }

    /// `⟨ψ|P|ψ⟩` for a single Pauli string given as a label.
    fn expect<const W: usize>(st: &StabilizerState<W>, label: &str) -> f64 {
        st.expectation_of(&gen_of::<W>(label).0)
    }

    // ---- Bell state: +XX, +ZZ ----

    /// `XX·ZZ = (X·Z)⊗(X·Z) = (-iY)⊗(-iY) = -YY`, so the group is
    /// `{+II, +XX, +ZZ, -YY}` and `⟨YY⟩ = -1`. Single-qubit Paulis are outside
    /// the group, hence `0`.
    #[test]
    fn bell_state_expectations_w1() {
        let bell = state::<1>(2, &["+XX", "+ZZ"]);
        assert_eq!(expect(&bell, "II"), 1.0);
        assert_eq!(expect(&bell, "XX"), 1.0);
        assert_eq!(expect(&bell, "ZZ"), 1.0);
        assert_eq!(expect(&bell, "YY"), -1.0);
        assert_eq!(expect(&bell, "ZI"), 0.0);
        assert_eq!(expect(&bell, "IZ"), 0.0);
        assert_eq!(expect(&bell, "XI"), 0.0);
        assert_eq!(expect(&bell, "XZ"), 0.0);
        assert_eq!(expect(&bell, "YI"), 0.0);
    }

    #[test]
    fn bell_state_expectations_w2() {
        let bell = state::<2>(2, &["XX", "ZZ"]);
        assert_eq!(expect(&bell, "XX"), 1.0);
        assert_eq!(expect(&bell, "ZZ"), 1.0);
        assert_eq!(expect(&bell, "YY"), -1.0);
        assert_eq!(expect(&bell, "ZI"), 0.0);
    }

    /// Flipping one generator's sign flips every group element containing it:
    /// `(-XX)(+ZZ) = -(XX·ZZ) = +YY`.
    #[test]
    fn bell_state_with_a_minus_generator_flips_xx_and_yy() {
        let bell = state::<1>(2, &["-XX", "+ZZ"]);
        assert_eq!(expect(&bell, "XX"), -1.0);
        assert_eq!(expect(&bell, "ZZ"), 1.0);
        assert_eq!(expect(&bell, "YY"), 1.0);
        assert_eq!(expect(&bell, "II"), 1.0);
    }

    /// `-Z` stabilizes `|1⟩`, so `⟨Z⟩ = -1` and `⟨X⟩ = ⟨Y⟩ = 0`.
    #[test]
    fn minus_z_generator_is_the_one_state() {
        let one = state::<1>(1, &["-Z"]);
        assert_eq!(expect(&one, "Z"), -1.0);
        assert_eq!(expect(&one, "X"), 0.0);
        assert_eq!(expect(&one, "Y"), 0.0);
        assert_eq!(expect(&one, "I"), 1.0);
    }

    // ---- GHZ state: +XXX, +ZZI, +IZZ ----

    /// `XXX·ZZI` acts as `X·Z = -iY` on qubits 0 and 1 and as `X·I = X` on
    /// qubit 2, so it equals `(-i)²·YYX = -YYX`; the group element being
    /// `-YYX` means `YYX|ψ⟩ = -|ψ⟩`.
    ///
    /// `ZZI·IZZ = ZIZ` with no phase (no X-bits meet a Z-bit), so `⟨ZIZ⟩ = 1`.
    /// `ZII` is outside the span of `{(x=111,z=000), (000,011), (000,110)}`,
    /// hence `0`.
    #[test]
    fn ghz_state_expectations_w1() {
        let ghz = state::<1>(3, &["+XXX", "+ZZI", "+IZZ"]);
        assert_eq!(expect(&ghz, "XXX"), 1.0);
        assert_eq!(expect(&ghz, "ZZI"), 1.0);
        assert_eq!(expect(&ghz, "IZZ"), 1.0);
        assert_eq!(expect(&ghz, "ZIZ"), 1.0);
        assert_eq!(expect(&ghz, "YYX"), -1.0);
        assert_eq!(expect(&ghz, "YXY"), -1.0);
        assert_eq!(expect(&ghz, "XYY"), -1.0);
        assert_eq!(expect(&ghz, "ZII"), 0.0);
        assert_eq!(expect(&ghz, "XXI"), 0.0);
        assert_eq!(expect(&ghz, "YYY"), 0.0);
        assert_eq!(expect(&ghz, "III"), 1.0);
    }

    #[test]
    fn ghz_state_expectations_w2() {
        let ghz = state::<2>(3, &["XXX", "ZZI", "IZZ"]);
        assert_eq!(expect(&ghz, "XXX"), 1.0);
        assert_eq!(expect(&ghz, "YYX"), -1.0);
        assert_eq!(expect(&ghz, "ZIZ"), 1.0);
        assert_eq!(expect(&ghz, "ZII"), 0.0);
    }

    /// Generator order must not matter: the same group, listed differently,
    /// gives the same signs.
    #[test]
    fn generator_order_does_not_change_the_state() {
        let a = state::<1>(3, &["XXX", "ZZI", "IZZ"]);
        let b = state::<1>(3, &["IZZ", "XXX", "ZZI"]);
        // ZIZ = ZZI·IZZ is a redundant *spelling* of the same group, too.
        let c = state::<1>(3, &["ZIZ", "IZZ", "XXX"]);
        for label in ["XXX", "YYX", "ZIZ", "ZZI", "ZII", "III", "XXI"] {
            let want = expect(&a, label);
            assert_eq!(expect(&b, label), want, "{label}");
            assert_eq!(expect(&c, label), want, "{label}");
        }
    }

    // ---- rejection cases ----

    #[test]
    fn anticommuting_generators_are_rejected() {
        let g = [gen_of::<1>("XI"), gen_of::<1>("ZI")];
        assert_eq!(
            StabilizerState::<1>::from_generators(2, &g).unwrap_err(),
            StabilizerError::NotCommuting {
                first: 0,
                second: 1
            },
        );
    }

    #[test]
    fn dependent_generators_are_rejected() {
        let g = [gen_of::<1>("ZI"), gen_of::<1>("ZI")];
        assert_eq!(
            StabilizerState::<1>::from_generators(2, &g).unwrap_err(),
            StabilizerError::Dependent { generator: 1 },
        );
    }

    /// `(+ZI)·(-ZI) = -II`, and `-I` stabilizes nothing. GF(2) dependence
    /// catches it.
    #[test]
    fn opposite_signs_on_one_key_are_rejected() {
        let g = [gen_of::<1>("+ZI"), gen_of::<1>("-ZI")];
        assert_eq!(
            StabilizerState::<1>::from_generators(2, &g).unwrap_err(),
            StabilizerError::Dependent { generator: 1 },
        );
    }

    /// A rank-2 set on 3 qubits: `ZZI` is `ZZZ·IIZ`.
    #[test]
    fn a_rank_deficient_triple_is_rejected() {
        let g = [gen_of::<1>("ZZZ"), gen_of::<1>("IIZ"), gen_of::<1>("ZZI")];
        assert_eq!(
            StabilizerState::<1>::from_generators(3, &g).unwrap_err(),
            StabilizerError::Dependent { generator: 2 },
        );
    }

    #[test]
    fn wrong_generator_count_is_rejected() {
        let g = [gen_of::<1>("ZI")];
        assert_eq!(
            StabilizerState::<1>::from_generators(2, &g).unwrap_err(),
            StabilizerError::GeneratorCount {
                expected: 2,
                found: 1
            },
        );
        let g3 = [gen_of::<1>("ZI"), gen_of::<1>("IZ"), gen_of::<1>("ZZ")];
        assert_eq!(
            StabilizerState::<1>::from_generators(2, &g3).unwrap_err(),
            StabilizerError::GeneratorCount {
                expected: 2,
                found: 3
            },
        );
    }

    #[test]
    fn a_generator_outside_num_qubits_is_rejected() {
        let g = [
            (PauliString::<1>::z(0), false),
            (PauliString::<1>::z(5), false),
        ];
        assert_eq!(
            StabilizerState::<1>::from_generators(2, &g).unwrap_err(),
            StabilizerError::QubitOutOfRange {
                generator: 1,
                num_qubits: 2
            },
        );
    }

    #[test]
    fn errors_display_without_panicking() {
        let e = StabilizerError::NotCommuting {
            first: 0,
            second: 1,
        };
        assert!(e.to_string().contains("anticommute"));
    }

    // ---- word-boundary coverage (W = 2, qubits 63/64) ----

    /// A Bell pair straddling the 64-qubit word boundary, with every other
    /// qubit in `|0⟩`. Same algebra as `bell_state_expectations_w1`, but the
    /// two X-bits and two Z-bits live in different `[u64; 2]` words.
    #[test]
    fn a_bell_pair_across_the_word_boundary() {
        let n = 66;
        let mut xx = PauliString::<2>::x(63);
        xx.mul_assign(&PauliString::<2>::x(64));
        let mut zz = PauliString::<2>::z(63);
        zz.mul_assign(&PauliString::<2>::z(64));
        let mut yy = PauliString::<2>::y(63);
        yy.mul_assign(&PauliString::<2>::y(64));

        let mut gens: Vec<(PauliString<2>, bool)> = Vec::with_capacity(n);
        for q in 0..n as u32 {
            match q {
                63 => gens.push((xx, false)),
                64 => gens.push((zz, false)),
                _ => gens.push((PauliString::<2>::z(q), false)),
            }
        }
        let st = StabilizerState::<2>::from_generators(n, &gens).expect("valid generators");

        assert_eq!(st.expectation_of(&xx), 1.0);
        assert_eq!(st.expectation_of(&zz), 1.0);
        assert_eq!(st.expectation_of(&yy), -1.0);
        assert_eq!(st.expectation_of(&PauliString::<2>::z(63)), 0.0);
        assert_eq!(st.expectation_of(&PauliString::<2>::z(64)), 0.0);
        assert_eq!(st.expectation_of(&PauliString::<2>::z(0)), 1.0);
        assert_eq!(st.expectation_of(&PauliString::<2>::z(65)), 1.0);
        assert_eq!(st.expectation_of(&PauliString::<2>::x(65)), 0.0);
        assert_eq!(st.expectation_of(&PauliString::<2>::identity()), 1.0);
    }

    /// A minus sign on qubit 64's generator: `|0…0 1 0…⟩` with the flip in the
    /// second word.
    #[test]
    fn a_minus_generator_in_the_second_word() {
        let n = 70;
        let mut gens: Vec<(PauliString<2>, bool)> = Vec::with_capacity(n);
        for q in 0..n as u32 {
            gens.push((PauliString::<2>::z(q), q == 64));
        }
        let st = StabilizerState::<2>::from_generators(n, &gens).unwrap();
        assert_eq!(st.expectation_of(&PauliString::<2>::z(64)), -1.0);
        assert_eq!(st.expectation_of(&PauliString::<2>::z(63)), 1.0);
        let mut z63_64 = PauliString::<2>::z(63);
        z63_64.mul_assign(&PauliString::<2>::z(64));
        assert_eq!(st.expectation_of(&z63_64), -1.0);
    }

    // ---- differential tests against the product-state path ----

    /// Diagonal generators `+Z_q` describe `|0…0⟩`, whose expectation the
    /// existing product-state scan already computes — an independent oracle.
    fn product_generators<const W: usize>(num_qubits: usize, axis: char) -> StabilizerState<W> {
        let gens: Vec<(PauliString<W>, bool)> = (0..num_qubits as u32)
            .map(|q| {
                let p = match axis {
                    'X' => PauliString::<W>::x(q),
                    'Y' => PauliString::<W>::y(q),
                    _ => PauliString::<W>::z(q),
                };
                (p, false)
            })
            .collect();
        StabilizerState::<W>::from_generators(num_qubits, &gens).unwrap()
    }

    #[test]
    fn uniform_product_generators_agree_with_the_product_state_scan_w1() {
        let sum = rand_sum::<1>(4000, 20, 0xB0);
        for (axis, st) in [
            ('X', ProductState::XPlus),
            ('Y', ProductState::YPlus),
            ('Z', ProductState::ZPlus),
        ] {
            let stab = product_generators::<1>(20, axis);
            let got = sum.expectation_stabilizer(&stab);
            let want = sum.expectation_product_state(st);
            assert!(
                (got - want).norm() < 1e-12,
                "axis {axis}: {got} vs {want} (product-state oracle)",
            );
        }
    }

    #[test]
    fn uniform_product_generators_agree_with_the_product_state_scan_w2() {
        let sum = rand_sum::<2>(8000, 100, 0xB1);
        for (axis, st) in [
            ('X', ProductState::XPlus),
            ('Y', ProductState::YPlus),
            ('Z', ProductState::ZPlus),
        ] {
            let stab = product_generators::<2>(100, axis);
            let got = sum.expectation_stabilizer(&stab);
            let want = sum.expectation_product_state(st);
            assert!(
                (got - want).norm() < 1e-12,
                "axis {axis}: {got} vs {want} (product-state oracle)",
            );
        }
    }

    /// The contraction is a per-bucket parallel reduction, so it must agree
    /// across partitions (to floating-point tolerance — partials are summed in
    /// bucket order, per the crate's determinism policy).
    #[test]
    fn the_contraction_is_partition_independent() {
        let sum = rand_sum::<1>(5000, 20, 0xB2);
        let stab = product_generators::<1>(20, 'Z');
        let want = sum.expectation_stabilizer(&stab);
        for bits in [0u8, 3, 7] {
            let b = sum.clone().with_hash(Gf2Hash::<1>::new(20, bits, 0xB3));
            let got = b.expectation_stabilizer(&stab);
            assert!((got - want).norm() < 1e-9, "bits={bits}: {got} vs {want}");
        }
    }

    /// Linear in the coefficients, imaginary parts included: the Bell state
    /// picks out `XX` (`+1`), `ZZ` (`+1`) and `YY` (`-1`) and drops `ZI`.
    #[test]
    fn the_contraction_is_linear_and_keeps_the_imaginary_part() {
        let sum = PauliSum::<1>::from_strings(&[
            ("XX", Complex64::new(2.0, 1.0)),
            ("ZZ", Complex64::new(0.5, 0.0)),
            ("YY", Complex64::new(4.0, 2.0)),
            ("ZI", Complex64::new(100.0, 100.0)),
        ]);
        let bell = state::<1>(2, &["XX", "ZZ"]);
        let got = sum.expectation_stabilizer(&bell);
        // (2 + i) + 0.5 - (4 + 2i) = -1.5 - i
        assert!(
            (got - Complex64::new(-1.5, -1.0)).norm() < 1e-12,
            "{got} vs -1.5 - 1i",
        );
    }

    #[test]
    fn an_empty_sum_contracts_to_zero() {
        let sum = PauliSum::<1>::empty(4);
        let stab = product_generators::<1>(4, 'Z');
        assert!(sum.expectation_stabilizer(&stab).norm() < 1e-15);
    }

    // ---- brute-force oracle for entangled states ----

    /// Enumerate all `2ⁿ` group elements by multiplying out every subset of
    /// the generators, and return `key -> sign` — an oracle that shares no code
    /// with the echelon reduction under test.
    ///
    /// Exponential in `n` by construction, which is exactly the thing
    /// [`StabilizerState`] exists to avoid; only usable at `n ≲ 12`.
    fn brute_force_group<const W: usize>(
        gens: &[(PauliString<W>, bool)],
    ) -> std::collections::HashMap<([u64; W], [u64; W]), f64> {
        let n = gens.len();
        let mut out = std::collections::HashMap::new();
        for subset in 0u64..(1u64 << n) {
            let mut key = PauliString::<W>::identity();
            let mut phase = Phase::ONE;
            let mut neg = false;
            for (i, (g, gneg)) in gens.iter().enumerate() {
                if subset >> i & 1 == 1 {
                    phase += key.mul_assign(g);
                    neg ^= gneg;
                }
            }
            assert_eq!(
                phase.exponent() & 1,
                0,
                "subset {subset:b} of commuting Hermitian generators is not Hermitian",
            );
            let sign = if neg ^ (phase == Phase::MINUS_ONE) {
                -1.0
            } else {
                1.0
            };
            assert!(
                out.insert((key.x, key.z), sign).is_none(),
                "subset {subset:b} repeats a group element: generators are dependent",
            );
        }
        out
    }

    /// `⟨ψ|O|ψ⟩` term by term against the enumerated group.
    fn brute_force_expectation<const W: usize>(
        sum: &PauliSum<W>,
        gens: &[(PauliString<W>, bool)],
    ) -> Complex64 {
        let group = brute_force_group(gens);
        let mut acc = Complex64::new(0.0, 0.0);
        for (x, z, c) in sum.iter() {
            if let Some(&s) = group.get(&(*x, *z)) {
                acc += s * c;
            }
        }
        acc
    }

    /// A 1-D cluster state: `K_q = Z_{q-1} X_q Z_{q+1}` (open boundaries),
    /// with the sign of generator `q` taken from `signs`.
    ///
    /// Genuinely entangled and not a Bell/GHZ special case, and every
    /// generator has both X- and Z-support, so the query-side reduction hits
    /// mixed products.
    fn cluster_generators<const W: usize>(
        num_qubits: usize,
        signs: &[bool],
    ) -> Vec<(PauliString<W>, bool)> {
        (0..num_qubits)
            .map(|q| {
                let mut p = PauliString::<W>::x(q as u32);
                if q > 0 {
                    p.mul_assign(&PauliString::<W>::z(q as u32 - 1));
                }
                if q + 1 < num_qubits {
                    p.mul_assign(&PauliString::<W>::z(q as u32 + 1));
                }
                (p, signs[q])
            })
            .collect()
    }

    #[test]
    fn cluster_state_contraction_matches_the_brute_force_group_w1() {
        let n = 8;
        for seed in [0xC0u64, 0xC1, 0xC2] {
            // Signs derived from the seed's bits, so all-plus and mixed-sign
            // states are both covered.
            let signs: Vec<bool> = (0..n).map(|q| seed >> q & 1 == 1).collect();
            let gens = cluster_generators::<1>(n, &signs);
            let stab = StabilizerState::<1>::from_generators(n, &gens).unwrap();
            let sum = rand_sum::<1>(3000, n, seed);
            let got = sum.expectation_stabilizer(&stab);
            let want = brute_force_expectation(&sum, &gens);
            assert!(
                (got - want).norm() < 1e-12,
                "seed {seed:#x}: {got} vs {want} (brute-force group)",
            );
        }
    }

    #[test]
    fn cluster_state_contraction_matches_the_brute_force_group_w2() {
        let n = 6;
        let signs: Vec<bool> = vec![false, true, true, false, false, true];
        let gens = cluster_generators::<2>(n, &signs);
        let stab = StabilizerState::<2>::from_generators(n, &gens).unwrap();
        let sum = rand_sum::<2>(2000, n, 0xC5);
        let got = sum.expectation_stabilizer(&stab);
        let want = brute_force_expectation(&sum, &gens);
        assert!((got - want).norm() < 1e-12, "{got} vs {want}");
    }

    /// GHZ, sign-randomized, against the same oracle — the state the hand
    /// computations above pin, checked over a whole random sum.
    #[test]
    fn ghz_contraction_matches_the_brute_force_group() {
        let n = 7;
        let mut gens: Vec<(PauliString<1>, bool)> = Vec::with_capacity(n);
        let mut all_x = PauliString::<1>::identity();
        for q in 0..n as u32 {
            all_x.mul_assign(&PauliString::<1>::x(q));
        }
        gens.push((all_x, true));
        for q in 1..n as u32 {
            let mut zz = PauliString::<1>::z(q - 1);
            zz.mul_assign(&PauliString::<1>::z(q));
            gens.push((zz, q % 3 == 0));
        }
        let stab = StabilizerState::<1>::from_generators(n, &gens).unwrap();
        let sum = rand_sum::<1>(3000, n, 0xC7);
        let got = sum.expectation_stabilizer(&stab);
        let want = brute_force_expectation(&sum, &gens);
        assert!((got - want).norm() < 1e-12, "{got} vs {want}");
    }

    /// Every one of the `2ⁿ` group elements must be reported with the sign the
    /// enumeration assigns it, and every non-member with `0` — a direct check
    /// of `sign_of` over the *whole* Pauli group at small `n`.
    #[test]
    fn sign_of_agrees_with_the_brute_force_group_over_every_pauli() {
        let n = 4;
        let signs = [true, false, true, true];
        let gens = cluster_generators::<1>(n, &signs);
        let stab = StabilizerState::<1>::from_generators(n, &gens).unwrap();
        let group = brute_force_group(&gens);
        let full = 1u64 << n;
        for x in 0..full {
            for z in 0..full {
                let key = PauliString::<1> { x: [x], z: [z] };
                let want = group.get(&([x], [z])).copied().unwrap_or(0.0);
                assert_eq!(
                    stab.expectation_of(&key),
                    want,
                    "x={x:#b} z={z:#b}: {want} expected",
                );
            }
        }
        // Sanity: the group really is 2^n of the 4^n Paulis.
        assert_eq!(group.len(), 1 << n);
    }

    #[test]
    #[should_panic(expected = "num_qubits mismatch")]
    fn contracting_against_a_differently_sized_state_panics() {
        let sum = PauliSum::<1>::from_strings(&[("XX", Complex64::new(1.0, 0.0))]);
        let stab = product_generators::<1>(3, 'Z');
        let _ = sum.expectation_stabilizer(&stab);
    }
}
