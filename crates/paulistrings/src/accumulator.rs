//! [`BuildAccumulator<W>`] — hashmap-based ingestion path.
//!
//! Used to incrementally build a [`PauliSum`] from unsorted inputs
//! (Hamiltonian parsing, Python dict construction, etc.). The accumulator is
//! **not** used during propagation — that path is sort-merge only (see
//! [`engine`]).
//!
//! See the [`PauliSum`] module for a worked example that walks
//! [`BuildAccumulator::new`] → [`BuildAccumulator::add_term`] →
//! [`BuildAccumulator::finalize`].
//!
//! See design doc §8.2.
//!
//! [`PauliSum`]: crate::PauliSum
//! [`engine`]: crate::engine

#![allow(unused)]

use crate::pauli_string::PauliString;
use crate::pauli_sum::PauliSum;
use crate::phase::Phase;
use hashbrown::HashMap;
use num_complex::Complex64;
use rustc_hash::FxBuildHasher;

/// Incremental builder for a `PauliSum`.
///
/// Uses `FxBuildHasher` rather than the default `SipHash` since Pauli
/// bitstrings are already high-entropy.
pub struct BuildAccumulator<const W: usize> {
    map: HashMap<PauliString<W>, Complex64, FxBuildHasher>,
    num_qubits: usize,
}

impl<const W: usize> BuildAccumulator<W> {
    /// New empty accumulator targeting `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            map: HashMap::with_hasher(FxBuildHasher),
            num_qubits,
        }
    }

    /// Allocate up-front for at least `cap` distinct Pauli keys.
    pub fn with_capacity(num_qubits: usize, cap: usize) -> Self {
        Self {
            map: HashMap::with_capacity_and_hasher(cap, FxBuildHasher),
            num_qubits,
        }
    }

    /// Add `phase · c · p` to the accumulator. The phase factor is folded
    /// into `c` before the upsert. `p` is taken as-is and used as the map
    /// key.
    ///
    /// # Examples
    ///
    /// ```
    /// use paulistrings::{BuildAccumulator, PauliString, Phase};
    /// use num_complex::Complex64;
    ///
    /// let mut acc = BuildAccumulator::<1>::new(2);
    /// // Add a Y-like term written as i · (X · Z) on qubit 0.
    /// acc.add_term(
    ///     PauliString::<1> { x: [1], z: [1] },
    ///     Phase::I,
    ///     Complex64::new(1.0, 0.0),
    /// );
    /// let sum = acc.finalize();
    /// assert_eq!(sum.coeff()[0], Complex64::new(0.0, 1.0));
    /// ```
    pub fn add_term(&mut self, p: PauliString<W>, phase: Phase, c: Complex64) {
        let contribution = phase.apply(c);
        self.map
            .entry(p)
            .and_modify(|e| *e += contribution)
            .or_insert(contribution);
    }

    /// Sort, deduplicate, and emit a `PauliSum`. Entries whose accumulated
    /// coefficient is exactly `0+0i` are dropped.
    pub fn finalize(self) -> PauliSum<W> {
        let zero = Complex64::new(0.0, 0.0);
        let mut entries: Vec<(PauliString<W>, Complex64)> = self
            .map
            .into_iter()
            .filter(|(_, c)| *c != zero)
            .collect();
        entries.sort_by(|a, b| (&a.0.x, &a.0.z).cmp(&(&b.0.x, &b.0.z)));
        let n = entries.len();
        let mut x = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut coeff = Vec::with_capacity(n);
        for (p, c) in entries {
            x.push(p.x);
            z.push(p.z);
            coeff.push(c);
        }
        PauliSum {
            x,
            z,
            coeff,
            num_qubits: self.num_qubits,
        }
    }
}

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::*;

    #[test]
    fn finalize_empty_accumulator_is_empty() {
        let acc = BuildAccumulator::<1>::new(4);
        let s = acc.finalize();
        assert!(s.is_empty());
        assert_eq!(s.num_qubits(), 4);
        s.assert_invariants();
    }

    #[test]
    fn add_term_dup_sums_coeffs() {
        let mut acc = BuildAccumulator::<1>::new(4);
        let p = PauliString::<1>::x(2);
        acc.add_term(p, Phase::ONE, Complex64::new(1.0, 0.0));
        acc.add_term(p, Phase::ONE, Complex64::new(2.5, -1.0));
        let s = acc.finalize();
        assert_eq!(s.len(), 1);
        assert_eq!(s.coeff()[0], Complex64::new(3.5, -1.0));
        s.assert_invariants();
    }

    #[test]
    fn add_term_phase_one_is_identity_factor() {
        let mut acc = BuildAccumulator::<1>::new(4);
        let p = PauliString::<1>::x(0);
        acc.add_term(p, Phase::ONE, Complex64::new(2.0, 3.0));
        let s = acc.finalize();
        assert_eq!(s.coeff()[0], Complex64::new(2.0, 3.0));
    }

    #[test]
    fn add_term_phase_i_multiplies_by_i() {
        let mut acc = BuildAccumulator::<1>::new(4);
        let p = PauliString::<1>::x(0);
        // (2 + 3i) * i = -3 + 2i
        acc.add_term(p, Phase::I, Complex64::new(2.0, 3.0));
        let s = acc.finalize();
        assert_eq!(s.coeff()[0], Complex64::new(-3.0, 2.0));
    }

    #[test]
    fn add_term_phase_minus_one_negates() {
        let mut acc = BuildAccumulator::<1>::new(4);
        let p = PauliString::<1>::x(0);
        acc.add_term(p, Phase::MINUS_ONE, Complex64::new(2.0, 3.0));
        let s = acc.finalize();
        assert_eq!(s.coeff()[0], Complex64::new(-2.0, -3.0));
    }

    #[test]
    fn add_term_phase_minus_i_multiplies_by_minus_i() {
        let mut acc = BuildAccumulator::<1>::new(4);
        let p = PauliString::<1>::x(0);
        // (2 + 3i) * -i = 3 - 2i
        acc.add_term(p, Phase::MINUS_I, Complex64::new(2.0, 3.0));
        let s = acc.finalize();
        assert_eq!(s.coeff()[0], Complex64::new(3.0, -2.0));
    }

    #[test]
    fn add_term_cancellation_drops() {
        let mut acc = BuildAccumulator::<1>::new(4);
        let p = PauliString::<1>::x(0);
        acc.add_term(p, Phase::ONE, Complex64::new(1.0, 0.0));
        acc.add_term(p, Phase::MINUS_ONE, Complex64::new(1.0, 0.0));
        let s = acc.finalize();
        assert!(s.is_empty());
        s.assert_invariants();
    }

    #[test]
    fn finalize_sorts_by_lex_key() {
        // Insert keys out of order and confirm finalize emits them sorted by
        // (x, z) lex. Use Z(0), X(0), X(1) — sorted: Z(0)=(0,1), X(0)=(1,0),
        // X(1)=(2,0) (lex on x first, then z).
        let mut acc = BuildAccumulator::<1>::new(4);
        acc.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(3.0, 0.0));
        acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
        acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(2.0, 0.0));
        let s = acc.finalize();
        assert_eq!(s.len(), 3);
        assert_eq!(s.x()[0], [0u64]);
        assert_eq!(s.z()[0], [1u64]);
        assert_eq!(s.coeff()[0], Complex64::new(1.0, 0.0));
        assert_eq!(s.x()[1], [1u64]);
        assert_eq!(s.coeff()[1], Complex64::new(2.0, 0.0));
        assert_eq!(s.x()[2], [2u64]);
        assert_eq!(s.coeff()[2], Complex64::new(3.0, 0.0));
        s.assert_invariants();
    }

    #[test]
    fn finalize_w2_across_word_boundary() {
        let mut acc = BuildAccumulator::<2>::new(128);
        acc.add_term(PauliString::<2>::x(64), Phase::ONE, Complex64::new(1.0, 0.0));
        acc.add_term(PauliString::<2>::x(0), Phase::ONE, Complex64::new(2.0, 0.0));
        acc.add_term(PauliString::<2>::z(127), Phase::ONE, Complex64::new(3.0, 0.0));
        let s = acc.finalize();
        assert_eq!(s.len(), 3);
        s.assert_invariants();
    }

    #[test]
    fn add_term_phase_new_wraps_mod_4() {
        // Phase::new(5) reduces to Phase::I — multiplies by i.
        let mut acc = BuildAccumulator::<1>::new(4);
        let p = PauliString::<1>::x(0);
        acc.add_term(p, Phase::new(5), Complex64::new(2.0, 0.0));
        let s = acc.finalize();
        assert_eq!(s.coeff()[0], Complex64::new(0.0, 2.0));
    }

    #[test]
    fn finalize_with_capacity_preallocated() {
        // with_capacity should behave identically to new() for correctness.
        let mut acc = BuildAccumulator::<1>::with_capacity(4, 16);
        acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(1.0, 0.0));
        let s = acc.finalize();
        assert_eq!(s.len(), 1);
        s.assert_invariants();
    }
}
