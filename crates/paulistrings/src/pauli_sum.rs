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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductState {
    /// `|+…+⟩`, the `+1` eigenstate of `X` on every qubit.
    XPlus,
    /// `|+i…+i⟩`, the `+1` eigenstate of `Y` on every qubit.
    YPlus,
    /// `|0…0⟩`, the `+1` eigenstate of `Z` on every qubit.
    ZPlus,
}

#[cfg(test)]
impl<const W: usize> PauliSum<W> {
    /// Test-only helper: build a `PauliSum<W>` from `(pauli_str, coeff)`
    /// pairs. Each `pauli_str` is a sequence of `I/X/Y/Z` characters where
    /// index `i` of the string corresponds to qubit `i`. `Y` characters
    /// fold one factor of `i` into the coefficient — the bitstring image
    /// of `Y_canonical` is `(x=1, z=1)` with an implicit `i` factor, so
    /// `Y_canonical = i · (x=1, z=1)`.
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
            let mut phase = Phase::ONE;
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
                        phase += Phase::I;
                    }
                    other => panic!("unexpected Pauli char {:?} (expected I/X/Y/Z)", other),
                }
            }
            let p = PauliString::<W> { x, z };
            acc.add_term(p, phase, *c);
        }
        acc.finalize()
    }
}
