//! [`BuildAccumulator<W>`] — hashmap-based ingestion path.
//!
//! Used to incrementally build a [`PauliSum`] from unsorted inputs (Hamiltonian parsing, Python dict construction, etc.); not used during propagation, which is sort-merge only (see [`engine`]).
//!
//! See the [`PauliSum`] module for a worked example: [`BuildAccumulator::new`] → [`BuildAccumulator::add_term`] → [`BuildAccumulator::finalize`].
//!
//! [`PauliSum`]: crate::PauliSum
//! [`engine`]: crate::engine

use crate::pauli_string::PauliString;
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::storage::{desired_bits, DEFAULT_HASH_SEED, DEFAULT_MIN_BUCKETS};
use crate::pauli_sum::PauliSum;
use crate::pauli_sum::DEFAULT_TARGET_BUCKET_LEN;
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

    /// Add `phase · c · p` to the accumulator. The phase factor is folded into `c` before the upsert; `p` is taken as-is and used as the map key.
    ///
    /// # Examples
    ///
    /// ```
    /// use paulistrings::{BuildAccumulator, PauliString, Phase};
    /// use num_complex::Complex64;
    ///
    /// let mut acc = BuildAccumulator::<1>::new(2);
    /// // Fold a product phase into the stored coefficient: Z·X = +i·Y, so the Y key (x=1, z=1) gets coefficient i.
    /// acc.add_term(
    ///     PauliString::<1> { x: [1], z: [1] },
    ///     Phase::I,
    ///     Complex64::new(1.0, 0.0),
    /// );
    /// let sum = acc.finalize();
    /// assert_eq!(sum.bucket(0).2[0], Complex64::new(0.0, 1.0));
    /// ```
    pub fn add_term(&mut self, p: PauliString<W>, phase: Phase, c: Complex64) {
        let contribution = phase.apply(c);
        self.map
            .entry(p)
            .and_modify(|e| *e += contribution)
            .or_insert(contribution);
    }

    /// Sort, deduplicate, and emit a `PauliSum`. Entries whose accumulated coefficient is exactly `0+0i` are dropped.
    ///
    /// The partition is chosen here, by `desired_bits` under the default seed, so a sum of at most 1024 terms gets a single bucket (plain lex canonical order) and a larger one starts out already sized for the engine.
    pub fn finalize(self) -> PauliSum<W> {
        let zero = Complex64::new(0.0, 0.0);
        let mut entries: Vec<(PauliString<W>, Complex64)> =
            self.map.into_iter().filter(|(_, c)| *c != zero).collect();
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
        let bits = desired_bits(n, DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS);
        let hash = Gf2Hash::new(self.num_qubits, bits, DEFAULT_HASH_SEED);
        PauliSum::from_key_sorted(&x, &z, &coeff, hash, self.num_qubits)
    }
}

#[cfg(test)]
mod tests;
