//! [`BuildAccumulator<W>`], the hashmap ingestion path from unsorted terms to a [`crate::PauliSum`].

use crate::pauli_string::PauliString;
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::storage::DEFAULT_TARGET_BUCKET_LEN;
use crate::pauli_sum::storage::{desired_bits, DEFAULT_HASH_SEED, DEFAULT_MIN_BUCKETS};
use crate::pauli_sum::PauliSum;
use crate::phase::Phase;
use hashbrown::HashMap;
use num_complex::Complex64;
use rustc_hash::FxBuildHasher;

/// Incremental builder for a [`crate::PauliSum`] from unsorted terms; repeated keys sum.
///
/// ```
/// use paulistrings::{BuildAccumulator, PauliString, Phase};
/// use num_complex::Complex64;
///
/// let mut accumulator = BuildAccumulator::<1>::new(2);
/// // The key (x=1, z=1) is Y; Phase::I folds a product's i into the coefficient.
/// accumulator.add_term(PauliString::<1> { x: [1], z: [1] }, Phase::I, Complex64::new(1.0, 0.0));
/// accumulator.add_term(PauliString::<1>::z(1), Phase::ONE, Complex64::new(0.5, 0.0));
/// accumulator.add_term(PauliString::<1>::z(1), Phase::ONE, Complex64::new(0.5, 0.0));
/// let sum = accumulator.finalize();
/// assert_eq!(sum.get(&[1], &[1]), Some(Complex64::new(0.0, 1.0)));
/// assert_eq!(sum.get(&[0], &[0b10]), Some(Complex64::new(1.0, 0.0)));
/// ```
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

    /// Allocate up-front for at least `capacity` distinct Pauli keys.
    pub fn with_capacity(num_qubits: usize, capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity_and_hasher(capacity, FxBuildHasher),
            num_qubits,
        }
    }

    /// Add `phase · c · p`.
    pub fn add_term(&mut self, p: PauliString<W>, phase: Phase, c: Complex64) {
        let contribution = phase.apply(c);
        self.map
            .entry(p)
            .and_modify(|e| *e += contribution)
            .or_insert(contribution);
    }

    /// Emit the [`crate::PauliSum`], dropping keys whose coefficients summed to exactly zero.
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
