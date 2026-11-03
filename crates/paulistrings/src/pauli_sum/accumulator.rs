//! [`BuildAccumulator<W>`], the hashmap ingestion path from unsorted terms to a [`crate::PauliSum`].

use crate::coefficient::Coefficient;
use crate::pauli_string::PauliString;
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::storage::DEFAULT_TARGET_BUCKET_LEN;
use crate::pauli_sum::storage::{desired_bits, DEFAULT_HASH_SEED, DEFAULT_MIN_BUCKETS};
use crate::pauli_sum::PauliSum;
use hashbrown::HashMap;
use rustc_hash::FxBuildHasher;

/// Incremental builder for a [`crate::PauliSum`] from unsorted terms; repeated keys sum.
pub struct BuildAccumulator<const W: usize, C: Coefficient = num_complex::Complex64> {
    map: HashMap<PauliString<W>, C, FxBuildHasher>,
    num_qubits: usize,
}

impl<const W: usize, C: Coefficient> BuildAccumulator<W, C> {
    /// New empty accumulator targeting `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Self {
        Self { map: HashMap::with_hasher(FxBuildHasher), num_qubits }
    }

    /// Allocate up-front for at least `capacity` distinct Pauli keys.
    pub fn with_capacity(num_qubits: usize, capacity: usize) -> Self {
        Self { map: HashMap::with_capacity_and_hasher(capacity, FxBuildHasher), num_qubits }
    }

    /// Add `c · p`.
    pub fn add_term(&mut self, p: PauliString<W>, c: C) {
        self.map.entry(p).and_modify(|e| *e += c).or_insert(c);
    }

    /// Emit the [`crate::PauliSum`], dropping keys whose coefficients summed to exactly zero.
    pub fn finalize(self) -> PauliSum<W, C> {
        let zero = C::ZERO;
        let mut entries: Vec<(PauliString<W>, C)> =
            self.map.into_iter().filter(|(_, c)| *c != zero).collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let n = entries.len();
        let mut x = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut coeff = Vec::with_capacity(n);
        for (p, c) in entries {
            x.push(p.x);
            z.push(p.z);
            coeff.push(c);
        }
        let bits = desired_bits(n, DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS)
            .min(crate::pauli_sum::hash::max_hash_bits(self.num_qubits));
        let hash = Gf2Hash::new(self.num_qubits, bits, DEFAULT_HASH_SEED);
        PauliSum::from_key_sorted(&x, &z, &coeff, hash, self.num_qubits)
    }
}

#[cfg(test)]
mod tests;
