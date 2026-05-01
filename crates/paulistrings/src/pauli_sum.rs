//! `PauliSum<W>` — weighted sum of Pauli strings in structure-of-arrays form.
//!
//! See design doc §3.2.
//!
//! Invariant: parallel arrays `x` and `z` are sorted in lexicographic order as
//! a single key, and no two entries share a key. Every public operation
//! either preserves this invariant or returns a fresh `PauliSum` that does.

#![allow(unused)]

use num_complex::Complex64;

/// Weighted sum of Pauli operators, stored SoA, sorted and deduplicated.
#[derive(Clone, Debug, Default)]
pub struct PauliSum<const W: usize> {
    pub(crate) x: Vec<[u64; W]>,
    pub(crate) z: Vec<[u64; W]>,
    pub(crate) coeff: Vec<Complex64>,
    pub(crate) num_qubits: usize,
}

impl<const W: usize> PauliSum<W> {
    /// Empty sum on `num_qubits` qubits. Caller is responsible for ensuring
    /// `num_qubits <= 64 * W`.
    pub fn empty(num_qubits: usize) -> Self {
        debug_assert!(num_qubits <= 64 * W);
        Self {
            x: Vec::new(),
            z: Vec::new(),
            coeff: Vec::new(),
            num_qubits,
        }
    }

    /// Number of qubits this sum is defined over.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of non-identity terms after deduplication.
    #[inline]
    pub fn len(&self) -> usize {
        self.coeff.len()
    }

    /// `true` iff the sum has no terms.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coeff.is_empty()
    }

    /// Read-only view of the X-part column.
    #[inline]
    pub fn x(&self) -> &[[u64; W]] {
        &self.x
    }

    /// Read-only view of the Z-part column.
    #[inline]
    pub fn z(&self) -> &[[u64; W]] {
        &self.z
    }

    /// Read-only view of the coefficient column.
    #[inline]
    pub fn coeff(&self) -> &[Complex64] {
        &self.coeff
    }

    /// Sum of two `PauliSum`s. Linear-time merge; preserves the sorted invariant.
    pub fn add(&self, _other: &Self) -> Self {
        todo!("§3.2: parallel-array merge of two sorted SoA sums")
    }

    /// Multiply every coefficient by `c` in place.
    pub fn scale(&mut self, _c: Complex64) {
        todo!("§3.2: rayon parallel iter over coeff column")
    }

    /// Locate a Pauli key by binary search; returns `Ok(idx)` if present,
    /// `Err(idx)` for the insertion point otherwise.
    pub fn find(&self, _x: &[u64; W], _z: &[u64; W]) -> Result<usize, usize> {
        todo!("§3.2: binary search on (x, z) lex key")
    }

    /// Drop terms whose coefficient magnitude is `<= eps`. Preserves sort.
    pub fn truncate_by_magnitude(&mut self, _eps: f64) {
        todo!("§3.2 / §7: filter coeff column, compact x/z accordingly")
    }

    /// Debug-only invariant check. No-op in release builds.
    #[cfg(debug_assertions)]
    pub fn assert_invariants(&self) {
        assert_eq!(self.x.len(), self.z.len());
        assert_eq!(self.x.len(), self.coeff.len());
        for i in 1..self.x.len() {
            let prev = (&self.x[i - 1], &self.z[i - 1]);
            let cur = (&self.x[i], &self.z[i]);
            assert!(prev < cur, "PauliSum out of order at {}", i);
        }
    }
}
