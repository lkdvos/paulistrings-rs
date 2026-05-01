//! `BuildAccumulator<W>` — hashmap-based ingestion path. See §8.2.
//!
//! Used to incrementally build a `PauliSum` from unsorted inputs (Hamiltonian
//! parsing, Python dict construction, etc.). The accumulator is **not** used
//! during propagation — that path is sort-merge only.

#![allow(unused)]

use crate::pauli_string::PauliString;
use crate::pauli_sum::PauliSum;
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

    /// Add `i^phase * c * p` to the accumulator. The caller provides the
    /// phase as a separate `u8` (in `0..4`); it is folded into `c` before
    /// the upsert. `p` is taken as-is and used as the map key.
    pub fn add_term(&mut self, _p: PauliString<W>, _phase: u8, _c: Complex64) {
        todo!("§8.2: fold i^phase into c, then upsert into map")
    }

    /// Sort, deduplicate, and emit a `PauliSum`.
    pub fn finalize(self) -> PauliSum<W> {
        todo!("§8.2: drain map into parallel Vecs, sort by (x, z) key")
    }
}
