//! Shared result type and the accumulate-into-a-`PauliSum` step.

use num_complex::Complex64;
use paulistrings::{BuildAccumulator, PauliString, PauliSum, PhaseStats, Phase};

use crate::workload::W;

pub type Key = ([u64; W], [u64; W]);

/// One measured propagation.
pub struct RunResult {
    pub sum: PauliSum<W>,
    pub wall_ns: u64,
    /// Resident terms after each layer.
    pub terms_out: Vec<usize>,
    pub layer_wall_ns: Option<Vec<u64>>,
    pub phase: Option<PhaseStats>,
    /// Realised bucket count of the returned sum (running max under grow-only rebucket).
    pub buckets: Option<usize>,
}

impl RunResult {
    pub fn peak_terms(&self) -> usize {
        self.terms_out.iter().copied().max().unwrap_or(self.sum.len())
    }
}

/// Materialise `(key, coeff)` rows into a `PauliSum` once, at the end of a run.
pub fn materialize<'a>(num_qubits: usize, rows: impl ExactSizeIterator<Item = (Key, Complex64)>) -> PauliSum<W> {
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, rows.len());
    for ((x, z), c) in rows {
        acc.add_term(PauliString::<W> { x, z }, Phase::ONE, c);
    }
    acc.finalize()
}

pub const ZERO: Complex64 = Complex64 { re: 0.0, im: 0.0 };
