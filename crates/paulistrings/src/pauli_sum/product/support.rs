//! [`SupportIndex`], a bit-sliced index of which terms of each bucket act on each qubit, so a Liouvillian visits only the pairs whose supports meet.

use rayon::prelude::*;

use crate::coefficient::Coefficient;
use crate::pauli_sum::storage::{BucketColumns, PauliSum};

/// Above this weight a term's candidate mask costs more to assemble than scanning the bucket.
pub(crate) const SCAN_ABOVE_WEIGHT: u32 = 64;

/// Per bucket, row `q` is a bitset over the bucket's terms with bit `i` set iff term `i` acts on qubit `q`.
pub(crate) struct SupportIndex {
    /// Bucket `b`'s rows, `num_qubits` of `words[b]` words each.
    rows: Vec<Vec<u64>>,
    words: Vec<usize>,
}

fn bucket_rows<const W: usize, C: Coefficient>(
    columns: &BucketColumns<W, C>,
    num_qubits: usize,
) -> (Vec<u64>, usize) {
    let words = columns.len().div_ceil(64);
    let mut rows = vec![0u64; num_qubits * words];
    for i in 0..columns.len() {
        for w in 0..W {
            let mut support = columns.x[i][w] | columns.z[i][w];
            while support != 0 {
                let q = w * 64 + support.trailing_zeros() as usize;
                rows[q * words + i / 64] |= 1 << (i % 64);
                support &= support - 1;
            }
        }
    }
    (rows, words)
}

impl SupportIndex {
    pub(crate) fn new<const W: usize, C: Coefficient>(sum: &PauliSum<W, C>) -> Self {
        let (rows, words) =
            sum.buckets().par_iter().map(|columns| bucket_rows(columns, sum.num_qubits())).unzip();
        Self { rows, words }
    }

    /// Overwrite `candidates` with the bitset of bucket `b`'s terms acting on some qubit of `support`.
    #[inline]
    pub(crate) fn candidates<const W: usize>(
        &self,
        b: usize,
        support: &[u64; W],
        candidates: &mut Vec<u64>,
    ) {
        let words = self.words[b];
        let rows = &self.rows[b];
        candidates.clear();
        candidates.resize(words, 0);
        for (w, &mask) in support.iter().enumerate() {
            let mut mask = mask;
            while mask != 0 {
                let q = w * 64 + mask.trailing_zeros() as usize;
                let row = &rows[q * words..(q + 1) * words];
                for (c, r) in candidates.iter_mut().zip(row) {
                    *c |= r;
                }
                mask &= mask - 1;
            }
        }
    }
}

#[cfg(test)]
mod tests;
