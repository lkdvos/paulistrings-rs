//! [`Circuit<W>`], an ordered list of channels.

use crate::channel::prepared::{Prepare, Prepared};
use crate::channel::Channel;
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::storage::DEFAULT_HASH_SEED;

/// A key shift `(x-mask, z-mask)`: a channel maps a term's key `k` to `k ⊕ shift`.
pub type KeyShift<const W: usize> = ([u64; W], [u64; W]);

/// An ordered list of channels on `num_qubits` qubits, user-defined [`crate::Channel`] impls included.
///
/// [`crate::Direction::Forward`] applies it in order, [`crate::Direction::Heisenberg`] in reverse with each channel's adjoint.
pub struct Circuit<const W: usize> {
    pub(crate) num_qubits: usize,
    /// In application order.
    pub(crate) channels: Vec<Box<dyn Channel<W>>>,
}

impl<const W: usize> Circuit<W> {
    /// Empty circuit on `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Self {
        Self { num_qubits, channels: Vec::new() }
    }

    /// Append a channel to the circuit.
    ///
    /// # Panics
    ///
    /// If the channel's support reaches a qubit at or above [`num_qubits`](Self::num_qubits).
    pub fn push<C: Channel<W> + 'static>(&mut self, channel: C) {
        self.push_boxed(Box::new(channel));
    }

    /// Append an already boxed channel, as [`push`](Self::push) does.
    ///
    /// # Panics
    ///
    /// If the channel's support reaches a qubit at or above [`num_qubits`](Self::num_qubits).
    pub fn push_boxed(&mut self, channel: Box<dyn Channel<W>>) {
        let support = channel.support();
        if let Some(word) = (0..W).rev().find(|&w| support[w] != 0) {
            let highest = 64 * word + 63 - support[word].leading_zeros() as usize;
            assert!(
                highest < self.num_qubits,
                "Circuit::push: {} acts on qubit {highest}, outside this {}-qubit circuit",
                channel.debug_name(),
                self.num_qubits,
            );
        }
        self.channels.push(channel);
    }

    /// Number of qubits this circuit acts on.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The channels in application order.
    pub fn channels(&self) -> &[Box<dyn Channel<W>>] {
        &self.channels
    }

    /// Number of channels in the circuit.
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// `true` iff no channels have been pushed.
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// Every nonzero key shift each channel applies to a term, as `(x-mask, z-mask)`, or `None` for a channel the partitioned engine declines.
    /// A channel moves terms between partitions exactly when the partition rows map one of its shifts to a nonzero label, the test `PartitionPlan` applies, so this is the static communication cost of a choice of rows.
    pub fn deltas(&self) -> Vec<Option<Vec<KeyShift<W>>>> {
        // The hash only sets the bucket deltas, which are not reported, so any hash gives the same shifts.
        let hash = Gf2Hash::<W>::new(self.num_qubits, 0, DEFAULT_HASH_SEED);
        let identity = ([0u64; W], [0u64; W]);
        self.channels
            .iter()
            .map(|channel| {
                channel.prepare(&hash, false).map(|prepared| match prepared {
                    Prepared::Local(ptm) => ptm
                        .deltas()
                        .iter()
                        .map(|delta| delta.mask())
                        .filter(|mask| *mask != identity)
                        .collect(),
                    Prepared::Rotation(rotation) => vec![rotation.generator_mask()],
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
