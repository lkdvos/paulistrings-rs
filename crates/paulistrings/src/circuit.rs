//! [`Circuit<W>`], an ordered list of channels.

use crate::channel::Channel;

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
        Self {
            num_qubits,
            channels: Vec::new(),
        }
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
}

#[cfg(test)]
mod tests;
