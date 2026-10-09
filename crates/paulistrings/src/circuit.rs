//! [`Circuit<W>`], an ordered list of channels.

use crate::channel::Channel;

/// An ordered list of channels on `num_qubits` qubits, user-defined [`crate::Channel`] impls included.
///
/// [`crate::Direction::Forward`] applies it in order, [`crate::Direction::Heisenberg`] in reverse with each channel's adjoint.
pub struct Circuit<const W: usize> {
    /// Number of qubits this circuit acts on.
    pub num_qubits: usize,
    /// Channels in application order.
    pub channels: Vec<Box<dyn Channel<W>>>,
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
    pub fn push<C: Channel<W> + 'static>(&mut self, channel: C) {
        self.channels.push(Box::new(channel));
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
