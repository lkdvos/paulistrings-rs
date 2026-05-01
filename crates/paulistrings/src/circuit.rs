//! `Circuit<W>` — an ordered sequence of channels.

#![allow(unused)]

use crate::channel::Channel;

/// A circuit on `num_qubits` qubits, stored as a heterogeneous list of
/// channels. Boxed `dyn Channel` rather than a generic enum so user-defined
/// channel types can be appended at runtime; the engine sees only the trait.
pub struct Circuit<const W: usize> {
    pub num_qubits: usize,
    pub channels: Vec<Box<dyn Channel<W>>>,
}

impl<const W: usize> Circuit<W> {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            channels: Vec::new(),
        }
    }

    /// Append a channel to the circuit.
    pub fn push<C: Channel<W> + 'static>(&mut self, c: C) {
        self.channels.push(Box::new(c));
    }

    /// Number of channels (gates + noise) currently in the circuit.
    #[inline]
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }
}
