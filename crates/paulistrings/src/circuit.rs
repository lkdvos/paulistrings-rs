//! [`Circuit<W>`] — an ordered sequence of channels.

use crate::channel::Channel;

/// A circuit on `num_qubits` qubits, stored as a heterogeneous list of
/// channels.
///
/// Channels are held as `Box<dyn `[`Channel`]`>` rather than a generic enum so
/// user-defined channel types can be appended at runtime; the engine sees only
/// the trait. The engine reads the list in order for
/// [`Direction::Forward`](crate::Direction::Forward) propagation and in
/// reverse (using each channel's adjoint) for
/// [`Direction::Heisenberg`](crate::Direction::Heisenberg).
///
/// # Examples
///
/// ```
/// use paulistrings::{Circuit, channel::Clifford1Q};
///
/// let mut c = Circuit::<1>::new(2);
/// c.push(Clifford1Q::h(0));
/// c.push(Clifford1Q::s(1));
/// assert_eq!(c.len(), 2);
/// ```
///
/// [`Channel`]: crate::Channel
pub struct Circuit<const W: usize> {
    /// Number of qubits this circuit acts on. Constrains the support of
    /// channels that can be pushed.
    pub num_qubits: usize,
    /// Channels in application order. Index `0` is applied first under
    /// [`Direction::Forward`](crate::Direction::Forward).
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
    pub fn push<C: Channel<W> + 'static>(&mut self, c: C) {
        self.channels.push(Box::new(c));
    }

    /// Number of channels (gates + noise) currently in the circuit.
    #[inline]
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// `true` iff no channels have been pushed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }
}
