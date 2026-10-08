//! `IdentityChannel` — no-op channel that emits its input unchanged.
//!
//! Useful as a sanity scaffold for the engine (ARCHITECTURE.md §Engine) and
//! as a neutral element when composing circuits.

use super::{Channel, OutputBuffer};
use num_complex::Complex64;

/// A channel that maps every input Pauli to itself with the same coefficient.
///
/// `support()` is empty, so the engine's bucket layout collapses to a single bucket and the only effect is to copy the input through. `max_fanout()` is `1`.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityChannel;

impl IdentityChannel {
    /// Construct an identity channel.
    pub const fn new() -> Self {
        Self
    }
}

impl<const W: usize> Channel<W> for IdentityChannel {
    #[inline]
    fn max_fanout(&self) -> usize {
        1
    }

    #[inline]
    fn support(&self) -> [u64; W] {
        [0; W]
    }

    #[inline]
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        out.push(*input_x, *input_z, coeff);
    }
}

#[cfg(test)]
mod tests;
