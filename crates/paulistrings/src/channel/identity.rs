//! [`IdentityChannel`], the channel that emits its input unchanged.

use super::{Channel, OutputBuffer};
use num_complex::Complex64;

/// The identity channel, with empty support.
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
