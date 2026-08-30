//! `IdentityChannel` — no-op channel that emits its input unchanged.
//!
//! Useful as a sanity scaffold for the sort-merge engine (§5) and as a
//! neutral element when composing circuits.

use super::{Channel, OutputBuffer};
use num_complex::Complex64;

/// A channel that maps every input Pauli to itself with the same coefficient.
///
/// `support()` is empty, so the engine's bucket layout collapses to a single
/// bucket and the only effect is to copy the input through. `max_fanout()`
/// is `1`.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityChannel {
    support: [u32; 0],
}

impl IdentityChannel {
    /// Construct an identity channel.
    pub const fn new() -> Self {
        Self { support: [] }
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
mod tests {
    use super::*;
    use crate::pauli_string::PauliString;

    #[test]
    fn max_fanout_is_one() {
        let id = IdentityChannel::new();
        assert_eq!(<IdentityChannel as Channel<1>>::max_fanout(&id), 1);
        assert_eq!(<IdentityChannel as Channel<2>>::max_fanout(&id), 1);
    }

    #[test]
    fn support_is_empty() {
        let id = IdentityChannel::new();
        assert_eq!(<IdentityChannel as Channel<1>>::support(&id), [0u64; 1]);
        assert_eq!(<IdentityChannel as Channel<2>>::support(&id), [0u64; 2]);
    }

    #[test]
    fn apply_emits_input_unchanged_w1() {
        let id = IdentityChannel::new();
        // Build XYZ on qubits 0,1,2 by composition.
        let mut input = PauliString::<1>::x(0);
        let _ = input.mul_assign(&PauliString::<1>::y(1));
        let _ = input.mul_assign(&PauliString::<1>::z(2));

        let mut x = vec![[0u64; 1]; 1];
        let mut z = vec![[0u64; 1]; 1];
        let mut c = vec![Complex64::new(0.0, 0.0); 1];
        let mut len = 0usize;
        let coeff = Complex64::new(2.0, 3.0);
        {
            let mut out = OutputBuffer::<1> {
                x: &mut x,
                z: &mut z,
                coeff: &mut c,
                len: &mut len,
            };
            id.apply(&input.x, &input.z, coeff, &mut out);
            assert_eq!(*out.len, 1);
        }
        assert_eq!(x[0], input.x);
        assert_eq!(z[0], input.z);
        assert_eq!(c[0], coeff);
    }

    #[test]
    fn apply_emits_input_unchanged_w2() {
        let id = IdentityChannel::new();
        // X on qubit 70 lands in word 1 — exercises the multi-word case.
        let input = PauliString::<2>::x(70);

        let mut x = vec![[0u64; 2]; 1];
        let mut z = vec![[0u64; 2]; 1];
        let mut c = vec![Complex64::new(0.0, 0.0); 1];
        let mut len = 0usize;
        let coeff = Complex64::new(-1.5, 0.25);
        {
            let mut out = OutputBuffer::<2> {
                x: &mut x,
                z: &mut z,
                coeff: &mut c,
                len: &mut len,
            };
            id.apply(&input.x, &input.z, coeff, &mut out);
            assert_eq!(*out.len, 1);
        }
        assert_eq!(x[0], input.x);
        assert_eq!(z[0], input.z);
        assert_eq!(c[0], coeff);
    }
}
