//! [`Channel<W>`] — unified abstraction for gates and noise.
//!
//! Every operation on a [`PauliSum`] (Clifford gate, Pauli rotation,
//! arbitrary unitary, noise channel) maps a single Pauli string to a small
//! weighted sum of Pauli strings. The trait formalizes that mapping; the
//! engine consumes it via the sort-merge pipeline (see [`engine`]).
//!
//! Built-ins in this module:
//!
//! - [`Clifford1Q`], [`Clifford2Q`] — table-driven Clifford gates with
//!   `MAX_FANOUT = 1`.
//! - [`PauliRotation`] — `exp(-i·θ·P/2)` with `MAX_FANOUT = 2` (commuting
//!   inputs collapse to fanout-1 at runtime).
//! - [`GeneralUnitary1Q`], [`GeneralUnitary2Q`] — generic unitaries stored
//!   as Pauli-expansion tables.
//! - [`Depolarizing`], [`Dephasing`] — coefficient-rescaling noise
//!   (`MAX_FANOUT = 1`).
//! - [`AmplitudeDamping`] — the one built-in with `MAX_FANOUT = 2`.
//! - [`IdentityChannel`] — pass-through, used in tests and as a neutral
//!   composition element.
//!
//! See design doc §6.
//!
//! # Implementing a custom channel
//!
//! Implement the trait directly; the engine treats your type as just another
//! `Box<dyn `[`Channel<W>`]`>` inside a [`Circuit`]. Three required methods
//! plus an optional [`Channel::apply_adjoint`] override.
//!
//! ```
//! use paulistrings::{Channel, OutputBuffer};
//! use num_complex::Complex64;
//!
//! /// Multiplies every input coefficient by a complex factor, with no
//! /// support and `MAX_FANOUT = 1`.
//! struct GlobalPhase {
//!     factor: Complex64,
//! }
//!
//! impl<const W: usize> Channel<W> for GlobalPhase {
//!     fn max_fanout(&self) -> usize { 1 }
//!     fn support(&self) -> [u64; W] { [0; W] }
//!     fn apply(
//!         &self,
//!         input_x: &[u64; W],
//!         input_z: &[u64; W],
//!         coeff: Complex64,
//!         out: &mut OutputBuffer<'_, W>,
//!     ) {
//!         out.push(*input_x, *input_z, coeff * self.factor);
//!     }
//! }
//!
//! let ch = GlobalPhase {
//!     factor: Complex64::new(0.0, 1.0),
//! };
//! let _: Box<dyn Channel<1>> = Box::new(ch);
//! ```
//!
//! [`PauliSum`]: crate::PauliSum
//! [`engine`]: crate::engine
//! [`Circuit`]: crate::Circuit

#![allow(unused)]

pub mod clifford;
pub mod identity;
pub mod noise;
pub mod prepared;
pub mod rotation;
pub mod unitary;

pub use clifford::{Clifford1Q, Clifford2Q};
pub use identity::IdentityChannel;
pub use noise::{AmplitudeDamping, Dephasing, Depolarizing};
pub use rotation::PauliRotation;
pub use unitary::{GeneralUnitary1Q, GeneralUnitary2Q};

use crate::bucket::hash::Gf2Hash;
use num_complex::Complex64;
use prepared::Prepared;

/// Pre-allocated, fixed-capacity SoA scratch buffer for channel outputs.
///
/// Sized by the engine to `n_in · channel.max_fanout()` so that `apply` can
/// write without dynamic growth. Required for GPU correctness and for CPU
/// hot-loop performance. Channel impls write via [`OutputBuffer::push`].
pub struct OutputBuffer<'a, const W: usize> {
    /// X-part column. Length equals the buffer's capacity.
    pub x: &'a mut [[u64; W]],
    /// Z-part column. Length equals the buffer's capacity.
    pub z: &'a mut [[u64; W]],
    /// Coefficient column. Length equals the buffer's capacity.
    pub coeff: &'a mut [Complex64],
    /// Cursor into the slices; `apply` writes at `len` and advances.
    pub len: &'a mut usize,
}

impl<'a, const W: usize> OutputBuffer<'a, W> {
    /// Append one term to the buffer at the current cursor.
    ///
    /// Capacity is `self.x.len()`; the engine sizes the slices to
    /// `channel.max_fanout()` per input term, so a `Channel::apply` body
    /// must not push more than its declared `max_fanout`. Out-of-range
    /// writes are caught by slice bounds-checking (and, in debug builds,
    /// by an explicit assertion with a clearer message).
    #[inline]
    pub fn push(&mut self, x: [u64; W], z: [u64; W], c: Complex64) {
        debug_assert!(
            *self.len < self.x.len(),
            "OutputBuffer overflow: {} pushes into a buffer of capacity {}",
            *self.len + 1,
            self.x.len()
        );
        let i = *self.len;
        self.x[i] = x;
        self.z[i] = z;
        self.coeff[i] = c;
        *self.len = i + 1;
    }

    /// Reset the cursor to zero so the same backing storage can be reused
    /// for the next input term without reallocation.
    #[inline]
    pub fn clear(&mut self) {
        *self.len = 0;
    }
}

/// Pack a list of qubit indices into a [`Channel::support`] bitmask.
///
/// Bit `q % 64` of word `q / 64` is set for each `q` in `qubits`. Order and
/// duplicates in `qubits` do not matter — the result is a plain set.
#[inline]
pub fn support_mask<const W: usize>(qubits: &[u32]) -> [u64; W] {
    let mut mask = [0u64; W];
    for &q in qubits {
        debug_assert!((q as usize) < 64 * W, "qubit {q} out of range for W={W}");
        mask[q as usize / 64] |= 1u64 << (q % 64);
    }
    mask
}

/// Anything that maps a Pauli string to a small weighted sum of Pauli strings.
///
/// [`Channel::max_fanout`] is a method (not an associated `const`) so the
/// trait stays `dyn`-compatible — [`Circuit`](crate::Circuit) stores
/// `Box<dyn Channel<W>>` to keep the channel set open for user extensions.
/// For built-in channels the returned value is a compile-time constant so
/// the engine still gets constant-folded buffer sizing once the concrete
/// type is in hand.
///
/// See the [module-level docs](self) for an `impl Channel` example.
pub trait Channel<const W: usize>: Send + Sync {
    /// Maximum number of output terms produced per input term. Used by the
    /// engine to size the scratch buffer up-front.
    fn max_fanout(&self) -> usize;

    /// Qubits this channel acts on, packed as one combined per-qubit bitmask
    /// (bit `q` set iff qubit `q` is in the support), one word per `W`.
    /// Outputs differ from inputs only at these bit positions; the engine
    /// uses this for bucket layout (v0.2 §2, v0.3 §2).
    ///
    /// Build one with [`support_mask`] from a list of qubit indices.
    fn support(&self) -> [u64; W];

    /// Apply the channel to a single input term, writing outputs to `out`.
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    );

    /// Apply the channel's *adjoint* to a single input term, writing outputs
    /// to `out`. Used by the engine in `Direction::Heisenberg` mode for
    /// backpropagating observables.
    ///
    /// The default implementation is `self.apply(...)`, i.e. assumes the
    /// channel is self-adjoint. Channels that are not self-adjoint
    /// (`PauliRotation`, `Clifford1Q::s`) override this. The design doc
    /// (§8) does not pin down a mechanism; this is the v0.1 convention.
    fn apply_adjoint(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        self.apply(input_x, input_z, coeff, out);
    }

    /// Prepare this channel for one layer of the bucketed engine.
    ///
    /// The default derives a dense local Pauli-transfer matrix by probing
    /// `apply` on the `4^|support|` local basis Paulis, so **a channel that
    /// implements `apply` gets the bucketed engine for free** — the research
    /// extensibility surface of §6 does not grow. Override only when the support
    /// is wider than [`prepared::MAX_LOCAL_SUPPORT`] and a tighter description
    /// exists; among the built-ins, only `PauliRotation` above generator weight
    /// 2 needs to.
    ///
    /// `None` means "this channel cannot be bucketed": the engine falls back to
    /// the v0.1 whole-sum path, which is correct but slower. It is a performance
    /// fallback, never a correctness compromise.
    ///
    /// # Contract
    ///
    /// Implementors must honour bounded support: the output amplitude may depend
    /// on the input only through its bits at [`Channel::support`] positions.
    /// Deriving assumes it; debug builds check it against an all-ones background,
    /// and a property test checks it against randomized full-width inputs.
    fn prepare(&self, hash: &Gf2Hash<W>, adjoint: bool) -> Option<Prepared<W>> {
        Prepared::derive_local(self, hash, adjoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- support_mask (v0.3 C.1) ----

    #[test]
    fn support_mask_packs_cross_word_qubits() {
        // Qubit 70 at W=2 lands in word 1, bit 6.
        let mask: [u64; 2] = support_mask(&[5, 70]);
        assert_eq!(mask[0], 1u64 << 5);
        assert_eq!(mask[1], 1u64 << 6);
    }

    #[test]
    fn support_mask_is_order_and_duplicate_insensitive() {
        let a: [u64; 1] = support_mask(&[3, 1, 3]);
        let b: [u64; 1] = support_mask(&[1, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn support_mask_of_empty_is_zero() {
        let mask: [u64; 2] = support_mask(&[]);
        assert_eq!(mask, [0, 0]);
    }

    #[allow(clippy::type_complexity)]
    fn alloc_bufs<const W: usize>(
        n: usize,
    ) -> (Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>, usize) {
        (
            vec![[0u64; W]; n],
            vec![[0u64; W]; n],
            vec![Complex64::new(0.0, 0.0); n],
            0usize,
        )
    }

    #[test]
    fn push_writes_at_cursor_w1() {
        let (mut x, mut z, mut c, mut len) = alloc_bufs::<1>(4);
        {
            let mut buf = OutputBuffer::<1> {
                x: &mut x,
                z: &mut z,
                coeff: &mut c,
                len: &mut len,
            };
            buf.push([0xAA], [0xBB], Complex64::new(1.0, 2.0));
            buf.push([0xCC], [0xDD], Complex64::new(3.0, 4.0));
            assert_eq!(*buf.len, 2);
        }
        assert_eq!(x[0], [0xAA]);
        assert_eq!(z[0], [0xBB]);
        assert_eq!(c[0], Complex64::new(1.0, 2.0));
        assert_eq!(x[1], [0xCC]);
        assert_eq!(z[1], [0xDD]);
        assert_eq!(c[1], Complex64::new(3.0, 4.0));
        // remaining slots untouched
        assert_eq!(x[2], [0]);
        assert_eq!(x[3], [0]);
        assert_eq!(c[3], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn push_writes_at_cursor_w2() {
        let (mut x, mut z, mut c, mut len) = alloc_bufs::<2>(3);
        {
            let mut buf = OutputBuffer::<2> {
                x: &mut x,
                z: &mut z,
                coeff: &mut c,
                len: &mut len,
            };
            buf.push([0x11, 0x22], [0x33, 0x44], Complex64::new(5.0, 6.0));
            assert_eq!(*buf.len, 1);
        }
        assert_eq!(x[0], [0x11, 0x22]);
        assert_eq!(z[0], [0x33, 0x44]);
        assert_eq!(c[0], Complex64::new(5.0, 6.0));
    }

    #[test]
    #[should_panic]
    fn push_panics_when_full() {
        let (mut x, mut z, mut c, mut len) = alloc_bufs::<1>(2);
        let mut buf = OutputBuffer::<1> {
            x: &mut x,
            z: &mut z,
            coeff: &mut c,
            len: &mut len,
        };
        buf.push([0; 1], [0; 1], Complex64::new(1.0, 0.0));
        buf.push([0; 1], [0; 1], Complex64::new(1.0, 0.0));
        buf.push([0; 1], [0; 1], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn clear_resets_cursor() {
        let (mut x, mut z, mut c, mut len) = alloc_bufs::<1>(4);
        {
            let mut buf = OutputBuffer::<1> {
                x: &mut x,
                z: &mut z,
                coeff: &mut c,
                len: &mut len,
            };
            buf.push([0xAA], [0xBB], Complex64::new(1.0, 0.0));
            buf.push([0xCC], [0xDD], Complex64::new(2.0, 0.0));
            assert_eq!(*buf.len, 2);
            buf.clear();
            assert_eq!(*buf.len, 0);
            buf.push([0xEE], [0xFF], Complex64::new(3.0, 0.0));
            assert_eq!(*buf.len, 1);
        }
        // The post-clear push lands at slot 0, overwriting the prior contents.
        assert_eq!(x[0], [0xEE]);
        assert_eq!(z[0], [0xFF]);
        assert_eq!(c[0], Complex64::new(3.0, 0.0));
        // Slot 1 was written before the clear and is left as-is.
        assert_eq!(x[1], [0xCC]);
        assert_eq!(z[1], [0xDD]);
    }

    #[test]
    fn reuse_does_not_grow_backing_vecs() {
        let cap = 4;
        let mut x: Vec<[u64; 1]> = vec![[0u64; 1]; cap];
        let mut z: Vec<[u64; 1]> = vec![[0u64; 1]; cap];
        let mut c: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); cap];
        assert_eq!(x.capacity(), cap);
        assert_eq!(z.capacity(), cap);
        assert_eq!(c.capacity(), cap);
        let mut len = 0usize;
        for i in 0..100u64 {
            len = 0;
            let mut buf = OutputBuffer::<1> {
                x: &mut x,
                z: &mut z,
                coeff: &mut c,
                len: &mut len,
            };
            buf.push([i], [0], Complex64::new(i as f64, 0.0));
            buf.push([i + 1], [0], Complex64::new((i + 1) as f64, 0.0));
        }
        assert_eq!(x.capacity(), cap);
        assert_eq!(z.capacity(), cap);
        assert_eq!(c.capacity(), cap);
    }
}
