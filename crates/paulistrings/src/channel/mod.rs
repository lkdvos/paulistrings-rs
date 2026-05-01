//! `Channel<W>` — unified abstraction for gates and noise. See §6.
//!
//! Every operation on a `PauliSum` (Clifford gate, Pauli rotation, arbitrary
//! unitary, noise channel) maps a single Pauli string to a small weighted sum
//! of Pauli strings. The trait formalizes that mapping; the engine consumes
//! it via the sort-merge pipeline (§5).

#![allow(unused)]

pub mod clifford;
pub mod noise;
pub mod rotation;
pub mod unitary;

pub use clifford::{Clifford1Q, Clifford2Q};
pub use noise::{AmplitudeDamping, Dephasing, Depolarizing};
pub use rotation::PauliRotation;
pub use unitary::{GeneralUnitary1Q, GeneralUnitary2Q};

use num_complex::Complex64;

/// Pre-allocated, fixed-capacity SoA scratch buffer for channel outputs.
///
/// Sized by the engine to `n_in * Channel::MAX_FANOUT` so that `apply` can
/// write without dynamic growth. Required for GPU correctness and for CPU
/// hot-loop performance.
pub struct OutputBuffer<'a, const W: usize> {
    pub x: &'a mut [[u64; W]],
    pub z: &'a mut [[u64; W]],
    pub coeff: &'a mut [Complex64],
    /// Cursor into the slices; `apply` writes at `len` and advances.
    pub len: &'a mut usize,
}

impl<'a, const W: usize> OutputBuffer<'a, W> {
    /// Append one term to the buffer.
    #[inline]
    pub fn push(&mut self, _x: [u64; W], _z: [u64; W], _c: Complex64) {
        todo!("§6: bounds-check against MAX_FANOUT and write at *self.len; bump cursor")
    }
}

/// Anything that maps a Pauli string to a small weighted sum of Pauli strings.
///
/// `max_fanout` is a method (not an associated `const`) so the trait stays
/// `dyn`-compatible — `Circuit` stores `Box<dyn Channel<W>>` to keep the
/// channel set open for user extensions (§6). For built-in channels the
/// returned value is a compile-time constant so the engine still gets
/// constant-folded buffer sizing once the concrete type is in hand.
pub trait Channel<const W: usize>: Send + Sync {
    /// Maximum number of output terms produced per input term. Used by the
    /// engine to size the scratch buffer up-front.
    fn max_fanout(&self) -> usize;

    /// Qubits this channel acts on. Outputs differ from inputs only at these
    /// bit positions; the engine uses this for bucket layout (§5).
    fn support(&self) -> &[u32];

    /// Apply the channel to a single input term, writing outputs to `out`.
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    );
}
