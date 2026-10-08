//! The [`Channel`] trait, its output buffer, and the built-in gates and noise channels.
//! See ARCHITECTURE.md §Channels.

pub(crate) mod clifford;
pub(crate) mod identity;
pub(crate) mod noise;
pub(crate) mod prepared;
pub(crate) mod rotation;
pub(crate) mod unitary;

pub use clifford::{Clifford1Q, Clifford2Q};
pub use identity::IdentityChannel;
pub use noise::{AmplitudeDamping, Dephasing, Depolarizing, Depolarizing2Q, PauliChannel};
pub use rotation::PauliRotation;
pub use unitary::{GeneralUnitary1Q, GeneralUnitary2Q};

use crate::pauli_sum::hash::Gf2Hash;
use num_complex::Complex64;
use prepared::Prepared;

/// Fixed-capacity structure-of-arrays buffer that [`Channel::apply`] writes its outputs into.
///
/// Capacity is the length of the columns, sized by the caller to at least [`Channel::max_fanout`] per input term.
pub struct OutputBuffer<'a, const W: usize> {
    /// X-part column.
    pub x: &'a mut [[u64; W]],
    /// Z-part column.
    pub z: &'a mut [[u64; W]],
    /// Coefficient column.
    pub coeff: &'a mut [Complex64],
    /// Number of terms written so far.
    pub len: &'a mut usize,
}

impl<'a, const W: usize> OutputBuffer<'a, W> {
    /// Append one term.
    ///
    /// Panics if the buffer is full; an `apply` body must not push more than its declared `max_fanout`.
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

    /// Reset the length to zero, keeping the storage.
    #[inline]
    pub fn clear(&mut self) {
        *self.len = 0;
    }
}

/// Pack qubit indices into a [`Channel::support`] bitmask; order and duplicates do not matter.
#[inline]
pub fn support_mask<const W: usize>(qubits: &[u32]) -> [u64; W] {
    let mut mask = [0u64; W];
    for &q in qubits {
        debug_assert!((q as usize) < 64 * W, "qubit {q} out of range for W={W}");
        mask[q as usize / 64] |= 1u64 << (q % 64);
    }
    mask
}

/// `(word, bit, 1 << bit)` of qubit `q`.
#[inline(always)]
fn qubit_loc(q: usize) -> (usize, usize, u64) {
    let word = q / 64;
    let bit = q % 64;
    (word, bit, 1u64 << bit)
}

/// The packed single-qubit Pauli index `x | (z << 1)` (`I=0, X=1, Z=2, Y=3`) at `(word, bit)`.
#[inline(always)]
fn read_pauli<const W: usize>(x: &[u64; W], z: &[u64; W], word: usize, bit: usize) -> usize {
    let x_bit = (x[word] >> bit) & 1;
    let z_bit = (z[word] >> bit) & 1;
    (x_bit | (z_bit << 1)) as usize
}

/// Overwrite the qubit at `(word, bit, mask)` with the packed Pauli index `p`.
#[inline(always)]
fn write_pauli<const W: usize>(
    x: &mut [u64; W],
    z: &mut [u64; W],
    word: usize,
    bit: usize,
    mask: u64,
    p: usize,
) {
    let x_bit = (p & 1) as u64;
    let z_bit = ((p >> 1) & 1) as u64;
    x[word] = (x[word] & !mask) | (x_bit << bit);
    z[word] = (z[word] & !mask) | (z_bit << bit);
}

/// Set or clear the bit at `(word, mask)` of one bit-plane.
#[inline(always)]
fn set_bit<const W: usize>(plane: &mut [u64; W], word: usize, mask: u64, value: bool) {
    if value {
        plane[word] |= mask;
    } else {
        plane[word] &= !mask;
    }
}

/// A gate or noise channel: maps a Pauli string to a small weighted sum of Pauli strings.
///
/// Built-ins: [`Clifford1Q`], [`Clifford2Q`], [`PauliRotation`], [`GeneralUnitary1Q`], [`GeneralUnitary2Q`], [`Depolarizing`], [`Dephasing`], [`PauliChannel`], [`Depolarizing2Q`], [`AmplitudeDamping`], [`IdentityChannel`].
/// A custom channel implements `max_fanout`, `support` and `apply`; the engine derives everything else (ARCHITECTURE.md §Prepared-Channels).
///
/// ```
/// use paulistrings::{Channel, OutputBuffer};
/// use num_complex::Complex64;
///
/// /// Multiplies every coefficient by a complex factor.
/// struct GlobalPhase {
///     factor: Complex64,
/// }
///
/// impl<const W: usize> Channel<W> for GlobalPhase {
///     fn max_fanout(&self) -> usize { 1 }
///     fn support(&self) -> [u64; W] { [0; W] }
///     fn apply(
///         &self,
///         input_x: &[u64; W],
///         input_z: &[u64; W],
///         coeff: Complex64,
///         out: &mut OutputBuffer<'_, W>,
///     ) {
///         out.push(*input_x, *input_z, coeff * self.factor);
///     }
/// }
///
/// let channel = GlobalPhase {
///     factor: Complex64::new(0.0, 1.0),
/// };
/// let _: Box<dyn Channel<1>> = Box::new(channel);
/// ```
pub trait Channel<const W: usize>: Send + Sync {
    /// Upper bound on the output terms per input term.
    fn max_fanout(&self) -> usize;

    /// Qubits this channel acts on, as a bitmask built with [`support_mask`].
    ///
    /// Outputs must differ from their input only at these qubits.
    fn support(&self) -> [u64; W];

    /// Short name for the per-layer progress log.
    ///
    /// The default is the type name without path or generics, which is unhelpful for tuples, closures and boxes; override it there.
    fn debug_name(&self) -> &'static str {
        let full = core::any::type_name::<Self>();
        let head = match full.find('<') {
            Some(i) => &full[..i],
            None => full,
        };
        match head.rfind("::") {
            Some(i) => &head[i + 2..],
            None => head,
        }
    }

    /// Apply the channel to one input term, pushing its outputs to `out`.
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    );

    /// Apply the channel's adjoint to one input term; run by `Direction::Heisenberg`.
    ///
    /// The default calls [`apply`](Self::apply), which is correct only for a self-adjoint channel.
    fn apply_adjoint(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        self.apply(input_x, input_z, coeff, out);
    }

    /// Prepare this channel for one engine layer (ARCHITECTURE.md §Prepared-Channels).
    ///
    /// The default probes `apply` on the local basis Paulis and supports at most two qubits; `None` makes `propagate` panic.
    /// It is exact only if the output amplitudes depend on the input solely through its bits at [`support`](Self::support), which debug builds check.
    fn prepare(&self, hash: &Gf2Hash<W>, adjoint: bool) -> Option<Prepared<W>> {
        Prepared::derive_local(self, hash, adjoint)
    }
}

#[cfg(test)]
mod tests;
