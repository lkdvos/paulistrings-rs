//! [`Channel<W>`] — unified abstraction for gates and noise.
//!
//! Built-ins: [`Clifford1Q`], [`Clifford2Q`], [`PauliRotation`], [`GeneralUnitary1Q`], [`GeneralUnitary2Q`], [`Depolarizing`], [`Dephasing`], [`PauliChannel`], [`Depolarizing2Q`], [`AmplitudeDamping`], [`IdentityChannel`]. See ARCHITECTURE.md §Channels.
//!
//! # Implementing a custom channel
//!
//! Implement the trait directly; the engine treats your type as just another `Box<dyn `[`Channel<W>`]`>` inside a [`Circuit`].
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

pub mod clifford;
pub mod identity;
pub mod noise;
pub mod prepared;
pub mod rotation;
pub mod unitary;

pub use clifford::{Clifford1Q, Clifford2Q};
pub use identity::IdentityChannel;
pub use noise::{AmplitudeDamping, Dephasing, Depolarizing, Depolarizing2Q, PauliChannel};
pub use rotation::PauliRotation;
pub use unitary::{GeneralUnitary1Q, GeneralUnitary2Q};

use crate::bucket::hash::Gf2Hash;
use num_complex::Complex64;
use prepared::Prepared;

/// Pre-allocated, fixed-capacity SoA scratch buffer for channel outputs.
///
/// Sized by the engine to `n_in · channel.max_fanout()` so `apply` writes without dynamic growth. Channel impls write via [`OutputBuffer::push`].
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
    /// Capacity is `self.x.len()`; a `Channel::apply` body must not push more than its declared `max_fanout`. Out-of-range writes are caught by slice bounds-checking (and, in debug builds, an explicit assertion).
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

    /// Reset the cursor to zero so the same backing storage can be reused for the next input term without reallocation.
    #[inline]
    pub fn clear(&mut self) {
        *self.len = 0;
    }
}

/// Pack a list of qubit indices into a [`Channel::support`] bitmask.
///
/// Bit `q % 64` of word `q / 64` is set for each `q` in `qubits`; order and duplicates do not matter.
#[inline]
pub fn support_mask<const W: usize>(qubits: &[u32]) -> [u64; W] {
    let mut mask = [0u64; W];
    for &q in qubits {
        debug_assert!((q as usize) < 64 * W, "qubit {q} out of range for W={W}");
        mask[q as usize / 64] |= 1u64 << (q % 64);
    }
    mask
}

// ---- Shared per-qubit bit extract/insert idiom ----
//
// Every built-in `Channel::apply`/`apply_adjoint` reads and/or overwrites the two bit-planes (`x`, `z`) at one or two support qubits; these four helpers are the common core, private to this module and its descendants.

/// Decompose qubit index `q` into `(word, bit, mask)`: the index into a `[u64; W]` array, the bit position within that word, and `1u64 << bit`.
#[inline(always)]
fn qubit_loc(q: usize) -> (usize, usize, u64) {
    let word = q / 64;
    let bit = q % 64;
    (word, bit, 1u64 << bit)
}

/// Read the packed single-qubit Pauli index (`x | (z << 1)`, i.e. `I=0, X=1, Z=2, Y=3` — the convention [`Clifford1Q`] and [`GeneralUnitary1Q`] both document) of the qubit at `(word, bit)`.
#[inline(always)]
fn read_pauli<const W: usize>(x: &[u64; W], z: &[u64; W], word: usize, bit: usize) -> usize {
    let x_bit = (x[word] >> bit) & 1;
    let z_bit = (z[word] >> bit) & 1;
    (x_bit | (z_bit << 1)) as usize
}

/// Overwrite the qubit at `(word, bit, mask)` of `(nx, nz)` with the packed Pauli index `p` (same `x | (z << 1)` encoding as [`read_pauli`]).
#[inline(always)]
fn write_pauli<const W: usize>(
    nx: &mut [u64; W],
    nz: &mut [u64; W],
    word: usize,
    bit: usize,
    mask: u64,
    p: usize,
) {
    let ox = (p & 1) as u64;
    let oz = ((p >> 1) & 1) as u64;
    nx[word] = (nx[word] & !mask) | (ox << bit);
    nz[word] = (nz[word] & !mask) | (oz << bit);
}

/// Set (`value = true`) or clear (`value = false`) the single bit at `(word, mask)` of one bit-plane array. Used where only one of `x`/`z` changes, e.g. amplitude damping's `I ↔ Z` fan-out.
#[inline(always)]
fn set_bit<const W: usize>(arr: &mut [u64; W], word: usize, mask: u64, value: bool) {
    if value {
        arr[word] |= mask;
    } else {
        arr[word] &= !mask;
    }
}

/// Anything that maps a Pauli string to a small weighted sum of Pauli strings.
///
/// [`Channel::max_fanout`] is a method, not an associated `const`, so the trait stays `dyn`-compatible — [`Circuit`](crate::Circuit) stores `Box<dyn Channel<W>>` to keep the channel set open for user extensions.
///
/// See the [module-level docs](self) for an `impl Channel` example.
pub trait Channel<const W: usize>: Send + Sync {
    /// Maximum number of output terms produced per input term. Used by the engine to size the scratch buffer up-front.
    fn max_fanout(&self) -> usize;

    /// Qubits this channel acts on, packed as one combined per-qubit bitmask (bit `q` set iff qubit `q` is in the support), one word per `W`.
    /// Outputs differ from inputs only at these bit positions; the engine uses this for bucket layout (ARCHITECTURE.md §Bucketing). Build one with [`support_mask`].
    fn support(&self) -> [u64; W];

    /// Short human-readable name, used only in the engine's per-layer progress log (see [`propagate`](crate::propagate)). Never parsed, never part of any output.
    ///
    /// The default returns [`core::any::type_name`] of the concrete type, trimmed to its last path segment — `Clifford1Q`, `PauliRotation`, `Depolarizing`.
    /// The trim is textual (cut at the first `<`, keep everything after the last `::`), so a type whose name is not a plain (possibly generic) named path — a tuple, a closure, a `Box<...>` — trims to something unhelpful; override if you need a name you can rely on.
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

    /// Apply the channel to a single input term, writing outputs to `out`.
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    );

    /// Apply the channel's adjoint to a single input term, writing outputs to `out`. Used by the engine in `Direction::Heisenberg` mode for backpropagating observables.
    ///
    /// The default is `self.apply(...)`, i.e. assumes the channel is self-adjoint. Channels that are not (`PauliRotation`, `Clifford1Q::s`) override this.
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
    /// The default derives a dense local Pauli-transfer matrix by probing `apply` on the `4^|support|` local basis Paulis, so a channel that implements `apply` gets the bucketed engine for free.
    /// Override only when the support is wider than [`prepared::MAX_LOCAL_SUPPORT`] and a tighter description exists (only `PauliRotation` above generator weight 2 does among the built-ins).
    /// `None` means "cannot be bucketed": `propagate` panics rather than proceed with an unsound preparation (ARCHITECTURE.md §Prepared-Channels).
    ///
    /// # Contract
    ///
    /// Implementors must honour bounded support: the output amplitude may depend on the input only through its bits at [`Channel::support`] positions.
    /// Deriving assumes it; debug builds check it against an all-ones background, and a property test checks it against randomized full-width inputs.
    fn prepare(&self, hash: &Gf2Hash<W>, adjoint: bool) -> Option<Prepared<W>> {
        Prepared::derive_local(self, hash, adjoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::alloc_bufs;

    // ---- support_mask ----

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

    // ---- debug_name (progress logging) ----

    /// The default `debug_name` must survive erasure to `dyn Channel<W>` — that is how the engine sees every channel — and must trim both the module path and any generic arguments.
    #[test]
    fn debug_name_through_dyn_trims_path_and_generics() {
        use crate::pauli_string::PauliString;

        let channels: Vec<(Box<dyn Channel<1>>, &str)> = vec![
            (Box::new(Clifford1Q::h(0)), "Clifford1Q"),
            (Box::new(Clifford2Q::cnot(0, 1)), "Clifford2Q"),
            (
                // Generic in `W`, so `type_name` carries a `<1>` to trim.
                Box::new(PauliRotation::new(PauliString::<1>::z(0), 0.3)),
                "PauliRotation",
            ),
            (
                Box::new(Depolarizing {
                    support: [0],
                    p: 0.1,
                }),
                "Depolarizing",
            ),
            (
                Box::new(Dephasing {
                    support: [0],
                    p: 0.1,
                }),
                "Dephasing",
            ),
            (
                Box::new(PauliChannel {
                    support: [0],
                    px: 0.1,
                    py: 0.1,
                    pz: 0.1,
                }),
                "PauliChannel",
            ),
            (
                Box::new(Depolarizing2Q {
                    support: [0, 1],
                    p: 0.1,
                }),
                "Depolarizing2Q",
            ),
            (
                Box::new(AmplitudeDamping {
                    support: [0],
                    gamma: 0.1,
                }),
                "AmplitudeDamping",
            ),
            (Box::new(IdentityChannel), "IdentityChannel"),
        ];
        for (ch, expected) in &channels {
            assert_eq!(ch.debug_name(), *expected);
        }
    }

    #[test]
    fn support_mask_of_empty_is_zero() {
        let mask: [u64; 2] = support_mask(&[]);
        assert_eq!(mask, [0, 0]);
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
        let mut len: usize;
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
