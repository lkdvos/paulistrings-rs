//! `PauliString<W>` — symplectic-encoded Pauli operator on up to `64*W` qubits.
//!
//! See design doc §3.1.
//!
//! Encoding: `I=(0,0)`, `X=(1,0)`, `Z=(0,1)`, `Y=(1,1)`. Multiplication is
//! bitwise XOR of the `(x, z)` parts with phase bookkeeping for the `i^k`
//! factors that arise where X- and Z-bits coincide.

#![allow(unused)]

use bytemuck::{Pod, Zeroable};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A Pauli operator on up to `64 * W` qubits.
///
/// Layout is `#[repr(C)]` with explicit padding so the type is `Pod` and can
/// be reinterpreted as bytes for serialization or upload to a GPU device.
///
/// `Ord` is the load-bearing trait (the engine is sort-based, not
/// hashmap-based); see §3.1.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct PauliString<const W: usize> {
    pub x: [u64; W],
    pub z: [u64; W],
    /// Phase as `i^phase`, i.e. one of `1, i, -1, -i`. Range: `0..=3`.
    pub phase: u8,
    pub _pad: [u8; 7],
}

unsafe impl<const W: usize> Zeroable for PauliString<W> {}
unsafe impl<const W: usize> Pod for PauliString<W> {}

impl<const W: usize> PauliString<W> {
    /// Identity Pauli string (all qubits `I`, phase `+1`).
    pub const fn identity() -> Self {
        Self {
            x: [0u64; W],
            z: [0u64; W],
            phase: 0,
            _pad: [0u8; 7],
        }
    }

    /// Number of non-identity qubits (Hamming weight of `x | z`).
    #[inline]
    pub fn weight(&self) -> u32 {
        todo!("§3.1: popcount of (x[i] | z[i]) across words")
    }

    /// Multiply `self * other` in place, updating `self.phase` accordingly.
    #[inline]
    pub fn mul_assign(&mut self, _other: &Self) {
        todo!("§3.1: XOR (x, z); accumulate i^k phase from X·Z = iY etc.")
    }

    /// `true` iff `self` and `other` commute.
    #[inline]
    pub fn commutes_with(&self, _other: &Self) -> bool {
        todo!("§3.1: parity of popcount(self.x & other.z) ^ popcount(self.z & other.x)")
    }

    /// Strip the phase, returning a key suitable for storage in a `PauliSum`,
    /// alongside the phase factor that must be folded into the coefficient.
    #[inline]
    pub fn split_phase(self) -> (Self, num_complex::Complex64) {
        todo!("§3.1: return (canonicalized PauliString with phase=0, i^phase as Complex64)")
    }
}

impl<const W: usize> Default for PauliString<W> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<const W: usize> Ord for PauliString<W> {
    /// Lexicographic compare on the concatenation `(x, z)` interpreted as an
    /// unsigned-integer array, low-to-high word order. Phase is **not** part
    /// of the key.
    fn cmp(&self, other: &Self) -> Ordering {
        for i in 0..W {
            match self.x[i].cmp(&other.x[i]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        for i in 0..W {
            match self.z[i].cmp(&other.z[i]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}

impl<const W: usize> PartialOrd for PauliString<W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const W: usize> Hash for PauliString<W> {
    /// Auxiliary; only used by `BuildAccumulator` (§8.2). Not on the hot path.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.z.hash(state);
    }
}
