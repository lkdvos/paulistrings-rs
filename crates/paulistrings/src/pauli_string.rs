//! `PauliString<W>` — symplectic-encoded Pauli operator on up to `64*W` qubits.
//!
//! See design doc §3.1.
//!
//! Encoding: `I=(0,0)`, `X=(1,0)`, `Z=(0,1)`, `Y=(1,1)`. Multiplication XORs
//! the `(x, z)` parts and returns the `i^k` phase factor — for `k` in `0..4`
//! — that arises where X- and Z-bits coincide. The phase is not stored on
//! the type; callers fold it into a `Complex64` coefficient at the boundary.

#![allow(unused)]

use bytemuck::{Pod, Zeroable};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use crate::phase::Phase;

/// A Pauli operator on up to `64 * W` qubits.
///
/// Layout is `#[repr(C)]` so the type is `Pod` and can be reinterpreted as
/// bytes for serialization or upload to a GPU device. There is no stored
/// phase: multiplication returns the `i^k` phase as a separate `u8` and
/// callers fold it into their coefficient at the boundary.
///
/// `Ord` is the load-bearing trait (the engine is sort-based, not
/// hashmap-based); see §3.1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct PauliString<const W: usize> {
    pub x: [u64; W],
    pub z: [u64; W],
}

unsafe impl<const W: usize> Zeroable for PauliString<W> {}
unsafe impl<const W: usize> Pod for PauliString<W> {}

impl<const W: usize> PauliString<W> {
    /// Identity Pauli string (all qubits `I`).
    pub const fn identity() -> Self {
        Self {
            x: [0u64; W],
            z: [0u64; W],
        }
    }

    /// Single-qubit `X` Pauli on `qubit`.
    #[inline]
    pub fn x(qubit: u32) -> Self {
        debug_assert!((qubit as usize) < 64 * W);
        let mut p = Self::identity();
        p.x[(qubit / 64) as usize] = 1u64 << (qubit % 64);
        p
    }

    /// Canonical Pauli `Y = (1, 1)` on `qubit`. The `i` factor in `Y = iXZ`
    /// is the caller's concern.
    #[inline]
    pub fn y(qubit: u32) -> Self {
        debug_assert!((qubit as usize) < 64 * W);
        let mut p = Self::identity();
        let w = (qubit / 64) as usize;
        let bit = 1u64 << (qubit % 64);
        p.x[w] = bit;
        p.z[w] = bit;
        p
    }

    /// Single-qubit `Z` Pauli on `qubit`.
    #[inline]
    pub fn z(qubit: u32) -> Self {
        debug_assert!((qubit as usize) < 64 * W);
        let mut p = Self::identity();
        p.z[(qubit / 64) as usize] = 1u64 << (qubit % 64);
        p
    }

    /// Number of non-identity qubits (Hamming weight of `x | z`).
    #[inline]
    pub fn weight(&self) -> u32 {
        (0..W).map(|i| (self.x[i] | self.z[i]).count_ones()).sum()
    }

    /// Multiply `self * other` in place. Returns the `i^k` phase factor such
    /// that the true product is `phase * self_after_xor`.
    #[inline]
    pub fn mul_assign(&mut self, other: &Self) -> Phase {
        // Per-qubit: P(a,b) · P(c,d) = i^δ · P(a⊕c, b⊕d) where
        //   δ = 2·(b·c) + a·b + c·d − (a⊕c)·(b⊕d)   (mod 4)
        // (derived from P(a,b) = i^{a·b} X^a Z^b and ZX = -XZ).
        let mut delta: u32 = 0;
        for i in 0..W {
            let a = self.x[i];
            let b = self.z[i];
            let c = other.x[i];
            let d = other.z[i];
            let zc_x = (b & c).count_ones();
            let y_self = (a & b).count_ones();
            let y_other = (c & d).count_ones();
            let y_result = ((a ^ c) & (b ^ d)).count_ones();
            delta = delta.wrapping_add(zc_x.wrapping_mul(2));
            delta = delta.wrapping_add(y_self);
            delta = delta.wrapping_add(y_other);
            delta = delta.wrapping_sub(y_result);
            self.x[i] ^= c;
            self.z[i] ^= d;
        }
        Phase::new(delta as u8)
    }

    /// Value-returning multiply: `(self * other, phase)`.
    #[inline]
    pub fn mul(mut self, other: &Self) -> (Self, Phase) {
        let phase = self.mul_assign(other);
        (self, phase)
    }

    /// `true` iff every set bit lies on a qubit index `< num_qubits`.
    ///
    /// The engine and built-in channels preserve this bound by construction
    /// (a channel only flips bits inside `support()`, which is bounded at
    /// `Circuit` build time), so this check is not on the hot path. Use it
    /// in `debug_assert!` at boundaries with custom `Channel` impls, in
    /// `PauliSum::assert_invariants`, and in tests that exercise the
    /// invariant directly.
    ///
    /// Panics in debug builds if `num_qubits > 64 * W`.
    #[inline]
    pub fn is_within(&self, num_qubits: usize) -> bool {
        debug_assert!(num_qubits <= 64 * W);
        let mut leak: u64 = 0;
        for i in 0..W {
            let lo = 64 * i;
            let in_bounds: u64 = if num_qubits >= lo + 64 {
                !0u64
            } else if num_qubits <= lo {
                0
            } else {
                (1u64 << (num_qubits - lo)) - 1
            };
            leak |= (self.x[i] | self.z[i]) & !in_bounds;
        }
        leak == 0
    }

    /// `true` iff `self` and `other` commute.
    #[inline]
    pub fn commutes_with(&self, other: &Self) -> bool {
        let mut parity: u32 = 0;
        for i in 0..W {
            parity ^= (self.x[i] & other.z[i]).count_ones();
            parity ^= (self.z[i] & other.x[i]).count_ones();
        }
        parity & 1 == 0
    }
}

impl<const W: usize> Default for PauliString<W> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<const W: usize> Ord for PauliString<W> {
    /// Lexicographic compare on the concatenation `(x, z)` interpreted as an
    /// unsigned-integer array, low-to-high word order.
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
