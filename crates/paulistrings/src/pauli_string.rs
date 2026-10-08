//! [`PauliString<W>`], the symplectic `(x, z)` encoding of a Pauli operator on up to `64·W` qubits (ARCHITECTURE.md §Data-Model, §Width).

use bytemuck::{Pod, Zeroable};
use num_complex::Complex64;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use crate::phase::Phase;

/// A Pauli operator on up to `64·W` qubits, `W` being the number of 64-bit words per `x`/`z` part.
///
/// Pick the smallest `W` that fits; the Python bindings monomorphize `W ∈ {1, 2, 4, 8, 16}`.
/// The phase of a product is not stored: [`Self::mul_assign`] returns it as a [`Phase`] for the caller to fold into a coefficient.
/// `Y` is the key `(x=1, z=1)` with no phase factor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct PauliString<const W: usize> {
    /// X-part bitmask: bit `q` set iff qubit `q` is `X` or `Y`.
    pub x: [u64; W],
    /// Z-part bitmask: bit `q` set iff qubit `q` is `Z` or `Y`.
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
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `qubit >= 64 · W`.
    #[inline]
    pub fn x(qubit: u32) -> Self {
        debug_assert!((qubit as usize) < 64 * W);
        let mut p = Self::identity();
        p.x[(qubit / 64) as usize] = 1u64 << (qubit % 64);
        p
    }

    /// Single-qubit `Y = (x=1, z=1)` on `qubit`; the `i` in `Y = i·X·Z` is the caller's concern.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `qubit >= 64 · W`.
    #[inline]
    pub fn y(qubit: u32) -> Self {
        debug_assert!((qubit as usize) < 64 * W);
        let mut p = Self::identity();
        let word = (qubit / 64) as usize;
        let bit = 1u64 << (qubit % 64);
        p.x[word] = bit;
        p.z[word] = bit;
        p
    }

    /// Single-qubit `Z` Pauli on `qubit`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `qubit >= 64 · W`.
    #[inline]
    pub fn z(qubit: u32) -> Self {
        debug_assert!((qubit as usize) < 64 * W);
        let mut p = Self::identity();
        p.z[(qubit / 64) as usize] = 1u64 << (qubit % 64);
        p
    }

    /// Number of non-identity qubits.
    #[inline]
    pub fn weight(&self) -> u32 {
        (0..W).map(|i| (self.x[i] | self.z[i]).count_ones()).sum()
    }

    /// Multiply `self * other` in place, returning the phase `i^k` such that the true product is `phase · self`.
    #[inline]
    pub fn mul_assign(&mut self, other: &Self) -> Phase {
        // Per qubit, P(a,b)·P(c,d) = i^δ·P(a⊕c, b⊕d) with δ = 2bc + ab + cd − (a⊕c)(b⊕d) mod 4, from P(a,b) = i^{ab}·X^a·Z^b.
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
    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub fn mul(mut self, other: &Self) -> (Self, Phase) {
        let phase = self.mul_assign(other);
        (self, phase)
    }

    /// `true` iff every set bit lies on a qubit index `< num_qubits`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `num_qubits > 64 · W`.
    #[inline]
    pub fn is_within(&self, num_qubits: usize) -> bool {
        debug_assert!(num_qubits <= 64 * W);
        let mut leak: u64 = 0;
        for i in 0..W {
            leak |= (self.x[i] | self.z[i]) & !word_mask(num_qubits, i);
        }
        leak == 0
    }

    /// `true` iff `self` and `other` commute as Pauli operators.
    #[inline]
    pub fn commutes_with(&self, other: &Self) -> bool {
        // Parity is GF(2)-linear, so XOR-fold the words and take one popcount.
        let mut folded: u64 = 0;
        for i in 0..W {
            folded ^= (self.x[i] & other.z[i]) ^ (self.z[i] & other.x[i]);
        }
        folded.count_ones() & 1 == 0
    }

    /// Commutator `[self, other] = self·other − other·self`, as `(product, coefficient)`.
    ///
    /// The coefficient is `2·i^k` when the strings anticommute and exactly `0` when they commute; the returned string is [`Self::mul`]'s product either way.
    #[inline]
    pub fn commutator(self, other: &Self) -> (Self, Complex64) {
        let vanishes = self.commutes_with(other);
        let (product, phase) = self.mul(other);
        (product, scaled_phase(phase, vanishes))
    }

    /// Anticommutator `{self, other} = self·other + other·self`, as `(product, coefficient)`.
    ///
    /// The coefficient is `2·i^k` when the strings commute and exactly `0` when they anticommute.
    #[inline]
    pub fn anticommutator(self, other: &Self) -> (Self, Complex64) {
        let vanishes = !self.commutes_with(other);
        let (product, phase) = self.mul(other);
        (product, scaled_phase(phase, vanishes))
    }
}

/// Mask of the live qubit bits in word `word`, given `num_qubits` total.
#[inline]
pub(crate) fn word_mask(num_qubits: usize, word: usize) -> u64 {
    let first_qubit = 64 * word;
    if num_qubits >= first_qubit + 64 {
        !0u64
    } else if num_qubits <= first_qubit {
        0
    } else {
        (1u64 << (num_qubits - first_qubit)) - 1
    }
}

/// `0` when the bracket vanishes, `2·i^k` otherwise.
#[inline]
fn scaled_phase(phase: Phase, vanishes: bool) -> Complex64 {
    if vanishes {
        Complex64::new(0.0, 0.0)
    } else {
        phase.to_complex() * 2.0
    }
}

impl<const W: usize> Default for PauliString<W> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<const W: usize> Ord for PauliString<W> {
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
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.z.hash(state);
    }
}

#[cfg(test)]
mod tests;
