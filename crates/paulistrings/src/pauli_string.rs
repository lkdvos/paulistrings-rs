//! [`PauliString<W>`] — symplectic-encoded Pauli operator on up to `64·W` qubits. See ARCHITECTURE.md §Data-Model and §Width.
//!
//! Multiplication XORs the `(x, z)` parts and returns the `i^k` phase factor as a [`Phase`]; the phase is not stored on the type, callers fold it into a coefficient at the boundary.
//!
//! # Examples
//!
//! `X · Z = -i·Y`: the XOR gives the Y bits and the returned [`Phase`] is the
//! `-i` factor.
//!
//! ```
//! use paulistrings::{PauliString, Phase};
//!
//! let mut p = PauliString::<1>::x(0);
//! let phase = p.mul_assign(&PauliString::<1>::z(0));
//! assert_eq!(p, PauliString::<1>::y(0));
//! assert_eq!(phase, Phase::MINUS_I);
//! ```
//!
//! [`PauliSum`]: crate::PauliSum
//! [`BuildAccumulator`]: crate::BuildAccumulator
//! [`Channel::apply`]: crate::Channel::apply
//! [`Phase`]: crate::Phase

use bytemuck::{Pod, Zeroable};
use num_complex::Complex64;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use crate::phase::Phase;

/// A Pauli operator on up to `64 · W` qubits. See ARCHITECTURE.md §Data-Model.
///
/// `#[repr(C)]` so the type is `Pod` and can be reinterpreted as bytes for serialization or GPU upload. [`Ord`] is the load-bearing trait — the engine is sort-based, not hashmap-based.
///
/// # Examples
///
/// ```
/// use paulistrings::PauliString;
///
/// let p = PauliString::<1>::x(3);
/// assert_eq!(p.weight(), 1);
/// assert!(p.commutes_with(&PauliString::<1>::x(0)));
/// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use paulistrings::PauliString;
    ///
    /// let id = PauliString::<1>::identity();
    /// assert_eq!(id.weight(), 0);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use paulistrings::PauliString;
    ///
    /// let p = PauliString::<1>::x(2);
    /// assert_eq!(p.x, [0b100]);
    /// assert_eq!(p.z, [0]);
    /// ```
    #[inline]
    pub fn x(qubit: u32) -> Self {
        debug_assert!((qubit as usize) < 64 * W);
        let mut p = Self::identity();
        p.x[(qubit / 64) as usize] = 1u64 << (qubit % 64);
        p
    }

    /// Canonical Pauli `Y = (1, 1)` on `qubit`. The `i` factor in `Y = i · X · Z` is the caller's concern.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `qubit >= 64 · W`.
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

    /// Number of non-identity qubits (Hamming weight of `x | z`).
    ///
    /// # Examples
    ///
    /// ```
    /// use paulistrings::PauliString;
    ///
    /// let mut p = PauliString::<1>::x(0);
    /// p.mul_assign(&PauliString::<1>::z(1));
    /// assert_eq!(p.weight(), 2);
    /// ```
    #[inline]
    pub fn weight(&self) -> u32 {
        (0..W).map(|i| (self.x[i] | self.z[i]).count_ones()).sum()
    }

    /// Multiply `self * other` in place. Returns the `i^k` phase factor such that the true product is `phase · self_after_xor`.
    ///
    /// # Examples
    ///
    /// `X · Z = -i·Y`: the bits XOR to `Y` and the phase is `-i`.
    ///
    /// ```
    /// use paulistrings::{PauliString, Phase};
    ///
    /// let mut p = PauliString::<1>::x(0);
    /// let phase = p.mul_assign(&PauliString::<1>::z(0));
    /// assert_eq!(p, PauliString::<1>::y(0));
    /// assert_eq!(phase, Phase::MINUS_I);
    /// ```
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
    ///
    /// Not `std::ops::Mul` because the operation also returns a `Phase`, which calling sites need to fold into a coefficient at the boundary.
    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub fn mul(mut self, other: &Self) -> (Self, Phase) {
        let phase = self.mul_assign(other);
        (self, phase)
    }

    /// `true` iff every set bit lies on a qubit index `< num_qubits`.
    ///
    /// The engine and built-in channels preserve this bound by construction, so this check is not on the hot path; used in `debug_assert!` at boundaries with custom [`Channel`] impls, in `PauliSum::assert_invariants`, and in tests.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `num_qubits > 64 · W`.
    ///
    /// [`Channel`]: crate::Channel
    /// [`Channel::support`]: crate::Channel::support
    /// [`Circuit`]: crate::Circuit
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

    /// `true` iff `self` and `other` commute as Pauli operators.
    ///
    /// # Examples
    ///
    /// ```
    /// use paulistrings::PauliString;
    ///
    /// // X and Z anticommute.
    /// assert!(!PauliString::<1>::x(0).commutes_with(&PauliString::<1>::z(0)));
    /// // X on disjoint qubits commutes with anything on the other qubit.
    /// assert!(PauliString::<1>::x(0).commutes_with(&PauliString::<1>::z(1)));
    /// ```
    ///
    /// Only the low bit of the symplectic form survives, and popcount parity is GF(2)-linear (`parity(a) ^ parity(b) == parity(a ^ b)`), so the masked words are XOR-folded first and reduced by a single `count_ones` instead of one per word.
    #[inline]
    pub fn commutes_with(&self, other: &Self) -> bool {
        let mut acc: u64 = 0;
        for i in 0..W {
            acc ^= (self.x[i] & other.z[i]) ^ (self.z[i] & other.x[i]);
        }
        acc.count_ones() & 1 == 0
    }

    /// Commutator `[self, other] = self·other − other·self`, as `(product, coefficient)`.
    ///
    /// Two Pauli strings either commute or anticommute, so the difference is either exactly `0` or twice the product: the coefficient is `2·i^k` when they anticommute and `0` when they commute. The returned string is [`Self::mul`]'s product either way, so a `0` coefficient is an exact algebraic zero rather than a missing term.
    ///
    /// # Examples
    ///
    /// ```
    /// use paulistrings::PauliString;
    /// use num_complex::Complex64;
    ///
    /// // [X, Z] = 2·X·Z = -2i·Y.
    /// let (product, coeff) = PauliString::<1>::x(0).commutator(&PauliString::<1>::z(0));
    /// assert_eq!(product, PauliString::<1>::y(0));
    /// assert_eq!(coeff, Complex64::new(0.0, -2.0));
    /// ```
    #[inline]
    pub fn commutator(self, other: &Self) -> (Self, Complex64) {
        let vanishes = self.commutes_with(other);
        let (product, phase) = self.mul(other);
        (product, scaled_phase(phase, vanishes))
    }

    /// Anticommutator `{self, other} = self·other + other·self`, as `(product, coefficient)`.
    ///
    /// [`Self::commutator`]'s mirror: the coefficient is `2·i^k` when the two strings commute and `0` when they anticommute, so the two together decompose `2·self·other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use paulistrings::PauliString;
    /// use num_complex::Complex64;
    ///
    /// // {X, X} = 2·I.
    /// let (product, coeff) = PauliString::<1>::x(0).anticommutator(&PauliString::<1>::x(0));
    /// assert_eq!(product, PauliString::<1>::identity());
    /// assert_eq!(coeff, Complex64::new(2.0, 0.0));
    /// ```
    #[inline]
    pub fn anticommutator(self, other: &Self) -> (Self, Complex64) {
        let vanishes = !self.commutes_with(other);
        let (product, phase) = self.mul(other);
        (product, scaled_phase(phase, vanishes))
    }
}

/// `0` when the bracket vanishes, `2·i^k` otherwise — the coefficient both brackets carry.
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
    /// Lexicographic compare on `(x, z)`, low-to-high word order.
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
    /// Auxiliary; only used by `BuildAccumulator` for ingestion. Not on the hot path.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.z.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    const ZERO: Complex64 = Complex64::new(0.0, 0.0);

    /// `X` and `Z` anticommute, so `[X, Z] = 2·X·Z = -2i·Y` and `{X, Z} = 0`.
    #[test]
    fn commutator_of_x_and_z_is_minus_two_i_y() {
        fn check<const W: usize>() {
            let x = PauliString::<W>::x(0);
            let z = PauliString::<W>::z(0);

            let (product, coeff) = x.commutator(&z);
            assert_eq!(product, PauliString::<W>::y(0));
            assert_eq!(coeff, Complex64::new(0.0, -2.0));

            let (product, coeff) = x.anticommutator(&z);
            assert_eq!(product, PauliString::<W>::y(0));
            assert_eq!(coeff, ZERO);
        }
        check::<1>();
        check::<2>();
    }

    /// Reversing the operands flips the commutator's sign: `[Z, X] = 2·Z·X = +2i·Y`.
    #[test]
    fn commutator_is_antisymmetric_on_x_and_z() {
        let (product, coeff) = PauliString::<1>::z(0).commutator(&PauliString::<1>::x(0));
        assert_eq!(product, PauliString::<1>::y(0));
        assert_eq!(coeff, Complex64::new(0.0, 2.0));
    }

    /// `X⊗I` and `I⊗X` commute, so `[P, Q] = 0` and `{P, Q} = 2·P·Q = 2·X⊗X`.
    #[test]
    fn commuting_strings_have_a_zero_commutator_and_twice_the_product() {
        fn check<const W: usize>() {
            let a = PauliString::<W>::x(0);
            let b = PauliString::<W>::x(1);
            let mut xx = PauliString::<W>::x(0);
            xx.x[0] |= 1u64 << 1;

            let (product, coeff) = a.commutator(&b);
            assert_eq!(product, xx);
            assert_eq!(coeff, ZERO);

            let (product, coeff) = a.anticommutator(&b);
            assert_eq!(product, xx);
            assert_eq!(coeff, Complex64::new(2.0, 0.0));
        }
        check::<1>();
        check::<2>();
    }

    /// Identity commutes with everything, so its anticommutator carries the whole `2·P`.
    #[test]
    fn identity_anticommutator_is_twice_the_other_string() {
        let id = PauliString::<2>::identity();
        let y = PauliString::<2>::y(64);

        let (product, coeff) = id.commutator(&y);
        assert_eq!(product, y);
        assert_eq!(coeff, ZERO);

        let (product, coeff) = id.anticommutator(&y);
        assert_eq!(product, y);
        assert_eq!(coeff, Complex64::new(2.0, 0.0));
    }

    /// The same `X`/`Z` case on qubit 64, i.e. entirely in the second word.
    #[test]
    fn commutator_multi_word() {
        let (product, coeff) = PauliString::<2>::x(64).commutator(&PauliString::<2>::z(64));
        assert_eq!(product, PauliString::<2>::y(64));
        assert_eq!(coeff, Complex64::new(0.0, -2.0));
    }

    mod props {
        use super::*;
        use proptest::prelude::*;

        fn arb_pauli_w2() -> impl Strategy<Value = PauliString<2>> {
            (any::<u64>(), any::<u64>(), any::<u64>(), any::<u64>()).prop_map(|(x0, x1, z0, z1)| {
                PauliString::<2> {
                    x: [x0, x1],
                    z: [z0, z1],
                }
            })
        }

        proptest! {
            /// `[P, Q] + {P, Q} = 2·P·Q` — the defining decomposition, and the reason exactly one of the two is nonzero.
            #[test]
            fn commutator_plus_anticommutator_is_twice_the_product(
                a in arb_pauli_w2(),
                b in arb_pauli_w2(),
            ) {
                let (product, phase) = a.mul(&b);
                let (comm_product, comm) = a.commutator(&b);
                let (anti_product, anti) = a.anticommutator(&b);

                prop_assert_eq!(comm_product, product);
                prop_assert_eq!(anti_product, product);
                prop_assert_eq!(comm + anti, phase.to_complex() * 2.0);
                prop_assert!((comm == ZERO) != (anti == ZERO));
            }
        }
    }
}
