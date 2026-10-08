//! [`Phase`] — `i^k` factors arising from Pauli algebra, with `k ∈ {0, 1, 2, 3}`.
//!
//! Multiplication of two Pauli bitstrings produces an `i^k` factor wherever X- and Z-bits coincide (see [`PauliString::mul_assign`]); callers fold this into the relevant `Complex64` coefficient at the boundary ([`PauliSum`], [`BuildAccumulator`], channel `apply`).
//!
//! # Examples
//!
//! Combine phases with `+` (mod 4) and fold one into a `Complex64` coefficient with [`Phase::apply`]:
//!
//! ```
//! use paulistrings::Phase;
//! use num_complex::Complex64;
//!
//! let p = Phase::I + Phase::I; // i · i = -1
//! assert_eq!(p, Phase::MINUS_ONE);
//!
//! let folded = Phase::I.apply(Complex64::new(2.0, 3.0)); // i · (2 + 3i) = -3 + 2i
//! assert_eq!(folded, Complex64::new(-3.0, 2.0));
//! ```
//!
//! [`PauliString::mul_assign`]: crate::PauliString::mul_assign
//! [`PauliSum`]: crate::PauliSum
//! [`BuildAccumulator`]: crate::BuildAccumulator

use num_complex::Complex64;
use std::ops::{Add, AddAssign};

/// A phase factor `i^k` where `k ∈ {0, 1, 2, 3}`.
///
/// Layout is `#[repr(transparent)] u8`, so `Phase` is zero-cost: a plain
/// arithmetic byte at the ABI level. Construction via `Phase::new` reduces
/// mod 4; the `Add` impl preserves that invariant.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(transparent)]
pub struct Phase(u8);

impl Phase {
    /// `i^0 = 1`.
    pub const ONE: Self = Self(0);
    /// `i^1 = i`.
    pub const I: Self = Self(1);
    /// `i^2 = -1`.
    pub const MINUS_ONE: Self = Self(2);
    /// `i^3 = -i`.
    pub const MINUS_I: Self = Self(3);

    /// Construct from a raw exponent. Reduces mod 4, so `Phase::new(5) ==
    /// Phase::I`.
    #[inline]
    pub const fn new(k: u8) -> Self {
        Self(k & 3)
    }

    /// The raw exponent in `0..=3`.
    #[inline]
    pub const fn exponent(self) -> u8 {
        self.0
    }

    /// `i^k` as a `Complex64`.
    #[inline]
    pub fn to_complex(self) -> Complex64 {
        match self.0 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        }
    }

    /// Multiply `c` by `i^k` without going through `to_complex`. Each branch
    /// is a single sign/swap on the `(re, im)` parts — no FP multiply.
    #[inline]
    pub fn apply(self, c: Complex64) -> Complex64 {
        match self.0 {
            0 => c,
            1 => Complex64::new(-c.im, c.re),
            2 => Complex64::new(-c.re, -c.im),
            3 => Complex64::new(c.im, -c.re),
            _ => unreachable!(),
        }
    }
}

impl Add for Phase {
    type Output = Phase;
    #[inline]
    fn add(self, other: Phase) -> Phase {
        // Both operands are already in `0..=3`, so the sum is in `0..=6` and
        // a single mask suffices.
        Phase((self.0 + other.0) & 3)
    }
}

impl AddAssign for Phase {
    #[inline]
    fn add_assign(&mut self, other: Phase) {
        *self = *self + other;
    }
}

#[cfg(test)]
mod tests;
