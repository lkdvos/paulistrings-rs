//! [`Phase`], the `i^k` factor a Pauli product returns.

use num_complex::Complex64;
use std::ops::{Add, AddAssign};

/// A phase factor `i^k` with `k ∈ {0, 1, 2, 3}`; `+` multiplies phases.
///
/// [`crate::PauliString::mul_assign`] returns one, and [`Phase::apply`] folds it into a coefficient.
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

    /// Construct from a raw exponent, reduced mod 4.
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

    /// Multiply `c` by `i^k` by a sign/swap of its parts, without a float multiply.
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
