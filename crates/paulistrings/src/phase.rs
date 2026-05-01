//! `Phase` — `i^k` factors arising from Pauli algebra, with `k ∈ {0, 1, 2, 3}`.
//!
//! Multiplication of two Pauli bitstrings produces an `i^k` factor wherever
//! X- and Z-bits coincide (see `PauliString::mul_assign`); callers fold this
//! into the relevant `Complex64` coefficient at the boundary (`PauliSum`,
//! `BuildAccumulator`, channel `apply`). `Phase` centralizes that arithmetic
//! so every site uses the same combine/apply rules instead of open-coded
//! `u8 & 3` masks and four-arm match statements.

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
mod tests {
    use super::*;

    #[test]
    fn constants_have_expected_exponents() {
        assert_eq!(Phase::ONE.exponent(), 0);
        assert_eq!(Phase::I.exponent(), 1);
        assert_eq!(Phase::MINUS_ONE.exponent(), 2);
        assert_eq!(Phase::MINUS_I.exponent(), 3);
    }

    #[test]
    fn new_reduces_mod_4() {
        assert_eq!(Phase::new(0), Phase::ONE);
        assert_eq!(Phase::new(1), Phase::I);
        assert_eq!(Phase::new(4), Phase::ONE);
        assert_eq!(Phase::new(5), Phase::I);
        assert_eq!(Phase::new(255), Phase::MINUS_I); // 255 & 3 == 3
    }

    #[test]
    fn to_complex_matches_i_powers() {
        assert_eq!(Phase::ONE.to_complex(), Complex64::new(1.0, 0.0));
        assert_eq!(Phase::I.to_complex(), Complex64::new(0.0, 1.0));
        assert_eq!(Phase::MINUS_ONE.to_complex(), Complex64::new(-1.0, 0.0));
        assert_eq!(Phase::MINUS_I.to_complex(), Complex64::new(0.0, -1.0));
    }

    #[test]
    fn apply_agrees_with_to_complex_times_c() {
        let c = Complex64::new(2.0, 3.0);
        for p in [Phase::ONE, Phase::I, Phase::MINUS_ONE, Phase::MINUS_I] {
            assert_eq!(p.apply(c), p.to_complex() * c);
        }
    }

    #[test]
    fn add_wraps_mod_4() {
        assert_eq!(Phase::I + Phase::I, Phase::MINUS_ONE);
        assert_eq!(Phase::MINUS_ONE + Phase::I, Phase::MINUS_I);
        assert_eq!(Phase::MINUS_I + Phase::I, Phase::ONE);
        assert_eq!(Phase::MINUS_I + Phase::MINUS_I, Phase::MINUS_ONE);
    }

    #[test]
    fn add_assign_wraps_mod_4() {
        let mut p = Phase::I;
        p += Phase::I;
        assert_eq!(p, Phase::MINUS_ONE);
        p += Phase::MINUS_I;
        assert_eq!(p, Phase::I);
    }

    #[test]
    fn repr_transparent_size_is_one_byte() {
        use std::mem::size_of;
        assert_eq!(size_of::<Phase>(), 1);
    }
}
