//! [`Coefficient`], the per-term scalar of a [`PauliSum`](crate::PauliSum): `f64`, `f32`, `Complex64` or `Complex32`.
//! Keys are Hermitian Pauli strings, so a real type is exact for a Hermitian operator under Hermiticity-preserving channels; the engine rejects a layer whose transfer table is not real before running it.

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

use num_complex::{Complex32, Complex64};

use crate::phase::Phase;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for num_complex::Complex64 {}
    impl Sealed for num_complex::Complex32 {}
}

/// Largest imaginary part a transfer-table amplitude may carry and still convert to a real coefficient type.
pub(crate) const REAL_TOLERANCE: f64 = 1e-12;

/// The scalar a [`PauliSum`](crate::PauliSum) stores per term; sealed, implemented for `f64`, `f32`, `Complex64` and `Complex32`.
pub trait Coefficient:
    sealed::Sealed
    + Copy
    + Send
    + Sync
    + Default
    + PartialEq
    + Debug
    + bytemuck::Pod
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Mul<Output = Self>
    + MulAssign
    + core::iter::Sum
    + Mul<Self::Real, Output = Self>
    + 'static
{
    /// The real scalar of the same precision, which rotation angles are multiplied in.
    type Real: Copy + Send + Sync + Debug + PartialEq + bytemuck::Pod + 'static;

    /// Whether the type has no imaginary part.
    const IS_REAL: bool;

    /// Additive identity.
    const ZERO: Self;

    /// Multiplicative identity.
    const ONE: Self;

    /// Short dtype name: `"f64"`, `"f32"`, `"c128"` or `"c64"`.
    const NAME: &'static str;

    /// `r` rounded to [`Self::Real`].
    fn real(r: f64) -> Self::Real;

    /// `r` as a coefficient.
    fn from_f64(r: f64) -> Self;

    /// `c` rounded to this type; `None` for a real type when `|c.im|` exceeds `tolerance`.
    fn from_complex64(c: Complex64, tolerance: f64) -> Option<Self>;

    /// Widened to `Complex64`.
    fn to_complex64(self) -> Complex64;

    /// `self · i^k`; a real type only ever sees an even `k`, since the product of two anticommuting Hermitian Pauli strings carries an odd phase that the rotation's `i` makes even.
    fn mul_phase(self, phase: Phase) -> Self;

    /// Complex conjugate.
    fn conj(self) -> Self;

    /// `|c|²`, in `f64`.
    fn norm_sqr(self) -> f64;

    /// `self != 0` without a short-circuit across parts: one flag, no branch.
    fn is_nonzero(self) -> bool;
}

macro_rules! real_coefficient {
    ($t:ty, $name:literal) => {
        impl Coefficient for $t {
            type Real = $t;
            const IS_REAL: bool = true;
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const NAME: &'static str = $name;

            #[inline]
            fn real(r: f64) -> $t {
                r as $t
            }

            #[inline]
            fn from_f64(r: f64) -> Self {
                r as $t
            }

            #[inline]
            fn from_complex64(c: Complex64, tolerance: f64) -> Option<Self> {
                (c.im.abs() <= tolerance).then_some(c.re as $t)
            }

            #[inline]
            fn to_complex64(self) -> Complex64 {
                Complex64::new(self as f64, 0.0)
            }

            #[inline]
            fn mul_phase(self, phase: Phase) -> Self {
                debug_assert!(phase.exponent() & 1 == 0, "odd phase on a real coefficient");
                if phase.exponent() & 2 == 0 {
                    self
                } else {
                    -self
                }
            }

            #[inline]
            fn conj(self) -> Self {
                self
            }

            #[inline]
            fn norm_sqr(self) -> f64 {
                let v = self as f64;
                v * v
            }

            #[inline]
            fn is_nonzero(self) -> bool {
                self != 0.0
            }
        }
    };
}

macro_rules! complex_coefficient {
    ($t:ty, $r:ty, $name:literal) => {
        impl Coefficient for $t {
            type Real = $r;
            const IS_REAL: bool = false;
            const ZERO: Self = <$t>::new(0.0, 0.0);
            const ONE: Self = <$t>::new(1.0, 0.0);
            const NAME: &'static str = $name;

            #[inline]
            fn real(r: f64) -> $r {
                r as $r
            }

            #[inline]
            fn from_f64(r: f64) -> Self {
                <$t>::new(r as $r, 0.0)
            }

            #[inline]
            fn from_complex64(c: Complex64, _tolerance: f64) -> Option<Self> {
                Some(<$t>::new(c.re as $r, c.im as $r))
            }

            #[inline]
            fn to_complex64(self) -> Complex64 {
                Complex64::new(self.re as f64, self.im as f64)
            }

            /// By a sign/swap of the parts, without a float multiply.
            #[inline]
            fn mul_phase(self, phase: Phase) -> Self {
                match phase.exponent() {
                    0 => self,
                    1 => <$t>::new(-self.im, self.re),
                    2 => <$t>::new(-self.re, -self.im),
                    _ => <$t>::new(self.im, -self.re),
                }
            }

            #[inline]
            fn conj(self) -> Self {
                <$t>::new(self.re, -self.im)
            }

            #[inline]
            fn norm_sqr(self) -> f64 {
                let wide = self.to_complex64();
                wide.re * wide.re + wide.im * wide.im
            }

            #[inline]
            fn is_nonzero(self) -> bool {
                (self.re != 0.0) | (self.im != 0.0)
            }
        }
    };
}

real_coefficient!(f64, "f64");
real_coefficient!(f32, "f32");
complex_coefficient!(Complex64, f64, "c128");
complex_coefficient!(Complex32, f32, "c64");

#[cfg(test)]
mod tests;
