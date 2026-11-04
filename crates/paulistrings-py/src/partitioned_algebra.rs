//! Operator products and linear algebra on a held split, through [`Split`], [`WidthSplit`] and [`SplitImpl`]: the steps of a Lanczos recursion on a `PartitionedPauliSum`.

use num_complex::Complex64;
use paulistrings::{BuiltinTruncation, PauliSum as CorePauliSum};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::dtype::{narrow, PyCoefficient, Scalar};
use crate::partitioned::{Split, SplitImpl, WidthSplit};
use crate::storage::{Pairing, PauliSumImpl, WidthSum};

impl<const W: usize, C: PyCoefficient> Split<W, C> {
    /// `pairing(generator, self)` as a new split on the same rows, truncated by `policy` as one layer. **Collective** under MPI, where the new split owns a duplicate of this one's communicator.
    fn paired(
        &self,
        generator: &CorePauliSum<W, C>,
        pairing: Pairing,
        policy: &BuiltinTruncation,
    ) -> Self {
        match self {
            Split::Process(split) => {
                let mut out = match pairing {
                    Pairing::Product => split.product_with(generator, policy),
                    Pairing::Liouvillian => split.liouvillian_with(generator, policy),
                    Pairing::Anticommutator => split.anticommutator_with(generator, policy),
                };
                out.enable_trace();
                Split::Process(out)
            }
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => {
                let transport = paulistrings::mpi::MpiTransport::from_communicator(
                    split.transport().communicator(),
                );
                let mut out = match pairing {
                    Pairing::Product => split.product_with(generator, policy, transport),
                    Pairing::Liouvillian => split.liouvillian_with(generator, policy, transport),
                    Pairing::Anticommutator => {
                        split.anticommutator_with(generator, policy, transport)
                    }
                };
                out.enable_trace();
                Split::Ranks(Box::new(out))
            }
        }
    }

    /// A copy; under MPI it owns a duplicate of this split's communicator. **Collective** under MPI.
    fn duplicate(&self) -> Self {
        match self {
            Split::Process(split) => {
                let mut out = split.clone();
                out.enable_trace();
                Split::Process(out)
            }
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => {
                let transport = paulistrings::mpi::MpiTransport::from_communicator(
                    split.transport().communicator(),
                );
                let mut out = split.duplicate(transport);
                out.enable_trace();
                Split::Ranks(Box::new(out))
            }
        }
    }

    fn scale(&mut self, c: C) {
        match self {
            Split::Process(split) => split.scale(c),
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => split.scale(c),
        }
    }

    /// Why `self` and `other` cannot be combined term by term, if they cannot.
    fn misaligned(&self, other: &Self) -> Option<&'static str> {
        match (self, other) {
            (Split::Process(a), Split::Process(b)) if a.rows() == b.rows() => None,
            #[cfg(feature = "mpi")]
            (Split::Ranks(a), Split::Ranks(b)) if a.rows() == b.rows() => None,
            #[cfg(feature = "mpi")]
            (Split::Process(_), Split::Ranks(_)) | (Split::Ranks(_), Split::Process(_)) => {
                Some("one is split in this process and the other across MPI ranks")
            }
            _ => {
                Some("they are split by different partition rows (repartition one to the other's)")
            }
        }
    }

    fn add_assign_scaled(&mut self, other: &Self, c: C) {
        match (self, other) {
            (Split::Process(a), Split::Process(b)) => a.add_assign_scaled(b, c),
            #[cfg(feature = "mpi")]
            (Split::Ranks(a), Split::Ranks(b)) => a.add_assign_scaled(b, c),
            #[cfg(feature = "mpi")]
            _ => unreachable!("checked by misaligned"),
        }
    }

    /// The whole sums' overlap, the same on every rank. **Collective** under MPI.
    fn overlap(&self, other: &Self) -> C {
        match (self, other) {
            (Split::Process(a), Split::Process(b)) => a.overlap(b),
            #[cfg(feature = "mpi")]
            (Split::Ranks(a), Split::Ranks(b)) => a.overlap(b),
            #[cfg(feature = "mpi")]
            _ => unreachable!("checked by misaligned"),
        }
    }
}

/// Bind the matching widths of a `WidthSplit` and a value of the width enum `$other`, with `$wrap` the `WidthSplit` constructor of that width; `None` on a mismatch.
macro_rules! width_pair {
    (($a:expr, $b:expr), $other:ident, |$x:ident, $y:ident, $wrap:ident| $body:expr) => {
        match ($a, $b) {
            (WidthSplit::W1($x), $other::W1($y)) => {
                let $wrap = Self::W1;
                Some($body)
            }
            (WidthSplit::W2($x), $other::W2($y)) => {
                let $wrap = Self::W2;
                Some($body)
            }
            (WidthSplit::W4($x), $other::W4($y)) => {
                let $wrap = Self::W4;
                Some($body)
            }
            (WidthSplit::W8($x), $other::W8($y)) => {
                let $wrap = Self::W8;
                Some($body)
            }
            (WidthSplit::W16($x), $other::W16($y)) => {
                let $wrap = Self::W16;
                Some($body)
            }
            _ => None,
        }
    };
}

impl<C: PyCoefficient> WidthSplit<C> {
    /// `None` when the generator was monomorphized at another width.
    fn paired(
        &self,
        generator: &WidthSum<C>,
        pairing: Pairing,
        policy: &BuiltinTruncation,
    ) -> Option<Self> {
        width_pair!((self, generator), WidthSum, |s, g, wrap| wrap(s.paired(g, pairing, policy)))
    }

    fn duplicate(&self) -> Self {
        match self {
            WidthSplit::W1(s) => WidthSplit::W1(s.duplicate()),
            WidthSplit::W2(s) => WidthSplit::W2(s.duplicate()),
            WidthSplit::W4(s) => WidthSplit::W4(s.duplicate()),
            WidthSplit::W8(s) => WidthSplit::W8(s.duplicate()),
            WidthSplit::W16(s) => WidthSplit::W16(s.duplicate()),
        }
    }

    fn scale(&mut self, c: C) {
        for_each_width!(self, |s| s.scale(c))
    }

    fn misaligned(&self, other: &Self) -> Option<&'static str> {
        width_pair!((self, other), WidthSplit, |a, b, _wrap| a.misaligned(b))
            .unwrap_or(Some("they hold different qubit counts"))
    }

    fn add_assign_scaled(&mut self, other: &Self, c: C) {
        width_pair!((self, other), WidthSplit, |a, b, _wrap| a.add_assign_scaled(b, c))
            .expect("checked by misaligned");
    }

    fn overlap(&self, other: &Self) -> C {
        width_pair!((self, other), WidthSplit, |a, b, _wrap| a.overlap(b))
            .expect("checked by misaligned")
    }
}

/// Bind the matching dtypes of two `SplitImpl`s, or of a `SplitImpl` and a `PauliSumImpl` already converted to its dtype.
macro_rules! dtype_pair {
    (($a:expr, $b:expr), $other:ident, |$x:ident, $y:ident, $wrap:ident| $body:expr) => {
        match ($a, $b) {
            (SplitImpl::F64($x), $other::F64($y)) => {
                let $wrap = SplitImpl::F64;
                Some($body)
            }
            (SplitImpl::F32($x), $other::F32($y)) => {
                let $wrap = SplitImpl::F32;
                Some($body)
            }
            (SplitImpl::C128($x), $other::C128($y)) => {
                let $wrap = SplitImpl::C128;
                Some($body)
            }
            (SplitImpl::C64($x), $other::C64($y)) => {
                let $wrap = SplitImpl::C64;
                Some($body)
            }
            _ => None,
        }
    };
}

impl SplitImpl {
    /// `pairing(generator, self)` as a new split, the generator converted to this split's dtype. **Collective** under MPI.
    ///
    /// Callers check the qubit counts and the policy first, before the GIL is released.
    pub(crate) fn paired(
        &self,
        generator: &PauliSumImpl,
        pairing: Pairing,
        policy: &BuiltinTruncation,
    ) -> PyResult<SplitImpl> {
        let dtype = self.dtype();
        if pairing == Pairing::Product && dtype.is_real() {
            return Err(PyTypeError::new_err(format!(
                "a product is not Hermitian, so a {} split cannot hold it; partition a complex sum",
                dtype.name(),
            )));
        }
        let generator = generator.cast(dtype).ok_or_else(|| {
            PyValueError::new_err(format!(
                "the generator has imaginary coefficients, which a {} split cannot take; partition a complex sum",
                dtype.name(),
            ))
        })?;
        dtype_pair!((self, &generator), PauliSumImpl, |s, g, wrap| s
            .paired(g, pairing, policy)
            .map(wrap))
        .flatten()
        .ok_or_else(|| PyValueError::new_err("the generator was monomorphized at another width"))
    }

    /// A copy. **Collective** under MPI.
    pub(crate) fn duplicate(&self) -> SplitImpl {
        for_each_dtype_convert!(SplitImpl => SplitImpl, self, |s| s.duplicate())
    }

    /// Multiply every coefficient by `factor`; a `ValueError` for an imaginary factor on a real split.
    pub(crate) fn scale(&mut self, factor: Complex64) -> PyResult<()> {
        for_each_dtype!(self, |s| {
            s.scale(narrow(factor, "the scale factor")?);
            Ok(())
        })
    }

    /// Why `self` and `other` cannot be combined term by term, if they cannot; the same answer on every rank.
    pub(crate) fn misaligned(&self, other: &SplitImpl) -> Option<String> {
        if self.dtype() != other.dtype() {
            return Some(format!(
                "their dtypes differ ({} vs {})",
                self.dtype().name(),
                other.dtype().name()
            ));
        }
        dtype_pair!((self, other), SplitImpl, |a, b, _wrap| a.misaligned(b))
            .flatten()
            .map(str::to_string)
    }

    /// `self += factor · other`, after [`misaligned`](Self::misaligned) found nothing.
    pub(crate) fn add_assign_scaled(
        &mut self,
        other: &SplitImpl,
        factor: Complex64,
    ) -> PyResult<()> {
        dtype_pair!((self, other), SplitImpl, |a, b, _wrap| {
            a.add_assign_scaled(b, narrow(factor, "the factor")?);
            Ok(())
        })
        .expect("checked by misaligned")
    }

    /// The whole sums' overlap, after [`misaligned`](Self::misaligned) found nothing. **Collective** under MPI.
    pub(crate) fn overlap(&self, other: &SplitImpl) -> Scalar {
        dtype_pair!((self, other), SplitImpl, |a, b, _wrap| Scalar::of(a.overlap(b)))
            .expect("checked by misaligned")
    }
}
