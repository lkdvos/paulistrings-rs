//! Operator products and linear algebra on a held split, through [`Split`], [`WidthSplit`] and [`SplitImpl`]: the steps of a Lanczos recursion on a `PartitionedPauliSum`.

use num_complex::Complex64;
use paulistrings::{BuiltinTruncation, PauliSum as CorePauliSum};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::dtype::{narrow, PyCoefficient, Scalar};
use crate::partitioned::{Split, SplitImpl, WidthSplit};
use crate::storage::{width_mismatch, Pairing, PauliSumImpl, WidthSum};

impl<const W: usize, C: PyCoefficient> Split<W, C> {
    /// An empty split on the same rows, the destination of a fresh product; under MPI it owns a duplicate of this split's communicator. **Collective** under MPI.
    fn empty_like(&self) -> Self {
        match self {
            Split::Process(split) => {
                let mut out = split.empty_like();
                out.enable_trace();
                Split::Process(out)
            }
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => {
                let transport = paulistrings::mpi::MpiTransport::from_communicator(
                    split.transport().communicator(),
                );
                let mut out = split.empty_like(transport);
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

    fn add_scaled(&mut self, other: &Self, alpha: C, beta: C) {
        match (self, other) {
            (Split::Process(a), Split::Process(b)) => a.add_scaled(b, alpha, beta),
            #[cfg(feature = "mpi")]
            (Split::Ranks(a), Split::Ranks(b)) => a.add_scaled(b, alpha, beta),
            #[cfg(feature = "mpi")]
            _ => unreachable!("checked by misaligned"),
        }
    }

    /// `self ← alpha · pairing(generator, operand) + beta · self`, after [`misaligned`](Self::misaligned) found nothing. **Collective** under MPI, over this split's communicator.
    fn add_paired(
        &mut self,
        generator: &CorePauliSum<W, C>,
        operand: &Self,
        pairing: Pairing,
        (alpha, beta): (C, C),
        policy: &BuiltinTruncation,
    ) {
        match (self, operand) {
            // A held split traces only for `propagate_with_stats`, so the product's own record is dropped.
            (Split::Process(d), Split::Process(o)) => {
                match pairing {
                    Pairing::Product => d.add_product(generator, o, alpha, beta, policy),
                    Pairing::Liouvillian => d.add_liouvillian(generator, o, alpha, beta, policy),
                    Pairing::Anticommutator => {
                        d.add_anticommutator(generator, o, alpha, beta, policy)
                    }
                }
                let _ = d.take_trace();
            }
            #[cfg(feature = "mpi")]
            (Split::Ranks(d), Split::Ranks(o)) => {
                match pairing {
                    Pairing::Product => d.add_product(generator, o, alpha, beta, policy),
                    Pairing::Liouvillian => d.add_liouvillian(generator, o, alpha, beta, policy),
                    Pairing::Anticommutator => {
                        d.add_anticommutator(generator, o, alpha, beta, policy)
                    }
                }
                let _ = d.take_trace();
            }
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

impl<C: PyCoefficient> WidthSplit<C> {
    fn empty_like(&self) -> Self {
        match self {
            WidthSplit::W1(s) => WidthSplit::W1(s.empty_like()),
            WidthSplit::W2(s) => WidthSplit::W2(s.empty_like()),
            WidthSplit::W4(s) => WidthSplit::W4(s.empty_like()),
            WidthSplit::W8(s) => WidthSplit::W8(s.empty_like()),
            WidthSplit::W16(s) => WidthSplit::W16(s.empty_like()),
        }
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
        for_each_width_pair!((self, other), |a, b| a.misaligned(b))
            .unwrap_or(Some("they hold different qubit counts"))
    }

    fn add_scaled(&mut self, other: &Self, alpha: C, beta: C) {
        for_each_width_pair!((self, other), |a, b| a.add_scaled(b, alpha, beta))
            .expect("checked by misaligned");
    }

    /// `None` when the generator was monomorphized at another width.
    fn add_paired(
        &mut self,
        generator: &WidthSum<C>,
        operand: &Self,
        pairing: Pairing,
        weights: (C, C),
        policy: &BuiltinTruncation,
    ) -> Option<()> {
        for_each_width_triple!((self, WidthSum: generator, operand), |d, g, o| {
            d.add_paired(g, o, pairing, weights, policy)
        })
    }

    fn overlap(&self, other: &Self) -> C {
        for_each_width_pair!((self, other), |a, b| a.overlap(b)).expect("checked by misaligned")
    }
}

impl SplitImpl {
    /// `generator` at this split's dtype for `pairing`: a `TypeError` for a product into a real split, a `ValueError` for an imaginary part a real split cannot take.
    fn generator_at_dtype(
        &self,
        generator: &PauliSumImpl,
        pairing: Pairing,
    ) -> PyResult<PauliSumImpl> {
        let dtype = self.dtype();
        if pairing == Pairing::Product && dtype.is_real() {
            return Err(PyTypeError::new_err(format!(
                "a product is not Hermitian, so a {} split cannot hold it; partition a complex sum",
                dtype.name(),
            )));
        }
        generator.cast(dtype).ok_or_else(|| {
            PyValueError::new_err(format!(
                "the generator has imaginary coefficients, which a {} split cannot take; partition a complex sum",
                dtype.name(),
            ))
        })
    }

    /// `pairing(generator, self)` as a new split on the same rows: [`add_paired`](Self::add_paired) into an empty one. **Collective** under MPI, where the new split owns a duplicate of this one's communicator.
    ///
    /// Callers check the qubit counts and the policy first, before the GIL is released.
    pub(crate) fn paired(
        &self,
        generator: &PauliSumImpl,
        pairing: Pairing,
        policy: &BuiltinTruncation,
    ) -> PyResult<SplitImpl> {
        let mut out = for_each_dtype_convert!(SplitImpl => SplitImpl, self, |s| s.empty_like());
        let fresh = (Complex64::ONE, Complex64::new(0.0, 0.0));
        out.add_paired(generator, self, pairing, fresh, policy)?;
        Ok(out)
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
        for_each_dtype_pair!((self, other), |a, b, _wrap| a.misaligned(b))
            .flatten()
            .map(str::to_string)
    }

    /// `self ← alpha · other + beta · self`, after [`misaligned`](Self::misaligned) found nothing.
    pub(crate) fn add_scaled(
        &mut self,
        other: &SplitImpl,
        alpha: Complex64,
        beta: Complex64,
    ) -> PyResult<()> {
        for_each_dtype_pair!((self, other), |a, b, _wrap| {
            a.add_scaled(b, narrow(alpha, "alpha")?, narrow(beta, "beta")?);
            Ok(())
        })
        .expect("checked by misaligned")
    }

    /// `self ← alpha · pairing(generator, operand) + beta · self`, the generator converted to this split's dtype, after [`misaligned`](Self::misaligned) found nothing. **Collective** under MPI.
    ///
    /// Callers check the qubit counts and the policy first, before the GIL is released.
    pub(crate) fn add_paired(
        &mut self,
        generator: &PauliSumImpl,
        operand: &SplitImpl,
        pairing: Pairing,
        (alpha, beta): (Complex64, Complex64),
        policy: &BuiltinTruncation,
    ) -> PyResult<()> {
        let generator = self.generator_at_dtype(generator, pairing)?;
        for_each_dtype_triple!((self, PauliSumImpl: &generator, operand), |d, g, o| {
            let weights = (narrow(alpha, "alpha")?, narrow(beta, "beta")?);
            d.add_paired(g, o, pairing, weights, policy).ok_or_else(width_mismatch)
        })
        .expect("checked by misaligned and converted to this split's dtype")
    }

    /// The whole sums' overlap, after [`misaligned`](Self::misaligned) found nothing. **Collective** under MPI.
    pub(crate) fn overlap(&self, other: &SplitImpl) -> Scalar {
        for_each_dtype_pair!((self, other), |a, b, _wrap| Scalar::of(a.overlap(b)))
            .expect("checked by misaligned")
    }
}
