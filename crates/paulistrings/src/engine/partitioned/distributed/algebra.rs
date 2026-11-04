//! Linear algebra and operator products on a [`DistributedSum`] held in host memory, the steps of a Lanczos recursion; a call that builds a new sum takes the endpoint it will own.

use num_complex::Complex64;
use std::marker::PhantomData;

#[cfg(feature = "phase-timing")]
use super::super::driver::DriverLaps;
use super::DistributedSum;
use crate::coefficient::Coefficient;
use crate::engine::backend::HostPartition;
use crate::engine::partitioned::operator_products::{
    anticommutator_share, liouvillian_share, product_share,
};
use crate::engine::transport::Transport;
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

impl<const W: usize, C: Coefficient, X: Transport> DistributedSum<W, X, C> {
    /// This rank's share as a new sum owning `transport`, an endpoint of a group with the same rank and size.
    fn with_share(&self, local: PauliSum<W, C>, transport: X) -> Self {
        assert_eq!(
            (transport.rank(), transport.size()),
            (self.transport.rank(), self.transport.size()),
            "DistributedSum: the new endpoint's rank and size differ from this sum's",
        );
        Self {
            local: HostPartition::new(local),
            coefficient: PhantomData,
            rows: self.rows.clone(),
            runtime: self.runtime.clone(),
            transport,
            trace: None,
            #[cfg(feature = "phase-timing")]
            laps: DriverLaps::new(0),
        }
    }

    /// Panics unless `other` is split by the same rows over a group of the same size.
    fn assert_aligned(&self, other: &Self, what: &str) {
        assert_eq!(
            self.rows, other.rows,
            "DistributedSum::{what}: the sums are split by different rows"
        );
        assert_eq!(
            self.transport.size(),
            other.transport.size(),
            "DistributedSum::{what}: group sizes differ"
        );
    }

    /// A copy owning `transport`, an endpoint of a group with the same rank and size.
    pub fn duplicate(&self, transport: X) -> Self {
        self.with_share(self.local.sum.clone(), transport)
    }

    /// Multiply every coefficient by `c`, in place; no communication.
    pub fn scale(&mut self, c: C) {
        self.local.sum.scale(c);
    }

    /// `self += c · other`, in place; no communication, since equal rows put a key on the same rank in both.
    ///
    /// # Panics
    ///
    /// If the two sums are split by different rows or over groups of different sizes.
    pub fn add_assign_scaled(&mut self, other: &Self, c: C) {
        self.assert_aligned(other, "add_assign_scaled");
        let mut scaled = other.local.sum.clone();
        scaled.scale(c);
        self.local.sum = self.local.sum.add(&scaled);
    }

    /// The Hilbert-Schmidt overlap `tr(self† · other) / 2ⁿ` of the whole sums, the same on every rank. **Collective.**
    ///
    /// # Panics
    ///
    /// As [`add_assign_scaled`](Self::add_assign_scaled).
    pub fn overlap(&self, other: &Self) -> C {
        self.assert_aligned(other, "overlap");
        let local = self.local.sum.overlap(&other.local.sum).to_complex64();
        let mut parts = [local.re, local.im];
        self.transport.allreduce_sum_f64(&mut parts);
        C::from_complex64(Complex64::new(parts[0], parts[1]), f64::INFINITY)
            .expect("an infinite tolerance always converts")
    }

    /// The Liouvillian `i[generator, self]`, truncated by `policy` as one layer, as a new sum owning `transport`. **Collective** over this sum's group.
    ///
    /// # Panics
    ///
    /// As [`liouvillian_share`], or if `transport`'s rank or size differ from this sum's.
    pub fn liouvillian_with<P>(&self, generator: &PauliSum<W, C>, policy: &P, transport: X) -> Self
    where
        P: TruncationPolicy<W, C> + ?Sized,
    {
        let share = self.runtime.install(|| {
            liouvillian_share(generator, &self.local.sum, &self.rows, &self.transport, policy)
        });
        self.with_share(share, transport)
    }

    /// The anticommutator `{generator, self}` as [`liouvillian_with`](Self::liouvillian_with) builds it. **Collective.**
    ///
    /// # Panics
    ///
    /// As [`anticommutator_share`].
    pub fn anticommutator_with<P>(
        &self,
        generator: &PauliSum<W, C>,
        policy: &P,
        transport: X,
    ) -> Self
    where
        P: TruncationPolicy<W, C> + ?Sized,
    {
        let share = self.runtime.install(|| {
            anticommutator_share(generator, &self.local.sum, &self.rows, &self.transport, policy)
        });
        self.with_share(share, transport)
    }

    /// The operator product `generator · self` as [`liouvillian_with`](Self::liouvillian_with) builds it. **Collective.**
    ///
    /// # Panics
    ///
    /// As [`product_share`].
    pub fn product_with<P>(&self, generator: &PauliSum<W, C>, policy: &P, transport: X) -> Self
    where
        P: TruncationPolicy<W, C> + ?Sized,
    {
        let share = self.runtime.install(|| {
            product_share(generator, &self.local.sum, &self.rows, &self.transport, policy)
        });
        self.with_share(share, transport)
    }
}
