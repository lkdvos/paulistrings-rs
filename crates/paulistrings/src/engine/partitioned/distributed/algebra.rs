//! Linear algebra and operator products on a [`DistributedSum`] held in host memory, the steps of a Lanczos recursion, each accumulating into a destination sum; a new sum takes the endpoint it will own.

use num_complex::Complex64;
use std::marker::PhantomData;

#[cfg(feature = "phase-timing")]
use super::super::driver::DriverLaps;
use super::DistributedSum;
use crate::coefficient::Coefficient;
use crate::engine::backend::HostPartition;
use crate::engine::partitioned::operator_products::pair_share_into;
use crate::engine::trace::assemble;
use crate::engine::transport::Transport;
use crate::pauli_sum::product::{Pairing, Weights};
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

    /// An empty sum split as this one and owning `transport`, the destination of a fresh product.
    pub fn empty_like(&self, transport: X) -> Self {
        let share = &self.local.sum;
        self.with_share(
            PauliSum::empty_with_hash(share.num_qubits(), share.hash().clone()),
            transport,
        )
    }

    /// Multiply every coefficient by `c`, in place; no communication.
    pub fn scale(&mut self, c: C) {
        self.local.sum.scale(c);
    }

    /// `self ← alpha · other + beta · self`, in place, bucket by bucket ([`PauliSum::add_scaled`]); no communication, since equal rows put a key on the same rank in both.
    ///
    /// # Panics
    ///
    /// If the two sums are split by different rows or over groups of different sizes.
    pub fn add_scaled(&mut self, other: &Self, alpha: C, beta: C) {
        self.assert_aligned(other, "add_scaled");
        self.local.sum.add_scaled(&other.local.sum, alpha, beta);
    }

    /// The Hilbert-Schmidt overlap `tr(self† · other) / 2ⁿ` of the whole sums, the same on every rank. **Collective.**
    ///
    /// # Panics
    ///
    /// As [`add_scaled`](Self::add_scaled).
    pub fn overlap(&self, other: &Self) -> C {
        self.assert_aligned(other, "overlap");
        let local = self.local.sum.overlap(&other.local.sum).to_complex64();
        let mut parts = [local.re, local.im];
        self.transport.allreduce_sum_f64(&mut parts);
        C::from_complex64(Complex64::new(parts[0], parts[1]), f64::INFINITY)
            .expect("an infinite tolerance always converts")
    }

    /// `self ← weights.alpha · pairing(generator, operand) + weights.beta · self` over this sum's group; a traced sum records this rank's row of the call as one layer. **Collective.**
    fn add_paired<P>(
        &mut self,
        generator: &PauliSum<W, C>,
        operand: &Self,
        pairing: Pairing,
        weights: Weights<C>,
        policy: &P,
    ) where
        P: TruncationPolicy<W, C> + ?Sized,
    {
        self.assert_aligned(operand, "add_paired");
        let (dest, rows, transport) = (&mut self.local.sum, &self.rows, &self.transport);
        let row = self.runtime.install(|| {
            pair_share_into(
                dest,
                generator,
                &operand.local.sum,
                rows,
                transport,
                (pairing, weights),
                policy,
            )
        });
        if let Some(trace) = self.trace.as_mut() {
            assemble(trace, vec![vec![row]], transport.size() as usize);
        }
    }

    /// `self ← alpha · i[generator, operand] + beta · self`, truncated by `policy` as one layer, with `operand` split by the same rows; `beta = 0` ignores `self`'s terms. **Collective** over this sum's group.
    ///
    /// # Panics
    ///
    /// If the two sums are split by different rows or over groups of different sizes, or if the ranks disagree about `generator` or `policy` has a layer pass with no partitioned form.
    pub fn add_liouvillian<P>(
        &mut self,
        generator: &PauliSum<W, C>,
        operand: &Self,
        alpha: C,
        beta: C,
        policy: &P,
    ) where
        P: TruncationPolicy<W, C> + ?Sized,
    {
        self.add_paired(generator, operand, Pairing::Liouvillian, Weights { alpha, beta }, policy);
    }

    /// `self ← alpha · {generator, operand} + beta · self`, as [`add_liouvillian`](Self::add_liouvillian) takes its arguments. **Collective.**
    ///
    /// # Panics
    ///
    /// As [`add_liouvillian`](Self::add_liouvillian).
    pub fn add_anticommutator<P>(
        &mut self,
        generator: &PauliSum<W, C>,
        operand: &Self,
        alpha: C,
        beta: C,
        policy: &P,
    ) where
        P: TruncationPolicy<W, C> + ?Sized,
    {
        let weights = Weights { alpha, beta };
        self.add_paired(generator, operand, Pairing::Anticommutator, weights, policy);
    }

    /// `self ← alpha · generator · operand + beta · self`, as [`add_liouvillian`](Self::add_liouvillian) takes its arguments. **Collective.**
    ///
    /// # Panics
    ///
    /// As [`add_liouvillian`](Self::add_liouvillian), and if `C` is real.
    pub fn add_product<P>(
        &mut self,
        generator: &PauliSum<W, C>,
        operand: &Self,
        alpha: C,
        beta: C,
        policy: &P,
    ) where
        P: TruncationPolicy<W, C> + ?Sized,
    {
        self.add_paired(generator, operand, Pairing::Product, Weights { alpha, beta }, policy);
    }
}
