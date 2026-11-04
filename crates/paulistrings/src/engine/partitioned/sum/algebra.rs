//! Linear algebra and operator products on a [`PartitionedSum`] held in host memory, the steps of a Lanczos recursion, each accumulating into a destination split.
//! Only the products communicate: equal rows put a key in the same partition of both sums.

use std::marker::PhantomData;
use std::sync::Arc;

use super::PartitionedSum;
use crate::coefficient::Coefficient;
use crate::engine::backend::HostPartition;
use crate::engine::partitioned::operator_products::pair_share_into;
use crate::engine::trace::assemble;
use crate::pauli_sum::product::{Pairing, Weights};
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

#[cfg(feature = "phase-timing")]
use super::super::driver::DriverLaps;

impl<const W: usize, C: Coefficient> Clone for PartitionedSum<W, C> {
    /// Partition by partition; the copy records no trace.
    fn clone(&self) -> Self {
        Self {
            parts: self.parts.iter().map(|part| HostPartition::new(part.sum.clone())).collect(),
            coefficient: PhantomData,
            rows: self.rows.clone(),
            runtime: self.runtime.clone(),
            trace: None,
            #[cfg(feature = "phase-timing")]
            laps: DriverLaps::new(0),
        }
    }
}

impl<const W: usize, C: Coefficient> PartitionedSum<W, C> {
    /// Panics unless `other` is split by the same rows.
    fn assert_aligned(&self, other: &Self, what: &str) {
        assert_eq!(
            self.rows, other.rows,
            "PartitionedSum::{what}: the sums are split by different rows"
        );
    }

    /// Multiply every coefficient by `c`, in place.
    pub fn scale(&mut self, c: C) {
        for part in &mut self.parts {
            part.sum.scale(c);
        }
    }

    /// `self ← alpha · other + beta · self`, in place, partition by partition and bucket by bucket ([`PauliSum::add_scaled`]).
    ///
    /// # Panics
    ///
    /// If the two sums are split by different rows.
    pub fn add_scaled(&mut self, other: &Self, alpha: C, beta: C) {
        self.assert_aligned(other, "add_scaled");
        for (part, theirs) in self.parts.iter_mut().zip(&other.parts) {
            part.sum.add_scaled(&theirs.sum, alpha, beta);
        }
    }

    /// The Hilbert-Schmidt overlap `tr(self† · other) / 2ⁿ`, the partitions' own values added in rank order.
    ///
    /// # Panics
    ///
    /// As [`add_scaled`](Self::add_scaled).
    pub fn overlap(&self, other: &Self) -> C {
        self.assert_aligned(other, "overlap");
        self.parts.iter().zip(&other.parts).map(|(a, b)| a.sum.overlap(&b.sum)).sum()
    }

    /// An empty split on the same runtime and rows.
    fn empty_like(&self) -> Self {
        Self {
            parts: self
                .parts
                .iter()
                .map(|part| {
                    let sum = &part.sum;
                    HostPartition::new(PauliSum::empty_with_hash(
                        sum.num_qubits(),
                        sum.hash().clone(),
                    ))
                })
                .collect(),
            coefficient: PhantomData,
            rows: self.rows.clone(),
            runtime: Arc::clone(&self.runtime),
            trace: None,
            #[cfg(feature = "phase-timing")]
            laps: DriverLaps::new(0),
        }
    }

    /// `self ← weights.alpha · pairing(generator, operand) + weights.beta · self`, every partition on its own pool with its own endpoint of one transport group; a traced split records the call as one layer.
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
        let rows = &self.rows;
        let items: Vec<_> =
            self.parts.iter_mut().zip(&operand.parts).map(|(d, o)| (&mut d.sum, &o.sum)).collect();
        let traced = self.runtime.map_partitions(items, |_, (dest, share), transport| {
            vec![pair_share_into(
                dest,
                generator,
                share,
                rows,
                transport,
                (pairing, weights),
                policy,
            )]
        });
        if let Some(trace) = self.trace.as_mut() {
            let partners = traced.len();
            assemble(trace, traced, partners);
        }
    }

    /// `self ← alpha · i[generator, operand] + beta · self`, truncated by `policy` as one layer; `beta = 0` ignores `self`'s terms.
    ///
    /// Each partition receives from a partner only the terms a generator term moving between the two can anticommute with ([`liouvillian_share`](crate::liouvillian_share)), and sums its pairs straight into its own buckets.
    ///
    /// # Panics
    ///
    /// If the two splits have different rows, or as [`liouvillian_share`](crate::liouvillian_share); a panic in any partition is re-raised on the calling thread.
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

    /// `self ← alpha · {generator, operand} + beta · self`, as [`add_liouvillian`](Self::add_liouvillian) takes its arguments.
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

    /// `self ← alpha · generator · operand + beta · self`, as [`add_liouvillian`](Self::add_liouvillian) takes its arguments.
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

    /// `pairing(generator, self)` as a new split on the same runtime and rows.
    fn paired_with<P>(&self, generator: &PauliSum<W, C>, pairing: Pairing, policy: &P) -> Self
    where
        P: TruncationPolicy<W, C> + ?Sized,
    {
        let mut out = self.empty_like();
        out.add_paired(generator, self, pairing, Weights::fresh(), policy);
        out
    }

    /// The Liouvillian `i[generator, self]` as a new split: [`add_liouvillian`](Self::add_liouvillian) into an empty one.
    ///
    /// # Panics
    ///
    /// As [`add_liouvillian`](Self::add_liouvillian).
    pub fn liouvillian_with<P>(&self, generator: &PauliSum<W, C>, policy: &P) -> Self
    where
        P: TruncationPolicy<W, C> + ?Sized,
    {
        self.paired_with(generator, Pairing::Liouvillian, policy)
    }

    /// The anticommutator `{generator, self}` as a new split, as [`liouvillian_with`](Self::liouvillian_with) builds it.
    ///
    /// # Panics
    ///
    /// As [`add_anticommutator`](Self::add_anticommutator).
    pub fn anticommutator_with<P>(&self, generator: &PauliSum<W, C>, policy: &P) -> Self
    where
        P: TruncationPolicy<W, C> + ?Sized,
    {
        self.paired_with(generator, Pairing::Anticommutator, policy)
    }

    /// The operator product `generator · self` as a new split, as [`liouvillian_with`](Self::liouvillian_with) builds it.
    ///
    /// # Panics
    ///
    /// As [`add_product`](Self::add_product).
    pub fn product_with<P>(&self, generator: &PauliSum<W, C>, policy: &P) -> Self
    where
        P: TruncationPolicy<W, C> + ?Sized,
    {
        self.paired_with(generator, Pairing::Product, policy)
    }
}
