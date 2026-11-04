//! Operator products accumulated into a destination under a [`TruncationPolicy`]: [`PauliSum::add_product`], [`PauliSum::add_liouvillian`] and [`PauliSum::add_anticommutator`], with `*_with` shorthands into a fresh sum.
//! One call is one layer to the policy: `keep_term` sees each output key's summed coefficient, the destination's own share included, as its bucket is drained, and the layer pass runs once on the result.

use crate::coefficient::Coefficient;
use crate::pauli_sum::product::{Pairing, Weights};
use crate::pauli_sum::storage::{DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

impl<const W: usize, C: Coefficient> PauliSum<W, C> {
    fn add_paired<P: TruncationPolicy<W, C> + ?Sized>(
        &mut self,
        generator: &Self,
        operand: &Self,
        pairing: Pairing,
        weights: Weights<C>,
        policy: &P,
    ) {
        self.pair_into(generator, operand, pairing, weights, &|x, z, c| policy.keep_term(x, z, c));
        if policy.finalizes_layer() {
            policy.finalize_layer(self);
            self.rebucket(DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS);
        }
    }

    /// `self ← alpha · generator · operand + beta · self`, truncated by `policy` as one layer; `beta = 0` ignores `self`'s terms.
    ///
    /// The pairs are summed straight into `self`'s buckets, one bucket at a time, and `self` ends partitioned under the larger operand's hash rows.
    ///
    /// # Panics
    ///
    /// If the three sums disagree about `num_qubits`, or if `C` is real: a product of anticommuting Hermitian strings carries an odd phase that no real coefficient holds.
    pub fn add_product<P: TruncationPolicy<W, C> + ?Sized>(
        &mut self,
        generator: &Self,
        operand: &Self,
        alpha: C,
        beta: C,
        policy: &P,
    ) {
        assert!(
            !C::IS_REAL,
            "PauliSum::add_product: a product is not Hermitian, so a real coefficient type ({}) cannot hold it; cast to a complex type first",
            C::NAME,
        );
        self.add_paired(generator, operand, Pairing::Product, Weights { alpha, beta }, policy);
    }

    /// `self ← alpha · i[generator, operand] + beta · self`, as [`Self::add_product`] accumulates; only anticommuting pairs whose supports meet are visited.
    ///
    /// # Panics
    ///
    /// If the three sums disagree about `num_qubits`.
    pub fn add_liouvillian<P: TruncationPolicy<W, C> + ?Sized>(
        &mut self,
        generator: &Self,
        operand: &Self,
        alpha: C,
        beta: C,
        policy: &P,
    ) {
        self.add_paired(generator, operand, Pairing::Liouvillian, Weights { alpha, beta }, policy);
    }

    /// `self ← alpha · {generator, operand} + beta · self`, as [`Self::add_product`] accumulates.
    ///
    /// # Panics
    ///
    /// If the three sums disagree about `num_qubits`.
    pub fn add_anticommutator<P: TruncationPolicy<W, C> + ?Sized>(
        &mut self,
        generator: &Self,
        operand: &Self,
        alpha: C,
        beta: C,
        policy: &P,
    ) {
        self.add_paired(
            generator,
            operand,
            Pairing::Anticommutator,
            Weights { alpha, beta },
            policy,
        );
    }

    /// `pairing(self, other)` under `policy` into an empty destination.
    fn fresh<P: TruncationPolicy<W, C> + ?Sized>(
        &self,
        other: &Self,
        pairing: Pairing,
        policy: &P,
    ) -> Self {
        let mut out = Self::empty_with_hash(self.num_qubits(), other.hash().clone());
        out.add_paired(self, other, pairing, Weights::fresh(), policy);
        out
    }

    /// [`Self::product`], truncated by `policy` as one layer.
    ///
    /// # Panics
    ///
    /// As [`Self::add_product`].
    pub fn product_with<P: TruncationPolicy<W, C> + ?Sized>(
        &self,
        other: &Self,
        policy: &P,
    ) -> Self {
        let mut out = Self::empty_with_hash(self.num_qubits(), other.hash().clone());
        out.add_product(self, other, C::ONE, C::ZERO, policy);
        out
    }

    /// [`Self::liouvillian`], truncated by `policy` as one layer.
    ///
    /// # Panics
    ///
    /// As [`Self::add_liouvillian`].
    pub fn liouvillian_with<P: TruncationPolicy<W, C> + ?Sized>(
        &self,
        other: &Self,
        policy: &P,
    ) -> Self {
        self.fresh(other, Pairing::Liouvillian, policy)
    }

    /// [`Self::anticommutator`], truncated by `policy` as one layer.
    ///
    /// # Panics
    ///
    /// As [`Self::add_anticommutator`].
    pub fn anticommutator_with<P: TruncationPolicy<W, C> + ?Sized>(
        &self,
        other: &Self,
        policy: &P,
    ) -> Self {
        self.fresh(other, Pairing::Anticommutator, policy)
    }
}

#[cfg(test)]
mod tests;
