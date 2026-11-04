//! Operator products under a [`TruncationPolicy`]: [`PauliSum::product_with`], [`PauliSum::liouvillian_with`] and [`PauliSum::anticommutator_with`].
//! One call is one layer to the policy: `keep_term` sees each output key's summed coefficient as its bucket is drained, and the layer pass runs once on the result.

use crate::coefficient::Coefficient;
use crate::pauli_sum::product::Pairing;
use crate::pauli_sum::storage::{DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

impl<const W: usize, C: Coefficient> PauliSum<W, C> {
    fn pair_with<P: TruncationPolicy<W, C> + ?Sized>(
        &self,
        other: &Self,
        pairing: Pairing,
        policy: &P,
    ) -> Self {
        let mut out = self.pair(other, pairing, &|x, z, c| policy.keep_term(x, z, c));
        if policy.finalizes_layer() {
            policy.finalize_layer(&mut out);
            out.rebucket(DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS);
        }
        out
    }

    /// [`Self::product`], truncated by `policy` as one layer.
    ///
    /// # Panics
    ///
    /// As [`Self::product`].
    pub fn product_with<P: TruncationPolicy<W, C> + ?Sized>(
        &self,
        other: &Self,
        policy: &P,
    ) -> Self {
        assert!(
            !C::IS_REAL,
            "PauliSum::product_with: a product is not Hermitian, so a real coefficient type ({}) cannot hold it; cast to a complex type first",
            C::NAME,
        );
        self.pair_with(other, Pairing::Product, policy)
    }

    /// [`Self::liouvillian`], truncated by `policy` as one layer.
    ///
    /// # Panics
    ///
    /// As [`Self::liouvillian`].
    pub fn liouvillian_with<P: TruncationPolicy<W, C> + ?Sized>(
        &self,
        other: &Self,
        policy: &P,
    ) -> Self {
        self.pair_with(other, Pairing::Liouvillian, policy)
    }

    /// [`Self::anticommutator`], truncated by `policy` as one layer.
    ///
    /// # Panics
    ///
    /// As [`Self::anticommutator`].
    pub fn anticommutator_with<P: TruncationPolicy<W, C> + ?Sized>(
        &self,
        other: &Self,
        policy: &P,
    ) -> Self {
        self.pair_with(other, Pairing::Anticommutator, policy)
    }
}

#[cfg(test)]
mod tests;
