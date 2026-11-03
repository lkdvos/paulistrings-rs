//! Read-outs of a propagated [`PauliSum`]: expectation values in product and stabilizer states.

pub(crate) mod product_state;
pub(crate) mod stabilizer;

use crate::coefficient::Coefficient;
use rayon::prelude::*;

use crate::pauli_sum::PauliSum;

pub use product_state::{PauliAxis, ProductBasis};
pub use stabilizer::{StabilizerError, StabilizerState};

impl<const W: usize, C: Coefficient> PauliSum<W, C> {
    /// `Σ f(x, z, c)` over the terms, buckets in parallel; the summation order is unspecified.
    pub(crate) fn sum_over_terms(&self, f: impl Fn(&[u64; W], &[u64; W], C) -> C + Sync) -> C {
        self.buckets()
            .par_iter()
            .map(|columns| {
                let mut partial = C::ZERO;
                for i in 0..columns.len() {
                    partial += f(&columns.x[i], &columns.z[i], columns.coeff[i]);
                }
                partial
            })
            .sum()
    }
}
