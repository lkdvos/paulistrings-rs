//! Read-outs of a propagated [`PauliSum`]: expectation values in product and stabilizer states.

pub(crate) mod product_state;
pub(crate) mod stabilizer;

use num_complex::Complex64;
use rayon::prelude::*;

use crate::pauli_sum::PauliSum;

pub use product_state::{PauliAxis, ProductBasis};
pub use stabilizer::{StabilizerError, StabilizerState};

impl<const W: usize> PauliSum<W> {
    /// `Σ f(x, z, c)` over the terms, buckets in parallel; the summation order is unspecified.
    pub(crate) fn sum_over_terms(
        &self,
        f: impl Fn(&[u64; W], &[u64; W], Complex64) -> Complex64 + Sync,
    ) -> Complex64 {
        self.buckets()
            .par_iter()
            .map(|columns| {
                let mut partial = Complex64::new(0.0, 0.0);
                for i in 0..columns.len() {
                    partial += f(&columns.x[i], &columns.z[i], columns.coeff[i]);
                }
                partial
            })
            .sum()
    }
}
