//! Built-in truncation policies and combinators. See §7.

#![allow(unused)]

use super::TruncationPolicy;
use crate::pauli_sum::PauliSum;
use num_complex::Complex64;

/// Drop terms whose coefficient magnitude falls below `epsilon`.
pub struct CoefficientThreshold(pub f64);

impl<const W: usize> TruncationPolicy<W> for CoefficientThreshold {
    #[inline]
    fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], c: Complex64) -> bool {
        c.norm() > self.0
    }
}

/// Drop terms whose Pauli weight (number of non-identity qubits) exceeds `k`.
pub struct WeightCutoff(pub u32);

impl<const W: usize> TruncationPolicy<W> for WeightCutoff {
    #[inline]
    fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], _c: Complex64) -> bool {
        todo!("§7: popcount(x[i] | z[i]) <= self.0")
    }
}

/// Retain only the `n` terms with largest coefficient magnitude. Implemented
/// as a `finalize_layer` partial sort (no per-term filter).
pub struct TopN(pub usize);

impl<const W: usize> TruncationPolicy<W> for TopN {
    fn finalize_layer(&self, _sum: &mut PauliSum<W>) {
        todo!("§7: select_nth_unstable_by on |coeff|, then truncate parallel arrays")
    }
}

/// Logical AND of two policies — both must accept.
pub struct And<A, B>(pub A, pub B);

impl<const W: usize, A, B> TruncationPolicy<W> for And<A, B>
where
    A: TruncationPolicy<W>,
    B: TruncationPolicy<W>,
{
    #[inline]
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
        self.0.keep_term(x, z, c) && self.1.keep_term(x, z, c)
    }

    fn finalize_layer(&self, sum: &mut PauliSum<W>) {
        self.0.finalize_layer(sum);
        self.1.finalize_layer(sum);
    }
}

/// Logical OR of two policies — either accepting is enough.
pub struct Or<A, B>(pub A, pub B);

impl<const W: usize, A, B> TruncationPolicy<W> for Or<A, B>
where
    A: TruncationPolicy<W>,
    B: TruncationPolicy<W>,
{
    #[inline]
    fn keep_term(&self, x: &[u64; W], z: &[u64; W], c: Complex64) -> bool {
        self.0.keep_term(x, z, c) || self.1.keep_term(x, z, c)
    }
}
