//! Operator Loschmidt-echo read-outs on a propagated operator `A`: the exact rotated overlap `2⁻ⁿ Tr(A V† A V)` and the anticommutation histogram behind its diagonal approximation.
//! `V = ⊗_{q ∈ sites} exp(-iδ G_q)` with `G_q` the single-qubit `X` or `Z` named by [`RotationAxis`].

use num_complex::Complex64;
use rayon::prelude::*;

use crate::pauli_sum::{PartitionRows, PauliSum};

/// The single-qubit generator `G_q` of an echo perturbation `V = ⊗_{q ∈ sites} exp(-iδ G_q)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationAxis {
    /// `G_q = X_q`.
    X,
    /// `G_q = Z_q`.
    Z,
}

impl RotationAxis {
    /// The key coordinates that conjugation by `V` can change, as `(x-mask, z-mask)`: the `z`-bits of `sites` for [`Z`](Self::Z), the `x`-bits for [`X`](Self::X).
    ///
    /// # Panics
    ///
    /// If a site is repeated or is not below `64 · W`.
    pub fn flip_mask<const W: usize>(self, sites: &[usize]) -> ([u64; W], [u64; W]) {
        let mask = site_mask::<W>(sites, 64 * W);
        match self {
            RotationAxis::Z => ([0; W], mask),
            RotationAxis::X => (mask, [0; W]),
        }
    }

    /// `(selector, flip)` halves of a key: `G_q` anticommutes with the term iff the selector bit is set on `q`, and `V` then flips the other half's bit there.
    #[inline]
    fn split<'a, const W: usize>(
        self,
        x: &'a [u64; W],
        z: &'a [u64; W],
    ) -> (&'a [u64; W], &'a [u64; W]) {
        match self {
            RotationAxis::Z => (x, z),
            RotationAxis::X => (z, x),
        }
    }
}

/// The diagonal echo `Σₙ wₙ cos(2δ)ⁿ / Σₙ wₙ` from an [`anticommute_histogram`](PauliSum::anticommute_histogram) `hist`.
///
/// It keeps the class-diagonal terms of [`rotated_overlap`](PauliSum::rotated_overlap), normalized by `Σ|c|²`; an all-zero histogram gives `NaN`.
pub fn diagonal_echo(hist: &[f64], delta: f64) -> f64 {
    let cos = (2.0 * delta).cos();
    let mut weight = 1.0;
    let mut numerator = 0.0;
    for &w in hist {
        numerator += w * weight;
        weight *= cos;
    }
    numerator / hist.iter().sum::<f64>()
}

/// The bit mask of `qubits`, each asserted below `num_qubits`.
pub(crate) fn qubit_mask<const W: usize>(
    qubits: impl IntoIterator<Item = usize>,
    num_qubits: usize,
) -> [u64; W] {
    let mut mask = [0u64; W];
    for q in qubits {
        assert!(
            q < num_qubits,
            "qubit {q} is outside the {num_qubits}-qubit register"
        );
        mask[q / 64] |= 1u64 << (q % 64);
    }
    mask
}

/// [`qubit_mask`] of `sites`, which must also be distinct.
fn site_mask<const W: usize>(sites: &[usize], num_qubits: usize) -> [u64; W] {
    let mask = qubit_mask(sites.iter().copied(), num_qubits);
    assert_eq!(
        popcount(&mask),
        sites.len(),
        "echo: a site is listed twice in {sites:?}"
    );
    mask
}

#[inline]
fn and<const W: usize>(a: &[u64; W], b: &[u64; W]) -> [u64; W] {
    std::array::from_fn(|w| a[w] & b[w])
}

#[inline]
fn popcount<const W: usize>(a: &[u64; W]) -> usize {
    a.iter().map(|w| w.count_ones() as usize).sum()
}

impl<const W: usize> PartitionRows<W> {
    /// `true` if no row reads a coordinate [`RotationAxis::flip_mask`] of `sites` names, so every [`PauliSum::rotated_overlap`] class lies in one partition.
    pub fn keeps_flip_classes(&self, sites: &[usize], axis: RotationAxis) -> bool {
        let (mask_x, mask_z) = axis.flip_mask(sites);
        self.avoids(&mask_x, &mask_z)
    }
}

/// One term that anticommutes with some generator, keyed by its class.
struct Member<const W: usize> {
    /// The key with the flippable bits cleared: equal for exactly the strings `V† · V` mixes.
    class: ([u64; W], [u64; W]),
    /// The flippable bits themselves, on the anticommuting sites `K`.
    flip: [u64; W],
    coeff: Complex64,
}

impl<const W: usize> PauliSum<W> {
    /// `w[n] = Σ |c_P|²` over the strings `P` that anticommute with exactly `n` of the generators `G_q`, `q ∈ sites`; length `sites.len() + 1`.
    ///
    /// Feed it to [`diagonal_echo`]; histograms of disjoint shares of a sum add.
    ///
    /// # Panics
    ///
    /// If a site is repeated or is not below [`num_qubits`](Self::num_qubits).
    pub fn anticommute_histogram(&self, sites: &[usize], axis: RotationAxis) -> Vec<f64> {
        let mask = site_mask::<W>(sites, self.num_qubits());
        let bins = sites.len() + 1;
        (0..self.num_buckets())
            .into_par_iter()
            .fold(
                || vec![0.0f64; bins],
                |mut hist, b| {
                    let (xs, zs, cs) = self.bucket(b);
                    for ((x, z), c) in xs.iter().zip(zs).zip(cs) {
                        hist[popcount(&and(axis.split(x, z).0, &mask))] += c.norm_sqr();
                    }
                    hist
                },
            )
            .reduce(
                || vec![0.0f64; bins],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            )
    }

    /// The operator Loschmidt echo `S_δ = 2⁻ⁿ Tr(A V† A V)` of this sum `A`, with `V = ⊗_{q ∈ sites} exp(-iδ G_q)`, computed without materializing `V† A V`.
    ///
    /// The value is the real part of `2⁻ⁿ Tr(A† V† A V)`, which is `S_δ` itself for a Hermitian `A`.
    /// Work is `O(k²)` per class of `k` strings agreeing off the anticommuting sites' flipped bits, so one large class dominates.
    ///
    /// # Panics
    ///
    /// If a site is repeated or is not below [`num_qubits`](Self::num_qubits).
    pub fn rotated_overlap(&self, sites: &[usize], delta: f64, axis: RotationAxis) -> f64 {
        self.rotated_overlap_complex(sites, delta, axis).re
    }

    /// `2⁻ⁿ Tr(A† V† A V)` in full, the sign convention included; [`Self::rotated_overlap`] is its real part.
    fn rotated_overlap_complex(
        &self,
        sites: &[usize],
        delta: f64,
        axis: RotationAxis,
    ) -> Complex64 {
        let mask = site_mask::<W>(sites, self.num_qubits());
        let (sin, cos) = (2.0 * delta).sin_cos();
        let cos_pow: Vec<f64> = (0..=sites.len()).map(|k| cos.powi(k as i32)).collect();
        let sin_pow: Vec<f64> = (0..=sites.len()).map(|k| sin.powi(k as i32)).collect();

        // A string commuting with every generator is its own class, and V leaves it alone.
        let (fixed, members): (Vec<f64>, Vec<Vec<Member<W>>>) = (0..self.num_buckets())
            .into_par_iter()
            .fold(
                || (0.0, Vec::new()),
                |(mut fixed, mut members), b| {
                    let (xs, zs, cs) = self.bucket(b);
                    for ((x, z), &coeff) in xs.iter().zip(zs).zip(cs) {
                        let (selector, flip) = axis.split(x, z);
                        let k = and(selector, &mask);
                        if popcount(&k) == 0 {
                            fixed += coeff.norm_sqr();
                            continue;
                        }
                        let cleared: [u64; W] = std::array::from_fn(|w| flip[w] & !k[w]);
                        let class = match axis {
                            RotationAxis::Z => (*x, cleared),
                            RotationAxis::X => (cleared, *z),
                        };
                        let flip = and(flip, &k);
                        members.push(Member { class, flip, coeff });
                    }
                    (fixed, members)
                },
            )
            .unzip();
        let mut members: Vec<Member<W>> = members.into_iter().flatten().collect();
        members.par_sort_unstable_by(|a, b| a.class.cmp(&b.class));

        let mixed: Complex64 = members
            .par_chunk_by(|a, b| a.class == b.class)
            .map(|class| {
                let (x, z) = &class[0].class;
                let sites_k = popcount(&and(axis.split(x, z).0, &mask));
                let mut partial = Complex64::new(0.0, 0.0);
                for p in class {
                    for q in class {
                        let d: [u64; W] = std::array::from_fn(|i| p.flip[i] ^ q.flip[i]);
                        let num_flipped = popcount(&d);
                        // R[P, Q] is `−sin 2δ` where P holds Y against Q's X (Z axis), or P holds Z against Q's Y (X axis).
                        let negative = match axis {
                            RotationAxis::Z => popcount(&and(&p.flip, &d)),
                            RotationAxis::X => popcount(&and(&q.flip, &d)),
                        } % 2
                            == 1;
                        let magnitude = cos_pow[sites_k - num_flipped] * sin_pow[num_flipped];
                        partial += p.coeff.conj()
                            * q.coeff
                            * if negative { -magnitude } else { magnitude };
                    }
                }
                partial
            })
            .sum();

        mixed + fixed.iter().sum::<f64>()
    }
}

#[cfg(test)]
mod tests;
