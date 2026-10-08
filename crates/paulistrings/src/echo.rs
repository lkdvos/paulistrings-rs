//! Operator Loschmidt-echo read-outs on a propagated operator `A`: the exact rotated overlap `2⁻ⁿ Tr(A V† A V)` and the anticommutation histogram behind its diagonal approximation.
//! `V = ⊗_{q ∈ sites} exp(-iδ G_q)` with `G_q` the single-qubit `X` or `Z` named by [`RotationAxis`].

use num_complex::Complex64;
use rayon::prelude::*;

use crate::pauli_sum::PauliSum;

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
/// It keeps the `P = Q` terms of [`rotated_overlap`](PauliSum::rotated_overlap) and normalizes by `Σ|c|²`: a string anticommuting with `n` of the generators keeps weight `cos(2δ)ⁿ` of itself under `V† · V`.
/// An all-zero histogram gives `NaN`.
///
/// ```
/// use paulistrings::diagonal_echo;
///
/// // Half the weight commutes with every generator, half anticommutes with two.
/// let s = diagonal_echo(&[0.5, 0.0, 0.5], 0.3);
/// assert!((s - (0.5 + 0.5 * (0.6f64).cos().powi(2))).abs() < 1e-15);
/// ```
pub fn diagonal_echo(hist: &[f64], delta: f64) -> f64 {
    let c = (2.0 * delta).cos();
    let mut weight = 1.0;
    let mut num = 0.0;
    for &w in hist {
        num += w * weight;
        weight *= c;
    }
    num / hist.iter().sum::<f64>()
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
    /// `G_q = Z_q` anticommutes with `P` iff `P` has an `X` or `Y` on `q` (its x-bit), `G_q = X_q` iff a `Y` or `Z` (its z-bit).
    /// Feed it to [`diagonal_echo`]; histograms of disjoint shares of a sum add.
    ///
    /// # Panics
    ///
    /// If a site is repeated or is not below [`num_qubits`](Self::num_qubits).
    ///
    /// ```
    /// use paulistrings::{BuildAccumulator, PauliString, Phase, RotationAxis};
    /// use num_complex::Complex64;
    ///
    /// let mut acc = BuildAccumulator::<1>::new(2);
    /// acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(0.6, 0.0));
    /// acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(0.8, 0.0));
    /// let w = acc.finalize().anticommute_histogram(&[0, 1], RotationAxis::Z);
    /// assert!((w[0] - 0.64).abs() < 1e-15 && (w[1] - 0.36).abs() < 1e-15 && w[2] == 0.0);
    /// ```
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
    /// Conjugation by `exp(-iδ Z)` fixes `I` and `Z` and rotates `X → cos 2δ X − sin 2δ Y`, `Y → cos 2δ Y + sin 2δ X` (for `X` read `Z → cos 2δ Z + sin 2δ Y`, `Y → cos 2δ Y − sin 2δ Z`), so `V† Q V` only flips the other half's bit on the sites `K` where `Q` anticommutes with `G`.
    /// Strings therefore fall into classes that agree off those bits, and `S_δ = Σ_class Σ_{P,Q} c̄_P c_Q Π_{q∈K} R_q[P_q, Q_q]`; the class-diagonal part is what [`anticommute_histogram`](Self::anticommute_histogram) keeps.
    ///
    /// The value is the real part of `2⁻ⁿ Tr(A† V† A V)`, which is `S_δ` itself for a Hermitian `A` (real coefficients in the crate's convention).
    ///
    /// # Cost
    ///
    /// A copy of every term with an `X`/`Y` (`Z` axis) or `Y`/`Z` (`X` axis) on a site, a parallel sort of those copies, and `O(k²)` work per class of `k` strings, so a class of many strings sharing everything off the sites dominates.
    ///
    /// # Panics
    ///
    /// If a site is repeated or is not below [`num_qubits`](Self::num_qubits).
    ///
    /// ```
    /// use paulistrings::{BuildAccumulator, PauliString, Phase, RotationAxis};
    /// use num_complex::Complex64;
    ///
    /// // Tr(X e^{iδZ} X e^{-iδZ}) / 2 = cos 2δ.
    /// let mut acc = BuildAccumulator::<1>::new(1);
    /// acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(1.0, 0.0));
    /// let s = acc.finalize().rotated_overlap(&[0], 0.3, RotationAxis::Z);
    /// assert!((s - (0.6f64).cos()).abs() < 1e-15);
    /// ```
    pub fn rotated_overlap(&self, sites: &[usize], delta: f64, axis: RotationAxis) -> f64 {
        self.rotated_overlap_complex(sites, delta, axis).re
    }

    /// `2⁻ⁿ Tr(A† V† A V)` in full, the sign convention included; [`Self::rotated_overlap`] is its real part.
    pub(crate) fn rotated_overlap_complex(
        &self,
        sites: &[usize],
        delta: f64,
        axis: RotationAxis,
    ) -> Complex64 {
        let mask = site_mask::<W>(sites, self.num_qubits());
        let (s, c) = (2.0 * delta).sin_cos();
        let cos_pow: Vec<f64> = (0..=sites.len()).map(|k| c.powi(k as i32)).collect();
        let sin_pow: Vec<f64> = (0..=sites.len()).map(|k| s.powi(k as i32)).collect();

        // A string commuting with every generator is its own class, and V leaves it alone.
        let (fixed, members): (Vec<f64>, Vec<Vec<Member<W>>>) = (0..self.num_buckets())
            .into_par_iter()
            .fold(
                || (0.0, Vec::new()),
                |(mut fixed, mut members), b| {
                    let (xs, zs, cs) = self.bucket(b);
                    for ((x, z), &coeff) in xs.iter().zip(zs).zip(cs) {
                        let (sel, flip) = axis.split(x, z);
                        let k = and(sel, &mask);
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
                let mut acc = Complex64::new(0.0, 0.0);
                for p in class {
                    for q in class {
                        let d: [u64; W] = std::array::from_fn(|i| p.flip[i] ^ q.flip[i]);
                        let nd = popcount(&d);
                        // R[P, Q] is `−sin 2δ` where P holds Y against Q's X (Z axis), or P holds Z against Q's Y (X axis).
                        let negative = match axis {
                            RotationAxis::Z => popcount(&and(&p.flip, &d)),
                            RotationAxis::X => popcount(&and(&q.flip, &d)),
                        } % 2
                            == 1;
                        let m = cos_pow[sites_k - nd] * sin_pow[nd];
                        acc += p.coeff.conj() * q.coeff * if negative { -m } else { m };
                    }
                }
                acc
            })
            .sum();

        mixed + fixed.iter().sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::rotation::PauliRotation;
    use crate::circuit::Circuit;
    use crate::engine::{propagate, Direction};
    use crate::pauli_string::PauliString;
    use crate::test_support::{rand_sum_on, KeepAll};
    use proptest::prelude::*;

    const DELTA: f64 = 0.3;

    /// Label index `i` is qubit `i`.
    fn sum_of<const W: usize>(terms: &[(&str, f64)]) -> PauliSum<W> {
        let terms: Vec<(&str, Complex64)> = terms
            .iter()
            .map(|&(l, c)| (l, Complex64::new(c, 0.0)))
            .collect();
        PauliSum::<W>::from_strings(&terms)
    }

    /// `2⁻ⁿ Tr(A† V† A V)` by brute force: `V† A V` materialized through `propagate` of `exp(-i (2δ) G_q / 2)` rotations in the Heisenberg picture, then [`PauliSum::overlap`].
    fn materialized<const W: usize>(
        a: &PauliSum<W>,
        sites: &[usize],
        delta: f64,
        axis: RotationAxis,
    ) -> Complex64 {
        let mut circuit = Circuit::<W>::new(a.num_qubits());
        for &q in sites {
            let g = match axis {
                RotationAxis::Z => PauliString::<W>::z(q as u32),
                RotationAxis::X => PauliString::<W>::x(q as u32),
            };
            circuit.push(PauliRotation::new(g, 2.0 * delta));
        }
        let b = propagate(&circuit, a.clone(), &KeepAll, Direction::Heisenberg);
        a.clone().with_hash(b.hash().clone()).overlap(&b)
    }

    #[test]
    fn single_qubit_hand_values() {
        let c = (2.0 * DELTA).cos();
        for (label, axis, sites, want) in [
            ("X", RotationAxis::Z, 0, c),
            ("Y", RotationAxis::Z, 0, c),
            ("Z", RotationAxis::Z, 0, 1.0),
            ("Z", RotationAxis::X, 0, c),
            ("Y", RotationAxis::X, 0, c),
            ("X", RotationAxis::X, 0, 1.0),
            // A generator off the string's support is invisible.
            ("XI", RotationAxis::Z, 1, 1.0),
        ] {
            let got = sum_of::<1>(&[(label, 1.0)]).rotated_overlap(&[sites], DELTA, axis);
            assert!(
                (got - want).abs() < 1e-15,
                "{label} {axis:?}: {got} vs {want}"
            );
        }
    }

    /// `XX` and `YY` form one class on sites `{0, 1}` (Z axis), with `R[XX, YY] = R[YY, XX] = sin² 2δ`, so `a XX + b YY` gives `(a² + b²) cos² 2δ + 2ab sin² 2δ`.
    /// `XX + YY` commutes with `Z ⊗ Z` rotations and returns its norm `2`; `XY − YX` does too, through the negative `R` entries.
    /// On the X axis `ZZ` and `YY` share a class, as do `ZY` and `YZ`.
    #[test]
    fn two_qubit_class_hand_values() {
        let (s, c) = (2.0 * DELTA).sin_cos();
        let (z, x) = (RotationAxis::Z, RotationAxis::X);
        let cases = [
            ([("XX", 0.7), ("YY", -0.4)], z, 0.65 * c * c - 0.56 * s * s),
            ([("XX", 1.0), ("YY", 1.0)], z, 2.0),
            ([("XY", 1.0), ("YX", -1.0)], z, 2.0),
            ([("XX", 1.0), ("YY", -1.0)], z, 2.0 * (4.0 * DELTA).cos()),
            ([("ZZ", 1.0), ("YY", 1.0)], x, 2.0),
            ([("ZY", 1.0), ("YZ", -1.0)], x, 2.0),
            ([("ZY", 1.0), ("YZ", 1.0)], x, 2.0 * (4.0 * DELTA).cos()),
        ];
        for (terms, axis, want) in cases {
            let got = sum_of::<1>(&terms).rotated_overlap(&[0, 1], DELTA, axis);
            assert!(
                (got - want).abs() < 1e-14,
                "{terms:?} {axis:?}: {got} vs {want}"
            );
        }
    }

    /// Complex coefficients so the sign convention of every off-diagonal entry shows in the imaginary part too.
    fn check_against_materialized<const W: usize>(num_qubits: usize, window: &[u32]) {
        for (seed, sites) in [
            (1u64, vec![window[0] as usize, window[2] as usize]),
            (2, window.iter().map(|&q| q as usize).collect::<Vec<_>>()),
            (3, vec![window[1] as usize]),
        ] {
            let a = rand_sum_on::<W>(300, num_qubits, window, seed);
            for axis in [RotationAxis::Z, RotationAxis::X] {
                let want = materialized(&a, &sites, DELTA, axis);
                let got = a.rotated_overlap_complex(&sites, DELTA, axis);
                assert!(
                    (got - want).norm() < 1e-10 * want.norm().max(1.0),
                    "W={W} seed {seed} {axis:?} sites {sites:?}: {got} vs {want}",
                );
            }
        }
    }

    #[test]
    fn agrees_with_materializing_the_rotation() {
        check_against_materialized::<1>(8, &[1, 2, 3, 4, 6]);
        check_against_materialized::<2>(70, &[62, 63, 64, 65, 67]);
    }

    /// With no class of two, the overlap is the histogram's diagonal sum: `aX0 + bZ0 + cY0Y1 + dI` on sites `{0, 1}`.
    #[test]
    fn histogram_hand_values_and_the_diagonal_limit() {
        let (a, b, c, d) = (0.5, -0.25, 0.75, 0.125);
        let sum = sum_of::<1>(&[("XI", a), ("ZI", b), ("YY", c), ("II", d)]);
        let w = sum.anticommute_histogram(&[0, 1], RotationAxis::Z);
        assert_eq!(w, vec![b * b + d * d, a * a, c * c]);
        let w = sum.anticommute_histogram(&[0, 1], RotationAxis::X);
        assert_eq!(w, vec![a * a + d * d, b * b, c * c]);

        for axis in [RotationAxis::Z, RotationAxis::X] {
            let w = sum.anticommute_histogram(&[0, 1], axis);
            let norm: f64 = w.iter().sum();
            let diag = diagonal_echo(&w, DELTA) * norm;
            let exact = sum.rotated_overlap(&[0, 1], DELTA, axis);
            assert!((diag - exact).abs() < 1e-15, "{axis:?}: {diag} vs {exact}");
        }
        // X_q under Z rotations keeps cos 2δ of itself.
        let w = sum_of::<1>(&[("X", 1.0)]).anticommute_histogram(&[0], RotationAxis::Z);
        assert!((diagonal_echo(&w, DELTA) - (2.0 * DELTA).cos()).abs() < 1e-15);
    }

    /// Bucketing is invisible to both read-outs, at both widths.
    #[test]
    fn read_outs_are_partition_independent() {
        use crate::pauli_sum::Gf2Hash;
        let a = rand_sum_on::<2>(400, 70, &[0, 5, 63, 64, 66], 9);
        let sites = [5usize, 63, 64];
        let spread = a.clone().with_hash(Gf2Hash::new(70, 4, 0xABC));
        assert!(spread.num_buckets() > 1);
        for axis in [RotationAxis::Z, RotationAxis::X] {
            let (x, y) = (
                a.rotated_overlap(&sites, DELTA, axis),
                spread.rotated_overlap(&sites, DELTA, axis),
            );
            assert!((x - y).abs() < 1e-12, "{axis:?}: {x} vs {y}");
            let (hx, hy) = (
                a.anticommute_histogram(&sites, axis),
                spread.anticommute_histogram(&sites, axis),
            );
            for (u, v) in hx.iter().zip(&hy) {
                assert!((u - v).abs() < 1e-12, "{axis:?}: {hx:?} vs {hy:?}");
            }
        }
    }

    #[test]
    fn flip_mask_names_the_other_half_of_the_sites() {
        assert_eq!(
            RotationAxis::Z.flip_mask::<2>(&[1, 64]),
            ([0, 0], [0b10, 1])
        );
        assert_eq!(RotationAxis::X.flip_mask::<1>(&[0, 3]), ([0b1001], [0]));
    }

    #[test]
    #[should_panic(expected = "listed twice")]
    fn a_repeated_site_is_rejected() {
        sum_of::<1>(&[("XX", 1.0)]).rotated_overlap(&[1, 1], DELTA, RotationAxis::Z);
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn an_out_of_range_site_is_rejected() {
        sum_of::<1>(&[("XX", 1.0)]).anticommute_histogram(&[2], RotationAxis::X);
    }

    proptest! {
        /// At `δ = 0`, `V` is the identity and the echo is `Σ|c|²`.
        #[test]
        fn zero_angle_is_the_squared_norm(seed in 0u64..1000, z_axis in any::<bool>()) {
            let a = rand_sum_on::<1>(120, 8, &[0, 1, 2, 5, 7], seed);
            let axis = if z_axis { RotationAxis::Z } else { RotationAxis::X };
            let norm: f64 = a.iter().map(|(_, _, c)| c.norm_sqr()).sum();
            let got = a.rotated_overlap(&[0, 2, 5, 7], 0.0, axis);
            prop_assert!((got - norm).abs() < 1e-12 * norm.max(1.0), "{} vs {}", got, norm);
        }
    }
}
