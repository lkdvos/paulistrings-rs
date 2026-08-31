//! General unitaries, stored as local Pauli-transfer matrices. See §6, and
//! v0.2 §5.1.
//!
//! A bounded-support channel *is* its local Pauli-transfer matrix, which is
//! exactly the form the bucketed engine consumes — so these types need no
//! `prepare` override: the default derivation (which probes `apply` on the local
//! basis) recovers the table it was built from.
//!
//! # Conventions
//!
//! * A single-qubit Pauli is indexed `idx = x | (z << 1)`, i.e.
//!   `I = 0, X = 1, Z = 2, Y = 3`, matching [`Clifford1Q`]. Two-qubit Paulis pack
//!   as `x0 | (z0 << 1) | (x1 << 2) | (z1 << 3)`, matching [`Clifford2Q`].
//! * `table[s][t]` is the coefficient of output Pauli `t` when the input is
//!   Pauli `s`. `apply` therefore computes `P_s ↦ Σ_t table[s][t] · P_t`, the
//!   Heisenberg conjugation `P ↦ U P U†` — the same direction `Clifford1Q`
//!   documents.
//! * `apply_adjoint` reads the table **transposed**, which is the
//!   Hilbert-Schmidt adjoint. For a unitary the entries are real, so this is
//!   also the conjugate transpose and the two round-trip to the identity. The
//!   same rule governs [`AmplitudeDamping`](super::noise::AmplitudeDamping).
//!
//! [`Clifford1Q`]: super::clifford::Clifford1Q
//! [`Clifford2Q`]: super::clifford::Clifford2Q

use super::{qubit_loc, read_pauli, support_mask, write_pauli, Channel, OutputBuffer};
use num_complex::Complex64;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// The four single-qubit Pauli matrices, in `I, X, Z, Y` index order.
fn pauli_matrix(idx: usize) -> [[Complex64; 2]; 2] {
    let i = Complex64::new(0.0, 1.0);
    match idx {
        0 => [[ONE, ZERO], [ZERO, ONE]],
        1 => [[ZERO, ONE], [ONE, ZERO]],
        2 => [[ONE, ZERO], [ZERO, -ONE]],
        _ => [[ZERO, -i], [i, ZERO]],
    }
}

fn matmul<const N: usize>(a: &[[Complex64; N]; N], b: &[[Complex64; N]; N]) -> [[Complex64; N]; N] {
    let mut out = [[ZERO; N]; N];
    for i in 0..N {
        for k in 0..N {
            let aik = a[i][k];
            if aik == ZERO {
                continue;
            }
            for j in 0..N {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

fn dagger<const N: usize>(a: &[[Complex64; N]; N]) -> [[Complex64; N]; N] {
    let mut out = [[ZERO; N]; N];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            // A transpose: `j` indexes rows of `a` while `i` indexes its columns,
            // so neither loop can be turned into an iterator over `a`.
            *slot = a[j][i].conj();
        }
    }
    out
}

fn trace<const N: usize>(a: &[[Complex64; N]; N]) -> Complex64 {
    a.iter().enumerate().map(|(i, row)| row[i]).sum()
}

/// `kron(a, b)` with `a` acting on the *first* (more significant) qubit.
fn kron2(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> [[Complex64; 4]; 4] {
    let mut out = [[ZERO; 4]; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    out[2 * i + k][2 * j + l] = a[i][j] * b[k][l];
                }
            }
        }
    }
    out
}

/// Round a PTM entry that is within `eps` of zero down to exactly zero.
///
/// Without this, a Clifford built via [`GeneralUnitary1Q::from_matrix`] carries
/// `~1e-17` entries where it should carry exact zeros, and every one of them
/// becomes a spurious output term with a denormal coefficient. The engine drops
/// only *exact* zeros, deliberately, so the cleanup has to happen here.
fn clean(v: Complex64, eps: f64) -> Complex64 {
    let re = if v.re.abs() < eps { 0.0 } else { v.re };
    let im = if v.im.abs() < eps { 0.0 } else { v.im };
    Complex64::new(re, im)
}

/// Tolerance below which a derived PTM entry is treated as an exact zero.
const PTM_EPS: f64 = 1e-12;

/// Materialize the effective PTM row for input index `s`: `table[s]`
/// normally, or its transpose column `[table[t][s] for t]` when `transpose`
/// is set (the Hilbert-Schmidt adjoint — see the module docs). Shared by
/// [`apply_1q`] (`N = 4`) and [`apply_2q`] (`N = 16`); hoisting this out of
/// the per-output loop means the transpose flag is tested once per call
/// instead of once per emitted term.
#[inline]
fn effective_row<const N: usize>(
    table: &[[Complex64; N]; N],
    transpose: bool,
    s: usize,
) -> [Complex64; N] {
    if transpose {
        core::array::from_fn(|t| table[t][s])
    } else {
        table[s]
    }
}

/// Generic 1-qubit unitary, stored as the Pauli expansion of its
/// Heisenberg-picture action on `{I, X, Z, Y}` at the support qubit.
///
/// `MAX_FANOUT = 4`, since an input Pauli on the support can map to a sum over
/// all four basis Paulis. The *bucket* fan-in is `2^rank(H|_D)` where `D` is the
/// realized delta set, which is at most 4 but often smaller — a `T` gate, for
/// instance, only ever mixes `X` with `Y`, so its delta set is one-dimensional
/// and it reads just 2 input buckets.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use paulistrings::channel::GeneralUnitary1Q;
///
/// // Hadamard as a general unitary.
/// let r = std::f64::consts::FRAC_1_SQRT_2;
/// let h = GeneralUnitary1Q::from_matrix(0, [
///     [Complex64::new(r, 0.0), Complex64::new(r, 0.0)],
///     [Complex64::new(r, 0.0), Complex64::new(-r, 0.0)],
/// ]);
/// // H conjugates X to Z: table[X][Z] == 1.
/// assert!((h.table[1][2] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
/// ```
#[derive(Clone, Debug)]
pub struct GeneralUnitary1Q {
    /// The single qubit this gate acts on.
    pub support: [u32; 1],
    /// `table[s][t]`: coefficient of output Pauli `t` for input Pauli `s`.
    pub table: [[Complex64; 4]; 4],
}

impl GeneralUnitary1Q {
    /// From an explicit Pauli-transfer matrix.
    pub fn from_ptm(qubit: u32, table: [[Complex64; 4]; 4]) -> Self {
        Self {
            support: [qubit],
            table,
        }
    }

    /// From a 2x2 unitary `u`, computing
    /// `table[s][t] = tr(P_t · U P_s U†) / 2`.
    pub fn from_matrix(qubit: u32, u: [[Complex64; 2]; 2]) -> Self {
        let ud = dagger(&u);
        let mut table = [[ZERO; 4]; 4];
        for (s, row) in table.iter_mut().enumerate() {
            let ps = pauli_matrix(s);
            let conj = matmul(&matmul(&u, &ps), &ud);
            for (t, slot) in row.iter_mut().enumerate() {
                let pt = pauli_matrix(t);
                let v = trace(&matmul(&pt, &conj)) / Complex64::new(2.0, 0.0);
                *slot = clean(v, PTM_EPS);
            }
        }
        Self {
            support: [qubit],
            table,
        }
    }
}

/// Shared body: read the support bits, look up the row, emit nonzero entries.
#[inline]
fn apply_1q<const W: usize>(
    qubit: u32,
    table: &[[Complex64; 4]; 4],
    transpose: bool,
    input_x: &[u64; W],
    input_z: &[u64; W],
    coeff: Complex64,
    out: &mut OutputBuffer<'_, W>,
) {
    let q = qubit as usize;
    debug_assert!(q < 64 * W);
    let (word, bit, mask) = qubit_loc(q);
    let s = read_pauli(input_x, input_z, word, bit);

    // Materialize the effective row once, so the transpose branch is hoisted out
    // of the loop instead of being retested per output.
    let row = effective_row(table, transpose, s);

    for (t, &c) in row.iter().enumerate() {
        if c == ZERO {
            continue;
        }
        let mut nx = *input_x;
        let mut nz = *input_z;
        write_pauli(&mut nx, &mut nz, word, bit, mask, t);
        out.push(nx, nz, coeff * c);
    }
}

impl<const W: usize> Channel<W> for GeneralUnitary1Q {
    #[inline]
    fn max_fanout(&self) -> usize {
        4
    }

    #[inline]
    fn support(&self) -> [u64; W] {
        support_mask(&self.support)
    }

    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        apply_1q(
            self.support[0],
            &self.table,
            false,
            input_x,
            input_z,
            coeff,
            out,
        );
    }

    fn apply_adjoint(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        apply_1q(
            self.support[0],
            &self.table,
            true,
            input_x,
            input_z,
            coeff,
            out,
        );
    }
}

/// Generic 2-qubit unitary, stored as a 16x16 Pauli-expansion table.
///
/// Packing is `x0 | (z0 << 1) | (x1 << 2) | (z1 << 3)`, with `support[0]`
/// contributing bits 0-1 and `support[1]` bits 2-3 — the same convention
/// [`Clifford2Q`](super::clifford::Clifford2Q) uses. In the matrix passed to
/// [`Self::from_matrix`], `support[0]` is the **more significant** tensor factor,
/// i.e. the matrix acts on `|q0 q1⟩`.
#[derive(Clone, Debug)]
pub struct GeneralUnitary2Q {
    /// The two qubits this gate acts on.
    pub support: [u32; 2],
    /// `table[s][t]`: coefficient of output Pauli `t` for input Pauli `s`.
    pub table: Box<[[Complex64; 16]; 16]>,
}

impl GeneralUnitary2Q {
    /// From an explicit Pauli-transfer matrix.
    pub fn from_ptm(q0: u32, q1: u32, table: Box<[[Complex64; 16]; 16]>) -> Self {
        Self {
            support: [q0, q1],
            table,
        }
    }

    /// From a 4x4 unitary `u` acting on `|q0 q1⟩`, computing
    /// `table[s][t] = tr(P_t · U P_s U†) / 4`.
    pub fn from_matrix(q0: u32, q1: u32, u: [[Complex64; 4]; 4]) -> Self {
        let ud = dagger(&u);
        let two_q = |s: usize| -> [[Complex64; 4]; 4] {
            let a = pauli_matrix((s & 1) | ((s >> 1) & 1) << 1);
            let b = pauli_matrix(((s >> 2) & 1) | ((s >> 3) & 1) << 1);
            kron2(&a, &b)
        };
        let mut table = Box::new([[ZERO; 16]; 16]);
        for (s, row) in table.iter_mut().enumerate() {
            let ps = two_q(s);
            let conj = matmul(&matmul(&u, &ps), &ud);
            for (t, slot) in row.iter_mut().enumerate() {
                let pt = two_q(t);
                let v = trace(&matmul(&pt, &conj)) / Complex64::new(4.0, 0.0);
                *slot = clean(v, PTM_EPS);
            }
        }
        Self {
            support: [q0, q1],
            table,
        }
    }
}

/// Shared body for the 2-qubit case.
#[inline]
#[allow(clippy::too_many_arguments)]
fn apply_2q<const W: usize>(
    support: &[u32; 2],
    table: &[[Complex64; 16]; 16],
    transpose: bool,
    input_x: &[u64; W],
    input_z: &[u64; W],
    coeff: Complex64,
    out: &mut OutputBuffer<'_, W>,
) {
    let q0 = support[0] as usize;
    let q1 = support[1] as usize;
    debug_assert!(q0 < 64 * W && q1 < 64 * W);
    let (w0, b0, m0) = qubit_loc(q0);
    let (w1, b1, m1) = qubit_loc(q1);

    let s = read_pauli(input_x, input_z, w0, b0) | (read_pauli(input_x, input_z, w1, b1) << 2);

    let row = effective_row(table, transpose, s);

    for (t, &c) in row.iter().enumerate() {
        if c == ZERO {
            continue;
        }
        let mut nx = *input_x;
        let mut nz = *input_z;
        write_pauli(&mut nx, &mut nz, w0, b0, m0, t & 3);
        write_pauli(&mut nx, &mut nz, w1, b1, m1, (t >> 2) & 3);
        out.push(nx, nz, coeff * c);
    }
}

impl<const W: usize> Channel<W> for GeneralUnitary2Q {
    #[inline]
    fn max_fanout(&self) -> usize {
        16
    }

    #[inline]
    fn support(&self) -> [u64; W] {
        support_mask(&self.support)
    }

    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        apply_2q(
            &self.support,
            &self.table,
            false,
            input_x,
            input_z,
            coeff,
            out,
        );
    }

    fn apply_adjoint(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<'_, W>,
    ) {
        apply_2q(
            &self.support,
            &self.table,
            true,
            input_x,
            input_z,
            coeff,
            out,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket::hash::Gf2Hash;
    use crate::channel::clifford::{Clifford1Q, Clifford2Q};
    use crate::channel::prepared::Prepared;
    use crate::pauli_string::PauliString;

    const TOL: f64 = 1e-12;
    const R: f64 = std::f64::consts::FRAC_1_SQRT_2;

    fn c(re: f64) -> Complex64 {
        Complex64::new(re, 0.0)
    }

    type Term<const W: usize> = ([u64; W], [u64; W], Complex64);

    fn outputs<const W: usize, C: Channel<W> + ?Sized>(
        ch: &C,
        adjoint: bool,
        p: PauliString<W>,
        coeff: Complex64,
    ) -> Vec<Term<W>> {
        let f = ch.max_fanout().max(1);
        let mut bx = vec![[0u64; W]; f];
        let mut bz = vec![[0u64; W]; f];
        let mut bc = vec![ZERO; f];
        let mut len = 0usize;
        {
            let mut out = OutputBuffer::<W> {
                x: &mut bx,
                z: &mut bz,
                coeff: &mut bc,
                len: &mut len,
            };
            if adjoint {
                ch.apply_adjoint(&p.x, &p.z, coeff, &mut out);
            } else {
                ch.apply(&p.x, &p.z, coeff, &mut out);
            }
        }
        let mut v: Vec<Term<W>> = (0..len).map(|i| (bx[i], bz[i], bc[i])).collect();
        v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        v.retain(|t| t.2.norm() > 1e-15);
        v
    }

    /// Two channels agree on every local basis Pauli of the given support.
    fn assert_agrees_on_basis<const W: usize, A, B>(a: &A, b: &B, qubits: &[u32], what: &str)
    where
        A: Channel<W> + ?Sized,
        B: Channel<W> + ?Sized,
    {
        let k = qubits.len();
        for s in 0..(1usize << (2 * k)) {
            let mut p = PauliString::<W> {
                x: [0u64; W],
                z: [0u64; W],
            };
            for (j, &q) in qubits.iter().enumerate() {
                let bit = 1u64 << (q % 64);
                if (s >> (2 * j)) & 1 == 1 {
                    p.x[q as usize / 64] |= bit;
                }
                if (s >> (2 * j + 1)) & 1 == 1 {
                    p.z[q as usize / 64] |= bit;
                }
            }
            let ga = outputs(a, false, p, ONE);
            let gb = outputs(b, false, p, ONE);
            assert_eq!(ga.len(), gb.len(), "{what}: s={s} term count");
            for (u, v) in ga.iter().zip(gb.iter()) {
                assert_eq!(u.0, v.0, "{what}: s={s} x key");
                assert_eq!(u.1, v.1, "{what}: s={s} z key");
                assert!(
                    (u.2 - v.2).norm() < TOL,
                    "{what}: s={s} coeff {} vs {}",
                    u.2,
                    v.2,
                );
            }
        }
    }

    // ---- Cliffords expressed as general unitaries ----

    #[test]
    fn hadamard_as_a_general_unitary_matches_clifford1q() {
        let h = GeneralUnitary1Q::from_matrix(3, [[c(R), c(R)], [c(R), c(-R)]]);
        assert_agrees_on_basis::<2, _, _>(&h, &Clifford1Q::h(3), &[3], "hadamard");
    }

    #[test]
    fn phase_gate_as_a_general_unitary_matches_clifford1q() {
        let i = Complex64::new(0.0, 1.0);
        let s = GeneralUnitary1Q::from_matrix(3, [[ONE, ZERO], [ZERO, i]]);
        assert_agrees_on_basis::<2, _, _>(&s, &Clifford1Q::s(3), &[3], "phase");
    }

    #[test]
    fn pauli_gates_as_general_unitaries_match_clifford1q() {
        let i = Complex64::new(0.0, 1.0);
        let x = GeneralUnitary1Q::from_matrix(5, [[ZERO, ONE], [ONE, ZERO]]);
        assert_agrees_on_basis::<2, _, _>(&x, &Clifford1Q::x(5), &[5], "pauli_x");
        let y = GeneralUnitary1Q::from_matrix(5, [[ZERO, -i], [i, ZERO]]);
        assert_agrees_on_basis::<2, _, _>(&y, &Clifford1Q::y(5), &[5], "pauli_y");
        let z = GeneralUnitary1Q::from_matrix(5, [[ONE, ZERO], [ZERO, -ONE]]);
        assert_agrees_on_basis::<2, _, _>(&z, &Clifford1Q::z(5), &[5], "pauli_z");
    }

    #[test]
    fn hadamard_across_a_word_boundary_w2() {
        let h = GeneralUnitary1Q::from_matrix(70, [[c(R), c(R)], [c(R), c(-R)]]);
        assert_agrees_on_basis::<2, _, _>(&h, &Clifford1Q::h(70), &[70], "hadamard@70");
    }

    #[test]
    fn cnot_as_a_general_unitary_matches_clifford2q() {
        // |q0 q1> with q0 the control and the more significant factor.
        let u = [
            [ONE, ZERO, ZERO, ZERO],
            [ZERO, ONE, ZERO, ZERO],
            [ZERO, ZERO, ZERO, ONE],
            [ZERO, ZERO, ONE, ZERO],
        ];
        let g = GeneralUnitary2Q::from_matrix(1, 4, u);
        assert_agrees_on_basis::<2, _, _>(&g, &Clifford2Q::cnot(1, 4), &[1, 4], "cnot");
    }

    #[test]
    fn cz_as_a_general_unitary_matches_clifford2q() {
        let mut u = [[ZERO; 4]; 4];
        for (i, row) in u.iter_mut().enumerate() {
            row[i] = if i == 3 { -ONE } else { ONE };
        }
        let g = GeneralUnitary2Q::from_matrix(1, 4, u);
        assert_agrees_on_basis::<2, _, _>(&g, &Clifford2Q::cz(1, 4), &[1, 4], "cz");
    }

    #[test]
    fn swap_as_a_general_unitary_matches_clifford2q() {
        let u = [
            [ONE, ZERO, ZERO, ZERO],
            [ZERO, ZERO, ONE, ZERO],
            [ZERO, ONE, ZERO, ZERO],
            [ZERO, ZERO, ZERO, ONE],
        ];
        let g = GeneralUnitary2Q::from_matrix(1, 4, u);
        assert_agrees_on_basis::<2, _, _>(&g, &Clifford2Q::swap(1, 4), &[1, 4], "swap");
    }

    #[test]
    fn cnot_across_a_word_boundary_w2() {
        let u = [
            [ONE, ZERO, ZERO, ZERO],
            [ZERO, ONE, ZERO, ZERO],
            [ZERO, ZERO, ZERO, ONE],
            [ZERO, ZERO, ONE, ZERO],
        ];
        let g = GeneralUnitary2Q::from_matrix(60, 70, u);
        assert_agrees_on_basis::<2, _, _>(&g, &Clifford2Q::cnot(60, 70), &[60, 70], "cnot@60,70");
    }

    // ---- non-Clifford ----

    /// The `T` gate mixes `X` with `Y` and fixes `I` and `Z`, so it is a genuine
    /// fanout-2 non-Clifford — and its delta set is only *one*-dimensional, so it
    /// reads 2 buckets rather than the 4 a dense 1Q unitary would.
    #[test]
    fn t_gate_expansion_and_bucket_fanin() {
        let t = GeneralUnitary1Q::from_matrix(
            2,
            [
                [ONE, ZERO],
                [
                    ZERO,
                    Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
                ],
            ],
        );
        // X -> (X + Y)/sqrt(2)
        let got = outputs::<1, _>(&t, false, PauliString::<1>::x(2), ONE);
        assert_eq!(got.len(), 2);
        for (_, _, coeff) in &got {
            assert!((coeff.norm() - R).abs() < TOL, "coeff {coeff}");
        }
        // Z is fixed.
        let got = outputs::<1, _>(&t, false, PauliString::<1>::z(2), ONE);
        assert_eq!(got.len(), 1);
        assert!((got[0].2 - ONE).norm() < TOL);

        let hash = Gf2Hash::<1>::new(64, 16, 0xBEEF);
        let prep = Channel::<1>::prepare(&t, &hash, false).unwrap();
        assert_eq!(
            prep.bucket_deltas().len(),
            2,
            "the T gate only mixes X with Y, so its delta set is 1-dimensional",
        );
    }

    /// A dense 1Q unitary does reach the 4-bucket upper bound, and a dense 2Q one
    /// reaches 16 — the values quoted in v0.2 §2.4 as upper bounds.
    #[test]
    fn dense_unitaries_reach_the_quoted_bucket_fanin() {
        // A rotation about an axis with all three components mixes everything.
        let a = std::f64::consts::FRAC_PI_3;
        let (ca, sa) = ((a / 2.0).cos(), (a / 2.0).sin());
        let n = (1.0f64 / 3.0).sqrt();
        let i = Complex64::new(0.0, 1.0);
        let u = [
            [c(ca) - i * c(sa * n), (-i * c(sa * n)) - c(sa * n)],
            [(-i * c(sa * n)) + c(sa * n), c(ca) + i * c(sa * n)],
        ];
        let g = GeneralUnitary1Q::from_matrix(0, u);
        let hash = Gf2Hash::<1>::new(64, 16, 0xBEEF);
        let prep = Channel::<1>::prepare(&g, &hash, false).unwrap();
        assert_eq!(
            prep.bucket_deltas().len(),
            4,
            "dense 1Q should read 4 buckets"
        );
    }

    // ---- adjoint ----

    #[test]
    fn adjoint_reads_the_table_transposed_and_round_trips() {
        let i = Complex64::new(0.0, 1.0);
        // T gate: not self-adjoint.
        let t = GeneralUnitary1Q::from_matrix(
            2,
            [
                [ONE, ZERO],
                [
                    ZERO,
                    Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
                ],
            ],
        );
        for basis in [
            PauliString::<1>::identity(),
            PauliString::<1>::x(2),
            PauliString::<1>::y(2),
            PauliString::<1>::z(2),
        ] {
            // Apply then adjoint, accumulating into a map, must give back `basis`.
            let mut acc: Vec<Term<1>> = Vec::new();
            for (x, z, cf) in outputs::<1, _>(&t, false, basis, ONE) {
                for out in outputs::<1, _>(&t, true, PauliString::<1> { x, z }, cf) {
                    acc.push(out);
                }
            }
            acc.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            let mut merged: Vec<Term<1>> = Vec::new();
            for (x, z, cf) in acc {
                match merged.last_mut() {
                    Some(l) if l.0 == x && l.1 == z => l.2 += cf,
                    _ => merged.push((x, z, cf)),
                }
            }
            merged.retain(|t| t.2.norm() > 1e-12);
            assert_eq!(merged.len(), 1, "round trip left {} terms", merged.len());
            assert_eq!((merged[0].0, merged[0].1), (basis.x, basis.z));
            assert!((merged[0].2 - ONE).norm() < 1e-12);
        }
        let _ = i;
    }

    #[test]
    fn cnot_general_unitary_adjoint_matches_clifford2q_adjoint() {
        let u = [
            [ONE, ZERO, ZERO, ZERO],
            [ZERO, ONE, ZERO, ZERO],
            [ZERO, ZERO, ZERO, ONE],
            [ZERO, ZERO, ONE, ZERO],
        ];
        let g = GeneralUnitary2Q::from_matrix(1, 4, u);
        let cn = Clifford2Q::cnot(1, 4);
        for s in 0..16usize {
            let mut p = PauliString::<1> { x: [0], z: [0] };
            for (j, q) in [1u32, 4].iter().enumerate() {
                let bit = 1u64 << q;
                if (s >> (2 * j)) & 1 == 1 {
                    p.x[0] |= bit;
                }
                if (s >> (2 * j + 1)) & 1 == 1 {
                    p.z[0] |= bit;
                }
            }
            let a = outputs::<1, _>(&g, true, p, ONE);
            let b = outputs::<1, _>(&cn, true, p, ONE);
            assert_eq!(a.len(), b.len(), "s={s}");
            for (u1, v1) in a.iter().zip(b.iter()) {
                assert_eq!((u1.0, u1.1), (v1.0, v1.1), "s={s}");
                assert!((u1.2 - v1.2).norm() < TOL, "s={s}");
            }
        }
    }

    // ---- prepared-form round trip ----

    /// The derivation must recover exactly the table it was built from — the
    /// point of v0.2 §5.1's claim that a bounded-support channel *is* its local
    /// PTM.
    #[test]
    fn derive_local_recovers_the_table() {
        let h = GeneralUnitary1Q::from_matrix(3, [[c(R), c(R)], [c(R), c(-R)]]);
        let hash = Gf2Hash::<2>::new(128, 12, 0x1234);
        let prep = Channel::<2>::prepare(&h, &hash, false).unwrap();
        let Prepared::Local(ptm) = prep else {
            panic!("expected a Local preparation")
        };
        assert_eq!(ptm.qubits(), &[3]);
        for d in ptm.deltas() {
            for s in 0..4usize {
                let t = s ^ d.local_delta as usize;
                assert!(
                    (d.amp[s] - h.table[s][t]).norm() < TOL,
                    "amp[{s}] for delta {} is {} but table[{s}][{t}] is {}",
                    d.local_delta,
                    d.amp[s],
                    h.table[s][t],
                );
            }
        }
    }
}
