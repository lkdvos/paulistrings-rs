//! General unitaries, stored as local Pauli-transfer matrices. See ARCHITECTURE.md §Channels and §Prepared-Channels.
//!
//! Pauli indexing matches [`Clifford1Q`]/[`Clifford2Q`] (`idx = x | (z << 1)`, two-qubit packs `x0 | (z0 << 1) | (x1 << 2) | (z1 << 3)`). `table[s][t]` is the coefficient of output Pauli `t` for input Pauli `s` (the Heisenberg conjugation `P ↦ U P U†`); `apply_adjoint` reads the table transposed (the Hilbert-Schmidt adjoint — [`AmplitudeDamping`](super::noise::AmplitudeDamping) follows the same rule).
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
            // A transpose: `j` indexes rows of `a` while `i` indexes its columns, so neither loop can be turned into an iterator over `a`.
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
/// Without this, a Clifford built via [`GeneralUnitary1Q::from_matrix`] carries `~1e-17` entries where it should carry exact zeros, each becoming a spurious output term; the engine only drops exact zeros, deliberately, so the cleanup has to happen here.
fn clean(v: Complex64, eps: f64) -> Complex64 {
    let re = if v.re.abs() < eps { 0.0 } else { v.re };
    let im = if v.im.abs() < eps { 0.0 } else { v.im };
    Complex64::new(re, im)
}

/// Tolerance below which a derived PTM entry is treated as an exact zero.
const PTM_EPS: f64 = 1e-12;

/// Materialize the effective PTM row for input index `s`: `table[s]` normally, or its transpose column when `transpose` is set (the Hilbert-Schmidt adjoint).
/// Shared by [`apply_1q`] (`N = 4`) and [`apply_2q`] (`N = 16`), so the transpose flag is tested once per call instead of once per emitted term.
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

/// Generic 1-qubit unitary, stored as the Pauli expansion of its Heisenberg-picture action on `{I, X, Z, Y}` at the support qubit.
///
/// `MAX_FANOUT = 4`, since an input Pauli on the support can map to a sum over all four basis Paulis; the bucket fan-in is `2^rank(H|_D)` for the realized delta set `D`, which is at most 4 but often smaller (a `T` gate only ever mixes `X` with `Y`, so it reads just 2 input buckets).
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use paulistrings::GeneralUnitary1Q;
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

    // Materialize the effective row once, so the transpose branch is hoisted out of the loop instead of being retested per output.
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
/// Packing is `x0 | (z0 << 1) | (x1 << 2) | (z1 << 3)`, `support[0]` in bits 0-1 and `support[1]` in bits 2-3 — the same convention [`Clifford2Q`](super::clifford::Clifford2Q) uses. In the matrix passed to [`Self::from_matrix`], `support[0]` is the more significant tensor factor, i.e. the matrix acts on `|q0 q1⟩`.
#[derive(Clone, Debug)]
pub struct GeneralUnitary2Q {
    /// The two qubits this gate acts on.
    pub support: [u32; 2],
    /// `table[s][t]`: coefficient of output Pauli `t` for input Pauli `s`.
    pub table: Box<[[Complex64; 16]; 16]>,
}

impl GeneralUnitary2Q {
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
mod tests;
