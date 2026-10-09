//! General one- and two-qubit unitaries, stored as Pauli-transfer matrices.

use super::{qubit_loc, read_pauli, support_mask, write_pauli, Channel, OutputBuffer};
use num_complex::Complex64;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// The four single-qubit Pauli matrices, in `I, X, Z, Y` index order.
fn pauli_matrix(index: usize) -> [[Complex64; 2]; 2] {
    let i = Complex64::new(0.0, 1.0);
    match index {
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

/// Round each part of `v` within `eps` of zero to exactly zero, since every nonzero PTM entry becomes an output term.
fn clean(v: Complex64, eps: f64) -> Complex64 {
    let re = if v.re.abs() < eps { 0.0 } else { v.re };
    let im = if v.im.abs() < eps { 0.0 } else { v.im };
    Complex64::new(re, im)
}

/// Tolerance below which a derived PTM entry is treated as an exact zero.
const PTM_EPS: f64 = 1e-12;

/// Row `s` of `table`, or column `s` when `transpose` is set.
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

/// General single-qubit unitary, stored as its Pauli-transfer matrix.
///
/// Paulis are indexed `x | (z << 1)`, so `I = 0, X = 1, Z = 2, Y = 3`; `apply` conjugates `P ↦ U P U†` and `apply_adjoint` reads the table transposed.
#[derive(Clone, Debug)]
pub struct GeneralUnitary1Q {
    /// The single qubit this gate acts on.
    pub support: [usize; 1],
    /// `table[s][t]`: coefficient of output Pauli `t` for input Pauli `s`.
    pub table: [[Complex64; 4]; 4],
}

impl GeneralUnitary1Q {
    /// From a 2x2 unitary `u`: `table[s][t] = tr(P_t · U P_s U†) / 2`.
    pub fn from_matrix(qubit: usize, u: [[Complex64; 2]; 2]) -> Self {
        let u_dagger = dagger(&u);
        let mut table = [[ZERO; 4]; 4];
        for (s, row) in table.iter_mut().enumerate() {
            let ps = pauli_matrix(s);
            let conjugated = matmul(&matmul(&u, &ps), &u_dagger);
            for (t, slot) in row.iter_mut().enumerate() {
                let pt = pauli_matrix(t);
                let v = trace(&matmul(&pt, &conjugated)) / Complex64::new(2.0, 0.0);
                *slot = clean(v, PTM_EPS);
            }
        }
        Self {
            support: [qubit],
            table,
        }
    }
}

/// Body of the one-qubit `apply` and `apply_adjoint`.
fn apply_1q<const W: usize>(
    qubit: usize,
    table: &[[Complex64; 4]; 4],
    transpose: bool,
    input_x: &[u64; W],
    input_z: &[u64; W],
    coeff: Complex64,
    out: &mut OutputBuffer<'_, W>,
) {
    let q = qubit;
    debug_assert!(q < 64 * W);
    let (word, bit, mask) = qubit_loc(q);
    let s = read_pauli(input_x, input_z, word, bit);

    let row = effective_row(table, transpose, s);

    for (t, &c) in row.iter().enumerate() {
        if c == ZERO {
            continue;
        }
        let mut new_x = *input_x;
        let mut new_z = *input_z;
        write_pauli(&mut new_x, &mut new_z, word, bit, mask, t);
        out.push(new_x, new_z, coeff * c);
    }
}

impl<const W: usize> Channel<W> for GeneralUnitary1Q {
    fn max_fanout(&self) -> usize {
        4
    }

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

/// General two-qubit unitary, stored as its 16x16 Pauli-transfer matrix.
///
/// Paulis are indexed `x0 | (z0 << 1) | (x1 << 2) | (z1 << 3)`, qubit 0 being `support[0]`, which is the more significant tensor factor of the matrix given to [`Self::from_matrix`].
#[derive(Clone, Debug)]
pub struct GeneralUnitary2Q {
    /// The two qubits this gate acts on.
    pub support: [usize; 2],
    /// `table[s][t]`: coefficient of output Pauli `t` for input Pauli `s`.
    pub table: Box<[[Complex64; 16]; 16]>,
}

impl GeneralUnitary2Q {
    /// From a 4x4 unitary `u` on `|q0 q1⟩`: `table[s][t] = tr(P_t · U P_s U†) / 4`.
    pub fn from_matrix(q0: usize, q1: usize, u: [[Complex64; 4]; 4]) -> Self {
        let u_dagger = dagger(&u);
        let two_qubit_pauli = |s: usize| -> [[Complex64; 4]; 4] {
            let a = pauli_matrix((s & 1) | ((s >> 1) & 1) << 1);
            let b = pauli_matrix(((s >> 2) & 1) | ((s >> 3) & 1) << 1);
            kron2(&a, &b)
        };
        let mut table = Box::new([[ZERO; 16]; 16]);
        for (s, row) in table.iter_mut().enumerate() {
            let ps = two_qubit_pauli(s);
            let conjugated = matmul(&matmul(&u, &ps), &u_dagger);
            for (t, slot) in row.iter_mut().enumerate() {
                let pt = two_qubit_pauli(t);
                let v = trace(&matmul(&pt, &conjugated)) / Complex64::new(4.0, 0.0);
                *slot = clean(v, PTM_EPS);
            }
        }
        Self {
            support: [q0, q1],
            table,
        }
    }
}

/// Body of the two-qubit `apply` and `apply_adjoint`.
#[allow(clippy::too_many_arguments)]
fn apply_2q<const W: usize>(
    support: &[usize; 2],
    table: &[[Complex64; 16]; 16],
    transpose: bool,
    input_x: &[u64; W],
    input_z: &[u64; W],
    coeff: Complex64,
    out: &mut OutputBuffer<'_, W>,
) {
    let q0 = support[0];
    let q1 = support[1];
    debug_assert!(q0 < 64 * W && q1 < 64 * W);
    let (word0, bit0, mask0) = qubit_loc(q0);
    let (word1, bit1, mask1) = qubit_loc(q1);

    let s = read_pauli(input_x, input_z, word0, bit0)
        | (read_pauli(input_x, input_z, word1, bit1) << 2);

    let row = effective_row(table, transpose, s);

    for (t, &c) in row.iter().enumerate() {
        if c == ZERO {
            continue;
        }
        let mut new_x = *input_x;
        let mut new_z = *input_z;
        write_pauli(&mut new_x, &mut new_z, word0, bit0, mask0, t & 3);
        write_pauli(&mut new_x, &mut new_z, word1, bit1, mask1, (t >> 2) & 3);
        out.push(new_x, new_z, coeff * c);
    }
}

impl<const W: usize> Channel<W> for GeneralUnitary2Q {
    fn max_fanout(&self) -> usize {
        16
    }

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
