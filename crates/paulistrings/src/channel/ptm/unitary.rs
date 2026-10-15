//! The general-unitary constructors of [`PtmChannel`], whose PTM is derived from the matrix.

use super::{assert_distinct, sparse_storage, PtmChannel, PtmKind, ONE, ZERO};
use num_complex::Complex64;

/// Tolerance below which a derived PTM entry is treated as an exact zero, since every nonzero entry becomes an output term.
const PTM_EPS: f64 = 1e-12;

/// The largest `|(U†U − I)_ij|` the general-unitary constructors accept; the Python bindings check the same bound first.
const UNITARITY_TOLERANCE: f64 = 1e-10;

impl PtmChannel {
    /// A general single-qubit unitary `U`: `ptm[s][t] = tr(P_t · U P_s U†) / 2`.
    ///
    /// # Panics
    ///
    /// Unless every entry of `U†U − I` is at most `1e-10` in absolute value.
    pub fn unitary_1q(qubit: usize, u: [[Complex64; 2]; 2]) -> Self {
        assert_unitary("unitary_1q", &u);
        let u_dagger = dagger(&u);
        let mut entries = Vec::new();
        for s in 0..4 {
            let conjugated = matmul(&matmul(&u, &pauli_matrix(s)), &u_dagger);
            for t in 0..4 {
                let value =
                    trace(&matmul(&pauli_matrix(t), &conjugated)) / Complex64::new(2.0, 0.0);
                entries.push(((s as u8, t as u8), clean(value)));
            }
        }
        Self::new(&[qubit], sparse_storage(entries), PtmKind::Unitary1q)
    }

    /// A general two-qubit unitary `U` on `|q0 q1⟩`, `q0` the more significant tensor factor: `ptm[s][t] = tr(P_t · U P_s U†) / 4`.
    ///
    /// # Panics
    ///
    /// If `q0 == q1`, or unless every entry of `U†U − I` is at most `1e-10` in absolute value.
    pub fn unitary_2q(q0: usize, q1: usize, u: [[Complex64; 4]; 4]) -> Self {
        assert_distinct("unitary_2q", q0, q1);
        assert_unitary("unitary_2q", &u);
        let u_dagger = dagger(&u);
        let two_qubit_pauli = |s: usize| kron(&pauli_matrix(s & 3), &pauli_matrix(s >> 2));
        let mut entries = Vec::new();
        for s in 0..16 {
            let conjugated = matmul(&matmul(&u, &two_qubit_pauli(s)), &u_dagger);
            for t in 0..16 {
                let value =
                    trace(&matmul(&two_qubit_pauli(t), &conjugated)) / Complex64::new(4.0, 0.0);
                entries.push(((s as u8, t as u8), clean(value)));
            }
        }
        Self::new(&[q0, q1], sparse_storage(entries), PtmKind::Unitary2q)
    }
}

/// Panics unless `u` is unitary to [`UNITARITY_TOLERANCE`]; a NaN entry fails too.
fn assert_unitary<const N: usize>(constructor: &str, u: &[[Complex64; N]; N]) {
    let product = matmul(&dagger(u), u);
    let deviation = (0..N)
        .flat_map(|i| (0..N).map(move |j| (i, j)))
        .map(|(i, j)| {
            let identity = if i == j { ONE } else { ZERO };
            (product[i][j] - identity).norm()
        })
        .fold(0.0, |largest: f64, entry| {
            if entry > largest || entry.is_nan() {
                entry
            } else {
                largest
            }
        });
    assert!(
        deviation <= UNITARITY_TOLERANCE,
        "PtmChannel::{constructor}: the matrix is not unitary (largest entry of U†U − I is {deviation:e}, tolerance {UNITARITY_TOLERANCE:e})"
    );
}

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
    core::array::from_fn(|i| core::array::from_fn(|j| a[j][i].conj()))
}

fn trace<const N: usize>(a: &[[Complex64; N]; N]) -> Complex64 {
    a.iter().enumerate().map(|(i, row)| row[i]).sum()
}

/// `kron(a, b)` with `a` acting on the more significant qubit.
fn kron(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> [[Complex64; 4]; 4] {
    core::array::from_fn(|row| {
        core::array::from_fn(|column| a[row / 2][column / 2] * b[row % 2][column % 2])
    })
}

/// Each part of `v` within [`PTM_EPS`] of zero rounded to exactly zero.
fn clean(v: Complex64) -> Complex64 {
    let snap = |part: f64| if part.abs() < PTM_EPS { 0.0 } else { part };
    Complex64::new(snap(v.re), snap(v.im))
}

#[cfg(test)]
mod tests;
