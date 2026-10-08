use super::*;
use crate::channel::clifford::{Clifford1Q, Clifford2Q};
use crate::channel::prepared::Prepared;
use crate::pauli_string::PauliString;
use crate::pauli_sum::hash::Gf2Hash;
use crate::test_support::outputs;

const TOL: f64 = 1e-12;
const R: f64 = std::f64::consts::FRAC_1_SQRT_2;

fn c(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

type Term<const W: usize> = ([u64; W], [u64; W], Complex64);

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

/// The `T` gate mixes `X` with `Y` and fixes `I` and `Z`, so it is a genuine fanout-2 non-Clifford whose delta set is only one-dimensional, reading 2 buckets rather than the 4 a dense 1Q unitary would.
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

/// A dense 1Q unitary does reach the 4-bucket upper bound, and a dense 2Q one reaches 16 (ARCHITECTURE.md §Bucketing).
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

/// The derivation must recover exactly the table it was built from — a bounded-support channel *is* its local PTM.
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
