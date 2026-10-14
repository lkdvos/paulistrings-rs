use super::*;
use crate::channel::Channel;
use crate::pauli_string::PauliString;
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

#[test]
fn hadamard_as_a_general_unitary_matches_clifford1q() {
    let h = PtmChannel::unitary_1q(3, [[c(R), c(R)], [c(R), c(-R)]]);
    assert_agrees_on_basis::<2, _, _>(&h, &PtmChannel::hadamard(3), &[3], "hadamard");
}

#[test]
fn phase_gate_as_a_general_unitary_matches_clifford1q() {
    let i = Complex64::new(0.0, 1.0);
    let s = PtmChannel::unitary_1q(3, [[ONE, ZERO], [ZERO, i]]);
    assert_agrees_on_basis::<2, _, _>(&s, &PtmChannel::phase_gate(3), &[3], "phase");
}

#[test]
fn pauli_gates_as_general_unitaries_match_clifford1q() {
    let i = Complex64::new(0.0, 1.0);
    let x = PtmChannel::unitary_1q(5, [[ZERO, ONE], [ONE, ZERO]]);
    assert_agrees_on_basis::<2, _, _>(&x, &PtmChannel::pauli_x(5), &[5], "pauli_x");
    let y = PtmChannel::unitary_1q(5, [[ZERO, -i], [i, ZERO]]);
    assert_agrees_on_basis::<2, _, _>(&y, &PtmChannel::pauli_y(5), &[5], "pauli_y");
    let z = PtmChannel::unitary_1q(5, [[ONE, ZERO], [ZERO, -ONE]]);
    assert_agrees_on_basis::<2, _, _>(&z, &PtmChannel::pauli_z(5), &[5], "pauli_z");
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
    let g = PtmChannel::unitary_2q(1, 4, u);
    assert_agrees_on_basis::<2, _, _>(&g, &PtmChannel::cnot(1, 4), &[1, 4], "cnot");
}

#[test]
fn cz_as_a_general_unitary_matches_clifford2q() {
    let mut u = [[ZERO; 4]; 4];
    for (i, row) in u.iter_mut().enumerate() {
        row[i] = if i == 3 { -ONE } else { ONE };
    }
    let g = PtmChannel::unitary_2q(1, 4, u);
    assert_agrees_on_basis::<2, _, _>(&g, &PtmChannel::cz(1, 4), &[1, 4], "cz");
}

#[test]
fn swap_as_a_general_unitary_matches_clifford2q() {
    let u = [
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, ZERO, ONE, ZERO],
        [ZERO, ONE, ZERO, ZERO],
        [ZERO, ZERO, ZERO, ONE],
    ];
    let g = PtmChannel::unitary_2q(1, 4, u);
    assert_agrees_on_basis::<2, _, _>(&g, &PtmChannel::swap(1, 4), &[1, 4], "swap");
}

#[test]
fn adjoint_reads_the_table_transposed_and_round_trips() {
    let i = Complex64::new(0.0, 1.0);
    // T gate: not self-adjoint.
    let t = PtmChannel::unitary_1q(
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
        let mut accumulator: Vec<Term<1>> = Vec::new();
        for (x, z, cf) in outputs::<1, _>(&t, false, basis, ONE) {
            for out in outputs::<1, _>(&t, true, PauliString::<1> { x, z }, cf) {
                accumulator.push(out);
            }
        }
        accumulator.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let mut merged: Vec<Term<1>> = Vec::new();
        for (x, z, cf) in accumulator {
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
    let g = PtmChannel::unitary_2q(1, 4, u);
    let cn = PtmChannel::cnot(1, 4);
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
