use super::*;
use crate::test_support::raw_outputs;
use num_complex::Complex64;

/// The one output of `gate` on `input` with coefficient 1.
fn apply_once<const W: usize>(
    gate: &PtmChannel,
    adjoint: bool,
    input: &PauliString<W>,
) -> (PauliString<W>, Complex64) {
    let out = raw_outputs(gate, adjoint, *input, Complex64::new(1.0, 0.0));
    assert_eq!(out.len(), 1, "a Clifford has fanout 1");
    out[0]
}

fn apply_1q<const W: usize>(
    gate: &PtmChannel,
    input: &PauliString<W>,
) -> (PauliString<W>, Complex64) {
    apply_once(gate, false, input)
}

fn apply_2q<const W: usize>(
    gate: &PtmChannel,
    input: &PauliString<W>,
) -> (PauliString<W>, Complex64) {
    apply_once(gate, false, input)
}

fn pauli_y<const W: usize>(qubit: usize) -> PauliString<W> {
    PauliString::<W>::y(qubit)
}

#[test]
fn h_on_qubit_0_w1() {
    let h = PtmChannel::hadamard(0);
    // I → I, +1
    let (out, c) = apply_1q::<1>(&h, &PauliString::<1>::identity());
    assert_eq!(out, PauliString::<1>::identity());
    assert_eq!(c, Complex64::new(1.0, 0.0));
    // X → Z, +1
    let (out, c) = apply_1q::<1>(&h, &PauliString::<1>::x(0));
    assert_eq!(out, PauliString::<1>::z(0));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    // Z → X, +1
    let (out, c) = apply_1q::<1>(&h, &PauliString::<1>::z(0));
    assert_eq!(out, PauliString::<1>::x(0));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    // Y → -Y, -1
    let (out, c) = apply_1q::<1>(&h, &pauli_y::<1>(0));
    assert_eq!(out, pauli_y::<1>(0));
    assert_eq!(c, Complex64::new(-1.0, 0.0));
}

#[test]
fn s_on_qubit_0_w1() {
    let s = PtmChannel::phase_gate(0);
    // X → Y, +1
    let (out, c) = apply_1q::<1>(&s, &PauliString::<1>::x(0));
    assert_eq!(out, pauli_y::<1>(0));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    // Z → Z, +1
    let (out, c) = apply_1q::<1>(&s, &PauliString::<1>::z(0));
    assert_eq!(out, PauliString::<1>::z(0));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    // Y → -X, -1
    let (out, c) = apply_1q::<1>(&s, &pauli_y::<1>(0));
    assert_eq!(out, PauliString::<1>::x(0));
    assert_eq!(c, Complex64::new(-1.0, 0.0));
}

#[test]
fn x_gate_w1() {
    let g = PtmChannel::pauli_x(0);
    // X → X, Z → -Z, Y → -Y
    let (o, c) = apply_1q::<1>(&g, &PauliString::<1>::x(0));
    assert_eq!(o, PauliString::<1>::x(0));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    let (o, c) = apply_1q::<1>(&g, &PauliString::<1>::z(0));
    assert_eq!(o, PauliString::<1>::z(0));
    assert_eq!(c, Complex64::new(-1.0, 0.0));
    let (o, c) = apply_1q::<1>(&g, &pauli_y::<1>(0));
    assert_eq!(o, pauli_y::<1>(0));
    assert_eq!(c, Complex64::new(-1.0, 0.0));
}

#[test]
fn y_gate_w1() {
    let g = PtmChannel::pauli_y(0);
    let (o, c) = apply_1q::<1>(&g, &PauliString::<1>::x(0));
    assert_eq!(o, PauliString::<1>::x(0));
    assert_eq!(c, Complex64::new(-1.0, 0.0));
    let (o, c) = apply_1q::<1>(&g, &PauliString::<1>::z(0));
    assert_eq!(o, PauliString::<1>::z(0));
    assert_eq!(c, Complex64::new(-1.0, 0.0));
    let (o, c) = apply_1q::<1>(&g, &pauli_y::<1>(0));
    assert_eq!(o, pauli_y::<1>(0));
    assert_eq!(c, Complex64::new(1.0, 0.0));
}

#[test]
fn z_gate_w1() {
    let g = PtmChannel::pauli_z(0);
    let (o, c) = apply_1q::<1>(&g, &PauliString::<1>::x(0));
    assert_eq!(o, PauliString::<1>::x(0));
    assert_eq!(c, Complex64::new(-1.0, 0.0));
    let (o, c) = apply_1q::<1>(&g, &PauliString::<1>::z(0));
    assert_eq!(o, PauliString::<1>::z(0));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    let (o, c) = apply_1q::<1>(&g, &pauli_y::<1>(0));
    assert_eq!(o, pauli_y::<1>(0));
    assert_eq!(c, Complex64::new(-1.0, 0.0));
}

#[test]
fn h_on_qubit_64_w2_word_boundary() {
    let h = PtmChannel::hadamard(64);
    // X(64) → Z(64), bits live in word 1.
    let (out, c) = apply_1q::<2>(&h, &PauliString::<2>::x(64));
    assert_eq!(out, PauliString::<2>::z(64));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    assert_eq!(out.x[0], 0); // word 0 untouched
    assert_eq!(out.z[0], 0);
    // Z(64) → X(64).
    let (out, _c) = apply_1q::<2>(&h, &PauliString::<2>::z(64));
    assert_eq!(out, PauliString::<2>::x(64));
}

#[test]
fn support_outside_bits_untouched_w2() {
    // Build X(0) · X(70) and apply H(70). Bit 0 must stay X, bit 70 must become Z, and the coefficient stays +1.
    let mut input = PauliString::<2>::x(0);
    let _ = input.mul_assign(&PauliString::<2>::x(70));
    let h = PtmChannel::hadamard(70);
    let (out, c) = apply_1q::<2>(&h, &input);

    // Expected: X(0) · Z(70).
    let mut expected = PauliString::<2>::x(0);
    let _ = expected.mul_assign(&PauliString::<2>::z(70));
    assert_eq!(out, expected);
    assert_eq!(c, Complex64::new(1.0, 0.0));
}

#[test]
fn h_squared_is_identity() {
    // Apply H twice; the result is the input with phase +1.
    let h = PtmChannel::hadamard(0);
    for input in [
        PauliString::<1>::identity(),
        PauliString::<1>::x(0),
        PauliString::<1>::z(0),
        pauli_y::<1>(0),
    ] {
        let (mid, c1) = apply_1q::<1>(&h, &input);
        let (out, c2) = apply_1q::<1>(&h, &mid);
        assert_eq!(out, input, "H·H should be identity on {:?}", input);
        assert_eq!(
            c1 * c2,
            Complex64::new(1.0, 0.0),
            "phase should square to +1"
        );
    }
}

/// Self-adjoint 1Q Cliffords round-trip through `adjoint()` to themselves.
#[test]
fn h_x_y_z_are_self_adjoint() {
    for gate in [
        PtmChannel::hadamard(0),
        PtmChannel::pauli_x(0),
        PtmChannel::pauli_y(0),
        PtmChannel::pauli_z(0),
    ] {
        assert_eq!(
            gate.adjoint().transfer_matrix(false),
            gate.transfer_matrix(false)
        );
    }
}

/// `S` is not self-adjoint, but `(S†)† = S`.
#[test]
fn s_adjoint_inverts_table_and_is_involutive() {
    let s = PtmChannel::phase_gate(0);
    let s_dagger = s.adjoint();
    // S†: I→I, X→-Y, Z→Z, Y→X.
    let table = s_dagger.transfer_matrix(false);
    assert_eq!(table[0][0], Complex64::new(1.0, 0.0));
    assert_eq!(table[1][3], Complex64::new(-1.0, 0.0));
    assert_eq!(table[2][2], Complex64::new(1.0, 0.0));
    assert_eq!(table[3][1], Complex64::new(1.0, 0.0));
    assert_eq!(s_dagger.transfer_matrix(false), s.transfer_matrix(true));
    assert_eq!(
        s_dagger.adjoint().transfer_matrix(false),
        s.transfer_matrix(false)
    );
}

/// `apply_adjoint` on `S` undoes `apply` on X.
#[test]
fn s_apply_then_apply_adjoint_round_trips() {
    let s = PtmChannel::phase_gate(0);
    let x_in = PauliString::<1>::x(0);
    let (mid, c1) = apply_1q::<1>(&s, &x_in);
    let (back, c2) = apply_once::<1>(&s, true, &mid);
    assert_eq!(back, x_in);
    assert_eq!(c1 * c2, Complex64::new(1.0, 0.0));
}

/// CNOT, CZ and SWAP are involutions, so their adjoint is themselves; the old table type silently relied on this.
#[test]
fn two_qubit_cliffords_are_self_adjoint() {
    for gate in [
        PtmChannel::cnot(1, 4),
        PtmChannel::cz(4, 1),
        PtmChannel::swap(1, 4),
    ] {
        assert_eq!(gate.transfer_matrix(true), gate.transfer_matrix(false));
    }
}

#[test]
#[should_panic(expected = "PtmChannel::cnot: the two qubits must differ (both are 3)")]
fn cnot_rejects_equal_qubits() {
    let _ = PtmChannel::cnot(3, 3);
}

/// Build `P_a ⊗ P_b` on qubits `(q0, q1)` of a `PauliString<W>` using `mul_assign`, where `pa` and `pb` are 2-bit single-qubit Pauli codes (`I=0, X=1, Z=2, Y=3`).
fn tensor<const W: usize>(q0: usize, q1: usize, pa: u8, pb: u8) -> PauliString<W> {
    let mut p = PauliString::<W>::identity();
    let put = |p: &mut PauliString<W>, q: usize, code: u8| {
        let g = match code {
            0 => return,
            1 => PauliString::<W>::x(q),
            2 => PauliString::<W>::z(q),
            3 => PauliString::<W>::y(q),
            _ => unreachable!(),
        };
        // Y on a fresh identity contributes phase 0, and the partial products are on disjoint qubits, so phases are always +1 here.
        let _ = p.mul_assign(&g);
    };
    put(&mut p, q0, pa);
    put(&mut p, q1, pb);
    p
}

#[test]
fn cnot_generator_rules_w1() {
    let cnot = PtmChannel::cnot(0, 1);
    // X⊗I → X⊗X
    let (o, c) = apply_2q::<1>(&cnot, &tensor::<1>(0, 1, 1, 0));
    assert_eq!(o, tensor::<1>(0, 1, 1, 1));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    // I⊗X → I⊗X
    let (o, c) = apply_2q::<1>(&cnot, &tensor::<1>(0, 1, 0, 1));
    assert_eq!(o, tensor::<1>(0, 1, 0, 1));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    // Z⊗I → Z⊗I
    let (o, c) = apply_2q::<1>(&cnot, &tensor::<1>(0, 1, 2, 0));
    assert_eq!(o, tensor::<1>(0, 1, 2, 0));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    // I⊗Z → Z⊗Z
    let (o, c) = apply_2q::<1>(&cnot, &tensor::<1>(0, 1, 0, 2));
    assert_eq!(o, tensor::<1>(0, 1, 2, 2));
    assert_eq!(c, Complex64::new(1.0, 0.0));
}

/// Reference for `CNOT (P⊗Q) CNOT†` derived from generator rules and `PauliString::mul_assign`. Each input Pauli on each qubit maps:
///   qubit 0:  I→I,  X→X⊗X,  Z→Z⊗I,  Y→i·X·Z → Y⊗X (with phase from XZ·X)
///   qubit 1:  I→I,  X→I⊗X,  Z→Z⊗Z,  Y→i·X·Z → Z⊗Y
/// The image is the product of these two qubit images, folding in any phase the multiplication picks up.
fn cnot_reference<const W: usize>(
    control: usize,
    target: usize,
    pa: u8,
    pb: u8,
) -> (PauliString<W>, Phase) {
    // Image of `pa` on `control`: a `PauliString<W>` plus a phase.
    let (img_a, ph_a) = match pa {
        0 => (PauliString::<W>::identity(), Phase::ONE),
        1 => {
            // X⊗X on (control, target).
            let mut p = PauliString::<W>::x(control);
            let _ = p.mul_assign(&PauliString::<W>::x(target));
            (p, Phase::ONE)
        }
        2 => (PauliString::<W>::z(control), Phase::ONE),
        3 => {
            // Y → i · X · Z: image of X is X⊗X, image of Z is Z⊗I; product picks up phase from mul_assign, then multiply by `i` for the Y-decomposition.
            let mut p = PauliString::<W>::x(control);
            let _ = p.mul_assign(&PauliString::<W>::x(target));
            let q = PauliString::<W>::z(control);
            let mp = p.mul_assign(&q);
            // Y = i · X · Z, so the image is i · (X-image)(Z-image).
            (p, Phase::I + mp)
        }
        _ => unreachable!(),
    };
    let (img_b, ph_b) = match pb {
        0 => (PauliString::<W>::identity(), Phase::ONE),
        1 => (PauliString::<W>::x(target), Phase::ONE),
        2 => {
            // I⊗Z → Z⊗Z.
            let mut p = PauliString::<W>::z(control);
            let _ = p.mul_assign(&PauliString::<W>::z(target));
            (p, Phase::ONE)
        }
        3 => {
            // I⊗Y → I⊗(i·X·Z) → i · (I⊗X) · (Z⊗Z) = i · X(target) · Z(c) · Z(t).
            let mut p = PauliString::<W>::x(target);
            let mut q = PauliString::<W>::z(control);
            let _ = q.mul_assign(&PauliString::<W>::z(target));
            let mp = p.mul_assign(&q);
            (p, Phase::I + mp)
        }
        _ => unreachable!(),
    };
    // Combine the two images: image of `pa ⊗ pb` is image(pa) · image(pb) since they commute as operators on different input qubits, but the image operators may overlap, so phases come from mul_assign.
    let mut prod = img_a;
    let mp = prod.mul_assign(&img_b);
    (prod, ph_a + ph_b + mp)
}

#[test]
fn cnot_full_table_w1() {
    let cnot = PtmChannel::cnot(0, 1);
    for pa in 0u8..4 {
        for pb in 0u8..4 {
            let input = tensor::<1>(0, 1, pa, pb);
            let (got_out, got_c) = apply_2q::<1>(&cnot, &input);
            let (exp_out, exp_phase) = cnot_reference::<1>(0, 1, pa, pb);
            assert_eq!(
                got_out, exp_out,
                "CNOT output mismatch on (pa={}, pb={})",
                pa, pb
            );
            assert_eq!(
                got_c,
                exp_phase.to_complex(),
                "CNOT phase mismatch on (pa={}, pb={})",
                pa,
                pb
            );
        }
    }
}

#[test]
fn cnot_word_boundary_w2() {
    // Control on qubit 63 (last bit of word 0), target on qubit 64 (first bit of word 1) — exercises the per-qubit word/bit math independently for each support qubit.
    let cnot = PtmChannel::cnot(63, 64);
    // X⊗X on (63, 64) → X⊗I (per CNOT rules: X⊗X → X⊗I).
    let input = tensor::<2>(63, 64, 1, 1);
    let (out, c) = apply_2q::<2>(&cnot, &input);
    let expected = tensor::<2>(63, 64, 1, 0);
    assert_eq!(out, expected);
    assert_eq!(c, Complex64::new(1.0, 0.0));
    // Y⊗X on (63, 64). Reference computes the image.
    let input = tensor::<2>(63, 64, 3, 1);
    let (got_out, got_c) = apply_2q::<2>(&cnot, &input);
    let (exp_out, exp_phase) = cnot_reference::<2>(63, 64, 3, 1);
    assert_eq!(got_out, exp_out);
    assert_eq!(got_c, exp_phase.to_complex());
}

#[test]
fn cz_symmetry_w1() {
    // CZ is symmetric in its qubits.
    let cz_ab = PtmChannel::cz(0, 1);
    let cz_ba = PtmChannel::cz(1, 0);
    for pa in 0u8..4 {
        for pb in 0u8..4 {
            let input = tensor::<1>(0, 1, pa, pb);
            let (a, ca) = apply_2q::<1>(&cz_ab, &input);
            let (b, cb) = apply_2q::<1>(&cz_ba, &input);
            assert_eq!(a, b, "CZ(0,1) and CZ(1,0) disagree on ({}, {})", pa, pb);
            assert_eq!(ca, cb);
        }
    }
}

#[test]
fn cz_generator_rules_w1() {
    let cz = PtmChannel::cz(0, 1);
    // X⊗I → X⊗Z
    let (o, c) = apply_2q::<1>(&cz, &tensor::<1>(0, 1, 1, 0));
    assert_eq!(o, tensor::<1>(0, 1, 1, 2));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    // I⊗X → Z⊗X
    let (o, c) = apply_2q::<1>(&cz, &tensor::<1>(0, 1, 0, 1));
    assert_eq!(o, tensor::<1>(0, 1, 2, 1));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    // Z⊗I → Z⊗I
    let (o, c) = apply_2q::<1>(&cz, &tensor::<1>(0, 1, 2, 0));
    assert_eq!(o, tensor::<1>(0, 1, 2, 0));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    // I⊗Z → I⊗Z
    let (o, c) = apply_2q::<1>(&cz, &tensor::<1>(0, 1, 0, 2));
    assert_eq!(o, tensor::<1>(0, 1, 0, 2));
    assert_eq!(c, Complex64::new(1.0, 0.0));
}

#[test]
fn swap_table_w1() {
    let swap = PtmChannel::swap(0, 1);
    // II → II, XI → IX, IY → YI, XZ → ZX
    let (o, c) = apply_2q::<1>(&swap, &tensor::<1>(0, 1, 0, 0));
    assert_eq!(o, tensor::<1>(0, 1, 0, 0));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    let (o, c) = apply_2q::<1>(&swap, &tensor::<1>(0, 1, 1, 0));
    assert_eq!(o, tensor::<1>(0, 1, 0, 1));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    let (o, c) = apply_2q::<1>(&swap, &tensor::<1>(0, 1, 0, 3));
    assert_eq!(o, tensor::<1>(0, 1, 3, 0));
    assert_eq!(c, Complex64::new(1.0, 0.0));
    let (o, c) = apply_2q::<1>(&swap, &tensor::<1>(0, 1, 1, 2));
    assert_eq!(o, tensor::<1>(0, 1, 2, 1));
    assert_eq!(c, Complex64::new(1.0, 0.0));
}
