use num_complex::Complex64;

use super::*;
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::{Gf2Hash, PauliSum};
use crate::test_support::{low_weight_sum, rand_sum, Xs64};

/// A sum of `terms` on `num_qubits` qubits.
fn build<const W: usize>(num_qubits: usize, terms: &[(PauliString<W>, Complex64)]) -> PauliSum<W> {
    let mut accumulator = BuildAccumulator::<W>::with_capacity(num_qubits, terms.len());
    for &(pp, c) in terms {
        accumulator.add_term(pp, c);
    }
    accumulator.finalize()
}

#[test]
fn bucketed_expectation_agrees_with_the_flat_version() {
    // Not bitwise: partials are combined in bucket order.
    for &weight in &[2usize, 4] {
        let sum = low_weight_sum::<2>(20_000, 100, weight, 0xE1 + weight as u64);
        let want = sum.expectation_product(&ProductBasis::uniform(PauliAxis::X));
        for bits in [0u8, 3, 7, 11] {
            let h = Gf2Hash::<2>::new(100, bits, 0xE2);
            let b = sum.clone().with_hash(h);
            let got = b.expectation_product(&ProductBasis::uniform(PauliAxis::X));
            assert!((got - want).norm() < 1e-9, "weight={weight} bits={bits}: {got} vs {want}",);
        }
    }
}

#[test]
fn bucketed_expectation_covers_all_three_states() {
    let sum = rand_sum::<1>(5000, 20, 0xE3);
    let h = Gf2Hash::<1>::new(20, 5, 0xE4);
    let b = sum.clone().with_hash(h);
    for state in [PauliAxis::X, PauliAxis::Y, PauliAxis::Z] {
        let got = b.expectation_product(&ProductBasis::uniform(state));
        let want = sum.expectation_product(&ProductBasis::uniform(state));
        assert!((got - want).norm() < 1e-9, "{state:?}: {got} vs {want}");
    }
}

#[test]
fn bucketed_expectation_of_an_empty_sum_is_zero() {
    let h = Gf2Hash::<1>::new(8, 3, 0xE5);
    let b = PauliSum::<1>::empty_with_hash(8, h);
    assert!(b.expectation_product(&ProductBasis::uniform(PauliAxis::X)).norm().abs() < 1e-15);
}

#[test]
fn expectation_of_single_paulis_in_each_product_state() {
    let cases = [
        (PauliString::<1>::identity(), 1.0, 1.0, 1.0),
        (PauliString::<1>::x(0), 1.0, 0.0, 0.0),
        (PauliString::<1>::y(0), 0.0, 1.0, 0.0),
        (PauliString::<1>::z(0), 0.0, 0.0, 1.0),
    ];
    for (pp, ex, ey, ez) in cases {
        let s = build::<1>(8, &[(pp, Complex64::new(1.0, 0.0))]);
        assert!(
            (s.expectation_product(&ProductBasis::uniform(PauliAxis::X)).re - ex).abs() < 1e-12,
            "XPlus for {pp:?}",
        );
        assert!(
            (s.expectation_product(&ProductBasis::uniform(PauliAxis::Y)).re - ey).abs() < 1e-12,
            "YPlus for {pp:?}",
        );
        assert!(
            (s.expectation_product(&ProductBasis::uniform(PauliAxis::Z)).re - ez).abs() < 1e-12,
            "ZPlus for {pp:?}",
        );
    }
}

#[test]
fn expectation_of_multi_qubit_products() {
    let mut xx = PauliString::<1>::x(0);
    xx.mul_assign(&PauliString::<1>::x(1));
    let mut xz = PauliString::<1>::x(0);
    xz.mul_assign(&PauliString::<1>::z(1));
    let mut yy = PauliString::<1>::y(0);
    yy.mul_assign(&PauliString::<1>::y(1));

    let s = build::<1>(
        8,
        &[
            (xx, Complex64::new(1.0, 0.0)),
            (xz, Complex64::new(10.0, 0.0)),
            (yy, Complex64::new(100.0, 0.0)),
        ],
    );
    assert!((s.expectation_product(&ProductBasis::uniform(PauliAxis::X)).re - 1.0).abs() < 1e-12);
    assert!((s.expectation_product(&ProductBasis::uniform(PauliAxis::Y)).re - 100.0).abs() < 1e-12);
    assert!(s.expectation_product(&ProductBasis::uniform(PauliAxis::Z)).re.abs() < 1e-12);
}

#[test]
fn expectation_is_linear_and_keeps_the_imaginary_part() {
    let s = build::<1>(
        8,
        &[
            (PauliString::<1>::x(0), Complex64::new(1.0, 2.0)),
            (PauliString::<1>::x(1), Complex64::new(3.0, -5.0)),
        ],
    );
    let e = s.expectation_product(&ProductBasis::uniform(PauliAxis::X));
    assert!((e - Complex64::new(4.0, -3.0)).norm() < 1e-12);
}

#[test]
fn expectation_across_a_word_boundary_w2() {
    let s = build::<2>(
        128,
        &[
            (PauliString::<2>::x(70), Complex64::new(2.0, 0.0)),
            (PauliString::<2>::z(70), Complex64::new(9.0, 0.0)),
        ],
    );
    assert!((s.expectation_product(&ProductBasis::uniform(PauliAxis::X)).re - 2.0).abs() < 1e-12);
    assert!((s.expectation_product(&ProductBasis::uniform(PauliAxis::Z)).re - 9.0).abs() < 1e-12);
}

/// The `|+…+⟩` expectation against a hand-rolled sum over Z-free terms.
#[test]
fn expectation_xplus_matches_the_hand_rolled_reference() {
    let mut rng = 0x2468u64 | 1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut accumulator =
        crate::pauli_sum::accumulator::BuildAccumulator::<1>::with_capacity(16, 500);
    for _ in 0..500 {
        let pp = PauliString::<1> { x: [next() & 0xFFFF], z: [next() & 0xFFFF] };
        let c = Complex64::new((next() as i64 as f64) / (i64::MAX as f64), 0.0);
        accumulator.add_term(pp, c);
    }
    let sum = accumulator.finalize();

    let mut want = 0.0f64;
    for i in 0..sum.len() {
        if sum.bucket(0).1[i] == [0u64] {
            want += sum.bucket(0).2[i].re;
        }
    }
    let got = sum.expectation_product(&ProductBasis::uniform(PauliAxis::X)).re;
    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
}

// Expected values below are products of single-qubit Bloch components:
//   |0⟩ = Z+:  ⟨Z⟩ = +1,  ⟨X⟩ = ⟨Y⟩ = 0
//   |1⟩ = Z-:  ⟨Z⟩ = -1,  ⟨X⟩ = ⟨Y⟩ = 0
//   |+⟩ = X+:  ⟨X⟩ = +1,  ⟨Y⟩ = ⟨Z⟩ = 0
//   |-⟩ = X-:  ⟨X⟩ = -1,  ⟨Y⟩ = ⟨Z⟩ = 0
//   |r⟩ = Y+ = (|0⟩ + i|1⟩)/√2:  ⟨Y⟩ = +1,  ⟨X⟩ = ⟨Z⟩ = 0
//   |l⟩ = Y- = (|0⟩ - i|1⟩)/√2:  ⟨Y⟩ = -1,  ⟨X⟩ = ⟨Z⟩ = 0
//
// and ⟨I⟩ = 1 in every state.

/// Label string → [`ProductBasis`] in qiskit's `from_label` alphabet (`0`/`1`, `+`/`-`, `r`/`l`), character `i` being qubit `i`; kept independent of the binding's parser.
fn basis_from_labels<const W: usize>(labels: &str) -> ProductBasis<W> {
    ProductBasis::<W>::from_axes(labels.chars().map(|ch| match ch {
        '0' => (PauliAxis::Z, false),
        '1' => (PauliAxis::Z, true),
        '+' => (PauliAxis::X, false),
        '-' => (PauliAxis::X, true),
        'r' => (PauliAxis::Y, false),
        'l' => (PauliAxis::Y, true),
        other => panic!("unexpected label {other:?}"),
    }))
}

/// `⟨ψ|O|ψ⟩` one qubit at a time from the Bloch table above.
fn naive_labelled_expectation<const W: usize>(sum: &PauliSum<W>, labels: &str) -> Complex64 {
    let mut total = Complex64::new(0.0, 0.0);
    for (x, z, c) in sum.iter() {
        let mut factor = 1.0f64;
        for (q, label) in labels.chars().enumerate() {
            let bx = (x[q / 64] >> (q % 64)) & 1 == 1;
            let bz = (z[q / 64] >> (q % 64)) & 1 == 1;
            factor *= match (bx, bz, label) {
                (false, false, _) => 1.0, // identity factor: no constraint
                (true, false, '+') => 1.0,
                (true, false, '-') => -1.0,
                (false, true, '0') => 1.0,
                (false, true, '1') => -1.0,
                (true, true, 'r') => 1.0,
                (true, true, 'l') => -1.0,
                _ => 0.0, // off-axis Pauli: zero overlap
            };
        }
        total += c * factor;
    }
    total
}

fn expect_close<const W: usize>(sum: &PauliSum<W>, labels: &str, want: f64) {
    let got = sum.expectation_product(&basis_from_labels::<W>(labels));
    assert!(
        (got - Complex64::new(want, 0.0)).norm() < 1e-12,
        "state {labels:?}: got {got}, want {want}",
    );
}

fn single_qubit_labels_against_every_pauli<const W: usize>() {
    // (label, ⟨I⟩, ⟨X⟩, ⟨Y⟩, ⟨Z⟩) — the Bloch table above, transposed.
    let cases = [
        ('0', 1.0, 0.0, 0.0, 1.0),
        ('1', 1.0, 0.0, 0.0, -1.0),
        ('+', 1.0, 1.0, 0.0, 0.0),
        ('-', 1.0, -1.0, 0.0, 0.0),
        ('r', 1.0, 0.0, 1.0, 0.0),
        ('l', 1.0, 0.0, -1.0, 0.0),
    ];
    for (label, ei, ex, ey, ez) in cases {
        let labels = label.to_string();
        for (pauli, want) in [("I", ei), ("X", ex), ("Y", ey), ("Z", ez)] {
            let s = PauliSum::<W>::from_strings(&[(pauli, Complex64::new(1.0, 0.0))]);
            let got = s.expectation_product(&basis_from_labels::<W>(&labels));
            assert!(
                (got - Complex64::new(want, 0.0)).norm() < 1e-12,
                "⟨{label}|{pauli}|{label}⟩ = {got}, want {want} (W={W})",
            );
        }
    }
}

#[test]
fn single_qubit_labels_against_every_pauli_w1() {
    single_qubit_labels_against_every_pauli::<1>();
}

#[test]
fn single_qubit_labels_against_every_pauli_w2() {
    single_qubit_labels_against_every_pauli::<2>();
}

fn multi_qubit_products_compose_per_qubit_signs<const W: usize>() {
    // ⟨01|Z⊗Z|01⟩ = ⟨0|Z|0⟩·⟨1|Z|1⟩ = (+1)(-1) = -1.
    let zz = PauliSum::<W>::from_strings(&[("ZZ", Complex64::new(1.0, 0.0))]);
    expect_close(&zz, "01", -1.0);
    expect_close(&zz, "10", -1.0);
    expect_close(&zz, "00", 1.0);
    expect_close(&zz, "11", 1.0); // (-1)(-1)

    // State |0⟩|+⟩|r⟩: axes Z, X, Y, every sign +1.
    let zxy = PauliSum::<W>::from_strings(&[("ZXY", Complex64::new(1.0, 0.0))]);
    expect_close(&zxy, "0+r", 1.0);
    // X on the Y-axis qubit is off-axis → the whole term drops.
    let zxx = PauliSum::<W>::from_strings(&[("ZXX", Complex64::new(1.0, 0.0))]);
    expect_close(&zxx, "0+r", 0.0);
    // Identity factors are ignored, whatever that qubit's label is.
    let ziy = PauliSum::<W>::from_strings(&[("ZIY", Complex64::new(1.0, 0.0))]);
    expect_close(&ziy, "0+r", 1.0);
    expect_close(&ziy, "0-r", 1.0);
    expect_close(&ziy, "01r", 1.0);

    // State |1⟩|-⟩|l⟩: the same axes with all three signs flipped, so a weight-3 term picks up (-1)^3 = -1.
    expect_close(&zxy, "1-l", -1.0);
    // Only the two non-identity sites' signs count: (-1)·(-1) = +1.
    expect_close(&ziy, "1-l", 1.0);
    // A single flipped site: -1.
    expect_close(&zxy, "1+r", -1.0);
    expect_close(&zxy, "0-r", -1.0);
    expect_close(&zxy, "0+l", -1.0);
    // Two flipped sites: +1.
    expect_close(&zxy, "1-r", 1.0);
}

#[test]
fn multi_qubit_products_compose_per_qubit_signs_w1() {
    multi_qubit_products_compose_per_qubit_signs::<1>();
}

#[test]
fn multi_qubit_products_compose_per_qubit_signs_w2() {
    multi_qubit_products_compose_per_qubit_signs::<2>();
}

fn an_off_axis_pauli_never_matches<const W: usize>() {
    // `X` on a Y-axis qubit must not contribute although the Y axis has its x-bit set: ⟨r|X|r⟩ = 0.
    let off_axis = [
        ('r', "X"),
        ('r', "Z"),
        ('l', "X"),
        ('l', "Z"),
        ('+', "Y"),
        ('+', "Z"),
        ('-', "Y"),
        ('0', "X"),
        ('0', "Y"),
        ('1', "X"),
        ('1', "Y"),
    ];
    for (label, pauli) in off_axis {
        let s = PauliSum::<W>::from_strings(&[(pauli, Complex64::new(3.0, -4.0))]);
        let got = s.expectation_product(&basis_from_labels::<W>(&label.to_string()));
        assert!(got.norm() < 1e-12, "⟨{label}|{pauli}|{label}⟩ = {got}, want 0 (W={W})",);
    }
    // Mixed: one off-axis factor kills a term whose other factors match.
    let s = PauliSum::<W>::from_strings(&[("XXX", Complex64::new(1.0, 0.0))]);
    expect_close(&s, "++r", 0.0);
    expect_close(&s, "+++", 1.0);
}

#[test]
fn an_off_axis_pauli_never_matches_w1() {
    an_off_axis_pauli_never_matches::<1>();
}

#[test]
fn an_off_axis_pauli_never_matches_w2() {
    an_off_axis_pauli_never_matches::<2>();
}

#[test]
fn labelled_expectation_is_linear_and_keeps_the_imaginary_part() {
    // ⟨1|Z|1⟩ = -1 and ⟨1|I|1⟩ = +1, so this is -(1+2i) + (3-5i).
    let s = PauliSum::<1>::from_strings(&[
        ("Z", Complex64::new(1.0, 2.0)),
        ("I", Complex64::new(3.0, -5.0)),
    ]);
    let got = s.expectation_product(&basis_from_labels::<1>("1"));
    assert!((got - Complex64::new(2.0, -7.0)).norm() < 1e-12, "{got}");
}

#[test]
fn labels_across_the_word_boundary_are_independent_w2() {
    // 128 qubits, |0…0⟩ except qubit 64 in |1⟩, whose sign bit lives in word 1.
    let mut labels: String = "0".repeat(128);
    labels.replace_range(64..65, "1");
    let basis = basis_from_labels::<2>(&labels);
    let cases = [
        (PauliString::<2>::z(0), 1.0),   // qubit 0 is |0⟩
        (PauliString::<2>::z(64), -1.0), // qubit 64 is |1⟩
        (PauliString::<2>::x(64), 0.0),  // off-axis on a Z qubit
    ];
    for (p, want) in cases {
        let s = build::<2>(128, &[(p, Complex64::new(1.0, 0.0))]);
        let got = s.expectation_product(&basis);
        assert!((got - Complex64::new(want, 0.0)).norm() < 1e-12, "{p:?}: got {got}, want {want}",);
    }
    // Z on both sides of the boundary: (+1)·(-1) = -1.
    let mut z0z64 = PauliString::<2>::z(0);
    z0z64.mul_assign(&PauliString::<2>::z(64));
    let s = build::<2>(128, &[(z0z64, Complex64::new(1.0, 0.0))]);
    let got = s.expectation_product(&basis);
    assert!((got - Complex64::new(-1.0, 0.0)).norm() < 1e-12, "{got}");
}

#[test]
fn labelled_expectation_of_an_empty_sum_is_zero() {
    let h = Gf2Hash::<1>::new(8, 3, 0xE6);
    let b = PauliSum::<1>::empty_with_hash(8, h);
    let got = b.expectation_product(&basis_from_labels::<1>("01+-rl01"));
    assert!(got.norm() < 1e-15, "{got}");
}

fn uniform_states_agree_with_their_label_spellings<const W: usize>() {
    let num_qubits = 50 * W;
    let sum = rand_sum::<W>(4000, num_qubits, 0xA40 + W as u64);
    for (state, label) in [(PauliAxis::X, '+'), (PauliAxis::Y, 'r'), (PauliAxis::Z, '0')] {
        let want = sum.expectation_product(&ProductBasis::uniform(state));
        let labels: String = std::iter::repeat_n(label, num_qubits).collect();
        let got = sum.expectation_product(&basis_from_labels::<W>(&labels));
        assert!((got - want).norm() < 1e-12, "{state:?} vs {label:?}: {got} vs {want} (W={W})",);
    }
}

#[test]
fn uniform_states_agree_with_their_label_spellings_w1() {
    uniform_states_agree_with_their_label_spellings::<1>();
}

#[test]
fn uniform_states_agree_with_their_label_spellings_w2() {
    uniform_states_agree_with_their_label_spellings::<2>();
}

fn labelled_expectation_agrees_with_the_naive_reference<const W: usize>() {
    // 33 qubits per word so W=2 straddles the boundary at 64.
    let num_qubits = 33 * W;
    let alphabet: Vec<char> = "01+-rl".chars().collect();
    let mut rng = Xs64::new(0xB40 + W as u64);
    let labels: String = (0..num_qubits).map(|_| alphabet[(rng.next_u64() % 6) as usize]).collect();
    for &weight in &[1usize, 2, 3] {
        let sum = low_weight_sum::<W>(3000, num_qubits, weight, 0xB50 + weight as u64);
        let want = naive_labelled_expectation(&sum, &labels);
        let got = sum.expectation_product(&basis_from_labels::<W>(&labels));
        assert!((got - want).norm() < 1e-9, "W={W} weight={weight}: {got} vs {want}",);
    }
}

#[test]
fn labelled_expectation_agrees_with_the_naive_reference_w1() {
    labelled_expectation_agrees_with_the_naive_reference::<1>();
}

#[test]
fn labelled_expectation_agrees_with_the_naive_reference_w2() {
    labelled_expectation_agrees_with_the_naive_reference::<2>();
}

#[test]
fn bucketed_labelled_expectation_agrees_across_partitions() {
    // A partition change must not move the value beyond float re-association.
    let alphabet: Vec<char> = "01+-rl".chars().collect();
    let mut rng = Xs64::new(0xB60);
    let labels: String = (0..100).map(|_| alphabet[(rng.next_u64() % 6) as usize]).collect();
    let basis = basis_from_labels::<2>(&labels);
    let sum = low_weight_sum::<2>(20_000, 100, 3, 0xB61);
    let want = sum.expectation_product(&basis);
    for bits in [0u8, 3, 7, 11] {
        let h = Gf2Hash::<2>::new(100, bits, 0xB62);
        let b = sum.clone().with_hash(h);
        let got = b.expectation_product(&basis);
        assert!((got - want).norm() < 1e-9, "bits={bits}: {got} vs {want}");
    }
}
