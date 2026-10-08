use super::*;

#[test]
fn finalize_empty_accumulator_is_empty() {
    let accumulator = BuildAccumulator::<1>::new(4);
    let s = accumulator.finalize();
    assert!(s.is_empty());
    assert_eq!(s.num_qubits(), 4);
    s.assert_invariants();
}

#[test]
fn add_term_dup_sums_coeffs() {
    let mut accumulator = BuildAccumulator::<1>::new(4);
    let p = PauliString::<1>::x(2);
    accumulator.add_term(p, Phase::ONE, Complex64::new(1.0, 0.0));
    accumulator.add_term(p, Phase::ONE, Complex64::new(2.5, -1.0));
    let s = accumulator.finalize();
    assert_eq!(s.len(), 1);
    assert_eq!(s.bucket(0).2[0], Complex64::new(3.5, -1.0));
    s.assert_invariants();
}

#[test]
fn add_term_phase_one_is_identity_factor() {
    let mut accumulator = BuildAccumulator::<1>::new(4);
    let p = PauliString::<1>::x(0);
    accumulator.add_term(p, Phase::ONE, Complex64::new(2.0, 3.0));
    let s = accumulator.finalize();
    assert_eq!(s.bucket(0).2[0], Complex64::new(2.0, 3.0));
}

#[test]
fn add_term_phase_i_multiplies_by_i() {
    let mut accumulator = BuildAccumulator::<1>::new(4);
    let p = PauliString::<1>::x(0);
    // (2 + 3i) * i = -3 + 2i
    accumulator.add_term(p, Phase::I, Complex64::new(2.0, 3.0));
    let s = accumulator.finalize();
    assert_eq!(s.bucket(0).2[0], Complex64::new(-3.0, 2.0));
}

#[test]
fn add_term_phase_minus_one_negates() {
    let mut accumulator = BuildAccumulator::<1>::new(4);
    let p = PauliString::<1>::x(0);
    accumulator.add_term(p, Phase::MINUS_ONE, Complex64::new(2.0, 3.0));
    let s = accumulator.finalize();
    assert_eq!(s.bucket(0).2[0], Complex64::new(-2.0, -3.0));
}

#[test]
fn add_term_phase_minus_i_multiplies_by_minus_i() {
    let mut accumulator = BuildAccumulator::<1>::new(4);
    let p = PauliString::<1>::x(0);
    // (2 + 3i) * -i = 3 - 2i
    accumulator.add_term(p, Phase::MINUS_I, Complex64::new(2.0, 3.0));
    let s = accumulator.finalize();
    assert_eq!(s.bucket(0).2[0], Complex64::new(3.0, -2.0));
}

#[test]
fn add_term_cancellation_drops() {
    let mut accumulator = BuildAccumulator::<1>::new(4);
    let p = PauliString::<1>::x(0);
    accumulator.add_term(p, Phase::ONE, Complex64::new(1.0, 0.0));
    accumulator.add_term(p, Phase::MINUS_ONE, Complex64::new(1.0, 0.0));
    let s = accumulator.finalize();
    assert!(s.is_empty());
    s.assert_invariants();
}

#[test]
fn finalize_sorts_by_lex_key() {
    // Insert keys out of order and confirm finalize emits them sorted by (x, z) lex: Z(0)=(0,1), X(0)=(1,0), X(1)=(2,0).
    let mut accumulator = BuildAccumulator::<1>::new(4);
    accumulator.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(3.0, 0.0));
    accumulator.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
    accumulator.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(2.0, 0.0));
    let s = accumulator.finalize();
    assert_eq!(s.len(), 3);
    assert_eq!(s.bucket(0).0[0], [0u64]);
    assert_eq!(s.bucket(0).1[0], [1u64]);
    assert_eq!(s.bucket(0).2[0], Complex64::new(1.0, 0.0));
    assert_eq!(s.bucket(0).0[1], [1u64]);
    assert_eq!(s.bucket(0).2[1], Complex64::new(2.0, 0.0));
    assert_eq!(s.bucket(0).0[2], [2u64]);
    assert_eq!(s.bucket(0).2[2], Complex64::new(3.0, 0.0));
    s.assert_invariants();
}

#[test]
fn finalize_w2_across_word_boundary() {
    let mut accumulator = BuildAccumulator::<2>::new(128);
    accumulator.add_term(
        PauliString::<2>::x(64),
        Phase::ONE,
        Complex64::new(1.0, 0.0),
    );
    accumulator.add_term(PauliString::<2>::x(0), Phase::ONE, Complex64::new(2.0, 0.0));
    accumulator.add_term(
        PauliString::<2>::z(127),
        Phase::ONE,
        Complex64::new(3.0, 0.0),
    );
    let s = accumulator.finalize();
    assert_eq!(s.len(), 3);
    s.assert_invariants();
}

#[test]
fn add_term_phase_new_wraps_mod_4() {
    // Phase::new(5) reduces to Phase::I — multiplies by i.
    let mut accumulator = BuildAccumulator::<1>::new(4);
    let p = PauliString::<1>::x(0);
    accumulator.add_term(p, Phase::new(5), Complex64::new(2.0, 0.0));
    let s = accumulator.finalize();
    assert_eq!(s.bucket(0).2[0], Complex64::new(0.0, 2.0));
}

#[test]
fn finalize_gives_one_bucket_at_or_below_1024_terms() {
    // 1024 distinct single-qubit-word keys on 11 qubits (x-patterns 1..=1024).
    let mut accumulator = BuildAccumulator::<1>::new(11);
    for k in 1..=1024u64 {
        accumulator.add_term(
            PauliString::<1> { x: [k], z: [0] },
            Phase::ONE,
            Complex64::new(1.0, 0.0),
        );
    }
    let s = accumulator.finalize();
    assert_eq!(s.len(), 1024);
    assert_eq!(s.num_buckets(), 1, "≤1024 terms must stay single-bucket");
    s.assert_invariants();
}

#[test]
fn finalize_picks_desired_bits_above() {
    use crate::pauli_sum::storage::DEFAULT_TARGET_BUCKET_LEN;
    use crate::pauli_sum::storage::{desired_bits, DEFAULT_MIN_BUCKETS};
    let mut accumulator = BuildAccumulator::<1>::new(12);
    for k in 1..=1500u64 {
        accumulator.add_term(
            PauliString::<1> { x: [k], z: [0] },
            Phase::ONE,
            Complex64::new(1.0, 0.0),
        );
    }
    let s = accumulator.finalize();
    assert_eq!(s.len(), 1500);
    let want_bits = desired_bits(1500, DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS);
    assert!(want_bits > 0, "1500 terms must split");
    assert_eq!(s.num_buckets(), 1usize << want_bits);
    s.assert_invariants();
}

#[test]
fn finalize_with_capacity_preallocated() {
    let mut accumulator = BuildAccumulator::<1>::with_capacity(4, 16);
    accumulator.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(1.0, 0.0));
    let s = accumulator.finalize();
    assert_eq!(s.len(), 1);
    s.assert_invariants();
}
