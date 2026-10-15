use super::*;
use crate::pauli_sum::PauliSum;
use crate::test_support::rand_sum;
use num_complex::Complex64;

/// Parse `"+XZY"` / `"-XZY"` / `"XZY"` into `(key, minus)`; character `i` is qubit `i`, Hermitian convention (`Y = (1, 1)`, no phase).
fn gen_of<const W: usize>(s: &str) -> (PauliString<W>, bool) {
    let (minus, body) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let mut p = PauliString::<W>::identity();
    for (i, ch) in body.chars().enumerate() {
        let w = i / 64;
        let bit = 1u64 << (i % 64);
        match ch {
            'I' => {}
            'X' => p.x[w] |= bit,
            'Z' => p.z[w] |= bit,
            'Y' => {
                p.x[w] |= bit;
                p.z[w] |= bit;
            }
            other => panic!("unexpected Pauli char {other:?}"),
        }
    }
    (p, minus)
}

fn state<const W: usize>(num_qubits: usize, gens: &[&str]) -> StabilizerState<W> {
    let g: Vec<_> = gens.iter().map(|s| gen_of::<W>(s)).collect();
    StabilizerState::<W>::from_generators(num_qubits, &g).expect("valid generators")
}

/// `⟨ψ|key|ψ⟩`: `0.0`, `1.0` or `-1.0`.
fn expectation_of<const W: usize>(stabilizer: &StabilizerState<W>, key: &PauliString<W>) -> f64 {
    match stabilizer.sign_of(key) {
        None => 0.0,
        Some(false) => 1.0,
        Some(true) => -1.0,
    }
}

/// `⟨ψ|P|ψ⟩` for a single Pauli string given as a label.
fn expect<const W: usize>(stabilizer: &StabilizerState<W>, label: &str) -> f64 {
    expectation_of(stabilizer, &gen_of::<W>(label).0)
}

#[test]
fn anticommuting_generators_are_rejected() {
    let g = [gen_of::<1>("XI"), gen_of::<1>("ZI")];
    assert_eq!(
        StabilizerState::<1>::from_generators(2, &g).unwrap_err(),
        StabilizerError::NotCommuting { first: 0, second: 1 },
    );
}

#[test]
fn dependent_generators_are_rejected() {
    let g = [gen_of::<1>("ZI"), gen_of::<1>("ZI")];
    assert_eq!(
        StabilizerState::<1>::from_generators(2, &g).unwrap_err(),
        StabilizerError::Dependent { generator: 1 },
    );
}

/// `(+ZI)·(-ZI) = -II`, and `-I` stabilizes nothing. GF(2) dependence catches it.
#[test]
fn opposite_signs_on_one_key_are_rejected() {
    let g = [gen_of::<1>("+ZI"), gen_of::<1>("-ZI")];
    assert_eq!(
        StabilizerState::<1>::from_generators(2, &g).unwrap_err(),
        StabilizerError::Dependent { generator: 1 },
    );
}

/// A rank-2 set on 3 qubits: `ZZI` is `ZZZ·IIZ`.
#[test]
fn a_rank_deficient_triple_is_rejected() {
    let g = [gen_of::<1>("ZZZ"), gen_of::<1>("IIZ"), gen_of::<1>("ZZI")];
    assert_eq!(
        StabilizerState::<1>::from_generators(3, &g).unwrap_err(),
        StabilizerError::Dependent { generator: 2 },
    );
}

#[test]
fn wrong_generator_count_is_rejected() {
    let g = [gen_of::<1>("ZI")];
    assert_eq!(
        StabilizerState::<1>::from_generators(2, &g).unwrap_err(),
        StabilizerError::GeneratorCount { expected: 2, found: 1 },
    );
    let g3 = [gen_of::<1>("ZI"), gen_of::<1>("IZ"), gen_of::<1>("ZZ")];
    assert_eq!(
        StabilizerState::<1>::from_generators(2, &g3).unwrap_err(),
        StabilizerError::GeneratorCount { expected: 2, found: 3 },
    );
}

#[test]
fn a_generator_outside_num_qubits_is_rejected() {
    let g = [(PauliString::<1>::z(0), false), (PauliString::<1>::z(5), false)];
    assert_eq!(
        StabilizerState::<1>::from_generators(2, &g).unwrap_err(),
        StabilizerError::QubitOutOfRange { generator: 1, num_qubits: 2 },
    );
}

#[test]
fn errors_display_without_panicking() {
    let e = StabilizerError::NotCommuting { first: 0, second: 1 };
    assert!(e.to_string().contains("anticommute"));
}

/// A minus sign on qubit 64's generator: `|0…0 1 0…⟩` with the flip in the second word.
#[test]
fn a_minus_generator_in_the_second_word() {
    let n = 70;
    let mut gens: Vec<(PauliString<2>, bool)> = Vec::with_capacity(n);
    for q in 0..n {
        gens.push((PauliString::<2>::z(q), q == 64));
    }
    let stabilizer = StabilizerState::<2>::from_generators(n, &gens).unwrap();
    assert_eq!(expectation_of(&stabilizer, &PauliString::<2>::z(64)), -1.0);
    assert_eq!(expectation_of(&stabilizer, &PauliString::<2>::z(63)), 1.0);
    let mut z63_64 = PauliString::<2>::z(63);
    z63_64.mul_assign(&PauliString::<2>::z(64));
    assert_eq!(expectation_of(&stabilizer, &z63_64), -1.0);
}

/// Diagonal generators `+Z_q` describe `|0…0⟩`, whose expectation the existing product-state scan already computes — an independent oracle.
fn product_generators<const W: usize>(num_qubits: usize, axis: char) -> StabilizerState<W> {
    let gens: Vec<(PauliString<W>, bool)> = (0..num_qubits)
        .map(|q| {
            let p = match axis {
                'X' => PauliString::<W>::x(q),
                'Y' => PauliString::<W>::y(q),
                _ => PauliString::<W>::z(q),
            };
            (p, false)
        })
        .collect();
    StabilizerState::<W>::from_generators(num_qubits, &gens).unwrap()
}

/// Linear in the coefficients, imaginary parts included: the Bell state picks out `XX` (`+1`), `ZZ` (`+1`) and `YY` (`-1`) and drops `ZI`.
#[test]
fn the_contraction_is_linear_and_keeps_the_imaginary_part() {
    let sum = PauliSum::<1>::from_strings(&[
        ("XX", Complex64::new(2.0, 1.0)),
        ("ZZ", Complex64::new(0.5, 0.0)),
        ("YY", Complex64::new(4.0, 2.0)),
        ("ZI", Complex64::new(100.0, 100.0)),
    ]);
    let bell = state::<1>(2, &["XX", "ZZ"]);
    let got = sum.expectation_stabilizer(&bell);
    // (2 + i) + 0.5 - (4 + 2i) = -1.5 - i
    assert!((got - Complex64::new(-1.5, -1.0)).norm() < 1e-12, "{got} vs -1.5 - 1i",);
}

#[test]
fn an_empty_sum_contracts_to_zero() {
    let sum = PauliSum::<1>::empty(4);
    let stab = product_generators::<1>(4, 'Z');
    assert!(sum.expectation_stabilizer(&stab).norm() < 1e-15);
}

/// `-XX` pivots the elimination of `+YY`: `(-XX)(+YY) = -(XX·YY) = +ZZ`, so the state is `(|00⟩ - |11⟩)/√2` with `⟨XX⟩ = -1`, `⟨YY⟩ = ⟨ZZ⟩ = +1`.
#[test]
fn a_minus_pivot_carries_its_sign_through_the_elimination() {
    let stabilizer = state::<1>(2, &["-XX", "+YY"]);
    assert_eq!(expect(&stabilizer, "XX"), -1.0);
    assert_eq!(expect(&stabilizer, "YY"), 1.0);
    assert_eq!(expect(&stabilizer, "ZZ"), 1.0);
}

/// All `2ⁿ` group elements as `key -> sign`, by multiplying out every subset of the generators; usable only at `n ≲ 12`.
fn brute_force_group<const W: usize>(
    gens: &[(PauliString<W>, bool)],
) -> std::collections::HashMap<([u64; W], [u64; W]), f64> {
    let n = gens.len();
    let mut out = std::collections::HashMap::new();
    for subset in 0u64..(1u64 << n) {
        let mut key = PauliString::<W>::identity();
        let mut phase = Phase::ONE;
        let mut neg = false;
        for (i, (g, gneg)) in gens.iter().enumerate() {
            if subset >> i & 1 == 1 {
                phase += key.mul_assign(g);
                neg ^= gneg;
            }
        }
        assert_eq!(
            phase.exponent() & 1,
            0,
            "subset {subset:b} of commuting Hermitian generators is not Hermitian",
        );
        let sign = if neg ^ (phase == Phase::MINUS_ONE) { -1.0 } else { 1.0 };
        assert!(
            out.insert((key.x, key.z), sign).is_none(),
            "subset {subset:b} repeats a group element: generators are dependent",
        );
    }
    out
}

/// `⟨ψ|O|ψ⟩` term by term against the enumerated group.
fn brute_force_expectation<const W: usize>(
    sum: &PauliSum<W>,
    gens: &[(PauliString<W>, bool)],
) -> Complex64 {
    let group = brute_force_group(gens);
    let mut acc = Complex64::new(0.0, 0.0);
    for (x, z, c) in sum.iter() {
        if let Some(&s) = group.get(&(*x, *z)) {
            acc += s * c;
        }
    }
    acc
}

/// A 1-D cluster state: `K_q = Z_{q-1} X_q Z_{q+1}` (open boundaries), with the sign of generator `q` taken from `signs`.
fn cluster_generators<const W: usize>(
    num_qubits: usize,
    signs: &[bool],
) -> Vec<(PauliString<W>, bool)> {
    (0..num_qubits)
        .map(|q| {
            let mut p = PauliString::<W>::x(q);
            if q > 0 {
                p.mul_assign(&PauliString::<W>::z(q - 1));
            }
            if q + 1 < num_qubits {
                p.mul_assign(&PauliString::<W>::z(q + 1));
            }
            (p, signs[q])
        })
        .collect()
}

#[test]
fn cluster_state_contraction_matches_the_brute_force_group_w1() {
    let n = 8;
    for seed in [0xC0u64, 0xC1, 0xC2] {
        let signs: Vec<bool> = (0..n).map(|q| seed >> q & 1 == 1).collect();
        let gens = cluster_generators::<1>(n, &signs);
        let stab = StabilizerState::<1>::from_generators(n, &gens).unwrap();
        let sum = rand_sum::<1>(3000, n, seed);
        let got = sum.expectation_stabilizer(&stab);
        let want = brute_force_expectation(&sum, &gens);
        assert!((got - want).norm() < 1e-12, "seed {seed:#x}: {got} vs {want} (brute-force group)",);
    }
}

/// GHZ, sign-randomized, against the same oracle — the state the hand computations above pin, checked over a whole random sum.
#[test]
fn ghz_contraction_matches_the_brute_force_group() {
    let n = 7;
    let mut gens: Vec<(PauliString<1>, bool)> = Vec::with_capacity(n);
    let mut all_x = PauliString::<1>::identity();
    for q in 0..n {
        all_x.mul_assign(&PauliString::<1>::x(q));
    }
    gens.push((all_x, true));
    for q in 1..n {
        let mut zz = PauliString::<1>::z(q - 1);
        zz.mul_assign(&PauliString::<1>::z(q));
        gens.push((zz, q % 3 == 0));
    }
    let stab = StabilizerState::<1>::from_generators(n, &gens).unwrap();
    let sum = rand_sum::<1>(3000, n, 0xC7);
    let got = sum.expectation_stabilizer(&stab);
    let want = brute_force_expectation(&sum, &gens);
    assert!((got - want).norm() < 1e-12, "{got} vs {want}");
}

/// `sign_of` over the whole Pauli group at small `n`: every group element with its enumerated sign, every non-member `None`.
#[test]
fn sign_of_agrees_with_the_brute_force_group_over_every_pauli() {
    let n = 4;
    let signs = [true, false, true, true];
    let gens = cluster_generators::<1>(n, &signs);
    let stab = StabilizerState::<1>::from_generators(n, &gens).unwrap();
    let group = brute_force_group(&gens);
    let full = 1u64 << n;
    for x in 0..full {
        for z in 0..full {
            let key = PauliString::<1> { x: [x], z: [z] };
            let want = group.get(&([x], [z])).copied().unwrap_or(0.0);
            assert_eq!(expectation_of(&stab, &key), want, "x={x:#b} z={z:#b}: {want} expected",);
        }
    }
    // Sanity: the group really is 2^n of the 4^n Paulis.
    assert_eq!(group.len(), 1 << n);
}

#[test]
#[should_panic(expected = "num_qubits mismatch")]
fn contracting_against_a_differently_sized_state_panics() {
    let sum = PauliSum::<1>::from_strings(&[("XX", Complex64::new(1.0, 0.0))]);
    let stab = product_generators::<1>(3, 'Z');
    let _ = sum.expectation_stabilizer(&stab);
}
