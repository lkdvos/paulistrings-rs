use super::*;
use crate::pauli_sum::PauliSum;
use crate::readout::ProductState;
use crate::test_support::rand_sum;
use crate::Gf2Hash;
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

/// `⟨ψ|P|ψ⟩` for a single Pauli string given as a label.
fn expect<const W: usize>(st: &StabilizerState<W>, label: &str) -> f64 {
    st.expectation_of(&gen_of::<W>(label).0)
}

/// `XX·ZZ = (X·Z)⊗(X·Z) = (-iY)⊗(-iY) = -YY`, so the group is `{+II, +XX, +ZZ, -YY}` and `⟨YY⟩ = -1`. Single-qubit Paulis are outside the group, hence `0`.
#[test]
fn bell_state_expectations_w1() {
    let bell = state::<1>(2, &["+XX", "+ZZ"]);
    assert_eq!(expect(&bell, "II"), 1.0);
    assert_eq!(expect(&bell, "XX"), 1.0);
    assert_eq!(expect(&bell, "ZZ"), 1.0);
    assert_eq!(expect(&bell, "YY"), -1.0);
    assert_eq!(expect(&bell, "ZI"), 0.0);
    assert_eq!(expect(&bell, "IZ"), 0.0);
    assert_eq!(expect(&bell, "XI"), 0.0);
    assert_eq!(expect(&bell, "XZ"), 0.0);
    assert_eq!(expect(&bell, "YI"), 0.0);
}

#[test]
fn bell_state_expectations_w2() {
    let bell = state::<2>(2, &["XX", "ZZ"]);
    assert_eq!(expect(&bell, "XX"), 1.0);
    assert_eq!(expect(&bell, "ZZ"), 1.0);
    assert_eq!(expect(&bell, "YY"), -1.0);
    assert_eq!(expect(&bell, "ZI"), 0.0);
}

/// Flipping one generator's sign flips every group element containing it: `(-XX)(+ZZ) = -(XX·ZZ) = +YY`.
#[test]
fn bell_state_with_a_minus_generator_flips_xx_and_yy() {
    let bell = state::<1>(2, &["-XX", "+ZZ"]);
    assert_eq!(expect(&bell, "XX"), -1.0);
    assert_eq!(expect(&bell, "ZZ"), 1.0);
    assert_eq!(expect(&bell, "YY"), 1.0);
    assert_eq!(expect(&bell, "II"), 1.0);
}

/// `-Z` stabilizes `|1⟩`, so `⟨Z⟩ = -1` and `⟨X⟩ = ⟨Y⟩ = 0`.
#[test]
fn minus_z_generator_is_the_one_state() {
    let one = state::<1>(1, &["-Z"]);
    assert_eq!(expect(&one, "Z"), -1.0);
    assert_eq!(expect(&one, "X"), 0.0);
    assert_eq!(expect(&one, "Y"), 0.0);
    assert_eq!(expect(&one, "I"), 1.0);
}

/// `XXX·ZZI` acts as `X·Z = -iY` on qubits 0 and 1 and as `X·I = X` on qubit 2, so it equals `(-i)²·YYX = -YYX`; the group element being `-YYX` means `YYX|ψ⟩ = -|ψ⟩`.
///
/// `ZZI·IZZ = ZIZ` with no phase (no X-bits meet a Z-bit), so `⟨ZIZ⟩ = 1`. `ZII` is outside the span of `{(x=111,z=000), (000,011), (000,110)}`, hence `0`.
#[test]
fn ghz_state_expectations_w1() {
    let ghz = state::<1>(3, &["+XXX", "+ZZI", "+IZZ"]);
    assert_eq!(expect(&ghz, "XXX"), 1.0);
    assert_eq!(expect(&ghz, "ZZI"), 1.0);
    assert_eq!(expect(&ghz, "IZZ"), 1.0);
    assert_eq!(expect(&ghz, "ZIZ"), 1.0);
    assert_eq!(expect(&ghz, "YYX"), -1.0);
    assert_eq!(expect(&ghz, "YXY"), -1.0);
    assert_eq!(expect(&ghz, "XYY"), -1.0);
    assert_eq!(expect(&ghz, "ZII"), 0.0);
    assert_eq!(expect(&ghz, "XXI"), 0.0);
    assert_eq!(expect(&ghz, "YYY"), 0.0);
    assert_eq!(expect(&ghz, "III"), 1.0);
}

#[test]
fn ghz_state_expectations_w2() {
    let ghz = state::<2>(3, &["XXX", "ZZI", "IZZ"]);
    assert_eq!(expect(&ghz, "XXX"), 1.0);
    assert_eq!(expect(&ghz, "YYX"), -1.0);
    assert_eq!(expect(&ghz, "ZIZ"), 1.0);
    assert_eq!(expect(&ghz, "ZII"), 0.0);
}

/// Generator order must not matter: the same group, listed differently, gives the same signs.
#[test]
fn generator_order_does_not_change_the_state() {
    let a = state::<1>(3, &["XXX", "ZZI", "IZZ"]);
    let b = state::<1>(3, &["IZZ", "XXX", "ZZI"]);
    // ZIZ = ZZI·IZZ is a redundant *spelling* of the same group, too.
    let c = state::<1>(3, &["ZIZ", "IZZ", "XXX"]);
    for label in ["XXX", "YYX", "ZIZ", "ZZI", "ZII", "III", "XXI"] {
        let want = expect(&a, label);
        assert_eq!(expect(&b, label), want, "{label}");
        assert_eq!(expect(&c, label), want, "{label}");
    }
}

#[test]
fn anticommuting_generators_are_rejected() {
    let g = [gen_of::<1>("XI"), gen_of::<1>("ZI")];
    assert_eq!(
        StabilizerState::<1>::from_generators(2, &g).unwrap_err(),
        StabilizerError::NotCommuting {
            first: 0,
            second: 1
        },
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
        StabilizerError::GeneratorCount {
            expected: 2,
            found: 1
        },
    );
    let g3 = [gen_of::<1>("ZI"), gen_of::<1>("IZ"), gen_of::<1>("ZZ")];
    assert_eq!(
        StabilizerState::<1>::from_generators(2, &g3).unwrap_err(),
        StabilizerError::GeneratorCount {
            expected: 2,
            found: 3
        },
    );
}

#[test]
fn a_generator_outside_num_qubits_is_rejected() {
    let g = [
        (PauliString::<1>::z(0), false),
        (PauliString::<1>::z(5), false),
    ];
    assert_eq!(
        StabilizerState::<1>::from_generators(2, &g).unwrap_err(),
        StabilizerError::QubitOutOfRange {
            generator: 1,
            num_qubits: 2
        },
    );
}

#[test]
fn errors_display_without_panicking() {
    let e = StabilizerError::NotCommuting {
        first: 0,
        second: 1,
    };
    assert!(e.to_string().contains("anticommute"));
}

/// A Bell pair straddling the 64-qubit word boundary, with every other qubit in `|0⟩`.
/// Same algebra as `bell_state_expectations_w1`, but the two X-bits and two Z-bits live in different `[u64; 2]` words.
#[test]
fn a_bell_pair_across_the_word_boundary() {
    let n = 66;
    let mut xx = PauliString::<2>::x(63);
    xx.mul_assign(&PauliString::<2>::x(64));
    let mut zz = PauliString::<2>::z(63);
    zz.mul_assign(&PauliString::<2>::z(64));
    let mut yy = PauliString::<2>::y(63);
    yy.mul_assign(&PauliString::<2>::y(64));

    let mut gens: Vec<(PauliString<2>, bool)> = Vec::with_capacity(n);
    for q in 0..n as u32 {
        match q {
            63 => gens.push((xx, false)),
            64 => gens.push((zz, false)),
            _ => gens.push((PauliString::<2>::z(q), false)),
        }
    }
    let st = StabilizerState::<2>::from_generators(n, &gens).expect("valid generators");

    assert_eq!(st.expectation_of(&xx), 1.0);
    assert_eq!(st.expectation_of(&zz), 1.0);
    assert_eq!(st.expectation_of(&yy), -1.0);
    assert_eq!(st.expectation_of(&PauliString::<2>::z(63)), 0.0);
    assert_eq!(st.expectation_of(&PauliString::<2>::z(64)), 0.0);
    assert_eq!(st.expectation_of(&PauliString::<2>::z(0)), 1.0);
    assert_eq!(st.expectation_of(&PauliString::<2>::z(65)), 1.0);
    assert_eq!(st.expectation_of(&PauliString::<2>::x(65)), 0.0);
    assert_eq!(st.expectation_of(&PauliString::<2>::identity()), 1.0);
}

/// A minus sign on qubit 64's generator: `|0…0 1 0…⟩` with the flip in the second word.
#[test]
fn a_minus_generator_in_the_second_word() {
    let n = 70;
    let mut gens: Vec<(PauliString<2>, bool)> = Vec::with_capacity(n);
    for q in 0..n as u32 {
        gens.push((PauliString::<2>::z(q), q == 64));
    }
    let st = StabilizerState::<2>::from_generators(n, &gens).unwrap();
    assert_eq!(st.expectation_of(&PauliString::<2>::z(64)), -1.0);
    assert_eq!(st.expectation_of(&PauliString::<2>::z(63)), 1.0);
    let mut z63_64 = PauliString::<2>::z(63);
    z63_64.mul_assign(&PauliString::<2>::z(64));
    assert_eq!(st.expectation_of(&z63_64), -1.0);
}

/// Diagonal generators `+Z_q` describe `|0…0⟩`, whose expectation the existing product-state scan already computes — an independent oracle.
fn product_generators<const W: usize>(num_qubits: usize, axis: char) -> StabilizerState<W> {
    let gens: Vec<(PauliString<W>, bool)> = (0..num_qubits as u32)
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

#[test]
fn uniform_product_generators_agree_with_the_product_state_scan_w1() {
    let sum = rand_sum::<1>(4000, 20, 0xB0);
    for (axis, st) in [
        ('X', ProductState::XPlus),
        ('Y', ProductState::YPlus),
        ('Z', ProductState::ZPlus),
    ] {
        let stab = product_generators::<1>(20, axis);
        let got = sum.expectation_stabilizer(&stab);
        let want = sum.expectation_product_state(st);
        assert!(
            (got - want).norm() < 1e-12,
            "axis {axis}: {got} vs {want} (product-state oracle)",
        );
    }
}

#[test]
fn uniform_product_generators_agree_with_the_product_state_scan_w2() {
    let sum = rand_sum::<2>(8000, 100, 0xB1);
    for (axis, st) in [
        ('X', ProductState::XPlus),
        ('Y', ProductState::YPlus),
        ('Z', ProductState::ZPlus),
    ] {
        let stab = product_generators::<2>(100, axis);
        let got = sum.expectation_stabilizer(&stab);
        let want = sum.expectation_product_state(st);
        assert!(
            (got - want).norm() < 1e-12,
            "axis {axis}: {got} vs {want} (product-state oracle)",
        );
    }
}

/// The contraction agrees across partitions to floating-point tolerance.
#[test]
fn the_contraction_is_partition_independent() {
    let sum = rand_sum::<1>(5000, 20, 0xB2);
    let stab = product_generators::<1>(20, 'Z');
    let want = sum.expectation_stabilizer(&stab);
    for bits in [0u8, 3, 7] {
        let b = sum.clone().with_hash(Gf2Hash::<1>::new(20, bits, 0xB3));
        let got = b.expectation_stabilizer(&stab);
        assert!((got - want).norm() < 1e-9, "bits={bits}: {got} vs {want}");
    }
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
    assert!(
        (got - Complex64::new(-1.5, -1.0)).norm() < 1e-12,
        "{got} vs -1.5 - 1i",
    );
}

#[test]
fn an_empty_sum_contracts_to_zero() {
    let sum = PauliSum::<1>::empty(4);
    let stab = product_generators::<1>(4, 'Z');
    assert!(sum.expectation_stabilizer(&stab).norm() < 1e-15);
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
        let sign = if neg ^ (phase == Phase::MINUS_ONE) {
            -1.0
        } else {
            1.0
        };
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
            let mut p = PauliString::<W>::x(q as u32);
            if q > 0 {
                p.mul_assign(&PauliString::<W>::z(q as u32 - 1));
            }
            if q + 1 < num_qubits {
                p.mul_assign(&PauliString::<W>::z(q as u32 + 1));
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
        assert!(
            (got - want).norm() < 1e-12,
            "seed {seed:#x}: {got} vs {want} (brute-force group)",
        );
    }
}

#[test]
fn cluster_state_contraction_matches_the_brute_force_group_w2() {
    let n = 6;
    let signs: Vec<bool> = vec![false, true, true, false, false, true];
    let gens = cluster_generators::<2>(n, &signs);
    let stab = StabilizerState::<2>::from_generators(n, &gens).unwrap();
    let sum = rand_sum::<2>(2000, n, 0xC5);
    let got = sum.expectation_stabilizer(&stab);
    let want = brute_force_expectation(&sum, &gens);
    assert!((got - want).norm() < 1e-12, "{got} vs {want}");
}

/// GHZ, sign-randomized, against the same oracle — the state the hand computations above pin, checked over a whole random sum.
#[test]
fn ghz_contraction_matches_the_brute_force_group() {
    let n = 7;
    let mut gens: Vec<(PauliString<1>, bool)> = Vec::with_capacity(n);
    let mut all_x = PauliString::<1>::identity();
    for q in 0..n as u32 {
        all_x.mul_assign(&PauliString::<1>::x(q));
    }
    gens.push((all_x, true));
    for q in 1..n as u32 {
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
            assert_eq!(
                stab.expectation_of(&key),
                want,
                "x={x:#b} z={z:#b}: {want} expected",
            );
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
