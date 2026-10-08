use super::*;
use crate::channel::clifford::{Clifford1Q, Clifford2Q};
use crate::channel::identity::IdentityChannel;
use crate::channel::noise::{AmplitudeDamping, Dephasing, Depolarizing};
use crate::channel::rotation::PauliRotation;
use crate::channel::unitary::{GeneralUnitary1Q, GeneralUnitary2Q};

use crate::test_support::{haar_su4_matrix, sqrt_swap_matrix, Xs64};

const TOL: f64 = 1e-12;

type Term<const W: usize> = ([u64; W], [u64; W], Complex64);

/// Outputs of `apply` / `apply_adjoint`, with exact zeros dropped and equal keys summed — the same normalization the merge phase performs.
fn via_apply<const W: usize, C: Channel<W> + ?Sized>(
    ch: &C,
    adjoint: bool,
    x: &[u64; W],
    z: &[u64; W],
    coeff: Complex64,
) -> Vec<Term<W>> {
    let f = ch.max_fanout().max(1);
    let mut bx = vec![[0u64; W]; f];
    let mut bz = vec![[0u64; W]; f];
    let mut bc = vec![ZERO; f];
    let mut len = 0usize;
    {
        let mut out = OutputBuffer::<W> {
            x: &mut bx,
            z: &mut bz,
            coeff: &mut bc,
            len: &mut len,
        };
        if adjoint {
            ch.apply_adjoint(x, z, coeff, &mut out);
        } else {
            ch.apply(x, z, coeff, &mut out);
        }
    }
    normalize((0..len).map(|i| (bx[i], bz[i], bc[i])).collect())
}

/// The same outputs, reconstructed from the prepared table.
fn via_prepared<const W: usize>(
    prep: &Prepared<W>,
    x: &[u64; W],
    z: &[u64; W],
    coeff: Complex64,
) -> Vec<Term<W>> {
    let mut out: Vec<Term<W>> = Vec::new();
    match prep {
        Prepared::Local(p) => {
            let s = p.support_bits(x, z);
            for m in p.deltas() {
                let a = m.amp[s];
                if a == ZERO {
                    continue;
                }
                let mut ox = *x;
                let mut oz = *z;
                for w in 0..W {
                    ox[w] ^= m.mask_x[w];
                    oz[w] ^= m.mask_z[w];
                }
                out.push((ox, oz, coeff * a));
            }
        }
        Prepared::Rotation(r) => {
            let input = PauliString::<W> { x: *x, z: *z };
            if input.commutes_with(&r.gen) {
                out.push((*x, *z, coeff));
            } else {
                out.push((*x, *z, coeff * r.cos));
                let mut prod = input;
                let phase = prod.mul_assign(&r.gen);
                let total = crate::phase::Phase::I + phase;
                out.push((prod.x, prod.z, total.apply(coeff) * r.sin));
            }
        }
    }
    normalize(out)
}

fn normalize<const W: usize>(mut v: Vec<Term<W>>) -> Vec<Term<W>> {
    v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    let mut out: Vec<Term<W>> = Vec::new();
    for (x, z, c) in v {
        match out.last_mut() {
            Some(last) if last.0 == x && last.1 == z => last.2 += c,
            _ => out.push((x, z, c)),
        }
    }
    out.retain(|t| t.2 != ZERO);
    out
}

fn assert_terms_eq<const W: usize>(a: &[Term<W>], b: &[Term<W>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: term count {a:?} vs {b:?}");
    for (p, q) in a.iter().zip(b.iter()) {
        assert_eq!(p.0, q.0, "{what}: x key");
        assert_eq!(p.1, q.1, "{what}: z key");
        assert!((p.2 - q.2).norm() < TOL, "{what}: coeff {} vs {}", p.2, q.2);
    }
}

/// The derived table reproduces `apply` on randomized full-width inputs, not just the basis probes.
fn check_agrees_on_random_inputs<const W: usize, C: Channel<W>>(
    ch: &C,
    num_qubits: usize,
    label: &str,
) {
    let hash = Gf2Hash::<W>::new(num_qubits, 10, 0xD00D);
    for &adjoint in &[false, true] {
        let prep = ch
            .prepare(&hash, adjoint)
            .unwrap_or_else(|| panic!("{label}: prepare returned None"));
        let mut rng = Xs64::new(0x5EED ^ (adjoint as u64));
        for _ in 0..400 {
            let mut x = [0u64; W];
            let mut z = [0u64; W];
            for w in 0..W {
                x[w] = rng.next_u64();
                z[w] = rng.next_u64();
            }
            let coeff = Complex64::new(
                (rng.next_u64() as i64 as f64) / (i64::MAX as f64),
                (rng.next_u64() as i64 as f64) / (i64::MAX as f64),
            );
            let direct = via_apply(ch, adjoint, &x, &z, coeff);
            let table = via_prepared(&prep, &x, &z, coeff);
            assert_terms_eq(&direct, &table, &format!("{label} adjoint={adjoint}"));
        }
    }
}

#[test]
fn derived_table_matches_apply_identity() {
    check_agrees_on_random_inputs::<2, _>(&IdentityChannel::new(), 128, "identity");
}

#[test]
fn derived_table_matches_apply_clifford1q() {
    for (name, ch) in [
        ("h", Clifford1Q::h(3)),
        ("s", Clifford1Q::s(3)),
        ("x", Clifford1Q::x(3)),
        ("y", Clifford1Q::y(3)),
        ("z", Clifford1Q::z(3)),
    ] {
        check_agrees_on_random_inputs::<2, _>(&ch, 128, name);
    }
    // Also across a word boundary.
    check_agrees_on_random_inputs::<2, _>(&Clifford1Q::h(70), 128, "h@70");
}

#[test]
fn derived_table_matches_apply_clifford2q() {
    for (name, ch) in [
        ("cnot", Clifford2Q::cnot(1, 4)),
        ("cz", Clifford2Q::cz(1, 4)),
        ("swap", Clifford2Q::swap(1, 4)),
    ] {
        check_agrees_on_random_inputs::<2, _>(&ch, 128, name);
    }
    // Straddling the word boundary.
    check_agrees_on_random_inputs::<2, _>(&Clifford2Q::cnot(60, 70), 128, "cnot@60,70");
}

#[test]
fn derived_table_matches_apply_noise() {
    check_agrees_on_random_inputs::<2, _>(
        &Depolarizing {
            support: [5],
            p: 0.07,
        },
        128,
        "depolarizing",
    );
    check_agrees_on_random_inputs::<2, _>(
        &Dephasing {
            support: [5],
            p: 0.07,
        },
        128,
        "dephasing",
    );
    check_agrees_on_random_inputs::<2, _>(
        &AmplitudeDamping {
            support: [5],
            gamma: 0.3,
        },
        128,
        "amplitude_damping",
    );
}

#[test]
fn derived_table_matches_apply_rotation_weight_1_and_2() {
    check_agrees_on_random_inputs::<2, _>(
        &PauliRotation::new(PauliString::<2>::z(9), 0.37),
        128,
        "rot_z",
    );
    check_agrees_on_random_inputs::<2, _>(
        &PauliRotation::new(PauliString::<2>::y(70), 0.37),
        128,
        "rot_y@70",
    );
    let mut zz = PauliString::<2>::z(9);
    zz.mul_assign(&PauliString::<2>::z(70));
    check_agrees_on_random_inputs::<2, _>(&PauliRotation::new(zz, 0.37), 128, "rot_zz");
}

#[test]
fn functional_form_matches_apply_for_a_wide_rotation() {
    // Weight 4 > MAX_LOCAL_SUPPORT, so this takes the Rotation variant.
    let mut gen = PauliString::<2>::z(1);
    for q in [5u32, 66, 100] {
        gen.mul_assign(&PauliString::<2>::z(q));
    }
    let rot = PauliRotation::new(gen, 0.41);
    assert_eq!(rot.weight(), 4);
    check_agrees_on_random_inputs::<2, _>(&rot, 128, "rot_weight4");
}

/// The same outputs, reconstructed one entry at a time through the emitters.
/// Mirrors `via_prepared`, but routes every row through [`DeltaEntry::emit`] / [`RotationPrep::emit_gen`].
fn via_emit<const W: usize>(
    prep: &Prepared<W>,
    x: &[u64; W],
    z: &[u64; W],
    coeff: Complex64,
) -> Vec<Term<W>> {
    let mut out: Vec<Term<W>> = Vec::new();
    match prep {
        Prepared::Local(p) => {
            let s = p.support_bits(x, z);
            for m in p.deltas() {
                if let Some(row) = m.emit(s, x, z, coeff) {
                    out.push(row);
                }
            }
        }
        Prepared::Rotation(r) => {
            // Entry 0, the identity pass, has no `DeltaEntry`: every term emits one such row, full coefficient when it commutes and `cos`-scaled when it does not.
            let input = PauliString::<W> { x: *x, z: *z };
            let commutes = input.commutes_with(&r.gen);
            out.push((*x, *z, if commutes { coeff } else { coeff * r.cos }));
            let gen_row = r.emit_gen(x, z, coeff);
            assert_eq!(
                gen_row.is_none(),
                commutes,
                "emit_gen must be None exactly when the term commutes",
            );
            if let Some(row) = gen_row {
                out.push(row);
            }
        }
    }
    normalize(out)
}

fn check_emit_agrees_on_random_inputs<const W: usize, C: Channel<W>>(
    ch: &C,
    num_qubits: usize,
    label: &str,
) {
    let hash = Gf2Hash::<W>::new(num_qubits, 10, 0xD00D);
    for &adjoint in &[false, true] {
        let prep = ch
            .prepare(&hash, adjoint)
            .unwrap_or_else(|| panic!("{label}: prepare returned None"));
        let mut rng = Xs64::new(0x5EED ^ (adjoint as u64));
        for _ in 0..400 {
            let mut x = [0u64; W];
            let mut z = [0u64; W];
            for w in 0..W {
                x[w] = rng.next_u64();
                z[w] = rng.next_u64();
            }
            let coeff = Complex64::new(
                (rng.next_u64() as i64 as f64) / (i64::MAX as f64),
                (rng.next_u64() as i64 as f64) / (i64::MAX as f64),
            );
            let direct = via_apply(ch, adjoint, &x, &z, coeff);
            let emitted = via_emit(&prep, &x, &z, coeff);
            assert_terms_eq(&direct, &emitted, &format!("{label} adjoint={adjoint}"));
        }
    }
}

#[test]
fn emit_matches_apply_for_cliffords() {
    for (name, ch) in [
        ("h", Clifford1Q::h(3)),
        ("s", Clifford1Q::s(3)),
        ("x", Clifford1Q::x(3)),
    ] {
        check_emit_agrees_on_random_inputs::<2, _>(&ch, 128, name);
    }
    for (name, ch) in [
        ("cnot", Clifford2Q::cnot(1, 4)),
        ("cz", Clifford2Q::cz(1, 4)),
        ("swap", Clifford2Q::swap(1, 4)),
    ] {
        check_emit_agrees_on_random_inputs::<2, _>(&ch, 128, name);
    }
}

#[test]
fn emit_matches_apply_for_general_unitaries() {
    // T gate: a non-Clifford 1Q PTM.
    let t = GeneralUnitary1Q::from_matrix(
        2,
        [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [
                Complex64::new(0.0, 0.0),
                Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
            ],
        ],
    );
    check_emit_agrees_on_random_inputs::<2, _>(&t, 128, "t_gate");

    check_emit_agrees_on_random_inputs::<2, _>(
        &GeneralUnitary2Q::from_matrix(1, 5, sqrt_swap_matrix()),
        128,
        "sqrt_swap",
    );
    // The dense fixture: all sixteen entries carry a nonzero amplitude.
    check_emit_agrees_on_random_inputs::<2, _>(
        &GeneralUnitary2Q::from_matrix(1, 5, haar_su4_matrix()),
        128,
        "haar_su4",
    );
}

#[test]
fn emit_matches_apply_for_noise() {
    check_emit_agrees_on_random_inputs::<2, _>(
        &AmplitudeDamping {
            support: [5],
            gamma: 0.3,
        },
        128,
        "amplitude_damping",
    );
}

#[test]
fn emit_matches_apply_for_rotations_at_every_width() {
    // Weight 1 and 2 take the tabulated path...
    check_emit_agrees_on_random_inputs::<2, _>(
        &PauliRotation::new(PauliString::<2>::z(9), 0.37),
        128,
        "rot_z",
    );
    let mut zz = PauliString::<2>::z(9);
    zz.mul_assign(&PauliString::<2>::z(70));
    check_emit_agrees_on_random_inputs::<2, _>(&PauliRotation::new(zz, 0.37), 128, "rot_zz");
    // ...weight 4 takes `Prepared::Rotation`, so `emit_gen` is exercised.
    let mut gen = PauliString::<2>::z(1);
    for q in [5u32, 66, 100] {
        gen.mul_assign(&PauliString::<2>::z(q));
    }
    let rot = PauliRotation::new(gen, 0.41);
    assert_eq!(rot.weight(), 4);
    check_emit_agrees_on_random_inputs::<2, _>(&rot, 128, "rot_weight4");
}

#[test]
fn emit_returns_none_exactly_on_a_zero_amplitude() {
    // CNOT's identity entry is nonzero on 4 of the 16 support patterns, so both branches of `emit` are reachable from one table.
    let hash = Gf2Hash::<2>::new(128, 8, 0x1);
    let Prepared::Local(p) = Clifford2Q::cnot(1, 4).prepare(&hash, false).unwrap() else {
        panic!("expected Local")
    };
    let one = Complex64::new(1.0, 0.0);
    let x = [0u64; 2];
    let z = [0u64; 2];
    let mut zero_seen = false;
    let mut some_seen = false;
    for m in p.deltas() {
        for s in 0..16usize {
            let got = m.emit(s, &x, &z, one);
            assert_eq!(
                got.is_none(),
                m.amp[s] == ZERO,
                "emit must be None exactly on a zero amplitude",
            );
            if got.is_none() {
                zero_seen = true;
            } else {
                some_seen = true;
            }
        }
    }
    assert!(zero_seen && some_seen, "both branches must be exercised");
}

#[test]
fn mask_returns_the_entrys_lifted_delta() {
    let hash = Gf2Hash::<2>::new(128, 8, 0x1);
    let Prepared::Local(p) = Clifford1Q::h(3).prepare(&hash, false).unwrap() else {
        panic!("expected Local")
    };
    // H's delta set is {0, XZ on qubit 3}.
    let masks: Vec<([u64; 2], [u64; 2])> = p.deltas().iter().map(|m| m.mask()).collect();
    assert_eq!(masks, vec![([0, 0], [0, 0]), ([1 << 3, 0], [1 << 3, 0])]);
}

#[test]
fn rotation_gen_mask_is_the_generator() {
    let mut gen = PauliString::<2>::z(1);
    for q in [5u32, 66, 100] {
        gen.mul_assign(&PauliString::<2>::z(q));
    }
    let hash = Gf2Hash::<2>::new(128, 8, 0x1);
    let Prepared::Rotation(r) = PauliRotation::new(gen, 0.41).prepare(&hash, false).unwrap() else {
        panic!("expected Rotation")
    };
    assert_eq!(r.gen_mask(), (gen.x, gen.z));
}

fn assert_entry_eq<const W: usize>(a: &DeltaEntry<W>, b: &DeltaEntry<W>, what: &str) {
    assert_eq!(a.bucket_delta, b.bucket_delta, "{what}: bucket_delta");
    assert_eq!(a.local_delta, b.local_delta, "{what}: local_delta");
    assert_eq!(a.mask_x, b.mask_x, "{what}: mask_x");
    assert_eq!(a.mask_z, b.mask_z, "{what}: mask_z");
    assert_eq!(a.amp, b.amp, "{what}: amp");
}

#[test]
fn retain_entries_keeps_the_selected_entries_in_order() {
    let hash = Gf2Hash::<2>::new(128, 8, 0x1);
    let Prepared::Local(p) = Clifford2Q::swap(1, 4).prepare(&hash, false).unwrap() else {
        panic!("expected Local")
    };
    assert_eq!(p.num_deltas(), 4);
    let keep = [true, false, true, false];
    let sub = p.retain_entries(&keep);
    assert_eq!(sub.k(), p.k());
    assert_eq!(sub.qubits(), p.qubits());
    assert_eq!(sub.num_deltas(), 2);
    assert_entry_eq(&sub.deltas()[0], &p.deltas()[0], "entry 0");
    assert_entry_eq(&sub.deltas()[1], &p.deltas()[2], "entry 1");
}

#[test]
fn a_retain_to_the_identity_alone_is_key_preserving() {
    let hash = Gf2Hash::<2>::new(128, 8, 0x1);
    let Prepared::Local(p) = Clifford2Q::cnot(1, 4).prepare(&hash, false).unwrap() else {
        panic!("expected Local")
    };
    assert!(!p.is_key_preserving());
    // Entry 0 is the identity by the ascending-`local_delta` construction.
    assert_eq!(p.deltas()[0].local_delta, 0);
    let keep: Vec<bool> = (0..p.num_deltas()).map(|e| e == 0).collect();
    assert!(p.retain_entries(&keep).is_key_preserving());
}

#[test]
fn retain_entries_keeping_everything_is_the_original() {
    let hash = Gf2Hash::<2>::new(128, 8, 0x1);
    let Prepared::Local(p) = Clifford2Q::cnot(1, 4).prepare(&hash, false).unwrap() else {
        panic!("expected Local")
    };
    let keep = vec![true; p.num_deltas()];
    let sub = p.retain_entries(&keep);
    assert_eq!(sub.num_deltas(), p.num_deltas());
    for (a, b) in sub.deltas().iter().zip(p.deltas()) {
        assert_entry_eq(a, b, "all-kept");
    }
    // And an empty selection is legal, if useless.
    assert_eq!(
        p.retain_entries(&vec![false; p.num_deltas()]).num_deltas(),
        0
    );
}

fn n_bucket_deltas<const W: usize, C: Channel<W>>(ch: &C, adjoint: bool) -> usize {
    // Plenty of bucket bits, so distinct key deltas do not collide.
    let hash = Gf2Hash::<W>::new(128, 16, 0xBEEF);
    ch.prepare(&hash, adjoint).unwrap().bucket_deltas().len()
}

#[test]
fn bucket_fanin_matches_the_design_table() {
    // 1 bucket: key-preserving channels.
    for (name, n) in [
        (
            "identity",
            n_bucket_deltas::<2, _>(&IdentityChannel::new(), false),
        ),
        (
            "depolarizing",
            n_bucket_deltas::<2, _>(
                &Depolarizing {
                    support: [5],
                    p: 0.1,
                },
                false,
            ),
        ),
        (
            "dephasing",
            n_bucket_deltas::<2, _>(
                &Dephasing {
                    support: [5],
                    p: 0.1,
                },
                false,
            ),
        ),
        ("pauli_x", n_bucket_deltas::<2, _>(&Clifford1Q::x(5), false)),
        ("pauli_y", n_bucket_deltas::<2, _>(&Clifford1Q::y(5), false)),
        ("pauli_z", n_bucket_deltas::<2, _>(&Clifford1Q::z(5), false)),
    ] {
        assert_eq!(n, 1, "{name} should read 1 input bucket per output bucket");
    }

    // 2 buckets: 1-dimensional delta sets.
    for (name, n) in [
        ("h", n_bucket_deltas::<2, _>(&Clifford1Q::h(5), false)),
        ("s", n_bucket_deltas::<2, _>(&Clifford1Q::s(5), false)),
        (
            "amplitude_damping",
            n_bucket_deltas::<2, _>(
                &AmplitudeDamping {
                    support: [5],
                    gamma: 0.3,
                },
                false,
            ),
        ),
        (
            "rot_z",
            n_bucket_deltas::<2, _>(&PauliRotation::new(PauliString::<2>::z(5), 0.3), false),
        ),
    ] {
        assert_eq!(n, 2, "{name} should read 2 input buckets per output bucket");
    }

    // 4 buckets: 2-dimensional delta sets.
    for (name, n) in [
        (
            "cnot",
            n_bucket_deltas::<2, _>(&Clifford2Q::cnot(1, 4), false),
        ),
        ("cz", n_bucket_deltas::<2, _>(&Clifford2Q::cz(1, 4), false)),
        (
            "swap",
            n_bucket_deltas::<2, _>(&Clifford2Q::swap(1, 4), false),
        ),
    ] {
        assert_eq!(n, 4, "{name} should read 4 input buckets per output bucket");
    }
}

#[test]
fn a_rotation_reads_two_buckets_at_any_generator_weight() {
    // The delta set {0, P} is 1-dimensional at every weight.
    for weight in 1..=6usize {
        let mut gen = PauliString::<2>::z(0);
        for q in 1..weight as u32 {
            gen.mul_assign(&PauliString::<2>::z(q * 13));
        }
        let rot = PauliRotation::new(gen, 0.3);
        assert_eq!(rot.weight(), weight);
        assert_eq!(
            n_bucket_deltas::<2, _>(&rot, false),
            2,
            "weight {weight} should still read 2 buckets",
        );
    }
}

#[test]
fn adjoint_preparations_have_the_same_fanin() {
    // Conjugating by G^-1 has delta set im(S^-1 ^ I), a different subspace of the same dimension, so the bucket count must not change.
    for (name, fwd, adj) in [
        (
            "s",
            n_bucket_deltas::<2, _>(&Clifford1Q::s(5), false),
            n_bucket_deltas::<2, _>(&Clifford1Q::s(5), true),
        ),
        (
            "cnot",
            n_bucket_deltas::<2, _>(&Clifford2Q::cnot(1, 4), false),
            n_bucket_deltas::<2, _>(&Clifford2Q::cnot(1, 4), true),
        ),
        (
            "rot_z",
            n_bucket_deltas::<2, _>(&PauliRotation::new(PauliString::<2>::z(5), 0.3), false),
            n_bucket_deltas::<2, _>(&PauliRotation::new(PauliString::<2>::z(5), 0.3), true),
        ),
        (
            "amp_damping",
            n_bucket_deltas::<2, _>(
                &AmplitudeDamping {
                    support: [5],
                    gamma: 0.3,
                },
                false,
            ),
            n_bucket_deltas::<2, _>(
                &AmplitudeDamping {
                    support: [5],
                    gamma: 0.3,
                },
                true,
            ),
        ),
    ] {
        assert_eq!(fwd, adj, "{name}: adjoint fan-in differs from forward");
    }
}

#[test]
fn key_preserving_channels_are_detected() {
    let hash = Gf2Hash::<2>::new(128, 8, 0x1);
    let yes: Vec<Box<dyn Channel<2>>> = vec![
        Box::new(IdentityChannel::new()),
        Box::new(Depolarizing {
            support: [5],
            p: 0.1,
        }),
        Box::new(Dephasing {
            support: [5],
            p: 0.1,
        }),
        Box::new(Clifford1Q::x(5)),
        Box::new(Clifford1Q::y(5)),
        Box::new(Clifford1Q::z(5)),
    ];
    for ch in &yes {
        match ch.prepare(&hash, false).unwrap() {
            Prepared::Local(p) => assert!(p.is_key_preserving()),
            _ => panic!("expected a Local preparation"),
        }
    }

    let no: Vec<Box<dyn Channel<2>>> = vec![
        Box::new(Clifford1Q::h(5)),
        Box::new(Clifford2Q::cnot(1, 4)),
        Box::new(PauliRotation::new(PauliString::<2>::z(5), 0.3)),
    ];
    for ch in &no {
        match ch.prepare(&hash, false).unwrap() {
            Prepared::Local(p) => assert!(!p.is_key_preserving()),
            _ => panic!("expected a Local preparation"),
        }
    }
}

#[test]
fn support_bits_use_the_clifford2q_packing() {
    let hash = Gf2Hash::<1>::new(64, 8, 0x1);
    let prep = Clifford2Q::cnot(2, 7).prepare(&hash, false).unwrap();
    let Prepared::Local(p) = prep else {
        panic!("expected Local")
    };
    assert_eq!(p.qubits(), &[2, 7]);
    // x on q2 -> bit 0; z on q2 -> bit 1; x on q7 -> bit 2; z on q7 -> bit 3.
    let x2 = PauliString::<1>::x(2);
    assert_eq!(p.support_bits(&x2.x, &x2.z), 0b0001);
    let z2 = PauliString::<1>::z(2);
    assert_eq!(p.support_bits(&z2.x, &z2.z), 0b0010);
    let x7 = PauliString::<1>::x(7);
    assert_eq!(p.support_bits(&x7.x, &x7.z), 0b0100);
    let z7 = PauliString::<1>::z(7);
    assert_eq!(p.support_bits(&z7.x, &z7.z), 0b1000);
    // Bits outside the support are ignored.
    let mut noise = PauliString::<1>::y(30);
    noise.mul_assign(&PauliString::<1>::x(2));
    assert_eq!(p.support_bits(&noise.x, &noise.z), 0b0001);
}

#[test]
fn colliding_deltas_share_a_group_rather_than_being_lost() {
    // With 1 bucket bit, CNOT's four key deltas collide on bucket deltas.
    let hash = Gf2Hash::<2>::new(128, 1, 0xC011);
    let prep = Clifford2Q::cnot(1, 4).prepare(&hash, false).unwrap();
    let Prepared::Local(p) = prep else {
        panic!("expected Local")
    };
    assert!(
        p.bucket_deltas().len() <= 2,
        "only 2 bucket values exist with 1 bit",
    );
    // No delta is dropped: the total member count is still |D| = 4.
    assert_eq!(p.num_deltas(), 4);
    // And the table still reproduces `apply`.
    check_agrees_on_random_inputs::<2, _>(&Clifford2Q::cnot(1, 4), 128, "cnot_collided");
}

#[test]
fn deltas_are_ascending_by_local_delta() {
    // Entries ascend by `local_delta` even when bucket deltas collide.
    for bits in [1u8, 4, 16] {
        let hash = Gf2Hash::<2>::new(128, bits, 0xC012);
        let prep = Clifford2Q::swap(1, 4).prepare(&hash, false).unwrap();
        let Prepared::Local(p) = prep else {
            panic!("expected Local")
        };
        for pair in p.deltas().windows(2) {
            assert!(
                pair[0].local_delta < pair[1].local_delta,
                "deltas not ascending at bits={bits}",
            );
        }
    }
}

#[test]
fn delta_iteration_order_is_independent_of_bucket_count() {
    // Changing `bits` must not reorder the entries.
    let order = |bits: u8| -> Vec<u8> {
        let hash = Gf2Hash::<2>::new(128, bits, 0xC013);
        let prep = Clifford2Q::cnot(1, 4).prepare(&hash, false).unwrap();
        let Prepared::Local(p) = prep else {
            panic!("expected Local")
        };
        p.deltas().iter().map(|m| m.local_delta).collect()
    };
    let reference = order(16);
    for bits in [1u8, 2, 4, 8, 12] {
        assert_eq!(order(bits), reference, "delta order changed at bits={bits}");
    }
}

#[test]
fn derive_refuses_support_wider_than_the_local_maximum() {
    // A weight-3 rotation: `derive_local` must decline, even though `PauliRotation::prepare` overrides to the functional form.
    let mut gen = PauliString::<1>::z(0);
    gen.mul_assign(&PauliString::<1>::z(1));
    gen.mul_assign(&PauliString::<1>::z(2));
    let rot = PauliRotation::new(gen, 0.3);
    let hash = Gf2Hash::<1>::new(64, 8, 0x1);
    assert!(Prepared::derive_local(&rot, &hash, false).is_none());
    // The override still produces a usable preparation.
    assert!(matches!(
        rot.prepare(&hash, false),
        Some(Prepared::Rotation(_))
    ));
}

/// The popcount check reads straight off the mask, independent of which qubits are set or which word they land in.
#[test]
fn derive_local_rejects_popcount_gt_2() {
    struct ThreeQubits;
    impl<const W: usize> Channel<W> for ThreeQubits {
        fn max_fanout(&self) -> usize {
            1
        }
        fn support(&self) -> [u64; W] {
            let mut mask = [0u64; W];
            mask[0] = 0b111; // qubits 0, 1, 2 -- popcount 3
            mask
        }
        fn apply(
            &self,
            input_x: &[u64; W],
            input_z: &[u64; W],
            coeff: Complex64,
            out: &mut OutputBuffer<'_, W>,
        ) {
            out.push(*input_x, *input_z, coeff);
        }
    }
    let hash = Gf2Hash::<1>::new(64, 8, 0x1);
    assert!(Prepared::derive_local(&ThreeQubits, &hash, false).is_none());
}

#[test]
fn derive_refuses_a_channel_that_writes_outside_its_support() {
    // Declares support [0] but also flips qubit 1, so derivation must decline.
    struct Liar;
    impl<const W: usize> Channel<W> for Liar {
        fn max_fanout(&self) -> usize {
            1
        }
        fn support(&self) -> [u64; W] {
            let mut mask = [0u64; W];
            mask[0] = 1;
            mask
        }
        fn apply(
            &self,
            input_x: &[u64; W],
            input_z: &[u64; W],
            coeff: Complex64,
            out: &mut OutputBuffer<'_, W>,
        ) {
            let mut x = *input_x;
            x[0] ^= 0b10; // qubit 1 — outside the declared support
            out.push(x, *input_z, coeff);
        }
    }
    let hash = Gf2Hash::<1>::new(64, 8, 0x1);
    assert!(Prepared::derive_local(&Liar, &hash, false).is_none());
    assert!(Liar.prepare(&hash, false).is_none());
}
