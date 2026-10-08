use super::*;
use crate::channel::clifford::Clifford2Q;
use crate::channel::noise::Depolarizing;
use crate::channel::rotation::PauliRotation;
use crate::channel::{Channel, GeneralUnitary2Q};
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::DEFAULT_HASH_SEED;
use crate::test_support::{haar_su4_matrix, sqrt_swap_matrix, zz_rotation};

fn table<const W: usize>(ch: &dyn Channel<W>, nq: usize) -> DevicePrepared<W> {
    let hash = Gf2Hash::<W>::new(nq, 6, DEFAULT_HASH_SEED);
    let prep = ch.prepare(&hash, false).expect("prepared");
    DevicePrepared::new(&prep, &hash, &FingerprintRows::new(hash.seed()), &[])
}

#[test]
fn classification_and_fanout_per_table() {
    let cnot = table::<1>(&Clifford2Q::cnot(1, 3), 8);
    assert_eq!(
        (cnot.fanout, cnot.dense, cnot.key_preserving),
        (cnot.entries, false, false)
    );
    assert!(cnot.entries > 1);
    let zz = table::<1>(&zz_rotation::<1>(1, 3, 0.3), 8);
    assert_eq!((zz.fanout, zz.dense), (2, false));
    let swap = table::<1>(&GeneralUnitary2Q::from_matrix(1, 3, sqrt_swap_matrix()), 8);
    assert!(swap.dense, "sqrt-swap averages 3.65 rows per pattern");
    let su4 = table::<1>(&GeneralUnitary2Q::from_matrix(1, 3, haar_su4_matrix()), 8);
    assert_eq!((su4.fanout, su4.dense), (16, true));
    let dep = table::<1>(
        &Depolarizing {
            support: [2],
            p: 0.1,
        },
        8,
    );
    assert!(dep.key_preserving && dep.fanout == 1 && !dep.dense);
    let mut gen = PauliString::<2>::x(3);
    gen.z[1] |= 1 << 2;
    gen.x[0] |= 1 << 40;
    let rot = table::<2>(&PauliRotation::new(gen, 0.7), 128);
    assert_eq!(
        (rot.mode, rot.entries, rot.fanout, rot.dense),
        (1, 2, 2, false)
    );
    assert_eq!(rot.mask[4..6], gen.x);
    assert_eq!(rot.mask[6..8], gen.z);
    assert_eq!(rot.bucket_deltas().len(), 2);
}

/// An identity-only retained table with received entries must not take the K5 path, and its position map spans the local deltas only.
#[test]
fn received_entries_disable_the_rescale_path_and_leave_the_local_span() {
    use crate::engine::partitioned::plan::PartitionPlan;
    use crate::pauli_sum::hash::PartitionRows;
    let hash = Gf2Hash::<1>::new(8, 4, DEFAULT_HASH_SEED);
    let fp = FingerprintRows::new(hash.seed());
    // `H` has the deltas `{0, X₁Z₁}`; a row reading qubit 1's x-bit makes the one non-identity entry remote.
    let ch = crate::channel::clifford::Clifford1Q::h(1);
    let prep = ch.prepare(&hash, false).unwrap();
    let rows = PartitionRows::<1>::from_rows(8, vec![[0b10u64]], vec![[0u64]]);
    let plan = PartitionPlan::new(&prep, &rows, 0);
    let Prepared::Local(ptm) = &prep else {
        unreachable!()
    };
    let retained = ptm.retain_entries(&plan.local_entries);
    assert!(
        retained.is_key_preserving() || plan.remote.is_empty(),
        "fixture: the retained table must be identity-only"
    );
    let t = DevicePrepared::new(&prep, &hash, &fp, &plan.remote);
    assert!(!plan.remote.is_empty());
    assert!(!t.key_preserving, "received rows force the full path");
    assert_eq!(t.n_remote, plan.remote.len());
    for r in &plan.remote {
        assert_ne!(t.rem[r.entry], NO_REMOTE);
    }
    assert_eq!(t.bucket_deltas(), plan.local_bucket_deltas);
    let local = DevicePrepared::new(&prep, &hash, &fp, &[]);
    assert!(!local.key_preserving && local.n_remote == 0);
}

/// A Clifford maps patterns bijectively, so no two of its entries share an output pattern; a dense SU(4) and `sqrt(SWAP)` do, and a rotation never restricts.
#[test]
fn collisions_between_entries_follow_the_output_patterns() {
    let all = |t: &DevicePrepared<1>| (0..t.entries).collect::<Vec<_>>();
    let cnot = table::<1>(&Clifford2Q::cnot(1, 3), 8);
    assert!(!cnot.entries_can_collide(&all(&cnot)));
    let su4 = table::<1>(&GeneralUnitary2Q::from_matrix(1, 3, haar_su4_matrix()), 8);
    assert!(su4.entries_can_collide(&all(&su4)));
    assert!(
        !su4.entries_can_collide(&[3]),
        "one entry cannot collide with itself"
    );
    let swap = table::<1>(&GeneralUnitary2Q::from_matrix(1, 3, sqrt_swap_matrix()), 8);
    assert!(swap.entries_can_collide(&all(&swap)));
    let mut gen = PauliString::<2>::x(3);
    gen.z[1] |= 1 << 2;
    gen.x[0] |= 1 << 40;
    let rot = table::<2>(&PauliRotation::new(gen, 0.7), 128);
    assert_eq!(rot.mode, 1);
    assert!(!rot.entries_can_collide(&[0, 1]));
}

/// A Clifford maps every support pattern to exactly one entry and no two entries onto one output pattern; a fanout-2 `T`, a dense SU(4), a rotation and a table with a received entry are not permutations.
#[test]
fn the_permutation_gate_holds_for_cliffords_alone() {
    use crate::channel::clifford::Clifford1Q;
    use crate::channel::GeneralUnitary1Q;
    let cliffords: Vec<(&str, Box<dyn Channel<1>>)> = vec![
        ("h", Box::new(Clifford1Q::h(3))),
        ("s", Box::new(Clifford1Q::s(3))),
        ("cnot", Box::new(Clifford2Q::cnot(1, 3))),
        ("cz", Box::new(Clifford2Q::cz(1, 3))),
        ("swap", Box::new(Clifford2Q::swap(1, 3))),
    ];
    for (name, ch) in &cliffords {
        let t = table::<1>(ch.as_ref(), 8);
        assert!(t.permutation && !t.key_preserving, "{name}");
        let dim = 1usize << (2 * t.kq);
        for s in 0..LOCAL_DIM {
            let emitting: Vec<usize> = (0..t.entries)
                .filter(|&e| (t.nz[e] >> s) & 1 != 0)
                .collect();
            if s < dim {
                assert_eq!(emitting, vec![t.entry_of[s] as usize], "{name} pattern {s}");
            } else {
                assert!(emitting.is_empty() && t.entry_of[s] == NO_ENTRY, "{name}");
            }
        }
    }
    // `X` is a Pauli gate: key-preserving, so K5 takes it before the permutation path.
    let x = table::<1>(&Clifford1Q::x(3), 8);
    assert!(x.key_preserving && x.permutation);
    let t = table::<1>(
        &GeneralUnitary1Q::from_matrix(
            3,
            [
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [
                    Complex64::new(0.0, 0.0),
                    Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
                ],
            ],
        ),
        8,
    );
    assert!(!t.permutation, "T emits two rows for X and for Y");
    let su4 = table::<1>(&GeneralUnitary2Q::from_matrix(1, 3, haar_su4_matrix()), 8);
    assert!(!su4.permutation);
    let rot = table::<1>(&zz_rotation::<1>(1, 3, 0.3), 8);
    assert!(!rot.permutation);
    let restricted = table::<1>(&Clifford2Q::cnot(1, 3), 8).restrict(&[0, 1]);
    assert!(!restricted.permutation);
}

#[test]
fn a_received_entry_disables_the_permutation_path() {
    use crate::engine::partitioned::plan::PartitionPlan;
    use crate::pauli_sum::hash::PartitionRows;
    let hash = Gf2Hash::<1>::new(8, 4, DEFAULT_HASH_SEED);
    let fp = FingerprintRows::new(hash.seed());
    let ch = crate::channel::clifford::Clifford1Q::h(1);
    let prep = ch.prepare(&hash, false).unwrap();
    let rows = PartitionRows::<1>::from_rows(8, vec![[0b10u64]], vec![[0u64]]);
    let plan = PartitionPlan::new(&prep, &rows, 0);
    assert!(!plan.remote.is_empty());
    assert!(!DevicePrepared::new(&prep, &hash, &fp, &plan.remote).permutation);
    assert!(DevicePrepared::new(&prep, &hash, &fp, &[]).permutation);
}

#[test]
fn restrict_renumbers_the_chosen_entries_and_sources_them_locally() {
    let su4 = table::<2>(
        &GeneralUnitary2Q::from_matrix(1, 70, haar_su4_matrix()),
        128,
    );
    let pick = [9usize, 2, 14];
    let r = su4.restrict(&pick);
    assert_eq!((r.entries, r.n_remote, r.key_preserving), (3, 0, false));
    assert_eq!(r.dense, su4.dense);
    for (j, &e) in pick.iter().enumerate() {
        let a = LOCAL_DIM * 2;
        assert_eq!(r.amp[j * a..(j + 1) * a], su4.amp[e * a..(e + 1) * a]);
        assert_eq!(r.mask[j * 4..(j + 1) * 4], su4.mask[e * 4..(e + 1) * 4]);
        assert_eq!(
            (r.nz[j], r.bucket_delta[j], r.gm[j]),
            (su4.nz[e], su4.bucket_delta[e], su4.gm[e])
        );
    }
    assert!(r.rem.iter().all(|&k| k == NO_REMOTE));
    assert!(r.nz[3..].iter().all(|&m| m == 0));
    assert_eq!(r.fanout, 3);
}

#[test]
fn rehash_tracks_the_hash_and_gm_is_the_fingerprint_of_the_mask() {
    let hash = Gf2Hash::<1>::new(8, 3, DEFAULT_HASH_SEED);
    let ch = Clifford2Q::cnot(1, 3);
    let prep = ch.prepare(&hash, false).unwrap();
    let fp = FingerprintRows::new(hash.seed());
    let mut t = DevicePrepared::new(&prep, &hash, &fp, &[]);
    let Prepared::Local(ptm) = &prep else {
        unreachable!()
    };
    for (e, d) in ptm.deltas().iter().enumerate() {
        assert_eq!(t.bucket_delta[e], d.bucket_delta);
        assert_eq!(t.gm[e], fp.fingerprint(&d.mask_x, &d.mask_z));
    }
    let mut refined = hash.clone();
    refined.refine();
    refined.refine();
    t.rehash(&refined);
    let prep2 = ch.prepare(&refined, false).unwrap();
    let Prepared::Local(ptm2) = &prep2 else {
        unreachable!()
    };
    for (e, d) in ptm2.deltas().iter().enumerate() {
        assert_eq!(t.bucket_delta[e], d.bucket_delta);
    }
    assert_eq!(t.bucket_deltas(), ptm2.bucket_deltas());
}
