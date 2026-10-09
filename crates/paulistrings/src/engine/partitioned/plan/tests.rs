use super::*;

use crate::channel::clifford::{Clifford1Q, Clifford2Q};
use crate::channel::rotation::PauliRotation;
use crate::channel::{Channel, GeneralUnitary2Q};
use crate::pauli_string::PauliString;
use crate::test_support::{haar_su4_matrix, Xs64};

const NQ: usize = 128;

fn hash() -> Gf2Hash<2> {
    Gf2Hash::<2>::new(NQ, 8, 0xB00C)
}

/// One partition row that is `Z` on `qubit`: `part(v)` is the z-bit of `v` there.
fn z_row(qubit: u32) -> PartitionRows<2> {
    let mut rz = [0u64; 2];
    rz[qubit as usize / 64] = 1u64 << (qubit % 64);
    PartitionRows::<2>::from_rows(NQ, vec![[0u64; 2]], vec![rz])
}

fn prep_of<C: Channel<2>>(channel: &C) -> Prepared<2> {
    channel.prepare(&hash(), false).unwrap()
}

#[test]
fn without_partition_rows_nothing_is_remote() {
    let rows = PartitionRows::<2>::none(NQ);
    for (name, channel) in [
        ("h", Box::new(Clifford1Q::h(3)) as Box<dyn Channel<2>>),
        ("cnot", Box::new(Clifford2Q::cnot(1, 4))),
        (
            "haar",
            Box::new(GeneralUnitary2Q::from_matrix(1, 5, haar_su4_matrix())),
        ),
    ] {
        let prepared = channel.prepare(&hash(), false).unwrap();
        let plan = PartitionPlan::new(&prepared, &rows, 0);
        assert!(!plan.has_remote(), "{name}: unexpected remote deltas");
        assert!(plan.remote.is_empty(), "{name}");
        assert_eq!(plan.partners().count(), 0, "{name}");
        assert!(plan.local_entries.iter().all(|b| *b), "{name}");
        assert_eq!(
            plan.local_bucket_deltas,
            prepared.bucket_deltas(),
            "{name}: local span input differs from the channel's own",
        );
        let Prepared::Local(ptm) = &prepared else {
            panic!("{name}: expected Local")
        };
        assert_eq!(plan.rest_streams_total, ptm.num_deltas() - 1, "{name}");
    }
}

#[test]
fn a_wide_rotation_without_partition_rows_is_all_local() {
    let mut gen = PauliString::<2>::z(1);
    for q in [5u32, 66, 100] {
        gen.mul_assign(&PauliString::<2>::z(q));
    }
    let prepared = prep_of(&PauliRotation::new(gen, 0.41));
    assert!(matches!(prepared, Prepared::Rotation(_)));
    let plan = PartitionPlan::new(&prepared, &PartitionRows::<2>::none(NQ), 0);
    assert_eq!(plan.local_entries, vec![true, true]);
    assert!(!plan.has_remote());
    assert_eq!(plan.local_bucket_deltas, prepared.bucket_deltas());
    assert_eq!(plan.rest_streams_total, 1);
}

#[test]
fn a_delta_whose_mask_sets_the_partition_bit_is_remote() {
    // H(3)'s delta set is {0, XZ on qubit 3}. A single partition row that is Z on qubit 3 reads the z-bit there, which that mask sets, so `part(XZ_3) = 1`.
    let rows = z_row(3);
    assert_eq!(rows.num_partitions(), 2);
    let prepared = prep_of(&Clifford1Q::h(3));
    let Prepared::Local(ptm) = &prepared else {
        panic!("expected Local")
    };
    assert_eq!(ptm.num_deltas(), 2);

    for rank in 0..2u32 {
        let plan = PartitionPlan::new(&prepared, &rows, rank);
        assert_eq!(plan.local_entries, vec![true, false], "rank {rank}");
        assert!(plan.has_remote(), "rank {rank}");
        assert_eq!(
            plan.remote,
            vec![RemoteDelta {
                entry: 1,
                partner: rank ^ 1,
                bucket_delta: ptm.deltas()[1].bucket_delta,
            }],
            "rank {rank}",
        );
        // Only the identity delta stays behind, so the local coset loop is over a single bucket.
        assert_eq!(plan.local_bucket_deltas, vec![0], "rank {rank}");
        // ...but the sort-kernel gate still sees the channel's full fanout.
        assert_eq!(plan.rest_streams_total, 1, "rank {rank}");
        assert_eq!(
            plan.partners().collect::<Vec<_>>(),
            vec![rank ^ 1],
            "rank {rank}",
        );
        assert_eq!(
            plan.remote_for_partner(rank ^ 1).count(),
            1,
            "rank {rank}: partner stream",
        );
        assert_eq!(plan.remote_for_partner(rank).count(), 0, "rank {rank}");
    }
}

#[test]
fn a_rotation_whose_generator_crosses_is_one_remote_delta() {
    let mut gen = PauliString::<2>::z(1);
    for q in [5u32, 66, 100] {
        gen.mul_assign(&PauliString::<2>::z(q));
    }
    // A Z row on qubit 1 reads the generator's z-bit there, which is set.
    let rows = z_row(1);
    let prepared = prep_of(&PauliRotation::new(gen, 0.41));
    let Prepared::Rotation(r) = &prepared else {
        panic!("expected Rotation")
    };

    let plan = PartitionPlan::new(&prepared, &rows, 1);
    assert_eq!(plan.local_entries, vec![true, false]);
    assert_eq!(plan.local_bucket_deltas, vec![0]);
    assert_eq!(
        plan.remote,
        vec![RemoteDelta {
            entry: 1,
            partner: 0,
            bucket_delta: r.bucket_delta_generator,
        }],
    );
    assert_eq!(plan.rest_streams_total, 1);
}

#[test]
fn local_and_remote_cover_every_entry_exactly_once() {
    let mut rng = Xs64::new(0xA11CE);
    let prepared = prep_of(&GeneralUnitary2Q::from_matrix(1, 5, haar_su4_matrix()));
    let Prepared::Local(ptm) = &prepared else {
        panic!("expected Local")
    };
    // The dense fixture realizes all sixteen deltas.
    assert_eq!(ptm.num_deltas(), 16);

    for trial in 0..32 {
        let bits = 1 + (trial % 3);
        let rows_x: Vec<[u64; 2]> = (0..bits).map(|_| rng.next_array::<2>()).collect();
        let rows_z: Vec<[u64; 2]> = (0..bits).map(|_| rng.next_array::<2>()).collect();
        let rows = PartitionRows::<2>::from_rows(NQ, rows_x, rows_z);
        let rank = (rng.next_u64() as u32) % rows.num_partitions() as u32;
        let plan = PartitionPlan::new(&prepared, &rows, rank);

        assert_eq!(plan.local_entries.len(), ptm.num_deltas());
        let n_local = plan.local_entries.iter().filter(|b| **b).count();
        assert_eq!(n_local + plan.remote.len(), ptm.num_deltas());
        // Remote entries are exactly the false slots, in ascending order.
        let remote_entries: Vec<usize> = plan.remote.iter().map(|r| r.entry).collect();
        let expected: Vec<usize> = plan
            .local_entries
            .iter()
            .enumerate()
            .filter(|(_, b)| !**b)
            .map(|(e, _)| e)
            .collect();
        assert_eq!(remote_entries, expected, "trial {trial}");
        for r in &plan.remote {
            let (mask_x, mask_z) = ptm.deltas()[r.entry].mask();
            assert_eq!(r.partner, rank ^ rows.partition_of(&mask_x, &mask_z));
            assert_ne!(r.partner, rank, "a partition never ships to itself");
            assert_eq!(r.bucket_delta, ptm.deltas()[r.entry].bucket_delta);
        }
        // Partners: distinct and ascending, and every remote delta's partner appears.
        let partners: Vec<u32> = plan.partners().collect();
        assert!(partners.windows(2).all(|w| w[0] < w[1]), "trial {trial}");
        for r in &plan.remote {
            assert!(partners.contains(&r.partner), "trial {trial}");
        }
        assert_eq!(
            partners
                .iter()
                .map(|&q| plan.remote_for_partner(q).count())
                .sum::<usize>(),
            plan.remote.len(),
            "trial {trial}",
        );
        // The local span input is exactly the local entries' bucket deltas, plus 0.
        let mut want: Vec<u32> = vec![0];
        for (e, keep) in plan.local_entries.iter().enumerate() {
            if *keep {
                want.push(ptm.deltas()[e].bucket_delta);
            }
        }
        want.sort_unstable();
        want.dedup();
        assert_eq!(plan.local_bucket_deltas, want, "trial {trial}");
        assert_eq!(plan.rest_streams_total, 15, "trial {trial}");
    }
}

#[test]
fn count_remote_deltas_reports_layers_in_application_order() {
    // Two layers with hand-checkable delta sets under a Z-row on qubit 3:
    //   H(3):  deltas {0, XZ_3} -> part = {0, 1}  -> (1 local, 1 remote)
    //   S(9):  deltas {0,  Z_9} -> part = {0, 0}  -> (2 local, 0 remote)
    // (S maps X -> Y and Y -> -X, so its only nonzero delta is a z-bit.)
    let rows = z_row(3);
    let h = hash();
    let mut circuit = Circuit::<2>::new(NQ);
    circuit.push(Clifford1Q::h(3));
    circuit.push(Clifford1Q::s(9));

    assert_eq!(
        count_remote_deltas(&circuit, &h, &rows, false),
        vec![(1, 1), (2, 0)],
    );
    // Heisenberg order is the reverse. `S` is not self-adjoint, but its adjoint's delta set is the same subspace, so the counts do not move.
    assert_eq!(
        count_remote_deltas(&circuit, &h, &rows, true),
        vec![(2, 0), (1, 1)],
    );
}

#[test]
fn count_remote_deltas_without_rows_is_all_local() {
    let rows = PartitionRows::<2>::none(NQ);
    let h = hash();
    let mut circuit = Circuit::<2>::new(NQ);
    circuit.push(Clifford2Q::cnot(1, 4));
    circuit.push(Clifford1Q::h(3));
    assert_eq!(
        count_remote_deltas(&circuit, &h, &rows, false),
        vec![(4, 0), (2, 0)],
    );
}

impl PartitionPlan {
    /// The partitions this layer exports to, distinct and ascending.
    fn partners(&self) -> impl Iterator<Item = u32> + '_ {
        let mut v: Vec<u32> = self.remote.iter().map(|r| r.partner).collect();
        v.sort_unstable();
        v.dedup();
        v.into_iter()
    }
}
