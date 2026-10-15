use super::*;

use crate::pauli_sum::hash::Gf2Hash;

use crate::channel::rotation::PauliRotation;
use crate::channel::Channel;
use crate::channel::Prepare;
use crate::pauli_string::PauliString;

const NQ: usize = 128;

fn hash() -> Gf2Hash<2> {
    Gf2Hash::<2>::new(NQ, 8, 0xB00C)
}

/// One partition row that is `Z` on `qubit`: `part(v)` is the z-bit of `v` there.
fn z_row(qubit: usize) -> PartitionRows<2> {
    let mut rz = [0u64; 2];
    rz[qubit / 64] = 1u64 << (qubit % 64);
    PartitionRows::<2>::from_rows(NQ, vec![[0u64; 2]], vec![rz])
}

fn prep_of<C: Channel<2>>(channel: &C) -> Prepared<2> {
    channel.prepare(&hash(), false).unwrap()
}

#[test]
fn a_rotation_whose_generator_crosses_is_one_remote_delta() {
    let mut gen = PauliString::<2>::z(1);
    for q in [5, 66, 100] {
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

/// A remote delta's block is its rank among the partner's remote deltas, on both the sending and the receiving side.
#[test]
fn block_indices_count_within_each_partner() {
    let remote = |entry: usize, partner: u32| RemoteDelta {
        entry,
        partner,
        bucket_delta: 0,
    };
    let plan = PartitionPlan {
        remote: vec![
            remote(1, 1),
            remote(2, 3),
            remote(4, 1),
            remote(5, 3),
            remote(7, 1),
        ],
        ..PartitionPlan::default()
    };
    let received: Vec<usize> = (0..5).map(|k| plan.received_block_index(k)).collect();
    let sent: Vec<usize> = (0..5).map(|k| plan.sent_block_index(k)).collect();
    assert_eq!(received, [0, 0, 1, 1, 2]);
    assert_eq!(sent, [0, 0, 1, 1, 2]);
}
