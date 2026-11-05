use super::*;
use crate::engine::partitioned::in_process::on_every_rank;

/// Each rank of four agrees `bits` within the exchange group of a layer whose partition deltas are `offsets`.
fn agreed_bits(offsets: &[u32], bits: [u8; 4]) -> Vec<u8> {
    on_every_rank(4, |transport| {
        let rank = transport.rank();
        let group = PartitionPlan::with_partition_deltas(rank, offsets).exchange_group(rank);
        exchange_bits(transport, &group, bits[rank as usize])
    })
}

/// Delta `1` groups `{0, 1}` and `{2, 3}`; deltas `1` and `2` group all four, partners of partners included.
#[test]
fn an_exchange_group_agrees_its_maximum() {
    assert_eq!(agreed_bits(&[1], [3, 5, 9, 4]), vec![5, 5, 9, 9]);
    assert_eq!(agreed_bits(&[1, 2], [3, 5, 4, 9]), vec![9; 4]);
}
