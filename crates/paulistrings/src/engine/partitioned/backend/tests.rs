use super::*;
use crate::test_support::{assert_same_terms, rand_sum};

fn detach_leaves_an_empty_partition_under_the_same_hash<const W: usize>(num_qubits: usize) {
    let mut sum = rand_sum::<W>(3_000, num_qubits, 0xDE7AC4 + W as u64);
    sum.refine();
    sum.refine();
    let want = sum.clone();
    let mut part = HostPartition::new(sum);

    let moved = part.detach();

    assert_eq!(part.len(), 0);
    assert!(part.sum.is_empty());
    assert_eq!(part.hash().bits(), want.hash().bits());
    assert!(part.hash().same_rows_as(want.hash()));
    assert_eq!(part.sum.num_qubits(), want.num_qubits());
    assert_eq!(part.sum.num_buckets(), want.num_buckets());
    part.sum.assert_invariants();

    assert_eq!(moved.len(), want.len());
    assert_same_terms(&moved.sum, &want, "detached partition");
    moved.sum.assert_invariants();
}

#[test]
fn detach_leaves_an_empty_partition_under_the_same_hash_w1() {
    detach_leaves_an_empty_partition_under_the_same_hash::<1>(40);
}

#[test]
fn detach_leaves_an_empty_partition_under_the_same_hash_w2() {
    detach_leaves_an_empty_partition_under_the_same_hash::<2>(100);
}
