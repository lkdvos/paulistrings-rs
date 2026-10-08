use super::*;

/// `P = 1` sheds nothing, so a scatter cannot change the bucket count and the partitioned run starts exactly where `propagate` would.
#[test]
fn scatter_bits_is_the_identity_at_one_partition() {
    for bits in 0u8..12 {
        for want in 0u8..12 {
            assert_eq!(scatter_bits(bits, 0, want), bits, "bits={bits} want={want}");
        }
    }
}

/// With `P` partitions the count sheds `log2(P)` bits, so the bucket count summed over partitions is the unpartitioned one — unless a partition's own share wants more.
#[test]
fn scatter_bits_sheds_at_most_log2_p() {
    // Incoming 10 bits, 4 partitions each wanting 8: shed exactly 2.
    assert_eq!(scatter_bits(10, 2, 8), 8);
    // Wanting more than the split leaves: the want wins, capped by what is there (the layer loop grows past it).
    assert_eq!(scatter_bits(10, 2, 9), 9);
    assert_eq!(scatter_bits(10, 2, 12), 10);
    // Wanting less than the split leaves: never shed more than log2(P).
    assert_eq!(scatter_bits(10, 2, 3), 8);
    // Fewer incoming bits than there are partitions: the floor saturates at a single bucket per partition, so the per-share want decides.
    assert_eq!(scatter_bits(1, 2, 0), 0);
    assert_eq!(scatter_bits(1, 2, 1), 1);
    assert_eq!(scatter_bits(0, 4, 0), 0);
}
