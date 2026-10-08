use super::*;

#[test]
fn scatter_bits_is_the_identity_at_one_partition() {
    for bits in 0u8..12 {
        for want in 0u8..12 {
            assert_eq!(scatter_bits(bits, 0, want), bits, "bits={bits} want={want}");
        }
    }
}

#[test]
fn scatter_bits_sheds_at_most_log2_p() {
    // Incoming 10 bits, 4 partitions each wanting 8: shed exactly 2.
    assert_eq!(scatter_bits(10, 2, 8), 8);
    // Wanting more than the split leaves: the want wins, capped by the incoming bits.
    assert_eq!(scatter_bits(10, 2, 9), 9);
    assert_eq!(scatter_bits(10, 2, 12), 10);
    // Wanting less than the split leaves: never shed more than log2(P).
    assert_eq!(scatter_bits(10, 2, 3), 8);
    // Fewer incoming bits than partition bits: the per-share want decides.
    assert_eq!(scatter_bits(1, 2, 0), 0);
    assert_eq!(scatter_bits(1, 2, 1), 1);
    assert_eq!(scatter_bits(0, 4, 0), 0);
}
