use super::*;
use crate::test_support::{low_weight_sum, rand_sum};

/// Every candidate bit against the direct support test, across buckets and for low- and high-weight probes.
fn assert_matches_brute_force<const W: usize>(sum: &PauliSum<W>, probes: &PauliSum<W>) {
    let index = SupportIndex::new(sum);
    let mut candidates = Vec::new();
    for (px, pz, _) in probes.iter() {
        let support: [u64; W] = std::array::from_fn(|w| px[w] | pz[w]);
        for (b, columns) in sum.buckets().iter().enumerate() {
            index.candidates(b, &support, &mut candidates);
            assert_eq!(candidates.len(), columns.len().div_ceil(64));
            for i in 0..columns.len() {
                let meets = (0..W).any(|w| (columns.x[i][w] | columns.z[i][w]) & support[w] != 0);
                let bit = candidates[i / 64] >> (i % 64) & 1 == 1;
                assert_eq!(bit, meets, "bucket {b} term {i}");
            }
        }
    }
}

#[test]
fn candidates_are_exactly_the_terms_whose_support_meets_w1() {
    let sum = low_weight_sum::<1>(20_000, 50, 4, 1);
    assert!(sum.num_buckets() > 1);
    assert_matches_brute_force(&sum, &low_weight_sum::<1>(30, 50, 2, 2));
}

#[test]
fn candidates_are_exactly_the_terms_whose_support_meets_w2() {
    let sum = low_weight_sum::<2>(20_000, 120, 4, 3);
    assert_matches_brute_force(&sum, &low_weight_sum::<2>(20, 120, 2, 4));
    assert_matches_brute_force(&sum, &rand_sum::<2>(5, 120, 5));
}
