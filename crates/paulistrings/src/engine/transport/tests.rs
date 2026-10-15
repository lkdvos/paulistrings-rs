use super::*;
use crate::engine::coset::Gf2Span;

use proptest::prelude::*;

fn order_of(bits: u8, deltas: &[u32]) -> CosetOrder {
    let mut order = CosetOrder::default();
    order.rebuild(&Gf2Span::new(deltas, bits), true);
    order
}

fn map_of(bits: u8, deltas: &[u32], chunks: usize) -> ChunkMap {
    ChunkMap::of_order(&order_of(bits, deltas), chunks)
}

/// `position_of` and `bucket_at` are mutually inverse permutations.
#[test]
fn the_destination_order_is_a_bijection() {
    for (bits, deltas) in [
        (0u8, vec![0u32]),
        (1, vec![0]),
        (4, vec![0]),
        (4, vec![0, 1]),
        (4, vec![0, 3, 5]),
        (6, vec![0, 1, 2, 3]),
        (6, vec![0, 9, 18, 27]),
    ] {
        let order = order_of(bits, &deltas);
        let n = 1u32 << bits;
        assert_eq!(order.positions(), n as usize);
        let mut seen = vec![false; n as usize];
        for beta in 0..n {
            let p = order.position_of(beta);
            assert!(p < n, "position {p} outside {n} for bits={bits}");
            assert!(!seen[p as usize], "position {p} claimed twice");
            seen[p as usize] = true;
            assert_eq!(order.bucket_at(p), beta, "bucket_at is not the inverse");
        }
        assert!(seen.iter().all(|&s| s), "bits={bits} left a position unfilled");
    }
}

/// The chunks tile the positions, and `chunk_of_position` inverts `bound`.
#[test]
fn the_chunks_tile_the_position_range() {
    for (bits, deltas) in [(4u8, vec![0u32]), (6, vec![0, 3]), (6, vec![0, 1, 2, 3])] {
        for chunks in [1usize, 2, 3, 5, 8, 16] {
            let map = map_of(bits, &deltas, chunks);
            let k = map.chunks();
            assert!(k >= 1 && k <= chunks);
            assert_eq!(map.bound(0), 0);
            assert_eq!(map.bound(k) as usize, map.positions());
            for i in 0..k {
                assert!(
                    map.bound(i) < map.bound(i + 1),
                    "chunk {i} of {k} is empty at bits={bits}",
                );
            }
            for p in 0..map.positions() as u32 {
                let c = map.chunk_of_position(p);
                assert!(
                    map.bound(c) <= p && p < map.bound(c + 1),
                    "position {p} says chunk {c}, whose range is                          {}..{}",
                    map.bound(c),
                    map.bound(c + 1),
                );
            }
        }
    }
}

/// A chunk boundary always falls between cosets.
#[test]
fn a_chunk_never_splits_a_coset() {
    for (bits, deltas) in [(6u8, vec![0u32, 3]), (6, vec![0, 1, 2, 3]), (5, vec![0, 7])] {
        let span = Gf2Span::new(&deltas, bits);
        let mut order = CosetOrder::default();
        order.rebuild(&span, true);
        for chunks in [1usize, 2, 4, 8, 64] {
            let map = ChunkMap::of_order(&order, chunks);
            let coset = span.coset_size() as u32;
            for k in 0..=map.chunks() {
                assert_eq!(
                    map.bound(k) % coset,
                    0,
                    "chunk bound {} is not on a coset boundary of {coset}",
                    map.bound(k),
                );
            }
            for beta in 0..(1u32 << bits) {
                let want = map.chunk_of_position(order.position_of(span.representative_of(beta)));
                assert_eq!(
                    map.chunk_of_position(order.position_of(beta)),
                    want,
                    "bucket {beta} is in another chunk than its coset representative",
                );
            }
        }
    }
}

/// The chunk count is clamped to the coset count.
#[test]
fn the_chunk_count_is_clamped_to_the_coset_count() {
    // 2^3 buckets, a span of dimension 2, so two cosets.
    let map = map_of(3, &[0, 1, 2, 3], 16);
    assert_eq!(map.chunks(), 2);
    assert_eq!(map.bound(0), 0);
    assert_eq!(map.bound(1), 4);
    assert_eq!(map.bound(2), 8);
}

proptest! {
    #[test]
    fn the_layout_is_a_bijection_and_its_chunks_tile_it(
        bits in 0u8..=7,
        deltas in prop::collection::vec(0u32..128, 0..5),
        chunks in 1usize..=20,
    ) {
        let hi = 1u32 << bits;
        let mut deltas: Vec<u32> = deltas.into_iter().map(|d| d % hi).chain([0]).collect();
        deltas.sort_unstable();
        deltas.dedup();
        let order = order_of(bits, &deltas);
        let map = ChunkMap::of_order(&order, chunks);
        let mut seen = vec![false; hi as usize];
        for beta in 0..hi {
            let p = order.position_of(beta);
            prop_assert!(p < hi);
            prop_assert!(!seen[p as usize]);
            seen[p as usize] = true;
            prop_assert_eq!(order.bucket_at(p), beta);
            let c = map.chunk_of_position(p);
            prop_assert!(map.bound(c) <= p && p < map.bound(c + 1));
        }
        prop_assert_eq!(map.bound(map.chunks()) as usize, map.positions());
    }
}
