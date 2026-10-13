use super::*;
use crate::engine::coset::{CosetOrder, Gf2Span};

use proptest::prelude::*;

fn order_of(bits: u8, deltas: &[u32]) -> CosetOrder {
    let mut order = CosetOrder::default();
    order.rebuild(&Gf2Span::new(deltas, bits), true);
    order
}

fn map_of(bits: u8, deltas: &[u32], chunks: usize) -> ChunkMap {
    ChunkMap::of_order(&order_of(bits, deltas), chunks)
}

/// A payload of `blocks` blocks over `2^bits` positions, filled deterministically.
fn payload_of<const W: usize>(bits: u8, blocks: usize, seed: u64) -> PartnerPayload<W> {
    let n = 1usize << bits;
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut payload = PartnerPayload::<W>::default();
    for e in 0..blocks {
        let counts: Vec<u32> = (0..n).map(|_| (next() % 4) as u32).collect();
        let mut block = ExchangeBlock::<W>::with_counts(e as u32, &counts);
        let rows = block.rows();
        block.x = (0..rows).map(|_| std::array::from_fn(|_| next())).collect();
        block.z = (0..rows).map(|_| std::array::from_fn(|_| next())).collect();
        block.coeff = (0..rows)
            .map(|_| Complex64::new(next() as f64 * 1e-18, next() as f64 * 1e-18))
            .collect();
        payload.blocks.push(block);
    }
    payload
}

/// The early parts are the headers and offsets, and each bulk part's chunks concatenate to its column, on both sides.
#[test]
fn the_early_and_bulk_parts_partition_the_wire() {
    for bits in [0u8, 1, 4, 5] {
        for blocks in [1usize, 3] {
            let mut payload = payload_of::<2>(bits, blocks, 0xB1A5 + u64::from(bits));
            let all: Vec<Vec<u8>> = payload
                .byte_parts()
                .into_iter()
                .map(<[u8]>::to_vec)
                .collect();
            for chunks in [1usize, 2, 3, 8, 64] {
                let map = map_of(bits, &[0], chunks);
                let early: Vec<Vec<u8>> = payload
                    .early_parts()
                    .into_iter()
                    .map(<[u8]>::to_vec)
                    .collect();
                assert_eq!(early.len(), 2 * blocks);
                for b in 0..blocks {
                    assert_eq!(early[2 * b], all[PARTS_PER_BLOCK * b], "block {b} header");
                    assert_eq!(
                        early[2 * b + 1],
                        all[PARTS_PER_BLOCK * b + 1],
                        "block {b} offsets",
                    );
                }

                let bulk = payload.bulk_parts(&map);
                assert_eq!(bulk.len(), 3 * blocks);
                for (i, pieces) in bulk.iter().enumerate() {
                    assert_eq!(pieces.len(), map.chunks(), "part {i} chunk count");
                    let joined: Vec<u8> = pieces.iter().flat_map(|p| p.iter().copied()).collect();
                    let want = &all[PARTS_PER_BLOCK * (i / 3) + 2 + i % 3];
                    assert_eq!(&joined, want, "part {i} at {chunks} chunks");
                }
                let want_lens: Vec<Vec<usize>> = bulk
                    .iter()
                    .map(|p| p.iter().map(|c| c.len()).collect())
                    .collect();
                let got_lens: Vec<Vec<usize>> = payload
                    .bulk_receive_into(&map)
                    .iter()
                    .map(|p| p.iter().map(|c| c.len()).collect())
                    .collect();
                assert_eq!(got_lens, want_lens, "receive side at {chunks} chunks");
            }
        }
    }
}

/// Deterministic xorshift64 row filler.
fn fill<const W: usize>(block: &mut ExchangeBlock<W>, seed: u64) {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for _ in 0..block.rows() {
        block.x.push(std::array::from_fn(|_| next()));
        block.z.push(std::array::from_fn(|_| next()));
        block
            .coeff
            .push(Complex64::new(next() as f64 * 1e-18, next() as f64 * 1e-18));
    }
}

#[test]
fn with_counts_builds_offsets_and_reserves_the_columns() {
    let block = ExchangeBlock::<2>::with_counts(3, &[2, 0, 5, 1]);

    assert_eq!(block.offsets, vec![0, 2, 2, 7, 8]);
    assert_eq!(
        block.header,
        BlockHeader {
            num_buckets: 4,
            rows: 8,
            width: 2,
            entry: 3,
        }
    );
    assert_eq!(block.rows(), 8);
    assert_eq!(block.num_buckets(), 4);
    assert_eq!(block.x.len(), 8);
    assert_eq!(block.z.len(), 8);
    assert_eq!(block.coeff.len(), 8);
}

#[test]
fn set_counts_reuses_the_columns_and_grows_only() {
    let mut block = ExchangeBlock::<1>::with_counts(0, &[4, 4]);
    let wide = block.x.as_ptr();
    assert_eq!(block.rows(), 8);

    block.set_counts(1, &[1, 2]);
    assert_eq!(block.rows(), 3);
    assert_eq!(block.offsets, vec![0, 1, 3]);
    assert_eq!(block.header.entry, 1);
    assert_eq!(block.columns().2.len(), 3, "three live rows");
    assert_eq!(block.x.len(), 8, "the storage of the wider layer is kept");
    assert_eq!(block.x.as_ptr(), wide, "and it is the same allocation");
    assert_eq!(block.bytes(), 16 + 3 * 4 + 3 * 8 + 3 * 8 + 3 * 16);

    block.set_counts(1, &[9, 9]);
    assert_eq!(block.rows(), 18);
    assert!(block.coeff.len() >= 18, "grown for the wider layer");
}

#[test]
fn with_counts_of_no_buckets_is_an_empty_block() {
    let block = ExchangeBlock::<1>::with_counts(0, &[]);
    assert_eq!(block.offsets, vec![0]);
    assert_eq!(block.rows(), 0);
    assert_eq!(block.num_buckets(), 0);
}

#[test]
fn segment_slices_the_columns_by_source_bucket() {
    let mut block = ExchangeBlock::<1>::with_counts(0, &[2, 0, 1]);
    fill(&mut block, 0xa5a5);

    let (x0, z0, c0) = block.segment(0);
    assert_eq!(x0.len(), 2);
    assert_eq!(x0, &block.x[0..2]);
    assert_eq!(z0, &block.z[0..2]);
    assert_eq!(c0, &block.coeff[0..2]);

    let (x1, z1, c1) = block.segment(1);
    assert!(x1.is_empty() && z1.is_empty() && c1.is_empty());

    let (x2, z2, c2) = block.segment(2);
    assert_eq!(x2, &block.x[2..3]);
    assert_eq!(z2, &block.z[2..3]);
    assert_eq!(c2, &block.coeff[2..3]);
}

#[test]
fn bytes_counts_the_wire_footprint() {
    let mut block = ExchangeBlock::<1>::with_counts(0, &[2, 1]);
    fill(&mut block, 7);
    // header 16 + offsets 3*4 + x 3*8 + z 3*8 + coeff 3*16
    assert_eq!(block.bytes(), 16 + 12 + 24 + 24 + 48);

    let mut wide = ExchangeBlock::<2>::with_counts(0, &[1]);
    fill(&mut wide, 7);
    // header 16 + offsets 2*4 + x 1*16 + z 1*16 + coeff 1*16
    assert_eq!(wide.bytes(), 16 + 8 + 16 + 16 + 16);
}

#[test]
fn byte_parts_are_five_borrowed_views_per_block() {
    let mut payload = PartnerPayload::<1>::default();
    let mut block = ExchangeBlock::<1>::with_counts(2, &[1, 1]);
    fill(&mut block, 11);
    payload.blocks.push(block);

    let parts = payload.byte_parts();
    assert_eq!(parts.len(), 5);
    assert_eq!(parts[0].len(), 16); // header
    assert_eq!(parts[1].len(), 12); // offsets: 3 × u32
    assert_eq!(parts[2].len(), 16); // x: 2 rows × 1 word
    assert_eq!(parts[3].len(), 16); // z
    assert_eq!(parts[4].len(), 32); // coeff: 2 × 16 B
    assert_eq!(parts.iter().map(|p| p.len()).sum::<usize>(), 92);
}

/// `byte_parts` on the sender, `receive_into` + `finish_receive` on a fresh receiver.
fn wire_round_trip<const W: usize>(payload: &PartnerPayload<W>) -> PartnerPayload<W> {
    let sent: Vec<Vec<u8>> = payload
        .byte_parts()
        .into_iter()
        .map(<[u8]>::to_vec)
        .collect();
    let lens: Vec<usize> = sent.iter().map(Vec::len).collect();
    let mut back = PartnerPayload::<W>::default();
    for (view, bytes) in back.receive_into(&lens).into_iter().zip(&sent) {
        view.copy_from_slice(bytes);
    }
    back.finish_receive();
    back
}

#[test]
fn an_empty_payload_round_trips_as_zero_parts() {
    let payload = PartnerPayload::<2>::default();
    assert!(payload.byte_parts().is_empty());
    assert_eq!(wire_round_trip(&payload), payload);
}

#[test]
fn payload_round_trips_through_byte_parts() {
    let mut payload = PartnerPayload::<2>::default();
    for (entry, counts) in [(0u32, &[2u32, 0, 1][..]), (1, &[0, 3, 0][..])] {
        let mut block = ExchangeBlock::<2>::with_counts(entry, counts);
        fill(&mut block, entry as u64 + 3);
        payload.blocks.push(block);
    }

    let back = wire_round_trip(&payload);
    assert_eq!(back, payload);
    assert_eq!(back.blocks[1].segment(1).0, payload.blocks[1].segment(1).0);
}

#[test]
#[should_panic(expected = "width")]
fn a_block_encoded_at_another_width_is_rejected() {
    let mut payload = payload_of::<2>(2, 1, 0x5);
    payload.blocks[0].header.width = 1;
    let _ = wire_round_trip(&payload);
}

/// Small random payloads: 0–2 blocks, 1–4 source buckets, 0–3 rows each.
fn arb_payload<const W: usize>() -> impl Strategy<Value = PartnerPayload<W>> {
    let block = (
        0u32..8,
        proptest::collection::vec(0u32..4, 1..5),
        any::<u64>(),
    )
        .prop_map(|(entry, counts, seed)| {
            let mut block = ExchangeBlock::<W>::with_counts(entry, &counts);
            fill(&mut block, seed);
            block
        });
    proptest::collection::vec(block, 0..3).prop_map(|blocks| PartnerPayload { blocks })
}

proptest! {
    #[test]
    fn arbitrary_payloads_round_trip_at_w1(payload in arb_payload::<1>()) {
        prop_assert_eq!(wire_round_trip(&payload), payload);
    }

    #[test]
    fn arbitrary_payloads_round_trip_at_w2(payload in arb_payload::<2>()) {
        prop_assert_eq!(wire_round_trip(&payload), payload);
    }
}
