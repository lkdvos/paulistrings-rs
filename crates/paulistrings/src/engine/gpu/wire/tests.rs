use super::*;
use crate::engine::partitioned::transport::Collectives;
use crate::engine::partitioned::InProcessTransport;

#[test]
fn the_timeout_knob_parses_positive_seconds() {
    assert_eq!(parse_timeout(None), DEFAULT_WIRE_TIMEOUT);
    assert_eq!(parse_timeout(Some("12")), Duration::from_secs(12));
    assert_eq!(parse_timeout(Some(" 0.5 ")), Duration::from_millis(500));
    for bad in ["0", "-3", "nan", "inf", "soon", ""] {
        assert_eq!(parse_timeout(Some(bad)), DEFAULT_WIRE_TIMEOUT, "{bad:?}");
    }
}

fn skeletons_of<const W: usize>(seed: u64, blocks: usize, b: usize) -> BlockSkeletons<W> {
    let mut out = BlockSkeletons::<W>::default();
    for j in 0..blocks {
        let mut offsets = vec![0u32];
        for p in 0..b as u64 {
            let n = (seed ^ (j as u64 * 31) ^ (p * 7)).wrapping_mul(0x9E37_79B9) >> 60;
            offsets.push(offsets.last().unwrap() + n as u32);
        }
        let rows = offsets[b];
        out.blocks.push(Skeleton {
            header: BlockHeader {
                num_buckets: b as u32,
                rows,
                w: W as u32,
                entry: 3 * j as u32 + 1,
            },
            offsets,
        });
    }
    out
}

/// The byte form a byte transport moves, decoded into a pooled payload of another shape.
fn byte_round_trip<const W: usize>(
    sent: &BlockSkeletons<W>,
    pooled: BlockSkeletons<W>,
) -> BlockSkeletons<W> {
    let parts: Vec<Vec<u8>> = sent.byte_parts().iter().map(|p| p.to_vec()).collect();
    let lens: Vec<usize> = parts.iter().map(Vec::len).collect();
    let mut got = pooled;
    for (dst, src) in got.recv_into(&lens).into_iter().zip(&parts) {
        dst.copy_from_slice(src);
    }
    got.finish_recv();
    got
}

#[test]
fn skeletons_round_trip_through_their_byte_form() {
    let sent = skeletons_of::<2>(0xA1, 3, 16);
    assert_eq!(sent.byte_parts().len(), 6);
    assert_eq!(byte_round_trip(&sent, BlockSkeletons::default()), sent);
    assert_eq!(byte_round_trip(&sent, skeletons_of::<2>(0xB2, 5, 64)), sent);
    let empty = BlockSkeletons::<1>::default();
    assert_eq!(byte_round_trip(&empty, skeletons_of::<1>(1, 2, 4)), empty);
}

#[test]
fn skeletons_travel_over_the_in_process_transport() {
    use crate::engine::partitioned::transport::Transport;
    let got: Vec<Vec<Option<BlockSkeletons<1>>>> = std::thread::scope(|s| {
        let hs: Vec<_> = InProcessTransport::group(4)
            .into_iter()
            .map(|t| {
                s.spawn(move || {
                    let me = t.rank();
                    let send = (0..4)
                        .map(|q| (q != me).then(|| skeletons_of::<1>(u64::from(me * 8 + q), 2, 8)))
                        .collect();
                    t.exchange(send, &mut Vec::new())
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (me, recv) in got.iter().enumerate() {
        for (q, r) in recv.iter().enumerate() {
            if q == me {
                assert!(r.is_none());
            } else {
                let want = skeletons_of::<1>((q * 8 + me) as u64, 2, 8);
                assert_eq!(r.as_ref(), Some(&want), "{q} -> {me}");
            }
        }
    }
}

#[test]
fn a_skeleton_whose_offsets_miss_its_header_is_refused() {
    let mut bad = skeletons_of::<1>(0xC3, 1, 8);
    bad.blocks[0].header.rows += 1;
    let r = std::panic::catch_unwind(|| byte_round_trip(&bad, BlockSkeletons::default()));
    assert!(r.is_err());
    let mut wide = skeletons_of::<1>(0xC4, 1, 8);
    wide.blocks[0].header.w = 2;
    assert!(
        std::panic::catch_unwind(|| byte_round_trip(&wide, BlockSkeletons::default())).is_err()
    );
    let mut descending = skeletons_of::<1>(0xC5, 1, 2);
    descending.blocks[0].offsets = vec![0, 5, 3];
    descending.blocks[0].header.rows = 3;
    assert!(
        std::panic::catch_unwind(|| byte_round_trip(&descending, BlockSkeletons::default()))
            .is_err()
    );
}

/// The pieces rank `me` sends to `to` as `(chunk, column, k, rows)`, and those `to` receives from `me`.
fn sends_to(ops: &[ScheduledOp], to: u32) -> Vec<(usize, WireColumn, usize, (usize, usize))> {
    ops.iter()
        .filter(|op| op.kind == WireOpKind::Send && op.peer == to)
        .map(|op| (op.chunk, op.column, op.k, op.rows))
        .collect()
}

fn recvs_from(ops: &[ScheduledOp], from: u32) -> Vec<(usize, WireColumn, usize, (usize, usize))> {
    ops.iter()
        .filter(|op| op.kind == WireOpKind::Recv && op.peer == from)
        .map(|op| (op.chunk, op.column, op.k, op.rows))
        .collect()
}

/// Per rank of a group of `size`, the schedule for remote deltas of partition deltas `pds`, where `counts[sender][k][p]` is the rows the sender's block for delta `k` puts at position `p`.
fn group_schedules(
    size: u32,
    pds: &[u32],
    counts: &[Vec<Vec<u32>>],
    map: &ChunkMap,
) -> Vec<Vec<ScheduledOp>> {
    let csr = |c: &Vec<u32>| -> Vec<u32> {
        let mut o = vec![0u32];
        for &n in c {
            o.push(o.last().unwrap() + n);
        }
        o
    };
    let offsets: Vec<Vec<Vec<u32>>> = counts
        .iter()
        .map(|per_k| per_k.iter().map(csr).collect())
        .collect();
    (0..size)
        .map(|r| {
            let partners: Vec<u32> = pds.iter().map(|&pd| r ^ pd).collect();
            let own: Vec<&[u32]> = offsets[r as usize].iter().map(Vec::as_slice).collect();
            let recv: Vec<&[u32]> = partners
                .iter()
                .enumerate()
                .map(|(k, &q)| offsets[q as usize][k].as_slice())
                .collect();
            schedule(&partners, &own, &recv, map)
        })
        .collect()
}

proptest::proptest! {
    /// Rank `a`'s sends to `b` are `b`'s receives from `a`, in order and in rows, chunk-major; within a chunk each column's receives tile the chunk's buffer in plan order.
    #[test]
    fn the_schedule_is_symmetric(
        size_bits in 1u32..=3,
        pd_seeds in proptest::collection::vec(0u32..1000, 1..=10),
        bits in 0u8..=4,
        chunks in proptest::sample::select(vec![1usize, 3, 8]),
        empty_every in 1usize..5,
        seed in 0u64..u64::MAX,
    ) {
        let size = 1u32 << size_bits;
        let pds: Vec<u32> = pd_seeds.iter().map(|s| 1 + s % (size - 1)).collect();
        let b = 1usize << bits;
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let counts: Vec<Vec<Vec<u32>>> = (0..size)
            .map(|r| {
                (0..pds.len())
                    .map(|k| {
                        (0..b)
                            .map(|_| {
                                if (r as usize + k).is_multiple_of(empty_every) { 0 } else { (next() % 5) as u32 }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let mut map = ChunkMap::default();
        map.rebuild(&crate::engine::coset::Gf2Span::new(&[], bits), b, chunks);
        let ops = group_schedules(size, &pds, &counts, &map);
        for a in 0..size {
            for bb in 0..size {
                let s = sends_to(&ops[a as usize], bb);
                let r = recvs_from(&ops[bb as usize], a);
                proptest::prop_assert_eq!(&s, &r, "{} -> {}", a, bb);
                proptest::prop_assert!(s.iter().all(|x| x.3 .1 > x.3 .0), "an empty piece posted");
            }
            let pds = &pds;
            let rows: u64 = counts.iter().enumerate().flat_map(|(r, per_k)| {
                per_k.iter().enumerate().filter(move |&(k, _)| r as u32 ^ pds[k] == a)
                    .map(|(_, c)| c.iter().map(|&n| u64::from(n)).sum::<u64>())
            }).sum();
            let got: u64 = ops[a as usize].iter().filter(|op| op.kind == WireOpKind::Recv && op.column == WireColumn::X)
                .map(|op| (op.rows.1 - op.rows.0) as u64).sum();
            proptest::prop_assert_eq!(got, rows, "rank {} receives every row once", a);
            for chunk in 0..map.chunks() {
                for column in WireColumn::ALL {
                    let ks: Vec<(usize, (usize, usize))> = ops[a as usize].iter()
                        .filter(|op| op.kind == WireOpKind::Recv && op.column == column && op.chunk == chunk)
                        .map(|op| (op.k, op.rows)).collect();
                    proptest::prop_assert!(ks.windows(2).all(|w| w[0].0 < w[1].0));
                    if chunk == 0 {
                        proptest::prop_assert!(ks.iter().all(|&(_, (lo, _))| lo == 0));
                    }
                }
            }
            for kind in [WireOpKind::Send, WireOpKind::Recv] {
                let chunk_of: Vec<usize> = ops[a as usize].iter().filter(|op| op.kind == kind).map(|op| op.chunk).collect();
                proptest::prop_assert!(chunk_of.windows(2).all(|w| w[0] <= w[1]), "{:?}s are chunk-major", kind);
            }
        }
    }
}

#[test]
fn a_group_records_its_ops_in_posting_order_with_carved_ranges() {
    crate::require_cuda!();
    let ctx = super::super::device::context(0).expect("a visible device");
    let stream = ctx.new_stream().expect("a stream");
    let src = stream.alloc_zeros::<f64>(6).expect("alloc");
    let mut dst = stream.alloc_zeros::<u64>(10).expect("alloc");
    let mut group = WireGroup::new();
    group.send(src.slice(2..6), 3, &stream);
    group.recv_parts(dst.as_view_mut(), &[(4, 1), (0, 2), (5, 1)], &stream);
    let ops = group.ops();
    let shape: Vec<_> = ops
        .iter()
        .map(|op| (op.kind(), op.peer(), op.bytes()))
        .collect();
    assert_eq!(
        shape,
        [
            (WireOpKind::Send, 3, 32),
            (WireOpKind::Recv, 1, 32),
            (WireOpKind::Recv, 2, 0),
            (WireOpKind::Recv, 1, 40),
        ]
    );
    assert_eq!(ops[2].ptr() - ops[1].ptr(), 32);
    assert_eq!(ops[3].ptr() - ops[1].ptr(), 32);
    assert!(ops.iter().all(|op| std::ptr::eq(op.stream(), &*stream)));
}
