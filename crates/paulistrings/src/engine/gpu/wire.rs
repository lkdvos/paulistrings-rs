//! The device exchange's wire protocol: [`DeviceWire`] and [`WireGroup`], the host half ([`BlockSkeletons`], [`schedule`]), and [`PeerWire`], the in-process wire.
//! See ARCHITECTURE.md §Partitioning.

mod peer;

use std::time::Duration;

use cudarc::driver::{CudaStream, CudaView, CudaViewMut, SyncOnDrop};

use super::error::GpuError;
use super::payload::DeviceBlock;
use crate::engine::partitioned::transport::{chunk_rows_of, BlockHeader, ChunkMap, Payload};

pub(crate) use peer::PeerWire;
#[cfg(test)]
pub(crate) use peer::{PeerFault, PeerTally};

/// The bound on every wire wait when `PAULISTRINGS_NCCL_TIMEOUT_S` is unset.
pub(crate) const DEFAULT_WIRE_TIMEOUT: Duration = Duration::from_secs(300);

/// `PAULISTRINGS_NCCL_TIMEOUT_S` as a positive number of seconds, else [`DEFAULT_WIRE_TIMEOUT`].
pub(crate) fn wire_timeout() -> Duration {
    parse_timeout(std::env::var("PAULISTRINGS_NCCL_TIMEOUT_S").ok().as_deref())
}

fn parse_timeout(raw: Option<&str>) -> Duration {
    raw.and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s > 0.0)
        .map_or(DEFAULT_WIRE_TIMEOUT, Duration::from_secs_f64)
}

/// Which way a [`WireOp`] moves bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WireOpKind {
    /// This rank's bytes to `peer`.
    Send,
    /// `peer`'s bytes into this rank's buffer.
    Recv,
}

/// One point-to-point transfer of a [`WireGroup`]: `bytes` bytes at device address `ptr`, to or from `peer`, ordered on `stream`.
/// Only a [`WireGroup`] builds one, which is what lets [`DeviceWire::post`] trust `ptr` without being `unsafe`.
#[derive(Clone, Copy)]
pub(crate) struct WireOp<'a> {
    kind: WireOpKind,
    peer: u32,
    ptr: u64,
    bytes: usize,
    stream: &'a CudaStream,
}

impl<'a> WireOp<'a> {
    pub(crate) fn kind(&self) -> WireOpKind {
        self.kind
    }
    pub(crate) fn peer(&self) -> u32 {
        self.peer
    }
    /// The device address, valid for [`bytes`](Self::bytes) bytes while the op's [`WireGroup`] lives.
    pub(crate) fn ptr(&self) -> u64 {
        self.ptr
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn stream(&self) -> &'a CudaStream {
        self.stream
    }
}

/// The ops of one wire group, holding every buffer borrowed until [`post`](Self::post) has enqueued them, which is what lets posting be safe: cudarc's per-buffer events are recorded only after the ops are on their streams.
#[derive(Default)]
pub(crate) struct WireGroup<'a> {
    ops: Vec<WireOp<'a>>,
    guards: Vec<SyncOnDrop<'a>>,
}

impl<'a> WireGroup<'a> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Send `src`'s bytes to `peer`, ordered on `stream`.
    pub(crate) fn send<T>(&mut self, src: CudaView<'a, T>, peer: u32, stream: &'a CudaStream) {
        let bytes = src.len() * std::mem::size_of::<T>();
        let (ptr, guard) = src.view_ptr(stream);
        self.push(WireOpKind::Send, peer, ptr, bytes, stream, guard);
    }

    /// Receive `dst.len()` elements' bytes from `peer` into `dst`, ordered on `stream`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn recv<T>(&mut self, dst: CudaViewMut<'a, T>, peer: u32, stream: &'a CudaStream) {
        let bytes = dst.len() * std::mem::size_of::<T>();
        let (ptr, guard) = dst.view_ptr(stream);
        self.push(WireOpKind::Recv, peer, ptr, bytes, stream, guard);
    }

    /// Receive consecutive ranges of `dst` in order, `parts[i] = (len, peer)` being `len` elements from `peer`, on `stream`; one view carved here, since a borrowed split of a cudarc view cannot outlive its parent.
    /// Panics if the parts are longer than `dst`.
    pub(crate) fn recv_parts<T>(
        &mut self,
        dst: CudaViewMut<'a, T>,
        parts: &[(usize, u32)],
        stream: &'a CudaStream,
    ) {
        let total: usize = parts.iter().map(|&(len, _)| len).sum();
        assert!(
            total <= dst.len(),
            "recv_parts: {total} elements into a view of {}",
            dst.len()
        );
        let elem = std::mem::size_of::<T>();
        let (base, guard) = dst.view_ptr(stream);
        let mut at = 0usize;
        for &(len, peer) in parts {
            self.ops.push(WireOp {
                kind: WireOpKind::Recv,
                peer,
                ptr: base + (at * elem) as u64,
                bytes: len * elem,
                stream,
            });
            at += len;
        }
        self.guards.push(guard);
    }

    fn push(
        &mut self,
        kind: WireOpKind,
        peer: u32,
        ptr: u64,
        bytes: usize,
        stream: &'a CudaStream,
        guard: SyncOnDrop<'a>,
    ) {
        self.ops.push(WireOp {
            kind,
            peer,
            ptr,
            bytes,
            stream,
        });
        self.guards.push(guard);
    }

    /// The ops in posting order.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn ops(&self) -> &[WireOp<'a>] {
        &self.ops
    }

    /// Post every op through `wire` as one group, then release the buffers' borrows.
    pub(crate) fn post(self, wire: &dyn DeviceWire) -> Result<(), GpuError> {
        self.post_with(|ops| wire.post(ops))
    }

    pub(crate) fn post_with(
        self,
        post: impl FnOnce(&[WireOp<'a>]) -> Result<(), GpuError>,
    ) -> Result<(), GpuError> {
        let posted = post(&self.ops);
        drop(self.guards);
        posted
    }
}

/// Point-to-point transfers of device memory within a group of ranks, one group of transfers at a time.
///
/// The matching contract an implementation must honour: a rank's sends to `q` match `q`'s receives from that rank one to one, in posting order, with equal byte counts, and a rank may be its own peer.
/// Every wait is bounded; a timed-out or failed wire is dead, and every later call on it returns an error.
pub(crate) trait DeviceWire: Send + Sync {
    /// This rank's index in the group.
    #[cfg_attr(not(test), allow(dead_code))]
    fn rank(&self) -> u32;
    /// Ranks in the group.
    #[cfg_attr(not(test), allow(dead_code))]
    fn size(&self) -> u32;
    /// Post `ops` as one group: when it returns `Ok`, every op is enqueued on its stream, so later work on that stream runs after it.
    /// Every rank a posted op names must post its matching group, or the transfers never complete and [`wait`](Self::wait) times out.
    fn post(&self, ops: &[WireOp<'_>]) -> Result<(), GpuError>;
    /// Block, bounded by the wire's timeout, until all work enqueued on `stream` so far has completed.
    fn wait(&self, stream: &CudaStream) -> Result<(), GpuError>;
    /// Give the wire up for good, without waiting on anything; every later call fails.
    fn abort(&self);
    /// Whether the wire can still carry a group: `false` once it failed, timed out or was aborted.
    fn is_healthy(&self) -> bool;
}

/// One exchange block without its columns: what a receiver sizes its receives and the fused layer's segments from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Skeleton {
    pub(crate) header: BlockHeader,
    pub(crate) offsets: Vec<u32>,
}

/// One partner's block skeletons in ascending remote-delta index, the host half of an exchange; not a column-less `PartnerPayload`, whose `finish_recv` holds a header to the columns that arrived with it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockSkeletons<const W: usize> {
    pub(crate) blocks: Vec<Skeleton>,
}

impl<const W: usize> BlockSkeletons<W> {
    /// The headers and offsets of `blocks`, reusing this payload's allocations.
    pub(crate) fn fill_from(&mut self, blocks: &[DeviceBlock<W>]) {
        self.blocks.truncate(blocks.len());
        self.blocks.resize_with(blocks.len(), Skeleton::default);
        for (s, b) in self.blocks.iter_mut().zip(blocks) {
            s.header = b.header;
            s.offsets.clear();
            s.offsets.extend_from_slice(&b.offsets);
        }
    }

    /// Append an empty block for remote-delta `entry` over `b` positions.
    pub(crate) fn push_empty(&mut self, entry: u32, b: usize) {
        self.blocks.push(Skeleton {
            header: BlockHeader {
                num_buckets: b as u32,
                rows: 0,
                w: W as u32,
                entry,
            },
            offsets: vec![0; b + 1],
        });
    }
}

impl<const W: usize> Payload for BlockSkeletons<W> {
    fn byte_parts(&self) -> Vec<&[u8]> {
        let mut parts = Vec::with_capacity(2 * self.blocks.len());
        for s in &self.blocks {
            parts.push(bytemuck::bytes_of(&s.header));
            parts.push(bytemuck::cast_slice(&s.offsets));
        }
        parts
    }

    fn recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
        assert_eq!(
            lens.len() % 2,
            0,
            "block skeletons: {} parts is not a whole number of two-part blocks",
            lens.len()
        );
        let n = lens.len() / 2;
        self.blocks.truncate(n);
        self.blocks.resize_with(n, Skeleton::default);
        let mut parts = Vec::with_capacity(lens.len());
        for (s, lens) in self.blocks.iter_mut().zip(lens.chunks_exact(2)) {
            assert_eq!(
                lens[0],
                std::mem::size_of::<BlockHeader>(),
                "block skeletons: a header of {} bytes",
                lens[0]
            );
            assert_eq!(
                lens[1] % 4,
                0,
                "block skeletons: {} offset bytes is not a whole number of u32",
                lens[1]
            );
            s.offsets.clear();
            s.offsets.resize(lens[1] / 4, 0);
            let Skeleton { header, offsets } = s;
            parts.push(bytemuck::bytes_of_mut(header));
            parts.push(bytemuck::cast_slice_mut(&mut offsets[..]));
        }
        parts
    }

    /// The receive sizes itself from these offsets, so each must be a CSR index that ends at the header's row count.
    fn finish_recv(&mut self) {
        for s in &self.blocks {
            let h = &s.header;
            assert_eq!(
                h.w as usize, W,
                "block skeletons: a block encoded at W={} decoded at W={W}",
                h.w
            );
            assert_eq!(
                s.offsets.len(),
                h.num_buckets as usize + 1,
                "block skeletons: a block of {} buckets arrived with {} offsets",
                h.num_buckets,
                s.offsets.len()
            );
            assert!(
                s.offsets[0] == 0 && s.offsets.windows(2).all(|w| w[0] <= w[1]),
                "block skeletons: offsets that are not a CSR index"
            );
            assert_eq!(
                s.offsets[h.num_buckets as usize], h.rows,
                "block skeletons: offsets end at {}, the header names {} rows",
                s.offsets[h.num_buckets as usize], h.rows
            );
        }
    }
}

/// A column of an exchange block as the wire moves it; the fingerprint column never travels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WireColumn {
    X,
    Z,
    Coeff,
}

impl WireColumn {
    pub(crate) const ALL: [WireColumn; 3] = [WireColumn::X, WireColumn::Z, WireColumn::Coeff];

    /// Device elements per row at width `W`: `u64` words for a key column, `f64` halves for the coefficient.
    pub(crate) fn elems_per_row<const W: usize>(self) -> usize {
        match self {
            WireColumn::X | WireColumn::Z => W,
            WireColumn::Coeff => 2,
        }
    }
}

/// One transfer of an exchange: rows `rows.0..rows.1` of `column` of the block remote delta `k` carries, in chunk `chunk`, to or from `peer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScheduledOp {
    pub(crate) kind: WireOpKind,
    pub(crate) peer: u32,
    pub(crate) chunk: usize,
    pub(crate) column: WireColumn,
    pub(crate) k: usize,
    pub(crate) rows: (usize, usize),
}

/// The transfers of one exchange: `partners[k]`, `own[k]` and `recv[k]` are remote delta `k`'s partner and the offsets of the blocks sent and received for it, `k` naming one block pair on both ends.
/// Per peer, sends and receives both run chunk, column, delta, the order a wire matches in; receives are column-major across peers so each column's tile its `recv_*` column, and an empty piece is posted by neither end.
pub(crate) fn schedule(
    partners: &[u32],
    own: &[&[u32]],
    recv: &[&[u32]],
    map: &ChunkMap,
) -> Vec<ScheduledOp> {
    assert_eq!(partners.len(), own.len());
    assert_eq!(partners.len(), recv.len());
    let own_rows: Vec<Vec<usize>> = own.iter().map(|o| chunk_rows_of(o, map)).collect();
    let recv_rows: Vec<Vec<usize>> = recv.iter().map(|o| chunk_rows_of(o, map)).collect();
    let mut peers: Vec<u32> = partners.to_vec();
    peers.sort_unstable();
    peers.dedup();
    let mut ops = Vec::new();
    for chunk in 0..map.chunks() {
        for &q in &peers {
            for column in WireColumn::ALL {
                for (k, _) in partners.iter().enumerate().filter(|&(_, &p)| p == q) {
                    let rows = (own_rows[k][chunk], own_rows[k][chunk + 1]);
                    if rows.1 > rows.0 {
                        ops.push(ScheduledOp {
                            kind: WireOpKind::Send,
                            peer: q,
                            chunk,
                            column,
                            k,
                            rows,
                        });
                    }
                }
            }
        }
    }
    for chunk in 0..map.chunks() {
        for column in WireColumn::ALL {
            for (k, &q) in partners.iter().enumerate() {
                let rows = (recv_rows[k][chunk], recv_rows[k][chunk + 1]);
                if rows.1 > rows.0 {
                    ops.push(ScheduledOp {
                        kind: WireOpKind::Recv,
                        peer: q,
                        chunk,
                        column,
                        k,
                        rows,
                    });
                }
            }
        }
    }
    ops
}

#[cfg(test)]
mod tests {
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
                            .map(|q| {
                                (q != me).then(|| skeletons_of::<1>(u64::from(me * 8 + q), 2, 8))
                            })
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
        assert!(std::panic::catch_unwind(|| byte_round_trip(
            &descending,
            BlockSkeletons::default()
        ))
        .is_err());
    }

    /// The pieces rank `me` sends to `to` as `(chunk, column, k, rows)`, and those `to` receives from `me`.
    fn sends_to(ops: &[ScheduledOp], to: u32) -> Vec<(usize, WireColumn, usize, (usize, usize))> {
        ops.iter()
            .filter(|op| op.kind == WireOpKind::Send && op.peer == to)
            .map(|op| (op.chunk, op.column, op.k, op.rows))
            .collect()
    }

    fn recvs_from(
        ops: &[ScheduledOp],
        from: u32,
    ) -> Vec<(usize, WireColumn, usize, (usize, usize))> {
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
}
