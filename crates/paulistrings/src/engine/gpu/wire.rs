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
// Only a `WireGroup` builds one, which is what lets `DeviceWire::post` trust `ptr` without being `unsafe`.
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

    /// Receive consecutive ranges of `dst`, `parts[i] = (len, peer)` being `len` elements from `peer`, on `stream`; panics if the parts are longer than `dst`.
    // Carved here from one view, since a borrowed split of a cudarc view cannot outlive its parent.
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
mod tests;
