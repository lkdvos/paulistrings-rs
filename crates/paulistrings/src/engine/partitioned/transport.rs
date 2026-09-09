//! Transport traits and the exchange wire format.
//!
//! The sum is split across `P = 2^p` partitions by designated *partition rows*
//! of the GF(2) hash (ARCHITECTURE.md §Bucketing gives the hash; the split is
//! §Partitioning). A partition is one NUMA domain in-process today and one MPI
//! rank later; both talk to the rest of the world only through the two traits
//! here — [`Collectives`] (rank/size, the two reductions, the barrier) and
//! [`Transport`] (the per-layer all-to-all [`Transport::exchange`]).
//!
//! # The collective-order invariant
//!
//! **Every partition issues the identical sequence of transport calls, in the
//! same order, on every layer.** Nothing in the transport reorders or matches
//! calls up by intent: a call's `n`-th message is paired with the partner's
//! `n`-th message positionally. A partition that skips an exchange because it
//! happens to have nothing to send, or that runs an extra reduction,
//! desynchronizes the whole group — so a layer's transport calls are driven by
//! the *plan* (which every partition computes identically from the channel and
//! the hash), never by local data. Empty is sent as `None`, not as silence.
//!
//! The invariant is *checked*, in every build. Each rank numbers its own
//! transport calls with a **generation** counter and publishes
//! `(generation, kind)` before it waits; a rank waiting for generation `g`
//! panics if the partner published a different kind at `g`, or ran past `g`
//! without publishing it. So a violation is a panic naming both partitions
//! rather than a hang or a silently crossed payload. Exchange messages carry
//! the same number as a debug-only stamp on the channel, checked on receive.
//!
//! # The in-process collectives are shared memory, not messages
//!
//! [`InProcessTransport`] keeps the [`Transport::exchange`] all-to-all on a
//! `P × P` matrix of `mpsc` channels — it moves a payload, and it runs only on
//! the layers that have something to move — but the three [`Collectives`]
//! operations run on **shared atomics with a spin wait**
//! ([`GroupState`]). They are on the per-layer critical path (the bucket-count
//! maximum is unconditional, ARCHITECTURE.md §Partitioning), they carry a
//! handful of words, and the `P` partition threads are pinned and dedicated
//! for the whole call — so a futex sleep/wake per round trip was the whole
//! cost. Measured on the reference host at `P = 2`: one `allreduce_max_u8`
//! costs ~2.9 µs on the channels and ~0.25 µs on the atomics. (The engine's
//! per-layer `collective_ns` is larger than either, because a partition that
//! finishes its layer first waits out its partner's skew inside the
//! collective; that part is load imbalance, not transport.)
//!
//! # The wire unit: a CSR block in the receiver's destination-coset order
//!
//! Per layer a prepared channel's delta set `D` splits into deltas that keep a
//! row inside its own partition and **remote deltas**, whose partition bits
//! `pd[e]` are non-zero. For a remote delta `e`, every row partition `R`
//! generates from its local bucket `β` lands in partner `R ⊕ pd[e]`, local
//! bucket `β ⊕ bd[e]` — one partner, one bucket offset, both known before a
//! single term is touched.
//!
//! So the natural unit is one [`ExchangeBlock`] per remote delta: the rows in
//! CSR order, one segment per destination bucket. The segments are ordered by
//! the **receiver's coset position** ([`ChunkMap`]) rather than by the
//! sender's bucket index, so
//!
//! ```text
//! block.segment(p)   holds the rows generated from source bucket  bucket_at(p) ^ bd[e]
//! ```
//!
//! and the receiver filling its output bucket `β′` reads
//! `block.segment(map.position_of(β′))`. Both sides derive the permutation
//! from the same local bucket deltas and the same collectively agreed bucket
//! count, so the sender pays one permutation of its count and offset arrays
//! and the receiver pays a table lookup. What it buys is that a coset's rows
//! are **contiguous** in the block, which is what lets the transfer be cut
//! into chunks the receiver consumes in order while the rest is still in
//! flight. A [`PartnerPayload`] is the blocks for one partner in ascending
//! remote-delta index (the block's [`BlockHeader::entry`]), so the receiver
//! walks its plan's remote deltas and the payload's blocks in lockstep.
//!
//! # Bytes, and why nothing is copied twice
//!
//! [`Payload::byte_parts`] hands out one borrowed, zero-copy `&[u8]` view per
//! column (`bytemuck::cast_slice`, no packing, no allocation). The receive side
//! is its exact mirror: [`Payload::recv_into`] sizes the payload's *own* typed
//! columns from the part lengths the wire declared and hands out mutable byte
//! views of them, so a transport receives straight into the storage the engine
//! reads and there is no decode pass at all. That pair is the **only** wire
//! path — there is no copying inverse to keep in step with it.
//!
//! Both sides are **pooled**: an [`ExchangeBlock`]'s columns are grow-only and
//! a payload the layer is finished with goes back into the caller's pool
//! ([`Transport::exchange`]'s `spare`), so a steady-state layer neither
//! allocates nor zeroes its megabytes again. That is worth more than it sounds:
//! measured at 2 ranks x 8 threads with 48 MB crossing per layer, faulting in
//! and zeroing the send and receive buffers cost 14 ms a layer and the word-by-
//! word decode another 8.5, against 13.7 ms of actual transfer.
//!
//! The in-process transport calls none of them — it moves the typed payload
//! through a channel — but an MPI transport implements the same traits by
//! sending the parts.

use std::cell::UnsafeCell;
use std::mem::size_of;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use num_complex::Complex64;

use crate::engine::coset::Gf2Span;

/// The order a layer's exchange blocks are laid out in, and the chunks the
/// bulk transfer is cut into.
///
/// # Destination-coset order
///
/// The receiver's unit of work is a **coset** of `span(h(D_local))`, and
/// `Gf2Span::perm_index` renumbers the bucket index so that a coset occupies a
/// contiguous run of *positions* (ARCHITECTURE.md §Engine). Both sides of an
/// exchange can compute that renumbering — the local bucket deltas are a
/// function of the channel and the hash, not of the rank, and the bucket count
/// is agreed by collective before the layer — so the **sender** lays its CSR
/// block out in the receiver's position order instead of its own source-bucket
/// order:
///
/// ```text
/// segment p  holds the rows generated from source bucket  bucket_at(p) ^ bd
/// ```
///
/// and the receiver filling output bucket `β′` reads `segment(position_of(β′))`
/// with no arithmetic of its own. Reordering costs the sender nothing: it is a
/// permutation of the count and offset arrays, applied before the fill pass
/// walks them.
///
/// # Chunks
///
/// Because a coset is contiguous in position space, a contiguous *range* of
/// positions is a whole number of cosets — so the block splits into `chunks`
/// pieces that the receiver can consume one at a time, in order, as they
/// arrive. Chunk `k` covers positions `bound(k)..bound(k + 1)`, always on a
/// coset boundary.
///
/// A chunk count above the coset count would produce empty chunks, so it is
/// clamped; `chunks == 1` is the un-pipelined layout.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkMap {
    /// `perm[β]` is bucket `β`'s destination position. Empty when the
    /// permutation is the identity (`r == 0`), which is what a layer whose
    /// local deltas are all zero — a rotation with a remote generator, the
    /// common case — produces.
    perm: Vec<u32>,
    /// `inv[p]` is the bucket at position `p`. Empty with [`Self::perm`].
    inv: Vec<u32>,
    /// Positions, i.e. buckets.
    positions: u32,
    /// `log2` of the coset size: coset `c` owns positions `c << r ..
    /// (c + 1) << r`.
    r: u32,
    /// Cosets, `positions >> r`.
    cosets: u32,
    /// Chunks the bulk transfer is cut into, `1..=cosets`.
    chunks: u32,
}

impl ChunkMap {
    /// Re-aim the map at `span` over `num_buckets` buckets, cut into at most
    /// `chunks` pieces, **keeping the permutation buffers' allocations**.
    ///
    /// # Panics
    ///
    /// If `num_buckets` is zero or not a power of two, or if `chunks` is zero.
    pub(crate) fn rebuild(&mut self, span: &Gf2Span, num_buckets: usize, chunks: usize) {
        assert!(
            num_buckets.is_power_of_two(),
            "ChunkMap: {num_buckets} buckets is not a power of two",
        );
        assert!(chunks > 0, "ChunkMap: a layer needs at least one chunk");
        let positions = num_buckets as u32;
        let r = span.r() as u32;
        debug_assert!(
            (1u32 << r) <= positions,
            "ChunkMap: a coset of {} buckets does not fit in {positions}",
            1u32 << r,
        );
        self.positions = positions;
        self.r = r;
        self.cosets = positions >> r;
        self.chunks = (chunks as u32).min(self.cosets).max(1);
        self.perm.clear();
        self.inv.clear();
        if r == 0 {
            // `perm_index` compresses over every bit, so it is the identity;
            // the empty vectors say so and save both the build and the
            // indirection.
            return;
        }
        self.perm.reserve(num_buckets);
        self.inv.resize(num_buckets, 0);
        for beta in 0..positions {
            let p = span.perm_index(beta);
            self.perm.push(p);
            self.inv[p as usize] = beta;
        }
    }

    /// Bucket `beta`'s destination position.
    #[inline]
    pub fn position_of(&self, beta: u32) -> u32 {
        if self.perm.is_empty() {
            beta
        } else {
            self.perm[beta as usize]
        }
    }

    /// The bucket at destination position `p` — the inverse of
    /// [`position_of`](Self::position_of).
    #[inline]
    pub fn bucket_at(&self, p: u32) -> u32 {
        if self.inv.is_empty() {
            p
        } else {
            self.inv[p as usize]
        }
    }

    /// Positions, i.e. buckets.
    pub fn positions(&self) -> usize {
        self.positions as usize
    }

    /// Chunks the bulk transfer is cut into.
    pub fn chunks(&self) -> usize {
        self.chunks as usize
    }

    /// The first position of chunk `k`; `bound(chunks())` is the position
    /// count, so chunk `k` is `bound(k)..bound(k + 1)`.
    ///
    /// Always a multiple of the coset size, so no coset straddles two chunks.
    ///
    /// # Panics
    ///
    /// If `k > chunks()`.
    pub fn bound(&self, k: usize) -> u32 {
        assert!(k <= self.chunks(), "ChunkMap: chunk {k} is out of range");
        let c = (k as u64 * u64::from(self.cosets)).div_ceil(u64::from(self.chunks)) as u32;
        c << self.r
    }

    /// The chunk position `p` belongs to.
    ///
    /// The inverse of [`bound`](Self::bound): `bound(k) <= p < bound(k + 1)`.
    #[inline]
    pub fn chunk_of_position(&self, p: u32) -> usize {
        let c = u64::from(p >> self.r);
        ((c * u64::from(self.chunks)) / u64::from(self.cosets)) as usize
    }
}

/// Fixed-size prefix describing one [`ExchangeBlock`] on the wire.
///
/// `#[repr(C)]` and `Pod`: four `u32`s, 16 bytes, no padding, so it casts to
/// bytes with no copy and reads back from an unaligned buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BlockHeader {
    /// Number of destination positions the block is indexed by — the group's
    /// agreed bucket count. `offsets` has `num_buckets + 1` entries.
    pub num_buckets: u32,
    /// Total rows in the block: `offsets[num_buckets]`, and the length of each
    /// column once the export pass has filled it.
    pub rows: u32,
    /// The width `W` the block was built at. Checked on decode: a payload
    /// encoded at one width is never silently reinterpreted at another.
    pub w: u32,
    /// Which of the layer plan's remote deltas this block carries. The
    /// receiver uses it to look up the delta's bucket offset `bd[e]` for the
    /// [`ExchangeBlock::segment`] rule.
    pub entry: u32,
}

/// The rows one remote delta moves from this partition to one partner, in CSR
/// order by the receiver's **destination position** ([`ChunkMap`]).
///
/// Columns are structure-of-arrays, matching the bucket storage they are
/// gathered from and scattered into: `x`, `z` and `coeff` are parallel and
/// their first [`rows`](Self::rows) entries are the block. `offsets` is the CSR
/// index, `num_buckets() + 1` entries, ascending, `offsets[0] == 0` and
/// `offsets[num_buckets] == rows`.
///
/// **The columns are grow-only, so they may be longer than `rows`.** A block is
/// reused across layers ([`set_counts`](Self::set_counts)) and the storage a
/// wider layer needed is kept rather than freed and re-faulted, so the row
/// count in the header is the one authority on how much of a column is live:
/// read the block through [`cols`](Self::cols) or [`segment`](Self::segment),
/// never through `x.len()`. Whatever sits past `rows` is a previous layer's
/// rows, and never travels.
///
/// The receiver never scans: for its output bucket `β′` it reads
/// [`segment`](Self::segment)`(map.position_of(β′))` and merges those rows into
/// that bucket. See the module docs for the ordering.
#[derive(Clone, Debug, Default)]
pub(crate) struct ExchangeBlock<const W: usize> {
    /// Wire prefix: source-bucket count, row count, width, remote-delta index.
    pub header: BlockHeader,
    /// CSR offsets by destination position, `num_buckets + 1` entries.
    pub offsets: Vec<u32>,
    /// X-part column, at least [`rows`](Self::rows) entries.
    pub x: Vec<[u64; W]>,
    /// Z-part column, at least [`rows`](Self::rows) entries.
    pub z: Vec<[u64; W]>,
    /// Coefficient column, at least [`rows`](Self::rows) entries.
    pub coeff: Vec<Complex64>,
}

/// Two blocks are equal when their **live** contents are: the header, the CSR
/// offsets, and the first `rows` entries of each column. The grow-only tail
/// past `rows` is a previous layer's scratch and is deliberately not compared —
/// a block built through a reused payload must equal the same block built
/// fresh.
impl<const W: usize> PartialEq for ExchangeBlock<W> {
    fn eq(&self, other: &Self) -> bool {
        self.header == other.header && self.offsets == other.offsets && self.cols() == other.cols()
    }
}

/// Everything one partner receives from this partition for one layer: the
/// blocks in ascending remote-delta index ([`BlockHeader::entry`]).
///
/// A partner with nothing to receive is sent `None` rather than an empty
/// payload (see the collective-order invariant in the module docs); an empty
/// `blocks` is legal and encodes to zero parts.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PartnerPayload<const W: usize> {
    /// One block per remote delta, ascending by [`BlockHeader::entry`].
    pub blocks: Vec<ExchangeBlock<W>>,
}

impl<const W: usize> ExchangeBlock<W> {
    /// Build the CSR skeleton for `counts[p]` rows at each destination
    /// position `p` under remote-delta index `entry`, and size the columns.
    ///
    /// [`set_counts`](Self::set_counts) on a fresh block: the columns come back
    /// `rows` long, ready for the export pass to write by index.
    ///
    /// The engine always re-aims a pooled block instead, so this is the tests'
    /// constructor.
    ///
    /// # Panics
    ///
    /// If the counts sum past `u32::MAX` rows.
    #[cfg(test)]
    pub fn with_counts(entry: u32, counts: &[u32]) -> Self {
        let mut block = Self::default();
        block.set_counts(entry, counts);
        block
    }

    /// Re-aim an existing block at `counts[p]` rows per destination position
    /// `p` under remote-delta index `entry`, **keeping every allocation**.
    ///
    /// The export pass then writes each row by index into the segment the
    /// offsets describe. The columns are only ever grown, never shrunk or
    /// re-zeroed (see the type docs): a steady-state layer re-aims a block it
    /// has already used at the same size, which touches nothing but the
    /// offsets. That is the whole point of holding the payloads across layers —
    /// the alternative is faulting in and zeroing the block's megabytes again
    /// every layer.
    ///
    /// # Panics
    ///
    /// If the counts sum past `u32::MAX` rows.
    pub fn set_counts(&mut self, entry: u32, counts: &[u32]) {
        self.offsets.clear();
        self.offsets.reserve(counts.len() + 1);
        self.offsets.push(0u32);
        let mut rows = 0u32;
        for &c in counts {
            rows = rows
                .checked_add(c)
                .expect("exchange block exceeds u32::MAX rows");
            self.offsets.push(rows);
        }
        self.header = BlockHeader {
            num_buckets: counts.len() as u32,
            rows,
            w: W as u32,
            entry,
        };
        self.grow_columns(rows as usize);
    }

    /// Make every column at least `rows` long, keeping what is there.
    fn grow_columns(&mut self, rows: usize) {
        if self.coeff.len() < rows {
            self.x.resize(rows, [0u64; W]);
            self.z.resize(rows, [0u64; W]);
            self.coeff.resize(rows, Complex64::new(0.0, 0.0));
        }
    }

    /// The block's live rows: the first [`rows`](Self::rows) entries of the
    /// three columns.
    pub fn cols(&self) -> (&[[u64; W]], &[[u64; W]], &[Complex64]) {
        let rows = self.rows();
        (&self.x[..rows], &self.z[..rows], &self.coeff[..rows])
    }

    /// The rows destined for position `p`, as parallel `x` / `z` / `coeff`
    /// slices.
    ///
    /// The receiver's rule for its own output bucket `β′` is
    /// `segment(map.position_of(β′))` (module docs). A position with no rows
    /// yields three empty slices.
    ///
    /// # Panics
    ///
    /// If `p >= num_buckets()`, or if the block's columns are shorter
    /// than [`rows`](Self::rows) — which only a hand-built block can be.
    pub fn segment(&self, p: u32) -> (&[[u64; W]], &[[u64; W]], &[Complex64]) {
        let lo = self.offsets[p as usize] as usize;
        let hi = self.offsets[p as usize + 1] as usize;
        (&self.x[lo..hi], &self.z[lo..hi], &self.coeff[lo..hi])
    }

    /// The row index each of `map`'s chunk boundaries falls at.
    ///
    /// `chunks + 1` ascending entries starting at 0 and ending at
    /// [`rows`](Self::rows): chunk `k` carries rows `out[k]..out[k + 1]` of
    /// every column.
    pub fn chunk_rows(&self, map: &ChunkMap) -> Vec<usize> {
        chunk_rows_of(&self.offsets, map)
    }

    /// Rows the block carries: `offsets[num_buckets]`, and the length of each
    /// column once the export pass has filled it.
    pub fn rows(&self) -> usize {
        self.header.rows as usize
    }

    /// Destination positions the block is indexed by — the group's agreed
    /// bucket count, which both sides hold.
    pub fn num_buckets(&self) -> u32 {
        self.header.num_buckets
    }

    /// Wire footprint in bytes: the header, the offsets, and the live rows of
    /// the three columns ([`Payload::byte_parts`] hands out exactly these
    /// bytes).
    pub fn bytes(&self) -> usize {
        size_of::<BlockHeader>()
            + self.offsets.len() * size_of::<u32>()
            + 2 * self.rows() * W * size_of::<u64>()
            + self.rows() * size_of::<Complex64>()
    }
}

/// Something a [`Transport`] can move between partitions as bytes.
///
/// [`byte_parts`](Self::byte_parts) is zero-copy — borrowed views of the
/// payload's own columns, one part per column — so a sending transport never
/// packs or allocates. [`recv_into`](Self::recv_into) +
/// [`finish_recv`](Self::finish_recv) is its mirror: the payload sizes its own
/// typed columns from the part lengths the wire declared and hands out mutable
/// byte views *of those columns*, so a transport receives straight into the
/// storage the engine will read, with no decode pass and no second buffer.
/// That is why a payload is `Default` — a transport pools them across layers,
/// so the steady state allocates nothing.
///
/// The in-process transport moves the typed value and calls none of them.
pub trait Payload: Default + Send + 'static {
    /// Borrowed byte views of this payload's columns, in wire order.
    fn byte_parts(&self) -> Vec<&[u8]>;

    /// Reshape this payload for an incoming message whose parts have byte
    /// lengths `lens`, and hand out one mutable byte view per part to receive
    /// into — the same parts, in the same order,
    /// [`byte_parts`](Self::byte_parts) would produce for it.
    ///
    /// Reuses whatever storage the payload already holds and only ever grows
    /// it. The views are of the payload's own typed columns, so filling them is
    /// the decode.
    ///
    /// # Panics
    ///
    /// If `lens` is not a shape this payload can take (a partial block, a part
    /// length that is not a whole number of elements).
    fn recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]>;

    /// Check what [`recv_into`](Self::recv_into) could not: the invariants that
    /// only hold once the bytes have arrived.
    ///
    /// Called by the transport after the receives complete, before the payload
    /// is handed to the engine.
    ///
    /// # Panics
    ///
    /// If the received payload is inconsistent — a block encoded at another
    /// width, offsets that disagree with the row count.
    fn finish_recv(&mut self);

    /// The parts the engine reads **before** it reads a single row, so a
    /// two-phase transport must have them in hand before it hands the payload
    /// over: for the layer's exchange blocks that is the block headers and the
    /// CSR offsets, a
    /// few tens of kilobytes against a layer's hundreds of megabytes, and all
    /// the engine needs to size a gather run (`ExtraRows::count`).
    ///
    /// A prefix of [`byte_parts`](Self::byte_parts) in the same order — the
    /// framing still declares every part's length, so the receiver can size the
    /// bulk columns from the header alone. The default is *every* part, which
    /// is what a payload with no interesting internal structure wants: a
    /// transport then behaves exactly as a blocking one.
    ///
    fn early_parts(&self) -> Vec<&[u8]> {
        self.byte_parts()
    }

    /// The rest of [`byte_parts`](Self::byte_parts), each split into `map`'s
    /// chunks: `bulk_parts()[i][k]` is part `i`'s slice for chunk `k`.
    ///
    /// A chunk is a contiguous range of the receiver's destination positions
    /// ([`ChunkMap`]), so a column's chunk is the byte range of the rows in
    /// those positions, which the CSR offsets give exactly. Both sides compute
    /// it from the same offsets, so the sender's pieces and the receiver's
    /// posted receives line up with no further exchange.
    ///
    /// The default is no bulk parts at all, the counterpart of
    /// [`early_parts`](Self::early_parts)'s default.
    fn bulk_parts(&self, map: &ChunkMap) -> Vec<Vec<&[u8]>> {
        let _ = map;
        Vec::new()
    }

    /// [`recv_into`](Self::recv_into), returning only the
    /// [`early_parts`](Self::early_parts) views.
    ///
    /// It still sizes *every* column from `lens` — that is what lets
    /// [`bulk_recv_into`](Self::bulk_recv_into) slice them afterwards — but the
    /// bulk views are not handed out yet, so the payload can be inspected
    /// ([`finish_recv`](Self::finish_recv)) between the two phases with no
    /// outstanding borrow.
    fn early_recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
        self.recv_into(lens)
    }

    /// The receive-side mirror of [`bulk_parts`](Self::bulk_parts): the same
    /// parts, in the same order, split into the same chunks.
    ///
    /// Called after the early parts have arrived, so the CSR offsets the split
    /// reads are the ones the sender used.
    fn bulk_recv_into(&mut self, map: &ChunkMap) -> Vec<Vec<&mut [u8]>> {
        let _ = map;
        Vec::new()
    }
}

/// What a coset task waits on before it reads a received row.
///
/// The partitioned layer hands one of these to its `RecvRows`, which calls
/// [`wait_chunk`](Self::wait_chunk) at the top of `append_into` with the chunk
/// its output bucket belongs to. A blocking transport's implementation is
/// empty: everything arrived before the body ever ran.
///
/// `Sync` because the coset loop calls it from every Rayon worker at once. An
/// implementation that talks to MPI must therefore serialize itself — one
/// thread in the library at a time is exactly `MPI_THREAD_SERIALIZED`.
pub trait ChunkWait: Sync {
    /// Block until every row of chunk `k` has arrived.
    ///
    /// Must be safe to call concurrently, repeatedly, and out of order — a
    /// coset task knows only its own chunk, and Rayon decides who runs when.
    fn wait_chunk(&self, k: usize);
}

/// The [`ChunkWait`] of a transport whose exchange already completed.
pub(crate) struct AlreadyHere;

impl ChunkWait for AlreadyHere {
    fn wait_chunk(&self, _k: usize) {}
}

/// Parts per block in the [`PartnerPayload`] encoding: header, offsets, x, z,
/// coeff.
const PARTS_PER_BLOCK: usize = 5;

/// The row index each of `map`'s chunk boundaries falls at, `chunks + 1`
/// ascending entries — the CSR offsets read at the chunks' destination
/// positions.
fn chunk_rows_of(offsets: &[u32], map: &ChunkMap) -> Vec<usize> {
    debug_assert_eq!(
        offsets.len(),
        map.positions() + 1,
        "the chunk map is built for a different bucket count than the block",
    );
    (0..=map.chunks())
        .map(|k| offsets[map.bound(k) as usize] as usize)
        .collect()
}

/// Cut `col` at `bounds` (row indices) and view each piece as bytes.
fn chunk_slices<'s, T, F>(col: &'s [T], bounds: &[usize], as_bytes: F) -> Vec<&'s [u8]>
where
    F: Fn(&'s [T]) -> &'s [u8],
{
    bounds
        .windows(2)
        .map(|w| as_bytes(&col[w[0]..w[1]]))
        .collect()
}

/// [`chunk_slices`] for the receive side: disjoint mutable pieces, in order.
fn chunk_slices_mut<'s, T, F>(col: &'s mut [T], bounds: &[usize], as_bytes: F) -> Vec<&'s mut [u8]>
where
    F: Fn(&'s mut [T]) -> &'s mut [u8],
{
    let mut rest = col;
    let mut out = Vec::with_capacity(bounds.len().saturating_sub(1));
    let mut at = bounds[0];
    for &edge in &bounds[1..] {
        let (head, tail) = rest.split_at_mut(edge - at);
        out.push(as_bytes(head));
        rest = tail;
        at = edge;
    }
    out
}

/// Panic unless a declared part length is a whole number of `stride`-byte
/// elements — the one thing a receiver can check about a part before its bytes
/// exist.
fn check_stride(len: usize, stride: usize, what: &str) {
    assert_eq!(
        len % stride,
        0,
        "exchange block {what}: {len} bytes is not a whole number of {stride}-byte entries",
    );
}

impl<const W: usize> Payload for PartnerPayload<W> {
    fn byte_parts(&self) -> Vec<&[u8]> {
        let mut parts = Vec::with_capacity(PARTS_PER_BLOCK * self.blocks.len());
        for block in &self.blocks {
            let (x, z, coeff) = block.cols();
            parts.push(bytemuck::bytes_of(&block.header));
            parts.push(bytemuck::cast_slice(&block.offsets));
            parts.push(bytemuck::cast_slice(x.as_flattened()));
            parts.push(bytemuck::cast_slice(z.as_flattened()));
            parts.push(bytemuck::cast_slice(coeff));
        }
        parts
    }

    /// Size the blocks from the declared part lengths — five parts per block,
    /// and each block's shape is readable from the lengths alone: `offsets`
    /// gives the bucket count, the `x` column the row count — then hand out the
    /// columns as bytes.
    ///
    /// The header travels into [`ExchangeBlock::header`] itself, so nothing
    /// about a block is known twice; [`finish_recv`](Payload::finish_recv)
    /// checks it against the shape sized here once it has arrived.
    fn recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
        assert_eq!(
            lens.len() % PARTS_PER_BLOCK,
            0,
            "partner payload: {} parts is not a whole number of {PARTS_PER_BLOCK}-part blocks",
            lens.len(),
        );
        let n = lens.len() / PARTS_PER_BLOCK;
        let key_stride = W * size_of::<u64>();
        // Grow-only, like the columns: a pooled payload keeps the blocks it
        // held last layer and re-aims them.
        // Truncate first, grow second: the views handed out below borrow the
        // blocks for as long as the caller holds them, so `self.blocks` cannot
        // be touched again after that.
        self.blocks.truncate(n);
        if self.blocks.len() < n {
            self.blocks.resize_with(n, ExchangeBlock::<W>::default);
        }
        let mut parts = Vec::with_capacity(lens.len());
        for (block, lens) in self
            .blocks
            .iter_mut()
            .zip(lens.chunks_exact(PARTS_PER_BLOCK))
        {
            assert_eq!(
                lens[0],
                size_of::<BlockHeader>(),
                "partner payload: block header is {} bytes, expected {}",
                lens[0],
                size_of::<BlockHeader>(),
            );
            check_stride(lens[1], size_of::<u32>(), "offsets");
            check_stride(lens[2], key_stride, "x column");
            check_stride(lens[3], key_stride, "z column");
            check_stride(lens[4], size_of::<Complex64>(), "coeff column");
            assert_eq!(
                lens[2], lens[3],
                "partner payload: x column is {} bytes and z column {}",
                lens[2], lens[3],
            );
            let rows = lens[2] / key_stride;
            assert_eq!(
                lens[4] / size_of::<Complex64>(),
                rows,
                "partner payload: coeff column carries {} rows where the keys carry {rows}",
                lens[4] / size_of::<Complex64>(),
            );
            block.offsets.clear();
            block.offsets.resize(lens[1] / size_of::<u32>(), 0);
            block.grow_columns(rows);
            // The row count the wire declared, so `cols` and `segment` see
            // exactly what arrives; the header's own copy is checked against it
            // in `finish_recv`.
            block.header.rows = rows as u32;
            let ExchangeBlock {
                header,
                offsets,
                x,
                z,
                coeff,
            } = block;
            parts.push(bytemuck::bytes_of_mut(header));
            parts.push(bytemuck::cast_slice_mut(&mut offsets[..]));
            parts.push(bytemuck::cast_slice_mut(x[..rows].as_flattened_mut()));
            parts.push(bytemuck::cast_slice_mut(z[..rows].as_flattened_mut()));
            parts.push(bytemuck::cast_slice_mut(&mut coeff[..rows]));
        }
        parts
    }

    fn finish_recv(&mut self) {
        for block in &self.blocks {
            assert_eq!(
                block.header.w as usize, W,
                "partner payload: block encoded at width W={} decoded at W={W}",
                block.header.w,
            );
            // The header travelled with the columns, so it could name more rows
            // than the columns the declared part lengths sized. It never does
            // between ranks running the same build; the check is what keeps a
            // garbled header an assertion rather than an out-of-bounds read.
            assert!(
                block.header.rows as usize <= block.coeff.len(),
                "partner payload: a block header claims {} rows but only {} arrived",
                block.header.rows,
                block.coeff.len(),
            );
            assert_eq!(
                block.offsets.len(),
                block.header.num_buckets as usize + 1,
                "partner payload: a block indexed by {} buckets arrived with {} offsets",
                block.header.num_buckets,
                block.offsets.len(),
            );
            assert_eq!(
                block.offsets.last().copied().unwrap_or(0),
                block.header.rows,
                "partner payload: offsets end at {:?}, the columns carry {} rows",
                block.offsets.last(),
                block.header.rows,
            );
        }
    }

    /// Parts 0 and 1 of every block — the header and the CSR offsets. Those
    /// are what `RecvRows::count` reads to size a gather run; the three columns
    /// after them are read only inside `append_into`.
    fn early_parts(&self) -> Vec<&[u8]> {
        let mut parts = Vec::with_capacity(2 * self.blocks.len());
        for block in &self.blocks {
            parts.push(bytemuck::bytes_of(&block.header));
            parts.push(bytemuck::cast_slice(&block.offsets));
        }
        parts
    }

    /// The three columns of each block, in block order, each cut at the chunks'
    /// destination-position boundaries.
    fn bulk_parts(&self, map: &ChunkMap) -> Vec<Vec<&[u8]>> {
        let mut parts = Vec::with_capacity(3 * self.blocks.len());
        for block in &self.blocks {
            let rows = block.rows();
            let (x, z, coeff) = (&block.x[..rows], &block.z[..rows], &block.coeff[..rows]);
            let bounds = block.chunk_rows(map);
            parts.push(chunk_slices(x, &bounds, |s| {
                bytemuck::cast_slice(s.as_flattened())
            }));
            parts.push(chunk_slices(z, &bounds, |s| {
                bytemuck::cast_slice(s.as_flattened())
            }));
            parts.push(chunk_slices(coeff, &bounds, bytemuck::cast_slice));
        }
        parts
    }

    fn early_recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
        let mut parts = self.recv_into(lens);
        // `recv_into` sized every column and handed out all five views per
        // block; keep the header and the offsets and drop the columns, which
        // `bulk_recv_into` hands out once their shape is known.
        let mut early = Vec::with_capacity(2 * parts.len() / PARTS_PER_BLOCK);
        for (i, view) in parts.drain(..).enumerate() {
            if i % PARTS_PER_BLOCK < 2 {
                early.push(view);
            }
        }
        early
    }

    fn bulk_recv_into(&mut self, map: &ChunkMap) -> Vec<Vec<&mut [u8]>> {
        let mut parts = Vec::with_capacity(3 * self.blocks.len());
        for block in &mut self.blocks {
            let rows = block.header.rows as usize;
            let bounds = chunk_rows_of(&block.offsets, map);
            let ExchangeBlock { x, z, coeff, .. } = block;
            parts.push(chunk_slices_mut(&mut x[..rows], &bounds, |s| {
                bytemuck::cast_slice_mut(s.as_flattened_mut())
            }));
            parts.push(chunk_slices_mut(&mut z[..rows], &bounds, |s| {
                bytemuck::cast_slice_mut(s.as_flattened_mut())
            }));
            parts.push(chunk_slices_mut(
                &mut coeff[..rows],
                &bounds,
                bytemuck::cast_slice_mut,
            ));
        }
        parts
    }
}

/// The laps a [`Transport::exchange_layer`] takes inside itself, for
/// [`Transport::drain_timings`] to hand the engine's [`PhaseStats`] (feature
/// `phase-timing`).
///
/// Atomics because `exchange` takes `&self`; uncontended, and one `Relaxed`
/// add per phase per layer, so the counters cost nothing measurable next to
/// the phases they measure. Every field is a *part of* `exchange_ns`.
///
/// [`PhaseStats`]: crate::engine::stats::PhaseStats
// Only the MPI transport takes these laps (the in-process one moves payloads),
// so without the `mpi` feature the type has no constructor and would be dead code.
#[cfg(all(feature = "phase-timing", feature = "mpi"))]
#[derive(Debug, Default)]
pub(crate) struct ExchangeTimings {
    /// Encoding the framing headers and posting every send.
    pub(crate) send_post_ns: AtomicU64,
    /// Blocking receive of each partner's framing header.
    pub(crate) hdr_wait_ns: AtomicU64,
    /// Sizing the receive buffers.
    pub(crate) recv_alloc_ns: AtomicU64,
    /// Posting the part receives and waiting out whatever the coset loop did
    /// not already drive to completion.
    pub(crate) data_wait_ns: AtomicU64,
}

#[cfg(all(feature = "phase-timing", feature = "mpi"))]
impl ExchangeTimings {
    /// Add `since.elapsed()` to `slot` and re-arm the stamp.
    pub(crate) fn lap(slot: &AtomicU64, since: &mut Instant) {
        let now = Instant::now();
        slot.fetch_add(
            now.duration_since(*since).as_nanos() as u64,
            Ordering::Relaxed,
        );
        *since = now;
    }

    /// Move every lap into `stats`, leaving the counters at zero.
    pub(crate) fn drain_into(&self, stats: &mut crate::engine::stats::PhaseStats) {
        stats.send_post_ns += self.send_post_ns.swap(0, Ordering::Relaxed);
        stats.hdr_wait_ns += self.hdr_wait_ns.swap(0, Ordering::Relaxed);
        stats.recv_alloc_ns += self.recv_alloc_ns.swap(0, Ordering::Relaxed);
        stats.data_wait_ns += self.data_wait_ns.swap(0, Ordering::Relaxed);
    }
}

/// The collective operations a partition needs outside the exchange itself:
/// its identity in the group, the two reductions a layer's truncation and
/// bookkeeping need, and a barrier.
///
/// Both reductions must return **the identical value on every partition** —
/// callers use them to agree on a global decision (a truncation threshold, a
/// term-count total), and a partition that computed a different answer would
/// diverge silently. Both are exact and order-independent (a maximum, and a
/// wrapping integer sum), so an implementation is free to combine in arrival
/// order; one whose reduction were *not* order-independent would have to
/// combine in rank order.
///
/// Every method obeys the collective-order invariant in the module docs: all
/// partitions call them in the same order, the same number of times.
pub trait Collectives: Send + Sync {
    /// This partition's index in the group, `0 <= rank < size`.
    fn rank(&self) -> u32;
    /// Number of partitions in the group.
    fn size(&self) -> u32;
    /// Maximum of `v` over the group. Same value on every partition.
    fn allreduce_max_u8(&self, v: u8) -> u8;
    /// Element-wise sum of `buf` over the group, in place. Every partition
    /// passes the same length and gets the same values back. Sums wrap rather
    /// than panic on overflow, so debug and release agree.
    fn allreduce_sum_u64(&self, buf: &mut [u64]);
    /// Block until every partition has arrived.
    fn barrier(&self);

    /// Panic unless every partition passed the same `fingerprint`.
    ///
    /// The intended fingerprint is whatever a run's partitions *must* agree on
    /// before they can be driven in lock-step — the channel count, the
    /// direction, the qubit count, the truncation policy's identity. Disagree
    /// on any of those and the group deadlocks on the first layer whose
    /// collectives no longer pair up; this turns that hang into a message. The
    /// driver calls it exactly once per propagation, before the first layer.
    ///
    /// One collective, and it obeys the collective-order invariant like the
    /// rest: every partition calls it, with its own fingerprint.
    ///
    /// # Panics
    ///
    /// If the fingerprints differ, naming a bit the group disagrees on and how
    /// many partitions set it.
    fn check_consistency(&self, fingerprint: u64) {
        // 64 counters, one per bit of the fingerprint: "how many partitions
        // have this bit set". A group that agrees answers 0 or `size` for
        // every bit; a group that disagrees cannot, because the counts pin
        // every partition's value bit by bit — so the test is exact, with no
        // false positives and no false negatives, and it needs nothing from
        // `Collectives` beyond the sum that is already there. 512 bytes once
        // per propagation.
        let mut counts = [0u64; 64];
        for (i, c) in counts.iter_mut().enumerate() {
            *c = (fingerprint >> i) & 1;
        }
        self.allreduce_sum_u64(&mut counts);
        let size = u64::from(self.size());
        for (bit, &count) in counts.iter().enumerate() {
            assert!(
                count == 0 || count == size,
                "the partitions disagree about the run: partition {} offered fingerprint \
                 {fingerprint:#018x}, and {count} of {size} partitions set bit {bit} of theirs. \
                 Every partition must be driven through the same circuit, in the same direction, \
                 under the same policy and options — a group that is not stays in step only by \
                 luck.",
                self.rank(),
            );
        }
    }
}

/// The per-layer all-to-all: each partition hands over what it exports and
/// gets back what its partners exported to it.
///
/// Not object-safe ([`exchange`](Self::exchange) is generic over the payload),
/// which is deliberate: the layer code is generic over the transport, so a
/// call monomorphizes into the partition's driving thread with no virtual
/// dispatch on a per-layer path.
pub trait Transport: Collectives {
    /// **The** exchange: send `send[q]` to partition `q`, run `body` on what
    /// the partners sent here, and give both back.
    ///
    /// It is **two-phase**. Everything the coset loop needs to size its gather
    /// runs — the block headers and the CSR offsets
    /// ([`Payload::early_parts`]) — has arrived before `body` starts; the rows
    /// themselves ([`Payload::bulk_parts`]) may still be in flight while it
    /// runs, and `body` blocks on the [`ChunkWait`] it is handed before it
    /// reads a chunk's rows. A transport with nothing to overlap hands over
    /// `AlreadyHere` and is a blocking exchange with extra steps — which is
    /// what [`InProcessTransport`] is, its "transfer" being a moved pointer.
    ///
    /// `send.len()` must be [`size`](Collectives::size) and
    /// `send[self.rank()]` must be `None`; the slots handed to `body` have the
    /// same length and `None` in the same self slot. A partner with nothing to
    /// send still participates, with `None` — silence would desynchronize the
    /// group (module docs).
    ///
    /// `map` is the destination-coset order both sides laid the blocks out in
    /// and the chunks the bulk transfer is cut into. Every partition passes the
    /// same one, for the same reason both sides agree on the bucket count.
    ///
    /// `spare` is the caller's **payload pool**, and it is what keeps a
    /// steady-state layer from allocating: a transport that has to materialize
    /// the received payloads takes them from here rather than building them
    /// fresh, and returns any payload from `send` it is finished with. The
    /// caller returns the results to the pool once the layer has consumed them.
    /// Pooled payloads are in an unspecified state — a taker reshapes one
    /// before it reads anything back — and an empty pool is always legal, so a
    /// transport must fall back to [`Default`]. [`InProcessTransport`] uses
    /// neither direction: it *moves* the sender's payload to the receiver, so
    /// the pool circulates through the partners instead.
    ///
    /// `body` returns whatever the caller needs out of the layer; the received
    /// payloads come back with it, for the caller to return to `spare`.
    fn exchange_layer<P, F, R>(
        &self,
        send: Vec<Option<P>>,
        spare: &mut Vec<P>,
        map: &ChunkMap,
        body: F,
    ) -> (Vec<Option<P>>, R)
    where
        P: Payload,
        F: FnOnce(&[Option<P>], &dyn ChunkWait) -> R;

    /// [`exchange_layer`](Self::exchange_layer) with nothing to overlap: the
    /// blocking all-to-all, for a caller that wants the payloads and no more.
    ///
    /// The empty [`ChunkMap`] cuts no chunks, so every part travels as an early
    /// one — which is what a payload with no interesting internal structure
    /// wants, and the gather's is the only one. A payload whose `bulk_parts`
    /// needs a real map goes through `exchange_layer`.
    fn exchange<P: Payload>(&self, send: Vec<Option<P>>, spare: &mut Vec<P>) -> Vec<Option<P>> {
        self.exchange_layer(send, spare, &ChunkMap::default(), |_, _| ())
            .0
    }

    /// Fold the sub-phase laps this transport took inside
    /// [`exchange_layer`](Self::exchange_layer) into `stats`, and reset them.
    ///
    /// Measurement only (feature `phase-timing`), called by the partitioned
    /// layer right after the exchange. The default records nothing, which is
    /// what a transport whose exchange has no interesting internal structure
    /// wants: [`InProcessTransport`] moves a typed payload through a channel
    /// and has no encode or wait to attribute. `MpiTransport` overrides it.
    #[cfg(feature = "phase-timing")]
    fn drain_timings(&self, _stats: &mut crate::engine::stats::PhaseStats) {}

    /// Collect every partition's `parts` on partition 0.
    ///
    /// `Some(v)` on rank 0, where `v[q]` is partition `q`'s parts in the order
    /// it passed them (`v[0]` being the caller's own); `None` everywhere else.
    /// This is the gather that ends a distributed run: each partition hands
    /// over its share of the sum as bytes and rank 0 reassembles.
    ///
    /// One collective. Unlike the exchange it is deliberately *not* symmetric —
    /// every partition talks to rank 0 and to nobody else — so a transport that
    /// infers its partner set from the `Some` positions (as the MPI one does)
    /// overrides this rather than inheriting it. The default body is the honest
    /// all-to-all: rank 0 sends nothing and receives from everyone.
    fn gather_to_root(&self, parts: Vec<&[u8]>) -> Option<Vec<Vec<Vec<u8>>>> {
        let n = self.size() as usize;
        let me = self.rank() as usize;
        let mine = ByteParts(parts.iter().map(|p| p.to_vec()).collect());

        if me != ROOT {
            let mut send: Vec<Option<ByteParts>> = (0..n).map(|_| None).collect();
            send[ROOT] = Some(mine);
            self.exchange(send, &mut Vec::new());
            return None;
        }

        let mut recv = self.exchange(
            (0..n).map(|_| None).collect::<Vec<Option<ByteParts>>>(),
            &mut Vec::new(),
        );
        let mut mine = Some(mine);
        Some(
            (0..n)
                .map(|q| {
                    let got = if q == ROOT {
                        mine.take()
                    } else {
                        recv[q].take()
                    };
                    got.unwrap_or_else(|| {
                        panic!(
                            "gather_to_root: partition {q} sent nothing (it must send its \
                                parts, empty or not)"
                        )
                    })
                    .0
                })
                .collect(),
        )
    }
}

/// The partition every gather lands on.
pub(crate) const ROOT: usize = 0;

/// A [`Payload`] that is already bytes: the gather's wire form.
///
/// [`Transport::gather_to_root`]'s default body moves one of these per
/// partition through [`Transport::exchange`], so a transport gets a gather for
/// free from the exchange it already implements.
#[derive(Default)]
pub(crate) struct ByteParts(pub(crate) Vec<Vec<u8>>);

impl Payload for ByteParts {
    fn byte_parts(&self) -> Vec<&[u8]> {
        self.0.iter().map(Vec::as_slice).collect()
    }

    /// Already bytes, so "receiving into the typed columns" is receiving into
    /// the parts themselves.
    fn recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
        self.0.resize_with(lens.len(), Vec::new);
        for (part, &len) in self.0.iter_mut().zip(lens) {
            part.clear();
            part.resize(len, 0);
        }
        self.0.iter_mut().map(Vec::as_mut_slice).collect()
    }

    fn finish_recv(&mut self) {}
}

// ---- the shared-memory collective state ------------------------------------

/// Which transport call a published generation word belongs to.
///
/// Stamped into every word a rank publishes, so a rank waiting for generation
/// `g` can tell "my partner has not arrived yet" from "my partner issued a
/// *different* call at `g`" — the collective-order invariant (module docs).
/// Never zero: a slot that was never written reads as generation 0, kind 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum CallKind {
    Barrier = 1,
    MaxU8 = 2,
    SumU64 = 3,
    Exchange = 4,
}

impl CallKind {
    /// The name used in a mismatch panic. Also the `op` string a partner-death
    /// panic names, so the two messages agree on what a call is called.
    fn name(self) -> &'static str {
        match self {
            CallKind::Barrier => "barrier",
            CallKind::MaxU8 => "allreduce_max_u8",
            CallKind::SumU64 => "allreduce_sum_u64",
            CallKind::Exchange => "exchange",
        }
    }

    /// The name of a raw kind byte off a published word, which may be a kind
    /// this build does not know (or the 0 of a never-written slot).
    fn name_of(raw: u8) -> &'static str {
        match raw {
            1 => "barrier",
            2 => "allreduce_max_u8",
            3 => "allreduce_sum_u64",
            4 => "exchange",
            _ => "no call",
        }
    }
}

/// `spin_loop` hints a waiting rank issues before it starts yielding instead.
///
/// The mechanism is a few hundred nanoseconds between dedicated threads, so
/// the fast path — the partner is already here, or arrives within a couple of
/// microseconds — never leaves this tier. Roughly 15 µs of `pause` on the
/// reference host.
const SPINS_BEFORE_YIELD: u32 = 1_000;

/// `yield_now` calls after the spin tier before the waiter starts sleeping.
///
/// Covers the ordinary case the spins do not: the partner is a few tens of
/// microseconds behind because its share of the layer was bigger. A `P = 16`
/// group on a smaller box (the test suite runs one) also makes progress here
/// rather than livelocking.
const YIELDS_BEFORE_SLEEP: u32 = 100;

/// How long a waiter sleeps per iteration once even yielding has not helped.
///
/// `sched_yield` in a loop is not free to the rest of the machine: it re-enters
/// the run queue and takes its fair share of the CPU, which on a partitioned
/// run is a share of the CPUs the partner's *own* workers are trying to finish
/// the layer on. Measured on a loaded reference host, a waiter that only spun
/// and yielded inflated its partner's coset loop by 10–60% and fed the skew
/// back into itself. Past a wait this long the partner is not close, so paying
/// up to one step of extra latency to stay off its cores is the right trade.
const SLEEP_STEP: Duration = Duration::from_micros(50);

/// Spin iterations between two checks of the departure mask and the deadline
/// while still in the spin tier. Past it, both are checked every iteration —
/// a yield or a sleep dwarfs two loads.
///
/// Both live on lines nobody writes in steady state, but keeping them out of
/// the tight loop leaves the fast path a single load. 64 iterations is well
/// under a microsecond, so a dead partner is still reported promptly.
const CHECKS_EVERY: u32 = 64;

/// How long a rank waits for a partner before declaring it dead.
///
/// The backstop, not the mechanism: a partner that panics drops its transport
/// and is reported within `CHECKS_EVERY` spins ([`InProcessTransport::drop`]).
/// This bound only catches a partner that is neither dead nor arriving — a
/// deadlock elsewhere in the process — so it is generous.
const WAIT_TIMEOUT: Duration = Duration::from_secs(10);

/// One rank's contribution to one generation, double-buffered by generation
/// parity (see [`GroupState`] for why two are enough).
#[derive(Default)]
struct ValueSlot {
    /// `(generation << 16) | (kind << 8) | u8 payload`, published last-but-one
    /// with `Release`. Generation 0 means "never written".
    tag: AtomicU64,
    /// Elements of `buf` that belong to this generation.
    len: AtomicUsize,
    /// The `allreduce_sum_u64` contribution.
    ///
    /// Written only by the owning rank and only while no partner can be
    /// reading it, which is what the parity split buys (see [`GroupState`]);
    /// the elements are atomics so that a *hypothetical* overlap is a stale
    /// read rather than undefined behaviour, and the `UnsafeCell` is there for
    /// the resize, which needs `&mut`. Allocated on first use and reused at
    /// the same length ever after.
    buf: UnsafeCell<Vec<AtomicU64>>,
}

/// SAFETY: `buf`'s exclusivity is established by the generation protocol
/// documented on [`GroupState`], not by Rust's borrow checker.
unsafe impl Sync for ValueSlot {}

impl ValueSlot {
    /// Store `values` as this generation's contribution.
    ///
    /// # Safety
    ///
    /// The caller must be the rank that owns this slot, and no partner may be
    /// reading it — [`GroupState`]'s parity argument.
    unsafe fn write_buf(&self, values: &[u64]) {
        let buf = &mut *self.buf.get();
        if buf.len() != values.len() {
            buf.clear();
            buf.resize_with(values.len(), AtomicU64::default);
        }
        for (slot, &v) in buf.iter().zip(values) {
            slot.store(v, Ordering::Relaxed);
        }
        self.len.store(values.len(), Ordering::Relaxed);
    }

    /// This generation's contribution.
    ///
    /// # Safety
    ///
    /// The caller must have observed the owner's `Release` of the generation
    /// it is reading (so the elements are visible) and must be inside the
    /// window in which the owner cannot be writing — [`GroupState`]'s parity
    /// argument.
    unsafe fn read_buf(&self) -> &[AtomicU64] {
        let buf: &Vec<AtomicU64> = &*self.buf.get();
        &buf[..]
    }
}

/// One rank's publication point: everything a partner reads to learn where
/// that rank is and what it contributed.
///
/// Padded to 128 bytes — two x86 cache lines, the granularity the hardware
/// prefetcher pairs — so `P` ranks polling each other never share a line.
#[repr(align(128))]
#[derive(Default)]
struct RankSlot {
    /// `(generation << 8) | kind` of the last call this rank published:
    /// **monotone**, overwritten every call, both parities.
    ///
    /// It is what makes a desynchronized partner a panic instead of a hang: a
    /// rank waiting for generation `g` sees a partner that ran *past* `g`
    /// here, even though the partner's `values` slot for `g`'s parity never
    /// got `g`'s tag.
    progress: AtomicU64,
    /// The value published at each generation parity.
    values: [ValueSlot; 2],
}

/// The collective state one [`InProcessTransport`] group shares.
///
/// # The protocol
///
/// Each rank numbers its own calls: generation 1, 2, 3, … in the order it
/// issues them. Since every rank issues the identical sequence (module docs),
/// the `g`-th call of every rank is the same call. To make call `g` of kind
/// `k`, a rank
///
/// 1. writes its contribution into `slots[rank].values[g % 2]` (the buffer
///    first, then `len`, both `Relaxed`),
/// 2. `Release`-stores `(g << 16) | (k << 8) | payload` into that slot's
///    `tag`,
/// 3. `Release`-stores `(g << 8) | k` into `slots[rank].progress`, and
/// 4. spins on every partner's `progress` until it reads generation `g` or
///    later, then reads that partner's `values[g % 2]`.
///
/// Nobody resets anything and no rank waits to be released, so back-to-back
/// calls cannot mix: the generation *is* the sense, and each rank owns the
/// word it publishes.
///
/// # Memory ordering
///
/// The `Release` on `progress` (3) and the `Acquire` that reads it (4) are the
/// only synchronization. A reader that observes generation `g` in `progress`
/// therefore also observes everything the writer did before that store — the
/// tag, `len`, and the buffer elements — so those may be loaded `Relaxed`
/// afterwards. (`tag` is stored `Release` and loaded `Acquire` as well, which
/// is redundant on the path through `progress` but costs nothing measurable
/// and keeps the slot readable on its own.) `departed` is `Release`/`Acquire`
/// for the same reason in the other direction: a panicking rank's mask bit
/// must not be observed before the writes that preceded it.
///
/// # Why two value buffers are enough
///
/// A rank may only overwrite `values[p]` when no partner can still be reading
/// it. Every call — including [`Transport::exchange`], which receives from
/// every partner — completes only after its rank has observed *all* partners
/// publish that generation. So:
///
/// > rank `r` finished call `g` ⟹ every rank `s` published `g` ⟹ every `s`
/// > had finished call `g − 1`, reads included.
///
/// Rank `r` starts call `g + 1` only after finishing `g`, and `g + 1` writes
/// parity `(g + 1) % 2`, whose previous use was `g − 1` — which the chain
/// above shows every partner has finished. The same argument bounds how far
/// ahead a partner can be: while `r` waits at `g`, no partner can be past
/// `g + 1`, so the tag `r` reads at parity `g % 2` is `g`'s and not `g + 2`'s.
struct GroupState {
    /// Ranks in the group.
    size: u32,
    /// Bit `q` set once rank `q`'s transport has been dropped — it will
    /// publish nothing further. `P ≤ 16`, so a `u32` mask is ample.
    departed: AtomicU32,
    /// One publication point per rank, in rank order.
    slots: Box<[RankSlot]>,
}

impl GroupState {
    fn new(size: u32) -> Self {
        Self {
            size,
            departed: AtomicU32::new(0),
            slots: (0..size).map(|_| RankSlot::default()).collect(),
        }
    }

    /// The slot rank `rank` publishes generation `gen` into.
    fn value_slot(&self, rank: u32, gen: u64) -> &ValueSlot {
        &self.slots[rank as usize].values[(gen & 1) as usize]
    }

    /// Publish generation `gen` of kind `kind` carrying `payload`.
    ///
    /// Steps 2 and 3 of the protocol; the caller has already done step 1 if
    /// its call carries a buffer.
    fn publish(&self, rank: u32, gen: u64, kind: CallKind, payload: u8) {
        let k = kind as u64;
        self.value_slot(rank, gen).tag.store(
            (gen << 16) | (k << 8) | u64::from(payload),
            Ordering::Release,
        );
        self.slots[rank as usize]
            .progress
            .store((gen << 8) | k, Ordering::Release);
    }

    /// Wait for rank `src` to publish generation `gen` of kind `kind`, and
    /// return its tag word (whose low byte is the `u8` payload).
    ///
    /// `waiter` is only used to name this rank in a mismatch panic.
    ///
    /// # Panics
    ///
    /// If `src` published a different call at `gen`, or ran past `gen` without
    /// publishing it (the collective-order invariant); if `src`'s transport
    /// has been dropped without publishing `gen` (it panicked); or if `src`
    /// has neither arrived nor died within [`WAIT_TIMEOUT`].
    fn wait(&self, waiter: u32, src: u32, gen: u64, kind: CallKind) -> u64 {
        let slot = &self.slots[src as usize];
        let value = &slot.values[(gen & 1) as usize];
        let mut spins: u32 = 0;
        let mut waiting_since: Option<Instant> = None;
        loop {
            let progress = slot.progress.load(Ordering::Acquire);
            if progress >> 8 >= gen {
                let tag = value.tag.load(Ordering::Acquire);
                if tag >> 16 == gen && (tag >> 8) & 0xff == kind as u64 {
                    return tag;
                }
                panic!(
                    "collective order mismatch: partition {waiter} is at transport call {gen} \
                     ({}) but partition {src} published {} at call {} — every partition must \
                     issue the identical sequence of transport calls per layer",
                    kind.name(),
                    CallKind::name_of(((tag >> 8) & 0xff) as u8),
                    tag >> 16,
                );
            }

            spins = spins.saturating_add(1);
            if spins >= SPINS_BEFORE_YIELD || spins.is_multiple_of(CHECKS_EVERY) {
                if self.departed.load(Ordering::Acquire) & (1 << src) != 0
                    // A rank that published this generation and then left the
                    // group is not a dead partner: re-read before condemning
                    // it, so the last collective of a call cannot race the
                    // partner's return.
                    && slot.progress.load(Ordering::Acquire) >> 8 < gen
                {
                    panic!(
                        "partition {src} terminated before completing the {} (it panicked)",
                        kind.name(),
                    );
                }
                let since = waiting_since.get_or_insert_with(Instant::now);
                if since.elapsed() > WAIT_TIMEOUT {
                    panic!(
                        "partition {src} terminated before completing the {}: no response in \
                         {} s (this partition is at transport call {gen}, that one at {})",
                        kind.name(),
                        WAIT_TIMEOUT.as_secs(),
                        slot.progress.load(Ordering::Acquire) >> 8,
                    );
                }
            }
            // Three tiers, cheapest first: burn a few microseconds where the
            // partner is about to arrive, hand the core over where it is a
            // layer's skew behind, and get off the machine entirely where it
            // is further than that.
            if spins < SPINS_BEFORE_YIELD {
                std::hint::spin_loop();
            } else if spins < SPINS_BEFORE_YIELD + YIELDS_BEFORE_SLEEP {
                std::thread::yield_now();
            } else {
                std::thread::sleep(SLEEP_STEP);
            }
        }
    }

    /// Wait for every partner of `rank` to publish generation `gen` of kind
    /// `kind`, folding their tag words in **rank order** through `fold`.
    fn wait_all(&self, rank: u32, gen: u64, kind: CallKind, mut fold: impl FnMut(u32, u64)) {
        for src in 0..self.size {
            if src != rank {
                fold(src, self.wait(rank, src, gen, kind));
            }
        }
    }
}

/// One message on an in-process channel: the payload, plus (debug builds) the
/// sender's collective sequence number.
struct Message {
    /// The sender's collective counter at the time of the send. Debug only —
    /// the check it feeds is a development tripwire, not a wire field.
    #[cfg(debug_assertions)]
    seq: u64,
    /// `Option<P>` for an exchange, the contribution for a reduction, `()` for
    /// a barrier. Typed on receive by [`downcast`].
    body: Box<dyn std::any::Any + Send>,
}

impl Message {
    fn new(seq: u64, body: Box<dyn std::any::Any + Send>) -> Self {
        let _ = seq;
        Self {
            #[cfg(debug_assertions)]
            seq,
            body,
        }
    }
}

/// Take a received message's body as `T`.
///
/// A failure means two partitions ran different transport calls at the same
/// step — the collective-order invariant (module docs) — so it is a panic, not
/// an error return.
fn downcast<T: 'static>(body: Box<dyn std::any::Any + Send>, from: usize, op: &str) -> T {
    match body.downcast::<T>() {
        Ok(value) => *value,
        Err(_) => panic!(
            "partition {from} sent a different kind of message during the {op}: the partitions \
             issued different transport calls (every partition must issue the identical sequence \
             of transport calls per layer)",
        ),
    }
}

/// In-process transport: `P` partitions sharing one `GroupState` for the
/// collectives, and wired as a `P × P` matrix of unbounded
/// `std::sync::mpsc` channels for the exchange.
///
/// Built as a group by [`group`](Self::group) and moved one per partition
/// thread. Deliberately Rayon-free: it is called from the partition's driving
/// thread between layers, never from inside a parallel region — and by *one*
/// thread per rank, since the generation counter is that thread's call index.
///
/// The three [`Collectives`] operations spin on shared atomics (module docs
/// and `GroupState`): a few hundred nanoseconds between dedicated threads,
/// where a channel round trip cost a few microseconds of futex sleep and wake
/// on the per-layer path. The exchange keeps the channels — it moves a
/// payload, and only on the layers that have one. Channels are unbounded, so a
/// send never blocks and the "send everything, then receive in rank order"
/// shape cannot deadlock. Its receives block; a partner that died is reported
/// by name rather than waited on forever (its dropped sender disconnects the
/// channel).
///
/// `size == 1` is a no-op path: no channels exist, no generation is consumed,
/// [`Transport::exchange`] returns one `None`, and the reductions return their
/// input.
pub struct InProcessTransport {
    /// This partition's index.
    rank: u32,
    /// Partitions in the group.
    size: u32,
    /// The group's shared collective state, one `Arc` per rank. It outlives
    /// every rank's endpoint, which is what lets a partner read a slot's
    /// buffer while its owner is on its way out.
    state: Arc<GroupState>,
    /// This rank's transport-call counter: one increment per call, the
    /// generation published to [`GroupState`] and (in debug builds) the stamp
    /// on every exchange message. Atomic only because the methods take
    /// `&self`; nothing but this rank's driving thread touches it.
    gen: AtomicU64,
    /// Sender to partition `q`, `None` in the self slot.
    outbox: Vec<Option<std::sync::mpsc::Sender<Message>>>,
    /// Receiver of what partition `q` sends here, `None` in the self slot.
    /// `Mutex` only to make the transport `Sync` — `Receiver` is `Send` but
    /// not `Sync`, and nothing here contends for it.
    inbox: Vec<Option<std::sync::Mutex<std::sync::mpsc::Receiver<Message>>>>,
}

/// Leaving the group marks this rank departed, so a partner spinning for a
/// generation this rank will never publish fails fast instead of waiting out
/// its `WAIT_TIMEOUT` backstop.
///
/// Unconditional rather than `if std::thread::panicking()`: a rank that
/// returns from its partition body with fewer collectives than its partners is
/// exactly as dead to them as one that panicked, and the panic message they
/// raise says so. It cannot fire spuriously at the end of a healthy call —
/// `GroupState::wait` re-reads the partner's progress before condemning it,
/// and a rank only leaves after publishing the group's last generation.
impl Drop for InProcessTransport {
    fn drop(&mut self) {
        self.state
            .departed
            .fetch_or(1 << self.rank, Ordering::Release);
    }
}

impl InProcessTransport {
    /// Build a group of `size` transports wired to each other, one per
    /// partition, in rank order.
    ///
    /// # Panics
    ///
    /// If `size` is zero.
    pub fn group(size: u32) -> Vec<InProcessTransport> {
        assert!(size > 0, "a transport group needs at least one partition");
        let n = size as usize;
        let mut senders: Vec<Vec<Option<std::sync::mpsc::Sender<Message>>>> =
            (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
        let mut receivers: Vec<Vec<Option<std::sync::mpsc::Receiver<Message>>>> =
            (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
        for src in 0..n {
            for dst in 0..n {
                if src != dst {
                    let (tx, rx) = std::sync::mpsc::channel();
                    senders[src][dst] = Some(tx);
                    receivers[src][dst] = Some(rx);
                }
            }
        }

        let state = Arc::new(GroupState::new(size));
        (0..n)
            .map(|rank| InProcessTransport {
                rank: rank as u32,
                size,
                state: Arc::clone(&state),
                gen: AtomicU64::new(0),
                outbox: std::mem::take(&mut senders[rank]),
                inbox: (0..n)
                    .map(|src| receivers[src][rank].take().map(std::sync::Mutex::new))
                    .collect(),
            })
            .collect()
    }

    /// The generation of the transport call starting now: one more than the
    /// last, and never zero (an unwritten slot reads as generation 0).
    fn next_gen(&self) -> u64 {
        self.gen.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Pretend one extra collective was issued, to test the order check.
    #[cfg(test)]
    fn skip_sequence_for_test(&self) {
        self.gen.fetch_add(1, Ordering::Relaxed);
    }

    /// Send one message to partition `dst`.
    fn send_to(&self, dst: usize, seq: u64, body: Box<dyn std::any::Any + Send>, op: &str) {
        let tx = self.outbox[dst]
            .as_ref()
            .expect("a partition has no channel to itself");
        if tx.send(Message::new(seq, body)).is_err() {
            panic!("partition {dst} terminated before completing the {op} (it panicked)");
        }
    }

    /// Receive this call's message from partition `src`, checking the
    /// sequence stamp in debug builds.
    fn recv_from(&self, src: usize, seq: u64, op: &str) -> Box<dyn std::any::Any + Send> {
        let rx = self.inbox[src]
            .as_ref()
            .expect("a partition has no channel to itself")
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match rx.recv() {
            Ok(message) => {
                #[cfg(debug_assertions)]
                assert_eq!(
                    message.seq, seq,
                    "collective order mismatch: partition {} is at transport call {seq} but \
                     partition {src} sent its call {} — every partition must issue the identical \
                     sequence of transport calls per layer",
                    self.rank, message.seq,
                );
                let _ = seq;
                message.body
            }
            Err(_) => {
                panic!("partition {src} terminated before completing the {op} (it panicked)")
            }
        }
    }
}

impl Collectives for InProcessTransport {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn size(&self) -> u32 {
        self.size
    }

    /// The maximum rides in the published word's payload byte, so the whole
    /// reduction is one store and `P − 1` loads. `max` is order-independent,
    /// so every partition returns the same byte however the arrivals
    /// interleave.
    fn allreduce_max_u8(&self, v: u8) -> u8 {
        if self.size == 1 {
            return v;
        }
        let gen = self.next_gen();
        self.state.publish(self.rank, gen, CallKind::MaxU8, v);

        let mut acc = v;
        self.state
            .wait_all(self.rank, gen, CallKind::MaxU8, |_, tag| {
                acc = acc.max((tag & 0xff) as u8);
            });
        acc
    }

    /// Each partition publishes its own contribution once and folds its
    /// partners' in place.
    ///
    /// The local sum runs in rank order, but it would not have to: wrapping
    /// `u64` addition is exact, associative and commutative, so every
    /// partition ends with the identical bits whatever order it folds in.
    fn allreduce_sum_u64(&self, buf: &mut [u64]) {
        if self.size == 1 {
            return;
        }
        let gen = self.next_gen();
        let slot = self.state.value_slot(self.rank, gen);
        // SAFETY: this rank owns the slot, and the generation parity keeps
        // every partner out of it — `GroupState`'s "why two value buffers are
        // enough". The write must precede the publish below, which is what
        // makes it visible to the partners at all.
        unsafe { slot.write_buf(buf) };
        self.state.publish(self.rank, gen, CallKind::SumU64, 0);

        let n = self.size;
        for src in 0..n {
            if src == self.rank {
                continue;
            }
            self.state.wait(self.rank, src, gen, CallKind::SumU64);
            let theirs = self.state.value_slot(src, gen);
            // SAFETY: `wait` returned, so this rank has observed `src`'s
            // `Release` of this generation — its buffer and length are visible
            // — and by the parity argument `src` cannot write the slot again
            // before this rank publishes its next generation.
            let values = unsafe { theirs.read_buf() };
            assert_eq!(
                values.len(),
                buf.len(),
                "allreduce_sum_u64: partition {src} contributed {} values, this partition {}",
                values.len(),
                buf.len(),
            );
            for (a, v) in buf.iter_mut().zip(values) {
                *a = a.wrapping_add(v.load(Ordering::Relaxed));
            }
        }
    }

    fn barrier(&self) {
        if self.size == 1 {
            return;
        }
        let gen = self.next_gen();
        self.state.publish(self.rank, gen, CallKind::Barrier, 0);
        self.state
            .wait_all(self.rank, gen, CallKind::Barrier, |_, _| {});
    }
}

impl Transport for InProcessTransport {
    /// A payload is *moved* to its partner rather than copied, so there is
    /// nothing for the two-phase shape to overlap: the transfer is complete
    /// before `body` runs and its [`ChunkWait`] is a no-op. `map` therefore
    /// goes unread, and `spare` is neither drawn from nor added to — a sender's
    /// blocks become the receiver's, and the pool circulates through the group
    /// rather than through the transport.
    fn exchange_layer<P, F, R>(
        &self,
        send: Vec<Option<P>>,
        _spare: &mut Vec<P>,
        _map: &ChunkMap,
        body: F,
    ) -> (Vec<Option<P>>, R)
    where
        P: Payload,
        F: FnOnce(&[Option<P>], &dyn ChunkWait) -> R,
    {
        let n = self.size as usize;
        assert_eq!(
            send.len(),
            n,
            "exchange: send has {} entries, expected one entry per partition ({n})",
            send.len(),
        );
        assert!(
            send[self.rank as usize].is_none(),
            "exchange: send[{}] is this partition's own slot and must be None",
            self.rank,
        );
        let recv: Vec<Option<P>> = if n == 1 {
            vec![None]
        } else {
            let seq = self.next_gen();
            // The exchange consumes a generation like any other call and
            // publishes it before sending, even though it waits on the channels
            // rather than on the slots: that is what lets a partner spinning in
            // a *collective* at the same generation see the kind mismatch and
            // panic, instead of the pair hanging on each other's different call.
            self.state.publish(self.rank, seq, CallKind::Exchange, 0);
            // Unbounded channels: every send completes before the first
            // receive, so no pair of partitions can block on each other.
            for (dst, payload) in send.into_iter().enumerate() {
                if dst != self.rank as usize {
                    self.send_to(dst, seq, Box::new(payload), "exchange");
                }
            }
            (0..n)
                .map(|src| {
                    if src == self.rank as usize {
                        None
                    } else {
                        downcast::<Option<P>>(self.recv_from(src, seq, "exchange"), src, "exchange")
                    }
                })
                .collect()
        };

        let out = body(&recv, &AlreadyHere);
        (recv, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;

    // ---- the destination-coset order and its chunks -----------------------

    /// The map for `deltas` over `2^bits` buckets, cut into `chunks`.
    fn map_of(bits: u8, deltas: &[u32], chunks: usize) -> ChunkMap {
        let mut map = ChunkMap::default();
        map.rebuild(&Gf2Span::new(deltas, bits), 1usize << bits, chunks);
        map
    }

    /// A payload of `blocks` blocks over `2^bits` positions, with a few rows
    /// per position, filled deterministically.
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

    /// The two-phase framing is a partition of the one-phase one: the early
    /// parts are the block headers and offsets, and each bulk part's chunks
    /// concatenate to exactly the column that part carries.
    ///
    /// That is the whole contract between the sender's `bulk_parts` and the
    /// receiver's `bulk_recv_into` — get it wrong and the two sides post
    /// different message sizes, which MPI reports as a truncated message rather
    /// than as wrong rows.
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
                        let joined: Vec<u8> =
                            pieces.iter().flat_map(|p| p.iter().copied()).collect();
                        let want = &all[PARTS_PER_BLOCK * (i / 3) + 2 + i % 3];
                        assert_eq!(&joined, want, "part {i} at {chunks} chunks");
                    }
                    // The receive side cuts the same column the same way.
                    let want_lens: Vec<Vec<usize>> = bulk
                        .iter()
                        .map(|p| p.iter().map(|c| c.len()).collect())
                        .collect();
                    let got_lens: Vec<Vec<usize>> = payload
                        .bulk_recv_into(&map)
                        .iter()
                        .map(|p| p.iter().map(|c| c.len()).collect())
                        .collect();
                    assert_eq!(got_lens, want_lens, "receive side at {chunks} chunks");
                }
            }
        }
    }

    /// Both directions of the layout are one permutation: every bucket has one
    /// position, every position one bucket, and nothing is dropped or
    /// duplicated. This is the property the sender's permuted CSR and the
    /// receiver's table lookup both rest on.
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
            let map = map_of(bits, &deltas, 1);
            let n = 1u32 << bits;
            assert_eq!(map.positions(), n as usize);
            let mut seen = vec![false; n as usize];
            for beta in 0..n {
                let p = map.position_of(beta);
                assert!(p < n, "position {p} outside {n} for bits={bits}");
                assert!(!seen[p as usize], "position {p} claimed twice");
                seen[p as usize] = true;
                assert_eq!(map.bucket_at(p), beta, "bucket_at is not the inverse");
            }
            assert!(
                seen.iter().all(|&s| s),
                "bits={bits} left a position unfilled"
            );
        }
    }

    /// The chunks tile the position range: ascending bounds from 0 to the
    /// position count, and `chunk_of_position` is their inverse.
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

    /// A chunk boundary always falls between cosets, so a coset task never has
    /// to wait for two chunks — which is what makes the receiver's per-chunk
    /// wait sound.
    #[test]
    fn a_chunk_never_splits_a_coset() {
        for (bits, deltas) in [(6u8, vec![0u32, 3]), (6, vec![0, 1, 2, 3]), (5, vec![0, 7])] {
            let span = Gf2Span::new(&deltas, bits);
            for chunks in [1usize, 2, 4, 8, 64] {
                let mut map = ChunkMap::default();
                map.rebuild(&span, 1usize << bits, chunks);
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
                    let want = map.chunk_of_position(map.position_of(span.rep_of(beta)));
                    assert_eq!(
                        map.chunk_of_position(map.position_of(beta)),
                        want,
                        "bucket {beta} is in another chunk than its coset representative",
                    );
                }
            }
        }
    }

    /// Asking for more chunks than there are cosets would make empty ones; the
    /// map clamps instead, so the pipeline degrades to one batch per coset.
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
        /// The two properties above over arbitrary spans and chunk counts.
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
            let map = map_of(bits, &deltas, chunks);
            let mut seen = vec![false; hi as usize];
            for beta in 0..hi {
                let p = map.position_of(beta);
                prop_assert!(p < hi);
                prop_assert!(!seen[p as usize]);
                seen[p as usize] = true;
                prop_assert_eq!(map.bucket_at(p), beta);
                let c = map.chunk_of_position(p);
                prop_assert!(map.bound(c) <= p && p < map.bound(c + 1));
            }
            prop_assert_eq!(map.bound(map.chunks()) as usize, map.positions());
        }
    }

    /// Deterministic pseudo-random row filler: xorshift64, so the tests carry
    /// no RNG dependency and a failing case is reproducible from its seed.
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
                w: 2,
                entry: 3,
            }
        );
        assert_eq!(block.rows(), 8);
        assert_eq!(block.num_buckets(), 4);
        // Sized, not filled: the export pass writes the rows by index.
        assert_eq!(block.x.len(), 8);
        assert_eq!(block.z.len(), 8);
        assert_eq!(block.coeff.len(), 8);
    }

    /// A block re-aimed at a new layer keeps its storage and reports the new
    /// shape: the columns grow to what the widest layer needed and stay there,
    /// so a narrower layer neither shrinks nor re-zeroes them — and `rows()`,
    /// not `x.len()`, is what says how much of a column is live.
    #[test]
    fn set_counts_reuses_the_columns_and_grows_only() {
        let mut block = ExchangeBlock::<1>::with_counts(0, &[4, 4]);
        let wide = block.x.as_ptr();
        assert_eq!(block.rows(), 8);

        block.set_counts(1, &[1, 2]);
        assert_eq!(block.rows(), 3);
        assert_eq!(block.offsets, vec![0, 1, 3]);
        assert_eq!(block.header.entry, 1);
        assert_eq!(block.cols().2.len(), 3, "three live rows");
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

        // An empty source bucket yields three empty slices, not a panic.
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

    /// Move `payload` over the wire and back: [`Payload::byte_parts`] on the
    /// sender, [`Payload::recv_into`] + [`Payload::finish_recv`] on a fresh
    /// receiver, with the bytes copied across the way a transport moves them.
    ///
    /// That pair is the only encode/decode path the engine has, so it is what
    /// the wire-format tests exercise.
    fn wire_round_trip<const W: usize>(payload: &PartnerPayload<W>) -> PartnerPayload<W> {
        let sent: Vec<Vec<u8>> = payload
            .byte_parts()
            .into_iter()
            .map(<[u8]>::to_vec)
            .collect();
        let lens: Vec<usize> = sent.iter().map(Vec::len).collect();
        let mut back = PartnerPayload::<W>::default();
        for (view, bytes) in back.recv_into(&lens).into_iter().zip(&sent) {
            view.copy_from_slice(bytes);
        }
        back.finish_recv();
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
        // And the CSR indexing survives byte-for-byte.
        assert_eq!(back.blocks[1].segment(1).0, payload.blocks[1].segment(1).0);
    }

    /// A block whose header says it was built at another width is rejected
    /// rather than reinterpreted. The declared part lengths cannot catch it —
    /// they are a whole number of rows either way — so
    /// [`Payload::finish_recv`] is what does.
    #[test]
    #[should_panic(expected = "width")]
    fn a_block_encoded_at_another_width_is_rejected() {
        let mut payload = payload_of::<2>(2, 1, 0x5);
        payload.blocks[0].header.w = 1;
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

    // ---- transport ------------------------------------------------------

    /// A minimal [`Payload`] for the transport tests: one column of `u64`.
    impl Payload for Vec<u64> {
        fn byte_parts(&self) -> Vec<&[u8]> {
            vec![bytemuck::cast_slice(&self[..])]
        }

        fn recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
            assert_eq!(lens.len(), 1, "test payload: expected one part");
            check_stride(lens[0], size_of::<u64>(), "test payload");
            self.clear();
            self.resize(lens[0] / size_of::<u64>(), 0);
            vec![bytemuck::cast_slice_mut(&mut self[..])]
        }

        fn finish_recv(&mut self) {}
    }

    /// What rank `from` sends to rank `to`: a `from`-long column of a code
    /// unique to the ordered pair, so a crossed delivery cannot pass.
    fn message(from: u32, to: u32) -> Vec<u64> {
        vec![(u64::from(from) << 32) | u64::from(to); from as usize + 1]
    }

    /// Rank 0 sends nothing to rank 1, so a `None` slot is exercised too.
    fn sends_nothing(from: u32, to: u32) -> bool {
        from == 0 && to == 1
    }

    fn exchange_round(size: u32) {
        let group = InProcessTransport::group(size);
        assert_eq!(group.len(), size as usize);

        std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| {
                    scope.spawn(move || {
                        let rank = transport.rank();
                        assert_eq!(transport.size(), size);
                        let send: Vec<Option<Vec<u64>>> = (0..size)
                            .map(|q| {
                                (q != rank && !sends_nothing(rank, q)).then(|| message(rank, q))
                            })
                            .collect();
                        (rank, transport.exchange(send, &mut Vec::new()))
                    })
                })
                .collect();

            for handle in handles {
                let (rank, recv) = handle.join().expect("rank thread panicked");
                assert_eq!(recv.len(), size as usize, "rank {rank}");
                for q in 0..size {
                    let expected = (q != rank && !sends_nothing(q, rank)).then(|| message(q, rank));
                    assert_eq!(recv[q as usize], expected, "rank {rank} slot {q}");
                }
            }
        });
    }

    #[test]
    fn exchange_delivers_each_payload_to_its_partner() {
        for size in [2u32, 4] {
            exchange_round(size);
        }
    }

    #[test]
    fn reductions_and_barrier_agree_on_every_rank() {
        let size = 4u32;
        let group = InProcessTransport::group(size);

        let mut results: Vec<(u32, u8, Vec<u64>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| {
                    scope.spawn(move || {
                        let rank = transport.rank();
                        transport.barrier();
                        // Contributions 0, 7, 14, 21 → max 21.
                        let max = transport.allreduce_max_u8((rank * 7) as u8);
                        // Columns (r+1, 10·(r+1)) → sums (10, 100).
                        let mut buf = vec![u64::from(rank) + 1, 10 * (u64::from(rank) + 1)];
                        transport.allreduce_sum_u64(&mut buf);
                        transport.barrier();
                        (rank, max, buf)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread panicked"))
                .collect()
        });

        results.sort_by_key(|(rank, _, _)| *rank);
        assert_eq!(results.len(), size as usize);
        for (rank, max, sum) in results {
            assert_eq!(max, 21, "rank {rank}");
            assert_eq!(sum, vec![10, 100], "rank {rank}");
        }
    }

    #[test]
    fn a_group_of_one_is_a_no_op() {
        let group = InProcessTransport::group(1);
        assert_eq!(group.len(), 1);
        let transport = &group[0];
        assert_eq!(transport.rank(), 0);
        assert_eq!(transport.size(), 1);

        let recv: Vec<Option<Vec<u64>>> = transport.exchange(vec![None], &mut Vec::new());
        assert_eq!(recv.len(), 1);
        assert!(recv[0].is_none());

        assert_eq!(transport.allreduce_max_u8(9), 9);
        let mut buf = vec![3, 4];
        transport.allreduce_sum_u64(&mut buf);
        assert_eq!(buf, vec![3, 4]);
        transport.barrier();
    }

    #[test]
    #[should_panic(expected = "must be None")]
    fn exchange_rejects_a_payload_addressed_to_this_partition() {
        let group = InProcessTransport::group(1);
        let _: Vec<Option<Vec<u64>>> = group[0].exchange(vec![Some(vec![1u64])], &mut Vec::new());
    }

    #[test]
    #[should_panic(expected = "one entry per partition")]
    fn exchange_rejects_a_wrongly_sized_send_vector() {
        let group = InProcessTransport::group(2);
        let _: Vec<Option<Vec<u64>>> = group[0].exchange(vec![None], &mut Vec::new());
    }

    /// A partition that issues one collective more than its partners is
    /// caught by the generation stamp rather than crossing payloads.
    ///
    /// The *in-step* rank is the one that names it: the desynchronized rank
    /// has published a generation its partner never reached, so the partner
    /// finds a call it did not issue in that generation's slot. The
    /// desynchronized rank is left waiting for a generation that will never
    /// come, and dies of its partner's departure instead — both panics are
    /// checked here, and both ranks own their endpoint inside their own thread
    /// so neither drop waits on the other.
    #[test]
    fn a_desynchronized_partition_is_caught_by_the_generation_stamp() {
        let mut group = InProcessTransport::group(2);
        let one = group.pop().expect("rank 1");
        let zero = group.pop().expect("rank 0");

        let (desynced, in_step) = std::thread::scope(|scope| {
            // Rank 0 behaves as if it had issued one extra collective.
            let desynced = scope.spawn(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    zero.skip_sequence_for_test();
                    zero.barrier();
                }))
            });
            let in_step = scope.spawn(move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || one.barrier()))
            });
            (
                desynced.join().expect("rank 0 thread"),
                in_step.join().expect("rank 1 thread"),
            )
        });

        let in_step = panic_message(
            in_step
                .expect_err("the in-step rank must reject the mismatched generation")
                .as_ref(),
        );
        assert!(in_step.contains("collective order mismatch"), "{in_step}");
        let desynced = panic_message(
            desynced
                .expect_err("the desynchronized rank waits for a call nobody makes")
                .as_ref(),
        );
        assert!(desynced.contains("terminated"), "{desynced}");
    }

    /// The panic message behind a `catch_unwind` payload.
    fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
        if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "<panic payload is not a string>".to_string()
        }
    }

    /// A rank that dies *between* two collectives must fail its partners fast
    /// — through the departure mask, not the 10 s backstop — and name itself.
    #[test]
    fn a_partner_that_dies_between_collectives_fails_the_others_promptly() {
        let mut group = InProcessTransport::group(2);
        let one = group.pop().expect("rank 1");
        let zero = group.pop().expect("rank 0");

        let (elapsed, message) = std::thread::scope(|scope| {
            scope.spawn(move || {
                // The transport is dropped *while unwinding*, which is what a
                // partitioned run does when a partition body panics.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    one.barrier();
                    panic!("rank 1 dies after its barrier");
                }));
            });
            zero.barrier();
            let started = Instant::now();
            let payload =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| zero.allreduce_max_u8(3)))
                    .expect_err("the surviving rank cannot complete a collective alone");
            (started.elapsed(), panic_message(payload.as_ref()))
        });

        assert!(message.contains("terminated"), "{message}");
        assert!(message.contains("partition 1"), "{message}");
        assert!(
            elapsed < Duration::from_secs(2),
            "the survivor waited {elapsed:?}, so it fell back on the {WAIT_TIMEOUT:?} timeout \
             instead of noticing the departure",
        );
    }

    /// A thousand back-to-back reductions on four ranks with rank- *and*
    /// round-dependent contributions: a generation that mixed with its
    /// neighbour shows up as a wrong sum, not as a hang.
    #[test]
    fn a_thousand_back_to_back_sums_never_mix_generations() {
        const ROUNDS: u64 = 1_000;
        let size = 4u32;
        let group = InProcessTransport::group(size);

        std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| {
                    scope.spawn(move || {
                        let rank = u64::from(transport.rank());
                        for round in 1..=ROUNDS {
                            let mut buf = vec![rank * round, round, rank];
                            transport.allreduce_sum_u64(&mut buf);
                            // Σ rank = 0+1+2+3 = 6, Σ round = 4·round.
                            assert_eq!(
                                buf,
                                vec![6 * round, 4 * round, 6],
                                "rank {rank}, round {round}",
                            );
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("rank thread panicked");
            }
        });
    }

    /// One round of the mixed script: the first `max`, the reduced buffer, and
    /// the flag `max`.
    type MixedRound = (u8, Vec<u64>, u8);
    /// What one rank came out of the mixed script with.
    type MixedScript = (u32, Vec<MixedRound>);

    /// Reductions and barriers interleaved: every rank runs the same mixed
    /// script and every rank must come out with the same hand-computed
    /// answers.
    #[test]
    fn mixed_collective_sequences_agree_on_every_rank() {
        const ROUNDS: u8 = 25;
        let size = 4u32;
        let group = InProcessTransport::group(size);

        let mut results: Vec<MixedScript> = std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| {
                    scope.spawn(move || {
                        let rank = transport.rank();
                        let mut rounds = Vec::new();
                        for round in 0..ROUNDS {
                            let hi = transport.allreduce_max_u8(rank as u8 * 3 + round);
                            transport.barrier();
                            let mut buf =
                                vec![u64::from(rank) + 1, u64::from(round), u64::from(rank) << 8];
                            transport.allreduce_sum_u64(&mut buf);
                            transport.barrier();
                            let flag = transport.allreduce_max_u8(if rank == 2 { 255 } else { 0 });
                            rounds.push((hi, buf, flag));
                        }
                        (rank, rounds)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread panicked"))
                .collect()
        });

        results.sort_by_key(|(rank, _)| *rank);
        for (rank, rounds) in &results {
            assert_eq!(rounds.len(), ROUNDS as usize, "rank {rank}");
            for (round, (hi, sum, flag)) in rounds.iter().enumerate() {
                let round = round as u8;
                // max over 3·rank + round is 9 + round; Σ(rank+1) = 10,
                // Σ round = 4·round, Σ(rank << 8) = 6·256.
                assert_eq!(*hi, 9 + round, "rank {rank}, round {round}");
                assert_eq!(
                    *sum,
                    vec![10, 4 * u64::from(round), 6 << 8],
                    "rank {rank}, round {round}",
                );
                assert_eq!(*flag, 255, "rank {rank}, round {round}");
            }
        }
        // And identical across ranks, not merely correct on each.
        for (rank, rounds) in &results[1..] {
            assert_eq!(rounds, &results[0].1, "rank {rank} disagrees with rank 0");
        }
    }

    /// Sixteen ranks — more than a CI box has cores — must finish rather than
    /// livelock: past [`SPINS_BEFORE_YIELD`] a waiting rank hands the core to
    /// the partner it is waiting for. Completing *is* the assertion.
    #[test]
    fn sixteen_ranks_complete_when_oversubscribed() {
        const ROUNDS: u64 = 50;
        let size = 16u32;
        let group = InProcessTransport::group(size);

        std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| {
                    scope.spawn(move || {
                        let rank = transport.rank();
                        for round in 1..=ROUNDS {
                            assert_eq!(transport.allreduce_max_u8(rank as u8 + 1), 16);
                            let mut buf = vec![u64::from(rank), round];
                            transport.allreduce_sum_u64(&mut buf);
                            // Σ_{r<16} r = 120.
                            assert_eq!(buf, vec![120, 16 * round], "rank {rank}, round {round}");
                            transport.barrier();
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().expect("rank thread panicked");
            }
        });
    }

    // ---- the two collectives every transport inherits ------------------

    /// Run `f` on every rank of a fresh group of `size`, joining in rank order.
    fn on_every_rank<O: Send>(
        size: u32,
        f: impl Fn(&InProcessTransport) -> O + Send + Sync,
    ) -> Vec<O> {
        let group = InProcessTransport::group(size);
        let f = &f;
        std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| scope.spawn(move || f(&transport)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread panicked"))
                .collect()
        })
    }

    #[test]
    fn check_consistency_passes_when_every_rank_agrees() {
        for size in [1u32, 2, 4] {
            on_every_rank(size, |transport| {
                transport.check_consistency(0xdead_beef_0000_0001);
            });
        }
    }

    /// One rank out of step is named, rather than left to deadlock two layers
    /// later. Every rank sees the mismatch (the reduction is symmetric), so
    /// rank 1 swallows its own panic and only rank 0's reaches the harness.
    #[test]
    #[should_panic(expected = "disagree about the run")]
    fn check_consistency_names_a_rank_that_disagrees() {
        let mut group = InProcessTransport::group(2);
        let one = group.pop().expect("rank 1");
        let zero = group.pop().expect("rank 0");

        std::thread::scope(|scope| {
            scope.spawn(move || {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    one.check_consistency(5)
                }));
            });
            zero.check_consistency(4);
        });
    }

    #[test]
    fn gather_to_root_collects_every_ranks_parts_in_rank_order() {
        for size in [1u32, 2, 4] {
            let got = on_every_rank(size, |transport| {
                let rank = transport.rank();
                // Rank `r` contributes `r + 1` parts, part `j` being `r + 1`
                // copies of the byte `10 · r + j`, so a crossed or reordered
                // delivery cannot pass.
                let owned: Vec<Vec<u8>> = (0..=rank)
                    .map(|j| vec![(10 * rank + j) as u8; rank as usize + 1])
                    .collect();
                let parts: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
                transport.gather_to_root(parts)
            });

            for (rank, out) in got.iter().enumerate() {
                if rank != 0 {
                    assert!(out.is_none(), "rank {rank} must not gather");
                    continue;
                }
                let all = out.as_ref().expect("rank 0 gathers");
                assert_eq!(all.len(), size as usize);
                for (r, parts) in all.iter().enumerate() {
                    let r = r as u32;
                    assert_eq!(parts.len(), r as usize + 1, "rank {r} part count");
                    for (j, part) in parts.iter().enumerate() {
                        assert_eq!(part, &vec![(10 * r + j as u32) as u8; r as usize + 1]);
                    }
                }
            }
        }
    }

    #[test]
    fn gather_to_root_of_no_parts_is_an_empty_vector_per_rank() {
        let got = on_every_rank(2, |transport| transport.gather_to_root(Vec::new()));
        assert_eq!(got[0], Some(vec![Vec::<Vec<u8>>::new(); 2]));
        assert_eq!(got[1], None);
    }

    /// A partner that died mid-layer must be reported, not waited on forever.
    #[test]
    #[should_panic(expected = "terminated")]
    fn a_partner_that_panicked_is_reported_rather_than_hanging() {
        let mut group = InProcessTransport::group(2);
        let one = group.pop().expect("rank 1");
        let zero = group.pop().expect("rank 0");

        std::thread::scope(|scope| {
            scope.spawn(move || {
                let _ = std::panic::catch_unwind(|| panic!("rank 1 dies before its exchange"));
                drop(one);
            });
            let _: Vec<Option<Vec<u64>>> =
                zero.exchange(vec![None, Some(vec![7u64])], &mut Vec::new());
        });
    }
}
