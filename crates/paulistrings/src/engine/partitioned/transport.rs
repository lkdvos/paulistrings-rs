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
//! same order, on every layer.** Nothing in the transport reorders, tags by
//! kind, or matches calls up: a call's `n`-th message is paired with the
//! partner's `n`-th message positionally. A partition that skips an exchange
//! because it happens to have nothing to send, or that runs an extra
//! reduction, desynchronizes the whole group — so a layer's transport calls
//! are driven by the *plan* (which every partition computes identically from
//! the channel and the hash), never by local data. Empty is sent as `None`,
//! not as silence. Debug builds carry a per-transport sequence counter on
//! every message and assert it on receive, so a violation is a panic naming
//! both partitions rather than a hang or a silently crossed payload.
//!
//! # The wire unit: a CSR block indexed by source bucket
//!
//! Per layer a prepared channel's delta set `D` splits into deltas that keep a
//! row inside its own partition and **remote deltas**, whose partition bits
//! `pd[e]` are non-zero. For a remote delta `e`, every row partition `R`
//! generates from its local bucket `β` lands in partner `R ⊕ pd[e]`, local
//! bucket `β ⊕ bd[e]` — one partner, one bucket offset, both known before a
//! single term is touched.
//!
//! So the natural unit is one [`ExchangeBlock`] per remote delta: the rows in
//! CSR order **by source bucket**, `offsets[β]..offsets[β + 1]` addressing the
//! rows generated from source bucket `β`. The receiver filling its output
//! bucket `β′` reads
//!
//! ```text
//! block.segment(β′ ^ bd[e])
//! ```
//!
//! which is the only place the delta's bucket offset appears on the receive
//! side — the sender never permutes. A [`PartnerPayload`] is the blocks for
//! one partner in ascending remote-delta index (the block's
//! [`BlockHeader::entry`]), so the receiver walks its plan's remote deltas and
//! the payload's blocks in lockstep.
//!
//! # Bytes
//!
//! [`Payload::byte_parts`] hands out one borrowed, zero-copy `&[u8]` view per
//! column (`bytemuck::cast_slice`, no packing, no allocation);
//! [`Payload::from_byte_parts`] is the copying inverse. The in-process
//! transport never calls either — it moves the typed payload through a channel
//! — but an MPI transport implements the same traits by sending the parts.
//! Both directions are exercised by tests, so the wire format is pinned before
//! the first `MPI_Isend` exists.

use std::mem::size_of;

use num_complex::Complex64;

/// Fixed-size prefix describing one [`ExchangeBlock`] on the wire.
///
/// `#[repr(C)]` and `Pod`: four `u32`s, 16 bytes, no padding, so it casts to
/// bytes with no copy and reads back from an unaligned buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockHeader {
    /// Number of **source** buckets the block is indexed by — the sender's
    /// local bucket count. `offsets` has `num_buckets + 1` entries.
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
/// order by **source** bucket.
///
/// Columns are structure-of-arrays, matching the bucket storage they are
/// gathered from and scattered into: `x`, `z` and `coeff` are parallel, each
/// `rows()` long once filled. `offsets` is the CSR index, `num_buckets() + 1`
/// entries, ascending, `offsets[0] == 0` and `offsets[num_buckets] == rows`.
///
/// The receiver never scans: for its output bucket `β′` under remote delta `e`
/// it reads [`segment`](Self::segment)`(β′ ^ bd[e])` and merges those rows into
/// that bucket. See the module docs for where `bd[e]` comes from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExchangeBlock<const W: usize> {
    /// Wire prefix: source-bucket count, row count, width, remote-delta index.
    pub header: BlockHeader,
    /// CSR offsets by **source** bucket, `num_buckets + 1` entries.
    pub offsets: Vec<u32>,
    /// X-part column, one entry per row.
    pub x: Vec<[u64; W]>,
    /// Z-part column, one entry per row.
    pub z: Vec<[u64; W]>,
    /// Coefficient column, one entry per row.
    pub coeff: Vec<Complex64>,
}

/// Everything one partner receives from this partition for one layer: the
/// blocks in ascending remote-delta index ([`BlockHeader::entry`]).
///
/// A partner with nothing to receive is sent `None` rather than an empty
/// payload (see the collective-order invariant in the module docs); an empty
/// `blocks` is legal and encodes to zero parts.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PartnerPayload<const W: usize> {
    /// One block per remote delta, ascending by [`BlockHeader::entry`].
    pub blocks: Vec<ExchangeBlock<W>>,
}

impl<const W: usize> ExchangeBlock<W> {
    /// Build the CSR skeleton for `counts[β]` rows from each source bucket `β`
    /// under remote-delta index `entry`, and reserve the columns.
    ///
    /// The columns come back **empty with capacity**: the export pass pushes
    /// exactly `counts[β]` rows per source bucket, in ascending `β`, so the
    /// offsets it was sized from stay true and no reallocation happens
    /// mid-export.
    ///
    /// # Panics
    ///
    /// If the counts sum past `u32::MAX` rows.
    pub fn with_counts(entry: u32, counts: &[u32]) -> Self {
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0u32);
        let mut rows = 0u32;
        for &c in counts {
            rows = rows
                .checked_add(c)
                .expect("exchange block exceeds u32::MAX rows");
            offsets.push(rows);
        }
        let cap = rows as usize;
        Self {
            header: BlockHeader {
                num_buckets: counts.len() as u32,
                rows,
                w: W as u32,
                entry,
            },
            offsets,
            x: Vec::with_capacity(cap),
            z: Vec::with_capacity(cap),
            coeff: Vec::with_capacity(cap),
        }
    }

    /// The rows generated from source bucket `src_bucket`, as parallel
    /// `x` / `z` / `coeff` slices.
    ///
    /// The receiver's rule for its own output bucket `β′` under remote delta
    /// `e` is `segment(β′ ^ bd[e])` (module docs). An empty source bucket
    /// yields three empty slices.
    ///
    /// # Panics
    ///
    /// If `src_bucket >= num_buckets()`, or if the block's columns are not
    /// filled to [`rows`](Self::rows).
    pub fn segment(&self, src_bucket: u32) -> (&[[u64; W]], &[[u64; W]], &[Complex64]) {
        let lo = self.offsets[src_bucket as usize] as usize;
        let hi = self.offsets[src_bucket as usize + 1] as usize;
        (&self.x[lo..hi], &self.z[lo..hi], &self.coeff[lo..hi])
    }

    /// Rows the block carries: `offsets[num_buckets]`, and the length of each
    /// column once the export pass has filled it.
    pub fn rows(&self) -> usize {
        self.header.rows as usize
    }

    /// Source buckets the block is indexed by — the *sender's* local bucket
    /// count, which is also the receiver's (every partition holds the same
    /// number of local buckets).
    pub fn num_buckets(&self) -> u32 {
        self.header.num_buckets
    }

    /// Wire footprint in bytes: the header, the offsets, and the columns as
    /// they stand ([`Payload::byte_parts`] hands out exactly these bytes).
    pub fn bytes(&self) -> usize {
        size_of::<BlockHeader>()
            + self.offsets.len() * size_of::<u32>()
            + (self.x.len() + self.z.len()) * W * size_of::<u64>()
            + self.coeff.len() * size_of::<Complex64>()
    }
}

/// Something a [`Transport`] can move between partitions as bytes.
///
/// [`byte_parts`](Self::byte_parts) is zero-copy — borrowed views of the
/// payload's own columns, one part per column — so a sending transport never
/// packs or allocates; [`from_byte_parts`](Self::from_byte_parts) is the
/// copying inverse, and must accept unaligned input (bytes off a network
/// buffer carry no alignment guarantee).
///
/// The in-process transport moves the typed value and calls neither.
pub trait Payload: Send + 'static {
    /// Borrowed byte views of this payload's columns, in decode order.
    fn byte_parts(&self) -> Vec<&[u8]>;
    /// Rebuild a payload from the parts [`byte_parts`](Self::byte_parts)
    /// produced, in the same order. Copies.
    fn from_byte_parts(parts: &[&[u8]]) -> Self;
}

/// Parts per block in the [`PartnerPayload`] encoding: header, offsets, x, z,
/// coeff.
const PARTS_PER_BLOCK: usize = 5;

/// Copy `len` values of `T` out of a possibly unaligned byte view.
///
/// `bytemuck::cast_slice` would be free but requires the input to be aligned
/// for `T`, which a received buffer need not be; the per-element
/// `pod_read_unaligned` costs one copy on a path that is already copying.
fn decode_column<T: bytemuck::Pod>(bytes: &[u8], len: usize, what: &str) -> Vec<T> {
    let stride = size_of::<T>();
    assert_eq!(
        bytes.len(),
        len * stride,
        "exchange block {what}: expected {} bytes for {len} entries, got {}",
        len * stride,
        bytes.len(),
    );
    bytes
        .chunks_exact(stride)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

impl<const W: usize> Payload for PartnerPayload<W> {
    fn byte_parts(&self) -> Vec<&[u8]> {
        let mut parts = Vec::with_capacity(PARTS_PER_BLOCK * self.blocks.len());
        for block in &self.blocks {
            parts.push(bytemuck::bytes_of(&block.header));
            parts.push(bytemuck::cast_slice(&block.offsets));
            parts.push(bytemuck::cast_slice(block.x.as_flattened()));
            parts.push(bytemuck::cast_slice(block.z.as_flattened()));
            parts.push(bytemuck::cast_slice(&block.coeff));
        }
        parts
    }

    fn from_byte_parts(parts: &[&[u8]]) -> Self {
        assert_eq!(
            parts.len() % PARTS_PER_BLOCK,
            0,
            "partner payload: {} parts is not a whole number of {PARTS_PER_BLOCK}-part blocks",
            parts.len(),
        );
        let mut blocks = Vec::with_capacity(parts.len() / PARTS_PER_BLOCK);
        for chunk in parts.chunks_exact(PARTS_PER_BLOCK) {
            assert_eq!(
                chunk[0].len(),
                size_of::<BlockHeader>(),
                "partner payload: block header is {} bytes, expected {}",
                chunk[0].len(),
                size_of::<BlockHeader>(),
            );
            let header: BlockHeader = bytemuck::pod_read_unaligned(chunk[0]);
            assert_eq!(
                header.w as usize, W,
                "partner payload: block encoded at width W={} decoded at W={W}",
                header.w,
            );
            let rows = header.rows as usize;
            let offsets =
                decode_column::<u32>(chunk[1], header.num_buckets as usize + 1, "offsets");
            assert_eq!(
                offsets.last().copied().unwrap_or(0),
                header.rows,
                "partner payload: offsets end at {:?}, header says {} rows",
                offsets.last(),
                header.rows,
            );
            let x = decode_rows::<W>(chunk[2], rows, "x column");
            let z = decode_rows::<W>(chunk[3], rows, "z column");
            let coeff = decode_column::<Complex64>(chunk[4], rows, "coeff column");
            blocks.push(ExchangeBlock {
                header,
                offsets,
                x,
                z,
                coeff,
            });
        }
        Self { blocks }
    }
}

/// Copy `rows` key words of width `W` out of a possibly unaligned byte view.
///
/// Separate from [`decode_column`] because `[u64; W]` for a generic `W` is not
/// `Pod` under the feature set this crate builds `bytemuck` with; the words are
/// read individually and assembled.
fn decode_rows<const W: usize>(bytes: &[u8], rows: usize, what: &str) -> Vec<[u64; W]> {
    let stride = W * size_of::<u64>();
    assert_eq!(
        bytes.len(),
        rows * stride,
        "exchange block {what}: expected {} bytes for {rows} rows, got {}",
        rows * stride,
        bytes.len(),
    );
    bytes
        .chunks_exact(stride)
        .map(|row| std::array::from_fn(|i| bytemuck::pod_read_unaligned(&row[i * 8..i * 8 + 8])))
        .collect()
}

/// The collective operations a partition needs outside the exchange itself:
/// its identity in the group, the two reductions a layer's truncation and
/// bookkeeping need, and a barrier.
///
/// Both reductions must return **the identical value on every partition** —
/// callers use them to agree on a global decision (a truncation threshold, a
/// term-count total), and a partition that computed a different answer would
/// diverge silently. Implementations therefore combine contributions in rank
/// order rather than in arrival order.
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
    /// Send `send[q]` to partition `q` and return what each partition sent
    /// here: `recv[q]` is `q`'s payload, `None` where `q` sent nothing.
    ///
    /// `send.len()` must be [`size`](Collectives::size) and
    /// `send[self.rank()]` must be `None`; the returned vector has the same
    /// length and `None` in the same self slot. A partner with nothing to send
    /// still participates, with `None` — silence would desynchronize the group
    /// (module docs).
    fn exchange<P: Payload>(&self, send: Vec<Option<P>>) -> Vec<Option<P>>;

    /// Collect every partition's `parts` on partition 0.
    ///
    /// `Some(v)` on rank 0, where `v[q]` is partition `q`'s parts in the order
    /// it passed them (`v[0]` being the caller's own); `None` everywhere else.
    /// This is the gather that ends a distributed run: each partition hands
    /// over its share of the sum as bytes and rank 0 reassembles.
    ///
    /// One collective. Unlike [`exchange`](Self::exchange) it is deliberately
    /// *not* symmetric — every partition talks to rank 0 and to nobody else —
    /// so a transport whose `exchange` infers its partner set from the `Some`
    /// positions (as the MPI one does) overrides this rather than inheriting
    /// it. The default body is the honest all-to-all: rank 0 sends nothing and
    /// receives from everyone.
    fn gather_to_root(&self, parts: Vec<&[u8]>) -> Option<Vec<Vec<Vec<u8>>>> {
        let n = self.size() as usize;
        let me = self.rank() as usize;
        let mine = ByteParts(parts.iter().map(|p| p.to_vec()).collect());

        if me != ROOT {
            let mut send: Vec<Option<ByteParts>> = (0..n).map(|_| None).collect();
            send[ROOT] = Some(mine);
            self.exchange(send);
            return None;
        }

        let mut recv = self.exchange((0..n).map(|_| None).collect::<Vec<Option<ByteParts>>>());
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
pub(crate) struct ByteParts(pub(crate) Vec<Vec<u8>>);

impl Payload for ByteParts {
    fn byte_parts(&self) -> Vec<&[u8]> {
        self.0.iter().map(Vec::as_slice).collect()
    }

    fn from_byte_parts(parts: &[&[u8]]) -> Self {
        Self(parts.iter().map(|p| p.to_vec()).collect())
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

/// In-process transport: `P` partitions wired as a `P × P` matrix of
/// unbounded `std::sync::mpsc` channels, one per ordered pair.
///
/// Built as a group by [`group`](Self::group) and moved one per partition
/// thread. Deliberately Rayon-free: it is called from the partition's driving
/// thread between layers, never from inside a parallel region.
///
/// Channels are unbounded, so a send never blocks and the "send everything,
/// then receive in rank order" shape every operation here uses cannot
/// deadlock. Receives block; a partner that died is reported by name rather
/// than waited on forever (its dropped sender disconnects the channel).
///
/// `size == 1` is a no-op path: no channels exist, [`Transport::exchange`]
/// returns one `None`, and the reductions return their input.
pub struct InProcessTransport {
    /// This partition's index.
    rank: u32,
    /// Partitions in the group.
    size: u32,
    /// Sender to partition `q`, `None` in the self slot.
    outbox: Vec<Option<std::sync::mpsc::Sender<Message>>>,
    /// Receiver of what partition `q` sends here, `None` in the self slot.
    /// `Mutex` only to make the transport `Sync` — `Receiver` is `Send` but
    /// not `Sync`, and nothing here contends for it.
    inbox: Vec<Option<std::sync::Mutex<std::sync::mpsc::Receiver<Message>>>>,
    /// Collective counter: one increment per transport call, stamped on every
    /// message and asserted on receive. Debug builds only.
    #[cfg(debug_assertions)]
    seq: std::sync::atomic::AtomicU64,
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

        (0..n)
            .map(|rank| InProcessTransport {
                rank: rank as u32,
                size,
                outbox: std::mem::take(&mut senders[rank]),
                inbox: (0..n)
                    .map(|src| receivers[src][rank].take().map(std::sync::Mutex::new))
                    .collect(),
                #[cfg(debug_assertions)]
                seq: std::sync::atomic::AtomicU64::new(0),
            })
            .collect()
    }

    /// The sequence number for the transport call starting now, and advance
    /// the counter. Always zero in release builds, where messages carry no
    /// sequence number.
    fn next_seq(&self) -> u64 {
        #[cfg(debug_assertions)]
        {
            self.seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        }
        #[cfg(not(debug_assertions))]
        {
            0
        }
    }

    /// Pretend one extra collective was issued, to test the sequence check.
    #[cfg(all(test, debug_assertions))]
    fn skip_sequence_for_test(&self) {
        self.seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

    /// Send one message to every partner, built fresh per partner.
    fn broadcast(
        &self,
        seq: u64,
        mut body: impl FnMut() -> Box<dyn std::any::Any + Send>,
        op: &str,
    ) {
        for dst in 0..self.size as usize {
            if dst != self.rank as usize {
                self.send_to(dst, seq, body(), op);
            }
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

    fn allreduce_max_u8(&self, v: u8) -> u8 {
        if self.size == 1 {
            return v;
        }
        let seq = self.next_seq();
        self.broadcast(seq, || Box::new(v), "allreduce_max_u8");

        let mut acc = v;
        for src in 0..self.size as usize {
            if src == self.rank as usize {
                continue;
            }
            let their = downcast::<u8>(
                self.recv_from(src, seq, "allreduce_max_u8"),
                src,
                "allreduce_max_u8",
            );
            acc = acc.max(their);
        }
        acc
    }

    fn allreduce_sum_u64(&self, buf: &mut [u64]) {
        if self.size == 1 {
            return;
        }
        let seq = self.next_seq();
        let mine = buf.to_vec();
        self.broadcast(seq, || Box::new(mine.clone()), "allreduce_sum_u64");

        // Received first, combined second, so the summation order is rank
        // order on every partition and all of them get the same bits back.
        let n = self.size as usize;
        let mut contributions: Vec<Option<Vec<u64>>> = Vec::with_capacity(n);
        for src in 0..n {
            contributions.push((src != self.rank as usize).then(|| {
                downcast::<Vec<u64>>(
                    self.recv_from(src, seq, "allreduce_sum_u64"),
                    src,
                    "allreduce_sum_u64",
                )
            }));
        }

        let mut acc = vec![0u64; buf.len()];
        for (src, contribution) in contributions.iter().enumerate() {
            let values = contribution.as_ref().unwrap_or(&mine);
            assert_eq!(
                values.len(),
                buf.len(),
                "allreduce_sum_u64: partition {src} contributed {} values, this partition {}",
                values.len(),
                buf.len(),
            );
            for (a, v) in acc.iter_mut().zip(values) {
                *a = a.wrapping_add(*v);
            }
        }
        buf.copy_from_slice(&acc);
    }

    fn barrier(&self) {
        if self.size == 1 {
            return;
        }
        let seq = self.next_seq();
        self.broadcast(seq, || Box::new(()), "barrier");
        for src in 0..self.size as usize {
            if src != self.rank as usize {
                downcast::<()>(self.recv_from(src, seq, "barrier"), src, "barrier");
            }
        }
    }
}

impl Transport for InProcessTransport {
    fn exchange<P: Payload>(&self, send: Vec<Option<P>>) -> Vec<Option<P>> {
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
        if n == 1 {
            return vec![None];
        }

        let seq = self.next_seq();
        // Unbounded channels: every send completes before the first receive,
        // so no pair of partitions can block on each other.
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use proptest::prelude::*;

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
        // Reserved, not filled: the export pass pushes the rows.
        assert!(block.x.is_empty() && block.z.is_empty() && block.coeff.is_empty());
        assert!(block.x.capacity() >= 8);
        assert!(block.coeff.capacity() >= 8);
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

    #[test]
    fn an_empty_payload_round_trips_as_zero_parts() {
        let payload = PartnerPayload::<2>::default();
        let parts = payload.byte_parts();
        assert!(parts.is_empty());
        assert_eq!(PartnerPayload::<2>::from_byte_parts(&parts), payload);
    }

    #[test]
    fn payload_round_trips_through_byte_parts() {
        let mut payload = PartnerPayload::<2>::default();
        for (entry, counts) in [(0u32, &[2u32, 0, 1][..]), (1, &[0, 3, 0][..])] {
            let mut block = ExchangeBlock::<2>::with_counts(entry, counts);
            fill(&mut block, entry as u64 + 3);
            payload.blocks.push(block);
        }

        let back = {
            let parts = payload.byte_parts();
            PartnerPayload::<2>::from_byte_parts(&parts)
        };
        assert_eq!(back, payload);
        // And the CSR indexing survives byte-for-byte.
        assert_eq!(back.blocks[1].segment(1).0, payload.blocks[1].segment(1).0);
    }

    #[test]
    #[should_panic(expected = "width")]
    fn decoding_at_the_wrong_width_panics() {
        let mut payload = PartnerPayload::<1>::default();
        let mut block = ExchangeBlock::<1>::with_counts(0, &[1]);
        fill(&mut block, 5);
        payload.blocks.push(block);

        let parts = payload.byte_parts();
        let _ = PartnerPayload::<2>::from_byte_parts(&parts);
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
            let back = {
                let parts = payload.byte_parts();
                PartnerPayload::<1>::from_byte_parts(&parts)
            };
            prop_assert_eq!(back, payload);
        }

        #[test]
        fn arbitrary_payloads_round_trip_at_w2(payload in arb_payload::<2>()) {
            let back = {
                let parts = payload.byte_parts();
                PartnerPayload::<2>::from_byte_parts(&parts)
            };
            prop_assert_eq!(back, payload);
        }
    }

    // ---- transport ------------------------------------------------------

    /// A minimal [`Payload`] for the transport tests: one column of `u64`.
    impl Payload for Vec<u64> {
        fn byte_parts(&self) -> Vec<&[u8]> {
            vec![bytemuck::cast_slice(&self[..])]
        }

        fn from_byte_parts(parts: &[&[u8]]) -> Self {
            assert_eq!(parts.len(), 1, "test payload: expected one part");
            decode_column::<u64>(parts[0], parts[0].len() / size_of::<u64>(), "test payload")
        }
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

    fn run_exchange(size: u32) {
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
                        (rank, transport.exchange(send))
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
    fn exchange_delivers_each_payload_to_its_partner_at_p2() {
        run_exchange(2);
    }

    #[test]
    fn exchange_delivers_each_payload_to_its_partner_at_p4() {
        run_exchange(4);
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

        let recv: Vec<Option<Vec<u64>>> = transport.exchange(vec![None]);
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
        let _: Vec<Option<Vec<u64>>> = group[0].exchange(vec![Some(vec![1u64])]);
    }

    #[test]
    #[should_panic(expected = "one entry per partition")]
    fn exchange_rejects_a_wrongly_sized_send_vector() {
        let group = InProcessTransport::group(2);
        let _: Vec<Option<Vec<u64>>> = group[0].exchange(vec![None]);
    }

    /// A partition that issues one collective more than its partners is caught
    /// on the next receive rather than crossing payloads. Debug builds only:
    /// the sequence counter is `cfg(debug_assertions)`.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "collective order mismatch")]
    fn a_desynchronized_partition_is_caught_on_receive() {
        let mut group = InProcessTransport::group(2);
        let one = group.pop().expect("rank 1");
        let zero = group.pop().expect("rank 0");

        std::thread::scope(|scope| {
            scope.spawn(move || {
                // Rank 1 is in step with itself but not with rank 0, so it sees
                // the mismatch too; swallow it so only rank 0's panic reaches
                // the test harness.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| one.barrier()));
            });
            // Rank 0 behaves as if it had issued one extra collective.
            zero.skip_sequence_for_test();
            zero.barrier();
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
            let _: Vec<Option<Vec<u64>>> = zero.exchange(vec![None, Some(vec![7u64])]);
        });
    }
}
