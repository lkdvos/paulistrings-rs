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

/// Something a transport can move between partitions as bytes.
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
}
