//! The exchange wire unit: [`ExchangeBlock`], one per remote delta, and the [`PartnerPayload`] that carries a partner's blocks.

use std::mem::size_of;

use num_complex::Complex64;

use super::{ChunkMap, Payload};

/// Fixed-size prefix describing one [`ExchangeBlock`] on the wire.
///
/// `#[repr(C)]` and `Pod`: four `u32`s, 16 bytes, no padding, so it casts to bytes with no copy and reads back from an unaligned buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BlockHeader {
    /// Number of destination positions the block is indexed by — the group's agreed bucket count. `offsets` has `num_buckets + 1` entries.
    pub num_buckets: u32,
    /// Total rows in the block: `offsets[num_buckets]`, and the length of each column once the export pass has filled it.
    pub rows: u32,
    /// The width `W` the block was built at. Checked on decode: a payload encoded at one width is never silently reinterpreted at another.
    pub w: u32,
    /// Which of the layer plan's remote deltas this block carries — the receiver looks it up for the delta's bucket offset `bd[e]` in the [`ExchangeBlock::segment`] rule.
    pub entry: u32,
}

/// The rows one remote delta moves from this partition to one partner, in CSR order by the receiver's **destination position** ([`ChunkMap`]).
///
/// Columns are structure-of-arrays, matching the bucket storage they are gathered from and scattered into: `x`, `z` and `coeff` are parallel and their first [`rows`](Self::rows) entries are the block.
/// `offsets` is the CSR index, `num_buckets() + 1` entries, ascending, `offsets[0] == 0` and `offsets[num_buckets] == rows`.
///
/// **The columns are grow-only, so they may be longer than `rows`.** A block is reused across layers ([`set_counts`](Self::set_counts)) and the storage a wider layer needed is kept rather than freed and re-faulted, so the row count in the header is the one authority on how much of a column is live: read the block through [`cols`](Self::cols) or [`segment`](Self::segment), never through `x.len()`.
/// Whatever sits past `rows` is a previous layer's rows, and never travels.
///
/// The receiver never scans: for its output bucket `β′` it reads [`segment`](Self::segment)`(map.position_of(β′))` and merges those rows into that bucket — see the module docs for the ordering.
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

/// Two blocks are equal when their **live** contents are: the header, the CSR offsets, and the first `rows` entries of each column.
/// The grow-only tail past `rows` is a previous layer's scratch and is deliberately not compared — a block built through a reused payload must equal the same block built fresh.
impl<const W: usize> PartialEq for ExchangeBlock<W> {
    fn eq(&self, other: &Self) -> bool {
        self.header == other.header && self.offsets == other.offsets && self.cols() == other.cols()
    }
}

/// Everything one partner receives from this partition for one layer: the blocks in ascending remote-delta index ([`BlockHeader::entry`]).
///
/// A partner with nothing to receive is sent `None` rather than an empty payload (see the collective-order invariant in the module docs); an empty `blocks` is legal and encodes to zero parts.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PartnerPayload<const W: usize> {
    /// One block per remote delta, ascending by [`BlockHeader::entry`].
    pub blocks: Vec<ExchangeBlock<W>>,
}

impl<const W: usize> ExchangeBlock<W> {
    /// Build the CSR skeleton for `counts[p]` rows at each destination position `p` under remote-delta index `entry`, and size the columns.
    ///
    /// [`set_counts`](Self::set_counts) on a fresh block: the columns come back `rows` long, ready for the export pass to write by index.
    ///
    /// The engine always re-aims a pooled block instead, so this is the tests' constructor.
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

    /// Re-aim an existing block at `counts[p]` rows per destination position `p` under remote-delta index `entry`, **keeping every allocation**.
    ///
    /// The export pass then writes each row by index into the segment the offsets describe.
    /// The columns are only ever grown, never shrunk or re-zeroed (see the type docs): a steady-state layer re-aims a block it has already used at the same size, which touches nothing but the offsets.
    /// That is the whole point of holding the payloads across layers — the alternative is faulting in and zeroing the block's megabytes again every layer.
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

    /// The block's live rows: the first [`rows`](Self::rows) entries of the three columns.
    pub fn cols(&self) -> (&[[u64; W]], &[[u64; W]], &[Complex64]) {
        let rows = self.rows();
        (&self.x[..rows], &self.z[..rows], &self.coeff[..rows])
    }

    /// The rows destined for position `p`, as parallel `x` / `z` / `coeff` slices.
    ///
    /// The receiver's rule for its own output bucket `β′` is `segment(map.position_of(β′))` (module docs).
    /// A position with no rows yields three empty slices.
    ///
    /// # Panics
    ///
    /// If `p >= num_buckets()`, or if the block's columns are shorter than [`rows`](Self::rows) — which only a hand-built block can be.
    pub fn segment(&self, p: u32) -> (&[[u64; W]], &[[u64; W]], &[Complex64]) {
        let lo = self.offsets[p as usize] as usize;
        let hi = self.offsets[p as usize + 1] as usize;
        (&self.x[lo..hi], &self.z[lo..hi], &self.coeff[lo..hi])
    }

    /// The row index each of `map`'s chunk boundaries falls at.
    ///
    /// `chunks + 1` ascending entries starting at 0 and ending at [`rows`](Self::rows): chunk `k` carries rows `out[k]..out[k + 1]` of every column.
    pub fn chunk_rows(&self, map: &ChunkMap) -> Vec<usize> {
        chunk_rows_of(&self.offsets, map)
    }

    /// Rows the block carries: `offsets[num_buckets]`, and the length of each column once the export pass has filled it.
    pub fn rows(&self) -> usize {
        self.header.rows as usize
    }

    /// Destination positions the block is indexed by — the group's agreed bucket count, which both sides hold.
    ///
    /// The engine reads it only to check a received block against its own count, which is a `debug_assert` (the count is a collective decision, so a mismatch is a driver bug, not a data-dependent outcome).
    #[cfg(any(test, debug_assertions))]
    pub fn num_buckets(&self) -> u32 {
        self.header.num_buckets
    }

    /// Wire footprint in bytes: the header, the offsets, and the live rows of the three columns ([`Payload::byte_parts`] hands out exactly these bytes).
    pub fn bytes(&self) -> usize {
        size_of::<BlockHeader>()
            + self.offsets.len() * size_of::<u32>()
            + 2 * self.rows() * W * size_of::<u64>()
            + self.rows() * size_of::<Complex64>()
    }
}

/// Parts per block in the [`PartnerPayload`] encoding: header, offsets, x, z,
/// coeff.
pub(super) const PARTS_PER_BLOCK: usize = 5;

/// The row index each of `map`'s chunk boundaries falls at, `chunks + 1` ascending entries — the CSR offsets read at the chunks' destination positions.
pub(crate) fn chunk_rows_of(offsets: &[u32], map: &ChunkMap) -> Vec<usize> {
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

/// Panic unless a declared part length is a whole number of `stride`-byte elements — the one thing a receiver can check about a part before its bytes exist.
pub(super) fn check_stride(len: usize, stride: usize, what: &str) {
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

    /// Size the blocks from the declared part lengths — five parts per block, and each block's shape is readable from the lengths alone: `offsets` gives the bucket count, the `x` column the row count — then hand out the columns as bytes.
    ///
    /// The header travels into [`ExchangeBlock::header`] itself, so nothing about a block is known twice; [`finish_recv`](Payload::finish_recv) checks it against the shape sized here once it has arrived.
    fn recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
        assert_eq!(
            lens.len() % PARTS_PER_BLOCK,
            0,
            "partner payload: {} parts is not a whole number of {PARTS_PER_BLOCK}-part blocks",
            lens.len(),
        );
        let n = lens.len() / PARTS_PER_BLOCK;
        let key_stride = W * size_of::<u64>();
        // Grow-only, like the columns: a pooled payload keeps the blocks it held last layer and re-aims them.
        // Truncate first, grow second: the views handed out below borrow the blocks for as long as the caller holds them, so `self.blocks` cannot be touched again after that.
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
            // The row count the wire declared, so `cols` and `segment` see exactly what arrives; the header's own copy is checked against it in `finish_recv`.
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
            // The header travelled with the columns, so it could name more rows than the columns the declared part lengths sized.
            // It never does between ranks running the same build; the check is what keeps a garbled header an assertion rather than an out-of-bounds read.
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

    /// Parts 0 and 1 of every block — the header and the CSR offsets.
    /// Those are what `RecvRows::count` reads to size a gather run; the three columns after them are read only inside `append_into`.
    fn early_parts(&self) -> Vec<&[u8]> {
        let mut parts = Vec::with_capacity(2 * self.blocks.len());
        for block in &self.blocks {
            parts.push(bytemuck::bytes_of(&block.header));
            parts.push(bytemuck::cast_slice(&block.offsets));
        }
        parts
    }

    /// The three columns of each block, in block order, each cut at the chunks' destination-position boundaries.
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
        // `recv_into` sized every column and handed out all five views per block; keep the header and the offsets and drop the columns, which `bulk_recv_into` hands out once their shape is known.
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
