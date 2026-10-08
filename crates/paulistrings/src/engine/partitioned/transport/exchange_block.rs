//! The exchange wire unit: [`ExchangeBlock`], one per remote delta, and the [`PartnerPayload`] that carries a partner's blocks.

use std::mem::size_of;

use num_complex::Complex64;

use super::{ChunkMap, Payload};

/// Fixed-size wire prefix of one [`ExchangeBlock`]: four `u32`s, no padding, so it casts to bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BlockHeader {
    /// Destination positions, the group's agreed bucket count; `offsets` has one more entry.
    pub num_buckets: u32,
    /// Live rows, `offsets[num_buckets]`.
    pub rows: u32,
    /// The width `W` the block was built at, checked on receive.
    pub w: u32,
    /// The layer plan's remote-delta index this block carries.
    pub entry: u32,
}

/// The rows one remote delta moves to one partner, CSR-indexed by the receiver's destination position ([`ChunkMap`]).
///
/// The columns are grow-only and may be longer than `header.rows`, which alone says how much is live: read through `cols` or `segment`, never `x.len()`.
#[derive(Clone, Debug, Default)]
pub(crate) struct ExchangeBlock<const W: usize> {
    pub header: BlockHeader,
    /// CSR offsets by destination position, `num_buckets + 1` entries.
    pub offsets: Vec<u32>,
    pub x: Vec<[u64; W]>,
    pub z: Vec<[u64; W]>,
    pub coeff: Vec<Complex64>,
}

/// Compares the live rows only, so a reused block equals the same block built fresh.
impl<const W: usize> PartialEq for ExchangeBlock<W> {
    fn eq(&self, other: &Self) -> bool {
        self.header == other.header && self.offsets == other.offsets && self.cols() == other.cols()
    }
}

/// Everything one partner receives from this partition for one layer: one block per remote delta, ascending by [`BlockHeader::entry`].
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PartnerPayload<const W: usize> {
    pub blocks: Vec<ExchangeBlock<W>>,
}

impl<const W: usize> ExchangeBlock<W> {
    /// `set_counts` on a fresh block.
    #[cfg(test)]
    pub fn with_counts(entry: u32, counts: &[u32]) -> Self {
        let mut block = Self::default();
        block.set_counts(entry, counts);
        block
    }

    /// Re-aim the block at `counts[p]` rows per destination position under remote-delta index `entry`, growing but never shrinking or re-zeroing the columns.
    pub(crate) fn set_counts(&mut self, entry: u32, counts: &[u32]) {
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

    fn grow_columns(&mut self, rows: usize) {
        if self.coeff.len() < rows {
            self.x.resize(rows, [0u64; W]);
            self.z.resize(rows, [0u64; W]);
            self.coeff.resize(rows, Complex64::new(0.0, 0.0));
        }
    }

    /// The live rows of the three columns.
    pub(crate) fn cols(&self) -> (&[[u64; W]], &[[u64; W]], &[Complex64]) {
        let rows = self.rows();
        (&self.x[..rows], &self.z[..rows], &self.coeff[..rows])
    }

    /// The rows destined for position `p`; output bucket `β′` reads `segment(map.position_of(β′))`.
    pub(crate) fn segment(&self, p: u32) -> (&[[u64; W]], &[[u64; W]], &[Complex64]) {
        let lo = self.offsets[p as usize] as usize;
        let hi = self.offsets[p as usize + 1] as usize;
        (&self.x[lo..hi], &self.z[lo..hi], &self.coeff[lo..hi])
    }

    /// The row index at each of `map`'s chunk boundaries, `chunks + 1` entries.
    pub(crate) fn chunk_rows(&self, map: &ChunkMap) -> Vec<usize> {
        chunk_rows_of(&self.offsets, map)
    }

    /// Live rows.
    pub(crate) fn rows(&self) -> usize {
        self.header.rows as usize
    }

    /// Destination positions the block is indexed by.
    #[cfg(any(test, debug_assertions))]
    pub(crate) fn num_buckets(&self) -> u32 {
        self.header.num_buckets
    }

    /// Wire footprint in bytes, exactly what `byte_parts` hands out.
    pub(crate) fn bytes(&self) -> usize {
        size_of::<BlockHeader>()
            + self.offsets.len() * size_of::<u32>()
            + 2 * self.rows() * W * size_of::<u64>()
            + self.rows() * size_of::<Complex64>()
    }
}

/// Parts per block on the wire: header, offsets, x, z, coeff.
pub(super) const PARTS_PER_BLOCK: usize = 5;

/// The CSR `offsets` read at `map`'s chunk boundaries, `chunks + 1` entries.
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

/// Panic unless a declared part length is a whole number of `stride`-byte elements.
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

    fn recv_into(&mut self, lens: &[usize]) -> Vec<&mut [u8]> {
        assert_eq!(
            lens.len() % PARTS_PER_BLOCK,
            0,
            "partner payload: {} parts is not a whole number of {PARTS_PER_BLOCK}-part blocks",
            lens.len(),
        );
        let n = lens.len() / PARTS_PER_BLOCK;
        let key_stride = W * size_of::<u64>();
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
            // Overwritten by the arriving header, which `finish_recv` checks against the columns.
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
            // A garbled header must fail here rather than as an out-of-bounds read.
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

    /// The header and the CSR offsets of every block.
    fn early_parts(&self) -> Vec<&[u8]> {
        let mut parts = Vec::with_capacity(2 * self.blocks.len());
        for block in &self.blocks {
            parts.push(bytemuck::bytes_of(&block.header));
            parts.push(bytemuck::cast_slice(&block.offsets));
        }
        parts
    }

    /// The three columns of each block, each cut at the chunk boundaries.
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
