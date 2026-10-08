//! [`DevicePayload`], one partner's exchange blocks with their columns in device memory. See ARCHITECTURE.md §Partitioning.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::columns::{grow, grow_keep};
use super::error::GpuError;
use crate::engine::partitioned::transport::BlockHeader;

/// One exchange block with its columns on a device: the header and CSR offsets on the host, `x`/`z`/`coeff` on the device the sender ran on.
/// The columns are grow-only and `header.rows` says how much of them is live.
pub(crate) struct DeviceBlock<const W: usize> {
    pub(crate) header: BlockHeader,
    pub(crate) offsets: Vec<u32>,
    pub(crate) x: CudaSlice<u64>,
    pub(crate) z: CudaSlice<u64>,
    pub(crate) c: CudaSlice<f64>,
}

impl<const W: usize> DeviceBlock<W> {
    fn new(stream: &Arc<CudaStream>) -> Result<Self, GpuError> {
        Ok(Self {
            header: BlockHeader {
                num_buckets: 0,
                rows: 0,
                w: W as u32,
                entry: 0,
            },
            offsets: Vec::new(),
            x: stream.alloc_zeros(W)?,
            z: stream.alloc_zeros(W)?,
            c: stream.alloc_zeros(2)?,
        })
    }

    /// The block's header and offsets for `rows` rows over `b` positions; `offsets` must already hold the `b + 1` CSR offsets.
    pub(crate) fn set_header(&mut self, entry: u32, b: usize) {
        debug_assert_eq!(self.offsets.len(), b + 1);
        self.header = BlockHeader {
            num_buckets: b as u32,
            rows: self.offsets[b],
            w: W as u32,
            entry,
        };
    }

    /// Room for `rows` rows in every column, discarding what is there.
    pub(crate) fn grow(
        &mut self,
        stream: &Arc<CudaStream>,
        rows: usize,
        ordinal: u32,
    ) -> Result<(), GpuError> {
        grow(stream, &mut self.x, rows * W, ordinal)?;
        grow(stream, &mut self.z, rows * W, ordinal)?;
        grow(stream, &mut self.c, 2 * rows, ordinal)
    }

    /// Room for `need` rows, keeping the first `live` rows of every column; grows geometrically.
    pub(crate) fn reserve_keep(
        &mut self,
        stream: &Arc<CudaStream>,
        live: usize,
        need: usize,
        ordinal: u32,
    ) -> Result<(), GpuError> {
        let capacity = self.c.len() / 2;
        if need <= capacity {
            return Ok(());
        }
        let rows = need.max(2 * capacity);
        let bytes = (rows * (2 * W + 2) * 8) as u64;
        grow_keep(stream, &mut self.x, rows * W, live * W, ordinal, bytes)?;
        grow_keep(stream, &mut self.z, rows * W, live * W, ordinal, bytes)?;
        grow_keep(stream, &mut self.c, 2 * rows, 2 * live, ordinal, bytes)
    }

    pub(crate) fn rows(&self) -> usize {
        self.header.rows as usize
    }

    /// Bytes the block occupies on a wire: header, offsets and the three live columns.
    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<BlockHeader>()
            + self.offsets.len() * std::mem::size_of::<u32>()
            + self.rows() * (2 * W + 2) * std::mem::size_of::<u64>()
    }
}

/// One partner's exchange blocks in ascending remote-delta index, pooled by the partition that fills them.
#[derive(Default)]
pub(crate) struct DevicePayload<const W: usize> {
    pub(crate) blocks: Vec<DeviceBlock<W>>,
}

impl<const W: usize> DevicePayload<W> {
    /// Block `index`, allocating on `stream` up to it.
    pub(crate) fn block_mut(
        &mut self,
        index: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<&mut DeviceBlock<W>, GpuError> {
        while self.blocks.len() <= index {
            self.blocks.push(DeviceBlock::new(stream)?);
        }
        Ok(&mut self.blocks[index])
    }
}
