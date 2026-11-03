//! [`DevicePayload`], one partner's exchange blocks with their columns in device memory.

use crate::coefficient::Coefficient;
use num_complex::Complex64;
use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::columns::{grow, grow_keep};
use super::error::GpuError;
use crate::engine::partitioned::exchange_block::BlockHeader;

/// One exchange block with its columns on a device: the header and CSR offsets on the host, `x`/`z`/`coeff` on the device the sender ran on.
/// The columns are grow-only and `header.rows` says how much of them is live.
pub(crate) struct DeviceBlock<const W: usize, C: Coefficient = Complex64> {
    pub(crate) header: BlockHeader,
    pub(crate) offsets: Vec<u32>,
    pub(crate) x: CudaSlice<u64>,
    pub(crate) z: CudaSlice<u64>,
    pub(crate) coefficient: CudaSlice<u8>,
    /// The type `coefficient` holds, `size_of::<C>()` bytes a row.
    coefficient_type: PhantomData<C>,
}

impl<const W: usize, C: Coefficient> DeviceBlock<W, C> {
    /// Bytes of one row's columns: both key columns and the coefficient.
    const ROW_BYTES: usize = 2 * W * size_of::<u64>() + size_of::<C>();

    fn new(stream: &Arc<CudaStream>) -> Result<Self, GpuError> {
        Ok(Self {
            header: BlockHeader { num_buckets: 0, rows: 0, width: W as u32, entry: 0 },
            offsets: Vec::new(),
            x: stream.alloc_zeros(W)?,
            z: stream.alloc_zeros(W)?,
            coefficient: stream.alloc_zeros(size_of::<C>())?,
            coefficient_type: PhantomData,
        })
    }

    /// The block's header and offsets for `rows` rows over `buckets` positions; `offsets` must already hold the `buckets + 1` CSR offsets.
    pub(crate) fn set_header(&mut self, entry: u32, buckets: usize) {
        debug_assert_eq!(self.offsets.len(), buckets + 1);
        self.header = BlockHeader {
            num_buckets: buckets as u32,
            rows: self.offsets[buckets],
            width: W as u32,
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
        grow(stream, &mut self.coefficient, rows * size_of::<C>(), ordinal)
    }

    /// Room for `need` rows, keeping the first `live` rows of every column; grows geometrically.
    pub(crate) fn reserve_keep(
        &mut self,
        stream: &Arc<CudaStream>,
        live: usize,
        need: usize,
        ordinal: u32,
    ) -> Result<(), GpuError> {
        let capacity = self.coefficient.len() / size_of::<C>();
        if need <= capacity {
            return Ok(());
        }
        let rows = need.max(2 * capacity);
        let bytes = (rows * Self::ROW_BYTES) as u64;
        grow_keep(stream, &mut self.x, rows * W, live * W, ordinal, bytes)?;
        grow_keep(stream, &mut self.z, rows * W, live * W, ordinal, bytes)?;
        let c = size_of::<C>();
        grow_keep(stream, &mut self.coefficient, rows * c, live * c, ordinal, bytes)
    }

    pub(crate) fn rows(&self) -> usize {
        self.header.rows as usize
    }

    /// Bytes the block occupies on a wire: header, offsets and the three live columns.
    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<BlockHeader>()
            + self.offsets.len() * std::mem::size_of::<u32>()
            + self.rows() * Self::ROW_BYTES
    }
}

/// One partner's exchange blocks in ascending remote-delta index, pooled by the partition that fills them.
#[derive(Default)]
pub(crate) struct DevicePayload<const W: usize, C: Coefficient = Complex64> {
    pub(crate) blocks: Vec<DeviceBlock<W, C>>,
}

impl<const W: usize, C: Coefficient> DevicePayload<W, C> {
    /// Block `index`, allocating on `stream` up to it.
    pub(crate) fn block_mut(
        &mut self,
        index: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<&mut DeviceBlock<W, C>, GpuError> {
        while self.blocks.len() <= index {
            self.blocks.push(DeviceBlock::new(stream)?);
        }
        Ok(&mut self.blocks[index])
    }
}
