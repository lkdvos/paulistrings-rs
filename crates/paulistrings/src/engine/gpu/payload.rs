//! [`DevicePayload`], exchange blocks whose columns stay in device memory, for a transport that moves objects. See ARCHITECTURE.md §Partitioning.

use std::any::Any;
use std::sync::{Arc, Mutex, PoisonError};

use cudarc::driver::{sys, CudaContext, CudaSlice, CudaStream};

use super::error::GpuError;
use super::layer::grow;
use crate::engine::partitioned::transport::{BlockHeader, Payload};

/// How a device group's exchange blocks travel: through the host `PartnerPayload` columns, or as device payloads whose columns never leave device memory.
///
/// `Device` needs a transport that moves objects (the in-process one) and a group of device partitions only; `PAULISTRINGS_GPU_EXCHANGE=host|device` sets the default a [`GpuPartitionedSum`](super::GpuPartitionedSum) starts with.
/// An MPI group and a group with a host member always use `Host`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuExchange {
    /// K10 stages every block through the host `PartnerPayload` and the receiver uploads it.
    #[default]
    Host,
    /// K10 fills device columns the receiver copies device-to-device, fingerprints included.
    Device,
}

/// The default for a [`GpuPartitionedSum`](super::GpuPartitionedSum): `Device` unless `PAULISTRINGS_GPU_EXCHANGE=host`, read once per process.
pub(crate) fn gpu_exchange_default() -> GpuExchange {
    static MODE: std::sync::OnceLock<GpuExchange> = std::sync::OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("PAULISTRINGS_GPU_EXCHANGE").as_deref() {
            Ok("host") => GpuExchange::Host,
            _ => GpuExchange::Device,
        },
    )
}

/// One exchange block with its columns on a device: the header and CSR offsets on the host, `x`/`z`/`coeff`/`g` on the device the sender ran on.
/// The columns are grow-only and `header.rows` says how much of them is live.
pub(crate) struct DeviceBlock<const W: usize> {
    pub(crate) header: BlockHeader,
    pub(crate) offsets: Vec<u32>,
    pub(crate) x: CudaSlice<u64>,
    pub(crate) z: CudaSlice<u64>,
    pub(crate) c: CudaSlice<f64>,
    pub(crate) g: CudaSlice<u64>,
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
            g: stream.alloc_zeros(1)?,
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

    /// An empty block over `b` positions.
    pub(crate) fn set_empty(&mut self, entry: u32, b: usize) {
        self.offsets.clear();
        self.offsets.resize(b + 1, 0);
        self.set_header(entry, b);
    }

    /// Room for `rows` rows in every column, keeping what is there.
    pub(crate) fn grow(
        &mut self,
        stream: &Arc<CudaStream>,
        rows: usize,
        ordinal: u32,
    ) -> Result<(), GpuError> {
        grow(stream, &mut self.x, rows * W, ordinal)?;
        grow(stream, &mut self.z, rows * W, ordinal)?;
        grow(stream, &mut self.c, 2 * rows, ordinal)?;
        grow(stream, &mut self.g, rows, ordinal)
    }

    pub(crate) fn rows(&self) -> usize {
        self.header.rows as usize
    }

    /// Bytes the block would occupy on a wire: header, offsets and the four live columns.
    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<BlockHeader>()
            + self.offsets.len() * std::mem::size_of::<u32>()
            + self.rows() * (2 * W + 3) * std::mem::size_of::<u64>()
    }
}

/// One partner's exchange blocks in ascending remote-delta index, columns resident on device `device`.
///
/// It implements [`Payload`] only to satisfy the transport bound: it has no byte form, so [`Payload::byte_parts`] and [`Payload::recv_into`] panic.
/// Only a transport that moves the typed value (`InProcessTransport`) may carry it, and `GpuPartitionedSum` is the one driver that sends it.
/// A group with a host member or an MPI transport uses `PartnerPayload`.
pub(crate) struct DevicePayload<const W: usize> {
    /// The ordinal every block's columns live on; `None` only for the `Default` shell.
    pub(crate) device: Option<u32>,
    pub(crate) blocks: Vec<DeviceBlock<W>>,
}

impl<const W: usize> Default for DevicePayload<W> {
    fn default() -> Self {
        Self {
            device: None,
            blocks: Vec::new(),
        }
    }
}

impl<const W: usize> DevicePayload<W> {
    /// Block `j`, allocating on `stream` up to it.
    pub(crate) fn block_mut(
        &mut self,
        j: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<&mut DeviceBlock<W>, GpuError> {
        let ordinal = stream.context().ordinal() as u32;
        debug_assert_eq!(self.device.unwrap_or(ordinal), ordinal);
        self.device = Some(ordinal);
        while self.blocks.len() <= j {
            self.blocks.push(DeviceBlock::new(stream)?);
        }
        Ok(&mut self.blocks[j])
    }
}

const NO_BYTE_FORM: &str = "DevicePayload has no byte form: it travels only through a transport that moves objects (InProcessTransport); an MPI group or a mixed host+device group must use GpuExchange::Host";

impl<const W: usize> Payload for DevicePayload<W> {
    fn byte_parts(&self) -> Vec<&[u8]> {
        panic!("{NO_BYTE_FORM}");
    }

    fn recv_into(&mut self, _lens: &[usize]) -> Vec<&mut [u8]> {
        panic!("{NO_BYTE_FORM}");
    }

    fn finish_recv(&mut self) {}
}

/// Payloads not in flight, per device, shared by every partition in the process.
/// A receiver adopts a payload by copying it and returns it here, so a sender on the payload's device reuses the columns next layer whichever partition received them.
static BIN: Mutex<Vec<(u32, Box<dyn Any + Send>)>> = Mutex::new(Vec::new());

/// A pooled payload for `device`, or a fresh empty one.
pub(crate) fn reclaim<const W: usize>(device: u32) -> DevicePayload<W> {
    let mut bin = BIN.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(i) = bin
        .iter()
        .position(|(d, p)| *d == device && p.is::<DevicePayload<W>>())
    {
        let (_, p) = bin.swap_remove(i);
        return *p.downcast().expect("checked by position");
    }
    DevicePayload {
        device: Some(device),
        blocks: Vec::new(),
    }
}

/// Return a payload to the pool once every copy out of it has completed.
pub(crate) fn recycle<const W: usize>(payload: DevicePayload<W>) {
    if let Some(device) = payload.device {
        BIN.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((device, Box::new(payload)));
    }
}

/// Free every pooled payload.
pub(crate) fn drain_bin() {
    BIN.lock().unwrap_or_else(PoisonError::into_inner).clear();
}

/// Enable direct access from `dst`'s context to `src`'s memory, once per ordered pair, when the devices allow it.
/// A pair that does not is still correct: `cuMemcpyPeerAsync` stages through the host, so nothing here is an error.
pub(crate) fn enable_peer_access(dst: &Arc<CudaContext>, src: &Arc<CudaContext>) {
    static DONE: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let pair = (dst.ordinal(), src.ordinal());
    if pair.0 == pair.1 {
        return;
    }
    let mut done = DONE.lock().unwrap_or_else(PoisonError::into_inner);
    if done.contains(&pair) {
        return;
    }
    done.push(pair);
    let mut can = 0i32;
    // SAFETY: driver queries on live ordinals; the raw handle of `src` is read while it is the bound context and used only while both contexts are alive.
    unsafe {
        if sys::cuDeviceCanAccessPeer(&mut can, pair.0 as i32, pair.1 as i32)
            .result()
            .is_err()
            || can == 0
            || src.bind_to_thread().is_err()
        {
            return;
        }
        let mut peer: sys::CUcontext = std::ptr::null_mut();
        if sys::cuCtxGetCurrent(&mut peer).result().is_err() || dst.bind_to_thread().is_err() {
            return;
        }
        let _ = sys::cuCtxEnablePeerAccess(peer, 0);
    }
}
