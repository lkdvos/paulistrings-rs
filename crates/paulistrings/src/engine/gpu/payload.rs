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

/// `PAULISTRINGS_GPU_EXCHANGE`'s value to a [`GpuExchange`] for an in-process group.
///
/// `nccl` means `Device` here, since in-process peer copies already go device to device; the MPI driver reads the raw value itself to tell `device` from `nccl`.
fn parse_gpu_exchange(raw: Option<&str>) -> GpuExchange {
    match raw {
        Some("host") => GpuExchange::Host,
        Some("nccl") => {
            log::info!(
                "gpu: PAULISTRINGS_GPU_EXCHANGE=nccl applies to MPI groups; an in-process group uses GpuExchange::Device"
            );
            GpuExchange::Device
        }
        _ => GpuExchange::Device,
    }
}

/// The default for a [`GpuPartitionedSum`](super::GpuPartitionedSum): `Device` unless `PAULISTRINGS_GPU_EXCHANGE=host`, read once per process.
pub(crate) fn gpu_exchange_default() -> GpuExchange {
    static MODE: std::sync::OnceLock<GpuExchange> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        parse_gpu_exchange(std::env::var("PAULISTRINGS_GPU_EXCHANGE").ok().as_deref())
    })
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

/// Whether a device-to-device copy between two devices goes direct or through the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PeerAccess {
    /// Both ends are the same device.
    SameDevice,
    /// The destination's context and the source's memory pool both map the source's memory into the destination, so copies go over NVLink or PCIe peer-to-peer.
    Enabled,
    /// The driver reports the pair cannot access each other; copies stage through the host.
    Unsupported,
    /// The driver allows the pair but enabling failed with this error; copies stage through the host.
    Failed(String),
}

/// Enable direct access from `dst`'s context to `src`'s memory, once per ordered pair, and report the outcome.
/// A pair without access is still correct: `cuMemcpyPeerAsync` stages through the host, so the outcome is logged rather than an error.
pub(crate) fn enable_peer_access(dst: &Arc<CudaContext>, src: &Arc<CudaContext>) -> PeerAccess {
    static DONE: Mutex<Vec<((usize, usize), PeerAccess)>> = Mutex::new(Vec::new());
    let pair = (dst.ordinal(), src.ordinal());
    if pair.0 == pair.1 {
        return PeerAccess::SameDevice;
    }
    let mut done = DONE.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, access)) = done.iter().find(|(p, _)| *p == pair) {
        return access.clone();
    }
    let access = try_enable_peer_access(dst, src, pair);
    match &access {
        PeerAccess::Enabled => log::info!("gpu: peer access {} -> {} enabled", pair.1, pair.0),
        other => log::warn!(
            "gpu: peer access {} -> {} is {other:?}; copies between them stage through the host",
            pair.1,
            pair.0
        ),
    }
    done.push((pair, access.clone()));
    access
}

fn try_enable_peer_access(
    dst: &Arc<CudaContext>,
    src: &Arc<CudaContext>,
    pair: (usize, usize),
) -> PeerAccess {
    let failed = |e: sys::CUresult| PeerAccess::Failed(format!("{e:?}"));
    let mut can = 0i32;
    // SAFETY: a driver query on two live ordinals.
    if let Err(e) =
        unsafe { sys::cuDeviceCanAccessPeer(&mut can, pair.0 as i32, pair.1 as i32) }.result()
    {
        return failed(e.0);
    }
    if can == 0 {
        return PeerAccess::Unsupported;
    }
    if let Err(e) = dst.bind_to_thread() {
        return failed(e.0);
    }
    // SAFETY: `dst` is the bound context and `src`'s handle stays valid while `src` is alive.
    match unsafe { sys::cuCtxEnablePeerAccess(src.cu_ctx(), 0) } {
        sys::CUresult::CUDA_SUCCESS | sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {}
        e => return failed(e),
    }
    // cudarc allocates through `cuMemAllocAsync`, and pool memory is mapped to a peer only by the pool's own access list: without this, peer copies stage through the host and peer loads fault.
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
    // SAFETY: a driver query on a live ordinal.
    if let Err(e) = unsafe { sys::cuDeviceGetMemPool(&mut pool, pair.1 as i32) }.result() {
        return failed(e.0);
    }
    let desc = sys::CUmemAccessDesc {
        location: sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: pair.0 as i32,
        },
        flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    // SAFETY: `pool` is `src`'s current pool and `desc` one valid entry.
    match unsafe { sys::cuMemPoolSetAccess(pool, &desc, 1) }.result() {
        Ok(()) => PeerAccess::Enabled,
        Err(e) => failed(e.0),
    }
}

/// Enable direct access from device `dst` to device `src`'s memory, as the device exchange does before its first copy, and report whether it took.
pub fn peer_access(dst: u32, src: u32) -> Result<PeerAccess, GpuError> {
    let dst = super::device::context(dst)?;
    let src = super::device::context(src)?;
    Ok(enable_peer_access(&dst, &src))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gpu_exchange_reads_the_knob() {
        assert_eq!(parse_gpu_exchange(None), GpuExchange::Device);
        assert_eq!(parse_gpu_exchange(Some("host")), GpuExchange::Host);
        assert_eq!(parse_gpu_exchange(Some("device")), GpuExchange::Device);
        assert_eq!(parse_gpu_exchange(Some("nccl")), GpuExchange::Device);
        assert_eq!(parse_gpu_exchange(Some("garbage")), GpuExchange::Device);
    }

    #[test]
    fn peer_access_is_reported_for_every_pair() {
        crate::require_cuda!();
        let n = super::super::device_count() as u32;
        for dst in 0..n {
            for src in 0..n {
                let access = peer_access(dst, src).expect("visible devices");
                if dst == src {
                    assert_eq!(access, PeerAccess::SameDevice);
                } else {
                    assert!(
                        !matches!(access, PeerAccess::Failed(_)),
                        "{dst} <- {src}: {access:?}"
                    );
                }
            }
        }
    }
}
