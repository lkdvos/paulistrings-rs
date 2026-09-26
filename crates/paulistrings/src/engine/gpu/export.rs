//! K10 export, the exchange, and the received rows' adoption for one device layer, through host or device payloads. See ARCHITECTURE.md §Partitioning.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};
use num_complex::Complex64;

use super::columns::DeviceColumns;
use super::error::GpuError;
use super::layer::{
    arena_batches, fused_variant, grow, launch_fused, thread_per, warp_per_bucket, xfer, FusedOut,
    FusedRecv, FusedTable, LayerScratch, Xfer, XferNs,
};
use super::module::MAX_BUCKET_LEN;
use super::payload::{self, DeviceBlock, DevicePayload, GpuExchange};
use super::prepared::DevicePrepared;
use super::scan::exclusive_scan_with_max_into;
use super::staging::HostStaging;
use super::sum::GpuSum;
use super::truncation::KeepProgram;
use crate::bucket::hash::PartitionRows;
use crate::engine::coset::Gf2Span;
use crate::engine::partitioned::layer::{exchange_chunks, LayerExchangeCounts};
use crate::engine::partitioned::plan::PartitionPlan;
use crate::engine::partitioned::transport::{
    BlockHeader, ChunkMap, ExchangeBlock, PartnerPayload, Transport,
};

#[cfg(feature = "nccl")]
use super::nccl::{
    nccl_schedule, BlockSkeletons, DeviceWire, Skeleton, WireColumn, WireGroup, WireOpKind,
};

/// How an exported block's columns come back to the host: a direct copy into the pooled `Vec`s, or through a pinned pool and one `memcpy`.
/// `PAULISTRINGS_GPU_STAGING=pageable|pinned` overrides the default, read once per process (`research/FINDINGS.md` for the measurement).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExportStaging {
    Pageable,
    Pinned,
}

pub(crate) fn export_staging() -> ExportStaging {
    static MODE: std::sync::OnceLock<ExportStaging> = std::sync::OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("PAULISTRINGS_GPU_STAGING").as_deref() {
            Ok("pageable") => ExportStaging::Pageable,
            Ok("pinned") => ExportStaging::Pinned,
            _ => ExportStaging::Pinned,
        },
    )
}

/// Grow-only device and host buffers of the export and receive passes, kept between layers.
pub(crate) struct DeviceExport<const W: usize> {
    counts: CudaSlice<u32>,
    off: CudaSlice<u32>,
    x: CudaSlice<u64>,
    z: CudaSlice<u64>,
    c: CudaSlice<f64>,
    pinned: HostStaging,
    /// Host payloads not in flight, the layer's own pool (`Transport::exchange_layer`); device payloads pool in `payload::BIN`.
    pub(crate) pool: Vec<PartnerPayload<W>>,
    /// The destination-position order and chunk edges of the current layer.
    pub(crate) chunks: ChunkMap,
    /// Received blocks, concatenated: `off` is `K × (B + 1)` CSR offsets, `base[k]` the first row of block `k`.
    pub(crate) recv_off: CudaSlice<u32>,
    pub(crate) recv_base: CudaSlice<u32>,
    pub(crate) recv_x: CudaSlice<u64>,
    pub(crate) recv_z: CudaSlice<u64>,
    pub(crate) recv_c: CudaSlice<f64>,
    pub(crate) recv_g: CudaSlice<u64>,
    off_host: Vec<u32>,
    base_host: Vec<u32>,
    /// The longest received segment of the current layer; must fit the tag's offset field.
    pub(crate) recv_max_segment: usize,
    pub(crate) recv_rows: usize,
    /// Which payload form the exchange uses; `Host` unless a `GpuPartitionedSum` chose otherwise.
    pub(crate) mode: GpuExchange,
    premerge: PremergeScratch,
    stream: Arc<CudaStream>,
    /// The device wire of [`GpuExchange::Nccl`], `Some` whenever that is the mode.
    #[cfg(feature = "nccl")]
    pub(crate) wire: Option<Arc<dyn DeviceWire>>,
    /// Send payloads of a group that failed after the vote: a peer may still be reading them, so they never return to the shared bin and are freed with the split, after `wire`.
    #[cfg(feature = "nccl")]
    quarantine: Vec<DevicePayload<W>>,
    /// Skeleton payloads not in flight.
    #[cfg(feature = "nccl")]
    skeletons: Vec<BlockSkeletons<W>>,
    /// Test hook: the next NCCL layer's receive growth fails as out of memory.
    #[cfg(all(feature = "nccl", any(test, feature = "test-utils")))]
    pub(crate) fail_recv_growth: bool,
}

/// Grow-only buffers of the sender-side merge: one partner's sub-table, its counts and CSR, and the split of its merged rows over the partner's blocks.
struct PremergeScratch {
    amp: CudaSlice<f64>,
    mask: CudaSlice<u64>,
    nz: CudaSlice<u32>,
    bd: CudaSlice<u32>,
    gm: CudaSlice<u64>,
    rem: CudaSlice<u32>,
    sel: CudaSlice<u32>,
    cnt: CudaSlice<u32>,
    rows: CudaSlice<u32>,
    start: CudaSlice<u32>,
    out_len_pos: CudaSlice<u32>,
    lens: CudaSlice<u32>,
    loff: CudaSlice<u32>,
    fallback: CudaSlice<u32>,
    tot: CudaSlice<u32>,
    /// Stands in for the fingerprint column a host-bound copy does not write.
    no_g: CudaSlice<u64>,
    start_host: Vec<u32>,
    loff_host: Vec<u32>,
}

impl PremergeScratch {
    fn new(s: &Arc<CudaStream>, w: usize) -> Result<Self, GpuError> {
        Ok(Self {
            amp: s.alloc_zeros(512)?,
            mask: s.alloc_zeros(32 * w)?,
            nz: s.alloc_zeros(16)?,
            bd: s.alloc_zeros(16)?,
            gm: s.alloc_zeros(16)?,
            rem: s.alloc_zeros(16)?,
            sel: s.alloc_zeros(16)?,
            cnt: s.alloc_zeros(1)?,
            rows: s.alloc_zeros(1)?,
            start: s.alloc_zeros(2)?,
            out_len_pos: s.alloc_zeros(1)?,
            lens: s.alloc_zeros(1)?,
            loff: s.alloc_zeros(2)?,
            fallback: s.alloc_zeros(2)?,
            tot: s.alloc_zeros(2)?,
            no_g: s.alloc_zeros(1)?,
            start_host: Vec::new(),
            loff_host: Vec::new(),
        })
    }
}

impl<const W: usize> DeviceExport<W> {
    pub(crate) fn new(sum: &GpuSum<W>) -> Result<Self, GpuError> {
        let s = &sum.stream;
        Ok(Self {
            counts: s.alloc_zeros(1)?,
            off: s.alloc_zeros(2)?,
            x: s.alloc_zeros(W)?,
            z: s.alloc_zeros(W)?,
            c: s.alloc_zeros(2)?,
            pinned: HostStaging::new(&sum.ctx),
            pool: Vec::new(),
            chunks: ChunkMap::default(),
            recv_off: s.alloc_zeros(2)?,
            recv_base: s.alloc_zeros(16)?,
            recv_x: s.alloc_zeros(W)?,
            recv_z: s.alloc_zeros(W)?,
            recv_c: s.alloc_zeros(2)?,
            recv_g: s.alloc_zeros(1)?,
            off_host: Vec::new(),
            base_host: Vec::new(),
            recv_max_segment: 0,
            recv_rows: 0,
            mode: GpuExchange::Host,
            premerge: PremergeScratch::new(s, W)?,
            stream: s.clone(),
            #[cfg(feature = "nccl")]
            wire: None,
            #[cfg(feature = "nccl")]
            quarantine: Vec::new(),
            #[cfg(feature = "nccl")]
            skeletons: Vec::new(),
            #[cfg(all(feature = "nccl", any(test, feature = "test-utils")))]
            fail_recv_growth: false,
        })
    }
}

/// The exchange call a partition owes its partners when its own layer failed before exporting: one well-formed empty block per remote delta the plan names, under the real chunk map, so a host or device receiver finishes the layer and the error surfaces after the loop.
pub(crate) fn pair_empty_exchange<const W: usize, X: Transport>(
    transport: &X,
    plan: &PartitionPlan,
    bits: u8,
    export: &mut DeviceExport<W>,
) {
    let size = transport.size();
    let b = 1usize << bits;
    export.chunks.rebuild(
        &Gf2Span::new(&plan.local_bucket_deltas, bits),
        b,
        exchange_chunks(),
    );
    let mut used = vec![0usize; size as usize];
    match export.mode {
        GpuExchange::Host => {
            let zero = vec![0u32; b];
            let mut send: Vec<Option<PartnerPayload<W>>> = (0..size).map(|_| None).collect();
            for r in &plan.remote {
                let q = r.partner as usize;
                let payload = send[q].get_or_insert_with(|| export.pool.pop().unwrap_or_default());
                let j = used[q];
                used[q] += 1;
                if payload.blocks.len() <= j {
                    payload
                        .blocks
                        .resize_with(j + 1, ExchangeBlock::<W>::default);
                }
                payload.blocks[j].set_counts(r.entry as u32, &zero);
            }
            for (q, payload) in send.iter_mut().enumerate() {
                if let Some(payload) = payload {
                    payload.blocks.truncate(used[q]);
                }
            }
            let (recv, ()) =
                transport.exchange_layer(send, &mut export.pool, &export.chunks, |_, _| ());
            export.pool.extend(recv.into_iter().flatten());
        }
        GpuExchange::Device => {
            let device = export.stream.context().ordinal() as u32;
            let mut send: Vec<Option<DevicePayload<W>>> = (0..size).map(|_| None).collect();
            for r in &plan.remote {
                let q = r.partner as usize;
                let payload = send[q].get_or_insert_with(|| payload::reclaim::<W>(device));
                let j = used[q];
                used[q] += 1;
                // An empty block holds 32 bytes of device memory; a partition that cannot allocate those has no way to pair its partners and lets them fail by name.
                payload
                    .block_mut(j, &export.stream)
                    .expect("allocating an empty device exchange block")
                    .set_empty(r.entry as u32, b);
            }
            for (q, payload) in send.iter_mut().enumerate() {
                if let Some(payload) = payload {
                    payload.blocks.truncate(used[q]);
                }
            }
            let (recv, ()) =
                transport.exchange_layer(send, &mut Vec::new(), &export.chunks, |_, _| ());
            recv.into_iter().flatten().for_each(payload::recycle);
        }
        #[cfg(feature = "nccl")]
        GpuExchange::Nccl => {
            let mut send: Vec<Option<BlockSkeletons<W>>> = (0..size).map(|_| None).collect();
            for r in &plan.remote {
                let skel = send[r.partner as usize].get_or_insert_with(|| {
                    let mut s = export.skeletons.pop().unwrap_or_default();
                    s.blocks.clear();
                    s
                });
                skel.push_empty(r.entry as u32, b);
            }
            let recv = transport.exchange(send, &mut export.skeletons);
            export.skeletons.extend(recv.into_iter().flatten());
            vote(transport, false);
        }
    }
}

/// The NCCL exchange's go/no-go: `true` when every rank of the group is `ready`. **Collective**: one `allreduce_sum_u64` of `size` words, rank `r`'s slot set when it is not ready.
#[cfg(feature = "nccl")]
fn vote<X: Transport>(transport: &X, ready: bool) -> bool {
    let mut votes = vec![0u64; transport.size() as usize];
    votes[transport.rank() as usize] = u64::from(!ready);
    transport.allreduce_sum_u64(&mut votes);
    votes.iter().all(|&v| v == 0)
}

/// Export → exchange → adoption of the received rows with their fingerprints, by the export's [`GpuExchange`] mode.
/// Runs after K1 has filled `cnt` at the agreed bucket count; the layer's K2/K3 then read `scratch.export.recv_*`.
pub(crate) fn exchange_rows<const W: usize, X: Transport>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    plan: &PartitionPlan,
    rows: &PartitionRows<W>,
    scratch: &mut LayerScratch<W>,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    match scratch.export.mode {
        GpuExchange::Host => exchange_rows_host(sum, table, plan, rows, scratch, transport),
        GpuExchange::Device => exchange_rows_device(sum, table, plan, scratch, transport),
        #[cfg(feature = "nccl")]
        GpuExchange::Nccl => exchange_rows_nccl(sum, table, plan, scratch, transport),
    }
}

/// The host-payload exchange: K10 stages through the host, the receiver uploads and fingerprints on device.
fn exchange_rows_host<const W: usize, X: Transport>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    plan: &PartitionPlan,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] rows: &PartitionRows<W>,
    scratch: &mut LayerScratch<W>,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    let size = transport.size();
    #[cfg(feature = "phase-timing")]
    let t_export = std::time::Instant::now();
    let (send, mut counts) = match export_blocks(sum, table, plan, scratch, size) {
        Ok(v) => v,
        Err(e) => {
            pair_empty_exchange::<W, X>(transport, plan, sum.hash.bits(), &mut scratch.export);
            return Err(e);
        }
    };
    #[cfg(feature = "phase-timing")]
    {
        scratch.laps.export_ns += t_export.elapsed().as_nanos() as u64;
        scratch.laps.rows_exported += counts.rows_sent.iter().sum::<u64>();
    }
    #[cfg(debug_assertions)]
    crate::engine::partitioned::export::debug_assert_exported_partitions(&send, rows);

    let b = sum.hash.num_buckets();
    let span = Gf2Span::new(&plan.local_bucket_deltas, sum.hash.bits());
    let export = &mut scratch.export;
    export.chunks.rebuild(&span, b, exchange_chunks());
    #[cfg(feature = "phase-timing")]
    let t_exchange = std::time::Instant::now();
    #[cfg(feature = "phase-timing")]
    let body_ns = std::cell::Cell::new(0u64);
    let DeviceExport {
        pool,
        chunks: map,
        recv_off,
        recv_base,
        recv_x,
        recv_z,
        recv_c,
        recv_g,
        off_host,
        base_host,
        recv_max_segment,
        recv_rows,
        ..
    } = export;
    let xfer_ns = &mut scratch.xfer_ns;
    #[cfg(feature = "phase-timing")]
    let laps = &mut scratch.laps;
    let (recv, body) = transport.exchange_layer(send, pool, map, |recv, wait| {
        #[cfg(feature = "phase-timing")]
        let t_body = std::time::Instant::now();
        // The whole transfer is waited out before the one upload, so the received columns are complete when K0 runs.
        for k in 0..map.chunks() {
            wait.wait_chunk(k);
        }
        #[cfg(feature = "phase-timing")]
        {
            laps.chunk_wait_ns += t_body.elapsed().as_nanos() as u64;
        }
        let blocks = paired_blocks(plan, recv, |p: &PartnerPayload<W>, j| p.blocks.get(j));
        let mut total = 0usize;
        let mut max_seg = 0usize;
        off_host.clear();
        base_host.clear();
        base_host.resize(16, 0);
        for (k, block) in blocks.iter().enumerate() {
            debug_assert_eq!(block.header.entry, plan.remote[k].entry as u32);
            assert_eq!(
                block.header.num_buckets as usize, b,
                "a partner sent a block indexed by {} buckets where this partition has {b}",
                block.header.num_buckets
            );
            base_host[k] = total as u32;
            off_host.extend_from_slice(&block.offsets[..b + 1]);
            max_seg = block.offsets[..b + 1]
                .windows(2)
                .map(|w| (w[1] - w[0]) as usize)
                .max()
                .unwrap_or(0)
                .max(max_seg);
            total += block.rows();
        }
        *recv_max_segment = max_seg;
        *recv_rows = total;
        let s = &sum.stream;
        let o = sum.device();
        grow(s, recv_off, off_host.len().max(2), o)?;
        grow(s, recv_x, total.max(1) * W, o)?;
        grow(s, recv_z, total.max(1) * W, o)?;
        grow(s, recv_c, 2 * total.max(1), o)?;
        grow(s, recv_g, total.max(1), o)?;
        #[cfg(feature = "phase-timing")]
        s.synchronize()?;
        let r: Result<(), GpuError> = xfer(xfer_ns, Xfer::H2d, || {
            s.memcpy_htod(&off_host[..], &mut recv_off.slice_mut(0..off_host.len()))?;
            s.memcpy_htod(&base_host[..], recv_base)?;
            for (k, block) in blocks.iter().enumerate() {
                let n = block.rows();
                if n == 0 {
                    continue;
                }
                let base = base_host[k] as usize;
                let (bx, bz, bc) = block.cols();
                s.memcpy_htod(
                    bx.as_flattened(),
                    &mut recv_x.slice_mut(base * W..(base + n) * W),
                )?;
                s.memcpy_htod(
                    bz.as_flattened(),
                    &mut recv_z.slice_mut(base * W..(base + n) * W),
                )?;
                s.memcpy_htod(
                    bytemuck::cast_slice::<Complex64, f64>(bc),
                    &mut recv_c.slice_mut(2 * base..2 * (base + n)),
                )?;
            }
            s.synchronize()?;
            Ok(())
        });
        r?;
        if total > 0 {
            let n32 = total as u32;
            // SAFETY: arguments match `k_fingerprint` in fingerprint.cu; `recv_g` holds `total` entries.
            unsafe {
                s.launch_builder(&sum.kernels.fingerprint)
                    .arg(&*recv_x)
                    .arg(&*recv_z)
                    .arg(&n32)
                    .arg(&sum.fp_rows)
                    .arg(&mut *recv_g)
                    .launch(thread_per(total, 256))?;
            }
        }
        #[cfg(feature = "phase-timing")]
        body_ns.set(t_body.elapsed().as_nanos() as u64);
        Ok::<u64, GpuError>(total as u64)
    });
    #[cfg(feature = "phase-timing")]
    {
        laps.exchange_ns += t_exchange.elapsed().as_nanos() as u64 - body_ns.get();
    }
    pool.extend(recv.into_iter().flatten());
    let rows_received = body?;
    #[cfg(feature = "phase-timing")]
    {
        laps.recv_rows += rows_received;
    }
    counts.rows_received = rows_received;
    counts.remote_deltas = plan.remote.len();
    Ok(counts)
}

/// The device-payload exchange: K10 fills device columns and fingerprints them, the payload moves, the receiver copies device-to-device.
fn exchange_rows_device<const W: usize, X: Transport>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    plan: &PartitionPlan,
    scratch: &mut LayerScratch<W>,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    let size = transport.size();
    #[cfg(feature = "phase-timing")]
    let t_export = std::time::Instant::now();
    let (send, mut counts) = match export_blocks_device(sum, table, plan, scratch, size, true) {
        Ok(v) => v,
        Err(e) => {
            pair_empty_exchange::<W, X>(transport, plan, sum.hash.bits(), &mut scratch.export);
            return Err(e);
        }
    };
    #[cfg(feature = "phase-timing")]
    {
        scratch.laps.export_ns += t_export.elapsed().as_nanos() as u64;
        scratch.laps.rows_exported += counts.rows_sent.iter().sum::<u64>();
    }
    let b = sum.hash.num_buckets();
    let export = &mut scratch.export;
    let xfer_ns = &mut scratch.xfer_ns;
    #[cfg(feature = "phase-timing")]
    let laps = &mut scratch.laps;
    export.chunks.rebuild(
        &Gf2Span::new(&plan.local_bucket_deltas, sum.hash.bits()),
        b,
        exchange_chunks(),
    );
    let map = std::mem::take(&mut export.chunks);
    #[cfg(feature = "phase-timing")]
    let t_exchange = std::time::Instant::now();
    let (recv, body) = transport.exchange_layer(send, &mut Vec::new(), &map, |recv, wait| {
        #[cfg(feature = "phase-timing")]
        let t_body = std::time::Instant::now();
        for k in 0..map.chunks() {
            wait.wait_chunk(k);
        }
        #[cfg(feature = "phase-timing")]
        {
            laps.chunk_wait_ns += t_body.elapsed().as_nanos() as u64;
        }
        let blocks = paired_blocks(plan, recv, |p: &DevicePayload<W>, j| p.blocks.get(j));
        adopt_blocks(sum, export, &blocks, xfer_ns)
    });
    export.chunks = map;
    // The move plus the device copies: there is no host body to subtract.
    #[cfg(feature = "phase-timing")]
    {
        laps.exchange_ns += t_exchange.elapsed().as_nanos() as u64;
    }
    recv.into_iter().flatten().for_each(payload::recycle);
    let rows_received = body?;
    #[cfg(feature = "phase-timing")]
    {
        laps.recv_rows += rows_received;
    }
    counts.rows_received = rows_received;
    counts.remote_deltas = plan.remote.len();
    Ok(counts)
}

/// The NCCL exchange (ARCHITECTURE.md §Partitioning): K10 fills device blocks, their skeletons cross over `transport`, the group votes, and on a unanimous yes one wire group moves every column straight into the concatenated `recv_*` columns, which the receiver then fingerprints.
/// A rank that cannot receive votes no and returns its error; on any no nobody posts, and a ready rank takes every received block as empty and fails nothing of its own.
#[cfg(feature = "nccl")]
fn exchange_rows_nccl<const W: usize, X: Transport>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    plan: &PartitionPlan,
    scratch: &mut LayerScratch<W>,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    let size = transport.size();
    #[cfg(feature = "phase-timing")]
    let t_export = std::time::Instant::now();
    let (send, mut counts) = match export_blocks_device(sum, table, plan, scratch, size, false) {
        Ok(v) => v,
        Err(e) => {
            pair_empty_exchange::<W, X>(transport, plan, sum.hash.bits(), &mut scratch.export);
            return Err(e);
        }
    };
    // The fingerprint column stays behind.
    for (q, p) in send.iter().enumerate() {
        counts.bytes_sent[q] = p.as_ref().map_or(0, |p| {
            p.blocks
                .iter()
                .map(|b| (b.bytes() - b.rows() * std::mem::size_of::<u64>()) as u64)
                .sum()
        });
    }
    #[cfg(feature = "phase-timing")]
    {
        scratch.laps.export_ns += t_export.elapsed().as_nanos() as u64;
        scratch.laps.rows_exported += counts.rows_sent.iter().sum::<u64>();
    }
    let b = sum.hash.num_buckets();
    let export = &mut scratch.export;
    let xfer_ns = &mut scratch.xfer_ns;
    export.chunks.rebuild(
        &Gf2Span::new(&plan.local_bucket_deltas, sum.hash.bits()),
        b,
        1,
    );
    #[cfg(feature = "phase-timing")]
    let t_exchange = std::time::Instant::now();
    let skeletons: Vec<Option<BlockSkeletons<W>>> = send
        .iter()
        .map(|p| {
            p.as_ref().map(|p| {
                let mut s = export.skeletons.pop().unwrap_or_default();
                s.fill_from(&p.blocks);
                s
            })
        })
        .collect();
    let recv = transport.exchange(skeletons, &mut export.skeletons);
    let mut posted = false;
    let rows = {
        let blocks = paired_blocks(plan, &recv, |p: &BlockSkeletons<W>, j| p.blocks.get(j));
        let ready = stage_receive(sum, export, &blocks, xfer_ns).and_then(|()| wire_ready(export));
        if vote(transport, ready.is_ok()) {
            posted = ready.is_ok();
            ready.and_then(|()| post_nccl_group(sum, plan, export, &send, &blocks, transport))
        } else {
            ready.and_then(|()| discard_received(export))
        }
    };
    export.skeletons.extend(recv.into_iter().flatten());
    if posted && rows.is_err() {
        export.quarantine.extend(send.into_iter().flatten());
    } else {
        send.into_iter().flatten().for_each(payload::recycle);
    }
    #[cfg(feature = "phase-timing")]
    {
        scratch.laps.exchange_ns += t_exchange.elapsed().as_nanos() as u64;
    }
    let rows_received = rows?;
    #[cfg(feature = "phase-timing")]
    {
        scratch.laps.recv_rows += rows_received;
    }
    counts.rows_received = rows_received;
    counts.remote_deltas = plan.remote.len();
    Ok(counts)
}

/// The receive layout from the skeletons in plan order, as `adopt_blocks` lays it out, checked against the segment cap before any row moves; grows `recv_*` and uploads the offsets and bases.
#[cfg(feature = "nccl")]
fn stage_receive<const W: usize>(
    sum: &GpuSum<W>,
    export: &mut DeviceExport<W>,
    blocks: &[&Skeleton],
    xfer_ns: &mut XferNs,
) -> Result<(), GpuError> {
    let b = sum.hash.num_buckets();
    let s = &sum.stream;
    let o = sum.device();
    let mut total = 0usize;
    let mut max_seg = 0usize;
    export.off_host.clear();
    export.base_host.clear();
    export.base_host.resize(16, 0);
    for (k, block) in blocks.iter().enumerate() {
        assert_eq!(
            block.header.num_buckets as usize, b,
            "a partner sent a block indexed by {} buckets where this partition has {b}",
            block.header.num_buckets
        );
        export.base_host[k] = total as u32;
        export.off_host.extend_from_slice(&block.offsets[..b + 1]);
        max_seg = block.offsets[..b + 1]
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0)
            .max(max_seg);
        total += block.header.rows as usize;
    }
    export.recv_max_segment = max_seg;
    export.recv_rows = total;
    if max_seg > MAX_RECV_SEGMENT {
        return Err(GpuError::Unsupported(
            "a fused-layer block or a received segment exceeds the record cap at the agreed bucket count",
        ));
    }
    #[cfg(any(test, feature = "test-utils"))]
    if std::mem::take(&mut export.fail_recv_growth) {
        return Err(GpuError::OutOfMemory {
            device: o,
            bytes: (total * (2 * W + 3) * std::mem::size_of::<u64>()) as u64,
        });
    }
    let DeviceExport {
        recv_off,
        recv_base,
        recv_x,
        recv_z,
        recv_c,
        recv_g,
        off_host,
        base_host,
        ..
    } = export;
    grow(s, recv_off, off_host.len().max(2), o)?;
    grow(s, recv_x, total.max(1) * W, o)?;
    grow(s, recv_z, total.max(1) * W, o)?;
    grow(s, recv_c, 2 * total.max(1), o)?;
    grow(s, recv_g, total.max(1), o)?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    xfer(xfer_ns, Xfer::H2d, || {
        s.memcpy_htod(&off_host[..], &mut recv_off.slice_mut(0..off_host.len()))?;
        s.memcpy_htod(&base_host[..], recv_base)?;
        Ok(())
    })
}

#[cfg(all(feature = "nccl", test))]
impl<const W: usize> DeviceExport<W> {
    /// Device addresses of every quarantined block's key column (test hook).
    pub(crate) fn quarantined_ptrs(&self) -> Vec<u64> {
        use cudarc::driver::DevicePtr;
        self.quarantine
            .iter()
            .flat_map(|p| &p.blocks)
            .map(|b| b.x.device_ptr(&self.stream).0)
            .collect()
    }
}

/// Whether this rank's wire can carry the group, so a failure it can predict is a no vote rather than an error after a yes.
#[cfg(feature = "nccl")]
fn wire_ready<const W: usize>(export: &DeviceExport<W>) -> Result<(), GpuError> {
    match &export.wire {
        None => Err(GpuError::Unsupported(
            "GpuExchange::Nccl without a device wire",
        )),
        Some(wire) if !wire.is_healthy() => Err(GpuError::Nccl {
            code: cudarc::nccl::sys::ncclResult_t::ncclInvalidUsage as i32,
            what: "an exchange on a failed or aborted device wire".to_string(),
        }),
        Some(_) => Ok(()),
    }
}

/// A ready rank's side of a no vote: every received block is empty, as a failed partner's would be on the host path.
#[cfg(feature = "nccl")]
fn discard_received<const W: usize>(export: &mut DeviceExport<W>) -> Result<u64, GpuError> {
    let n = export.off_host.len();
    export
        .stream
        .memset_zeros(&mut export.recv_off.slice_mut(0..n))?;
    export.recv_rows = 0;
    export.recv_max_segment = 0;
    Ok(0)
}

/// One wire group of [`nccl_schedule`]'s transfers, completed, then the received rows' fingerprints; returns the rows received.
#[cfg(feature = "nccl")]
fn post_nccl_group<const W: usize, X: Transport>(
    sum: &GpuSum<W>,
    plan: &PartitionPlan,
    export: &mut DeviceExport<W>,
    send: &[Option<DevicePayload<W>>],
    recv: &[&Skeleton],
    transport: &X,
) -> Result<u64, GpuError> {
    let wire = export.wire.clone().ok_or(GpuError::Unsupported(
        "GpuExchange::Nccl without a device wire",
    ))?;
    assert_eq!(
        (wire.rank(), wire.size()),
        (transport.rank(), transport.size()),
        "the device wire's ranks are not the transport's"
    );
    assert_eq!(
        export.chunks.chunks(),
        1,
        "the NCCL exchange receives whole blocks into the concatenated columns"
    );
    let mut used = vec![0usize; send.len()];
    let own: Vec<&DeviceBlock<W>> = plan
        .remote
        .iter()
        .map(|r| {
            let q = r.partner as usize;
            let j = used[q];
            used[q] += 1;
            &send[q]
                .as_ref()
                .expect("a payload for every partner")
                .blocks[j]
        })
        .collect();
    let partners: Vec<u32> = plan.remote.iter().map(|r| r.partner).collect();
    let own_off: Vec<&[u32]> = own.iter().map(|b| b.offsets.as_slice()).collect();
    let recv_off: Vec<&[u32]> = recv.iter().map(|b| b.offsets.as_slice()).collect();
    let ops = nccl_schedule(&partners, &own_off, &recv_off, &export.chunks);
    let s: &CudaStream = &sum.stream;
    let total = export.recv_rows;
    let mut group = WireGroup::new();
    for op in ops.iter().filter(|op| op.kind == WireOpKind::Send) {
        let block = own[op.k];
        let (lo, hi) = op.rows;
        let e = op.column.elems_per_row::<W>();
        match op.column {
            WireColumn::X => group.send(block.x.slice(lo * e..hi * e), op.peer, s),
            WireColumn::Z => group.send(block.z.slice(lo * e..hi * e), op.peer, s),
            WireColumn::Coeff => group.send(block.c.slice(lo * e..hi * e), op.peer, s),
        }
    }
    let parts = |column: WireColumn| -> Vec<(usize, u32)> {
        let e = column.elems_per_row::<W>();
        ops.iter()
            .filter(|op| op.kind == WireOpKind::Recv && op.column == column)
            .map(|op| ((op.rows.1 - op.rows.0) * e, op.peer))
            .collect()
    };
    let DeviceExport {
        recv_x,
        recv_z,
        recv_c,
        recv_g,
        ..
    } = export;
    group.recv_parts(recv_x.slice_mut(0..total * W), &parts(WireColumn::X), s);
    group.recv_parts(recv_z.slice_mut(0..total * W), &parts(WireColumn::Z), s);
    group.recv_parts(recv_c.slice_mut(0..2 * total), &parts(WireColumn::Coeff), s);
    group.post(&*wire)?;
    wire.wait(s)?;
    if total > 0 {
        let n32 = total as u32;
        // SAFETY: arguments match `k_fingerprint` in fingerprint.cu; `recv_g` holds `total` entries.
        unsafe {
            s.launch_builder(&sum.kernels.fingerprint)
                .arg(&*recv_x)
                .arg(&*recv_z)
                .arg(&n32)
                .arg(&sum.fp_rows)
                .arg(&mut *recv_g)
                .launch(thread_per(total, 256))?;
        }
    }
    Ok(total as u64)
}

/// Received device blocks into the concatenated `recv_*` columns, one device-to-device copy per column; returns the rows adopted.
/// Every copy has completed on return, so the blocks may go back to their pool.
pub(crate) fn adopt_blocks<const W: usize>(
    sum: &GpuSum<W>,
    export: &mut DeviceExport<W>,
    blocks: &[&DeviceBlock<W>],
    xfer_ns: &mut XferNs,
) -> Result<u64, GpuError> {
    let b = sum.hash.num_buckets();
    let s = &sum.stream;
    let o = sum.device();
    let DeviceExport {
        recv_off,
        recv_base,
        recv_x,
        recv_z,
        recv_c,
        recv_g,
        off_host,
        base_host,
        recv_max_segment,
        recv_rows,
        ..
    } = export;
    let mut total = 0usize;
    let mut max_seg = 0usize;
    off_host.clear();
    base_host.clear();
    base_host.resize(16, 0);
    for (k, block) in blocks.iter().enumerate() {
        assert_eq!(
            block.header.num_buckets as usize, b,
            "a partner sent a block indexed by {} buckets where this partition has {b}",
            block.header.num_buckets
        );
        base_host[k] = total as u32;
        off_host.extend_from_slice(&block.offsets[..b + 1]);
        max_seg = block.offsets[..b + 1]
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0)
            .max(max_seg);
        total += block.rows();
    }
    *recv_max_segment = max_seg;
    *recv_rows = total;
    grow(s, recv_off, off_host.len().max(2), o)?;
    grow(s, recv_x, total.max(1) * W, o)?;
    grow(s, recv_z, total.max(1) * W, o)?;
    grow(s, recv_c, 2 * total.max(1), o)?;
    grow(s, recv_g, total.max(1), o)?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    xfer(xfer_ns, Xfer::H2d, || {
        s.memcpy_htod(&off_host[..], &mut recv_off.slice_mut(0..off_host.len()))?;
        s.memcpy_htod(&base_host[..], recv_base)?;
        Ok(())
    })?;
    for (k, block) in blocks.iter().enumerate() {
        let n = block.rows();
        if n == 0 {
            continue;
        }
        if block.x.context().ordinal() != sum.ctx.ordinal() {
            let _ = payload::enable_peer_access(&sum.ctx, block.x.context());
        }
        let base = base_host[k] as usize;
        s.memcpy_dtod(
            &block.x.slice(0..n * W),
            &mut recv_x.slice_mut(base * W..(base + n) * W),
        )?;
        s.memcpy_dtod(
            &block.z.slice(0..n * W),
            &mut recv_z.slice_mut(base * W..(base + n) * W),
        )?;
        s.memcpy_dtod(
            &block.c.slice(0..2 * n),
            &mut recv_c.slice_mut(2 * base..2 * (base + n)),
        )?;
        s.memcpy_dtod(&block.g.slice(0..n), &mut recv_g.slice_mut(base..base + n))?;
    }
    s.synchronize()?;
    Ok(total as u64)
}

/// The received block per remote delta in plan order: the `j`-th of `remote_for_partner(q)` is the `j`-th block of `recv[q]`, as `RecvRows::new`.
fn paired_blocks<'a, P, B>(
    plan: &PartitionPlan,
    recv: &'a [Option<P>],
    block: impl Fn(&'a P, usize) -> Option<&'a B>,
) -> Vec<&'a B> {
    plan.remote
        .iter()
        .map(|r| {
            let j = plan
                .remote_for_partner(r.partner)
                .position(|other| other.entry == r.entry)
                .expect("a remote delta is in its own partner's list");
            recv[r.partner as usize]
                .as_ref()
                .and_then(|payload| block(payload, j))
                .unwrap_or_else(|| {
                    panic!(
                        "partition {} sent no block for remote delta {j} (entry {})",
                        r.partner, r.entry
                    )
                })
        })
        .collect()
}

/// K10's count and scan for entry `e` at bucket delta `bd`, leaving the CSR offsets in `scratch.export.off` and their copy in `offsets`; returns the block's rows.
fn export_offsets<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    scratch: &mut LayerScratch<W>,
    bd: u32,
    e: u32,
    offsets: &mut Vec<u32>,
) -> Result<usize, GpuError> {
    let s = &sum.stream;
    let k = &sum.kernels;
    let b = sum.hash.num_buckets();
    let (e32, b32) = (table.entries as u32, b as u32);
    // SAFETY: arguments match `k_export_counts` in export.cu; `counts` holds `b` entries.
    unsafe {
        s.launch_builder(&k.export_counts)
            .arg(&scratch.cnt)
            .arg(&scratch.bucket_at)
            .arg(&bd)
            .arg(&e)
            .arg(&e32)
            .arg(&b32)
            .arg(&mut scratch.export.counts)
            .launch(thread_per(b, 1024))?;
    }
    super::scan::exclusive_scan_with_max_into(
        s,
        k,
        &scratch.export.counts.slice(0..b),
        &mut scratch.export.off.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut scratch.tot_a,
    )?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    offsets.clear();
    offsets.resize(b + 1, 0);
    let off = &scratch.export.off;
    xfer(&mut scratch.xfer_ns, Xfer::D2h, || {
        s.memcpy_dtoh(&off.slice(0..b + 1), &mut offsets[..])?;
        s.synchronize()?;
        Ok(())
    })?;
    Ok(offsets[b] as usize)
}

/// The scratch buffers K10's fill reads, borrowed apart from the columns it writes.
struct FillCtx<'a> {
    bucket_at: &'a CudaSlice<u32>,
    amp: &'a CudaSlice<f64>,
    mask: &'a CudaSlice<u64>,
    nz: &'a CudaSlice<u32>,
    off: &'a CudaSlice<u32>,
}

/// K10's fill of entry `e` at bucket delta `bd` into `(x, z, c)`, which hold the rows `ctx.off` names.
#[allow(clippy::too_many_arguments)]
fn export_fill<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    ctx: FillCtx<'_>,
    bd: u32,
    e: u32,
    x: &mut CudaSlice<u64>,
    z: &mut CudaSlice<u64>,
    c: &mut CudaSlice<f64>,
) -> Result<(), GpuError> {
    let s = &sum.stream;
    let b = sum.hash.num_buckets();
    let (e32, b32) = (table.entries as u32, b as u32);
    // SAFETY: arguments match `k_export_fill` in export.cu; the columns hold the block's rows and `off` its offsets.
    unsafe {
        s.launch_builder(&sum.kernels.export_fill)
            .arg(&sum.cols.x)
            .arg(&sum.cols.z)
            .arg(&sum.cols.coeff)
            .arg(&sum.cols.start)
            .arg(&sum.cols.lens)
            .arg(ctx.bucket_at)
            .arg(&table.mode)
            .arg(&e32)
            .arg(&table.kq)
            .arg(&table.q0)
            .arg(&table.q1)
            .arg(&table.rot_cos)
            .arg(&table.rot_sin)
            .arg(ctx.amp)
            .arg(ctx.mask)
            .arg(ctx.nz)
            .arg(&bd)
            .arg(&e)
            .arg(&b32)
            .arg(ctx.off)
            .arg(x)
            .arg(z)
            .arg(c)
            .launch(warp_per_bucket(b))?;
    }
    Ok(())
}

/// K10 for every remote delta: one host block per delta into pooled payloads, in ascending remote-delta index per partner.
pub(crate) fn export_blocks<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    plan: &PartitionPlan,
    scratch: &mut LayerScratch<W>,
    size: u32,
) -> Result<(Vec<Option<PartnerPayload<W>>>, LayerExchangeCounts), GpuError> {
    let mut send: Vec<Option<PartnerPayload<W>>> = (0..size).map(|_| None).collect();
    let mut counts = LayerExchangeCounts::none(size);
    let b = sum.hash.num_buckets();
    let s: Arc<CudaStream> = sum.stream.clone();
    let o = sum.device();
    grow(&s, &mut scratch.export.counts, b, o)?;
    grow(&s, &mut scratch.export.off, b + 1, o)?;
    let groups = premerge_groups(sum, table, plan, scratch, size)?;
    let mut merged = vec![false; size as usize];
    for (q, entries) in groups.iter().enumerate() {
        if entries.is_empty() {
            continue;
        }
        let payload = send[q].get_or_insert_with(|| scratch.export.pool.pop().unwrap_or_default());
        if payload.blocks.len() < entries.len() {
            payload
                .blocks
                .resize_with(entries.len(), ExchangeBlock::<W>::default);
        }
        let blocks = &mut payload.blocks[..entries.len()];
        if let Some(rows) = premerge_partner(sum, table, entries, scratch, MergeInto::Host(blocks))?
        {
            merged[q] = true;
            counts.rows_sent[q] += rows;
            counts.bytes_sent[q] += blocks.iter().map(|b| b.bytes() as u64).sum::<u64>();
        }
    }
    let mut blocks_used = vec![0usize; size as usize];
    for r in &plan.remote {
        let q = r.partner as usize;
        let payload = send[q].get_or_insert_with(|| scratch.export.pool.pop().unwrap_or_default());
        let j = blocks_used[q];
        blocks_used[q] += 1;
        if merged[q] {
            continue;
        }
        if payload.blocks.len() <= j {
            payload
                .blocks
                .resize_with(j + 1, ExchangeBlock::<W>::default);
        }
        let block = &mut payload.blocks[j];
        let (bd, e) = (r.bucket_delta, r.entry as u32);
        let t0 = scratch.event(sum)?;
        let rows = export_offsets(sum, table, scratch, bd, e, &mut block.offsets)?;
        block.header = BlockHeader {
            num_buckets: b as u32,
            rows: rows as u32,
            w: W as u32,
            entry: e,
        };
        if block.coeff.len() < rows {
            block.x.resize(rows, [0u64; W]);
            block.z.resize(rows, [0u64; W]);
            block.coeff.resize(rows, Complex64::new(0.0, 0.0));
        }
        if rows > 0 {
            grow(&s, &mut scratch.export.x, rows * W, o)?;
            grow(&s, &mut scratch.export.z, rows * W, o)?;
            grow(&s, &mut scratch.export.c, 2 * rows, o)?;
            {
                let LayerScratch {
                    bucket_at,
                    amp,
                    mask,
                    nz,
                    export,
                    ..
                } = &mut *scratch;
                let DeviceExport {
                    off,
                    x: ex,
                    z: ez,
                    c: ec,
                    ..
                } = export;
                let ctx = FillCtx {
                    bucket_at,
                    amp,
                    mask,
                    nz,
                    off,
                };
                export_fill(sum, table, ctx, bd, e, ex, ez, ec)?;
            }
            scratch.lap(sum, t0, |m| &mut m.export)?;
            #[cfg(feature = "phase-timing")]
            s.synchronize()?;
            let DeviceExport {
                x: ex,
                z: ez,
                c: ec,
                pinned,
                ..
            } = &mut scratch.export;
            let (ex, ez, ec): (&CudaSlice<u64>, &CudaSlice<u64>, &CudaSlice<f64>) = (ex, ez, ec);
            xfer(&mut scratch.xfer_ns, Xfer::D2h, || {
                download_block(&s, rows, (ex, ez, ec), pinned, block)
            })?;
        } else {
            scratch.lap(sum, t0, |m| &mut m.export)?;
        }
        counts.rows_sent[q] += rows as u64;
        counts.bytes_sent[q] += block.bytes() as u64;
    }
    for (q, payload) in send.iter_mut().enumerate() {
        if let Some(payload) = payload {
            payload.blocks.truncate(blocks_used[q]);
        }
    }
    Ok((send, counts))
}

/// K10 for every remote delta into device blocks, each fingerprinted on the sender when `fingerprint`: one block per delta into pooled [`DevicePayload`]s, in ascending remote-delta index per partner.
pub(crate) fn export_blocks_device<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    plan: &PartitionPlan,
    scratch: &mut LayerScratch<W>,
    size: u32,
    fingerprint: bool,
) -> Result<(Vec<Option<DevicePayload<W>>>, LayerExchangeCounts), GpuError> {
    let mut send: Vec<Option<DevicePayload<W>>> = (0..size).map(|_| None).collect();
    let mut counts = LayerExchangeCounts::none(size);
    let b = sum.hash.num_buckets();
    let s: Arc<CudaStream> = sum.stream.clone();
    let o = sum.device();
    grow(&s, &mut scratch.export.counts, b, o)?;
    grow(&s, &mut scratch.export.off, b + 1, o)?;
    let groups = premerge_groups(sum, table, plan, scratch, size)?;
    let mut merged = vec![false; size as usize];
    for (q, entries) in groups.iter().enumerate() {
        if entries.is_empty() {
            continue;
        }
        let payload = send[q].get_or_insert_with(|| payload::reclaim::<W>(o));
        payload.block_mut(entries.len() - 1, &s)?;
        let blocks = &mut payload.blocks[..entries.len()];
        if let Some(rows) = premerge_partner(
            sum,
            table,
            entries,
            scratch,
            MergeInto::Device(blocks, fingerprint),
        )? {
            merged[q] = true;
            counts.rows_sent[q] += rows;
            counts.bytes_sent[q] += blocks.iter().map(|b| b.bytes() as u64).sum::<u64>();
        }
    }
    let mut blocks_used = vec![0usize; size as usize];
    for r in &plan.remote {
        let q = r.partner as usize;
        let payload = send[q].get_or_insert_with(|| payload::reclaim::<W>(o));
        let j = blocks_used[q];
        blocks_used[q] += 1;
        if merged[q] {
            continue;
        }
        let block = payload.block_mut(j, &s)?;
        let (bd, e) = (r.bucket_delta, r.entry as u32);
        let t0 = scratch.event(sum)?;
        let rows = export_offsets(sum, table, scratch, bd, e, &mut block.offsets)?;
        block.set_header(e, b);
        if rows > 0 {
            block.grow(&s, rows, o)?;
            let ctx = FillCtx {
                bucket_at: &scratch.bucket_at,
                amp: &scratch.amp,
                mask: &scratch.mask,
                nz: &scratch.nz,
                off: &scratch.export.off,
            };
            export_fill(
                sum,
                table,
                ctx,
                bd,
                e,
                &mut block.x,
                &mut block.z,
                &mut block.c,
            )?;
            if fingerprint {
                let n32 = rows as u32;
                // SAFETY: arguments match `k_fingerprint` in fingerprint.cu; the block's columns hold `rows` rows.
                unsafe {
                    s.launch_builder(&sum.kernels.fingerprint)
                        .arg(&block.x)
                        .arg(&block.z)
                        .arg(&n32)
                        .arg(&sum.fp_rows)
                        .arg(&mut block.g)
                        .launch(thread_per(rows, 256))?;
                }
            }
        }
        scratch.lap(sum, t0, |m| &mut m.export)?;
        counts.rows_sent[q] += rows as u64;
        counts.bytes_sent[q] += block.bytes() as u64;
    }
    for (q, payload) in send.iter_mut().enumerate() {
        if let Some(payload) = payload {
            payload.blocks.truncate(blocks_used[q]);
        }
    }
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    Ok((send, counts))
}

/// One partner's blocks as the sender-side merge fills them, in the partner's remote-delta order; a device block's fingerprints are written when the flag is set.
enum MergeInto<'a, const W: usize> {
    Host(&'a mut [ExchangeBlock<W>]),
    Device(&'a mut [DeviceBlock<W>], bool),
}

/// Elements `exclusive_scan_with_max_into` handles, so the split's `K × positions` offsets must fit.
const SCAN_LIMIT: usize = 1 << 24;

/// Per partner, the remote entries the sender-side merge applies to, empty where it does not: at least two entries that can emit one key (`DevicePrepared::entries_can_collide`) and every source bucket inside the tag's offset field.
fn premerge_groups<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    plan: &PartitionPlan,
    scratch: &mut LayerScratch<W>,
    size: u32,
) -> Result<Vec<Vec<usize>>, GpuError> {
    let mut groups = vec![Vec::new(); size as usize];
    if !scratch.options.premerge {
        return Ok(groups);
    }
    for r in &plan.remote {
        groups[r.partner as usize].push(r.entry);
    }
    let b = sum.hash.num_buckets();
    for g in &mut groups {
        if g.len() < 2 || g.len() * b > SCAN_LIMIT || !table.entries_can_collide(g) {
            g.clear();
        }
    }
    if groups.iter().all(Vec::is_empty) {
        return Ok(groups);
    }
    // K3's tag addresses 4096 rows of a source bucket; a longer one fails the layer after the exchange, and must not reach K3 before it.
    let s = &sum.stream;
    let o = sum.device();
    let pm = &mut scratch.export.premerge;
    grow(s, &mut pm.start, b + 1, o)?;
    exclusive_scan_with_max_into(
        s,
        &sum.kernels,
        &sum.cols.lens.slice(0..b),
        &mut pm.start.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut pm.tot,
    )?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    let tot = &pm.tot;
    let longest = xfer(&mut scratch.xfer_ns, Xfer::D2h, || {
        let v = s.clone_dtoh(tot)?;
        s.synchronize()?;
        Ok(v[1] as usize)
    })?;
    if longest > MAX_BUCKET_LEN {
        groups.iter_mut().for_each(Vec::clear);
    }
    Ok(groups)
}

/// The sender-side merge of one partner's blocks (ARCHITECTURE.md §Partitioning): K3 over the sub-table of the partner's remote `entries` under the keep-everything program, so equal keys sum and exact zeros drop but nothing truncates; then each position's merged rows are split greedily over the blocks in entry order, never more than a block's unmerged count there, so every segment still fits the receiver's tag.
/// Returns the rows written, or `None` having written nothing when a position's records exceed the fused kernel's cap.
fn premerge_partner<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    entries: &[usize],
    scratch: &mut LayerScratch<W>,
    mut into: MergeInto<'_, W>,
) -> Result<Option<u64>, GpuError> {
    let s: Arc<CudaStream> = sum.stream.clone();
    let k = sum.kernels.clone();
    let o = sum.device();
    let b = sum.hash.num_buckets();
    let nk = entries.len();
    let sub = table.restrict(entries);
    let mut sel = [0u32; 16];
    for (j, &e) in entries.iter().enumerate() {
        sel[j] = e as u32;
    }
    let (b32, k32, e32) = (b as u32, nk as u32, table.entries as u32);
    let t0 = scratch.event(sum)?;
    {
        let pm = &mut scratch.export.premerge;
        grow(&s, &mut pm.cnt, b * nk, o)?;
        grow(&s, &mut pm.rows, b, o)?;
        grow(&s, &mut pm.start, b + 1, o)?;
        grow(&s, &mut pm.out_len_pos, b, o)?;
        #[cfg(feature = "phase-timing")]
        s.synchronize()?;
        xfer(&mut scratch.xfer_ns, Xfer::H2d, || {
            s.memcpy_htod(&sub.amp, &mut pm.amp)?;
            s.memcpy_htod(&sub.mask, &mut pm.mask)?;
            s.memcpy_htod(&sub.nz, &mut pm.nz)?;
            s.memcpy_htod(&sub.bucket_delta, &mut pm.bd)?;
            s.memcpy_htod(&sub.gm, &mut pm.gm)?;
            s.memcpy_htod(&sub.rem, &mut pm.rem)?;
            s.memcpy_htod(&sel[..], &mut pm.sel)?;
            Ok(())
        })?;
        // SAFETY: arguments match `k_premerge_counts` in export.cu; `cnt` holds the layer's `b × E` counts and `pm.cnt` room for `b × K`.
        unsafe {
            s.launch_builder(&k.premerge_counts)
                .arg(&scratch.cnt)
                .arg(&e32)
                .arg(&pm.sel)
                .arg(&k32)
                .arg(&b32)
                .arg(&mut pm.cnt)
                .launch(thread_per(b * nk, 256))?;
        }
        // SAFETY: arguments match `k_rows` in count.cu; every `rem` entry is `NO_REMOTE`, so `recv_off` is never read.
        unsafe {
            s.launch_builder(&k.rows)
                .arg(&pm.cnt)
                .arg(&scratch.bucket_at)
                .arg(&pm.bd)
                .arg(&pm.rem)
                .arg(&scratch.export.recv_off)
                .arg(&mut pm.rows)
                .arg(&b32)
                .arg(&k32)
                .launch(thread_per(b, 1024))?;
        }
        exclusive_scan_with_max_into(
            &s,
            &k,
            &pm.rows.slice(0..b),
            &mut pm.start.slice_mut(0..b + 1),
            b,
            &mut scratch.scan,
            &mut pm.tot,
        )?;
    }
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    let (total, most) = {
        let pm = &mut scratch.export.premerge;
        pm.start_host.resize(b + 1, 0);
        let (start, start_host, tot) = (&pm.start, &mut pm.start_host, &pm.tot);
        xfer(&mut scratch.xfer_ns, Xfer::D2h, || {
            let v = s.clone_dtoh(tot)?;
            s.memcpy_dtoh(&start.slice(0..b + 1), &mut start_host[..])?;
            s.synchronize()?;
            Ok((v[0] as u64, v[1] as usize))
        })?
    };
    let cap = k.layer_cap();
    if most > cap {
        scratch.lap(sum, t0, |m| &mut m.export)?;
        return Ok(None);
    }
    let (func, _, smem) = fused_variant(&k, W, most, sub.dense);
    let (batches, max_batch_rows) = arena_batches::<W>(
        &scratch.export.premerge.start_host,
        scratch.options.arena_bytes,
        cap,
    );
    let mut arena = match scratch.arena.take() {
        Some(a) => a,
        None => DeviceColumns::<W>::with_capacity(&s, o, max_batch_rows, 1)?,
    };
    arena.len = 0;
    arena.buckets = 0;
    arena.reserve(max_batch_rows, 1)?;
    let mut running = vec![0u32; nk];
    match &mut into {
        MergeInto::Host(blocks) => blocks.iter_mut().for_each(|bl| {
            bl.offsets.clear();
            bl.offsets.resize(b + 1, 0);
        }),
        MergeInto::Device(blocks, _) => blocks.iter_mut().for_each(|bl| {
            bl.offsets.clear();
            bl.offsets.resize(b + 1, 0);
        }),
    }
    let r = premerge_batches(
        sum,
        &sub,
        scratch,
        &mut arena,
        &batches,
        (func, smem),
        &mut running,
        &mut into,
    );
    scratch.arena = Some(arena);
    r?;
    let mut merged = 0u64;
    for (j, &e) in entries.iter().enumerate() {
        let rows = running[j];
        merged += u64::from(rows);
        match &mut into {
            MergeInto::Host(blocks) => {
                let bl = &mut blocks[j];
                bl.offsets[b] = rows;
                bl.header = BlockHeader {
                    num_buckets: b32,
                    rows,
                    w: W as u32,
                    entry: e as u32,
                };
            }
            MergeInto::Device(blocks, _) => {
                blocks[j].offsets[b] = rows;
                blocks[j].set_header(e as u32, b);
            }
        }
    }
    let fb = s.clone_dtoh(&scratch.export.premerge.fallback)?;
    s.synchronize()?;
    scratch.counters.fallback_hi += fb[0];
    scratch.counters.fallback_key += fb[1];
    scratch.counters.rows_premerged += total - merged;
    scratch.lap(sum, t0, |m| &mut m.export)?;
    Ok(Some(merged))
}

/// The merge's batch loop: K3 into the arena, the split and its scan, the block offsets, and the copy of every block's share.
#[allow(clippy::too_many_arguments)]
fn premerge_batches<const W: usize>(
    sum: &GpuSum<W>,
    sub: &DevicePrepared<W>,
    scratch: &mut LayerScratch<W>,
    arena: &mut DeviceColumns<W>,
    batches: &[(usize, usize)],
    kernel: (&cudarc::driver::CudaFunction, u32),
    running: &mut [u32],
    into: &mut MergeInto<'_, W>,
) -> Result<(), GpuError> {
    let s = &sum.stream;
    let k = &sum.kernels;
    let o = sum.device();
    let nk = running.len();
    let k32 = nk as u32;
    let LayerScratch {
        bucket_at,
        scan,
        export,
        xfer_ns,
        ..
    } = scratch;
    let DeviceExport {
        x: ex,
        z: ez,
        c: ec,
        pinned,
        premerge: pm,
        recv_off,
        recv_base,
        recv_x,
        recv_z,
        recv_c,
        recv_g,
        ..
    } = export;
    s.memset_zeros(&mut pm.fallback)?;
    for &(p0, p1) in batches {
        let n = p1 - p0;
        let (p0u, n32) = (p0 as u32, n as u32);
        let fused = FusedTable {
            cnt: &pm.cnt,
            seg_start: &pm.start,
            amp: &pm.amp,
            mask: &pm.mask,
            nz: &pm.nz,
            bd: &pm.bd,
            gm: &pm.gm,
            rem: &pm.rem,
        };
        let recv = FusedRecv {
            off: recv_off,
            base: recv_base,
            x: recv_x,
            z: recv_z,
            c: recv_c,
            g: recv_g,
        };
        let written = FusedOut {
            arena: &mut *arena,
            out_len_pos: &mut pm.out_len_pos,
            fallback: &mut pm.fallback,
        };
        launch_fused(
            sum,
            sub,
            &fused,
            &recv,
            bucket_at,
            &KeepProgram::KEEP,
            kernel,
            (p0u, n32),
            written,
        )?;
        grow(s, &mut pm.lens, nk * n, o)?;
        grow(s, &mut pm.loff, nk * n + 1, o)?;
        // SAFETY: arguments match `k_premerge_split` in export.cu; `lens` holds `K × n` entries.
        unsafe {
            s.launch_builder(&k.premerge_split)
                .arg(&pm.out_len_pos)
                .arg(&pm.cnt)
                .arg(&*bucket_at)
                .arg(&pm.bd)
                .arg(&k32)
                .arg(&p0u)
                .arg(&n32)
                .arg(&mut pm.lens)
                .launch(thread_per(n, 256))?;
        }
        exclusive_scan_with_max_into(
            s,
            k,
            &pm.lens.slice(0..nk * n),
            &mut pm.loff.slice_mut(0..nk * n + 1),
            nk * n,
            scan,
            &mut pm.tot,
        )?;
        #[cfg(feature = "phase-timing")]
        s.synchronize()?;
        pm.loff_host.resize(nk * n + 1, 0);
        {
            let (loff, loff_host) = (&pm.loff, &mut pm.loff_host);
            xfer(xfer_ns, Xfer::D2h, || {
                s.memcpy_dtoh(&loff.slice(0..nk * n + 1), &mut loff_host[..])?;
                s.synchronize()?;
                Ok(())
            })?;
        }
        let lo = &pm.loff_host;
        let share = |j: usize| (lo[j * n] as usize, (lo[(j + 1) * n] - lo[j * n]) as usize);
        let set_offsets = |off: &mut [u32], j: usize, run: u32| {
            for i in 0..n {
                off[p0 + i] = run + lo[j * n + i] - lo[j * n];
            }
        };
        let copy = |j: usize,
                    dst0: u32,
                    with_g: u32,
                    dst: (
            &mut CudaSlice<u64>,
            &mut CudaSlice<u64>,
            &mut CudaSlice<f64>,
            &mut CudaSlice<u64>,
        )|
         -> Result<(), GpuError> {
            let j32 = j as u32;
            // SAFETY: arguments match `k_premerge_copy` in export.cu; the destination holds `dst0` plus block `j`'s share of the batch, and `og` is written only when `with_g`.
            unsafe {
                s.launch_builder(&k.premerge_copy)
                    .arg(&arena.x)
                    .arg(&arena.z)
                    .arg(&arena.coeff)
                    .arg(&arena.g)
                    .arg(&pm.start)
                    .arg(&pm.lens)
                    .arg(&pm.loff)
                    .arg(&j32)
                    .arg(&p0u)
                    .arg(&n32)
                    .arg(&dst0)
                    .arg(&with_g)
                    .arg(dst.0)
                    .arg(dst.1)
                    .arg(dst.2)
                    .arg(dst.3)
                    .launch(warp_per_bucket(n))?;
            }
            Ok(())
        };
        match into {
            MergeInto::Device(blocks, with_g) => {
                for (j, bl) in blocks.iter_mut().enumerate() {
                    let (_, rows) = share(j);
                    set_offsets(&mut bl.offsets, j, running[j]);
                    if rows > 0 {
                        let live = running[j] as usize;
                        bl.reserve_keep(s, live, live + rows, o)?;
                        let DeviceBlock { x, z, c, g, .. } = bl;
                        copy(j, running[j], u32::from(*with_g), (x, z, c, g))?;
                    }
                    running[j] += rows as u32;
                }
            }
            MergeInto::Host(blocks) => {
                let batch = lo[nk * n] as usize;
                if batch > 0 {
                    grow(s, ex, batch * W, o)?;
                    grow(s, ez, batch * W, o)?;
                    grow(s, ec, 2 * batch, o)?;
                    for j in 0..nk {
                        let (at, rows) = share(j);
                        if rows > 0 {
                            copy(
                                j,
                                at as u32,
                                0,
                                (&mut *ex, &mut *ez, &mut *ec, &mut pm.no_g),
                            )?;
                        }
                    }
                }
                #[cfg(feature = "phase-timing")]
                s.synchronize()?;
                xfer(xfer_ns, Xfer::D2h, || {
                    download_shares(s, (ex, ez, ec), pinned, blocks, batch, &share, running)
                })?;
                for (j, bl) in blocks.iter_mut().enumerate() {
                    set_offsets(&mut bl.offsets, j, running[j]);
                    running[j] += share(j).1 as u32;
                }
            }
        }
    }
    Ok(())
}

/// One batch's merged rows, staged contiguously on the device, into each host block's columns after its `running` rows.
fn download_shares<const W: usize>(
    s: &Arc<CudaStream>,
    cols: (&CudaSlice<u64>, &CudaSlice<u64>, &CudaSlice<f64>),
    pinned: &mut HostStaging,
    blocks: &mut [ExchangeBlock<W>],
    batch: usize,
    share: &dyn Fn(usize) -> (usize, usize),
    running: &[u32],
) -> Result<(), GpuError> {
    if batch == 0 {
        return Ok(());
    }
    let (ex, ez, ec) = cols;
    for (j, bl) in blocks.iter_mut().enumerate() {
        let need = running[j] as usize + share(j).1;
        if bl.coeff.len() < need {
            bl.x.resize(need, [0u64; W]);
            bl.z.resize(need, [0u64; W]);
            bl.coeff.resize(need, Complex64::new(0.0, 0.0));
        }
    }
    let staging = export_staging();
    if staging == ExportStaging::Pinned {
        pinned.ensure(batch, W)?;
        s.memcpy_dtoh(&ex.slice(0..batch * W), pinned.x.slice_mut(batch * W))?;
        s.memcpy_dtoh(&ez.slice(0..batch * W), pinned.z.slice_mut(batch * W))?;
        s.memcpy_dtoh(&ec.slice(0..2 * batch), pinned.coeff.slice_mut(2 * batch))?;
        s.synchronize()?;
    }
    for (j, bl) in blocks.iter_mut().enumerate() {
        let (at, rows) = share(j);
        if rows == 0 {
            continue;
        }
        let r0 = running[j] as usize;
        let bx = bl.x[r0..r0 + rows].as_flattened_mut();
        let bz = bl.z[r0..r0 + rows].as_flattened_mut();
        let bc = bytemuck::cast_slice_mut::<Complex64, f64>(&mut bl.coeff[r0..r0 + rows]);
        match staging {
            ExportStaging::Pinned => {
                bx.copy_from_slice(&pinned.x.slice(batch * W)[at * W..(at + rows) * W]);
                bz.copy_from_slice(&pinned.z.slice(batch * W)[at * W..(at + rows) * W]);
                bc.copy_from_slice(&pinned.coeff.slice(2 * batch)[2 * at..2 * (at + rows)]);
            }
            ExportStaging::Pageable => {
                s.memcpy_dtoh(&ex.slice(at * W..(at + rows) * W), bx)?;
                s.memcpy_dtoh(&ez.slice(at * W..(at + rows) * W), bz)?;
                s.memcpy_dtoh(&ec.slice(2 * at..2 * (at + rows)), bc)?;
            }
        }
    }
    s.synchronize()?;
    Ok(())
}

/// The three exported columns into the block's pooled `Vec`s, by the configured [`ExportStaging`].
fn download_block<const W: usize>(
    s: &Arc<CudaStream>,
    rows: usize,
    cols: (&CudaSlice<u64>, &CudaSlice<u64>, &CudaSlice<f64>),
    pinned: &mut HostStaging,
    block: &mut ExchangeBlock<W>,
) -> Result<(), GpuError> {
    let (ex, ez, ec) = cols;
    let bx = block.x[..rows].as_flattened_mut();
    let bz = block.z[..rows].as_flattened_mut();
    let bc = bytemuck::cast_slice_mut::<Complex64, f64>(&mut block.coeff[..rows]);
    match export_staging() {
        ExportStaging::Pageable => {
            s.memcpy_dtoh(&ex.slice(0..rows * W), bx)?;
            s.memcpy_dtoh(&ez.slice(0..rows * W), bz)?;
            s.memcpy_dtoh(&ec.slice(0..2 * rows), bc)?;
            s.synchronize()?;
        }
        ExportStaging::Pinned => {
            pinned.ensure(rows, W)?;
            s.memcpy_dtoh(&ex.slice(0..rows * W), pinned.x.slice_mut(rows * W))?;
            s.memcpy_dtoh(&ez.slice(0..rows * W), pinned.z.slice_mut(rows * W))?;
            s.memcpy_dtoh(&ec.slice(0..2 * rows), pinned.coeff.slice_mut(2 * rows))?;
            s.synchronize()?;
            bx.copy_from_slice(pinned.x.slice(rows * W));
            bz.copy_from_slice(pinned.z.slice(rows * W));
            bc.copy_from_slice(pinned.coeff.slice(2 * rows));
        }
    }
    Ok(())
}

/// Rows the tag's offset field can address in one received segment.
pub(crate) const MAX_RECV_SEGMENT: usize = MAX_BUCKET_LEN;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket::hash::PartitionRows;
    use crate::channel::clifford::Clifford2Q;
    use crate::channel::{Channel, GeneralUnitary2Q};
    use crate::engine::gpu::fingerprint::FingerprintRows;
    use crate::engine::gpu::layer::GpuLayerOptions;
    use crate::engine::partitioned::export::{export_layer, ExportScratch};
    use crate::test_support::{haar_su4_matrix, rand_sum, zz_rotation};

    /// The three channels of the export fixtures: a dense SU(4), a Clifford and a rotation.
    fn fixture_channels<const W: usize>() -> Vec<(&'static str, Box<dyn Channel<W>>)> {
        vec![
            (
                "su4",
                Box::new(GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix())),
            ),
            ("cnot", Box::new(Clifford2Q::cnot(1, 3))),
            ("zz", Box::new(zz_rotation::<W>(0, 2, 0.3))),
        ]
    }

    /// A layer scratch with the sender-side merge on or off.
    fn scratch_with<const W: usize>(dev: &GpuSum<W>, premerge: bool) -> LayerScratch<W> {
        scratch_arena(dev, premerge, crate::engine::gpu::DEFAULT_ARENA_BYTES)
    }

    /// As [`scratch_with`] under an arena of `arena_bytes`; one byte batches the merge by the record cap.
    fn scratch_arena<const W: usize>(
        dev: &GpuSum<W>,
        premerge: bool,
        arena_bytes: usize,
    ) -> LayerScratch<W> {
        let opts = GpuLayerOptions {
            premerge,
            arena_bytes,
            ..GpuLayerOptions::default()
        };
        LayerScratch::new(dev, opts).expect("scratch")
    }

    /// K10's blocks against `export_layer`'s on every partition: header, offsets and columns bitwise, since a freshly uploaded sum keeps the host's within-bucket order.
    fn export_matches_host<const W: usize>(num_qubits: usize, n: usize, seed: u64, pbits: u8) {
        let input = rand_sum::<W>(n, num_qubits, seed);
        let rows = PartitionRows::<W>::from_seed(num_qubits, pbits, seed ^ 0x77);
        let size = rows.num_partitions() as u32;
        let mut remote_layers = 0;
        let mut multi_block_payloads = 0;
        for (name, ch) in &fixture_channels::<W>() {
            for rank in 0..size {
                let local = input.filter_partition(&rows, rank);
                let prep = ch.prepare(local.hash(), false).expect("prepared");
                let plan = PartitionPlan::new(&prep, &rows, rank);
                if !plan.has_remote() {
                    continue;
                }
                remote_layers += 1;
                let nb = local.num_buckets();
                let mut map = ChunkMap::default();
                map.rebuild(
                    &Gf2Span::new(&plan.local_bucket_deltas, local.hash().bits()),
                    nb,
                    1,
                );
                let (want, want_counts) = export_layer(
                    &local,
                    &prep,
                    &plan,
                    size,
                    &map,
                    &mut ExportScratch::default(),
                );
                let dev = GpuSum::from_host(&local, 0).expect("upload");
                let mut scratch = scratch_with(&dev, false);
                let fp = FingerprintRows::<W>::new(dev.hash().seed());
                let table = DevicePrepared::new(&prep, dev.hash(), &fp, &plan.remote);
                scratch.upload_table(&dev, &table).expect("table");
                scratch.count_local(&dev, &table).expect("count");
                let (got, got_counts) =
                    export_blocks(&dev, &table, &plan, &mut scratch, size).expect("export");
                dev.stream.synchronize().expect("sync");
                let what = format!("W={W} {name} rank {rank}");
                assert_eq!(got_counts.rows_sent, want_counts.rows_to, "{what}: rows");
                assert_eq!(got_counts.bytes_sent, want_counts.bytes_to, "{what}: bytes");
                assert_eq!(got.len(), want.len());
                for (q, (g, w)) in got.iter().zip(&want).enumerate() {
                    match (g, w) {
                        (None, None) => {}
                        (Some(g), Some(w)) => {
                            if g.blocks.len() >= 2 {
                                multi_block_payloads += 1;
                            }
                            assert_eq!(g.blocks.len(), w.blocks.len(), "{what}: blocks to {q}");
                            for (j, (gb, wb)) in g.blocks.iter().zip(&w.blocks).enumerate() {
                                assert_eq!(gb.header, wb.header, "{what}: header {j} to {q}");
                                assert_eq!(gb.offsets, wb.offsets, "{what}: offsets {j} to {q}");
                                assert_eq!(gb.cols(), wb.cols(), "{what}: columns {j} to {q}");
                            }
                        }
                        _ => panic!("{what}: payload presence to {q} differs"),
                    }
                }
            }
        }
        assert!(remote_layers > 0, "the fixture must export something");
        assert!(
            multi_block_payloads > 0,
            "the SU(4) layer must ship several remote deltas to one partner"
        );
    }

    /// A device block's columns on the host, `g` included.
    fn download<const W: usize>(
        s: &Arc<CudaStream>,
        block: &DeviceBlock<W>,
    ) -> (Vec<u64>, Vec<u64>, Vec<f64>, Vec<u64>) {
        let n = block.rows();
        let x = s.clone_dtoh(&block.x.slice(0..n * W)).unwrap();
        let z = s.clone_dtoh(&block.z.slice(0..n * W)).unwrap();
        let c = s.clone_dtoh(&block.c.slice(0..2 * n)).unwrap();
        let g = s.clone_dtoh(&block.g.slice(0..n)).unwrap();
        s.synchronize().unwrap();
        (x, z, c, g)
    }

    /// The device payloads equal the host blocks bitwise, their `g` is the host fingerprint, and adopting every block lands the concatenation in `recv_*` with the offsets and bases the host path would compute.
    fn device_export_matches_host_and_adopts<const W: usize>(
        num_qubits: usize,
        n: usize,
        seed: u64,
        pbits: u8,
    ) {
        let input = rand_sum::<W>(n, num_qubits, seed);
        let rows = PartitionRows::<W>::from_seed(num_qubits, pbits, seed ^ 0x77);
        let size = rows.num_partitions() as u32;
        let mut adopted_blocks = 0usize;
        let mut multi_block_payloads = 0;
        for (name, ch) in &fixture_channels::<W>() {
            for rank in 0..size {
                let local = input.filter_partition(&rows, rank);
                let prep = ch.prepare(local.hash(), false).expect("prepared");
                let plan = PartitionPlan::new(&prep, &rows, rank);
                if !plan.has_remote() {
                    continue;
                }
                let nb = local.num_buckets();
                let mut map = ChunkMap::default();
                map.rebuild(
                    &Gf2Span::new(&plan.local_bucket_deltas, local.hash().bits()),
                    nb,
                    1,
                );
                let (want, want_counts) = export_layer(
                    &local,
                    &prep,
                    &plan,
                    size,
                    &map,
                    &mut ExportScratch::default(),
                );
                let dev = GpuSum::from_host(&local, 0).expect("upload");
                let mut scratch = scratch_with(&dev, false);
                let fp = FingerprintRows::<W>::new(dev.hash().seed());
                let table = DevicePrepared::new(&prep, dev.hash(), &fp, &plan.remote);
                scratch.upload_table(&dev, &table).expect("table");
                scratch.count_local(&dev, &table).expect("count");
                let (got, got_counts) =
                    export_blocks_device(&dev, &table, &plan, &mut scratch, size, true)
                        .expect("export");
                let what = format!("W={W} {name} rank {rank}");
                assert_eq!(got_counts.rows_sent, want_counts.rows_to, "{what}: rows");
                let mut all_x = Vec::new();
                let mut all_z = Vec::new();
                let mut all_c = Vec::new();
                let mut all_g = Vec::new();
                let mut all_off = Vec::new();
                let mut bases = Vec::new();
                let mut blocks: Vec<&DeviceBlock<W>> = Vec::new();
                for (q, (g, w)) in got.iter().zip(&want).enumerate() {
                    match (g, w) {
                        (None, None) => {}
                        (Some(g), Some(w)) => {
                            assert_eq!(g.device, Some(0));
                            if g.blocks.len() >= 2 {
                                multi_block_payloads += 1;
                            }
                            assert_eq!(g.blocks.len(), w.blocks.len(), "{what}: blocks to {q}");
                            for (j, (gb, wb)) in g.blocks.iter().zip(&w.blocks).enumerate() {
                                assert_eq!(gb.header, wb.header, "{what}: header {j} to {q}");
                                assert_eq!(gb.offsets, wb.offsets, "{what}: offsets {j} to {q}");
                                let (x, z, c, gfp) = download(&dev.stream, gb);
                                let (wx, wz, wc) = wb.cols();
                                assert_eq!(x, wx.as_flattened(), "{what}: x {j} to {q}");
                                assert_eq!(z, wz.as_flattened(), "{what}: z {j} to {q}");
                                assert_eq!(
                                    c,
                                    bytemuck::cast_slice::<Complex64, f64>(wc),
                                    "{what}: coeff {j} to {q}"
                                );
                                let want_g: Vec<u64> = wx
                                    .iter()
                                    .zip(wz)
                                    .map(|(x, z)| fp.fingerprint(x, z))
                                    .collect();
                                assert_eq!(gfp, want_g, "{what}: fingerprints {j} to {q}");
                                bases.push(all_g.len() as u32);
                                all_off.extend_from_slice(&gb.offsets);
                                all_x.extend_from_slice(&x);
                                all_z.extend_from_slice(&z);
                                all_c.extend_from_slice(&c);
                                all_g.extend_from_slice(&gfp);
                                blocks.push(gb);
                            }
                        }
                        _ => panic!("{what}: payload presence to {q} differs"),
                    }
                }
                let total = adopt_blocks(&dev, &mut scratch.export, &blocks, &mut scratch.xfer_ns)
                    .expect("adopt") as usize;
                assert_eq!(total, all_g.len(), "{what}: rows adopted");
                assert_eq!(scratch.export.recv_rows, total);
                let s = &dev.stream;
                let e = &scratch.export;
                let off = s.clone_dtoh(&e.recv_off.slice(0..all_off.len())).unwrap();
                let base = s.clone_dtoh(&e.recv_base.slice(0..bases.len())).unwrap();
                let x = s.clone_dtoh(&e.recv_x.slice(0..total * W)).unwrap();
                let z = s.clone_dtoh(&e.recv_z.slice(0..total * W)).unwrap();
                let c = s.clone_dtoh(&e.recv_c.slice(0..2 * total)).unwrap();
                let g = s.clone_dtoh(&e.recv_g.slice(0..total)).unwrap();
                s.synchronize().unwrap();
                assert_eq!(off, all_off, "{what}: adopted offsets");
                assert_eq!(base, bases, "{what}: adopted bases");
                assert_eq!((x, z), (all_x, all_z), "{what}: adopted keys");
                assert_eq!(c, all_c, "{what}: adopted coefficients");
                assert_eq!(g, all_g, "{what}: adopted fingerprints");
                adopted_blocks += blocks.len();
                got.into_iter().flatten().for_each(payload::recycle);
            }
        }
        assert!(adopted_blocks > 0, "the fixture must export something");
        assert!(
            multi_block_payloads > 0,
            "the SU(4) layer must ship several remote deltas to one partner"
        );
    }

    /// A partition with no terms still ships one empty block per remote delta, bitwise the host's.
    #[test]
    fn an_empty_partition_ships_empty_blocks() {
        crate::require_cuda!();
        let nq = 12;
        let seed = 0xE55u64;
        let hash = crate::bucket::hash::Gf2Hash::<1>::new(nq, 3, seed);
        let local = crate::PauliSum::<1>::empty_with_hash(nq, hash.clone());
        let rows = PartitionRows::<1>::from_seed(nq, 1, seed ^ 0x77);
        let ch = GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix());
        let prep = ch.prepare(&hash, false).expect("prepared");
        let plan = PartitionPlan::new(&prep, &rows, 0);
        assert!(plan.has_remote(), "fixture: the SU(4) must cross");
        let mut map = ChunkMap::default();
        map.rebuild(&Gf2Span::new(&plan.local_bucket_deltas, 3), 8, 1);
        let (want, _) = export_layer(&local, &prep, &plan, 2, &map, &mut ExportScratch::default());
        let dev = GpuSum::from_host(&local, 0).expect("upload");
        let mut scratch = LayerScratch::new(&dev, GpuLayerOptions::default()).expect("scratch");
        let fp = FingerprintRows::<1>::new(seed);
        let table = DevicePrepared::new(&prep, &hash, &fp, &plan.remote);
        scratch.upload_table(&dev, &table).expect("table");
        scratch.count_local(&dev, &table).expect("count");
        let (got, counts) = export_blocks(&dev, &table, &plan, &mut scratch, 2).expect("export");
        let payload = got[1].as_ref().expect("a payload for the partner");
        assert_eq!(payload.blocks.len(), plan.remote.len());
        for block in &payload.blocks {
            assert_eq!(block.rows(), 0);
            assert!(block.offsets.iter().all(|&o| o == 0));
            assert_eq!(block.offsets.len(), 9);
        }
        assert_eq!(payload, want[1].as_ref().unwrap());
        assert_eq!(counts.rows_sent, vec![0, 0]);
        let (got, counts) = export_blocks_device(&dev, &table, &plan, &mut scratch, 2, true)
            .expect("device export");
        let payload = got[1].as_ref().expect("a device payload for the partner");
        assert_eq!(payload.blocks.len(), plan.remote.len());
        for (block, want) in payload.blocks.iter().zip(&want[1].as_ref().unwrap().blocks) {
            assert_eq!(block.header, want.header);
            assert_eq!(block.offsets, want.offsets);
        }
        assert_eq!(counts.rows_sent, vec![0, 0]);
    }

    #[test]
    fn export_blocks_match_export_layer_bitwise_w1() {
        crate::require_cuda!();
        export_matches_host::<1>(12, 6000, 0xE7, 1);
        export_matches_host::<1>(12, 6000, 0xE8, 2);
    }

    #[test]
    fn export_blocks_match_export_layer_bitwise_w2() {
        crate::require_cuda!();
        export_matches_host::<2>(100, 5000, 0xE9, 1);
        export_matches_host::<2>(100, 5000, 0xEA, 2);
    }

    #[test]
    fn device_payloads_match_export_layer_bitwise_and_adopt_w1() {
        crate::require_cuda!();
        device_export_matches_host_and_adopts::<1>(12, 6000, 0xE7, 1);
        device_export_matches_host_and_adopts::<1>(12, 6000, 0xE8, 2);
    }

    #[test]
    fn device_payloads_match_export_layer_bitwise_and_adopt_w2() {
        crate::require_cuda!();
        device_export_matches_host_and_adopts::<2>(100, 5000, 0xE9, 1);
        device_export_matches_host_and_adopts::<2>(100, 5000, 0xEA, 2);
    }

    /// Per destination position, one partner's rows keyed by `(x, z)` with their coefficients summed across the partner's blocks, and the largest number of times one key occurs.
    type PositionSums<const W: usize> =
        Vec<std::collections::BTreeMap<([u64; W], [u64; W]), Complex64>>;

    /// One block's `(offsets, x, z, coeff)`, live rows only.
    type BlockCols<const W: usize> = (Vec<u32>, Vec<[u64; W]>, Vec<[u64; W]>, Vec<Complex64>);

    /// Occurrences per key at each position.
    type PositionCounts<const W: usize> =
        Vec<std::collections::BTreeMap<([u64; W], [u64; W]), usize>>;

    fn position_sums<const W: usize>(
        blocks: &[BlockCols<W>],
        b: usize,
    ) -> (PositionSums<W>, usize) {
        let mut out: PositionSums<W> = vec![Default::default(); b];
        let mut seen: PositionCounts<W> = vec![Default::default(); b];
        let mut most = 0;
        for (off, x, z, c) in blocks {
            for p in 0..b {
                for i in off[p] as usize..off[p + 1] as usize {
                    *out[p]
                        .entry((x[i], z[i]))
                        .or_insert(Complex64::new(0.0, 0.0)) += c[i];
                    let n = seen[p].entry((x[i], z[i])).or_insert(0);
                    *n += 1;
                    most = most.max(*n);
                }
            }
        }
        (out, most)
    }

    /// Two per-position key sums agree to `tol`, a key absent on one side counting as zero.
    fn assert_position_sums_close<const W: usize>(
        got: &PositionSums<W>,
        want: &PositionSums<W>,
        tol: f64,
        what: &str,
    ) {
        let zero = Complex64::new(0.0, 0.0);
        for (p, (g, w)) in got.iter().zip(want).enumerate() {
            for (k, gc) in g {
                let wc = w.get(k).copied().unwrap_or(zero);
                assert!(
                    (gc - wc).norm() <= tol,
                    "{what}: position {p} key {k:?}: {gc} vs {wc}"
                );
            }
            for (k, wc) in w {
                if !g.contains_key(k) {
                    assert!(
                        wc.norm() <= tol,
                        "{what}: position {p} lost key {k:?} ({wc})"
                    );
                }
            }
        }
    }

    /// Host blocks as `(offsets, x, z, coeff)`, live rows only.
    fn host_block_cols<const W: usize>(b: &ExchangeBlock<W>) -> BlockCols<W> {
        let (x, z, c) = b.cols();
        (b.offsets.clone(), x.to_vec(), z.to_vec(), c.to_vec())
    }

    /// Device blocks the same way, checking every `g` is the host fingerprint of its key.
    fn device_block_cols<const W: usize>(
        s: &Arc<CudaStream>,
        b: &DeviceBlock<W>,
        fp: &FingerprintRows<W>,
        fp_mask: u64,
        what: &str,
    ) -> BlockCols<W> {
        let (x, z, c, g) = download(s, b);
        let x: Vec<[u64; W]> = x.chunks(W).map(|w| w.try_into().unwrap()).collect();
        let z: Vec<[u64; W]> = z.chunks(W).map(|w| w.try_into().unwrap()).collect();
        let c: Vec<Complex64> = c.chunks(2).map(|p| Complex64::new(p[0], p[1])).collect();
        for i in 0..x.len() {
            assert_eq!(
                g[i],
                fp.fingerprint(&x[i], &z[i]) & fp_mask,
                "{what}: fingerprint {i}"
            );
        }
        (b.offsets.clone(), x, z, c)
    }

    /// With the sender-side merge on, both payload forms carry per position what `export_layer` carries summed by key, no key twice across one partner's blocks, and no segment longer than its unmerged one.
    /// Returns `(rows sent, unmerged rows, fallback blocks)` per channel name, summed over ranks; `nvrtc` and `fp_mask` are the collision hook's options and the fingerprint mask they imply.
    fn premerge_matches_host_by_key<const W: usize>(
        input: &crate::PauliSum<W>,
        rows: &PartitionRows<W>,
        channels: &[(&'static str, Box<dyn Channel<W>>)],
        arena_bytes: usize,
        (nvrtc, fp_mask): (&[String], u64),
    ) -> Vec<(&'static str, u64, u64, (u32, u32))> {
        let size = rows.num_partitions() as u32;
        let mut totals = Vec::new();
        for (name, ch) in channels {
            let (mut sent, mut unmerged, mut fallbacks) = (0u64, 0u64, (0u32, 0u32));
            for rank in 0..size {
                let local = input.filter_partition(rows, rank);
                let prep = ch.prepare(local.hash(), false).expect("prepared");
                let plan = PartitionPlan::new(&prep, rows, rank);
                if !plan.has_remote() {
                    continue;
                }
                let nb = local.num_buckets();
                let mut map = ChunkMap::default();
                map.rebuild(
                    &Gf2Span::new(&plan.local_bucket_deltas, local.hash().bits()),
                    nb,
                    1,
                );
                let (want, want_counts) = export_layer(
                    &local,
                    &prep,
                    &plan,
                    size,
                    &map,
                    &mut ExportScratch::default(),
                );
                let dev = GpuSum::from_host_with_options(&local, 0, nvrtc).expect("upload");
                let fp = FingerprintRows::<W>::new(dev.hash().seed());
                let table = DevicePrepared::new(&prep, dev.hash(), &fp, &plan.remote);
                for device in [false, true] {
                    let mut scratch = scratch_arena(&dev, true, arena_bytes);
                    scratch.upload_table(&dev, &table).expect("table");
                    scratch.count_local(&dev, &table).expect("count");
                    let what =
                        format!("W={W} {name} rank {rank} device={device} arena={arena_bytes}");
                    let mut got: Vec<Option<Vec<_>>> = vec![None; size as usize];
                    let counts = if device {
                        let (g, c) =
                            export_blocks_device(&dev, &table, &plan, &mut scratch, size, true)
                                .expect("device export");
                        for (q, p) in g.iter().enumerate() {
                            got[q] = p.as_ref().map(|p| {
                                p.blocks
                                    .iter()
                                    .map(|b| {
                                        assert_eq!(b.header.num_buckets as usize, nb);
                                        assert_eq!(b.header.rows as usize, b.offsets[nb] as usize);
                                        device_block_cols(&dev.stream, b, &fp, fp_mask, &what)
                                    })
                                    .collect()
                            });
                        }
                        g.into_iter().flatten().for_each(payload::recycle);
                        c
                    } else {
                        let (g, c) =
                            export_blocks(&dev, &table, &plan, &mut scratch, size).expect("export");
                        #[cfg(debug_assertions)]
                        crate::engine::partitioned::export::debug_assert_exported_partitions(
                            &g, rows,
                        );
                        for (q, p) in g.iter().enumerate() {
                            got[q] = p
                                .as_ref()
                                .map(|p| p.blocks.iter().map(host_block_cols).collect());
                        }
                        c
                    };
                    for (q, w) in want.iter().enumerate() {
                        let (Some(w), Some(g)) = (w, &got[q]) else {
                            assert!(w.is_none() && got[q].is_none(), "{what}: presence to {q}");
                            continue;
                        };
                        assert_eq!(g.len(), w.blocks.len(), "{what}: blocks to {q}");
                        for (j, (gb, wb)) in g.iter().zip(&w.blocks).enumerate() {
                            assert_eq!(gb.0.len(), nb + 1, "{what}: offsets {j} to {q}");
                            for p in 0..nb {
                                assert!(
                                    gb.0[p + 1] - gb.0[p] <= wb.offsets[p + 1] - wb.offsets[p],
                                    "{what}: block {j} to {q} grew at position {p}"
                                );
                            }
                        }
                        let wcols: Vec<_> = w.blocks.iter().map(host_block_cols).collect();
                        let (want_sums, _) = position_sums(&wcols, nb);
                        let (got_sums, most) = position_sums(g, nb);
                        assert_position_sums_close(&got_sums, &want_sums, 1e-12, &what);
                        if w.blocks.len() >= 2 && matches!(*name, "su4") {
                            assert_eq!(most, 1, "{what}: a key twice across the blocks to {q}");
                        }
                    }
                    assert_eq!(
                        counts.rows_sent.iter().sum::<u64>(),
                        got.iter()
                            .flatten()
                            .flatten()
                            .map(|b| b.1.len() as u64)
                            .sum::<u64>(),
                        "{what}: counted rows"
                    );
                    if !device {
                        sent += counts.rows_sent.iter().sum::<u64>();
                        unmerged += want_counts.rows_to.iter().sum::<u64>();
                        fallbacks.0 += scratch.counters.fallback_hi;
                        fallbacks.1 += scratch.counters.fallback_key;
                    }
                }
            }
            totals.push((*name, sent, unmerged, fallbacks));
        }
        totals
    }

    fn premerge_fixture<const W: usize>(num_qubits: usize, n: usize, seed: u64, pbits: u8) {
        // A dense sum on qubits 0 and 1: every support pattern occurs for each remaining key, so the SU(4)'s remote rows collide.
        let base = rand_sum::<W>(n / 16, num_qubits, seed);
        let mut acc = crate::accumulator::BuildAccumulator::<W>::new(num_qubits);
        for (x, z, c) in base.iter() {
            for s in 0..16u64 {
                let (mut x, mut z) = (*x, *z);
                x[0] = (x[0] & !0b11) | (s & 0b11);
                z[0] = (z[0] & !0b11) | (s >> 2);
                acc.add_term(
                    crate::pauli_string::PauliString::<W> { x, z },
                    crate::phase::Phase::ONE,
                    c * (1.0 + s as f64),
                );
            }
        }
        let input = acc.finalize();
        let rows = PartitionRows::<W>::from_seed(num_qubits, pbits, seed ^ 0x77);
        for arena in [crate::engine::gpu::DEFAULT_ARENA_BYTES, 1] {
            for (name, sent, unmerged, fallbacks) in
                premerge_matches_host_by_key(&input, &rows, &fixture_channels(), arena, (&[], !0))
            {
                check_shrink(name, sent, unmerged, pbits);
                assert_eq!(
                    fallbacks,
                    (0, 0),
                    "{name}: a 64-bit fingerprint needs no fallback"
                );
            }
        }
        // Every merge block takes a fallback: the `g_hi32` passes with the low word cleared, the full-key sort with no fingerprint at all.
        for (opt, mask, full_key) in [
            ("-DFP_ZERO_LO", !0xFFFF_FFFFu64, false),
            ("-DFP_BITS=0", 0, true),
        ] {
            let nvrtc = [opt.to_string()];
            for (name, sent, unmerged, fallbacks) in premerge_matches_host_by_key(
                &input,
                &rows,
                &fixture_channels(),
                crate::engine::gpu::DEFAULT_ARENA_BYTES,
                (&nvrtc, mask),
            ) {
                check_shrink(name, sent, unmerged, pbits);
                if name == "su4" {
                    let taken = if full_key { fallbacks.1 } else { fallbacks.0 };
                    assert!(taken > 0, "{opt}: su4 merged without the expected fallback");
                }
            }
        }
    }

    /// The SU(4) ships under half its unmerged rows; the Clifford and the rotation have nothing to merge.
    fn check_shrink(name: &str, sent: u64, unmerged: u64, pbits: u8) {
        if name == "su4" {
            assert!(
                sent * 2 < unmerged,
                "P={}: su4 sent {sent} of {unmerged} rows",
                1 << pbits
            );
        } else {
            assert_eq!(sent, unmerged, "{name}: nothing to merge");
        }
    }

    #[test]
    fn premerged_blocks_match_export_layer_by_key_and_shrink_w1() {
        crate::require_cuda!();
        premerge_fixture::<1>(12, 6400, 0xF7, 1);
        premerge_fixture::<1>(12, 6400, 0xF8, 2);
    }

    #[test]
    fn premerged_blocks_match_export_layer_by_key_and_shrink_w2() {
        crate::require_cuda!();
        premerge_fixture::<2>(100, 6400, 0xF9, 1);
        premerge_fixture::<2>(100, 6400, 0xFA, 2);
    }

    /// Two terms whose remote rows to one key cancel exactly: the merged export drops the key, the unmerged one ships both rows.
    /// Unit coefficients against equal-magnitude amplitudes keep every product exact whatever the kernel's FMA contraction.
    #[test]
    fn exactly_cancelling_remote_rows_are_not_shipped() {
        crate::require_cuda!();
        use crate::channel::prepared::Prepared;
        use crate::test_support::sqrt_swap_matrix;
        let nq = 8;
        let hash = crate::bucket::hash::Gf2Hash::<1>::new(nq, 2, 0xCA);
        // The row reads x on qubit 0, so a delta flipping qubit 0's x-bit is remote.
        let rows = PartitionRows::<1>::from_rows(nq, vec![[0b1u64]], vec![[0u64]]);
        let ch = GeneralUnitary2Q::from_matrix(0, 1, sqrt_swap_matrix());
        let prep = ch.prepare(&hash, false).expect("prepared");
        let Prepared::Local(ptm) = &prep else {
            unreachable!()
        };
        let plan = PartitionPlan::new(&prep, &rows, 0);
        let d = ptm.deltas();
        let zero = Complex64::new(0.0, 0.0);
        let mut pick = None;
        'outer: for (i, a) in plan.remote.iter().enumerate() {
            for b in &plan.remote[i + 1..] {
                let (da, db) = (&d[a.entry], &d[b.entry]);
                for sa in (0..16usize).step_by(2) {
                    let sb = sa ^ (da.local_delta ^ db.local_delta) as usize;
                    let (aa, ab) = (da.amp[sa], db.amp[sb]);
                    if sb & 1 == 0 && aa != zero && (aa == ab || aa == -ab) {
                        pick = Some((a.entry, sa, b.entry, sb));
                        break 'outer;
                    }
                }
            }
        }
        let (ea, sa, eb, sb) =
            pick.expect("sqrt(SWAP) has two colliding remote entries of equal magnitude");
        let key = |s: usize| {
            let x = (s & 1) as u64 | (((s >> 2) & 1) as u64) << 1 | 1 << 5;
            let z = ((s >> 1) & 1) as u64 | (((s >> 3) & 1) as u64) << 1;
            crate::pauli_string::PauliString::<1> { x: [x], z: [z] }
        };
        let (aa, ab) = (d[ea].amp[sa], d[eb].amp[sb]);
        let cb = if aa == ab { -1.0 } else { 1.0 };
        let mut acc = crate::accumulator::BuildAccumulator::<1>::new(nq);
        acc.add_term(key(sa), crate::phase::Phase::ONE, Complex64::new(1.0, 0.0));
        acc.add_term(key(sb), crate::phase::Phase::ONE, Complex64::new(cb, 0.0));
        let local = acc.finalize().with_hash(hash.clone());
        let (ka, kb) = (key(sa), key(sb));
        assert_eq!(rows.partition_of(&ka.x, &ka.z), 0);
        assert_eq!(rows.partition_of(&kb.x, &kb.z), 0);
        let (tx, tz) = (ka.x[0] ^ d[ea].mask_x[0], ka.z[0] ^ d[ea].mask_z[0]);
        assert_eq!(
            (tx, tz),
            (kb.x[0] ^ d[eb].mask_x[0], kb.z[0] ^ d[eb].mask_z[0])
        );
        let dev = GpuSum::from_host(&local, 0).expect("upload");
        let fp = FingerprintRows::<1>::new(hash.seed());
        let table = DevicePrepared::new(&prep, &hash, &fp, &plan.remote);
        let shipped = |premerge: bool| {
            let mut scratch = scratch_with(&dev, premerge);
            scratch.upload_table(&dev, &table).expect("table");
            scratch.count_local(&dev, &table).expect("count");
            let (got, counts) =
                export_blocks(&dev, &table, &plan, &mut scratch, 2).expect("export");
            let keys: Vec<(u64, u64)> = got[1]
                .as_ref()
                .unwrap()
                .blocks
                .iter()
                .flat_map(|b| {
                    let (x, z, _) = b.cols();
                    x.iter()
                        .zip(z)
                        .map(|(x, z)| (x[0], z[0]))
                        .collect::<Vec<_>>()
                })
                .collect();
            (keys, counts.rows_sent[1])
        };
        let (plain, plain_rows) = shipped(false);
        assert_eq!(plain.iter().filter(|k| **k == (tx, tz)).count(), 2);
        let (merged, merged_rows) = shipped(true);
        assert!(!merged.contains(&(tx, tz)), "the cancelled key travelled");
        assert!(
            merged_rows + 2 <= plain_rows,
            "{merged_rows} vs {plain_rows}"
        );
    }

    /// A source bucket longer than [`MAX_BUCKET_LEN`] forces `premerge_groups` to clear every group (export.rs's tag-overflow guard), so the sender falls back to the unmerged, bitwise-`export_layer`-matching export.
    /// A single bucket (`bits = 0`) with enough rows still exceeds the limit after `filter_partition` halves it.
    #[test]
    fn oversize_source_bucket_falls_back_to_unmerged_export() {
        crate::require_cuda!();
        let nq = 12;
        let seed = 0xB16u64;
        let hash = crate::bucket::hash::Gf2Hash::<1>::new(nq, 0, seed);
        let input = rand_sum::<1>(12_000, nq, seed).with_hash(hash);
        assert_eq!(input.num_buckets(), 1);
        let rows = PartitionRows::<1>::from_seed(nq, 1, seed ^ 0x77);
        let size = rows.num_partitions() as u32;
        let ch = GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix());
        let mut remote_layers = 0;
        for rank in 0..size {
            let local = input.filter_partition(&rows, rank);
            assert!(
                local.bucket(0).2.len() > MAX_BUCKET_LEN,
                "fixture: the single bucket must exceed the tag's offset field"
            );
            let prep = ch.prepare(local.hash(), false).expect("prepared");
            let plan = PartitionPlan::new(&prep, &rows, rank);
            if !plan.has_remote() {
                continue;
            }
            remote_layers += 1;
            let nb = local.num_buckets();
            let mut map = ChunkMap::default();
            map.rebuild(
                &Gf2Span::new(&plan.local_bucket_deltas, local.hash().bits()),
                nb,
                1,
            );
            let (want, want_counts) = export_layer(
                &local,
                &prep,
                &plan,
                size,
                &map,
                &mut ExportScratch::default(),
            );
            let dev = GpuSum::from_host(&local, 0).expect("upload");
            let mut scratch = scratch_with(&dev, true);
            let fp = FingerprintRows::<1>::new(dev.hash().seed());
            let table = DevicePrepared::new(&prep, dev.hash(), &fp, &plan.remote);
            scratch.upload_table(&dev, &table).expect("table");
            scratch.count_local(&dev, &table).expect("count");
            let (got, got_counts) =
                export_blocks(&dev, &table, &plan, &mut scratch, size).expect("export");
            dev.stream.synchronize().expect("sync");
            let what = format!("rank {rank}");
            assert_eq!(
                got_counts.rows_sent, want_counts.rows_to,
                "{what}: the oversize bucket must skip the merge, not shrink it"
            );
            assert_eq!(
                scratch.counters.rows_premerged, 0,
                "{what}: no rows were premerged"
            );
            for (q, (g, w)) in got.iter().zip(&want).enumerate() {
                match (g, w) {
                    (None, None) => {}
                    (Some(g), Some(w)) => {
                        assert_eq!(g.blocks.len(), w.blocks.len(), "{what}: blocks to {q}");
                        for (j, (gb, wb)) in g.blocks.iter().zip(&w.blocks).enumerate() {
                            assert_eq!(gb.header, wb.header, "{what}: header {j} to {q}");
                            assert_eq!(gb.offsets, wb.offsets, "{what}: offsets {j} to {q}");
                            assert_eq!(gb.cols(), wb.cols(), "{what}: columns {j} to {q}");
                        }
                    }
                    _ => panic!("{what}: payload presence to {q} differs"),
                }
            }
        }
        assert!(remote_layers > 0, "the fixture must export something");
    }

    /// A merge position whose selected entries' combined record count exceeds the fused kernel's record cap: `premerge_partner` returns `Ok(None)` without writing, and the sender falls back to the unmerged, bitwise-`export_layer`-matching export for that partner.
    /// `-DTEST_SHARED_LIMIT` (module.rs's test hook) shrinks the loaded record cap to its smallest variant so a modest single bucket exceeds it, while staying under [`MAX_BUCKET_LEN`] so the tag-overflow guard does not fire first.
    #[test]
    fn oversize_merge_position_falls_back_to_unmerged_export() {
        crate::require_cuda!();
        let nq = 12;
        let seed = 0xCA9u64;
        let hash = crate::bucket::hash::Gf2Hash::<1>::new(nq, 0, seed);
        let input = rand_sum::<1>(6000, nq, seed).with_hash(hash);
        assert_eq!(input.num_buckets(), 1);
        let rows = PartitionRows::<1>::from_seed(nq, 1, seed ^ 0x77);
        let size = rows.num_partitions() as u32;
        let ch = GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix());
        let small_cap = ["-DTEST_SHARED_LIMIT=17000".to_string()];
        let mut remote_layers = 0;
        for rank in 0..size {
            let local = input.filter_partition(&rows, rank);
            assert!(
                local.bucket(0).2.len() <= MAX_BUCKET_LEN,
                "fixture: the bucket must stay under the tag limit so only the record cap trips"
            );
            let prep = ch.prepare(local.hash(), false).expect("prepared");
            let plan = PartitionPlan::new(&prep, &rows, rank);
            if !plan.has_remote() {
                continue;
            }
            remote_layers += 1;
            let nb = local.num_buckets();
            let mut map = ChunkMap::default();
            map.rebuild(
                &Gf2Span::new(&plan.local_bucket_deltas, local.hash().bits()),
                nb,
                1,
            );
            let (want, want_counts) = export_layer(
                &local,
                &prep,
                &plan,
                size,
                &map,
                &mut ExportScratch::default(),
            );
            let dev = GpuSum::from_host_with_options(&local, 0, &small_cap).expect("upload");
            let mut scratch = scratch_with(&dev, true);
            let fp = FingerprintRows::<1>::new(dev.hash().seed());
            let table = DevicePrepared::new(&prep, dev.hash(), &fp, &plan.remote);
            scratch.upload_table(&dev, &table).expect("table");
            scratch.count_local(&dev, &table).expect("count");
            let (got, got_counts) =
                export_blocks(&dev, &table, &plan, &mut scratch, size).expect("export");
            dev.stream.synchronize().expect("sync");
            let what = format!("rank {rank}");
            assert_eq!(
                got_counts.rows_sent, want_counts.rows_to,
                "{what}: the oversize merge position must skip the merge, not shrink it"
            );
            assert_eq!(
                scratch.counters.rows_premerged, 0,
                "{what}: no rows were premerged"
            );
            for (q, (g, w)) in got.iter().zip(&want).enumerate() {
                match (g, w) {
                    (None, None) => {}
                    (Some(g), Some(w)) => {
                        assert_eq!(g.blocks.len(), w.blocks.len(), "{what}: blocks to {q}");
                        for (j, (gb, wb)) in g.blocks.iter().zip(&w.blocks).enumerate() {
                            assert_eq!(gb.header, wb.header, "{what}: header {j} to {q}");
                            assert_eq!(gb.offsets, wb.offsets, "{what}: offsets {j} to {q}");
                            assert_eq!(gb.cols(), wb.cols(), "{what}: columns {j} to {q}");
                        }
                    }
                    _ => panic!("{what}: payload presence to {q} differs"),
                }
            }
        }
        assert!(remote_layers > 0, "the fixture must export something");
    }

    /// A partner group whose `entries × buckets` product exceeds [`SCAN_LIMIT`]: `premerge_groups` clears it before ever touching K3, so the sender falls back to the unmerged, bitwise-`export_layer`-matching export.
    /// `B_MAX_BITS` buckets (the largest a hash can address) times the SU(4)'s eight-entry group already clears `SCAN_LIMIT`; the terms themselves can be a small dense-collision fixture, since the guard reads only entry count and bucket count.
    #[test]
    fn oversize_partner_group_falls_back_to_unmerged_export() {
        crate::require_cuda!();
        let nq = 32;
        let seed = 0xB5CAu64;
        let base = rand_sum::<1>(400, nq, seed);
        let mut acc = crate::accumulator::BuildAccumulator::<1>::new(nq);
        for (x, z, c) in base.iter() {
            for s in 0..16u64 {
                let (mut x, mut z) = (*x, *z);
                x[0] = (x[0] & !0b11) | (s & 0b11);
                z[0] = (z[0] & !0b11) | (s >> 2);
                acc.add_term(
                    crate::pauli_string::PauliString::<1> { x, z },
                    crate::phase::Phase::ONE,
                    c * (1.0 + s as f64),
                );
            }
        }
        let hash =
            crate::bucket::hash::Gf2Hash::<1>::new(nq, crate::bucket::hash::B_MAX_BITS, seed);
        let input = acc.finalize().with_hash(hash);
        assert!(
            input.num_buckets() * 8 > SCAN_LIMIT,
            "fixture: entries × buckets must exceed SCAN_LIMIT"
        );
        let rows = PartitionRows::<1>::from_seed(nq, 1, seed ^ 0x77);
        let size = rows.num_partitions() as u32;
        let ch = GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix());
        let mut remote_layers = 0;
        for rank in 0..size {
            let local = input.filter_partition(&rows, rank);
            let prep = ch.prepare(local.hash(), false).expect("prepared");
            let plan = PartitionPlan::new(&prep, &rows, rank);
            if !plan.has_remote() {
                continue;
            }
            remote_layers += 1;
            let nb = local.num_buckets();
            let mut map = ChunkMap::default();
            map.rebuild(
                &Gf2Span::new(&plan.local_bucket_deltas, local.hash().bits()),
                nb,
                1,
            );
            let (want, want_counts) = export_layer(
                &local,
                &prep,
                &plan,
                size,
                &map,
                &mut ExportScratch::default(),
            );
            let dev = GpuSum::from_host(&local, 0).expect("upload");
            let mut scratch = scratch_with(&dev, true);
            let fp = FingerprintRows::<1>::new(dev.hash().seed());
            let table = DevicePrepared::new(&prep, dev.hash(), &fp, &plan.remote);
            scratch.upload_table(&dev, &table).expect("table");
            scratch.count_local(&dev, &table).expect("count");
            let (got, got_counts) =
                export_blocks(&dev, &table, &plan, &mut scratch, size).expect("export");
            dev.stream.synchronize().expect("sync");
            let what = format!("rank {rank}");
            assert_eq!(
                got_counts.rows_sent, want_counts.rows_to,
                "{what}: the oversize group must skip the merge, not shrink it"
            );
            assert_eq!(
                scratch.counters.rows_premerged, 0,
                "{what}: no rows were premerged"
            );
            for (q, (g, w)) in got.iter().zip(&want).enumerate() {
                match (g, w) {
                    (None, None) => {}
                    (Some(g), Some(w)) => {
                        assert_eq!(g.blocks.len(), w.blocks.len(), "{what}: blocks to {q}");
                        for (j, (gb, wb)) in g.blocks.iter().zip(&w.blocks).enumerate() {
                            assert_eq!(gb.header, wb.header, "{what}: header {j} to {q}");
                            assert_eq!(gb.offsets, wb.offsets, "{what}: offsets {j} to {q}");
                            assert_eq!(gb.cols(), wb.cols(), "{what}: columns {j} to {q}");
                        }
                    }
                    _ => panic!("{what}: payload presence to {q} differs"),
                }
            }
        }
        assert!(remote_layers > 0, "the fixture must export something");
    }

    /// A recycled payload comes back for its own device and keeps its blocks; another device's does not.
    #[test]
    fn the_payload_bin_is_keyed_by_device() {
        crate::require_cuda!();
        let sum = rand_sum::<1>(10, 8, 0xB1);
        let dev = GpuSum::from_host(&sum, 0).expect("upload");
        payload::drain_bin();
        let mut p = payload::reclaim::<1>(0);
        p.block_mut(1, &dev.stream).expect("blocks");
        payload::recycle(p);
        assert!(payload::reclaim::<1>(7).blocks.is_empty());
        assert!(payload::reclaim::<2>(0).blocks.is_empty());
        let p = payload::reclaim::<1>(0);
        assert_eq!((p.device, p.blocks.len()), (Some(0), 2));
        payload::recycle(p);
        payload::drain_bin();
        assert!(payload::reclaim::<1>(0).blocks.is_empty());
    }
}
