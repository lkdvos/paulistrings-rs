//! K10 export, the exchange, and the received rows' upload for one device layer. See ARCHITECTURE.md §Partitioning.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};
use num_complex::Complex64;

use super::error::GpuError;
use super::layer::{grow, thread_per, warp_per_bucket, xfer, LayerScratch, Xfer};
use super::module::MAX_BUCKET_LEN;
use super::prepared::DevicePrepared;
use super::staging::HostStaging;
use super::sum::GpuSum;
use crate::bucket::hash::PartitionRows;
use crate::engine::coset::Gf2Span;
use crate::engine::partitioned::layer::{exchange_chunks, LayerExchangeCounts};
use crate::engine::partitioned::plan::PartitionPlan;
use crate::engine::partitioned::transport::{
    BlockHeader, ChunkMap, ExchangeBlock, PartnerPayload, Transport,
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
    /// Payloads not in flight, the layer's own pool (`Transport::exchange_layer`).
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
        })
    }
}

/// The exchange call a partition owes its partners when its own layer failed before exporting: `None` to everyone, so the group stays in step and reports the error after the loop.
pub(crate) fn pair_empty_exchange<const W: usize, X: Transport>(transport: &X) {
    let send: Vec<Option<PartnerPayload<W>>> = (0..transport.size()).map(|_| None).collect();
    let _ = transport.exchange(send, &mut Vec::new());
}

/// Export → exchange → upload of the received rows, with their fingerprints computed on device.
/// Runs after K1 has filled `cnt` at the agreed bucket count; the layer's K2/K3 then read `scratch.export.recv_*`.
pub(crate) fn exchange_rows<const W: usize, X: Transport>(
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
            pair_empty_exchange::<W, X>(transport);
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
        // v1 waits for the whole transfer before the one upload; a chunked upload overlapping the tail is v2.
        for k in 0..map.chunks() {
            wait.wait_chunk(k);
        }
        #[cfg(feature = "phase-timing")]
        {
            laps.chunk_wait_ns += t_body.elapsed().as_nanos() as u64;
        }
        let blocks = paired_blocks(plan, recv);
        let mut total = 0usize;
        let mut max_seg = 0usize;
        off_host.clear();
        base_host.clear();
        base_host.resize(16, 0);
        for (k, block) in blocks.iter().enumerate() {
            debug_assert_eq!(block.offsets.len(), b + 1);
            base_host[k] = total as u32;
            off_host.extend_from_slice(&block.offsets[..b + 1]);
            max_seg = block
                .offsets
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

/// The received block per remote delta in plan order: the `j`-th of `remote_for_partner(q)` is the `j`-th block of `recv[q]`, as `RecvRows::new`.
fn paired_blocks<'a, const W: usize>(
    plan: &PartitionPlan,
    recv: &'a [Option<PartnerPayload<W>>],
) -> Vec<&'a ExchangeBlock<W>> {
    plan.remote
        .iter()
        .map(|r| {
            let j = plan
                .remote_for_partner(r.partner)
                .position(|other| other.entry == r.entry)
                .expect("a remote delta is in its own partner's list");
            let block = recv[r.partner as usize]
                .as_ref()
                .and_then(|payload| payload.blocks.get(j))
                .unwrap_or_else(|| {
                    panic!(
                        "partition {} sent no block for remote delta {j} (entry {})",
                        r.partner, r.entry
                    )
                });
            debug_assert_eq!(block.header.entry, r.entry as u32);
            block
        })
        .collect()
}

/// K10 for every remote delta: one block per delta into pooled payloads, in ascending remote-delta index per partner.
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
    let k = sum.kernels.clone();
    let o = sum.device();
    let e32 = table.entries as u32;
    let b32 = b as u32;
    grow(&s, &mut scratch.export.counts, b, o)?;
    grow(&s, &mut scratch.export.off, b + 1, o)?;
    let mut blocks_used = vec![0usize; size as usize];
    for r in &plan.remote {
        let q = r.partner as usize;
        let payload = send[q].get_or_insert_with(|| scratch.export.pool.pop().unwrap_or_default());
        let j = blocks_used[q];
        blocks_used[q] += 1;
        if payload.blocks.len() <= j {
            payload
                .blocks
                .resize_with(j + 1, ExchangeBlock::<W>::default);
        }
        let block = &mut payload.blocks[j];
        let (bd, e) = (r.bucket_delta, r.entry as u32);
        let t0 = scratch.event(sum)?;
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
            &s,
            &k,
            &scratch.export.counts.slice(0..b),
            &mut scratch.export.off.slice_mut(0..b + 1),
            b,
            &mut scratch.scan,
            &mut scratch.tot_a,
        )?;
        #[cfg(feature = "phase-timing")]
        s.synchronize()?;
        block.offsets.clear();
        block.offsets.resize(b + 1, 0);
        let off = &scratch.export.off;
        xfer(&mut scratch.xfer_ns, Xfer::D2h, || {
            s.memcpy_dtoh(&off.slice(0..b + 1), &mut block.offsets[..])?;
            s.synchronize()?;
            Ok(())
        })?;
        let rows = block.offsets[b] as usize;
        block.header = BlockHeader {
            num_buckets: b32,
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
            // SAFETY: arguments match `k_export_fill` in export.cu; the columns hold `rows` rows and `off` the block's offsets.
            unsafe {
                s.launch_builder(&k.export_fill)
                    .arg(&sum.cols.x)
                    .arg(&sum.cols.z)
                    .arg(&sum.cols.coeff)
                    .arg(&sum.cols.start)
                    .arg(&sum.cols.lens)
                    .arg(&scratch.bucket_at)
                    .arg(&table.mode)
                    .arg(&e32)
                    .arg(&table.kq)
                    .arg(&table.q0)
                    .arg(&table.q1)
                    .arg(&table.rot_cos)
                    .arg(&table.rot_sin)
                    .arg(&scratch.amp)
                    .arg(&scratch.mask)
                    .arg(&scratch.nz)
                    .arg(&bd)
                    .arg(&e)
                    .arg(&b32)
                    .arg(&scratch.export.off)
                    .arg(&mut scratch.export.x)
                    .arg(&mut scratch.export.z)
                    .arg(&mut scratch.export.c)
                    .launch(warp_per_bucket(b))?;
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

    /// K10's blocks against `export_layer`'s on every partition: header, offsets and columns bitwise, since a freshly uploaded sum keeps the host's within-bucket order.
    fn export_matches_host<const W: usize>(num_qubits: usize, n: usize, seed: u64, pbits: u8) {
        let input = rand_sum::<W>(n, num_qubits, seed);
        let rows = PartitionRows::<W>::from_seed(num_qubits, pbits, seed ^ 0x77);
        let size = rows.num_partitions() as u32;
        let channels: Vec<(&str, Box<dyn Channel<W>>)> = vec![
            (
                "su4",
                Box::new(GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix())),
            ),
            ("cnot", Box::new(Clifford2Q::cnot(1, 3))),
            ("zz", Box::new(zz_rotation::<W>(0, 2, 0.3))),
        ];
        let mut remote_layers = 0;
        for (name, ch) in &channels {
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
                let mut scratch =
                    LayerScratch::new(&dev, GpuLayerOptions::default()).expect("scratch");
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
}
