//! One layer on the device: bucket policy, count, export and exchange, fused layer, compaction, and the key-preserving rescale. See ARCHITECTURE.md §Engine, §Partitioning and §GPU-Readiness.

use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, PushKernelArg};

use super::columns::{grow, DeviceColumns};
use super::error::GpuError;
use super::export::{exchange_rows, finish_receive, receive_chunk, DeviceExport, MAX_RECV_SEGMENT};
use super::module::{thread_per, warp_per_bucket, KernelSet, MAX_BUCKET_LEN};
use super::prepared::DevicePrepared;
use super::scan::{exclusive_scan, ScanScratch};
use super::sum::GpuSum;
use super::truncation::KeepProgram;
use crate::channel::prepared::Prepared;
use crate::engine::coset::Gf2Span;
use crate::engine::partitioned::layer::LayerExchangeCounts;
use crate::engine::partitioned::plan::PartitionPlan;
use crate::engine::partitioned::transport::Transport;
use crate::pauli_sum::hash::B_MAX_BITS;
use crate::truncation::builtin::APPROX_BINS;

mod fast_paths;
mod fused;
mod options;

use fast_paths::{permute_device, rescale_device};
pub(super) use fused::{
    arena_batches, fused_variant, launch_fused, FusedOut, FusedRecv, FusedTable,
};
pub(crate) use options::{gpu_desired_bits, prepared_fanout};
pub use options::{
    GpuBucketPolicy, GpuLayerOptions, DEFAULT_ARENA_BYTES, DEFAULT_RECORDS_PER_BLOCK,
};

/// Per-layer counters of the most recent device layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuLayerCounters {
    /// Bucket bits the layer ran at.
    pub bits: u8,
    /// Refine passes the layer itself issued.
    pub refine_passes: u32,
    /// Pre-dedup records over every block.
    pub records: u64,
    /// The largest block's records.
    pub records_max: u32,
    /// Record capacity of the launched variant.
    pub record_capacity: u32,
    /// Arena batches.
    pub batches: u32,
    /// Blocks that fell back to the `g_hi32` passes, the sender-side merge's included.
    pub fallback_hi: u32,
    /// Blocks that fell back to the full-key sort, the sender-side merge's included.
    pub fallback_key: u32,
    /// The layer took the rescale path.
    pub rescaled: bool,
    /// The layer took the permutation path, K12–K14, with `records` its input rows.
    pub permuted: bool,
    /// The layer reduced by the segmented scan rather than the head-serial walk.
    pub dense: bool,
    /// Rows received from partners and merged by the fused layer.
    pub rows_received: u64,
    /// Exported rows the sender-side merge folded away before the exchange.
    pub rows_premerged: u64,
    /// Chunks the received rows moved in ([`GpuLayerOptions::exchange_bytes`]); zero on a local layer.
    pub recv_chunks: u32,
}

/// Kernel milliseconds per family, accumulated under `phase-timing` and folded into the partition's `PhaseStats`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct GpuKernelMs {
    /// K1, the count table.
    pub count: f64,
    /// K2, segment sizes and their scans.
    pub sizes: f64,
    /// K3, the fused layer over every batch.
    pub layer: f64,
    /// K4, compaction and its scans.
    pub compact: f64,
    /// K5, the key-preserving rescale.
    pub rescale: f64,
    /// K14, the permutation scatter; its K12 count and K13 sizes land in `count` and `sizes`.
    pub permute: f64,
    /// K6, refines the layer issued.
    pub refine: f64,
    /// K7, the `ApproxTopN` histogram and retain.
    pub truncate: f64,
    /// K10, the export blocks' counts and fill.
    pub export: f64,
}

/// The exchange's laps of one partition, drained into its `PhaseStats` per layer.
#[cfg(feature = "phase-timing")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ExchangeLaps {
    pub export_ns: u64,
    pub exchange_ns: u64,
    pub rows_exported: u64,
    pub recv_rows: u64,
}

/// A [`DevicePrepared`]'s per-entry tables on the device.
pub(super) struct TableBuffers {
    pub(super) amp: CudaSlice<f64>,
    pub(super) mask: CudaSlice<u64>,
    pub(super) nz: CudaSlice<u32>,
    pub(super) bucket_delta: CudaSlice<u32>,
    pub(super) gm: CudaSlice<u64>,
    pub(super) rem: CudaSlice<u32>,
}

impl TableBuffers {
    pub(super) fn new(stream: &Arc<CudaStream>, w: usize) -> Result<Self, GpuError> {
        Ok(Self {
            amp: stream.alloc_zeros(512)?,
            mask: stream.alloc_zeros(32 * w)?,
            nz: stream.alloc_zeros(16)?,
            bucket_delta: stream.alloc_zeros(16)?,
            gm: stream.alloc_zeros(16)?,
            rem: stream.alloc_zeros(16)?,
        })
    }

    /// Copy every table of `table` up; the copies are synchronous.
    pub(super) fn upload<const W: usize>(
        &mut self,
        stream: &Arc<CudaStream>,
        table: &DevicePrepared<W>,
    ) -> Result<(), GpuError> {
        stream.memcpy_htod(&table.amp, &mut self.amp)?;
        stream.memcpy_htod(&table.mask, &mut self.mask)?;
        stream.memcpy_htod(&table.nz, &mut self.nz)?;
        stream.memcpy_htod(&table.bucket_delta, &mut self.bucket_delta)?;
        stream.memcpy_htod(&table.gm, &mut self.gm)?;
        stream.memcpy_htod(&table.rem, &mut self.rem)?;
        Ok(())
    }
}

/// Grow-only device buffers one partition keeps between layers.
pub(crate) struct LayerScratch<const W: usize> {
    pub(super) bucket_at: CudaSlice<u32>,
    pub(super) counts: CudaSlice<u32>,
    rows: CudaSlice<u32>,
    segment_start: CudaSlice<u32>,
    out_len_pos: CudaSlice<u32>,
    pub(super) destination_offsets: CudaSlice<u32>,
    /// K7's `[len, bins…]`.
    pub(super) hist: CudaSlice<u64>,
    /// K8's 256-bin radix digit histogram, reused across passes.
    pub(super) radix_histogram: CudaSlice<u64>,
    /// K8's small scalar outputs: the extracted singleton bits, or `[above, equal]` counts.
    pub(super) radix_out: CudaSlice<u64>,
    fallback: CudaSlice<u32>,
    pub(super) table: TableBuffers,
    entry_of: CudaSlice<u32>,
    pub(super) arena: Option<DeviceColumns<W>>,
    segment_start_host: Vec<u32>,
    bucket_at_host: Vec<u32>,
    /// `(bits, bucket deltas)` the resident `bucket_at` was built for; the map is a function of those two alone.
    bucket_at_key: Option<(u8, Vec<u32>)>,
    /// The longest source bucket at the current bucket count, once a pass of the layer has read it.
    longest: Option<u32>,
    pub(super) scan: ScanScratch,
    /// Scan totals, two slots so a count's pair of scans can be read together.
    pub(super) totals: CudaSlice<u32>,
    length_totals: CudaSlice<u32>,
    /// Export, exchange and receive buffers.
    pub(super) export: DeviceExport<W>,
    /// Highest live row index plus one; differs from `len` only after a rescale, whose output keeps the input's offsets.
    pub(crate) extent: usize,
    pub(crate) options: GpuLayerOptions,
    pub(crate) counters: GpuLayerCounters,
    pub(crate) kernel_ms: GpuKernelMs,
    /// Timing events, reused across layers.
    events: Vec<CudaEvent>,
    next_event: usize,
    /// Recorded event pairs not yet read; [`Self::resolve`] reads them behind one synchronization instead of one per kernel.
    pending: Vec<(usize, usize, KernelSlot)>,
    /// Host-to-device and device-to-host copy time inside the current layer.
    pub(crate) transfer_ns: TransferNs,
    #[cfg(feature = "phase-timing")]
    pub(crate) laps: ExchangeLaps,
}

type KernelSlot = fn(&mut GpuKernelMs) -> &mut f64;

#[cfg(feature = "phase-timing")]
pub(crate) type TransferNs = [u64; 2];
#[cfg(not(feature = "phase-timing"))]
pub(crate) type TransferNs = ();

/// Which direction a timed copy in [`timed_transfer`] runs.
#[derive(Clone, Copy)]
pub(super) enum Transfer {
    H2d,
    D2h,
}

/// Run `f`, a synchronous host<->device copy on `stream`, timing it into `h2d_ns` / `d2h_ns` under `phase-timing`.
#[inline]
pub(super) fn timed_transfer<T>(
    stream: &CudaStream,
    ns: &mut TransferNs,
    dir: Transfer,
    f: impl FnOnce() -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    #[cfg(feature = "phase-timing")]
    {
        // The copy would otherwise absorb the wait for queued kernels, so the timed region starts after its own sync.
        stream.synchronize()?;
        let t0 = std::time::Instant::now();
        let r = f();
        ns[dir as usize] += t0.elapsed().as_nanos() as u64;
        r
    }
    #[cfg(not(feature = "phase-timing"))]
    {
        let _ = (stream, ns, dir);
        f()
    }
}

impl<const W: usize> LayerScratch<W> {
    pub(crate) fn new(sum: &GpuSum<W>, options: GpuLayerOptions) -> Result<Self, GpuError> {
        let stream = &sum.stream;
        Ok(Self {
            bucket_at: stream.alloc_zeros(1)?,
            counts: stream.alloc_zeros(1)?,
            rows: stream.alloc_zeros(1)?,
            segment_start: stream.alloc_zeros(2)?,
            out_len_pos: stream.alloc_zeros(1)?,
            destination_offsets: stream.alloc_zeros(2)?,
            hist: stream.alloc_zeros(1 + APPROX_BINS)?,
            radix_histogram: stream.alloc_zeros(256)?,
            radix_out: stream.alloc_zeros(2)?,
            fallback: stream.alloc_zeros(2)?,
            table: TableBuffers::new(stream, W)?,
            entry_of: stream.alloc_zeros(16)?,
            arena: None,
            segment_start_host: Vec::new(),
            bucket_at_host: Vec::new(),
            bucket_at_key: None,
            longest: None,
            scan: ScanScratch::new(stream)?,
            totals: stream.alloc_zeros(2)?,
            length_totals: stream.alloc_zeros(2)?,
            export: DeviceExport::new(sum)?,
            extent: sum.len(),
            options,
            counters: GpuLayerCounters::default(),
            kernel_ms: GpuKernelMs::default(),
            events: Vec::new(),
            next_event: 0,
            pending: Vec::new(),
            transfer_ns: TransferNs::default(),
            #[cfg(feature = "phase-timing")]
            laps: ExchangeLaps::default(),
        })
    }

    pub(super) fn event(&mut self, sum: &GpuSum<W>) -> Result<Option<usize>, GpuError> {
        if !cfg!(feature = "phase-timing") {
            return Ok(None);
        }
        if self.next_event == self.events.len() {
            self.events.push(
                sum.context
                    .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
            );
        }
        let i = self.next_event;
        self.events[i].record(&sum.stream)?;
        self.next_event += 1;
        Ok(Some(i))
    }

    /// Close the interval opened by `from` with a new event; read at the next [`Self::resolve`].
    pub(super) fn lap(
        &mut self,
        sum: &GpuSum<W>,
        from: Option<usize>,
        into: KernelSlot,
    ) -> Result<(), GpuError> {
        let Some(i) = from else { return Ok(()) };
        let Some(j) = self.event(sum)? else {
            return Ok(());
        };
        self.pending.push((i, j, into));
        Ok(())
    }

    /// Read every pending interval into `kernel_ms` behind one synchronization and free the events for reuse.
    pub(super) fn resolve(&mut self, sum: &GpuSum<W>) -> Result<(), GpuError> {
        if self.pending.is_empty() {
            self.next_event = 0;
            return Ok(());
        }
        sum.stream.synchronize()?;
        for (i, j, into) in std::mem::take(&mut self.pending) {
            let ms = f64::from(self.events[i].elapsed_ms(&self.events[j])?);
            *into(&mut self.kernel_ms) += ms;
        }
        self.next_event = 0;
        Ok(())
    }

    /// Upload the table and the position map (over the local deltas) for the current bucket count.
    pub(super) fn upload_table(
        &mut self,
        sum: &GpuSum<W>,
        table: &DevicePrepared<W>,
    ) -> Result<(), GpuError> {
        let stream = &sum.stream;
        let ordinal = sum.device();
        let b = sum.hash.num_buckets();
        let e = table.entries;
        let key = (sum.hash.bits(), table.bucket_deltas());
        let map_stale = self.bucket_at_key.as_ref() != Some(&key);
        if map_stale {
            let span = Gf2Span::new(&key.1, key.0);
            self.bucket_at_host.clear();
            self.bucket_at_host.resize(b, 0);
            for beta in 0..b as u32 {
                self.bucket_at_host[span.permuted_index(beta) as usize] = beta;
            }
            grow(stream, &mut self.bucket_at, b, ordinal)?;
        }
        grow(stream, &mut self.counts, b * e, ordinal)?;
        grow(stream, &mut self.rows, b, ordinal)?;
        grow(stream, &mut self.segment_start, b + 1, ordinal)?;
        grow(stream, &mut self.destination_offsets, b + 1, ordinal)?;
        let (bucket_at_host, bucket_at) = (&self.bucket_at_host, &mut self.bucket_at);
        let buffers = &mut self.table;
        timed_transfer(stream, &mut self.transfer_ns, Transfer::H2d, || {
            if map_stale {
                stream.memcpy_htod(bucket_at_host, &mut bucket_at.slice_mut(0..b))?;
            }
            buffers.upload(stream, table)
        })?;
        if map_stale {
            self.bucket_at_key = Some(key);
        }
        Ok(())
    }

    /// K1 over the local buckets.
    pub(super) fn count_local(
        &mut self,
        sum: &GpuSum<W>,
        table: &DevicePrepared<W>,
    ) -> Result<(), GpuError> {
        let stream = &sum.stream;
        let b = sum.hash.num_buckets();
        let (b32, e32) = (b as u32, table.entries as u32);
        let t0 = self.event(sum)?;
        // SAFETY: arguments match `k_count` in count.cu; `counts` holds `b * e` entries.
        unsafe {
            stream
                .launch_builder(&sum.kernels.count)
                .arg(&sum.columns.x)
                .arg(&sum.columns.z)
                .arg(&sum.columns.start)
                .arg(&sum.columns.lens)
                .arg(&b32)
                .arg(&table.mode)
                .arg(&e32)
                .arg(&table.kq)
                .arg(&table.q0)
                .arg(&table.q1)
                .arg(&table.rot_cos)
                .arg(&table.rot_sin)
                .arg(&self.table.amp)
                .arg(&self.table.mask)
                .arg(&self.table.nz)
                .arg(&mut self.counts)
                .launch(warp_per_bucket(b))?;
        }
        self.lap(sum, t0, |m| &mut m.count)
    }

    /// The longest source bucket, read once per bucket count and layer.
    pub(super) fn longest_bucket(&mut self, sum: &GpuSum<W>) -> Result<u32, GpuError> {
        if let Some(l) = self.longest {
            return Ok(l);
        }
        self.scan_lens(sum)?;
        let (stream, length_totals) = (&sum.stream, &self.length_totals);
        let l = timed_transfer(stream, &mut self.transfer_ns, Transfer::D2h, || {
            let v = stream.clone_dtoh(length_totals)?;
            stream.synchronize()?;
            Ok(v[1])
        })?;
        self.longest = Some(l);
        Ok(l)
    }

    /// The scan of the bucket lengths whose maximum `length_totals[1]` is the longest bucket.
    fn scan_lens(&mut self, sum: &GpuSum<W>) -> Result<(), GpuError> {
        let b = sum.hash.num_buckets();
        exclusive_scan(
            &sum.stream,
            &sum.kernels,
            &sum.columns.lens.slice(0..b),
            &mut self.destination_offsets.slice_mut(0..b + 1),
            b,
            &mut self.scan,
            &mut self.length_totals,
        )
    }

    /// K2 and its scans, over the local counts and the received offsets; returns `(records total, records max, longest local bucket)`.
    fn sizes(
        &mut self,
        sum: &GpuSum<W>,
        table: &DevicePrepared<W>,
    ) -> Result<(u32, u32, u32), GpuError> {
        let stream = &sum.stream;
        let kernels = &sum.kernels;
        let b = sum.hash.num_buckets();
        let (b32, e32) = (b as u32, table.entries as u32);
        let t1 = self.event(sum)?;
        // SAFETY: arguments match `k_rows` in count.cu; `received_offsets` holds `K × (b + 1)` entries for the `K` received entries `rem` names.
        unsafe {
            stream
                .launch_builder(&kernels.rows)
                .arg(&self.counts)
                .arg(&self.bucket_at)
                .arg(&self.table.bucket_delta)
                .arg(&self.table.rem)
                .arg(&self.export.received_offsets)
                .arg(&mut self.rows)
                .arg(&b32)
                .arg(&e32)
                .launch(thread_per(b, 1024))?;
        }
        exclusive_scan(
            stream,
            kernels,
            &self.rows.slice(0..b),
            &mut self.segment_start.slice_mut(0..b + 1),
            b,
            &mut self.scan,
            &mut self.totals,
        )?;
        let known = self.longest;
        if known.is_none() {
            self.scan_lens(sum)?;
        }
        self.lap(sum, t1, |m| &mut m.sizes)?;
        self.segment_start_host.resize(b + 1, 0);
        let (segment_start, segment_start_host) =
            (&self.segment_start, &mut self.segment_start_host);
        let (totals, length_totals) = (&self.totals, &self.length_totals);
        let (tm, longest) = timed_transfer(stream, &mut self.transfer_ns, Transfer::D2h, || {
            let tm = stream.clone_dtoh(totals)?;
            let longest = match known {
                Some(l) => l,
                None => stream.clone_dtoh(length_totals)?[1],
            };
            stream.memcpy_dtoh(&segment_start.slice(0..b + 1), &mut segment_start_host[..])?;
            stream.synchronize()?;
            Ok((tm, longest))
        })?;
        self.longest = Some(longest);
        Ok((tm[0], tm[1], longest))
    }
}

/// Apply `prepared` to `sum` on its device under `keep` at the group's agreed `target_bits`, leaving the previous columns as `sum.spare`; remote deltas exchange over `transport`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_layer_device<const W: usize, X: Transport>(
    sum: &mut GpuSum<W>,
    prepared: &Prepared<W>,
    plan: &PartitionPlan,
    keep: &KeepProgram,
    scratch: &mut LayerScratch<W>,
    target_bits: u8,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    let r = apply_layer_body(sum, prepared, plan, keep, scratch, target_bits, transport);
    // A failed layer still completes its pending receive, so every peer's chunked receive completes too.
    let finished = finish_receive(sum, scratch);
    let resolved = scratch.resolve(sum);
    r.and_then(|c| finished.and(resolved).map(|()| c))
}

#[allow(clippy::too_many_arguments)]
fn apply_layer_body<const W: usize, X: Transport>(
    sum: &mut GpuSum<W>,
    prepared: &Prepared<W>,
    plan: &PartitionPlan,
    keep: &KeepProgram,
    scratch: &mut LayerScratch<W>,
    target_bits: u8,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    let size = transport.size();
    scratch.next_event = 0;
    let mut table = DevicePrepared::new(prepared, &sum.hash, &sum.fingerprints, &plan.remote);
    let has_remote = plan.has_remote();
    debug_assert_eq!(table.n_remote, plan.remote.len());
    scratch.counters = GpuLayerCounters {
        bits: sum.hash.bits(),
        dense: table.dense,
        ..GpuLayerCounters::default()
    };
    scratch.export.received_rows = 0;
    scratch.export.received_max_segment = 0;
    scratch.longest = None;
    debug_assert!(scratch.export.pending.is_none());
    let refine =
        |sum: &mut GpuSum<W>, scratch: &mut LayerScratch<W>, bits: u8| -> Result<(), GpuError> {
            let t = scratch.event(sum)?;
            sum.refine_to(bits)?;
            scratch.lap(sum, t, |m| &mut m.refine)?;
            scratch.extent = sum.len();
            scratch.longest = None;
            scratch.counters.refine_passes += 1;
            Ok(())
        };
    if table.key_preserving {
        debug_assert!(!has_remote);
        if target_bits > sum.hash.bits() {
            refine(sum, scratch, target_bits)?;
        }
        rescale_device(sum, &table, keep, scratch)?;
        return Ok(LayerExchangeCounts::none(size));
    }
    if table.permutation && scratch.options.clifford {
        debug_assert!(!has_remote);
        if target_bits > sum.hash.bits() {
            refine(sum, scratch, target_bits)?;
            table.rehash(&sum.hash);
        }
        permute_device(sum, &table, keep, scratch)?;
        return Ok(LayerExchangeCounts::none(size));
    }
    let solo = size == 1;
    let n_in = sum.len();
    // Every failure before the exchange still owes the partners their call.
    let pair = |scratch: &mut LayerScratch<W>, bits: u8| {
        if has_remote {
            super::export::pair_empty_exchange::<W, X>(transport, plan, bits, &mut scratch.export);
        }
    };
    if n_in.saturating_mul(table.fanout.max(1)) >= u32::MAX as usize {
        pair(scratch, sum.hash.bits());
        return Err(GpuError::Unsupported(
            "more than 2^32 pre-dedup records in one layer",
        ));
    }
    // A partition of a group runs at exactly `target_bits`, steered through `proposed_bits` alone, and never refines off-schedule (ARCHITECTURE.md §Partitioning).
    let max_bits = if solo {
        scratch.options.max_bits.min(B_MAX_BITS)
    } else {
        target_bits.max(sum.hash.bits())
    };
    let want = if solo {
        gpu_desired_bits(
            n_in,
            table.fanout,
            scratch.options.bucket_policy,
            sum.hash.bits(),
        )
        .min(max_bits)
        .max(target_bits)
    } else {
        target_bits
    };
    if want > sum.hash.bits() {
        if let Err(e) = refine(sum, scratch, want) {
            pair(scratch, sum.hash.bits());
            return Err(e);
        }
        table.rehash(&sum.hash);
    }
    let record_cap = sum.kernels.layer_cap();
    let mut counts = LayerExchangeCounts::none(size);
    // Oversize blocks and over-long source buckets are known from the count table; a lone partition refines one more bit, which halves both.
    let (records, records_max) = loop {
        let counted = scratch
            .upload_table(sum, &table)
            .and_then(|()| scratch.count_local(sum, &table));
        if let Err(e) = counted {
            pair(scratch, sum.hash.bits());
            return Err(e);
        }
        if has_remote {
            counts = exchange_rows(sum, &table, plan, scratch, transport)?;
        }
        let (total, max_segment, max_len) = scratch.sizes(sum, &table)?;
        if max_segment as usize <= record_cap
            && max_len as usize <= MAX_BUCKET_LEN
            && scratch.export.received_max_segment <= MAX_RECV_SEGMENT
        {
            break (total, max_segment);
        }
        if !solo {
            return Err(GpuError::Unsupported(
                "a fused-layer block or a received segment exceeds the record cap at the agreed bucket count",
            ));
        }
        if sum.hash.bits() >= max_bits {
            return Err(GpuError::Unsupported(
                "a fused-layer block exceeds the record cap at the bucket-bit limit",
            ));
        }
        refine(sum, scratch, sum.hash.bits() + 1)?;
        table.rehash(&sum.hash);
    };
    let bits = sum.hash.bits();
    let b = sum.hash.num_buckets();
    scratch.counters.bits = bits;
    scratch.counters.records = u64::from(records);
    scratch.counters.records_max = records_max;
    scratch.counters.rows_received = counts.rows_received;

    let kernels: Arc<KernelSet> = sum.kernels.clone();
    let (func, record_capacity, smem) =
        fused_variant(&kernels, W, records_max as usize, table.dense);
    scratch.counters.record_capacity = record_capacity as u32;

    // Batches: contiguous position ranges whose pre-dedup rows fit the arena, never straddling a chunk of the pending receive.
    let chunk_starts: Vec<usize> = scratch.export.pending.as_ref().map_or_else(Vec::new, |p| {
        (0..p.map.chunks())
            .map(|c| p.map.bound(c) as usize)
            .collect()
    });
    let (batches, max_batch_rows) = arena_batches::<W>(
        &scratch.segment_start_host,
        scratch.options.arena_bytes,
        record_cap,
        &chunk_starts,
    );
    scratch.counters.batches = batches.len() as u32;
    scratch.counters.recv_chunks = chunk_starts.len() as u32;

    let stream = sum.stream.clone();
    let ordinal = sum.device();
    let mut arena = match scratch.arena.take() {
        Some(a) => a,
        None => DeviceColumns::<W>::with_capacity(&stream, ordinal, max_batch_rows, 1)?,
    };
    arena.len = 0;
    arena.buckets = 0;
    arena.reserve(max_batch_rows, 1)?;
    let mut out = match sum.spare.take() {
        Some(spare) => spare,
        None => DeviceColumns::<W>::with_capacity(&stream, ordinal, n_in, b)?,
    };
    out.len = 0;
    out.buckets = 0;
    out.reserve(out.term_capacity(), b)?;
    grow(&stream, &mut scratch.out_len_pos, b, ordinal)?;
    stream.memset_zeros(&mut scratch.fallback)?;
    let mut running = 0u32;
    let mut next_chunk = 0usize;
    for &(p0, p1) in &batches {
        if chunk_starts.get(next_chunk) == Some(&p0) {
            receive_chunk(sum, scratch, next_chunk)?;
            next_chunk += 1;
        }
        let num_blocks = (p1 - p0) as u32;
        let p0u = p0 as u32;
        let t0 = scratch.event(sum)?;
        let fused = FusedTable {
            counts: &scratch.counts,
            segment_start: &scratch.segment_start,
            table: &scratch.table,
        };
        let recv = FusedRecv {
            offsets: &scratch.export.received_offsets,
            base: &scratch.export.received_base,
            x: &scratch.export.received_x,
            z: &scratch.export.received_z,
            coefficient: &scratch.export.received_coefficient,
            g: &scratch.export.received_fingerprint,
        };
        let written = FusedOut {
            arena: &mut arena,
            out_len_pos: &mut scratch.out_len_pos,
            fallback: &mut scratch.fallback,
        };
        launch_fused(
            sum,
            &table,
            &fused,
            &recv,
            &scratch.bucket_at,
            keep,
            (func, smem),
            (p0u, num_blocks),
            written,
        )?;
        scratch.lap(sum, t0, |m| &mut m.layer)?;
        let n = p1 - p0;
        let t1 = scratch.event(sum)?;
        exclusive_scan(
            &stream,
            &kernels,
            &scratch.out_len_pos.slice(p0..p1),
            &mut scratch.destination_offsets.slice_mut(0..n + 1),
            n,
            &mut scratch.scan,
            &mut scratch.totals,
        )?;
        let totals = &scratch.totals;
        let batch_out = timed_transfer(&stream, &mut scratch.transfer_ns, Transfer::D2h, || {
            let v = stream.clone_dtoh(totals)?;
            stream.synchronize()?;
            Ok(v[0])
        })?;
        out.len = running as usize;
        let need = (running + batch_out) as usize;
        if need > out.term_capacity() {
            let grown = (2 * out.term_capacity())
                .max(need)
                .min((records as usize).max(need));
            out.reserve(grown, b)?;
        }
        // SAFETY: arguments match `k_compact` in compact.cu; `out` holds `running + batch_out` terms.
        unsafe {
            stream
                .launch_builder(&kernels.compact)
                .arg(&arena.x)
                .arg(&arena.z)
                .arg(&arena.coeff)
                .arg(&arena.g)
                .arg(&scratch.segment_start)
                .arg(&scratch.destination_offsets)
                .arg(&scratch.out_len_pos)
                .arg(&scratch.bucket_at)
                .arg(&p0u)
                .arg(&running)
                .arg(&mut out.x)
                .arg(&mut out.z)
                .arg(&mut out.coeff)
                .arg(&mut out.g)
                .arg(&mut out.start)
                .arg(&mut out.lens)
                .launch(thread_per(num_blocks as usize * 256, 256))?;
        }
        scratch.lap(sum, t1, |m| &mut m.compact)?;
        running += batch_out;
    }
    debug_assert_eq!(next_chunk, chunk_starts.len());
    let fallbacks = stream.clone_dtoh(&scratch.fallback)?;
    stream.synchronize()?;
    scratch.counters.fallback_hi += fallbacks[0];
    scratch.counters.fallback_key += fallbacks[1];
    out.len = running as usize;
    out.buckets = b;
    scratch.arena = Some(arena);
    sum.spare = Some(std::mem::replace(&mut sum.columns, out));
    scratch.extent = sum.len();
    sum.debug_check();
    Ok(counts)
}

#[cfg(test)]
mod tests;
