//! One layer on the device: bucket policy, count, export and exchange, fused layer, compaction, and the key-preserving rescale. See ARCHITECTURE.md §Engine, §Partitioning and §GPU-Readiness.

use std::sync::Arc;

#[cfg(feature = "phase-timing")]
use cudarc::driver::sys::CUevent_flags;
#[cfg(feature = "phase-timing")]
use cudarc::driver::CudaEvent;
use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};

use super::columns::{grow, DeviceColumns};
use super::error::GpuError;
use super::export::{
    exchange_rows, finish_receive, receive_chunk, DeviceExport, MAX_RECEIVED_SEGMENT,
};
use super::module::{thread_per, warp_per_bucket, KernelSet, MAX_BUCKET_LEN};
use super::prepared::DevicePrepared;
use super::scan::{exclusive_scan, ScanScratch};
use super::sum::{GpuSum, REFINE_MAX_DELTA};
use super::truncation::KeepProgram;
use crate::channel::prepared::Prepared;
use crate::engine::coset::{CosetOrder, Gf2Span};
use crate::engine::partitioned::layer::LayerExchangeCounts;
use crate::engine::plan::PartitionPlan;
use crate::engine::timing::timed;
use crate::engine::trace::DeviceLayerRecord;
use crate::engine::transport::Transport;
use crate::pauli_sum::hash::B_MAX_BITS;
use crate::truncation::builtin::APPROX_BINS;

mod fast_paths;
mod fused;
mod options;

use fast_paths::{permute_device, rescale_device};
pub(super) use fused::{
    arena_batches, fused_variant, launch_fused, FusedOut, FusedReceived, FusedTable,
};
pub(crate) use options::{gpu_desired_bits, prepared_fanout};
pub use options::{GpuLayerOptions, DEFAULT_ARENA_BYTES, DEFAULT_RECORDS_PER_BLOCK};

/// The partition's phase counters under `phase-timing` (ARCHITECTURE.md §GPU-Readiness, `benchmarks/PROFILING.md` contract (a)), nothing otherwise.
#[cfg(feature = "phase-timing")]
pub(crate) type DeviceStats = crate::engine::stats::PhaseStats;
#[cfg(not(feature = "phase-timing"))]
pub(crate) type DeviceStats = ();

/// The `PhaseStats` field a kernel interval lands in.
#[cfg(feature = "phase-timing")]
type KernelSlot = fn(&mut DeviceStats) -> &mut u64;

/// A [`DevicePrepared`]'s per-entry tables on the device.
pub(super) struct TableBuffers {
    pub(super) amplitude: CudaSlice<f64>,
    pub(super) mask: CudaSlice<u64>,
    pub(super) nonzero_patterns: CudaSlice<u32>,
    pub(super) bucket_delta: CudaSlice<u32>,
    pub(super) fingerprint_delta: CudaSlice<u64>,
    pub(super) received_slot: CudaSlice<u32>,
}

impl TableBuffers {
    pub(super) fn new(stream: &Arc<CudaStream>, w: usize) -> Result<Self, GpuError> {
        Ok(Self {
            amplitude: stream.alloc_zeros(512)?,
            mask: stream.alloc_zeros(32 * w)?,
            nonzero_patterns: stream.alloc_zeros(16)?,
            bucket_delta: stream.alloc_zeros(16)?,
            fingerprint_delta: stream.alloc_zeros(16)?,
            received_slot: stream.alloc_zeros(16)?,
        })
    }

    /// Copy every table of `table` up; the copies are synchronous.
    pub(super) fn upload<const W: usize>(
        &mut self,
        stream: &Arc<CudaStream>,
        table: &DevicePrepared<W>,
    ) -> Result<(), GpuError> {
        stream.memcpy_htod(&table.amplitude, &mut self.amplitude)?;
        stream.memcpy_htod(&table.mask, &mut self.mask)?;
        stream.memcpy_htod(&table.nonzero_patterns, &mut self.nonzero_patterns)?;
        stream.memcpy_htod(&table.bucket_delta, &mut self.bucket_delta)?;
        stream.memcpy_htod(&table.fingerprint_delta, &mut self.fingerprint_delta)?;
        stream.memcpy_htod(&table.received_slot, &mut self.received_slot)?;
        Ok(())
    }
}

/// Grow-only device buffers one partition keeps between layers.
pub(crate) struct LayerScratch<const W: usize> {
    pub(super) bucket_at: CudaSlice<u32>,
    pub(super) counts: CudaSlice<u32>,
    rows: CudaSlice<u32>,
    segment_start: CudaSlice<u32>,
    rows_per_position: CudaSlice<u32>,
    pub(super) destination_offsets: CudaSlice<u32>,
    /// The octave histogram's `[len, bins…]`.
    pub(super) histogram: CudaSlice<u64>,
    /// The radix-select's 256-bin digit histogram, reused across passes.
    pub(super) radix_histogram: CudaSlice<u64>,
    /// The radix-select's extracted singleton bits.
    pub(super) radix_out: CudaSlice<u64>,
    fallback: CudaSlice<u32>,
    pub(super) table: TableBuffers,
    entry_of: CudaSlice<u32>,
    pub(super) arena: Option<DeviceColumns<W>>,
    segment_start_host: Vec<u32>,
    /// The coset order `bucket_at` uploads the inverse of, and that inverse on the host.
    order: CosetOrder,
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
    /// The current layer's device counters, read into its trace row.
    pub(crate) record: DeviceLayerRecord,
    /// The current layer's refine passes.
    pub(crate) refine_passes: u32,
    /// The current layer's receive chunks, planned before the first moves, so a failure mid-receive keeps it.
    pub(crate) receive_chunks: u32,
    /// Kernel intervals, copies and exchange laps land here directly.
    pub(crate) stats: DeviceStats,
    /// Timing events, reused across layers.
    #[cfg(feature = "phase-timing")]
    events: Vec<CudaEvent>,
    #[cfg(feature = "phase-timing")]
    next_event: usize,
    /// Recorded event pairs not yet read; [`Self::resolve`] reads them behind one synchronization instead of one per kernel.
    #[cfg(feature = "phase-timing")]
    pending: Vec<(usize, usize, KernelSlot)>,
}

/// Which direction a timed copy in [`timed_transfer`] runs.
#[derive(Clone, Copy)]
pub(super) enum Transfer {
    H2d,
    D2h,
}

/// Run `f`, a synchronous host<->device copy on `stream`, timing it into `h2d_ns` / `d2h_ns` under `phase-timing`.
pub(super) fn timed_transfer<T>(
    stream: &CudaStream,
    stats: &mut DeviceStats,
    dir: Transfer,
    f: impl FnOnce() -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    #[cfg(feature = "phase-timing")]
    {
        // The copy would otherwise absorb the wait for queued kernels, so the timed region starts after its own sync.
        stream.synchronize()?;
        let t0 = std::time::Instant::now();
        let r = f();
        let ns = t0.elapsed().as_nanos() as u64;
        match dir {
            Transfer::H2d => stats.h2d_ns += ns,
            Transfer::D2h => stats.d2h_ns += ns,
        }
        r
    }
    #[cfg(not(feature = "phase-timing"))]
    {
        let _ = (stream, stats, dir);
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
            rows_per_position: stream.alloc_zeros(1)?,
            destination_offsets: stream.alloc_zeros(2)?,
            histogram: stream.alloc_zeros(1 + APPROX_BINS)?,
            radix_histogram: stream.alloc_zeros(256)?,
            radix_out: stream.alloc_zeros(2)?,
            fallback: stream.alloc_zeros(2)?,
            table: TableBuffers::new(stream, W)?,
            entry_of: stream.alloc_zeros(16)?,
            arena: None,
            segment_start_host: Vec::new(),
            order: CosetOrder::default(),
            bucket_at_host: Vec::new(),
            bucket_at_key: None,
            longest: None,
            scan: ScanScratch::new(stream)?,
            totals: stream.alloc_zeros(2)?,
            length_totals: stream.alloc_zeros(2)?,
            export: DeviceExport::new(sum)?,
            extent: sum.len(),
            options,
            record: DeviceLayerRecord::default(),
            refine_passes: 0,
            receive_chunks: 0,
            stats: DeviceStats::default(),
            #[cfg(feature = "phase-timing")]
            events: Vec::new(),
            #[cfg(feature = "phase-timing")]
            next_event: 0,
            #[cfg(feature = "phase-timing")]
            pending: Vec::new(),
        })
    }

    /// Record a timing event on the sum's stream, opening an interval for [`Self::lap`].
    #[cfg(feature = "phase-timing")]
    pub(super) fn event(&mut self, sum: &GpuSum<W>) -> Result<usize, GpuError> {
        if self.next_event == self.events.len() {
            self.events.push(
                sum.context
                    .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?,
            );
        }
        let i = self.next_event;
        self.events[i].record(&sum.stream)?;
        self.next_event += 1;
        Ok(i)
    }

    /// Close the interval opened by `from` with a new event; read into `into` at the next [`Self::resolve`].
    #[cfg(feature = "phase-timing")]
    pub(super) fn lap(
        &mut self,
        sum: &GpuSum<W>,
        from: usize,
        into: KernelSlot,
    ) -> Result<(), GpuError> {
        let to = self.event(sum)?;
        self.pending.push((from, to, into));
        Ok(())
    }

    /// Read every pending interval into `stats`, in nanoseconds, behind one synchronization and free the events for reuse.
    pub(super) fn resolve(&mut self, sum: &GpuSum<W>) -> Result<(), GpuError> {
        #[cfg(feature = "phase-timing")]
        {
            if !self.pending.is_empty() {
                sum.stream.synchronize()?;
            }
            for (from, to, into) in std::mem::take(&mut self.pending) {
                let ms = f64::from(self.events[from].elapsed_ms(&self.events[to])?);
                *into(&mut self.stats) += (ms.max(0.0) * 1e6) as u64;
            }
            self.next_event = 0;
        }
        #[cfg(not(feature = "phase-timing"))]
        let _ = sum;
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
            self.order.rebuild(&Gf2Span::new(&key.1, key.0), true);
            let order = &self.order;
            self.bucket_at_host.clear();
            self.bucket_at_host
                .extend((0..b as u32).map(|position| order.bucket_at(position)));
            grow(stream, &mut self.bucket_at, b, ordinal)?;
        }
        grow(stream, &mut self.counts, b * e, ordinal)?;
        grow(stream, &mut self.rows, b, ordinal)?;
        grow(stream, &mut self.segment_start, b + 1, ordinal)?;
        grow(stream, &mut self.destination_offsets, b + 1, ordinal)?;
        let (bucket_at_host, bucket_at) = (&self.bucket_at_host, &mut self.bucket_at);
        let buffers = &mut self.table;
        timed_transfer(stream, &mut self.stats, Transfer::H2d, || {
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

    /// `k_count` over the local buckets.
    pub(super) fn count_local(
        &mut self,
        sum: &GpuSum<W>,
        table: &DevicePrepared<W>,
    ) -> Result<(), GpuError> {
        let stream = &sum.stream;
        let b = sum.hash.num_buckets();
        let (b32, e32) = (b as u32, table.entries as u32);
        timed!(let t0 = self.event(sum)?);
        // SAFETY: arguments match `k_count` in count.cu; `counts` holds `b * e` entries.
        unsafe {
            stream
                .launch_builder(&sum.kernels.count)
                .arg(&sum.columns.x)
                .arg(&sum.columns.z)
                .arg(&sum.columns.start)
                .arg(&sum.columns.lengths)
                .arg(&b32)
                .arg(&table.mode)
                .arg(&e32)
                .arg(&table.support_size)
                .arg(&table.qubit_0)
                .arg(&table.qubit_1)
                .arg(&table.rotation_cos)
                .arg(&table.rotation_sin)
                .arg(&self.table.amplitude)
                .arg(&self.table.mask)
                .arg(&self.table.nonzero_patterns)
                .arg(&mut self.counts)
                .launch(warp_per_bucket(b))?;
        }
        timed!(self.lap(sum, t0, |stats| &mut stats.gather_ns)?);
        Ok(())
    }

    /// The longest source bucket, read once per bucket count and layer.
    pub(super) fn longest_bucket(&mut self, sum: &GpuSum<W>) -> Result<u32, GpuError> {
        if let Some(l) = self.longest {
            return Ok(l);
        }
        self.scan_lens(sum)?;
        let (stream, length_totals) = (&sum.stream, &self.length_totals);
        let l = timed_transfer(stream, &mut self.stats, Transfer::D2h, || {
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
            &sum.columns.lengths.slice(0..b),
            &mut self.destination_offsets.slice_mut(0..b + 1),
            b,
            &mut self.scan,
            &mut self.length_totals,
        )
    }

    /// `k_rows` and its scans, over the local counts and the received offsets; returns `(records total, records max, longest local bucket)`.
    fn sizes(
        &mut self,
        sum: &GpuSum<W>,
        table: &DevicePrepared<W>,
    ) -> Result<(u32, u32, u32), GpuError> {
        let stream = &sum.stream;
        let kernels = &sum.kernels;
        let b = sum.hash.num_buckets();
        let (b32, e32) = (b as u32, table.entries as u32);
        timed!(let t1 = self.event(sum)?);
        // SAFETY: arguments match `k_rows` in count.cu; `received_offsets` holds `K × (b + 1)` entries for the `K` received entries `received_slot` names.
        unsafe {
            stream
                .launch_builder(&kernels.rows)
                .arg(&self.counts)
                .arg(&self.bucket_at)
                .arg(&self.table.bucket_delta)
                .arg(&self.table.received_slot)
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
        timed!(self.lap(sum, t1, |stats| &mut stats.gather_ns)?);
        self.segment_start_host.resize(b + 1, 0);
        let (segment_start, segment_start_host) =
            (&self.segment_start, &mut self.segment_start_host);
        let (totals, length_totals) = (&self.totals, &self.length_totals);
        let (tm, longest) = timed_transfer(stream, &mut self.stats, Transfer::D2h, || {
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

/// One layer's inputs besides the sum, its scratch and the transport.
pub(crate) struct DeviceLayer<'a, const W: usize> {
    pub(crate) prepared: &'a Prepared<W>,
    pub(crate) plan: &'a PartitionPlan,
    /// The per-term half of the run's policy.
    pub(crate) keep: &'a KeepProgram,
    /// The group's agreed bucket bits.
    pub(crate) target_bits: u8,
}

/// Apply `layer` to `sum` on its device, leaving the previous columns as `sum.spare`; remote deltas exchange over `transport`.
pub(crate) fn apply_layer_device<const W: usize, X: Transport>(
    sum: &mut GpuSum<W>,
    layer: &DeviceLayer<'_, W>,
    scratch: &mut LayerScratch<W>,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    let r = apply_layer_body(sum, layer, scratch, transport);
    // A failed layer still completes its pending receive, so every peer's chunked receive completes too.
    let finished = finish_receive(sum, scratch);
    let resolved = scratch.resolve(sum);
    r.and_then(|c| finished.and(resolved).map(|()| c))
}

fn apply_layer_body<const W: usize, X: Transport>(
    sum: &mut GpuSum<W>,
    layer: &DeviceLayer<'_, W>,
    scratch: &mut LayerScratch<W>,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    let &DeviceLayer {
        prepared,
        plan,
        keep,
        target_bits,
    } = layer;
    let size = transport.size();
    let mut table = DevicePrepared::new(prepared, &sum.hash, &sum.fingerprints, &plan.remote);
    let has_remote = plan.has_remote();
    debug_assert_eq!(table.received_entries, plan.remote.len());
    scratch.record = DeviceLayerRecord {
        dense: table.dense,
        ..DeviceLayerRecord::default()
    };
    scratch.refine_passes = 0;
    scratch.receive_chunks = 0;
    scratch.export.received_rows = 0;
    scratch.export.received_max_segment = 0;
    scratch.longest = None;
    debug_assert!(scratch.export.pending.is_none());
    let refine =
        |sum: &mut GpuSum<W>, scratch: &mut LayerScratch<W>, bits: u8| -> Result<(), GpuError> {
            timed!(let t = scratch.event(sum)?);
            let before = sum.hash.bits();
            sum.refine_to(bits)?;
            timed!(scratch.lap(sum, t, |stats| &mut stats.rebucket_ns)?);
            scratch.extent = sum.len();
            scratch.longest = None;
            scratch.refine_passes += u32::from(bits - before).div_ceil(u32::from(REFINE_MAX_DELTA));
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
            scratch.options.records_per_block,
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
            && scratch.export.received_max_segment <= MAX_RECEIVED_SEGMENT
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
    let b = sum.hash.num_buckets();
    scratch.record.records = u64::from(records);
    scratch.record.max_block_records = records_max;

    let kernels: Arc<KernelSet> = sum.kernels.clone();
    let (func, record_capacity, smem) =
        fused_variant(&kernels, W, records_max as usize, table.dense);
    scratch.record.record_capacity = record_capacity as u32;

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
    scratch.record.batches = batches.len() as u32;
    scratch.receive_chunks = chunk_starts.len() as u32;
    counts.receive_chunks = scratch.receive_chunks;

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
    grow(&stream, &mut scratch.rows_per_position, b, ordinal)?;
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
        timed!(let t0 = scratch.event(sum)?);
        let fused = FusedTable {
            counts: &scratch.counts,
            segment_start: &scratch.segment_start,
            table: &scratch.table,
        };
        let received = FusedReceived {
            offsets: &scratch.export.received_offsets,
            base: &scratch.export.received_base,
            x: &scratch.export.received_x,
            z: &scratch.export.received_z,
            coefficient: &scratch.export.received_coefficient,
            fingerprint: &scratch.export.received_fingerprint,
        };
        let written = FusedOut {
            arena: &mut arena,
            rows_per_position: &mut scratch.rows_per_position,
            fallback: &mut scratch.fallback,
        };
        launch_fused(
            sum,
            &table,
            &fused,
            &received,
            &scratch.bucket_at,
            keep,
            (func, smem),
            (p0u, num_blocks),
            written,
        )?;
        timed!(scratch.lap(sum, t0, |stats| &mut stats.merge_ns)?);
        let n = p1 - p0;
        timed!(let t1 = scratch.event(sum)?);
        exclusive_scan(
            &stream,
            &kernels,
            &scratch.rows_per_position.slice(p0..p1),
            &mut scratch.destination_offsets.slice_mut(0..n + 1),
            n,
            &mut scratch.scan,
            &mut scratch.totals,
        )?;
        let totals = &scratch.totals;
        let batch_out = timed_transfer(&stream, &mut scratch.stats, Transfer::D2h, || {
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
                .arg(&arena.fingerprint)
                .arg(&scratch.segment_start)
                .arg(&scratch.destination_offsets)
                .arg(&scratch.rows_per_position)
                .arg(&scratch.bucket_at)
                .arg(&p0u)
                .arg(&running)
                .arg(&mut out.x)
                .arg(&mut out.z)
                .arg(&mut out.coeff)
                .arg(&mut out.fingerprint)
                .arg(&mut out.start)
                .arg(&mut out.lengths)
                .launch(thread_per(num_blocks as usize * 256, 256))?;
        }
        timed!(scratch.lap(sum, t1, |stats| &mut stats.compact_ns)?);
        running += batch_out;
    }
    debug_assert_eq!(next_chunk, chunk_starts.len());
    let fallbacks = stream.clone_dtoh(&scratch.fallback)?;
    stream.synchronize()?;
    scratch.record.high_word_fallbacks += fallbacks[0];
    scratch.record.full_key_fallbacks += fallbacks[1];
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
