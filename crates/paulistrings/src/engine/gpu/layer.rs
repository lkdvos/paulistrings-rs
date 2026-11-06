//! One layer on the device: bucket policy, count, export and exchange, fused layer, compaction, and the key-preserving rescale.
//!
//! Within a bucket the device keeps unique keys in no particular order; the fused layer sorts records by fingerprint and falls back to the full key on a collision, so equal keys always end adjacent.
//! A block's records and a source bucket's rows are checked against `LAYER_CAP` and `MAX_BUCKET_LEN` before the launch: a lone partition or a local layer refines past a violation up to `GpuLayerOptions::max_bits`, while a remote layer reports `Unsupported`, since its count is agreed.
//! A permutation table takes the scatter path instead of the fused layer only when the layer has no received rows.

use crate::coefficient::Coefficient;
use num_complex::Complex64;
use std::sync::Arc;

#[cfg(feature = "phase-timing")]
use cudarc::driver::sys::CUevent_flags;
#[cfg(feature = "phase-timing")]
use cudarc::driver::CudaEvent;
use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};

use super::collapse::CollapseScratch;
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
use crate::channel::prepared::{Prepared, LOCAL_DIM};
use crate::engine::coset::{CosetOrder, Gf2Span};
use crate::engine::partitioned::layer::LayerExchangeCounts;
use crate::engine::plan::PartitionPlan;
use crate::engine::timing::timed;
use crate::engine::trace::DeviceLayerRecord;
use crate::engine::transport::Transport;
use crate::truncation::builtin::APPROX_BINS;

mod fast_paths;
mod fused;
mod join;
mod options;
mod rotation_run;
mod run;

use fast_paths::{permute_device, rescale_device};
pub(super) use fused::{
    arena_batches, fused_variant, launch_fused, FusedOut, FusedReceived, FusedTable,
};
use join::join_layer;
pub(crate) use options::{gpu_desired_bits, prepared_fanout};
pub use options::{GpuLayerOptions, DEFAULT_ARENA_BYTES, DEFAULT_RECORDS_PER_BLOCK};
pub(crate) use rotation_run::apply_shared_bytes as rotation_run_shared_bytes;
pub(crate) use run::{permutation_run_device, rotation_run_device};

/// The partition's phase counters under `phase-timing` (`benchmarks/PROFILING.md` contract (a)), nothing otherwise.
#[cfg(feature = "phase-timing")]
pub(crate) type DeviceStats = crate::engine::stats::PhaseStats;
#[cfg(not(feature = "phase-timing"))]
pub(crate) type DeviceStats = ();

/// The `PhaseStats` field a kernel interval lands in.
#[cfg(feature = "phase-timing")]
type KernelSlot = fn(&mut DeviceStats) -> &mut u64;

/// Bytes of a [`DevicePrepared`]'s amplitudes in the widest coefficient type, so one allocation serves every type.
const AMPLITUDE_BYTES: usize = 16 * LOCAL_DIM * std::mem::size_of::<num_complex::Complex64>();

/// A [`DevicePrepared`]'s per-entry tables on the device, and the packed upload [`Self::upload_packed`] stages them through.
pub(super) struct TableBuffers {
    packed: CudaSlice<u32>,
    staging: Vec<u32>,
    /// The map segment's destination when no map is uploaded.
    no_map: CudaSlice<u32>,
    pub(super) amplitude: CudaSlice<u8>,
    pub(super) mask: CudaSlice<u64>,
    pub(super) nonzero_patterns: CudaSlice<u32>,
    pub(super) bucket_delta: CudaSlice<u32>,
    pub(super) fingerprint_delta: CudaSlice<u64>,
    pub(super) received_slot: CudaSlice<u32>,
}

impl TableBuffers {
    pub(super) fn new(stream: &Arc<CudaStream>, w: usize) -> Result<Self, GpuError> {
        Ok(Self {
            packed: stream.alloc_zeros(1)?,
            staging: Vec::new(),
            no_map: stream.alloc_zeros(1)?,
            amplitude: stream.alloc_zeros(AMPLITUDE_BYTES)?,
            mask: stream.alloc_zeros(32 * w)?,
            nonzero_patterns: stream.alloc_zeros(16)?,
            bucket_delta: stream.alloc_zeros(16)?,
            fingerprint_delta: stream.alloc_zeros(16)?,
            received_slot: stream.alloc_zeros(16)?,
        })
    }

    /// Copy every table of `table` up, and `bucket_at` into `map` if given, as one copy and one `k_unpack_table`.
    pub(super) fn upload_packed<const W: usize, C: Coefficient>(
        &mut self,
        stream: &Arc<CudaStream>,
        kernels: &KernelSet,
        table: &DevicePrepared<W, C>,
        map: Option<(&[u32], &mut CudaSlice<u32>)>,
    ) -> Result<(), GpuError> {
        let staging = &mut self.staging;
        staging.clear();
        let mut lengths = [0u32; 7];
        let mut push = |segment: usize, words: &[u32]| {
            staging.extend_from_slice(words);
            lengths[segment] = words.len() as u32;
        };
        push(0, bytemuck::cast_slice::<C, u32>(&table.amplitude));
        push(1, bytemuck::cast_slice::<u64, u32>(&table.mask));
        push(2, &table.nonzero_patterns);
        push(3, &table.bucket_delta);
        push(4, bytemuck::cast_slice::<u64, u32>(&table.fingerprint_delta));
        push(5, &table.received_slot);
        let ordinal = stream.context().ordinal() as u32;
        let (map_host, map) = match map {
            Some((host, device)) => (host, Some(device)),
            None => (&[][..], None),
        };
        push(6, map_host);
        let words = staging.len();
        super::columns::grow(stream, &mut self.packed, words, ordinal)?;
        stream.memcpy_htod(&staging[..], &mut self.packed.slice_mut(0..words))?;
        let map = match map {
            Some(map) => map,
            None => &mut self.no_map,
        };
        let [n0, n1, n2, n3, n4, n5, n6] = lengths;
        // SAFETY: arguments match `k_unpack_table` in count.cu; every destination holds its segment's words: the table buffers their `DevicePrepared` sizes, `map` the bucket count, and a skipped segment has length 0.
        unsafe {
            stream
                .launch_builder(&kernels.unpack_table)
                .arg(&self.packed)
                .arg(&mut self.amplitude)
                .arg(&n0)
                .arg(&mut self.mask)
                .arg(&n1)
                .arg(&mut self.nonzero_patterns)
                .arg(&n2)
                .arg(&mut self.bucket_delta)
                .arg(&n3)
                .arg(&mut self.fingerprint_delta)
                .arg(&n4)
                .arg(&mut self.received_slot)
                .arg(&n5)
                .arg(&mut *map)
                .arg(&n6)
                .launch(thread_per(words.min(1 << 16), 256))?;
        }
        Ok(())
    }

    /// Copy every table of `table` up; the copies are synchronous.
    pub(super) fn upload<const W: usize, C: Coefficient>(
        &mut self,
        stream: &Arc<CudaStream>,
        table: &DevicePrepared<W, C>,
    ) -> Result<(), GpuError> {
        stream.memcpy_htod(bytemuck::cast_slice::<C, u8>(&table.amplitude), &mut self.amplitude)?;
        stream.memcpy_htod(&table.mask, &mut self.mask)?;
        stream.memcpy_htod(&table.nonzero_patterns, &mut self.nonzero_patterns)?;
        stream.memcpy_htod(&table.bucket_delta, &mut self.bucket_delta)?;
        stream.memcpy_htod(&table.fingerprint_delta, &mut self.fingerprint_delta)?;
        stream.memcpy_htod(&table.received_slot, &mut self.received_slot)?;
        Ok(())
    }
}

/// Grow-only device buffers one partition keeps between layers.
pub(crate) struct LayerScratch<const W: usize, C: Coefficient = Complex64> {
    pub(super) bucket_at: CudaSlice<u32>,
    pub(super) counts: CudaSlice<u32>,
    rows: CudaSlice<u32>,
    segment_start: CudaSlice<u32>,
    rows_per_position: CudaSlice<u32>,
    pub(super) destination_offsets: CudaSlice<u32>,
    /// The octave histogram's `[len, bins…]`, and whether it is all zero, as `k_octave_edge` leaves it.
    pub(super) histogram: CudaSlice<u64>,
    pub(super) histogram_clean: bool,
    /// Whether `histogram` is the current sum's, as a join that counted its output leaves it; every mutating layer clears it.
    pub(super) histogram_fresh: bool,
    /// Whether a join counts its output's octave histogram, for the lone `ApproxTopN` pass after it.
    pub(crate) join_histogram: bool,
    /// The radix-select's 256-bin digit histogram, reused across passes.
    pub(super) radix_histogram: CudaSlice<u64>,
    /// The radix-select's extracted singleton bits.
    pub(super) radix_out: CudaSlice<u64>,
    /// `k_cut`'s row flags, one byte per row of the columns' capacity, its `[kept terms, buckets taken]`, and its look-back prefix per bucket.
    pub(super) cut_flag: CudaSlice<u8>,
    pub(super) cut_counters: CudaSlice<u64>,
    pub(super) cut_prefix: CudaSlice<u64>,
    /// `k_cut`'s decision, `CUT_KEEP_ALL`… in truncate.cu.
    pub(super) cut_decision: CudaSlice<u64>,
    fallback: CudaSlice<u32>,
    pub(super) table: TableBuffers,
    entry_of: CudaSlice<u32>,
    pub(super) arena: Option<DeviceColumns<W, C>>,
    /// A permutation run's step tables, allocated by the first run.
    run: Option<run::RunBuffers>,
    /// A fused rotation run's buffers, allocated by the first.
    rotation_run: Option<rotation_run::RotationRunScratch<W, C>>,
    segment_start_host: Vec<u32>,
    /// The coset order `bucket_at` uploads the inverse of, and that inverse on the host.
    order: CosetOrder,
    bucket_at_host: Vec<u32>,
    /// `(bits, bucket deltas)` the resident `bucket_at` was built for; the map is a function of those two alone.
    bucket_at_key: Option<(u8, Vec<u32>)>,
    /// The longest source bucket at the current bucket count, once a pass of the layer has read it.
    longest: Option<u32>,
    pub(super) scan: ScanScratch,
    /// `CollapseSample`'s buffers.
    pub(super) collapse: CollapseScratch,
    /// Scan totals, two slots so a count's pair of scans can be read together.
    pub(super) totals: CudaSlice<u32>,
    length_totals: CudaSlice<u32>,
    /// `k_join`'s `[output terms, longest output bucket]`, zeroed by `k_join_rows`.
    pub(super) join_totals: CudaSlice<u32>,
    /// Export, exchange and receive buffers.
    pub(super) export: DeviceExport<W, C>,
    /// Highest live row index plus one; differs from `len` only after a rescale, whose output keeps the input's offsets.
    pub(crate) extent: usize,
    pub(crate) options: GpuLayerOptions,
    /// The run policy's ceiling on a layer's input (`TruncationPolicy::run_length_limit`), which caps how far an output over-reserves; `None` without one.
    pub(crate) ceiling: Option<usize>,
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

impl<const W: usize, C: Coefficient> LayerScratch<W, C> {
    pub(crate) fn new(sum: &GpuSum<W, C>, options: GpuLayerOptions) -> Result<Self, GpuError> {
        let stream = &sum.stream;
        Ok(Self {
            bucket_at: stream.alloc_zeros(1)?,
            counts: stream.alloc_zeros(1)?,
            rows: stream.alloc_zeros(1)?,
            segment_start: stream.alloc_zeros(2)?,
            rows_per_position: stream.alloc_zeros(1)?,
            destination_offsets: stream.alloc_zeros(2)?,
            histogram: stream.alloc_zeros(1 + APPROX_BINS)?,
            histogram_clean: true,
            histogram_fresh: false,
            join_histogram: false,
            radix_histogram: stream.alloc_zeros(256)?,
            radix_out: stream.alloc_zeros(2)?,
            cut_flag: stream.alloc_zeros(1)?,
            cut_counters: stream.alloc_zeros(2)?,
            cut_prefix: stream.alloc_zeros(1)?,
            cut_decision: stream.alloc_zeros(4)?,
            fallback: stream.alloc_zeros(2)?,
            table: TableBuffers::new(stream, W)?,
            entry_of: stream.alloc_zeros(16)?,
            arena: None,
            run: None,
            rotation_run: None,
            segment_start_host: Vec::new(),
            order: CosetOrder::default(),
            bucket_at_host: Vec::new(),
            bucket_at_key: None,
            longest: None,
            scan: ScanScratch::new(stream)?,
            collapse: CollapseScratch::new(stream)?,
            totals: stream.alloc_zeros(2)?,
            length_totals: stream.alloc_zeros(2)?,
            join_totals: stream.alloc_zeros(2)?,
            export: DeviceExport::new(sum)?,
            extent: sum.len(),
            options,
            ceiling: None,
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
    pub(super) fn event(&mut self, sum: &GpuSum<W, C>) -> Result<usize, GpuError> {
        if self.next_event == self.events.len() {
            self.events.push(sum.context.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?);
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
        sum: &GpuSum<W, C>,
        from: usize,
        into: KernelSlot,
    ) -> Result<(), GpuError> {
        let to = self.event(sum)?;
        self.pending.push((from, to, into));
        Ok(())
    }

    /// Read every pending interval into `stats`, in nanoseconds, behind one synchronization and free the events for reuse.
    pub(super) fn resolve(&mut self, sum: &GpuSum<W, C>) -> Result<(), GpuError> {
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
        sum: &GpuSum<W, C>,
        table: &DevicePrepared<W, C>,
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
            self.bucket_at_host.extend((0..b as u32).map(|position| order.bucket_at(position)));
            grow(stream, &mut self.bucket_at, b, ordinal)?;
        }
        grow(stream, &mut self.counts, b * e, ordinal)?;
        grow(stream, &mut self.rows, b, ordinal)?;
        grow(stream, &mut self.segment_start, b + 1, ordinal)?;
        grow(stream, &mut self.destination_offsets, b + 1, ordinal)?;
        let (bucket_at_host, bucket_at) = (&self.bucket_at_host, &mut self.bucket_at);
        let buffers = &mut self.table;
        let map = map_stale.then_some((&bucket_at_host[..], bucket_at));
        timed_transfer(stream, &mut self.stats, Transfer::H2d, || {
            buffers.upload_packed(stream, &sum.kernels, table, map)
        })?;
        if map_stale {
            self.bucket_at_key = Some(key);
        }
        Ok(())
    }

    /// `k_count` over the local buckets.
    pub(super) fn count_local(
        &mut self,
        sum: &GpuSum<W, C>,
        table: &DevicePrepared<W, C>,
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

    /// The original bucket at each position of the resident `bucket_at`, for a host-side debug check.
    #[cfg(debug_assertions)]
    pub(super) fn bucket_at_host(&self) -> &[u32] {
        &self.bucket_at_host
    }

    /// The longest source bucket, read once per bucket count and layer.
    pub(super) fn longest_bucket(&mut self, sum: &GpuSum<W, C>) -> Result<u32, GpuError> {
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
    fn scan_lens(&mut self, sum: &GpuSum<W, C>) -> Result<(), GpuError> {
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
        sum: &GpuSum<W, C>,
        table: &DevicePrepared<W, C>,
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
pub(crate) struct DeviceLayer<'a, const W: usize, C: Coefficient> {
    pub(crate) prepared: &'a Prepared<W, C>,
    pub(crate) plan: &'a PartitionPlan,
    /// The per-term half of the run's policy.
    pub(crate) keep: &'a KeepProgram,
    /// The group's agreed bucket bits.
    pub(crate) target_bits: u8,
}

/// Apply `layer` to `sum` on its device, leaving the previous columns as `sum.spare`; remote deltas exchange over `transport`.
pub(crate) fn apply_layer_device<const W: usize, C: Coefficient, X: Transport>(
    sum: &mut GpuSum<W, C>,
    layer: &DeviceLayer<'_, W, C>,
    scratch: &mut LayerScratch<W, C>,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    scratch.histogram_fresh = false;
    let r = apply_layer_body(sum, layer, scratch, transport);
    // A failed layer still completes its pending receive, so every peer's chunked receive completes too.
    let finished = finish_receive(sum, scratch);
    let resolved = scratch.resolve(sum);
    r.and_then(|c| finished.and(resolved).map(|()| c))
}

fn apply_layer_body<const W: usize, C: Coefficient, X: Transport>(
    sum: &mut GpuSum<W, C>,
    layer: &DeviceLayer<'_, W, C>,
    scratch: &mut LayerScratch<W, C>,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    let &DeviceLayer { prepared, plan, keep, target_bits } = layer;
    let size = transport.size();
    let mut table = DevicePrepared::new(prepared, &sum.hash, &sum.fingerprints, &plan.remote);
    let has_remote = plan.has_remote();
    debug_assert_eq!(table.received_entries, plan.remote.len());
    scratch.record = DeviceLayerRecord { dense: table.dense, ..DeviceLayerRecord::default() };
    scratch.refine_passes = 0;
    scratch.receive_chunks = 0;
    scratch.export.received_rows = 0;
    scratch.export.received_max_segment = 0;
    scratch.longest = None;
    debug_assert!(scratch.export.pending.is_none());
    let refine = |sum: &mut GpuSum<W, C>,
                  scratch: &mut LayerScratch<W, C>,
                  bits: u8|
     -> Result<(), GpuError> {
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
    let pair = |scratch: &mut LayerScratch<W, C>, bits: u8| {
        if has_remote {
            super::export::pair_empty_exchange::<W, C, X>(
                transport,
                plan,
                bits,
                &mut scratch.export,
            );
        }
    };
    if n_in.saturating_mul(table.fanout.max(1)) >= u32::MAX as usize {
        pair(scratch, sum.hash.bits());
        return Err(GpuError::Unsupported("more than 2^32 pre-dedup records in one layer"));
    }
    // A remote layer runs at exactly the agreed `target_bits`, since partners index its blocks by the count; a local layer refines freely, as counts already differ mid-call (engine/driver.rs).
    let refines = solo || !has_remote;
    let max_bits = if refines {
        scratch.options.max_bits.min(sum.max_bits())
    } else {
        target_bits.max(sum.hash.bits())
    };
    let want = if refines {
        let estimate = sum.len_estimate();
        gpu_desired_bits(estimate, table.fanout, scratch.options.records_per_block, sum.hash.bits())
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
    let join = table.join.filter(|_| scratch.options.join);
    scratch.record.joined = join.is_some();
    if let Some(join) = join {
        debug_assert!(!has_remote, "a join table has no received entry");
        // A join writes straight into the output, so only its source buckets' length is capped: `k_join`'s 13-bit row index; a bound past the cap is settled before it refines.
        let longest = loop {
            scratch.upload_table(sum, &table)?;
            let longest = match sum.longest_bound() {
                Some(bound) if bound as usize <= MAX_BUCKET_LEN => bound,
                _ => {
                    sum.settle_lengths()?;
                    sum.longest_bound().expect("settled lengths know the longest bucket")
                }
            };
            if longest as usize <= MAX_BUCKET_LEN {
                scratch.join_sizes(sum)?;
                break longest;
            }
            if sum.hash.bits() >= max_bits {
                return Err(GpuError::Unsupported(
                    "a source bucket exceeds the join's row index at the bucket-bit limit",
                ));
            }
            refine(sum, scratch, sum.hash.bits() + 1)?;
            table.rehash(&sum.hash);
        };
        // `k_join_rows` totals exactly twice the input, so twice its bound bounds the output.
        let records = (2 * sum.len()) as u32;
        scratch.record.records = u64::from(records);
        scratch.record.batches = 1;
        join_layer(sum, scratch, &table, join, keep, longest, records)?;
        return Ok(counts);
    }
    // Oversize blocks and over-long source buckets are known from the count table; a layer that may refine takes one more bit, which halves both.
    let (records, records_max, _) = loop {
        let counted =
            scratch.upload_table(sum, &table).and_then(|()| scratch.count_local(sum, &table));
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
            break (total, max_segment, max_len);
        }
        if !refines {
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
    let chunk_starts: Vec<usize> =
        scratch.export.pending.as_ref().map_or_else(Vec::new, |p| {
            (0..p.map.chunks()).map(|c| p.map.bound(c) as usize).collect()
        });
    let (batches, max_batch_rows) = arena_batches::<W, C>(
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
        None => DeviceColumns::<W, C>::with_capacity(&stream, ordinal, max_batch_rows, 1)?,
    };
    arena.len = 0;
    arena.buckets = 0;
    arena.reserve(max_batch_rows, 1)?;
    let mut out = match sum.spare.take() {
        Some(spare) => spare,
        None => DeviceColumns::<W, C>::with_capacity(&stream, ordinal, n_in, b)?,
    };
    out.len = 0;
    out.buckets = 0;
    out.reserve(out.term_capacity(), b)?;
    grow(&stream, &mut scratch.rows_per_position, b, ordinal)?;
    stream.memset_zeros(&mut scratch.fallback)?;
    // A lone batch whose pre-dedup rows already fit the output compacts at offset 0 without its total, which is read once after the compaction.
    let deferred_total = batches.len() == 1
        && (scratch.segment_start_host[b] - scratch.segment_start_host[0]) as usize
            <= out.term_capacity();
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
        let batch_out = if deferred_total {
            0
        } else {
            let totals = &scratch.totals;
            let batch_out = timed_transfer(&stream, &mut scratch.stats, Transfer::D2h, || {
                let v = stream.clone_dtoh(totals)?;
                stream.synchronize()?;
                Ok(v[0])
            })?;
            out.len = running as usize;
            let need = (running + batch_out) as usize;
            if need > out.term_capacity() {
                let grown = (2 * out.term_capacity()).max(need).min((records as usize).max(need));
                out.reserve(grown, b)?;
            }
            batch_out
        };
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
    let (totals, fallback) = (&scratch.totals, &scratch.fallback);
    let (total, fallbacks) = timed_transfer(&stream, &mut scratch.stats, Transfer::D2h, || {
        let total = if deferred_total { stream.clone_dtoh(totals)?[0] } else { 0 };
        let fallbacks = stream.clone_dtoh(fallback)?;
        stream.synchronize()?;
        Ok((total, fallbacks))
    })?;
    running += total;
    scratch.record.high_word_fallbacks += fallbacks[0];
    scratch.record.full_key_fallbacks += fallbacks[1];
    out.len = running as usize;
    out.buckets = b;
    scratch.arena = Some(arena);
    sum.spare = Some(std::mem::replace(&mut sum.columns, out));
    sum.record_exact(running as usize, None);
    scratch.extent = sum.len();
    sum.debug_check();
    Ok(counts)
}

#[cfg(test)]
mod tests;
