//! One layer on the device: bucket policy, count, export and exchange, fused layer, compaction, and the key-preserving rescale. See ARCHITECTURE.md §Engine, §Partitioning and §GPU-Readiness.

use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{
    CudaEvent, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};

use super::columns::DeviceColumns;
use super::error::GpuError;
use super::export::{exchange_rows, DeviceExport, MAX_RECV_SEGMENT};
use super::fingerprint::FingerprintRows;
use super::module::{layer_shared_bytes, layer_threads, KernelSet, MAX_BUCKET_LEN};
use super::prepared::DevicePrepared;
use super::scan::{exclusive_scan_with_max_into, ScanScratch};
use super::sum::GpuSum;
use super::truncation::KeepProgram;
use crate::bucket::hash::{PartitionRows, B_MAX_BITS};
use crate::bucket::sum::desired_bits;
use crate::channel::prepared::Prepared;
use crate::engine::coset::Gf2Span;
use crate::engine::partitioned::layer::LayerExchangeCounts;
use crate::engine::partitioned::plan::PartitionPlan;
use crate::engine::partitioned::transport::Transport;
use crate::truncation::builtin::APPROX_BINS;

/// Records per fused block the default bucket policy aims for.
pub const DEFAULT_RECORDS_PER_BLOCK: usize = 4096;

/// Default cap on the loose output arena, in bytes.
pub const DEFAULT_ARENA_BYTES: usize = 4 << 30;

/// How the device chooses its bucket count before a layer; grow-only either way, as [`PauliSum::rebucket`](crate::PauliSum::rebucket).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBucketPolicy {
    /// `fanout × terms per bucket ≈ target`, so a fused block sees about `target` records whatever the channel's fanout.
    RecordsPerBlock(usize),
    /// A fixed `desired_bits(len, target, 1)`, independent of the channel.
    TermsPerBucket(usize),
}

impl Default for GpuBucketPolicy {
    fn default() -> Self {
        GpuBucketPolicy::RecordsPerBlock(DEFAULT_RECORDS_PER_BLOCK)
    }
}

/// Knobs of the device layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuLayerOptions {
    /// The bucket count the device refines to before a layer.
    pub bucket_policy: GpuBucketPolicy,
    /// Bytes the loose output arena may hold; positions are batched so no batch's pre-dedup rows exceed it.
    pub arena_bytes: usize,
    /// Bucket bits a layer may refine to before an oversize block is [`GpuError::Unsupported`]; `B_MAX_BITS` by default.
    pub max_bits: u8,
    /// Merge one partner's exported rows by key on the sender before the exchange (ARCHITECTURE.md §Partitioning); on unless `PAULISTRINGS_GPU_PREMERGE=off`.
    pub premerge: bool,
}

/// The default of [`GpuLayerOptions::premerge`]: on unless `PAULISTRINGS_GPU_PREMERGE=off`, read once per process.
fn premerge_default() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PAULISTRINGS_GPU_PREMERGE").as_deref() != Ok("off"))
}

impl Default for GpuLayerOptions {
    fn default() -> Self {
        Self {
            bucket_policy: GpuBucketPolicy::default(),
            arena_bytes: DEFAULT_ARENA_BYTES,
            max_bits: B_MAX_BITS,
            premerge: premerge_default(),
        }
    }
}

/// The bucket bits a layer of `fanout` entries over `len` terms wants under `policy`, never below `current`.
pub(crate) fn gpu_desired_bits(
    len: usize,
    fanout: usize,
    policy: GpuBucketPolicy,
    current: u8,
) -> u8 {
    let want = match policy {
        GpuBucketPolicy::TermsPerBucket(target) => desired_bits(len, target.max(1), 1),
        GpuBucketPolicy::RecordsPerBlock(target) => {
            let records = len.saturating_mul(fanout.max(1));
            let mut b = 0u8;
            while b < B_MAX_BITS && records > target.max(1).saturating_mul(1usize << b) {
                b += 1;
            }
            b
        }
    };
    want.max(current).min(B_MAX_BITS)
}

/// The entries `prep` emits records for, local and received alike: the fanout [`gpu_desired_bits`] sizes a block by.
pub(crate) fn prepared_fanout<const W: usize>(prep: &Prepared<W>) -> usize {
    match prep {
        Prepared::Local(ptm) => ptm
            .deltas()
            .iter()
            .filter(|d| {
                d.amp
                    .iter()
                    .any(|a| *a != num_complex::Complex64::new(0.0, 0.0))
            })
            .count(),
        Prepared::Rotation(_) => 2,
    }
}

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
    pub n_cap: u32,
    /// Arena batches.
    pub batches: u32,
    /// Blocks that fell back to the `g_hi32` passes, the sender-side merge's included.
    pub fallback_hi: u32,
    /// Blocks that fell back to the full-key sort, the sender-side merge's included.
    pub fallback_key: u32,
    /// The layer took the rescale path.
    pub rescaled: bool,
    /// The layer reduced by the segmented scan rather than the head-serial walk.
    pub dense: bool,
    /// Rows received from partners and merged by the fused layer.
    pub rows_received: u64,
    /// Exported rows the sender-side merge folded away before the exchange.
    pub rows_premerged: u64,
}

/// Kernel milliseconds per family, accumulated while [`GpuPauliSum::set_kernel_timing`](super::GpuPauliSum::set_kernel_timing) is on.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuKernelMs {
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
    pub chunk_wait_ns: u64,
    pub rows_exported: u64,
    pub recv_rows: u64,
}

/// Grow-only device buffers one partition keeps between layers.
pub(crate) struct LayerScratch<const W: usize> {
    pub(super) bucket_at: CudaSlice<u32>,
    pub(super) cnt: CudaSlice<u32>,
    rows: CudaSlice<u32>,
    seg_start: CudaSlice<u32>,
    out_len_pos: CudaSlice<u32>,
    pub(super) dst_off: CudaSlice<u32>,
    /// K7's `[len, bins…]`.
    pub(super) hist: CudaSlice<u64>,
    fallback: CudaSlice<u32>,
    pub(super) amp: CudaSlice<f64>,
    pub(super) mask: CudaSlice<u64>,
    pub(super) nz: CudaSlice<u32>,
    bd: CudaSlice<u32>,
    gm: CudaSlice<u64>,
    rem: CudaSlice<u32>,
    pub(super) arena: Option<DeviceColumns<W>>,
    seg_host: Vec<u32>,
    bucket_at_host: Vec<u32>,
    /// `(bits, bucket deltas)` the resident `bucket_at` was built for; the map is a function of those two alone.
    bucket_at_key: Option<(u8, Vec<u32>)>,
    pub(super) scan: ScanScratch,
    /// Scan totals, two slots so a count's pair of scans can be read together.
    pub(super) tot_a: CudaSlice<u32>,
    tot_b: CudaSlice<u32>,
    /// Export, exchange and receive buffers.
    pub(super) export: DeviceExport<W>,
    /// Highest live row index plus one; differs from `len` only after a rescale, whose output keeps the input's offsets.
    pub(crate) extent: usize,
    pub(crate) options: GpuLayerOptions,
    pub(crate) counters: GpuLayerCounters,
    pub(crate) time_kernels: bool,
    pub(crate) kernel_ms: GpuKernelMs,
    /// Timing events, reused across layers.
    events: Vec<CudaEvent>,
    next_event: usize,
    /// Recorded event pairs not yet read; [`Self::resolve`] reads them behind one synchronization instead of one per kernel.
    pending: Vec<(usize, usize, KernelSlot)>,
    /// Host-to-device and device-to-host copy time inside the current layer.
    pub(crate) xfer_ns: XferNs,
    #[cfg(feature = "phase-timing")]
    pub(crate) laps: ExchangeLaps,
}

type KernelSlot = fn(&mut GpuKernelMs) -> &mut f64;

#[cfg(feature = "phase-timing")]
pub(crate) type XferNs = [u64; 2];
#[cfg(not(feature = "phase-timing"))]
pub(crate) type XferNs = ();

/// Which direction a timed copy in [`xfer`] runs.
#[derive(Clone, Copy)]
pub(super) enum Xfer {
    H2d,
    D2h,
}

/// Run `f`, a synchronous host<->device copy, timing it into `h2d_ns` / `d2h_ns` under `phase-timing`.
#[inline]
pub(super) fn xfer<T>(
    ns: &mut XferNs,
    dir: Xfer,
    f: impl FnOnce() -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    #[cfg(feature = "phase-timing")]
    {
        let t0 = std::time::Instant::now();
        let r = f();
        ns[dir as usize] += t0.elapsed().as_nanos() as u64;
        r
    }
    #[cfg(not(feature = "phase-timing"))]
    {
        let _ = (ns, dir);
        f()
    }
}

pub(super) fn grow<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    s: &mut CudaSlice<T>,
    n: usize,
    ordinal: u32,
) -> Result<(), GpuError> {
    if s.len() >= n {
        return Ok(());
    }
    // SAFETY: every kernel writes a buffer before reading it, and the host never reads one.
    *s = unsafe { stream.alloc::<T>(n) }
        .map_err(|e| GpuError::from_alloc(e, ordinal, (n * std::mem::size_of::<T>()) as u64))?;
    Ok(())
}

pub(super) fn warp_per_bucket(b: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((b as u32).div_ceil(8).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(super) fn thread_per(n: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(threads).max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl<const W: usize> LayerScratch<W> {
    pub(crate) fn new(sum: &GpuSum<W>, options: GpuLayerOptions) -> Result<Self, GpuError> {
        let s = &sum.stream;
        Ok(Self {
            bucket_at: s.alloc_zeros(1)?,
            cnt: s.alloc_zeros(1)?,
            rows: s.alloc_zeros(1)?,
            seg_start: s.alloc_zeros(2)?,
            out_len_pos: s.alloc_zeros(1)?,
            dst_off: s.alloc_zeros(2)?,
            hist: s.alloc_zeros(1 + APPROX_BINS)?,
            fallback: s.alloc_zeros(2)?,
            amp: s.alloc_zeros(512)?,
            mask: s.alloc_zeros(32 * W)?,
            nz: s.alloc_zeros(16)?,
            bd: s.alloc_zeros(16)?,
            gm: s.alloc_zeros(16)?,
            rem: s.alloc_zeros(16)?,
            arena: None,
            seg_host: Vec::new(),
            bucket_at_host: Vec::new(),
            bucket_at_key: None,
            scan: ScanScratch::new(s)?,
            tot_a: s.alloc_zeros(2)?,
            tot_b: s.alloc_zeros(2)?,
            export: DeviceExport::new(sum)?,
            extent: sum.len(),
            options,
            counters: GpuLayerCounters::default(),
            time_kernels: false,
            kernel_ms: GpuKernelMs::default(),
            events: Vec::new(),
            next_event: 0,
            pending: Vec::new(),
            xfer_ns: XferNs::default(),
            #[cfg(feature = "phase-timing")]
            laps: ExchangeLaps::default(),
        })
    }

    /// Whether kernel events are recorded this layer: on request, or always under `phase-timing`.
    fn timing(&self) -> bool {
        self.time_kernels || cfg!(feature = "phase-timing")
    }

    pub(super) fn event(&mut self, sum: &GpuSum<W>) -> Result<Option<usize>, GpuError> {
        if !self.timing() {
            return Ok(None);
        }
        if self.next_event == self.events.len() {
            self.events
                .push(sum.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?);
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
        let s = &sum.stream;
        let o = sum.device();
        let b = sum.hash.num_buckets();
        let e = table.entries;
        let key = (sum.hash.bits(), table.bucket_deltas());
        // The position map depends on the bucket count and the deltas alone, so a steady-state layer reuses the resident one.
        let map_stale = self.bucket_at_key.as_ref() != Some(&key);
        if map_stale {
            let span = Gf2Span::new(&key.1, key.0);
            self.bucket_at_host.clear();
            self.bucket_at_host.resize(b, 0);
            for beta in 0..b as u32 {
                self.bucket_at_host[span.perm_index(beta) as usize] = beta;
            }
            grow(s, &mut self.bucket_at, b, o)?;
        }
        grow(s, &mut self.cnt, b * e, o)?;
        grow(s, &mut self.rows, b, o)?;
        grow(s, &mut self.seg_start, b + 1, o)?;
        grow(s, &mut self.dst_off, b + 1, o)?;
        // A pageable upload synchronizes the stream first, so the copy's wall includes any refine still running; the timed region starts after its own sync.
        #[cfg(feature = "phase-timing")]
        s.synchronize()?;
        let (bucket_at_host, bucket_at) = (&self.bucket_at_host, &mut self.bucket_at);
        let (amp, mask, nz, bd, gm, rem) = (
            &mut self.amp,
            &mut self.mask,
            &mut self.nz,
            &mut self.bd,
            &mut self.gm,
            &mut self.rem,
        );
        xfer(&mut self.xfer_ns, Xfer::H2d, || {
            if map_stale {
                s.memcpy_htod(bucket_at_host, &mut bucket_at.slice_mut(0..b))?;
            }
            s.memcpy_htod(&table.amp, amp)?;
            s.memcpy_htod(&table.mask, mask)?;
            s.memcpy_htod(&table.nz, nz)?;
            s.memcpy_htod(&table.bucket_delta, bd)?;
            s.memcpy_htod(&table.gm, gm)?;
            s.memcpy_htod(&table.rem, rem)?;
            Ok(())
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
        let s = &sum.stream;
        let b = sum.hash.num_buckets();
        let (b32, e32) = (b as u32, table.entries as u32);
        let t0 = self.event(sum)?;
        // SAFETY: arguments match `k_count` in count.cu; `cnt` holds `b * e` entries.
        unsafe {
            s.launch_builder(&sum.kernels.count)
                .arg(&sum.cols.x)
                .arg(&sum.cols.z)
                .arg(&sum.cols.start)
                .arg(&sum.cols.lens)
                .arg(&b32)
                .arg(&table.mode)
                .arg(&e32)
                .arg(&table.kq)
                .arg(&table.q0)
                .arg(&table.q1)
                .arg(&table.rot_cos)
                .arg(&table.rot_sin)
                .arg(&self.amp)
                .arg(&self.mask)
                .arg(&self.nz)
                .arg(&mut self.cnt)
                .launch(warp_per_bucket(b))?;
        }
        self.lap(sum, t0, |m| &mut m.count)
    }

    /// K2 and its scans, over the local counts and the received offsets; returns `(records total, records max, longest local bucket)`.
    fn sizes(
        &mut self,
        sum: &GpuSum<W>,
        table: &DevicePrepared<W>,
    ) -> Result<(u32, u32, u32), GpuError> {
        let s = &sum.stream;
        let k = &sum.kernels;
        let b = sum.hash.num_buckets();
        let (b32, e32) = (b as u32, table.entries as u32);
        let t1 = self.event(sum)?;
        // SAFETY: arguments match `k_rows` in count.cu; `recv_off` holds `K × (b + 1)` entries for the `K` received entries `rem` names.
        unsafe {
            s.launch_builder(&k.rows)
                .arg(&self.cnt)
                .arg(&self.bucket_at)
                .arg(&self.bd)
                .arg(&self.rem)
                .arg(&self.export.recv_off)
                .arg(&mut self.rows)
                .arg(&b32)
                .arg(&e32)
                .launch(thread_per(b, 1024))?;
        }
        exclusive_scan_with_max_into(
            s,
            k,
            &self.rows.slice(0..b),
            &mut self.seg_start.slice_mut(0..b + 1),
            b,
            &mut self.scan,
            &mut self.tot_a,
        )?;
        exclusive_scan_with_max_into(
            s,
            k,
            &sum.cols.lens.slice(0..b),
            &mut self.dst_off.slice_mut(0..b + 1),
            b,
            &mut self.scan,
            &mut self.tot_b,
        )?;
        self.lap(sum, t1, |m| &mut m.sizes)?;
        // Downloads wait for the scans, so the timed region again starts after its own sync.
        #[cfg(feature = "phase-timing")]
        s.synchronize()?;
        self.seg_host.resize(b + 1, 0);
        let (seg_start, seg_host) = (&self.seg_start, &mut self.seg_host);
        let (tot_a, tot_b) = (&self.tot_a, &self.tot_b);
        let (tm, lm) = xfer(&mut self.xfer_ns, Xfer::D2h, || {
            let tm = s.clone_dtoh(tot_a)?;
            let lm = s.clone_dtoh(tot_b)?;
            s.memcpy_dtoh(&seg_start.slice(0..b + 1), &mut seg_host[..])?;
            s.synchronize()?;
            Ok((tm, lm))
        })?;
        Ok((tm[0], tm[1], lm[1]))
    }
}

/// Apply `prep` to `sum` on its device under `keep`, leaving the previous columns as `sum.spare`.
/// `target_bits` is the bucket count the group settled on; a lone partition refines to it and to its bucket policy in one pass, a partition of a group runs at exactly `target_bits` and reports `Unsupported` rather than refine off-schedule.
/// A layer with remote deltas exports through K10, exchanges over `transport`, and merges the received rows in the fused layer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_layer_device<const W: usize, X: Transport>(
    sum: &mut GpuSum<W>,
    prep: &Prepared<W>,
    plan: &PartitionPlan,
    rows: &PartitionRows<W>,
    keep: &KeepProgram,
    scratch: &mut LayerScratch<W>,
    target_bits: u8,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    let r = apply_layer_body(sum, prep, plan, rows, keep, scratch, target_bits, transport);
    let resolved = scratch.resolve(sum);
    r.and_then(|c| resolved.map(|()| c))
}

#[allow(clippy::too_many_arguments)]
fn apply_layer_body<const W: usize, X: Transport>(
    sum: &mut GpuSum<W>,
    prep: &Prepared<W>,
    plan: &PartitionPlan,
    rows: &PartitionRows<W>,
    keep: &KeepProgram,
    scratch: &mut LayerScratch<W>,
    target_bits: u8,
    transport: &X,
) -> Result<LayerExchangeCounts, GpuError> {
    let size = transport.size();
    scratch.next_event = 0;
    let fp = FingerprintRows::<W>::new(sum.hash.seed());
    let mut table = DevicePrepared::new(prep, &sum.hash, &fp, &plan.remote);
    let has_remote = plan.has_remote();
    debug_assert_eq!(table.n_remote, plan.remote.len());
    scratch.counters = GpuLayerCounters {
        bits: sum.hash.bits(),
        dense: table.dense,
        ..GpuLayerCounters::default()
    };
    scratch.export.recv_rows = 0;
    scratch.export.recv_max_segment = 0;
    let refine =
        |sum: &mut GpuSum<W>, scratch: &mut LayerScratch<W>, bits: u8| -> Result<(), GpuError> {
            let t = scratch.event(sum)?;
            sum.refine_to(bits)?;
            scratch.lap(sum, t, |m| &mut m.refine)?;
            scratch.extent = sum.len();
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
    // In a group every layer runs at exactly the agreed count and nobody refines off-schedule (ARCHITECTURE.md §Partitioning); the device steers the count through `proposed_bits` alone.
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
    let cap = sum.kernels.layer_cap();
    let mut counts = LayerExchangeCounts::none(size);
    let (records, records_max) = if has_remote {
        if let Err(e) = scratch
            .upload_table(sum, &table)
            .and_then(|()| scratch.count_local(sum, &table))
        {
            pair(scratch, sum.hash.bits());
            return Err(e);
        }
        counts = exchange_rows(sum, &table, plan, rows, scratch, transport)?;
        let (total, max_seg, max_len) = scratch.sizes(sum, &table)?;
        if max_seg as usize > cap
            || max_len as usize > MAX_BUCKET_LEN
            || scratch.export.recv_max_segment > MAX_RECV_SEGMENT
        {
            return Err(GpuError::Unsupported(
                "a fused-layer block or a received segment exceeds the record cap at the agreed bucket count",
            ));
        }
        (total, max_seg)
    } else if solo {
        // Oversize blocks and over-long source buckets are known from the count table; one more bit halves both.
        loop {
            scratch.upload_table(sum, &table)?;
            scratch.count_local(sum, &table)?;
            let (total, max_seg, max_len) = scratch.sizes(sum, &table)?;
            if max_seg as usize <= cap && max_len as usize <= MAX_BUCKET_LEN {
                break (total, max_seg);
            }
            if sum.hash.bits() >= max_bits {
                return Err(GpuError::Unsupported(
                    "a fused-layer block exceeds the record cap at the bucket-bit limit",
                ));
            }
            refine(sum, scratch, sum.hash.bits() + 1)?;
            table.rehash(&sum.hash);
        }
    } else {
        scratch.upload_table(sum, &table)?;
        scratch.count_local(sum, &table)?;
        let (total, max_seg, max_len) = scratch.sizes(sum, &table)?;
        if max_seg as usize > cap || max_len as usize > MAX_BUCKET_LEN {
            return Err(GpuError::Unsupported(
                "a fused-layer block exceeds the record cap at the agreed bucket count",
            ));
        }
        (total, max_seg)
    };
    let bits = sum.hash.bits();
    let b = sum.hash.num_buckets();
    scratch.counters.bits = bits;
    scratch.counters.records = u64::from(records);
    scratch.counters.records_max = records_max;
    scratch.counters.rows_received = counts.rows_received;

    let kernels = sum.kernels.clone();
    let (func, n_cap, smem) = fused_variant(&kernels, W, records_max as usize, table.dense);
    scratch.counters.n_cap = n_cap as u32;

    // Batches: contiguous position ranges whose pre-dedup rows fit the arena.
    let (batches, max_batch_rows) =
        arena_batches::<W>(&scratch.seg_host, scratch.options.arena_bytes, cap);
    scratch.counters.batches = batches.len() as u32;

    let s = sum.stream.clone();
    let k: Arc<KernelSet> = sum.kernels.clone();
    let o = sum.device();
    let mut arena = match scratch.arena.take() {
        Some(a) => a,
        None => DeviceColumns::<W>::with_capacity(&s, o, max_batch_rows, 1)?,
    };
    arena.len = 0;
    arena.buckets = 0;
    arena.reserve(max_batch_rows, 1)?;
    let mut out = match sum.spare.take() {
        Some(spare) => spare,
        None => DeviceColumns::<W>::with_capacity(&s, o, n_in, b)?,
    };
    out.len = 0;
    out.buckets = 0;
    out.reserve(out.term_capacity(), b)?;
    grow(&s, &mut scratch.out_len_pos, b, o)?;
    s.memset_zeros(&mut scratch.fallback)?;
    let mut running = 0u32;
    for &(p0, p1) in &batches {
        let nblk = (p1 - p0) as u32;
        let p0u = p0 as u32;
        let t0 = scratch.event(sum)?;
        let fused = FusedTable {
            cnt: &scratch.cnt,
            seg_start: &scratch.seg_start,
            amp: &scratch.amp,
            mask: &scratch.mask,
            nz: &scratch.nz,
            bd: &scratch.bd,
            gm: &scratch.gm,
            rem: &scratch.rem,
        };
        let recv = FusedRecv {
            off: &scratch.export.recv_off,
            base: &scratch.export.recv_base,
            x: &scratch.export.recv_x,
            z: &scratch.export.recv_z,
            c: &scratch.export.recv_c,
            g: &scratch.export.recv_g,
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
            (p0u, nblk),
            written,
        )?;
        scratch.lap(sum, t0, |m| &mut m.layer)?;
        let n = p1 - p0;
        let t1 = scratch.event(sum)?;
        exclusive_scan_with_max_into(
            &s,
            &k,
            &scratch.out_len_pos.slice(p0..p1),
            &mut scratch.dst_off.slice_mut(0..n + 1),
            n,
            &mut scratch.scan,
            &mut scratch.tot_a,
        )?;
        // The download would otherwise absorb the wait for the fused kernel.
        #[cfg(feature = "phase-timing")]
        s.synchronize()?;
        let tot = &scratch.tot_a;
        let batch_out = xfer(&mut scratch.xfer_ns, Xfer::D2h, || {
            let v = s.clone_dtoh(tot)?;
            s.synchronize()?;
            Ok(v[0])
        })?;
        out.len = running as usize;
        let need = (running + batch_out) as usize;
        if need > out.term_capacity() {
            // Geometric growth, capped by the pre-dedup total no output can exceed.
            let grown = (2 * out.term_capacity())
                .max(need)
                .min((records as usize).max(need));
            out.reserve(grown, b)?;
        }
        // SAFETY: arguments match `k_compact` in compact.cu; `out` holds `running + batch_out` terms.
        unsafe {
            s.launch_builder(&k.compact)
                .arg(&arena.x)
                .arg(&arena.z)
                .arg(&arena.coeff)
                .arg(&arena.g)
                .arg(&scratch.seg_start)
                .arg(&scratch.dst_off)
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
                .launch(thread_per(nblk as usize * 256, 256))?;
        }
        scratch.lap(sum, t1, |m| &mut m.compact)?;
        running += batch_out;
    }
    let fb = s.clone_dtoh(&scratch.fallback)?;
    s.synchronize()?;
    scratch.counters.fallback_hi += fb[0];
    scratch.counters.fallback_key += fb[1];
    out.len = running as usize;
    out.buckets = b;
    scratch.arena = Some(arena);
    sum.spare = Some(std::mem::replace(&mut sum.cols, out));
    scratch.extent = sum.len();
    sum.debug_check();
    Ok(counts)
}

/// Contiguous position ranges whose pre-dedup rows (`seg`, `b + 1` CSR offsets) fit an arena of `arena_bytes`, and the largest range's rows.
pub(super) fn arena_batches<const W: usize>(
    seg: &[u32],
    arena_bytes: usize,
    cap: usize,
) -> (Vec<(usize, usize)>, usize) {
    let b = seg.len() - 1;
    let cap_rows = (arena_bytes / DeviceColumns::<W>::BYTES_PER_TERM).max(cap);
    let mut batches: Vec<(usize, usize)> = Vec::new();
    let mut p0 = 0usize;
    while p0 < b {
        let mut p1 = p0 + 1;
        while p1 < b && (seg[p1 + 1] - seg[p0]) as usize <= cap_rows {
            p1 += 1;
        }
        batches.push((p0, p1));
        p0 = p1;
    }
    let max_rows = batches
        .iter()
        .map(|&(a, c)| (seg[c] - seg[a]) as usize)
        .max()
        .unwrap_or(0);
    (batches, max_rows)
}

/// The fused-layer variant for blocks of up to `records_max` records: the kernel, its record capacity and its dynamic shared bytes.
pub(super) fn fused_variant(
    k: &KernelSet,
    w: usize,
    records_max: usize,
    dense: bool,
) -> (&CudaFunction, usize, u32) {
    let threads = layer_threads(w) as usize;
    let n_cap = records_max.max(threads).next_power_of_two();
    let variant = &k.layer[(n_cap / threads).trailing_zeros() as usize];
    debug_assert_eq!(variant.items * threads, n_cap);
    let func = if dense {
        &variant.segscan
    } else {
        &variant.serial
    };
    (func, n_cap, layer_shared_bytes(n_cap, w))
}

/// The table-side buffers one fused-layer launch reads, in the uploaded layout of a [`DevicePrepared`].
pub(super) struct FusedTable<'a> {
    pub(super) cnt: &'a CudaSlice<u32>,
    pub(super) seg_start: &'a CudaSlice<u32>,
    pub(super) amp: &'a CudaSlice<f64>,
    pub(super) mask: &'a CudaSlice<u64>,
    pub(super) nz: &'a CudaSlice<u32>,
    pub(super) bd: &'a CudaSlice<u32>,
    pub(super) gm: &'a CudaSlice<u64>,
    pub(super) rem: &'a CudaSlice<u32>,
}

/// The concatenated received blocks a fused-layer launch reads for its received entries.
pub(super) struct FusedRecv<'a> {
    pub(super) off: &'a CudaSlice<u32>,
    pub(super) base: &'a CudaSlice<u32>,
    pub(super) x: &'a CudaSlice<u64>,
    pub(super) z: &'a CudaSlice<u64>,
    pub(super) c: &'a CudaSlice<f64>,
    pub(super) g: &'a CudaSlice<u64>,
}

/// What a fused-layer launch writes: the loose arena, rows per position, and the fallback counters.
pub(super) struct FusedOut<'a, const W: usize> {
    pub(super) arena: &'a mut DeviceColumns<W>,
    pub(super) out_len_pos: &'a mut CudaSlice<u32>,
    pub(super) fallback: &'a mut CudaSlice<u32>,
}

/// K3 over positions `p0..p0 + nblk` with `kernel = (function, shared bytes)` from [`fused_variant`].
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_fused<const W: usize>(
    sum: &GpuSum<W>,
    table: &DevicePrepared<W>,
    t: &FusedTable<'_>,
    r: &FusedRecv<'_>,
    bucket_at: &CudaSlice<u32>,
    keep: &KeepProgram,
    kernel: (&CudaFunction, u32),
    (p0, nblk): (u32, u32),
    out: FusedOut<'_, W>,
) -> Result<(), GpuError> {
    let (func, smem) = kernel;
    let e32 = table.entries as u32;
    let b32 = sum.hash.num_buckets() as u32;
    let FusedOut {
        arena,
        out_len_pos,
        fallback,
    } = out;
    // SAFETY: arguments match the `LAYER_KERNEL` signature in layer.cu; the arena holds the batch's pre-dedup rows, `out_len_pos` has `b` entries, `t` is `table`'s upload with `cnt`/`seg_start` sized for it, and the received columns hold every row `t.rem` names.
    unsafe {
        sum.stream
            .launch_builder(func)
            .arg(&sum.cols.x)
            .arg(&sum.cols.z)
            .arg(&sum.cols.coeff)
            .arg(&sum.cols.g)
            .arg(&sum.cols.start)
            .arg(&sum.cols.lens)
            .arg(bucket_at)
            .arg(t.cnt)
            .arg(t.seg_start)
            .arg(&table.mode)
            .arg(&e32)
            .arg(&table.kq)
            .arg(&table.q0)
            .arg(&table.q1)
            .arg(&table.rot_cos)
            .arg(&table.rot_sin)
            .arg(t.amp)
            .arg(t.mask)
            .arg(t.nz)
            .arg(t.bd)
            .arg(t.gm)
            .arg(t.rem)
            .arg(r.off)
            .arg(r.base)
            .arg(&b32)
            .arg(r.x)
            .arg(r.z)
            .arg(r.c)
            .arg(r.g)
            .arg(keep)
            .arg(&p0)
            .arg(&mut arena.x)
            .arg(&mut arena.z)
            .arg(&mut arena.coeff)
            .arg(&mut arena.g)
            .arg(out_len_pos)
            .arg(fallback)
            .launch(LaunchConfig {
                grid_dim: (nblk, 1, 1),
                block_dim: (layer_threads(W), 1, 1),
                shared_mem_bytes: smem,
            })?;
    }
    Ok(())
}

/// K5: the key-preserving fast path, into the spare columns at the input's offsets.
fn rescale_device<const W: usize>(
    sum: &mut GpuSum<W>,
    table: &DevicePrepared<W>,
    keep: &KeepProgram,
    scratch: &mut LayerScratch<W>,
) -> Result<(), GpuError> {
    let s = sum.stream.clone();
    let k = sum.kernels.clone();
    let o = sum.device();
    let b = sum.hash.num_buckets();
    let extent = scratch.extent.max(sum.len());
    let mut out = match sum.spare.take() {
        Some(spare) => spare,
        None => DeviceColumns::<W>::with_capacity(&s, o, extent, b)?,
    };
    out.len = 0;
    out.buckets = 0;
    out.reserve(extent, b)?;
    grow(&s, &mut scratch.dst_off, b + 1, o)?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    let amp = &mut scratch.amp;
    xfer(&mut scratch.xfer_ns, Xfer::H2d, || {
        s.memcpy_htod(&table.amp, amp)?;
        Ok(())
    })?;
    let b32 = b as u32;
    let t0 = scratch.event(sum)?;
    // SAFETY: arguments match `k_rescale` in rescale.cu; `out` has room for the input's extent.
    unsafe {
        s.launch_builder(&k.rescale)
            .arg(&sum.cols.x)
            .arg(&sum.cols.z)
            .arg(&sum.cols.coeff)
            .arg(&sum.cols.g)
            .arg(&sum.cols.start)
            .arg(&sum.cols.lens)
            .arg(&b32)
            .arg(&table.kq)
            .arg(&table.q0)
            .arg(&table.q1)
            .arg(&scratch.amp)
            .arg(keep)
            .arg(&mut out.x)
            .arg(&mut out.z)
            .arg(&mut out.coeff)
            .arg(&mut out.g)
            .arg(&mut out.start)
            .arg(&mut out.lens)
            .launch(warp_per_bucket(b))?;
    }
    exclusive_scan_with_max_into(
        &s,
        &k,
        &out.lens.slice(0..b),
        &mut scratch.dst_off.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut scratch.tot_a,
    )?;
    scratch.lap(sum, t0, |m| &mut m.rescale)?;
    #[cfg(feature = "phase-timing")]
    s.synchronize()?;
    let tot = &scratch.tot_a;
    let total = xfer(&mut scratch.xfer_ns, Xfer::D2h, || {
        let v = s.clone_dtoh(tot)?;
        s.synchronize()?;
        Ok(v[0])
    })?;
    out.len = total as usize;
    out.buckets = b;
    sum.spare = Some(std::mem::replace(&mut sum.cols, out));
    scratch.counters.rescaled = true;
    sum.debug_check();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket::hash::Gf2Hash;
    use crate::channel::{Channel, GeneralUnitary2Q};
    use crate::engine::partitioned::transport::InProcessTransport;
    use crate::test_support::{
        assert_terms_close, haar_su4_matrix, naive_apply_layer, rand_sum, KeepAll,
    };
    use num_complex::Complex64;

    /// `SWAP·CNOT` as a matrix: its symplectic map has no fixed nonzero vector, so all 16 deltas are realized and every row emits exactly one record.
    fn sixteen_delta_permutation() -> GeneralUnitary2Q {
        let e = |i: usize| -> [Complex64; 4] {
            let mut r = [Complex64::new(0.0, 0.0); 4];
            r[i] = Complex64::new(1.0, 0.0);
            r
        };
        GeneralUnitary2Q::from_matrix(0, 1, [e(0), e(3), e(1), e(2)])
    }

    /// One layer on a single-bucket device sum under `opts`, against the naive oracle; returns the counters.
    /// With `x0`, every term carries `X` on qubit 0, so a two-qubit table on `(0, 1)` never sees the identity pattern and every entry emits for every row.
    fn single_bucket_layer(
        n: usize,
        x0: bool,
        ch: &dyn Channel<2>,
        opts: GpuLayerOptions,
    ) -> Result<GpuLayerCounters, GpuError> {
        let mut input = rand_sum::<2>(n, 128, 0x4096 + n as u64);
        if x0 {
            let mut acc = crate::accumulator::BuildAccumulator::<2>::new(128);
            for (x, z, c) in input.iter() {
                let mut x = *x;
                x[0] |= 1;
                acc.add_term(
                    crate::pauli_string::PauliString::<2> { x, z: *z },
                    crate::phase::Phase::ONE,
                    c,
                );
            }
            input = acc.finalize();
        }
        let input = input.with_hash(Gf2Hash::new(128, 0, crate::bucket::sum::DEFAULT_HASH_SEED));
        assert_eq!(input.len(), n);
        let mut sum = GpuSum::from_host(&input, 0)?;
        let mut scratch = LayerScratch::new(&sum, opts)?;
        let prep = ch.prepare(sum.hash(), false).expect("prepared");
        let rows = PartitionRows::<2>::none(128);
        let plan = PartitionPlan::new(&prep, &rows, 0);
        let solo = InProcessTransport::group(1);
        apply_layer_device(
            &mut sum,
            &prep,
            &plan,
            &rows,
            &KeepProgram::KEEP,
            &mut scratch,
            0,
            &solo[0],
        )?;
        let want = naive_apply_layer(&input, ch, &KeepAll, false);
        assert_terms_close(&sum.to_host()?, &want, 1e-11, "single bucket");
        Ok(scratch.counters)
    }

    #[test]
    fn a_full_source_bucket_and_a_full_block_fit_and_one_more_row_refines() {
        crate::require_cuda!();
        let opts = GpuLayerOptions {
            bucket_policy: GpuBucketPolicy::TermsPerBucket(1 << 20),
            ..GpuLayerOptions::default()
        };
        let perm = sixteen_delta_permutation();
        let c = single_bucket_layer(MAX_BUCKET_LEN, false, &perm, opts).unwrap();
        assert_eq!(
            (c.bits, c.refine_passes, c.records, c.records_max),
            (0, 0, 4096, 4096)
        );
        let c = single_bucket_layer(MAX_BUCKET_LEN + 1, false, &perm, opts).unwrap();
        assert!(c.refine_passes > 0 && c.bits > 0, "{c:?}");

        // A Haar SU(4) row with a non-identity pattern emits 15 records: the one entry mapping it onto `I⊗I` is exactly zero.
        let su4 = GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix());
        let cap = crate::engine::gpu::module::kernel_set(0, 2)
            .unwrap()
            .layer_cap();
        let c = single_bucket_layer(cap / 15, true, &su4, opts).unwrap();
        assert_eq!(
            (
                c.bits,
                c.refine_passes,
                c.records_max as usize,
                c.n_cap as usize
            ),
            (0, 0, 15 * (cap / 15), cap)
        );
        let c = single_bucket_layer(cap / 15 + 1, true, &su4, opts).unwrap();
        assert!(c.refine_passes > 0 && c.bits > 0, "{c:?}");
        let capped = GpuLayerOptions {
            max_bits: 0,
            ..opts
        };
        assert!(matches!(
            single_bucket_layer(cap / 15 + 1, true, &su4, capped),
            Err(GpuError::Unsupported(_))
        ));
    }

    #[test]
    fn records_per_block_policy_scales_with_fanout_and_is_grow_only() {
        let p = GpuBucketPolicy::RecordsPerBlock(4096);
        // 1e6 terms: fanout 1 wants 2^8 buckets (3906 records), fanout 2 one more bit, fanout 16 four more.
        assert_eq!(gpu_desired_bits(1_000_000, 1, p, 0), 8);
        assert_eq!(gpu_desired_bits(1_000_000, 2, p, 0), 9);
        assert_eq!(gpu_desired_bits(1_000_000, 16, p, 0), 12);
        assert_eq!(
            gpu_desired_bits(1_000_000, 1, p, 11),
            11,
            "never below the current bits"
        );
        assert_eq!(gpu_desired_bits(0, 16, p, 0), 0);
        assert_eq!(gpu_desired_bits(4096, 1, p, 0), 0);
        assert_eq!(gpu_desired_bits(4097, 1, p, 0), 1);
        assert_eq!(gpu_desired_bits(usize::MAX / 2, 16, p, 0), B_MAX_BITS);
        let fixed = GpuBucketPolicy::TermsPerBucket(256);
        assert_eq!(gpu_desired_bits(1_000_000, 16, fixed, 0), 12);
        assert_eq!(gpu_desired_bits(1_000_000, 1, fixed, 0), 12);
        assert_eq!(gpu_desired_bits(256, 1, fixed, 0), 0);
        assert_eq!(gpu_desired_bits(257, 1, fixed, 0), 1);
    }
}
