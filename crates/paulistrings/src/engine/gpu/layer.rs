//! One layer on the device: bucket policy, count, fused layer, compaction, and the key-preserving rescale. See ARCHITECTURE.md §Engine and §GPU-Readiness.

use std::sync::Arc;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg};

use super::columns::DeviceColumns;
use super::error::GpuError;
use super::fingerprint::FingerprintRows;
use super::module::{layer_shared_bytes, layer_threads, KernelSet, MAX_BUCKET_LEN};
use super::prepared::DevicePrepared;
use super::scan::exclusive_scan_with_max;
use super::sum::GpuSum;
use super::truncation::KeepProgram;
use crate::bucket::hash::B_MAX_BITS;
use crate::bucket::sum::desired_bits;
use crate::channel::prepared::Prepared;
use crate::engine::coset::Gf2Span;
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
}

impl Default for GpuLayerOptions {
    fn default() -> Self {
        Self {
            bucket_policy: GpuBucketPolicy::default(),
            arena_bytes: DEFAULT_ARENA_BYTES,
            max_bits: B_MAX_BITS,
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
    /// Blocks that fell back to the `g_hi32` passes.
    pub fallback_hi: u32,
    /// Blocks that fell back to the full-key sort.
    pub fallback_key: u32,
    /// The layer took the rescale path.
    pub rescaled: bool,
    /// The layer reduced by the segmented scan rather than the head-serial walk.
    pub dense: bool,
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
}

/// Grow-only device buffers one partition keeps between layers.
pub(crate) struct LayerScratch<const W: usize> {
    bucket_at: CudaSlice<u32>,
    cnt: CudaSlice<u32>,
    rows: CudaSlice<u32>,
    seg_start: CudaSlice<u32>,
    out_len_pos: CudaSlice<u32>,
    pub(super) dst_off: CudaSlice<u32>,
    /// K7's `[len, bins…]`.
    pub(super) hist: CudaSlice<u64>,
    fallback: CudaSlice<u32>,
    amp: CudaSlice<f64>,
    mask: CudaSlice<u64>,
    nz: CudaSlice<u32>,
    bd: CudaSlice<u32>,
    gm: CudaSlice<u64>,
    arena: Option<DeviceColumns<W>>,
    seg_host: Vec<u32>,
    bucket_at_host: Vec<u32>,
    /// Highest live row index plus one; differs from `len` only after a rescale, whose output keeps the input's offsets.
    pub(crate) extent: usize,
    pub(crate) options: GpuLayerOptions,
    pub(crate) counters: GpuLayerCounters,
    pub(crate) time_kernels: bool,
    pub(crate) kernel_ms: GpuKernelMs,
    /// Timing events, reused across layers.
    events: Vec<CudaEvent>,
    next_event: usize,
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

fn thread_per(n: usize, threads: u32) -> LaunchConfig {
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
            arena: None,
            seg_host: Vec::new(),
            bucket_at_host: Vec::new(),
            extent: sum.len(),
            options,
            counters: GpuLayerCounters::default(),
            time_kernels: false,
            kernel_ms: GpuKernelMs::default(),
            events: Vec::new(),
            next_event: 0,
        })
    }

    pub(super) fn event(&mut self, sum: &GpuSum<W>) -> Result<Option<usize>, GpuError> {
        if !self.time_kernels {
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

    pub(super) fn lap(
        &mut self,
        sum: &GpuSum<W>,
        from: Option<usize>,
        into: fn(&mut GpuKernelMs) -> &mut f64,
    ) -> Result<(), GpuError> {
        let Some(i) = from else { return Ok(()) };
        let Some(j) = self.event(sum)? else {
            return Ok(());
        };
        sum.stream.synchronize()?;
        let ms = f64::from(self.events[i].elapsed_ms(&self.events[j])?);
        *into(&mut self.kernel_ms) += ms;
        Ok(())
    }

    /// Upload the table and the position map for `bits`, then run K1 and K2; returns `(records total, records max, longest bucket)`.
    fn count(
        &mut self,
        sum: &GpuSum<W>,
        table: &DevicePrepared<W>,
    ) -> Result<(u32, u32, u32), GpuError> {
        let s = &sum.stream;
        let k = &sum.kernels;
        let o = sum.device();
        let b = sum.hash.num_buckets();
        let e = table.entries;
        let span = Gf2Span::new(&table.bucket_deltas(), sum.hash.bits());
        self.bucket_at_host.clear();
        self.bucket_at_host.resize(b, 0);
        for beta in 0..b as u32 {
            self.bucket_at_host[span.perm_index(beta) as usize] = beta;
        }
        grow(s, &mut self.bucket_at, b, o)?;
        grow(s, &mut self.cnt, b * e, o)?;
        grow(s, &mut self.rows, b, o)?;
        grow(s, &mut self.seg_start, b + 1, o)?;
        grow(s, &mut self.dst_off, b + 1, o)?;
        s.memcpy_htod(&self.bucket_at_host, &mut self.bucket_at.slice_mut(0..b))?;
        s.memcpy_htod(&table.amp, &mut self.amp)?;
        s.memcpy_htod(&table.mask, &mut self.mask)?;
        s.memcpy_htod(&table.nz, &mut self.nz)?;
        s.memcpy_htod(&table.bucket_delta, &mut self.bd)?;
        s.memcpy_htod(&table.gm, &mut self.gm)?;
        let (b32, e32) = (b as u32, e as u32);
        let t0 = self.event(sum)?;
        // SAFETY: arguments match `k_count` in count.cu; `cnt` holds `b * e` entries.
        unsafe {
            s.launch_builder(&k.count)
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
        self.lap(sum, t0, |m| &mut m.count)?;
        let t1 = self.event(sum)?;
        // SAFETY: arguments match `k_rows` in count.cu.
        unsafe {
            s.launch_builder(&k.rows)
                .arg(&self.cnt)
                .arg(&self.bucket_at)
                .arg(&self.bd)
                .arg(&mut self.rows)
                .arg(&b32)
                .arg(&e32)
                .launch(thread_per(b, 1024))?;
        }
        let tot_max = exclusive_scan_with_max(
            s,
            k,
            &self.rows.slice(0..b),
            &mut self.seg_start.slice_mut(0..b + 1),
            b,
        )?;
        let lens_max = exclusive_scan_with_max(
            s,
            k,
            &sum.cols.lens.slice(0..b),
            &mut self.dst_off.slice_mut(0..b + 1),
            b,
        )?;
        self.lap(sum, t1, |m| &mut m.sizes)?;
        let tm = s.clone_dtoh(&tot_max)?;
        let lm = s.clone_dtoh(&lens_max)?;
        self.seg_host.resize(b + 1, 0);
        s.memcpy_dtoh(&self.seg_start.slice(0..b + 1), &mut self.seg_host[..])?;
        s.synchronize()?;
        Ok((tm[0], tm[1], lm[1]))
    }
}

/// Apply `prep` to `sum` on its device under `keep`, leaving the previous columns as `sum.spare`.
/// `target_bits` is the bucket count the driver settled on; the device refines to it and to the bucket policy in one pass.
pub(crate) fn apply_layer_device<const W: usize>(
    sum: &mut GpuSum<W>,
    prep: &Prepared<W>,
    keep: &KeepProgram,
    scratch: &mut LayerScratch<W>,
    target_bits: u8,
) -> Result<(), GpuError> {
    scratch.next_event = 0;
    let fp = FingerprintRows::<W>::new(sum.hash.seed());
    let mut table = DevicePrepared::new(prep, &sum.hash, &fp);
    scratch.counters = GpuLayerCounters {
        bits: sum.hash.bits(),
        dense: table.dense,
        ..GpuLayerCounters::default()
    };
    let max_bits = scratch.options.max_bits.min(B_MAX_BITS);
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
        if target_bits > sum.hash.bits() {
            refine(sum, scratch, target_bits)?;
        }
        return rescale_device(sum, &table, keep, scratch);
    }
    let n_in = sum.len();
    if n_in.saturating_mul(table.fanout.max(1)) >= u32::MAX as usize {
        return Err(GpuError::Unsupported(
            "more than 2^32 pre-dedup records in one layer",
        ));
    }
    let want = gpu_desired_bits(
        n_in,
        table.fanout,
        scratch.options.bucket_policy,
        sum.hash.bits(),
    )
    .min(max_bits)
    .max(target_bits);
    if want > sum.hash.bits() {
        refine(sum, scratch, want)?;
        table.rehash(&sum.hash);
    }
    // Oversize blocks and over-long source buckets are known from the count table; one more bit halves both.
    let cap = sum.kernels.layer_cap();
    let (records, records_max) = loop {
        let (total, max_seg, max_len) = scratch.count(sum, &table)?;
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
    };
    let bits = sum.hash.bits();
    let b = sum.hash.num_buckets();
    scratch.counters.bits = bits;
    scratch.counters.records = u64::from(records);
    scratch.counters.records_max = records_max;

    let threads = layer_threads(W) as usize;
    let n_cap = (records_max as usize).max(threads).next_power_of_two();
    let variant = &sum.kernels.layer[(n_cap / threads).trailing_zeros() as usize];
    debug_assert_eq!(variant.items * threads, n_cap);
    let func = if table.dense {
        &variant.segscan
    } else {
        &variant.serial
    };
    let smem = layer_shared_bytes(n_cap, W);
    scratch.counters.n_cap = n_cap as u32;

    // Batches: contiguous position ranges whose pre-dedup rows fit the arena.
    let seg = &scratch.seg_host;
    let cap_rows = (scratch.options.arena_bytes / DeviceColumns::<W>::BYTES_PER_TERM).max(cap);
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
    let max_batch_rows = batches
        .iter()
        .map(|&(a, c)| (seg[c] - seg[a]) as usize)
        .max()
        .unwrap_or(0);
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
    let e32 = table.entries as u32;

    let mut running = 0u32;
    for &(p0, p1) in &batches {
        let nblk = (p1 - p0) as u32;
        let p0u = p0 as u32;
        let t0 = scratch.event(sum)?;
        // SAFETY: arguments match the `LAYER_KERNEL` signature in layer.cu; the arena holds this batch's pre-dedup rows and `out_len_pos` has `b` entries.
        unsafe {
            s.launch_builder(func)
                .arg(&sum.cols.x)
                .arg(&sum.cols.z)
                .arg(&sum.cols.coeff)
                .arg(&sum.cols.g)
                .arg(&sum.cols.start)
                .arg(&sum.cols.lens)
                .arg(&scratch.bucket_at)
                .arg(&scratch.cnt)
                .arg(&scratch.seg_start)
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
                .arg(&scratch.bd)
                .arg(&scratch.gm)
                .arg(keep)
                .arg(&p0u)
                .arg(&mut arena.x)
                .arg(&mut arena.z)
                .arg(&mut arena.coeff)
                .arg(&mut arena.g)
                .arg(&mut scratch.out_len_pos)
                .arg(&mut scratch.fallback)
                .launch(LaunchConfig {
                    grid_dim: (nblk, 1, 1),
                    block_dim: (threads as u32, 1, 1),
                    shared_mem_bytes: smem,
                })?;
        }
        scratch.lap(sum, t0, |m| &mut m.layer)?;
        let n = p1 - p0;
        let t1 = scratch.event(sum)?;
        let tot = exclusive_scan_with_max(
            &s,
            &k,
            &scratch.out_len_pos.slice(p0..p1),
            &mut scratch.dst_off.slice_mut(0..n + 1),
            n,
        )?;
        let batch_out = s.clone_dtoh(&tot)?[0];
        s.synchronize()?;
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
    scratch.counters.fallback_hi = fb[0];
    scratch.counters.fallback_key = fb[1];
    out.len = running as usize;
    out.buckets = b;
    scratch.arena = Some(arena);
    sum.spare = Some(std::mem::replace(&mut sum.cols, out));
    scratch.extent = sum.len();
    sum.debug_check();
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
    s.memcpy_htod(&table.amp, &mut scratch.amp)?;
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
    let tot = exclusive_scan_with_max(
        &s,
        &k,
        &out.lens.slice(0..b),
        &mut scratch.dst_off.slice_mut(0..b + 1),
        b,
    )?;
    scratch.lap(sum, t0, |m| &mut m.rescale)?;
    let total = s.clone_dtoh(&tot)?[0];
    s.synchronize()?;
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
        apply_layer_device(&mut sum, &prep, &KeepProgram::KEEP, &mut scratch, 0)?;
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
