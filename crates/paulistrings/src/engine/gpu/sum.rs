//! [`GpuSum`], a bucketed Pauli sum resident on one CUDA device, with its upload, download and refine.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, PushKernelArg};
use num_complex::Complex64;
use rayon::prelude::*;

use super::columns::DeviceColumns;
use super::device;
use super::error::GpuError;
use super::fingerprint::FingerprintRows;
use super::length::{Growth, LengthTracker, Lengths, LONE_LAG};
use super::module::{self, thread_per, warp_per_bucket, CoefficientLayout, KernelSet};
use super::scan::{exclusive_scan, ScanScratch};
use super::staging::HostStaging;
use crate::coefficient::Coefficient;
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::storage::BucketColumns;
use crate::pauli_sum::PauliSum;

/// Hash bits one refine pass adds at most; must match `REFINE_MAX_DELTA` in `kernels/refine.cu`.
pub(super) const REFINE_MAX_DELTA: u8 = 4;

/// The most bucket bits a device sum refines to: the scan kernel (`kernels/scan.cu`) takes at most `2^24` elements, and a layer scans one entry per bucket plus a sentinel.
pub const GPU_MAX_BITS: u8 = 23;

/// GF(2) rows in the device layout of `kernels/prelude.cuh`, padded so an empty matrix still allocates.
fn flat_rows<const W: usize>(rows: impl Iterator<Item = ([u64; W], [u64; W])>) -> Vec<u64> {
    let mut v = Vec::new();
    for (x, z) in rows {
        v.extend_from_slice(&x);
        v.extend_from_slice(&z);
    }
    if v.is_empty() {
        v.resize(2 * W, 0);
    }
    v
}

/// A bucketed Pauli sum resident on one CUDA device.
///
/// Within a bucket keys are unique in no particular order; [`Self::to_host`] restores the host order.
/// [`Self::from_host`] and [`Self::to_host`] round-trip bitwise, bucket by bucket, under the host sum's [`Gf2Hash`].
/// Resident device memory is `16·W + 8` bytes per term plus its coefficient (`size_of::<C>()`, 4 to 16 bytes) and `8` per bucket, doubled once a refine has run.
/// Every fallible operation returns a [`GpuError`], device allocation failure as [`GpuError::OutOfMemory`].
pub struct GpuSum<const W: usize, C: Coefficient = Complex64> {
    pub(super) context: Arc<CudaContext>,
    pub(super) stream: Arc<CudaStream>,
    pub(super) kernels: Arc<KernelSet>,
    pub(super) hash: Gf2Hash<W>,
    pub(super) num_qubits: usize,
    pub(super) columns: DeviceColumns<W, C>,
    /// The previous columns, kept as the next refine's or layer's target.
    pub(super) spare: Option<DeviceColumns<W, C>>,
    /// The rows of `hash` up to [`Self::max_bits`], so a refine uploads nothing.
    pub(super) hash_rows: CudaSlice<u64>,
    /// `FingerprintRows::new(hash.seed())`, and its rows in device layout.
    pub(super) fingerprints: FingerprintRows<W>,
    pub(super) fingerprint_rows: CudaSlice<u64>,
    /// The refine's scan buffers and `[total, max]`.
    scan: (ScanScratch, CudaSlice<u32>),
    /// The scan [`Self::settle_lengths`] writes, one entry per bucket plus one.
    lengths_scan: CudaSlice<u32>,
    /// The host's bound on `columns.len` and the longest bucket; `columns.len` is always its bound.
    pub(super) length: LengthTracker,
    /// Page-locked download staging, allocated on first [`Self::to_host`] and reused.
    staging: Mutex<HostStaging<C>>,
}

/// A context object of its own on `ordinal`'s primary context, with cudarc's event tracking off, so a launch costs no per-argument event wait and record (research/FINDINGS.md §Lone device sums skip cudarc's event tracking).
///
/// Contract: every buffer allocated through it is used on the one stream its sum creates, in stream order, so no event is needed; a group member's buffers meet other streams and stay on the tracked [`device::context`].
fn untracked_context(ordinal: u32) -> Result<Arc<CudaContext>, GpuError> {
    device::context(ordinal)?;
    let context = CudaContext::new(ordinal as usize)?;
    // SAFETY: no slice exists on the new context object yet, and the contract above keeps every later one on a single stream.
    unsafe { context.disable_event_tracking() };
    Ok(context)
}

/// `k_fingerprint`: the fingerprint of rows `0..n` of `(x, z)` into `g`, enqueued on `stream`.
pub(super) fn launch_fingerprint(
    stream: &Arc<CudaStream>,
    kernels: &KernelSet,
    fingerprint_rows: &CudaSlice<u64>,
    (x, z): (&CudaSlice<u64>, &CudaSlice<u64>),
    g: &mut CudaSlice<u64>,
    n: usize,
) -> Result<(), GpuError> {
    if n == 0 {
        return Ok(());
    }
    let n32 = n as u32;
    // SAFETY: arguments match `k_fingerprint` in fingerprint.cu; every column holds `n` rows.
    unsafe {
        stream
            .launch_builder(&kernels.fingerprint)
            .arg(x)
            .arg(z)
            .arg(&n32)
            .arg(fingerprint_rows)
            .arg(g)
            .launch(thread_per(n, 256))?;
    }
    Ok(())
}

impl<const W: usize, C: Coefficient> GpuSum<W, C> {
    /// Upload `sum` to device `ordinal`, bucket by bucket in the host's order, and compute every term's fingerprint on the device.
    /// The first call per `(ordinal, W, C)` compiles the kernels through NVRTC.
    pub fn from_host(sum: &PauliSum<W, C>, ordinal: u32) -> Result<Self, GpuError> {
        let kernels = module::kernel_set(ordinal, W, CoefficientLayout::of::<C>())?;
        Self::upload(sum, device::context(ordinal)?, kernels, false)
    }

    /// As [`Self::from_host`] with extra NVRTC options, the `-DFINGERPRINT_BITS=<b>` collision hook; `alone` puts the sum on [`untracked_context`], for a sum no other stream ever touches.
    pub(crate) fn from_host_with_options(
        sum: &PauliSum<W, C>,
        ordinal: u32,
        extra_options: &[String],
        alone: bool,
    ) -> Result<Self, GpuError> {
        let context = if alone { untracked_context(ordinal)? } else { device::context(ordinal)? };
        Self::upload(
            sum,
            context,
            module::kernel_set_with_options(
                ordinal,
                W,
                CoefficientLayout::of::<C>(),
                extra_options,
            )?,
            alone,
        )
    }

    fn upload(
        sum: &PauliSum<W, C>,
        context: Arc<CudaContext>,
        kernels: Arc<KernelSet>,
        alone: bool,
    ) -> Result<Self, GpuError> {
        let ordinal = context.ordinal() as u32;
        let stream = context.new_stream()?;
        let hash = sum.hash().clone();
        let (n, b) = (sum.len(), hash.num_buckets());
        let mut columns = DeviceColumns::<W, C>::with_capacity(&stream, ordinal, n, b)?;

        let mut x = Vec::with_capacity(n * W);
        let mut z = Vec::with_capacity(n * W);
        let mut c: Vec<C> = Vec::with_capacity(n);
        let mut start = Vec::with_capacity(b + 1);
        let mut lens = Vec::with_capacity(b);
        for bucket in 0..b {
            let (bucket_x, bucket_z, bucket_coeff) = sum.bucket(bucket);
            start.push((x.len() / W) as u32);
            lens.push(bucket_coeff.len() as u32);
            x.extend_from_slice(bucket_x.as_flattened());
            z.extend_from_slice(bucket_z.as_flattened());
            c.extend_from_slice(bucket_coeff);
        }
        start.push(n as u32);
        if n > 0 {
            stream.memcpy_htod(&x, &mut columns.x)?;
            stream.memcpy_htod(&z, &mut columns.z)?;
            stream.memcpy_htod(bytemuck::cast_slice::<C, u8>(&c), &mut columns.coeff)?;
        }
        stream.memcpy_htod(&start, &mut columns.start)?;
        stream.memcpy_htod(&lens, &mut columns.lengths)?;
        columns.len = n;
        columns.buckets = b;

        let usable = hash.max_bits().min(GPU_MAX_BITS) as usize;
        let hash_rows = stream.clone_htod(&flat_rows::<W>((0..usable).map(|i| hash.row(i))))?;
        let fingerprints = FingerprintRows::<W>::new(hash.seed());
        let fingerprint_rows = stream.clone_htod(&fingerprints.flat())?;
        launch_fingerprint(
            &stream,
            &kernels,
            &fingerprint_rows,
            (&columns.x, &columns.z),
            &mut columns.fingerprint,
            n,
        )?;
        let scan = (ScanScratch::new(&stream)?, stream.alloc_zeros::<u32>(2)?);
        let lengths_scan = stream.alloc_zeros::<u32>(1)?;
        stream.synchronize()?;
        let longest = lens.iter().copied().max().unwrap_or(0);
        let lag = if alone { LONE_LAG } else { 0 };
        let length =
            LengthTracker::new(context.clone(), Lengths { len: n, longest: Some(longest) }, lag);
        let staging = Mutex::new(HostStaging::new(&context));
        let uploaded = Self {
            context,
            stream,
            kernels,
            num_qubits: sum.num_qubits(),
            hash,
            columns,
            spare: None,
            hash_rows,
            fingerprints,
            fingerprint_rows,
            scan,
            lengths_scan,
            length,
            staging,
        };
        uploaded.debug_check();
        Ok(uploaded)
    }

    /// Download into a host [`PauliSum`] under the same hash and bucket count, re-sorting each bucket to the host's lexicographic order.
    /// The first call allocates page-locked staging for the whole sum, which later calls reuse.
    pub fn to_host(&self) -> Result<PauliSum<W, C>, GpuError> {
        let b = self.hash.num_buckets();
        let start = self.stream.clone_dtoh(&self.columns.start.slice(0..b))?;
        let lens = self.stream.clone_dtoh(&self.columns.lengths.slice(0..b))?;
        self.stream.synchronize()?;
        let extent = start
            .iter()
            .zip(&lens)
            .map(|(&first, &len)| first as usize + len as usize)
            .max()
            .unwrap_or(0);
        let mut staging = self.staging.lock().expect("staging mutex poisoned");
        staging.ensure(extent, W)?;
        if extent > 0 {
            let stream = &self.stream;
            stream.memcpy_dtoh(
                &self.columns.x.slice(0..extent * W),
                staging.x.slice_mut(extent * W),
            )?;
            stream.memcpy_dtoh(
                &self.columns.z.slice(0..extent * W),
                staging.z.slice_mut(extent * W),
            )?;
            stream.memcpy_dtoh(
                &self.columns.coeff.slice(0..extent * DeviceColumns::<W, C>::COEFFICIENT_BYTES),
                bytemuck::cast_slice_mut::<C, u8>(staging.coeff.slice_mut(extent)),
            )?;
            stream.synchronize()?;
        }
        let buckets = gather_sorted::<W, C>(
            &start,
            &lens,
            staging.x.slice(extent * W),
            staging.z.slice(extent * W),
            staging.coeff.slice(extent),
        );
        Ok(PauliSum::from_buckets(buckets, self.hash.clone(), self.num_qubits))
    }

    /// Total number of terms, or an upper bound on it after a layer whose count is still on the device; exact once the call that ran the layer returns.
    pub fn len(&self) -> usize {
        self.columns.len
    }

    /// `true` if the sum has no terms; `false` may be an upper bound's, as [`Self::len`].
    pub fn is_empty(&self) -> bool {
        self.columns.len == 0
    }

    /// An upper bound on the longest bucket, `None` when no step since the last anchor bounds it.
    pub(crate) fn longest_bound(&self) -> Option<u32> {
        self.length.bound().longest
    }

    /// The last anchor's length carried through the steps since with joins taken as keeping it: deterministic, for heuristics, and no bound.
    pub(crate) fn len_estimate(&self) -> usize {
        self.length.estimate().len
    }

    /// A step bounded by `growth`, after which `published` holds the device's `[len, longest]` (`LengthTracker::step`).
    pub(super) fn record_step(
        &mut self,
        growth: Growth,
        published: Option<&CudaSlice<u32>>,
    ) -> Result<(), GpuError> {
        self.length.step(growth, published.map(|device| (&self.stream, device)))?;
        self.columns.len = self.length.bound().len;
        Ok(())
    }

    /// The sum holds exactly `len` terms, read synchronously, and its longest bucket is at most `longest`.
    pub(super) fn record_exact(&mut self, len: usize, longest: Option<u32>) {
        self.length.set_exact(Lengths { len, longest });
        self.columns.len = len;
    }

    /// The sum holds exactly `len` terms, read synchronously, after a step that only shortened buckets; the longest bucket stays a bound, which [`Self::settle_lengths`] still scans.
    pub(super) fn record_exact_len(&mut self, len: usize) {
        self.length.set_exact_len(len);
        self.columns.len = len;
    }

    /// The spare columns emptied, with room for `terms` terms and `buckets` buckets; fresh columns hold at least the sum's length.
    pub(super) fn take_spare(
        &mut self,
        terms: usize,
        buckets: usize,
    ) -> Result<DeviceColumns<W, C>, GpuError> {
        let mut out = match self.spare.take() {
            Some(spare) => spare,
            None => DeviceColumns::with_capacity(
                &self.stream,
                self.device(),
                terms.max(self.columns.len),
                buckets,
            )?,
        };
        out.len = 0;
        out.buckets = 0;
        out.reserve(terms, buckets)?;
        Ok(out)
    }

    /// `out`, holding `len` terms over the current buckets, becomes the columns, and the previous columns the spare.
    pub(super) fn swap_in(&mut self, mut out: DeviceColumns<W, C>, len: usize) {
        out.len = len;
        out.buckets = self.hash.num_buckets();
        self.spare = Some(std::mem::replace(&mut self.columns, out));
    }

    /// Make [`Self::len`] and the longest bucket exact: consume every read-back and, if a step published none, scan the bucket lengths behind one synchronization.
    pub(crate) fn settle_lengths(&mut self) -> Result<(), GpuError> {
        self.length.consume(0)?;
        if self.length.is_settled() {
            self.columns.len = self.length.bound().len;
            return Ok(());
        }
        let (len, longest) = self.scan_lengths()?;
        self.record_exact(len, Some(longest));
        Ok(())
    }

    /// The exact number of terms, scanned behind one synchronization unless the bound is exact; the length tracker is left as it was, so a trace that reads it changes no decision.
    pub(crate) fn read_len(&mut self) -> Result<usize, GpuError> {
        if self.length.is_exact() {
            return Ok(self.columns.len);
        }
        Ok(self.scan_lengths()?.0)
    }

    /// The bucket lengths' total and maximum, scanned and read behind one synchronization.
    fn scan_lengths(&mut self) -> Result<(usize, u32), GpuError> {
        let b = self.hash.num_buckets();
        let ordinal = self.device();
        super::columns::grow(&self.stream, &mut self.lengths_scan, b + 1, ordinal)?;
        let (scan, totals) = &mut self.scan;
        exclusive_scan(
            &self.stream,
            &self.kernels,
            &self.columns.lengths.slice(0..b),
            &mut self.lengths_scan.slice_mut(0..b + 1),
            b,
            scan,
            totals,
        )?;
        let read = self.stream.clone_dtoh(totals)?;
        self.stream.synchronize()?;
        Ok((read[0] as usize, read[1]))
    }

    /// The partitioning hash, identical to the host sum's.
    pub fn hash(&self) -> &Gf2Hash<W> {
        &self.hash
    }

    /// Number of qubits this sum acts on.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Active bucket bits, `hash().bits()`.
    pub fn bits(&self) -> u8 {
        self.hash.bits()
    }

    /// The device ordinal the sum lives on.
    pub fn device(&self) -> u32 {
        self.context.ordinal() as u32
    }

    /// Double the bucket count as the host sum does: bucket `β` splits into `β` and `β + B`, each inheriting `β`'s order.
    /// Returns [`GpuError::Unsupported`] at [`Self::max_bits`].
    pub fn refine(&mut self) -> Result<(), GpuError> {
        if self.hash.bits() >= self.max_bits() {
            return Err(GpuError::Unsupported("refine beyond the device's bucket bits"));
        }
        self.refine_pass(1)
    }

    /// The most bucket bits this sum refines to: its hash's rows, capped at [`GPU_MAX_BITS`].
    pub fn max_bits(&self) -> u8 {
        self.hash.max_bits().min(GPU_MAX_BITS)
    }

    /// Refine until the bucket count is `1 << bits`, up to four bits per counting pass, as repeated [`Self::refine`] would; a no-op at or above it, [`GpuError::Unsupported`] past [`Self::max_bits`] with the sum unchanged.
    pub(crate) fn refine_to(&mut self, bits: u8) -> Result<(), GpuError> {
        if bits > self.max_bits() {
            return Err(GpuError::Unsupported("refine_to beyond the device's bucket bits"));
        }
        while self.hash.bits() < bits {
            self.refine_pass((bits - self.hash.bits()).min(REFINE_MAX_DELTA))?;
        }
        Ok(())
    }

    /// One counting pass adding `delta` bits into the spare columns, which then swap in.
    fn refine_pass(&mut self, delta: u8) -> Result<(), GpuError> {
        let bits_old = self.hash.bits();
        let b_old = self.hash.num_buckets();
        let b_new = b_old << delta;
        let n = self.columns.len;
        let mut out = self.take_spare(n, b_new)?;
        let (b_old32, bits32, delta32) = (b_old as u32, u32::from(bits_old), u32::from(delta));
        let (kernels, stream, columns) = (&self.kernels, &self.stream, &self.columns);
        // SAFETY: arguments match `k_refine_count` in refine.cu; `out.lengths` holds `b_new` entries.
        unsafe {
            stream
                .launch_builder(&kernels.refine_count)
                .arg(&columns.x)
                .arg(&columns.z)
                .arg(&columns.start)
                .arg(&columns.lengths)
                .arg(&self.hash_rows)
                .arg(&b_old32)
                .arg(&bits32)
                .arg(&delta32)
                .arg(&mut out.lengths)
                .launch(warp_per_bucket(b_old))?;
        }
        let (scan, totals) = &mut self.scan;
        exclusive_scan(
            stream,
            kernels,
            &out.lengths.slice(0..b_new),
            &mut out.start.slice_mut(0..b_new + 1),
            b_new,
            scan,
            totals,
        )?;
        // SAFETY: arguments match `k_refine_scatter`; `out` holds `n` terms, the scan's total.
        unsafe {
            stream
                .launch_builder(&kernels.refine_scatter)
                .arg(&columns.x)
                .arg(&columns.z)
                .arg(&columns.coeff)
                .arg(&columns.fingerprint)
                .arg(&columns.start)
                .arg(&columns.lengths)
                .arg(&self.hash_rows)
                .arg(&b_old32)
                .arg(&bits32)
                .arg(&delta32)
                .arg(&out.start)
                .arg(&mut out.x)
                .arg(&mut out.z)
                .arg(&mut out.coeff)
                .arg(&mut out.fingerprint)
                .launch(warp_per_bucket(b_old))?;
        }
        for _ in 0..delta {
            self.hash.refine();
        }
        self.swap_in(out, n);
        self.length.step(Growth::Shrink { cap: usize::MAX }, Some((&self.stream, &self.scan.1)))?;
        self.columns.len = self.length.bound().len;
        self.debug_check();
        Ok(())
    }

    /// The device-side analogue of the [`PauliSum`] structural invariant, order within a bucket free: the first class of violation, or the device error that stopped the check.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fn assert_invariants_device(&self) -> Result<(), String> {
        self.check_invariants().map_err(|e| e.to_string())?
    }

    fn check_invariants(&self) -> Result<Result<(), String>, GpuError> {
        let b = self.hash.num_buckets();
        if self.columns.buckets != b {
            return Ok(Err(format!(
                "GpuSum: {} bucket entries for a hash with {b} buckets",
                self.columns.buckets
            )));
        }
        let stream = &self.stream;
        let mut bad = stream.clone_htod(&[0u32, 0, 0, 0, u32::MAX])?;
        let (bits32, b32, nq32) = (u32::from(self.hash.bits()), b as u32, self.num_qubits as u32);
        let columns = &self.columns;
        // SAFETY: arguments match `k_check_invariants` in invariants.cu.
        unsafe {
            stream
                .launch_builder(&self.kernels.check_invariants)
                .arg(&columns.x)
                .arg(&columns.z)
                .arg(&columns.fingerprint)
                .arg(&columns.start)
                .arg(&columns.lengths)
                .arg(&self.hash_rows)
                .arg(&bits32)
                .arg(&b32)
                .arg(&self.fingerprint_rows)
                .arg(&nq32)
                .arg(&mut bad)
                .launch(warp_per_bucket(b))?;
        }
        let bad = stream.clone_dtoh(&bad)?;
        let start = stream.clone_dtoh(&columns.start.slice(0..b))?;
        let lens = stream.clone_dtoh(&columns.lengths.slice(0..b))?;
        stream.synchronize()?;
        if bad[..4].iter().any(|&v| v != 0) {
            return Ok(Err(format!(
                "GpuSum: {} misplaced, {} duplicate keys, {} beyond num_qubits, {} stale fingerprints (first in bucket {})",
                bad[0], bad[1], bad[2], bad[3], bad[4]
            )));
        }
        let total: usize = lens.iter().map(|&l| l as usize).sum();
        let longest = lens.iter().copied().max().unwrap_or(0);
        if self.longest_bound().is_some_and(|bound| longest > bound) {
            return Ok(Err(format!(
                "GpuSum: a bucket of {longest} terms above the longest-bucket bound {:?}",
                self.longest_bound()
            )));
        }
        let exact = self.length.is_exact();
        if total > columns.len || (exact && total != columns.len) {
            return Ok(Err(format!(
                "GpuSum: bucket lengths sum to {total}, cached len is {}{}",
                columns.len,
                if exact { "" } else { " (a bound)" }
            )));
        }
        let mut spans: Vec<(usize, usize, usize)> = (0..b)
            .filter(|&i| lens[i] > 0)
            .map(|i| (start[i] as usize, lens[i] as usize, i))
            .collect();
        spans.sort_unstable();
        for w in spans.windows(2) {
            if w[0].0 + w[0].1 > w[1].0 {
                return Ok(Err(format!("GpuSum: buckets {} and {} overlap", w[0].2, w[1].2)));
            }
        }
        if let Some(&(first, len, i)) = spans.last() {
            if first + len > columns.term_capacity() {
                return Ok(Err(format!(
                    "GpuSum: bucket {i} ends at {} beyond capacity {}",
                    first + len,
                    columns.term_capacity()
                )));
            }
        }
        Ok(Ok(()))
    }

    /// Debug builds run the device invariant check after every structural change.
    pub(super) fn debug_check(&self) {
        #[cfg(debug_assertions)]
        if let Ok(Err(msg)) = self.check_invariants() {
            panic!("{msg}");
        }
    }
}

/// Gather every bucket from the staged columns, sorting any bucket that is not already strictly ascending.
fn gather_sorted<const W: usize, C: Coefficient>(
    start: &[u32],
    lens: &[u32],
    staged_x: &[u64],
    staged_z: &[u64],
    staged_coeff: &[C],
) -> Vec<BucketColumns<W, C>> {
    (0..lens.len())
        .into_par_iter()
        .map(|bucket| {
            let (first, len) = (start[bucket] as usize, lens[bucket] as usize);
            let key = |column: &[u64], r: usize| -> [u64; W] {
                std::array::from_fn(|w| column[r * W + w])
            };
            let coeff = |r: usize| staged_coeff[r];
            let ascending = (first + 1..first + len).all(|r| {
                (key(staged_x, r - 1), key(staged_z, r - 1)) < (key(staged_x, r), key(staged_z, r))
            });
            let build = |rows: &mut dyn Iterator<Item = usize>| {
                let mut columns = BucketColumns::<W, C>::default();
                columns.x.reserve_exact(len);
                columns.z.reserve_exact(len);
                columns.coeff.reserve_exact(len);
                for r in rows {
                    columns.x.push(key(staged_x, r));
                    columns.z.push(key(staged_z, r));
                    columns.coeff.push(coeff(r));
                }
                columns
            };
            if ascending {
                build(&mut (first..first + len))
            } else {
                let mut rows: Vec<usize> = (first..first + len).collect();
                rows.sort_unstable_by_key(|&r| (key(staged_x, r), key(staged_z, r)));
                build(&mut rows.into_iter())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
