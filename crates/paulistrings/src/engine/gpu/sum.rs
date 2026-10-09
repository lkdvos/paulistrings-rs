//! [`GpuSum`], a bucketed Pauli sum resident on one CUDA device, with its upload, download and refine.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, PushKernelArg};
use num_complex::Complex64;
use rayon::prelude::*;

use super::columns::DeviceColumns;
use super::device;
use super::error::GpuError;
use super::fingerprint::FingerprintRows;
use super::module::{self, thread_per, warp_per_bucket, KernelSet};
use super::scan::{exclusive_scan, ScanScratch};
use super::staging::HostStaging;
use crate::pauli_sum::hash::{Gf2Hash, B_MAX_BITS};
use crate::pauli_sum::storage::BucketColumns;
use crate::pauli_sum::PauliSum;

/// Hash bits one refine pass adds at most; must match `REFINE_MAX_DELTA` in `kernels/refine.cu`.
const REFINE_MAX_DELTA: u8 = 4;

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
/// [`Self::from_host`] and [`Self::to_host`] round-trip bitwise, bucket by bucket, under the host sum's [`Gf2Hash`].
/// Resident device memory is `16·W + 24` bytes per term plus `8` per bucket, doubled once a refine has run.
/// Every fallible operation returns a [`GpuError`], device allocation failure as [`GpuError::OutOfMemory`].
pub struct GpuSum<const W: usize> {
    pub(super) context: Arc<CudaContext>,
    pub(super) stream: Arc<CudaStream>,
    pub(super) kernels: Arc<KernelSet>,
    pub(super) hash: Gf2Hash<W>,
    pub(super) num_qubits: usize,
    pub(super) columns: DeviceColumns<W>,
    /// The previous columns, kept as the next refine's or layer's target.
    pub(super) spare: Option<DeviceColumns<W>>,
    /// All `B_MAX_BITS` rows of `hash`, so a refine uploads nothing.
    pub(super) hash_rows: CudaSlice<u64>,
    /// `FingerprintRows::new(hash.seed())`, and its rows in device layout.
    pub(super) fingerprints: FingerprintRows<W>,
    pub(super) fingerprint_rows: CudaSlice<u64>,
    /// The refine's scan buffers and `[total, max]`.
    scan: (ScanScratch, CudaSlice<u32>),
    /// Page-locked download staging, allocated on first [`Self::to_host`] and reused.
    staging: Mutex<HostStaging>,
}

/// K0: the fingerprint of rows `0..n` of `(x, z)` into `g`, enqueued on `stream`.
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

impl<const W: usize> GpuSum<W> {
    /// Upload `sum` to device `ordinal`, bucket by bucket in the host's order, and compute every term's fingerprint on the device.
    /// The first call per `(ordinal, W)` compiles the kernels through NVRTC.
    pub fn from_host(sum: &PauliSum<W>, ordinal: u32) -> Result<Self, GpuError> {
        Self::upload(sum, ordinal, module::kernel_set(ordinal, W)?)
    }

    /// As [`Self::from_host`] with extra NVRTC options, the `-DFP_BITS=<b>` collision hook.
    pub(crate) fn from_host_with_options(
        sum: &PauliSum<W>,
        ordinal: u32,
        extra_options: &[String],
    ) -> Result<Self, GpuError> {
        Self::upload(
            sum,
            ordinal,
            module::kernel_set_with_options(ordinal, W, extra_options)?,
        )
    }

    fn upload(sum: &PauliSum<W>, ordinal: u32, kernels: Arc<KernelSet>) -> Result<Self, GpuError> {
        let context = device::context(ordinal)?;
        let stream = context.new_stream()?;
        let hash = sum.hash().clone();
        let (n, b) = (sum.len(), hash.num_buckets());
        let mut columns = DeviceColumns::<W>::with_capacity(&stream, ordinal, n, b)?;

        let mut x = Vec::with_capacity(n * W);
        let mut z = Vec::with_capacity(n * W);
        let mut c: Vec<f64> = Vec::with_capacity(2 * n);
        let mut start = Vec::with_capacity(b + 1);
        let mut lens = Vec::with_capacity(b);
        for bucket in 0..b {
            let (bucket_x, bucket_z, bucket_coeff) = sum.bucket(bucket);
            start.push((x.len() / W) as u32);
            lens.push(bucket_coeff.len() as u32);
            x.extend_from_slice(bucket_x.as_flattened());
            z.extend_from_slice(bucket_z.as_flattened());
            c.extend_from_slice(bytemuck::cast_slice::<Complex64, f64>(bucket_coeff));
        }
        start.push(n as u32);
        if n > 0 {
            stream.memcpy_htod(&x, &mut columns.x)?;
            stream.memcpy_htod(&z, &mut columns.z)?;
            stream.memcpy_htod(&c, &mut columns.coeff)?;
        }
        stream.memcpy_htod(&start, &mut columns.start)?;
        stream.memcpy_htod(&lens, &mut columns.lens)?;
        columns.len = n;
        columns.buckets = b;

        let hash_rows = stream.clone_htod(&flat_rows::<W>(
            (0..B_MAX_BITS as usize).map(|i| hash.row(i)),
        ))?;
        let fingerprints = FingerprintRows::<W>::new(hash.seed());
        let fingerprint_rows = stream.clone_htod(&fingerprints.flat())?;
        launch_fingerprint(
            &stream,
            &kernels,
            &fingerprint_rows,
            (&columns.x, &columns.z),
            &mut columns.g,
            n,
        )?;
        let scan = (ScanScratch::new(&stream)?, stream.alloc_zeros::<u32>(2)?);
        stream.synchronize()?;
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
            staging,
        };
        uploaded.debug_check();
        Ok(uploaded)
    }

    /// Download into a host [`PauliSum`] under the same hash and bucket count, re-sorting each bucket to the host's lexicographic order.
    /// The first call allocates page-locked staging for the whole sum, which later calls reuse.
    pub fn to_host(&self) -> Result<PauliSum<W>, GpuError> {
        let b = self.hash.num_buckets();
        let start = self.stream.clone_dtoh(&self.columns.start.slice(0..b))?;
        let lens = self.stream.clone_dtoh(&self.columns.lens.slice(0..b))?;
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
                &self.columns.coeff.slice(0..2 * extent),
                staging.coeff.slice_mut(2 * extent),
            )?;
            stream.synchronize()?;
        }
        let buckets = gather_sorted::<W>(
            &start,
            &lens,
            staging.x.slice(extent * W),
            staging.z.slice(extent * W),
            staging.coeff.slice(2 * extent),
        );
        Ok(PauliSum::from_buckets(
            buckets,
            self.hash.clone(),
            self.num_qubits,
        ))
    }

    /// Total number of terms.
    pub fn len(&self) -> usize {
        self.columns.len
    }

    /// `true` if the sum has no terms.
    pub fn is_empty(&self) -> bool {
        self.columns.len == 0
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
    /// Returns [`GpuError::Unsupported`] at `B_MAX_BITS`.
    pub fn refine(&mut self) -> Result<(), GpuError> {
        if self.hash.bits() >= B_MAX_BITS {
            return Err(GpuError::Unsupported("refine beyond B_MAX_BITS"));
        }
        self.refine_pass(1)
    }

    /// Refine until the bucket count is `1 << bits`, up to four bits per counting pass; a no-op at or above it.
    /// The result is the one repeated [`Self::refine`] gives.
    /// Returns [`GpuError::Unsupported`] past `B_MAX_BITS`, leaving the sum unchanged.
    pub fn refine_to(&mut self, bits: u8) -> Result<(), GpuError> {
        if bits > B_MAX_BITS {
            return Err(GpuError::Unsupported("refine_to beyond B_MAX_BITS"));
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
        let mut out = match self.spare.take() {
            Some(spare) => spare,
            None => DeviceColumns::with_capacity(&self.stream, self.device(), n, b_new)?,
        };
        out.len = 0;
        out.buckets = 0;
        out.reserve(n, b_new)?;
        let (b_old32, bits32, delta32) = (b_old as u32, u32::from(bits_old), u32::from(delta));
        let (kernels, stream, columns) = (&self.kernels, &self.stream, &self.columns);
        // SAFETY: arguments match `k_refine_count` in refine.cu; `out.lens` holds `b_new` entries.
        unsafe {
            stream
                .launch_builder(&kernels.refine_count)
                .arg(&columns.x)
                .arg(&columns.z)
                .arg(&columns.start)
                .arg(&columns.lens)
                .arg(&self.hash_rows)
                .arg(&b_old32)
                .arg(&bits32)
                .arg(&delta32)
                .arg(&mut out.lens)
                .launch(warp_per_bucket(b_old))?;
        }
        let (scan, totals) = &mut self.scan;
        exclusive_scan(
            stream,
            kernels,
            &out.lens.slice(0..b_new),
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
                .arg(&columns.g)
                .arg(&columns.start)
                .arg(&columns.lens)
                .arg(&self.hash_rows)
                .arg(&b_old32)
                .arg(&bits32)
                .arg(&delta32)
                .arg(&out.start)
                .arg(&mut out.x)
                .arg(&mut out.z)
                .arg(&mut out.coeff)
                .arg(&mut out.g)
                .launch(warp_per_bucket(b_old))?;
        }
        out.len = n;
        out.buckets = b_new;
        self.spare = Some(std::mem::replace(&mut self.columns, out));
        for _ in 0..delta {
            self.hash.refine();
        }
        self.debug_check();
        Ok(())
    }

    /// Check the device-side analogue of the [`PauliSum`] structural invariant, with order within a bucket free; the error describes the first class of violation, or the device error that stopped the check.
    pub fn assert_invariants_device(&self) -> Result<(), String> {
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
        let (bits32, b32, nq32) = (
            u32::from(self.hash.bits()),
            b as u32,
            self.num_qubits as u32,
        );
        let columns = &self.columns;
        // SAFETY: arguments match `k_check_invariants` in invariants.cu.
        unsafe {
            stream
                .launch_builder(&self.kernels.check_invariants)
                .arg(&columns.x)
                .arg(&columns.z)
                .arg(&columns.g)
                .arg(&columns.start)
                .arg(&columns.lens)
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
        let lens = stream.clone_dtoh(&columns.lens.slice(0..b))?;
        stream.synchronize()?;
        if bad[..4].iter().any(|&v| v != 0) {
            return Ok(Err(format!(
                "GpuSum: {} misplaced, {} duplicate keys, {} beyond num_qubits, {} stale fingerprints (first in bucket {})",
                bad[0], bad[1], bad[2], bad[3], bad[4]
            )));
        }
        let total: usize = lens.iter().map(|&l| l as usize).sum();
        if total != columns.len {
            return Ok(Err(format!(
                "GpuSum: bucket lengths sum to {total}, cached len is {}",
                columns.len
            )));
        }
        let mut spans: Vec<(usize, usize, usize)> = (0..b)
            .filter(|&i| lens[i] > 0)
            .map(|i| (start[i] as usize, lens[i] as usize, i))
            .collect();
        spans.sort_unstable();
        for w in spans.windows(2) {
            if w[0].0 + w[0].1 > w[1].0 {
                return Ok(Err(format!(
                    "GpuSum: buckets {} and {} overlap",
                    w[0].2, w[1].2
                )));
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
fn gather_sorted<const W: usize>(
    start: &[u32],
    lens: &[u32],
    staged_x: &[u64],
    staged_z: &[u64],
    staged_coeff: &[f64],
) -> Vec<BucketColumns<W>> {
    (0..lens.len())
        .into_par_iter()
        .map(|bucket| {
            let (first, len) = (start[bucket] as usize, lens[bucket] as usize);
            let key = |column: &[u64], r: usize| -> [u64; W] {
                std::array::from_fn(|w| column[r * W + w])
            };
            let coeff = |r: usize| Complex64::new(staged_coeff[2 * r], staged_coeff[2 * r + 1]);
            let ascending = (first + 1..first + len).all(|r| {
                (key(staged_x, r - 1), key(staged_z, r - 1)) < (key(staged_x, r), key(staged_z, r))
            });
            let build = |rows: &mut dyn Iterator<Item = usize>| {
                let mut columns = BucketColumns::<W>::default();
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
