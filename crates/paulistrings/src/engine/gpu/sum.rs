//! [`GpuSum`], a bucketed Pauli sum resident on one CUDA device, with its upload, download and refine.

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use num_complex::Complex64;
use rayon::prelude::*;

use super::columns::DeviceColumns;
use super::device;
use super::error::GpuError;
use super::fingerprint::FingerprintRows;
use super::module::{self, KernelSet};
use super::scan::exclusive_scan;
use super::staging::HostStaging;
use crate::bucket::hash::{Gf2Hash, B_MAX_BITS};
use crate::bucket::sum::BucketCols;
use crate::pauli_sum::PauliSum;

const TERM_THREADS: u32 = 256;
const BUCKET_THREADS: u32 = 256;
/// Hash bits one refine pass adds at most; must match `REFINE_MAX_DELTA` in `kernels/refine.cu`.
const REFINE_MAX_DELTA: u8 = 4;

/// One thread per term.
fn per_term(m: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((m as u32).div_ceil(TERM_THREADS).max(1), 1, 1),
        block_dim: (TERM_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One warp per bucket.
fn per_bucket(b: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((b as u32).div_ceil(BUCKET_THREADS / 32).max(1), 1, 1),
        block_dim: (BUCKET_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

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
/// Built from a host [`PauliSum`] by [`Self::from_host`] and read back by [`Self::to_host`]; the round trip is bitwise, bucket by bucket.
/// On the device a bucket holds unique keys under the same [`Gf2Hash`] partition as the host sum, and every term carries a 64-bit GF(2)-linear fingerprint used only for ordering. See ARCHITECTURE.md §GPU-Readiness.
///
/// Resident device memory is `16·W + 24` bytes per term plus `8` per bucket, doubled once a refine has run, since the sum keeps the previous columns as the next refine's target.
/// Every fallible operation returns a [`GpuError`], including device allocation failure as [`GpuError::OutOfMemory`].
pub struct GpuSum<const W: usize> {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: Arc<KernelSet>,
    hash: Gf2Hash<W>,
    num_qubits: usize,
    cols: DeviceColumns<W>,
    /// The previous columns, kept as the next refine's target.
    spare: Option<DeviceColumns<W>>,
    /// All `B_MAX_BITS` rows of `hash`, so a refine uploads nothing.
    hash_rows: CudaSlice<u64>,
    /// The fingerprint rows `FingerprintRows::new(hash.seed())` in device layout.
    fp_rows: CudaSlice<u64>,
    /// Page-locked download staging, allocated on first [`Self::to_host`] and reused.
    staging: Mutex<HostStaging>,
}

impl<const W: usize> GpuSum<W> {
    /// Upload `sum` to device `ordinal`, bucket by bucket in the host's order, and compute every term's fingerprint on the device.
    /// The first call per `(ordinal, W)` compiles the kernels through NVRTC.
    pub fn from_host(sum: &PauliSum<W>, ordinal: u32) -> Result<Self, GpuError> {
        Self::upload(sum, ordinal, module::kernel_set(ordinal, W)?)
    }

    /// As [`Self::from_host`] with extra NVRTC options, the `-DFP_BITS=<b>` collision hook.
    #[cfg(test)]
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
        let ctx = device::context(ordinal)?;
        let stream = ctx.new_stream()?;
        let hash = sum.hash().clone();
        let (n, b) = (sum.len(), hash.num_buckets());
        let mut cols = DeviceColumns::<W>::with_capacity(&stream, ordinal, n, b)?;

        let mut x = Vec::with_capacity(n * W);
        let mut z = Vec::with_capacity(n * W);
        let mut c: Vec<f64> = Vec::with_capacity(2 * n);
        let mut start = Vec::with_capacity(b + 1);
        let mut lens = Vec::with_capacity(b);
        for bucket in 0..b {
            let (bx, bz, bc) = sum.bucket(bucket);
            start.push((x.len() / W) as u32);
            lens.push(bc.len() as u32);
            x.extend_from_slice(bx.as_flattened());
            z.extend_from_slice(bz.as_flattened());
            c.extend_from_slice(bytemuck::cast_slice::<Complex64, f64>(bc));
        }
        start.push(n as u32);
        if n > 0 {
            stream.memcpy_htod(&x, &mut cols.x)?;
            stream.memcpy_htod(&z, &mut cols.z)?;
            stream.memcpy_htod(&c, &mut cols.coeff)?;
        }
        stream.memcpy_htod(&start, &mut cols.start)?;
        stream.memcpy_htod(&lens, &mut cols.lens)?;
        cols.len = n;
        cols.buckets = b;

        let hash_rows = stream.clone_htod(&flat_rows::<W>(
            (0..B_MAX_BITS as usize).map(|i| hash.row(i)),
        ))?;
        let fp_rows = stream.clone_htod(&FingerprintRows::<W>::new(hash.seed()).flat())?;
        if n > 0 {
            let n32 = n as u32;
            // SAFETY: arguments match `k_fingerprint` in fingerprint.cu; `g` holds `n` entries.
            unsafe {
                stream
                    .launch_builder(&kernels.fingerprint)
                    .arg(&cols.x)
                    .arg(&cols.z)
                    .arg(&n32)
                    .arg(&fp_rows)
                    .arg(&mut cols.g)
                    .launch(per_term(n))?;
            }
        }
        stream.synchronize()?;
        let staging = Mutex::new(HostStaging::new(&ctx));
        let s = Self {
            ctx,
            stream,
            kernels,
            num_qubits: sum.num_qubits(),
            hash,
            cols,
            spare: None,
            hash_rows,
            fp_rows,
            staging,
        };
        s.debug_check();
        Ok(s)
    }

    /// Download into a host [`PauliSum`] under the same hash and bucket count, re-sorting each bucket to the host's lexicographic order.
    /// The first call allocates page-locked staging for the whole sum, which later calls reuse.
    pub fn to_host(&self) -> Result<PauliSum<W>, GpuError> {
        let b = self.hash.num_buckets();
        let start = self.stream.clone_dtoh(&self.cols.start.slice(0..b))?;
        let lens = self.stream.clone_dtoh(&self.cols.lens.slice(0..b))?;
        self.stream.synchronize()?;
        let extent = start
            .iter()
            .zip(&lens)
            .map(|(&s, &l)| s as usize + l as usize)
            .max()
            .unwrap_or(0);
        let mut staging = self.staging.lock().expect("staging mutex poisoned");
        staging.ensure(extent, W)?;
        if extent > 0 {
            let s = &self.stream;
            s.memcpy_dtoh(
                &self.cols.x.slice(0..extent * W),
                staging.x.slice_mut(extent * W),
            )?;
            s.memcpy_dtoh(
                &self.cols.z.slice(0..extent * W),
                staging.z.slice_mut(extent * W),
            )?;
            s.memcpy_dtoh(
                &self.cols.coeff.slice(0..2 * extent),
                staging.coeff.slice_mut(2 * extent),
            )?;
            s.synchronize()?;
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
        self.cols.len
    }

    /// `true` if the sum has no terms.
    pub fn is_empty(&self) -> bool {
        self.cols.len == 0
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
        self.ctx.ordinal() as u32
    }

    /// Double the bucket count, as [`PauliSum::refine`]: bucket `β` splits into `β` and `β + B`, each inheriting `β`'s order.
    /// Returns [`GpuError::Unsupported`] at [`B_MAX_BITS`].
    pub fn refine(&mut self) -> Result<(), GpuError> {
        if self.hash.bits() >= B_MAX_BITS {
            return Err(GpuError::Unsupported("refine beyond B_MAX_BITS"));
        }
        self.refine_pass(1)
    }

    /// Refine until the bucket count is `1 << bits`, up to four bits per counting pass; a no-op at or above it.
    /// The result is the one repeated [`Self::refine`] gives. Returns [`GpuError::Unsupported`] past [`B_MAX_BITS`], leaving the sum unchanged.
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
        let n = self.cols.len;
        let mut out = match self.spare.take() {
            Some(spare) => spare,
            None => DeviceColumns::with_capacity(&self.stream, self.device(), n, b_new)?,
        };
        out.len = 0;
        out.buckets = 0;
        out.reserve(n, b_new)?;
        let (b_old32, bits32, delta32) = (b_old as u32, u32::from(bits_old), u32::from(delta));
        let (k, s, cols) = (&self.kernels, &self.stream, &self.cols);
        // SAFETY: arguments match `k_refine_count` in refine.cu; `out.lens` holds `b_new` entries.
        unsafe {
            s.launch_builder(&k.refine_count)
                .arg(&cols.x)
                .arg(&cols.z)
                .arg(&cols.start)
                .arg(&cols.lens)
                .arg(&self.hash_rows)
                .arg(&b_old32)
                .arg(&bits32)
                .arg(&delta32)
                .arg(&mut out.lens)
                .launch(per_bucket(b_old))?;
        }
        exclusive_scan(
            s,
            k,
            &out.lens.slice(0..b_new),
            &mut out.start.slice_mut(0..b_new + 1),
            b_new,
        )?;
        // SAFETY: arguments match `k_refine_scatter`; `out` holds `n` terms, the scan's total.
        unsafe {
            s.launch_builder(&k.refine_scatter)
                .arg(&cols.x)
                .arg(&cols.z)
                .arg(&cols.coeff)
                .arg(&cols.g)
                .arg(&cols.start)
                .arg(&cols.lens)
                .arg(&self.hash_rows)
                .arg(&b_old32)
                .arg(&bits32)
                .arg(&delta32)
                .arg(&out.start)
                .arg(&mut out.x)
                .arg(&mut out.z)
                .arg(&mut out.coeff)
                .arg(&mut out.g)
                .launch(per_bucket(b_old))?;
        }
        out.len = n;
        out.buckets = b_new;
        self.spare = Some(std::mem::replace(&mut self.cols, out));
        for _ in 0..delta {
            self.hash.refine();
        }
        self.debug_check();
        Ok(())
    }

    /// Check the device-side invariant, the analogue of [`PauliSum::assert_invariants`]: every term in its hash bucket, each bucket strictly ascending in `(x, z)`, every key within `num_qubits`, every fingerprint current, and the bucket table consistent with [`Self::len`].
    /// Returns a description of the first class of violation, or of the device error that stopped the check.
    pub fn assert_invariants_device(&self) -> Result<(), String> {
        self.check_invariants().map_err(|e| e.to_string())?
    }

    fn check_invariants(&self) -> Result<Result<(), String>, GpuError> {
        let b = self.hash.num_buckets();
        if self.cols.buckets != b {
            return Ok(Err(format!(
                "GpuSum: {} bucket entries for a hash with {b} buckets",
                self.cols.buckets
            )));
        }
        let s = &self.stream;
        let mut bad = s.clone_htod(&[0u32, 0, 0, 0, u32::MAX])?;
        let (bits32, b32, nq32) = (
            u32::from(self.hash.bits()),
            b as u32,
            self.num_qubits as u32,
        );
        let cols = &self.cols;
        // SAFETY: arguments match `k_check_invariants` in invariants.cu.
        unsafe {
            s.launch_builder(&self.kernels.check_invariants)
                .arg(&cols.x)
                .arg(&cols.z)
                .arg(&cols.g)
                .arg(&cols.start)
                .arg(&cols.lens)
                .arg(&self.hash_rows)
                .arg(&bits32)
                .arg(&b32)
                .arg(&self.fp_rows)
                .arg(&nq32)
                .arg(&mut bad)
                .launch(per_bucket(b))?;
        }
        let bad = s.clone_dtoh(&bad)?;
        let start = s.clone_dtoh(&cols.start.slice(0..b))?;
        let lens = s.clone_dtoh(&cols.lens.slice(0..b))?;
        s.synchronize()?;
        if bad[..4].iter().any(|&v| v != 0) {
            return Ok(Err(format!(
                "GpuSum: {} misplaced, {} out of order, {} beyond num_qubits, {} stale fingerprints (first in bucket {})",
                bad[0], bad[1], bad[2], bad[3], bad[4]
            )));
        }
        let total: usize = lens.iter().map(|&l| l as usize).sum();
        if total != cols.len {
            return Ok(Err(format!(
                "GpuSum: bucket lengths sum to {total}, cached len is {}",
                cols.len
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
        if let Some(&(s0, l0, i)) = spans.last() {
            if s0 + l0 > cols.term_capacity() {
                return Ok(Err(format!(
                    "GpuSum: bucket {i} ends at {} beyond capacity {}",
                    s0 + l0,
                    cols.term_capacity()
                )));
            }
        }
        Ok(Ok(()))
    }

    /// Debug builds run the device invariant check after every structural change.
    fn debug_check(&self) {
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
    sx: &[u64],
    sz: &[u64],
    sc: &[f64],
) -> Vec<BucketCols<W>> {
    (0..lens.len())
        .into_par_iter()
        .map(|bucket| {
            let (s, l) = (start[bucket] as usize, lens[bucket] as usize);
            let key =
                |col: &[u64], r: usize| -> [u64; W] { std::array::from_fn(|w| col[r * W + w]) };
            let coeff = |r: usize| Complex64::new(sc[2 * r], sc[2 * r + 1]);
            let ascending =
                (s + 1..s + l).all(|r| (key(sx, r - 1), key(sz, r - 1)) < (key(sx, r), key(sz, r)));
            let build = |rows: &mut dyn Iterator<Item = usize>| {
                let mut cols = BucketCols::<W>::default();
                cols.x.reserve_exact(l);
                cols.z.reserve_exact(l);
                cols.coeff.reserve_exact(l);
                for r in rows {
                    cols.x.push(key(sx, r));
                    cols.z.push(key(sz, r));
                    cols.coeff.push(coeff(r));
                }
                cols
            };
            if ascending {
                build(&mut (s..s + l))
            } else {
                let mut rows: Vec<usize> = (s..s + l).collect();
                rows.sort_unstable_by_key(|&r| (key(sx, r), key(sz, r)));
                build(&mut rows.into_iter())
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulator::BuildAccumulator;
    use crate::bucket::hash::PartitionRows;
    use crate::bucket::sum::DEFAULT_HASH_SEED;
    use crate::pauli_string::PauliString;
    use crate::phase::Phase;
    use crate::test_support::{assert_same_terms, low_weight_sum, rand_sum};

    /// The device columns in storage order, gathered by nothing: valid while the columns are compact.
    struct Raw<const W: usize> {
        x: Vec<[u64; W]>,
        z: Vec<[u64; W]>,
        g: Vec<u64>,
    }

    impl<const W: usize> GpuSum<W> {
        fn download_raw(&self) -> Raw<W> {
            let n = self.cols.len;
            let s = &self.stream;
            let words = |v: Vec<u64>| -> Vec<[u64; W]> {
                v.chunks_exact(W)
                    .map(|c| std::array::from_fn(|w| c[w]))
                    .collect()
            };
            let x = s.clone_dtoh(&self.cols.x.slice(0..n * W)).unwrap();
            let z = s.clone_dtoh(&self.cols.z.slice(0..n * W)).unwrap();
            let g = s.clone_dtoh(&self.cols.g.slice(0..n)).unwrap();
            s.synchronize().unwrap();
            Raw {
                x: words(x),
                z: words(z),
                g,
            }
        }

        /// `out[i] = gf2_image(row i)` from the hash or partition kernel over rows `rows` with `bits` active.
        fn device_image(&self, partition: Option<&PartitionRows<W>>) -> Vec<u32> {
            let n = self.cols.len;
            let s = &self.stream;
            let mut out = s.alloc_zeros::<u32>(n.max(1)).unwrap();
            let (f, rows, bits) = match partition {
                None => (&self.kernels.bucket_of, None, u32::from(self.hash.bits())),
                Some(p) => {
                    let (rx, rz) = p.rows();
                    let flat = flat_rows::<W>(rx.iter().copied().zip(rz.iter().copied()));
                    (
                        &self.kernels.partition_of,
                        Some(s.clone_htod(&flat).unwrap()),
                        u32::from(p.bits()),
                    )
                }
            };
            let n32 = n as u32;
            unsafe {
                s.launch_builder(f)
                    .arg(&self.cols.x)
                    .arg(&self.cols.z)
                    .arg(&n32)
                    .arg(rows.as_ref().unwrap_or(&self.hash_rows))
                    .arg(&bits)
                    .arg(&mut out)
                    .launch(per_term(n))
                    .unwrap();
            }
            let v = s.clone_dtoh(&out.slice(0..n)).unwrap();
            s.synchronize().unwrap();
            v
        }
    }

    /// Same hash and bucket count, and every bucket bitwise equal including coefficient bit patterns.
    fn assert_same_buckets<const W: usize>(got: &PauliSum<W>, want: &PauliSum<W>, what: &str) {
        got.assert_invariants();
        assert!(got.hash().same_rows_as(want.hash()), "{what}: hash rows");
        assert_eq!(got.hash().bits(), want.hash().bits(), "{what}: bucket bits");
        assert_eq!(got.num_qubits(), want.num_qubits(), "{what}: num_qubits");
        for b in 0..want.num_buckets() {
            let (gx, gz, gc) = got.bucket(b);
            let (wx, wz, wc) = want.bucket(b);
            assert_eq!(gx, wx, "{what}: bucket {b} x");
            assert_eq!(gz, wz, "{what}: bucket {b} z");
            let bits = |c: &[Complex64]| -> Vec<(u64, u64)> {
                c.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
            };
            assert_eq!(bits(gc), bits(wc), "{what}: bucket {b} coeff bits");
        }
        assert_same_terms(got, want, what);
    }

    fn round_trip<const W: usize>(sum: &PauliSum<W>, what: &str) {
        let dev = GpuSum::from_host(sum, 0).expect("upload");
        assert_eq!(dev.len(), sum.len(), "{what}: len");
        assert_eq!(dev.bits(), sum.hash().bits(), "{what}: bits");
        dev.assert_invariants_device()
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_same_buckets(&dev.to_host().expect("download"), sum, what);
    }

    fn one_term<const W: usize>(num_qubits: usize) -> PauliSum<W> {
        let mut acc = BuildAccumulator::<W>::new(num_qubits);
        acc.add_term(
            PauliString::<W>::y(num_qubits as u32 - 1),
            Phase::ONE,
            Complex64::new(0.5, -0.25),
        );
        acc.finalize()
    }

    fn single_bucket<const W: usize>(num_qubits: usize) -> PauliSum<W> {
        let sum = rand_sum::<W>(5000, num_qubits, 0x5B).with_hash(Gf2Hash::new(
            num_qubits,
            0,
            DEFAULT_HASH_SEED,
        ));
        assert_eq!(sum.num_buckets(), 1);
        sum
    }

    fn round_trips<const W: usize>() {
        let nq = 64 * W;
        round_trip(&rand_sum::<W>(10_000, nq, 0x1111), "rand 1e4");
        round_trip(&low_weight_sum::<W>(20_000, nq, 2, 0x5151), "low weight");
        round_trip(&PauliSum::<W>::empty(nq), "empty");
        round_trip(&one_term::<W>(nq), "one term");
        round_trip(&single_bucket::<W>(nq), "single bucket");
    }

    #[test]
    fn round_trip_is_bitwise_w1() {
        crate::require_cuda!();
        round_trips::<1>();
    }

    #[test]
    fn round_trip_is_bitwise_w2() {
        crate::require_cuda!();
        round_trips::<2>();
    }

    #[test]
    fn round_trip_is_bitwise_w4() {
        crate::require_cuda!();
        round_trip(&rand_sum::<4>(10_000, 250, 0x4444), "W=4 rand 1e4");
    }

    #[test]
    fn round_trip_is_bitwise_at_one_million_terms() {
        crate::require_cuda!();
        round_trip(&rand_sum::<2>(1_000_000, 128, 0xCAFE), "rand 1e6");
    }

    #[test]
    fn to_host_twice_reuses_the_staging_and_agrees() {
        crate::require_cuda!();
        let sum = rand_sum::<1>(3000, 64, 0x77);
        let dev = GpuSum::from_host(&sum, 0).expect("upload");
        assert_same_buckets(&dev.to_host().unwrap(), &sum, "first");
        assert_same_buckets(&dev.to_host().unwrap(), &sum, "second");
    }

    fn check_fingerprints<const W: usize>(sum: &PauliSum<W>, mask: u64, dev: &GpuSum<W>) {
        let raw = dev.download_raw();
        assert_eq!(raw.g.len(), sum.len());
        let fp = FingerprintRows::<W>::new(sum.hash().seed());
        for i in 0..raw.g.len() {
            assert_eq!(
                raw.g[i],
                fp.fingerprint(&raw.x[i], &raw.z[i]) & mask,
                "term {i}"
            );
        }
    }

    fn device_fingerprints<const W: usize>() {
        let nq = 64 * W;
        for sum in [
            rand_sum::<W>(100_000, nq, 0xF00D),
            low_weight_sum::<W>(100_000, nq, 2, 0x6262),
        ] {
            let dev = GpuSum::from_host(&sum, 0).expect("upload");
            check_fingerprints(&sum, !0, &dev);
        }
    }

    #[test]
    fn device_fingerprint_matches_host_w1() {
        crate::require_cuda!();
        device_fingerprints::<1>();
    }

    #[test]
    fn device_fingerprint_matches_host_w2() {
        crate::require_cuda!();
        device_fingerprints::<2>();
    }

    fn hash_kernels<const W: usize>() {
        let nq = 64 * W - 3;
        for sum in [
            rand_sum::<W>(50_000, nq, 0xAB),
            low_weight_sum::<W>(50_000, nq, 2, 0xCD),
        ] {
            let dev = GpuSum::from_host(&sum, 0).expect("upload");
            let raw = dev.download_raw();
            let got = dev.device_image(None);
            assert_eq!(got.len(), sum.len());
            for (i, &got) in got.iter().enumerate() {
                assert_eq!(
                    got,
                    sum.hash().bucket_of(&raw.x[i], &raw.z[i]),
                    "bucket_of {i}"
                );
            }
            for bits in [1u8, 3, 6] {
                let rows = PartitionRows::<W>::from_seed(nq, bits, 0x1234);
                let got = dev.device_image(Some(&rows));
                assert_eq!(got.len(), sum.len());
                for (i, &got) in got.iter().enumerate() {
                    assert_eq!(
                        got,
                        rows.partition_of(&raw.x[i], &raw.z[i]),
                        "partition_of {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn hash_kernels_match_host_w1() {
        crate::require_cuda!();
        hash_kernels::<1>();
    }

    #[test]
    fn hash_kernels_match_host_w2() {
        crate::require_cuda!();
        hash_kernels::<2>();
    }

    fn refine_against_host<const W: usize>(sum: PauliSum<W>, what: &str) {
        let mut host = sum.clone();
        let mut dev = GpuSum::from_host(&sum, 0).expect("upload");
        dev.refine().expect("refine");
        host.refine();
        dev.assert_invariants_device()
            .unwrap_or_else(|e| panic!("{what} refine: {e}"));
        assert_same_buckets(&dev.to_host().unwrap(), &host, &format!("{what} refine"));
        let target = dev.bits() + 4;
        dev.refine_to(target).expect("refine_to");
        for _ in 0..4 {
            host.refine();
        }
        assert_eq!(dev.bits(), target);
        dev.assert_invariants_device()
            .unwrap_or_else(|e| panic!("{what} refine_to: {e}"));
        assert_same_buckets(&dev.to_host().unwrap(), &host, &format!("{what} refine_to"));
        let target = dev.bits() + 6;
        dev.refine_to(target).expect("refine_to across two passes");
        for _ in 0..6 {
            host.refine();
        }
        dev.assert_invariants_device()
            .unwrap_or_else(|e| panic!("{what} two-pass refine_to: {e}"));
        assert_same_buckets(
            &dev.to_host().unwrap(),
            &host,
            &format!("{what} two passes"),
        );
    }

    fn refines<const W: usize>() {
        let nq = 64 * W;
        refine_against_host(rand_sum::<W>(100_000, nq, 0x31), "rand");
        refine_against_host(low_weight_sum::<W>(50_000, nq, 2, 0x32), "low weight");
        refine_against_host(single_bucket::<W>(nq), "single bucket");
        refine_against_host(PauliSum::<W>::empty(nq), "empty");
    }

    #[test]
    fn refine_matches_host_w1() {
        crate::require_cuda!();
        refines::<1>();
    }

    #[test]
    fn refine_matches_host_w2() {
        crate::require_cuda!();
        refines::<2>();
    }

    #[test]
    fn refine_to_beyond_b_max_bits_is_unsupported() {
        crate::require_cuda!();
        let sum = rand_sum::<1>(1000, 64, 0x9);
        let mut dev = GpuSum::from_host(&sum, 0).expect("upload");
        let bits = dev.bits();
        assert!(matches!(
            dev.refine_to(B_MAX_BITS + 1),
            Err(GpuError::Unsupported(_))
        ));
        assert_eq!(dev.bits(), bits);
        dev.refine_to(bits).expect("a no-op");
        assert_same_buckets(&dev.to_host().unwrap(), &sum, "unchanged");
    }

    #[test]
    fn eight_bit_fingerprints_still_round_trip_and_refine() {
        crate::require_cuda!();
        let opts = ["-DFP_BITS=8".to_string()];
        let sum = rand_sum::<2>(50_000, 128, 0x88);
        let mut dev = GpuSum::from_host_with_options(&sum, 0, &opts).expect("upload");
        check_fingerprints(&sum, 0xFF, &dev);
        dev.assert_invariants_device().expect("invariants");
        assert_same_buckets(&dev.to_host().unwrap(), &sum, "FP_BITS=8 round trip");
        let mut host = sum.clone();
        dev.refine_to(dev.bits() + 3).expect("refine_to");
        for _ in 0..3 {
            host.refine();
        }
        dev.assert_invariants_device()
            .expect("invariants after refine");
        assert_same_buckets(&dev.to_host().unwrap(), &host, "FP_BITS=8 refine");
    }

    /// Device order within a bucket is free, so a download must restore the host's lex order itself.
    #[test]
    fn to_host_re_sorts_a_bucket_out_of_lex_order() {
        crate::require_cuda!();
        let sum = rand_sum::<2>(20_000, 128, 0x43);
        let b0 = (0..sum.num_buckets())
            .find(|&b| sum.bucket_len(b) >= 3)
            .unwrap();
        let r0: usize = (0..b0).map(|b| sum.bucket_len(b)).sum();
        let (bx, bz, bc) = sum.bucket(b0);
        let l = bx.len();
        let rev = |col: &[[u64; 2]]| -> Vec<u64> { col.iter().rev().flat_map(|k| *k).collect() };
        let c: Vec<f64> = bc.iter().rev().flat_map(|c| [c.re, c.im]).collect();
        let mut dev = GpuSum::from_host(&sum, 0).expect("upload");
        let s = dev.stream.clone();
        s.memcpy_htod(&rev(bx), &mut dev.cols.x.slice_mut(2 * r0..2 * (r0 + l)))
            .unwrap();
        s.memcpy_htod(&rev(bz), &mut dev.cols.z.slice_mut(2 * r0..2 * (r0 + l)))
            .unwrap();
        s.memcpy_htod(&c, &mut dev.cols.coeff.slice_mut(2 * r0..2 * (r0 + l)))
            .unwrap();
        assert!(dev.assert_invariants_device().is_err());
        assert_same_buckets(&dev.to_host().unwrap(), &sum, "reversed bucket");
    }

    #[test]
    fn invariants_kernel_reports_corruption() {
        crate::require_cuda!();
        let sum = rand_sum::<1>(20_000, 64, 0x42);
        let full: Vec<usize> = (0..sum.num_buckets())
            .filter(|&b| sum.bucket_len(b) >= 2)
            .take(2)
            .collect();
        let (b0, b1) = (full[0], full[1]);

        let mut dev = GpuSum::from_host(&sum, 0).expect("upload");
        let (bx, bz, _) = sum.bucket(b0);
        let swapped = [bx[1][0], bx[0][0]];
        let z_swapped = [bz[1][0], bz[0][0]];
        let r0: usize = (0..b0).map(|b| sum.bucket_len(b)).sum();
        dev.stream
            .memcpy_htod(&swapped, &mut dev.cols.x.slice_mut(r0..r0 + 2))
            .unwrap();
        dev.stream
            .memcpy_htod(&z_swapped, &mut dev.cols.z.slice_mut(r0..r0 + 2))
            .unwrap();
        let fp = FingerprintRows::<1>::new(sum.hash().seed());
        let g = [
            fp.fingerprint(&bx[1], &bz[1]),
            fp.fingerprint(&bx[0], &bz[0]),
        ];
        dev.stream
            .memcpy_htod(&g, &mut dev.cols.g.slice_mut(r0..r0 + 2))
            .unwrap();
        let msg = dev.assert_invariants_device().unwrap_err();
        assert!(msg.contains("0 misplaced, 1 out of order"), "{msg}");
        assert!(msg.contains(&format!("bucket {b0})")), "{msg}");

        let mut dev = GpuSum::from_host(&sum, 0).expect("upload");
        let (cx, cz, _) = sum.bucket(b1);
        dev.stream
            .memcpy_htod(&cx[0], &mut dev.cols.x.slice_mut(r0..r0 + 1))
            .unwrap();
        dev.stream
            .memcpy_htod(&cz[0], &mut dev.cols.z.slice_mut(r0..r0 + 1))
            .unwrap();
        let msg = dev.assert_invariants_device().unwrap_err();
        assert!(msg.contains("1 misplaced"), "{msg}");
        assert!(msg.contains("1 stale fingerprints"), "{msg}");

        let mut dev = GpuSum::from_host(&sum, 0).expect("upload");
        dev.cols.len += 1;
        let msg = dev.assert_invariants_device().unwrap_err();
        assert!(msg.contains("bucket lengths sum to"), "{msg}");
    }
}
