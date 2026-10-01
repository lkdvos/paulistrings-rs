//! Device-resident storage of one sum: flat structure-of-arrays term columns plus a per-bucket CSR.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};

use super::error::GpuError;

/// Term columns and bucket CSR on one device, grow-only.
/// Term `i` is `x[iW..(i+1)W]`, `z[iW..(i+1)W]`, `coeff[2i..2i+2]` (a `Complex64`'s `repr(C)` bytes) and `g[i]`.
/// Bucket `β` owns terms `start[β]..start[β] + lens[β]`; buckets need not be contiguous or in index order.
pub(crate) struct DeviceColumns<const W: usize> {
    pub(crate) x: CudaSlice<u64>,
    pub(crate) z: CudaSlice<u64>,
    pub(crate) coeff: CudaSlice<f64>,
    pub(crate) g: CudaSlice<u64>,
    /// One entry per bucket plus one, so an exclusive scan of `lens` can land here directly.
    pub(crate) start: CudaSlice<u32>,
    pub(crate) lens: CudaSlice<u32>,
    /// Live terms.
    pub(crate) len: usize,
    /// Live CSR entries.
    pub(crate) buckets: usize,
    term_cap: usize,
    bucket_cap: usize,
    stream: Arc<CudaStream>,
    ordinal: u32,
}

/// `n` uninitialized elements; `bytes` is what an out-of-memory error reports.
fn alloc<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    n: usize,
    ordinal: u32,
    bytes: u64,
) -> Result<CudaSlice<T>, GpuError> {
    // SAFETY: device memory is never read on the host before a kernel or copy wrote it; every kernel reads only rows below the live length.
    unsafe { stream.alloc::<T>(n.max(1)) }.map_err(|e| GpuError::from_alloc(e, ordinal, bytes))
}

/// Room for `n` elements in `s`, keeping its first `keep`; a no-op when it already has room, and `s` untouched on failure.
/// `bytes` is what an out-of-memory error reports.
pub(crate) fn grow_keep<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    s: &mut CudaSlice<T>,
    n: usize,
    keep: usize,
    ordinal: u32,
    bytes: u64,
) -> Result<(), GpuError> {
    if s.len() >= n {
        return Ok(());
    }
    let mut next = alloc(stream, n, ordinal, bytes)?;
    if keep > 0 {
        stream.memcpy_dtod(&s.slice(0..keep), &mut next.slice_mut(0..keep))?;
    }
    *s = next;
    Ok(())
}

/// Room for `n` elements in `s`, discarding its contents.
pub(crate) fn grow<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    s: &mut CudaSlice<T>,
    n: usize,
    ordinal: u32,
) -> Result<(), GpuError> {
    grow_keep(
        stream,
        s,
        n,
        0,
        ordinal,
        (n * std::mem::size_of::<T>()) as u64,
    )
}

impl<const W: usize> DeviceColumns<W> {
    /// Resident device bytes per term: `x`, `z`, `coeff` and `g`.
    pub(crate) const BYTES_PER_TERM: usize = 16 * W + 24;

    /// Empty columns with room for `terms` terms and `buckets` buckets.
    pub(crate) fn with_capacity(
        stream: &Arc<CudaStream>,
        ordinal: u32,
        terms: usize,
        buckets: usize,
    ) -> Result<Self, GpuError> {
        let bytes = Self::request_bytes(terms, buckets)?;
        Ok(Self {
            x: alloc(stream, terms * W, ordinal, bytes)?,
            z: alloc(stream, terms * W, ordinal, bytes)?,
            coeff: alloc(stream, 2 * terms, ordinal, bytes)?,
            g: alloc(stream, terms, ordinal, bytes)?,
            start: alloc(stream, buckets + 1, ordinal, bytes)?,
            lens: alloc(stream, buckets, ordinal, bytes)?,
            len: 0,
            buckets: 0,
            term_cap: terms,
            bucket_cap: buckets,
            stream: stream.clone(),
            ordinal,
        })
    }

    pub(crate) fn term_capacity(&self) -> usize {
        self.term_cap
    }

    /// Bytes a `(terms, buckets)` allocation needs, or `Unsupported` past the `u32` index range.
    fn request_bytes(terms: usize, buckets: usize) -> Result<u64, GpuError> {
        if terms > u32::MAX as usize || buckets >= u32::MAX as usize {
            return Err(GpuError::Unsupported(
                "more than u32::MAX terms or buckets on one device",
            ));
        }
        let csr = if buckets > 0 {
            (2 * buckets as u64 + 1) * 4
        } else {
            0
        };
        Ok(terms as u64 * Self::BYTES_PER_TERM as u64 + csr)
    }

    /// Grow to hold at least `terms` terms and `buckets` buckets, keeping the live terms and CSR entries.
    /// The term columns and the CSR grow independently, so a refine that only adds buckets never copies a term.
    /// An allocation failure reports `OutOfMemory` with the bytes of the whole request, the capacities unchanged.
    pub(crate) fn reserve(&mut self, terms: usize, buckets: usize) -> Result<(), GpuError> {
        let grow_terms = terms > self.term_cap;
        let grow_buckets = buckets > self.bucket_cap;
        if !grow_terms && !grow_buckets {
            return Ok(());
        }
        let (st, o) = (&self.stream, self.ordinal);
        let bytes = Self::request_bytes(
            if grow_terms { terms } else { 0 },
            if grow_buckets { buckets } else { 0 },
        )?;
        let (n, b) = (self.len, self.buckets);
        if grow_terms {
            grow_keep(st, &mut self.x, terms * W, n * W, o, bytes)?;
            grow_keep(st, &mut self.z, terms * W, n * W, o, bytes)?;
            grow_keep(st, &mut self.coeff, 2 * terms, 2 * n, o, bytes)?;
            grow_keep(st, &mut self.g, terms, n, o, bytes)?;
            self.term_cap = terms;
        }
        if grow_buckets {
            let live = if b > 0 { b + 1 } else { 0 };
            grow_keep(st, &mut self.start, buckets + 1, live, o, bytes)?;
            grow_keep(st, &mut self.lens, buckets, b, o, bytes)?;
            self.bucket_cap = buckets;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::gpu::device;

    fn stream() -> Arc<CudaStream> {
        device::context(0).expect("context").default_stream()
    }

    #[test]
    fn reserve_of_an_absurd_term_count_is_out_of_memory() {
        crate::require_cuda!();
        let s = stream();
        let mut cols = DeviceColumns::<16>::with_capacity(&s, 0, 1, 1).expect("tiny alloc");
        let terms = 4_000_000_000usize;
        match cols.reserve(terms, 1) {
            Err(GpuError::OutOfMemory { device, bytes }) => {
                assert_eq!(device, 0);
                assert_eq!(bytes, terms as u64 * (16 * 16 + 24));
            }
            other => panic!("expected OutOfMemory, got {other:?}"),
        }
        assert_eq!(
            cols.term_capacity(),
            1,
            "a failed reserve leaves the columns untouched"
        );
    }

    #[test]
    fn reserve_past_the_u32_index_range_is_unsupported() {
        crate::require_cuda!();
        let s = stream();
        let mut cols = DeviceColumns::<1>::with_capacity(&s, 0, 1, 1).expect("tiny alloc");
        assert!(matches!(
            cols.reserve(1usize << 33, 1),
            Err(GpuError::Unsupported(_))
        ));
    }

    #[test]
    fn reserve_keeps_the_live_terms_and_csr() {
        crate::require_cuda!();
        let s = stream();
        let mut cols = DeviceColumns::<2>::with_capacity(&s, 0, 3, 2).expect("alloc");
        let x: Vec<u64> = (0..6).collect();
        let z: Vec<u64> = (10..16).collect();
        let c: Vec<f64> = (0..6).map(|i| i as f64 + 0.5).collect();
        let g: Vec<u64> = vec![7, 8, 9];
        s.memcpy_htod(&x, &mut cols.x).unwrap();
        s.memcpy_htod(&z, &mut cols.z).unwrap();
        s.memcpy_htod(&c, &mut cols.coeff).unwrap();
        s.memcpy_htod(&g, &mut cols.g).unwrap();
        s.memcpy_htod(&[0u32, 1, 3], &mut cols.start).unwrap();
        s.memcpy_htod(&[1u32, 2], &mut cols.lens).unwrap();
        cols.len = 3;
        cols.buckets = 2;
        cols.reserve(1000, 64).expect("grow");
        assert_eq!(cols.term_capacity(), 1000);
        cols.reserve(10, 4096).expect("grow the CSR alone");
        assert_eq!(cols.term_capacity(), 1000);
        assert_eq!(s.clone_dtoh(&cols.x.slice(0..6)).unwrap(), x);
        assert_eq!(s.clone_dtoh(&cols.z.slice(0..6)).unwrap(), z);
        assert_eq!(s.clone_dtoh(&cols.coeff.slice(0..6)).unwrap(), c);
        assert_eq!(s.clone_dtoh(&cols.g.slice(0..3)).unwrap(), g);
        assert_eq!(
            s.clone_dtoh(&cols.start.slice(0..3)).unwrap(),
            vec![0, 1, 3]
        );
        assert_eq!(s.clone_dtoh(&cols.lens.slice(0..2)).unwrap(), vec![1, 2]);
        assert_eq!((cols.len, cols.buckets), (3, 2));
    }
}
