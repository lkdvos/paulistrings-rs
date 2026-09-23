//! Grow-only page-locked host buffers for device-to-host staging.

use std::sync::Arc;

use cudarc::driver::{result, CudaContext};

use super::error::GpuError;

/// A page-locked host buffer of `T`, allocated cached rather than write-combined.
/// cudarc's `alloc_pinned` is write-combined, which makes every host read of a download uncached.
pub(crate) struct PinnedBuf<T: Copy> {
    ctx: Arc<CudaContext>,
    ptr: *mut T,
    cap: usize,
}

// SAFETY: the buffer is plain host memory owned by this value; every access goes through `&self`/`&mut self`.
unsafe impl<T: Copy + Send> Send for PinnedBuf<T> {}
unsafe impl<T: Copy + Sync> Sync for PinnedBuf<T> {}

impl<T: Copy> PinnedBuf<T> {
    pub(crate) fn new(ctx: Arc<CudaContext>) -> Self {
        Self {
            ctx,
            ptr: std::ptr::null_mut(),
            cap: 0,
        }
    }

    /// Grow to at least `n` elements, discarding the contents; a no-op when already large enough.
    pub(crate) fn ensure(&mut self, n: usize) -> Result<(), GpuError> {
        if n <= self.cap {
            return Ok(());
        }
        self.free();
        let bytes = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(GpuError::Unsupported("host staging larger than usize"))?;
        let ordinal = self.ctx.ordinal() as u32;
        self.ctx.bind_to_thread().map_err(GpuError::from)?;
        // SAFETY: flag 0 is a plain portable-less, cached page-locked allocation of `bytes` bytes.
        let ptr = unsafe { result::malloc_host(bytes, 0) }
            .map_err(|e| GpuError::from_alloc(e, ordinal, bytes as u64))?
            as *mut T;
        // SAFETY: `ptr` holds `bytes` writable bytes; zeroing makes every later `&[T]` view initialized.
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes) };
        self.ptr = ptr;
        self.cap = n;
        Ok(())
    }

    pub(crate) fn slice(&self, n: usize) -> &[T] {
        assert!(n <= self.cap, "PinnedBuf: {n} beyond capacity {}", self.cap);
        if n == 0 {
            return &[];
        }
        // SAFETY: `ptr` is live, initialized and `cap >= n` elements long.
        unsafe { std::slice::from_raw_parts(self.ptr, n) }
    }

    pub(crate) fn slice_mut(&mut self, n: usize) -> &mut [T] {
        assert!(n <= self.cap, "PinnedBuf: {n} beyond capacity {}", self.cap);
        if n == 0 {
            return &mut [];
        }
        // SAFETY: as `slice`, and `&mut self` makes the view unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, n) }
    }

    fn free(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        // Freeing needs the context current; a failure here leaks the buffer rather than panicking in drop.
        if self.ctx.bind_to_thread().is_ok() {
            // SAFETY: `ptr` came from `malloc_host` and no copy into it is in flight, since every writer synchronizes before returning.
            let _ = unsafe { result::free_host(self.ptr as *mut std::ffi::c_void) };
        }
        self.ptr = std::ptr::null_mut();
        self.cap = 0;
    }
}

impl<T: Copy> Drop for PinnedBuf<T> {
    fn drop(&mut self) {
        self.free();
    }
}

/// The three staged columns of one download.
pub(crate) struct HostStaging {
    pub(crate) x: PinnedBuf<u64>,
    pub(crate) z: PinnedBuf<u64>,
    pub(crate) coeff: PinnedBuf<f64>,
}

impl HostStaging {
    pub(crate) fn new(ctx: &Arc<CudaContext>) -> Self {
        Self {
            x: PinnedBuf::new(ctx.clone()),
            z: PinnedBuf::new(ctx.clone()),
            coeff: PinnedBuf::new(ctx.clone()),
        }
    }

    /// Room for `terms` terms of width `w`.
    pub(crate) fn ensure(&mut self, terms: usize, w: usize) -> Result<(), GpuError> {
        self.x.ensure(terms * w)?;
        self.z.ensure(terms * w)?;
        self.coeff.ensure(2 * terms)
    }
}
