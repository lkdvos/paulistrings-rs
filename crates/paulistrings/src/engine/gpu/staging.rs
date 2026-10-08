//! Grow-only page-locked host buffers for device-to-host staging.

use std::sync::Arc;

use cudarc::driver::{result, CudaContext};

use super::error::GpuError;

/// A page-locked host buffer of `T`, allocated cached: cudarc's write-combined `alloc_pinned` makes every host read of a download uncached.
pub(crate) struct PinnedBuffer<T: Copy> {
    context: Arc<CudaContext>,
    ptr: *mut T,
    capacity: usize,
}

// SAFETY: the buffer is plain host memory owned by this value; every access goes through `&self`/`&mut self`.
unsafe impl<T: Copy + Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for PinnedBuffer<T> {}

impl<T: Copy> PinnedBuffer<T> {
    pub(crate) fn new(context: Arc<CudaContext>) -> Self {
        Self {
            context,
            ptr: std::ptr::null_mut(),
            capacity: 0,
        }
    }

    /// Grow to at least `n` elements, discarding the contents; a no-op when already large enough.
    pub(crate) fn ensure(&mut self, n: usize) -> Result<(), GpuError> {
        if n <= self.capacity {
            return Ok(());
        }
        self.free();
        let bytes = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(GpuError::Unsupported("host staging larger than usize"))?;
        let ordinal = self.context.ordinal() as u32;
        self.context.bind_to_thread().map_err(GpuError::from)?;
        // SAFETY: flag 0 is a plain portable-less, cached page-locked allocation of `bytes` bytes.
        let ptr = unsafe { result::malloc_host(bytes, 0) }
            .map_err(|e| GpuError::from_alloc(e, ordinal, bytes as u64))?
            as *mut T;
        self.ptr = ptr;
        self.capacity = n;
        Ok(())
    }

    pub(crate) fn slice(&self, n: usize) -> &[T] {
        assert!(
            n <= self.capacity,
            "PinnedBuffer: {n} beyond capacity {}",
            self.capacity
        );
        if n == 0 {
            return &[];
        }
        // SAFETY: `ptr` is live and `cap >= n` elements long; the only reader, `GpuSum::to_host`, reads what its download just wrote.
        unsafe { std::slice::from_raw_parts(self.ptr, n) }
    }

    pub(crate) fn slice_mut(&mut self, n: usize) -> &mut [T] {
        assert!(
            n <= self.capacity,
            "PinnedBuffer: {n} beyond capacity {}",
            self.capacity
        );
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
        if self.context.bind_to_thread().is_ok() {
            // SAFETY: `ptr` came from `malloc_host` and no copy into it is in flight, since every writer synchronizes before returning.
            let _ = unsafe { result::free_host(self.ptr as *mut std::ffi::c_void) };
        }
        self.ptr = std::ptr::null_mut();
        self.capacity = 0;
    }
}

impl<T: Copy> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        self.free();
    }
}

/// The three staged columns of one download.
pub(crate) struct HostStaging {
    pub(crate) x: PinnedBuffer<u64>,
    pub(crate) z: PinnedBuffer<u64>,
    pub(crate) coeff: PinnedBuffer<f64>,
}

impl HostStaging {
    pub(crate) fn new(context: &Arc<CudaContext>) -> Self {
        Self {
            x: PinnedBuffer::new(context.clone()),
            z: PinnedBuffer::new(context.clone()),
            coeff: PinnedBuffer::new(context.clone()),
        }
    }

    /// Room for `terms` terms of width `w`.
    pub(crate) fn ensure(&mut self, terms: usize, w: usize) -> Result<(), GpuError> {
        self.x.ensure(terms * w)?;
        self.z.ensure(terms * w)?;
        self.coeff.ensure(2 * terms)
    }
}
