//! [`ChunkPipeline`], the in-flight half of a two-phase MPI exchange that the coset loop waits on chunk by chunk.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

use ::mpi::request::RequestCollection;

use crate::engine::transport::ChunkWait;

/// Slot of a send request, waited out by [`ChunkPipeline::finish`].
pub(super) const SLOT_SEND: usize = usize::MAX;

/// Slot of an early receive, waited out before the body starts.
pub(super) const SLOT_EARLY: usize = usize::MAX - 1;

/// Record that request `i` belongs to `slot`, a chunk index or one of the two sentinels.
pub(super) fn note_slot(slot_of: &mut Vec<usize>, i: usize, slot: usize) {
    if slot_of.len() <= i {
        slot_of.resize(i + 1, SLOT_SEND);
    }
    slot_of[i] = slot;
}

/// The posted bulk receives and sends of a two-phase exchange, waited on chunk by chunk as a [`ChunkWait`].
///
/// One waiter at a time holds the driver role and drives `MPI_Waitsome` over every request of the rank until its own chunk completes, notifying the condvar on each completion and on release; the others sleep on it.
/// One role holder is exactly `MPI_THREAD_SERIALIZED`, and it cannot deadlock: a waiter sleeps only while the role is held, so a rank with an incomplete chunk always has a thread in `MPI_Waitsome`; every send of a call is posted before its first receive, so a waiting rank services its partner; and a rank whose coset loop never asks for a chunk still reaches the closing wait.
pub(super) struct ChunkPipeline<'a, 'c> {
    /// Locked only by the driver-role holder, so never contended.
    inner: Mutex<PipelineInner<'a, 'c>>,
    /// Outstanding receives per chunk.
    remaining: Vec<AtomicUsize>,
    /// Whether a thread holds the driver role; never held across an MPI call.
    driving: Mutex<bool>,
    /// Notified when a chunk completes and when the driver role is released.
    wake: Condvar,
}

struct PipelineInner<'a, 'c> {
    requests: &'c mut RequestCollection<'a, [u8]>,
    /// Slot per request index.
    slot_of: Vec<usize>,
    done: Vec<(usize, ::mpi::point_to_point::Status, &'a [u8])>,
}

// SAFETY: `MPI_Request` is a raw handle, so `RequestCollection` is neither `Send` nor `Sync`.
// Every worker reaches it only through `inner`'s mutex, taken only by the one driver-role holder, so at most one thread is inside MPI, as `MPI_THREAD_SERIALIZED` allows.
unsafe impl Send for ChunkPipeline<'_, '_> {}
unsafe impl Sync for ChunkPipeline<'_, '_> {}

/// The driver role, released and announced on drop, unwinding included.
struct DriverRole<'p, 'a, 'c>(&'p ChunkPipeline<'a, 'c>);

impl Drop for DriverRole<'_, '_, '_> {
    fn drop(&mut self) {
        *self.0.lock_driving() = false;
        self.0.wake.notify_all();
    }
}

impl<'a, 'c> ChunkPipeline<'a, 'c> {
    /// Wrap `requests`, whose request `i` belongs to chunk `slot_of[i]`, with `chunks` chunk counters.
    pub(super) fn new(
        requests: &'c mut RequestCollection<'a, [u8]>,
        slot_of: Vec<usize>,
        chunks: usize,
    ) -> Self {
        let remaining: Vec<_> = (0..chunks).map(|_| AtomicUsize::new(0)).collect();
        for &slot in &slot_of {
            if slot < chunks {
                remaining[slot].fetch_add(1, Ordering::Relaxed);
            }
        }
        Self {
            inner: Mutex::new(PipelineInner {
                requests,
                slot_of,
                done: Vec::new(),
            }),
            remaining,
            driving: Mutex::new(false),
            wake: Condvar::new(),
        }
    }

    fn lock_driving(&self) -> MutexGuard<'_, bool> {
        self.driving.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn complete(&self, k: usize) -> bool {
        self.remaining[k].load(Ordering::Acquire) == 0
    }

    /// As the driver, `MPI_Waitsome` until chunk `k` completes or nothing is outstanding, announcing every chunk that completes.
    fn drive_until(&self, k: usize) {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let PipelineInner {
            requests,
            slot_of,
            done,
        } = &mut *inner;
        while !self.complete(k) && requests.incomplete() > 0 {
            requests.wait_some(done);
            let mut completed = false;
            for &(index, _, _) in done.iter() {
                let slot = slot_of[index];
                if slot < self.remaining.len() {
                    completed |= self.remaining[slot].fetch_sub(1, Ordering::Release) == 1;
                }
            }
            if completed {
                // Taking the lock orders the decrement before a waiter's check-then-sleep.
                drop(self.lock_driving());
                self.wake.notify_all();
            }
        }
    }

    fn wait_slot(&self, k: usize) {
        if self.complete(k) {
            return;
        }
        let mut driving = self.lock_driving();
        while !self.complete(k) {
            if *driving {
                driving = self
                    .wake
                    .wait(driving)
                    .unwrap_or_else(PoisonError::into_inner);
                continue;
            }
            *driving = true;
            drop(driving);
            let role = DriverRole(self);
            self.drive_until(k);
            drop(role);
            return;
        }
    }

    /// Wait out every outstanding request, sends included.
    pub(super) fn finish(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(PoisonError::into_inner);
        inner.requests.wait_all(&mut inner.done);
        for slot in &self.remaining {
            slot.store(0, Ordering::Release);
        }
    }
}

impl ChunkWait for ChunkPipeline<'_, '_> {
    fn wait_chunk(&self, k: usize) {
        self.wait_slot(k);
    }
}

#[cfg(test)]
mod tests;
