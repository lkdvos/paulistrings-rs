//! [`ChunkPipeline`], the in-flight half of a two-phase MPI exchange that the coset loop waits on chunk by chunk.

use std::sync::Mutex;

use ::mpi::request::RequestCollection;

use super::super::transport::ChunkWait;

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

/// Spin tier of a worker waiting on another's MPI drive, before it yields.
const PIPELINE_SPINS: u32 = 256;

/// Yield tier, before it sleeps.
const PIPELINE_YIELDS: u32 = 64;

/// Sleep tier step, which keeps waiters off the cores the driving worker copies on.
const PIPELINE_SLEEP: std::time::Duration = std::time::Duration::from_micros(20);

/// The posted bulk receives and sends of a two-phase exchange, waited on chunk by chunk as a [`ChunkWait`].
///
/// Whichever worker takes the mutex drives `MPI_Waitsome` over every request and credits completions to per-chunk counters the others watch; why that is `MPI_THREAD_SERIALIZED`-safe and deadlock-free: ARCHITECTURE.md §Partitioning.
pub(super) struct ChunkPipeline<'a, 'c> {
    /// Only `try_lock`ed: a waiter must not queue behind a driver blocked inside MPI.
    inner: Mutex<PipelineInner<'a, 'c>>,
    /// Outstanding receives per chunk.
    remaining: Vec<std::sync::atomic::AtomicUsize>,
}

struct PipelineInner<'a, 'c> {
    coll: &'c mut RequestCollection<'a, [u8]>,
    /// Slot per request index.
    slot_of: Vec<usize>,
    done: Vec<(usize, ::mpi::point_to_point::Status, &'a [u8])>,
}

// SAFETY: `MPI_Request` is a raw handle, so `RequestCollection` is neither `Send` nor `Sync`.
// Every worker reaches it only through `inner`'s mutex, so at most one thread is inside MPI, as `MPI_THREAD_SERIALIZED` allows.
unsafe impl Send for ChunkPipeline<'_, '_> {}
unsafe impl Sync for ChunkPipeline<'_, '_> {}

impl<'a, 'c> ChunkPipeline<'a, 'c> {
    /// Wrap `coll`, whose request `i` belongs to chunk `slot_of[i]`, with `chunks` chunk counters.
    pub(super) fn new(
        coll: &'c mut RequestCollection<'a, [u8]>,
        slot_of: Vec<usize>,
        chunks: usize,
    ) -> Self {
        let remaining: Vec<_> = (0..chunks)
            .map(|_| std::sync::atomic::AtomicUsize::new(0))
            .collect();
        for &slot in &slot_of {
            if slot < chunks {
                remaining[slot].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Self {
            inner: Mutex::new(PipelineInner {
                coll,
                slot_of,
                done: Vec::new(),
            }),
            remaining,
        }
    }

    /// Drive MPI once if nobody else is: `None` when another thread holds it, `Some(false)` when nothing is outstanding.
    fn try_drive(&self) -> Option<bool> {
        let mut inner = match self.inner.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return None,
        };
        let PipelineInner {
            coll,
            slot_of,
            done,
        } = &mut *inner;
        if coll.incomplete() == 0 {
            return Some(false);
        }
        coll.wait_some(done);
        for &(index, _, _) in done.iter() {
            let slot = slot_of[index];
            if slot < self.remaining.len() {
                self.remaining[slot].fetch_sub(1, std::sync::atomic::Ordering::Release);
            }
        }
        Some(true)
    }

    fn wait_slot(&self, k: usize) {
        let mut spins: u32 = 0;
        loop {
            if self.remaining[k].load(std::sync::atomic::Ordering::Acquire) == 0 {
                return;
            }
            match self.try_drive() {
                Some(false) => return,
                Some(true) => spins = 0,
                None => {
                    spins = spins.saturating_add(1);
                    if spins <= PIPELINE_SPINS {
                        std::hint::spin_loop();
                    } else if spins <= PIPELINE_SPINS + PIPELINE_YIELDS {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(PIPELINE_SLEEP);
                    }
                }
            }
        }
    }

    /// Wait out every outstanding request, sends included.
    pub(super) fn finish(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(|e| e.into_inner());
        inner.coll.wait_all(&mut inner.done);
        for slot in &self.remaining {
            slot.store(0, std::sync::atomic::Ordering::Release);
        }
    }
}

impl ChunkWait for ChunkPipeline<'_, '_> {
    fn wait_chunk(&self, k: usize) {
        self.wait_slot(k);
    }
}
