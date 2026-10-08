//! [`ChunkPipeline`], the in-flight half of a two-phase MPI exchange that the coset loop waits on chunk by chunk.

use std::sync::Mutex;

use ::mpi::request::RequestCollection;

use super::super::transport::ChunkWait;

/// Slot number a send request carries: not a chunk, and never waited on by name — the closing [`ChunkPipeline::finish`] waits it out with the rest.
pub(super) const SLOT_SEND: usize = usize::MAX;

/// Slot number an early receive carries: waited out by the driving thread before the body starts, so the pipeline never counts it.
pub(super) const SLOT_EARLY: usize = usize::MAX - 1;

/// Record that request `i` belongs to `slot` — a chunk index, [`SLOT_SEND`] or [`SLOT_EARLY`] — growing the table as the request collection grows.
///
/// `RequestCollection::add` hands out consecutive indices, so the resize is a push; it is written as one anyway, because nothing in the collection's contract promises that.
pub(super) fn note_slot(slot_of: &mut Vec<usize>, i: usize, slot: usize) {
    if slot_of.len() <= i {
        slot_of.resize(i + 1, SLOT_SEND);
    }
    slot_of[i] = slot;
}

/// Spin iterations a waiting Rayon worker burns before it starts yielding.
///
/// Only one thread may be inside MPI at a time
/// (`MPI_THREAD_SERIALIZED`), so the worker that wins the pipeline's mutex
/// drives the transfer for everybody and the rest wait on its published
/// counters. A chunk is milliseconds of copying, so this tier exists only for
/// the case where the chunk landed while this worker was on its way in.
const PIPELINE_SPINS: u32 = 256;

/// `yield_now` calls after the spin tier before a waiter starts sleeping.
const PIPELINE_YIELDS: u32 = 64;

/// How long a waiter sleeps per iteration once yielding has not helped.
///
/// A yield loop is not free to the rest of the machine: it takes its share of the very CPUs the driving worker's copy is running on.
/// Past this point the chunk is not close, so paying a step of latency to stay off those cores is the right trade — the same argument as the in-process collectives' three wait tiers in [`transport`](super::transport).
const PIPELINE_SLEEP: std::time::Duration = std::time::Duration::from_micros(20);

/// The in-flight half of a two-phase exchange: the posted bulk receives, this rank's sends, and which chunk each receive belongs to.
///
/// Handed to the coset loop as a [`ChunkWait`].
/// A task that reaches `append_into` for a bucket in chunk `k` calls `wait_chunk(k)`; whichever worker takes the mutex drives MPI until something completes, credits it to its chunk and releases, and the waiters see the counter fall.
///
/// # One thread at a time is enough
///
/// The application requested `MPI_THREAD_SERIALIZED`, so exactly one thread may be inside the library at once, which the mutex enforces.
/// Nothing is lost by serializing: MPI's progress engine moves every outstanding request of this rank, receives and sends alike, so one thread inside `MPI_Waitsome` is driving the whole layer's transfer.
///
/// # It cannot deadlock
///
/// Every send of the call is posted before the call's first receive, on every rank, so a partner's rendezvous always has a matching posted receive to land in.
/// A rank blocked in `wait_chunk` is inside `MPI_Waitsome` over *all* of its requests, so it services its partner's transfer as well as its own.
/// A rank whose coset loop never asks for a chunk still reaches [`finish`](Self::finish), which waits everything out.
/// And a coset task waits for exactly one chunk (`ChunkMap` puts a whole coset in one), so no task holds a wait on a chunk behind a wait on another.
pub(super) struct ChunkPipeline<'a, 'c> {
    /// The request collection and its bookkeeping.
    /// `try_lock`ed, never blocked on: a worker that cannot get in must not queue behind the driver, which is itself blocked inside MPI.
    inner: Mutex<PipelineInner<'a, 'c>>,
    /// Outstanding receives per chunk. Sends are not counted.
    remaining: Vec<std::sync::atomic::AtomicUsize>,
}

struct PipelineInner<'a, 'c> {
    coll: &'c mut RequestCollection<'a, [u8]>,
    /// Chunk per request index, [`SLOT_SEND`] for a send or an early part.
    slot_of: Vec<usize>,
    /// Reused `wait_some` result buffer.
    done: Vec<(usize, ::mpi::point_to_point::Status, &'a [u8])>,
}

// SAFETY: `RequestCollection` is neither `Send` nor `Sync`, because `MPI_Request`
// is a raw handle. It is reachable here from any worker of the partition's Rayon
// pool, but only through `inner`'s mutex, so at most one thread is ever inside
// MPI — exactly what `MPI_THREAD_SERIALIZED` allows and what this module's docs
// require of the application. Same argument as the `unsafe impl`s on
// `MpiTransport`.
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

    /// Drive MPI once from this thread if nobody else is inside it.
    ///
    /// `None` when another thread holds the pipeline, `Some(false)` when nothing is outstanding at all.
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

    /// Block until every receive of chunk `k` has completed.
    fn wait_slot(&self, k: usize) {
        let mut spins: u32 = 0;
        loop {
            if self.remaining[k].load(std::sync::atomic::Ordering::Acquire) == 0 {
                return;
            }
            match self.try_drive() {
                // Nothing outstanding: this chunk arrived (or never existed).
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

    /// Wait every outstanding request out: the receives the coset loop never reached, and this rank's own sends.
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
