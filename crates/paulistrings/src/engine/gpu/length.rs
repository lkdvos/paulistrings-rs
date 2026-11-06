//! [`LengthTracker`], the host's bound on a device sum's length and longest bucket, anchored by lagged read-backs.
//!
//! Contracts: every decision reads the bound or a consumed anchor, never a read-back that merely happens to have landed, so decisions do not depend on timing.
//! The bound is a true upper bound on the device's live terms and longest bucket; a step that cannot bound the longest bucket makes it unknown until the next anchor.

use std::collections::VecDeque;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream};

use super::error::GpuError;
use super::staging::PinnedBuffer;

/// Read-backs a lone sum leaves in flight; a group member's sum consumes every read-back at once.
pub(super) const LONE_LAG: usize = 1;

/// How one device step moves the length and the longest bucket, as far as the host can tell without reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Growth {
    /// A join: at most twice the terms, and no output bucket longer than two source buckets.
    Join,
    /// Terms only drop, to at most `cap`, and buckets only shorten: a cut, a rescale, a refine.
    Shrink { cap: usize },
    /// Terms only drop, but rows move between buckets, so the longest is unknown: a permutation.
    Regroup,
}

/// A sum's length and longest bucket, exactly or as upper bounds; `longest` is `None` when unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Lengths {
    pub(crate) len: usize,
    pub(crate) longest: Option<u32>,
}

impl Lengths {
    fn after(self, growth: Growth) -> Self {
        match growth {
            Growth::Join => Self {
                len: self.len.saturating_mul(2),
                longest: self.longest.map(|l| l.saturating_mul(2)),
            },
            Growth::Shrink { cap } => Self { len: self.len.min(cap), longest: self.longest },
            Growth::Regroup => Self { len: self.len, longest: None },
        }
    }

    /// As [`Self::after`], but a join keeps the length: the stale estimate a bucket-count heuristic reads.
    fn estimate_after(self, growth: Growth) -> Self {
        match growth {
            Growth::Join => self,
            _ => self.after(growth),
        }
    }
}

/// An asynchronous copy of the device's `[len, longest]` after one step into pinned memory, and the event that marks it landed.
struct Readback {
    host: PinnedBuffer<u32>,
    event: CudaEvent,
}

/// The host's view of a device sum's length: an exact anchor, the steps applied since, and the read-backs of the states after some of them.
pub(crate) struct LengthTracker {
    anchor: Lengths,
    /// Steps since the anchor's state, oldest first, each with the read-back of the state after it, if it published one.
    steps: VecDeque<(Growth, Option<Readback>)>,
    /// Read-backs no step holds, for reuse.
    pool: Vec<Readback>,
    /// Read-backs left in flight after a step publishes one.
    lag: usize,
    context: Arc<CudaContext>,
}

impl LengthTracker {
    pub(crate) fn new(context: Arc<CudaContext>, exact: Lengths, lag: usize) -> Self {
        Self { anchor: exact, steps: VecDeque::new(), pool: Vec::new(), lag, context }
    }

    /// The upper bound on the current state.
    pub(crate) fn bound(&self) -> Lengths {
        self.steps.iter().fold(self.anchor, |known, &(growth, _)| known.after(growth))
    }

    /// The anchor carried through the steps with every join taken as keeping the length: a deterministic estimate for heuristics, not a bound.
    pub(crate) fn estimate(&self) -> Lengths {
        self.steps.iter().fold(self.anchor, |known, &(growth, _)| known.estimate_after(growth))
    }

    /// Whether the bound is the exact state.
    pub(crate) fn is_exact(&self) -> bool {
        self.steps.is_empty()
    }

    /// The current state is exactly `lengths`, read synchronously by the step that made it.
    pub(crate) fn set_exact(&mut self, lengths: Lengths) {
        self.anchor = lengths;
        while let Some((_, readback)) = self.steps.pop_front() {
            self.pool.extend(readback);
        }
    }

    /// A step the host bounds by `growth`; with `published`, the device's `[len, longest]` after it is copied back asynchronously on `stream`, and every read-back but the latest `lag` is consumed.
    pub(crate) fn step(
        &mut self,
        growth: Growth,
        published: Option<(&Arc<CudaStream>, &CudaSlice<u32>)>,
    ) -> Result<(), GpuError> {
        let readback = match published {
            Some((stream, device)) => Some(self.publish(stream, device)?),
            None => None,
        };
        self.steps.push_back((growth, readback));
        self.consume(self.lag)
    }

    fn publish(
        &mut self,
        stream: &Arc<CudaStream>,
        device: &CudaSlice<u32>,
    ) -> Result<Readback, GpuError> {
        let mut readback = match self.pool.pop() {
            Some(readback) => readback,
            None => Readback {
                host: PinnedBuffer::new(self.context.clone()),
                event: self.context.new_event(None)?,
            },
        };
        readback.host.ensure(2)?;
        stream.memcpy_dtoh(&device.slice(0..2), readback.host.slice_mut(2))?;
        readback.event.record(stream)?;
        Ok(readback)
    }

    /// A test's unpublished regrouping step, so the longest bucket is unknown until the next anchor.
    #[cfg(test)]
    pub(crate) fn forget_longest(&mut self) {
        self.steps.push_back((Growth::Regroup, None));
    }

    /// Consume every read-back but the latest `keep`, oldest first, waiting on each; its state becomes the anchor.
    pub(crate) fn consume(&mut self, keep: usize) -> Result<(), GpuError> {
        let published = self.steps.iter().filter(|(_, readback)| readback.is_some()).count();
        for _ in keep..published {
            let at = self
                .steps
                .iter()
                .position(|(_, readback)| readback.is_some())
                .expect("a published step");
            let readback = self.steps[at].1.as_ref().expect("a published step");
            readback.event.synchronize()?;
            let read = readback.host.slice(2);
            let exact = Lengths { len: read[0] as usize, longest: Some(read[1]) };
            for _ in 0..=at {
                let (_, readback) = self.steps.pop_front().expect("steps up to the read-back");
                self.pool.extend(readback);
            }
            self.anchor = exact;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
