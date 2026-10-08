//! [`PartitionRuntime`]: the resolved placement, one pinned pool per partition, and the scoped fan-out every in-process partitioned call runs inside (ARCHITECTURE.md §Partitioning).
//! A layer runs inside `ThreadPool::install`, so transport calls come from a pool worker, which is why the MPI transport needs `MPI_THREAD_SERIALIZED`.

use std::sync::Arc;

use super::topology::{
    bind_current_thread_memory, build_pool, pin_current_thread, PartitionConfig, PartitionSlot,
    TopologyError,
};
use super::transport::InProcessTransport;

const LOG_TARGET: &str = "paulistrings::partitioned";

/// The resolved placement and its pinned pools, built once and shared behind an [`Arc`] by every [`PartitionedSum`](crate::PartitionedSum) that runs on it.
pub struct PartitionRuntime {
    /// In rank order.
    slots: Vec<PartitionSlot>,
    /// In rank order.
    pools: Vec<rayon::ThreadPool>,
    bind_memory: bool,
    /// How long a partition waits for a partner's collective before declaring it dead.
    wait_timeout: std::time::Duration,
}

pub(crate) const DEFAULT_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Longer, since device partitions may share a device and its queue.
pub(crate) const DEVICE_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

impl PartitionRuntime {
    /// Resolves `config` against the machine and builds one pinned pool per partition.
    ///
    /// # Errors
    ///
    /// Whatever [`PartitionConfig::resolve`] reports (a partition count that is not a power of two, an empty or out-of-mask explicit CPU set), or [`TopologyError::Io`] if a pool cannot be built.
    pub fn new(config: &PartitionConfig) -> Result<Arc<Self>, TopologyError> {
        Self::with_threads_per_partition(config, None)
    }

    /// [`new`](Self::new), with `Some(t)` giving every partition a `t`-worker pool (at least one) on the CPU set the placement resolved.
    ///
    /// A measurement knob for holding the total thread count fixed across partition counts; in production prefer [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// Whatever [`PartitionConfig::resolve`] reports (a partition count that is not a power of two, an empty or out-of-mask explicit CPU set), or [`TopologyError::Io`] if a pool cannot be built.
    pub fn with_threads_per_partition(
        config: &PartitionConfig,
        threads_per_partition: Option<usize>,
    ) -> Result<Arc<Self>, TopologyError> {
        let mut slots = config.resolve()?;
        if let Some(threads) = threads_per_partition {
            for slot in &mut slots {
                slot.threads = threads.max(1);
            }
        }
        let mut pools = Vec::with_capacity(slots.len());
        for (rank, slot) in slots.iter().enumerate() {
            pools.push(build_pool(
                slot,
                config.bind_memory,
                &format!("ps-part{rank}"),
            )?);
        }
        let wait_timeout = if slots.iter().any(|s| s.device.is_some()) {
            DEVICE_WAIT_TIMEOUT
        } else {
            DEFAULT_WAIT_TIMEOUT
        };
        Ok(Arc::new(Self {
            slots,
            pools,
            bind_memory: config.bind_memory,
            wait_timeout,
        }))
    }

    /// Partitions in the group — always a power of two.
    pub fn num_partitions(&self) -> usize {
        self.slots.len()
    }

    /// The resolved slots, in rank order.
    pub fn slots(&self) -> &[PartitionSlot] {
        &self.slots
    }

    /// `log2` of the partition count.
    pub(crate) fn partition_bits(&self) -> u8 {
        debug_assert!(self.slots.len().is_power_of_two());
        self.slots.len().trailing_zeros() as u8
    }

    /// A one-line summary of the placement, for the entry log line.
    pub(crate) fn placement_summary(&self) -> String {
        let mut summary = String::new();
        for (rank, slot) in self.slots.iter().enumerate() {
            if rank > 0 {
                summary.push_str(", ");
            }
            match (&slot.cpus, slot.device) {
                (_, Some(device)) => {
                    summary.push_str(&format!("{rank}:gpu{device} x{}", slot.threads))
                }
                (Some(cpus), None) => summary.push_str(&format!("{rank}:{cpus}x{}", slot.threads)),
                (None, None) => summary.push_str(&format!("{rank}:unpinned x{}", slot.threads)),
            }
        }
        summary
    }

    /// Runs `f` on partition 0's pool: the distributed shape of [`map_partitions`](Self::map_partitions).
    pub(crate) fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        self.pools[0].install(f)
    }

    /// Runs `f(rank, item, transport)` once per partition inside its own pool, concurrently, and returns the results in rank order; a partition's panic is re-raised here.
    pub(crate) fn map_partitions<I, O, F>(&self, items: Vec<I>, f: F) -> Vec<O>
    where
        I: Send,
        O: Send,
        F: Fn(usize, I, &InProcessTransport) -> O + Send + Sync,
    {
        let size = self.num_partitions();
        assert_eq!(items.len(), size, "map_partitions: one item per partition");
        // A fresh group per call, each partition owning its endpoint, so a partner's panic drops its senders and surfaces as a panic instead of a hang.
        let transports = InProcessTransport::group_with_timeout(size as u32, self.wait_timeout);
        let f = &f;

        let mut items = items.into_iter();
        let mut transports = transports.into_iter();
        let item0 = items.next().expect("at least one partition");
        let transport0 = transports.next().expect("at least one transport");

        std::thread::scope(|scope| {
            let handles: Vec<_> = items
                .zip(transports)
                .enumerate()
                .map(|(i, (item, transport))| {
                    let rank = i + 1;
                    let slot = &self.slots[rank];
                    let pool = &self.pools[rank];
                    let bind_memory = self.bind_memory;
                    scope.spawn(move || {
                        place_current_thread(slot, bind_memory);
                        pool.install(move || f(rank, item, &transport))
                    })
                })
                .collect();

            // The calling thread is not ours to pin; the work runs on pool 0's pinned workers.
            let mut out = Vec::with_capacity(size);
            out.push(self.pools[0].install(|| f(0, item0, &transport0)));
            for handle in handles {
                match handle.join() {
                    Ok(o) => out.push(o),
                    Err(payload) => std::panic::resume_unwind(payload),
                }
            }
            out
        })
    }
}

/// Pins a partition's driving thread to its slot, warning rather than failing when the platform declines.
fn place_current_thread(slot: &PartitionSlot, bind_memory: bool) {
    #[cfg(feature = "cuda")]
    if let Some(device) = slot.device {
        crate::engine::cuda_context::bind_device_context(device);
    }
    if let Some(cpus) = &slot.cpus {
        if let Err(err) = pin_current_thread(cpus) {
            log::warn!(target: LOG_TARGET, "failed to pin partition driver to {cpus}: {err}");
        }
    }
    if bind_memory {
        if let Err(err) = bind_current_thread_memory(slot.node) {
            log::warn!(
                target: LOG_TARGET,
                "failed to bind partition driver memory to {:?}: {err}",
                slot.node,
            );
        }
    }
}

#[cfg(test)]
mod tests;
