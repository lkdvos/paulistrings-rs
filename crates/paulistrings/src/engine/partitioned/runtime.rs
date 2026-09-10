//! [`PartitionRuntime`]: the placement resolved once, the pinned pools built
//! once, and the scoped fan-out every partitioned call runs inside.
//!
//! A partitioned run has two kinds of thread. Each partition has one
//! **driving** thread that walks the layers, issues the collectives and calls
//! into the engine; partition 0's driving thread is the *calling* thread, and
//! partitions `1..P` get a scoped thread each. Inside a partition, the layer
//! itself runs on that partition's own pinned Rayon pool — the driving thread
//! enters it with `ThreadPool::install`, so every allocation a layer makes is
//! first-touched by a worker in the partition's own NUMA domain
//! (ARCHITECTURE.md §Parallelism for why the layer needs no synchronization of
//! its own). That `install` is also why the MPI transport requires
//! `MPI_THREAD_SERIALIZED` rather than `FUNNELED`: a distributed rank's
//! collectives are issued from a pool worker, not from the process's main
//! thread.
//!
//! There is no synchronization between partitions other than the transport:
//! every partition issues the identical sequence of transport calls per layer
//! (see [`transport`](super::transport)'s collective-order invariant), so the
//! group stays in step without a barrier.

use std::sync::Arc;

use super::topology::{
    bind_current_thread_memory, build_pool, pin_current_thread, PartitionConfig, PartitionSlot,
    TopologyError,
};
use super::transport::InProcessTransport;

/// `log` target for placement diagnostics, matching
/// [`topology`](super::topology).
const LOG_TARGET: &str = "paulistrings::partitioned";

/// The resolved placement and its pools, shared by every
/// [`PartitionedSum`](super::PartitionedSum) that runs on it.
///
/// Built once — pools are not cheap — and held behind an [`Arc`], so a driver
/// stepping an observable through many circuits, or several sums propagated in
/// turn, pay for the pinned pools once. Cloning the `Arc` is the intended way
/// to share it.
///
/// # Examples
///
/// ```
/// use paulistrings::engine::partitioned::{PartitionConfig, PartitionRuntime, Placement};
///
/// let config = PartitionConfig {
///     placement: Placement::Unpinned { partitions: 2, threads_per_partition: Some(1) },
///     bind_memory: false,
///     partition_row_seed: None,
/// };
/// let runtime = PartitionRuntime::new(&config).expect("topology resolves");
/// assert_eq!(runtime.num_partitions(), 2);
/// assert_eq!(runtime.slots().len(), 2);
/// ```
pub struct PartitionRuntime {
    /// One slot per partition, in rank order.
    slots: Vec<PartitionSlot>,
    /// One pinned pool per partition, in rank order.
    pools: Vec<rayon::ThreadPool>,
    /// Whether driving threads and pool workers bind their allocations to the
    /// slot's NUMA node.
    bind_memory: bool,
}

impl PartitionRuntime {
    /// Resolves `config` against the machine and builds one pinned pool per
    /// partition.
    ///
    /// # Errors
    ///
    /// Whatever [`PartitionConfig::resolve`] reports (a partition count that
    /// is not a power of two, an empty or out-of-mask explicit CPU set), or
    /// [`TopologyError::Io`] if a pool cannot be built.
    pub fn new(config: &PartitionConfig) -> Result<Arc<Self>, TopologyError> {
        Self::with_threads_per_partition(config, None)
    }

    /// [`new`](Self::new) with the pool width chosen by the caller instead of
    /// by the placement.
    ///
    /// `Some(t)` gives **every** partition a `t`-worker pool (at least one),
    /// leaving the CPU set each pool is pinned to exactly as
    /// [`PartitionConfig::resolve`] derived it; `None` keeps the resolved
    /// widths ([`Placement::Auto`](super::Placement::Auto) and
    /// [`Explicit`](super::Placement::Explicit) size a pool by the number of
    /// CPUs in its set).
    ///
    /// This is the knob a measurement harness needs to hold the *total* thread
    /// count fixed across partition counts — `T` threads unpartitioned against
    /// `P` pools of `T / P` on the same CPUs — without shrinking the CPU masks
    /// and changing what is being compared. In production, prefer
    /// [`new`](Self::new): a pool narrower than its CPU set leaves cores idle,
    /// and a wider one oversubscribes them.
    ///
    /// # Errors
    ///
    /// Whatever [`PartitionConfig::resolve`] reports (a partition count that
    /// is not a power of two, an empty or out-of-mask explicit CPU set), or
    /// [`TopologyError::Io`] if a pool cannot be built.
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
        Ok(Arc::new(Self {
            slots,
            pools,
            bind_memory: config.bind_memory,
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

    /// `log2` of the partition count: the number of GF(2) rows a
    /// [`PartitionRows`](crate::PartitionRows) needs to name a partition.
    pub(crate) fn partition_bits(&self) -> u8 {
        debug_assert!(self.slots.len().is_power_of_two());
        self.slots.len().trailing_zeros() as u8
    }

    /// A one-line summary of the placement, for the entry log line.
    pub(crate) fn placement_summary(&self) -> String {
        let mut s = String::new();
        for (rank, slot) in self.slots.iter().enumerate() {
            if rank > 0 {
                s.push_str(", ");
            }
            match &slot.cpus {
                Some(cpus) => s.push_str(&format!("{rank}:{cpus}x{}", slot.threads)),
                None => s.push_str(&format!("{rank}:unpinned x{}", slot.threads)),
            }
        }
        s
    }

    /// Runs `f` on partition 0's pool and returns its result.
    ///
    /// The distributed shape of [`map_partitions`](Self::map_partitions): a
    /// process that *is* one partition has nothing to fan out to, but its work
    /// still belongs on the pinned pool, so that every allocation is
    /// first-touched by a worker inside the process's own domain. As with
    /// `map_partitions`, the calling thread is not re-affinitized — the work
    /// runs on pool 0's workers, which `build_pool` pinned.
    ///
    /// `rayon::ThreadPool::install` blocks the caller and runs `f` **on a pool
    /// worker**, so `f` (and any MPI call inside it) is not on the process's
    /// main thread. That is why the MPI transport documents
    /// `MPI_THREAD_SERIALIZED` rather than `FUNNELED`.
    pub(crate) fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        self.pools[0].install(f)
    }

    /// Runs `f(rank, item, transport)` once per partition, concurrently, and
    /// returns the results in rank order.
    ///
    /// `items` carries one value per partition — the partition's local sum and
    /// scratch — moved in and handed back, so nothing is shared between
    /// partitions but the transport. Partition 0 runs on the calling thread;
    /// partitions `1..P` on scoped threads that pin themselves to their slot
    /// first. Every partition's body runs inside its own pool
    /// (`ThreadPool::install`), so a `rayon` call inside `f` lands on the
    /// partition's own workers.
    ///
    /// # Panics
    ///
    /// A panic in any partition is propagated to the caller with its original
    /// payload, after the scope has joined the others. It cannot hang the
    /// group: the transport group is *moved* into the partitions, so a dying
    /// partition drops its endpoints and its partners' next collective reports
    /// a dead partner rather than blocking (see
    /// [`transport`](super::transport)).
    pub(crate) fn map_partitions<I, O, F>(&self, items: Vec<I>, f: F) -> Vec<O>
    where
        I: Send,
        O: Send,
        F: Fn(usize, I, &InProcessTransport) -> O + Send + Sync,
    {
        let size = self.num_partitions();
        assert_eq!(items.len(), size, "map_partitions: one item per partition");
        // A fresh group per call, rather than one held in the runtime: each
        // partition *owns* its endpoint for the duration, which is what turns a
        // partner's panic into a panic (dropped senders) instead of a hang. It
        // also means an aborted call cannot leave the group's collective
        // counter out of step for the next one.
        let transports = InProcessTransport::group(size as u32);
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

            // Partition 0 on the calling thread — deliberately *not* pinned:
            // the caller's thread is not ours to re-affinitize, and the work
            // itself runs on pool 0's workers, which `build_pool` pinned.
            let mut out = Vec::with_capacity(size);
            out.push(self.pools[0].install(|| f(0, item0, &transport0)));
            for handle in handles {
                match handle.join() {
                    Ok(o) => out.push(o),
                    // Re-raise the partition's own payload, so the caller sees
                    // the original message rather than a join error.
                    Err(payload) => std::panic::resume_unwind(payload),
                }
            }
            out
        })
    }
}

/// Pins a partition's driving thread to its slot, warning (not failing) when
/// the platform declines — the same degradation policy `build_pool` applies to
/// pool workers.
fn place_current_thread(slot: &PartitionSlot, bind_memory: bool) {
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
mod tests {
    use super::*;
    use crate::engine::partitioned::topology::{allowed_cpus, Placement};
    use crate::engine::partitioned::transport::Collectives;

    fn config(partitions: usize) -> PartitionConfig {
        PartitionConfig {
            placement: Placement::Unpinned {
                partitions,
                threads_per_partition: Some(2),
            },
            bind_memory: false,
            partition_row_seed: None,
        }
    }

    #[test]
    fn partition_bits_is_log2_of_the_count() {
        for (p, bits) in [(1usize, 0u8), (2, 1), (4, 2), (8, 3)] {
            let runtime = PartitionRuntime::new(&config(p)).expect("resolve");
            assert_eq!(runtime.num_partitions(), p);
            assert_eq!(runtime.partition_bits(), bits);
        }
    }

    /// The pool-width override replaces the resolved worker count on every
    /// slot and leaves the CPU sets alone — the probe's "total threads split
    /// over P partitions" knob.
    #[test]
    fn with_threads_per_partition_overrides_the_resolved_width() {
        let cpus = allowed_cpus();
        let placement = Placement::Explicit(vec![cpus.clone(), cpus.clone()]);
        let config = PartitionConfig {
            placement,
            bind_memory: false,
            partition_row_seed: None,
        };

        let resolved = PartitionRuntime::new(&config).expect("resolve");
        for slot in resolved.slots() {
            assert_eq!(slot.threads, cpus.len());
        }

        let narrowed = PartitionRuntime::with_threads_per_partition(&config, Some(1))
            .expect("resolve with an explicit width");
        for slot in narrowed.slots() {
            assert_eq!(slot.threads, 1);
            assert_eq!(slot.cpus.as_ref(), Some(&cpus), "the CPU set is untouched");
        }
    }

    #[test]
    fn map_partitions_returns_results_in_rank_order() {
        let runtime = PartitionRuntime::new(&config(4)).expect("resolve");
        let got = runtime.map_partitions(vec![10usize, 20, 30, 40], |rank, item, transport| {
            assert_eq!(transport.rank() as usize, rank);
            assert_eq!(transport.size(), 4);
            // Inside the partition's own pool.
            assert!(rayon::current_thread_index().is_some());
            (rank, item)
        });
        assert_eq!(got, vec![(0, 10), (1, 20), (2, 30), (3, 40)]);
    }

    #[test]
    fn map_partitions_runs_collectives_between_partitions() {
        let runtime = PartitionRuntime::new(&config(4)).expect("resolve");
        let got = runtime.map_partitions(vec![3u8, 7, 1, 5], |_, item, transport| {
            transport.allreduce_max_u8(item)
        });
        assert_eq!(got, vec![7, 7, 7, 7]);
    }

    /// A partition that panics while its partners are inside a collective must
    /// surface as a panic, not a deadlock. (The test itself would hang on
    /// failure; that is the assertion.)
    ///
    /// *Which* panic surfaces is a race, so the assertion is bare
    /// `should_panic` with no expected text. The first `Err` the joins reach in
    /// rank order is usually a partner's "partition 2 terminated" from the
    /// transport, but a partner that noticed rank 2's death first can itself
    /// die before rank 0 reads rank 2's channel, and rank 0 then reports *that*
    /// partner instead — observed as roughly one `cargo test --workspace` run
    /// in five under full parallelism. Every one of those outcomes is the
    /// behaviour under test: the group terminates instead of blocking forever.
    #[test]
    #[should_panic]
    fn a_panicking_partition_does_not_hang_the_group() {
        let runtime = PartitionRuntime::new(&config(4)).expect("resolve");
        runtime.map_partitions(vec![0usize; 4], |rank, _, transport| {
            if rank == 2 {
                panic!("rank 2 fell over");
            }
            // The partners keep collecting, so they are blocked on the dead
            // partition until its endpoint drops.
            for _ in 0..4 {
                transport.allreduce_max_u8(rank as u8);
            }
        });
    }

    #[test]
    fn placement_summary_names_every_slot() {
        let runtime = PartitionRuntime::new(&config(2)).expect("resolve");
        let summary = runtime.placement_summary();
        assert!(summary.contains("0:"), "{summary}");
        assert!(summary.contains("1:"), "{summary}");
    }
}
