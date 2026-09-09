//! CPU sets, NUMA node discovery, thread/memory pinning, and the pinned Rayon
//! pool one partition runs on. See ARCHITECTURE.md §Partitioning.
//!
//! A partition owns one Rayon pool whose workers are pinned to one NUMA
//! domain's CPUs and whose allocations are bound to that domain's memory. Work
//! stealing therefore stays *inside* a socket: the coset loop keeps its
//! straggler tolerance (static coset→worker placement was measured 1.25–1.9×
//! slower — `research/notes/2026-08-30-static-coset-placement.md`) while
//! first-touch of a partition's columns lands on the socket that reads them.
//!
//! Discovery is sysfs + `libc` (`sched_{get,set}affinity`, `sched_getcpu`,
//! `set_mempolicy`); there is no hwloc or libnuma dependency. Every entry
//! point compiles on non-Linux targets, where it reports one node covering
//! `available_parallelism` CPUs and pins nothing.

use std::fmt;
use std::io;

/// `log` target for topology events (pinning failures, overlapping explicit
/// sets). Separate from the engine's progress target so a consumer can filter
/// placement diagnostics on their own.
const LOG_TARGET: &str = "paulistrings::partitioned";

/// A set of logical CPU indices, kept sorted and deduplicated.
///
/// The tuple field is public so callers can build a set literally; the
/// constructors ([`CpuSet::parse`], [`CpuSet::intersect`]) and every consumer
/// in this module normalize, so an unsorted literal is tolerated rather than
/// relied upon.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuSet(
    /// The CPU indices, ascending and unique.
    pub Vec<usize>,
);

impl CpuSet {
    /// Parses a Linux cpulist, the `"0-7,16-23"` form written by sysfs.
    ///
    /// Surrounding and per-token whitespace is trimmed and empty tokens (a
    /// trailing comma, a trailing newline) are skipped.
    ///
    /// # Errors
    ///
    /// [`TopologyError::EmptySet`] if the list names no CPU at all, and
    /// [`TopologyError::InvalidCpuList`] if a token is not a decimal index or
    /// an ascending `lo-hi` range.
    pub fn parse(s: &str) -> Result<Self, TopologyError> {
        let invalid = || TopologyError::InvalidCpuList(s.trim().to_string());
        let index = |token: &str| token.trim().parse::<usize>().map_err(|_| invalid());

        let mut cpus = Vec::new();
        for token in s.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            match token.split_once('-') {
                None => cpus.push(index(token)?),
                Some((lo, hi)) => {
                    let (lo, hi) = (index(lo)?, index(hi)?);
                    if hi < lo {
                        return Err(invalid());
                    }
                    cpus.extend(lo..=hi);
                }
            }
        }
        if cpus.is_empty() {
            return Err(TopologyError::EmptySet);
        }
        cpus.sort_unstable();
        cpus.dedup();
        Ok(Self(cpus))
    }

    /// The number of distinct CPUs in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.normalized().len()
    }

    /// Whether the set names no CPU.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The CPUs present in both sets.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        let other = other.normalized();
        Self(
            self.normalized()
                .into_iter()
                .filter(|cpu| other.binary_search(cpu).is_ok())
                .collect(),
        )
    }

    /// Whether every CPU of `self` is also in `other`.
    fn is_subset_of(&self, other: &Self) -> bool {
        let other = other.normalized();
        self.0.iter().all(|cpu| other.binary_search(cpu).is_ok())
    }

    /// A sorted, deduplicated copy of the indices.
    fn normalized(&self) -> Vec<usize> {
        let mut cpus = self.0.clone();
        cpus.sort_unstable();
        cpus.dedup();
        cpus
    }

    /// The union of two sets.
    fn union(&self, other: &Self) -> Self {
        let mut cpus = self.normalized();
        cpus.extend(other.normalized());
        cpus.sort_unstable();
        cpus.dedup();
        Self(cpus)
    }
}

impl fmt::Display for CpuSet {
    /// Writes the sysfs cpulist form, compressing runs into `lo-hi` ranges.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cpus = self.normalized();
        let mut first = true;
        let mut i = 0;
        while i < cpus.len() {
            let mut j = i;
            while j + 1 < cpus.len() && cpus[j + 1] == cpus[j] + 1 {
                j += 1;
            }
            if !first {
                write!(f, ",")?;
            }
            first = false;
            if i == j {
                write!(f, "{}", cpus[i])?;
            } else {
                write!(f, "{}-{}", cpus[i], cpus[j])?;
            }
            i = j + 1;
        }
        Ok(())
    }
}

/// The CPUs this process may run on: the current affinity mask.
///
/// Falls back to `0..available_parallelism` when the mask cannot be read and
/// on non-Linux targets.
#[must_use]
pub(crate) fn allowed_cpus() -> CpuSet {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `set` is a valid, correctly sized `cpu_set_t`; pid 0 is the
        // calling thread.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) == 0 {
                let cpus: Vec<usize> = (0..libc::CPU_SETSIZE as usize)
                    .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
                    .collect();
                if !cpus.is_empty() {
                    return CpuSet(cpus);
                }
            }
        }
    }
    CpuSet((0..available_parallelism()).collect())
}

/// The NUMA nodes visible in the current affinity mask, ascending by node id.
///
/// Each node's CPU set is intersected with the process's affinity mask, and
/// nodes left empty by that intersection are dropped. When sysfs exposes no
/// usable node — a kernel without NUMA, a sandbox without `/sys`, a non-Linux
/// target — the whole affinity mask is reported as node 0.
#[must_use]
pub fn numa_nodes() -> Vec<(usize, CpuSet)> {
    let allowed = allowed_cpus();
    let mut nodes = sysfs_numa_nodes(&allowed);
    nodes.sort_unstable_by_key(|(id, _)| *id);
    if nodes.is_empty() {
        vec![(0, allowed)]
    } else {
        nodes
    }
}

#[cfg(target_os = "linux")]
fn sysfs_numa_nodes(allowed: &CpuSet) -> Vec<(usize, CpuSet)> {
    const ROOT: &str = "/sys/devices/system/node";
    let Ok(entries) = std::fs::read_dir(ROOT) else {
        return Vec::new();
    };
    let mut nodes = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(id) = name.to_str().and_then(|n| n.strip_prefix("node")) else {
            continue;
        };
        let Ok(id) = id.parse::<usize>() else {
            continue;
        };
        let Ok(list) = std::fs::read_to_string(entry.path().join("cpulist")) else {
            continue;
        };
        let Ok(cpus) = CpuSet::parse(&list) else {
            continue;
        };
        let cpus = cpus.intersect(allowed);
        if !cpus.is_empty() {
            nodes.push((id, cpus));
        }
    }
    nodes
}

#[cfg(not(target_os = "linux"))]
fn sysfs_numa_nodes(_allowed: &CpuSet) -> Vec<(usize, CpuSet)> {
    Vec::new()
}

/// Pins the calling thread to `set`.
///
/// A no-op returning `Ok(())` on non-Linux targets.
///
/// # Errors
///
/// [`io::ErrorKind::InvalidInput`] if the set is empty or names a CPU at or
/// beyond the kernel's `CPU_SETSIZE`; otherwise the `sched_setaffinity` error.
pub(crate) fn pin_current_thread(set: &CpuSet) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        if set.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot pin a thread to an empty CPU set",
            ));
        }
        // SAFETY: `mask` is a valid, zeroed `cpu_set_t`; every index is
        // checked against `CPU_SETSIZE` before `CPU_SET` touches it.
        unsafe {
            let mut mask: libc::cpu_set_t = std::mem::zeroed();
            for &cpu in &set.0 {
                if cpu >= libc::CPU_SETSIZE as usize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("CPU {cpu} is beyond CPU_SETSIZE"),
                    ));
                }
                libc::CPU_SET(cpu, &mut mask);
            }
            if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mask) != 0 {
                return Err(io::Error::last_os_error());
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = set;
    Ok(())
}

/// Binds the calling thread's allocations to one NUMA node, or restores the
/// default policy.
///
/// `Some(node)` installs `MPOL_BIND` on that node alone, so pages this thread
/// first-touches come from its own domain; `None` restores `MPOL_DEFAULT`. A
/// no-op returning `Ok(())` on non-Linux targets.
///
/// # Errors
///
/// The `set_mempolicy` error — notably [`io::ErrorKind::Unsupported`] on a
/// kernel built without NUMA support.
pub(crate) fn bind_current_thread_memory(node: Option<usize>) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        // `libc` exposes no `set_mempolicy` wrapper, so go through `syscall`.
        const BITS: usize = 8 * std::mem::size_of::<libc::c_ulong>();
        let rc = match node {
            Some(node) => {
                let words = node / BITS + 1;
                let mut mask = vec![0 as libc::c_ulong; words];
                mask[node / BITS] = 1 << (node % BITS);
                // SAFETY: `mask` outlives the call and holds `words` words,
                // which is exactly the `maxnode` bits the kernel reads.
                unsafe {
                    libc::syscall(
                        libc::SYS_set_mempolicy,
                        libc::MPOL_BIND,
                        mask.as_ptr(),
                        (words * BITS) as libc::c_ulong,
                    )
                }
            }
            // SAFETY: `MPOL_DEFAULT` takes a null mask of zero length.
            None => unsafe {
                libc::syscall(
                    libc::SYS_set_mempolicy,
                    libc::MPOL_DEFAULT,
                    std::ptr::null::<libc::c_ulong>(),
                    0 as libc::c_ulong,
                )
            },
        };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = node;
    Ok(())
}

/// The CPU the calling thread is running on right now, if the platform can
/// say. `None` on non-Linux targets.
///
/// Only [`build_pool`]'s own test asks — the engine pins and then trusts the
/// kernel — so it is compiled for tests alone.
#[cfg(test)]
fn current_cpu() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `sched_getcpu` takes no arguments and cannot fail beyond
        // returning a negative value.
        let cpu = unsafe { libc::sched_getcpu() };
        usize::try_from(cpu).ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// One resolved partition: where its pool's workers run and how many there
/// are.
#[derive(Clone, Debug)]
pub struct PartitionSlot {
    /// CPUs to pin the pool's workers to, or `None` to leave them unpinned.
    pub cpus: Option<CpuSet>,
    /// NUMA node to bind the workers' allocations to, when the slot sits in
    /// exactly one node.
    pub node: Option<usize>,
    /// Worker count for the slot's Rayon pool.
    pub threads: usize,
}

/// Builds the Rayon pool for one partition, pinning each worker to the slot's
/// CPUs and (when `bind_memory`) binding its allocations to the slot's node.
///
/// Pinning happens on the worker itself, before it enters Rayon's main loop,
/// so the pool's own stacks and per-worker allocations are first-touched
/// locally. A worker whose pinning call fails logs a warning and runs
/// unpinned: a placement failure degrades locality, it does not stop the run.
///
/// # Errors
///
/// [`TopologyError::Io`] wrapping the spawn or pool-build failure.
pub(crate) fn build_pool(
    slot: &PartitionSlot,
    bind_memory: bool,
    name: &str,
) -> Result<rayon::ThreadPool, TopologyError> {
    let cpus = slot.cpus.clone();
    let node = slot.node;
    let pool_name = name.to_string();
    rayon::ThreadPoolBuilder::new()
        .num_threads(slot.threads)
        .thread_name(move |index| format!("{pool_name}-{index}"))
        .spawn_handler(move |thread| {
            let cpus = cpus.clone();
            let mut builder = std::thread::Builder::new();
            if let Some(name) = thread.name() {
                builder = builder.name(name.to_string());
            }
            if let Some(size) = thread.stack_size() {
                builder = builder.stack_size(size);
            }
            builder.spawn(move || {
                if let Some(cpus) = &cpus {
                    if let Err(err) = pin_current_thread(cpus) {
                        log::warn!(target: LOG_TARGET, "failed to pin worker to {cpus}: {err}");
                    }
                }
                if bind_memory {
                    if let Err(err) = bind_current_thread_memory(node) {
                        log::warn!(target: LOG_TARGET, "failed to bind worker memory to {node:?}: {err}");
                    }
                }
                thread.run();
            })?;
            Ok(())
        })
        .build()
        .map_err(|err| TopologyError::Io(io::Error::other(err.to_string())))
}

/// How partitions map onto the machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    /// One partition per NUMA node in the affinity mask.
    ///
    /// The partition count is the node count rounded **down** to a power of
    /// two (at least one); when that is fewer than the node count, adjacent
    /// nodes are merged into a partition rather than dropped, so every allowed
    /// CPU stays in play. Each partition takes as many threads as it has CPUs.
    Auto {
        /// Upper bound on the partition count, itself rounded down to a power
        /// of two. `None` for no bound.
        max_partitions: Option<usize>,
    },
    /// One partition per listed CPU set, exactly as given.
    ///
    /// Sets may overlap — the resolve logs a warning and proceeds — but each
    /// must be non-empty and a subset of the process's affinity mask.
    Explicit(
        /// The CPU sets, one per partition; the count must be a power of two.
        Vec<CpuSet>,
    ),
    /// Partitions with no pinning at all: the shape of a partitioned run
    /// without its placement, for tests and CI.
    Unpinned {
        /// Number of partitions; must be a power of two.
        partitions: usize,
        /// Threads per partition, or `None` to split `available_parallelism`
        /// evenly (at least one thread each).
        threads_per_partition: Option<usize>,
    },
}

/// Placement plus the knobs the partitioned engine reads alongside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionConfig {
    /// How partitions map onto the machine.
    pub placement: Placement,
    /// Whether each pool's workers bind their allocations to the partition's
    /// NUMA node (`MPOL_BIND`, through `set_mempolicy`).
    ///
    /// The Python surface spells this `pin_memory=`, and the probe's JSON
    /// sidecar follows the Python name; everything on the Rust side is
    /// `bind_memory`, because it is a memory policy and not a thread affinity.
    pub bind_memory: bool,
    /// Seed selecting which GF(2) hash rows designate the partition, or
    /// `None` to take the engine's default choice.
    pub partition_row_seed: Option<u64>,
}

impl Default for PartitionConfig {
    /// One partition per NUMA node, memory bound to it, default partition
    /// rows.
    fn default() -> Self {
        Self {
            placement: Placement::Auto {
                max_partitions: None,
            },
            bind_memory: true,
            partition_row_seed: None,
        }
    }
}

impl PartitionConfig {
    /// Resolves the placement against the machine into one slot per partition.
    ///
    /// The slot count is always a power of two, which is what makes a
    /// partition index a fixed set of GF(2) hash rows.
    ///
    /// # Errors
    ///
    /// [`TopologyError::NotPowerOfTwo`] if an explicit or unpinned partition
    /// count is not a power of two, [`TopologyError::EmptySet`] if an explicit
    /// set is empty, and [`TopologyError::CpuNotAllowed`] if an explicit set
    /// names a CPU outside the process's affinity mask.
    pub fn resolve(&self) -> Result<Vec<PartitionSlot>, TopologyError> {
        match &self.placement {
            Placement::Auto { max_partitions } => Ok(resolve_auto(*max_partitions)),
            Placement::Explicit(sets) => resolve_explicit(sets),
            Placement::Unpinned {
                partitions,
                threads_per_partition,
            } => resolve_unpinned(*partitions, *threads_per_partition),
        }
    }
}

fn resolve_auto(max_partitions: Option<usize>) -> Vec<PartitionSlot> {
    let nodes = numa_nodes();
    let mut count = prev_power_of_two(nodes.len());
    if let Some(max) = max_partitions {
        count = prev_power_of_two(count.min(max));
    }

    // Merge adjacent nodes when the count was rounded down, so no allowed CPU
    // is left out of the run: the first `nodes.len() % count` partitions take
    // one node more than the rest.
    let (base, remainder) = (nodes.len() / count, nodes.len() % count);
    let mut slots = Vec::with_capacity(count);
    let mut rest = nodes.as_slice();
    for partition in 0..count {
        let take = base + usize::from(partition < remainder);
        let (group, tail) = rest.split_at(take);
        rest = tail;
        let cpus = group
            .iter()
            .fold(CpuSet(Vec::new()), |acc, (_, set)| acc.union(set));
        slots.push(PartitionSlot {
            node: (group.len() == 1).then(|| group[0].0),
            threads: cpus.len(),
            cpus: Some(cpus),
        });
    }
    slots
}

fn resolve_explicit(sets: &[CpuSet]) -> Result<Vec<PartitionSlot>, TopologyError> {
    if !sets.len().is_power_of_two() {
        return Err(TopologyError::NotPowerOfTwo(sets.len()));
    }
    let allowed = allowed_cpus();
    let nodes = numa_nodes();

    let mut slots = Vec::with_capacity(sets.len());
    for set in sets {
        if set.is_empty() {
            return Err(TopologyError::EmptySet);
        }
        let cpus = CpuSet(set.normalized());
        if let Some(&cpu) = cpus.0.iter().find(|cpu| !allowed.0.contains(cpu)) {
            return Err(TopologyError::CpuNotAllowed(cpu));
        }
        let mut containing = nodes
            .iter()
            .filter(|(_, node)| cpus.is_subset_of(node))
            .map(|(id, _)| *id);
        let node = match (containing.next(), containing.next()) {
            (Some(id), None) => Some(id),
            _ => None,
        };
        slots.push(PartitionSlot {
            node,
            threads: cpus.len(),
            cpus: Some(cpus),
        });
    }

    for (i, slot) in slots.iter().enumerate() {
        for other in slots.iter().skip(i + 1) {
            let (a, b) = (slot.cpus.as_ref(), other.cpus.as_ref());
            if let (Some(a), Some(b)) = (a, b) {
                let shared = a.intersect(b);
                if !shared.is_empty() {
                    log::warn!(
                        target: LOG_TARGET,
                        "explicit partitions {a} and {b} overlap on {shared}; \
                         their pools will contend for those CPUs",
                    );
                }
            }
        }
    }
    Ok(slots)
}

fn resolve_unpinned(
    partitions: usize,
    threads_per_partition: Option<usize>,
) -> Result<Vec<PartitionSlot>, TopologyError> {
    if !partitions.is_power_of_two() {
        return Err(TopologyError::NotPowerOfTwo(partitions));
    }
    let threads = threads_per_partition
        .unwrap_or(available_parallelism() / partitions)
        .max(1);
    Ok(vec![
        PartitionSlot {
            cpus: None,
            node: None,
            threads,
        };
        partitions
    ])
}

/// The largest power of two `<= n`, and 1 for `n == 0`.
fn prev_power_of_two(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        1 << (usize::BITS - 1 - n.leading_zeros())
    }
}

fn available_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// What can go wrong discovering topology or resolving a [`PartitionConfig`].
#[derive(Debug)]
pub enum TopologyError {
    /// A cpulist string was not the `"0-7,16-23"` form.
    InvalidCpuList(
        /// The offending list, trimmed.
        String,
    ),
    /// A partition count was not a power of two.
    NotPowerOfTwo(
        /// The rejected count.
        usize,
    ),
    /// A CPU set was empty where a partition needs at least one CPU.
    EmptySet,
    /// A syscall or sysfs read failed, or a pool could not be built.
    Io(
        /// The underlying error.
        io::Error,
    ),
    /// An explicit set named a CPU outside the process's affinity mask.
    CpuNotAllowed(
        /// The CPU index that is outside the process's affinity mask.
        usize,
    ),
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCpuList(list) => {
                write!(f, "{list:?} is not a cpulist of the form \"0-7,16-23\"")
            }
            Self::NotPowerOfTwo(count) => write!(
                f,
                "partition count {count} is not a power of two; a partition index is a fixed \
                 set of GF(2) hash rows",
            ),
            Self::EmptySet => write!(f, "a partition needs a non-empty CPU set"),
            Self::Io(err) => write!(f, "topology I/O failed: {err}"),
            Self::CpuNotAllowed(cpu) => {
                write!(f, "CPU {cpu} is outside this process's affinity mask",)
            }
        }
    }
}

impl std::error::Error for TopologyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for TopologyError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_ranges_and_singletons() {
        assert_eq!(CpuSet::parse("0-7,16-23").unwrap().0, {
            let mut v: Vec<usize> = (0..8).collect();
            v.extend(16..24);
            v
        });
        assert_eq!(CpuSet::parse("3").unwrap().0, vec![3]);
        assert_eq!(CpuSet::parse("5-5").unwrap().0, vec![5]);
        assert_eq!(CpuSet::parse(" 0-1,\n").unwrap().0, vec![0, 1]);
    }

    #[test]
    fn parse_rejects_malformed_lists() {
        assert!(matches!(CpuSet::parse(""), Err(TopologyError::EmptySet)));
        assert!(matches!(
            CpuSet::parse("a"),
            Err(TopologyError::InvalidCpuList(_))
        ));
        assert!(matches!(
            CpuSet::parse("7-3"),
            Err(TopologyError::InvalidCpuList(_))
        ));
    }

    #[test]
    fn display_compresses_ranges_and_round_trips() {
        let set = CpuSet::parse("0-7,16-23").unwrap();
        assert_eq!(set.to_string(), "0-7,16-23");
        assert_eq!(CpuSet::parse(&set.to_string()).unwrap(), set);
        assert_eq!(CpuSet::parse("3,1,2,2").unwrap().to_string(), "1-3");
        assert_eq!(CpuSet::parse("1,3,5-6").unwrap().to_string(), "1,3,5-6");
    }

    #[test]
    fn intersect_keeps_common_cpus() {
        let a = CpuSet::parse("0-7,16-23").unwrap();
        let b = CpuSet::parse("4-18").unwrap();
        assert_eq!(a.intersect(&b).to_string(), "4-7,16-18");
        assert!(a.intersect(&CpuSet::parse("100").unwrap()).is_empty());
        assert_eq!(a.len(), 16);
    }

    #[test]
    fn numa_nodes_are_disjoint_subsets_of_the_affinity_mask() {
        let allowed = allowed_cpus();
        let nodes = numa_nodes();
        assert!(!nodes.is_empty());
        for (i, (_, set)) in nodes.iter().enumerate() {
            assert!(!set.is_empty());
            assert_eq!(set.intersect(&allowed), *set, "node cpus outside affinity");
            for (_, other) in nodes.iter().skip(i + 1) {
                assert!(set.intersect(other).is_empty(), "nodes overlap");
            }
        }
        let ids: Vec<usize> = nodes.iter().map(|(id, _)| *id).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn unpinned_placement_requires_a_power_of_two() {
        let cfg = PartitionConfig {
            placement: Placement::Unpinned {
                partitions: 3,
                threads_per_partition: None,
            },
            ..PartitionConfig::default()
        };
        assert!(matches!(
            cfg.resolve(),
            Err(TopologyError::NotPowerOfTwo(3))
        ));
    }

    #[test]
    fn unpinned_placement_resolves_to_unpinned_slots() {
        let cfg = PartitionConfig {
            placement: Placement::Unpinned {
                partitions: 4,
                threads_per_partition: Some(2),
            },
            ..PartitionConfig::default()
        };
        let slots = cfg.resolve().unwrap();
        assert_eq!(slots.len(), 4);
        for slot in &slots {
            assert_eq!(slot.threads, 2);
            assert!(slot.cpus.is_none());
            assert!(slot.node.is_none());
        }
    }

    #[test]
    fn explicit_placement_validates_its_sets() {
        let one = CpuSet(vec![allowed_cpus().0[0]]);
        let cfg = |sets: Vec<CpuSet>| PartitionConfig {
            placement: Placement::Explicit(sets),
            ..PartitionConfig::default()
        };
        assert!(matches!(
            cfg(vec![one.clone(), one.clone(), one.clone()]).resolve(),
            Err(TopologyError::NotPowerOfTwo(3))
        ));
        assert!(matches!(
            cfg(vec![one.clone(), CpuSet(vec![])]).resolve(),
            Err(TopologyError::EmptySet)
        ));
        assert!(matches!(
            cfg(vec![one, CpuSet(vec![1_000_000])]).resolve(),
            Err(TopologyError::CpuNotAllowed(1_000_000))
        ));
    }

    #[test]
    fn explicit_placement_resolves_disjoint_singletons() {
        let allowed = allowed_cpus();
        if allowed.len() < 2 {
            eprintln!("skipping: fewer than two allowed CPUs ({allowed})");
            return;
        }
        let cfg = PartitionConfig {
            placement: Placement::Explicit(vec![
                CpuSet(vec![allowed.0[0]]),
                CpuSet(vec![allowed.0[1]]),
            ]),
            ..PartitionConfig::default()
        };
        let slots = cfg.resolve().unwrap();
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].cpus.as_ref().unwrap().0, vec![allowed.0[0]]);
        assert_eq!(slots[1].cpus.as_ref().unwrap().0, vec![allowed.0[1]]);
        for slot in &slots {
            assert_eq!(slot.threads, 1);
        }
    }

    #[test]
    fn auto_placement_is_a_power_of_two_of_disjoint_sets() {
        let slots = PartitionConfig::default().resolve().unwrap();
        assert!(!slots.is_empty());
        assert!(slots.len().is_power_of_two());
        for (i, slot) in slots.iter().enumerate() {
            let cpus = slot.cpus.as_ref().expect("auto slots are pinned");
            assert!(!cpus.is_empty());
            assert_eq!(slot.threads, cpus.len());
            for other in slots.iter().skip(i + 1) {
                let other = other.cpus.as_ref().unwrap();
                assert!(cpus.intersect(other).is_empty(), "partitions overlap");
            }
        }
    }

    #[test]
    fn auto_placement_honours_max_partitions() {
        let cfg = PartitionConfig {
            placement: Placement::Auto {
                max_partitions: Some(1),
            },
            ..PartitionConfig::default()
        };
        assert_eq!(cfg.resolve().unwrap().len(), 1);
    }

    #[test]
    fn a_pinned_pool_runs_every_worker_on_the_pinned_cpu() {
        let allowed = allowed_cpus();
        if allowed.len() < 2 {
            eprintln!("skipping: fewer than two allowed CPUs ({allowed})");
            return;
        }
        let cpu = allowed.0[0];
        let slot = PartitionSlot {
            cpus: Some(CpuSet(vec![cpu])),
            node: None,
            threads: 2,
        };
        let pool = build_pool(&slot, false, "test-pinned").unwrap();
        let seen: Vec<Option<usize>> = pool.broadcast(|_| current_cpu());
        assert_eq!(seen.len(), 2);
        if cfg!(target_os = "linux") {
            for got in seen {
                assert_eq!(got, Some(cpu));
            }
        }
    }

    #[test]
    fn an_unpinned_pool_builds_and_runs() {
        let slot = PartitionSlot {
            cpus: None,
            node: None,
            threads: 2,
        };
        let pool = build_pool(&slot, false, "test-unpinned").unwrap();
        assert_eq!(pool.install(|| (0..100).sum::<usize>()), 4950);
    }

    #[test]
    fn memory_binding_round_trips_on_a_scratch_thread() {
        // On a scratch thread so the test runner's own memory policy, which
        // every other test in this process shares, is left alone.
        let node = numa_nodes()[0].0;
        std::thread::scope(|scope| {
            scope.spawn(|| match bind_current_thread_memory(Some(node)) {
                Ok(()) => bind_current_thread_memory(None).unwrap(),
                // A kernel built without NUMA has no `set_mempolicy`.
                Err(err) if err.kind() == io::ErrorKind::Unsupported => {
                    eprintln!("skipping: set_mempolicy unavailable ({err})");
                }
                Err(err) => panic!("binding to node {node} failed: {err}"),
            });
        });
    }
}
