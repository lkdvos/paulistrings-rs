//! CPU sets, NUMA node discovery, thread and memory pinning, and the pinned Rayon pool one partition runs on (ARCHITECTURE.md §Partitioning).
//! Discovery is sysfs plus `libc`; on non-Linux targets every entry point reports one node and pins nothing.

use std::fmt;
use std::io;

/// Separate from the progress target, so placement diagnostics filter on their own.
const LOG_TARGET: &str = "paulistrings::partitioned";

/// A set of logical CPU indices; an unsorted literal is tolerated, since every consumer normalizes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuSet(
    /// The CPU indices, ascending and unique.
    pub Vec<usize>,
);

impl CpuSet {
    /// Parses a Linux cpulist, the `"0-7,16-23"` form written by sysfs, skipping whitespace and empty tokens.
    ///
    /// # Errors
    ///
    /// [`TopologyError::EmptySet`] if the list names no CPU, and [`TopologyError::InvalidCpuList`] if a token is not a decimal index or an ascending `lo-hi` range.
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

    fn is_subset_of(&self, other: &Self) -> bool {
        let other = other.normalized();
        self.0.iter().all(|cpu| other.binary_search(cpu).is_ok())
    }

    fn normalized(&self) -> Vec<usize> {
        let mut cpus = self.0.clone();
        cpus.sort_unstable();
        cpus.dedup();
        cpus
    }

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

/// The current affinity mask, or `0..available_parallelism` where it cannot be read.
#[must_use]
pub(crate) fn allowed_cpus() -> CpuSet {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `set` is a valid, correctly sized `cpu_set_t`; pid 0 is the calling thread.
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

/// The NUMA nodes visible in the current affinity mask, ascending by node id, each intersected with the mask.
///
/// Where sysfs exposes no usable node, the whole affinity mask is reported as node 0.
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
pub(crate) fn sysfs_numa_nodes(allowed: &CpuSet) -> Vec<(usize, CpuSet)> {
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
pub(crate) fn sysfs_numa_nodes(_allowed: &CpuSet) -> Vec<(usize, CpuSet)> {
    Vec::new()
}

/// Pins the calling thread to `set`; a no-op on non-Linux targets.
pub(super) fn pin_current_thread(set: &CpuSet) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        if set.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot pin a thread to an empty CPU set",
            ));
        }
        // SAFETY: `mask` is a valid, zeroed `cpu_set_t`; every index is checked against `CPU_SETSIZE` before `CPU_SET` touches it.
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

/// Binds the calling thread's allocations to one NUMA node (`MPOL_BIND`), or restores `MPOL_DEFAULT` for `None`; a no-op on non-Linux targets.
pub(super) fn bind_current_thread_memory(node: Option<usize>) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        const BITS: usize = 8 * std::mem::size_of::<libc::c_ulong>();
        let rc = match node {
            Some(node) => {
                let words = node / BITS + 1;
                let mut mask = vec![0 as libc::c_ulong; words];
                mask[node / BITS] = 1 << (node % BITS);
                // SAFETY: `mask` outlives the call and holds `words` words, exactly the `maxnode` bits the kernel reads.
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

/// One resolved partition: where its pool's workers run and how many there are.
#[derive(Clone, Debug)]
pub struct PartitionSlot {
    /// CPUs to pin the pool's workers to, or `None` to leave them unpinned.
    pub cpus: Option<CpuSet>,
    /// NUMA node to bind the workers' allocations to, when the slot sits in exactly one node.
    pub node: Option<usize>,
    /// Worker count for the slot's Rayon pool.
    pub threads: usize,
    /// The CUDA device a device partition runs on, `None` for a host partition.
    pub device: Option<u32>,
}

/// Builds one partition's Rayon pool, each worker pinning itself before Rayon's main loop and only warning when pinning fails.
pub(super) fn build_pool(
    slot: &PartitionSlot,
    bind_memory: bool,
    name: &str,
) -> Result<rayon::ThreadPool, TopologyError> {
    let cpus = slot.cpus.clone();
    let node = slot.node;
    #[cfg(feature = "cuda")]
    let device = slot.device;
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
                #[cfg(feature = "cuda")]
                if let Some(device) = device {
                    crate::engine::cuda_context::bind_device_context(device);
                }
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
    /// The partition count is the node count rounded **down** to a power of two (at least one); when that is fewer than the node count, adjacent nodes are merged into a partition rather than dropped, so every allowed CPU stays in play.
    /// Each partition takes as many threads as it has CPUs.
    Auto {
        /// Upper bound on the partition count, itself rounded down to a power of two.
        max_partitions: Option<usize>,
    },
    /// One partition per listed CPU set, exactly as given.
    ///
    /// Sets may overlap (the resolve logs a warning and proceeds), but each must be non-empty and a subset of the process's affinity mask.
    Explicit(
        /// The CPU sets, one per partition; the count must be a power of two.
        Vec<CpuSet>,
    ),
    /// Partitions with no pinning at all, for tests and CI.
    Unpinned {
        /// Number of partitions; must be a power of two.
        partitions: usize,
        /// Threads per partition, or `None` to split `available_parallelism` evenly (at least one each).
        threads_per_partition: Option<usize>,
    },
    /// `per_device` partitions on each listed CUDA device, in rank order, for the `cuda` backend's [`GpuPartitionedSum`](crate::gpu::GpuPartitionedSum).
    ///
    /// `devices.len() × per_device` must be a power of two; each slot is unpinned with a small host pool for the export and receive plumbing, and every worker of that pool binds the device's context when it starts.
    #[cfg(feature = "cuda")]
    Devices {
        /// Device ordinals, one entry per device.
        devices: Vec<u32>,
        /// Partitions sharing each device; `1` in production, more to test the exchange on one device.
        per_device: usize,
    },
}

/// Host workers of a device partition's pool.
#[cfg(feature = "cuda")]
const DEVICE_PARTITION_THREADS: usize = 4;

/// Placement plus the knobs the partitioned engine reads alongside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionConfig {
    /// How partitions map onto the machine.
    pub placement: Placement,
    /// Whether each pool's workers bind their allocations to the partition's NUMA node (Python's `pin_memory=`).
    pub bind_memory: bool,
    /// Seed of the partition rows, or `None` for the sum's own hash seed.
    pub partition_row_seed: Option<u64>,
}

impl Default for PartitionConfig {
    /// One partition per NUMA node, memory bound to it, default partition rows.
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
    /// Resolves the placement against the machine into one slot per partition, a power of two of them.
    ///
    /// # Errors
    ///
    /// [`TopologyError::NotPowerOfTwo`] if an explicit or unpinned partition count is not a power of two, [`TopologyError::EmptySet`] if an explicit set is empty, and [`TopologyError::CpuNotAllowed`] if an explicit set names a CPU outside the affinity mask.
    pub fn resolve(&self) -> Result<Vec<PartitionSlot>, TopologyError> {
        match &self.placement {
            Placement::Auto { max_partitions } => Ok(resolve_auto(*max_partitions)),
            Placement::Explicit(sets) => resolve_explicit(sets),
            Placement::Unpinned {
                partitions,
                threads_per_partition,
            } => resolve_unpinned(*partitions, *threads_per_partition),
            #[cfg(feature = "cuda")]
            Placement::Devices {
                devices,
                per_device,
            } => resolve_devices(devices, *per_device),
        }
    }
}

#[cfg(feature = "cuda")]
fn resolve_devices(
    devices: &[u32],
    per_device: usize,
) -> Result<Vec<PartitionSlot>, TopologyError> {
    let count = devices.len() * per_device;
    if count == 0 || !count.is_power_of_two() {
        return Err(TopologyError::NotPowerOfTwo(count));
    }
    Ok(devices
        .iter()
        .flat_map(|&d| {
            std::iter::repeat_n(
                PartitionSlot {
                    cpus: None,
                    node: None,
                    threads: DEVICE_PARTITION_THREADS,
                    device: Some(d),
                },
                per_device,
            )
        })
        .collect())
}

fn resolve_auto(max_partitions: Option<usize>) -> Vec<PartitionSlot> {
    let nodes = numa_nodes();
    let mut count = prev_power_of_two(nodes.len());
    if let Some(max) = max_partitions {
        count = prev_power_of_two(count.min(max));
    }

    // Merge adjacent nodes when the count was rounded down, the first `nodes.len() % count` partitions taking one node more.
    let (base, remainder) = (nodes.len() / count, nodes.len() % count);
    let mut slots = Vec::with_capacity(count);
    let mut rest = nodes.as_slice();
    for partition in 0..count {
        let take = base + usize::from(partition < remainder);
        let (group, tail) = rest.split_at(take);
        rest = tail;
        let cpus = group
            .iter()
            .fold(CpuSet(Vec::new()), |accumulator, (_, set)| {
                accumulator.union(set)
            });
        slots.push(PartitionSlot {
            node: (group.len() == 1).then(|| group[0].0),
            threads: cpus.len(),
            cpus: Some(cpus),
            device: None,
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
            device: None,
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
            device: None,
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
mod tests;
