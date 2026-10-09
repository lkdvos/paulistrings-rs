//! [`GpuPartitionedSum`], [`PartitionedSum`] over device partitions, its one-device form [`GpuPauliSum`], and their front doors.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use super::error::GpuError;
use super::layer::{GpuLayerCounters, GpuLayerOptions};
use super::partition::DevicePartition;
use super::sum::GpuSum;
use super::truncation::KeepProgram;
use super::wire::PeerWire;
use crate::circuit::Circuit;
use crate::collectives::Collectives;
use crate::engine::partitioned::backend::PartitionStorage;
use crate::engine::partitioned::driver::scatter_local;
#[cfg(feature = "phase-timing")]
use crate::engine::partitioned::PartitionPhaseStats;
use crate::engine::partitioned::{
    PartitionConfig, PartitionRowPolicy, PartitionRuntime, PartitionedSum, Placement,
    ScatterOptions, ScatterRows,
};
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows};
use crate::pauli_sum::PauliSum;
use crate::truncation::BuiltinTruncation;

const LOG_TARGET: &str = "paulistrings::propagate";

/// A [`PauliSum`] split across the device partitions of a [`PartitionRuntime`] resolved from [`Placement::Devices`] (ARCHITECTURE.md §Partitioning): [`PartitionedSum`] over the device backend.
///
/// Rows a layer moves between partitions go device to device and never touch the host.
/// Several partitions may share one device (`per_device > 1`), which is a testing shape; one partition per device is the production one.
/// Held across calls: scatter once with [`scatter_to_devices`](Self::scatter_to_devices) (or [`from_host`](Self::from_host) for one device), step many times with [`propagate`](Self::propagate), read back with [`gather`](Self::gather).
///
/// A policy is a [`BuiltinTruncation`], which every builtin converts into: per-term filters run inside the fused layer, `ApproxTopN` as a device layer pass with the host's collective, `And`/`Or` as on the host, and an exact `TopN` as a local radix-select (K8) at one partition only, since the `n`-th largest of a split sum has no collective form.
///
/// A device error mid-run leaves a one-partition sum holding the last completed layer's output, and a later call resumes from it; above one partition the split is **poisoned** instead (see [`propagate_with`](Self::propagate_with)).
pub type GpuPartitionedSum<const W: usize> = PartitionedSum<W, DevicePartition<W>>;

/// A [`PauliSum`] resident on one CUDA device: a [`GpuPartitionedSum`] of one partition, built by [`from_host`](GpuPartitionedSum::from_host) (ARCHITECTURE.md §GPU-Readiness).
pub type GpuPauliSum<const W: usize> = GpuPartitionedSum<W>;

/// The policy's per-term program, after checking that it lowers, that every channel prepares, and that an exact `TopN` runs only at `single_partition`.
pub(super) fn lower_for_run<const W: usize>(
    circuit: &Circuit<W>,
    policy: &BuiltinTruncation,
    direction: Direction,
    hash: &Gf2Hash<W>,
    single_partition: bool,
) -> Result<KeepProgram, GpuError> {
    if policy.contains_exact_top_n() && !single_partition {
        return Err(GpuError::Unsupported("exact TopN on device"));
    }
    if policy.contains_collapse_sample() {
        return Err(GpuError::Unsupported("CollapseSample on device"));
    }
    let keep = KeepProgram::lower(policy)?;
    let adjoint = matches!(direction, Direction::Heisenberg);
    if circuit
        .channels
        .iter()
        .any(|ch| ch.prepare(hash, adjoint).is_none())
    {
        return Err(GpuError::Unsupported(
            "channel with support wider than MAX_LOCAL_SUPPORT",
        ));
    }
    Ok(keep)
}

/// `(rank, code)` of the first rank of `collectives`'s group whose `code` is non-zero: every rank's failure agreed in one all-reduce of `size` words. **Collective.**
pub fn first_failure(collectives: &dyn Collectives, code: u64) -> Option<(usize, u64)> {
    let mut codes = vec![0u64; collectives.size() as usize];
    codes[collectives.rank() as usize] = code;
    collectives.allreduce_sum_u64(&mut codes);
    first_nonzero(&codes)
}

fn first_nonzero(codes: &[u64]) -> Option<(usize, u64)> {
    codes
        .iter()
        .position(|&code| code != 0)
        .map(|rank| (rank, codes[rank]))
}

/// The one-partition runtime on `device`, built once per process and device so a caller scattering every call pays for its pool once.
pub(super) fn device_runtime(device: u32) -> Result<Arc<PartitionRuntime>, GpuError> {
    static RUNTIMES: Mutex<Vec<(u32, Arc<PartitionRuntime>)>> = Mutex::new(Vec::new());
    let mut cache = RUNTIMES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, runtime)) = cache.iter().find(|(cached, _)| *cached == device) {
        return Ok(runtime.clone());
    }
    let config = PartitionConfig {
        placement: Placement::Devices {
            devices: vec![device],
            per_device: 1,
        },
        bind_memory: false,
        partition_row_seed: None,
    };
    let runtime = PartitionRuntime::new(&config).map_err(GpuError::Topology)?;
    cache.push((device, runtime.clone()));
    Ok(runtime)
}

/// Rank `rank` of a `size`-partition group: its share of `sum` under `rows` uploaded to `device`. **Collective** above one partition, through [`scatter_local`].
pub(super) fn upload_share<const W: usize>(
    sum: &PauliSum<W>,
    rows: &PartitionRows<W>,
    (rank, size): (u32, u32),
    device: u32,
    collectives: &dyn Collectives,
    extra_options: &[String],
) -> Result<DevicePartition<W>, GpuError> {
    let device_sum = if size == 1 {
        GpuSum::from_host_with_options(sum, device, extra_options)?
    } else {
        let local = scatter_local(sum, rows, rank, collectives);
        GpuSum::from_host_with_options(&local, device, extra_options)?
    };
    let mut partition = DevicePartition::new(device_sum, GpuLayerOptions::default())?;
    partition.group_size = size;
    Ok(partition)
}

impl<const W: usize> PartitionedSum<W, DevicePartition<W>> {
    /// Upload `sum` whole to device `ordinal`, as a one-partition split.
    ///
    /// # Errors
    ///
    /// [`GpuError::Topology`] if the one-device placement does not resolve, and any device error of the upload.
    pub fn from_host(sum: &PauliSum<W>, ordinal: u32) -> Result<Self, GpuError> {
        Self::one_device(sum, ordinal, &[])
    }

    /// As [`Self::from_host`] with extra NVRTC options, the `-DFP_BITS=<b>` collision hook.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn from_host_with_options(
        sum: &PauliSum<W>,
        ordinal: u32,
        extra_options: &[String],
    ) -> Result<Self, GpuError> {
        Self::one_device(sum, ordinal, extra_options)
    }

    fn one_device(
        sum: &PauliSum<W>,
        ordinal: u32,
        extra_options: &[String],
    ) -> Result<Self, GpuError> {
        let runtime = device_runtime(ordinal)?;
        Self::upload(
            sum,
            PartitionRows::none(sum.num_qubits()),
            runtime,
            extra_options,
        )
    }

    /// Splits `sum` across the device partitions `config` places, by seeded rows as [`PartitionedSum::scatter`] draws them.
    ///
    /// # Errors
    ///
    /// [`GpuError::Topology`] if `config` does not resolve, [`GpuError::Unsupported`] if a slot names no device, and any device error of the uploads.
    pub fn scatter_to_devices(
        sum: &PauliSum<W>,
        config: &PartitionConfig,
    ) -> Result<Self, GpuError> {
        let options = ScatterOptions {
            runtime: PartitionRuntime::new(config).map_err(GpuError::Topology)?,
            rows: ScatterRows::Policy(PartitionRowPolicy::Seeded(config.partition_row_seed)),
        };
        Self::scatter_to_devices_with(sum, options)
    }

    /// [`scatter_to_devices`](Self::scatter_to_devices) onto a caller-built runtime, split by the rows `options` names.
    ///
    /// # Errors
    ///
    /// As [`Self::scatter_to_devices`], [`GpuError::Topology`] aside.
    ///
    /// # Panics
    ///
    /// If the rows do not name the runtime's partition count or the sum's qubit count.
    pub fn scatter_to_devices_with(
        sum: &PauliSum<W>,
        options: ScatterOptions<W>,
    ) -> Result<Self, GpuError> {
        let ScatterOptions { runtime, rows } = options;
        let rows = rows.resolve(sum, runtime.partition_bits());
        Self::upload(sum, rows, runtime, &[])
    }

    fn upload(
        sum: &PauliSum<W>,
        rows: PartitionRows<W>,
        runtime: Arc<PartitionRuntime>,
        extra_options: &[String],
    ) -> Result<Self, GpuError> {
        let size = runtime.num_partitions();
        rows.assert_splits(sum.hash(), sum.num_qubits(), size);
        let devices: Vec<u32> = runtime
            .slots()
            .iter()
            .map(|slot| {
                slot.device.ok_or(GpuError::Unsupported(
                    "GpuPartitionedSum needs a runtime resolved from Placement::Devices",
                ))
            })
            .collect::<Result<_, _>>()?;
        let started = Instant::now();
        let partitions: Vec<Result<DevicePartition<W>, GpuError>> = {
            let (rows, devices) = (&rows, &devices);
            let wires = PeerWire::group(size as u32);
            runtime.map_partitions(wires, |rank, wire, transport| {
                let ids = (rank as u32, size as u32);
                let mut partition =
                    upload_share(sum, rows, ids, devices[rank], transport, extra_options)?;
                partition.scratch_mut().export.wire = Some(Arc::new(wire));
                Ok(partition)
            })
        };
        let partitions = partitions.into_iter().collect::<Result<Vec<_>, _>>()?;
        log::info!(
            target: LOG_TARGET,
            "scatter_gpu: {} terms over {size} device partitions [{}], {} bucket bits, {:.3} s",
            sum.len(),
            runtime.placement_summary(),
            partitions[0].hash().bits(),
            started.elapsed().as_secs_f64(),
        );
        let scatter_ns = started.elapsed().as_nanos() as u64;
        Ok(Self::from_backend(partitions, rows, runtime, scatter_ns))
    }

    /// Propagate through `circuit` with the default [`PropagateOptions`].
    pub fn propagate(
        &mut self,
        circuit: &Circuit<W>,
        policy: impl Into<BuiltinTruncation>,
        direction: Direction,
    ) -> Result<(), GpuError> {
        self.propagate_with(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagate through `circuit`, every partition on its own thread and device, in lock-step through the shared layer loop; `options` drive the host-side bucket schedule, which the device bucket policy of [`Self::set_layer_options`] refines on top of.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] before any layer if the policy's per-term part lowers to more than 15 nodes, if a channel's support is wider than `MAX_LOCAL_SUPPORT`, or if the policy holds an exact `TopN` and the split has more than one partition.
    /// A group member runs every layer at the agreed bucket count and never refines on its own, so a fused-layer block or a received segment over the kernel's cap is [`GpuError::Unsupported`] rather than a retry.
    /// A device error on any partition is returned after the loop, the first by rank, the partners having finished the call on empty exchange blocks.
    /// Above one partition the split is then **poisoned**: its partitions no longer hold one consistent sum, and every later `propagate` or [`Self::gather`] returns [`GpuError::Poisoned`] until the caller scatters again.
    pub fn propagate_with(
        &mut self,
        circuit: &Circuit<W>,
        policy: impl Into<BuiltinTruncation>,
        direction: Direction,
        options: PropagateOptions,
    ) -> Result<(), GpuError> {
        let policy = policy.into();
        let partitions = self.backends_mut();
        partitions[0].check_poison()?;
        let single = partitions.len() == 1;
        let keep = lower_for_run(circuit, &policy, direction, partitions[0].hash(), single)?;
        for partition in partitions.iter_mut() {
            partition.take_error()?;
            partition.keep = keep;
        }
        self.propagate_on_backend(circuit, &policy, direction, options);
        let partitions = self.backends_mut();
        let (mut outcomes, codes): (Vec<_>, Vec<u64>) = partitions
            .iter_mut()
            .map(DevicePartition::take_failure)
            .unzip();
        let Some((rank, code)) = first_nonzero(&codes) else {
            return Ok(());
        };
        if !single {
            for partition in partitions.iter_mut() {
                partition.poison(rank, code as usize - 1);
            }
        }
        std::mem::replace(&mut outcomes[rank], Ok(()))
    }

    /// Fail partition `rank` before the exchange of its `layer`-th layer of the next call (test hook).
    #[cfg(any(test, feature = "test-utils"))]
    pub fn inject_failure(&mut self, rank: usize, layer: usize) {
        self.backends_mut()[rank].fail_at_layer = Some(layer);
    }

    /// Make partition `rank`'s next chunked receive fail as out of memory once chunk `chunk` has moved, mid-layer (test hook).
    #[cfg(any(test, feature = "test-utils"))]
    pub fn inject_chunk_oom(&mut self, rank: usize, chunk: usize) {
        self.backends_mut()[rank]
            .scratch_mut()
            .export
            .fail_after_chunk = Some(chunk);
    }

    /// Download every partition, each bucket re-sorted to the host's order, and merge them back into one sum.
    ///
    /// # Errors
    ///
    /// [`GpuError::Poisoned`] after a failed call, otherwise any download error.
    pub fn gather(&self) -> Result<PauliSum<W>, GpuError> {
        let partitions = self.backends();
        partitions[0].check_poison()?;
        #[cfg(feature = "phase-timing")]
        let started = Instant::now();
        let shares = std::thread::scope(|scope| {
            let downloads: Vec<_> = partitions
                .iter()
                .map(|partition| scope.spawn(|| partition.sum().to_host()))
                .collect();
            downloads
                .into_iter()
                .map(|download| download.join().expect("a download thread panicked"))
                .collect::<Result<Vec<_>, _>>()
        })?;
        let out = PauliSum::merge_partitions(shares);
        #[cfg(feature = "phase-timing")]
        self.lap_gather(started.elapsed().as_nanos() as u64);
        Ok(out)
    }

    /// Each partition's device ordinal, in rank order.
    pub fn devices(&self) -> Vec<u32> {
        self.backends()
            .iter()
            .map(|partition| partition.sum().device())
            .collect()
    }

    /// The device layer's knobs on every partition; takes effect from the next layer.
    pub fn set_layer_options(&mut self, options: GpuLayerOptions) {
        for partition in self.backends_mut() {
            partition.scratch_mut().options = options;
        }
    }

    /// The counters of the most recent layer on partition `rank`.
    pub fn last_layer_counters(&self, rank: usize) -> GpuLayerCounters {
        self.backends()[rank].counters()
    }

    /// Drain the per-phase counters: one [`PhaseStats`](crate::PhaseStats) per partition plus the driver's scatter and gather laps (feature `phase-timing`).
    ///
    /// Kernel families land on the host phases they replace: K1+K2 in `gather_ns`, the fused K3 in `merge_ns` (`sort_ns` stays zero, the sort being inside it), K4 in `compact_ns`, K5 in `rescale_ns`, refines in `rebucket_ns`; `coset_loop_ns` is the driving thread's wall over the fused path.
    #[cfg(feature = "phase-timing")]
    pub fn take_stats(&mut self) -> PartitionPhaseStats {
        let per_partition = self
            .backends_mut()
            .iter_mut()
            .map(|partition| std::mem::take(partition.stats()))
            .collect();
        let (scatter_ns, gather_ns, layers) = self.take_driver_laps();
        PartitionPhaseStats {
            per_partition,
            scatter_ns,
            gather_ns,
            layers,
        }
    }
}

/// Propagate `sum` through `circuit` on device `device` and return the result on the host.
///
/// One-shot convenience over [`GpuPauliSum`]; a caller stepping repeatedly should hold the resident sum instead.
pub fn propagate_gpu<const W: usize>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: impl Into<BuiltinTruncation>,
    direction: Direction,
    device: u32,
) -> Result<PauliSum<W>, GpuError> {
    let mut resident = GpuPauliSum::from_host(sum, device)?;
    resident.propagate(circuit, policy, direction)?;
    resident.gather()
}

/// Propagate `sum` through `circuit` on the device partitions `config` places, and gather the result.
///
/// One-shot convenience over [`GpuPartitionedSum`].
///
/// # Errors
///
/// [`GpuError::Topology`] if `config` does not resolve, then as [`GpuPartitionedSum::propagate`].
pub fn propagate_gpu_partitioned<const W: usize>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: impl Into<BuiltinTruncation>,
    direction: Direction,
    config: &PartitionConfig,
) -> Result<PauliSum<W>, GpuError> {
    let mut split = GpuPartitionedSum::scatter_to_devices(sum, config)?;
    split.propagate(circuit, policy, direction)?;
    split.gather()
}
