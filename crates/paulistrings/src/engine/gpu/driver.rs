//! [`GpuPauliSum`], a sum propagated on one CUDA device, [`GpuPartitionedSum`], a sum split across device partitions, and their front doors. Both run the shared partitioned layer loop.

use std::sync::Arc;
use std::time::Instant;

use super::error::GpuError;
use super::layer::{GpuKernelMs, GpuLayerCounters, GpuLayerOptions};
use super::partition::DevicePartition;
use super::payload::{self, GpuExchange};
use super::sum::GpuSum;
use super::truncation::DevicePolicy;
use crate::bucket::hash::{Gf2Hash, PartitionRows};
use crate::circuit::Circuit;
use crate::engine::partitioned::backend::PartitionStorage;
use crate::engine::partitioned::driver::{run_layers, scatter_local, PartitionCtx, PartitionWork};
use crate::engine::partitioned::trace::{assemble, PartitionTrace};
use crate::engine::partitioned::transport::InProcessTransport;
use crate::engine::partitioned::truncation::PartitionedTruncation;
#[cfg(feature = "phase-timing")]
use crate::engine::partitioned::PartitionPhaseStats;
use crate::engine::partitioned::{PartitionConfig, PartitionRuntime};
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::PauliSum;

const LOG_TARGET: &str = "paulistrings::propagate";

/// A [`PauliSum`] resident on one CUDA device, propagated layer by layer with the fused device layer (ARCHITECTURE.md §GPU-Readiness).
///
/// Built by [`Self::from_host`], stepped by [`Self::propagate`], read back by [`Self::to_host`].
/// The layer loop is the partitioned engine's at `P = 1`, so the bucket-count schedule, trace and log lines are those of [`PartitionedSum`](crate::engine::partitioned::PartitionedSum).
/// A policy runs here through its [`BuiltinTruncation`](crate::truncation::BuiltinTruncation) tree from [`TruncationPolicy::device_policy`](crate::TruncationPolicy::device_policy): per-term filters inside the fused layer, `ApproxTopN` as a device layer pass with the host's collective, exact `TopN` as a local radix-select (K8, no collective needed at one partition), `And`/`Or` as on the host.
/// `propagate` fails with [`GpuError::Unsupported`] before touching the device if the policy has no tree or if its per-term part lowers to more than 15 nodes; unlike [`GpuPartitionedSum`] and `gpu::MpiGpuSum`, an exact `TopN` runs here, since this sum is always one partition.
///
/// A device error mid-run leaves the sum in the state of the last completed layer and is returned from `propagate`.
pub struct GpuPauliSum<const W: usize> {
    part: DevicePartition<W>,
    rows: PartitionRows<W>,
    trace: Option<PartitionTrace>,
}

/// The checks every device propagation makes before its first layer: the policy lowers, its layer pass agrees with `finalizes_layer`, and every channel prepares.
///
/// `single_partition` gates an exact `TopN` in the tree: `true` for [`GpuPauliSum`] alone, `false` for [`GpuPartitionedSum`] and `gpu::MpiGpuSum`, whose partitions have no collective `n`-th-largest.
pub(super) fn lower_for_run<const W: usize, T>(
    circuit: &Circuit<W>,
    policy: &T,
    direction: Direction,
    hash: &Gf2Hash<W>,
    single_partition: bool,
) -> Result<DevicePolicy, GpuError>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    let tree = policy.device_policy().ok_or(GpuError::Unsupported(
        "truncation policy without a device form",
    ))?;
    if tree.contains_exact_top_n() && !single_partition {
        return Err(GpuError::Unsupported("exact TopN on device"));
    }
    // The driver runs the layer pass iff `policy` says so, so a tree that disagrees would be skipped or run wrongly.
    if <_ as crate::TruncationPolicy<W>>::finalizes_layer(&tree) != policy.finalizes_layer() {
        return Err(GpuError::Unsupported(
            "a device_policy whose layer pass disagrees with finalizes_layer",
        ));
    }
    let lowered = DevicePolicy::lower(tree)?;
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
    Ok(lowered)
}

impl<const W: usize> GpuPauliSum<W> {
    /// Upload `sum` to device `ordinal`.
    pub fn from_host(sum: &PauliSum<W>, ordinal: u32) -> Result<Self, GpuError> {
        Self::wrap(GpuSum::from_host(sum, ordinal)?, sum.num_qubits())
    }

    /// As [`Self::from_host`] with extra NVRTC options, the `-DFP_BITS=<b>` collision hook.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn from_host_with_options(
        sum: &PauliSum<W>,
        ordinal: u32,
        extra_options: &[String],
    ) -> Result<Self, GpuError> {
        Self::wrap(
            GpuSum::from_host_with_options(sum, ordinal, extra_options)?,
            sum.num_qubits(),
        )
    }

    fn wrap(sum: GpuSum<W>, num_qubits: usize) -> Result<Self, GpuError> {
        Ok(Self {
            part: DevicePartition::new(sum, GpuLayerOptions::default())?,
            rows: PartitionRows::none(num_qubits),
            trace: None,
        })
    }

    /// Propagate through `circuit` with the default [`PropagateOptions`].
    pub fn propagate<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
    ) -> Result<(), GpuError>
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        self.propagate_with_options(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagate through `circuit`; `options` drive the host-side bucket schedule, the device policy in [`Self::set_layer_options`] refines on top of it.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] before any layer if `policy` cannot run on device (see the type docs) or a channel's support is wider than `MAX_LOCAL_SUPPORT`; any device error from a layer, after which the sum holds the last completed layer's output and a later call resumes from it.
    pub fn propagate_with_options<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
    ) -> Result<(), GpuError>
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        let lowered = lower_for_run(circuit, policy, direction, self.part.hash(), true)?;
        self.part.take_error()?;
        self.part.policy = lowered;
        let n = circuit.channels.len();
        let terms_in = self.len();
        let started = Instant::now();
        log::info!(
            target: LOG_TARGET,
            "propagate_gpu: {terms_in} terms through {n} channels ({direction:?}) on device {}",
            self.device(),
        );
        if n > 0 {
            let tracing = self.trace.is_some();
            let mut work = PartitionWork::take(&mut self.part, n, tracing);
            let transports = InProcessTransport::group(1);
            let ctx = PartitionCtx {
                rows: &self.rows,
                rank: 0,
                size: 1,
                tracing,
            };
            run_layers(
                circuit,
                policy,
                direction,
                options,
                ctx,
                &mut work,
                &transports[0],
            );
            self.part = work.local;
            if let Some(trace) = self.trace.as_mut() {
                assemble(trace, vec![work.rows]);
            }
        }
        log::info!(
            target: LOG_TARGET,
            "propagate_gpu: {n} layers applied, {terms_in} -> {} terms, {:.3} s",
            self.len(),
            started.elapsed().as_secs_f64(),
        );
        self.part.take_error()
    }

    /// Download the sum, each bucket re-sorted to the host's order.
    pub fn to_host(&self) -> Result<PauliSum<W>, GpuError> {
        self.part.sum().to_host()
    }

    /// Terms on the device.
    pub fn len(&self) -> usize {
        self.part.sum().len()
    }

    /// `true` if the sum has no terms.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Active bucket bits.
    pub fn bits(&self) -> u8 {
        self.part.sum().bits()
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.part.sum().num_qubits()
    }

    /// The device ordinal.
    pub fn device(&self) -> u32 {
        self.part.sum().device()
    }

    /// The device layer's knobs; takes effect from the next layer.
    pub fn set_layer_options(&mut self, options: GpuLayerOptions) {
        self.part.scratch_mut().options = options;
    }

    /// Record CUDA-event timings per kernel family into [`Self::take_kernel_ms`]; costs a synchronization per kernel.
    pub fn set_kernel_timing(&mut self, on: bool) {
        self.part.scratch_mut().time_kernels = on;
    }

    /// Kernel milliseconds accumulated since the last call, zeroing the counters.
    pub fn take_kernel_ms(&mut self) -> GpuKernelMs {
        std::mem::take(&mut self.part.scratch_mut().kernel_ms)
    }

    /// The counters of the most recent layer.
    pub fn last_layer_counters(&self) -> GpuLayerCounters {
        self.part.counters()
    }

    /// Drain the per-phase counters accumulated since the last call (feature `phase-timing`).
    ///
    /// Kernel families land on the host phases they replace: K1+K2 in `gather_ns`, the fused K3 in `merge_ns` (`sort_ns` stays zero, the sort being inside it), K4 in `compact_ns`, K5 in `rescale_ns`, refines in `rebucket_ns`; `coset_loop_ns` is the driving thread's wall over the fused path.
    #[cfg(feature = "phase-timing")]
    pub fn take_stats(&mut self) -> crate::PhaseStats {
        std::mem::take(self.part.stats())
    }

    /// Start recording one [`PartitionTrace`] record per layer.
    pub fn enable_trace(&mut self) {
        self.trace.get_or_insert_with(PartitionTrace::default);
    }

    /// Drain the per-layer records, `Some` iff tracing is on.
    pub fn take_trace(&mut self) -> Option<PartitionTrace> {
        self.trace.as_mut().map(std::mem::take)
    }
}

/// Propagate `sum` through `circuit` on device `device` and return the result on the host.
///
/// One-shot convenience over [`GpuPauliSum`]; a caller stepping repeatedly should hold the resident sum instead.
pub fn propagate_gpu<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    direction: Direction,
    device: u32,
) -> Result<PauliSum<W>, GpuError>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    let mut dev = GpuPauliSum::from_host(sum, device)?;
    dev.propagate(circuit, policy, direction)?;
    dev.to_host()
}

/// A [`PauliSum`] split across the device partitions of a [`PartitionRuntime`] resolved from [`Placement::Devices`](crate::engine::partitioned::Placement::Devices) (ARCHITECTURE.md §Partitioning).
///
/// Every partition is a [`GpuPauliSum`]'s worth of device state driven by the shared layer loop, with the rows a layer moves across partitions exported by K10, exchanged through the in-process transport and merged by the fused layer.
/// The exchange moves device-resident blocks by default ([`GpuExchange::Device`]): the receiver copies them device-to-device, across devices through a peer copy, and no row touches the host; [`Self::set_exchange`] or `PAULISTRINGS_GPU_EXCHANGE=host` selects the host-staged form.
/// Several partitions may share one device (`per_device > 1`), which is the testing shape; one partition per device is the production one.
/// Held across calls like [`PartitionedSum`](crate::engine::partitioned::PartitionedSum): scatter once, step many times, gather once.
pub struct GpuPartitionedSum<const W: usize> {
    parts: Vec<DevicePartition<W>>,
    rows: PartitionRows<W>,
    runtime: Arc<PartitionRuntime>,
    trace: Option<PartitionTrace>,
    /// `(rank, layer)` of the first partition error; set once, refuses every later call.
    poison: Option<(usize, usize)>,
    #[cfg(feature = "phase-timing")]
    scatter_ns: u64,
    #[cfg(feature = "phase-timing")]
    gather_ns: std::sync::atomic::AtomicU64,
    #[cfg(feature = "phase-timing")]
    layers: u64,
}

impl<const W: usize> GpuPartitionedSum<W> {
    /// Splits `sum` across `runtime`'s device partitions, deriving the partition rows from `config` as [`PartitionedSum::scatter`](crate::engine::partitioned::PartitionedSum::scatter) does.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] if a slot of `runtime` names no device, and any device error of the uploads.
    pub fn scatter(
        sum: PauliSum<W>,
        runtime: Arc<PartitionRuntime>,
        config: &PartitionConfig,
    ) -> Result<Self, GpuError> {
        let seed = config
            .partition_row_seed
            .unwrap_or_else(|| sum.hash().seed());
        let rows = PartitionRows::<W>::from_seed(sum.num_qubits(), runtime.partition_bits(), seed);
        Self::scatter_with_rows(sum, rows, runtime)
    }

    /// Splits `sum` across `runtime`'s device partitions with caller-supplied rows; each partition filters its share on its own thread and uploads it to its slot's device.
    ///
    /// # Errors
    ///
    /// As [`Self::scatter`].
    ///
    /// # Panics
    ///
    /// If `rows` does not name the runtime's partition count or the sum's qubit count.
    pub fn scatter_with_rows(
        sum: PauliSum<W>,
        rows: PartitionRows<W>,
        runtime: Arc<PartitionRuntime>,
    ) -> Result<Self, GpuError> {
        let size = runtime.num_partitions();
        assert_eq!(
            rows.num_partitions(),
            size,
            "partition rows name {} partitions but the runtime has {size}",
            rows.num_partitions(),
        );
        assert_eq!(
            rows.num_qubits(),
            sum.num_qubits(),
            "partition rows are for {} qubits, the sum for {}",
            rows.num_qubits(),
            sum.num_qubits(),
        );
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
        let exchange = payload::gpu_exchange_default();
        let parts: Vec<Result<DevicePartition<W>, GpuError>> = {
            let (sum, rows, devices) = (&sum, &rows, &devices);
            runtime.map_partitions((0..size).collect(), |rank, _, transport| {
                let local = scatter_local(sum, rows, rank as u32, transport);
                let dev = GpuSum::from_host(&local, devices[rank])?;
                let mut part = DevicePartition::new(dev, GpuLayerOptions::default())?;
                part.group_size = size as u32;
                part.scratch_mut().export.mode = exchange;
                Ok(part)
            })
        };
        let parts = parts.into_iter().collect::<Result<Vec<_>, _>>()?;
        log::info!(
            target: LOG_TARGET,
            "scatter_gpu: {} terms over {size} device partitions [{}], {} bucket bits, {:.3} s",
            sum.len(),
            runtime.placement_summary(),
            parts[0].hash().bits(),
            started.elapsed().as_secs_f64(),
        );
        Ok(Self {
            parts,
            rows,
            runtime,
            trace: None,
            poison: None,
            #[cfg(feature = "phase-timing")]
            scatter_ns: started.elapsed().as_nanos() as u64,
            #[cfg(feature = "phase-timing")]
            gather_ns: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "phase-timing")]
            layers: 0,
        })
    }

    /// Propagate through `circuit` with the default [`PropagateOptions`].
    pub fn propagate<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
    ) -> Result<(), GpuError>
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        self.propagate_with_options(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagate through `circuit`, every partition on its own thread and device, in lock-step through the shared layer loop.
    ///
    /// # Errors
    ///
    /// As [`GpuPauliSum::propagate_with_options`] before the first layer, except that an exact `TopN` anywhere in the tree is always [`GpuError::Unsupported`] here: every partition holds a disjoint slice, and the `n`-th largest of the whole split sum has no collective form.
    /// A group member runs every layer at the agreed bucket count and never refines on its own, so a fused-layer block or a received segment over the kernel's cap is [`GpuError::Unsupported`] rather than a retry; a device error on any partition is returned after the loop, the first by rank, the partners having finished the call on empty exchange blocks.
    /// The split is then **poisoned**: its partitions no longer hold one consistent sum, and every later `propagate` or [`Self::gather`] returns [`GpuError::Poisoned`] until the caller scatters again.
    pub fn propagate_with_options<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
    ) -> Result<(), GpuError>
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        self.check_poison()?;
        let lowered = lower_for_run(circuit, policy, direction, self.parts[0].hash(), false)?;
        for part in &mut self.parts {
            part.take_error()?;
            part.policy = lowered.clone();
        }
        let n = circuit.channels.len();
        let size = self.parts.len();
        let terms_in = self.len();
        let started = Instant::now();
        log::info!(
            target: LOG_TARGET,
            "propagate_gpu_partitioned: {terms_in} terms through {n} channels ({direction:?}) on {size} device partitions [{}]",
            self.runtime.placement_summary(),
        );
        if n > 0 {
            let tracing = self.trace.is_some();
            let items: Vec<PartitionWork<DevicePartition<W>>> = self
                .parts
                .iter_mut()
                .map(|part| PartitionWork::take(part, n, tracing))
                .collect();
            let runtime = Arc::clone(&self.runtime);
            let rows = &self.rows;
            let done = runtime.map_partitions(items, |rank, mut work, transport| {
                let ctx = PartitionCtx {
                    rows,
                    rank,
                    size,
                    tracing,
                };
                run_layers(
                    circuit, policy, direction, options, ctx, &mut work, transport,
                );
                work
            });
            let mut traced = Vec::with_capacity(if tracing { size } else { 0 });
            for (rank, work) in done.into_iter().enumerate() {
                self.parts[rank] = work.local;
                if tracing {
                    traced.push(work.rows);
                }
            }
            if let Some(trace) = self.trace.as_mut() {
                assemble(trace, traced);
            }
            #[cfg(feature = "phase-timing")]
            {
                self.layers += n as u64;
            }
        }
        log::info!(
            target: LOG_TARGET,
            "propagate_gpu_partitioned: {n} layers applied, {terms_in} -> {} terms, {:.3} s",
            self.len(),
            started.elapsed().as_secs_f64(),
        );
        let mut first = Ok(());
        for (rank, part) in self.parts.iter_mut().enumerate() {
            let layer = part.failed_layer.take();
            let r = part.take_error();
            if r.is_err() && first.is_ok() {
                self.poison = Some((rank, layer.unwrap_or(0)));
                first = r;
            }
        }
        first
    }

    fn check_poison(&self) -> Result<(), GpuError> {
        match self.poison {
            Some((rank, layer)) => Err(GpuError::Poisoned { rank, layer }),
            None => Ok(()),
        }
    }

    /// Fail partition `rank` before the exchange of its `layer`-th layer of the next call (test hook).
    #[cfg(any(test, feature = "test-utils"))]
    pub fn inject_failure(&mut self, rank: usize, layer: usize) {
        self.parts[rank].fail_at_layer = Some(layer);
    }

    /// Download every partition and merge them back into one sum.
    ///
    /// # Errors
    ///
    /// [`GpuError::Poisoned`] after a failed call, otherwise any download error.
    pub fn gather(&self) -> Result<PauliSum<W>, GpuError> {
        self.check_poison()?;
        #[cfg(feature = "phase-timing")]
        let started = Instant::now();
        let parts = self
            .parts
            .iter()
            .map(|p| p.sum().to_host())
            .collect::<Result<Vec<_>, _>>()?;
        let out = PauliSum::merge_partitions(parts);
        #[cfg(feature = "phase-timing")]
        self.gather_ns.fetch_add(
            started.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(out)
    }

    /// Terms over every partition.
    pub fn len(&self) -> usize {
        self.parts.iter().map(|p| p.len()).sum()
    }

    /// `true` if no partition holds a term.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Partitions in the group.
    pub fn num_partitions(&self) -> usize {
        self.parts.len()
    }

    /// The bucket bits every partition shares between calls.
    pub fn bits(&self) -> u8 {
        self.parts[0].hash().bits()
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.parts[0].sum().num_qubits()
    }

    /// The rows deciding which partition a key belongs to.
    pub fn rows(&self) -> &PartitionRows<W> {
        &self.rows
    }

    /// The runtime this sum runs on.
    pub fn runtime(&self) -> &Arc<PartitionRuntime> {
        &self.runtime
    }

    /// The device layer's knobs on every partition; takes effect from the next layer.
    pub fn set_layer_options(&mut self, options: GpuLayerOptions) {
        for part in &mut self.parts {
            part.scratch_mut().options = options;
        }
    }

    /// The counters of the most recent layer on partition `rank`.
    pub fn last_layer_counters(&self, rank: usize) -> GpuLayerCounters {
        self.parts[rank].counters()
    }

    /// How exchange blocks travel between the partitions from the next layer on; see [`GpuExchange`].
    ///
    /// `Nccl` means `Device` here, since an in-process group's peer copies already go device to device.
    pub fn set_exchange(&mut self, mode: GpuExchange) {
        #[cfg(feature = "nccl")]
        let mode = match mode {
            GpuExchange::Nccl => GpuExchange::Device,
            other => other,
        };
        for part in &mut self.parts {
            part.scratch_mut().export.mode = mode;
        }
    }

    /// The exchange mode the partitions run under.
    pub fn exchange(&self) -> GpuExchange {
        self.parts[0].scratch().export.mode
    }

    /// Start recording one [`PartitionTrace`] record per layer.
    pub fn enable_trace(&mut self) {
        self.trace.get_or_insert_with(PartitionTrace::default);
    }

    /// Drain the per-layer records, `Some` iff tracing is on.
    pub fn take_trace(&mut self) -> Option<PartitionTrace> {
        self.trace.as_mut().map(std::mem::take)
    }

    /// Drain the per-phase counters: one [`PhaseStats`](crate::PhaseStats) per partition plus the driver's scatter and gather laps (feature `phase-timing`).
    #[cfg(feature = "phase-timing")]
    pub fn take_stats(&mut self) -> PartitionPhaseStats {
        PartitionPhaseStats {
            per_partition: self
                .parts
                .iter_mut()
                .map(|part| std::mem::take(part.stats()))
                .collect(),
            scatter_ns: std::mem::take(&mut self.scatter_ns),
            gather_ns: self.gather_ns.swap(0, std::sync::atomic::Ordering::Relaxed),
            layers: std::mem::take(&mut self.layers),
        }
    }
}

/// The pooled device payloads outlive any one split, so a dropped split frees them; a live split allocates its own again on its next remote layer.
impl<const W: usize> Drop for GpuPartitionedSum<W> {
    fn drop(&mut self) {
        payload::drain_bin();
    }
}

/// Propagate `sum` through `circuit` on the device partitions `config` places, and gather the result.
///
/// One-shot convenience over [`GpuPartitionedSum`].
///
/// # Errors
///
/// [`GpuError::Topology`] if `config` does not resolve, then as [`GpuPartitionedSum::propagate`].
pub fn propagate_gpu_partitioned<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    config: &PartitionConfig,
) -> Result<PauliSum<W>, GpuError>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    let runtime = PartitionRuntime::new(config).map_err(GpuError::Topology)?;
    let mut split = GpuPartitionedSum::scatter(sum, runtime, config)?;
    split.propagate(circuit, policy, direction)?;
    split.gather()
}
