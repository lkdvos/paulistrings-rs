//! [`GpuDistributedSum`], one device partition per process of a transport's group, with the MPI front door `MpiGpuSum` (feature `mpi`) and the device picker [`local_device_for_rank`]. See ARCHITECTURE.md §Partitioning.

use std::time::Instant;

use super::device::device_count;
use super::driver::lower_for_run;
use super::error::GpuError;
use super::layer::{GpuKernelMs, GpuLayerCounters, GpuLayerOptions};
use super::partition::DevicePartition;
use super::payload::GpuExchange;
use super::sum::GpuSum;
use crate::bucket::hash::PartitionRows;
use crate::circuit::Circuit;
use crate::engine::partitioned::backend::PartitionStorage;
use crate::engine::partitioned::distributed::gather_share;
use crate::engine::partitioned::driver::scatter_local;
use crate::engine::partitioned::transport::{Collectives, Transport};
use crate::engine::partitioned::truncation::PartitionedTruncation;
#[cfg(feature = "phase-timing")]
use crate::engine::partitioned::PartitionPhaseStats;
use crate::engine::partitioned::{
    DistributedSum, PartitionConfig, PartitionRowPolicy, PartitionRuntime, PartitionTrace,
    Placement,
};
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::PauliSum;

#[cfg(feature = "mpi")]
use crate::engine::partitioned::mpi::{rsmpi::topology::Communicator, MpiTransport};

const LOG_TARGET: &str = "paulistrings::propagate";

/// Environment variables naming a process's rank among the processes on its node, read in this order.
///
/// `SLURM_LOCALID` comes last because a non-Slurm launcher inside an allocation (`mpirun` in a batch script) leaves every rank the batch step's stale value.
const LOCAL_RANK_VARS: [&str; 4] = [
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "SLURM_LOCALID",
];

/// The CUDA device this process should drive: its node-local rank from the launcher's environment, or `rank` when no launcher variable is set, modulo the visible device count.
///
/// Under `srun --gpus-per-task=1` each process sees one device, so the answer is `0` on every rank; under `mpirun` on a node of `k` devices, local rank `i` gets device `i % k`.
///
/// # Errors
///
/// [`GpuError::NoDevice`] if no device is visible (including a build or host without CUDA libraries).
pub fn local_device_for_rank(rank: u32) -> Result<u32, GpuError> {
    let count = device_count();
    if count == 0 {
        return Err(GpuError::NoDevice);
    }
    let local = LOCAL_RANK_VARS
        .iter()
        .find_map(|var| std::env::var(var).ok()?.trim().parse::<u32>().ok());
    Ok(pick_device(local, rank, count))
}

fn pick_device(local: Option<u32>, rank: u32, count: usize) -> u32 {
    (local.unwrap_or(rank) as usize % count) as u32
}

/// The first failing rank of the group and the layer it names, `None` if every rank passed `None`. **Collective**: one all-reduce of `size` words.
fn first_failure(coll: &dyn Collectives, failed_at: Option<usize>) -> Option<(usize, usize)> {
    let mut buf = vec![0u64; coll.size() as usize];
    if let Some(layer) = failed_at {
        buf[coll.rank() as usize] = layer as u64 + 1;
    }
    coll.allreduce_sum_u64(&mut buf);
    buf.iter()
        .position(|&v| v != 0)
        .map(|rank| (rank, buf[rank] as usize - 1))
}

/// `r` agreed over the group: this rank's own error, [`GpuError::Poisoned`] naming the first failing peer, or `r`'s value when every rank succeeded. **Collective.**
fn agree<T>(coll: &dyn Collectives, r: Result<T, GpuError>, layer: usize) -> Result<T, GpuError> {
    let failure = first_failure(coll, r.is_err().then_some(layer));
    match (r, failure) {
        (Err(e), _) => Err(e),
        (Ok(v), None) => Ok(v),
        (Ok(_), Some((rank, layer))) => Err(GpuError::Poisoned { rank, layer }),
    }
}

/// One process's partition of a sum split across a [`Transport`]'s group, held on one CUDA device: [`DistributedSum`] with the device backend of [`GpuPartitionedSum`](super::GpuPartitionedSum).
///
/// The contract is [`DistributedSum`]'s — replicated input, `D = 1`, rank 0 gathers, a per-rank trace — and every method named as collective there is collective here.
/// The exchange mode is agreed over the group at scatter ([`Self::exchange`]): built with `nccl`, a group of more than one rank on distinct devices that can all start NCCL moves its exchange columns device to device over NCCL unless `PAULISTRINGS_GPU_EXCHANGE=host`, and `PAULISTRINGS_GPU_EXCHANGE=nccl` on any rank makes the host fallback an error on every rank.
/// Device failures are agreed over the group: a scatter, propagate or gather that fails on any rank fails on every rank, with that rank's own error on the failing one and [`GpuError::Poisoned`] naming it on its peers, so the group never falls out of step.
/// After a failed `propagate` the split is poisoned, as a [`GpuPartitionedSum`](super::GpuPartitionedSum) is, until the caller scatters again.
///
/// `MpiGpuSum` (feature `mpi`) is this type over MPI, one GPU per rank; the in-process transport drives it for testing.
pub struct GpuDistributedSum<const W: usize, X: Transport> {
    inner: DistributedSum<W, X, DevicePartition<W>>,
    /// `(rank, layer)` of the group's first failure; set on every rank at once.
    poison: Option<(usize, usize)>,
}

/// [`GpuDistributedSum`] over MPI: one CUDA device per rank.
///
/// The universe is the application's, as for [`MpiSum`](crate::engine::partitioned::mpi::MpiSum); pick each rank's device with [`local_device_for_rank`].
///
/// ```no_run
/// # use paulistrings::engine::partitioned::PartitionRowPolicy;
/// # use paulistrings::gpu::{local_device_for_rank, MpiGpuSum};
/// # use paulistrings::mpi::{rsmpi, MpiTransport};
/// # use paulistrings::engine::partitioned::Collectives;
/// # use paulistrings::{Circuit, Direction, PauliSum};
/// # use paulistrings::truncation::ApproxTopN;
/// # fn go(circuit: &Circuit<1>, sum: PauliSum<1>) -> Result<(), paulistrings::gpu::GpuError> {
/// let (universe, _) =
///     rsmpi::initialize_with_threading(rsmpi::Threading::Serialized).expect("MPI initializes");
/// let transport = MpiTransport::from_communicator(&universe.world());
/// let device = local_device_for_rank(transport.rank())?;
/// let mut split: MpiGpuSum<1> =
///     MpiGpuSum::scatter(sum, transport, device, &PartitionRowPolicy::Seeded(None))?;
/// for _ in 0..10 {
///     split.propagate(circuit, &ApproxTopN(1_000_000), Direction::Heisenberg)?;
/// }
/// if let Some(out) = split.gather()? {
///     println!("{} terms", out.len());
/// }
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "mpi")]
pub type MpiGpuSum<const W: usize> = GpuDistributedSum<W, MpiTransport>;

impl<const W: usize, X: Transport> GpuDistributedSum<W, X> {
    /// Split the replicated `sum` across `transport`'s group by the rows `policy` names, and upload this rank's share to `device`. **Collective.**
    ///
    /// # Errors
    ///
    /// Any rank's placement or upload error, agreed over the group as the type docs describe.
    ///
    /// # Panics
    ///
    /// If the group size is not a power of two, and as [`PartitionRows::cut`] for a [`Cut`](PartitionRowPolicy::Cut) policy.
    pub fn scatter(
        sum: PauliSum<W>,
        transport: X,
        device: u32,
        policy: &PartitionRowPolicy,
    ) -> Result<Self, GpuError> {
        let size = transport.size();
        assert!(
            size.is_power_of_two(),
            "a group of {size} ranks cannot be a partitioning: a partition is named by log2(P) \
             GF(2) rows, so the rank count must be a power of two",
        );
        let rows = match policy {
            PartitionRowPolicy::Seeded(seed) => PartitionRows::<W>::from_seed(
                sum.num_qubits(),
                size.trailing_zeros() as u8,
                seed.unwrap_or_else(|| sum.hash().seed()),
            ),
            PartitionRowPolicy::Cut(blocks) => PartitionRows::<W>::cut(sum.num_qubits(), blocks),
        };
        Self::scatter_with_rows(sum, transport, device, rows)
    }

    /// [`scatter`](Self::scatter) with caller-supplied rows, which every rank must pass identically. **Collective.**
    ///
    /// # Errors
    ///
    /// As [`scatter`](Self::scatter).
    ///
    /// # Panics
    ///
    /// If `rows` does not name one partition per rank or is for a different qubit count than `sum`.
    pub fn scatter_with_rows(
        sum: PauliSum<W>,
        transport: X,
        device: u32,
        rows: PartitionRows<W>,
    ) -> Result<Self, GpuError> {
        let (rank, size) = (transport.rank(), transport.size());
        assert_eq!(
            rows.num_partitions(),
            size as usize,
            "partition rows name {} partitions but the group has {size} ranks",
            rows.num_partitions(),
        );
        assert_eq!(
            rows.num_qubits(),
            sum.num_qubits(),
            "partition rows are for {} qubits, the sum for {}",
            rows.num_qubits(),
            sum.num_qubits(),
        );
        debug_assert!(
            rows.is_independent_of(sum.hash()),
            "partition rows are dependent on the bucket hash rows — the split will correlate \
             with the bucket partition and load-balance badly",
        );
        let started = Instant::now();
        let config = PartitionConfig {
            placement: Placement::Devices {
                devices: vec![device],
                per_device: 1,
            },
            bind_memory: false,
            partition_row_seed: None,
        };
        let runtime = agree(
            &transport,
            PartitionRuntime::new(&config).map_err(GpuError::Topology),
            0,
        )?;
        let part = {
            let (sum, rows, transport) = (&sum, &rows, &transport);
            runtime.install(move || -> Result<DevicePartition<W>, GpuError> {
                let local = scatter_local(sum, rows, rank, transport);
                let mut part = DevicePartition::new(
                    GpuSum::from_host(&local, device)?,
                    GpuLayerOptions::default(),
                )?;
                part.group_size = size;
                Ok(part)
            })
        };
        let part = agree(&transport, part, 0)?;
        #[cfg(feature = "nccl")]
        let part = {
            let mut part = part;
            start_nccl(&transport, &mut part)?;
            part
        };
        log::info!(
            target: LOG_TARGET,
            "scatter_gpu: rank {rank}/{size} on device {device}, {} terms in, {} kept locally, {} bucket bits, {:.3} s",
            sum.len(),
            part.len(),
            part.hash().bits(),
            started.elapsed().as_secs_f64(),
        );
        let scatter_ns = started.elapsed().as_nanos() as u64;
        Ok(Self {
            inner: DistributedSum::from_backend(part, rows, runtime, transport, scatter_ns),
            poison: None,
        })
    }

    /// Propagate through `circuit` under `policy` with the default [`PropagateOptions`]. **Collective.**
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

    /// Propagate through `circuit` under `policy` with explicit [`PropagateOptions`]. **Collective**, with the same circuit, direction and options on every rank.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] on every rank before any layer if `policy` or a channel cannot run on device (as [`GpuPauliSum::propagate_with_options`](super::GpuPauliSum::propagate_with_options)), or if the policy contains an exact `TopN` — every rank here holds one partition of a distributed sum, and the `n`-th largest of that sum has no collective form.
    /// A device error on any rank is agreed after the loop, the partners having finished the call on empty exchange blocks, and poisons the split on every rank.
    ///
    /// # Panics
    ///
    /// As [`DistributedSum::propagate_with_options`], including the ranks disagreeing about the run's shape.
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
        let lowered = lower_for_run(
            circuit,
            policy,
            direction,
            self.inner.backend().hash(),
            false,
        )?;
        let part = self.inner.backend_mut();
        let stale = part.take_error();
        debug_assert!(stale.is_ok(), "a device error survived the previous call");
        part.policy = lowered;

        self.inner
            .propagate_on_backend(circuit, policy, direction, options);

        let part = self.inner.backend_mut();
        let layer = part.failed_layer.take();
        let own = part.take_error();
        let failed = own.is_err().then(|| layer.unwrap_or(0));
        match first_failure(self.inner.transport(), failed) {
            None => Ok(()),
            Some((rank, layer)) => {
                // A poisoned split is never used again, so its wire goes now rather than waiting on a finalize at drop.
                #[cfg(feature = "nccl")]
                if let Some(wire) = &self.inner.backend().scratch().export.wire {
                    wire.abort();
                }
                self.poison = Some((rank, layer));
                own.and(Err(GpuError::Poisoned { rank, layer }))
            }
        }
    }

    fn check_poison(&self) -> Result<(), GpuError> {
        match self.poison {
            Some((rank, layer)) => Err(GpuError::Poisoned { rank, layer }),
            None => Ok(()),
        }
    }

    /// Fail this rank before the exchange of its `layer`-th layer of the next call (test hook).
    #[cfg(any(test, feature = "test-utils"))]
    pub fn inject_failure(&mut self, layer: usize) {
        self.inner.backend_mut().fail_at_layer = Some(layer);
    }

    /// Make the next NCCL exchange's receive growth on this rank fail as out of memory, after the skeletons and before the vote (test hook).
    #[cfg(all(feature = "nccl", any(test, feature = "test-utils")))]
    pub fn inject_recv_oom(&mut self) {
        self.inner
            .backend_mut()
            .scratch_mut()
            .export
            .fail_recv_growth = true;
    }

    /// Exchange over NCCL's protocol with `wire` standing in for the communicator, from the next layer on (test hook); every rank of the group must switch together, each with its own rank's wire.
    ///
    /// # Panics
    ///
    /// If `wire` is not for this rank of a group of this size.
    #[cfg(all(feature = "nccl", any(test, feature = "test-utils")))]
    pub fn use_loopback_wire(&mut self, wire: super::nccl::LoopbackWire) {
        use super::nccl::DeviceWire;
        assert_eq!(
            (wire.rank(), wire.size()),
            (self.rank(), self.size()),
            "a loopback wire for another rank"
        );
        let export = &mut self.inner.backend_mut().scratch_mut().export;
        export.mode = GpuExchange::Nccl;
        export.wire = Some(std::sync::Arc::new(wire));
    }

    /// How this rank's exchange blocks travel, as agreed over the group at scatter.
    pub fn exchange(&self) -> GpuExchange {
        self.inner.backend().scratch().export.mode
    }

    /// Download every rank's share and collect the whole sum on rank 0: `Ok(Some(sum))` there, `Ok(None)` elsewhere. **Collective.**
    ///
    /// # Errors
    ///
    /// [`GpuError::Poisoned`] after a failed `propagate`, otherwise any rank's download error, agreed over the group.
    pub fn gather(&self) -> Result<Option<PauliSum<W>>, GpuError> {
        self.check_poison()?;
        let started = Instant::now();
        let share = agree(self.inner.transport(), self.local_to_host(), 0)?;
        let out = gather_share(&share, self.inner.transport());
        #[cfg(feature = "phase-timing")]
        self.inner.lap_gather(started.elapsed().as_nanos() as u64);
        #[cfg(not(feature = "phase-timing"))]
        let _ = started;
        Ok(out)
    }

    /// This rank's share, downloaded: a valid [`PauliSum`] under the group's shared hash holding exactly the keys of partition [`rank`](Self::rank). Local.
    pub fn local_to_host(&self) -> Result<PauliSum<W>, GpuError> {
        self.inner.backend().sum().to_host()
    }

    /// Terms this rank holds. Local.
    pub fn len_local(&self) -> usize {
        self.inner.backend().len()
    }

    /// Terms in the whole sum. **Collective**: one all-reduce.
    pub fn len(&self) -> usize {
        let mut buf = [self.len_local() as u64];
        self.inner.transport().allreduce_sum_u64(&mut buf);
        buf[0] as usize
    }

    /// Whether the whole sum is empty. **Collective**, via [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bucket bits every rank shares between calls.
    pub fn bits(&self) -> u8 {
        self.inner.backend().hash().bits()
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.inner.backend().sum().num_qubits()
    }

    /// This rank's index in the group.
    pub fn rank(&self) -> u32 {
        self.inner.rank()
    }

    /// Ranks in the group.
    pub fn size(&self) -> u32 {
        self.inner.size()
    }

    /// The device this rank's share lives on.
    pub fn device(&self) -> u32 {
        self.inner.backend().sum().device()
    }

    /// This rank's endpoint, for a caller's own collectives.
    pub fn transport(&self) -> &X {
        self.inner.transport()
    }

    /// The rows deciding which rank a key belongs to.
    pub fn rows(&self) -> &PartitionRows<W> {
        self.inner.rows()
    }

    /// The device layer's knobs; takes effect from the next layer.
    ///
    /// Every rank must set the same options, since the bucket-count proposal reads them.
    pub fn set_layer_options(&mut self, options: GpuLayerOptions) {
        self.inner.backend_mut().scratch_mut().options = options;
    }

    /// Record CUDA-event timings per kernel family into [`Self::take_kernel_ms`]; costs a synchronization per kernel.
    pub fn set_kernel_timing(&mut self, on: bool) {
        self.inner.backend_mut().scratch_mut().time_kernels = on;
    }

    /// Kernel milliseconds accumulated since the last call, zeroing the counters.
    pub fn take_kernel_ms(&mut self) -> GpuKernelMs {
        std::mem::take(&mut self.inner.backend_mut().scratch_mut().kernel_ms)
    }

    /// The counters of this rank's most recent layer.
    pub fn last_layer_counters(&self) -> GpuLayerCounters {
        self.inner.backend().counters()
    }

    /// Start recording this rank's [`PartitionTrace`], as [`DistributedSum::enable_trace`].
    pub fn enable_trace(&mut self) {
        self.inner.enable_trace();
    }

    /// Drain this rank's records, as [`DistributedSum::take_trace`].
    pub fn take_trace(&mut self) -> Option<PartitionTrace> {
        self.inner.take_trace()
    }

    /// Drain this rank's phase counters in the in-process driver's shape, `per_partition` holding this rank's one entry (feature `phase-timing`).
    #[cfg(feature = "phase-timing")]
    pub fn take_stats(&mut self) -> PartitionPhaseStats {
        let per = std::mem::take(self.inner.backend_mut().stats());
        let (scatter_ns, gather_ns, layers) = self.inner.take_driver_laps();
        PartitionPhaseStats {
            per_partition: vec![per],
            scatter_ns,
            gather_ns,
            layers,
        }
    }
}

/// Agree the group's exchange mode and, for [`GpuExchange::Nccl`], bootstrap and warm up the communicator. **Collective.**
#[cfg(feature = "nccl")]
fn start_nccl<const W: usize>(
    coll: &dyn Collectives,
    part: &mut DevicePartition<W>,
) -> Result<(), GpuError> {
    use super::nccl::{self, ExchangeKnob, NcclComm, NcclWire};
    let knob =
        nccl::parse_exchange_knob(std::env::var("PAULISTRINGS_GPU_EXCHANGE").ok().as_deref());
    let ctx = part.sum().ctx.clone();
    let wants = coll.size() > 1 && knob != ExchangeKnob::Host;
    let device = if wants && super::device::nccl_available() {
        nccl::device_uuid(&ctx).ok()
    } else {
        None
    };
    let plan = nccl::agree_exchange(coll, knob, device.is_some(), device.unwrap_or_default())?;
    if !plan.nccl {
        return Ok(());
    }
    let stream = part.sum().stream.clone();
    let started = bootstrap(
        coll,
        plan.strict,
        || NcclComm::init(coll, &ctx),
        |comm| comm.warm_up(&stream),
        NcclComm::abort,
    )?;
    if let Some(comm) = started {
        let export = &mut part.scratch_mut().export;
        export.mode = GpuExchange::Nccl;
        export.wire = Some(std::sync::Arc::new(NcclWire::new(std::sync::Arc::new(
            comm,
        ))));
    }
    Ok(())
}

/// Start a communicator with `init`, warm it up with `warm`, and agree each step over the group. **Collective**: two `allreduce_sum_u64`.
/// A communicator whose own warm-up failed or whose peers failed is aborted with `abort` rather than left to finalize against dead peers; on any failure a `strict` group (agreed, never one rank's knob) errs on every rank and any other group returns `None` on every rank.
#[cfg(feature = "nccl")]
fn bootstrap<C>(
    coll: &dyn Collectives,
    strict: bool,
    init: impl FnOnce() -> Result<C, GpuError>,
    warm: impl FnOnce(&C) -> Result<(), GpuError>,
    abort: impl Fn(&C),
) -> Result<Option<C>, GpuError> {
    let agreed = |r: Result<C, GpuError>| -> Result<C, GpuError> {
        let failure = first_failure(coll, r.is_err().then_some(0));
        match (r, failure) {
            (Err(e), _) => Err(e),
            (Ok(c), None) => Ok(c),
            (Ok(c), Some((rank, layer))) => {
                abort(&c);
                Err(GpuError::Poisoned { rank, layer })
            }
        }
    };
    let started = agreed(init()).and_then(|c| {
        let warmed = match warm(&c) {
            Ok(()) => Ok(c),
            Err(e) => {
                abort(&c);
                Err(e)
            }
        };
        agreed(warmed)
    });
    match started {
        Ok(c) => Ok(Some(c)),
        Err(e) if strict => Err(e),
        Err(e) => {
            log::warn!(
                target: LOG_TARGET,
                "scatter_gpu: rank {} of {}: NCCL did not start ({e}); exchanging through the host",
                coll.rank(),
                coll.size()
            );
            Ok(None)
        }
    }
}

/// Propagate `sum` through `circuit` with one CUDA device per rank of `comm`, and gather the result on rank 0.
///
/// **Collective**, with the replicated input and contract of [`propagate_mpi`](crate::engine::partitioned::mpi::propagate_mpi).
/// `device` is this rank's ordinal, or [`local_device_for_rank`] at `None`.
/// One-shot convenience over [`MpiGpuSum`]; a caller stepping repeatedly should hold the split instead.
///
/// # Errors
///
/// As [`GpuDistributedSum::scatter`], [`propagate`](GpuDistributedSum::propagate) and [`gather`](GpuDistributedSum::gather), plus [`GpuError::NoDevice`] from the device pick — every one agreed over the group.
#[cfg(feature = "mpi")]
pub fn propagate_mpi_gpu<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    options: PropagateOptions,
    comm: &impl Communicator,
    device: Option<u32>,
) -> Result<Option<PauliSum<W>>, GpuError>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    let transport = MpiTransport::from_communicator(comm);
    let picked = match device {
        Some(d) => Ok(d),
        None => local_device_for_rank(transport.rank()),
    };
    let device = agree(&transport, picked, 0)?;
    let mut split =
        MpiGpuSum::<W>::scatter(sum, transport, device, &PartitionRowPolicy::Seeded(None))?;
    split.propagate_with_options(circuit, policy, direction, options)?;
    split.gather()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::partitioned::InProcessTransport;
    use crate::test_support::{
        assert_terms_close, rand_sum, rand_sum_real, random_circuit, trotter_circuit, KeepAll,
    };
    use crate::truncation::ApproxTopN;

    #[test]
    fn the_device_pick_is_the_local_rank_modulo_the_devices() {
        assert_eq!(pick_device(Some(3), 7, 4), 3);
        assert_eq!(pick_device(Some(5), 0, 4), 1);
        assert_eq!(pick_device(None, 6, 4), 2);
        assert_eq!(pick_device(Some(3), 3, 1), 0);
    }

    /// `size` in-process ranks on device 0: each scatters, propagates and gathers; rank 0's result and every rank's outcome come back.
    fn run_group<const W: usize, T>(
        size: u32,
        circuit: &Circuit<W>,
        input: &PauliSum<W>,
        policy: &T,
        direction: Direction,
        fail: Option<(u32, usize)>,
    ) -> Vec<Result<Option<PauliSum<W>>, GpuError>>
    where
        T: PartitionedTruncation<W> + Sync,
    {
        let group =
            InProcessTransport::group_with_timeout(size, std::time::Duration::from_secs(120));
        std::thread::scope(|s| {
            let handles: Vec<_> = group
                .into_iter()
                .map(|transport| {
                    s.spawn(move || {
                        let rank = transport.rank();
                        let mut split = GpuDistributedSum::scatter(
                            input.clone(),
                            transport,
                            0,
                            &PartitionRowPolicy::Seeded(Some(0x5EED)),
                        )?;
                        if let Some((r, layer)) = fail {
                            if r == rank {
                                split.inject_failure(layer);
                            }
                        }
                        let ran = split.propagate(circuit, policy, direction);
                        let gathered = split.gather();
                        ran?;
                        gathered
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }

    fn differential<const W: usize, T>(
        circuit: &Circuit<W>,
        input: &PauliSum<W>,
        policy: &T,
        what: &str,
    ) where
        T: PartitionedTruncation<W> + Sync,
    {
        for direction in [Direction::Forward, Direction::Heisenberg] {
            let want = crate::propagate(circuit, input.clone(), policy, direction);
            for size in [1u32, 2, 4] {
                let mut out = run_group(size, circuit, input, policy, direction, None);
                let rest: Vec<_> = out.drain(1..).collect();
                let got = out.pop().unwrap().expect("rank 0").expect("rank 0 gathers");
                for r in rest {
                    assert!(r.expect("peer rank").is_none(), "only rank 0 gathers");
                }
                let what = format!("{what} ranks={size} {direction:?}");
                assert_eq!(got.len(), want.len(), "{what}: term count");
                assert_terms_close(&got, &want, 1e-11, &what);
            }
        }
    }

    #[test]
    fn device_ranks_match_propagate() {
        crate::require_cuda!();
        let dense = random_circuit::<1>(8, 12, 0xD15, true);
        differential(&dense, &rand_sum::<1>(400, 8, 0xD16), &KeepAll, "w1 dense");
        let trotter = trotter_circuit::<2>(24, 0.1);
        differential(
            &trotter,
            &rand_sum_real::<2>(900, 24, 0xD17),
            &ApproxTopN(1_200),
            "w2 trotter",
        );
    }

    /// A rank failing mid-run fails the call on every rank: its own error there, `Poisoned` naming it on the peers, and the gather refused everywhere.
    #[test]
    fn a_device_failure_on_one_rank_is_agreed_over_the_group() {
        crate::require_cuda!();
        let dense = random_circuit::<1>(8, 6, 0xD18, true);
        let input = rand_sum::<1>(300, 8, 0xD19);
        let out = run_group(
            2,
            &dense,
            &input,
            &KeepAll,
            Direction::Forward,
            Some((1, 2)),
        );
        assert!(
            matches!(out[0], Err(GpuError::Poisoned { rank: 1, layer: 2 })),
            "rank 0: {:?}",
            out[0].as_ref().map(|s| s.as_ref().map(PauliSum::len))
        );
        assert!(
            matches!(
                out[1],
                Err(GpuError::Unsupported("injected before the exchange"))
            ),
            "rank 1: {:?}",
            out[1].as_ref().map(|s| s.as_ref().map(PauliSum::len))
        );
    }

    #[cfg(feature = "nccl")]
    mod nccl {
        use std::sync::{Arc, Mutex};

        use super::super::super::nccl::{LoopbackTally, LoopbackWire};
        use super::*;
        use crate::engine::partitioned::transport::{ChunkMap, ChunkWait, Payload};
        use crate::engine::partitioned::{PartitionRuntime, TopologyError};
        use crate::test_support::{
            rows_reading_z63, unpinned_partitions, x0_terms_identity_on_q63, zz_rotation,
        };

        type Outcome<const W: usize> = Result<Option<PauliSum<W>>, GpuError>;

        /// Each `(transport, wire)` rank scatters `input` under `rows` onto device 0, switches to the NCCL protocol over its loopback wire if it has one, runs `setup`, propagates under `options` and gathers.
        #[allow(clippy::too_many_arguments)]
        fn run_split<const W: usize, T, X>(
            group: Vec<(X, Option<LoopbackWire>)>,
            input: &PauliSum<W>,
            rows: &PartitionRows<W>,
            circuit: &Circuit<W>,
            policy: &T,
            direction: Direction,
            options: PropagateOptions,
            setup: &(dyn Fn(&mut GpuDistributedSum<W, X>) + Sync),
        ) -> Vec<Outcome<W>>
        where
            T: PartitionedTruncation<W> + Sync,
            X: Transport + 'static,
        {
            std::thread::scope(|s| {
                let hs: Vec<_> = group
                    .into_iter()
                    .map(|(transport, wire)| {
                        s.spawn(move || {
                            let mut split = GpuDistributedSum::scatter_with_rows(
                                input.clone(),
                                transport,
                                0,
                                rows.clone(),
                            )?;
                            assert_eq!(
                                split.exchange(),
                                GpuExchange::Host,
                                "one device agrees Host"
                            );
                            if let Some(wire) = wire {
                                split.use_loopback_wire(wire);
                                assert_eq!(split.exchange(), GpuExchange::Nccl);
                            }
                            setup(&mut split);
                            let ran =
                                split.propagate_with_options(circuit, policy, direction, options);
                            let gathered = split.gather();
                            ran?;
                            gathered
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            })
        }

        fn loopback_group(size: u32) -> Vec<(InProcessTransport, Option<LoopbackWire>)> {
            loopback_tallied(size).0
        }

        /// As [`loopback_group`], with the tally of the groups its wires post.
        fn loopback_tallied(
            size: u32,
        ) -> (
            Vec<(InProcessTransport, Option<LoopbackWire>)>,
            LoopbackTally,
        ) {
            let wires = LoopbackWire::group(size);
            let tally = wires[0].tally();
            let group =
                InProcessTransport::group_with_timeout(size, std::time::Duration::from_secs(120))
                    .into_iter()
                    .zip(wires)
                    .map(|(t, w)| (t, Some(w)))
                    .collect();
            (group, tally)
        }

        fn seeded_rows<const W: usize>(nq: usize, size: u32) -> PartitionRows<W> {
            PartitionRows::<W>::from_seed(nq, size.trailing_zeros() as u8, 0x5EED)
        }

        fn nccl_differential<const W: usize, T>(
            circuit: &Circuit<W>,
            input: &PauliSum<W>,
            policy: &T,
            what: &str,
        ) where
            T: PartitionedTruncation<W> + Sync,
        {
            for direction in [Direction::Forward, Direction::Heisenberg] {
                let want = crate::propagate(circuit, input.clone(), policy, direction);
                for size in [2u32, 4] {
                    let rows = seeded_rows::<W>(input.num_qubits(), size);
                    let mut out = run_split(
                        loopback_group(size),
                        input,
                        &rows,
                        circuit,
                        policy,
                        direction,
                        PropagateOptions::default(),
                        &|_| {},
                    );
                    let rest: Vec<_> = out.drain(1..).collect();
                    let got = out.pop().unwrap().expect("rank 0").expect("rank 0 gathers");
                    for r in rest {
                        assert!(r.expect("peer rank").is_none(), "only rank 0 gathers");
                    }
                    let what = format!("{what} nccl ranks={size} {direction:?}");
                    assert_eq!(got.len(), want.len(), "{what}: term count");
                    assert_terms_close(&got, &want, 1e-11, &what);
                }
            }
        }

        /// The NCCL protocol over the loopback wire against `propagate`, the sender-side merge on and off.
        #[test]
        fn nccl_ranks_match_propagate() {
            crate::require_cuda!();
            let dense = random_circuit::<1>(8, 12, 0xD15, true);
            nccl_differential(&dense, &rand_sum::<1>(400, 8, 0xD16), &KeepAll, "w1 dense");
            let trotter = trotter_circuit::<2>(24, 0.1);
            nccl_differential(
                &trotter,
                &rand_sum_real::<2>(900, 24, 0xD17),
                &ApproxTopN(1_200),
                "w2 trotter",
            );
            let input = rand_sum::<1>(400, 8, 0xD18);
            let rows = seeded_rows::<1>(8, 2);
            let unmerged = |split: &mut GpuDistributedSum<1, InProcessTransport>| {
                split.set_layer_options(GpuLayerOptions {
                    premerge: false,
                    ..GpuLayerOptions::default()
                });
            };
            let want = crate::propagate(&dense, input.clone(), &KeepAll, Direction::Forward);
            let out = run_split(
                loopback_group(2),
                &input,
                &rows,
                &dense,
                &KeepAll,
                Direction::Forward,
                PropagateOptions::default(),
                &unmerged,
            );
            let got = out
                .into_iter()
                .next()
                .unwrap()
                .expect("rank 0")
                .expect("gathers");
            assert_terms_close(&got, &want, 1e-11, "w1 dense nccl, merge off");
        }

        /// `(rank, error)` per rank, for the failure nets.
        fn errors<const W: usize>(out: Vec<Outcome<W>>) -> Vec<GpuError> {
            out.into_iter()
                .enumerate()
                .map(|(r, o)| match o {
                    Err(e) => e,
                    Ok(s) => panic!("rank {r} succeeded with {:?} terms", s.map(|s| s.len())),
                })
                .collect()
        }

        /// A failure before the exchange takes the NCCL arm of the empty pairing: every rank fails, the culprit with its own error and every peer naming it.
        #[test]
        fn an_nccl_failure_before_the_exchange_is_agreed_over_the_group() {
            crate::require_cuda!();
            let dense = random_circuit::<1>(8, 6, 0xD18, true);
            let input = rand_sum::<1>(300, 8, 0xD19);
            for size in [2u32, 4] {
                let rows = seeded_rows::<1>(8, size);
                let fail = |split: &mut GpuDistributedSum<1, InProcessTransport>| {
                    if split.rank() == 1 {
                        split.inject_failure(2);
                    }
                };
                let out = run_split(
                    loopback_group(size),
                    &input,
                    &rows,
                    &dense,
                    &KeepAll,
                    Direction::Forward,
                    PropagateOptions::default(),
                    &fail,
                );
                for (r, e) in errors(out).iter().enumerate() {
                    if r == 1 {
                        assert!(
                            matches!(e, GpuError::Unsupported("injected before the exchange")),
                            "{e:?}"
                        );
                    } else {
                        assert!(
                            matches!(e, GpuError::Poisoned { rank: 1, layer: 2 }),
                            "rank {r}: {e:?}"
                        );
                    }
                }
            }
        }

        /// A rank whose receive growth fails votes no: nobody moves a row, it returns the out-of-memory error, and every peer names it at the same layer.
        #[test]
        fn a_receive_oom_votes_no_and_is_agreed_over_the_group() {
            crate::require_cuda!();
            let dense = random_circuit::<1>(8, 6, 0xD1A, true);
            let input = rand_sum::<1>(300, 8, 0xD1B);
            for size in [2u32, 4] {
                let rows = seeded_rows::<1>(8, size);
                let culprit = size - 1;
                let oom = move |split: &mut GpuDistributedSum<1, InProcessTransport>| {
                    if split.rank() == culprit {
                        split.inject_recv_oom();
                    }
                };
                let (group, tally) = loopback_tallied(size);
                let out = run_split(
                    group,
                    &input,
                    &rows,
                    &dense,
                    &KeepAll,
                    Direction::Forward,
                    PropagateOptions::default(),
                    &oom,
                );
                assert_eq!(
                    tally.groups(),
                    0,
                    "a no vote posts nothing, and the culprit votes no from then on"
                );
                let errs = errors(out);
                let GpuError::Poisoned { rank, layer } = errs[0] else {
                    panic!("rank 0: {:?}", errs[0]);
                };
                assert_eq!(rank, culprit as usize);
                for (r, e) in errs.iter().enumerate() {
                    if r == culprit as usize {
                        assert!(
                            matches!(e, GpuError::OutOfMemory { device: 0, .. }),
                            "{e:?}"
                        );
                    } else {
                        assert!(
                            matches!(e, GpuError::Poisoned { rank: q, layer: l } if *q == rank && *l == layer),
                            "rank {r}: {e:?}"
                        );
                    }
                }
            }
        }

        /// One source bucket of `n` rows crossing from rank 1 to rank 0 at one bucket: `MAX_BUCKET_LEN` rows fit, one more makes the receiver vote no.
        /// A device sender's own over-long bucket fails it after the exchange, so the receiver is rank 0, the culprit the agreement names first.
        fn one_segment(n: usize) -> (Vec<Outcome<1>>, u64) {
            let mut c = Circuit::<1>::new(64);
            c.push(zz_rotation::<1>(0, 63, 0.3));
            let mut acc = crate::accumulator::BuildAccumulator::<1>::new(64);
            for (x, z, coeff) in x0_terms_identity_on_q63(n, 0xF1 + n as u64).iter() {
                let p = crate::pauli_string::PauliString::<1> {
                    x: *x,
                    z: [z[0] | 1 << 63],
                };
                acc.add_term(p, crate::phase::Phase::ONE, coeff);
            }
            let input = acc
                .finalize()
                .with_hash(crate::bucket::hash::Gf2Hash::new(64, 0, 0xF0));
            let options = PropagateOptions {
                target_bucket_len: 1 << 20,
                min_buckets: 1,
                ..PropagateOptions::default()
            };
            let wide = |split: &mut GpuDistributedSum<1, InProcessTransport>| {
                split.set_layer_options(GpuLayerOptions {
                    bucket_policy: super::super::super::layer::GpuBucketPolicy::TermsPerBucket(
                        1 << 20,
                    ),
                    ..GpuLayerOptions::default()
                });
            };
            let (group, tally) = loopback_tallied(2);
            let out = run_split(
                group,
                &input,
                &rows_reading_z63(),
                &c,
                &KeepAll,
                Direction::Forward,
                options,
                &wide,
            );
            (out, tally.groups())
        }

        #[test]
        fn an_oversize_received_segment_votes_no_and_names_the_receiver() {
            crate::require_cuda!();
            use super::super::super::module::MAX_BUCKET_LEN;
            let (fits, groups) = one_segment(MAX_BUCKET_LEN);
            assert_eq!(groups, 2, "one group per rank");
            let got = fits
                .into_iter()
                .next()
                .unwrap()
                .expect("rank 0")
                .expect("gathers");
            assert_eq!(got.len(), 2 * MAX_BUCKET_LEN, "every term splits into two");
            let (over, groups) = one_segment(MAX_BUCKET_LEN + 1);
            assert_eq!(groups, 0, "the receiver voted no before any row moved");
            let errs = errors(over);
            assert!(
                matches!(errs[0], GpuError::Unsupported(m) if m.contains("received segment")),
                "{:?}",
                errs[0]
            );
            assert!(
                matches!(errs[1], GpuError::Unsupported(_)),
                "the sender's own bucket: {:?}",
                errs[1]
            );
        }

        /// A wire failing on one rank after a unanimous yes, in its post or its wait: every rank fails the call within the wire's bound, the culprit with the wire's error and its peers with their timed-out waits, and no later layer posts again.
        #[test]
        fn a_wire_failure_after_the_vote_fails_every_rank_and_none_hangs() {
            crate::require_cuda!();
            use super::super::super::nccl::LoopbackFault;
            let dense = random_circuit::<1>(8, 6, 0xD1C, true);
            let input = rand_sum::<1>(300, 8, 0xD1D);
            for size in [2u32, 4] {
                for fault in [LoopbackFault::Post, LoopbackFault::Wait] {
                    let culprit = size - 1;
                    let wires = LoopbackWire::group_with_fault(
                        size,
                        culprit,
                        fault,
                        std::time::Duration::from_secs(2),
                    );
                    let tally = wires[0].tally();
                    let group = InProcessTransport::group_with_timeout(
                        size,
                        std::time::Duration::from_secs(120),
                    )
                    .into_iter()
                    .zip(wires)
                    .map(|(t, w)| (t, Some(w)))
                    .collect();
                    let rows = seeded_rows::<1>(8, size);
                    let start = std::time::Instant::now();
                    let out = run_split(
                        group,
                        &input,
                        &rows,
                        &dense,
                        &KeepAll,
                        Direction::Forward,
                        PropagateOptions::default(),
                        &|_| {},
                    );
                    let elapsed = start.elapsed();
                    let what = format!("size {size} {fault:?}");
                    assert!(
                        elapsed < std::time::Duration::from_secs(60),
                        "{what}: {elapsed:?}"
                    );
                    let posted = if fault == LoopbackFault::Post {
                        size - 1
                    } else {
                        size
                    };
                    assert_eq!(
                        tally.groups(),
                        u64::from(posted),
                        "{what}: one failed group, then only no votes"
                    );
                    for (r, e) in errors(out).iter().enumerate() {
                        if r == culprit as usize {
                            assert!(
                                matches!(e, GpuError::Nccl { what, .. } if what.contains("injected")),
                                "{what}: {e:?}"
                            );
                        } else {
                            assert!(
                                matches!(e, GpuError::Timeout { .. }),
                                "{what}: rank {r}: {e:?}"
                            );
                        }
                    }
                }
            }
        }

        /// A rank whose wire is already dead votes no instead of failing after a yes: nothing is posted, it returns the wire's error and its peers name it.
        #[test]
        fn a_dead_wire_votes_no() {
            crate::require_cuda!();
            use super::super::super::nccl::DeviceWire;
            let dense = random_circuit::<1>(8, 6, 0xD1E, true);
            let input = rand_sum::<1>(300, 8, 0xD1F);
            let size = 2u32;
            let (group, tally) = loopback_tallied(size);
            group[1].1.as_ref().expect("a wire").abort();
            let rows = seeded_rows::<1>(8, size);
            let out = run_split(
                group,
                &input,
                &rows,
                &dense,
                &KeepAll,
                Direction::Forward,
                PropagateOptions::default(),
                &|_| {},
            );
            assert_eq!(tally.groups(), 0, "a no vote posts nothing");
            let errs = errors(out);
            assert!(
                matches!(&errs[1], GpuError::Nccl { what, .. } if what.contains("failed or aborted device wire")),
                "{:?}",
                errs[1]
            );
            assert!(
                matches!(errs[0], GpuError::Poisoned { rank: 1, .. }),
                "{:?}",
                errs[0]
            );
        }

        /// Mixed knobs with one rank's init or warm-up failing: every rank reaches the same outcome (an error everywhere when any rank is strict, the host exchange everywhere otherwise), and every communicator that was made is aborted.
        #[test]
        fn a_failed_start_is_one_outcome_on_every_rank_whatever_the_knobs() {
            use super::super::super::nccl::{agree_exchange, ExchangeKnob};
            use std::sync::atomic::{AtomicU32, Ordering};
            use ExchangeKnob::{Auto, Nccl};
            #[derive(Clone, Copy, Debug, PartialEq)]
            enum Fault {
                Init,
                WarmUp,
            }
            for size in [2u32, 4] {
                for strict_rank in [None, Some(0), Some(size - 1)] {
                    for (fault, culprit) in [
                        (Fault::Init, 1),
                        (Fault::WarmUp, 1),
                        (Fault::Init, 0),
                        (Fault::WarmUp, size - 1),
                    ] {
                        let aborted = AtomicU32::new(0);
                        let made = AtomicU32::new(0);
                        let out: Vec<Result<bool, String>> = std::thread::scope(|s| {
                            let hs: Vec<_> = InProcessTransport::group(size)
                                .into_iter()
                                .map(|t| {
                                    let (aborted, made) = (&aborted, &made);
                                    s.spawn(move || {
                                        let me = t.rank();
                                        let knob =
                                            if strict_rank == Some(me) { Nccl } else { Auto };
                                        let plan =
                                            agree_exchange(&t, knob, true, [u64::from(me) + 1, 0])
                                                .map_err(|e| e.to_string())?;
                                        assert!(plan.nccl);
                                        assert_eq!(
                                            plan.strict,
                                            strict_rank.is_some(),
                                            "strictness is agreed"
                                        );
                                        bootstrap(
                                            &t,
                                            plan.strict,
                                            || {
                                                if fault == Fault::Init && me == culprit {
                                                    return Err(GpuError::Unsupported(
                                                        "injected init failure",
                                                    ));
                                                }
                                                made.fetch_add(1, Ordering::Relaxed);
                                                Ok(me)
                                            },
                                            |_| {
                                                if fault == Fault::WarmUp && me == culprit {
                                                    return Err(GpuError::Unsupported(
                                                        "injected warm-up failure",
                                                    ));
                                                }
                                                Ok(())
                                            },
                                            |_| {
                                                aborted.fetch_add(1, Ordering::Relaxed);
                                            },
                                        )
                                        .map(|c| c.is_some())
                                        .map_err(|e| e.to_string())
                                    })
                                })
                                .collect();
                            hs.into_iter().map(|h| h.join().unwrap()).collect()
                        });
                        let what =
                            format!("size {size}, strict {strict_rank:?}, {fault:?} on {culprit}");
                        if strict_rank.is_some() {
                            assert!(out.iter().all(Result::is_err), "{what}: {out:?}");
                            let own = if fault == Fault::Init {
                                "injected init failure"
                            } else {
                                "injected warm-up failure"
                            };
                            assert!(
                                out[culprit as usize].as_ref().unwrap_err().contains(own),
                                "{what}: {out:?}"
                            );
                        } else {
                            assert!(
                                out.iter().all(|o| o == &Ok(false)),
                                "{what}: every rank falls back: {out:?}"
                            );
                        }
                        assert_eq!(
                            aborted.load(Ordering::Relaxed),
                            made.load(Ordering::Relaxed),
                            "{what}: every made communicator is aborted"
                        );
                    }
                }
            }
        }

        /// Every transport call a rank issues, in order.
        #[derive(Clone, Default)]
        struct Log(Arc<Mutex<Vec<&'static str>>>);

        impl Log {
            fn push(&self, call: &'static str) {
                self.0.lock().unwrap().push(call);
            }
            fn take(&self) -> Vec<&'static str> {
                std::mem::take(&mut *self.0.lock().unwrap())
            }
        }

        struct Counting {
            inner: InProcessTransport,
            log: Log,
        }

        impl Collectives for Counting {
            fn rank(&self) -> u32 {
                self.inner.rank()
            }
            fn size(&self) -> u32 {
                self.inner.size()
            }
            fn allreduce_max_u8(&self, v: u8) -> u8 {
                self.log.push("allreduce_max_u8");
                self.inner.allreduce_max_u8(v)
            }
            fn allreduce_sum_u64(&self, buf: &mut [u64]) {
                self.log.push("allreduce_sum_u64");
                self.inner.allreduce_sum_u64(buf);
            }
            fn barrier(&self) {
                self.log.push("barrier");
                self.inner.barrier();
            }
        }

        impl Transport for Counting {
            fn exchange_layer<P, F, R>(
                &self,
                send: Vec<Option<P>>,
                spare: &mut Vec<P>,
                map: &ChunkMap,
                body: F,
            ) -> (Vec<Option<P>>, R)
            where
                P: Payload,
                F: FnOnce(&[Option<P>], &dyn ChunkWait) -> R,
            {
                self.log.push("exchange_layer");
                self.inner.exchange_layer(send, spare, map, body)
            }
            fn exchange<P: Payload>(
                &self,
                send: Vec<Option<P>>,
                spare: &mut Vec<P>,
            ) -> Vec<Option<P>> {
                self.log.push("exchange");
                self.inner.exchange(send, spare)
            }
        }

        fn counting_group(size: u32) -> (Vec<Counting>, Vec<Log>) {
            let logs: Vec<Log> = (0..size).map(|_| Log::default()).collect();
            let group =
                InProcessTransport::group_with_timeout(size, std::time::Duration::from_secs(120))
                    .into_iter()
                    .zip(&logs)
                    .map(|(inner, log)| Counting {
                        inner,
                        log: log.clone(),
                    })
                    .collect();
            (group, logs)
        }

        /// The host `DistributedSum`'s propagate-time calls per rank.
        fn host_calls(
            size: u32,
            input: &PauliSum<1>,
            rows: &PartitionRows<1>,
            circuit: &Circuit<1>,
        ) -> Vec<Vec<&'static str>> {
            let (group, logs) = counting_group(size);
            std::thread::scope(|s| {
                let hs: Vec<_> = group
                    .into_iter()
                    .zip(&logs)
                    .map(|(t, log)| {
                        s.spawn(move || -> Result<Vec<&'static str>, TopologyError> {
                            let runtime = PartitionRuntime::new(&unpinned_partitions(1, 2, 0))?;
                            let mut split = DistributedSum::scatter_with_rows(
                                input.clone(),
                                t,
                                runtime,
                                rows.clone(),
                            );
                            log.take();
                            split.propagate_with_options(
                                circuit,
                                &ApproxTopN(900),
                                Direction::Forward,
                                PropagateOptions::default(),
                            );
                            Ok(log.take())
                        })
                    })
                    .collect();
                hs.into_iter()
                    .map(|h| h.join().unwrap().expect("topology"))
                    .collect()
            })
        }

        /// The device split's propagate-time calls per rank, over the NCCL protocol when `nccl`.
        fn device_calls(
            size: u32,
            input: &PauliSum<1>,
            rows: &PartitionRows<1>,
            circuit: &Circuit<1>,
            nccl: bool,
        ) -> Vec<Vec<&'static str>> {
            let (group, logs) = counting_group(size);
            let wires: Vec<Option<LoopbackWire>> = if nccl {
                LoopbackWire::group(size).into_iter().map(Some).collect()
            } else {
                (0..size).map(|_| None).collect()
            };
            let clear = |split: &mut GpuDistributedSum<1, Counting>| {
                logs[split.rank() as usize].take();
            };
            let out = run_split(
                group.into_iter().zip(wires).collect(),
                input,
                rows,
                circuit,
                &ApproxTopN(900),
                Direction::Forward,
                PropagateOptions::default(),
                &clear,
            );
            for o in out {
                o.expect("device rank");
            }
            logs.iter()
                .map(|l| {
                    let mut calls = l.take();
                    let gather = calls
                        .iter()
                        .rposition(|&c| c == "allreduce_sum_u64")
                        .expect("the gather agrees its download");
                    calls.truncate(gather);
                    calls
                })
                .collect()
        }

        /// Every rank issues the host's calls with each remote layer's `exchange_layer` replaced by the skeleton `exchange` and one `allreduce_sum_u64` vote, and nothing extra on a local layer; the host exchange issues the host's calls unchanged.
        /// Both device runs end on the propagate's own failure agreement, which the host driver has no counterpart of.
        #[test]
        fn every_remote_layer_adds_one_vote_and_a_local_layer_nothing() {
            crate::require_cuda!();
            let circuit = trotter_circuit::<1>(24, 0.1);
            let input = rand_sum_real::<1>(700, 24, 0xC0C0);
            for size in [2u32, 4] {
                let rows = seeded_rows::<1>(24, size);
                let host = host_calls(size, &input, &rows, &circuit);
                let staged = device_calls(size, &input, &rows, &circuit, false);
                let nccl = device_calls(size, &input, &rows, &circuit, true);
                for r in 0..size as usize {
                    let remote = host[r].iter().filter(|&&c| c == "exchange_layer").count();
                    assert!(
                        remote > 0 && remote < circuit.channels.len(),
                        "rank {r}: {remote} remote layers"
                    );
                    let mut want_staged = host[r].clone();
                    want_staged.push("allreduce_sum_u64");
                    assert_eq!(
                        staged[r], want_staged,
                        "rank {r} of {size}: the host exchange"
                    );
                    let mut want_nccl: Vec<&'static str> = host[r]
                        .iter()
                        .flat_map(|&c| match c {
                            "exchange_layer" => vec!["exchange", "allreduce_sum_u64"],
                            other => vec![other],
                        })
                        .collect();
                    want_nccl.push("allreduce_sum_u64");
                    assert_eq!(nccl[r], want_nccl, "rank {r} of {size}: the NCCL exchange");
                }
            }
        }
    }
}
