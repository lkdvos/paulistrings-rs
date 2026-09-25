//! [`GpuDistributedSum`], one device partition per process of a transport's group, with the MPI front door `MpiGpuSum` (feature `mpi`) and the device picker [`local_device_for_rank`]. See ARCHITECTURE.md §Partitioning.

use std::time::Instant;

use super::device::device_count;
use super::driver::lower_for_run;
use super::error::GpuError;
use super::layer::{GpuKernelMs, GpuLayerCounters, GpuLayerOptions};
use super::partition::DevicePartition;
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
}
