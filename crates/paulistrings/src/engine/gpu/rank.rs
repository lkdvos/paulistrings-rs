//! [`GpuDistributedSum`], [`DistributedSum`] over one device partition per process, with the MPI front door `MpiGpuSum` and the device pick `local_device_for_comm` (feature `mpi`). See ARCHITECTURE.md §Partitioning.

use std::time::Instant;

use super::driver::{device_runtime, first_failure, lower_for_run, upload_share};
use super::error::GpuError;
use super::layer::{GpuLayerCounters, GpuLayerOptions};
use super::partition::DevicePartition;
use crate::circuit::Circuit;
use crate::engine::partitioned::backend::PartitionStorage;
use crate::engine::partitioned::distributed::{gather_share, group_bits};
use crate::engine::partitioned::transport::{Collectives, Transport};
#[cfg(feature = "phase-timing")]
use crate::engine::partitioned::PartitionPhaseStats;
use crate::engine::partitioned::{DistributedSum, PartitionRowPolicy};
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::hash::PartitionRows;
use crate::pauli_sum::PauliSum;
use crate::truncation::BuiltinTruncation;

#[cfg(feature = "mpi")]
use crate::engine::partitioned::mpi::{rsmpi::topology::Communicator, MpiTransport};

const LOG_TARGET: &str = "paulistrings::propagate";

#[cfg(feature = "mpi")]
mod affinity;
#[cfg(feature = "mpi")]
pub use affinity::local_device_for_comm;

/// `r` agreed over the group: this rank's own error, [`GpuError::Poisoned`] naming the first failing peer (after handing this rank's value to `discard`), or `r`'s value when every rank succeeded. **Collective.**
fn agree_with<T>(
    coll: &dyn Collectives,
    r: Result<T, GpuError>,
    discard: impl FnOnce(T),
) -> Result<T, GpuError> {
    match (first_failure(coll, u64::from(r.is_err())), r) {
        (_, Err(e)) => Err(e),
        (None, Ok(v)) => Ok(v),
        (Some((rank, _)), Ok(v)) => {
            discard(v);
            Err(GpuError::Poisoned { rank, layer: 0 })
        }
    }
}

fn agree<T>(coll: &dyn Collectives, r: Result<T, GpuError>) -> Result<T, GpuError> {
    agree_with(coll, r, drop)
}

/// One process's partition of a sum split across a [`Transport`]'s group, held on one CUDA device: [`DistributedSum`] over the device backend of [`GpuPartitionedSum`](super::GpuPartitionedSum).
///
/// The contract is [`DistributedSum`]'s — replicated input, `D = 1`, rank 0 gathers, a per-rank trace — and every method named as collective there is collective here.
/// A group of more than one rank moves its exchange columns device to device over NCCL (features `cuda` and `mpi`), started at scatter: a rank that cannot load NCCL, or two ranks on one device, fail the scatter on every rank.
/// Device failures are agreed over the group: a scatter, propagate or gather that fails on any rank fails on every rank, with that rank's own error on the failing one and [`GpuError::Poisoned`] naming it on its peers, so the group never falls out of step.
/// After a failed `propagate` a group of more than one rank is poisoned, as a [`GpuPartitionedSum`](super::GpuPartitionedSum) is, until the caller scatters again.
pub type GpuDistributedSum<const W: usize, X> = DistributedSum<W, X, DevicePartition<W>>;

/// [`GpuDistributedSum`] over MPI: one CUDA device per rank.
///
/// The universe is the application's, as for [`MpiSum`](crate::engine::partitioned::mpi::MpiSum); pick each rank's device with [`local_device_for_comm`].
#[cfg(feature = "mpi")]
pub type MpiGpuSum<const W: usize> = GpuDistributedSum<W, MpiTransport>;

impl<const W: usize, X: Transport> DistributedSum<W, X, DevicePartition<W>> {
    /// Split the replicated `sum` across `transport`'s group by the rows `policy` names, and upload this rank's share to `device`. **Collective.**
    ///
    /// # Errors
    ///
    /// Any rank's placement or upload error, agreed over the group as the type docs describe.
    ///
    /// # Panics
    ///
    /// If the group size is not a power of two, and as [`PartitionRows::cut`] for a [`Cut`](PartitionRowPolicy::Cut) policy.
    pub fn scatter_to_device(
        sum: &PauliSum<W>,
        transport: X,
        device: u32,
        policy: &PartitionRowPolicy,
    ) -> Result<Self, GpuError> {
        let rows = policy.rows(
            sum.num_qubits(),
            group_bits(transport.size()),
            sum.hash().seed(),
        );
        Self::scatter_to_device_with_rows(sum, transport, device, rows)
    }

    /// [`scatter_to_device`](Self::scatter_to_device) with caller-supplied rows, which every rank must pass identically. **Collective.**
    ///
    /// # Errors
    ///
    /// As [`scatter_to_device`](Self::scatter_to_device).
    ///
    /// # Panics
    ///
    /// If `rows` does not name one partition per rank or is for a different qubit count than `sum`.
    pub fn scatter_to_device_with_rows(
        sum: &PauliSum<W>,
        transport: X,
        device: u32,
        rows: PartitionRows<W>,
    ) -> Result<Self, GpuError> {
        Self::scatter_then(sum, transport, device, rows, start_exchange)
    }

    fn scatter_then(
        sum: &PauliSum<W>,
        transport: X,
        device: u32,
        rows: PartitionRows<W>,
        start: impl FnOnce(&X, &mut DevicePartition<W>) -> Result<(), GpuError>,
    ) -> Result<Self, GpuError> {
        let (rank, size) = (transport.rank(), transport.size());
        rows.assert_splits(sum.hash(), sum.num_qubits(), size as usize);
        let started = Instant::now();
        let runtime = agree(&transport, device_runtime(device))?;
        let part = {
            let (rows, transport) = (&rows, &transport);
            runtime.install(move || upload_share(sum, rows, (rank, size), device, transport, &[]))
        };
        let mut part = agree(&transport, part)?;
        start(&transport, &mut part)?;
        log::info!(
            target: LOG_TARGET,
            "scatter_gpu: rank {rank}/{size} on device {device}, {} terms in, {} kept locally, {} bucket bits, {:.3} s",
            sum.len(),
            part.len(),
            part.hash().bits(),
            started.elapsed().as_secs_f64(),
        );
        let scatter_ns = started.elapsed().as_nanos() as u64;
        Ok(Self::from_backend(
            part, rows, runtime, transport, scatter_ns,
        ))
    }

    /// Propagate through `circuit` under `policy` with the default [`PropagateOptions`]. **Collective.**
    pub fn propagate(
        &mut self,
        circuit: &Circuit<W>,
        policy: impl Into<BuiltinTruncation>,
        direction: Direction,
    ) -> Result<(), GpuError> {
        self.propagate_with(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagate through `circuit` under `policy` with explicit [`PropagateOptions`]. **Collective**, with the same circuit, direction and options on every rank.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] on every rank before any layer if `policy` or a channel cannot run on device (as [`GpuPartitionedSum::propagate_with`](super::GpuPartitionedSum::propagate_with)), or if the policy contains an exact `TopN` and the group has more than one rank.
    /// A device error on any rank is agreed after the loop, the partners having finished the call on empty exchange blocks, and poisons a group of more than one rank on every rank.
    ///
    /// # Panics
    ///
    /// As [`DistributedSum::propagate_with`], including the ranks disagreeing about the run's shape.
    pub fn propagate_with(
        &mut self,
        circuit: &Circuit<W>,
        policy: impl Into<BuiltinTruncation>,
        direction: Direction,
        options: PropagateOptions,
    ) -> Result<(), GpuError> {
        let policy = policy.into();
        let single = self.size() == 1;
        let part = self.backend_mut();
        part.check_poison()?;
        let keep = lower_for_run(circuit, &policy, direction, part.hash(), single)?;
        let stale = part.take_error();
        debug_assert!(stale.is_ok(), "a device error survived the previous call");
        part.keep = keep;

        self.propagate_on_backend(circuit, &policy, direction, options);

        let (own, code) = self.backend_mut().take_failure();
        match first_failure(self.transport(), code) {
            None => Ok(()),
            Some((rank, code)) => {
                let layer = code as usize - 1;
                if !single {
                    self.backend_mut().poison(rank, layer);
                }
                own.and(Err(GpuError::Poisoned { rank, layer }))
            }
        }
    }

    /// Fail this rank before the exchange of its `layer`-th layer of the next call (test hook).
    #[cfg(any(test, feature = "test-utils"))]
    pub fn inject_failure(&mut self, layer: usize) {
        self.backend_mut().fail_at_layer = Some(layer);
    }

    /// Make the next exchange's receive growth on this rank fail as out of memory, after the skeletons and before the vote (test hook).
    #[cfg(any(test, feature = "test-utils"))]
    pub fn inject_recv_oom(&mut self) {
        self.backend_mut().scratch_mut().export.fail_recv_growth = true;
    }

    /// Make this rank's next chunked receive fail as out of memory once chunk `chunk` has moved, after the vote and mid-layer (test hook).
    #[cfg(any(test, feature = "test-utils"))]
    pub fn inject_chunk_oom(&mut self, chunk: usize) {
        self.backend_mut().scratch_mut().export.fail_after_chunk = Some(chunk);
    }

    /// Download every rank's share and collect the whole sum on rank 0: `Ok(Some(sum))` there, `Ok(None)` elsewhere. **Collective.**
    ///
    /// # Errors
    ///
    /// [`GpuError::Poisoned`] after a failed `propagate`, otherwise any rank's download error, agreed over the group.
    pub fn gather(&self) -> Result<Option<PauliSum<W>>, GpuError> {
        self.backend().check_poison()?;
        let started = Instant::now();
        let share = agree(self.transport(), self.local_to_host())?;
        let out = gather_share(&share, self.transport());
        #[cfg(feature = "phase-timing")]
        self.lap_gather(started.elapsed().as_nanos() as u64);
        #[cfg(not(feature = "phase-timing"))]
        let _ = started;
        Ok(out)
    }

    /// This rank's share, downloaded: a valid [`PauliSum`] under the group's shared hash holding exactly the keys of partition [`rank`](Self::rank). Local.
    pub fn local_to_host(&self) -> Result<PauliSum<W>, GpuError> {
        self.backend().sum().to_host()
    }

    /// The device this rank's share lives on.
    pub fn device(&self) -> u32 {
        self.backend().sum().device()
    }

    /// The device layer's knobs; takes effect from the next layer.
    ///
    /// Every rank must set the same options, since the bucket-count proposal reads them.
    pub fn set_layer_options(&mut self, options: GpuLayerOptions) {
        self.backend_mut().scratch_mut().options = options;
    }

    /// The counters of this rank's most recent layer.
    pub fn last_layer_counters(&self) -> GpuLayerCounters {
        self.backend().counters()
    }

    /// Drain this rank's phase counters in the in-process driver's shape, `per_partition` holding this rank's one entry (feature `phase-timing`).
    #[cfg(feature = "phase-timing")]
    pub fn take_stats(&mut self) -> PartitionPhaseStats {
        let per = std::mem::take(self.backend_mut().stats());
        let (scatter_ns, gather_ns, layers) = self.take_driver_laps();
        PartitionPhaseStats {
            per_partition: vec![per],
            scatter_ns,
            gather_ns,
            layers,
        }
    }
}

/// Nothing at one rank; above it, start NCCL for the group (feature `mpi`), which is the only exchange a device group over a transport has. **Collective.**
fn start_exchange<const W: usize, X: Transport>(
    transport: &X,
    part: &mut DevicePartition<W>,
) -> Result<(), GpuError> {
    if transport.size() == 1 {
        return Ok(());
    }
    #[cfg(feature = "mpi")]
    return start_nccl(transport, part);
    #[cfg(not(feature = "mpi"))]
    {
        let _ = part;
        Err(GpuError::Unsupported(
            "a device group of more than one rank exchanges over NCCL, which needs the mpi feature",
        ))
    }
}

/// Agree that the group can start NCCL, then bootstrap and warm up the communicator. **Collective.**
#[cfg(feature = "mpi")]
fn start_nccl<const W: usize>(
    coll: &dyn Collectives,
    part: &mut DevicePartition<W>,
) -> Result<(), GpuError> {
    use super::nccl::{self, NcclComm, NcclWire};
    let ctx = part.sum().ctx.clone();
    let device = if super::device::nccl_available() {
        nccl::device_uuid(&ctx).ok()
    } else {
        None
    };
    nccl::agree_start(coll, device.is_some(), device.unwrap_or_default())?;
    let stream = part.sum().stream.clone();
    let comm = bootstrap(
        coll,
        || NcclComm::init(coll, &ctx),
        |comm| comm.warm_up(&stream),
        NcclComm::abort,
    )?;
    part.scratch_mut().export.wire = Some(std::sync::Arc::new(NcclWire::new(std::sync::Arc::new(
        comm,
    ))));
    Ok(())
}

/// Start a communicator with `init` and warm it up with `warm`, each step agreed over the group (**collective**: two `allreduce_sum_u64`); on any failure every rank errs and its communicator is aborted, not finalized against dead peers.
#[cfg(feature = "mpi")]
fn bootstrap<C>(
    coll: &dyn Collectives,
    init: impl FnOnce() -> Result<C, GpuError>,
    warm: impl FnOnce(&C) -> Result<(), GpuError>,
    abort: impl Fn(&C),
) -> Result<C, GpuError> {
    agree_with(coll, init(), |c| abort(&c)).and_then(|c| {
        let warmed = match warm(&c) {
            Ok(()) => Ok(c),
            Err(e) => {
                abort(&c);
                Err(e)
            }
        };
        agree_with(coll, warmed, |c| abort(&c))
    })
}

/// Propagate `sum` through `circuit` with one CUDA device per rank of `comm`, and gather the result on rank 0.
///
/// **Collective**, with the replicated input and contract of [`propagate_mpi`](crate::engine::partitioned::mpi::propagate_mpi).
/// `device` is this rank's ordinal, or [`local_device_for_comm`] at `None`.
/// One-shot convenience over [`MpiGpuSum`]; a caller stepping repeatedly should hold the split instead.
///
/// # Errors
///
/// As [`GpuDistributedSum::scatter_to_device`], [`propagate`](GpuDistributedSum::propagate) and [`gather`](GpuDistributedSum::gather), plus [`GpuError::NoDevice`] from the device pick — every one agreed over the group.
#[cfg(feature = "mpi")]
pub fn propagate_mpi_gpu<const W: usize>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: impl Into<BuiltinTruncation>,
    direction: Direction,
    options: PropagateOptions,
    comm: &impl Communicator,
    device: Option<u32>,
) -> Result<Option<PauliSum<W>>, GpuError> {
    let transport = MpiTransport::from_communicator(comm);
    let picked = match device {
        Some(d) => Ok(d),
        None => local_device_for_comm(comm),
    };
    let device = agree(&transport, picked)?;
    let mut split = MpiGpuSum::<W>::scatter_to_device(
        sum,
        transport,
        device,
        &PartitionRowPolicy::Seeded(None),
    )?;
    split.propagate_with(circuit, policy, direction, options)?;
    split.gather()
}

#[cfg(test)]
mod tests;
