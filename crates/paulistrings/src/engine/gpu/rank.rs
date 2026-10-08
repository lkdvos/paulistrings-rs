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
///
/// `MpiGpuSum` (feature `mpi`) is this type over MPI, one GPU per rank; the in-process transport drives it for testing.
pub type GpuDistributedSum<const W: usize, X> = DistributedSum<W, X, DevicePartition<W>>;

/// [`GpuDistributedSum`] over MPI: one CUDA device per rank.
///
/// The universe is the application's, as for [`MpiSum`](crate::engine::partitioned::mpi::MpiSum); pick each rank's device with [`local_device_for_comm`].
///
/// ```no_run
/// # use paulistrings::engine::partitioned::PartitionRowPolicy;
/// # use paulistrings::gpu::{local_device_for_comm, MpiGpuSum};
/// # use paulistrings::mpi::{rsmpi, MpiTransport};
/// # use paulistrings::{Circuit, Direction, PauliSum};
/// # use paulistrings::truncation::ApproxTopN;
/// # fn go(circuit: &Circuit<1>, sum: PauliSum<1>) -> Result<(), paulistrings::gpu::GpuError> {
/// let (universe, _) =
///     rsmpi::initialize_with_threading(rsmpi::Threading::Serialized).expect("MPI initializes");
/// let world = universe.world();
/// let device = local_device_for_comm(&world)?;
/// let transport = MpiTransport::from_communicator(&world);
/// let mut split: MpiGpuSum<1> =
///     MpiGpuSum::scatter_to_device(&sum, transport, device, &PartitionRowPolicy::Seeded(None))?;
/// for _ in 0..10 {
///     split.propagate(circuit, ApproxTopN(1_000_000), Direction::Heisenberg)?;
/// }
/// if let Some(out) = split.gather()? {
///     println!("{} terms", out.len());
/// }
/// # Ok(())
/// # }
/// ```
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

    /// [`scatter_to_device_with_rows`](Self::scatter_to_device_with_rows) exchanging over `wire`, this rank's of an in-process group, instead of NCCL (test hook).
    #[cfg(test)]
    pub(crate) fn scatter_with_wire(
        sum: &PauliSum<W>,
        transport: X,
        device: u32,
        rows: PartitionRows<W>,
        wire: super::wire::PeerWire,
    ) -> Result<Self, GpuError> {
        use super::wire::DeviceWire;
        assert_eq!(
            (wire.rank(), wire.size()),
            (transport.rank(), transport.size())
        );
        Self::scatter_then(sum, transport, device, rows, move |_, part| {
            part.scratch_mut().export.wire = Some(std::sync::Arc::new(wire));
            Ok(())
        })
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
        self.propagate_with_options(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagate through `circuit` under `policy` with explicit [`PropagateOptions`]. **Collective**, with the same circuit, direction and options on every rank.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] on every rank before any layer if `policy` or a channel cannot run on device (as [`GpuPartitionedSum::propagate_with_options`](super::GpuPartitionedSum::propagate_with_options)), or if the policy contains an exact `TopN` and the group has more than one rank.
    /// A device error on any rank is agreed after the loop, the partners having finished the call on empty exchange blocks, and poisons a group of more than one rank on every rank.
    ///
    /// # Panics
    ///
    /// As [`DistributedSum::propagate_with_options`], including the ranks disagreeing about the run's shape.
    pub fn propagate_with_options(
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

/// Start a communicator with `init`, warm it up with `warm`, and agree each step over the group. **Collective**: two `allreduce_sum_u64`.
/// A communicator whose own warm-up failed or whose peers failed is aborted with `abort` rather than left to finalize against dead peers, and every rank errs.
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
    split.propagate_with_options(circuit, policy, direction, options)?;
    split.gather()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::partitioned::InProcessTransport;
    use crate::test_support::rand_sum;

    /// A group of more than one rank without a way to start NCCL (no `mpi` feature, or every rank on device 0) fails the scatter on every rank.
    #[test]
    fn a_group_that_cannot_start_nccl_fails_the_scatter_on_every_rank() {
        crate::require_cuda!();
        let input = rand_sum::<1>(100, 8, 0xD19);
        for size in [2u32, 4] {
            let group =
                InProcessTransport::group_with_timeout(size, std::time::Duration::from_secs(120));
            let out: Vec<Result<(), GpuError>> = std::thread::scope(|s| {
                let hs: Vec<_> = group
                    .into_iter()
                    .map(|t| {
                        let input = &input;
                        s.spawn(move || {
                            GpuDistributedSum::scatter_to_device(
                                input,
                                t,
                                0,
                                &PartitionRowPolicy::Seeded(Some(0x5EED)),
                            )
                            .map(|_| ())
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for (r, o) in out.iter().enumerate() {
                assert!(
                    matches!(o, Err(GpuError::Unsupported(_))),
                    "size {size} rank {r}: {o:?}"
                );
            }
        }
    }

    /// The exchange protocol over the in-process [`PeerWire`](super::super::wire::PeerWire).
    mod protocol {
        use std::time::{Duration, Instant};

        use super::super::super::layer::GpuBucketPolicy;
        use super::super::super::module::MAX_BUCKET_LEN;
        use super::super::super::wire::{DeviceWire, PeerFault, PeerTally, PeerWire};
        use super::*;
        use crate::engine::partitioned::PartitionRuntime;
        use crate::test_support::{
            assert_terms_close, rand_sum_real, random_circuit, rows_reading_z63, su4_chain,
            trotter_circuit, unpinned_partitions, x0_terms_identity_on_q63, zz_rotation, CallLog,
            LoggingTransport,
        };
        use crate::truncation::ApproxTopN;

        type Outcome<const W: usize> = Result<Option<PauliSum<W>>, GpuError>;

        /// `transports` each paired with its rank's wire, and the tally of the groups the wires post.
        fn over_wires<X>(
            transports: Vec<X>,
            wires: Vec<PeerWire>,
        ) -> (Vec<(X, PeerWire)>, PeerTally) {
            let tally = wires[0].tally();
            (transports.into_iter().zip(wires).collect(), tally)
        }

        /// A group of `size` in-process ranks over a healthy peer wire.
        fn wired(size: u32) -> (Vec<(InProcessTransport, PeerWire)>, PeerTally) {
            let transports = InProcessTransport::group_with_timeout(size, Duration::from_secs(120));
            over_wires(transports, PeerWire::group(size))
        }

        fn seeded_rows<const W: usize>(nq: usize, size: u32) -> PartitionRows<W> {
            PartitionRows::<W>::from_seed(nq, size.trailing_zeros() as u8, 0x5EED)
        }

        /// One device run of `circuit` over `input` split by `rows`: `KeepAll`, forward, default options unless a case says otherwise.
        struct Run<'a, const W: usize> {
            input: &'a PauliSum<W>,
            rows: PartitionRows<W>,
            circuit: &'a Circuit<W>,
            policy: BuiltinTruncation,
            direction: Direction,
            options: PropagateOptions,
        }

        impl<'a, const W: usize> Run<'a, W> {
            fn new(
                input: &'a PauliSum<W>,
                rows: PartitionRows<W>,
                circuit: &'a Circuit<W>,
            ) -> Self {
                Self {
                    input,
                    rows,
                    circuit,
                    policy: BuiltinTruncation::Keep,
                    direction: Direction::Forward,
                    options: PropagateOptions::default(),
                }
            }

            /// The host oracle.
            fn want(&self) -> PauliSum<W> {
                crate::propagate(
                    self.circuit,
                    self.input.clone(),
                    &self.policy,
                    self.direction,
                )
            }

            /// Every rank scatters onto device 0 over its wire, runs `setup`, propagates and gathers; its outcome and last layer's counters.
            fn on<X: Transport + 'static>(
                &self,
                group: Vec<(X, PeerWire)>,
                setup: &(dyn Fn(&mut GpuDistributedSum<W, X>) + Sync),
            ) -> Vec<(Outcome<W>, GpuLayerCounters)> {
                std::thread::scope(|s| {
                    let hs: Vec<_> = group
                        .into_iter()
                        .map(|(transport, wire)| {
                            s.spawn(move || {
                                let rows = self.rows.clone();
                                let mut split = match GpuDistributedSum::scatter_with_wire(
                                    self.input, transport, 0, rows, wire,
                                ) {
                                    Ok(split) => split,
                                    Err(e) => return (Err(e), GpuLayerCounters::default()),
                                };
                                setup(&mut split);
                                let ran = split.propagate_with_options(
                                    self.circuit,
                                    &self.policy,
                                    self.direction,
                                    self.options,
                                );
                                let counters = split.last_layer_counters();
                                (ran.and_then(|()| split.gather()), counters)
                            })
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                })
            }

            /// [`Self::on`] a fresh healthy group of `size`, rank 0's result checked against [`Self::want`].
            fn check(&self, size: u32, what: &str) {
                let got = rank0(self.on(wired(size).0, &|_| {}));
                let want = self.want();
                assert_eq!(got.len(), want.len(), "{what}: term count");
                assert_terms_close(&got, &want, 1e-11, what);
            }
        }

        /// Rank 0's gathered sum, every other rank having succeeded with nothing.
        fn rank0<const W: usize>(out: Vec<(Outcome<W>, GpuLayerCounters)>) -> PauliSum<W> {
            let mut out = out
                .into_iter()
                .map(|(o, _)| o.expect("every rank succeeds"));
            let got = out.next().unwrap().expect("rank 0 gathers");
            assert!(out.all(|o| o.is_none()), "only rank 0 gathers");
            got
        }

        /// Every rank's error, for the failure nets.
        fn errors<const W: usize>(out: Vec<(Outcome<W>, GpuLayerCounters)>) -> Vec<GpuError> {
            out.into_iter()
                .enumerate()
                .map(|(r, (o, _))| match o {
                    Err(e) => e,
                    Ok(s) => panic!("rank {r} succeeded with {:?} terms", s.map(|s| s.len())),
                })
                .collect()
        }

        /// The culprit ended on `own` and every peer on a `Poisoned` naming it.
        fn agreed(errs: &[GpuError], culprit: u32, own: impl Fn(&GpuError) -> bool) {
            for (r, e) in errs.iter().enumerate() {
                if r == culprit as usize {
                    assert!(own(e), "rank {r}: {e:?}");
                } else {
                    assert!(
                        matches!(e, GpuError::Poisoned { rank, .. } if *rank == culprit as usize),
                        "rank {r}: {e:?}"
                    );
                }
            }
        }

        /// The protocol over the peer wire against `propagate` at 1, 2 and 4 ranks in both directions, and with the sender-side merge off.
        #[test]
        fn ranks_match_propagate() {
            crate::require_cuda!();
            let dense = random_circuit::<1>(8, 12, 0xD15, true);
            let input = rand_sum::<1>(400, 8, 0xD16);
            let trotter = trotter_circuit::<2>(24, 0.1);
            let real = rand_sum_real::<2>(900, 24, 0xD17);
            for direction in [Direction::Forward, Direction::Heisenberg] {
                for size in [1u32, 2, 4] {
                    let what = format!("ranks={size} {direction:?}");
                    let run = Run {
                        direction,
                        ..Run::new(&input, seeded_rows(8, size), &dense)
                    };
                    run.check(size, &format!("w1 dense {what}"));
                    let run = Run {
                        direction,
                        policy: BuiltinTruncation::ApproxTopN(1_200),
                        ..Run::new(&real, seeded_rows(24, size), &trotter)
                    };
                    run.check(size, &format!("w2 trotter {what}"));
                }
            }
            let run = Run::new(&input, seeded_rows(8, 2), &dense);
            let unmerged = |split: &mut GpuDistributedSum<1, InProcessTransport>| {
                split.set_layer_options(GpuLayerOptions {
                    premerge: false,
                    ..GpuLayerOptions::default()
                });
            };
            let got = rank0(run.on(wired(2).0, &unmerged));
            assert_terms_close(&got, &run.want(), 1e-11, "w1 dense, merge off");
        }

        /// Many small buckets, so a remote layer has positions to cut, and a receive cap of `rows` rows.
        fn capped(rows: usize) -> GpuLayerOptions {
            GpuLayerOptions {
                bucket_policy: GpuBucketPolicy::TermsPerBucket(8),
                exchange_bytes: rows
                    .saturating_mul(super::super::super::export::recv_row_bytes::<1>()),
                ..GpuLayerOptions::default()
            }
        }

        /// The receive cut into chunks over the peer wire agrees with `propagate`: one group per rank per chunk, so an uncapped run posts one per remote layer and caps of 128 and 8 rows at least 3 and 8 on some layer.
        /// `W = 2` runs the same protocol in `propagate_gpu_partitioned`'s chunked net.
        #[test]
        fn a_capped_receive_moves_in_chunk_groups_and_agrees() {
            crate::require_cuda!();
            let input = rand_sum::<1>(1_500, 10, 0xC4B1);
            let circuit = su4_chain::<1>(10);
            for direction in [Direction::Forward, Direction::Heisenberg] {
                for size in [2u32, 4] {
                    let run = Run {
                        direction,
                        ..Run::new(&input, seeded_rows(10, size), &circuit)
                    };
                    let want = run.want();
                    let mut plain = 0u64;
                    for (cap, least) in [(usize::MAX, 1u64), (128, 3), (8, 8)] {
                        let what = format!("ranks={size} {direction:?} cap={cap}");
                        let (group, tally) = wired(size);
                        let options = capped(cap);
                        let got = rank0(run.on(group, &|split| split.set_layer_options(options)));
                        assert_eq!(got.len(), want.len(), "{what}: term count");
                        assert_terms_close(&got, &want, 1e-11, &what);
                        let groups = tally.groups();
                        assert_eq!(
                            groups % u64::from(size),
                            0,
                            "{what}: every rank posts every group"
                        );
                        if cap == usize::MAX {
                            plain = groups;
                            assert!(plain > 0, "{what}: the circuit must cross");
                        } else {
                            assert!(
                                groups >= plain + (least - 1) * u64::from(size),
                                "{what}: {groups} groups against {plain} uncapped"
                            );
                        }
                    }
                }
            }
        }

        /// Ranks that disagree on their own receive cap still agree on the chunk count: the group takes the largest any rank asked for.
        #[test]
        fn a_capped_receive_agrees_the_largest_of_different_per_rank_caps() {
            crate::require_cuda!();
            let circuit = su4_chain::<1>(10);
            let input = rand_sum::<1>(1_500, 10, 0xC4B5);
            for (size, caps) in [
                (2u32, vec![usize::MAX, 8]),
                (4, vec![usize::MAX, 128, 32, 8]),
            ] {
                let run = Run::new(&input, seeded_rows(10, size), &circuit);
                let setup = |split: &mut GpuDistributedSum<1, InProcessTransport>| {
                    split.set_layer_options(capped(caps[split.rank() as usize]));
                };
                let out = run.on(wired(size).0, &setup);
                let chunks: Vec<u32> = out.iter().map(|(_, c)| c.recv_chunks).collect();
                let what = format!("ranks={size} caps={caps:?}");
                assert!(
                    chunks[0] > 1,
                    "{what}: the tightest cap forces chunks, got {chunks:?}"
                );
                assert!(chunks.iter().all(|&c| c == chunks[0]), "{what}: {chunks:?}");
                assert_terms_close(&rank0(out), &run.want(), 1e-11, &what);
            }
        }

        /// A rank failing after its second chunk moved still posts every later chunk's group, so its peers' receives complete: the culprit returns its error and every peer names it, with no wire timing out.
        /// The culprit's own `recv_chunks`, set before the chunk loop runs, proves the failure landed on a layer of several chunks.
        #[test]
        fn a_mid_receive_failure_drains_its_groups_and_is_agreed_over_the_group() {
            crate::require_cuda!();
            let input = rand_sum::<1>(1_500, 10, 0xC4B3);
            let circuit = su4_chain::<1>(10);
            for size in [2u32, 4] {
                let culprit = size - 1;
                let setup = move |split: &mut GpuDistributedSum<1, InProcessTransport>| {
                    split.set_layer_options(capped(8));
                    if split.rank() == culprit {
                        split.inject_chunk_oom(1);
                    }
                };
                let start = Instant::now();
                let out =
                    Run::new(&input, seeded_rows(10, size), &circuit).on(wired(size).0, &setup);
                assert!(start.elapsed() < Duration::from_secs(60));
                assert!(
                    out[culprit as usize].1.recv_chunks > 1,
                    "ranks={size}: {:?}",
                    out[culprit as usize].1
                );
                agreed(&errors(out), culprit, |e| {
                    matches!(e, GpuError::OutOfMemory { device: 0, .. })
                });
            }
        }

        /// A failure before the exchange takes the empty pairing: every rank fails, the culprit with its own error and every peer naming it at its layer.
        #[test]
        fn a_failure_before_the_exchange_is_agreed_over_the_group() {
            crate::require_cuda!();
            let dense = random_circuit::<1>(8, 6, 0xD18, true);
            let input = rand_sum::<1>(300, 8, 0xD19);
            for size in [2u32, 4] {
                let fail = |split: &mut GpuDistributedSum<1, InProcessTransport>| {
                    if split.rank() == 1 {
                        split.inject_failure(2);
                    }
                };
                let errs =
                    errors(Run::new(&input, seeded_rows(8, size), &dense).on(wired(size).0, &fail));
                agreed(&errs, 1, |e| {
                    matches!(e, GpuError::Unsupported("injected before the exchange"))
                });
                assert!(errs
                    .iter()
                    .enumerate()
                    .all(|(r, e)| r == 1 || matches!(e, GpuError::Poisoned { layer: 2, .. })));
            }
        }

        /// A rank whose receive growth fails votes no: nobody moves a row, it returns the out-of-memory error, and every peer names it.
        #[test]
        fn a_receive_oom_votes_no_and_is_agreed_over_the_group() {
            crate::require_cuda!();
            let dense = random_circuit::<1>(8, 6, 0xD1A, true);
            let input = rand_sum::<1>(300, 8, 0xD1B);
            for size in [2u32, 4] {
                let culprit = size - 1;
                let oom = move |split: &mut GpuDistributedSum<1, InProcessTransport>| {
                    if split.rank() == culprit {
                        split.inject_recv_oom();
                    }
                };
                let (group, tally) = wired(size);
                let errs = errors(Run::new(&input, seeded_rows(8, size), &dense).on(group, &oom));
                assert_eq!(
                    tally.groups(),
                    0,
                    "a no vote posts nothing, and the culprit votes no from then on"
                );
                agreed(&errs, culprit, |e| {
                    matches!(e, GpuError::OutOfMemory { device: 0, .. })
                });
            }
        }

        /// One source bucket of `n` rows crossing from rank 1 to rank 0 at one bucket: `MAX_BUCKET_LEN` rows fit, one more makes the receiver vote no.
        /// A device sender's own over-long bucket fails it after the exchange, so the receiver is rank 0, the culprit the agreement names first.
        #[test]
        fn an_oversize_received_segment_votes_no_and_names_the_receiver() {
            crate::require_cuda!();
            let mut c = Circuit::<1>::new(64);
            c.push(zz_rotation::<1>(0, 63, 0.3));
            let one_segment = |n: usize| {
                let mut acc = crate::pauli_sum::accumulator::BuildAccumulator::<1>::new(64);
                for (x, z, coeff) in x0_terms_identity_on_q63(n, 0xF1 + n as u64).iter() {
                    let p = crate::pauli_string::PauliString::<1> {
                        x: *x,
                        z: [z[0] | 1 << 63],
                    };
                    acc.add_term(p, crate::phase::Phase::ONE, coeff);
                }
                let input = acc
                    .finalize()
                    .with_hash(crate::pauli_sum::hash::Gf2Hash::new(64, 0, 0xF0));
                let run = Run {
                    options: PropagateOptions {
                        target_bucket_len: 1 << 20,
                        min_buckets: 1,
                        ..PropagateOptions::default()
                    },
                    ..Run::new(&input, rows_reading_z63(), &c)
                };
                let wide = |split: &mut GpuDistributedSum<1, InProcessTransport>| {
                    split.set_layer_options(GpuLayerOptions {
                        bucket_policy: GpuBucketPolicy::TermsPerBucket(1 << 20),
                        ..GpuLayerOptions::default()
                    });
                };
                let (group, tally) = wired(2);
                let out = run.on(group, &wide);
                (out, tally.groups())
            };
            let (fits, groups) = one_segment(MAX_BUCKET_LEN);
            assert_eq!(groups, 2, "one group per rank");
            assert_eq!(
                rank0(fits).len(),
                2 * MAX_BUCKET_LEN,
                "every term splits into two"
            );
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

        /// A wire failing on one rank after a unanimous yes, in its post or its wait: every rank fails within the wire's bound, the culprit with the wire's error and its peers with their timed-out waits, and no later layer posts again.
        #[test]
        fn a_wire_failure_after_the_vote_fails_every_rank_and_none_hangs() {
            crate::require_cuda!();
            let dense = random_circuit::<1>(8, 6, 0xD1C, true);
            let input = rand_sum::<1>(300, 8, 0xD1D);
            for size in [2u32, 4] {
                for fault in [PeerFault::Post, PeerFault::Wait] {
                    let culprit = size - 1;
                    let wires =
                        PeerWire::group_with_fault(size, culprit, fault, Duration::from_secs(2));
                    let transports =
                        InProcessTransport::group_with_timeout(size, Duration::from_secs(120));
                    let (group, tally) = over_wires(transports, wires);
                    let start = Instant::now();
                    let out = Run::new(&input, seeded_rows(8, size), &dense).on(group, &|_| {});
                    let what = format!("size {size} {fault:?}");
                    assert!(start.elapsed() < Duration::from_secs(60), "{what}");
                    let posted = if fault == PeerFault::Post {
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
                                matches!(e, GpuError::Wire(m) if m.contains("injected")),
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
            let dense = random_circuit::<1>(8, 6, 0xD1E, true);
            let input = rand_sum::<1>(300, 8, 0xD1F);
            let (group, tally) = wired(2);
            group[1].1.abort();
            let errs = errors(Run::new(&input, seeded_rows(8, 2), &dense).on(group, &|_| {}));
            assert_eq!(tally.groups(), 0, "a no vote posts nothing");
            agreed(
                &errs,
                1,
                |e| matches!(e, GpuError::Wire(m) if m.contains("failed or aborted device wire")),
            );
        }

        /// One rank's init or warm-up failing: every rank errs, the culprit with its own error, and every communicator that was made is aborted.
        #[cfg(feature = "mpi")]
        #[test]
        fn a_failed_start_errs_on_every_rank_and_aborts_every_communicator() {
            use std::sync::atomic::{AtomicU32, Ordering};
            for size in [2u32, 4] {
                for (warm_up, culprit) in [(false, 1), (true, 1), (false, 0), (true, size - 1)] {
                    let (aborted, made) = (AtomicU32::new(0), AtomicU32::new(0));
                    let own = if warm_up {
                        "injected warm-up failure"
                    } else {
                        "injected init failure"
                    };
                    let out: Vec<Result<u32, String>> = std::thread::scope(|s| {
                        let hs: Vec<_> = InProcessTransport::group(size)
                            .into_iter()
                            .map(|t| {
                                let (aborted, made) = (&aborted, &made);
                                s.spawn(move || {
                                    let me = t.rank();
                                    let fails = |at_warm_up| me == culprit && warm_up == at_warm_up;
                                    bootstrap(
                                        &t,
                                        || {
                                            if fails(false) {
                                                return Err(GpuError::Unsupported(own));
                                            }
                                            made.fetch_add(1, Ordering::Relaxed);
                                            Ok(me)
                                        },
                                        |_| {
                                            if fails(true) {
                                                Err(GpuError::Unsupported(own))
                                            } else {
                                                Ok(())
                                            }
                                        },
                                        |_| {
                                            aborted.fetch_add(1, Ordering::Relaxed);
                                        },
                                    )
                                    .map_err(|e| e.to_string())
                                })
                            })
                            .collect();
                        hs.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                    let what = format!("size {size}, warm-up {warm_up} on {culprit}");
                    assert!(out.iter().all(Result::is_err), "{what}: {out:?}");
                    assert!(
                        out[culprit as usize].as_ref().unwrap_err().contains(own),
                        "{what}: {out:?}"
                    );
                    assert_eq!(
                        aborted.load(Ordering::Relaxed),
                        made.load(Ordering::Relaxed),
                        "{what}: every made communicator is aborted"
                    );
                }
            }
        }

        /// Every rank issues the host's calls with each remote layer's `exchange_layer` replaced by the skeleton `exchange` and one `allreduce_sum_u64` vote, and nothing extra on a local layer.
        /// The device run ends on the propagate's own failure agreement, which the host driver has no counterpart of.
        #[test]
        fn every_remote_layer_adds_one_vote_and_a_local_layer_nothing() {
            crate::require_cuda!();
            let circuit = trotter_circuit::<1>(24, 0.1);
            let input = rand_sum_real::<1>(700, 24, 0xC0C0);
            for size in [2u32, 4] {
                let rows = seeded_rows::<1>(24, size);
                let group = LoggingTransport::group(size);
                let host: Vec<Vec<&'static str>> = std::thread::scope(|s| {
                    let hs: Vec<_> = group
                        .into_iter()
                        .map(|t| {
                            let (input, rows, circuit) = (&input, &rows, &circuit);
                            s.spawn(move || {
                                let runtime = PartitionRuntime::new(&unpinned_partitions(1, 2, 0))
                                    .expect("topology");
                                let mut split = DistributedSum::scatter_with_rows(
                                    input.clone(),
                                    t,
                                    runtime,
                                    rows.clone(),
                                );
                                split.transport().log.take();
                                split.propagate(circuit, &ApproxTopN(900), Direction::Forward);
                                split.transport().log.take()
                            })
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let run = Run {
                    policy: BuiltinTruncation::ApproxTopN(900),
                    ..Run::new(&input, rows.clone(), &circuit)
                };
                let group = LoggingTransport::group(size);
                let device_logs: Vec<CallLog> = group.iter().map(|t| t.log.clone()).collect();
                let (group, _) = over_wires(group, PeerWire::group(size));
                rank0(run.on(group, &|split| {
                    split.transport().log.take();
                }));
                for (r, log) in device_logs.iter().enumerate() {
                    let mut device = log.take();
                    let gather = device
                        .iter()
                        .rposition(|&c| c == "allreduce_sum_u64")
                        .expect("the gather agrees its download");
                    device.truncate(gather);
                    let remote = host[r].iter().filter(|&&c| c == "exchange_layer").count();
                    assert!(
                        remote > 0 && remote < circuit.channels.len(),
                        "rank {r}: {remote} remote layers"
                    );
                    let mut want: Vec<&'static str> = host[r]
                        .iter()
                        .flat_map(|&c| match c {
                            "exchange_layer" => vec!["exchange", "allreduce_sum_u64"],
                            other => vec![other],
                        })
                        .collect();
                    want.push("allreduce_sum_u64");
                    assert_eq!(device, want, "rank {r} of {size}: the device exchange");
                }
            }
        }
    }
}
