//! The MPI surface of the bindings: `comm=` / `result=` on the two propagate entry points. Compiled only with `--features mpi`.
//!
//! The library never calls `MPI_Init`; `mpi4py` owns init/finalize, and this module only adopts a communicator the interpreter already has via [`transport_from_comm`], which duplicates it.
//! The adoption is collective (happens with the GIL held, before `allow_threads`, after every check that could raise), and the duplicate must not outlive `MPI_Finalize` — [`MpiRun`] drops both the transport and the `DistributedSum` before returning.
//! The layer loop runs inside `rayon::ThreadPool::install`, so MPI calls come off a pool worker: this needs at least `MPI_THREAD_SERIALIZED`, which [`transport_from_comm`] enforces.

#[cfg(feature = "cuda")]
use paulistrings::engine::partitioned::Collectives;
use paulistrings::mpi::{default_config, MpiError, MpiSum, MpiTransport};
use paulistrings::{
    Circuit as CoreCircuit, Direction, PartitionRowPolicy, PartitionTrace, PartitionedTruncation,
    PauliSum as CorePauliSum, PropagateOptions, TopologyError,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// A core [`MpiError`] as a Python exception.
///
/// | variant | exception |
/// |---|---|
/// | `SizeNotPowerOfTwo` | `ValueError` — the caller launched the wrong rank count |
/// | `NullCommunicator` | `ValueError` — the caller passed `MPI.COMM_NULL` |
/// | `NotInitialized` | `RuntimeError` — MPI is not up (or already finalized) |
/// | `DuplicateFailed` | `RuntimeError` — `MPI_Comm_dup` reported an error |
fn mpi_error(err: MpiError) -> PyErr {
    match err {
        MpiError::SizeNotPowerOfTwo(_) | MpiError::NullCommunicator => {
            PyValueError::new_err(err.to_string())
        }
        MpiError::NotInitialized | MpiError::DuplicateFailed(_) => {
            PyRuntimeError::new_err(err.to_string())
        }
    }
}

/// Adopt an mpi4py communicator as an [`MpiTransport`]. Collective over `comm`, since `MPI_Comm_dup` is collective; every check that can raise happens before the duplicate, so a rejected call rejects on all ranks alike.
/// `comm` is duck-typed — anything mpi4py's `_sizeof`/`_handleof` accept. The handle is duplicated, so ownership stays with Python.
pub fn transport_from_comm(py: Python<'_>, comm: &Bound<'_, PyAny>) -> PyResult<MpiTransport> {
    let mpi = py.import_bound("mpi4py.MPI").map_err(|err| {
        PyRuntimeError::new_err(format!(
            "comm= needs mpi4py, which could not be imported: {err}"
        ))
    })?;

    if !mpi.call_method0("Is_initialized")?.extract::<bool>()? {
        return Err(PyRuntimeError::new_err(
            "MPI is not initialized: paulistrings never calls MPI_Init, so `from mpi4py import \
             MPI` (which does) must happen before comm= — and MPI must not already have been \
             finalized",
        ));
    }

    // The layer loop calls MPI from a pool worker, not necessarily the same one each layer: that is MPI_THREAD_SERIALIZED, not FUNNELED.
    let provided: i32 = mpi.call_method0("Query_thread")?.extract()?;
    let serialized: i32 = mpi.getattr("THREAD_SERIALIZED")?.extract()?;
    if provided < serialized {
        return Err(PyRuntimeError::new_err(format!(
            "MPI provides thread level {provided}, but a distributed propagate needs at least \
             MPI_THREAD_SERIALIZED ({serialized}): its layer loop calls MPI from a pinned Rayon \
             pool worker. Set the level before MPI is initialized:\n\
             \n    import mpi4py\n    mpi4py.rc.thread_level = \"serialized\"\n    from mpi4py \
             import MPI\n\n\
             (mpi4py's default is \"multiple\", which is also fine; a level below SERIALIZED \
             usually means the MPI build itself cannot provide more.)"
        )));
    }

    // `_sizeof` guards against an ABI mismatch between mpi4py's MPI and this extension's: reading a 4-byte handle as an 8-byte pointer (or the reverse) would be silent corruption.
    let want = std::mem::size_of::<paulistrings::mpi::rsmpi::ffi::MPI_Comm>();
    let got: usize = mpi
        .call_method1("_sizeof", (comm,))
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "comm= must be an mpi4py communicator (mpi4py.MPI.Comm), got {}",
                comm.get_type()
                    .name()
                    .map_or_else(|_| "an unknown type".to_string(), |name| name.to_string()),
            ))
        })?
        .extract()?;
    if got != want {
        return Err(PyRuntimeError::new_err(format!(
            "mpi4py's MPI_Comm is {got} bytes but this extension was built against an MPI whose \
             MPI_Comm is {want}: the two are different MPI libraries. Rebuild the extension \
             against the same MPI mpi4py uses (`maturin develop --release --features mpi` with \
             that MPI's mpicc on PATH)."
        )));
    }
    let handle: usize = mpi.call_method1("_handleof", (comm,))?.extract()?;

    // SAFETY: `handle` is the live `MPI_Comm` of an mpi4py communicator the interpreter holds, of the width just checked; `from_raw_handle` duplicates it and never frees the original.
    unsafe { MpiTransport::from_raw_handle(handle) }.map_err(mpi_error)
}

/// The group size of an mpi4py communicator.
///
/// `MPI_Comm_size` is **local**, so this is readable before the collective duplicate in [`transport_from_comm`] and gives every rank the same number — which is what lets `parse_run_mode` validate `partition_row_blocks=` against the rank count and still reject on every rank alike.
pub fn comm_size(comm: &Bound<'_, PyAny>) -> PyResult<usize> {
    let size: i64 = comm
        .call_method0("Get_size")
        .map_err(|err| {
            PyTypeError::new_err(format!(
                "comm= must be a live mpi4py communicator (mpi4py.MPI.Comm); Get_size() on {} \
                 failed: {err}",
                comm.get_type()
                    .name()
                    .map_or_else(|_| "an unknown type".to_string(), |name| name.to_string()),
            ))
        })?
        .extract()?;
    usize::try_from(size)
        .map_err(|_| PyValueError::new_err(format!("comm= reports a group size of {size}")))
}

/// One distributed propagate: the adopted transport, the rows to split by, and what to hand back. Constructed with the GIL held and consumed inside `allow_threads`, keeping the communicator's lifetime inside the call.
pub struct MpiRun {
    /// The duplicated communicator, moved into the `DistributedSum`.
    transport: MpiTransport,
    /// Which rows decide a term's rank. Resolved to a `PartitionRows<W>` inside the width dispatch, where `W` is known.
    rows: PartitionRowPolicy,
    /// `true` for `result="gather"`, `false` for `result="local"`.
    gather: bool,
}

impl MpiRun {
    pub fn new(transport: MpiTransport, rows: PartitionRowPolicy, gather: bool) -> Self {
        Self {
            transport,
            rows,
            gather,
        }
    }

    /// Scatter, propagate, and take this rank's answer. **Collective.**
    pub fn propagate<const W: usize, T>(
        self,
        circuit: &CoreCircuit<W>,
        sum: &CorePauliSum<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
    ) -> Result<CorePauliSum<W>, TopologyError>
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        let Self {
            transport,
            rows,
            gather,
        } = self;
        let mut split =
            MpiSum::<W>::scatter_with_policy(sum.clone(), transport, &default_config(), &rows)?;
        split.propagate_with_options(circuit, policy, direction, options);
        let out = harvest(&split, gather);
        // Explicit: frees the duplicated communicator here, before the interpreter finalizes MPI.
        drop(split);
        Ok(out)
    }

    /// [`propagate`](Self::propagate), also draining this rank's
    /// [`PartitionTrace`]. Returns `(sum, trace, rank, size)`.
    pub fn propagate_traced<const W: usize, T>(
        self,
        circuit: &CoreCircuit<W>,
        sum: &CorePauliSum<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
    ) -> Result<(CorePauliSum<W>, PartitionTrace, u32, u32), TopologyError>
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        let Self {
            transport,
            rows,
            gather,
        } = self;
        let mut split =
            MpiSum::<W>::scatter_with_policy(sum.clone(), transport, &default_config(), &rows)?;
        split.enable_trace();
        split.propagate_with_options(circuit, policy, direction, options);
        let (rank, size) = (split.rank(), split.size());
        // An empty trace is the honest fallback for a zero-layer circuit, which records nothing.
        let trace = split.take_trace().unwrap_or_default();
        let out = harvest(&split, gather);
        drop(split);
        Ok((out, trace, rank, size))
    }
}

/// What this rank returns: the gathered sum on rank 0 (an empty sum elsewhere), or its own partition.
fn harvest<const W: usize>(split: &MpiSum<W>, gather: bool) -> CorePauliSum<W> {
    if gather {
        // Collective; `Some` on rank 0 only.
        split
            .gather()
            .unwrap_or_else(|| CorePauliSum::<W>::empty(split.num_qubits()))
    } else {
        split.local().clone()
    }
}

/// The first rank of the group whose `code` is non-zero, with that code. **Collective**: one all-reduce of `size` words.
#[cfg(feature = "cuda")]
fn first_nonzero(coll: &dyn Collectives, code: u64) -> Option<(usize, u64)> {
    let mut buf = vec![0u64; coll.size() as usize];
    buf[coll.rank() as usize] = code;
    coll.allreduce_sum_u64(&mut buf);
    buf.iter()
        .position(|&v| v != 0)
        .map(|rank| (rank, buf[rank]))
}

/// This rank's device pick `local`, agreed over `transport`'s group, so a rank that cannot use its device fails the call on every rank. **Collective.**
/// The failing rank raises its own exception; its peers raise one of the same type naming it.
#[cfg(feature = "cuda")]
pub fn agree_device(
    py: Python<'_>,
    transport: &MpiTransport,
    local: PyResult<u32>,
    shown: &str,
) -> PyResult<u32> {
    let code = match &local {
        Ok(_) => 0,
        Err(err) if err.is_instance_of::<PyValueError>(py) => 1,
        Err(_) => 2,
    };
    match (local, first_nonzero(transport, code)) {
        (Err(err), _) => Err(err),
        (Ok(device), None) => Ok(device),
        (Ok(_), Some((rank, code))) => {
            let msg = format!(
                "{shown}: rank {rank} cannot use its CUDA device, so no rank runs (rank {rank} \
                 raises the reason)"
            );
            Err(if code == 1 {
                PyValueError::new_err(msg)
            } else {
                PyRuntimeError::new_err(msg)
            })
        }
    }
}

/// What can fail inside a one-GPU-per-rank run: a core error, already agreed over the group, or this rank's peer failing to download its `result="local"` share.
#[cfg(feature = "cuda")]
pub enum MpiGpuFailure {
    Gpu(paulistrings::gpu::GpuError),
    PeerDownload(usize),
}

/// One distributed propagate with one CUDA device per rank (`comm=` with `device=`), [`MpiRun`]'s twin over `MpiGpuSum`.
#[cfg(feature = "cuda")]
pub struct MpiGpuRun {
    transport: MpiTransport,
    rows: PartitionRowPolicy,
    gather: bool,
    device: u32,
}

#[cfg(feature = "cuda")]
impl MpiGpuRun {
    pub fn new(
        transport: MpiTransport,
        rows: PartitionRowPolicy,
        gather: bool,
        device: u32,
    ) -> Self {
        Self {
            transport,
            rows,
            gather,
            device,
        }
    }

    /// Scatter, propagate and take this rank's answer, plus this rank's `(trace, rank, size, device)` when `traced`. **Collective**; every error is agreed over the group.
    #[allow(clippy::type_complexity)]
    pub fn propagate<const W: usize, T>(
        self,
        circuit: &CoreCircuit<W>,
        sum: &CorePauliSum<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
        traced: bool,
    ) -> Result<(CorePauliSum<W>, Option<(PartitionTrace, u32, u32, u32)>), MpiGpuFailure>
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        use paulistrings::gpu::MpiGpuSum;
        let Self {
            transport,
            rows,
            gather,
            device,
        } = self;
        let mut split = MpiGpuSum::<W>::scatter(sum.clone(), transport, device, &rows)
            .map_err(MpiGpuFailure::Gpu)?;
        if traced {
            split.enable_trace();
        }
        split
            .propagate_with_options(circuit, policy, direction, options)
            .map_err(MpiGpuFailure::Gpu)?;
        let (rank, size) = (split.rank(), split.size());
        let trace = traced.then(|| (split.take_trace().unwrap_or_default(), rank, size, device));
        let out = if gather {
            split
                .gather()
                .map_err(MpiGpuFailure::Gpu)?
                .unwrap_or_else(|| CorePauliSum::<W>::empty(split.num_qubits()))
        } else {
            // A local download: agreed here so a rank whose download fails does not leave its peers a call ahead.
            let share = split.local_to_host();
            let failed = first_nonzero(split.transport(), u64::from(share.is_err()));
            match (share, failed) {
                (Err(err), _) => return Err(MpiGpuFailure::Gpu(err)),
                (Ok(_), Some((rank, _))) => return Err(MpiGpuFailure::PeerDownload(rank)),
                (Ok(share), None) => share,
            }
        };
        // Explicit: frees the duplicated communicator here, before the interpreter finalizes MPI.
        drop(split);
        Ok((out, trace))
    }
}
