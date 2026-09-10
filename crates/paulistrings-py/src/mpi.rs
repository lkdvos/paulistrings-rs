//! The MPI surface of the bindings: `comm=` / `result=` on the two propagate
//! entry points. Compiled only with `--features mpi`.
//!
//! The library never calls `MPI_Init`. `mpi4py` owns initialization and
//! finalization, and this module only *adopts* a communicator the interpreter
//! already has: [`transport_from_comm`] reads the raw `MPI_Comm` out of an
//! mpi4py communicator and hands it to
//! [`MpiTransport::from_raw_handle`](paulistrings::mpi::MpiTransport::from_raw_handle),
//! which duplicates it. Two consequences shape everything here:
//!
//! - **The adoption is collective**, so it happens with the GIL held and
//!   *before* `allow_threads`, after every cheap check that could raise. A
//!   rank that raises before the duplicate is a rank that never entered a
//!   collective, so the group stays in step.
//! - **The duplicate must not outlive `MPI_Finalize`.** [`MpiRun`] owns the
//!   transport, moves it into the `DistributedSum` and drops both before it
//!   returns, so nothing MPI-shaped survives the call — a module-level global
//!   holding one would be finalized in the wrong order at interpreter
//!   teardown.
//!
//! The thread level is the other non-obvious requirement: the layer loop runs
//! inside `rayon::ThreadPool::install`, so MPI calls come off a pool worker
//! rather than the main thread. That is exactly `MPI_THREAD_SERIALIZED`, and
//! [`transport_from_comm`] refuses anything weaker.

use paulistrings::mpi::{default_config, MpiError, MpiSum, MpiTransport};
use paulistrings::{
    Circuit as CoreCircuit, Direction, PartitionTrace, PartitionedTruncation,
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

/// Adopt an mpi4py communicator as an [`MpiTransport`].
///
/// **Collective** over `comm`: every rank must reach this call, because
/// `MPI_Comm_dup` is collective. Every check that can raise happens before the
/// duplicate, so a rejected call rejects on all ranks alike.
///
/// `comm` is duck-typed — anything mpi4py's `_sizeof` / `_handleof` accept, so
/// `MPI.COMM_WORLD`, a `Split()`, a `Create_cart()` and a subclass all work.
/// The handle is duplicated, so ownership stays with Python.
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

    // The layer loop runs inside `rayon::ThreadPool::install`, so the thread
    // that calls MPI is a pool worker and need not be the same one on every
    // layer. Only ever one at a time, which is MPI_THREAD_SERIALIZED;
    // FUNNELED would be a false claim and is undefined behaviour here.
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

    // `_sizeof` is the guard against an ABI mismatch between the mpi4py in
    // this interpreter and the MPI this extension linked: `_handleof` returns
    // the handle as a Python int, and reading a 4-byte `int` handle as an
    // 8-byte pointer (or the reverse) would be silent corruption.
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

    // SAFETY: `handle` is the live `MPI_Comm` of an mpi4py communicator the
    // interpreter holds, of the width just checked, and mpi4py's
    // communicators are intra-communicators unless the caller built an
    // inter-communicator (which `scatter` would then reject on its size).
    // `from_raw_handle` duplicates it and never frees the original.
    unsafe { MpiTransport::from_raw_handle(handle) }.map_err(mpi_error)
}

/// One distributed propagate: the adopted transport plus what to hand back.
///
/// Constructed with the GIL held and consumed inside `allow_threads`, which is
/// what keeps the collective `MPI_Comm_dup` out of the GIL-released region and
/// the communicator's lifetime inside the call.
pub struct MpiRun {
    /// The duplicated communicator, moved into the `DistributedSum`.
    transport: MpiTransport,
    /// `true` for `result="gather"` (rank 0 gets the whole sum, everyone else
    /// an empty one), `false` for `result="local"` (each rank gets its share).
    gather: bool,
}

impl MpiRun {
    pub fn new(transport: MpiTransport, gather: bool) -> Self {
        Self { transport, gather }
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
        let Self { transport, gather } = self;
        let mut split = MpiSum::<W>::scatter(sum.clone(), transport, &default_config())?;
        split.propagate_with_options(circuit, policy, direction, options);
        let out = harvest(&split, gather);
        // Explicit, not incidental: the duplicated communicator is freed here,
        // inside the call, long before the interpreter finalizes MPI.
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
        let Self { transport, gather } = self;
        let mut split = MpiSum::<W>::scatter(sum.clone(), transport, &default_config())?;
        split.enable_trace();
        split.propagate_with_options(circuit, policy, direction, options);
        let (rank, size) = (split.rank(), split.size());
        // `enable_trace` was called before the layer loop, so a `None` here
        // would be a core bug; an empty trace is the honest fallback either
        // way (a zero-layer circuit records nothing).
        let trace = split.take_trace().unwrap_or_default();
        let out = harvest(&split, gather);
        drop(split);
        Ok((out, trace, rank, size))
    }
}

/// What this rank returns: the gathered sum on rank 0 (an empty sum of the
/// same width and qubit count everywhere else, so downstream code still gets a
/// `PauliSum`), or its own partition.
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
