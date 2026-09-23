//! Python bindings for `paulistrings`.
//!
//! See ARCHITECTURE.md §Python-Bindings. The user-visible Python surface is a
//! small set of classes (`PauliSum`, `Circuit`) plus three factory submodules
//! (`gates`, `noise`, `truncation`). The width parameter `W` is monomorphized
//! at a fixed set `{1, 2, 4, 8, 16}` and dispatched outside any hot loop
//! (ARCHITECTURE.md §Width).

// PyO3 0.22's `#[pymethods]` expansion converts return values via `.into()` even for a `PyResult<T>`, which clippy flags; the lint is on the macro output, not our code.
#![allow(clippy::useless_conversion)]

#[macro_use]
mod macros;

mod channel_spec;
mod circuit;
mod gates;
#[cfg(feature = "mpi")]
mod mpi;
mod noise;
mod pauli_string;
mod sum;
mod truncation;
mod truncation_spec;

use pyo3::prelude::*;
use std::sync::OnceLock;

/// Handle returned by `pyo3_log::try_init`, kept so `reset_log_cache` can clear pyo3-log's per-logger level cache. Set exactly once, at module import.
static LOG_RESET: OnceLock<pyo3_log::ResetHandle> = OnceLock::new();

/// Drop the cached Python log levels of the Rust->Python log bridge.
///
/// pyo3-log caches each logger's effective level. Call this after changing Python log levels mid-process (e.g. `logging.getLogger("paulistrings").setLevel(logging.DEBUG)`), otherwise the new level is not picked up.
/// A no-op if some other logger claimed the `log` facade before this module was imported.
#[pyfunction]
fn reset_log_cache() {
    if let Some(handle) = LOG_RESET.get() {
        handle.reset();
    }
}

/// The NUMA nodes this process may run on, as one list of CPU indices each, in ascending node order.
///
/// This is what `PauliSum.propagate(partitions="auto")` places against: `"auto"` takes one partition per entry (rounded down to a power of two), and `partitions=k` is refused unless there are at least `k` entries. Each list is intersected with the process's CPU affinity mask, so a cgroup- or `taskset`-confined run sees only what it may use.
/// A machine with no NUMA information reports the whole affinity mask as a single node. The lists are a snapshot: a later affinity change is not reflected until the next call.
#[pyfunction]
fn numa_nodes() -> Vec<Vec<usize>> {
    paulistrings::engine::partitioned::numa_nodes()
        .into_iter()
        .map(|(_id, cpus)| cpus.0)
        .collect()
}

/// Whether this build of the extension can run `PauliSum.propagate(comm=...)`.
///
/// `True` only if compiled with the `mpi` cargo feature (`maturin develop --release --features mpi`); the default wheel lacks it and a `comm=` there raises `RuntimeError`.
/// Importing `paulistrings` never imports `mpi4py` and never touches MPI, so this is safe to call anywhere:
///
/// ```python
/// if paulistrings.mpi_available():
///     from mpi4py import MPI
///     evolved = observable.propagate(circuit, comm=MPI.COMM_WORLD)
/// ```
#[pyfunction]
fn mpi_available() -> bool {
    cfg!(feature = "mpi")
}

/// Whether this build of the extension can run `PauliSum.propagate(device=...)`.
///
/// `True` only if compiled with the `cuda` cargo feature (`maturin develop --release --features cuda`)
/// *and* a CUDA device is visible to this process; the default wheel lacks the feature.
#[cfg(feature = "cuda")]
#[pyfunction]
fn cuda_available() -> bool {
    paulistrings::gpu::cuda_available()
}

/// Whether this build of the extension can run `PauliSum.propagate(device=...)`.
///
/// Always `False`: this build lacks the `cuda` cargo feature.
#[cfg(not(feature = "cuda"))]
#[pyfunction]
fn cuda_available() -> bool {
    false
}

/// Shorthand for `PauliString.from_label(label)`, for writing one down by hand.
///
/// ```python
/// from paulistrings import p
/// p("XYZ").weight   # 3
/// ```
#[pyfunction]
fn p(label: &str) -> PyResult<pauli_string::PauliString> {
    pauli_string::PauliString::parse(label)
}

#[pymodule]
fn _paulistrings(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Route the core crate's `log` records to Python's `logging`. Fails only if some other logger is already installed, which is not an import error.
    if let Ok(handle) = pyo3_log::try_init() {
        let _ = LOG_RESET.set(handle);
    }
    m.add_function(wrap_pyfunction!(reset_log_cache, m)?)?;
    m.add_function(wrap_pyfunction!(numa_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(mpi_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(p, m)?)?;

    // Re-exported from the core so the Python default cannot drift from the Rust one.
    m.add(
        "DEFAULT_SMALL_SUM_THRESHOLD",
        paulistrings::DEFAULT_SMALL_SUM_THRESHOLD,
    )?;

    m.add_class::<pauli_string::PauliString>()?;
    m.add_class::<sum::PauliSum>()?;
    m.add_class::<sum::PropagationStats>()?;
    m.add_class::<sum::PartitionStats>()?;
    m.add_class::<circuit::Circuit>()?;
    m.add_class::<channel_spec::PyChannel>()?;
    m.add_class::<truncation_spec::PyTruncation>()?;

    let gates_mod = PyModule::new_bound(py, "gates")?;
    gates::register(&gates_mod)?;
    m.add_submodule(&gates_mod)?;

    let noise_mod = PyModule::new_bound(py, "noise")?;
    noise::register(&noise_mod)?;
    m.add_submodule(&noise_mod)?;

    let truncation_mod = PyModule::new_bound(py, "truncation")?;
    truncation::register(&truncation_mod)?;
    m.add_submodule(&truncation_mod)?;

    Ok(())
}
