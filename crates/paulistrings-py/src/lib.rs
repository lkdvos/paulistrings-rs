//! Python bindings for `paulistrings`.
//!
//! See ARCHITECTURE.md §Python-Bindings. The user-visible Python surface is a
//! small set of classes (`PauliSum`, `Circuit`) plus three factory submodules
//! (`gates`, `noise`, `truncation`). The width parameter `W` is monomorphized
//! at a fixed set `{1, 2, 4, 8, 16}` and dispatched outside any hot loop
//! (ARCHITECTURE.md §Width).

// PyO3 0.22's `#[pymethods]` expansion converts return values via
// `.into()` even when the method already returns a `PyResult<T>`, which
// clippy flags as a useless conversion. The lint is on the macro output, not
// our code, so we silence it crate-wide.
#![allow(clippy::useless_conversion)]

#[macro_use]
mod macros;

mod channel_spec;
mod circuit;
mod gates;
mod noise;
mod sum;
mod truncation;
mod truncation_spec;

use pyo3::prelude::*;
use std::sync::OnceLock;

/// Handle returned by `pyo3_log::try_init`, kept so `reset_log_cache` can
/// clear pyo3-log's per-logger level cache. Set exactly once, at module
/// import.
static LOG_RESET: OnceLock<pyo3_log::ResetHandle> = OnceLock::new();

/// Drop the cached Python log levels of the Rust->Python log bridge.
///
/// pyo3-log caches each logger's effective level, because looking it up on
/// every record would be far too slow. Call this after changing Python log
/// levels mid-process -- e.g. after
/// `logging.getLogger("paulistrings").setLevel(logging.DEBUG)` -- otherwise
/// the new level is not picked up. A no-op if some other logger claimed the
/// `log` facade before this module was imported.
#[pyfunction]
fn reset_log_cache() {
    if let Some(handle) = LOG_RESET.get() {
        handle.reset();
    }
}

/// The NUMA nodes this process may run on, as one list of CPU indices each,
/// in ascending node order.
///
/// This is what `PauliSum.propagate(partitions="auto")` places against: `"auto"`
/// takes one partition per entry, rounded down to a power of two, and
/// `partitions=k` is refused unless there are at least `k` entries. Each list
/// is intersected with the process's CPU affinity mask, so a run inside a
/// cgroup or under `taskset` sees only the CPUs it may actually use, and a node
/// left empty by that intersection does not appear at all.
///
/// A machine with no NUMA information — no `/sys/devices/system/node`, a
/// kernel without NUMA, a non-Linux host — reports the whole affinity mask as
/// a single node, which is the honest answer: there is one domain to place in.
///
/// The lists are a snapshot: an affinity change (``os.sched_setaffinity``)
/// after the call is not reflected until the next one.
#[pyfunction]
fn numa_nodes() -> Vec<Vec<usize>> {
    paulistrings::engine::partitioned::numa_nodes()
        .into_iter()
        .map(|(_id, cpus)| cpus.0)
        .collect()
}

#[pymodule]
fn _paulistrings(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Route the core crate's `log` records (target `paulistrings::propagate`,
    // INFO entry/exit + DEBUG per layer) to Python's `logging`. Fails only if
    // some other logger is already installed, which is not an import error.
    if let Ok(handle) = pyo3_log::try_init() {
        let _ = LOG_RESET.set(handle);
    }
    m.add_function(wrap_pyfunction!(reset_log_cache, m)?)?;
    m.add_function(wrap_pyfunction!(numa_nodes, m)?)?;

    // Default for `PauliSum.propagate(small_sum_threshold=...)`, re-exported
    // from the core so the Python default cannot drift from the Rust one.
    m.add(
        "DEFAULT_SMALL_SUM_THRESHOLD",
        paulistrings::DEFAULT_SMALL_SUM_THRESHOLD,
    )?;

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
