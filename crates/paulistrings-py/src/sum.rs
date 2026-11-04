//! Python `PauliSum` class over the dtype- and width-monomorphized storage of [`crate::storage`].

use crate::dtype::{Dtype, PyCoefficient, Scalar};
pub(crate) use crate::storage::{
    extract_complex, parse_pauli_key, width_mismatch, Pairing, PauliSumImpl, WidthSum,
};
#[cfg(feature = "cuda")]
use crate::truncation_spec::COLLAPSE_SAMPLE_DEVICE_MSG;
use crate::truncation_spec::{collapse_count, collapses_since, PyTruncation, TOPN_PARTITIONED_MSG};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDescr, PyArrayMethods, PyReadonlyArray2};
use paulistrings::max_hash_bits;
use paulistrings::BuiltinTruncation;
use paulistrings::{numa_nodes, CpuSet};
use paulistrings::{
    propagate_with, Circuit as CoreCircuit, Direction, LayerScratch, PartitionConfig,
    PartitionRowPolicy, PartitionRuntime, PartitionTrace, PartitionedSum, PauliAxis,
    PauliSum as CorePauliSum, Placement, ProductBasis, PropagateOptions, ScatterOptions,
    ScatterRows, TopologyError,
};
use paulistrings::{FineLayerPhases, LayerPhases};
use pyo3::exceptions::{PyNotImplementedError, PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict};
use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

/// A product state as `PauliSum.expectation` spells it: a uniform name or one per-qubit label.
pub(crate) enum ProductState {
    Uniform(PauliAxis),
    Labels(Vec<(PauliAxis, bool)>),
}

impl ProductState {
    pub(crate) fn basis<const W: usize>(&self) -> ProductBasis<W> {
        match self {
            ProductState::Uniform(axis) => ProductBasis::uniform(*axis),
            ProductState::Labels(axes) => ProductBasis::from_axes(axes.iter().copied()),
        }
    }
}

/// `state` for a sum over `num_qubits` qubits, with `PauliSum.expectation`'s errors.
pub(crate) fn parse_product_state(state: &str, num_qubits: usize) -> PyResult<ProductState> {
    // The uniform names win first, case-insensitively, so `"x+"` keeps
    // meaning |+...+> at any qubit count. They cannot collide with a label
    // string: `x`, `y` and `z` are not in the per-qubit alphabet.
    let uniform = match state.to_ascii_lowercase().as_str() {
        "x+" => Some(PauliAxis::X),
        "y+" => Some(PauliAxis::Y),
        "z+" => Some(PauliAxis::Z),
        _ => None,
    };
    if let Some(st) = uniform {
        return Ok(ProductState::Uniform(st));
    }
    let mut axes: Vec<(PauliAxis, bool)> = Vec::with_capacity(num_qubits);
    for (q, ch) in state.chars().enumerate() {
        match parse_state_label(ch) {
            Some(entry) => axes.push(entry),
            None => {
                return Err(PyValueError::new_err(format!(
                    "unknown product state {state:?}: {ch:?} at qubit {q} is not a per-qubit \
                     label; expected a character from \"01+-rl\" (0/1 = Z±, +/- = X±, \
                     r/l = Y±), or one of the uniform names \"x+\", \"y+\", \"z+\"",
                )))
            }
        }
    }
    if axes.len() != num_qubits {
        return Err(PyValueError::new_err(format!(
            "unknown product state {state:?}: a per-qubit label string over \"01+-rl\" needs \
             one character per qubit (got {}, num_qubits is {num_qubits}); the uniform names \
             are \"x+\", \"y+\", \"z+\"",
            axes.len(),
        )));
    }
    Ok(ProductState::Labels(axes))
}

/// One character of a per-qubit product-state label, in qiskit's `Statevector.from_label` alphabet. `None` for anything else.
/// Returns `(axis, minus)`, exactly what `ProductBasis::from_axes` consumes.
fn parse_state_label(ch: char) -> Option<(PauliAxis, bool)> {
    Some(match ch {
        '0' => (PauliAxis::Z, false),
        '1' => (PauliAxis::Z, true),
        '+' => (PauliAxis::X, false),
        '-' => (PauliAxis::X, true),
        'r' => (PauliAxis::Y, false),
        'l' => (PauliAxis::Y, true),
        _ => return None,
    })
}

/// `"forward"` (the default when `None`) or `"heisenberg"`, shared by `propagate` and `propagate_with_stats` so the accepted spellings and error message cannot drift apart.
pub(crate) fn parse_direction(direction: Option<&str>) -> PyResult<Direction> {
    match direction.unwrap_or("forward") {
        "forward" => Ok(Direction::Forward),
        "heisenberg" => Ok(Direction::Heisenberg),
        other => Err(PyValueError::new_err(format!(
            "direction must be 'forward' or 'heisenberg', got {:?}",
            other
        ))),
    }
}

/// The per-layer bucket-sizing knobs as a core [`PropagateOptions`], a `ValueError` with [`PropagateOptions::validate`]'s message unless it accepts them.
/// The pyo3 signatures spell the defaults as literals (`1024`, `128`) so `help()` shows them; `literal_defaults_are_the_engines` pins them to [`PropagateOptions::default`].
pub(crate) fn propagate_options(
    target_bucket_len: usize,
    min_buckets: usize,
) -> PyResult<PropagateOptions> {
    let options = PropagateOptions { target_bucket_len, min_buckets };
    options.validate().map_err(PyValueError::new_err)?;
    Ok(options)
}

/// `partitions=` / `pin_memory=` → an optional core [`PartitionConfig`]; `None` means the classic unpartitioned path.
/// Accepted: `None`/`1` (classic), `"auto"` (one partition per NUMA node, or classic on a single-node box), an `int` power of two `>= 2` (capped at that many partitions), or `list[list[int]]` (one partition per CPU list). Anything else is a `TypeError`; a malformed value of an accepted shape is a `ValueError`.
/// Resolved against the machine here, before the GIL is released, so a bad CPU list is an exception rather than a failure inside the run; resolved again by [`PartitionRuntime::new`] (see [`runtime_for`]).
pub(crate) fn parse_partitions(
    partitions: Option<&Bound<'_, PyAny>>,
    pin_memory: bool,
) -> PyResult<Option<PartitionConfig>> {
    let Some(obj) = partitions else {
        return Ok(None);
    };
    if obj.is_none() {
        return Ok(None);
    }

    let placement = if let Ok(name) = obj.extract::<String>() {
        match name.as_str() {
            "auto" => Placement::Auto { max_partitions: None },
            other => {
                return Err(PyValueError::new_err(format!(
                    "partitions must be 'auto', got {other:?}"
                )))
            }
        }
    } else if obj.is_instance_of::<PyBool>() {
        // `bool` is an `int` subclass, so `partitions=True` would otherwise parse as `1` and silently mean "unpartitioned".
        return Err(PyTypeError::new_err(
            "partitions must be None, an int, a list of CPU lists, or 'auto', not a bool",
        ));
    } else if let Ok(count) = obj.extract::<usize>() {
        match count {
            0 => {
                return Err(PyValueError::new_err(
                    "partitions=0: a run needs at least one partition (pass None or 1 for the \
                     unpartitioned path)",
                ))
            }
            1 => return Ok(None),
            k if !k.is_power_of_two() => {
                return Err(PyValueError::new_err(format!(
                    "partitions={k} is not a power of two; a partition index is a fixed set of \
                     GF(2) hash rows, so the count must be 1, 2, 4, 8, ..."
                )))
            }
            k => {
                let nodes = numa_nodes().len();
                if nodes < k {
                    return Err(PyValueError::new_err(format!(
                        "partitions={k} needs {k} NUMA nodes in this process's CPU affinity mask, \
                         which has {nodes}; pass explicit CPU lists (e.g. \
                         partitions=[[0, 1], [2, 3]]) to place more partitions than there are \
                         nodes"
                    )));
                }
                Placement::Auto { max_partitions: Some(k) }
            }
        }
    } else if let Ok(sets) = obj.extract::<Vec<Vec<usize>>>() {
        if sets.is_empty() {
            return Err(PyValueError::new_err(
                "partitions=[]: pass at least one CPU list, or None for the unpartitioned path",
            ));
        }
        let mut seen: HashSet<usize> = HashSet::new();
        for (rank, cpus) in sets.iter().enumerate() {
            if cpus.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "partitions[{rank}] is empty; a partition needs at least one CPU"
                )));
            }
            for &cpu in cpus {
                if !seen.insert(cpu) {
                    return Err(PyValueError::new_err(format!(
                        "CPU {cpu} appears in more than one partition; the CPU lists must be \
                         disjoint, so that no two partitions share a core"
                    )));
                }
            }
        }
        Placement::Explicit(sets.into_iter().map(CpuSet).collect())
    } else {
        return Err(PyTypeError::new_err(
            "partitions must be None, an int power of two, a list of CPU lists \
             (e.g. [[0, 1], [2, 3]]), or 'auto'",
        ));
    };

    let config =
        PartitionConfig { placement, bind_memory: pin_memory, threads_per_partition: None };
    let num_partitions = config.num_partitions().map_err(topology_error)?;
    if num_partitions == 1 && matches!(config.placement, Placement::Auto { .. }) {
        // A single-node box: nothing to partition, so run the classic path rather than pay for a pool build and pin a thread nobody asked to pin. An explicit one-element CPU list is still honoured.
        return Ok(None);
    }
    Ok(Some(config))
}

/// `partition_row_blocks=` → an explicit "cut" policy, one disjoint qubit block per partition, fed straight to the core [`PartitionRows::cut`](paulistrings::PartitionRows::cut).
/// `None` (the default) means no override — the caller's `partition_row_seed`/the sum's own hash seed picks GF(2)-random rows instead, unchanged from before this knob existed.
fn parse_partition_row_blocks(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Vec<usize>>>> {
    let Some(obj) = obj else {
        return Ok(None);
    };
    if obj.is_none() {
        return Ok(None);
    }
    let blocks: Vec<Vec<usize>> = obj.extract().map_err(|_| {
        PyTypeError::new_err(
            "partition_row_blocks must be None or a list of disjoint qubit-index lists, one \
             per partition, e.g. [[0, 1, ..., 63], [64, ..., 126]] for a 2-partition cut",
        )
    })?;
    if blocks.is_empty() {
        return Err(PyValueError::new_err(
            "partition_row_blocks=[]: pass one block per partition, or None to use the default \
             (seeded) rows",
        ));
    }
    Ok(Some(blocks))
}

/// Validates `partition_row_blocks` against the partition count and qubit count, `what` naming where that count came from (`partitions=` on the in-process path, the MPI group size under `comm=`).
/// Plain Rust (`Result<(), String>`), not `PyResult`, so it — and its `#[cfg(test)]` coverage — never touch the Python C API: this crate is `extension-module`-only (see the `partition_row_knob_tests` module comment), so a test that formats a `PyErr` fails to link.
/// `validate_partition_row_blocks` (below) is the `PyResult` wrapper `parse_run_mode` actually calls.
fn validate_partition_row_blocks_impl(
    blocks: &[Vec<usize>],
    num_qubits: usize,
    num_partitions: usize,
    what: &str,
) -> Result<(), String> {
    if blocks.len() != num_partitions {
        return Err(format!(
            "partition_row_blocks has {} block(s), but {what} needs exactly one block per \
             partition",
            blocks.len()
        ));
    }
    let mut seen = vec![false; num_qubits];
    for (b, qubits) in blocks.iter().enumerate() {
        for &q in qubits {
            if q >= num_qubits {
                return Err(format!(
                    "partition_row_blocks[{b}] names qubit {q}, out of range for \
                     num_qubits={num_qubits}"
                ));
            }
            if seen[q] {
                return Err(format!(
                    "partition_row_blocks: qubit {q} appears in more than one block; \
                     blocks must be disjoint"
                ));
            }
            seen[q] = true;
        }
    }
    if !num_partitions.is_power_of_two() {
        return Err(format!(
            "partition_row_blocks cannot cut {what}: a partition is named by log2(P) GF(2) rows, \
             so the count must be a power of two"
        ));
    }
    // `PartitionRows::cut` panics on a row no qubit can set, which would leave
    // half the partitions permanently empty; the caller sees a `ValueError`
    // instead.
    for bit in 0..num_partitions.trailing_zeros() {
        let reachable =
            blocks.iter().enumerate().any(|(b, qubits)| (b >> bit) & 1 == 1 && !qubits.is_empty());
        if !reachable {
            return Err(format!(
                "partition_row_blocks: no qubit lies in a block whose index has bit {bit} set, so \
                 that partition bit is constant and half the partitions would stay empty"
            ));
        }
    }
    Ok(())
}

/// Validates `partition_row_blocks` against the resolved partition count and qubit count, with the GIL held — mirrors `parse_run_mode`'s "raise before the GIL is released" discipline, so a caller mistake surfaces as a `ValueError` rather than a panic inside `allow_threads` (`PartitionRows::cut` itself panics on a malformed block set).
fn validate_partition_row_blocks(
    blocks: &[Vec<usize>],
    num_qubits: usize,
    num_partitions: usize,
    what: &str,
) -> PyResult<()> {
    validate_partition_row_blocks_impl(blocks, num_qubits, num_partitions, what)
        .map_err(PyValueError::new_err)
}

/// `partition_row_exclude=` → the `(x, z)` qubit lists no partition row may read, checked against the register with the GIL held.
fn parse_partition_row_exclude(
    obj: Option<&Bound<'_, PyAny>>,
    num_qubits: usize,
) -> PyResult<Option<(Vec<usize>, Vec<usize>)>> {
    let Some(obj) = obj.filter(|obj| !obj.is_none()) else {
        return Ok(None);
    };
    let shape = "partition_row_exclude must be None or a dict {\"x\": [qubits], \"z\": [qubits]} \
                 naming the qubits whose x-bit / z-bit no partition row may read";
    let dict = obj.downcast::<PyDict>().map_err(|_| PyTypeError::new_err(shape))?;
    let (mut exclude_x, mut exclude_z) = (Vec::new(), Vec::new());
    for (key, value) in dict.iter() {
        let key: String = key.extract().map_err(|_| PyTypeError::new_err(shape))?;
        let qubits: Vec<usize> = value.extract().map_err(|_| PyTypeError::new_err(shape))?;
        if let Some(q) = qubits.iter().find(|&&q| q >= num_qubits) {
            return Err(PyValueError::new_err(format!(
                "partition_row_exclude[{key:?}] names qubit {q}, out of range for \
                 num_qubits={num_qubits}"
            )));
        }
        match key.as_str() {
            "x" => exclude_x = qubits,
            "z" => exclude_z = qubits,
            other => {
                return Err(PyValueError::new_err(format!(
                    "partition_row_exclude has key {other:?}; the keys are \"x\" and \"z\""
                )))
            }
        }
    }
    Ok(Some((exclude_x, exclude_z)))
}

/// A core [`TopologyError`] as a Python exception: `OSError` for a failed syscall or sysfs read, `ValueError` for everything the caller spelled wrong.
pub(crate) fn topology_error(err: TopologyError) -> PyErr {
    match err {
        TopologyError::Io(_) => PyOSError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

/// One [`PartitionRuntime`] per distinct [`PartitionConfig`], for the life of the process.
/// A runtime owns one pinned Rayon pool per partition, too expensive to build per call, so it is cached keyed by config; a linear scan is fine since a process uses one or two placements, and the mutex is held across the build so two racing first calls build one pool set.
type RuntimeCache = Mutex<Vec<(PartitionConfig, Arc<PartitionRuntime>)>>;

static PARTITION_RUNTIMES: OnceLock<RuntimeCache> = OnceLock::new();

pub(crate) fn runtime_for(
    config: &PartitionConfig,
) -> Result<Arc<PartitionRuntime>, TopologyError> {
    let cache = PARTITION_RUNTIMES.get_or_init(|| Mutex::new(Vec::new()));
    // A poisoned lock means an earlier caller panicked; the cache is append-only and never half-updated, so recover rather than poison every later call.
    let mut cache = cache.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some((_, runtime)) = cache.iter().find(|(cached, _)| cached == config) {
        return Ok(Arc::clone(runtime));
    }
    let runtime = PartitionRuntime::new(config)?;
    cache.push((config.clone(), Arc::clone(&runtime)));
    Ok(runtime)
}

/// What can go wrong inside the GIL-released region of a propagate call.
/// Every variant is turned into a Python exception after the GIL is reacquired; none can be raised from inside `allow_threads`.
pub(crate) enum PropagateFailure {
    /// The sum and the circuit monomorphized at different widths — impossible, but surfaced as an error rather than a panic.
    WidthMismatch,
    /// The partitioned placement could not be realized on this machine.
    Topology(TopologyError),
    /// Under `comm=`, the ranks hold different sums that are not already split by the call's rows.
    #[cfg(feature = "mpi")]
    MisplacedShares,
    /// A device run failed; see [`crate::gpu::gpu_error`] for the exception each variant becomes.
    #[cfg(feature = "cuda")]
    Gpu(paulistrings::gpu::GpuError),
    /// Under `comm=` with `device=`, rank `.0` failed to download its `result="local"` share.
    #[cfg(all(feature = "cuda", feature = "mpi"))]
    PeerDownload(usize),
}

#[cfg(all(feature = "cuda", feature = "mpi"))]
impl From<crate::mpi::MpiGpuFailure> for PropagateFailure {
    fn from(err: crate::mpi::MpiGpuFailure) -> Self {
        match err {
            crate::mpi::MpiGpuFailure::Gpu(err) => PropagateFailure::Gpu(err),
            crate::mpi::MpiGpuFailure::PeerDownload(rank) => PropagateFailure::PeerDownload(rank),
        }
    }
}

impl From<PropagateFailure> for PyErr {
    fn from(err: PropagateFailure) -> Self {
        match err {
            PropagateFailure::WidthMismatch => {
                PyValueError::new_err("internal: PauliSum and Circuit width mismatch")
            }
            PropagateFailure::Topology(err) => topology_error(err),
            #[cfg(feature = "mpi")]
            PropagateFailure::MisplacedShares => PyValueError::new_err(
                "comm= takes the same sum on every rank, but the ranks hold different terms that the call's partition rows would move between ranks, so the scatter would drop them; keep a split between calls with PauliSum.partition(comm=...) and change its rows with PartitionedPauliSum.repartition",
            ),
            #[cfg(feature = "cuda")]
            PropagateFailure::Gpu(err) => crate::gpu::gpu_error(err),
            #[cfg(all(feature = "cuda", feature = "mpi"))]
            PropagateFailure::PeerDownload(rank) => pyo3::exceptions::PyRuntimeError::new_err(
                format!("rank {rank} failed to download its result=\"local\" share from its CUDA device; that rank raises the reason"),
            ),
        }
    }
}

/// The `NotImplementedError` a partitioned run with an exact `topn` raises, naming the kwarg that put the call in partitioned mode.
fn topn_partitioned_error(partitions: Option<&Bound<'_, PyAny>>) -> PyErr {
    let shown = match partitions {
        // A distributed run: `comm=`'s repr says nothing useful, so the kwarg's name is the actionable part.
        None => "comm=<mpi4py communicator>".to_string(),
        Some(obj) => {
            let repr = obj.repr().map_or_else(|_| "…".to_string(), |repr| repr.to_string());
            format!("partitions={repr}")
        }
    };
    PyNotImplementedError::new_err(format!("{shown}: {TOPN_PARTITIONED_MSG}"))
}

/// `result=`: `"gather"` (the default) is `true`, `"local"` is `false`.
///
/// Parsed even when `comm=` is absent, so a typo is an error rather than a
/// silently ignored kwarg; the value itself is only read on a distributed run.
fn parse_result(result: &str) -> PyResult<bool> {
    match result {
        "gather" => Ok(true),
        "local" => Ok(false),
        other => {
            Err(PyValueError::new_err(format!("result must be 'gather' or 'local', got {other:?}")))
        }
    }
}

/// The `DeprecationWarning` `result="local"` raises, pointing at the split that replaces it.
fn warn_local_result(py: Python<'_>) -> PyResult<()> {
    PyErr::warn_bound(
        py,
        &py.get_type_bound::<pyo3::exceptions::PyDeprecationWarning>(),
        "result=\"local\" is deprecated: keep the sum split between calls with PauliSum.partition(comm=...), which reads expectation values without gathering and changes its rows with PartitionedPauliSum.repartition",
        1,
    )
}

/// Whether `comm=` names an actual communicator (`None` and an omitted kwarg
/// both mean "not distributed").
fn comm_requested(comm: Option<&Bound<'_, PyAny>>) -> bool {
    comm.is_some_and(|comm| !comm.is_none())
}

/// The `RuntimeError` a `comm=` raises in a build without the `mpi` feature.
#[cfg(not(feature = "mpi"))]
pub(crate) fn mpi_unavailable_error() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(
        "paulistrings was built without MPI support, which no wheel includes; build it from source as in \
         https://lkdvos.github.io/paulistrings-rs/installation.html#mpi-and-multi-node-runs",
    )
}

/// How one propagate call runs. Decided at the boundary, with the GIL held,
/// and consumed once inside the width dispatch.
enum RunMode {
    /// One pool over the whole process, unpartitioned.
    Classic,
    /// One pinned pool per NUMA domain, in this process (`partitions=`).
    Partitioned(PartitionConfig, PartitionRowPolicy),
    /// One partition per MPI rank (`comm=`). Carries the adopted communicator,
    /// so building the mode is the collective step and dropping it frees the
    /// duplicate.
    #[cfg(feature = "mpi")]
    Distributed(crate::mpi::MpiRun),
    /// One partition per listed CUDA device (`device=`), in this process, uploaded and downloaded around the run; the rows are `Partitioned`'s.
    #[cfg(feature = "cuda")]
    Devices(PartitionConfig, Vec<u32>, PartitionRowPolicy),
    /// One CUDA device per MPI rank (`comm=` with `device=`); owns the adopted communicator as `Distributed` does.
    #[cfg(all(feature = "cuda", feature = "mpi"))]
    DistributedDevice(crate::mpi::MpiGpuRun),
}

/// The per-layer records a run produced, and where its partitions ran.
pub(crate) struct RunTrace {
    pub(crate) trace: PartitionTrace,
    pub(crate) placement: TracePlacement,
}

/// Where a traced run's partitions ran, for `PropagationStats.partition`.
pub(crate) struct TracePlacement {
    pub(crate) partitions: usize,
    /// `(rank, size)` of a `comm=` run, whose trace holds this rank's records only.
    pub(crate) ranks: Option<(u32, u32)>,
    /// Each partition's device ordinal, for a device run.
    pub(crate) devices: Option<Vec<u32>>,
}

impl TracePlacement {
    pub(crate) fn partitions(partitions: usize) -> Self {
        Self { partitions, ranks: None, devices: None }
    }
}

/// What one run hands back besides the sum.
struct RunOutput<const W: usize, C: PyCoefficient> {
    sum: CorePauliSum<W, C>,
    trace: Option<RunTrace>,
    /// The call's collapses, `None` without a sampler in the policy.
    collapses: Option<u64>,
}

#[cfg(feature = "mpi")]
impl<const W: usize, C: PyCoefficient> RunOutput<W, C> {
    /// This rank's side of a `comm=` run, host or device.
    fn from_mpi(out: crate::mpi::MpiOutcome<W, C>) -> Self {
        let (rank, size) = (out.rank, out.size);
        Self {
            sum: out.sum,
            trace: out.trace.map(|trace| RunTrace {
                trace,
                placement: TracePlacement {
                    partitions: size as usize,
                    ranks: Some((rank, size)),
                    devices: out.device.map(|device| vec![device]),
                },
            }),
            collapses: out.collapses,
        }
    }
}

impl RunMode {
    /// Run `circuit` over `sum`, recording the per-layer trace when `traced`. Called inside `allow_threads`; consumes the mode, so a distributed run's communicator is freed before it returns.
    fn run<const W: usize, C: PyCoefficient>(
        self,
        circuit: &CoreCircuit<W>,
        sum: CorePauliSum<W, C>,
        policy: &BuiltinTruncation,
        direction: Direction,
        options: PropagateOptions,
        traced: bool,
    ) -> Result<RunOutput<W, C>, PropagateFailure> {
        let before = collapse_count(policy);
        match self {
            RunMode::Classic => {
                let mut scratch = LayerScratch::<W, C>::new();
                if traced {
                    scratch.enable_trace();
                }
                let out = with_lowered_policy!(policy, |lowered| propagate_with(
                    circuit,
                    sum,
                    lowered,
                    direction,
                    &mut scratch,
                    options,
                ));
                Ok(RunOutput {
                    sum: out,
                    trace: scratch
                        .take_trace()
                        .map(|trace| RunTrace { trace, placement: TracePlacement::partitions(1) }),
                    collapses: collapses_since(policy, before),
                })
            }
            RunMode::Partitioned(config, rows) => {
                // The runtime (and its pinned pools) is cached per config, so
                // a Trotter loop of many short calls builds it once.
                let runtime = runtime_for(&config).map_err(PropagateFailure::Topology)?;
                let mut split = PartitionedSum::scatter_with(
                    sum,
                    ScatterOptions { runtime, rows: ScatterRows::Policy(rows.clone()) },
                );
                if traced {
                    split.enable_trace();
                }
                with_lowered_policy!(policy, |lowered| split
                    .propagate_with(circuit, lowered, direction, options));
                // The partition count from the runtime, not the trace: a zero-layer circuit records no layer, but the placement is still worth reporting.
                let partitions = split.num_partitions();
                let trace = traced.then(|| RunTrace {
                    trace: split.take_trace().unwrap_or_default(),
                    placement: TracePlacement::partitions(partitions),
                });
                Ok(RunOutput {
                    sum: split.into_gathered(),
                    trace,
                    collapses: collapses_since(policy, before),
                })
            }
            #[cfg(feature = "mpi")]
            RunMode::Distributed(run) => run
                .propagate(circuit, sum, policy, direction, options, traced)
                .map(RunOutput::from_mpi),
            #[cfg(feature = "cuda")]
            RunMode::Devices(config, devices, rows) => {
                let (out, trace) =
                    run_devices(&config, &rows, circuit, sum, policy, direction, options, traced)?;
                Ok(RunOutput {
                    sum: out,
                    trace: trace.map(|trace| RunTrace {
                        trace,
                        placement: TracePlacement {
                            partitions: devices.len(),
                            ranks: None,
                            devices: Some(devices),
                        },
                    }),
                    collapses: collapses_since(policy, before),
                })
            }
            #[cfg(all(feature = "cuda", feature = "mpi"))]
            RunMode::DistributedDevice(run) => Ok(RunOutput::from_mpi(
                run.propagate(circuit, sum, policy, direction, options, traced)?,
            )),
        }
    }
}

/// Scatter `sum` over the device partitions `config` places by `rows`, dropping the host copy once uploaded, propagate, and gather.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_devices<const W: usize, C: PyCoefficient>(
    config: &PartitionConfig,
    rows: &PartitionRowPolicy,
    circuit: &CoreCircuit<W>,
    sum: CorePauliSum<W, C>,
    policy: &BuiltinTruncation,
    direction: Direction,
    options: PropagateOptions,
    traced: bool,
) -> Result<(CorePauliSum<W, C>, Option<PartitionTrace>), PropagateFailure> {
    use paulistrings::gpu::{GpuLayerOptions, GpuPartitionedSum, GpuScatterOptions};
    // Cached like a host runtime: each slot's pool binds its device context once, not per call.
    let runtime = runtime_for(config).map_err(PropagateFailure::Topology)?;
    let split = GpuPartitionedSum::scatter_to_devices_with(
        &sum,
        GpuScatterOptions {
            devices: runtime,
            rows: ScatterRows::Policy(rows.clone()),
            layer: GpuLayerOptions::default(),
        },
    );
    drop(sum);
    let mut split = split.map_err(PropagateFailure::Gpu)?;
    if traced {
        split.enable_trace();
    }
    split.propagate_with(circuit, policy, direction, options).map_err(PropagateFailure::Gpu)?;
    let trace = split.take_trace();
    let out = split.gather().map_err(PropagateFailure::Gpu)?;
    Ok((out, trace))
}

/// Turn the placement kwargs into a [`RunMode`], with the GIL held.
/// Order matters: everything that can raise on the caller's spelling is decided before the communicator is adopted, since adopting it is collective — a rank that raises early never enters the collective, so the whole group raises together.
#[allow(clippy::too_many_arguments)]
fn parse_run_mode(
    py: Python<'_>,
    partitions: Option<&Bound<'_, PyAny>>,
    pin_memory: bool,
    rows: RowKwargs,
    comm: Option<&Bound<'_, PyAny>>,
    gather: bool,
    device: Option<&Bound<'_, PyAny>>,
    policy: &BuiltinTruncation,
    sum: &PauliSumImpl,
) -> PyResult<RunMode> {
    let num_qubits = sum.num_qubits();
    let distributed = comm_requested(comm);
    if let Some(device) = device.filter(|obj| !obj.is_none()) {
        return parse_device_mode(
            py,
            device,
            partitions.is_some_and(|obj| !obj.is_none()),
            comm.filter(|_| distributed),
            gather,
            rows,
            policy,
            sum,
        );
    }
    // Before `parse_partitions`, so the conflict is reported as a conflict
    // whatever the placement would have resolved to on this machine.
    if distributed && partitions.is_some_and(|obj| !obj.is_none()) {
        return Err(PyValueError::new_err(
            "comm= and partitions= are alternatives: comm= already places one partition per MPI \
             rank, so pass the placement to the launcher (mpirun --map-by ppr:1:numa --bind-to \
             numa) rather than to propagate",
        ));
    }
    // Checked against the kwarg rather than the resolved config: `partitions="auto"` on a one-node box runs unpartitioned, where the rows are merely moot.
    if let Some(keyword) = rows.split_only_keyword() {
        if !distributed && partitions.is_none_or(|obj| obj.is_none()) {
            return Err(PyValueError::new_err(format!(
                "{keyword}= needs partitions= or comm= (it has no effect on the unpartitioned path)"
            )));
        }
    }
    let config = parse_partitions(partitions, pin_memory)?;
    if (distributed || config.is_some()) && !policy.supports_partitioned() {
        return Err(topn_partitioned_error(if distributed { None } else { partitions }));
    }
    if distributed {
        #[cfg(feature = "mpi")]
        {
            let comm = comm.expect("comm is Some");
            // `MPI_Comm_size` is local and every rank reads the same number, so validating here still rejects on every rank alike — before the collective duplicate below.
            let ranks = crate::mpi::comm_size(comm)?;
            rows.validate_count(num_qubits, ranks, &format!("comm= with {ranks} rank(s)"))?;
            let row_policy = rows.policy();
            warn_partition_rows(py, sum, &row_policy, ranks)?;
            let transport = crate::mpi::transport_from_comm(py, comm)?;
            return Ok(RunMode::Distributed(crate::mpi::MpiRun::new(
                transport, row_policy, gather,
            )));
        }
        #[cfg(not(feature = "mpi"))]
        {
            let _ = (py, gather, rows);
            return Err(mpi_unavailable_error());
        }
    }
    Ok(match config {
        Some(config) => {
            // `num_partitions()` builds no pool, unlike `PartitionRuntime::new`.
            let num_partitions = config.num_partitions().map_err(topology_error)?;
            rows.validate_count(
                num_qubits,
                num_partitions,
                &format!("partitions={num_partitions}"),
            )?;
            let row_policy = rows.policy();
            warn_partition_rows(py, sum, &row_policy, num_partitions)?;
            RunMode::Partitioned(config, row_policy)
        }
        None => RunMode::Classic,
    })
}

/// `parse_run_mode`'s `device=` branch: every conflict and spelling error is a `ValueError` or `TypeError` before anything is resolved, then exact `topn`, then whether this build and machine can honour the request.
/// Under `comm=` every check before the communicator is adopted reads only the arguments, so the group raises together; the rank-local device pick after it is agreed over the group.
#[allow(clippy::too_many_arguments)]
fn parse_device_mode(
    py: Python<'_>,
    device: &Bound<'_, PyAny>,
    partitioned: bool,
    comm: Option<&Bound<'_, PyAny>>,
    gather: bool,
    rows: RowKwargs,
    policy: &BuiltinTruncation,
    sum: &PauliSumImpl,
) -> PyResult<RunMode> {
    let num_qubits = sum.num_qubits();
    if partitioned {
        return Err(PyValueError::new_err(
            "device= and partitions= are alternatives: a device list already places one \
             partition per listed device",
        ));
    }
    if comm.is_none() && !gather {
        return Err(PyValueError::new_err(
            "result=\"local\" is a comm= option; a device= run returns the whole sum",
        ));
    }
    let request = crate::gpu::parse_device(device)?;
    let shown = device.repr().map_or_else(|_| "…".to_string(), |repr| repr.to_string());
    let shown = format!("device={shown}");
    if let (Some(_), crate::gpu::DeviceRequest::Ordinals(list)) = (comm, &request) {
        if list.len() > 1 {
            return Err(PyValueError::new_err(format!(
                "{shown} with comm=: each MPI rank drives one CUDA device, so pass one ordinal \
                 or 'auto' (a device near the rank's CPUs)"
            )));
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (py, request, rows, num_qubits, policy, sum);
        match comm {
            Some(_) => Err(crate::gpu::device_comm_unavailable_error()),
            None => Err(crate::gpu::cuda_unavailable_error()),
        }
    }
    #[cfg(feature = "cuda")]
    {
        let devices = match comm {
            Some(_) => None,
            None => Some(crate::gpu::resolve_devices(&request, &shown, num_qubits)?),
        };
        // Exact TopN runs at one partition only: one resolved device, never a comm= group (one partition per rank).
        if !policy.supports_partitioned() && devices.as_ref().is_none_or(|d| d.len() > 1) {
            return Err(PyNotImplementedError::new_err(format!(
                "{shown}: {}",
                crate::gpu::TOPN_DEVICE_MSG
            )));
        }
        // A collapse is a layer pass no device backend runs, at any partition count.
        if policy.contains_collapse_sample() {
            return Err(PyNotImplementedError::new_err(format!(
                "{shown}: {COLLAPSE_SAMPLE_DEVICE_MSG}"
            )));
        }
        let (Some(devices), None) = (devices, comm) else {
            return parse_distributed_device_mode(
                py,
                comm.expect("a device group without devices is a comm= run"),
                &request,
                &shown,
                gather,
                rows,
                sum,
            );
        };
        if let Some(keyword) = rows.split_only_keyword() {
            if devices.len() == 1 {
                return Err(PyValueError::new_err(format!(
                    "{keyword}= needs partitions=, comm= or several devices (it has no effect on \
                     the one-device run {shown})"
                )));
            }
        }
        rows.validate_count(
            num_qubits,
            devices.len(),
            &format!("{shown} ({} partitions)", devices.len()),
        )?;
        let row_policy = rows.policy();
        warn_partition_rows(py, sum, &row_policy, devices.len())?;
        let config = PartitionConfig {
            placement: Placement::Devices { devices: devices.clone(), per_device: 1 },
            bind_memory: false,
            threads_per_partition: None,
        };
        Ok(RunMode::Devices(config, devices, row_policy))
    }
}

/// `parse_device_mode` under `comm=`: validate the rows against the group size, adopt the communicator, and agree this rank's device over the group.
#[cfg(feature = "cuda")]
fn parse_distributed_device_mode(
    py: Python<'_>,
    comm: &Bound<'_, PyAny>,
    request: &crate::gpu::DeviceRequest,
    shown: &str,
    gather: bool,
    rows: RowKwargs,
    sum: &PauliSumImpl,
) -> PyResult<RunMode> {
    #[cfg(feature = "mpi")]
    {
        let ranks = crate::mpi::comm_size(comm)?;
        rows.validate_count(sum.num_qubits(), ranks, &format!("comm= with {ranks} rank(s)"))?;
        let row_policy = rows.policy();
        warn_partition_rows(py, sum, &row_policy, ranks)?;
        let transport = crate::mpi::transport_from_comm(py, comm)?;
        let rank = paulistrings::Collectives::rank(&transport);
        // Collective on every rank whatever each asked for, so an `"auto"` rank never waits on one that named an ordinal.
        let auto = paulistrings::gpu::local_device_for_comm(transport.communicator());
        let local = crate::gpu::resolve_rank_device(request, shown, rank, auto);
        let device = crate::mpi::agree_device(py, &transport, local, shown)?;
        Ok(RunMode::DistributedDevice(crate::mpi::MpiGpuRun::new(
            transport, row_policy, gather, device,
        )))
    }
    #[cfg(not(feature = "mpi"))]
    {
        let _ = (py, comm, request, shown, gather, rows, sum);
        Err(crate::gpu::device_comm_unavailable_error())
    }
}

/// Every propagation entry point requires the sum and the circuit to agree on the qubit count (they would otherwise be monomorphized at different widths, which the width dispatch cannot pair up); `what` names the sum's class in the message.
pub(crate) fn check_num_qubits(
    what: &str,
    num_qubits: usize,
    circuit: &crate::circuit::Circuit,
) -> PyResult<()> {
    if num_qubits != circuit.inner.num_qubits() {
        return Err(PyValueError::new_err(format!(
            "{what}.num_qubits ({num_qubits}) != Circuit.num_qubits ({})",
            circuit.inner.num_qubits()
        )));
    }
    Ok(())
}

/// A real-dtype sum propagates only through a circuit whose every channel preserves Hermiticity; the core panics on any other, so every entry point raises this `ValueError` first.
pub(crate) fn check_dtype_carries(
    what: &str,
    dtype: Dtype,
    circuit: &crate::circuit::Circuit,
) -> PyResult<()> {
    match hermiticity_refusal(what, dtype, &circuit.inner) {
        Some(message) => Err(PyValueError::new_err(message)),
        None => Ok(()),
    }
}

/// [`check_dtype_carries`]'s message, `None` when `circuit` can carry a `dtype` sum.
pub(crate) fn hermiticity_refusal(
    what: &str,
    dtype: Dtype,
    circuit: &crate::circuit::CircuitImpl,
) -> Option<String> {
    (dtype.is_real() && !circuit.preserves_hermiticity()).then(|| {
        format!(
            "a {} {what} cannot propagate through this circuit: one of its channels does not preserve Hermiticity, so the coefficients would turn imaginary; convert with astype(\"complex128\") (or \"complex64\") first",
            dtype.name()
        )
    })
}

/// Per-layer term counts from `PauliSum.propagate_with_stats`.
///
/// A plain record with read-only attributes; `terms_in` and `terms_out` have
/// one entry per layer applied, in application order (so *reverse* circuit
/// order under `direction="heisenberg"`).
#[pyclass(frozen, module = "paulistrings._paulistrings", name = "PropagationStats")]
pub struct PropagationStats {
    layers: usize,
    terms_in: Vec<usize>,
    terms_out: Vec<usize>,
    peak_terms: usize,
    final_terms: usize,
    partition: PartitionStats,
    circuit_index: Vec<u32>,
    application_index: Vec<u32>,
    gate_name: Vec<&'static str>,
    nanos: Vec<u64>,
    collapses: Option<u64>,
}

#[pymethods]
impl PropagationStats {
    /// Number of layers (channels) applied.
    #[getter]
    fn layers(&self) -> usize {
        self.layers
    }

    /// Term count before each layer. `terms_in[k + 1] == terms_out[k]`.
    #[getter]
    fn terms_in(&self) -> Vec<usize> {
        self.terms_in.clone()
    }

    /// Term count after each layer, i.e. **after** that layer's truncation.
    #[getter]
    fn terms_out(&self) -> Vec<usize> {
        self.terms_out.clone()
    }

    /// Peak *resident* term count between layers: `max(terms_in[0], terms_out...)`, or the input's count for a zero-layer circuit.
    /// The transient in-layer expansion (after fanout, before merge/truncation) is not measured, since that needs instrumenting the engine's hot loop; read peak RSS from `/proc/self/status` for a memory figure.
    #[getter]
    fn peak_terms(&self) -> usize {
        self.peak_terms
    }

    /// Term count of the returned sum: `terms_out[-1]`, or the input's count
    /// for a zero-layer circuit.
    #[getter]
    fn final_terms(&self) -> usize {
        self.final_terms
    }

    /// The run's per-partition record: one partition for an unpartitioned call (including `partitions="auto"` on a single-NUMA-node box), one per partition, rank or device otherwise, `devices` naming the devices of a `device=` run.
    /// Its per-layer lists are indexed the same way as `terms_in` / `terms_out`.
    #[getter]
    fn partition(&self) -> PartitionStats {
        self.partition.clone()
    }

    /// This layer's position in the circuit as written, independent of `direction`. One entry per layer, in application order.
    #[getter]
    fn circuit_index(&self) -> Vec<u32> {
        self.circuit_index.clone()
    }

    /// This layer's position in the propagation loop, i.e. the loop counter `k`: always `0..layers` regardless of `direction`. Pair with `circuit_index` to recover the circuit position without knowing the direction or circuit length.
    #[getter]
    fn application_index(&self) -> Vec<u32> {
        self.application_index.clone()
    }

    /// The applied channel's debug name (its type name, stripped of path and generics) for each layer.
    #[getter]
    fn gate_name(&self) -> Vec<&str> {
        self.gate_name.to_vec()
    }

    /// Elapsed wall-clock nanoseconds for each layer's complete gate application (rebucket/prepare through merge, truncation, and finalization).
    /// For an unpartitioned run this is exact. For a `partitions=` or `comm=` run this is the **maximum over partitions/ranks** — a critical-rank proxy, since partitions are not synchronized mid-layer; see `partition.nanos` for the raw per-partition timings that let you compute min/median/max yourself.
    #[getter]
    fn nanos(&self) -> Vec<u64> {
        self.nanos.clone()
    }

    /// How many times a `truncation.collapse_sample` in the policy collapsed the sum during this call, or `None` if the policy has none.
    /// Under `comm=` the count is rank 0's (the only rank whose sampler counts), all-reduced so every rank reports the same number.
    #[getter]
    fn collapses(&self) -> Option<u64> {
        self.collapses
    }

    /// The five term-count fields, in getter order, for a readable REPL/log line. `partition` is deliberately not here (the format is pinned by `test_propagation_stats.py`); print `stats.partition` for that.
    fn __repr__(&self) -> String {
        format!(
            "PropagationStats(layers={}, terms_in={:?}, terms_out={:?}, \
             peak_terms={}, final_terms={})",
            self.layers, self.terms_in, self.terms_out, self.peak_terms, self.final_terms
        )
    }
}

impl PropagationStats {
    /// The Python-facing record from a run's [`PartitionTrace`], plus the length of the propagated sum (what "peak" falls back to when no layer ran) and the per-partition detail in `partition`.
    /// The layer-level term counts are the per-partition ones summed, so a partitioned run is comparable to an unpartitioned one field by field.
    /// `nanos` is the **maximum** over partitions per layer (a critical-rank proxy, documented on the getter); `circuit_index`/`application_index`/`gate_name` are identical across partitions by construction (lock-step application).
    /// A distributed run's records are this rank's only, so the term-count sum is a sum of one (documented on `PartitionStats.size`) and `nanos` is that one rank's own timing.
    pub(crate) fn from_run_trace(
        run: RunTrace,
        final_terms: usize,
        collapses: Option<u64>,
    ) -> Self {
        let RunTrace { trace, placement } = run;
        let sum_of = |counts: &[usize]| counts.iter().sum::<usize>();
        let layers = &trace.layers;
        Self {
            layers: layers.len(),
            terms_in: layers.iter().map(|l| sum_of(&l.terms_in)).collect(),
            terms_out: layers.iter().map(|l| sum_of(&l.terms_out)).collect(),
            peak_terms: trace.peak_terms().unwrap_or(final_terms),
            final_terms,
            circuit_index: layers.iter().map(|l| l.circuit_index).collect(),
            application_index: layers.iter().map(|l| l.application_index).collect(),
            gate_name: layers.iter().map(|l| l.gate_name).collect(),
            nanos: layers.iter().map(|l| l.nanos.iter().copied().max().unwrap_or(0)).collect(),
            partition: PartitionStats::from_trace(
                &trace,
                placement.partitions,
                placement.ranks,
                placement.devices,
            ),
            collapses,
        }
    }

    /// A one-device run's record: one partition, `partition.devices == [device]`.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_device_trace(
        trace: PartitionTrace,
        device: u32,
        final_terms: usize,
    ) -> Self {
        let run = RunTrace {
            trace,
            placement: TracePlacement { partitions: 1, ranks: None, devices: Some(vec![device]) },
        };
        Self::from_run_trace(run, final_terms, None)
    }
}

/// Per-layer, per-partition record of a propagation — the `partition` attribute of every [`PropagationStats`], one partition for an unpartitioned call.
/// Every list is one entry per layer applied, in application order; `terms_in`/`terms_out` entries are themselves one entry per partition, in rank order.
/// Answers the two questions a partitioned run raises: how much crossed a partition boundary (`rows_exported`, `bytes_exported`, `local`), and how evenly terms were spread (`imbalance`).
/// The phase fields split each `nanos[k][r]`, each shaped like `nanos`: the seven coarse phases (`rebucket_ns`, `prepare_ns`, `collective_ns`, `export_ns`, `exchange_ns`, `coset_loop_ns`, `finalize_ns`) are disjoint wall-clock spans recorded on every run, so they sum to at most `nanos[k][r]`, short of it only by the record's own bookkeeping.
/// The fine phases (`gather_ns`, `sort_ns`, `merge_ns`, `append_ns`, `chunk_wait_ns`) are `None` unless the extension was built with the `phase-timing` feature.
#[pyclass(frozen, module = "paulistrings._paulistrings", name = "PartitionStats")]
#[derive(Clone)]
pub struct PartitionStats {
    partitions: usize,
    rank: Option<u32>,
    size: Option<u32>,
    devices: Option<Vec<u32>>,
    local: Vec<bool>,
    rows_exported: Vec<u64>,
    bytes_exported: Vec<u64>,
    terms_in: Vec<Vec<usize>>,
    terms_out: Vec<Vec<usize>>,
    imbalance: Vec<f64>,
    nanos: Vec<Vec<u64>>,
    phases: Vec<Vec<LayerPhases>>,
}

#[pymethods]
impl PartitionStats {
    /// Number of partitions the run was split across — always a power of two. For a distributed (`comm=`) run this is the MPI group size (`size` below): one partition per rank.
    #[getter]
    fn partitions(&self) -> usize {
        self.partitions
    }

    /// This process's rank in the `comm=` group, or `None` for a run that is not distributed.
    #[getter]
    fn rank(&self) -> Option<u32> {
        self.rank
    }

    /// The `comm=` group's size, or `None` for a run that is not distributed.
    /// A distributed run's per-layer lists hold this rank's entry only (`terms_in[k]`/`terms_out[k]`, `nanos[k]` and the phase fields one-element, `rows_exported[k]`/`bytes_exported[k]` this rank's, `imbalance[k]` always `1.0`) — nothing gathers the group's counters, since that would add a collective per layer for a diagnostic; reduce over `comm` yourself. The in-process case is unchanged: lists are `partitions` long.
    #[getter]
    fn size(&self) -> Option<u32> {
        self.size
    }

    /// The CUDA device ordinal of each partition for a `device=` run (this rank's one device under `comm=`), or `None` for a host run.
    #[getter]
    fn devices(&self) -> Option<Vec<u32>> {
        self.devices.clone()
    }

    /// Whether each layer was planned local: the partition rows map none of its key shifts to another partition, as `paulistrings.hashes.remote_layers` predicts.
    /// A layer planned remote can still move no row (`rows_exported[k] == 0`) when none of the terms it touches takes a crossing shift.
    #[getter]
    fn local(&self) -> Vec<bool> {
        self.local.clone()
    }

    /// Rows sent across partition boundaries in each layer, summed over every sender/receiver pair. Compare against `PropagationStats.terms_in` for the fraction of the sum that moved.
    #[getter]
    fn rows_exported(&self) -> Vec<u64> {
        self.rows_exported.clone()
    }

    /// Wire bytes behind `rows_exported`, per layer, including per-block headers.
    #[getter]
    fn bytes_exported(&self) -> Vec<u64> {
        self.bytes_exported.clone()
    }

    /// Terms each partition held before each layer: `terms_in[k][r]` for layer `k`, rank `r`. The row sums are `PropagationStats.terms_in`.
    #[getter]
    fn terms_in(&self) -> Vec<Vec<usize>> {
        self.terms_in.clone()
    }

    /// Terms each partition held after each layer's truncation. The row sums are `PropagationStats.terms_out`.
    #[getter]
    fn terms_out(&self) -> Vec<Vec<usize>> {
        self.terms_out.clone()
    }

    /// Load imbalance of `terms_in` per layer: the maximum over partitions divided by their mean. `1.0` is perfect balance; `partitions` is the worst case.
    #[getter]
    fn imbalance(&self) -> Vec<f64> {
        self.imbalance.clone()
    }

    /// Each partition's own elapsed wall time for the layer's complete gate, in nanoseconds: `nanos[k][r]` for layer `k`, partition `r`. `PropagationStats.nanos[k]` is `max(nanos[k])`; take `min`/statistics::median yourself for the rest of the skew picture.
    /// For a distributed (`comm=`) run each entry is this rank's own timing only (one-element per layer, like `terms_in`/`terms_out` above).
    #[getter]
    fn nanos(&self) -> Vec<Vec<u64>> {
        self.nanos.clone()
    }

    /// Coarse phase, shaped like `nanos`: raising or lowering the bucket count.
    #[getter]
    fn rebucket_ns(&self) -> Vec<Vec<u64>> {
        self.coarse(|phases| phases.rebucket_ns)
    }

    /// Coarse phase, shaped like `nanos`: channel preparation and partition planning.
    #[getter]
    fn prepare_ns(&self) -> Vec<Vec<u64>> {
        self.coarse(|phases| phases.prepare_ns)
    }

    /// Coarse phase, shaped like `nanos`: the bucket-count agreement and the truncation policy's own collectives, including the wait for the slowest partition or rank to arrive.
    #[getter]
    fn collective_ns(&self) -> Vec<Vec<u64>> {
        self.coarse(|phases| phases.collective_ns)
    }

    /// Coarse phase, shaped like `nanos`: building the rows to send; zero on a `local` layer.
    #[getter]
    fn export_ns(&self) -> Vec<Vec<u64>> {
        self.coarse(|phases| phases.export_ns)
    }

    /// Coarse phase, shaped like `nanos`: the transfer the layer's compute did not hide, waiting for a partner included; zero on a `local` layer.
    #[getter]
    fn exchange_ns(&self) -> Vec<Vec<u64>> {
        self.coarse(|phases| phases.exchange_ns)
    }

    /// Coarse phase, shaped like `nanos`: the layer's own compute, gather through merge.
    /// A device (`device=`) run counts its export and exchange here, leaving `export_ns` and `exchange_ns` zero.
    #[getter]
    fn coset_loop_ns(&self) -> Vec<Vec<u64>> {
        self.coarse(|phases| phases.coset_loop_ns)
    }

    /// Coarse phase, shaped like `nanos`: the truncation policy's pass after the layer, less its collectives.
    #[getter]
    fn finalize_ns(&self) -> Vec<Vec<u64>> {
        self.coarse(|phases| phases.finalize_ns)
    }

    /// Fine phase, shaped like `nanos`, or `None` unless the extension was built with the `phase-timing` feature: gathering each coset's rows, received rows included.
    /// The fine phases are busy time summed over the worker threads, so under parallelism they can exceed `coset_loop_ns`; on a device they are kernel times.
    #[getter]
    fn gather_ns(&self) -> Option<Vec<Vec<u64>>> {
        self.fine(|fine| fine.gather_ns)
    }

    /// Fine phase, shaped like `nanos`, or `None` without `phase-timing`: sorting the rows that need it.
    #[getter]
    fn sort_ns(&self) -> Option<Vec<Vec<u64>>> {
        self.fine(|fine| fine.sort_ns)
    }

    /// Fine phase, shaped like `nanos`, or `None` without `phase-timing`: merging and reducing duplicate strings.
    #[getter]
    fn merge_ns(&self) -> Option<Vec<Vec<u64>>> {
        self.fine(|fine| fine.merge_ns)
    }

    /// Fine phase, shaped like `nanos`, or `None` without `phase-timing`: appending received rows, part of `gather_ns`; zero on a `local` layer.
    #[getter]
    fn append_ns(&self) -> Option<Vec<Vec<u64>>> {
        self.fine(|fine| fine.append_ns)
    }

    /// Fine phase, shaped like `nanos`, or `None` without `phase-timing`: blocked on received rows still in flight, part of `append_ns`.
    #[getter]
    fn chunk_wait_ns(&self) -> Option<Vec<Vec<u64>>> {
        self.fine(|fine| fine.chunk_wait_ns)
    }

    /// Every field but `nanos` and its phases, in getter order; `terms_in`/`terms_out` nested one level deeper (per layer per partition).
    fn __repr__(&self) -> String {
        // Spelled with Python's `True`/`False`/`None` rather than Rust's `Debug`, so the line pastes back into a REPL.
        let local = self
            .local
            .iter()
            .map(|&local| if local { "True" } else { "False" })
            .collect::<Vec<_>>()
            .join(", ");
        let show = |value: Option<u32>| value.map_or_else(|| "None".to_string(), |v| v.to_string());
        let devices =
            self.devices.as_ref().map_or_else(|| "None".to_string(), |d| format!("{d:?}"));
        format!(
            "PartitionStats(partitions={}, rank={}, size={}, devices={}, local=[{}], \
             rows_exported={:?}, bytes_exported={:?}, terms_in={:?}, terms_out={:?}, \
             imbalance={:?})",
            self.partitions,
            show(self.rank),
            show(self.size),
            devices,
            local,
            self.rows_exported,
            self.bytes_exported,
            self.terms_in,
            self.terms_out,
            self.imbalance,
        )
    }
}

impl PartitionStats {
    /// Transpose a core [`PartitionTrace`] into the Python-facing record.
    /// `partitions` comes from the runtime (or group size), not the trace, so a zero-layer circuit still reports its placement. `ranks` is `Some((rank, size))` for a distributed run and `devices` `Some` for a device run, both `None` for an in-process host run.
    fn from_trace(
        trace: &PartitionTrace,
        partitions: usize,
        ranks: Option<(u32, u32)>,
        devices: Option<Vec<u32>>,
    ) -> Self {
        let total = |matrix: &[Vec<u64>]| matrix.iter().flat_map(|row| row.iter()).sum::<u64>();
        Self {
            partitions,
            rank: ranks.map(|(rank, _)| rank),
            size: ranks.map(|(_, size)| size),
            devices,
            local: trace.layers.iter().map(|l| l.remote_deltas == 0).collect(),
            rows_exported: trace.layers.iter().map(|l| total(&l.rows_sent)).collect(),
            bytes_exported: trace.layers.iter().map(|l| total(&l.bytes_sent)).collect(),
            terms_in: trace.layers.iter().map(|l| l.terms_in.clone()).collect(),
            terms_out: trace.layers.iter().map(|l| l.terms_out.clone()).collect(),
            imbalance: trace.imbalance(),
            nanos: trace.layers.iter().map(|l| l.nanos.clone()).collect(),
            phases: trace.layers.iter().map(|l| l.phases.clone()).collect(),
        }
    }

    fn coarse(&self, field: impl Fn(&LayerPhases) -> u64) -> Vec<Vec<u64>> {
        self.phases.iter().map(|layer| layer.iter().map(&field).collect()).collect()
    }

    /// `None` on a build without `phase-timing`, whatever the layers hold.
    fn fine(&self, field: impl Fn(&FineLayerPhases) -> u64) -> Option<Vec<Vec<u64>>> {
        cfg!(feature = "phase-timing").then(|| {
            self.phases
                .iter()
                .map(|layer| layer.iter().map(|p| field(&p.fine.unwrap_or_default())).collect())
                .collect()
        })
    }
}

#[pyclass(module = "paulistrings._paulistrings", name = "PauliSum")]
pub struct PauliSum {
    pub(crate) inner: PauliSumImpl,
}

impl PauliSum {
    /// A sum with no place in a distributed split.
    pub(crate) fn plain(inner: PauliSumImpl) -> Self {
        Self { inner }
    }

    /// The qubit counts checked before a merge — the core's merge asserts on a mismatch, and two different qubit counts can still share a width band.
    fn check_same_qubits(&self, other: &PauliSumImpl) -> PyResult<()> {
        if self.inner.num_qubits() != other.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "num_qubits mismatch ({} vs {})",
                self.inner.num_qubits(),
                other.num_qubits(),
            )));
        }
        Ok(())
    }

    /// The operator product `pairing` names with `self` the left factor, a `PauliSum` or a `PartitionedPauliSum` as `other` is.
    fn paired_with(
        &self,
        py: Python<'_>,
        other: Operand<'_>,
        pairing: Pairing,
        policy: Option<&PyTruncation>,
    ) -> PyResult<PyObject> {
        Ok(match other {
            Operand::Sum(other) => self.paired(py, &other.inner, pairing, policy)?.into_py(py),
            Operand::Split(split) => split.paired_by(py, &self.inner, pairing, policy)?.into_py(py),
        })
    }

    /// The operator product `pairing` names, at the promoted dtype and truncated by `policy` as one layer, with the GIL released.
    fn paired(
        &self,
        py: Python<'_>,
        other: &PauliSumImpl,
        pairing: Pairing,
        policy: Option<&PyTruncation>,
    ) -> PyResult<Self> {
        self.check_same_qubits(other)?;
        let policy = PyTruncation::tree_of(policy);
        let inner = py.allow_threads(|| self.inner.pair_promoted(other, pairing, &policy));
        Ok(Self::plain(inner.ok_or_else(width_mismatch)?))
    }

    /// `self ± other` at the promoted dtype.
    fn checked_add(&self, other: &PauliSumImpl, subtract: bool) -> PyResult<PauliSumImpl> {
        self.check_same_qubits(other)?;
        self.inner.add_promoted(other, subtract).ok_or_else(width_mismatch)
    }

    /// `self` scaled by a Python number. An exact-zero factor gives the empty sum, keeping the "no stored zero coefficient" invariant `from_strings` and the merge both hold.
    fn scaled(&self, factor: &Bound<'_, PyAny>) -> PyResult<PauliSumImpl> {
        let factor = scalar_factor(factor)?;
        if factor == Complex64::new(0.0, 0.0) {
            return PauliSumImpl::empty_for(self.inner.num_qubits(), self.inner.dtype())
                .ok_or_else(|| PyValueError::new_err("internal: width band lost"));
        }
        let mut inner = self.inner.clone();
        inner.scale(factor)?;
        Ok(inner)
    }

    /// `slf ± other`, in place; merged bucket by bucket when `slf` already has the promoted dtype, otherwise replaced by the promoted sum.
    fn add_in_place(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, Self>,
        subtract: bool,
    ) -> PyResult<()> {
        let sign = Complex64::new(if subtract { -1.0 } else { 1.0 }, 0.0);
        let that = Held::of(slf, other);
        let mut this = slf.borrow_mut();
        if this.inner.dtype().promote(that.dtype()) == this.inner.dtype() {
            this.check_same_qubits(&that)?;
            let (inner, that) = (&mut this.inner, &*that);
            return slf.py().allow_threads(move || inner.add_scaled(that, sign, Complex64::ONE));
        }
        this.inner = this.checked_add(&that, subtract)?;
        Ok(())
    }

    /// `slf ← alpha · pairing(generator, operand) + beta · slf`, the body of `add_product`, `add_liouvillian` and `add_anticommutator`.
    fn add_paired_into(
        slf: &Bound<'_, Self>,
        generator: &Bound<'_, Self>,
        operand: &Bound<'_, Self>,
        pairing: Pairing,
        weights: (Complex64, Complex64),
        policy: Option<&PyTruncation>,
    ) -> PyResult<()> {
        let (generator, operand) = (Held::of(slf, generator), Held::of(slf, operand));
        let mut this = slf.borrow_mut();
        this.check_same_qubits(&generator)?;
        this.check_same_qubits(&operand)?;
        let policy = PyTruncation::tree_of(policy);
        let (inner, generator, operand) = (&mut this.inner, &*generator, &*operand);
        slf.py().allow_threads(move || {
            inner.add_paired(generator, operand, pairing, weights, &policy)
        })
    }
}

/// The sum an in-place call on `slf` reads from `other`: a copy when `other` is `slf`, so the call never borrows one cell twice, otherwise a borrow.
enum Held<'py> {
    Copy(PauliSumImpl),
    Borrowed(PyRef<'py, PauliSum>),
}

impl<'py> Held<'py> {
    fn of(slf: &Bound<'py, PauliSum>, other: &Bound<'py, PauliSum>) -> Self {
        if slf.is(other) {
            Held::Copy(slf.borrow().inner.clone())
        } else {
            Held::Borrowed(other.borrow())
        }
    }
}

impl std::ops::Deref for Held<'_> {
    type Target = PauliSumImpl;

    fn deref(&self) -> &PauliSumImpl {
        match self {
            Held::Copy(inner) => inner,
            Held::Borrowed(sum) => &sum.inner,
        }
    }
}

/// How many terms `__repr__`/`__str__` show before falling back to `... (N more terms)`.
const PREVIEW_TERMS: usize = 4;

/// `PauliSum.__repr__`/`__str__`'s body: the first [`PREVIEW_TERMS`] terms as `coefficient*label`, `+`-joined, with a trailing count of however many more there are.
/// `0` for the empty sum — the zero operator, not "no terms".
fn format_sum(inner: &PauliSumImpl) -> String {
    let len = inner.len();
    if len == 0 {
        return "0".to_string();
    }
    let shown = inner.preview(PREVIEW_TERMS);
    let mut parts: Vec<String> = shown.iter().map(|(label, c)| format!("{c}*{label}")).collect();
    if len > shown.len() {
        parts.push(format!(
            "... ({} more term{})",
            len - shown.len(),
            if len - shown.len() == 1 { "" } else { "s" }
        ));
    }
    parts.join(" + ")
}

/// One row's `2 · num_qubits` coordinate flags, x-bits first.
pub(crate) fn coordinate_flags<const W: usize>(
    x: &[u64; W],
    z: &[u64; W],
    num_qubits: usize,
) -> Vec<bool> {
    let bit = |half: &[u64; W], q: usize| (half[q / 64] >> (q % 64)) & 1 == 1;
    (0..num_qubits).map(|q| bit(x, q)).chain((0..num_qubits).map(|q| bit(z, q))).collect()
}

/// A matrix of shape `(k, 2 · num_qubits)`, `k <= max_rows`, whose entries are booleans or `0`/`1`, as its rows of flags; `name` is the keyword in errors.
pub(crate) fn parse_row_matrix(
    py: Python<'_>,
    rows: &Bound<'_, PyAny>,
    num_qubits: usize,
    max_rows: usize,
    name: &str,
) -> PyResult<Vec<Vec<bool>>> {
    let numpy = py.import_bound("numpy")?;
    let array = numpy.call_method1("asarray", (rows,)).map_err(|err| {
        PyValueError::new_err(format!(
            "{name} must be a boolean array of shape (k, 2 * num_qubits): {err}"
        ))
    })?;
    let shape: Vec<usize> = array.getattr("shape")?.extract()?;
    let width = 2 * num_qubits;
    if shape.len() != 2 || shape[1] != width || shape[0] > max_rows {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (k, {width}) with k <= {max_rows} — one column per coordinate, the {num_qubits} x-bits then the {num_qubits} z-bits — got shape {shape:?}"
        )));
    }
    let is_bool = array.getattr("dtype")?.eq(numpy.getattr("bool_")?)?;
    if !is_bool {
        let binary = numpy.call_method1("isin", (&array, (0, 1)))?.call_method0("all")?;
        if !binary.is_truthy()? {
            return Err(PyValueError::new_err(format!(
                "{name} must hold booleans or 0/1 entries, one flag per coordinate"
            )));
        }
    }
    array.call_method1("astype", ("bool",))?.call_method0("tolist")?.extract()
}

/// A Python `RuntimeWarning`, attributed to the caller's line.
pub(crate) fn warn_runtime(py: Python<'_>, message: &str) -> PyResult<()> {
    PyErr::warn_bound(py, &py.get_type_bound::<pyo3::exceptions::PyRuntimeWarning>(), message, 1)
}

/// The partition-row keywords of `propagate`, parsed and checked against each other with the GIL held.
pub(crate) struct RowKwargs {
    seed: Option<u64>,
    blocks: Option<Vec<Vec<usize>>>,
    exclude: Option<(Vec<usize>, Vec<usize>)>,
    rows: Option<Vec<Vec<bool>>>,
}

impl RowKwargs {
    pub(crate) fn parse(
        py: Python<'_>,
        seed: Option<u64>,
        blocks: Option<&Bound<'_, PyAny>>,
        exclude: Option<&Bound<'_, PyAny>>,
        rows: Option<&Bound<'_, PyAny>>,
        num_qubits: usize,
    ) -> PyResult<Self> {
        let blocks = parse_partition_row_blocks(blocks)?;
        if seed.is_some() && blocks.is_some() {
            return Err(PyValueError::new_err(
                "partition_row_seed= and partition_row_blocks= are alternatives (a seeded random \
                 draw vs. an explicit locality cut); pass at most one",
            ));
        }
        let exclude = parse_partition_row_exclude(exclude, num_qubits)?;
        if exclude.is_some() && blocks.is_some() {
            return Err(PyValueError::new_err(
                "partition_row_blocks= and partition_row_exclude= are alternatives (an explicit \
                 locality cut vs. a seeded draw that skips some coordinates); pass at most one",
            ));
        }
        let rows = rows
            .filter(|obj| !obj.is_none())
            .map(|obj| {
                parse_row_matrix(
                    py,
                    obj,
                    num_qubits,
                    max_hash_bits(num_qubits) as usize,
                    "partition_rows",
                )
            })
            .transpose()?;
        if rows.is_some() && (seed.is_some() || blocks.is_some() || exclude.is_some()) {
            return Err(PyValueError::new_err(
                "partition_rows= gives the rows outright, so it takes none of partition_row_seed=, \
                 partition_row_blocks= or partition_row_exclude=",
            ));
        }
        Ok(Self { seed, blocks, exclude, rows })
    }

    /// The keyword that only means something when the sum is split, if one was given.
    fn split_only_keyword(&self) -> Option<&'static str> {
        if self.blocks.is_some() {
            Some("partition_row_blocks")
        } else if self.rows.is_some() {
            Some("partition_rows")
        } else if self.exclude.is_some() {
            Some("partition_row_exclude")
        } else {
            None
        }
    }

    /// Check what can be checked once the partition count is known: a cut's blocks or the explicit rows' count; `what` names where the count came from.
    pub(crate) fn validate_count(
        &self,
        num_qubits: usize,
        partitions: usize,
        what: &str,
    ) -> PyResult<()> {
        if let Some(blocks) = &self.blocks {
            validate_partition_row_blocks(blocks, num_qubits, partitions, what)?;
        }
        if let Some(rows) = &self.rows {
            let bits = partitions.trailing_zeros() as usize;
            if rows.len() != bits {
                return Err(PyValueError::new_err(format!(
                    "partition_rows has {} rows, but {what} needs {bits}: one per partition bit",
                    rows.len()
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn policy(&self) -> PartitionRowPolicy {
        let (exclude_x, exclude_z) = self.exclude.clone().unwrap_or_default();
        match (&self.rows, &self.blocks, &self.exclude) {
            (Some(rows), ..) => PartitionRowPolicy::Matrix(rows.clone()),
            (None, Some(blocks), _) => PartitionRowPolicy::Cut(blocks.clone()),
            (None, None, Some(_)) => {
                PartitionRowPolicy::SeededExcluding { seed: self.seed, exclude_x, exclude_z }
            }
            (None, None, None) => PartitionRowPolicy::Seeded(self.seed),
        }
    }
}

/// A `RuntimeWarning` when the rows `policy` names for `partitions` partitions are dependent; overlap with the bucket rows is the engine's to log, at the layer it starts to matter.
pub(crate) fn warn_partition_rows(
    py: Python<'_>,
    sum: &PauliSumImpl,
    policy: &PartitionRowPolicy,
    partitions: usize,
) -> PyResult<()> {
    if partitions <= 1 {
        return Ok(());
    }
    match sum.partition_rank_deficiency(policy, partitions.trailing_zeros() as u8) {
        Some(reason) => warn_runtime(py, &reason),
        None => Ok(()),
    }
}

/// The scalar `*` accepts. Naming the two-sum case explicitly, since `a * b` on two sums is the one multiplication a reader is most likely to expect and the least likely to get.
/// The right factor of an operator product: a sum held whole, or a split one.
#[derive(FromPyObject)]
pub(crate) enum Operand<'py> {
    Sum(PyRef<'py, PauliSum>),
    Split(PyRef<'py, crate::partitioned::PartitionedPauliSum>),
}

fn scalar_factor(factor: &Bound<'_, PyAny>) -> PyResult<Complex64> {
    if factor.downcast::<PauliSum>().is_ok() {
        return Err(PyTypeError::new_err(
            "PauliSum * PauliSum is not supported: * and *= take a complex or real number; the \
             operator product is a @ b",
        ));
    }
    extract_complex(factor).map_err(|_| {
        PyTypeError::new_err("PauliSum * x: x must be a complex or real number (scalar scaling)")
    })
}

#[pymethods]
impl PauliSum {
    /// Empty Pauli sum on `num_qubits` qubits.
    ///
    /// `dtype` is the coefficient type: `float64`, `float32`, `complex128` (the default) or `complex64`, in any spelling `numpy.dtype` accepts.
    #[new]
    #[pyo3(signature = (num_qubits, *, dtype=None))]
    fn new(py: Python<'_>, num_qubits: usize, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let dtype = Dtype::parse(py, dtype)?;
        PauliSumImpl::empty_for(num_qubits, dtype).map(Self::plain).ok_or_else(|| {
            PyValueError::new_err("num_qubits exceeds largest monomorphized width (1024)")
        })
    }

    /// Build from a `{pauli_string: coefficient}` dict, or from `(labels, coefficients)` — two equal-length sequences.
    ///
    /// Each label is a string of `I/X/Y/Z` characters, one per qubit (index
    /// `i` addresses qubit `i`). Coefficients multiply the literal Hermitian
    /// Pauli string, so a Hermitian observable has real coefficients.
    /// `num_qubits` is inferred from the first label's length when omitted.
    /// A label repeated in the two-sequence form accumulates, unlike a dict's keys.
    /// `dtype` is as for `PauliSum(num_qubits, dtype=...)`; a real dtype raises `ValueError` on a coefficient with a nonzero imaginary part.
    #[classmethod]
    #[pyo3(signature = (terms, coefficients=None, *, num_qubits=None, dtype=None))]
    fn from_strings(
        cls: &Bound<'_, pyo3::types::PyType>,
        terms: &Bound<'_, PyAny>,
        coefficients: Option<&Bound<'_, PyAny>>,
        num_qubits: Option<usize>,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dtype = Dtype::parse(cls.py(), dtype)?;
        let inner = match coefficients {
            Some(coefficients) => {
                if terms.downcast::<PyDict>().is_ok() {
                    return Err(PyTypeError::new_err(
                        "PauliSum.from_strings: pass either a dict, or (labels, coefficients) as two sequences — not a dict with coefficients also given",
                    ));
                }
                PauliSumImpl::from_label_list(terms, coefficients, num_qubits, dtype)?
            }
            None => {
                let dict = terms.downcast::<PyDict>().map_err(|_| {
                    PyTypeError::new_err(
                        "PauliSum.from_strings: pass a dict, or (labels, coefficients) as two equal-length sequences",
                    )
                })?;
                PauliSumImpl::from_strings_dict(dict, num_qubits, dtype)?
            }
        };
        Ok(Self::plain(inner))
    }

    /// Build from raw symplectic `(x, z, coefficients)` arrays — the inverse of `x_array` / `z_array` / `coefficients_array`.
    ///
    /// `x`/`z` are `uint64` arrays of shape `(n_terms, w)`, `w` from `1` up to the band width `num_qubits` picks; a narrower array is zero-padded, so a sum round-trips exactly. `coefficients` is a 1-D `float64`, `float32`, `complex128` or `complex64` array of length `n_terms`.
    /// Rows are symplectic keys in the Hermitian convention (no phase, matching `from_strings`); duplicate `(x, z)` rows sum their coefficients and exact-zero rows are dropped. A set bit at or beyond qubit `num_qubits` is a `ValueError`.
    /// `dtype` is the sum's, as for `PauliSum(num_qubits, dtype=...)`, whatever the array's own dtype; `to_arrays` returns the sum's.
    #[classmethod]
    #[pyo3(signature = (x, z, coefficients, num_qubits, *, dtype=None))]
    fn from_arrays(
        cls: &Bound<'_, pyo3::types::PyType>,
        x: PyReadonlyArray2<'_, u64>,
        z: PyReadonlyArray2<'_, u64>,
        coefficients: &Bound<'_, PyAny>,
        num_qubits: usize,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dtype = Dtype::parse(cls.py(), dtype)?;
        let inner = PauliSumImpl::from_arrays(num_qubits, &x, &z, coefficients, dtype)?;
        Ok(Self::plain(inner))
    }

    /// The coefficient type, a `numpy.dtype`: `float64`, `float32`, `complex128` or `complex64`.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDescr> {
        self.inner.dtype().numpy(py)
    }

    /// A copy with coefficients of type `dtype`, in any spelling `numpy.dtype` accepts.
    ///
    /// Raises `ValueError` when `dtype` is real and a coefficient has a nonzero imaginary part; a narrower float rounds.
    fn astype(&self, py: Python<'_>, dtype: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dtype = Dtype::parse(py, Some(dtype))?;
        self.inner.cast(dtype).map(Self::plain).ok_or_else(|| {
            crate::dtype::imaginary_error(
                &format!("astype({}): a coefficient", dtype.name()),
                dtype,
            )
        })
    }

    /// An independent copy, to keep an input that `propagate` would otherwise evolve in place.
    fn copy(&self) -> Self {
        Self::plain(self.inner.clone())
    }

    fn __copy__(&self) -> Self {
        self.copy()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.copy()
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The first few terms as `coefficient*label`, `+`-joined; `0` for the empty sum.
    /// Always a prefix in canonical storage order, never sorted by magnitude — that would cost `O(len() log len())` just to print, on a type whose whole point is staying cheap at a huge term count.
    fn __repr__(&self) -> String {
        format_sum(&self.inner)
    }

    fn __str__(&self) -> String {
        format_sum(&self.inner)
    }

    /// Current bucket count the sum's storage is partitioned into.
    /// Realized, not requested: reflects where the last `propagate` left it, which refines past `target_bucket_len` terms per bucket and coarsens only below an eighth of it, so it can differ from a `min_buckets`/`target_bucket_len` request passed to `propagate`.
    #[getter]
    fn num_buckets(&self) -> usize {
        self.inner.num_buckets()
    }

    /// The seed the bucket hash's rows were drawn from.
    #[getter]
    fn hash_seed(&self) -> u64 {
        self.inner.hash_seed()
    }

    /// Terms per bucket, in bucket order, as a NumPy `uint64` array: how evenly the hash spreads this sum.
    fn bucket_lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        self.inner
            .bucket_lengths()
            .into_iter()
            .map(|n| n as u64)
            .collect::<Vec<u64>>()
            .into_pyarray_bound(py)
    }

    /// The bucket hash's rows as a boolean NumPy array of shape `(min(2 * num_qubits, 31), 2 * num_qubits)`: entry `[i, c]` is set when row `i` reads coordinate `c`, the x-bit of qubit `c` for `c < num_qubits` and the z-bit of qubit `c - num_qubits` above. The first `log2(num_buckets)` rows are active.
    fn hash_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let rows = self.inner.hash_rows();
        Ok(PyArray2::from_vec2_bound(py, &rows)?)
    }

    /// A copy with the same terms, partitioned into buckets by another GF(2)-linear hash; `propagate` keeps the hash a sum carries.
    ///
    /// `rows` is a boolean (or 0/1) array of shape `(k, 2 * num_qubits)` with `k` at most `hash_rows().shape[0]`, laid out as `hash_rows()` returns it; they become the first `k` rows, and dense random rows drawn from `seed` (default: this sum's) fill the rest, each independent of the rows before it.
    /// `paulistrings.hashes` builds such matrices; `rows=None` draws every row from `seed`.
    /// Linearly dependent rows are accepted with a `RuntimeWarning`: past their rank, some buckets can never hold a term.
    /// Every hash gives the same results; they differ only in how evenly they spread the terms over buckets, and hence in speed.
    #[pyo3(signature = (rows=None, seed=None))]
    fn with_hash(
        &self,
        py: Python<'_>,
        rows: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let num_qubits = self.inner.num_qubits();
        let seed = seed.unwrap_or_else(|| self.inner.hash_seed());
        let rows = match rows.filter(|obj| !obj.is_none()) {
            Some(rows) => {
                parse_row_matrix(py, rows, num_qubits, max_hash_bits(num_qubits) as usize, "rows")?
            }
            None => Vec::new(),
        };
        let rehashed = self.inner.with_hash_rows(&rows, seed);
        if let Some(reason) = rehashed.hash_rank_deficiency() {
            warn_runtime(py, &reason)?;
        }
        Ok(Self::plain(rehashed))
    }

    /// Snapshot of the coefficient column as a list of Python `float` (real dtype) or `complex` values.
    fn coefficients(&self, py: Python<'_>) -> PyObject {
        self.inner.coefficients(py)
    }

    /// Expectation value in a single-qubit product state.
    ///
    /// `state` is either a uniform name — `"x+"`, `"y+"`, `"z+"`, the `+1` eigenstate of that Pauli on every qubit, matched case-insensitively — or a per-qubit label string of exactly `num_qubits` characters in qiskit's `Statevector.from_label` alphabet (`0`/`1` = Z±, `+`/`-` = X±, `r`/`l` = Y±, case-sensitive). Qubit indexing matches `from_strings`.
    /// Cost is one masked pass over the terms either way, never an expansion over basis states. Returns a Python `float` for a real dtype and a `complex` otherwise; take `.real` of the latter when the operator is Hermitian.
    #[pyo3(signature = (state="x+"))]
    fn expectation(&self, state: &str) -> PyResult<Scalar> {
        Ok(match parse_product_state(state, self.inner.num_qubits())? {
            ProductState::Uniform(axis) => self.inner.expectation_uniform(axis),
            ProductState::Labels(axes) => self.inner.expectation_labels(&axes),
        })
    }

    /// Expectation value in a stabilizer state given by its generators.
    ///
    /// `generators` is a list of exactly `num_qubits` signed Pauli strings — `"+XX"`, `"-ZZ"`, or a bare `"ZIZ"` for `+` — in the same `I/X/Y/Z` alphabet and qubit indexing as `from_strings`. They must be pairwise commuting and independent over GF(2); anything else is a `ValueError`.
    /// Reads any stabilizer state (Bell, GHZ, cluster, Clifford-circuit output; see `paulistrings.interop.stabilizers_from_stim`), where `expectation` reads only product states. Cost is `O(terms · num_qubits² / 64)` after a one-off `O(num_qubits³ / 64)` reduction of the generators; prefer `expectation` when the state factorizes.
    /// Returns a Python `float` for a real dtype and a `complex` otherwise, as `expectation` does.
    ///
    /// ```python
    /// bell = ["XX", "ZZ"]                      # (|00> + |11>) / sqrt(2)
    /// PauliSum.from_strings({"YY": 1.0}, num_qubits=2).expectation_stabilizer(bell)
    /// # (-1+0j), since XX·ZZ = -YY
    /// ```
    fn expectation_stabilizer(&self, generators: Vec<String>) -> PyResult<Scalar> {
        self.inner.expectation_stabilizer(&generators)
    }

    /// Hilbert-Schmidt overlap `tr(self* . other) / 2^n`.
    ///
    /// On the Pauli basis this is `sum(conj(a_i) * b_i)` over shared keys, computed at the two dtypes' NumPy promotion and returned as a `float` when that is real.
    fn overlap(&self, other: &Self) -> PyResult<Scalar> {
        if self.inner.num_qubits() != other.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "overlap: num_qubits mismatch ({} vs {})",
                self.inner.num_qubits(),
                other.inner.num_qubits(),
            )));
        }
        self.inner.overlap(&other.inner).ok_or_else(|| {
            PyValueError::new_err("overlap: sums were monomorphized at different widths")
        })
    }

    /// Coefficient of the identity term, i.e. `tr(O) / 2^n`; a `float` for a real dtype.
    fn identity_coefficient(&self) -> Scalar {
        self.inner.identity_coefficient()
    }

    /// `self + other`: coefficients added on matching strings, the rest kept. Both operands are left untouched.
    /// The result's dtype is NumPy's promotion of the two: complex if either is, double precision if either is.
    fn __add__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self::plain(self.checked_add(&other.inner, false)?))
    }

    /// `self - other`, the same merge with `other`'s coefficients negated.
    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self::plain(self.checked_add(&other.inner, true)?))
    }

    /// `self += other`, in place, at the dtype `+` would give.
    fn __iadd__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        Self::add_in_place(slf, other, false)
    }

    /// `self -= other`, in place, as `+=`.
    fn __isub__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        Self::add_in_place(slf, other, true)
    }

    /// `self = alpha * other + beta * self`, in place, merging bucket by bucket rather than building a new sum.
    ///
    /// `self` keeps its dtype: `other` is converted to it, and a real `self` raises `ValueError` on an imaginary coefficient or factor.
    /// `beta=0` ignores `self`'s current terms; strings whose coefficients cancel exactly are dropped.
    ///
    /// ```python
    /// y = PauliSum.from_strings({"X": 1.0, "Z": 2.0}, num_qubits=1)
    /// y.add(PauliSum.from_strings({"Z": 1.0}, num_qubits=1), alpha=-2.0)
    /// y                                        # 1.0 * X: the Z terms cancel exactly
    /// ```
    #[pyo3(signature = (other, alpha=Complex64::ONE, beta=Complex64::ONE))]
    fn add(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, Self>,
        alpha: Complex64,
        beta: Complex64,
    ) -> PyResult<()> {
        let that = Held::of(slf, other);
        let mut this = slf.borrow_mut();
        this.check_same_qubits(&that)?;
        let (inner, that) = (&mut this.inner, &*that);
        slf.py().allow_threads(move || inner.add_scaled(that, alpha, beta))
    }

    /// `self = alpha * (generator @ operand) + beta * self`, in place: the pairs are summed straight into `self`, one bucket at a time, so no intermediate product is built.
    ///
    /// `self` keeps its dtype, which must be complex; the operands are converted to it.
    /// `policy` truncates the result as one propagation layer would, and its filter sees each string's coefficient after `beta * self` is added.
    /// `beta=0` ignores `self`'s current terms, so an empty sum of the right dtype is a fresh destination.
    #[pyo3(signature = (generator, operand, alpha=Complex64::ONE, beta=Complex64::ONE, policy=None))]
    fn add_product(
        slf: &Bound<'_, Self>,
        generator: &Bound<'_, Self>,
        operand: &Bound<'_, Self>,
        alpha: Complex64,
        beta: Complex64,
        policy: Option<&PyTruncation>,
    ) -> PyResult<()> {
        Self::add_paired_into(slf, generator, operand, Pairing::Product, (alpha, beta), policy)
    }

    /// `self = alpha * i[generator, operand] + beta * self`, in place, as `add_product` accumulates; a real `self` stays real for Hermitian operands.
    ///
    /// One Lanczos step overwrites the older vector, so only two operators are ever held:
    ///
    /// ```python
    /// h = PauliSum.from_strings({"ZZ": 1.0, "XI": 0.5}, dtype="float64")
    /// previous, current = PauliSum(2, dtype="float64"), PauliSum.from_strings({"ZI": 1.0}, dtype="float64")
    /// previous.add_liouvillian(h, current, beta=0.0)   # previous = i[h, current]
    /// previous                                         # 1.0 * YI, since i[0.5 X, Z] = Y
    /// ```
    #[pyo3(signature = (generator, operand, alpha=Complex64::ONE, beta=Complex64::ONE, policy=None))]
    fn add_liouvillian(
        slf: &Bound<'_, Self>,
        generator: &Bound<'_, Self>,
        operand: &Bound<'_, Self>,
        alpha: Complex64,
        beta: Complex64,
        policy: Option<&PyTruncation>,
    ) -> PyResult<()> {
        Self::add_paired_into(slf, generator, operand, Pairing::Liouvillian, (alpha, beta), policy)
    }

    /// `self = alpha * {generator, operand} + beta * self`, in place, as `add_product` accumulates.
    #[pyo3(signature = (generator, operand, alpha=Complex64::ONE, beta=Complex64::ONE, policy=None))]
    fn add_anticommutator(
        slf: &Bound<'_, Self>,
        generator: &Bound<'_, Self>,
        operand: &Bound<'_, Self>,
        alpha: Complex64,
        beta: Complex64,
        policy: Option<&PyTruncation>,
    ) -> PyResult<()> {
        let pairing = Pairing::Anticommutator;
        Self::add_paired_into(slf, generator, operand, pairing, (alpha, beta), policy)
    }

    /// The operator product `self @ other`, with every pair of strings multiplied and equal strings summed.
    ///
    /// The result is complex at the operands' precision, since a product of anticommuting Pauli strings carries a factor of `i`.
    /// It costs one string product per pair of terms; for a commutator with a Hermitian generator use `liouvillian`, which does less work and keeps a real dtype real.
    /// `policy` truncates the result as one propagation layer would: a filter sees each string's summed coefficient, and a layer pass such as `topn` runs once on the result.
    ///
    /// `other` may be a `PartitionedPauliSum`, with `self` the same on every partition or rank: the result is then a new split by the same rows, each partition receiving from another only the strings a generator term moving between them can reach.
    /// A split operand keeps its dtype rather than promoting, so a product needs a complex split, and partitioned exact `truncation.topn` raises `NotImplementedError` as `propagate` does. **Collective** under MPI.
    ///
    /// ```python
    /// x = PauliSum.from_strings({"X": 1.0}, num_qubits=1)
    /// z = PauliSum.from_strings({"Z": 1.0}, num_qubits=1)
    /// x.product(z)                             # -1j * Y
    /// ```
    #[pyo3(signature = (other, policy=None))]
    fn product(
        &self,
        py: Python<'_>,
        other: Operand<'_>,
        policy: Option<&PyTruncation>,
    ) -> PyResult<PyObject> {
        self.paired_with(py, other, Pairing::Product, policy)
    }

    /// `self @ other`, the operator product; see `product`.
    fn __matmul__(&self, py: Python<'_>, other: Operand<'_>) -> PyResult<PyObject> {
        self.product(py, other, None)
    }

    /// The Liouvillian `i[self, other] = i(self @ other - other @ self)`, the step of a Lanczos recursion or a Heisenberg-picture Taylor series with generator `self`.
    ///
    /// Only anticommuting string pairs contribute.
    /// For Hermitian `self` and `other` the result is Hermitian, so a real dtype stays real; the dtype is the operands' promotion.
    /// Only pairs acting on a common qubit are visited, so a local generator against a large operator costs far fewer than `len(self) * len(other)` string products.
    /// `policy` truncates the result as `product`'s does; a nested commutator of a symmetric generator cancels many strings only to rounding, so a coefficient threshold belongs in every step of a recursion.
    /// `other` may be a `PartitionedPauliSum`, as for `product`.
    ///
    /// ```python
    /// h = PauliSum.from_strings({"Z": 1.0}, num_qubits=1, dtype="float64")
    /// o = PauliSum.from_strings({"X": 1.0}, num_qubits=1, dtype="float64")
    /// h.liouvillian(o)                         # -2.0 * Y, since ZX = iY
    /// ```
    #[pyo3(signature = (other, policy=None))]
    fn liouvillian(
        &self,
        py: Python<'_>,
        other: Operand<'_>,
        policy: Option<&PyTruncation>,
    ) -> PyResult<PyObject> {
        self.paired_with(py, other, Pairing::Liouvillian, policy)
    }

    /// The anticommutator `{self, other} = self @ other + other @ self`.
    ///
    /// Only commuting string pairs contribute.
    /// For Hermitian operands the result is Hermitian, so a real dtype stays real; the dtype is the operands' promotion.
    /// `policy` truncates the result as `product`'s does, and `other` may be a `PartitionedPauliSum`, as for `product`.
    #[pyo3(signature = (other, policy=None))]
    fn anticommutator(
        &self,
        py: Python<'_>,
        other: Operand<'_>,
        policy: Option<&PyTruncation>,
    ) -> PyResult<PyObject> {
        self.paired_with(py, other, Pairing::Anticommutator, policy)
    }

    /// `self * scalar`: every coefficient scaled by a complex or real number, keeping the dtype.
    ///
    /// A real dtype raises `ValueError` on a scalar with a nonzero imaginary part; `astype` to a complex dtype first.
    /// Scalar-only; the operator product of two sums is `product`, or `self @ other`.
    fn __mul__(&self, factor: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::plain(self.scaled(factor)?))
    }

    /// `scalar * self`, identical to `self * scalar`.
    fn __rmul__(&self, factor: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(factor)
    }

    /// `self *= scalar`, in place.
    fn __imul__(&mut self, factor: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner = self.scaled(factor)?;
        Ok(())
    }

    /// Snapshot of the coefficient column as a 1-D NumPy array of the sum's dtype.
    fn coefficients_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.inner.coefficients_array(py)
    }

    /// Snapshot of the X-part column as a 2-D NumPy `uint64` array of shape
    /// `(len, W)` where `W` is the monomorphized width chosen for this sum.
    /// One row per term; column `j` holds the bit-word covering qubits
    /// `64*j .. 64*(j+1)`.
    fn x_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        let (w, x_flat, _z_flat) = self.inner.xz_flat();
        let n = x_flat.len() / w;
        x_flat.into_pyarray_bound(py).reshape([n, w]).expect("flat length is n*w by construction")
    }

    /// Snapshot of the Z-part column as a 2-D NumPy `uint64` array. See
    /// `x_array` for the layout.
    fn z_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        let (w, _x_flat, z_flat) = self.inner.xz_flat();
        let n = z_flat.len() / w;
        z_flat.into_pyarray_bound(py).reshape([n, w]).expect("flat length is n*w by construction")
    }

    /// `(x, z, coefficients)` as NumPy arrays, the exact inverse of `from_arrays`: `x`/`z` are `uint64` of shape `(len, width)` and `coefficients` is of the sum's dtype and shape `(len,)`, row `i` of each describing the same term.
    ///
    /// On a `propagate(..., comm=..., result="local")` share it is this rank's terms only.
    #[allow(clippy::type_complexity)]
    fn to_arrays<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<u64>>, Bound<'py, PyArray2<u64>>, Bound<'py, PyAny>) {
        let (w, x_flat, z_flat, coeffs) = self.inner.columns(py);
        let n = x_flat.len() / w;
        let reshape = |flat: Vec<u64>| {
            flat.into_pyarray_bound(py).reshape([n, w]).expect("flat length is n*w by construction")
        };
        (reshape(x_flat), reshape(z_flat), coeffs)
    }

    /// Active monomorphized width `W` (number of `u64` words per term).
    /// Useful when paired with `x_array` / `z_array` for downstream bit-twiddling.
    #[getter]
    fn width(&self) -> usize {
        self.inner.xz_flat().0
    }

    /// Propagate `self` through `circuit`, in place.
    ///
    /// The engine takes the terms over instead of copying them, so propagate a `copy()` to keep the input; if the run raises, `self` is left empty.
    /// `direction` is `"forward"` (default) or `"heisenberg"`. `policy` is an optional `Truncation`; `None` applies no per-term filtering beyond the engine's own exact-zero drop.
    /// `target_bucket_len` (at least 1) and `min_buckets` (at least 16) are the engine's per-layer bucket-sizing knobs, raising `ValueError` below those bounds. `PauliSum.num_buckets` reads back the realized count, which can differ from a request since the count moves only outside a gap of 8× around the target.
    /// The sum keeps its dtype; a real dtype raises `ValueError` unless every channel of `circuit` preserves Hermiticity, as every `gates` and `noise` factory does.
    /// The GIL is released for the duration.
    ///
    /// `partitions` splits the sum across NUMA domains: `None`/`1` (default) is unpartitioned; `"auto"` is one partition per NUMA node; an `int` power of two caps it at that many nodes; `list[list[int]]` gives explicit disjoint CPU lists. `pin_memory` (default `True`) binds each partition's allocations to its node.
    /// In partitioned mode `RAYON_NUM_THREADS` is ignored, and `truncation.topn` raises `NotImplementedError` (use `approx_topn`).
    ///
    /// `partition_row_seed` picks which GF(2) rows decide a term's partition (`None`, the default, falls back to the sum's own hash seed — unchanged from before this knob existed). `partition_row_blocks`, an alternative to the seed, gives one disjoint qubit block per partition (`PartitionRows::cut`, e.g. `[[0, ..., 63], [64, ..., 126]]` for a 2-partition cut) — a term's partition is then the XOR of the blocks in which it has odd Z-weight, a locality cut rather than a GF(2)-random draw. The two are mutually exclusive with each other, and either needs `partitions=` or `comm=`.
    /// Under `comm=` the block count must equal the MPI group size, and the blocks must be identical on every rank (the split is a local filter every rank computes for itself).
    /// `partition_row_exclude`, a dict `{"x": [qubits], "z": [qubits]}` and the other alternative to `partition_row_blocks`, keeps the seeded draw (`partition_row_seed` still applies) but clears the named x-bits and z-bits from every row, so strings differing only there share a partition.
    /// `partition_rows`, a boolean (or 0/1) array of shape `(log2(P), 2 * num_qubits)` laid out as `hash_rows()` (see `paulistrings.hashes`), gives the rows outright and takes no other row keyword; under `comm=` every rank must pass the same rows.
    /// Rows that leave partitions unreachable run with a `RuntimeWarning`; rows that overlap the span of the bucket hash's active rows are logged as a warning (`paulistrings` logger) at the layer whose bucket count reaches the overlap.
    ///
    /// `comm` takes an `mpi4py` communicator and runs one partition per rank, as an alternative to `partitions` (place via the launcher, e.g. `mpirun --map-by ppr:1:numa --bind-to numa`). Requires `MPI_THREAD_SERIALIZED` set before importing MPI, a power-of-two rank count, and every rank calling with the same replicated input in the same order.
    /// `result="gather"` (default) leaves the whole sum on rank 0 and an empty one elsewhere; `"local"` leaves each rank its own disjoint share. Raises `RuntimeError` without the `mpi` feature.
    ///
    /// `device` runs the propagation on CUDA devices: an `int` ordinal holds the whole sum on that device, uploaded before the first layer and downloaded after the last.
    /// A `list[int]` of a power-of-two length places one partition per entry, split and exchanged like `partitions=` (`partition_row_seed`/`partition_row_blocks` apply, and a repeated ordinal puts several partitions on one device); `"auto"` takes devices `0..k` for the largest power of two `k` visible.
    /// With `comm`, each MPI rank drives one device: an `int` ordinal or `"auto"` (a device near the rank's CPUs, distinct per rank on a node while there are enough), with `result=` as for a host `comm=` run; this needs both the `cuda` and `mpi` features.
    /// `device` is an alternative to `partitions`, runs `complex128` sums only (`TypeError` for another dtype), and raises `RuntimeError` without the `cuda` feature. `truncation.topn` runs exactly when `device` resolves to one device (an `int`, or `"auto"`/a one-entry list on a one-GPU box); a device list of more than one entry or `comm=` with `device=` raises `NotImplementedError` (use `approx_topn`), since the `n`-th largest of a split sum has no collective form. `PauliSum.to_device` keeps a one-device sum resident across calls instead.
    ///
    /// ```python
    /// observable.propagate(circuit, policy, direction="heisenberg", partitions="auto")
    /// ```
    #[pyo3(signature = (circuit, policy=None, direction=None, target_bucket_len=1024, min_buckets=128, partitions=None, pin_memory=true, partition_row_seed=None, partition_row_blocks=None, partition_row_exclude=None, partition_rows=None, comm=None, result="gather", device=None))]
    #[allow(clippy::too_many_arguments)]
    fn propagate(
        &mut self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
        target_bucket_len: usize,
        min_buckets: usize,
        partitions: Option<&Bound<'_, PyAny>>,
        pin_memory: bool,
        partition_row_seed: Option<u64>,
        partition_row_blocks: Option<&Bound<'_, PyAny>>,
        partition_row_exclude: Option<&Bound<'_, PyAny>>,
        partition_rows: Option<&Bound<'_, PyAny>>,
        comm: Option<&Bound<'_, PyAny>>,
        result: &str,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let dir = parse_direction(direction)?;
        let options = propagate_options(target_bucket_len, min_buckets)?;
        let gather = parse_result(result)?;
        if !gather {
            warn_local_result(py)?;
        }
        check_num_qubits("PauliSum", self.inner.num_qubits(), circuit)?;
        check_dtype_carries("PauliSum", self.inner.dtype(), circuit)?;
        let policy = PyTruncation::tree_of(policy);
        // Last, because adopting a communicator is collective: every check
        // above raises on all ranks alike, before any of them has entered MPI.
        let mode = parse_run_mode(
            py,
            partitions,
            pin_memory,
            RowKwargs::parse(
                py,
                partition_row_seed,
                partition_row_blocks,
                partition_row_exclude,
                partition_rows,
                self.inner.num_qubits(),
            )?,
            comm,
            gather,
            device,
            &policy,
            &self.inner,
        )?;
        self.run(py, circuit, mode, &policy, dir, options, false)?;
        Ok(())
    }

    /// Propagate `self` through `circuit` in place, returning a `PropagationStats`.
    ///
    /// Arguments and semantics are `propagate`'s; the only difference is that the engine also records per-layer term counts (before each layer, and after its truncation), so the evolved sum agrees with `propagate`'s to floating-point tolerance. See `PropagationStats.peak_terms` for what "peak" does and does not mean.
    /// `PropagationStats.partition` holds the per-partition detail, summed to the layer-level `terms_in`/`terms_out`; an unpartitioned call reports one partition.
    /// A distributed (`comm=`) call fills it too, but its per-layer lists hold **this rank's entry only** — gathering the group's counters would add a collective per layer for a diagnostic. Reduce over `comm` for the group's picture.
    /// A device (`device=`) call fills it with one partition per device, `PartitionStats.devices` naming them; under `comm=` it is this rank's record with this rank's device.
    #[pyo3(signature = (circuit, policy=None, direction=None, target_bucket_len=1024, min_buckets=128, partitions=None, pin_memory=true, partition_row_seed=None, partition_row_blocks=None, partition_row_exclude=None, partition_rows=None, comm=None, result="gather", device=None))]
    #[allow(clippy::too_many_arguments)]
    fn propagate_with_stats(
        &mut self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
        target_bucket_len: usize,
        min_buckets: usize,
        partitions: Option<&Bound<'_, PyAny>>,
        pin_memory: bool,
        partition_row_seed: Option<u64>,
        partition_row_blocks: Option<&Bound<'_, PyAny>>,
        partition_row_exclude: Option<&Bound<'_, PyAny>>,
        partition_rows: Option<&Bound<'_, PyAny>>,
        comm: Option<&Bound<'_, PyAny>>,
        result: &str,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PropagationStats> {
        let dir = parse_direction(direction)?;
        let options = propagate_options(target_bucket_len, min_buckets)?;
        let gather = parse_result(result)?;
        if !gather {
            warn_local_result(py)?;
        }
        check_num_qubits("PauliSum", self.inner.num_qubits(), circuit)?;
        check_dtype_carries("PauliSum", self.inner.dtype(), circuit)?;
        let policy = PyTruncation::tree_of(policy);
        // Last, because adopting a communicator is collective: every check
        // above raises on all ranks alike, before any of them has entered MPI.
        let mode = parse_run_mode(
            py,
            partitions,
            pin_memory,
            RowKwargs::parse(
                py,
                partition_row_seed,
                partition_row_blocks,
                partition_row_exclude,
                partition_rows,
                self.inner.num_qubits(),
            )?,
            comm,
            gather,
            device,
            &policy,
            &self.inner,
        )?;
        let (trace, collapses) = self.run(py, circuit, mode, &policy, dir, options, true)?;
        Ok(PropagationStats::from_run_trace(
            trace.expect("a traced run always records a trace"),
            self.inner.len(),
            collapses,
        ))
    }

    /// Split `self` into a `PartitionedPauliSum` that stays split between calls, so a run can change its partition or bucket rows mid-circuit without gathering; `self` is left untouched.
    ///
    /// `partitions` (default `"auto"`) and `pin_memory` place the partitions in this process as for `propagate`, where one partition is a one-pool split; `comm=` instead splits over the ranks of an MPI group, one partition per rank, and is then collective.
    /// The row keywords are `propagate`'s and pick the initial partition rows.
    #[pyo3(signature = (partitions=None, pin_memory=true, partition_row_seed=None, partition_row_blocks=None, partition_row_exclude=None, partition_rows=None, comm=None))]
    #[allow(clippy::too_many_arguments)]
    fn partition(
        &self,
        py: Python<'_>,
        partitions: Option<&Bound<'_, PyAny>>,
        pin_memory: bool,
        partition_row_seed: Option<u64>,
        partition_row_blocks: Option<&Bound<'_, PyAny>>,
        partition_row_exclude: Option<&Bound<'_, PyAny>>,
        partition_rows: Option<&Bound<'_, PyAny>>,
        comm: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<crate::partitioned::PartitionedPauliSum> {
        let rows = RowKwargs::parse(
            py,
            partition_row_seed,
            partition_row_blocks,
            partition_row_exclude,
            partition_rows,
            self.inner.num_qubits(),
        )?;
        crate::partitioned::partition(py, &self.inner, partitions, pin_memory, rows, comm)
    }

    /// Upload `self` to CUDA device `device`, returning a resident `GpuPauliSum`; `self` is left untouched.
    ///
    /// The resident sum is stepped in place by `GpuPauliSum.propagate` and read back by `GpuPauliSum.to_host`, so a loop of many short propagations pays one upload and one download rather than one of each per call.
    /// Raises `RuntimeError` without the `cuda` feature or with no visible device, `TypeError` for a dtype other than `complex128`, `ValueError` for an ordinal this process cannot see, and `MemoryError` if the device cannot hold the sum.
    #[pyo3(signature = (device=0))]
    fn to_device(&self, py: Python<'_>, device: i64) -> PyResult<PyObject> {
        crate::gpu::to_device(py, &self.inner, device)
    }
}

impl PauliSum {
    /// The GIL-released half of both propagate entry points: move the sum out of `self`, run `mode` over it and put the result back, so no copy of the input stays alive.
    /// `self` is left empty if the run fails.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        mode: RunMode,
        policy: &BuiltinTruncation,
        dir: Direction,
        options: PropagateOptions,
        traced: bool,
    ) -> PyResult<(Option<RunTrace>, Option<u64>)> {
        let empty = PauliSumImpl::empty_for(self.inner.num_qubits(), self.inner.dtype())
            .expect("the sum's own width and dtype");
        let input = std::mem::replace(&mut self.inner, empty);
        // The whole simulation runs without the GIL: everything the engine
        // touches is plain Rust data (`PauliSumImpl`, `CircuitImpl` and
        // `BuiltinTruncation` are all `Send + Sync`), so nothing here needs Python.
        // Releasing it lets Python `logging` handlers — the consumers of the
        // engine's per-layer progress records, bridged by `pyo3-log` — and any
        // other Python thread run while a long propagate is in flight.
        //
        // The closure is `move` (it owns `mode`, whose distributed variant
        // owns the communicator), so the side outputs escape through borrows
        // taken before it; the (unreachable) width-mismatch arm bails out with
        // an error turned into a Python exception after the GIL is reacquired.
        let (mut trace, mut collapses) = (None, None);
        let slots = (&mut trace, &mut collapses);
        let inner = py.allow_threads(move || -> Result<PauliSumImpl, PropagateFailure> {
            Ok(for_each_dtype_convert!(PauliSumImpl => PauliSumImpl, input, |sum| {
                for_each_width_propagate!(
                    WidthSum,
                    sum,
                    &circuit.inner,
                    |s, c, W, wrap| {
                        let out = mode.run::<W, _>(c, s, policy, dir, options, traced)?;
                        *slots.0 = out.trace;
                        *slots.1 = out.collapses;
                        wrap(out.sum)
                    },
                    else {
                        return Err(PropagateFailure::WidthMismatch);
                    }
                )
            }))
        })?;
        self.inner = inner;
        Ok((trace, collapses))
    }
}

// `parse_partitions`/`parse_run_mode` themselves are not unit-tested here:
// exercising the `partitions="auto"`/`int` branches needs a live `Bound<PyAny>`,
// which needs a real Python runtime linked in — this crate is built
// `extension-module`-only (loaded *by* Python, never embedding it), so a
// `Python::with_gil` call in a `cargo test` binary fails to link (undefined
// `PyErr_*`/`PyUnicode_*` symbols that the embedding interpreter would
// normally provide). The seed/blocks plumbing itself (`PartitionConfig`
// construction, `validate_partition_row_blocks`) is
// plain Rust and tested below; the end-to-end Python-facing behavior is
// covered by `python/paulistrings/tests/test_partitioned.py`.
#[cfg(test)]
mod partition_row_knob_tests {
    use super::*;
    use paulistrings::PartitionRows;

    /// Explicit "cut" blocks round-trip through `validate_partition_row_blocks` +
    /// `PartitionRows::cut` into a `PartitionRows` that actually assigns qubits to the
    /// blocks named, and two different cuts assign at least one term to different partitions.
    #[test]
    fn explicit_cut_blocks_round_trip_and_differ_from_each_other() {
        use paulistrings::PauliString;

        let num_qubits = 4;
        let num_partitions = 2;

        let half_low = vec![vec![0usize, 1], vec![2usize, 3]];
        validate_partition_row_blocks_impl(&half_low, num_qubits, num_partitions, "partitions=2")
            .expect("two disjoint blocks covering all 4 qubits validate cleanly");
        let rows_low = PartitionRows::<1>::cut(num_qubits, &half_low);

        let half_alt = vec![vec![0usize, 2], vec![1usize, 3]];
        validate_partition_row_blocks_impl(&half_alt, num_qubits, num_partitions, "partitions=2")
            .expect("an alternative disjoint cut also validates cleanly");
        let rows_alt = PartitionRows::<1>::cut(num_qubits, &half_alt);

        // Round-trip: qubit 2 sits in block 1 under `half_low`, block 0 under `half_alt`.
        let z2 = PauliString::<1>::z(2);
        assert_eq!(rows_low.partition_of(&z2.x, &z2.z), 1);
        assert_eq!(rows_alt.partition_of(&z2.x, &z2.z), 0);

        // The two cuts disagree on at least this term, so they are genuinely different row
        // sets, not two spellings of the same partition.
        assert_ne!(rows_low.partition_of(&z2.x, &z2.z), rows_alt.partition_of(&z2.x, &z2.z));
    }

    /// A block count that doesn't match the partition count is a `ValueError`, not a panic —
    /// the whole point of validating before `allow_threads` releases the GIL.
    #[test]
    fn mismatched_block_count_is_a_value_error_not_a_panic() {
        let one_block = vec![vec![0usize, 1, 2, 3]];
        let err = validate_partition_row_blocks_impl(&one_block, 4, 2, "partitions=2")
            .expect_err("1 block for 2 partitions must be rejected");
        assert!(err.contains("partition_row_blocks"));
        assert!(err.contains("partitions=2"));
    }

    /// A block set `PartitionRows::cut` would panic on is a `ValueError` too:
    /// a partition bit no qubit can set leaves half the partitions empty.
    #[test]
    fn a_block_set_that_cannot_name_every_partition_is_a_value_error() {
        let empty_second = vec![vec![0usize, 1, 2, 3], Vec::new()];
        let err = validate_partition_row_blocks_impl(&empty_second, 4, 2, "partitions=2")
            .expect_err("an empty block 1 leaves partition bit 0 constant");
        assert!(err.contains("bit 0"), "{err}");

        // Block 1 empty out of four is fine — block 3 still sets bit 0.
        let one_empty = vec![vec![0usize], Vec::new(), vec![1usize], vec![2usize]];
        validate_partition_row_blocks_impl(&one_empty, 4, 4, "partitions=4")
            .expect("every bit is set by some non-empty block");
    }

    /// A placement whose partition count is not a power of two cannot be cut by
    /// `log2(P)` rows at all.
    #[test]
    fn a_partition_count_that_is_not_a_power_of_two_is_a_value_error() {
        let three = vec![vec![0usize], vec![1usize], vec![2usize]];
        let err = validate_partition_row_blocks_impl(&three, 3, 3, "partitions=3")
            .expect_err("three partitions cannot be named by GF(2) rows");
        assert!(err.contains("power of two"), "{err}");
    }

    /// The count a distributed run validates against is the MPI group size, so
    /// the message names the group rather than a `partitions=` the caller never
    /// passed.
    #[test]
    fn a_mismatched_block_count_names_the_group_under_comm() {
        let two_blocks = vec![vec![0usize, 1], vec![2usize, 3]];
        let err = validate_partition_row_blocks_impl(&two_blocks, 4, 4, "comm= with 4 rank(s)")
            .expect_err("2 blocks for a 4-rank group must be rejected");
        assert!(err.contains("comm= with 4 rank(s)"), "{err}");
    }

    /// A qubit named in two blocks is rejected before it ever reaches `PartitionRows::cut`
    /// (which would otherwise panic on the same condition).
    #[test]
    fn overlapping_blocks_are_a_value_error() {
        let overlapping = vec![vec![0usize, 1], vec![1usize, 2]];
        let err = validate_partition_row_blocks_impl(&overlapping, 3, 2, "partitions=2")
            .expect_err("qubit 1 in two blocks must be rejected");
        assert!(err.contains("more than one block"));
    }
}

#[cfg(test)]
mod propagate_options_tests {
    use super::*;

    #[test]
    fn literal_defaults_are_the_engines() {
        let literal = PropagateOptions { target_bucket_len: 1024, min_buckets: 128 };
        assert_eq!(literal, PropagateOptions::default());
    }
}

#[cfg(test)]
mod tests;
