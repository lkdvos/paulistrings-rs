//! Python `PartitionedPauliSum`: a sum kept split across calls, over NUMA partitions in this process or over the ranks of an MPI group.

use num_complex::Complex64;
use numpy::PyArray2;
use paulistrings::{
    max_hash_bits, BuiltinTruncation, Circuit as CoreCircuit, Direction, Gf2Hash, PartitionConfig,
    PartitionRowPolicy, PartitionRows, PartitionedSum, PauliSum as CorePauliSum, Placement,
    ProductBasis, PropagateOptions, ScatterOptions, ScatterRows,
};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::sum::{
    check_num_qubits, coordinate_flags, parse_direction, parse_partitions, parse_product_state,
    parse_row_matrix, propagate_options, runtime_for, topology_error, warn_partition_rows,
    warn_runtime, PauliSum, PauliSumImpl, ProductState, PropagateFailure, PropagationStats,
    RowKwargs, RunTrace, TracePlacement,
};
use crate::truncation_spec::{collapse_count, collapses_since, PyTruncation, TOPN_PARTITIONED_MSG};

/// One width's split: pinned pools over this process's NUMA domains, or this rank's share of an MPI group.
pub(crate) enum Split<const W: usize> {
    Process(PartitionedSum<W>),
    #[cfg(feature = "mpi")]
    Ranks(Box<paulistrings::mpi::MpiSum<W>>),
}

impl<const W: usize> Split<W> {
    fn rows(&self) -> &PartitionRows<W> {
        match self {
            Split::Process(split) => split.rows(),
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => split.rows(),
        }
    }

    fn rank(&self) -> Option<u32> {
        match self {
            Split::Process(_) => None,
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => Some(split.rank()),
        }
    }

    /// The bucket hash, the same on every partition and rank.
    fn hash(&self) -> &Gf2Hash<W> {
        match self {
            Split::Process(split) => split.partition(0).hash(),
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => split.local().hash(),
        }
    }

    /// Terms in the whole sum. **Collective** under MPI.
    fn len(&self) -> usize {
        match self {
            Split::Process(split) => split.len(),
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => split.len(),
        }
    }

    /// Terms this process holds: the whole sum in one process, this rank's share under MPI.
    fn held(&self) -> usize {
        match self {
            Split::Process(split) => split.len(),
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => split.local().len(),
        }
    }

    /// The terms this process holds: every partition's in one process, this rank's under MPI.
    fn local(&self) -> CorePauliSum<W> {
        match self {
            Split::Process(split) => split.gather(),
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => split.local().clone(),
        }
    }

    /// The whole sum here, and on rank 0 only under MPI (an empty sum elsewhere). **Collective** under MPI.
    fn gather(&self) -> CorePauliSum<W> {
        match self {
            Split::Process(split) => split.gather(),
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => {
                split.gather().unwrap_or_else(|| CorePauliSum::empty(split.num_qubits()))
            }
        }
    }

    /// The whole sum's expectation in `basis`, the same on every rank. **Collective** under MPI.
    fn expectation(&self, basis: &ProductBasis<W>) -> Complex64 {
        match self {
            Split::Process(split) => split.expectation_product(basis),
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => {
                use paulistrings::Collectives;
                let value = split.local().expectation_product(basis);
                let mut parts = [value.re, value.im];
                split.transport().allreduce_sum_f64(&mut parts);
                Complex64::new(parts[0], parts[1])
            }
        }
    }

    /// Step through `circuit`, returning this call's trace when `traced`, and its collapses. **Collective** under MPI.
    fn propagate(
        &mut self,
        circuit: &CoreCircuit<W>,
        policy: &BuiltinTruncation,
        direction: Direction,
        options: PropagateOptions,
        traced: bool,
    ) -> (Option<RunTrace>, Option<u64>) {
        let before = collapse_count(policy);
        match self {
            Split::Process(split) => {
                with_lowered_policy!(policy, |lowered| split
                    .propagate_with(circuit, lowered, direction, options));
                // Drained every call, so a split stepped many times never accumulates records.
                let trace = split.take_trace().unwrap_or_default();
                let placement = TracePlacement::partitions(split.num_partitions());
                (traced.then_some(RunTrace { trace, placement }), collapses_since(policy, before))
            }
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => {
                with_lowered_policy!(policy, |lowered| split
                    .propagate_with(circuit, lowered, direction, options));
                let collapses = crate::mpi::agreed_collapses(policy, before, split.transport());
                let trace = split.take_trace().unwrap_or_default();
                let placement = TracePlacement {
                    partitions: split.size() as usize,
                    ranks: Some((split.rank(), split.size())),
                    devices: None,
                };
                (traced.then_some(RunTrace { trace, placement }), collapses)
            }
        }
    }

    /// The rows `policy` names for this split's partition count.
    fn resolve(&self, policy: &PartitionRowPolicy) -> PartitionRows<W> {
        let bits = self.rows().num_partitions().trailing_zeros() as u8;
        policy.rows::<W>(self.rows().num_qubits(), bits, self.hash().seed())
    }

    /// Move every term to the partition `rows` names. **Collective** under MPI.
    fn repartition(&mut self, rows: PartitionRows<W>) {
        match self {
            Split::Process(split) => split.repartition(rows),
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => split.repartition(rows),
        }
    }

    /// Rebucket every partition by `hash`. **Collective** under MPI.
    fn rehash(&mut self, hash: Gf2Hash<W>) {
        match self {
            Split::Process(split) => split.rehash_buckets(hash),
            #[cfg(feature = "mpi")]
            Split::Ranks(split) => split.rehash_buckets(hash),
        }
    }
}

pub(crate) enum SplitImpl {
    W1(Split<1>),
    W2(Split<2>),
    W4(Split<4>),
    W8(Split<8>),
    W16(Split<16>),
}

impl SplitImpl {
    fn num_qubits(&self) -> usize {
        for_each_width!(self, |s| s.rows().num_qubits())
    }

    fn num_partitions(&self) -> usize {
        for_each_width!(self, |s| s.rows().num_partitions())
    }

    fn rank(&self) -> Option<u32> {
        for_each_width!(self, |s| s.rank())
    }

    fn num_buckets(&self) -> usize {
        for_each_width!(self, |s| s.hash().num_buckets())
    }

    fn hash_seed(&self) -> u64 {
        for_each_width!(self, |s| s.hash().seed())
    }

    fn len(&self) -> usize {
        for_each_width!(self, |s| s.len())
    }

    fn held(&self) -> usize {
        for_each_width!(self, |s| s.held())
    }

    fn partition_flags(&self) -> Vec<Vec<bool>> {
        let num_qubits = self.num_qubits();
        for_each_width!(self, |s| {
            let (rows_x, rows_z) = s.rows().rows();
            rows_x.iter().zip(rows_z).map(|(x, z)| coordinate_flags(x, z, num_qubits)).collect()
        })
    }

    fn hash_flags(&self) -> Vec<Vec<bool>> {
        let num_qubits = self.num_qubits();
        for_each_width!(self, |s| {
            let (rows_x, rows_z) = s.hash().rows();
            rows_x.iter().zip(rows_z).map(|(x, z)| coordinate_flags(x, z, num_qubits)).collect()
        })
    }

    fn expectation(&self, state: &ProductState) -> Complex64 {
        for_each_width!(self, |s| s.expectation(&state.basis()))
    }

    fn gather(&self) -> PauliSumImpl {
        for_each_width_convert!(SplitImpl => PauliSumImpl, self, |s| s.gather())
    }

    fn local(&self) -> PauliSumImpl {
        for_each_width_convert!(SplitImpl => PauliSumImpl, self, |s| s.local())
    }

    /// Why the rows `policy` names leave partitions unreachable, if they do.
    fn partition_shortfall(&self, policy: &PartitionRowPolicy) -> Option<String> {
        for_each_width!(self, |s| s.resolve(policy).rank_deficiency())
    }

    fn repartition(&mut self, policy: &PartitionRowPolicy) {
        for_each_width!(self, |s| {
            let rows = s.resolve(policy);
            s.repartition(rows)
        })
    }

    /// The hash `rows` lead and dense rows from `seed` (default: the current hash's) complete, at the current bucket count.
    fn rehash_target(&self, rows: &[Vec<bool>], seed: Option<u64>) -> HashTarget {
        let num_qubits = self.num_qubits();
        for_each_width_convert!(SplitImpl => HashTarget, self, |s| {
            let current = s.hash();
            Gf2Hash::from_matrix(num_qubits, current.bits(), seed.unwrap_or(current.seed()), rows)
        })
    }

    fn rehash(&mut self, target: HashTarget) -> Result<(), PropagateFailure> {
        match (self, target) {
            (SplitImpl::W1(s), HashTarget::W1(hash)) => s.rehash(hash),
            (SplitImpl::W2(s), HashTarget::W2(hash)) => s.rehash(hash),
            (SplitImpl::W4(s), HashTarget::W4(hash)) => s.rehash(hash),
            (SplitImpl::W8(s), HashTarget::W8(hash)) => s.rehash(hash),
            (SplitImpl::W16(s), HashTarget::W16(hash)) => s.rehash(hash),
            _ => return Err(PropagateFailure::WidthMismatch),
        }
        Ok(())
    }
}

/// A bucket hash at the width of the split it was built for.
enum HashTarget {
    W1(Gf2Hash<1>),
    W2(Gf2Hash<2>),
    W4(Gf2Hash<4>),
    W8(Gf2Hash<8>),
    W16(Gf2Hash<16>),
}

impl HashTarget {
    fn rank_deficiency(&self) -> Option<String> {
        for_each_width!(self, |hash| hash.rank_deficiency())
    }
}

/// `PauliSum.partition`'s body: split `sum` by the parsed `rows`, over `comm`'s ranks when given, else over the placement `partitions` names.
/// Every check that can raise runs before the communicator is adopted, so a rejected call raises on every rank alike.
pub(crate) fn partition(
    py: Python<'_>,
    sum: &PauliSumImpl,
    partitions: Option<&Bound<'_, PyAny>>,
    pin_memory: bool,
    rows: RowKwargs,
    comm: Option<&Bound<'_, PyAny>>,
) -> PyResult<PartitionedPauliSum> {
    let num_qubits = sum.num_qubits();
    if let Some(comm) = comm.filter(|comm| !comm.is_none()) {
        if partitions.is_some_and(|obj| !obj.is_none()) {
            return Err(PyValueError::new_err(
                "comm= and partitions= are alternatives: comm= already places one partition per MPI rank",
            ));
        }
        return partition_ranks(py, sum, rows, comm);
    }
    let auto = PyString::new_bound(py, "auto").into_any();
    let placement = partitions.filter(|obj| !obj.is_none()).unwrap_or(&auto);
    // `propagate` runs one partition unpartitioned; a split keeps it as a one-pool split, which still repartitions and rehashes.
    let config = parse_partitions(Some(placement), pin_memory)?.unwrap_or(PartitionConfig {
        placement: Placement::Auto { max_partitions: Some(1) },
        bind_memory: pin_memory,
        threads_per_partition: None,
    });
    let count = config.num_partitions().map_err(topology_error)?;
    rows.validate_count(num_qubits, count, &format!("{count} partition(s)"))?;
    let policy = rows.policy();
    warn_partition_rows(py, sum, &policy, count)?;
    let runtime = runtime_for(&config).map_err(topology_error)?;
    let inner = py.allow_threads(move || {
        for_each_width_convert!(PauliSumImpl => SplitImpl, sum, |s| {
            let options = ScatterOptions { runtime, rows: ScatterRows::Policy(policy) };
            let mut split = PartitionedSum::scatter_with(s.clone(), options);
            split.enable_trace();
            Split::Process(split)
        })
    });
    Ok(PartitionedPauliSum { inner })
}

#[cfg(feature = "mpi")]
fn partition_ranks(
    py: Python<'_>,
    sum: &PauliSumImpl,
    rows: RowKwargs,
    comm: &Bound<'_, PyAny>,
) -> PyResult<PartitionedPauliSum> {
    let num_qubits = sum.num_qubits();
    let ranks = crate::mpi::comm_size(comm)?;
    if !ranks.is_power_of_two() {
        return Err(PyValueError::new_err(format!(
            "comm= has {ranks} ranks, but a partition is named by log2(P) GF(2) rows, so the group size must be a power of two"
        )));
    }
    rows.validate_count(num_qubits, ranks, &format!("comm= with {ranks} rank(s)"))?;
    let policy = rows.policy();
    warn_partition_rows(py, sum, &policy, ranks)?;
    let transport = crate::mpi::transport_from_comm(py, comm)?;
    let inner = py.allow_threads(move || -> Result<SplitImpl, paulistrings::TopologyError> {
        Ok(for_each_width_convert!(PauliSumImpl => SplitImpl, sum, |s| {
            let mut split = crate::mpi::scatter(s.clone(), transport, policy)?;
            split.enable_trace();
            Split::Ranks(Box::new(split))
        }))
    });
    Ok(PartitionedPauliSum { inner: inner.map_err(topology_error)? })
}

#[cfg(not(feature = "mpi"))]
fn partition_ranks(
    _: Python<'_>,
    _: &PauliSumImpl,
    _: RowKwargs,
    _: &Bound<'_, PyAny>,
) -> PyResult<PartitionedPauliSum> {
    Err(crate::sum::mpi_unavailable_error())
}

/// A `PauliSum` kept split between calls: over this process's NUMA domains (`PauliSum.partition(partitions=...)`) or over the ranks of an MPI group (`PauliSum.partition(comm=...)`).
///
/// Built by `PauliSum.partition`; there is no public constructor.
/// `propagate` steps every partition in place, so a run can change its partition rows (`repartition`) or bucket rows (`rehash`) between segments of a circuit without gathering, for instance to keep the next gates' communication low (`paulistrings.hashes.remote_layers`).
/// Under MPI every method that reads or changes the whole sum is collective, `len()` and `expectation` included, and every rank must make the same calls with the same arguments; `repr()` and the properties are local.
///
/// ```python
/// split = observable.partition(partitions="auto", partition_rows=first_rows)
/// split.propagate(circuit[:40], policy, direction="heisenberg")
/// split.repartition(partition_rows=second_rows)
/// split.propagate(circuit[40:], policy, direction="heisenberg")
/// value = split.expectation("z+")
/// ```
#[pyclass(module = "paulistrings._paulistrings", name = "PartitionedPauliSum")]
pub struct PartitionedPauliSum {
    inner: SplitImpl,
}

#[pymethods]
impl PartitionedPauliSum {
    /// Propagate every partition through `circuit`, in place; arguments are `PauliSum.propagate`'s of the same name.
    /// Exact `truncation.topn` raises `NotImplementedError`, as for any partitioned run (use `approx_topn`). The GIL is released for the duration.
    #[pyo3(signature = (circuit, policy=None, direction=None, target_bucket_len=1024, min_buckets=128))]
    fn propagate(
        &mut self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
        target_bucket_len: usize,
        min_buckets: usize,
    ) -> PyResult<()> {
        self.step(py, circuit, policy, direction, target_bucket_len, min_buckets, false)?;
        Ok(())
    }

    /// `propagate`, returning a `PropagationStats` for this call's layers; under MPI its per-layer lists hold this rank's entry only, as for `PauliSum.propagate_with_stats(comm=...)`.
    #[pyo3(signature = (circuit, policy=None, direction=None, target_bucket_len=1024, min_buckets=128))]
    fn propagate_with_stats(
        &mut self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
        target_bucket_len: usize,
        min_buckets: usize,
    ) -> PyResult<PropagationStats> {
        let (trace, collapses) =
            self.step(py, circuit, policy, direction, target_bucket_len, min_buckets, true)?;
        Ok(PropagationStats::from_run_trace(
            trace.expect("a traced step always records a trace"),
            self.inner.held(),
            collapses,
        ))
    }

    /// Move every term to the partition new rows name, in one exchange between all partitions and without gathering; the bucket rows stay.
    /// Takes `PauliSum.propagate`'s row keywords (`partition_row_seed`, `partition_row_blocks`, `partition_row_exclude`, `partition_rows`), defaulting to the seeded dense draw at the sum's own hash seed, and warns about dependent rows as it does.
    /// Under MPI every rank must pass the same rows; a disagreement fails on every rank.
    #[pyo3(signature = (*, partition_row_seed=None, partition_row_blocks=None, partition_row_exclude=None, partition_rows=None))]
    fn repartition(
        &mut self,
        py: Python<'_>,
        partition_row_seed: Option<u64>,
        partition_row_blocks: Option<&Bound<'_, PyAny>>,
        partition_row_exclude: Option<&Bound<'_, PyAny>>,
        partition_rows: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let num_qubits = self.inner.num_qubits();
        let count = self.inner.num_partitions();
        let rows = RowKwargs::parse(
            py,
            partition_row_seed,
            partition_row_blocks,
            partition_row_exclude,
            partition_rows,
            num_qubits,
        )?;
        rows.validate_count(num_qubits, count, &format!("{count} partition(s)"))?;
        let policy = rows.policy();
        if let Some(reason) = self.inner.partition_shortfall(&policy) {
            warn_runtime(py, &reason)?;
        }
        let inner = &mut self.inner;
        py.allow_threads(move || inner.repartition(&policy));
        Ok(())
    }

    /// Rebucket every partition by another GF(2)-linear hash, as `PauliSum.with_hash` takes it: `rows` a boolean (or 0/1) array `(k, 2 * num_qubits)` leading the hash, dense rows drawn from `seed` (default: the current hash's) filling the rest, at the current bucket count.
    /// No term changes partition, so nothing is exchanged; under MPI every rank must pass the same rows (checked). Dependent rows warn.
    #[pyo3(signature = (rows=None, seed=None))]
    fn rehash(
        &mut self,
        py: Python<'_>,
        rows: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let num_qubits = self.inner.num_qubits();
        let rows = match rows.filter(|obj| !obj.is_none()) {
            Some(rows) => {
                parse_row_matrix(py, rows, num_qubits, max_hash_bits(num_qubits) as usize, "rows")?
            }
            None => Vec::new(),
        };
        let target = self.inner.rehash_target(&rows, seed);
        if let Some(reason) = target.rank_deficiency() {
            warn_runtime(py, &reason)?;
        }
        let inner = &mut self.inner;
        py.allow_threads(move || inner.rehash(target))?;
        Ok(())
    }

    /// The whole sum as a `PauliSum`: here, or on rank 0 under MPI (an empty sum on every other rank). **Collective** under MPI.
    fn gather(&self, py: Python<'_>) -> PauliSum {
        let inner = &self.inner;
        PauliSum::plain(py.allow_threads(move || inner.gather()))
    }

    /// The terms this process holds as a `PauliSum`: the whole sum in one process, this rank's share under MPI; not collective.
    fn local(&self, py: Python<'_>) -> PauliSum {
        let inner = &self.inner;
        PauliSum::plain(py.allow_threads(move || inner.local()))
    }

    /// The whole sum's expectation value in a product state, as `PauliSum.expectation` takes it; the same value on every rank. **Collective** under MPI.
    #[pyo3(signature = (state="x+"))]
    fn expectation(&self, py: Python<'_>, state: &str) -> PyResult<Complex64> {
        let parsed = parse_product_state(state, self.inner.num_qubits())?;
        let inner = &self.inner;
        Ok(py.allow_threads(move || inner.expectation(&parsed)))
    }

    /// Terms in the whole sum. **Collective** under MPI.
    fn __len__(&self, py: Python<'_>) -> usize {
        let inner = &self.inner;
        py.allow_threads(move || inner.len())
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Partitions the sum is split into: NUMA partitions in one process, the group size under MPI.
    #[getter]
    fn num_partitions(&self) -> usize {
        self.inner.num_partitions()
    }

    /// This process's rank under MPI, `None` for a split in one process.
    #[getter]
    fn rank(&self) -> Option<u32> {
        self.inner.rank()
    }

    /// The bucket count, equal on every partition.
    #[getter]
    fn num_buckets(&self) -> usize {
        self.inner.num_buckets()
    }

    /// The seed the bucket hash's drawn rows came from.
    #[getter]
    fn hash_seed(&self) -> u64 {
        self.inner.hash_seed()
    }

    /// The partition rows as a boolean array of shape `(log2(num_partitions), 2 * num_qubits)`, laid out as `PauliSum.hash_rows()`.
    fn partition_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<bool>>> {
        flag_array(py, self.inner.partition_flags(), 2 * self.inner.num_qubits())
    }

    /// The bucket hash's rows, as `PauliSum.hash_rows()` returns them.
    fn hash_rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<bool>>> {
        flag_array(py, self.inner.hash_flags(), 2 * self.inner.num_qubits())
    }

    fn __repr__(&self) -> String {
        let place = match self.inner.rank() {
            Some(rank) => format!("rank {rank} of {}", self.inner.num_partitions()),
            None => format!("{} partitions", self.inner.num_partitions()),
        };
        format!("PartitionedPauliSum(num_qubits={}, {place})", self.inner.num_qubits())
    }
}

impl PartitionedPauliSum {
    /// The shared body of `propagate` and `propagate_with_stats`: every check raises before the GIL is released, so under MPI a rejected call raises on every rank alike.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
        target_bucket_len: usize,
        min_buckets: usize,
        traced: bool,
    ) -> PyResult<(Option<RunTrace>, Option<u64>)> {
        let direction = parse_direction(direction)?;
        let options = propagate_options(target_bucket_len, min_buckets)?;
        check_num_qubits("PartitionedPauliSum", self.inner.num_qubits(), circuit)?;
        let policy = PyTruncation::tree_of(policy);
        if !policy.supports_partitioned() {
            return Err(PyNotImplementedError::new_err(format!(
                "PartitionedPauliSum.propagate: {TOPN_PARTITIONED_MSG}"
            )));
        }
        let inner = &mut self.inner;
        let outcome = py.allow_threads(move || -> Result<_, PropagateFailure> {
            Ok(for_each_width_propagate!(
                SplitImpl,
                inner,
                &circuit.inner,
                |s, c, _W, _wrap| s.propagate(c, &policy, direction, options, traced),
                else {
                    return Err(PropagateFailure::WidthMismatch);
                }
            ))
        })?;
        Ok(outcome)
    }
}

/// A `(rows, width)` boolean array, `width` columns even when there are no rows.
fn flag_array(
    py: Python<'_>,
    flags: Vec<Vec<bool>>,
    width: usize,
) -> PyResult<Bound<'_, PyArray2<bool>>> {
    if flags.is_empty() {
        return Ok(PyArray2::<bool>::zeros_bound(py, [0, width], false));
    }
    Ok(PyArray2::from_vec2_bound(py, &flags)?)
}
