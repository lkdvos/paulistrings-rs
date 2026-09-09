//! Python `PauliSum` class with width-monomorphized backing storage. See
//! ARCHITECTURE.md §Width and ARCHITECTURE.md §Python-Bindings.

use crate::truncation_spec::{
    spec_has_exact_topn, PolicySpec, PyTruncation, SpecPolicy, TOPN_PARTITIONED_MSG,
};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use paulistrings::accumulator::BuildAccumulator;
use paulistrings::engine::partitioned::{numa_nodes, CpuSet};
use paulistrings::pauli_string::PauliString;
use paulistrings::phase::Phase;
use paulistrings::{
    propagate_with_options, propagate_with_scratch_and_options, Circuit as CoreCircuit, Direction,
    EngineSelection, LayerScratch, PartitionConfig, PartitionRuntime, PartitionTrace,
    PartitionedSum, PauliAxis, PauliSum as CorePauliSum, Placement, ProductBasis, ProductState,
    PropagateOptions, StabilizerState, TermTrace, TopologyError, DEFAULT_SMALL_SUM_THRESHOLD,
};
use pyo3::exceptions::{PyNotImplementedError, PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyComplex, PyDict};
use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

/// Width-dispatch enum. The Python boundary picks the smallest width that
/// fits `num_qubits` and stores the appropriately monomorphized `PauliSum`.
pub enum PauliSumImpl {
    W1(CorePauliSum<1>),
    W2(CorePauliSum<2>),
    W4(CorePauliSum<4>),
    W8(CorePauliSum<8>),
    W16(CorePauliSum<16>),
}

impl PauliSumImpl {
    /// Pick the smallest supported width for `num_qubits`. Returns `None` if
    /// `num_qubits` exceeds the largest monomorphized width (1024 qubits).
    pub fn empty_for(num_qubits: usize) -> Option<Self> {
        for_num_qubits!(num_qubits, |W| CorePauliSum::<W>::empty(num_qubits))
    }

    pub fn num_qubits(&self) -> usize {
        for_each_width!(self, |s| s.num_qubits())
    }

    pub fn len(&self) -> usize {
        for_each_width!(self, |s| s.len())
    }

    /// Uniform product state: the same `+1` eigenstate on every qubit.
    pub fn expectation_uniform(&self, state: ProductState) -> Complex64 {
        for_each_width!(self, |s| s.expectation_product_state(state))
    }

    /// Per-qubit product state: entry `q` is qubit `q`'s `(axis, minus)`. The
    /// caller has already checked that there is exactly one entry per qubit,
    /// so the resulting masks have no bit set past `num_qubits`.
    pub fn expectation_labels(&self, axes: &[(PauliAxis, bool)]) -> Complex64 {
        for_each_width!(self, |s| s.expectation_product_basis(
            &ProductBasis::from_axes(axes.iter().copied())
        ))
    }

    /// Stabilizer state given by one signed Pauli generator per qubit. The
    /// generator strings are parsed at the active width and validated by the
    /// core, so both a malformed string and an invalid generator set surface
    /// as a `ValueError`.
    pub fn expectation_stabilizer(&self, generators: &[String]) -> PyResult<Complex64> {
        for_each_width!(self, |s| stabilizer_expectation(s, generators))
    }

    pub fn identity_coefficient(&self) -> Complex64 {
        for_each_width!(self, |s| s.identity_coefficient())
    }

    /// `None` when the two sums were monomorphized at different widths, which
    /// can only happen if their qubit counts fall in different dispatch bands.
    pub fn overlap(&self, other: &Self) -> Option<Complex64> {
        for_each_width_pair!((self, other), |a, b| a.overlap(b))
    }

    /// Snapshot of the coefficient column, in the sum's canonical order
    /// (partition-bucket index ascending, then lexicographic `(x, z)`; equal
    /// to plain lex order for sums of ≤ 1024 terms).
    pub fn coeffs(&self) -> Vec<Complex64> {
        fn coeffs_of<const W: usize>(s: &CorePauliSum<W>) -> Vec<Complex64> {
            let (_, _, c) = s.to_arrays();
            c
        }
        for_each_width!(self, |s| coeffs_of(s))
    }

    /// `(width, x_flat, z_flat)` snapshot of the SoA columns, in the sum's
    /// canonical order (see [`Self::coeffs`]) — the same order across the
    /// three exported arrays, since the order is a deterministic function of
    /// the sum. Both `x_flat` and `z_flat` have length `len() * width`, and
    /// `width` is the active monomorphization's `W`. Caller reshapes to
    /// `(len, width)`.
    pub fn xz_flat(&self) -> (usize, Vec<u64>, Vec<u64>) {
        fn flatten<const W: usize>(rows: &[[u64; W]]) -> Vec<u64> {
            // Flat-copy via iteration. The W is small (≤16) and the array
            // length is `len()`; this is not on the hot path.
            let mut out = Vec::with_capacity(rows.len() * W);
            for r in rows {
                out.extend_from_slice(r);
            }
            out
        }
        fn xz_of<const W: usize>(s: &CorePauliSum<W>) -> (usize, Vec<u64>, Vec<u64>) {
            let (x, z, _) = s.to_arrays();
            (W, flatten(&x), flatten(&z))
        }
        for_each_width!(self, |s| xz_of(s))
    }

    /// Build from a `{pauli_string: coefficient}` Python dict at the requested
    /// width. The width must already match `num_qubits` (caller's job).
    pub fn from_strings_dict(num_qubits: usize, terms: &Bound<'_, PyDict>) -> PyResult<Self> {
        for_num_qubits!(num_qubits, |W| parse_terms::<W>(num_qubits, terms)?).ok_or_else(|| {
            PyValueError::new_err("num_qubits exceeds largest monomorphized width (1024)")
        })
    }

    /// Build from raw symplectic `(x, z, coefficients)` arrays — the inverse
    /// of [`Self::xz_flat`] / [`Self::coeffs`]. See `PauliSum::from_arrays`
    /// for the shape/dtype contract; this just picks the width band for
    /// `num_qubits` and delegates.
    pub fn from_arrays(
        num_qubits: usize,
        x: &PyReadonlyArray2<'_, u64>,
        z: &PyReadonlyArray2<'_, u64>,
        coefficients: &[Complex64],
    ) -> PyResult<Self> {
        let x_shape = x.as_array().shape().to_vec();
        let z_shape = z.as_array().shape().to_vec();
        if x_shape != z_shape {
            return Err(PyValueError::new_err(format!(
                "PauliSum.from_arrays: x.shape {:?} != z.shape {:?}",
                x_shape, z_shape
            )));
        }
        if x_shape[0] != coefficients.len() {
            return Err(PyValueError::new_err(format!(
                "PauliSum.from_arrays: coefficients length {} != x/z row count {}",
                coefficients.len(),
                x_shape[0]
            )));
        }
        for_num_qubits!(num_qubits, |W| build_from_arrays::<W>(
            num_qubits,
            x,
            z,
            coefficients
        )?)
        .ok_or_else(|| {
            PyValueError::new_err("num_qubits exceeds largest monomorphized width (1024)")
        })
    }
}

/// Build a `PauliSum<W>` from a `{pauli_string: coefficient}` Python dict.
///
/// Pauli-string format matches the test helper in `pauli_sum.rs`: the
/// character at index `i` describes qubit `i`. Coefficients multiply the
/// literal Hermitian Pauli string — `Y` maps to the symplectic key
/// `(x=1, z=1)` with no phase factor, so a Hermitian observable keeps
/// real coefficients (ARCHITECTURE.md §Data-Model).
fn parse_terms<const W: usize>(
    num_qubits: usize,
    terms: &Bound<'_, PyDict>,
) -> PyResult<CorePauliSum<W>> {
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, terms.len());
    for (key, val) in terms.iter() {
        let s: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("PauliSum.from_strings keys must be str"))?;
        if s.len() != num_qubits {
            return Err(PyValueError::new_err(format!(
                "Pauli string {:?} has length {}, expected {} (length must match num_qubits)",
                s,
                s.len(),
                num_qubits
            )));
        }
        let c = extract_complex(&val)?;
        acc.add_term(parse_pauli_key::<W>(&s)?, Phase::ONE, c);
    }
    Ok(acc.finalize())
}

/// Parse an `I/X/Y/Z` label into a symplectic key: character `i` addresses
/// qubit `i`, `Y` maps to `(x=1, z=1)` with no phase factor (the crate's
/// Hermitian convention).
///
/// The caller checks the label's length against `num_qubits` first — this only
/// rejects characters outside the alphabet.
fn parse_pauli_key<const W: usize>(s: &str) -> PyResult<PauliString<W>> {
    let mut x = [0u64; W];
    let mut z = [0u64; W];
    for (i, ch) in s.chars().enumerate() {
        let word = i / 64;
        let bit = 1u64 << (i % 64);
        match ch {
            'I' => {}
            'X' => x[word] |= bit,
            'Z' => z[word] |= bit,
            'Y' => {
                x[word] |= bit;
                z[word] |= bit;
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unexpected Pauli character {:?} (expected I/X/Y/Z)",
                    other
                )));
            }
        }
    }
    Ok(PauliString::<W> { x, z })
}

/// Contract `sum` against the stabilizer state spelled by `generators`.
///
/// Generators are signed Pauli strings — `"+XX"`, `"-ZZ"`, or a bare `"ZIZ"`
/// for `+` — with the same `I/X/Y/Z` alphabet and qubit indexing as
/// `from_strings`. The core validates the set (count, range, commutation,
/// GF(2) independence) and its `StabilizerError` becomes the `ValueError`
/// message verbatim.
fn stabilizer_expectation<const W: usize>(
    sum: &CorePauliSum<W>,
    generators: &[String],
) -> PyResult<Complex64> {
    let num_qubits = sum.num_qubits();
    let mut gens: Vec<(PauliString<W>, bool)> = Vec::with_capacity(generators.len());
    for (i, spec) in generators.iter().enumerate() {
        let (minus, body) = match spec.as_bytes().first() {
            Some(b'-') => (true, &spec[1..]),
            Some(b'+') => (false, &spec[1..]),
            _ => (false, spec.as_str()),
        };
        if body.len() != num_qubits {
            return Err(PyValueError::new_err(format!(
                "expectation_stabilizer: generator {i} {spec:?} has length {} after the optional \
                 sign, expected {num_qubits} (one character per qubit)",
                body.len(),
            )));
        }
        let key = parse_pauli_key::<W>(body).map_err(|e| {
            PyValueError::new_err(format!(
                "expectation_stabilizer: generator {i} {spec:?}: {e}"
            ))
        })?;
        gens.push((key, minus));
    }
    let state = StabilizerState::<W>::from_generators(num_qubits, &gens)
        .map_err(|e| PyValueError::new_err(format!("expectation_stabilizer: {e}")))?;
    Ok(sum.expectation_stabilizer(&state))
}

/// Bit mask of the qubits `word` (a `64·word .. 64·(word+1)` slice) actually
/// covers within `num_qubits` — `!0u64` for a word entirely below
/// `num_qubits`, `0` for one entirely at or above it, and a low-bits mask for
/// the boundary word. Any set bit outside this mask addresses a qubit that
/// does not exist.
fn word_mask(word: usize, num_qubits: usize) -> u64 {
    let start = word * 64;
    if start >= num_qubits {
        0
    } else {
        let bits_in_word = (num_qubits - start).min(64);
        if bits_in_word == 64 {
            !0u64
        } else {
            (1u64 << bits_in_word) - 1
        }
    }
}

/// Build a `PauliSum<W>` from raw `(x, z, coefficients)` arrays.
///
/// `x`/`z` rows are symplectic keys in the Hermitian convention (no phase —
/// every row is folded in with `Phase::ONE`, matching `parse_terms`). A row
/// narrower than `W` words is zero-padded on the high side; a row wider than
/// `W` is a `ValueError`, since silently truncating would drop data. Ingest
/// goes through `BuildAccumulator`, so duplicate `(x, z)` rows sum their
/// coefficients and rows that cancel to exact `0+0i` are dropped.
fn build_from_arrays<const W: usize>(
    num_qubits: usize,
    x: &PyReadonlyArray2<'_, u64>,
    z: &PyReadonlyArray2<'_, u64>,
    coefficients: &[Complex64],
) -> PyResult<CorePauliSum<W>> {
    let xa = x.as_array();
    let za = z.as_array();
    let n = xa.shape()[0];
    let w_in = xa.shape()[1];
    if w_in > W {
        return Err(PyValueError::new_err(format!(
            "PauliSum.from_arrays: array width {w_in} exceeds the band width {W} \
             for num_qubits={num_qubits}"
        )));
    }
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n);
    for row in 0..n {
        let mut px = [0u64; W];
        let mut pz = [0u64; W];
        for j in 0..w_in {
            px[j] = xa[[row, j]];
            pz[j] = za[[row, j]];
        }
        for j in 0..w_in {
            let mask = word_mask(j, num_qubits);
            if px[j] & !mask != 0 || pz[j] & !mask != 0 {
                return Err(PyValueError::new_err(format!(
                    "PauliSum.from_arrays: row {row} has a bit set at or beyond qubit \
                     {num_qubits} (word {j}, x={:#x}, z={:#x})",
                    px[j], pz[j]
                )));
            }
        }
        acc.add_term(
            PauliString::<W> { x: px, z: pz },
            Phase::ONE,
            coefficients[row],
        );
    }
    Ok(acc.finalize())
}

/// Extract a Python 1-D array of `complex128` or a real-float dtype
/// (`float64`, cast to a zero-imaginary complex) into a `Vec<Complex64>`.
fn extract_complex_array(val: &Bound<'_, PyAny>) -> PyResult<Vec<Complex64>> {
    if let Ok(arr) = val.extract::<PyReadonlyArray1<Complex64>>() {
        return Ok(arr.as_array().to_vec());
    }
    if let Ok(arr) = val.extract::<PyReadonlyArray1<f64>>() {
        return Ok(arr
            .as_array()
            .iter()
            .map(|&re| Complex64::new(re, 0.0))
            .collect());
    }
    Err(PyTypeError::new_err(
        "PauliSum.from_arrays: coefficients must be a complex128 or float64 NumPy array",
    ))
}

/// Extract a Python complex/float/int into `Complex64`.
fn extract_complex(val: &Bound<'_, PyAny>) -> PyResult<Complex64> {
    if let Ok(c) = val.downcast::<PyComplex>() {
        return Ok(Complex64::new(c.real(), c.imag()));
    }
    if let Ok(f) = val.extract::<f64>() {
        return Ok(Complex64::new(f, 0.0));
    }
    Err(PyTypeError::new_err(
        "expected complex, float, or int coefficient",
    ))
}

/// One character of a per-qubit product-state label, in qiskit's
/// `Statevector.from_label` alphabet: `0`/`1` are the `Z` eigenstates, `+`/`-`
/// the `X` ones and `r`/`l` the `Y` ones. `None` for anything else.
///
/// Returns `(axis, minus)`, which is exactly what `ProductBasis::from_axes`
/// consumes.
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

/// `"forward"` (the default when `None`) or `"heisenberg"`.
///
/// Shared by `propagate` and `propagate_with_stats` so the accepted spellings
/// and the error message cannot drift apart.
fn parse_direction(direction: Option<&str>) -> PyResult<Direction> {
    match direction.unwrap_or("forward") {
        "forward" => Ok(Direction::Forward),
        "heisenberg" => Ok(Direction::Heisenberg),
        other => Err(PyValueError::new_err(format!(
            "direction must be 'forward' or 'heisenberg', got {:?}",
            other
        ))),
    }
}

/// `"sorted"` (the default when `None`), `"auto"` or `"direct"`, paired with an
/// optional small-sum threshold, as a core [`PropagateOptions`].
///
/// `None`/`None` is `PropagateOptions::default()` exactly, which the core
/// documents as bit-for-bit today's `propagate` — so the kwargs are additive and
/// omitting them changes nothing. The parse happens once at the boundary,
/// outside the width dispatch and every loop.
///
/// Shared by `propagate` and `propagate_with_stats` so the accepted spellings
/// and the error message cannot drift apart, exactly as `parse_direction` is.
fn parse_engine(
    engine: Option<&str>,
    small_sum_threshold: Option<usize>,
) -> PyResult<PropagateOptions> {
    let engine = match engine.unwrap_or("sorted") {
        "sorted" => EngineSelection::SortedOnly,
        "auto" => EngineSelection::Auto,
        "direct" => EngineSelection::SmallSumDirect,
        other => {
            return Err(PyValueError::new_err(format!(
                "engine must be 'sorted', 'auto', or 'direct', got {:?}",
                other
            )))
        }
    };
    Ok(PropagateOptions {
        engine,
        small_sum_threshold: small_sum_threshold.unwrap_or(DEFAULT_SMALL_SUM_THRESHOLD),
        ..PropagateOptions::default()
    })
}

/// `partitions=` / `pin_memory=` → an optional core [`PartitionConfig`], where
/// `None` means the classic unpartitioned path — bit for bit today's
/// behaviour, so the kwargs stay additive.
///
/// Accepted spellings, and what each resolves to:
///
/// | `partitions` | placement |
/// |---|---|
/// | `None`, `1` | `None` — the classic path |
/// | `"auto"` | one partition per NUMA node in the affinity mask, or the classic path on a single-node box |
/// | an `int` power of two `>= 2` | the same, capped at that many partitions; rejected unless the box has that many nodes |
/// | `list[list[int]]` | one partition per CPU list, exactly as given |
///
/// Anything else is a `TypeError`; a malformed value of an accepted shape (a
/// count that is not a power of two, an empty or overlapping CPU list, a CPU
/// outside the affinity mask) is a `ValueError`.
///
/// The placement is resolved against the machine **here**, at the boundary and
/// before the GIL is released, so a bad CPU list is an exception rather than a
/// failure inside the run. It is resolved a second time by
/// [`PartitionRuntime::new`]; that costs one sysfs walk, once per distinct
/// config per process (see [`runtime_for`]).
fn parse_partitions(
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
            "auto" => Placement::Auto {
                max_partitions: None,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "partitions must be 'auto', got {other:?}"
                )))
            }
        }
    } else if obj.is_instance_of::<PyBool>() {
        // `bool` is an `int` subclass, so `partitions=True` would otherwise
        // parse as `1` and silently mean "unpartitioned".
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
                Placement::Auto {
                    max_partitions: Some(k),
                }
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

    let config = PartitionConfig {
        placement,
        bind_memory: pin_memory,
        partition_row_seed: None,
    };
    let slots = config.resolve().map_err(topology_error)?;
    if slots.len() == 1 && matches!(config.placement, Placement::Auto { .. }) {
        // A single-node box (a laptop, a cgroup pinned inside one node): there
        // is nothing to partition, so run the classic path rather than pay for
        // a pool build and pin a thread nobody asked to pin. An explicit
        // one-element CPU list is honoured — that caller *did* ask.
        return Ok(None);
    }
    Ok(Some(config))
}

/// A core [`TopologyError`] as a Python exception: `OSError` for a failed
/// syscall or sysfs read, `ValueError` for everything the caller spelled
/// wrong.
fn topology_error(err: TopologyError) -> PyErr {
    match err {
        TopologyError::Io(_) => PyOSError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

/// One [`PartitionRuntime`] per distinct [`PartitionConfig`], for the life of
/// the process.
///
/// A runtime owns one pinned Rayon pool per partition, which is far too
/// expensive to build per call: a Trotter driver stepping an observable
/// through many short circuits would otherwise spawn and tear down every
/// pinned pool on every step. Keyed by the config itself, so two different
/// placements coexist and the same placement is built once.
///
/// The `Vec` is short by construction — a process uses one or two placements —
/// so a linear scan under a mutex is the right shape, and the mutex is held
/// across the build so two threads racing on a first call build one pool set
/// rather than two.
type RuntimeCache = Mutex<Vec<(PartitionConfig, Arc<PartitionRuntime>)>>;

static PARTITION_RUNTIMES: OnceLock<RuntimeCache> = OnceLock::new();

fn runtime_for(config: &PartitionConfig) -> Result<Arc<PartitionRuntime>, TopologyError> {
    let cache = PARTITION_RUNTIMES.get_or_init(|| Mutex::new(Vec::new()));
    // A poisoned lock means an earlier caller panicked; the cache is a plain
    // append-only `Vec` that is never left half-updated, so recover rather
    // than poison every later call.
    let mut cache = cache.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some((_, runtime)) = cache.iter().find(|(cached, _)| cached == config) {
        return Ok(Arc::clone(runtime));
    }
    let runtime = PartitionRuntime::new(config)?;
    cache.push((config.clone(), Arc::clone(&runtime)));
    Ok(runtime)
}

/// What can go wrong inside the GIL-released region of a propagate call.
///
/// Both variants are turned into Python exceptions after the GIL is
/// reacquired; neither can be raised from inside `allow_threads`.
enum PropagateFailure {
    /// The sum and the circuit monomorphized at different widths — impossible
    /// (both width pickers map `num_qubits` to the same arm), but surfaced as
    /// an error rather than a panic.
    WidthMismatch,
    /// The partitioned placement could not be realized on this machine.
    Topology(TopologyError),
}

impl From<PropagateFailure> for PyErr {
    fn from(err: PropagateFailure) -> Self {
        match err {
            PropagateFailure::WidthMismatch => {
                PyValueError::new_err("internal: PauliSum and Circuit width mismatch")
            }
            PropagateFailure::Topology(err) => topology_error(err),
        }
    }
}

/// The `NotImplementedError` a partitioned run with an exact `topn` raises,
/// naming the kwarg that put the call in partitioned mode.
fn topn_partitioned_error(partitions: Option<&Bound<'_, PyAny>>) -> PyErr {
    let shown = match partitions {
        // A distributed run: `comm=` is a communicator whose repr says
        // nothing useful, and the kwarg's *name* is the actionable part.
        None => "comm=<mpi4py communicator>".to_string(),
        Some(obj) => {
            let repr = obj
                .repr()
                .map_or_else(|_| "…".to_string(), |repr| repr.to_string());
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
        other => Err(PyValueError::new_err(format!(
            "result must be 'gather' or 'local', got {other:?}"
        ))),
    }
}

/// Whether `comm=` names an actual communicator (`None` and an omitted kwarg
/// both mean "not distributed").
fn comm_requested(comm: Option<&Bound<'_, PyAny>>) -> bool {
    comm.is_some_and(|comm| !comm.is_none())
}

/// The `RuntimeError` a `comm=` raises in a build without the `mpi` feature.
#[cfg(not(feature = "mpi"))]
fn mpi_unavailable_error() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(
        "paulistrings was built without MPI support; rebuild with `maturin develop --features mpi`",
    )
}

/// How one propagate call runs. Decided at the boundary, with the GIL held,
/// and consumed once inside the width dispatch.
enum RunMode {
    /// One pool over the whole process — today's path, bit for bit.
    Classic,
    /// One pinned pool per NUMA domain, in this process (`partitions=`).
    Partitioned(PartitionConfig),
    /// One partition per MPI rank (`comm=`). Carries the adopted communicator,
    /// so building the mode is the collective step and dropping it frees the
    /// duplicate.
    #[cfg(feature = "mpi")]
    Distributed(crate::mpi::MpiRun),
}

/// The per-layer records a run produced, tagged by which engine produced them.
enum RunTrace {
    /// The unpartitioned engine's term counts.
    Term(TermTrace),
    /// An in-process partitioned run's records, plus the partition count.
    Partition(PartitionTrace, usize),
    /// A distributed run's records — **this rank's only** — plus `(rank, size)`.
    #[cfg(feature = "mpi")]
    Distributed(PartitionTrace, u32, u32),
}

impl RunMode {
    /// Run `circuit` over `sum`. Called inside `allow_threads`; consumes the
    /// mode, so a distributed run's communicator is freed before it returns.
    fn run<const W: usize>(
        self,
        circuit: &CoreCircuit<W>,
        sum: &CorePauliSum<W>,
        spec: &PolicySpec,
        direction: Direction,
        options: PropagateOptions,
    ) -> Result<CorePauliSum<W>, PropagateFailure> {
        let policy = SpecPolicy::<W>(spec);
        match self {
            RunMode::Classic => Ok(propagate_with_options(
                circuit,
                sum.clone(),
                &policy,
                direction,
                options,
            )),
            RunMode::Partitioned(config) => {
                // The runtime (and its pinned pools) is cached per config, so
                // a Trotter loop of many short calls builds it once.
                let runtime = runtime_for(&config).map_err(PropagateFailure::Topology)?;
                let mut split = PartitionedSum::<W>::scatter(sum.clone(), runtime, &config);
                split.propagate_with_options(circuit, &policy, direction, options);
                Ok(split.into_gathered())
            }
            #[cfg(feature = "mpi")]
            RunMode::Distributed(run) => run
                .propagate(circuit, sum, &policy, direction, options)
                .map_err(PropagateFailure::Topology),
        }
    }

    /// [`run`](Self::run), also recording the per-layer counts.
    fn run_traced<const W: usize>(
        self,
        circuit: &CoreCircuit<W>,
        sum: &CorePauliSum<W>,
        spec: &PolicySpec,
        direction: Direction,
        options: PropagateOptions,
    ) -> Result<(CorePauliSum<W>, RunTrace), PropagateFailure> {
        let policy = SpecPolicy::<W>(spec);
        match self {
            RunMode::Classic => {
                let mut scratch = LayerScratch::<W>::new();
                scratch.enable_term_trace();
                let out = propagate_with_scratch_and_options(
                    circuit,
                    sum.clone(),
                    &policy,
                    direction,
                    &mut scratch,
                    options,
                );
                let trace = scratch
                    .take_term_trace()
                    .expect("the trace is enabled before the layer loop runs");
                Ok((out, RunTrace::Term(trace)))
            }
            RunMode::Partitioned(config) => {
                let runtime = runtime_for(&config).map_err(PropagateFailure::Topology)?;
                let mut split = PartitionedSum::<W>::scatter(sum.clone(), runtime, &config);
                split.enable_trace();
                split.propagate_with_options(circuit, &policy, direction, options);
                // From the runtime, not the trace: a zero-layer circuit
                // records no layer, but the placement is still worth
                // reporting.
                let partitions = split.num_partitions();
                let trace = split.take_trace().unwrap_or_default();
                Ok((
                    split.into_gathered(),
                    RunTrace::Partition(trace, partitions),
                ))
            }
            #[cfg(feature = "mpi")]
            RunMode::Distributed(run) => {
                let (out, trace, rank, size) = run
                    .propagate_traced(circuit, sum, &policy, direction, options)
                    .map_err(PropagateFailure::Topology)?;
                Ok((out, RunTrace::Distributed(trace, rank, size)))
            }
        }
    }
}

/// Turn the placement kwargs into a [`RunMode`], with the GIL held.
///
/// Order matters. Everything that can raise on the caller's spelling —
/// `comm=` together with `partitions=`, a bad CPU list, an exact `topn` — is
/// decided *before* the communicator is adopted, because adopting it is
/// collective: a rank that raises early is a rank that never entered a
/// collective, so the whole group raises together and none of them hangs.
fn parse_run_mode(
    py: Python<'_>,
    partitions: Option<&Bound<'_, PyAny>>,
    pin_memory: bool,
    comm: Option<&Bound<'_, PyAny>>,
    gather: bool,
    spec: &PolicySpec,
) -> PyResult<RunMode> {
    let distributed = comm_requested(comm);
    // Before `parse_partitions`, so the conflict is reported as a conflict
    // whatever the placement would have resolved to on this machine.
    if distributed && partitions.is_some_and(|obj| !obj.is_none()) {
        return Err(PyValueError::new_err(
            "comm= and partitions= are alternatives: comm= already places one partition per MPI \
             rank, so pass the placement to the launcher (mpirun --map-by ppr:1:numa --bind-to \
             numa) rather than to propagate",
        ));
    }
    let config = parse_partitions(partitions, pin_memory)?;
    if (distributed || config.is_some()) && spec_has_exact_topn(spec) {
        return Err(topn_partitioned_error(if distributed {
            None
        } else {
            partitions
        }));
    }
    if distributed {
        #[cfg(feature = "mpi")]
        {
            let transport = crate::mpi::transport_from_comm(py, comm.expect("comm is Some"))?;
            return Ok(RunMode::Distributed(crate::mpi::MpiRun::new(
                transport, gather,
            )));
        }
        #[cfg(not(feature = "mpi"))]
        {
            let _ = (py, gather);
            return Err(mpi_unavailable_error());
        }
    }
    Ok(match config {
        Some(config) => RunMode::Partitioned(config),
        None => RunMode::Classic,
    })
}

/// Both propagation entry points require the sum and the circuit to agree on
/// the qubit count (they would otherwise be monomorphized at different widths,
/// which the width dispatch cannot pair up).
fn check_num_qubits(sum: &PauliSumImpl, circuit: &crate::circuit::Circuit) -> PyResult<()> {
    if sum.num_qubits() != circuit.inner.num_qubits() {
        return Err(PyValueError::new_err(format!(
            "PauliSum.num_qubits ({}) != Circuit.num_qubits ({})",
            sum.num_qubits(),
            circuit.inner.num_qubits()
        )));
    }
    Ok(())
}

/// Per-layer term counts from `PauliSum.propagate_with_stats`.
///
/// A plain record with read-only attributes; `terms_in` and `terms_out` have
/// one entry per layer applied, in application order (so *reverse* circuit
/// order under `direction="heisenberg"`).
#[pyclass(
    frozen,
    module = "paulistrings._paulistrings",
    name = "PropagationStats"
)]
pub struct PropagationStats {
    layers: usize,
    terms_in: Vec<usize>,
    terms_out: Vec<usize>,
    peak_terms: usize,
    final_terms: usize,
    partition: Option<PartitionStats>,
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

    /// Peak *resident* term count: `max(terms_in[0], terms_out...)`, or the
    /// input's term count for a zero-layer circuit.
    ///
    /// This is how large the sum ever got *between* layers. The transient
    /// in-layer expansion — after a channel's fanout, before the merge
    /// deduplicates and truncation filters — is deliberately not measured;
    /// capturing it would mean instrumenting the engine's hot loop. For a
    /// memory figure, read peak RSS from `/proc/self/status` instead.
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

    /// The partitioned run's own record, or `None` for an unpartitioned call.
    ///
    /// `Some(PartitionStats)` exactly when `propagate_with_stats` was given a
    /// `partitions=` that put the call in partitioned mode — note that
    /// `partitions="auto"` on a single-NUMA-node box runs unpartitioned and so
    /// reports `None` here. The per-layer lists it carries are indexed the same
    /// way as `terms_in` / `terms_out`: one entry per layer, in application
    /// order.
    #[getter]
    fn partition(&self) -> Option<PartitionStats> {
        self.partition.clone()
    }

    /// The five term-count fields, in the order the getters are declared, so a
    /// stats record printed from a REPL or a log line is readable without
    /// poking at it attribute by attribute. The per-layer lists are printed in
    /// full — they are one entry per layer, not per term.
    ///
    /// `partition` is deliberately **not** here: the format is pinned by
    /// `test_propagation_stats.py`, and a partitioned record's per-partition
    /// lists are `P` times longer again. Print `stats.partition` for those.
    fn __repr__(&self) -> String {
        format!(
            "PropagationStats(layers={}, terms_in={:?}, terms_out={:?}, \
             peak_terms={}, final_terms={})",
            self.layers, self.terms_in, self.terms_out, self.peak_terms, self.final_terms
        )
    }
}

impl PropagationStats {
    /// Derive the Python-facing record from a core [`TermTrace`] plus the
    /// length of the propagated sum (which is what "peak" falls back to when
    /// no layer ran).
    fn from_trace(trace: TermTrace, final_terms: usize) -> Self {
        debug_assert_eq!(trace.terms_in.len(), trace.terms_out.len());
        Self {
            layers: trace.terms_out.len(),
            peak_terms: trace.peak_terms().unwrap_or(final_terms),
            final_terms,
            terms_in: trace.terms_in,
            terms_out: trace.terms_out,
            partition: None,
        }
    }

    /// The same record from a partitioned run's [`PartitionTrace`], plus the
    /// per-partition detail in `partition`.
    ///
    /// The layer-level counts are the per-partition ones summed over
    /// partitions, which is what the unpartitioned engine would have recorded
    /// for the same layer: partitions hold disjoint term sets, and a layer's
    /// `terms_in`/`terms_out` are read at the same two points in the layer
    /// loop. So the two `propagate_with_stats` paths report comparable
    /// numbers, and a partitioned run can be checked against an unpartitioned
    /// one field by field.
    /// A distributed run's records are this rank's only, so the sum over the
    /// "partition" dimension is a sum of one — the layer-level counts are then
    /// this rank's, not the group's (documented on `PartitionStats.size`).
    fn from_partition_trace(
        trace: &PartitionTrace,
        partitions: usize,
        final_terms: usize,
        ranks: Option<(u32, u32)>,
    ) -> Self {
        let sum_of = |counts: &[usize]| counts.iter().sum::<usize>();
        let term_trace = TermTrace {
            terms_in: trace.layers.iter().map(|l| sum_of(&l.terms_in)).collect(),
            terms_out: trace.layers.iter().map(|l| sum_of(&l.terms_out)).collect(),
        };
        Self {
            partition: Some(PartitionStats::from_trace(trace, partitions, ranks)),
            ..Self::from_trace(term_trace, final_terms)
        }
    }

    /// Whichever of the three traces the run recorded, as one record.
    fn from_run_trace(trace: RunTrace, final_terms: usize) -> Self {
        match trace {
            RunTrace::Term(trace) => Self::from_trace(trace, final_terms),
            RunTrace::Partition(trace, partitions) => {
                Self::from_partition_trace(&trace, partitions, final_terms, None)
            }
            #[cfg(feature = "mpi")]
            RunTrace::Distributed(trace, rank, size) => {
                Self::from_partition_trace(&trace, size as usize, final_terms, Some((rank, size)))
            }
        }
    }
}

/// Per-layer, per-partition record of a partitioned propagation — the
/// `partition` attribute of a [`PropagationStats`] from a `partitions=` call.
///
/// A plain record with read-only attributes, like `PropagationStats`. Every
/// list is one entry per layer applied, in application order (so *reverse*
/// circuit order under `direction="heisenberg"`); the entries of `terms_in` /
/// `terms_out` are themselves one entry per partition, in rank order.
///
/// This is the instrument for the two questions a partitioned run raises:
/// how much of the sum crossed a partition boundary (`rows_exported`,
/// `bytes_exported`, `local`), and how evenly the terms were spread
/// (`imbalance`).
#[pyclass(frozen, module = "paulistrings._paulistrings", name = "PartitionStats")]
#[derive(Clone)]
pub struct PartitionStats {
    partitions: usize,
    rank: Option<u32>,
    size: Option<u32>,
    local: Vec<bool>,
    rows_exported: Vec<u64>,
    bytes_exported: Vec<u64>,
    terms_in: Vec<Vec<usize>>,
    terms_out: Vec<Vec<usize>>,
    imbalance: Vec<f64>,
}

#[pymethods]
impl PartitionStats {
    /// Number of partitions the run was split across — always a power of two.
    ///
    /// For a distributed (`comm=`) run this is the MPI group size, i.e. `size`
    /// below: one partition per rank.
    #[getter]
    fn partitions(&self) -> usize {
        self.partitions
    }

    /// This process's rank in the `comm=` group, or `None` for an in-process
    /// (`partitions=`) run.
    ///
    /// It is the index into the per-partition dimension that the lists below
    /// *would* carry if a rank could see the whole group — see `size` for why
    /// it cannot.
    #[getter]
    fn rank(&self) -> Option<u32> {
        self.rank
    }

    /// The `comm=` group's size, or `None` for an in-process (`partitions=`)
    /// run.
    ///
    /// **A distributed run's per-layer lists hold this rank's entry only.**
    /// `terms_in[k]` and `terms_out[k]` are one-element lists (this rank's
    /// count for layer `k`), `rows_exported[k]` / `bytes_exported[k]` are what
    /// *this* rank sent, and `imbalance[k]` is therefore always `1.0` —
    /// nothing gathers the group's counters, because a per-layer all-reduce
    /// would be a collective added to every layer for a diagnostic. Reduce
    /// them yourself over `comm` when you want the group's picture. The
    /// in-process case is unchanged: there the lists are `partitions` long.
    #[getter]
    fn size(&self) -> Option<u32> {
        self.size
    }

    /// Whether each layer was purely local, i.e. moved no row across a
    /// partition boundary and made no transport call for the exchange.
    ///
    /// A layer is local when the channel's deltas all keep the partition rows
    /// of a key fixed — every single-qubit channel on a qubit outside the
    /// partition rows, and every diagonal one. `local[k]` is exactly
    /// `rows_exported[k] == 0`.
    #[getter]
    fn local(&self) -> Vec<bool> {
        self.local.clone()
    }

    /// Rows sent across partition boundaries in each layer, summed over every
    /// sender/receiver pair.
    ///
    /// The traffic figure: one row is one key plus one coefficient, written by
    /// the sender and read by the receiver's merge. Compare against
    /// `PropagationStats.terms_in` for the fraction of the sum that moved.
    #[getter]
    fn rows_exported(&self) -> Vec<u64> {
        self.rows_exported.clone()
    }

    /// Wire bytes behind `rows_exported`, per layer — the same rows counted in
    /// their exchange-block encoding, including the per-block headers.
    #[getter]
    fn bytes_exported(&self) -> Vec<u64> {
        self.bytes_exported.clone()
    }

    /// Terms each partition held before each layer: `terms_in[k][r]` for layer
    /// `k`, rank `r`. The row sums are `PropagationStats.terms_in`.
    #[getter]
    fn terms_in(&self) -> Vec<Vec<usize>> {
        self.terms_in.clone()
    }

    /// Terms each partition held after each layer, i.e. after that layer's
    /// truncation. The row sums are `PropagationStats.terms_out`.
    #[getter]
    fn terms_out(&self) -> Vec<Vec<usize>> {
        self.terms_out.clone()
    }

    /// Load imbalance of `terms_in` per layer: the maximum over partitions
    /// divided by their mean.
    ///
    /// `1.0` is perfect balance (and the answer for a layer where every
    /// partition was empty); `partitions` is the worst case, one partition
    /// holding everything. Random partition rows on a large sum sit within a
    /// percent or two of `1.0` — a persistent excursion is the signal that the
    /// circuit drove the sum's support into one partition's rows.
    #[getter]
    fn imbalance(&self) -> Vec<f64> {
        self.imbalance.clone()
    }

    /// All nine fields, in the order the getters are declared. The per-layer
    /// lists are printed in full, `terms_in` / `terms_out` nested one level
    /// deeper — one entry per layer per partition, never per term.
    fn __repr__(&self) -> String {
        // `local` is spelled with Python's `True`/`False` rather than Rust's
        // `Debug`, so the line can be pasted back into a REPL. `rank`/`size`
        // are spelled the same way: `None`, not Rust's `None`-in-`Debug`
        // (which happens to match) or `Some(0)`.
        let local = self
            .local
            .iter()
            .map(|&local| if local { "True" } else { "False" })
            .collect::<Vec<_>>()
            .join(", ");
        let show = |value: Option<u32>| value.map_or_else(|| "None".to_string(), |v| v.to_string());
        format!(
            "PartitionStats(partitions={}, rank={}, size={}, local=[{}], rows_exported={:?}, \
             bytes_exported={:?}, terms_in={:?}, terms_out={:?}, imbalance={:?})",
            self.partitions,
            show(self.rank),
            show(self.size),
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
    ///
    /// `partitions` comes from the runtime (or, distributed, from the group
    /// size) rather than the trace, so a zero-layer circuit — which records
    /// nothing — still reports the placement it ran on. `ranks` is
    /// `Some((rank, size))` for a distributed run and `None` for an in-process
    /// one, which is the only thing distinguishing the two records.
    fn from_trace(trace: &PartitionTrace, partitions: usize, ranks: Option<(u32, u32)>) -> Self {
        let total = |matrix: &[Vec<u64>]| matrix.iter().flat_map(|row| row.iter()).sum::<u64>();
        Self {
            partitions,
            rank: ranks.map(|(rank, _)| rank),
            size: ranks.map(|(_, size)| size),
            local: trace.layers.iter().map(|l| l.remote_deltas == 0).collect(),
            rows_exported: trace.layers.iter().map(|l| total(&l.rows_sent)).collect(),
            bytes_exported: trace.layers.iter().map(|l| total(&l.bytes_sent)).collect(),
            terms_in: trace.layers.iter().map(|l| l.terms_in.clone()).collect(),
            terms_out: trace.layers.iter().map(|l| l.terms_out.clone()).collect(),
            imbalance: trace.imbalance(),
        }
    }
}

#[pyclass(module = "paulistrings._paulistrings", name = "PauliSum")]
pub struct PauliSum {
    pub(crate) inner: PauliSumImpl,
}

#[pymethods]
impl PauliSum {
    /// Empty Pauli sum on `num_qubits` qubits.
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        PauliSumImpl::empty_for(num_qubits)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                PyValueError::new_err("num_qubits exceeds largest monomorphized width (1024)")
            })
    }

    /// Build from a `{pauli_string: coefficient}` dict.
    ///
    /// Each key is a string of `I/X/Y/Z` characters, one per qubit (index
    /// `i` addresses qubit `i`). Coefficients multiply the literal Hermitian
    /// Pauli string, so a Hermitian observable has real coefficients.
    #[classmethod]
    fn from_strings(
        _cls: &Bound<'_, pyo3::types::PyType>,
        terms: &Bound<'_, PyDict>,
        num_qubits: usize,
    ) -> PyResult<Self> {
        let inner = PauliSumImpl::from_strings_dict(num_qubits, terms)?;
        Ok(Self { inner })
    }

    /// Build from raw symplectic `(x, z, coefficients)` arrays — the inverse
    /// of `x_array` / `z_array` / `coefficients_array`.
    ///
    /// `x` and `z` are `uint64` arrays of shape `(n_terms, w)`; `w` may be
    /// anywhere from `1` up to the band width `num_qubits` picks (the same
    /// width `.width` would report), and a narrower array is zero-padded on
    /// ingest, so a sum exported at its own band and re-imported round-trips
    /// exactly. `coefficients` is a 1-D array of length `n_terms`,
    /// `complex128` or a real-float dtype (cast to a zero-imaginary complex).
    ///
    /// Rows are symplectic keys in the Hermitian convention — no phase is
    /// applied, matching `from_strings`. Ingest routes through the same
    /// `BuildAccumulator` `from_strings` uses: duplicate `(x, z)` rows sum
    /// their coefficients, and rows whose accumulated coefficient is exact
    /// `0+0i` are dropped. A set bit at or beyond qubit index `num_qubits` in
    /// any row is a `ValueError`.
    #[classmethod]
    fn from_arrays(
        _cls: &Bound<'_, pyo3::types::PyType>,
        x: PyReadonlyArray2<'_, u64>,
        z: PyReadonlyArray2<'_, u64>,
        coefficients: &Bound<'_, PyAny>,
        num_qubits: usize,
    ) -> PyResult<Self> {
        let coeffs = extract_complex_array(coefficients)?;
        let inner = PauliSumImpl::from_arrays(num_qubits, &x, &z, &coeffs)?;
        Ok(Self { inner })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Snapshot of the coefficient column as a list of Python complex values.
    fn coefficients(&self) -> Vec<Complex64> {
        self.inner.coeffs()
    }

    /// Expectation value in a single-qubit product state.
    ///
    /// `state` is either a **uniform** name — `"x+"` (`|+...+>`), `"y+"`
    /// (`|+i...+i>`) or `"z+"` (`|0...0>`), each the `+1` eigenstate of that
    /// Pauli on every qubit, matched case-insensitively — or a **per-qubit
    /// label string** of exactly `num_qubits` characters, where character `i`
    /// gives qubit `i`'s state in qiskit's `Statevector.from_label` alphabet:
    ///
    /// | label | state | axis |
    /// |---|---|---|
    /// | `0` / `1` | `\|0>` / `\|1>` | Z ± |
    /// | `+` / `-` | `\|+>` / `\|->` | X ± |
    /// | `r` / `l` | `\|+i>` / `\|-i>` | Y ± |
    ///
    /// The label characters are case-sensitive (`r`/`l`, not `R`/`L`), so a
    /// mistyped uniform name is an error rather than a silent reinterpretation.
    /// Qubit indexing matches `from_strings`.
    ///
    /// Cost is one masked pass over the terms in either case — never an
    /// expansion over basis states. Returns a Python complex; take `.real`
    /// when the operator is Hermitian.
    #[pyo3(signature = (state="x+"))]
    fn expectation(&self, state: &str) -> PyResult<Complex64> {
        // The uniform names win first, case-insensitively, so `"x+"` keeps
        // meaning |+...+> at any qubit count. They cannot collide with a label
        // string: `x`, `y` and `z` are not in the per-qubit alphabet.
        let uniform = match state.to_ascii_lowercase().as_str() {
            "x+" => Some(ProductState::XPlus),
            "y+" => Some(ProductState::YPlus),
            "z+" => Some(ProductState::ZPlus),
            _ => None,
        };
        if let Some(st) = uniform {
            return Ok(self.inner.expectation_uniform(st));
        }
        let num_qubits = self.inner.num_qubits();
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
        Ok(self.inner.expectation_labels(&axes))
    }

    /// Expectation value in a stabilizer state given by its generators.
    ///
    /// `generators` is a list of exactly `num_qubits` signed Pauli strings —
    /// `"+XX"`, `"-ZZ"`, or a bare `"ZIZ"` for `+` — each of length
    /// `num_qubits`, in the same `I/X/Y/Z` alphabet and qubit indexing as
    /// `from_strings` (character `i` is qubit `i`, `Y` Hermitian with no phase
    /// factor). They must be pairwise commuting and independent over GF(2);
    /// anything else is a `ValueError`, including a set implying `-I` is a
    /// stabilizer.
    ///
    /// This reads any stabilizer state — Bell, GHZ, cluster, the output of a
    /// Clifford circuit (see `paulistrings.interop.stabilizers_from_stim`) —
    /// where `expectation` reads only single-qubit product states. Each term
    /// `P` contributes `+c_P` or `-c_P` when `±P` is in the stabilizer group
    /// and `0` otherwise, so the cost is `O(terms · num_qubits² / 64)` word
    /// operations after a one-off `O(num_qubits³ / 64)` reduction of the
    /// generators — never an expansion over basis states. For a state that
    /// factorizes, `expectation`'s masked scan is `num_qubits` times cheaper
    /// per term; prefer it there.
    ///
    /// Returns a Python complex; take `.real` when the operator is Hermitian.
    ///
    /// ```python
    /// bell = ["XX", "ZZ"]                      # (|00> + |11>) / sqrt(2)
    /// PauliSum.from_strings({"YY": 1.0}, num_qubits=2).expectation_stabilizer(bell)
    /// # (-1+0j), since XX·ZZ = -YY
    /// ```
    fn expectation_stabilizer(&self, generators: Vec<String>) -> PyResult<Complex64> {
        self.inner.expectation_stabilizer(&generators)
    }

    /// Hilbert-Schmidt overlap `tr(self* . other) / 2^n`.
    ///
    /// On the Pauli basis this is `sum(conj(a_i) * b_i)` over shared keys.
    fn overlap(&self, other: &Self) -> PyResult<Complex64> {
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

    /// Coefficient of the identity term, i.e. `tr(O) / 2^n`.
    fn identity_coefficient(&self) -> Complex64 {
        self.inner.identity_coefficient()
    }

    /// Snapshot of the coefficient column as a 1-D NumPy `complex128` array.
    fn coefficients_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
        self.inner.coeffs().into_pyarray_bound(py)
    }

    /// Snapshot of the X-part column as a 2-D NumPy `uint64` array of shape
    /// `(len, W)` where `W` is the monomorphized width chosen for this sum.
    /// One row per term; column `j` holds the bit-word covering qubits
    /// `64*j .. 64*(j+1)`.
    fn x_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        let (w, x_flat, _z_flat) = self.inner.xz_flat();
        let n = x_flat.len() / w;
        x_flat
            .into_pyarray_bound(py)
            .reshape([n, w])
            .expect("flat length is n*w by construction")
    }

    /// Snapshot of the Z-part column as a 2-D NumPy `uint64` array. See
    /// `x_array` for the layout.
    fn z_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        let (w, _x_flat, z_flat) = self.inner.xz_flat();
        let n = z_flat.len() / w;
        z_flat
            .into_pyarray_bound(py)
            .reshape([n, w])
            .expect("flat length is n*w by construction")
    }

    /// Active monomorphized width `W` (number of `u64` words per term).
    /// Useful when paired with `x_array` / `z_array` for downstream bit-twiddling.
    #[getter]
    fn width(&self) -> usize {
        self.inner.xz_flat().0
    }

    /// Propagate `self` through `circuit`.
    ///
    /// `direction`: `"forward"` (default) or `"heisenberg"`. `policy` is an
    /// optional `Truncation` from the `truncation` submodule; if `None`, no
    /// per-term filtering is applied (the engine's merge phase still drops
    /// exact-zero terms).
    ///
    /// `engine` picks the layer engine, and defaults to the bucketed sorting
    /// engine at every term count — today's behaviour, unchanged:
    ///
    /// | `engine` | layers on the small-sum direct path |
    /// |---|---|
    /// | `"sorted"` (default) | none |
    /// | `"auto"` | the leading ones, while the sum is within `small_sum_threshold` **and** the policy has no layer pass |
    /// | `"direct"` | the leading ones, while the sum is within `small_sum_threshold`, whatever the policy |
    ///
    /// The direct path applies each layer term by term into a hash map, skipping
    /// the bucketed machinery and its per-layer fixed cost; it is 1.5–2.4× faster
    /// below a few hundred terms and slower above a few thousand, which is what
    /// `small_sum_threshold` (default 2048, `paulistrings.DEFAULT_SMALL_SUM_THRESHOLD`)
    /// prices. The transition is one-way: once a layer leaves the sum above the
    /// threshold the rest of the circuit runs on the sorting engine. Entry is
    /// re-decided on every call, so a Trotter driver stepping a small observable
    /// through many short circuits gets it each time.
    ///
    /// Only the *speed* differs. Both engines apply the same truncation in the
    /// same place and emit the same per-layer term counts and progress records;
    /// the results agree to floating-point tolerance, since equal-key summation
    /// order is unspecified (ARCHITECTURE.md §Determinism). `"auto"` declines a
    /// policy with a layer pass — `topn`, `approx_topn`, or an `&` composition
    /// containing one — because the round trip through a materialized sum would
    /// eat the win; `"direct"` takes it anyway, and stays correct.
    ///
    /// The GIL is released for the duration of the propagation, so Python
    /// threads — including `logging` handlers draining the engine's per-layer
    /// progress records — run while a long simulation is in flight.
    ///
    /// # Partitioned mode
    ///
    /// `partitions` splits the sum across NUMA domains: each partition holds a
    /// disjoint share of the terms, selected by designated rows of the GF(2)
    /// hash, and runs on its own Rayon pool pinned to that domain's CPUs, with
    /// only the rows a layer moves across a boundary exchanged between them.
    /// It is off by default, and `partitions=None` is bit for bit today's
    /// path.
    ///
    /// | `partitions` | placement |
    /// |---|---|
    /// | `None`, `1` (default) | unpartitioned — one pool, today's engine |
    /// | `"auto"` | one partition per NUMA node in the affinity mask (unpartitioned on a single-node box) |
    /// | an `int` power of two `>= 2` | the same, capped at that many partitions; a `ValueError` unless the box has that many nodes |
    /// | `list[list[int]]` | one partition per CPU list, e.g. `[[0, 1], [2, 3]]`; the lists must be non-empty, disjoint, and inside the process's affinity mask |
    ///
    /// `pin_memory` (default `True`) binds each pool's allocations to its
    /// partition's NUMA node, which is the point of the placement; pass
    /// `False` to pin threads but not memory.
    ///
    /// Three things behave differently in partitioned mode:
    ///
    /// - **`RAYON_NUM_THREADS` is ignored.** Each partition builds its own
    ///   pool sized from its CPU list, so the thread count is the placement's,
    ///   not the environment's.
    /// - **`engine` is ignored.** Every layer runs on the bucketed sorting
    ///   engine; there is no partitioned small-sum direct path.
    /// - **`truncation.topn` is unsupported** and raises
    ///   `NotImplementedError`, anywhere in the policy tree. Exact top-n needs
    ///   the n-th largest magnitude of the whole layer — a distributed k-th
    ///   selection, not one reduction. Use `truncation.approx_topn`, whose
    ///   octave histogram all-reduces exactly, or `partitions=None`.
    ///
    /// Results agree with the unpartitioned path to floating-point tolerance,
    /// as the two engines do (ARCHITECTURE.md §Determinism).
    ///
    /// # Distributed mode (MPI)
    ///
    /// `comm` takes an `mpi4py` communicator and runs **one partition per
    /// rank** — the same layer loop as `partitions=`, with the in-process
    /// channel matrix replaced by point-to-point MPI. `comm` and `partitions`
    /// are alternatives, not a pair: pass the placement to the launcher
    /// instead (one rank per NUMA domain). Everything the partitioned section
    /// above says still holds, `truncation.topn` included.
    ///
    /// ```python
    /// import mpi4py
    /// mpi4py.rc.thread_level = "serialized"      # before mpi4py.MPI is imported
    /// from mpi4py import MPI
    /// import paulistrings
    ///
    /// evolved = observable.propagate(circuit, policy, comm=MPI.COMM_WORLD)
    /// if MPI.COMM_WORLD.Get_rank() == 0:
    ///     print(len(evolved), evolved.expectation("z+"))
    /// ```
    ///
    /// Launch it with one rank per NUMA domain, e.g.
    /// `mpirun -n 4 --map-by ppr:1:numa --bind-to numa python script.py`, or
    /// under Slurm `srun --ntasks-per-node=4 --cpu-bind=ldoms --mpi=pmix
    /// python script.py`.
    ///
    /// | `result` | what each rank gets back |
    /// |---|---|
    /// | `"gather"` (default) | rank 0 the whole evolved sum; every other rank an **empty** `PauliSum` of the same `num_qubits`, so downstream code still type-checks |
    /// | `"local"` | this rank's own share. The shares are disjoint, so a global reduction is `comm.allreduce(local.expectation(...))` and the term count is `comm.allreduce(len(local))` |
    ///
    /// Four requirements, none of them checkable from inside a single rank:
    ///
    /// - **Thread level at least `MPI_THREAD_SERIALIZED`.** The layer loop
    ///   runs inside a pinned Rayon pool, so MPI is called from a pool worker
    ///   rather than the main thread. Set `mpi4py.rc.thread_level =
    ///   "serialized"` (or `"multiple"`, which is mpi4py's default) *before*
    ///   `from mpi4py import MPI`; a weaker level is a `RuntimeError` here
    ///   rather than undefined behaviour later.
    /// - **A power-of-two rank count.** A partition is named by `log2(P)`
    ///   GF(2) hash rows, so `mpirun -n 3` is a `ValueError`.
    /// - **A replicated input.** Every rank must call this with the *same*
    ///   `self`, circuit, policy, direction and options; the scatter is a
    ///   local filter of the replicated sum, not a distribution of one rank's
    ///   copy. Nothing checks the terms — build them from the same seed, or
    ///   load the same file, on every rank. (The run's *shape* is checked: a
    ///   disagreement about the circuit or the bucket count aborts.)
    /// - **A collective call.** Every rank of `comm` must reach it, in the
    ///   same order; a rank that skips one hangs the rest.
    ///
    /// Without the `mpi` cargo feature (`paulistrings.mpi_available()` is
    /// `False`), any `comm` other than `None` raises `RuntimeError`.
    #[pyo3(signature = (circuit, policy=None, direction=None, engine=None, small_sum_threshold=None, partitions=None, pin_memory=true, comm=None, result="gather"))]
    #[allow(clippy::too_many_arguments)]
    fn propagate(
        &self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
        engine: Option<&str>,
        small_sum_threshold: Option<usize>,
        partitions: Option<&Bound<'_, PyAny>>,
        pin_memory: bool,
        comm: Option<&Bound<'_, PyAny>>,
        result: &str,
    ) -> PyResult<Self> {
        let dir = parse_direction(direction)?;
        let options = parse_engine(engine, small_sum_threshold)?;
        let gather = parse_result(result)?;
        check_num_qubits(&self.inner, circuit)?;
        let no_op = PolicySpec::NoOp;
        let spec: &PolicySpec = match policy {
            Some(p) => &p.spec,
            None => &no_op,
        };
        // Last, because adopting a communicator is collective: every check
        // above raises on all ranks alike, before any of them has entered MPI.
        let mode = parse_run_mode(py, partitions, pin_memory, comm, gather, spec)?;
        // The whole simulation runs without the GIL: everything the engine
        // touches is plain Rust data (`PauliSumImpl`, `CircuitImpl` and
        // `PolicySpec` are all `Send + Sync`), so nothing here needs Python.
        // Releasing it lets Python `logging` handlers — the consumers of the
        // engine's per-layer progress records, bridged by `pyo3-log` — and any
        // other Python thread run while a long propagate is in flight.
        //
        // The closure returns a `Result` so the (unreachable) width-mismatch
        // arm can bail out of it; the error is turned into a Python exception
        // after the GIL is reacquired.
        let inner = py.allow_threads(move || -> Result<PauliSumImpl, PropagateFailure> {
            Ok(for_each_width_propagate!(
                &self.inner,
                &circuit.inner,
                |s, c, W| mode.run::<W>(c, s, spec, dir, options)?,
                else {
                    // Same num_qubits but different widths is impossible
                    // because both width pickers map num_qubits to the
                    // same arm.
                    return Err(PropagateFailure::WidthMismatch);
                }
            ))
        })?;
        Ok(Self { inner })
    }

    /// Propagate `self` through `circuit`, returning
    /// `(evolved, PropagationStats)`.
    ///
    /// Arguments and semantics are `propagate`'s, exactly — the only
    /// difference is that the engine records per-layer term counts (before
    /// each layer, and after each layer's truncation) on the calling thread.
    /// The counts come from length reads the layer loop already performs, so
    /// the propagation itself is untouched: `evolved` agrees with
    /// `propagate`'s result to floating-point tolerance.
    ///
    /// See `PropagationStats.peak_terms` for what "peak" does and does not
    /// mean.
    ///
    /// `engine` and `small_sum_threshold` work exactly as in `propagate`, and
    /// the recorded counts are the same records in the same order whichever
    /// engine ran the layer — which is what makes this the way to check the two
    /// engines against each other.
    ///
    /// `partitions` and `pin_memory` work exactly as in `propagate` too. A
    /// partitioned call additionally fills `PropagationStats.partition` with a
    /// `PartitionStats`: per layer, which partition held how many terms and how
    /// many rows crossed a boundary. The layer-level `terms_in`/`terms_out`
    /// are then the per-partition counts summed, so they stay comparable with
    /// an unpartitioned run of the same circuit.
    ///
    /// `comm` and `result` likewise. A distributed call fills
    /// `PropagationStats.partition` with a `PartitionStats` whose `partitions`
    /// and `size` are the group size and whose `rank` is this rank — but whose
    /// per-layer lists hold **this rank's entry only**, because gathering the
    /// group's counters would mean a collective per layer for a diagnostic.
    /// The layer-level `terms_in`/`terms_out` are therefore this rank's too,
    /// and `final_terms` is the length of what this rank got back (zero on a
    /// non-root rank under `result="gather"`). Reduce over `comm` for the
    /// group's picture.
    #[pyo3(signature = (circuit, policy=None, direction=None, engine=None, small_sum_threshold=None, partitions=None, pin_memory=true, comm=None, result="gather"))]
    #[allow(clippy::too_many_arguments)]
    fn propagate_with_stats(
        &self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
        engine: Option<&str>,
        small_sum_threshold: Option<usize>,
        partitions: Option<&Bound<'_, PyAny>>,
        pin_memory: bool,
        comm: Option<&Bound<'_, PyAny>>,
        result: &str,
    ) -> PyResult<(Self, PropagationStats)> {
        let dir = parse_direction(direction)?;
        let options = parse_engine(engine, small_sum_threshold)?;
        let gather = parse_result(result)?;
        check_num_qubits(&self.inner, circuit)?;
        let no_op = PolicySpec::NoOp;
        let spec: &PolicySpec = match policy {
            Some(p) => &p.spec,
            None => &no_op,
        };
        let mode = parse_run_mode(py, partitions, pin_memory, comm, gather, spec)?;
        // GIL released for the propagation, as in `propagate` above. The trace
        // is produced inside the closure and moved out with the sum —
        // `LayerScratch` is not `Send`-shared with anything, it is built and
        // dropped within this call.
        let mut recorded: Option<RunTrace> = None;
        // The closure is `move` (it owns `mode`, whose distributed variant
        // owns the communicator), so the trace escapes through a borrow taken
        // before it rather than by capturing `recorded` itself.
        let slot = &mut recorded;
        let inner = py.allow_threads(move || -> Result<PauliSumImpl, PropagateFailure> {
            Ok(for_each_width_propagate!(
                &self.inner,
                &circuit.inner,
                |s, c, W| {
                    let (out, trace) = mode.run_traced::<W>(c, s, spec, dir, options)?;
                    *slot = Some(trace);
                    out
                },
                else {
                    // Unreachable for the same reason as in `propagate`.
                    return Err(PropagateFailure::WidthMismatch);
                }
            ))
        })?;
        let stats = PropagationStats::from_run_trace(
            recorded.expect("the layer loop always records a trace"),
            inner.len(),
        );
        Ok((Self { inner }, stats))
    }
}
