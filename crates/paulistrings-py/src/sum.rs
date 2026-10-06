//! Python `PauliSum` class with width-monomorphized backing storage. See
//! ARCHITECTURE.md §Width and ARCHITECTURE.md §Python-Bindings.

#[cfg(feature = "cuda")]
use crate::truncation_spec::COLLAPSE_SAMPLE_DEVICE_MSG;
use crate::truncation_spec::{collapse_count, collapses_since, PyTruncation, TOPN_PARTITIONED_MSG};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use paulistrings::accumulator::BuildAccumulator;
use paulistrings::engine::partitioned::{numa_nodes, CpuSet};
use paulistrings::pauli_string::PauliString;
use paulistrings::phase::Phase;
use paulistrings::truncation::BuiltinTruncation;
use paulistrings::{
    propagate_with_scratch_and_options, Circuit as CoreCircuit, Direction, EngineSelection,
    GateTrace, LayerScratch, PartitionConfig, PartitionRowPolicy, PartitionRows, PartitionRuntime,
    PartitionTrace, PartitionedSum, PauliAxis, PauliSum as CorePauliSum, Placement, ProductBasis,
    ProductState, PropagateOptions, RotationAxis, StabilizerState, TopologyError,
    DEFAULT_SMALL_SUM_THRESHOLD,
};
use pyo3::exceptions::{PyNotImplementedError, PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyComplex, PyDict};
use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

/// Width-dispatch enum. The Python boundary picks the smallest width that fits `num_qubits` and stores the appropriately monomorphized `PauliSum`.
#[derive(Clone)]
pub enum PauliSumImpl {
    W1(CorePauliSum<1>),
    W2(CorePauliSum<2>),
    W4(CorePauliSum<4>),
    W8(CorePauliSum<8>),
    W16(CorePauliSum<16>),
}

impl PauliSumImpl {
    /// Pick the smallest supported width for `num_qubits`. Returns `None` if `num_qubits` exceeds the largest monomorphized width (1024 qubits).
    pub fn empty_for(num_qubits: usize) -> Option<Self> {
        for_num_qubits!(num_qubits, |W| CorePauliSum::<W>::empty(num_qubits))
    }

    pub fn num_qubits(&self) -> usize {
        for_each_width!(self, |s| s.num_qubits())
    }

    pub fn len(&self) -> usize {
        for_each_width!(self, |s| s.len())
    }

    /// Current bucket count, `1 << hash().bits()`. Reflects whatever a prior `propagate` call left resident (`rebucket` is grow-only), not a request.
    pub fn num_buckets(&self) -> usize {
        for_each_width!(self, |s| s.num_buckets())
    }

    /// Uniform product state: the same `+1` eigenstate on every qubit.
    pub fn expectation_uniform(&self, state: ProductState) -> Complex64 {
        for_each_width!(self, |s| s.expectation_product_state(state))
    }

    /// Per-qubit product state: entry `q` is qubit `q`'s `(axis, minus)`. Caller already checked one entry per qubit, so the resulting masks have no bit set past `num_qubits`.
    pub fn expectation_labels(&self, axes: &[(PauliAxis, bool)]) -> Complex64 {
        for_each_width!(self, |s| s.expectation_product_basis(
            &ProductBasis::from_axes(axes.iter().copied())
        ))
    }

    /// Stabilizer state given by one signed Pauli generator per qubit. Parsed at the active width and validated by the core, so a malformed string or an invalid generator set surfaces as a `ValueError`.
    pub fn expectation_stabilizer(&self, generators: &[String]) -> PyResult<Complex64> {
        for_each_width!(self, |s| stabilizer_expectation(s, generators))
    }

    pub fn identity_coefficient(&self) -> Complex64 {
        for_each_width!(self, |s| s.identity_coefficient())
    }

    /// Caller checked `sites` with [`check_sites`], since the core panics on a bad list.
    pub fn anticommute_histogram(&self, sites: &[usize], axis: RotationAxis) -> Vec<f64> {
        for_each_width!(self, |s| s.anticommute_histogram(sites, axis))
    }

    /// Caller checked `sites` with [`check_sites`], since the core panics on a bad list.
    pub fn rotated_overlap(&self, sites: &[usize], delta: f64, axis: RotationAxis) -> f64 {
        for_each_width!(self, |s| s.rotated_overlap(sites, delta, axis))
    }

    /// `None` when the two sums were monomorphized at different widths, which can only happen if their qubit counts fall in different dispatch bands.
    pub fn overlap(&self, other: &Self) -> Option<Complex64> {
        for_each_width_pair!((self, other), |a, b| a.overlap(b))
    }

    /// `self + factor · other`, matching strings combined and the rest kept; `None` on a width mismatch.
    /// Caller checks the qubit counts first: equal widths are not equal qubit counts, and the core's merge asserts on the latter.
    pub fn add_scaled(&self, other: &Self, factor: Complex64) -> Option<Self> {
        for_each_width_pair_rewrap!((self, other), |a, b, wrap| {
            if factor == Complex64::new(1.0, 0.0) {
                wrap(a.add(b))
            } else {
                let mut scaled = b.clone();
                scaled.scale(factor);
                wrap(a.add(&scaled))
            }
        })
    }

    /// Multiply every coefficient by `factor`, in place.
    pub fn scale(&mut self, factor: Complex64) {
        for_each_width!(self, |s| s.scale(factor))
    }

    /// First `k` terms in canonical order, decoded to `(label, coefficient)` — `O(k)`, not `O(len())`, so `__repr__`/`__str__` stay cheap on a huge sum.
    pub fn preview(&self, k: usize) -> Vec<(String, Complex64)> {
        fn preview_of<const W: usize>(
            s: &CorePauliSum<W>,
            num_qubits: usize,
            k: usize,
        ) -> Vec<(String, Complex64)> {
            s.iter()
                .take(k)
                .map(|(x, z, c)| {
                    let key = PauliString::<W> { x: *x, z: *z };
                    (crate::pauli_string::label_of(&key, num_qubits), c)
                })
                .collect()
        }
        let num_qubits = self.num_qubits();
        for_each_width!(self, |s| preview_of(s, num_qubits, k))
    }

    /// Snapshot of the coefficient column, in the sum's canonical order (partition-bucket index ascending, then lexicographic `(x, z)`; equal to plain lex order for sums of ≤ 1024 terms).
    pub fn coeffs(&self) -> Vec<Complex64> {
        fn coeffs_of<const W: usize>(s: &CorePauliSum<W>) -> Vec<Complex64> {
            let (_, _, c) = s.to_arrays();
            c
        }
        for_each_width!(self, |s| coeffs_of(s))
    }

    /// `(width, x_flat, z_flat)` snapshot of the SoA columns, in the sum's canonical order (see [`Self::coeffs`]), the same across the three arrays since the order is a deterministic function of the sum.
    /// `x_flat`/`z_flat` have length `len() * width`; caller reshapes to `(len, width)`.
    pub fn xz_flat(&self) -> (usize, Vec<u64>, Vec<u64>) {
        fn flatten<const W: usize>(rows: &[[u64; W]]) -> Vec<u64> {
            // Flat-copy via iteration; W is small (≤16) and this is not on the hot path.
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

    /// Build from a `{pauli_string: coefficient}` Python dict. `num_qubits` is inferred from the first key's length when `None`.
    pub fn from_strings_dict(
        terms: &Bound<'_, PyDict>,
        num_qubits: Option<usize>,
    ) -> PyResult<Self> {
        let num_qubits = match num_qubits {
            Some(n) => n,
            None => match terms.iter().next() {
                Some((key, _)) => {
                    let s: String = key.extract().map_err(|_| {
                        PyTypeError::new_err("PauliSum.from_strings keys must be str")
                    })?;
                    s.chars().count()
                }
                None => {
                    return Err(PyValueError::new_err(
                        "PauliSum.from_strings: cannot infer num_qubits from an empty dict; pass num_qubits explicitly",
                    ))
                }
            },
        };
        for_num_qubits!(num_qubits, |W| parse_terms::<W>(num_qubits, terms)?).ok_or_else(|| {
            PyValueError::new_err("num_qubits exceeds largest monomorphized width (1024)")
        })
    }

    /// Build from two equal-length sequences, `labels` (`I/X/Y/Z` strings) and `coefficients`. `num_qubits` is inferred from the first label's length when `None`. A label repeated in `labels` accumulates rather than overwriting, unlike a dict's keys.
    pub fn from_label_list(
        labels: &Bound<'_, PyAny>,
        coefficients: &Bound<'_, PyAny>,
        num_qubits: Option<usize>,
    ) -> PyResult<Self> {
        let labels: Vec<String> = labels.extract().map_err(|_| {
            PyTypeError::new_err("PauliSum.from_strings: labels must be a sequence of str")
        })?;
        let n_coeffs = coefficients.len()?;
        if labels.len() != n_coeffs {
            return Err(PyValueError::new_err(format!(
                "PauliSum.from_strings: {} labels but {} coefficients",
                labels.len(),
                n_coeffs
            )));
        }
        let num_qubits = match num_qubits {
            Some(n) => n,
            None => labels.first().map(|s| s.chars().count()).ok_or_else(|| {
                PyValueError::new_err(
                    "PauliSum.from_strings: cannot infer num_qubits from zero terms; pass num_qubits explicitly",
                )
            })?,
        };
        for_num_qubits!(num_qubits, |W| parse_label_list::<W>(
            num_qubits,
            &labels,
            coefficients
        )?)
        .ok_or_else(|| {
            PyValueError::new_err("num_qubits exceeds largest monomorphized width (1024)")
        })
    }

    /// Single-term sum from a `PauliString`'s key and a coefficient — `PauliString.__mul__`'s body.
    /// `coeff` may be exactly zero; `BuildAccumulator::finalize` already drops it, giving the empty sum.
    pub fn from_single(
        term: &crate::pauli_string::PauliStringImpl,
        num_qubits: usize,
        coeff: Complex64,
    ) -> Self {
        fn build<const W: usize>(
            p: &PauliString<W>,
            num_qubits: usize,
            coeff: Complex64,
        ) -> CorePauliSum<W> {
            let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, 1);
            acc.add_term(*p, Phase::ONE, coeff);
            acc.finalize()
        }
        use crate::pauli_string::PauliStringImpl as PS;
        match term {
            PS::W1(p) => PauliSumImpl::W1(build(p, num_qubits, coeff)),
            PS::W2(p) => PauliSumImpl::W2(build(p, num_qubits, coeff)),
            PS::W4(p) => PauliSumImpl::W4(build(p, num_qubits, coeff)),
            PS::W8(p) => PauliSumImpl::W8(build(p, num_qubits, coeff)),
            PS::W16(p) => PauliSumImpl::W16(build(p, num_qubits, coeff)),
        }
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
/// Format matches the test helper in `pauli_sum.rs`: character `i` describes qubit `i`, coefficients multiply the literal Hermitian Pauli string (ARCHITECTURE.md §Data-Model).
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

/// Build a `PauliSum<W>` from parallel `labels`/`coefficients` sequences. Unlike [`parse_terms`]'s dict, a label repeated in `labels` is not an error — `BuildAccumulator::add_term` sums it, the same accumulation the manual's Hamiltonian example does by hand with a dict.
fn parse_label_list<const W: usize>(
    num_qubits: usize,
    labels: &[String],
    coefficients: &Bound<'_, PyAny>,
) -> PyResult<CorePauliSum<W>> {
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, labels.len());
    for (i, s) in labels.iter().enumerate() {
        if s.len() != num_qubits {
            return Err(PyValueError::new_err(format!(
                "Pauli string {:?} has length {}, expected {} (length must match num_qubits)",
                s,
                s.len(),
                num_qubits
            )));
        }
        let c = extract_complex(&coefficients.get_item(i)?)?;
        acc.add_term(parse_pauli_key::<W>(s)?, Phase::ONE, c);
    }
    Ok(acc.finalize())
}

/// Parse an `I/X/Y/Z` label into a symplectic key (the crate's Hermitian convention: `Y` maps to `(x=1, z=1)` with no phase factor).
/// Caller checks the label's length against `num_qubits` first; this only rejects characters outside the alphabet.
pub(crate) fn parse_pauli_key<const W: usize>(s: &str) -> PyResult<PauliString<W>> {
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

/// Contract `sum` against the stabilizer state spelled by `generators` (signed Pauli strings, same alphabet and indexing as `from_strings`).
/// The core validates the set and its `StabilizerError` becomes the `ValueError` message verbatim.
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

/// Bit mask of the qubits `word` (a `64·word .. 64·(word+1)` slice) actually covers within `num_qubits`; any set bit outside this mask addresses a qubit that does not exist.
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
/// Rows are symplectic keys in the Hermitian convention (matching `parse_terms`); a row narrower than `W` words is zero-padded, a wider one is a `ValueError`.
/// Ingest goes through `BuildAccumulator`, so duplicate `(x, z)` rows sum their coefficients and exact-`0+0i` rows are dropped.
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
pub(crate) fn extract_complex(val: &Bound<'_, PyAny>) -> PyResult<Complex64> {
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

/// `"x"` or `"z"`, either case, as the echo read-outs' generator axis.
fn parse_axis(axis: &str) -> PyResult<RotationAxis> {
    match axis {
        "x" | "X" => Ok(RotationAxis::X),
        "z" | "Z" => Ok(RotationAxis::Z),
        other => Err(PyValueError::new_err(format!(
            "axis must be 'x' or 'z', got {other:?}"
        ))),
    }
}

/// The echo read-outs' `sites`: distinct qubits of the register, which the core asserts.
fn check_sites(sites: &[usize], num_qubits: usize) -> PyResult<()> {
    let mut seen = HashSet::new();
    for &q in sites {
        if q >= num_qubits {
            return Err(PyValueError::new_err(format!(
                "site {q} is out of range for num_qubits={num_qubits}"
            )));
        }
        if !seen.insert(q) {
            return Err(PyValueError::new_err(format!("site {q} is listed twice")));
        }
    }
    Ok(())
}

/// `local()` on this rank, summed elementwise over `comm` with the core's all-reduce, which gives every rank the same bits. **Collective.**
#[cfg(feature = "mpi")]
fn reduce_over_comm(
    py: Python<'_>,
    comm: &Bound<'_, PyAny>,
    local: impl FnOnce() -> Vec<f64> + Send,
) -> PyResult<Vec<f64>> {
    use paulistrings::engine::partitioned::Collectives;
    let transport = crate::mpi::transport_from_comm(py, comm)?;
    Ok(py.allow_threads(move || {
        let mut values = local();
        transport.allreduce_sum_f64(&mut values);
        values
    }))
}

#[cfg(not(feature = "mpi"))]
fn reduce_over_comm(
    _py: Python<'_>,
    _comm: &Bound<'_, PyAny>,
    _local: impl FnOnce() -> Vec<f64> + Send,
) -> PyResult<Vec<f64>> {
    Err(mpi_unavailable_error())
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

/// `"sorted"` (default), `"auto"` or `"direct"`, paired with an optional small-sum threshold and the per-layer bucket-sizing knobs, as a core [`PropagateOptions`].
/// `None`/`None`/`None`/`None` is `PropagateOptions::default()` exactly, so the kwargs are additive and omitting them changes nothing; parsed once at the boundary, outside the width dispatch.
/// Shared by `propagate` and `propagate_with_stats`, like `parse_direction`.
pub(crate) fn parse_engine(
    engine: Option<&str>,
    small_sum_threshold: Option<usize>,
    target_bucket_len: Option<usize>,
    min_buckets: Option<usize>,
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
    let defaults = PropagateOptions::default();
    Ok(PropagateOptions {
        engine,
        small_sum_threshold: small_sum_threshold.unwrap_or(DEFAULT_SMALL_SUM_THRESHOLD),
        target_bucket_len: target_bucket_len.unwrap_or(defaults.target_bucket_len),
        min_buckets: min_buckets.unwrap_or(defaults.min_buckets),
    })
}

/// `partitions=` / `pin_memory=` / `partition_row_seed=` → an optional core [`PartitionConfig`]; `None` means the classic unpartitioned path, bit for bit today's behaviour.
/// Accepted: `None`/`1` (classic), `"auto"` (one partition per NUMA node, or classic on a single-node box), an `int` power of two `>= 2` (capped at that many partitions), or `list[list[int]]` (one partition per CPU list). Anything else is a `TypeError`; a malformed value of an accepted shape is a `ValueError`.
/// `partition_row_seed=None` (the default) reproduces today's behaviour exactly: the engine falls back to the sum's own hash seed, same as before this knob existed.
/// Resolved against the machine here, before the GIL is released, so a bad CPU list is an exception rather than a failure inside the run; resolved again by [`PartitionRuntime::new`] (see [`runtime_for`]).
fn parse_partitions(
    partitions: Option<&Bound<'_, PyAny>>,
    pin_memory: bool,
    partition_row_seed: Option<u64>,
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
        partition_row_seed,
    };
    let slots = config.resolve().map_err(topology_error)?;
    if slots.len() == 1 && matches!(config.placement, Placement::Auto { .. }) {
        // A single-node box: nothing to partition, so run the classic path rather than pay for a pool build and pin a thread nobody asked to pin. An explicit one-element CPU list is still honoured.
        return Ok(None);
    }
    Ok(Some(config))
}

/// `partition_row_blocks=` → an explicit "cut" policy, one disjoint qubit block per partition, fed straight to the core [`PartitionRows::cut`](paulistrings::PartitionRows::cut).
/// `None` (the default) means no override — the caller's `partition_row_seed`/the sum's own hash seed picks GF(2)-random rows instead, unchanged from before this knob existed.
fn parse_partition_row_blocks(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Vec<u32>>>> {
    let Some(obj) = obj else {
        return Ok(None);
    };
    if obj.is_none() {
        return Ok(None);
    }
    let blocks: Vec<Vec<u32>> = obj.extract().map_err(|_| {
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
    blocks: &[Vec<u32>],
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
            let qi = q as usize;
            if qi >= num_qubits {
                return Err(format!(
                    "partition_row_blocks[{b}] names qubit {q}, out of range for \
                     num_qubits={num_qubits}"
                ));
            }
            if seen[qi] {
                return Err(format!(
                    "partition_row_blocks: qubit {q} appears in more than one block; \
                     blocks must be disjoint"
                ));
            }
            seen[qi] = true;
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
        let reachable = blocks
            .iter()
            .enumerate()
            .any(|(b, qubits)| (b >> bit) & 1 == 1 && !qubits.is_empty());
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
    blocks: &[Vec<u32>],
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
) -> PyResult<Option<(Vec<u32>, Vec<u32>)>> {
    let Some(obj) = obj.filter(|obj| !obj.is_none()) else {
        return Ok(None);
    };
    let shape = "partition_row_exclude must be None or a dict {\"x\": [qubits], \"z\": [qubits]} \
                 naming the qubits whose x-bit / z-bit no partition row may read";
    let dict = obj
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err(shape))?;
    let (mut exclude_x, mut exclude_z) = (Vec::new(), Vec::new());
    for (key, value) in dict.iter() {
        let key: String = key.extract().map_err(|_| PyTypeError::new_err(shape))?;
        let qubits: Vec<u32> = value.extract().map_err(|_| PyTypeError::new_err(shape))?;
        if let Some(q) = qubits.iter().find(|&&q| q as usize >= num_qubits) {
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
    let covers = |qubits: &[u32]| qubits.iter().collect::<HashSet<_>>().len() == num_qubits;
    if num_qubits > 0 && covers(&exclude_x) && covers(&exclude_z) {
        return Err(PyValueError::new_err(
            "partition_row_exclude excludes every coordinate of the register, leaving no row to \
             name a partition by",
        ));
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

fn runtime_for(config: &PartitionConfig) -> Result<Arc<PartitionRuntime>, TopologyError> {
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

/// The per-layer records a run produced, tagged by which engine produced them.
enum RunTrace {
    /// The unpartitioned engine's per-gate trace: term counts, gate identity and elapsed time.
    Term(GateTrace),
    /// An in-process partitioned run's records, plus the partition count.
    Partition(PartitionTrace, usize),
    /// A distributed run's records — **this rank's only** — plus `(rank, size)` and this rank's device, if it ran on one.
    #[cfg(feature = "mpi")]
    Distributed(PartitionTrace, u32, u32, Option<u32>),
    /// A device run's records, plus each partition's device ordinal.
    #[cfg(feature = "cuda")]
    Device(PartitionTrace, Vec<u32>),
}

/// A [`PartitionRows`] at the width of the sum it splits.
#[derive(Clone)]
#[cfg_attr(not(feature = "mpi"), allow(dead_code))]
enum RowsImpl {
    W1(PartitionRows<1>),
    W2(PartitionRows<2>),
    W4(PartitionRows<4>),
    W8(PartitionRows<8>),
    W16(PartitionRows<16>),
}

macro_rules! rows_impl_from {
    ($($w:literal => $variant:ident),*) => {
        $(impl From<PartitionRows<$w>> for RowsImpl {
            fn from(rows: PartitionRows<$w>) -> Self {
                Self::$variant(rows)
            }
        })*
    };
}
rows_impl_from!(1 => W1, 2 => W2, 4 => W4, 8 => W8, 16 => W16);

#[cfg_attr(not(feature = "mpi"), allow(dead_code))]
impl RowsImpl {
    /// [`PartitionRows::keeps_flip_classes`] at this width.
    fn keeps_flip_classes(&self, sites: &[usize], axis: RotationAxis) -> bool {
        for_each_width!(self, |rows| rows.keeps_flip_classes(sites, axis))
    }
}

/// Where a `comm=` result sits in its group: what lets a collective read-out know that per-rank values add up to the whole.
#[derive(Clone)]
#[cfg_attr(not(feature = "mpi"), allow(dead_code))]
pub(crate) struct Share {
    rank: u32,
    size: u32,
    /// The rows the group's sum is split by; no rows for a gathered result, which rank 0 holds whole.
    rows: RowsImpl,
}

/// What one run hands back besides the sum.
struct RunOutput<const W: usize> {
    sum: CorePauliSum<W>,
    trace: Option<RunTrace>,
    /// A distributed run's place in its group.
    share: Option<Share>,
    /// The call's collapses, `None` without a sampler in the policy.
    collapses: Option<u64>,
}

#[cfg(feature = "mpi")]
impl<const W: usize> RunOutput<W>
where
    RowsImpl: From<PartitionRows<W>>,
{
    /// This rank's side of a `comm=` run, host or device.
    fn from_mpi(out: crate::mpi::MpiOutcome<W>) -> Self {
        let (rank, size) = (out.rank, out.size);
        Self {
            sum: out.sum,
            trace: out
                .trace
                .map(|trace| RunTrace::Distributed(trace, rank, size, out.device)),
            share: Some(Share {
                rank,
                size,
                rows: RowsImpl::from(out.rows),
            }),
            collapses: out.collapses,
        }
    }
}

impl RunMode {
    /// Run `circuit` over `sum`, recording the per-layer trace when `traced`. Called inside `allow_threads`; consumes the mode, so a distributed run's communicator is freed before it returns.
    fn run<const W: usize>(
        self,
        circuit: &CoreCircuit<W>,
        sum: &CorePauliSum<W>,
        policy: &BuiltinTruncation,
        direction: Direction,
        options: PropagateOptions,
        traced: bool,
    ) -> Result<RunOutput<W>, PropagateFailure>
    where
        RowsImpl: From<PartitionRows<W>>,
    {
        let before = collapse_count(policy);
        match self {
            RunMode::Classic => {
                let mut scratch = LayerScratch::<W>::new();
                if traced {
                    scratch.enable_gate_trace();
                }
                let out = propagate_with_scratch_and_options(
                    circuit,
                    sum.clone(),
                    policy,
                    direction,
                    &mut scratch,
                    options,
                );
                Ok(RunOutput {
                    sum: out,
                    trace: scratch.take_gate_trace().map(RunTrace::Term),
                    share: None,
                    collapses: collapses_since(policy, before),
                })
            }
            RunMode::Partitioned(config, rows) => {
                // The runtime (and its pinned pools) is cached per config, so
                // a Trotter loop of many short calls builds it once.
                let runtime = runtime_for(&config).map_err(PropagateFailure::Topology)?;
                let bits = runtime.num_partitions().trailing_zeros() as u8;
                let rows = rows.rows::<W>(sum.num_qubits(), bits, sum.hash().seed());
                let mut split = PartitionedSum::scatter_with_rows(sum.clone(), rows, runtime);
                if traced {
                    split.enable_trace();
                }
                split.propagate_with_options(circuit, policy, direction, options);
                // The partition count from the runtime, not the trace: a zero-layer circuit records no layer, but the placement is still worth reporting.
                let partitions = split.num_partitions();
                let trace = traced.then(|| {
                    RunTrace::Partition(split.take_trace().unwrap_or_default(), partitions)
                });
                Ok(RunOutput {
                    sum: split.into_gathered(),
                    trace,
                    share: None,
                    collapses: collapses_since(policy, before),
                })
            }
            #[cfg(feature = "mpi")]
            RunMode::Distributed(run) => run
                .propagate(circuit, sum, policy, direction, options, traced)
                .map(RunOutput::from_mpi)
                .map_err(PropagateFailure::Topology),
            #[cfg(feature = "cuda")]
            RunMode::Devices(config, devices, rows) => {
                let (out, trace) = run_devices(
                    &config, &rows, circuit, sum, policy, direction, options, traced,
                )?;
                Ok(RunOutput {
                    sum: out,
                    trace: trace.map(|trace| RunTrace::Device(trace, devices)),
                    share: None,
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

/// Scatter `sum` over the device partitions `config` places by `rows`, propagate, and gather.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_devices<const W: usize>(
    config: &PartitionConfig,
    rows: &PartitionRowPolicy,
    circuit: &CoreCircuit<W>,
    sum: &CorePauliSum<W>,
    policy: &BuiltinTruncation,
    direction: Direction,
    options: PropagateOptions,
    traced: bool,
) -> Result<(CorePauliSum<W>, Option<PartitionTrace>), PropagateFailure> {
    use paulistrings::gpu::GpuPartitionedSum;
    // Cached like a host runtime: each slot's pool binds its device context once, not per call.
    let runtime = runtime_for(config).map_err(PropagateFailure::Topology)?;
    let bits = runtime.num_partitions().trailing_zeros() as u8;
    let rows = rows.rows::<W>(sum.num_qubits(), bits, sum.hash().seed());
    let split = GpuPartitionedSum::scatter_to_devices_with_rows(sum, rows, runtime);
    let mut split = split.map_err(PropagateFailure::Gpu)?;
    if traced {
        split.enable_trace();
    }
    split
        .propagate_with_options(circuit, policy, direction, options)
        .map_err(PropagateFailure::Gpu)?;
    let trace = split.take_trace();
    let out = split.gather().map_err(PropagateFailure::Gpu)?;
    Ok((out, trace))
}

/// The rows a partitioned run splits by: a cut, the seeded draw minus the excluded coordinates, or the plain seeded draw.
fn row_policy(
    row_blocks: Option<Vec<Vec<u32>>>,
    exclude: Option<(Vec<u32>, Vec<u32>)>,
    seed: Option<u64>,
) -> PartitionRowPolicy {
    match (row_blocks, exclude) {
        (Some(blocks), _) => PartitionRowPolicy::Cut(blocks),
        (None, Some((exclude_x, exclude_z))) => PartitionRowPolicy::SeededExcluding {
            seed,
            exclude_x,
            exclude_z,
        },
        (None, None) => PartitionRowPolicy::Seeded(seed),
    }
}

/// Turn the placement kwargs into a [`RunMode`], with the GIL held.
/// Order matters: everything that can raise on the caller's spelling is decided before the communicator is adopted, since adopting it is collective — a rank that raises early never enters the collective, so the whole group raises together.
#[allow(clippy::too_many_arguments)]
fn parse_run_mode(
    py: Python<'_>,
    partitions: Option<&Bound<'_, PyAny>>,
    pin_memory: bool,
    partition_row_seed: Option<u64>,
    partition_row_blocks: Option<&Bound<'_, PyAny>>,
    partition_row_exclude: Option<&Bound<'_, PyAny>>,
    comm: Option<&Bound<'_, PyAny>>,
    gather: bool,
    device: Option<&Bound<'_, PyAny>>,
    policy: &BuiltinTruncation,
    num_qubits: usize,
) -> PyResult<RunMode> {
    let distributed = comm_requested(comm);
    if let Some(device) = device.filter(|obj| !obj.is_none()) {
        return parse_device_mode(
            py,
            device,
            partitions.is_some_and(|obj| !obj.is_none()),
            comm.filter(|_| distributed),
            gather,
            partition_row_seed,
            partition_row_blocks,
            partition_row_exclude,
            policy,
            num_qubits,
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
    let row_blocks = parse_partition_row_blocks(partition_row_blocks)?;
    if partition_row_seed.is_some() && row_blocks.is_some() {
        return Err(PyValueError::new_err(
            "partition_row_seed= and partition_row_blocks= are alternatives (a seeded random \
             draw vs. an explicit locality cut); pass at most one",
        ));
    }
    let exclude = parse_partition_row_exclude(partition_row_exclude, num_qubits)?;
    if exclude.is_some() && row_blocks.is_some() {
        return Err(PyValueError::new_err(
            "partition_row_blocks= and partition_row_exclude= are alternatives (an explicit \
             locality cut vs. a seeded draw that skips some coordinates); pass at most one",
        ));
    }
    // Checked against the kwarg rather than the resolved config: `partitions="auto"` on a one-node box runs unpartitioned, where an exclusion is merely moot.
    if exclude.is_some() && !distributed && partitions.is_none_or(|obj| obj.is_none()) {
        return Err(PyValueError::new_err(
            "partition_row_exclude= needs partitions= or comm= (it has no effect on the \
             unpartitioned path)",
        ));
    }
    let config = parse_partitions(partitions, pin_memory, partition_row_seed)?;
    if (distributed || config.is_some()) && policy.contains_exact_top_n() {
        return Err(topn_partitioned_error(if distributed {
            None
        } else {
            partitions
        }));
    }
    let rows = row_policy(row_blocks, exclude, partition_row_seed);
    if distributed {
        #[cfg(feature = "mpi")]
        {
            let comm = comm.expect("comm is Some");
            if let PartitionRowPolicy::Cut(blocks) = &rows {
                // `MPI_Comm_size` is local and every rank reads the same
                // number, so validating here still rejects on every rank alike
                // — before the collective duplicate below.
                let ranks = crate::mpi::comm_size(comm)?;
                validate_partition_row_blocks(
                    blocks,
                    num_qubits,
                    ranks,
                    &format!("comm= with {ranks} rank(s)"),
                )?;
            }
            let transport = crate::mpi::transport_from_comm(py, comm)?;
            return Ok(RunMode::Distributed(crate::mpi::MpiRun::new(
                transport, rows, gather,
            )));
        }
        #[cfg(not(feature = "mpi"))]
        {
            let _ = (py, gather, rows);
            return Err(mpi_unavailable_error());
        }
    }
    if let (Some(config), PartitionRowPolicy::Cut(row_blocks)) = (&config, &rows) {
        // `resolve()` alone (not `PartitionRuntime::new`, which builds pinned
        // pools) is enough to learn the partition count for validation.
        let num_partitions = config.resolve().map_err(topology_error)?.len();
        validate_partition_row_blocks(
            row_blocks,
            num_qubits,
            num_partitions,
            &format!("partitions={num_partitions}"),
        )?;
    } else if config.is_none() && matches!(rows, PartitionRowPolicy::Cut(_)) {
        return Err(PyValueError::new_err(
            "partition_row_blocks= needs partitions= or comm= (it has no effect on the \
             unpartitioned path)",
        ));
    }
    Ok(match config {
        Some(config) => RunMode::Partitioned(config, rows),
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
    partition_row_seed: Option<u64>,
    partition_row_blocks: Option<&Bound<'_, PyAny>>,
    partition_row_exclude: Option<&Bound<'_, PyAny>>,
    policy: &BuiltinTruncation,
    num_qubits: usize,
) -> PyResult<RunMode> {
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
    let shown = device
        .repr()
        .map_or_else(|_| "…".to_string(), |repr| repr.to_string());
    let shown = format!("device={shown}");
    let row_blocks = parse_partition_row_blocks(partition_row_blocks)?;
    if partition_row_seed.is_some() && row_blocks.is_some() {
        return Err(PyValueError::new_err(
            "partition_row_seed= and partition_row_blocks= are alternatives (a seeded random \
             draw vs. an explicit locality cut); pass at most one",
        ));
    }
    let exclude = parse_partition_row_exclude(partition_row_exclude, num_qubits)?;
    if exclude.is_some() && row_blocks.is_some() {
        return Err(PyValueError::new_err(
            "partition_row_blocks= and partition_row_exclude= are alternatives (an explicit \
             locality cut vs. a seeded draw that skips some coordinates); pass at most one",
        ));
    }
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
        let _ = (py, request, row_blocks, exclude, num_qubits, policy);
        match comm {
            Some(_) => Err(crate::gpu::device_comm_unavailable_error()),
            None => Err(crate::gpu::cuda_unavailable_error()),
        }
    }
    #[cfg(feature = "cuda")]
    {
        let devices = match comm {
            Some(_) => None,
            None => Some(crate::gpu::resolve_devices(&request, &shown)?),
        };
        // Exact TopN runs at one partition only: one resolved device, never a comm= group (one partition per rank).
        if policy.contains_exact_top_n() && devices.as_ref().is_none_or(|d| d.len() > 1) {
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
                partition_row_seed,
                row_blocks,
                exclude,
                num_qubits,
            );
        };
        match &row_blocks {
            Some(_) if devices.len() == 1 => {
                return Err(PyValueError::new_err(format!(
                    "partition_row_blocks= needs partitions=, comm= or several devices (it has \
                     no effect on the one-device run {shown})"
                )));
            }
            Some(blocks) => validate_partition_row_blocks(
                blocks,
                num_qubits,
                devices.len(),
                &format!("{shown} ({} partitions)", devices.len()),
            )?,
            None => {}
        }
        if exclude.is_some() && devices.len() == 1 {
            return Err(PyValueError::new_err(format!(
                "partition_row_exclude= needs partitions=, comm= or several devices (it has no \
                 effect on the one-device run {shown})"
            )));
        }
        let rows = row_policy(row_blocks, exclude, partition_row_seed);
        let config = PartitionConfig {
            placement: Placement::Devices {
                devices: devices.clone(),
                per_device: 1,
            },
            bind_memory: false,
            partition_row_seed,
        };
        Ok(RunMode::Devices(config, devices, rows))
    }
}

/// `parse_device_mode` under `comm=`: validate the blocks against the group size, adopt the communicator, and agree this rank's device over the group.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn parse_distributed_device_mode(
    py: Python<'_>,
    comm: &Bound<'_, PyAny>,
    request: &crate::gpu::DeviceRequest,
    shown: &str,
    gather: bool,
    partition_row_seed: Option<u64>,
    row_blocks: Option<Vec<Vec<u32>>>,
    exclude: Option<(Vec<u32>, Vec<u32>)>,
    num_qubits: usize,
) -> PyResult<RunMode> {
    #[cfg(feature = "mpi")]
    {
        if let Some(blocks) = &row_blocks {
            let ranks = crate::mpi::comm_size(comm)?;
            validate_partition_row_blocks(
                blocks,
                num_qubits,
                ranks,
                &format!("comm= with {ranks} rank(s)"),
            )?;
        }
        let rows = row_policy(row_blocks, exclude, partition_row_seed);
        let transport = crate::mpi::transport_from_comm(py, comm)?;
        let rank = paulistrings::engine::partitioned::Collectives::rank(&transport);
        // Collective on every rank whatever each asked for, so an `"auto"` rank never waits on one that named an ordinal.
        let auto = paulistrings::gpu::local_device_for_comm(transport.communicator());
        let local = crate::gpu::resolve_rank_device(request, shown, rank, auto);
        let device = crate::mpi::agree_device(py, &transport, local, shown)?;
        Ok(RunMode::DistributedDevice(crate::mpi::MpiGpuRun::new(
            transport, rows, gather, device,
        )))
    }
    #[cfg(not(feature = "mpi"))]
    {
        let _ = (
            py,
            comm,
            request,
            shown,
            gather,
            partition_row_seed,
            row_blocks,
            exclude,
            num_qubits,
        );
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

    /// The partitioned run's own record, or `None` for an unpartitioned call (including `partitions="auto"` on a single-NUMA-node box).
    /// A `device=` run fills it with one partition per device, `devices` naming them.
    /// Its per-layer lists are indexed the same way as `terms_in` / `terms_out`.
    #[getter]
    fn partition(&self) -> Option<PartitionStats> {
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
    /// Derive the Python-facing record from a core [`GateTrace`] plus the
    /// length of the propagated sum (which is what "peak" falls back to when
    /// no layer ran).
    fn from_trace(trace: GateTrace, final_terms: usize) -> Self {
        debug_assert_eq!(trace.terms_in.len(), trace.terms_out.len());
        let peak_terms = trace
            .terms_in
            .first()
            .copied()
            .into_iter()
            .chain(trace.terms_out.iter().copied())
            .max()
            .unwrap_or(final_terms);
        Self {
            layers: trace.terms_out.len(),
            peak_terms,
            final_terms,
            terms_in: trace.terms_in,
            terms_out: trace.terms_out,
            partition: None,
            circuit_index: trace.circuit_index,
            application_index: trace.application_index,
            gate_name: trace.gate_name,
            nanos: trace.nanos,
            collapses: None,
        }
    }

    /// The same record from a partitioned run's [`PartitionTrace`], plus the per-partition detail in `partition`.
    /// The layer-level term counts are the per-partition ones summed, matching what the unpartitioned engine would record for the same layer — so a partitioned run is comparable to an unpartitioned one field by field.
    /// `nanos` is the **maximum** over partitions per layer (a critical-rank proxy, documented on the getter); `circuit_index`/`application_index`/`gate_name` are single values, identical across partitions by construction (lock-step application).
    /// A distributed run's records are this rank's only, so the term-count sum is a sum of one (documented on `PartitionStats.size`) and `nanos` is that one rank's own timing.
    fn from_partition_trace(
        trace: &PartitionTrace,
        partitions: usize,
        final_terms: usize,
        ranks: Option<(u32, u32)>,
        devices: Option<Vec<u32>>,
    ) -> Self {
        let sum_of = |counts: &[usize]| counts.iter().sum::<usize>();
        let gate_trace = GateTrace {
            circuit_index: trace.layers.iter().map(|l| l.circuit_index).collect(),
            application_index: trace.layers.iter().map(|l| l.application_index).collect(),
            gate_name: trace.layers.iter().map(|l| l.gate_name).collect(),
            terms_in: trace.layers.iter().map(|l| sum_of(&l.terms_in)).collect(),
            terms_out: trace.layers.iter().map(|l| sum_of(&l.terms_out)).collect(),
            nanos: trace
                .layers
                .iter()
                .map(|l| l.nanos.iter().copied().max().unwrap_or(0))
                .collect(),
        };
        Self {
            partition: Some(PartitionStats::from_trace(
                trace, partitions, ranks, devices,
            )),
            ..Self::from_trace(gate_trace, final_terms)
        }
    }

    /// Whichever of the three traces the run recorded, as one record.
    fn from_run_trace(trace: RunTrace, final_terms: usize, collapses: Option<u64>) -> Self {
        let stats = match trace {
            RunTrace::Term(trace) => Self::from_trace(trace, final_terms),
            RunTrace::Partition(trace, partitions) => {
                Self::from_partition_trace(&trace, partitions, final_terms, None, None)
            }
            #[cfg(feature = "mpi")]
            RunTrace::Distributed(trace, rank, size, device) => Self::from_partition_trace(
                &trace,
                size as usize,
                final_terms,
                Some((rank, size)),
                device.map(|d| vec![d]),
            ),
            #[cfg(feature = "cuda")]
            RunTrace::Device(trace, devices) => {
                Self::from_partition_trace(&trace, devices.len(), final_terms, None, Some(devices))
            }
        };
        Self { collapses, ..stats }
    }

    /// A one-device run's record: one partition, `partition.devices == [device]`.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_device_trace(
        trace: &PartitionTrace,
        device: u32,
        final_terms: usize,
    ) -> Self {
        Self::from_partition_trace(trace, 1, final_terms, None, Some(vec![device]))
    }
}

/// Per-layer, per-partition record of a partitioned propagation — the `partition` attribute of a [`PropagationStats`] from a `partitions=` or `comm=` call.
/// Every list is one entry per layer applied, in application order; `terms_in`/`terms_out` entries are themselves one entry per partition, in rank order.
/// Answers the two questions a partitioned run raises: how much crossed a partition boundary (`rows_exported`, `bytes_exported`, `local`), and how evenly terms were spread (`imbalance`).
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
}

#[pymethods]
impl PartitionStats {
    /// Number of partitions the run was split across — always a power of two. For a distributed (`comm=`) run this is the MPI group size (`size` below): one partition per rank.
    #[getter]
    fn partitions(&self) -> usize {
        self.partitions
    }

    /// This process's rank in the `comm=` group, or `None` for an in-process (`partitions=`) run.
    #[getter]
    fn rank(&self) -> Option<u32> {
        self.rank
    }

    /// The `comm=` group's size, or `None` for an in-process (`partitions=`) run.
    /// A distributed run's per-layer lists hold this rank's entry only (`terms_in[k]`/`terms_out[k]` one-element, `rows_exported[k]`/`bytes_exported[k]` this rank's, `imbalance[k]` always `1.0`) — nothing gathers the group's counters, since that would add a collective per layer for a diagnostic; reduce over `comm` yourself. The in-process case is unchanged: lists are `partitions` long.
    #[getter]
    fn size(&self) -> Option<u32> {
        self.size
    }

    /// The CUDA device ordinal of each partition for a `device=` run (this rank's one device under `comm=`), or `None` for a host run.
    #[getter]
    fn devices(&self) -> Option<Vec<u32>> {
        self.devices.clone()
    }

    /// Whether each layer was purely local: moved no row across a partition boundary. `local[k]` is exactly `rows_exported[k] == 0`.
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

    /// Every field but `nanos`, in getter order; `terms_in`/`terms_out` nested one level deeper (per layer per partition).
    fn __repr__(&self) -> String {
        // Spelled with Python's `True`/`False`/`None` rather than Rust's `Debug`, so the line pastes back into a REPL.
        let local = self
            .local
            .iter()
            .map(|&local| if local { "True" } else { "False" })
            .collect::<Vec<_>>()
            .join(", ");
        let show = |value: Option<u32>| value.map_or_else(|| "None".to_string(), |v| v.to_string());
        let devices = self
            .devices
            .as_ref()
            .map_or_else(|| "None".to_string(), |d| format!("{d:?}"));
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
        }
    }
}

#[pyclass(module = "paulistrings._paulistrings", name = "PauliSum")]
pub struct PauliSum {
    pub(crate) inner: PauliSumImpl,
    /// Set only on a `comm=` propagate result; anything that could break the split clears it.
    pub(crate) share: Option<Share>,
}

impl PauliSum {
    /// This sum's place in `comm`'s split, or the `ValueError` a collective read-out raises on a sum that has none. Every check is local, so a rejection happens on every rank alike before any rank enters the collective.
    #[cfg(feature = "mpi")]
    fn check_share(&self, comm: &Bound<'_, PyAny>, method: &str) -> PyResult<&Share> {
        let share = self.share.as_ref().ok_or_else(|| {
            PyValueError::new_err(format!(
                "{method}(comm=...) adds the per-rank values of a sum split across ranks, but \
                 this sum is not a propagate(..., comm=...) result (or was since combined with \
                 another sum); call it without comm= for this sum alone"
            ))
        })?;
        let size: u32 = comm.call_method0("Get_size")?.extract()?;
        let rank: u32 = comm.call_method0("Get_rank")?.extract()?;
        if (rank, size) != (share.rank, share.size) {
            return Err(PyValueError::new_err(format!(
                "{method}(comm=...): this sum is rank {} of a {}-rank split, but comm is rank \
                 {rank} of {size}; pass the communicator it was propagated over",
                share.rank, share.size,
            )));
        }
        Ok(share)
    }

    #[cfg(not(feature = "mpi"))]
    fn check_share(&self, _comm: &Bound<'_, PyAny>, _method: &str) -> PyResult<&Share> {
        Err(mpi_unavailable_error())
    }

    /// A sum with no place in a distributed split.
    pub(crate) fn plain(inner: PauliSumImpl) -> Self {
        Self { inner, share: None }
    }

    /// `self + factor · other`, with the qubit counts checked first — the core's merge asserts on a mismatch, and two different qubit counts can still share a width band.
    fn checked_add(&self, other: &PauliSumImpl, factor: Complex64) -> PyResult<PauliSumImpl> {
        if self.inner.num_qubits() != other.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "num_qubits mismatch ({} vs {})",
                self.inner.num_qubits(),
                other.num_qubits(),
            )));
        }
        self.inner
            .add_scaled(other, factor)
            .ok_or_else(|| PyValueError::new_err("sums were monomorphized at different widths"))
    }

    /// `self` scaled by a Python number. An exact-zero factor gives the empty sum, keeping the "no stored zero coefficient" invariant `from_strings` and the merge both hold.
    fn scaled(&self, factor: &Bound<'_, PyAny>) -> PyResult<PauliSumImpl> {
        let factor = scalar_factor(factor)?;
        if factor == Complex64::new(0.0, 0.0) {
            return PauliSumImpl::empty_for(self.inner.num_qubits())
                .ok_or_else(|| PyValueError::new_err("internal: width band lost"));
        }
        let mut inner = self.inner.clone();
        inner.scale(factor);
        Ok(inner)
    }

    /// `slf += factor · other`, in place.
    /// `a += a` aliases one Python object into both operands, so that case merges against a snapshot rather than taking two borrows of the same cell.
    fn add_in_place(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, Self>,
        factor: Complex64,
    ) -> PyResult<()> {
        let combined = if slf.is(other) {
            let this = slf.borrow();
            let snapshot = this.inner.clone();
            this.checked_add(&snapshot, factor)?
        } else {
            let this = slf.borrow();
            let that = other.borrow();
            this.checked_add(&that.inner, factor)?
        };
        let mut this = slf.borrow_mut();
        this.inner = combined;
        this.share = None;
        Ok(())
    }
}

/// How many terms `__repr__`/`__str__` show before falling back to `... (N more terms)`.
const PREVIEW_TERMS: usize = 4;

/// `0.25` for a real coefficient, `(0.25+0.5j)` otherwise — dropping the `+0j` tail Python's own `complex.__repr__` always carries, since most coefficients in this library are real.
fn format_coeff(c: Complex64) -> String {
    if c.im == 0.0 {
        format!("{}", c.re)
    } else {
        format!(
            "({}{}{}j)",
            c.re,
            if c.im < 0.0 { "-" } else { "+" },
            c.im.abs()
        )
    }
}

/// `PauliSum.__repr__`/`__str__`'s body: the first [`PREVIEW_TERMS`] terms as `coefficient*label`, `+`-joined, with a trailing count of however many more there are.
/// `0` for the empty sum — the zero operator, not "no terms".
fn format_sum(inner: &PauliSumImpl) -> String {
    let len = inner.len();
    if len == 0 {
        return "0".to_string();
    }
    let shown = inner.preview(PREVIEW_TERMS);
    let mut parts: Vec<String> = shown
        .iter()
        .map(|(label, c)| format!("{}*{}", format_coeff(*c), label))
        .collect();
    if len > shown.len() {
        parts.push(format!(
            "... ({} more term{})",
            len - shown.len(),
            if len - shown.len() == 1 { "" } else { "s" }
        ));
    }
    parts.join(" + ")
}

/// The scalar `*` accepts. Naming the two-sum case explicitly, since `a * b` on two sums is the one multiplication a reader is most likely to expect and the least likely to get.
fn scalar_factor(factor: &Bound<'_, PyAny>) -> PyResult<Complex64> {
    if factor.downcast::<PauliSum>().is_ok() {
        return Err(PyTypeError::new_err(
            "PauliSum * PauliSum is not supported: that is a full operator product, not a scalar \
             scaling; * and *= take a complex or real number",
        ));
    }
    extract_complex(factor).map_err(|_| {
        PyTypeError::new_err("PauliSum * x: x must be a complex or real number (scalar scaling)")
    })
}

#[pymethods]
impl PauliSum {
    /// Empty Pauli sum on `num_qubits` qubits.
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        PauliSumImpl::empty_for(num_qubits)
            .map(Self::plain)
            .ok_or_else(|| {
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
    #[classmethod]
    #[pyo3(signature = (terms, coefficients=None, *, num_qubits=None))]
    fn from_strings(
        _cls: &Bound<'_, pyo3::types::PyType>,
        terms: &Bound<'_, PyAny>,
        coefficients: Option<&Bound<'_, PyAny>>,
        num_qubits: Option<usize>,
    ) -> PyResult<Self> {
        let inner = match coefficients {
            Some(coefficients) => {
                if terms.downcast::<PyDict>().is_ok() {
                    return Err(PyTypeError::new_err(
                        "PauliSum.from_strings: pass either a dict, or (labels, coefficients) as two sequences — not a dict with coefficients also given",
                    ));
                }
                PauliSumImpl::from_label_list(terms, coefficients, num_qubits)?
            }
            None => {
                let dict = terms.downcast::<PyDict>().map_err(|_| {
                    PyTypeError::new_err(
                        "PauliSum.from_strings: pass a dict, or (labels, coefficients) as two equal-length sequences",
                    )
                })?;
                PauliSumImpl::from_strings_dict(dict, num_qubits)?
            }
        };
        Ok(Self::plain(inner))
    }

    /// Build from raw symplectic `(x, z, coefficients)` arrays — the inverse of `x_array` / `z_array` / `coefficients_array`.
    ///
    /// `x`/`z` are `uint64` arrays of shape `(n_terms, w)`, `w` from `1` up to the band width `num_qubits` picks; a narrower array is zero-padded, so a sum round-trips exactly. `coefficients` is a 1-D array of length `n_terms`, `complex128` or a real-float dtype.
    /// Rows are symplectic keys in the Hermitian convention (no phase, matching `from_strings`); duplicate `(x, z)` rows sum their coefficients and exact-zero rows are dropped. A set bit at or beyond qubit `num_qubits` is a `ValueError`.
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
        Ok(Self::plain(inner))
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
    /// Realized, not requested: reflects the running max of `desired_bits` over every `propagate`/`rebucket` call so far (grow-only), so it can differ from a `min_buckets`/`target_bucket_len` request passed to `propagate`.
    #[getter]
    fn num_buckets(&self) -> usize {
        self.inner.num_buckets()
    }

    /// Snapshot of the coefficient column as a list of Python complex values.
    fn coefficients(&self) -> Vec<Complex64> {
        self.inner.coeffs()
    }

    /// Expectation value in a single-qubit product state.
    ///
    /// `state` is either a uniform name — `"x+"`, `"y+"`, `"z+"`, the `+1` eigenstate of that Pauli on every qubit, matched case-insensitively — or a per-qubit label string of exactly `num_qubits` characters in qiskit's `Statevector.from_label` alphabet (`0`/`1` = Z±, `+`/`-` = X±, `r`/`l` = Y±, case-sensitive). Qubit indexing matches `from_strings`.
    /// Cost is one masked pass over the terms either way, never an expansion over basis states. Returns a Python complex; take `.real` when the operator is Hermitian.
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
    /// `generators` is a list of exactly `num_qubits` signed Pauli strings — `"+XX"`, `"-ZZ"`, or a bare `"ZIZ"` for `+` — in the same `I/X/Y/Z` alphabet and qubit indexing as `from_strings`. They must be pairwise commuting and independent over GF(2); anything else is a `ValueError`.
    /// Reads any stabilizer state (Bell, GHZ, cluster, Clifford-circuit output; see `paulistrings.interop.stabilizers_from_stim`), where `expectation` reads only product states. Cost is `O(terms · num_qubits² / 64)` after a one-off `O(num_qubits³ / 64)` reduction of the generators; prefer `expectation` when the state factorizes.
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

    /// `w[n]`: the total `|c|**2` of the strings that anticommute with exactly `n` of the generators `G_q`, `q in sites`, where `G = X` (`axis="x"`) or `Z` (`axis="z"`); `len(sites) + 1` entries.
    ///
    /// `X_q` anticommutes with a `Y` or `Z` on `q`, `Z_q` with an `X` or `Y`. `sum(w) == sum(|c|**2)`, and `sum(w[n] * cos(2 delta)**n) / sum(w)` is the diagonal approximation of the echo `rotated_overlap` computes exactly (`paulistrings.diagonal_echo`).
    /// With `comm`, `self` must be a `propagate(..., comm=...)` result and the call is collective: every rank gets the histogram of the whole distributed sum.
    #[pyo3(signature = (sites, axis="x", comm=None))]
    fn anticommute_histogram(
        &self,
        py: Python<'_>,
        sites: Vec<usize>,
        axis: &str,
        comm: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<f64>> {
        let axis = parse_axis(axis)?;
        check_sites(&sites, self.inner.num_qubits())?;
        let inner = &self.inner;
        let local = move || inner.anticommute_histogram(&sites, axis);
        if let Some(comm) = comm.filter(|comm| !comm.is_none()) {
            self.check_share(comm, "anticommute_histogram")?;
            return reduce_over_comm(py, comm, local);
        }
        Ok(py.allow_threads(local))
    }

    /// The operator Loschmidt echo `2**-n Tr(A V^dag A V)` of this sum `A`, with `V = prod_{q in sites} exp(-i delta G_q)` and `G = X` (`axis="x"`) or `Z` (`axis="z"`); exact, and without materializing `V^dag A V`.
    ///
    /// Strings that `V^dag . V` mixes (equal away from the flipped bits on `sites`) are grouped and paired, so the cost is dominated by the largest such class. At `delta = 0` the value is `sum(|c|**2)`; divide by that for the normalized echo.
    /// With `comm`, `self` must be a `propagate(..., comm=...)` result whose partition rows read none of the coordinates `V` flips (the x-bits of `sites` for `axis="x"`, the z-bits for `"z"`), so every class lies on one rank; propagate with `partition_row_exclude={axis: sites}` to get such rows (`partition_row_blocks=` rows read only z-bits, so they always qualify for `axis="x"`, as does any `result="gather"`). Otherwise it raises `ValueError` on every rank. The call is collective and every rank gets the whole sum's value.
    #[pyo3(signature = (sites, delta, axis="x", comm=None))]
    fn rotated_overlap(
        &self,
        py: Python<'_>,
        sites: Vec<usize>,
        delta: f64,
        axis: &str,
        comm: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        let parsed = parse_axis(axis)?;
        check_sites(&sites, self.inner.num_qubits())?;
        if let Some(comm) = comm.filter(|comm| !comm.is_none()) {
            let share = self.check_share(comm, "rotated_overlap")?;
            if !share.rows.keeps_flip_classes(&sites, parsed) {
                let (half, cut) = match parsed {
                    RotationAxis::X => (
                        "x",
                        " (or partition_row_blocks=, whose rows read only z-bits)",
                    ),
                    RotationAxis::Z => ("z", ""),
                };
                return Err(PyValueError::new_err(format!(
                    "rotated_overlap(comm=...): this sum's partition rows read the {half}-bits of \
                     sites the axis={axis:?} rotation flips, so strings it mixes can sit on \
                     different ranks and the per-rank values do not add up; propagate with \
                     partition_row_exclude={{{half:?}: sites}}{cut}"
                )));
            }
            let inner = &self.inner;
            let value = reduce_over_comm(py, comm, move || {
                vec![inner.rotated_overlap(&sites, delta, parsed)]
            })?;
            return Ok(value[0]);
        }
        let inner = &self.inner;
        Ok(py.allow_threads(move || inner.rotated_overlap(&sites, delta, parsed)))
    }

    /// `self + other`: coefficients added on matching strings, the rest kept. Both operands are left untouched.
    fn __add__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self::plain(
            self.checked_add(&other.inner, Complex64::new(1.0, 0.0))?,
        ))
    }

    /// `self - other`, the same merge with `other`'s coefficients negated.
    fn __sub__(&self, other: &Self) -> PyResult<Self> {
        Ok(Self::plain(
            self.checked_add(&other.inner, Complex64::new(-1.0, 0.0))?,
        ))
    }

    /// `self += other`, in place.
    fn __iadd__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        Self::add_in_place(slf, other, Complex64::new(1.0, 0.0))
    }

    /// `self -= other`, in place.
    fn __isub__(slf: &Bound<'_, Self>, other: &Bound<'_, Self>) -> PyResult<()> {
        Self::add_in_place(slf, other, Complex64::new(-1.0, 0.0))
    }

    /// `self * scalar`: every coefficient scaled by a complex or real number.
    ///
    /// Scalar-only. Multiplying two sums is a full operator product, a much larger operation this class does not implement.
    fn __mul__(&self, factor: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: self.scaled(factor)?,
            // Scaling moves no term, so a share keeps its place in the split.
            share: self.share.clone(),
        })
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
    /// `direction` is `"forward"` (default) or `"heisenberg"`. `policy` is an optional `Truncation`; `None` applies no per-term filtering beyond the engine's own exact-zero drop.
    /// `engine` is `"sorted"` (default, always bucketed), `"auto"` (a term-by-term hash-map path below `small_sum_threshold`, unless the policy has a layer pass like `topn`), or `"direct"` (same threshold, always). Results agree to floating-point tolerance across engines (ARCHITECTURE.md §Determinism).
    /// `target_bucket_len` and `min_buckets` are the sorting engine's per-layer bucket-sizing knobs (`None` for either keeps that field at `PropagateOptions::default()`, `1024`/`128`); see `PropagateOptions` for the tradeoff. `PauliSum.num_buckets` reads back the realized count, which can differ from a request since `rebucket` only ever grows a sum's partition.
    /// The GIL is released for the duration.
    ///
    /// `partitions` splits the sum across NUMA domains: `None`/`1` (default) is unpartitioned and bit-for-bit today's path; `"auto"` is one partition per NUMA node; an `int` power of two caps it at that many nodes; `list[list[int]]` gives explicit disjoint CPU lists. `pin_memory` (default `True`) binds each partition's allocations to its node.
    /// In partitioned mode `RAYON_NUM_THREADS` and `engine` are ignored, and `truncation.topn` raises `NotImplementedError` (use `approx_topn`).
    ///
    /// `partition_row_seed` picks which GF(2) rows decide a term's partition (`None`, the default, falls back to the sum's own hash seed — unchanged from before this knob existed). `partition_row_blocks`, an alternative to the seed, gives one disjoint qubit block per partition (`PartitionRows::cut`, e.g. `[[0, ..., 63], [64, ..., 126]]` for a 2-partition cut) — a term's partition is then the XOR of the blocks in which it has odd Z-weight, a locality cut rather than a GF(2)-random draw. The two are mutually exclusive with each other, and either needs `partitions=` or `comm=`.
    /// Under `comm=` the block count must equal the MPI group size, and the blocks must be identical on every rank (the split is a local filter every rank computes for itself).
    /// `partition_row_exclude`, a dict `{"x": [qubits], "z": [qubits]}` and the other alternative to `partition_row_blocks`, keeps the seeded draw (`partition_row_seed` still applies) but clears the named x-bits and z-bits from every row, so strings differing only there share a partition; `{"x": sites}` is what `rotated_overlap(sites, axis="x", comm=...)` needs of a `result="local"` share.
    ///
    /// `comm` takes an `mpi4py` communicator and runs one partition per rank, as an alternative to `partitions` (place via the launcher, e.g. `mpirun --map-by ppr:1:numa --bind-to numa`). Requires `MPI_THREAD_SERIALIZED` set before importing MPI, a power-of-two rank count, and every rank calling with the same replicated input in the same order.
    /// `result="gather"` (default) returns the whole sum on rank 0 and an empty one elsewhere; `"local"` returns each rank's own disjoint share, which remembers its rows for the `comm=` read-outs (`anticommute_histogram`, `rotated_overlap`). Raises `RuntimeError` without the `mpi` feature.
    ///
    /// `device` runs the propagation on CUDA devices: an `int` ordinal holds the whole sum on that device, uploaded before the first layer and downloaded after the last.
    /// A `list[int]` of a power-of-two length places one partition per entry, split and exchanged like `partitions=` (`partition_row_seed`/`partition_row_blocks` apply, and a repeated ordinal puts several partitions on one device); `"auto"` takes devices `0..k` for the largest power of two `k` visible.
    /// With `comm`, each MPI rank drives one device: an `int` ordinal or `"auto"` (a device near the rank's CPUs, distinct per rank on a node while there are enough), with `result=` as for a host `comm=` run; this needs both the `cuda` and `mpi` features.
    /// `device` is an alternative to `partitions`, ignores `engine`, and raises `RuntimeError` without the `cuda` feature. `truncation.topn` runs exactly when `device` resolves to one device (an `int`, or `"auto"`/a one-entry list on a one-GPU box); a device list of more than one entry or `comm=` with `device=` raises `NotImplementedError` (use `approx_topn`), since the `n`-th largest of a split sum has no collective form. `PauliSum.to_device` keeps a one-device sum resident across calls instead.
    ///
    /// ```python
    /// evolved = observable.propagate(circuit, policy, direction="heisenberg", partitions="auto")
    /// ```
    #[pyo3(signature = (circuit, policy=None, direction=None, engine=None, small_sum_threshold=None, target_bucket_len=None, min_buckets=None, partitions=None, pin_memory=true, partition_row_seed=None, partition_row_blocks=None, partition_row_exclude=None, comm=None, result="gather", device=None))]
    #[allow(clippy::too_many_arguments)]
    fn propagate(
        &self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
        engine: Option<&str>,
        small_sum_threshold: Option<usize>,
        target_bucket_len: Option<usize>,
        min_buckets: Option<usize>,
        partitions: Option<&Bound<'_, PyAny>>,
        pin_memory: bool,
        partition_row_seed: Option<u64>,
        partition_row_blocks: Option<&Bound<'_, PyAny>>,
        partition_row_exclude: Option<&Bound<'_, PyAny>>,
        comm: Option<&Bound<'_, PyAny>>,
        result: &str,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let dir = parse_direction(direction)?;
        let options = parse_engine(engine, small_sum_threshold, target_bucket_len, min_buckets)?;
        let gather = parse_result(result)?;
        check_num_qubits("PauliSum", self.inner.num_qubits(), circuit)?;
        let policy = PyTruncation::tree_of(policy);
        // Last, because adopting a communicator is collective: every check
        // above raises on all ranks alike, before any of them has entered MPI.
        let mode = parse_run_mode(
            py,
            partitions,
            pin_memory,
            partition_row_seed,
            partition_row_blocks,
            partition_row_exclude,
            comm,
            gather,
            device,
            &policy,
            self.inner.num_qubits(),
        )?;
        let (sum, _, _) = self.run(py, circuit, mode, &policy, dir, options, false)?;
        Ok(sum)
    }

    /// Propagate `self` through `circuit`, returning `(evolved, PropagationStats)`.
    ///
    /// Arguments and semantics are `propagate`'s; the only difference is that the engine also records per-layer term counts (before each layer, and after its truncation), so `evolved` agrees with `propagate`'s result to floating-point tolerance. See `PropagationStats.peak_terms` for what "peak" does and does not mean.
    /// A partitioned (`partitions=`) call additionally fills `PropagationStats.partition` with per-partition detail, summed to the same layer-level `terms_in`/`terms_out` an unpartitioned run would report.
    /// A distributed (`comm=`) call fills it too, but its per-layer lists hold **this rank's entry only** — gathering the group's counters would add a collective per layer for a diagnostic. Reduce over `comm` for the group's picture.
    /// A device (`device=`) call fills it with one partition per device, `PartitionStats.devices` naming them; under `comm=` it is this rank's record with this rank's device.
    #[pyo3(signature = (circuit, policy=None, direction=None, engine=None, small_sum_threshold=None, target_bucket_len=None, min_buckets=None, partitions=None, pin_memory=true, partition_row_seed=None, partition_row_blocks=None, partition_row_exclude=None, comm=None, result="gather", device=None))]
    #[allow(clippy::too_many_arguments)]
    fn propagate_with_stats(
        &self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
        engine: Option<&str>,
        small_sum_threshold: Option<usize>,
        target_bucket_len: Option<usize>,
        min_buckets: Option<usize>,
        partitions: Option<&Bound<'_, PyAny>>,
        pin_memory: bool,
        partition_row_seed: Option<u64>,
        partition_row_blocks: Option<&Bound<'_, PyAny>>,
        partition_row_exclude: Option<&Bound<'_, PyAny>>,
        comm: Option<&Bound<'_, PyAny>>,
        result: &str,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PropagationStats)> {
        let dir = parse_direction(direction)?;
        let options = parse_engine(engine, small_sum_threshold, target_bucket_len, min_buckets)?;
        let gather = parse_result(result)?;
        check_num_qubits("PauliSum", self.inner.num_qubits(), circuit)?;
        let policy = PyTruncation::tree_of(policy);
        // Last, because adopting a communicator is collective: every check
        // above raises on all ranks alike, before any of them has entered MPI.
        let mode = parse_run_mode(
            py,
            partitions,
            pin_memory,
            partition_row_seed,
            partition_row_blocks,
            partition_row_exclude,
            comm,
            gather,
            device,
            &policy,
            self.inner.num_qubits(),
        )?;
        let (sum, trace, collapses) = self.run(py, circuit, mode, &policy, dir, options, true)?;
        let stats = PropagationStats::from_run_trace(
            trace.expect("a traced run always records a trace"),
            sum.inner.len(),
            collapses,
        );
        Ok((sum, stats))
    }

    /// Upload `self` to CUDA device `device`, returning a resident `GpuPauliSum`; `self` is left untouched.
    ///
    /// The resident sum is stepped in place by `GpuPauliSum.propagate` and read back by `GpuPauliSum.to_host`, so a loop of many short propagations pays one upload and one download rather than one of each per call.
    /// Raises `RuntimeError` without the `cuda` feature or with no visible device, `ValueError` for an ordinal this process cannot see, and `MemoryError` if the device cannot hold the sum.
    #[pyo3(signature = (device=0))]
    fn to_device(&self, py: Python<'_>, device: i64) -> PyResult<PyObject> {
        crate::gpu::to_device(py, &self.inner, device)
    }
}

impl PauliSum {
    /// The GIL-released half of both propagate entry points: run `mode` over `self` and wrap the result, with its place in a distributed split.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        mode: RunMode,
        policy: &BuiltinTruncation,
        dir: Direction,
        options: PropagateOptions,
        traced: bool,
    ) -> PyResult<(Self, Option<RunTrace>, Option<u64>)> {
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
        let (mut trace, mut share, mut collapses) = (None, None, None);
        let slots = (&mut trace, &mut share, &mut collapses);
        let inner = py.allow_threads(move || -> Result<PauliSumImpl, PropagateFailure> {
            Ok(for_each_width_propagate!(
                PauliSumImpl,
                &self.inner,
                &circuit.inner,
                |s, c, W, wrap| {
                    let out = mode.run::<W>(c, s, policy, dir, options, traced)?;
                    *slots.0 = out.trace;
                    *slots.1 = out.share;
                    *slots.2 = out.collapses;
                    wrap(out.sum)
                },
                else {
                    return Err(PropagateFailure::WidthMismatch);
                }
            ))
        })?;
        Ok((Self { inner, share }, trace, collapses))
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

    /// Explicit "cut" blocks round-trip through `validate_partition_row_blocks` +
    /// `PartitionRows::cut` into a `PartitionRows` that actually assigns qubits to the
    /// blocks named, and two different cuts assign at least one term to different partitions.
    #[test]
    fn explicit_cut_blocks_round_trip_and_differ_from_each_other() {
        use paulistrings::pauli_string::PauliString;

        let num_qubits = 4;
        let num_partitions = 2;

        let half_low = vec![vec![0u32, 1], vec![2u32, 3]];
        validate_partition_row_blocks_impl(&half_low, num_qubits, num_partitions, "partitions=2")
            .expect("two disjoint blocks covering all 4 qubits validate cleanly");
        let rows_low = PartitionRows::<1>::cut(num_qubits, &half_low);

        let half_alt = vec![vec![0u32, 2], vec![1u32, 3]];
        validate_partition_row_blocks_impl(&half_alt, num_qubits, num_partitions, "partitions=2")
            .expect("an alternative disjoint cut also validates cleanly");
        let rows_alt = PartitionRows::<1>::cut(num_qubits, &half_alt);

        // Round-trip: qubit 2 sits in block 1 under `half_low`, block 0 under `half_alt`.
        let z2 = PauliString::<1>::z(2);
        assert_eq!(rows_low.partition_of_pauli(&z2), 1);
        assert_eq!(rows_alt.partition_of_pauli(&z2), 0);

        // The two cuts disagree on at least this term, so they are genuinely different row
        // sets, not two spellings of the same partition.
        assert_ne!(
            rows_low.partition_of_pauli(&z2),
            rows_alt.partition_of_pauli(&z2)
        );
    }

    /// A block count that doesn't match the partition count is a `ValueError`, not a panic —
    /// the whole point of validating before `allow_threads` releases the GIL.
    #[test]
    fn mismatched_block_count_is_a_value_error_not_a_panic() {
        let one_block = vec![vec![0u32, 1, 2, 3]];
        let err = validate_partition_row_blocks_impl(&one_block, 4, 2, "partitions=2")
            .expect_err("1 block for 2 partitions must be rejected");
        assert!(err.contains("partition_row_blocks"));
        assert!(err.contains("partitions=2"));
    }

    /// A block set `PartitionRows::cut` would panic on is a `ValueError` too:
    /// a partition bit no qubit can set leaves half the partitions empty.
    #[test]
    fn a_block_set_that_cannot_name_every_partition_is_a_value_error() {
        let empty_second = vec![vec![0u32, 1, 2, 3], Vec::new()];
        let err = validate_partition_row_blocks_impl(&empty_second, 4, 2, "partitions=2")
            .expect_err("an empty block 1 leaves partition bit 0 constant");
        assert!(err.contains("bit 0"), "{err}");

        // Block 1 empty out of four is fine — block 3 still sets bit 0.
        let one_empty = vec![vec![0u32], Vec::new(), vec![1u32], vec![2u32]];
        validate_partition_row_blocks_impl(&one_empty, 4, 4, "partitions=4")
            .expect("every bit is set by some non-empty block");
    }

    /// A placement whose partition count is not a power of two cannot be cut by
    /// `log2(P)` rows at all.
    #[test]
    fn a_partition_count_that_is_not_a_power_of_two_is_a_value_error() {
        let three = vec![vec![0u32], vec![1u32], vec![2u32]];
        let err = validate_partition_row_blocks_impl(&three, 3, 3, "partitions=3")
            .expect_err("three partitions cannot be named by GF(2) rows");
        assert!(err.contains("power of two"), "{err}");
    }

    /// The count a distributed run validates against is the MPI group size, so
    /// the message names the group rather than a `partitions=` the caller never
    /// passed.
    #[test]
    fn a_mismatched_block_count_names_the_group_under_comm() {
        let two_blocks = vec![vec![0u32, 1], vec![2u32, 3]];
        let err = validate_partition_row_blocks_impl(&two_blocks, 4, 4, "comm= with 4 rank(s)")
            .expect_err("2 blocks for a 4-rank group must be rejected");
        assert!(err.contains("comm= with 4 rank(s)"), "{err}");
    }

    /// A qubit named in two blocks is rejected before it ever reaches `PartitionRows::cut`
    /// (which would otherwise panic on the same condition).
    #[test]
    fn overlapping_blocks_are_a_value_error() {
        let overlapping = vec![vec![0u32, 1], vec![1u32, 2]];
        let err = validate_partition_row_blocks_impl(&overlapping, 3, 2, "partitions=2")
            .expect_err("qubit 1 in two blocks must be rejected");
        assert!(err.contains("more than one block"));
    }
}
