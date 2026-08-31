//! Python `PauliSum` class with width-monomorphized backing storage. See
//! ARCHITECTURE.md §Width and ARCHITECTURE.md §Python-Bindings.

use crate::truncation_spec::{PolicySpec, PyTruncation, SpecPolicy};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use paulistrings::accumulator::BuildAccumulator;
use paulistrings::pauli_string::PauliString;
use paulistrings::phase::Phase;
use paulistrings::{propagate, Direction, PauliSum as CorePauliSum, ProductState};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict};

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

    pub fn expectation(&self, state: ProductState) -> Complex64 {
        for_each_width!(self, |s| s.expectation_product_state(state))
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
        acc.add_term(PauliString::<W> { x, z }, Phase::ONE, c);
    }
    Ok(acc.finalize())
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

    /// Expectation value in a uniform single-qubit product state.
    ///
    /// `state` is one of `"x+"` (`|+...+>`), `"y+"` (`|+i...+i>`) or `"z+"`
    /// (`|0...0>`) — each the `+1` eigenstate of that Pauli on every qubit.
    /// Returns a Python complex; take `.real` when the operator is Hermitian.
    #[pyo3(signature = (state="x+"))]
    fn expectation(&self, state: &str) -> PyResult<Complex64> {
        let st = match state {
            "x+" | "X+" => ProductState::XPlus,
            "y+" | "Y+" => ProductState::YPlus,
            "z+" | "Z+" => ProductState::ZPlus,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown product state {other:?}; expected \"x+\", \"y+\" or \"z+\"",
                )))
            }
        };
        Ok(self.inner.expectation(st))
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
    /// The GIL is released for the duration of the propagation, so Python
    /// threads — including `logging` handlers draining the engine's per-layer
    /// progress records — run while a long simulation is in flight.
    #[pyo3(signature = (circuit, policy=None, direction=None))]
    fn propagate(
        &self,
        py: Python<'_>,
        circuit: &crate::circuit::Circuit,
        policy: Option<&PyTruncation>,
        direction: Option<&str>,
    ) -> PyResult<Self> {
        let dir = match direction.unwrap_or("forward") {
            "forward" => Direction::Forward,
            "heisenberg" => Direction::Heisenberg,
            other => {
                return Err(PyValueError::new_err(format!(
                    "direction must be 'forward' or 'heisenberg', got {:?}",
                    other
                )));
            }
        };
        if self.inner.num_qubits() != circuit.inner.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "PauliSum.num_qubits ({}) != Circuit.num_qubits ({})",
                self.inner.num_qubits(),
                circuit.inner.num_qubits()
            )));
        }
        let no_op = PolicySpec::NoOp;
        let spec: &PolicySpec = match policy {
            Some(p) => &p.spec,
            None => &no_op,
        };
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
        let inner = py
            .allow_threads(|| -> Result<PauliSumImpl, &'static str> {
                Ok(for_each_width_propagate!(
                    &self.inner,
                    &circuit.inner,
                    |s, c, W| propagate(c, s.clone(), &SpecPolicy::<W>(spec), dir),
                    else {
                        // Same num_qubits but different widths is impossible
                        // because both width pickers map num_qubits to the
                        // same arm.
                        return Err("internal: PauliSum and Circuit width mismatch");
                    }
                ))
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }
}
