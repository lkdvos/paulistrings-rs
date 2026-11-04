//! The storage behind a Python `PauliSum`: [`WidthSum`], one coefficient type's sum monomorphized over the widths, and [`PauliSumImpl`], the dtype enum over it.

use crate::dtype::{imaginary_error, narrow, Dtype, PyCoefficient, Scalar};
use num_complex::{Complex32, Complex64};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use paulistrings::{
    BuildAccumulator, BuiltinTruncation, Gf2Hash, PartitionRowPolicy, PauliAxis, PauliString,
    PauliSum as CorePauliSum, ProductBasis, StabilizerState,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyComplex, PyDict};
use std::borrow::Cow;

/// Which operator product [`WidthSum::pair`] takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pairing {
    /// `A·B`.
    Product,
    /// `i[A, B]`.
    Liouvillian,
    /// `{A, B}`.
    Anticommutator,
}

/// Width-dispatch enum over sums of one coefficient type. The Python boundary picks the smallest width that fits `num_qubits`.
#[derive(Clone)]
pub enum WidthSum<C: PyCoefficient> {
    W1(CorePauliSum<1, C>),
    W2(CorePauliSum<2, C>),
    W4(CorePauliSum<4, C>),
    W8(CorePauliSum<8, C>),
    W16(CorePauliSum<16, C>),
}

impl<C: PyCoefficient> WidthSum<C> {
    /// Pick the smallest supported width for `num_qubits`. Returns `None` if `num_qubits` exceeds the largest monomorphized width (1024 qubits).
    pub fn empty_for(num_qubits: usize) -> Option<Self> {
        for_num_qubits!(num_qubits, |W| CorePauliSum::<W, C>::empty(num_qubits))
    }

    forward! {
        for_each_width;
        pub fn num_qubits(&self) -> usize;
        pub fn len(&self) -> usize;
        /// Current bucket count, `1 << hash().bits()`. Reflects whatever a prior `propagate` call left resident, not a request.
        pub fn num_buckets(&self) -> usize;
        pub fn identity_coefficient(&self) -> C;
        pub fn bucket_lengths(&self) -> Vec<usize>;
    }

    /// Uniform product state: the `+1` eigenstate of `axis` on every qubit.
    pub fn expectation_uniform(&self, axis: PauliAxis) -> C {
        for_each_width!(self, |s| s.expectation_product(&ProductBasis::uniform(axis)))
    }

    /// Per-qubit product state: entry `q` is qubit `q`'s `(axis, minus)`. Caller already checked one entry per qubit, so the resulting masks have no bit set past `num_qubits`.
    pub fn expectation_labels(&self, axes: &[(PauliAxis, bool)]) -> C {
        for_each_width!(self, |s| s
            .expectation_product(&ProductBasis::from_axes(axes.iter().copied())))
    }

    /// Stabilizer state given by one signed Pauli generator per qubit. Parsed at the active width and validated by the core, so a malformed string or an invalid generator set surfaces as a `ValueError`.
    pub fn expectation_stabilizer(&self, generators: &[String]) -> PyResult<C> {
        for_each_width!(self, |s| stabilizer_expectation(s, generators))
    }

    /// The bucket hash's rows as coordinate flags: row `i`, column `c` is set when row `i` reads coordinate `c` (x-bits first, then z-bits).
    pub fn hash_rows(&self) -> Vec<Vec<bool>> {
        for_each_width!(self, |s| {
            let (rows_x, rows_z) = s.hash().rows();
            rows_x
                .iter()
                .zip(rows_z)
                .map(|(x, z)| crate::sum::coordinate_flags(x, z, s.num_qubits()))
                .collect()
        })
    }

    pub fn hash_rank_deficiency(&self) -> Option<String> {
        for_each_width!(self, |s| s.hash().rank_deficiency())
    }

    /// The same terms bucketed by the flag-matrix `rows` completed with dense rows from `seed`, at the current bucket count.
    pub fn with_hash_rows(&self, rows: &[Vec<bool>], seed: u64) -> Self {
        for_each_width_convert!(WidthSum => WidthSum, self.clone(), |s| {
            let hash = Gf2Hash::from_matrix(s.num_qubits(), s.hash().bits(), seed, rows);
            s.with_hash(hash)
        })
    }

    /// Why the rows `policy` names for `2^bits` partitions leave some partitions unreachable, if they do.
    pub fn partition_rank_deficiency(
        &self,
        policy: &PartitionRowPolicy,
        bits: u8,
    ) -> Option<String> {
        fn shortfall<const W: usize, C: PyCoefficient>(
            s: &CorePauliSum<W, C>,
            policy: &PartitionRowPolicy,
            bits: u8,
        ) -> Option<String> {
            policy.rows::<W>(s.num_qubits(), bits, s.hash().seed()).rank_deficiency()
        }
        for_each_width!(self, |s| shortfall(s, policy, bits))
    }

    pub fn hash_seed(&self) -> u64 {
        for_each_width!(self, |s| s.hash().seed())
    }

    /// `None` when the two sums were monomorphized at different widths, which can only happen if their qubit counts fall in different dispatch bands.
    pub fn overlap(&self, other: &Self) -> Option<C> {
        for_each_width_pair!((self, other), |a, b| a.overlap(b))
    }

    /// `self + other`, or `self - other` when `subtract`, matching strings combined and the rest kept; `None` on a width mismatch.
    /// Caller checks the qubit counts first: equal widths are not equal qubit counts, and the core's merge asserts on the latter.
    pub fn add_signed(&self, other: &Self, subtract: bool) -> Option<Self> {
        for_each_width_pair_rewrap!((self, other), |a, b, wrap| {
            if subtract {
                let mut negated = b.clone();
                negated.scale(-C::ONE);
                wrap(a.add(&negated))
            } else {
                wrap(a.add(b))
            }
        })
    }

    /// `self ← alpha · other + beta · self`, in place; `None` on a width mismatch.
    /// Caller checks the qubit counts first, as for [`WidthSum::add_signed`].
    pub fn add_scaled(&mut self, other: &Self, alpha: C, beta: C) -> Option<()> {
        for_each_width_pair!((self, other), |a, b| a.add_scaled(b, alpha, beta))
    }

    /// `self ← alpha · pairing(generator, operand) + beta · self`, truncated by `policy` as one layer; `None` on a width mismatch.
    /// Caller checks the qubit counts first and passes a complex `C` for [`Pairing::Product`].
    pub fn add_paired(
        &mut self,
        generator: &Self,
        operand: &Self,
        pairing: Pairing,
        (alpha, beta): (C, C),
        policy: &BuiltinTruncation,
    ) -> Option<()> {
        for_each_width_triple!((self, generator, operand), |d, g, o| match pairing {
            Pairing::Product => d.add_product(g, o, alpha, beta, policy),
            Pairing::Liouvillian => d.add_liouvillian(g, o, alpha, beta, policy),
            Pairing::Anticommutator => d.add_anticommutator(g, o, alpha, beta, policy),
        })
    }

    /// Multiply every coefficient by `factor`, in place.
    pub fn scale(&mut self, factor: C) {
        for_each_width!(self, |s| s.scale(factor))
    }

    /// First `k` terms in canonical order, decoded to `(label, coefficient)` — `O(k)`, not `O(len())`, so `__repr__`/`__str__` stay cheap on a huge sum.
    pub fn preview(&self, k: usize) -> Vec<(String, C)> {
        fn preview_of<const W: usize, C: PyCoefficient>(
            s: &CorePauliSum<W, C>,
            k: usize,
        ) -> Vec<(String, C)> {
            s.iter()
                .take(k)
                .map(|(x, z, c)| {
                    let key = PauliString::<W> { x: *x, z: *z };
                    (crate::pauli_string::label_of(&key, s.num_qubits()), c)
                })
                .collect()
        }
        for_each_width!(self, |s| preview_of(s, k))
    }

    /// Snapshot of the coefficient column, in the sum's canonical order (partition-bucket index ascending, then lexicographic `(x, z)`; equal to plain lex order for sums of ≤ 1024 terms).
    pub fn coeffs(&self) -> Vec<C> {
        for_each_width!(self, |s| s.to_arrays().2)
    }

    /// `(width, x_flat, z_flat)` snapshot of the SoA columns, in the sum's canonical order (see [`Self::coeffs`]), the same across the three arrays since the order is a deterministic function of the sum.
    /// `x_flat`/`z_flat` have length `len() * width`; caller reshapes to `(len, width)`.
    pub fn xz_flat(&self) -> (usize, Vec<u64>, Vec<u64>) {
        let (width, x, z, _) = self.columns_flat();
        (width, x, z)
    }

    /// `(width, x_flat, z_flat, coefficients)` from one snapshot of the columns; see [`Self::xz_flat`].
    pub fn columns_flat(&self) -> (usize, Vec<u64>, Vec<u64>, Vec<C>) {
        fn columns_of<const W: usize, C: PyCoefficient>(
            s: &CorePauliSum<W, C>,
        ) -> (usize, Vec<u64>, Vec<u64>, Vec<C>) {
            let (x, z, c) = s.to_arrays();
            (W, x.concat(), z.concat(), c)
        }
        for_each_width!(self, |s| columns_of(s))
    }

    /// The same terms as a `T` sum; `None` when `T` is real and a coefficient has a nonzero imaginary part.
    pub fn cast<T: PyCoefficient>(&self) -> Option<WidthSum<T>> {
        Some(for_each_width_convert!(WidthSum => WidthSum, self, |s| s.cast::<T>(0.0)?))
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
        for_num_qubits!(num_qubits, |W| parse_terms::<W, C>(num_qubits, terms)?)
            .ok_or_else(too_many_qubits)
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
        for_num_qubits!(num_qubits, |W| parse_label_list::<W, C>(
            num_qubits,
            &labels,
            coefficients
        )?)
        .ok_or_else(too_many_qubits)
    }

    /// Single-term sum from a `PauliString`'s key and a coefficient — `PauliString.__mul__`'s body.
    /// `coeff` may be exactly zero; `BuildAccumulator::finalize` already drops it, giving the empty sum.
    pub fn from_single(
        term: &crate::pauli_string::PauliStringImpl,
        num_qubits: usize,
        coeff: C,
    ) -> Self {
        fn build<const W: usize, C: PyCoefficient>(
            p: &PauliString<W>,
            num_qubits: usize,
            coeff: C,
        ) -> CorePauliSum<W, C> {
            let mut acc = BuildAccumulator::<W, C>::with_capacity(num_qubits, 1);
            acc.add_term(*p, coeff);
            acc.finalize()
        }
        use crate::pauli_string::PauliStringImpl as PS;
        match term {
            PS::W1(p) => WidthSum::W1(build(p, num_qubits, coeff)),
            PS::W2(p) => WidthSum::W2(build(p, num_qubits, coeff)),
            PS::W4(p) => WidthSum::W4(build(p, num_qubits, coeff)),
            PS::W8(p) => WidthSum::W8(build(p, num_qubits, coeff)),
            PS::W16(p) => WidthSum::W16(build(p, num_qubits, coeff)),
        }
    }

    /// Build from raw symplectic `(x, z, coefficients)` arrays — the inverse of [`Self::columns_flat`].
    /// See `PauliSum::from_arrays` for the shape/dtype contract; this just picks the width band for `num_qubits` and delegates.
    pub fn from_arrays(
        num_qubits: usize,
        x: &PyReadonlyArray2<'_, u64>,
        z: &PyReadonlyArray2<'_, u64>,
        coefficients: &[C],
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
        for_num_qubits!(num_qubits, |W| build_from_arrays::<W, C>(num_qubits, x, z, coefficients)?)
            .ok_or_else(too_many_qubits)
    }
}

fn too_many_qubits() -> PyErr {
    PyValueError::new_err("num_qubits exceeds largest monomorphized width (1024)")
}

/// Dtype-dispatch enum: the Python boundary holds one [`WidthSum`] per coefficient type, `complex128` unless `dtype=` says otherwise.
#[derive(Clone)]
pub enum PauliSumImpl {
    F64(WidthSum<f64>),
    F32(WidthSum<f32>),
    C128(WidthSum<Complex64>),
    C64(WidthSum<Complex32>),
}

impl PauliSumImpl {
    pub fn empty_for(num_qubits: usize, dtype: Dtype) -> Option<Self> {
        for_dtype!(PauliSumImpl, dtype, |C, wrap| WidthSum::<C>::empty_for(num_qubits).map(wrap))
    }

    pub fn dtype(&self) -> Dtype {
        fn of<C: PyCoefficient>(_: &WidthSum<C>) -> Dtype {
            C::DTYPE
        }
        for_each_dtype!(self, |s| of(s))
    }

    forward! {
        for_each_dtype;
        pub fn num_qubits(&self) -> usize;
        pub fn len(&self) -> usize;
        pub fn num_buckets(&self) -> usize;
        pub fn hash_seed(&self) -> u64;
        pub fn bucket_lengths(&self) -> Vec<usize>;
        pub fn hash_rows(&self) -> Vec<Vec<bool>>;
        pub fn hash_rank_deficiency(&self) -> Option<String>;
        pub fn xz_flat(&self) -> (usize, Vec<u64>, Vec<u64>);
        pub fn partition_rank_deficiency(&self, policy: &PartitionRowPolicy, bits: u8) -> Option<String>;
    }

    pub fn with_hash_rows(&self, rows: &[Vec<bool>], seed: u64) -> Self {
        for_each_dtype_convert!(PauliSumImpl => PauliSumImpl, self, |s| s.with_hash_rows(rows, seed))
    }

    pub fn expectation_uniform(&self, axis: PauliAxis) -> Scalar {
        for_each_dtype!(self, |s| Scalar::of(s.expectation_uniform(axis)))
    }

    pub fn expectation_labels(&self, axes: &[(PauliAxis, bool)]) -> Scalar {
        for_each_dtype!(self, |s| Scalar::of(s.expectation_labels(axes)))
    }

    pub fn expectation_stabilizer(&self, generators: &[String]) -> PyResult<Scalar> {
        for_each_dtype!(self, |s| s.expectation_stabilizer(generators).map(Scalar::of))
    }

    pub fn identity_coefficient(&self) -> Scalar {
        for_each_dtype!(self, |s| Scalar::of(s.identity_coefficient()))
    }

    /// [`WidthSum::preview`] with each coefficient formatted at its own precision.
    pub fn preview(&self, k: usize) -> Vec<(String, String)> {
        for_each_dtype!(self, |s| s
            .preview(k)
            .into_iter()
            .map(|(label, c)| (label, c.format()))
            .collect())
    }

    /// The coefficient column as a Python list of `float` or `complex`.
    pub fn coefficients(&self, py: Python<'_>) -> PyObject {
        for_each_dtype!(self, |s| s.coeffs().into_py(py))
    }

    /// The coefficient column as a 1-D NumPy array of the sum's dtype.
    pub fn coefficients_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        for_each_dtype!(self, |s| s.coeffs().into_pyarray_bound(py).into_any())
    }

    /// `(width, x_flat, z_flat, coefficients)` from one snapshot, the coefficients a NumPy array of the sum's dtype.
    pub fn columns<'py>(&self, py: Python<'py>) -> (usize, Vec<u64>, Vec<u64>, Bound<'py, PyAny>) {
        for_each_dtype!(self, |s| {
            let (width, x, z, c) = s.columns_flat();
            (width, x, z, c.into_pyarray_bound(py).into_any())
        })
    }

    /// The same terms at `dtype`; `None` when `dtype` is real and a coefficient has a nonzero imaginary part.
    pub fn cast(&self, dtype: Dtype) -> Option<Self> {
        for_each_dtype!(self, |s| for_dtype!(PauliSumImpl, dtype, |T, wrap| s
            .cast::<T>()
            .map(wrap)))
    }

    /// `self` at `dtype`, borrowed when it already is; `None` as for [`Self::cast`].
    fn converted(&self, dtype: Dtype) -> Option<Cow<'_, Self>> {
        if self.dtype() == dtype {
            Some(Cow::Borrowed(self))
        } else {
            self.cast(dtype).map(Cow::Owned)
        }
    }

    /// `self ± other` at the two dtypes' NumPy promotion; `None` on a width mismatch.
    /// Caller checks the qubit counts first, as for [`WidthSum::add_signed`].
    pub fn add_promoted(&self, other: &Self, subtract: bool) -> Option<Self> {
        let dtype = self.dtype().promote(other.dtype());
        let a = self.converted(dtype).expect("promotion never narrows");
        let b = other.converted(dtype).expect("promotion never narrows");
        for_each_dtype_pair!((a.as_ref(), b.as_ref()), |x, y, wrap| x
            .add_signed(y, subtract)
            .map(wrap))
        .flatten()
    }

    /// The operator product `pairing` names at the two dtypes' promotion, made complex for [`Pairing::Product`]: [`add_paired`](Self::add_paired) into an empty sum.
    /// Caller checks the qubit counts first, as for [`WidthSum::add_signed`].
    pub fn pair_promoted(
        &self,
        other: &Self,
        pairing: Pairing,
        policy: &BuiltinTruncation,
    ) -> PyResult<Self> {
        let mut dtype = self.dtype().promote(other.dtype());
        if pairing == Pairing::Product {
            dtype = dtype.complex();
        }
        let mut out = Self::empty_for(self.num_qubits(), dtype).ok_or_else(width_mismatch)?;
        out.add_paired(self, other, pairing, (Complex64::ONE, Complex64::new(0.0, 0.0)), policy)?;
        Ok(out)
    }

    /// `other` at this sum's dtype, a `ValueError` naming it `what` when it has an imaginary part a real dtype cannot hold.
    pub fn at_dtype_of<'a>(&self, other: &'a Self, what: &str) -> PyResult<Cow<'a, Self>> {
        let dtype = self.dtype();
        other.converted(dtype).ok_or_else(|| imaginary_error(what, dtype))
    }

    /// `self ← alpha · other + beta · self`, in place at this sum's dtype; a `ValueError` for an imaginary part a real dtype cannot hold.
    /// Caller checks the qubit counts first, as for [`WidthSum::add_signed`].
    pub fn add_scaled(&mut self, other: &Self, alpha: Complex64, beta: Complex64) -> PyResult<()> {
        let other = self.at_dtype_of(other, "the added sum")?;
        for_each_dtype_pair!((self, other.as_ref()), |a, b, _wrap| {
            let (alpha, beta) = (narrow(alpha, "alpha")?, narrow(beta, "beta")?);
            a.add_scaled(b, alpha, beta).ok_or_else(width_mismatch)
        })
        .expect("converted to this sum's dtype")
    }

    /// `self ← alpha · pairing(generator, operand) + beta · self`, in place at this sum's dtype, truncated by `policy` as one layer.
    /// A product into a real sum is a `TypeError`, an imaginary part a real dtype cannot hold a `ValueError`; caller checks the qubit counts first.
    pub fn add_paired(
        &mut self,
        generator: &Self,
        operand: &Self,
        pairing: Pairing,
        (alpha, beta): (Complex64, Complex64),
        policy: &BuiltinTruncation,
    ) -> PyResult<()> {
        let dtype = self.dtype();
        if pairing == Pairing::Product && dtype.is_real() {
            return Err(PyTypeError::new_err(format!(
                "a product is not Hermitian, so a {} PauliSum cannot hold it; use a complex destination",
                dtype.name(),
            )));
        }
        let generator = self.at_dtype_of(generator, "the generator")?;
        let operand = self.at_dtype_of(operand, "the operand")?;
        for_each_dtype_triple!((self, generator.as_ref(), operand.as_ref()), |d, g, o| {
            let weights = (narrow(alpha, "alpha")?, narrow(beta, "beta")?);
            d.add_paired(g, o, pairing, weights, policy).ok_or_else(width_mismatch)
        })
        .expect("converted to this sum's dtype")
    }

    /// `tr(self† · other) / 2^n` at the two dtypes' promotion; `None` on a width mismatch.
    pub fn overlap(&self, other: &Self) -> Option<Scalar> {
        let dtype = self.dtype().promote(other.dtype());
        let a = self.converted(dtype).expect("promotion never narrows");
        let b = other.converted(dtype).expect("promotion never narrows");
        for_each_dtype_pair!((a.as_ref(), b.as_ref()), |x, y, _wrap| x.overlap(y).map(Scalar::of))
            .flatten()
    }

    /// Multiply every coefficient by `factor`, in place; a `ValueError` when the dtype is real and `factor` has an imaginary part.
    pub fn scale(&mut self, factor: Complex64) -> PyResult<()> {
        for_each_dtype!(self, |s| {
            s.scale(narrow(factor, "the scale factor")?);
            Ok(())
        })
    }

    pub fn from_strings_dict(
        terms: &Bound<'_, PyDict>,
        num_qubits: Option<usize>,
        dtype: Dtype,
    ) -> PyResult<Self> {
        for_dtype!(PauliSumImpl, dtype, |C, wrap| WidthSum::<C>::from_strings_dict(
            terms, num_qubits
        )
        .map(wrap))
    }

    pub fn from_label_list(
        labels: &Bound<'_, PyAny>,
        coefficients: &Bound<'_, PyAny>,
        num_qubits: Option<usize>,
        dtype: Dtype,
    ) -> PyResult<Self> {
        for_dtype!(PauliSumImpl, dtype, |C, wrap| WidthSum::<C>::from_label_list(
            labels,
            coefficients,
            num_qubits
        )
        .map(wrap))
    }

    pub fn from_arrays(
        num_qubits: usize,
        x: &PyReadonlyArray2<'_, u64>,
        z: &PyReadonlyArray2<'_, u64>,
        coefficients: &Bound<'_, PyAny>,
        dtype: Dtype,
    ) -> PyResult<Self> {
        for_dtype!(PauliSumImpl, dtype, |C, wrap| {
            let coefficients = extract_coefficient_array::<C>(coefficients)?;
            WidthSum::<C>::from_arrays(num_qubits, x, z, &coefficients).map(wrap)
        })
    }
}

/// Build a `PauliSum<W, C>` from a `{pauli_string: coefficient}` Python dict.
/// Format matches the test helper in `pauli_sum.rs`: character `i` describes qubit `i`, coefficients multiply the literal Hermitian Pauli string.
fn parse_terms<const W: usize, C: PyCoefficient>(
    num_qubits: usize,
    terms: &Bound<'_, PyDict>,
) -> PyResult<CorePauliSum<W, C>> {
    let mut acc = BuildAccumulator::<W, C>::with_capacity(num_qubits, terms.len());
    for (key, val) in terms.iter() {
        let s: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("PauliSum.from_strings keys must be str"))?;
        check_label_length(&s, num_qubits)?;
        let c = coefficient::<C>(&val, || format!("the coefficient of {s:?}"))?;
        acc.add_term(parse_pauli_key::<W>(&s)?, c);
    }
    Ok(acc.finalize())
}

/// Build a `PauliSum<W, C>` from parallel `labels`/`coefficients` sequences. Unlike [`parse_terms`]'s dict, a label repeated in `labels` is not an error — `BuildAccumulator::add_term` sums it, the same accumulation the manual's Hamiltonian example does by hand with a dict.
fn parse_label_list<const W: usize, C: PyCoefficient>(
    num_qubits: usize,
    labels: &[String],
    coefficients: &Bound<'_, PyAny>,
) -> PyResult<CorePauliSum<W, C>> {
    let mut acc = BuildAccumulator::<W, C>::with_capacity(num_qubits, labels.len());
    for (i, s) in labels.iter().enumerate() {
        check_label_length(s, num_qubits)?;
        let c = coefficient::<C>(&coefficients.get_item(i)?, || format!("coefficients[{i}]"))?;
        acc.add_term(parse_pauli_key::<W>(s)?, c);
    }
    Ok(acc.finalize())
}

fn check_label_length(s: &str, num_qubits: usize) -> PyResult<()> {
    if s.len() != num_qubits {
        return Err(PyValueError::new_err(format!(
            "Pauli string {:?} has length {}, expected {} (length must match num_qubits)",
            s,
            s.len(),
            num_qubits
        )));
    }
    Ok(())
}

/// A Python number as a `C`, `what` naming it in the `ValueError` a real `C` raises on an imaginary part.
fn coefficient<C: PyCoefficient>(
    val: &Bound<'_, PyAny>,
    what: impl FnOnce() -> String,
) -> PyResult<C> {
    C::from_complex64(extract_complex(val)?, 0.0).ok_or_else(|| imaginary_error(&what(), C::DTYPE))
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
fn stabilizer_expectation<const W: usize, C: PyCoefficient>(
    sum: &CorePauliSum<W, C>,
    generators: &[String],
) -> PyResult<C> {
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
            PyValueError::new_err(format!("expectation_stabilizer: generator {i} {spec:?}: {e}"))
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

/// Build a `PauliSum<W, C>` from raw `(x, z, coefficients)` arrays.
/// Rows are symplectic keys in the Hermitian convention (matching `parse_terms`); a row narrower than `W` words is zero-padded, a wider one is a `ValueError`.
/// Ingest goes through `BuildAccumulator`, so duplicate `(x, z)` rows sum their coefficients and exact-zero rows are dropped.
fn build_from_arrays<const W: usize, C: PyCoefficient>(
    num_qubits: usize,
    x: &PyReadonlyArray2<'_, u64>,
    z: &PyReadonlyArray2<'_, u64>,
    coefficients: &[C],
) -> PyResult<CorePauliSum<W, C>> {
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
    let mut acc = BuildAccumulator::<W, C>::with_capacity(num_qubits, n);
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
        acc.add_term(PauliString::<W> { x: px, z: pz }, coefficients[row]);
    }
    Ok(acc.finalize())
}

/// A 1-D NumPy array of any of the four dtypes as a `Vec<C>`, a `ValueError` when `C` is real and an entry has an imaginary part.
fn extract_coefficient_array<C: PyCoefficient>(val: &Bound<'_, PyAny>) -> PyResult<Vec<C>> {
    if let Ok(arr) = val.extract::<PyReadonlyArray1<C>>() {
        return Ok(arr.as_array().to_vec());
    }
    fn widened<T: PyCoefficient>(val: &Bound<'_, PyAny>) -> Option<Vec<Complex64>> {
        let arr = val.extract::<PyReadonlyArray1<T>>().ok()?;
        let wide = arr.as_array().iter().map(|c| c.to_complex64()).collect();
        Some(wide)
    }
    let wide = widened::<Complex64>(val)
        .or_else(|| widened::<f64>(val))
        .or_else(|| widened::<f32>(val))
        .or_else(|| widened::<Complex32>(val))
        .ok_or_else(|| {
            PyTypeError::new_err(
                "PauliSum.from_arrays: coefficients must be a 1-D float64, float32, complex128 or complex64 NumPy array",
            )
        })?;
    wide.into_iter()
        .enumerate()
        .map(|(i, c)| {
            C::from_complex64(c, 0.0)
                .ok_or_else(|| imaginary_error(&format!("coefficients[{i}]"), C::DTYPE))
        })
        .collect()
}

/// Extract a Python complex/float/int into `Complex64`.
pub(crate) fn extract_complex(val: &Bound<'_, PyAny>) -> PyResult<Complex64> {
    if let Ok(c) = val.downcast::<PyComplex>() {
        return Ok(Complex64::new(c.real(), c.imag()));
    }
    if let Ok(f) = val.extract::<f64>() {
        return Ok(Complex64::new(f, 0.0));
    }
    Err(PyTypeError::new_err("expected complex, float, or int coefficient"))
}

/// The error for operands monomorphized at different widths, which only differing qubit counts produce.
pub(crate) fn width_mismatch() -> PyErr {
    PyValueError::new_err("sums were monomorphized at different widths")
}
