//! Python `PauliString` class with width-monomorphized backing storage. See
//! ARCHITECTURE.md §Width and ARCHITECTURE.md §Python-Bindings.

use crate::sum::parse_pauli_key;
use num_complex::Complex64;
use paulistrings::pauli_string::PauliString as CorePauliString;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Width-dispatch enum, `PauliSumImpl`'s counterpart for a single string: the boundary picks the smallest width that fits `num_qubits`.
/// The core type carries no qubit count of its own, so the Python class keeps it alongside.
#[derive(Clone, Copy)]
pub enum PauliStringImpl {
    W1(CorePauliString<1>),
    W2(CorePauliString<2>),
    W4(CorePauliString<4>),
    W8(CorePauliString<8>),
    W16(CorePauliString<16>),
}

/// Which single-site constructor a `PauliString.x/y/z` call wants, chosen before the width is known.
#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
    Z,
}

/// Which product a binary method reports the coefficient of.
#[derive(Clone, Copy)]
enum Bracket {
    Mul,
    Commutator,
    Anticommutator,
}

impl PauliStringImpl {
    /// Identity at the smallest width fitting `num_qubits`. `None` above 1024 qubits.
    fn identity_for(num_qubits: usize) -> Option<Self> {
        for_num_qubits!(num_qubits, |W| CorePauliString::<W>::identity())
    }

    /// Single-site `X`/`Y`/`Z`. Caller has already checked `qubit < num_qubits`.
    fn single(axis: Axis, qubit: usize, num_qubits: usize) -> Option<Self> {
        for_num_qubits!(num_qubits, |W| match axis {
            Axis::X => CorePauliString::<W>::x(qubit as u32),
            Axis::Y => CorePauliString::<W>::y(qubit as u32),
            Axis::Z => CorePauliString::<W>::z(qubit as u32),
        })
    }

    /// Parse an `IXYZ` label, one character per qubit, through the same parser `PauliSum.from_strings` uses.
    fn from_label(label: &str) -> PyResult<Option<Self>> {
        Ok(for_num_qubits!(
            label.chars().count(),
            |W| parse_pauli_key::<W>(label)?
        ))
    }

    fn weight(&self) -> u32 {
        for_each_width!(self, |p| p.weight())
    }

    /// The `IXYZ` label, one character per qubit of `num_qubits`.
    fn label(&self, num_qubits: usize) -> String {
        for_each_width!(self, |p| label_of(p, num_qubits))
    }

    /// `None` when the two strings were monomorphized at different widths, which can only happen if their qubit counts fall in different dispatch bands.
    fn commutes_with(&self, other: &Self) -> Option<bool> {
        for_each_width_pair!((self, other), |a, b| a.commutes_with(b))
    }

    /// `(coefficient, product)` for whichever bracket `op` names, `None` on a width mismatch.
    fn bracket(&self, other: &Self, op: Bracket) -> Option<(Complex64, Self)> {
        for_each_width_pair_rewrap!((self, other), |a, b, wrap| {
            let (product, coeff) = match op {
                Bracket::Mul => {
                    let (product, phase) = a.mul(b);
                    (product, phase.to_complex())
                }
                Bracket::Commutator => a.commutator(b),
                Bracket::Anticommutator => a.anticommutator(b),
            };
            (coeff, wrap(product))
        })
    }

    /// Symplectic key, for hashing and equality.
    fn key(&self) -> (&[u64], &[u64]) {
        for_each_width!(self, |p| (&p.x[..], &p.z[..]))
    }
}

/// Decode a symplectic key back into its `IXYZ` label — `parse_pauli_key`'s inverse, in the same Hermitian convention (`Y` is the `(x=1, z=1)` key, with no phase factor).
fn label_of<const W: usize>(p: &CorePauliString<W>, num_qubits: usize) -> String {
    (0..num_qubits)
        .map(|q| {
            let word = q / 64;
            let bit = 1u64 << (q % 64);
            match (p.x[word] & bit != 0, p.z[word] & bit != 0) {
                (false, false) => 'I',
                (true, false) => 'X',
                (false, true) => 'Z',
                (true, true) => 'Y',
            }
        })
        .collect()
}

/// The `ValueError` every constructor raises above the largest monomorphized width.
fn too_wide() -> PyErr {
    PyValueError::new_err("num_qubits exceeds largest monomorphized width (1024)")
}

/// A single Pauli string: the symplectic `(x, z)` key of one `IXYZ` word, with no coefficient attached.
///
/// Immutable and hashable. `str` and `repr` both print the label, and two strings compare equal when their labels and `num_qubits` agree.
/// The five constructors are `identity`, `x`, `y`, `z` and `from_label`; `paulistrings.p(label)` is shorthand for the last.
#[pyclass(frozen, module = "paulistrings._paulistrings", name = "PauliString")]
pub struct PauliString {
    inner: PauliStringImpl,
    num_qubits: usize,
}

impl PauliString {
    fn build(inner: Option<PauliStringImpl>, num_qubits: usize) -> PyResult<Self> {
        Ok(Self {
            inner: inner.ok_or_else(too_wide)?,
            num_qubits,
        })
    }

    /// `from_label`'s body, shared with the module-level `p()` shorthand.
    pub(crate) fn parse(label: &str) -> PyResult<Self> {
        Self::build(PauliStringImpl::from_label(label)?, label.chars().count())
    }

    /// Reject a qubit index the string cannot address, mirroring `Circuit.append`'s message.
    fn check_qubit(qubit: usize, num_qubits: usize) -> PyResult<()> {
        if qubit >= num_qubits {
            return Err(PyValueError::new_err(format!(
                "qubit index {qubit} is out of range for a {num_qubits}-qubit Pauli string"
            )));
        }
        Ok(())
    }

    /// Both operands of a binary method must describe the same number of qubits; anything else is a `ValueError` rather than a silently padded comparison.
    fn check_same_width(&self, other: &Self, what: &str) -> PyResult<()> {
        if self.num_qubits != other.num_qubits {
            return Err(PyValueError::new_err(format!(
                "{what}: num_qubits mismatch ({} vs {})",
                self.num_qubits, other.num_qubits
            )));
        }
        Ok(())
    }

    fn bracket(&self, other: &Self, op: Bracket, what: &str) -> PyResult<(Complex64, Self)> {
        self.check_same_width(other, what)?;
        let (coeff, product) = self
            .inner
            .bracket(&other.inner, op)
            .ok_or_else(|| PyValueError::new_err(format!("{what}: width mismatch")))?;
        Ok((
            coeff,
            Self {
                inner: product,
                num_qubits: self.num_qubits,
            },
        ))
    }
}

#[pymethods]
impl PauliString {
    /// The identity string on `num_qubits` qubits, `"III..."`.
    #[classmethod]
    fn identity(_cls: &Bound<'_, PyType>, num_qubits: usize) -> PyResult<Self> {
        Self::build(PauliStringImpl::identity_for(num_qubits), num_qubits)
    }

    /// `X` on `qubit`, identity elsewhere.
    #[classmethod]
    fn x(_cls: &Bound<'_, PyType>, qubit: usize, num_qubits: usize) -> PyResult<Self> {
        Self::check_qubit(qubit, num_qubits)?;
        Self::build(
            PauliStringImpl::single(Axis::X, qubit, num_qubits),
            num_qubits,
        )
    }

    /// `Y` on `qubit`, identity elsewhere. Hermitian convention: the `(x=1, z=1)` key carries no phase of its own.
    #[classmethod]
    fn y(_cls: &Bound<'_, PyType>, qubit: usize, num_qubits: usize) -> PyResult<Self> {
        Self::check_qubit(qubit, num_qubits)?;
        Self::build(
            PauliStringImpl::single(Axis::Y, qubit, num_qubits),
            num_qubits,
        )
    }

    /// `Z` on `qubit`, identity elsewhere.
    #[classmethod]
    fn z(_cls: &Bound<'_, PyType>, qubit: usize, num_qubits: usize) -> PyResult<Self> {
        Self::check_qubit(qubit, num_qubits)?;
        Self::build(
            PauliStringImpl::single(Axis::Z, qubit, num_qubits),
            num_qubits,
        )
    }

    /// Parse an `IXYZ` label; `num_qubits` is its length. Same alphabet and qubit indexing as `PauliSum.from_strings` (character `i` addresses qubit `i`).
    #[classmethod]
    fn from_label(_cls: &Bound<'_, PyType>, label: &str) -> PyResult<Self> {
        Self::parse(label)
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of non-identity factors — the quantity `truncation.weight` caps.
    #[getter]
    fn weight(&self) -> u32 {
        self.inner.weight()
    }

    /// The `IXYZ` label, `num_qubits` characters long.
    #[getter]
    fn label(&self) -> String {
        self.inner.label(self.num_qubits)
    }

    fn __str__(&self) -> String {
        self.label()
    }

    fn __repr__(&self) -> String {
        self.label()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.downcast::<Self>() {
            Ok(other) => {
                let other = other.get();
                self.num_qubits == other.num_qubits && self.inner.key() == other.inner.key()
            }
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.num_qubits.hash(&mut hasher);
        let (x, z) = self.inner.key();
        x.hash(&mut hasher);
        z.hash(&mut hasher);
        hasher.finish()
    }

    /// Whether the two strings commute, i.e. whether their symplectic inner product is even.
    fn commutes_with(&self, other: &Self) -> PyResult<bool> {
        self.check_same_width(other, "commutes_with")?;
        self.inner
            .commutes_with(&other.inner)
            .ok_or_else(|| PyValueError::new_err("commutes_with: width mismatch"))
    }

    /// Exactly `not commutes_with(other)`: a pair of Pauli strings always does one or the other.
    fn anticommutes_with(&self, other: &Self) -> PyResult<bool> {
        Ok(!self.commutes_with(other)?)
    }

    /// Product `self · other` as `(coefficient, string)`, the coefficient being the `i^k` phase — e.g. `X·Z = -i·Y`.
    fn mul(&self, other: &Self) -> PyResult<(Complex64, Self)> {
        self.bracket(other, Bracket::Mul, "mul")
    }

    /// Commutator `[self, other]` as `(coefficient, string)`: `2 · mul(self, other)`'s coefficient when the two anticommute, exactly `0` when they commute.
    fn commutator(&self, other: &Self) -> PyResult<(Complex64, Self)> {
        self.bracket(other, Bracket::Commutator, "commutator")
    }

    /// Anticommutator `{self, other}` as `(coefficient, string)` — `commutator`'s mirror: `2 · mul`'s coefficient when the two commute, exactly `0` when they anticommute.
    fn anticommutator(&self, other: &Self) -> PyResult<(Complex64, Self)> {
        self.bracket(other, Bracket::Anticommutator, "anticommutator")
    }
}
