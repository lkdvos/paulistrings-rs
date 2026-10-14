//! `PauliTerms`: an ordered, width-free list of sparse Pauli terms.
//! Bound to a register by `PauliSum(num_qubits, terms)` or `Circuit.trotter`. See ARCHITECTURE.md §Python-Bindings.

use crate::sparse::{expand_sites, extract_sites, parse_sparse, sparse_key, Axis};
use crate::sum::{extract_complex, format_coeff};
use num_complex::Complex64;
use paulistrings::accumulator::BuildAccumulator;
use paulistrings::phase::Phase;
use paulistrings::PauliSum as CorePauliSum;
use pyo3::exceptions::{PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList, PyTuple};
use std::sync::Arc;

/// One term `coeff · P`.
/// `paulis` is built only by `parse_sparse` in this module, so it is non-empty with distinct sites.
#[derive(Clone)]
pub(crate) struct Term {
    paulis: Arc<[(u32, Axis)]>,
    coeff: Complex64,
}

impl Term {
    pub(crate) fn paulis(&self) -> &[(u32, Axis)] {
        &self.paulis
    }

    pub(crate) fn coeff(&self) -> Complex64 {
        self.coeff
    }
}

/// An ordered list of sparse Pauli terms `c·P`, not bound to a qubit count.
///
/// Built by `pt(label, sites)` and combined with `+`, `-` and scaling by a number.
/// Order and repeated strings are kept, since the list is also the gate list of `Circuit.trotter`.
/// `PauliSum(num_qubits, terms)` binds it to a register and merges repeated strings.
/// Iterating yields `(label, sites, coefficient)` triples.
#[pyclass(frozen, module = "paulistrings._paulistrings", name = "PauliTerms")]
#[derive(Clone)]
pub struct PauliTerms {
    terms: Vec<Term>,
}

impl PauliTerms {
    /// `pt(label, sites)`: one coefficient-1 term per site tuple.
    pub(crate) fn broadcast(label: &str, sites: &Bound<'_, PyAny>) -> PyResult<Self> {
        let k = label.chars().count();
        let terms = expand_sites(extract_sites(sites)?, k)
            .iter()
            .map(|s| {
                Ok(Term {
                    paulis: parse_sparse("pt", label, s)?.into(),
                    coeff: Complex64::new(1.0, 0.0),
                })
            })
            .collect::<PyResult<_>>()?;
        Ok(Self { terms })
    }

    pub(crate) fn terms(&self) -> &[Term] {
        &self.terms
    }

    fn scaled(&self, factor: Complex64) -> impl Iterator<Item = Term> + '_ {
        self.terms.iter().map(move |t| Term {
            paulis: Arc::clone(&t.paulis),
            coeff: t.coeff * factor,
        })
    }

    /// Raise before any width-dependent work if a term reaches past `num_qubits`.
    pub(crate) fn check_width(&self, num_qubits: usize, what: &str) -> PyResult<()> {
        let max = self
            .terms
            .iter()
            .flat_map(|t| t.paulis.iter().map(|&(q, _)| q))
            .max();
        match max {
            Some(q) if q as usize >= num_qubits => Err(PyValueError::new_err(format!(
                "{what}: qubit index {q} is out of range for {num_qubits} qubits"
            ))),
            _ => Ok(()),
        }
    }

    /// Accumulate at width `W`; repeated strings sum and exact zeros drop. Caller has run `check_width`.
    pub(crate) fn accumulate<const W: usize>(&self, num_qubits: usize) -> CorePauliSum<W> {
        let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, self.terms.len());
        for t in &self.terms {
            acc.add_term(sparse_key::<W>(&t.paulis), Phase::ONE, t.coeff);
        }
        acc.finalize()
    }
}

/// A Python number as a factor, naming the operator in the error.
fn factor(value: &Bound<'_, PyAny>, op: &str) -> PyResult<Complex64> {
    if value.downcast::<PauliTerms>().is_ok() {
        return Err(PyTypeError::new_err(format!(
            "PauliTerms {op} PauliTerms is not supported: {op} takes a complex or real number"
        )));
    }
    extract_complex(value).map_err(|_| {
        PyTypeError::new_err(format!(
            "PauliTerms {op} x: x must be a complex or real number"
        ))
    })
}

#[pymethods]
impl PauliTerms {
    /// The empty list, the start value for `sum(..., PauliTerms())`.
    #[new]
    fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// `None`: a NumPy scalar on the left defers to `__rmul__` instead of iterating this object into an array.
    #[classattr]
    fn __array_ufunc__(py: Python<'_>) -> PyObject {
        py.None()
    }

    fn __len__(&self) -> usize {
        self.terms.len()
    }

    fn __add__(&self, other: &Self) -> Self {
        Self {
            terms: self
                .terms
                .iter()
                .cloned()
                .chain(other.terms.iter().cloned())
                .collect(),
        }
    }

    fn __sub__(&self, other: &Self) -> Self {
        Self {
            terms: self
                .terms
                .iter()
                .cloned()
                .chain(other.scaled(Complex64::new(-1.0, 0.0)))
                .collect(),
        }
    }

    /// `sum()` starts from the integer `0`.
    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if other.extract::<i64>().is_ok_and(|v| v == 0) {
            return Ok(Py::new(py, self.clone())?.into_any());
        }
        Ok(py.NotImplemented())
    }

    fn __neg__(&self) -> Self {
        Self {
            terms: self.scaled(Complex64::new(-1.0, 0.0)).collect(),
        }
    }

    fn __mul__(&self, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            terms: self.scaled(factor(value, "*")?).collect(),
        })
    }

    fn __rmul__(&self, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(value)
    }

    fn __truediv__(&self, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let d = factor(value, "/")?;
        if d == Complex64::new(0.0, 0.0) {
            return Err(PyZeroDivisionError::new_err("PauliTerms / 0"));
        }
        Ok(Self {
            terms: self.scaled(Complex64::new(1.0, 0.0) / d).collect(),
        })
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let items = self.terms.iter().map(|t| -> PyObject {
            let label: String = t.paulis.iter().map(|&(_, axis)| axis.as_char()).collect();
            let sites = PyTuple::new_bound(py, t.paulis.iter().map(|&(q, _)| q));
            (label, sites, t.coeff).into_py(py)
        });
        PyList::new_bound(py, items).into_any().iter()
    }

    /// The first few terms as `coefficient*Pj...`, `+`-joined; `0` for the empty list.
    fn __repr__(&self) -> String {
        const SHOWN: usize = 4;
        if self.terms.is_empty() {
            return "0".to_string();
        }
        let mut parts: Vec<String> = self
            .terms
            .iter()
            .take(SHOWN)
            .map(|t| {
                let string: String = t
                    .paulis
                    .iter()
                    .map(|&(q, axis)| format!("{}{q}", axis.as_char()))
                    .collect();
                format!("{}*{string}", format_coeff(t.coeff))
            })
            .collect();
        let rest = self.terms.len().saturating_sub(SHOWN);
        if rest > 0 {
            parts.push(format!(
                "... ({rest} more term{})",
                if rest == 1 { "" } else { "s" }
            ));
        }
        parts.join(" + ")
    }
}
