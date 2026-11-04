//! The coefficient dtypes a Python sum holds: the runtime [`Dtype`] tag, the [`PyCoefficient`] binding-side bound on the core's `Coefficient`, and the [`Scalar`] a readout returns.

use num_complex::{Complex32, Complex64};
use numpy::{PyArrayDescr, PyArrayDescrMethods};
use paulistrings::Coefficient;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// One of the four coefficient types, as Python names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dtype {
    F64,
    F32,
    C128,
    C64,
}

impl Dtype {
    /// The NumPy name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Dtype::F64 => "float64",
            Dtype::F32 => "float32",
            Dtype::C128 => "complex128",
            Dtype::C64 => "complex64",
        }
    }

    pub(crate) fn is_real(self) -> bool {
        matches!(self, Dtype::F64 | Dtype::F32)
    }

    fn is_double(self) -> bool {
        matches!(self, Dtype::F64 | Dtype::C128)
    }

    /// The complex dtype of the same precision.
    pub(crate) fn complex(self) -> Self {
        match self {
            Dtype::F64 | Dtype::C128 => Dtype::C128,
            Dtype::F32 | Dtype::C64 => Dtype::C64,
        }
    }

    /// NumPy's `result_type` restricted to the four: complex if either is, double if either is.
    pub(crate) fn promote(self, other: Self) -> Self {
        match (self.is_real() && other.is_real(), self.is_double() || other.is_double()) {
            (true, true) => Dtype::F64,
            (true, false) => Dtype::F32,
            (false, true) => Dtype::C128,
            (false, false) => Dtype::C64,
        }
    }

    /// `dtype=` as anything `numpy.dtype` accepts, `None` meaning `complex128`; any other dtype is a `TypeError`.
    pub(crate) fn parse(py: Python<'_>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let Some(dtype) = dtype.filter(|obj| !obj.is_none()) else {
            return Ok(Dtype::C128);
        };
        let unsupported = |shown: String| {
            PyTypeError::new_err(format!(
                "dtype={shown}: a PauliSum holds float64, float32, complex128 or complex64 coefficients"
            ))
        };
        let shown = || dtype.repr().map_or_else(|_| "…".to_string(), |repr| repr.to_string());
        let descr = py
            .import_bound("numpy")?
            .getattr("dtype")?
            .call1((dtype,))
            .map_err(|_| unsupported(shown()))?;
        let descr = descr.downcast::<PyArrayDescr>().map_err(|_| unsupported(shown()))?;
        [Dtype::F64, Dtype::F32, Dtype::C128, Dtype::C64]
            .into_iter()
            .find(|candidate| descr.is_equiv_to(&candidate.numpy(py)))
            .ok_or_else(|| unsupported(shown()))
    }

    /// The `numpy.dtype` object.
    pub(crate) fn numpy(self, py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        match self {
            Dtype::F64 => numpy::dtype_bound::<f64>(py),
            Dtype::F32 => numpy::dtype_bound::<f32>(py),
            Dtype::C128 => numpy::dtype_bound::<Complex64>(py),
            Dtype::C64 => numpy::dtype_bound::<Complex32>(py),
        }
    }
}

/// A core [`Coefficient`] the bindings can hand to NumPy and print.
pub(crate) trait PyCoefficient: Coefficient + numpy::Element + IntoPy<PyObject> {
    const DTYPE: Dtype;

    /// `0.25` for a real value, `(0.25+0.5j)` otherwise, at the type's own precision.
    fn format(self) -> String;
}

macro_rules! py_real {
    ($t:ty, $dtype:expr) => {
        impl PyCoefficient for $t {
            const DTYPE: Dtype = $dtype;

            fn format(self) -> String {
                format!("{self}")
            }
        }
    };
}

macro_rules! py_complex {
    ($t:ty, $dtype:expr) => {
        impl PyCoefficient for $t {
            const DTYPE: Dtype = $dtype;

            fn format(self) -> String {
                if self.im == 0.0 {
                    format!("{}", self.re)
                } else {
                    let sign = if self.im < 0.0 { "-" } else { "+" };
                    format!("({}{sign}{}j)", self.re, self.im.abs())
                }
            }
        }
    };
}

py_real!(f64, Dtype::F64);
py_real!(f32, Dtype::F32);
py_complex!(Complex64, Dtype::C128);
py_complex!(Complex32, Dtype::C64);

/// `value` as a `C`, a `ValueError` naming `what` when `C` is real and `value` has an imaginary part.
pub(crate) fn narrow<C: PyCoefficient>(value: Complex64, what: &str) -> PyResult<C> {
    C::from_complex64(value, 0.0).ok_or_else(|| imaginary_error(what, C::DTYPE))
}

/// The `ValueError` for an imaginary part a real dtype cannot hold.
pub(crate) fn imaginary_error(what: &str, dtype: Dtype) -> PyErr {
    PyValueError::new_err(format!(
        "{what} has a nonzero imaginary part, which a {} PauliSum cannot hold; use dtype=complex128 or complex64",
        dtype.name()
    ))
}

/// A readout value: a Python `float` from a real sum, a `complex` from a complex one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Scalar {
    Real(f64),
    Complex(Complex64),
}

impl Scalar {
    pub(crate) fn of<C: Coefficient>(value: C) -> Self {
        let wide = value.to_complex64();
        if C::IS_REAL {
            Scalar::Real(wide.re)
        } else {
            Scalar::Complex(wide)
        }
    }
}

impl IntoPy<PyObject> for Scalar {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Scalar::Real(value) => value.into_py(py),
            Scalar::Complex(value) => value.into_py(py),
        }
    }
}

#[cfg(test)]
mod tests;
