//! Python bindings for `paulistrings`.
//!
//! See design doc §11. The user-visible Python surface is a small set of
//! classes (`PauliSum`, `Circuit`) plus three factory submodules (`gates`,
//! `noise`, `truncation`). The width parameter `W` is monomorphized at a
//! fixed set `{1, 2, 4, 8, 16}` and dispatched outside any hot loop (§4).

#![allow(unused)]
// PyO3 0.22's `#[pymethods]` expansion converts return values via
// `.into()` even when the method already returns a `PyResult<T>`, which
// clippy flags as a useless conversion. The lint is on the macro output, not
// our code, so we silence it crate-wide.
#![allow(clippy::useless_conversion)]

mod channel_spec;
mod circuit;
mod gates;
mod noise;
mod sum;
mod truncation;
mod truncation_spec;

use pyo3::prelude::*;

#[pymodule]
fn _paulistrings(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<sum::PauliSum>()?;
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
