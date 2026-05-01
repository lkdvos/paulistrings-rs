//! Python bindings for `paulistrings`.
//!
//! See design doc §11. The user-visible Python surface is a small set of
//! classes (`PauliSum`, `Circuit`) plus three factory submodules (`gates`,
//! `noise`, `truncation`). The width parameter `W` is monomorphized at a
//! fixed set `{1, 2, 4, 8, 16}` and dispatched outside any hot loop (§4).

#![allow(unused)]

mod circuit;
mod gates;
mod noise;
mod sum;
mod truncation;

use pyo3::prelude::*;

#[pymodule]
fn _paulistrings(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<sum::PauliSum>()?;
    m.add_class::<circuit::Circuit>()?;

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
