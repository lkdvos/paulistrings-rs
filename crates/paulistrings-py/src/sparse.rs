//! Sparse Pauli strings: a label of `X`/`Y`/`Z` paired with distinct sites, identity elsewhere.
//! Shared by `pt(label, sites)`, `PauliTerms`, `pauli_rotation` and the single-site `PauliString` constructors.

use paulistrings::pauli_string::PauliString;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

/// The Pauli on one site of a sparse string.
/// No `I`: an identity site is expressed by omitting it, so every string has one spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub(crate) fn as_char(self) -> char {
        match self {
            Self::X => 'X',
            Self::Y => 'Y',
            Self::Z => 'Z',
        }
    }
}

/// A `sites=` argument as spelled, before it is matched against a label's length.
pub(crate) enum Sites {
    One(u32),
    Flat(Vec<u32>),
    Nested(Vec<Vec<u32>>),
}

/// Read `sites=` at the Python boundary.
pub(crate) fn extract_sites(obj: &Bound<'_, PyAny>) -> PyResult<Sites> {
    // `bool` is an `int` subclass, so `True` would otherwise mean qubit 1.
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err("sites must be ints, not a bool"));
    }
    if let Ok(q) = obj.extract::<u32>() {
        return Ok(Sites::One(q));
    }
    if let Ok(flat) = obj.extract::<Vec<u32>>() {
        return Ok(Sites::Flat(flat));
    }
    obj.extract::<Vec<Vec<u32>>>()
        .map(Sites::Nested)
        .map_err(|_| {
            PyTypeError::new_err(
                "sites must be an int, a sequence of ints, or a sequence of int sequences",
            )
        })
}

/// The site tuples `sites` names for a `k`-character label.
/// A flat sequence is one tuple, or one tuple per entry when `k == 1`; an empty sequence names none.
pub(crate) fn expand_sites(sites: Sites, k: usize) -> Vec<Vec<u32>> {
    match sites {
        Sites::One(q) => vec![vec![q]],
        Sites::Flat(flat) if flat.is_empty() && k > 0 => Vec::new(),
        Sites::Flat(flat) if k == 1 => flat.into_iter().map(|q| vec![q]).collect(),
        Sites::Flat(flat) => vec![flat],
        Sites::Nested(nested) => nested,
    }
}

/// `label[k]` on `sites[k]`, validated.
/// Plain Rust so `cargo test` covers it; `parse_sparse` is the `PyResult` wrapper.
pub(crate) fn parse_sparse_impl(
    what: &str,
    label: &str,
    sites: &[u32],
) -> Result<Vec<(u32, Axis)>, String> {
    let k = label.chars().count();
    if k != sites.len() {
        return Err(format!(
            "{what}: pauli and qubits must have the same length (got {k} characters and {} qubits)",
            sites.len()
        ));
    }
    if sites.is_empty() {
        return Err(format!("{what}: pauli and qubits must be non-empty"));
    }
    let mut paulis = Vec::with_capacity(k);
    for (ch, &qubit) in label.chars().zip(sites) {
        let axis = match ch {
            'X' => Axis::X,
            'Y' => Axis::Y,
            'Z' => Axis::Z,
            other => {
                return Err(format!(
                    "{what}: unexpected Pauli character {other:?} (expected X/Y/Z; identity positions are expressed by omission)"
                ))
            }
        };
        paulis.push((qubit, axis));
    }
    // A repeated site would make `sparse_key` OR two Paulis into a phase-free product: silently the wrong operator.
    for i in 0..paulis.len() {
        if paulis[i + 1..].iter().any(|&(q, _)| q == paulis[i].0) {
            return Err(format!(
                "{what}: qubits must be distinct (index {} appears twice)",
                paulis[i].0
            ));
        }
    }
    Ok(paulis)
}

pub(crate) fn parse_sparse(what: &str, label: &str, sites: &[u32]) -> PyResult<Vec<(u32, Axis)>> {
    parse_sparse_impl(what, label, sites).map_err(PyValueError::new_err)
}

/// Symplectic key of a sparse string: X sets the x bit, Z the z bit, Y both.
/// Sites must be distinct (`parse_sparse` guarantees it), so no phase arises.
pub(crate) fn sparse_key<const W: usize>(paulis: &[(u32, Axis)]) -> PauliString<W> {
    let mut key = PauliString::<W>::identity();
    for &(qubit, axis) in paulis {
        let (word, bit) = (qubit as usize / 64, 1u64 << (qubit % 64));
        if axis != Axis::Z {
            key.x[word] |= bit;
        }
        if axis != Axis::X {
            key.z[word] |= bit;
        }
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_key_encodes_x_y_z() {
        let key = sparse_key::<1>(&[(0, Axis::X), (1, Axis::Y), (2, Axis::Z)]);
        assert_eq!((key.x, key.z), ([0b011], [0b110]));
    }

    #[test]
    fn sparse_key_reaches_the_second_word() {
        let key = sparse_key::<2>(&[(70, Axis::Y)]);
        assert_eq!((key.x, key.z), ([0, 1 << 6], [0, 1 << 6]));
    }

    #[test]
    fn expand_sites_follows_the_three_cases() {
        assert_eq!(expand_sites(Sites::One(3), 1), vec![vec![3]]);
        assert_eq!(
            expand_sites(Sites::Flat(vec![0, 1, 2]), 1),
            vec![vec![0], vec![1], vec![2]]
        );
        assert_eq!(expand_sites(Sites::Flat(vec![0, 1]), 2), vec![vec![0, 1]]);
        assert_eq!(
            expand_sites(Sites::Nested(vec![vec![0, 1], vec![1, 2]]), 2),
            vec![vec![0, 1], vec![1, 2]]
        );
        assert!(expand_sites(Sites::Flat(vec![]), 2).is_empty());
    }

    #[test]
    fn parse_sparse_rejects_identity_repeats_and_mismatch() {
        assert_eq!(
            parse_sparse_impl("pt", "XZ", &[4, 1]).unwrap(),
            vec![(4, Axis::X), (1, Axis::Z)]
        );
        assert!(parse_sparse_impl("pt", "ZI", &[0, 1])
            .unwrap_err()
            .contains("unexpected Pauli"));
        assert!(parse_sparse_impl("pt", "ZZ", &[3, 3])
            .unwrap_err()
            .contains("appears twice"));
        assert!(parse_sparse_impl("pt", "ZZ", &[0])
            .unwrap_err()
            .contains("same length"));
    }
}
