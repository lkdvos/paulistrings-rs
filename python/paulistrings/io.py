"""Pauli-sum file I/O: the ``paulistrings-npz-v1`` format.

Design source: ``research/notes/2026-09-01-python-api-extensions.md`` §A3.
Pure Python glue around ``PauliSum.x_array`` / ``z_array`` /
``coefficients_array`` (export) and ``PauliSum.from_arrays`` (import) — a
small versioned ``.npz`` container, no pickle and no serde, so an evolved
observable can cross a process boundary (the B5 motivating flow:
propagate -> save -> load -> propagate further).

Format ``paulistrings-npz-v1``: an ``np.savez_compressed`` archive with keys
``format`` (this module's version string), ``num_qubits`` (int), ``x``,
``z`` (the ``uint64`` symplectic arrays) and ``coefficients``
(``complex128``). ``load`` hard-errors on a missing or unrecognized
``format`` key rather than guessing — a format change bumps the version
string instead of silently reinterpreting old files.
"""

from __future__ import annotations

import os

import numpy as np

from ._paulistrings import PauliSum

FORMAT = "paulistrings-npz-v1"

__all__ = ["FORMAT", "save", "load"]


def save(path: str | os.PathLike, pauli_sum: PauliSum) -> None:
    """Write `pauli_sum` to `path` in the ``paulistrings-npz-v1`` format.

    `path` gets a ``.npz`` suffix appended by ``numpy.savez_compressed`` if
    it does not already end in one (see the `numpy.savez_compressed` docs);
    `load` accepts back whatever path that produces.
    """
    np.savez_compressed(
        path,
        format=FORMAT,
        num_qubits=np.asarray(pauli_sum.num_qubits, dtype=np.int64),
        x=pauli_sum.x_array(),
        z=pauli_sum.z_array(),
        coefficients=pauli_sum.coefficients_array(),
    )


def load(path: str | os.PathLike) -> PauliSum:
    """Read a `PauliSum` previously written by `save`.

    Raises ``ValueError`` if `path` has no ``format`` field, or one that is
    not ``paulistrings-npz-v1`` — never silently reinterprets an unknown or
    future format.
    """
    with np.load(path, allow_pickle=False) as data:
        if "format" not in data:
            raise ValueError(
                f"{os.fspath(path)!r}: missing 'format' field; not a paulistrings npz file"
            )
        fmt = str(data["format"])
        if fmt != FORMAT:
            raise ValueError(
                f"{os.fspath(path)!r}: unknown format {fmt!r}, expected {FORMAT!r}"
            )
        num_qubits = int(data["num_qubits"])
        x = data["x"]
        z = data["z"]
        coefficients = data["coefficients"]
    return PauliSum.from_arrays(x, z, coefficients, num_qubits)
