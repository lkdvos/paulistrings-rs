"""``PauliSum.anticommute_histogram`` / ``PauliSum.rotated_overlap`` and ``paulistrings.diagonal_echo``.

Hand values on one- and two-string sums, then a dense ``2**-n Tr(A V^dag A V)`` oracle on random three-qubit sums.
The ``partitions=`` / ``comm=`` paths are in ``test_partitioned.py`` and ``test_mpi.py``.
"""

import math
from functools import reduce

import numpy as np
import pytest

import paulistrings
from paulistrings import PauliSum

from .test_stabilizer import _PAULI, _dense_pauli

DELTA = 0.3
TOL = 1e-12


def _one(label, coeff=1.0):
    return PauliSum.from_strings({label: coeff}, num_qubits=len(label))


@pytest.mark.parametrize(
    "label, axis, anticommutes",
    [
        ("X", "z", True),
        ("Y", "z", True),
        ("Z", "z", False),
        ("Z", "x", True),
        ("Y", "x", True),
        ("X", "x", False),
        ("X", "Z", True),  # case-insensitive
        ("Z", None, True),  # the default axis is x
    ],
)
def test_a_single_string_echoes_cos_two_delta_iff_it_anticommutes(label, axis, anticommutes):
    s, kw = _one(label), {} if axis is None else {"axis": axis}
    assert s.rotated_overlap([0], DELTA, **kw) == pytest.approx(math.cos(2 * DELTA) if anticommutes else 1.0, abs=TOL)
    assert s.anticommute_histogram([0], **kw) == ([0.0, 1.0] if anticommutes else [1.0, 0.0])


def test_the_histogram_counts_anticommuting_sites():
    s = PauliSum.from_strings({"XXI": 0.6, "ZIZ": 0.8}, num_qubits=3)
    assert s.anticommute_histogram([0, 1, 2], axis="z") == pytest.approx([0.64, 0.0, 0.36, 0.0], abs=TOL)
    assert s.anticommute_histogram([0, 1, 2], axis="x") == pytest.approx([0.36, 0.0, 0.64, 0.0], abs=TOL)
    assert s.anticommute_histogram([], axis="x") == pytest.approx([1.0], abs=TOL)


def test_at_zero_delta_the_echo_is_the_squared_norm():
    rng = np.random.default_rng(7)
    labels = ["".join(rng.choice(list("IXYZ"), size=5)) for _ in range(40)]
    s = PauliSum.from_strings({lab: float(rng.normal()) for lab in labels}, num_qubits=5)
    norm = sum(abs(c) ** 2 for c in s.coefficients())
    for axis in ("x", "z"):
        assert s.rotated_overlap([0, 2, 3], 0.0, axis=axis) == pytest.approx(norm, rel=1e-12)
        assert sum(s.anticommute_histogram([0, 2, 3], axis=axis)) == pytest.approx(norm, rel=1e-12)


def test_the_exact_echo_keeps_what_the_diagonal_drops():
    """``XX + YY`` commutes with ``Z_0 + Z_1``, so the exact echo is ``2`` at every ``delta``; the diagonal approximation, which drops the ``XX``/``YY`` cross terms, is ``2 cos(2 delta)**2``."""
    s = PauliSum.from_strings({"XX": 1.0, "YY": 1.0}, num_qubits=2)
    assert s.rotated_overlap([0, 1], DELTA, axis="z") == pytest.approx(2.0, abs=TOL)
    hist = s.anticommute_histogram([0, 1], axis="z")
    assert hist == pytest.approx([0.0, 0.0, 2.0], abs=TOL)
    assert paulistrings.diagonal_echo(hist, DELTA) == pytest.approx(math.cos(2 * DELTA) ** 2, abs=TOL)


def test_diagonal_echo():
    assert paulistrings.diagonal_echo([0.5, 0.0, 0.5], DELTA) == pytest.approx(
        0.5 + 0.5 * math.cos(2 * DELTA) ** 2, abs=TOL
    )
    assert math.isnan(paulistrings.diagonal_echo([0.0, 0.0], DELTA))


def _dense_echo(terms, sites, delta, axis):
    """``2**-n Tr(A V^dag A V)`` from matrices, qubit ``i`` the ``i``-th Kronecker factor like label character ``i``."""
    n = len(next(iter(terms)))
    A = sum(c * _dense_pauli(label) for label, c in terms.items())
    g = _PAULI[axis.upper()]
    rot = math.cos(delta) * _PAULI["I"] - 1j * math.sin(delta) * g
    V = reduce(np.kron, [rot if q in sites else _PAULI["I"] for q in range(n)])
    return float(np.real(np.trace(A @ V.conj().T @ A @ V)) / 2**n)


@pytest.mark.parametrize("axis", ["x", "z"])
@pytest.mark.parametrize("seed", range(4))
def test_rotated_overlap_matches_a_dense_trace(axis, seed):
    rng = np.random.default_rng(seed)
    terms = {"".join(rng.choice(list("IXYZ"), size=3)): float(rng.normal()) for _ in range(24)}
    s = PauliSum.from_strings(terms, num_qubits=3)
    for sites in ([0], [0, 2], [0, 1, 2]):
        assert s.rotated_overlap(sites, 0.37, axis=axis) == pytest.approx(
            _dense_echo(terms, sites, 0.37, axis), abs=1e-12
        )


@pytest.mark.parametrize("axis", ["y", "", "xz"])
def test_an_unknown_axis_is_a_value_error(axis):
    s = _one("X")
    for call in (lambda: s.rotated_overlap([0], DELTA, axis=axis), lambda: s.anticommute_histogram([0], axis=axis)):
        with pytest.raises(ValueError, match="axis must be 'x' or 'z'"):
            call()


@pytest.mark.parametrize("sites, message", [([3], "out of range"), ([0, 0], "listed twice")])
def test_bad_sites_are_value_errors(sites, message):
    s = PauliSum.from_strings({"XYZ": 1.0}, num_qubits=3)
    for call in (lambda: s.rotated_overlap(sites, DELTA), lambda: s.anticommute_histogram(sites)):
        with pytest.raises(ValueError, match=message):
            call()


@pytest.mark.skipif(paulistrings.mpi_available(), reason="the mpi build answers comm= itself (test_mpi.py)")
def test_comm_without_the_mpi_feature_is_a_runtime_error():
    s = _one("X")
    for call in (lambda: s.rotated_overlap([0], DELTA, comm=object()), lambda: s.anticommute_histogram([0], comm=object())):
        with pytest.raises(RuntimeError, match="without MPI support"):
            call()
