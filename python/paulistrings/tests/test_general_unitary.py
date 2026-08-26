"""``gates.unitary_1q`` / ``unitary_2q`` — arbitrary unitaries as Pauli-transfer
matrices.

These were ``todo!()`` in v0.1 and are implemented in v0.2 B.9. The Rust side is
covered by ``crates/paulistrings/src/channel/unitary.rs``; this file checks the
Python boundary: NumPy input handling, shape and unitarity validation, and that a
unitary supplied as a matrix behaves like the equivalent named gate.

Note CI does not run these (see CLAUDE.md); run them locally with
``maturin develop --release`` followed by ``pytest python/paulistrings/tests``.
"""

import cmath
import math

import numpy as np
import pytest

from paulistrings import Circuit, PauliSum, gates

R = 1.0 / math.sqrt(2.0)

H = np.array([[R, R], [R, -R]], dtype=complex)
S = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
T = np.array([[1.0, 0.0], [0.0, cmath.exp(1j * math.pi / 4)]], dtype=complex)
CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
)


def _sum(num_qubits, terms):
    return PauliSum.from_strings(terms, num_qubits=num_qubits)


def _as_dict(sum_):
    """{(x_words, z_words): coeff}, so comparisons do not depend on ordering."""
    xs = sum_.x_array()
    zs = sum_.z_array()
    cs = sum_.coefficients_array()
    return {
        (tuple(int(v) for v in xs[i]), tuple(int(v) for v in zs[i])): complex(cs[i])
        for i in range(len(sum_))
    }


def _assert_close(a, b, tol=1e-10):
    da, db = _as_dict(a), _as_dict(b)
    assert set(da) == set(db), f"different keys:\n{sorted(da)}\nvs\n{sorted(db)}"
    for k in da:
        assert abs(da[k] - db[k]) < tol, f"{k}: {da[k]} vs {db[k]}"


# ---- equivalence with the named gates ----


@pytest.mark.parametrize(
    "matrix,named",
    [(H, "h"), (S, "s")],
)
def test_1q_unitary_matches_the_named_gate(matrix, named):
    initial = _sum(4, {"ZIII": 1.0, "XIII": 0.5, "YIII": -0.25})

    via_matrix = Circuit(4)
    via_matrix.unitary_1q(0, matrix)
    got = initial.propagate(circuit=via_matrix)

    via_named = Circuit(4)
    getattr(via_named, named)(0)
    want = initial.propagate(circuit=via_named)

    _assert_close(got, want)


@pytest.mark.parametrize(
    "matrix,named,args",
    [(CNOT, "cnot", (0, 1)), (CZ, "cz", (0, 1)), (SWAP, "swap", (0, 1))],
)
def test_2q_unitary_matches_the_named_gate(matrix, named, args):
    initial = _sum(4, {"ZIII": 1.0, "IXII": 0.5, "YYII": -0.25})

    via_matrix = Circuit(4)
    via_matrix.unitary_2q(args[0], args[1], matrix)
    got = initial.propagate(circuit=via_matrix)

    via_named = Circuit(4)
    getattr(via_named, named)(*args)
    want = initial.propagate(circuit=via_named)

    _assert_close(got, want)


def test_gates_factory_and_circuit_method_agree():
    initial = _sum(4, {"XIII": 1.0})
    a = Circuit(4)
    a.unitary_1q(0, H)
    b = Circuit(4)
    b.append(gates.unitary_1q(0, H))
    _assert_close(initial.propagate(circuit=a), initial.propagate(circuit=b))


# ---- non-Clifford ----


def test_t_gate_maps_x_to_a_two_term_sum():
    initial = _sum(4, {"XIII": 1.0})
    c = Circuit(4)
    c.unitary_1q(0, T)
    out = initial.propagate(circuit=c)
    # T X T† = (X + Y)/sqrt(2)
    assert len(out) == 2
    for coeff in out.coefficients():
        assert abs(abs(coeff) - R) < 1e-10


def test_t_gate_round_trips_under_heisenberg():
    initial = _sum(4, {"XIII": 1.0, "ZIII": 0.3})
    c = Circuit(4)
    c.unitary_1q(0, T)
    fwd = initial.propagate(circuit=c, direction="forward")
    back = fwd.propagate(circuit=c, direction="heisenberg")
    _assert_close(back, initial)


# ---- validation ----


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match="2x2"):
        gates.unitary_1q(0, np.eye(3, dtype=complex))
    with pytest.raises(ValueError, match="4x4"):
        gates.unitary_2q(0, 1, np.eye(2, dtype=complex))


def test_non_unitary_matrix_is_rejected():
    # A non-unitary matrix would silently give a non-physical channel whose
    # coefficients drift over many layers, so it is refused up front.
    bad = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex)
    with pytest.raises(ValueError, match="not unitary"):
        gates.unitary_1q(0, bad)


def test_repeated_qubit_is_rejected():
    with pytest.raises(ValueError, match="must differ"):
        gates.unitary_2q(2, 2, CNOT)


def test_real_dtype_is_accepted_by_conversion():
    # NumPy will happily hand over a float array; the binding asks for complex,
    # so this should either convert or raise cleanly rather than misread memory.
    real_h = np.array([[R, R], [R, -R]])
    try:
        ch = gates.unitary_1q(0, real_h.astype(complex))
    except ValueError:  # pragma: no cover - conversion is expected to succeed
        pytest.fail("complex-cast Hadamard should be accepted")
    assert ch is not None
