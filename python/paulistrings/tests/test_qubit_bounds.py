"""Qubit-index validation at the Python boundary.

Every `Channel` factory is width- and circuit-agnostic by design, so a qubit
index can only be checked against a concrete width when the channel is appended
to a `Circuit`. These tests pin that check on `Circuit.append` and on every
convenience method, plus the construction-time distinctness check on the
two-qubit gates.

Before this existed an out-of-range index either silently misbehaved or panicked
inside the core, and `cnot(q, q)` built a nonsense channel; both are now clean
`ValueError`s.
"""

import math

import numpy as np
import pytest

from paulistrings import Circuit, PauliSum, gates, noise


H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)
CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


# ---- bounds on append ----


@pytest.mark.parametrize(
    "channel",
    [
        gates.h(2),
        gates.s(2),
        gates.x(2),
        gates.y(2),
        gates.z(2),
        gates.cnot(0, 2),
        gates.cnot(2, 0),
        gates.cz(1, 7),
        gates.swap(9, 0),
        gates.rz(0.1, 2),
        gates.rx(0.1, 2),
        gates.ry(0.1, 2),
        gates.pauli_rotation("ZZ", [0, 2], 0.1),
        noise.depolarize(0.1, 2),
        noise.dephase(0.1, 2),
        noise.amplitude_damping(0.1, 2),
    ],
)
def test_append_rejects_an_out_of_range_qubit(channel):
    c = Circuit(2)
    with pytest.raises(ValueError, match="out of range"):
        c.append(channel)
    assert len(c) == 0


def test_append_rejects_an_out_of_range_unitary():
    c = Circuit(2)
    with pytest.raises(ValueError, match="out of range"):
        c.append(gates.unitary_1q(5, H))
    with pytest.raises(ValueError, match="out of range"):
        c.append(gates.unitary_2q(0, 5, CNOT))
    assert len(c) == 0


def test_append_accepts_the_largest_valid_index():
    c = Circuit(2)
    c.append(gates.h(1))
    c.append(gates.cnot(1, 0))
    assert len(c) == 2


# ---- bounds on the convenience methods ----


def test_convenience_methods_reject_an_out_of_range_qubit():
    c = Circuit(2)
    for call in (
        lambda: c.h(2),
        lambda: c.s(2),
        lambda: c.x(2),
        lambda: c.y(2),
        lambda: c.z(2),
        lambda: c.cnot(0, 2),
        lambda: c.cz(2, 1),
        lambda: c.swap(0, 3),
        lambda: c.rz(0.1, 2),
        lambda: c.rx(0.1, 2),
        lambda: c.ry(0.1, 2),
        lambda: c.pauli_rotation("Z", [2], 0.1),
        lambda: c.depolarize(0.1, [0, 2]),
        lambda: c.dephase(0.1, [2]),
        lambda: c.amplitude_damping(0.1, [2]),
        lambda: c.unitary_1q(2, H),
        lambda: c.unitary_2q(0, 2, CNOT),
    ):
        with pytest.raises(ValueError, match="out of range"):
            call()


def test_a_broadcast_noise_method_stops_at_the_bad_index():
    # `depolarize(p, [0, 5])` pushes qubit 0, then rejects qubit 5. The partial
    # push is documented behavior, not silence: the error names the index.
    c = Circuit(2)
    with pytest.raises(ValueError, match="out of range"):
        c.depolarize(0.1, [0, 5])
    assert len(c) == 1


def test_the_error_names_the_offending_index_and_the_width():
    c = Circuit(3)
    with pytest.raises(ValueError, match=r"11.*3-qubit"):
        c.h(11)


# ---- distinctness on the two-qubit gates ----


@pytest.mark.parametrize("factory", [gates.cnot, gates.cz, gates.swap])
def test_two_qubit_gates_reject_a_repeated_qubit(factory):
    with pytest.raises(ValueError, match="must differ"):
        factory(2, 2)


def test_two_qubit_circuit_methods_reject_a_repeated_qubit():
    c = Circuit(4)
    with pytest.raises(ValueError, match="must differ"):
        c.cnot(1, 1)
    with pytest.raises(ValueError, match="must differ"):
        c.cz(1, 1)
    with pytest.raises(ValueError, match="must differ"):
        c.swap(3, 3)
    assert len(c) == 0


# ---- the checks do not disturb valid circuits ----


def test_a_valid_circuit_still_propagates():
    s = PauliSum.from_strings({"ZI": 1.0}, num_qubits=2)
    c = Circuit(2)
    c.h(0)
    c.cnot(0, 1)
    out = s.propagate(circuit=c, direction="forward")
    assert len(out) == 1
