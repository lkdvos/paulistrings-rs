"""``Circuit`` introspection, slicing/composition, ``adjoint()``, and ``sdg``.

Follow-ups named in PR #1. Four capabilities, one test section each:

1. ``Circuit.gates`` — the channel list as JSON-native dicts in task-JSON
   schema v1's gate vocabulary (``research/notes/2026-09-01-python-api-extensions.md``
   §A5), so ``Circuit(...)`` -> ``.gates`` -> task JSON -> ``Circuit`` closes.
2. ``sdg`` — the named ``S^dagger`` spelling, an addition to that vocabulary
   and the one gate ``adjoint()`` could not otherwise express.
3. ``Circuit[...]`` slicing plus ``extend``/``+``.
4. ``Circuit.adjoint()`` — the circuit whose *forward* application equals this
   one's Heisenberg application.
"""

import json
import math

import numpy as np
import pytest

from paulistrings import Circuit, PauliSum, gates, interop, noise


TOL = 1e-12

#: The T gate. Not self-adjoint, so it makes `adjoint()` do real work on the
#: `unitary_1q` path.
T_MATRIX = np.array(
    [[1.0, 0.0], [0.0, complex(math.cos(math.pi / 4), math.sin(math.pi / 4))]],
    dtype=complex,
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


def _assert_close(a, b, tol=TOL):
    da, db = _as_dict(a), _as_dict(b)
    assert set(da) == set(db), f"different keys:\n{sorted(da)}\nvs\n{sorted(db)}"
    for k in da:
        assert abs(da[k] - db[k]) < tol, f"{k}: {da[k]} vs {db[k]}"


def _haar_unitary(rng, n):
    """A Haar-random `n x n` unitary (QR with the phases fixed)."""
    z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(z)
    return q * (np.diagonal(r) / np.abs(np.diagonal(r)))


def _matrix_from_json(nested):
    return np.array([[complex(re, im) for re, im in row] for row in nested], dtype=complex)


def _unitary_circuit(n=4, seed=7):
    """A circuit exercising every unitary gate in the vocabulary."""
    rng = np.random.default_rng(seed)
    c = Circuit(n)
    c.h(0)
    c.s(1)
    c.sdg(2)
    c.x(3)
    c.y(0)
    c.z(1)
    c.cnot(0, 1)
    c.cz(1, 2)
    c.swap(2, 3)
    c.rz(0.37, 0)
    c.rx(-0.81, 1)
    c.ry(1.23, 2)
    c.pauli_rotation("XYZ", [3, 0, 1], 0.44)
    c.unitary_1q(2, T_MATRIX)
    c.unitary_2q(0, 3, _haar_unitary(rng, 4))
    c.append(gates.pauli_rotation("ZZ", [1, 2], -math.pi / 2))
    return c


# =============================================================================
# 1. Gate-list introspection
# =============================================================================


def test_gates_dicts_are_the_schema_v1_gate_vocabulary():
    """Hand-written expected dicts, field for field.

    This is the pinned wire shape: JSON-native scalars, `qubits` always a list
    of ints in the gate's own argument order, and a `matrix` as nested rows of
    `[re, im]` pairs (never a NumPy array, so `json.dumps` needs no encoder).
    """
    c = Circuit(3)
    c.h(0)
    c.s(1)
    c.sdg(2)
    c.x(0)
    c.y(1)
    c.z(2)
    c.cnot(2, 0)
    c.cz(1, 2)
    c.swap(0, 2)
    c.rz(0.25, 0)
    c.rx(0.5, 1)
    c.ry(-0.75, 2)
    c.pauli_rotation("XYZ", [2, 0, 1], 0.125)
    c.depolarize(0.1, [0, 1])
    c.dephase(0.2, [2])
    c.amplitude_damping(0.3, [0])
    c.pauli_channel(0.01, 0.02, 0.03, [1])
    c.depolarize2(0.05, [(0, 2)])

    assert c.gates == [
        {"name": "h", "qubits": [0]},
        {"name": "s", "qubits": [1]},
        {"name": "sdg", "qubits": [2]},
        {"name": "x", "qubits": [0]},
        {"name": "y", "qubits": [1]},
        {"name": "z", "qubits": [2]},
        # `cnot`'s qubits are [control, target] -- not sorted.
        {"name": "cnot", "qubits": [2, 0]},
        {"name": "cz", "qubits": [1, 2]},
        {"name": "swap", "qubits": [0, 2]},
        {"name": "rz", "qubits": [0], "theta": 0.25},
        {"name": "rx", "qubits": [1], "theta": 0.5},
        {"name": "ry", "qubits": [2], "theta": -0.75},
        # The compact generator form, in the order the qubits were given.
        {"name": "pauli_rotation", "qubits": [2, 0, 1], "pauli": "XYZ", "theta": 0.125},
        # One dict per channel: the broadcast `depolarize` is two channels.
        {"name": "depolarize", "qubits": [0], "p": 0.1},
        {"name": "depolarize", "qubits": [1], "p": 0.1},
        {"name": "dephase", "qubits": [2], "p": 0.2},
        {"name": "amplitude_damping", "qubits": [0], "gamma": 0.3},
        {"name": "pauli_channel", "qubits": [1], "px": 0.01, "py": 0.02, "pz": 0.03},
        {"name": "depolarize2", "qubits": [0, 2], "p": 0.05},
    ]
    assert len(c.gates) == len(c)


def test_gates_matrices_are_nested_re_im_pairs():
    rng = np.random.default_rng(3)
    u2 = _haar_unitary(rng, 4)
    c = Circuit(2)
    c.unitary_1q(1, T_MATRIX)
    c.unitary_2q(1, 0, u2)

    g1, g2 = c.gates
    assert g1["name"] == "unitary_1q" and g1["qubits"] == [1]
    assert g2["name"] == "unitary_2q" and g2["qubits"] == [1, 0]
    # Nested lists, not arrays: exactly the JSON form the schema specifies.
    assert isinstance(g1["matrix"], list)
    assert isinstance(g1["matrix"][0], list)
    assert g1["matrix"][0][0] == [1.0, 0.0]
    assert np.array_equal(_matrix_from_json(g1["matrix"]), T_MATRIX)
    assert np.array_equal(_matrix_from_json(g2["matrix"]), u2)


def test_gates_returns_a_fresh_list_of_fresh_dicts():
    c = Circuit(1)
    c.rz(0.5, 0)
    first = c.gates
    first[0]["theta"] = 99.0
    first.append({"name": "h", "qubits": [0]})
    assert c.gates == [{"name": "rz", "qubits": [0], "theta": 0.5}]


def test_gates_round_trip_through_task_json():
    """`Circuit` -> `.gates` -> JSON text -> `interop.circuit_from_json`.

    The comparison is an evolved sum rather than a gate list, so a lost angle,
    a transposed matrix, or a swapped qubit order fails here as well.
    """
    c = _unitary_circuit()
    document = {"gates": c.gates}
    reloaded = interop.circuit_from_json(
        json.loads(json.dumps(document)), c.num_qubits
    )
    assert reloaded.gates == c.gates

    observable = _sum(c.num_qubits, {"ZIIX": 1.0, "IXYI": 0.5j})
    _assert_close(
        observable.propagate(reloaded, None, direction="heisenberg"),
        observable.propagate(c, None, direction="heisenberg"),
    )


def test_gates_round_trip_of_a_noisy_circuit():
    c = Circuit(2)
    c.h(0)
    c.depolarize(0.1, [0])
    c.dephase(0.2, [1])
    c.amplitude_damping(0.3, [0])
    c.pauli_channel(0.01, 0.02, 0.03, [1])
    c.depolarize2(0.05, [(0, 1)])
    reloaded = interop.circuit_from_json({"gates": c.gates}, 2)
    observable = _sum(2, {"XZ": 1.0, "YY": 0.25})
    _assert_close(
        observable.propagate(reloaded, None, direction="forward"),
        observable.propagate(c, None, direction="forward"),
    )


def test_empty_circuit_has_no_gates():
    assert Circuit(2).gates == []


def test_noise_factories_still_reach_the_gate_list():
    """`noise.*` channels appended as objects introspect the same way."""
    c = Circuit(2)
    c.append(noise.depolarize(0.1, 0))
    c.append(noise.depolarize2(0.05, 0, 1))
    assert c.gates == [
        {"name": "depolarize", "qubits": [0], "p": 0.1},
        {"name": "depolarize2", "qubits": [0, 1], "p": 0.05},
    ]


# =============================================================================
# 2. sdg
# =============================================================================


def test_s_and_sdg_conjugate_x_with_opposite_signs():
    """Hand-derived: `S X S^dagger = +Y` and `S^dagger X S = -Y`.

    With `S = diag(1, i)`: `S X S^dagger = [[0, -i], [i, 0]] = +Y`, and
    `S^dagger X S = [[0, i], [-i, 0]] = -Y`. Forward propagation here is
    `P -> U P U^dagger` (interop.py's convention note), so forward `s` sends
    `X -> +Y` and forward `sdg` sends `X -> -Y`. `Y` is the symplectic key
    `(x=1, z=1)` with no phase factor (the Hermitian convention).
    """
    x = _sum(1, {"X": 1.0})
    y_key = ((1,), (1,))

    forward_s = Circuit(1)
    forward_s.s(0)
    assert _as_dict(x.propagate(forward_s, None, direction="forward")) == {
        y_key: 1.0 + 0j
    }

    forward_sdg = Circuit(1)
    forward_sdg.sdg(0)
    assert _as_dict(x.propagate(forward_sdg, None, direction="forward")) == {
        y_key: -1.0 + 0j
    }

    # And the Heisenberg direction swaps the two, since S^dagger is S's adjoint.
    assert _as_dict(x.propagate(forward_s, None, direction="heisenberg")) == {
        y_key: -1.0 + 0j
    }
    assert _as_dict(x.propagate(forward_sdg, None, direction="heisenberg")) == {
        y_key: 1.0 + 0j
    }


def test_s_then_sdg_is_the_identity():
    c = Circuit(1)
    c.s(0)
    c.sdg(0)
    initial = _sum(1, {"X": 1.0, "Y": 0.5, "Z": 0.25})
    _assert_close(initial.propagate(c, None, direction="forward"), initial)


def test_sdg_factory_and_method_agree():
    from_method = Circuit(2)
    from_method.sdg(1)
    from_factory = Circuit(2)
    from_factory.append(gates.sdg(1))
    assert from_method.gates == from_factory.gates == [{"name": "sdg", "qubits": [1]}]


def test_sdg_is_bounds_checked_like_every_other_gate():
    c = Circuit(2)
    with pytest.raises(ValueError, match="out of range"):
        c.sdg(2)
    with pytest.raises(ValueError, match="out of range"):
        c.append(gates.sdg(5))


def test_sdg_survives_the_task_json_round_trip():
    c = Circuit(1)
    c.sdg(0)
    reloaded = interop.circuit_from_json(json.loads(json.dumps({"gates": c.gates})), 1)
    x = _sum(1, {"X": 1.0})
    _assert_close(
        x.propagate(reloaded, None, direction="forward"),
        x.propagate(c, None, direction="forward"),
    )


# =============================================================================
# 3. Slicing and composition
# =============================================================================


def test_slice_returns_a_circuit_of_the_selected_channels():
    c = _unitary_circuit()
    whole = c.gates
    head = c[:5]
    tail = c[5:]
    assert isinstance(head, Circuit)
    assert head.num_qubits == c.num_qubits
    assert len(head) == 5 and len(tail) == len(c) - 5
    assert head.gates == whole[:5]
    assert tail.gates == whole[5:]
    assert c[:].gates == whole
    assert c[len(c) :].gates == []
    assert c[100:200].gates == []
    assert c[-2:].gates == whole[-2:]
    assert c[::2].gates == whole[::2]
    assert c[::-1].gates == whole[::-1]


def test_front_and_tail_slices_recompose_to_the_whole_circuit():
    """B5's front/tail split, which used to be rebuilt gate by gate."""
    c = _unitary_circuit()
    observable = _sum(c.num_qubits, {"ZIIZ": 1.0})
    whole = observable.propagate(c, None, direction="heisenberg")
    for k in (0, 1, 7, len(c)):
        # Heisenberg through the whole circuit == Heisenberg through the tail,
        # then Heisenberg through the front.
        via_split = observable.propagate(c[k:], None, direction="heisenberg")
        via_split = via_split.propagate(c[:k], None, direction="heisenberg")
        _assert_close(via_split, whole)
        # ... and the two halves concatenate back to the original.
        _assert_close(
            observable.propagate(c[:k] + c[k:], None, direction="heisenberg"), whole
        )


def test_integer_index_returns_the_channel():
    c = Circuit(2)
    c.h(0)
    c.cnot(0, 1)
    assert "Cnot" in repr(c[1])
    assert "H" in repr(c[0])
    assert "H" in repr(c[-2])
    # A Channel taken out of one circuit is appendable to another.
    other = Circuit(2)
    other.append(c[1])
    assert other.gates == [{"name": "cnot", "qubits": [0, 1]}]
    with pytest.raises(IndexError):
        _ = c[2]
    with pytest.raises(IndexError):
        _ = c[-3]


def test_extend_appends_in_place():
    a = Circuit(2)
    a.h(0)
    b = Circuit(2)
    b.cnot(0, 1)
    b.rz(0.3, 1)
    a.extend(b)
    assert a.gates == [
        {"name": "h", "qubits": [0]},
        {"name": "cnot", "qubits": [0, 1]},
        {"name": "rz", "qubits": [1], "theta": 0.3},
    ]
    assert len(b) == 2  # the argument is untouched


def test_add_leaves_both_operands_alone():
    a = Circuit(2)
    a.h(0)
    b = Circuit(2)
    b.cnot(0, 1)
    total = a + b
    assert total.gates == a.gates + b.gates
    assert len(a) == 1 and len(b) == 1
    # Self-concatenation is fine (the specs are cloned).
    assert (a + a).gates == a.gates + a.gates
    a.extend(a)
    assert len(a) == 2


def test_extend_and_add_require_matching_widths():
    a = Circuit(2)
    b = Circuit(3)
    with pytest.raises(ValueError, match="2-qubit circuit.*3-qubit circuit"):
        a.extend(b)
    with pytest.raises(ValueError, match="2-qubit circuit.*3-qubit circuit"):
        _ = a + b


# =============================================================================
# 4. adjoint()
# =============================================================================


def test_adjoint_forward_equals_heisenberg():
    """The pinned semantics of `adjoint()`, on every unitary gate at once."""
    c = _unitary_circuit()
    observable = _sum(c.num_qubits, {"XIZI": 1.0, "IYIZ": -0.5, "ZZZZ": 0.25j})
    _assert_close(
        observable.propagate(c.adjoint(), None, direction="forward"),
        observable.propagate(c, None, direction="heisenberg"),
    )
    # ... and symmetrically, since adjoint() is an involution.
    _assert_close(
        observable.propagate(c.adjoint(), None, direction="heisenberg"),
        observable.propagate(c, None, direction="forward"),
    )


def test_adjoint_reverses_the_gate_list_and_daggers_each_gate():
    c = _unitary_circuit()
    names = [g["name"] for g in c.gates]
    adj_names = [g["name"] for g in c.adjoint().gates]
    # Reversed order, with `s` and `sdg` swapped and nothing else renamed.
    swapped = {"s": "sdg", "sdg": "s"}
    assert adj_names == [swapped.get(n, n) for n in reversed(names)]

    by_name = {g["name"]: g for g in c.adjoint().gates}
    assert by_name["rz"]["theta"] == -0.37
    assert by_name["rx"]["theta"] == 0.81
    assert by_name["ry"]["theta"] == -1.23
    assert by_name["pauli_rotation"]["theta"] == -0.44
    assert by_name["pauli_rotation"]["pauli"] == "XYZ"
    assert by_name["pauli_rotation"]["qubits"] == [3, 0, 1]
    assert np.allclose(
        _matrix_from_json(by_name["unitary_1q"]["matrix"]), T_MATRIX.conj().T
    )


def test_adjoint_is_an_involution_on_the_gate_list():
    c = _unitary_circuit()
    assert c.adjoint().adjoint().gates == c.gates


def test_adjoint_of_a_slice_composes():
    c = _unitary_circuit()
    k = 6
    # (A B)^dagger = B^dagger A^dagger.
    assert (c[k:].adjoint() + c[:k].adjoint()).gates == c.adjoint().gates


@pytest.mark.parametrize(
    "push, name",
    [
        (lambda c: c.depolarize(0.1, [0]), "depolarize"),
        (lambda c: c.dephase(0.1, [0]), "dephase"),
        (lambda c: c.amplitude_damping(0.1, [0]), "amplitude_damping"),
        (lambda c: c.pauli_channel(0.01, 0.02, 0.03, [0]), "pauli_channel"),
        (lambda c: c.depolarize2(0.05, [(0, 1)]), "depolarize2"),
    ],
)
def test_adjoint_refuses_noise_channels_by_name(push, name):
    c = Circuit(2)
    c.h(0)
    push(c)
    with pytest.raises(ValueError, match=name):
        c.adjoint()


def test_adjoint_of_an_empty_circuit_is_empty():
    c = Circuit(3)
    assert c.adjoint().gates == []
    assert c.adjoint().num_qubits == 3
