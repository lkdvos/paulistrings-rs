"""``paulistrings.interop`` — stim / qiskit importers and the task-JSON schema.

Design source: ``research/notes/2026-09-01-python-api-extensions.md`` §A5.
The stim and qiskit sections ``pytest.importorskip`` their backend so CI
(numpy-only) stays green; the task-JSON section has no optional dependency
and always runs.

Note CI does not run the stim/qiskit sections (see CLAUDE.md); run them
locally with ``maturin develop --release`` followed by
``pytest python/paulistrings/tests``.
"""

import json
import math

import pytest

from paulistrings import Circuit, PauliSum, gates, interop


TOL = 1e-10


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


def _full_dict(sum_, num_qubits):
    """{full_length_label: coeff}, decoding the symplectic key per qubit."""
    xs = sum_.x_array()
    zs = sum_.z_array()
    cs = sum_.coefficients_array()
    out = {}
    for i in range(len(sum_)):
        chars = []
        for q in range(num_qubits):
            word, bit = q // 64, 1 << (q % 64)
            xb = int(xs[i][word]) & bit
            zb = int(zs[i][word]) & bit
            if xb and zb:
                chars.append("Y")
            elif xb:
                chars.append("X")
            elif zb:
                chars.append("Z")
            else:
                chars.append("I")
        out["".join(chars)] = complex(cs[i])
    return out


# =============================================================================
# stim
# =============================================================================


def test_stim_hermitian_y_convention_matches_stim():
    # The known-by-hand conjugation cited in the module docstring and
    # research/notes/2026-08-31-python-test-triage.md: S X S^-1 = +Y in
    # stim's (Hermitian) convention. Propagating X forward through S in this
    # library must land on the same key with the same +1 sign, with no extra
    # phase to reconcile.
    stim = pytest.importorskip("stim")
    tab = stim.Tableau.from_named_gate("S")
    want = tab(stim.PauliString("+X"))
    assert str(want) == "+Y"

    circuit, observable = interop.circuit_from_stim("S 0")
    assert observable is None
    out = _sum(1, {"X": 1.0}).propagate(circuit=circuit, direction="forward")
    got = _full_dict(out, 1)
    assert got == {"Y": pytest.approx(1.0 + 0.0j)}


def test_stim_round_trip_matches_tableau_conjugation():
    stim = pytest.importorskip("stim")
    src = "H 0\nS 0\nCX 0 1"
    circuit, observable = interop.circuit_from_stim(src)
    assert observable is None
    tab = stim.Circuit(src).to_tableau()

    for label in ("ZI", "IX", "XI", "YZ", "ZY"):
        got = _full_dict(
            _sum(2, {label: 1.0}).propagate(circuit=circuit, direction="forward"), 2
        )
        want_p = tab(stim.PauliString("+" + label))
        want_key = "".join("IXYZ"[want_p[i]] for i in range(len(want_p)))
        want_sign = want_p.sign.real
        assert got == {want_key: pytest.approx(want_sign + 0.0j)}, label


def test_stim_repeat_block_is_expanded():
    pytest.importorskip("stim")
    src = """
    REPEAT 4 {
    H 0
    }
    """
    circuit, _observable = interop.circuit_from_stim(src)
    assert len(circuit) == 4


def test_stim_depolarize1_matches_hand_computed_factor():
    pytest.importorskip("stim")
    circuit, observable = interop.circuit_from_stim("DEPOLARIZE1(0.3) 0")
    assert observable is None
    # stim's DEPOLARIZE1(p) is a uniform-Pauli-p/3 channel; this library's
    # depolarize(p) dual is 1 - 4p/3 = 1 - 0.4 = 0.6.
    out = _sum(1, {"X": 1.0}).propagate(circuit=circuit, direction="forward")
    assert abs(out.coefficients()[0] - (0.6 + 0.0j)) < TOL


def test_stim_depolarize2_matches_the_binding():
    pytest.importorskip("stim")
    from paulistrings import noise

    circuit, _observable = interop.circuit_from_stim("DEPOLARIZE2(0.3) 0 1")
    via_binding = Circuit(2)
    via_binding.append(noise.depolarize2(0.3, 0, 1))
    initial = _sum(2, {"XZ": 1.0, "II": 2.0})
    _assert_close(
        initial.propagate(circuit=circuit, direction="forward"),
        initial.propagate(circuit=via_binding, direction="forward"),
    )


@pytest.mark.parametrize(
    "instr,slot",
    [("X_ERROR", 0), ("Y_ERROR", 1), ("Z_ERROR", 2)],
)
def test_stim_pauli_error_channels_hand_computed(instr, slot):
    pytest.importorskip("stim")
    p = 0.3
    circuit, observable = interop.circuit_from_stim(f"{instr}({p}) 0")
    assert observable is None
    # Heisenberg dual of a single-Pauli error channel: the matching Pauli is
    # left invariant (it commutes with its own error operator), the other
    # two anticommute and pick up (1 - 2p).
    letters = "XYZ"
    same = letters[slot]
    others = [c for c in letters if c != same]
    out_same = _sum(1, {same: 1.0}).propagate(circuit=circuit, direction="forward")
    assert abs(out_same.coefficients()[0] - 1.0) < TOL
    for other in others:
        out = _sum(1, {other: 1.0}).propagate(circuit=circuit, direction="forward")
        assert abs(out.coefficients()[0] - (1 - 2 * p)) < TOL


def test_stim_named_gates_broadcast_over_repeated_target():
    # "H 0 0 0" from a merged REPEAT block: three independent H pushes on the
    # same qubit, not one push with a length-3 support.
    pytest.importorskip("stim")
    circuit, _observable = interop.circuit_from_stim("H 0 0 0")
    assert len(circuit) == 3
    manual = Circuit(1)
    manual.h(0)
    manual.h(0)
    manual.h(0)
    initial = _sum(1, {"X": 1.0})
    _assert_close(
        initial.propagate(circuit=circuit, direction="forward"),
        initial.propagate(circuit=manual, direction="forward"),
    )


def test_stim_observable_include_builds_the_observable():
    pytest.importorskip("stim")
    circuit, observable = interop.circuit_from_stim(
        "H 0\nOBSERVABLE_INCLUDE(0) X0 Y1"
    )
    assert observable is not None
    assert _full_dict(observable, 2) == {"XY": pytest.approx(1.0 + 0.0j)}


def test_stim_multiple_observable_include_sum():
    pytest.importorskip("stim")
    circuit, observable = interop.circuit_from_stim(
        "OBSERVABLE_INCLUDE(0) X0\nOBSERVABLE_INCLUDE(1) Z1"
    )
    d = _full_dict(observable, 2)
    assert d == {"XI": pytest.approx(1.0 + 0.0j), "IZ": pytest.approx(1.0 + 0.0j)}


def test_stim_string_source_and_path_both_work(tmp_path):
    pytest.importorskip("stim")
    src = "H 0\nCX 0 1"
    from_text, _ = interop.circuit_from_stim(src)

    p = tmp_path / "prog.stim"
    p.write_text(src)
    from_path, _ = interop.circuit_from_stim(p)
    from_str_path, _ = interop.circuit_from_stim(str(p))

    initial = _sum(2, {"XI": 1.0, "IZ": 0.5})
    want = initial.propagate(circuit=from_text, direction="forward")
    _assert_close(initial.propagate(circuit=from_path, direction="forward"), want)
    _assert_close(initial.propagate(circuit=from_str_path, direction="forward"), want)


@pytest.mark.parametrize(
    "name,src",
    [
        ("M", "M 0"),
        ("MR", "MR 0"),
        ("R", "R 0"),
        ("DETECTOR", "DETECTOR(0,0) rec[-1]"),
        ("E", "CORRELATED_ERROR(0.1) X0 Y1"),
        ("PAULI_CHANNEL_1", "PAULI_CHANNEL_1(0.1,0.1,0.1) 0"),
        ("MPP", "MPP X0*X1"),
    ],
)
def test_stim_unsupported_instructions_hard_error_naming_the_instruction(name, src):
    pytest.importorskip("stim")
    with pytest.raises(ValueError, match=name):
        interop.circuit_from_stim(src)


def test_stim_observable_include_with_measurement_record_hard_errors():
    pytest.importorskip("stim")
    with pytest.raises(ValueError, match="measurement-record"):
        interop.circuit_from_stim("OBSERVABLE_INCLUDE(0) rec[-1]")


def test_stim_unknown_string_source_is_a_parse_error_not_a_silent_skip():
    stim = pytest.importorskip("stim")
    with pytest.raises(Exception):
        # Garbage stim source: neither a valid path nor valid stim syntax.
        interop.circuit_from_stim("NOT_A_REAL_INSTRUCTION 0 1 2")


def test_stim_wrong_type_raises_type_error():
    pytest.importorskip("stim")
    with pytest.raises(TypeError):
        interop.circuit_from_stim(12345)


# =============================================================================
# qiskit
# =============================================================================


def test_qiskit_named_gates_map_directly():
    qiskit = pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.s(1)
    qc.cx(0, 1)
    qc.cz(1, 2)
    qc.swap(0, 2)
    qc.rz(0.3, 1)

    imported = interop.circuit_from_qiskit(qc)

    manual = Circuit(3)
    manual.h(0)
    manual.s(1)
    manual.cnot(0, 1)
    manual.cz(1, 2)
    manual.swap(0, 2)
    manual.rz(0.3, 1)

    initial = _sum(3, {"XYZ": 1.0, "IZI": 0.5})
    _assert_close(
        initial.propagate(circuit=imported, direction="forward"),
        initial.propagate(circuit=manual, direction="forward"),
    )


def test_qiskit_rzz_matches_pauli_rotation():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    theta = 0.83
    qc = QuantumCircuit(2)
    qc.rzz(theta, 0, 1)
    imported = interop.circuit_from_qiskit(qc)

    native = Circuit(2)
    native.pauli_rotation("ZZ", [0, 1], theta)

    initial = _sum(2, {"XI": 1.0, "IY": 0.5})
    _assert_close(
        initial.propagate(circuit=imported, direction="forward"),
        initial.propagate(circuit=native, direction="forward"),
    )


@pytest.mark.parametrize("name,pauli", [("rxx", "XX"), ("ryy", "YY")])
def test_qiskit_rxx_ryy_match_pauli_rotation(name, pauli):
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    theta = -0.42
    qc = QuantumCircuit(2)
    getattr(qc, name)(theta, 0, 1)
    imported = interop.circuit_from_qiskit(qc)

    native = Circuit(2)
    native.pauli_rotation(pauli, [0, 1], theta)

    initial = _sum(2, {"ZI": 1.0, "IZ": 0.25})
    _assert_close(
        initial.propagate(circuit=imported, direction="forward"),
        initial.propagate(circuit=native, direction="forward"),
    )


def test_qiskit_t_gate_fallback_matches_general_unitary_conjugation():
    # T X T^dagger = (X + Y)/sqrt(2), the same identity
    # test_general_unitary.py::test_t_gate_maps_x_to_a_two_term_sum pins.
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.t(0)
    imported = interop.circuit_from_qiskit(qc)

    out = _sum(1, {"X": 1.0}).propagate(circuit=imported, direction="forward")
    assert len(out) == 2
    r = 1.0 / math.sqrt(2.0)
    for coeff in out.coefficients():
        assert abs(abs(coeff) - r) < TOL


def test_qiskit_sdg_maps_to_the_named_sdg_gate():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.sdg(0)
    imported = interop.circuit_from_qiskit(qc)
    # Named, not the unitary_1q fallback it used before `sdg` existed.
    assert imported.gates == [{"name": "sdg", "qubits": [0]}]

    forward_s = Circuit(1)
    forward_s.s(0)

    initial = _sum(1, {"X": 1.0})
    # S^dagger X S = -Y, i.e. running Sdg forward is the same as running S in
    # the heisenberg (adjoint) direction.
    want = initial.propagate(circuit=forward_s, direction="heisenberg")
    got = initial.propagate(circuit=imported, direction="forward")
    _assert_close(got, want)


def test_qiskit_2q_unitary_fallback_matches_named_cnot():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import CXGate, UnitaryGate

    # qiskit's own little-endian CX matrix (qargs[1] as the more-significant
    # tensor factor) -- deliberately *not* this library's own CNOT fixture,
    # since the point of this test is exercising the basis-convention
    # permutation the fallback path applies.
    cx_matrix = CXGate().to_matrix()
    qc = QuantumCircuit(2)
    qc.append(UnitaryGate(cx_matrix), [0, 1])
    imported = interop.circuit_from_qiskit(qc)

    native = Circuit(2)
    native.cnot(0, 1)

    initial = _sum(2, {"XI": 1.0, "IZ": 0.5, "YY": -0.25})
    _assert_close(
        initial.propagate(circuit=imported, direction="forward"),
        initial.propagate(circuit=native, direction="forward"),
    )


def test_qiskit_barrier_is_ignored():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.barrier()
    qc.cx(0, 1)
    imported = interop.circuit_from_qiskit(qc)
    assert len(imported) == 2


def test_qiskit_measurement_hard_errors():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    with pytest.raises(ValueError, match="measure"):
        interop.circuit_from_qiskit(qc)


def test_qiskit_reset_hard_errors():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.reset(0)
    with pytest.raises(ValueError, match="reset"):
        interop.circuit_from_qiskit(qc)


def test_qiskit_more_than_two_qubit_gate_hard_errors():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    with pytest.raises(ValueError, match="ccx"):
        interop.circuit_from_qiskit(qc)


def test_qiskit_classically_conditioned_instruction_hard_errors():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    op = qc.data[0].operation.to_mutable()
    op.condition = (qc.cregs[0], 1)
    qc.data[0] = qc.data[0].replace(operation=op)

    with pytest.raises(ValueError, match="conditioned"):
        interop.circuit_from_qiskit(qc)


# =============================================================================
# task-JSON schema v1 (no optional dependency)
# =============================================================================


def _small_task_dict():
    return {
        "version": 1,
        "n_qubits": 2,
        "circuit": {
            "gates": [
                {"name": "h", "qubits": [0]},
                {"name": "cnot", "qubits": [0, 1]},
                {"name": "rz", "qubits": [1], "theta": 0.4},
            ]
        },
        "observable": {"ZI": 1.0, "IZ": [0.5, 0.0]},
        "truncation": {"max_weight": 2, "min_abs_coeff": 1e-9},
        "run": {"direction": "forward", "threads": 1, "state": "z+"},
    }


def test_load_task_from_dict_builds_a_consistent_task():
    task = interop.load_task(_small_task_dict())
    assert task.n_qubits == 2
    assert task.direction == "forward"
    assert task.threads == 1
    assert task.state == "z+"
    assert task.truncation is not None
    assert task.observable is not None
    assert _full_dict(task.observable, 2) == {
        "ZI": pytest.approx(1.0 + 0.0j),
        "IZ": pytest.approx(0.5 + 0.0j),
    }

    manual = Circuit(2)
    manual.h(0)
    manual.cnot(0, 1)
    manual.rz(0.4, 1)
    initial = _sum(2, {"XI": 1.0, "IY": 0.3})
    _assert_close(
        initial.propagate(circuit=task.circuit, direction="forward"),
        initial.propagate(circuit=manual, direction="forward"),
    )


def test_load_task_from_file_round_trip_build_run_expectation(tmp_path):
    # A full round trip on a hand-computed 2-qubit case: H;CNOT on |00>
    # (state="z+") measuring Z0*Z1 (a Bell pair) gives expectation +1.
    task_dict = {
        "version": 1,
        "n_qubits": 2,
        "circuit": {
            "gates": [
                {"name": "h", "qubits": [0]},
                {"name": "cnot", "qubits": [0, 1]},
            ]
        },
        "observable": {"ZZ": 1.0},
        "run": {"direction": "heisenberg", "state": "z+"},
    }
    p = tmp_path / "task.json"
    p.write_text(json.dumps(task_dict))

    task = interop.load_task(p)
    assert task.threads == 1  # default
    assert task.truncation is None  # omitted

    evolved = task.observable.propagate(circuit=task.circuit, direction=task.direction)
    got = evolved.expectation(state=task.state)
    assert abs(got.real - 1.0) < 1e-10


def test_load_task_truncation_alias_truth_table():
    both = interop.load_task(_small_task_dict())
    assert both.truncation is not None

    d = _small_task_dict()
    del d["truncation"]["min_abs_coeff"]
    weight_only = interop.load_task(d)
    assert weight_only.truncation is not None

    d = _small_task_dict()
    del d["truncation"]
    neither = interop.load_task(d)
    assert neither.truncation is None


def test_load_task_unknown_top_level_key_hard_errors():
    d = _small_task_dict()
    d["bogus"] = 1
    with pytest.raises(ValueError, match="bogus"):
        interop.load_task(d)


@pytest.mark.parametrize("missing", ["version", "n_qubits", "circuit", "run"])
def test_load_task_missing_required_key_hard_errors(missing):
    d = _small_task_dict()
    del d[missing]
    with pytest.raises(ValueError, match=missing):
        interop.load_task(d)


def test_load_task_wrong_version_hard_errors():
    d = _small_task_dict()
    d["version"] = 2
    with pytest.raises(ValueError, match="version"):
        interop.load_task(d)


def test_load_task_direction_is_required_never_defaulted():
    d = _small_task_dict()
    del d["run"]["direction"]
    with pytest.raises(ValueError, match="direction"):
        interop.load_task(d)


def test_load_task_unknown_run_key_hard_errors():
    d = _small_task_dict()
    d["run"]["bogus"] = 1
    with pytest.raises(ValueError, match="bogus"):
        interop.load_task(d)


def test_load_task_invalid_direction_value_hard_errors():
    d = _small_task_dict()
    d["run"]["direction"] = "sideways"
    with pytest.raises(ValueError, match="sideways"):
        interop.load_task(d)


def test_load_task_unknown_gate_name_hard_errors():
    d = _small_task_dict()
    d["circuit"]["gates"].append({"name": "bogus_gate", "qubits": [0]})
    with pytest.raises(ValueError, match="bogus_gate"):
        interop.load_task(d)


def test_load_task_gate_missing_required_field_hard_errors():
    d = _small_task_dict()
    d["circuit"]["gates"].append({"name": "rz", "qubits": [0]})  # no "theta"
    with pytest.raises(ValueError, match="theta"):
        interop.load_task(d)


def test_load_task_pauli_rotation_and_channel_gates():
    d = {
        "version": 1,
        "n_qubits": 2,
        "circuit": {
            "gates": [
                {"name": "pauli_rotation", "pauli": "ZZ", "qubits": [0, 1], "theta": 0.5},
                {"name": "pauli_channel", "px": 0.1, "py": 0.0, "pz": 0.0, "qubits": [0]},
                {"name": "depolarize2", "p": 0.05, "qubits": [0, 1]},
            ]
        },
        "run": {"direction": "forward"},
    }
    task = interop.load_task(d)
    manual = Circuit(2)
    manual.pauli_rotation("ZZ", [0, 1], 0.5)
    manual.pauli_channel(0.1, 0.0, 0.0, [0])
    manual.depolarize2(0.05, [(0, 1)])
    initial = _sum(2, {"XI": 1.0, "IZ": 0.2})
    _assert_close(
        initial.propagate(circuit=task.circuit, direction="forward"),
        initial.propagate(circuit=manual, direction="forward"),
    )


def test_load_task_unitary_gates_from_nested_matrix():
    r = 1.0 / math.sqrt(2.0)
    h_matrix = [[[r, 0.0], [r, 0.0]], [[r, 0.0], [-r, 0.0]]]
    d = {
        "version": 1,
        "n_qubits": 1,
        "circuit": {"gates": [{"name": "unitary_1q", "qubits": [0], "matrix": h_matrix}]},
        "run": {"direction": "forward"},
    }
    task = interop.load_task(d)
    manual = Circuit(1)
    manual.h(0)
    initial = _sum(1, {"X": 1.0})
    _assert_close(
        initial.propagate(circuit=task.circuit, direction="forward"),
        initial.propagate(circuit=manual, direction="forward"),
    )


def test_circuit_from_json_matches_load_task_for_the_same_gates():
    d = _small_task_dict()
    from_load_task = interop.load_task(d).circuit
    from_direct = interop.circuit_from_json(d["circuit"], d["n_qubits"])
    initial = _sum(2, {"YI": 1.0})
    _assert_close(
        initial.propagate(circuit=from_load_task, direction="forward"),
        initial.propagate(circuit=from_direct, direction="forward"),
    )


def test_circuit_from_json_rejects_both_gates_and_stim_file():
    with pytest.raises(ValueError, match="gates.*stim_file|stim_file.*gates"):
        interop.circuit_from_json({"gates": [], "stim_file": "x.stim"}, 1)


def test_circuit_from_json_stim_file_round_trip(tmp_path):
    pytest.importorskip("stim")
    stim_path = tmp_path / "prog.stim"
    stim_path.write_text("H 0\nCX 0 1")

    task_dict = {
        "version": 1,
        "n_qubits": 2,
        "circuit": {"stim_file": "prog.stim"},
        "observable": {"ZI": 1.0},
        "run": {"direction": "forward"},
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task_dict))

    task = interop.load_task(task_path)
    manual = Circuit(2)
    manual.h(0)
    manual.cnot(0, 1)
    initial = _sum(2, {"XI": 1.0})
    _assert_close(
        initial.propagate(circuit=task.circuit, direction="forward"),
        initial.propagate(circuit=manual, direction="forward"),
    )


def test_load_task_stim_file_observable_include_is_used_when_json_omits_it(tmp_path):
    pytest.importorskip("stim")
    stim_path = tmp_path / "prog.stim"
    stim_path.write_text("H 0\nOBSERVABLE_INCLUDE(0) X0 X1")

    task_dict = {
        "version": 1,
        "n_qubits": 2,
        "circuit": {"stim_file": "prog.stim"},
        "run": {"direction": "forward"},
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task_dict))

    task = interop.load_task(task_path)
    assert task.observable is not None
    assert _full_dict(task.observable, 2) == {"XX": pytest.approx(1.0 + 0.0j)}


def test_load_task_n_qubits_mismatch_with_stim_file_hard_errors(tmp_path):
    pytest.importorskip("stim")
    stim_path = tmp_path / "prog.stim"
    stim_path.write_text("H 0\nCX 0 1")

    task_dict = {
        "version": 1,
        "n_qubits": 5,
        "circuit": {"stim_file": "prog.stim"},
        "run": {"direction": "forward"},
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task_dict))

    with pytest.raises(ValueError, match="n_qubits"):
        interop.load_task(task_path)
