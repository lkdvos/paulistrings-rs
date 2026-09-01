"""Circuit/task importers: ``stim``, ``qiskit``, and the frozen task-JSON schema.

Design source: ``research/notes/2026-09-01-python-api-extensions.md`` §A5. This
module is shipped API (consumed by the examples & benchmarks suite and by
``benchmarks/julia/runner.jl``'s task-JSON counterpart), not example code, so
its behavior — in particular the "never skip an unsupported instruction"
rule and the frozen task-JSON schema below — is load-bearing.

Conventions
-----------
Every Pauli string here follows the core's Hermitian convention (CLAUDE.md
§Known gaps): a coefficient multiplies the literal Hermitian Pauli string,
and ``Y`` is the symplectic key ``(x=1, z=1)`` with no phase factor. This
happens to line up exactly with stim's own Pauli convention, which is also
Hermitian (stim's ``Y = [[0, -i], [i, 0]]``, not a phased "canonical" Y).
Concretely, ``S X S^-1 = +Y`` in stim's tableau formalism
(``stim.Tableau.from_named_gate("S")(stim.PauliString("+X")) == stim.PauliString("+Y")``),
and propagating ``X`` through ``Circuit.s`` in this library forward gives the
term at key ``(x=1, z=1)`` with coefficient ``+1`` — the same result, with no
extra phase to reconcile. This is the resolved outcome of the phase-convention
conflict recorded in
``research/notes/2026-08-31-python-test-triage.md``: both sides now agree
because the *parser's* Y-phase folding (not stim's convention, which was
always Hermitian) was the thing that got fixed.

``circuit_from_stim`` and ``circuit_from_qiskit`` both lazily import their
respective optional dependency inside the function body, so
``import paulistrings`` and this module's own import never require ``stim``
or ``qiskit`` to be installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._paulistrings import Circuit, PauliSum
from . import truncation as _truncation_mod

__all__ = [
    "circuit_from_stim",
    "circuit_from_qiskit",
    "circuit_from_json",
    "load_task",
    "Task",
]


# =============================================================================
# stim importer
# =============================================================================

# Named single-qubit unitaries: stim's canonical name is already the
# lower-cased Circuit method name.
_STIM_1Q_NAMED = ("H", "S", "X", "Y", "Z")

# Named two-qubit unitaries: stim canonical name -> Circuit method name.
# stim canonicalizes "CNOT" to "CX" on parse, but both spellings are accepted
# defensively (research note §A5's mapping table lists "CX/CNOT").
_STIM_2Q_NAMED = {"CX": "cnot", "CNOT": "cnot", "CZ": "cz", "SWAP": "swap"}

# Annotations that carry no operational meaning for propagation.
_STIM_ANNOTATIONS = ("TICK", "QUBIT_COORDS", "SHIFT_COORDS")

_STIM_ERROR_SLOT = {"X_ERROR": 0, "Y_ERROR": 1, "Z_ERROR": 2}


def _load_stim_circuit(src):
    import stim

    if isinstance(src, stim.Circuit):
        return src
    if isinstance(src, os.PathLike):
        return stim.Circuit.from_file(str(src))
    if isinstance(src, str):
        # A string is either a filesystem path or raw stim program text. If a
        # file exists at that path, prefer reading it; otherwise parse the
        # string itself as stim source.
        path = Path(src)
        if path.exists():
            return stim.Circuit.from_file(str(path))
        return stim.Circuit(src)
    raise TypeError(
        "circuit_from_stim expects a stim.Circuit, a path (str/os.PathLike), "
        f"or stim program text; got {type(src).__name__}"
    )


def _stim_plain_qubits(name, index, targets):
    """Extract plain qubit-target indices, hard-erroring on anything else.

    A sweep-bit, measurement-record, or combined target on an otherwise
    supported instruction is exactly the kind of thing this importer must
    never skip silently.
    """
    qubits = []
    for t in targets:
        if not t.is_qubit_target:
            raise ValueError(
                f"circuit_from_stim: {name!r} has a sweep-bit, measurement-record, "
                f"or combined target, which is not supported (instruction #{index})"
            )
        qubits.append(t.value)
    return qubits


def _stim_require_even(name, index, qubits):
    if len(qubits) % 2 != 0:
        raise ValueError(
            f"circuit_from_stim: {name!r} has an odd number of qubit targets "
            f"(instruction #{index})"
        )


def circuit_from_stim(src):
    """Import a stim circuit as a ``(Circuit, PauliSum | None)`` pair.

    ``src`` is a ``stim.Circuit``, a path to a ``.stim`` file (``str`` or
    ``os.PathLike``), or stim program text (``str``). ``REPEAT`` blocks are
    expanded via ``stim.Circuit.flattened()`` before translation, so they
    reach the instruction loop below already unrolled.

    Supported instructions: ``H``, ``S``, ``X``, ``Y``, ``Z``, ``CX``/``CNOT``,
    ``CZ``, ``SWAP``, ``DEPOLARIZE1(p)``, ``DEPOLARIZE2(p)``,
    ``X_ERROR``/``Y_ERROR``/``Z_ERROR(p)``, ``I`` (skipped), the annotations
    ``TICK``/``QUBIT_COORDS``/``SHIFT_COORDS`` (skipped, not operations), and
    ``OBSERVABLE_INCLUDE`` with Pauli targets (surfaced as the returned
    observable). Anything else — measurements/resets (``M``, ``MR``, ``R``,
    ``MPP``), ``DETECTOR``, ``OBSERVABLE_INCLUDE`` with measurement-record
    targets, ``CORRELATED_ERROR``/``ELSE_CORRELATED_ERROR``,
    ``PAULI_CHANNEL_1``/``PAULI_CHANNEL_2`` (not yet mapped), sweep/combined
    targets, or any unlisted instruction — is a hard ``ValueError`` naming the
    instruction; nothing is ever skipped silently.

    Multiple ``OBSERVABLE_INCLUDE`` instructions (any index) each contribute
    one term — the product of their Pauli targets, coefficient 1.0 — to the
    returned observable; duplicate keys sum (``PauliSum.from_strings``
    semantics). Returns ``(circuit, None)`` if the circuit carries no
    ``OBSERVABLE_INCLUDE``.
    """
    raw = _load_stim_circuit(src)
    flat = raw.flattened()
    n_qubits = flat.num_qubits
    circuit = Circuit(n_qubits)
    observable_terms: dict[str, complex] = {}

    for index, instr in enumerate(flat):
        name = instr.name
        targets = instr.targets_copy()
        args = instr.gate_args_copy()

        if name == "I" or name in _STIM_ANNOTATIONS:
            continue

        if name in _STIM_1Q_NAMED:
            method = getattr(circuit, name.lower())
            for q in _stim_plain_qubits(name, index, targets):
                method(q)
            continue

        if name in _STIM_2Q_NAMED:
            method = getattr(circuit, _STIM_2Q_NAMED[name])
            qs = _stim_plain_qubits(name, index, targets)
            _stim_require_even(name, index, qs)
            for a, b in zip(qs[0::2], qs[1::2]):
                method(a, b)
            continue

        if name == "DEPOLARIZE1":
            (p,) = args
            for q in _stim_plain_qubits(name, index, targets):
                circuit.depolarize(p, [q])
            continue

        if name == "DEPOLARIZE2":
            (p,) = args
            qs = _stim_plain_qubits(name, index, targets)
            _stim_require_even(name, index, qs)
            for a, b in zip(qs[0::2], qs[1::2]):
                circuit.depolarize2(p, [(a, b)])
            continue

        if name in _STIM_ERROR_SLOT:
            (p,) = args
            probs = [0.0, 0.0, 0.0]
            probs[_STIM_ERROR_SLOT[name]] = p
            for q in _stim_plain_qubits(name, index, targets):
                circuit.pauli_channel(probs[0], probs[1], probs[2], [q])
            continue

        if name == "OBSERVABLE_INCLUDE":
            chars = ["I"] * n_qubits
            for t in targets:
                if t.is_x_target:
                    chars[t.value] = "X"
                elif t.is_y_target:
                    chars[t.value] = "Y"
                elif t.is_z_target:
                    chars[t.value] = "Z"
                elif t.is_measurement_record_target:
                    raise ValueError(
                        "circuit_from_stim: OBSERVABLE_INCLUDE with a "
                        f"measurement-record target is not supported (instruction #{index})"
                    )
                else:
                    raise ValueError(
                        "circuit_from_stim: OBSERVABLE_INCLUDE with an unsupported "
                        f"target is not supported (instruction #{index})"
                    )
            key = "".join(chars)
            observable_terms[key] = observable_terms.get(key, 0.0) + 1.0
            continue

        raise ValueError(
            f"circuit_from_stim: unsupported stim instruction {name!r} (instruction #{index})"
        )

    observable = (
        PauliSum.from_strings(observable_terms, num_qubits=n_qubits)
        if observable_terms
        else None
    )
    return circuit, observable


# =============================================================================
# qiskit importer
# =============================================================================

_QISKIT_1Q_NAMED = ("h", "s", "sdg", "x", "y", "z")
_QISKIT_1Q_ROT = ("rz", "rx", "ry")
_QISKIT_2Q_NAMED = {"cx": "cnot", "cz": "cz", "swap": "swap"}
_QISKIT_2Q_ROT = {"rzz": "ZZ", "rxx": "XX", "ryy": "YY"}

# Reversing which of the two qubits is the more-significant tensor factor.
# qiskit's `Operator(gate).data` is written in its native little-endian
# convention: for a gate applied to qargs = (a, b), the matrix's basis index
# has `b` (the *second* argument) as the more significant bit and `a` as the
# less significant bit. This library's `unitary_2q(q0, q1, matrix)` takes the
# opposite convention — `q0` (the *first* argument) is the more significant
# tensor factor (docstring in `gates.rs`: "matrix acts on |q0 q1>"). Swapping
# the two middle basis indices (1 <-> 2) converts one convention to the
# other; verified against qiskit's `CXGate` against this library's own CNOT
# matrix fixture (`test_general_unitary.py`) — both bases have `qargs[0]`
# as the CNOT control, "q0" in this library's `unitary_2q(0, 1, CNOT)`, and
# the permuted matrices agree exactly.
_TWO_Q_BASIS_SWAP = (0, 2, 1, 3)


def _permute_2q_matrix(matrix):
    m = np.asarray(matrix, dtype=complex)
    return m[np.ix_(_TWO_Q_BASIS_SWAP, _TWO_Q_BASIS_SWAP)]


def _qiskit_operator_matrix(op):
    from qiskit.quantum_info import Operator

    try:
        return Operator(op).data
    except Exception as exc:  # pragma: no cover - depends on the failing gate
        raise ValueError(
            f"circuit_from_qiskit: cannot convert instruction {op.name!r} to a "
            f"unitary matrix: {exc}"
        ) from exc


def circuit_from_qiskit(qc):
    """Import a ``qiskit.QuantumCircuit`` as a paulistrings ``Circuit``.

    Named mapping where exact: ``h s sdg x y z cx cz swap rz rx ry``, plus
    ``rzz/rxx/ryy`` (mapped to ``pauli_rotation("ZZ"/"XX"/"YY", ...)``) and
    ``t/tdg`` (mapped through the checked ``unitary_1q`` fallback, since
    they have no direct spelling here). Any other 1- or 2-qubit gate that
    exposes a unitary via ``qiskit.quantum_info.Operator`` falls back to
    ``unitary_1q``/``unitary_2q`` — the binding's own unitarity check is the
    safety net for that path. ``barrier`` is ignored (not an operation).
    Measurements, resets, classically conditioned instructions, and any gate
    on more than two qubits are hard ``ValueError``s.

    Qubit index = ``qc.find_bit(q).index`` (handles multi-register circuits
    correctly, unlike assuming register-local indices).
    """
    n_qubits = qc.num_qubits
    circuit = Circuit(n_qubits)

    for index, instruction in enumerate(qc.data):
        op = instruction.operation
        name = op.name

        if name == "barrier":
            continue

        if getattr(op, "condition", None) is not None:
            raise ValueError(
                f"circuit_from_qiskit: classically conditioned instruction {name!r} "
                f"at index {index} is not supported"
            )

        if name in ("measure", "reset"):
            raise ValueError(
                f"circuit_from_qiskit: unsupported instruction {name!r} at index "
                f"{index} (measurement/reset is not supported)"
            )

        qubits = [qc.find_bit(q).index for q in instruction.qubits]

        if len(qubits) > 2:
            raise ValueError(
                f"circuit_from_qiskit: unsupported {len(qubits)}-qubit instruction "
                f"{name!r} at index {index} (only 1- and 2-qubit gates are supported)"
            )

        if name in _QISKIT_1Q_NAMED:
            getattr(circuit, name)(qubits[0])
        elif name in _QISKIT_1Q_ROT:
            theta = op.params[0]
            getattr(circuit, name)(theta, qubits[0])
        elif name in _QISKIT_2Q_NAMED:
            getattr(circuit, _QISKIT_2Q_NAMED[name])(qubits[0], qubits[1])
        elif name in _QISKIT_2Q_ROT:
            theta = op.params[0]
            circuit.pauli_rotation(_QISKIT_2Q_ROT[name], qubits, theta)
        elif len(qubits) == 1:
            matrix = _qiskit_operator_matrix(op)
            circuit.unitary_1q(qubits[0], matrix)
        else:
            matrix = _permute_2q_matrix(_qiskit_operator_matrix(op))
            circuit.unitary_2q(qubits[0], qubits[1], matrix)

    return circuit


# =============================================================================
# Task-JSON schema v1
# =============================================================================


@dataclass(frozen=True)
class Task:
    """A parsed task-JSON (schema v1) job.

    ``raw`` is the exact parsed JSON dict, kept for provenance echoing (e.g.
    writing it back out alongside results).
    """

    n_qubits: int
    circuit: Circuit
    observable: PauliSum | None
    truncation: Any | None
    direction: str
    threads: int
    state: str | None
    raw: dict = field(default_factory=dict)


_TASK_TOP_KEYS = {"version", "n_qubits", "circuit", "observable", "truncation", "run"}
_TASK_REQUIRED_KEYS = {"version", "n_qubits", "circuit", "run"}
_RUN_KEYS = {"direction", "threads", "state"}
_TRUNCATION_KEYS = {"max_weight", "min_abs_coeff"}

# `sdg` joined this table with `Circuit.adjoint()`, which needs a named
# spelling for `S^dagger`; it is an *addition* to schema v1's gate vocabulary
# (a reader that predates it hard-errors on the name, which is the schema's
# documented behavior for an unknown gate — `benchmarks/julia/runner.jl` does
# not implement it).
_TASK_1Q = ("h", "s", "sdg", "x", "y", "z")
_TASK_1Q_ROT = ("rz", "rx", "ry")
_TASK_2Q_NAMED = ("cz", "swap")
_TASK_1Q_NOISE_P = ("depolarize", "dephase")


def _matrix_from_json(nested):
    """A JSON-native nested-list matrix (rows of ``[re, im]`` pairs) as a
    complex128 NumPy array."""
    return np.array(
        [[complex(re, im) for re, im in row] for row in nested],
        dtype=complex,
    )


def _push_task_gate(circuit, gate):
    name = gate["name"]
    if name in _TASK_1Q:
        (q,) = gate["qubits"]
        getattr(circuit, name)(q)
    elif name == "cnot":
        control, target = gate["qubits"]
        circuit.cnot(control, target)
    elif name in _TASK_2Q_NAMED:
        q0, q1 = gate["qubits"]
        getattr(circuit, name)(q0, q1)
    elif name in _TASK_1Q_ROT:
        (q,) = gate["qubits"]
        getattr(circuit, name)(gate["theta"], q)
    elif name == "pauli_rotation":
        circuit.pauli_rotation(gate["pauli"], gate["qubits"], gate["theta"])
    elif name in _TASK_1Q_NOISE_P:
        getattr(circuit, name)(gate["p"], gate["qubits"])
    elif name == "amplitude_damping":
        circuit.amplitude_damping(gate["gamma"], gate["qubits"])
    elif name == "pauli_channel":
        circuit.pauli_channel(gate["px"], gate["py"], gate["pz"], gate["qubits"])
    elif name == "depolarize2":
        q0, q1 = gate["qubits"]
        circuit.depolarize2(gate["p"], [(q0, q1)])
    elif name == "unitary_1q":
        (q,) = gate["qubits"]
        circuit.unitary_1q(q, _matrix_from_json(gate["matrix"]))
    elif name == "unitary_2q":
        q0, q1 = gate["qubits"]
        circuit.unitary_2q(q0, q1, _matrix_from_json(gate["matrix"]))
    else:
        raise ValueError(f"unknown task gate name {name!r}")


def _circuit_from_gate_list(gate_list, n_qubits):
    circuit = Circuit(n_qubits)
    for i, gate in enumerate(gate_list):
        if not isinstance(gate, dict) or "name" not in gate:
            raise ValueError(f'task gate #{i} must be an object with a "name" key')
        name = gate["name"]
        try:
            _push_task_gate(circuit, gate)
        except KeyError as exc:
            raise ValueError(
                f"task gate #{i} ({name!r}) is missing required field {exc}"
            ) from exc
    return circuit


def circuit_from_json(obj: dict, n_qubits: int, *, base_dir=None) -> Circuit:
    """Build a ``Circuit`` from the task-JSON schema v1 ``"circuit"`` object.

    ``obj`` is ``{"gates": [...]}`` (a list of gate objects, in application
    order — see the schema table in the module docstring of this file's
    design note) or ``{"stim_file": "relative/path.stim"}``. ``base_dir``
    resolves a relative ``stim_file`` path (``load_task`` passes the task
    file's own directory); it defaults to the current working directory.

    A ``stim_file`` circuit's ``OBSERVABLE_INCLUDE`` (if any) is *not*
    returned here — this function's return type is ``Circuit`` only, matching
    the schema's "circuit object" scope. Callers that need that observable
    (``load_task`` does) should get it from ``circuit_from_stim`` directly.
    """
    if not isinstance(obj, dict):
        raise ValueError("circuit_from_json: circuit object must be a JSON object")
    has_gates = "gates" in obj
    has_stim = "stim_file" in obj
    if has_gates and has_stim:
        raise ValueError(
            'circuit_from_json: circuit object must have exactly one of "gates" '
            'or "stim_file", not both'
        )
    if has_gates:
        return _circuit_from_gate_list(obj["gates"], n_qubits)
    if has_stim:
        base = Path(base_dir) if base_dir is not None else Path(".")
        circuit, _observable = circuit_from_stim(base / obj["stim_file"])
        if circuit.num_qubits != n_qubits:
            raise ValueError(
                f"circuit_from_json: n_qubits ({n_qubits}) does not match the stim "
                f"file's qubit count ({circuit.num_qubits})"
            )
        return circuit
    raise ValueError(
        'circuit_from_json: circuit object must have a "gates" or "stim_file" key'
    )


def _parse_task_coeff(value):
    if isinstance(value, bool):
        raise ValueError(
            f"observable coefficient must be a number or [re, im] pair, got {value!r}"
        )
    if isinstance(value, (int, float)):
        return complex(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        re, im = value
        return complex(re, im)
    raise ValueError(
        f"observable coefficient must be a number or [re, im] pair, got {value!r}"
    )


def _observable_from_task_dict(d, n_qubits):
    terms = {k: _parse_task_coeff(v) for k, v in d.items()}
    return PauliSum.from_strings(terms, num_qubits=n_qubits)


def _truncation_from_task_dict(t):
    """`weight(w) & coeff(eps)` when both knobs are present, the single one
    when one is, `None` when neither — mirrors the harness's `make_policy`
    alias (A7) without depending on `examples/`, since this module is shipped
    API and A7 is example-only code."""
    max_weight = t.get("max_weight")
    min_abs_coeff = t.get("min_abs_coeff")
    policy = None
    if max_weight is not None:
        policy = _truncation_mod.weight(max_weight)
    if min_abs_coeff is not None:
        c = _truncation_mod.coeff(min_abs_coeff)
        policy = c if policy is None else (policy & c)
    return policy


def load_task(path) -> Task:
    """Load a task-JSON (schema v1) file, or an already-parsed dict, as a `Task`.

    This is the frozen interchange format shared with
    ``benchmarks/julia/runner.jl`` — see the schema table and gate-object
    vocabulary in ``research/notes/2026-09-01-python-api-extensions.md`` §A5.
    Unknown top-level keys, unknown gate names, and missing required keys are
    hard errors; ``run.direction`` is required with no default (the stale
    README "Heisenberg by default" trap this schema deliberately avoids).
    """
    if isinstance(path, dict):
        raw = path
        base_dir = Path(".")
    else:
        p = Path(path)
        raw = json.loads(p.read_text())
        base_dir = p.parent

    if not isinstance(raw, dict):
        raise ValueError("task JSON: top level must be a JSON object")

    unknown = set(raw) - _TASK_TOP_KEYS
    if unknown:
        raise ValueError(f"task JSON: unknown top-level key(s) {sorted(unknown)}")
    missing = _TASK_REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(f"task JSON: missing required key(s) {sorted(missing)}")
    if raw["version"] != 1:
        raise ValueError(
            f"task JSON: unsupported schema version {raw['version']!r} "
            "(only version 1 is implemented)"
        )

    n_qubits = raw["n_qubits"]
    circuit_obj = raw["circuit"]
    if not isinstance(circuit_obj, dict):
        raise ValueError('task JSON: "circuit" must be a JSON object')

    observable = None
    if "stim_file" in circuit_obj:
        stim_path = base_dir / circuit_obj["stim_file"]
        circuit, observable = circuit_from_stim(stim_path)
        if circuit.num_qubits != n_qubits:
            raise ValueError(
                f"task JSON: n_qubits ({n_qubits}) does not match the stim file's "
                f"qubit count ({circuit.num_qubits})"
            )
    else:
        circuit = circuit_from_json(circuit_obj, n_qubits, base_dir=base_dir)

    if "observable" in raw:
        observable = _observable_from_task_dict(raw["observable"], n_qubits)

    truncation = None
    if "truncation" in raw:
        t = raw["truncation"]
        if not isinstance(t, dict):
            raise ValueError('task JSON: "truncation" must be a JSON object')
        unknown_t = set(t) - _TRUNCATION_KEYS
        if unknown_t:
            raise ValueError(f"task JSON: unknown truncation key(s) {sorted(unknown_t)}")
        truncation = _truncation_from_task_dict(t)

    run = raw["run"]
    if not isinstance(run, dict):
        raise ValueError('task JSON: "run" must be a JSON object')
    unknown_run = set(run) - _RUN_KEYS
    if unknown_run:
        raise ValueError(f"task JSON: unknown run key(s) {sorted(unknown_run)}")
    if "direction" not in run:
        raise ValueError("task JSON: run.direction is required (no default)")
    direction = run["direction"]
    if direction not in ("forward", "heisenberg"):
        raise ValueError(
            f"task JSON: run.direction must be \"forward\" or \"heisenberg\", got {direction!r}"
        )
    threads = run.get("threads", 1)
    state = run.get("state")

    return Task(
        n_qubits=n_qubits,
        circuit=circuit,
        observable=observable,
        truncation=truncation,
        direction=direction,
        threads=threads,
        state=state,
        raw=raw,
    )
