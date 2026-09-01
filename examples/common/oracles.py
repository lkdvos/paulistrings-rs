"""Ground-truth oracles for the `examples/` benchmark & showcase suite.

Handoff item P0c; see `research/plans/2026-08-31-examples-benchmarks-suite.md`
§6 Part 0.3, and global rule 1 in §7: *"every numeric claim is computed by an
oracle or loaded from a provenance-tagged reference file. Only Clifford-point
integers and oracle outputs may be asserted directly."* This module is the
"computed by an oracle" half of that rule.

Four oracles, plus one optional:

`statevector_expectation`
    Dense reference for `n <= ~26-28`, run by **qiskit Aer** (no simulator is
    hand-rolled here). Unitary circuits only.
`stim_clifford_exact`
    Exact reference at any `n` for circuits that are Clifford, run by stim's
    tableau simulator. Accepts a `.stim` file, so one checked-in file drives
    both the engine and the oracle (plan §7 rule 6).
`light_cone_exact`
    Exact reference for shallow circuits at any `n`, by backward causal-cone
    reduction followed by an exact evaluation *of the reduced problem* --
    statevector when the cone is small enough, otherwise untruncated Pauli
    propagation. The cone is computed from the gate list, never tabulated, and
    by default is commutation-aware (see `light_cone`), which is what makes it
    tight enough to be useful on the heavy-hex kicked Ising.
`load_published_reference`
    Loader for provenance-tagged reference files under
    `examples/data/references/`. Refuses any file whose header does not record
    `source`, `method` and `accuracy`.
`tsim_low_magic_exact`
    Optional low-magic oracle, behind a capability check: raises `SkipOracle`
    when `tsim` is not installed. Nothing in the suite depends on it.

How a circuit reaches an oracle
-------------------------------
Every oracle here consumes a `CircuitSpec` -- a plain gate list in the
*already frozen* task-JSON gate-object vocabulary
(`python/paulistrings/interop.py`, schema v1), which is also what
`benchmarks/julia/runner.jl` consumes. `CircuitSpec.to_circuit()` builds the
engine's `Circuit` from that same list through `interop.circuit_from_json`, so
engine and oracle are driven from one description rather than two
transcriptions of it.

`as_circuit_spec` accepts a `paulistrings.Circuit` **directly**: `Circuit.gates`
returns its channel list in exactly that vocabulary, so a builder that writes
into a real `Circuit` (everything in `examples/common/circuits.py` does) is read
back with no stand-in, no shimmed factories and no rebinding of module globals.
`record_gates(builder, *args)` is the thin convenience around that -- it runs
the builder and coerces the result -- and `RecordingCircuit` survives only as a
deprecated wrapper for call sites that still name it.

Conventions, and the traps between them
---------------------------------------
Three conversions in this module are easy to get silently wrong; each is pinned
by a test in `python/paulistrings/tests/test_examples_oracles.py`.

1. **Qubit-label endianness.** A paulistrings / stim Pauli label has character
   `i` on qubit `i`. A qiskit `SparsePauliOp` / `Pauli` label is MSB-first:
   character `-1-i` is on qubit `i`. So the qiskit conversion **reverses the
   label** (`_to_qiskit_label`) and the stim conversion does not. Qubit
   *indices* are never renumbered by any of this -- only labels are.
   The same asymmetry applies to two-qubit unitaries: this library's
   `unitary_2q(q0, q1, m)` has `q0` as the more significant tensor factor,
   while `QuantumCircuit.unitary(m, qargs)` has `qargs[0]` as the least
   significant, so the emitted qargs are `[q1, q0]`.
2. **The Hermitian-Y convention.** Here `Y` is the symplectic key `(x=1, z=1)`
   with no phase factor, i.e. the literal Hermitian `Y = [[0, -i], [i, 0]]`
   (CLAUDE.md §Known gaps). qiskit's `Pauli("Y")` and stim's `Y` are the *same*
   Hermitian matrix, so the conversion carries **no phase factor** in either
   direction -- reversal for qiskit, identity for stim, and that is all. The
   tests assert the matrices, not the claim.
3. **The rotation convention.** `pauli_rotation(pauli, qubits, theta)` is
   `U = exp(-i·theta·P/2)` with `pauli[k]` acting on `qubits[k]`. In qiskit that
   is `PauliEvolutionGate(SparsePauliOp(pauli[::-1]), time=theta/2)` -- note
   both the reversed label and the halved time. In stim it is a Clifford only
   when `theta` is a multiple of `pi/2`, and then it is built as a tableau from
   its own conjugation action (`_clifford_rotation_tableau`), which reproduces
   `SQRT_ZZ` / `SQRT_X` / `S` / ... exactly and also covers the mixed and
   higher-weight generators that have no named stim gate.

Initial states
--------------
All three computed oracles evaluate `<psi| U^dagger O U |psi>` -- the
expectation of `O` in the state prepared by running the circuit forward on
`|psi>`. The equivalent Pauli-propagation call is
`observable.propagate(circuit, policy, direction="heisenberg").expectation(state)`,
and the two paths agreeing to ~1e-15 on random circuits is what makes these
oracles trustworthy (`test_examples_oracles.py::test_statevector_matches_pauli_propagation`).

`initial_state` is a **product state**, spelled exactly the way
`PauliSum.expectation(state=...)` spells it, so suite code has one spelling:

- `None` -- `|0...0>`. This is the default here, and it is deliberately *not*
  `PauliSum.expectation`'s own default of `"x+"`; suite code passes the state
  explicitly everywhere (plan decision D9).
- one of the uniform names `"z+"` (`|0...0>`), `"x+"` (`|+...+>`), `"y+"`
  (`|+i...+i>`).
- a per-qubit label string of length `n` over `"01+-rl"`: `0`/`1` are `Z±`,
  `+`/`-` are `X±`, `r`/`l` are `Y±`. Both oracles and the engine take this
  form, so a non-uniform state (a domain wall, a staggered pattern) needs no
  translation between them.
- a sequence of `n` per-qubit entries, each either one of those characters or a
  uniform name -- the same thing, spelled for readability in code that builds a
  pattern qubit by qubit.

Non-product initial states, and density-matrix inputs, are out of scope for all
of them.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from paulistrings import Circuit, PauliSum, interop

__all__ = [
    "CLIFFORD_ANGLE_ATOL",
    "DEFAULT_MAX_STATEVECTOR_QUBITS",
    "PRODUCT_STATE_LABELS",
    "REFERENCES_DIR",
    "UNIFORM_PRODUCT_STATES",
    "CircuitSpec",
    "ConeTooLarge",
    "LightCone",
    "NonCliffordGate",
    "OracleError",
    "PublishedReference",
    "RecordingCircuit",
    "SkipOracle",
    "as_circuit_spec",
    "light_cone",
    "light_cone_exact",
    "load_published_reference",
    "pauli_terms",
    "record_gates",
    "spec_to_qiskit",
    "statevector_expectation",
    "stim_clifford_exact",
    "tsim_low_magic_exact",
]


class OracleError(RuntimeError):
    """An oracle cannot answer for this input (and must not guess)."""


class SkipOracle(OracleError):
    """An optional oracle's dependency is not installed.

    Distinct from `OracleError` so a caller can skip cleanly (the suite must run
    without any optional oracle) instead of treating a missing package as a
    failed cross-check.
    """


class NonCliffordGate(OracleError):
    """A gate outside the Clifford group reached the stabilizer oracle."""


class ConeTooLarge(OracleError):
    """A causal cone exceeded the evaluation budget of the chosen method."""


#: Per-qubit product-state labels, in `PauliSum.expectation`'s own spelling
#: (`0`/`1` = `Z±`, `+`/`-` = `X±`, `r`/`l` = `Y±`), mapped to the gate sequence
#: preparing that state from `|0>`. `|+i> = S H |0>` and `|-i> = S H X |0>`.
PRODUCT_STATE_LABELS: dict[str, tuple[str, ...]] = {
    "0": (),
    "1": ("x",),
    "+": ("h",),
    "-": ("x", "h"),
    "r": ("h", "s"),
    "l": ("x", "h", "s"),
}

#: The uniform state names, as their per-qubit label.
UNIFORM_PRODUCT_STATES: dict[str, str] = {"z+": "0", "x+": "+", "y+": "r"}

#: Statevector memory is `2**n * 16` bytes, and Aer's copy plus the returned
#: array means roughly twice that in flight: 4.3 GiB at n=28, 34 GiB at n=31.
#: 28 is the plan's stated ceiling (§6 Part 0.3); pass `max_qubits=` to move it.
DEFAULT_MAX_STATEVECTOR_QUBITS = 28

#: How far an angle may sit from a multiple of `pi/2` and still be treated as a
#: Clifford rotation. Angles in this suite are exact multiples built from
#: `math.pi`, so the only error is the double rounding of `pi/2` itself.
CLIFFORD_ANGLE_ATOL = 1e-9

#: Provenance-tagged published reference values. Ships with a README and no
#: data files -- see `load_published_reference`.
REFERENCES_DIR = Path(__file__).resolve().parents[1] / "data" / "references"


# =============================================================================
# Gate-list circuit description
# =============================================================================

# Gate vocabulary: name -> (qubit count or None for variadic, required extra
# fields). This is task-JSON schema v1's gate-object vocabulary, mirrored from
# `interop._push_task_gate`; a name added there must be added here too, and
# `_GATE_SPECS` is asserted against `interop`'s own tables by
# `test_examples_oracles.py::test_gate_vocabulary_matches_interop`.
_GATE_SPECS: dict[str, tuple[int | None, tuple[str, ...]]] = {
    "h": (1, ()),
    "s": (1, ()),
    "sdg": (1, ()),
    "x": (1, ()),
    "y": (1, ()),
    "z": (1, ()),
    "cnot": (2, ()),
    "cz": (2, ()),
    "swap": (2, ()),
    "rz": (1, ("theta",)),
    "rx": (1, ("theta",)),
    "ry": (1, ("theta",)),
    "pauli_rotation": (None, ("pauli", "theta")),
    "depolarize": (1, ("p",)),
    "dephase": (1, ("p",)),
    "amplitude_damping": (1, ("gamma",)),
    "pauli_channel": (1, ("px", "py", "pz")),
    "depolarize2": (2, ("p",)),
    "unitary_1q": (1, ("matrix",)),
    "unitary_2q": (2, ("matrix",)),
}

#: Gate names that are not unitary. The statevector and stabilizer oracles
#: refuse them; `light_cone_exact`'s Pauli path handles them, since the engine
#: propagates channels of either kind.
_NOISE_GATE_NAMES = frozenset(
    {"depolarize", "dephase", "amplitude_damping", "pauli_channel", "depolarize2"}
)

#: Gate names that are Clifford for every parameter value.
_ALWAYS_CLIFFORD = frozenset({"h", "s", "sdg", "x", "y", "z", "cnot", "cz", "swap"})

#: Gate names whose stim `TableauSimulator` method is not just the name.
_STIM_SIM_METHOD = {"sdg": "s_dag"}

#: `rz`/`rx`/`ry` as the weight-1 spelling of `pauli_rotation`.
_ROTATION_AXIS = {"rz": "Z", "rx": "X", "ry": "Y"}


def _gate_qubits(gate: Mapping[str, Any]) -> tuple[int, ...]:
    """The qubits a gate object acts on, as a tuple of ints."""
    return tuple(int(q) for q in gate["qubits"])


def _validate_gate(gate: Mapping[str, Any], num_qubits: int, index: int) -> None:
    if not isinstance(gate, Mapping) or "name" not in gate:
        raise ValueError(f'gate #{index} must be a mapping with a "name" key, got {gate!r}')
    name = gate["name"]
    if name not in _GATE_SPECS:
        raise ValueError(
            f"gate #{index}: unknown gate name {name!r}; known names are "
            f"{sorted(_GATE_SPECS)}"
        )
    arity, required = _GATE_SPECS[name]
    if "qubits" not in gate:
        raise ValueError(f'gate #{index} ({name!r}) is missing "qubits"')
    qubits = _gate_qubits(gate)
    if arity is not None and len(qubits) != arity:
        raise ValueError(
            f"gate #{index} ({name!r}) takes {arity} qubit(s), got {len(qubits)}"
        )
    if not qubits:
        raise ValueError(f"gate #{index} ({name!r}) has no qubits")
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"gate #{index} ({name!r}) repeats a qubit index: {qubits}")
    for q in qubits:
        if not 0 <= q < num_qubits:
            raise ValueError(
                f"gate #{index} ({name!r}): qubit index {q} is out of range for a "
                f"{num_qubits}-qubit circuit"
            )
    for key in required:
        if key not in gate:
            raise ValueError(f"gate #{index} ({name!r}) is missing required field {key!r}")
    if name == "pauli_rotation":
        pauli = str(gate["pauli"])
        if len(pauli) != len(qubits):
            raise ValueError(
                f"gate #{index} (pauli_rotation): pauli {pauli!r} has {len(pauli)} "
                f"characters but {len(qubits)} qubits"
            )
        bad = sorted(set(pauli) - set("XYZ"))
        if bad:
            raise ValueError(
                f"gate #{index} (pauli_rotation): unexpected Pauli character(s) {bad} "
                "(expected X/Y/Z; identity positions are expressed by omission)"
            )


def _matrix_to_json(matrix) -> list[list[list[float]]]:
    m = np.asarray(matrix, dtype=complex)
    return [[[float(v.real), float(v.imag)] for v in row] for row in m]


def _gate_to_json(gate: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in gate.items()}
    if "matrix" in out:
        out["matrix"] = _matrix_to_json(out["matrix"])
    out["qubits"] = list(_gate_qubits(gate))
    return out


@dataclass(frozen=True)
class CircuitSpec:
    """A circuit as an ordered gate list, in task-JSON schema v1's vocabulary.

    `gates` is a tuple of gate objects -- plain mappings `{"name": ..., "qubits":
    [...], ...}` with the extra fields each name requires (`theta`, `pauli`,
    `p`, `gamma`, `px`/`py`/`pz`, `matrix`). A `matrix` field holds a NumPy
    array: that is the in-memory form every consumer here wants, and
    `as_circuit_spec` converts the JSON nested `[re, im]` pair form (which is
    what `Circuit.gates` and task JSON carry) on the way in.

    One gate object per gate, matching the suite's one-gate-per-channel
    construction rule (plan §5, decision D10): the gate list index *is* the
    channel index, so a cone's `gate_indices` and the engine's layer indices
    line up.

    Treat instances as immutable -- `gates` is a tuple, but the mappings inside
    it are not deep-copied.
    """

    num_qubits: int
    gates: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {self.num_qubits}")
        object.__setattr__(self, "gates", tuple(self.gates))
        for i, gate in enumerate(self.gates):
            _validate_gate(gate, self.num_qubits, i)

    def __len__(self) -> int:
        return len(self.gates)

    @property
    def gate_names(self) -> tuple[str, ...]:
        return tuple(str(g["name"]) for g in self.gates)

    @property
    def is_unitary(self) -> bool:
        """`True` when no gate is a noise channel."""
        return not any(g["name"] in _NOISE_GATE_NAMES for g in self.gates)

    def support(self, index: int) -> tuple[int, ...]:
        """The qubits gate `index` acts on."""
        return _gate_qubits(self.gates[index])

    def to_circuit_json(self) -> dict[str, Any]:
        """The task-JSON schema v1 `"circuit"` object for this gate list."""
        return {"gates": [_gate_to_json(g) for g in self.gates]}

    def to_circuit(self) -> Circuit:
        """Build the engine's `Circuit` from this gate list.

        Goes through `paulistrings.interop.circuit_from_json`, the shipped
        task-JSON builder, rather than a second gate-dispatch table: the oracle
        and the engine then disagree only if `interop` itself is wrong, and that
        has its own tests.
        """
        return interop.circuit_from_json(self.to_circuit_json(), self.num_qubits)

    def restrict(self, qubits: Sequence[int]) -> CircuitSpec:
        """This circuit with only the gates supported inside `qubits`, renumbered.

        `qubits` is renumbered to `0..len(qubits)-1` in the given order. A gate
        that touches a qubit outside `qubits` is dropped -- so this is only a
        faithful restriction when applied to a causal cone, where no dropped
        gate can influence the observable (see `light_cone`).
        """
        keep = {q: i for i, q in enumerate(qubits)}
        gates: list[Mapping[str, Any]] = []
        for gate in self.gates:
            qs = _gate_qubits(gate)
            if all(q in keep for q in qs):
                remapped = {k: v for k, v in gate.items()}
                remapped["qubits"] = [keep[q] for q in qs]
                gates.append(remapped)
        return CircuitSpec(num_qubits=len(keep), gates=tuple(gates))


class RecordingCircuit:
    """Deprecated thin wrapper around a real `paulistrings.Circuit`.

    It exists only so call sites that still name it keep working. `Circuit` is
    introspectable now (`Circuit.gates`), so there is nothing left to record:
    this builds a real `Circuit`, forwards every attribute to it, and adds the
    one thing the old stand-in had that `Circuit` does not -- a `spec` property,
    which is just `as_circuit_spec(self.circuit)`.

    New code should build a `paulistrings.Circuit` and pass it straight to
    `as_circuit_spec` (or to any oracle, which calls that itself).
    """

    __slots__ = ("circuit",)

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        self.circuit = Circuit(int(num_qubits))

    def __getattr__(self, name: str) -> Any:
        # Reached only for names not found normally, i.e. everything the real
        # Circuit provides: h/s/sdg/.../unitary_2q, append, num_qubits.
        return getattr(self.circuit, name)

    def __len__(self) -> int:
        return len(self.circuit)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RecordingCircuit({self.circuit.num_qubits}, {len(self.circuit)} gates)"

    @property
    def spec(self) -> CircuitSpec:
        return as_circuit_spec(self.circuit)


def record_gates(builder, *args: Any, **kwargs: Any) -> CircuitSpec:
    """Run a `Circuit`-building function and return its gate list.

    `builder` is any function that constructs and returns a
    `paulistrings.Circuit` -- every builder in `examples/common/circuits.py`.
    Since `Circuit.gates` exposes the channel list, this is now just
    `as_circuit_spec(builder(*args, **kwargs))`: the builder runs untouched,
    against the real API, and nothing is rebound. (It used to run against a
    recording stand-in installed into `builder.__globals__`, which was neither
    thread-safe nor able to see names the builder did not resolve as module
    globals; both caveats are gone with the mechanism.)

    Kept as a named function because the suite's call sites read as "capture
    this builder's gate list", and because it fails loudly -- `OracleError`,
    never a spec that is missing gates -- when the builder returns something
    that is not a circuit.
    """
    result = builder(*args, **kwargs)
    if not isinstance(result, (Circuit, RecordingCircuit)):
        raise OracleError(
            f"{getattr(builder, '__qualname__', builder)} returned "
            f"{type(result).__name__}, not a paulistrings.Circuit, so it has no gate "
            "list to read. Pass a CircuitSpec built another way."
        )
    return as_circuit_spec(result)


def as_circuit_spec(circuit: Any, num_qubits: int | None = None) -> CircuitSpec:
    """Coerce a supported circuit description to a `CircuitSpec`.

    Accepted: a `CircuitSpec`, a `paulistrings.Circuit`, a `RecordingCircuit`,
    or a task-JSON circuit object `{"gates": [...]}` (then `num_qubits` is
    required).

    A `Circuit` is read through `Circuit.gates`, which emits exactly the
    task-JSON gate objects this module's vocabulary is, so the `Circuit` and
    the JSON paths are the same conversion -- including turning a `matrix`
    from the JSON `[re, im]` pair form into a complex array.
    """
    if isinstance(circuit, CircuitSpec):
        return circuit
    if isinstance(circuit, RecordingCircuit):
        return circuit.spec
    if isinstance(circuit, Circuit):
        return CircuitSpec(
            num_qubits=circuit.num_qubits,
            gates=tuple(_json_gate_to_spec_gate(g) for g in circuit.gates),
        )
    if isinstance(circuit, Mapping) and "gates" in circuit:
        if num_qubits is None:
            raise TypeError(
                "num_qubits is required when passing a task-JSON circuit object"
            )
        gates = [_json_gate_to_spec_gate(g) for g in circuit["gates"]]
        return CircuitSpec(num_qubits=num_qubits, gates=tuple(gates))
    raise TypeError(
        f"unsupported circuit description {type(circuit).__name__}; expected a "
        "CircuitSpec, a paulistrings.Circuit, or a task-JSON circuit object"
    )


def _json_gate_to_spec_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in gate.items()}
    if "matrix" in out and not isinstance(out["matrix"], np.ndarray):
        out["matrix"] = np.array(
            [[complex(re, im) for re, im in row] for row in out["matrix"]],
            dtype=complex,
        )
    return out


# =============================================================================
# Observables and product states
# =============================================================================

# (x, z) bit pair -> Pauli character. `(1, 1) -> "Y"` with no phase factor is
# the repo-wide Hermitian convention (CLAUDE.md §Known gaps).
_XZ_CODE_TO_CHAR = np.array(list("IXZY"))  # index = x + 2*z


def pauli_terms(
    observable: Any, num_qubits: int | None = None
) -> list[tuple[str, complex]]:
    """`observable` as a list of `(pauli_label, coefficient)` pairs.

    Accepts a `PauliSum` (decoded from its `x_array`/`z_array`/coefficient
    export), a `{label: coefficient}` mapping, a bare label string (coefficient
    1), or an already-decoded sequence of `(label, coefficient)` pairs -- so the
    function is idempotent and the oracles can hand each other term lists
    without re-decoding. Labels are full length with character `i` on qubit `i`,
    in the Hermitian convention -- the same spelling `PauliSum.from_strings`
    takes.

    Term order follows the source: a `PauliSum`'s bucketed storage order, which
    is unspecified and free to change (CLAUDE.md §Determinism policy), so sort
    before comparing two term lists.
    """
    if isinstance(observable, PauliSum):
        n = observable.num_qubits
        if num_qubits is not None and num_qubits != n:
            raise ValueError(
                f"observable has {n} qubits, but num_qubits={num_qubits} was given"
            )
        coeffs = np.asarray(observable.coefficients_array())
        if coeffs.size == 0:
            return []
        xs = np.asarray(observable.x_array())
        zs = np.asarray(observable.z_array())
        bits = np.arange(n)
        words = bits // 64
        shifts = (bits % 64).astype(np.uint64)
        xb = (xs[:, words] >> shifts) & np.uint64(1)
        zb = (zs[:, words] >> shifts) & np.uint64(1)
        codes = (xb + 2 * zb).astype(np.intp)
        chars = _XZ_CODE_TO_CHAR[codes]
        return [
            ("".join(row), complex(c)) for row, c in zip(chars, coeffs, strict=True)
        ]
    if isinstance(observable, str):
        observable = {observable: 1.0}
    if isinstance(observable, Mapping):
        terms = [(str(k), complex(v)) for k, v in observable.items()]
    elif isinstance(observable, Sequence):
        terms = []
        for entry in observable:
            if isinstance(entry, str) or not isinstance(entry, Sequence) or len(entry) != 2:
                raise TypeError(
                    "a sequence observable must hold (label, coefficient) pairs, got "
                    f"{entry!r}"
                )
            label, coefficient = entry
            terms.append((str(label), complex(coefficient)))
    else:
        raise TypeError(
            f"unsupported observable {type(observable).__name__}; expected a PauliSum, a "
            "{label: coefficient} mapping, a label string, or a sequence of "
            "(label, coefficient) pairs"
        )
    for label, _ in terms:
        bad = sorted(set(label) - set("IXYZ"))
        if bad:
            raise ValueError(f"Pauli label {label!r} has unexpected character(s) {bad}")
    lengths = {len(label) for label, _ in terms}
    if len(lengths) > 1:
        raise ValueError(f"Pauli labels have mixed lengths {sorted(lengths)}")
    if num_qubits is not None and lengths and lengths != {num_qubits}:
        raise ValueError(
            f"Pauli labels have length {lengths.pop()}, expected {num_qubits}"
        )
    return terms


def _observable_support(terms: Iterable[tuple[str, complex]]) -> set[int]:
    support: set[int] = set()
    for label, _ in terms:
        support.update(i for i, ch in enumerate(label) if ch != "I")
    return support


def _to_qiskit_label(label: str) -> str:
    """Reverse a paulistrings label into qiskit's MSB-first spelling.

    No phase factor is involved: both conventions use the Hermitian `Y`. See
    the module docstring, trap 2.
    """
    return label[::-1]


_STATE_SPELLING_HELP = (
    'expected None, a uniform name ("z+", "x+", "y+"), a per-qubit label string '
    'over "01+-rl" (0/1 = Z±, +/- = X±, r/l = Y±), or a sequence of per-qubit '
    "entries in either of those spellings"
)


def _normalize_initial_state(initial_state: Any, num_qubits: int) -> str:
    """`initial_state` as a per-qubit label string of length `num_qubits`.

    The returned string is exactly what `PauliSum.expectation` accepts, so the
    Pauli path can hand it straight to the engine and the statevector/stabilizer
    paths can turn it into preparation gates -- one canonical form, no second
    spelling to keep in sync.
    """
    if initial_state is None:
        return "0" * num_qubits
    if isinstance(initial_state, str):
        lowered = initial_state.lower()
        if lowered in UNIFORM_PRODUCT_STATES:
            return UNIFORM_PRODUCT_STATES[lowered] * num_qubits
        bad = sorted(set(lowered) - set(PRODUCT_STATE_LABELS))
        if bad:
            raise ValueError(
                f"unknown product state {initial_state!r}: character(s) {bad} are not "
                f"per-qubit labels; {_STATE_SPELLING_HELP}"
            )
        if len(lowered) != num_qubits:
            raise ValueError(
                f"product state {initial_state!r} has {len(lowered)} per-qubit labels "
                f"for a {num_qubits}-qubit system"
            )
        return lowered
    if isinstance(initial_state, Sequence):
        if len(initial_state) != num_qubits:
            raise ValueError(
                f"initial_state has {len(initial_state)} entries for a {num_qubits}-"
                "qubit system"
            )
        out: list[str] = []
        for q, entry in enumerate(initial_state):
            key = str(entry).lower()
            if key in UNIFORM_PRODUCT_STATES:
                out.append(UNIFORM_PRODUCT_STATES[key])
            elif key in PRODUCT_STATE_LABELS:
                out.append(key)
            else:
                raise ValueError(
                    f"unknown product state {entry!r} at qubit {q}; "
                    f"{_STATE_SPELLING_HELP}"
                )
        return "".join(out)
    raise TypeError(
        f"unsupported initial_state {type(initial_state).__name__}; "
        f"{_STATE_SPELLING_HELP}"
    )


def _state_prep_gates(labels: str) -> list[dict[str, Any]]:
    """Gate objects preparing the product state `labels` from `|0...0>`."""
    gates: list[dict[str, Any]] = []
    for q, label in enumerate(labels):
        for name in PRODUCT_STATE_LABELS[label]:
            gates.append({"name": name, "qubits": [q]})
    return gates


def _engine_state_argument(labels: str) -> str:
    """`labels` as the shortest spelling `PauliSum.expectation` accepts.

    A uniform pattern goes through as its uniform name so the engine takes its
    dedicated uniform reduction; anything else goes through verbatim.
    """
    unique = set(labels)
    if len(unique) == 1:
        char = unique.pop()
        for name, label in UNIFORM_PRODUCT_STATES.items():
            if label == char:
                return name
    return labels


# =============================================================================
# 1. Statevector oracle (qiskit Aer)
# =============================================================================


def _import_aer():
    """`(QuantumCircuit, SparsePauliOp, Statevector, AerSimulator)`, or `SkipOracle`.

    Lazy, so importing this module never requires qiskit; and a missing package
    surfaces as `SkipOracle`, which callers treat the way `importorskip` treats
    one, rather than as a failed cross-check.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp, Statevector
        from qiskit_aer import AerSimulator
    except ImportError as exc:  # pragma: no cover - exercised by importorskip
        raise SkipOracle(
            "the statevector oracle needs qiskit and qiskit-aer "
            "(pip install '.[examples]'); "
            f"import failed: {exc}"
        ) from exc
    return QuantumCircuit, SparsePauliOp, Statevector, AerSimulator


@lru_cache(maxsize=256)
def _pauli_rotation_matrix(pauli: str, theta: float) -> np.ndarray:
    """`exp(-i·theta·P/2)` as a dense matrix in qiskit's little-endian basis.

    Built by qiskit itself, from `PauliEvolutionGate(SparsePauliOp(label),
    time=theta/2)` -- the halved time and the reversed label are the two halves
    of trap 3 in the module docstring. Emitting the dense matrix (rather than
    appending the `PauliEvolutionGate` and transpiling) keeps every gate this
    module hands Aer inside Aer's native basis, so no transpiler pass runs; the
    cache makes the ~12 ms qiskit synthesis a per-`(pauli, theta)` cost rather
    than a per-gate one.
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Operator, SparsePauliOp

    gate = PauliEvolutionGate(SparsePauliOp(_to_qiskit_label(pauli)), time=theta / 2.0)
    matrix = np.array(Operator(gate).data, dtype=complex)
    matrix.flags.writeable = False
    return matrix


def _emit_qiskit_gate(qc, gate: Mapping[str, Any]) -> None:
    name = str(gate["name"])
    qubits = _gate_qubits(gate)
    if name in _NOISE_GATE_NAMES:
        raise OracleError(
            f"the statevector oracle is unitary-only, but the circuit contains the "
            f"noise channel {name!r} on qubits {list(qubits)}. Use light_cone_exact's "
            "Pauli path (or a density-matrix reference, which this module does not "
            "provide) for noisy circuits."
        )
    # qiskit spells `S^dagger` `sdg` too, so the name goes through unchanged.
    if name in ("h", "s", "sdg", "x", "y", "z"):
        getattr(qc, name)(qubits[0])
    elif name == "cnot":
        qc.cx(qubits[0], qubits[1])
    elif name in ("cz", "swap"):
        getattr(qc, name)(qubits[0], qubits[1])
    elif name in ("rz", "rx", "ry"):
        getattr(qc, name)(float(gate["theta"]), qubits[0])
    elif name == "pauli_rotation":
        matrix = _pauli_rotation_matrix(str(gate["pauli"]), float(gate["theta"]))
        # `qc.unitary(m, qargs)` reads `qargs[0]` as the matrix's least
        # significant factor, which is exactly how `Operator(gate).data` above
        # is laid out over the gate's own qubits -- so the qubit list goes
        # through in order.
        qc.unitary(np.array(matrix), list(qubits))
    elif name == "unitary_1q":
        qc.unitary(np.asarray(gate["matrix"], dtype=complex), [qubits[0]])
    elif name == "unitary_2q":
        # This library's `unitary_2q(q0, q1, m)` has `q0` as the *more*
        # significant tensor factor; qiskit's `qargs[0]` is the least
        # significant. Hence the reversed qargs (module docstring, trap 1).
        qc.unitary(np.asarray(gate["matrix"], dtype=complex), [qubits[1], qubits[0]])
    else:  # pragma: no cover - _validate_gate rejects unknown names first
        raise OracleError(f"no qiskit translation for gate {name!r}")


def spec_to_qiskit(circuit: Any, initial_state: Any = None):
    """Translate a `CircuitSpec` to a `qiskit.QuantumCircuit`.

    The state preparation for `initial_state` is prepended as gates (`h`, `h s`)
    rather than injected as a state vector, so the returned circuit is a plain
    unitary circuit that any qiskit tool can consume. Raises `OracleError` on a
    noise channel: this direction is unitary-only.

    Exposed (not private) because it is the natural handle for cross-checking
    the conversion itself, and for feeding a suite circuit to any other qiskit
    based tool.
    """
    QuantumCircuit, _SparsePauliOp, _Statevector, _AerSimulator = _import_aer()
    spec = as_circuit_spec(circuit)
    labels = _normalize_initial_state(initial_state, spec.num_qubits)
    qc = QuantumCircuit(spec.num_qubits)
    for gate in _state_prep_gates(labels):
        _emit_qiskit_gate(qc, gate)
    for gate in spec.gates:
        _emit_qiskit_gate(qc, gate)
    return qc


def statevector_expectation(
    circuit: Any,
    observable: Any,
    initial_state: Any = None,
    *,
    max_qubits: int = DEFAULT_MAX_STATEVECTOR_QUBITS,
) -> complex:
    """Exact `<psi| U^dagger O U |psi>` by dense statevector simulation.

    `circuit` is a `CircuitSpec` (or anything `as_circuit_spec` accepts),
    `observable` anything `pauli_terms` accepts, and `initial_state` a product
    state (default `|0...0>`; see the module docstring). Unitary circuits only.

    The simulation is run by **qiskit Aer** (`AerSimulator(method="statevector")`
    plus `save_statevector`); the contraction with `O` is
    `qiskit.quantum_info.Statevector.expectation_value` on a `SparsePauliOp`.
    Nothing here implements time evolution or an inner product itself.

    `max_qubits` guards the memory cliff: the state is `2**n * 16` bytes, of
    which Aer holds one copy and Python another. Above the guard this raises
    `ConeTooLarge` rather than trying.
    """
    spec = as_circuit_spec(circuit)
    if not spec.is_unitary:
        noisy = sorted(
            {str(g["name"]) for g in spec.gates if g["name"] in _NOISE_GATE_NAMES}
        )
        # Refuse before _import_aer(): the refusal is a property of the request,
        # not of the environment, so it must win over SkipOracle when qiskit is
        # absent (CI's numpy-only job sees this ordering).
        raise OracleError(
            f"the statevector oracle is unitary-only, but the circuit contains the "
            f"noise channel(s) {noisy}. Use light_cone_exact's Pauli path (or a "
            "density-matrix reference, which this module does not provide) for "
            "noisy circuits."
        )
    _QuantumCircuit, SparsePauliOp, Statevector, AerSimulator = _import_aer()
    n = spec.num_qubits
    if n > max_qubits:
        raise ConeTooLarge(
            f"statevector_expectation refuses n={n} > max_qubits={max_qubits}: the "
            f"state alone would need {2 ** n * 16 / 2**30:.1f} GiB. Raise max_qubits "
            "deliberately, or use stim_clifford_exact / light_cone_exact instead."
        )
    terms = pauli_terms(observable, n)
    if not terms:
        return 0j

    qc = spec_to_qiskit(spec, initial_state)
    qc.save_statevector()
    result = AerSimulator(method="statevector").run(qc).result()
    state = Statevector(np.asarray(result.data()["statevector"]))
    op = SparsePauliOp(
        [_to_qiskit_label(label) for label, _ in terms],
        np.array([c for _, c in terms], dtype=complex),
    )
    return complex(state.expectation_value(op))


# =============================================================================
# 2. Clifford oracle (stim tableau simulator)
# =============================================================================


def _import_stim():
    try:
        import stim
    except ImportError as exc:  # pragma: no cover - exercised by importorskip
        raise SkipOracle(
            f"the Clifford oracle needs the stim package; import failed: {exc}"
        ) from exc
    return stim


def _clifford_quarter_turns(theta: float, what: str) -> int:
    """`theta / (pi/2)` as an integer mod 4, or raise if it is not one.

    A rotation `exp(-i·theta·P/2)` is Clifford exactly when `theta` is a
    multiple of `pi/2`; the returned `k` selects which of the four:
    `0` identity, `1` `sqrt(P)`, `2` `P`, `3` `sqrt(P)^dagger`.
    """
    quarters = theta / (math.pi / 2.0)
    k = round(quarters)
    if abs(quarters - k) > CLIFFORD_ANGLE_ATOL:
        raise NonCliffordGate(
            f"{what}: theta={theta!r} is {quarters:.6f} quarter-turns, not a multiple "
            "of pi/2, so the rotation is not Clifford. The stabilizer oracle only "
            "applies at the Clifford points (for the kicked Ising: theta_h in {0, pi/2} "
            "with theta_zz = -pi/2); use light_cone_exact or statevector_expectation "
            "elsewhere."
        )
    return k % 4


def _clifford_rotation_tableau(stim, pauli: str, k: int):
    """The `stim.Tableau` of `exp(-i·theta·P/2)` for `theta = k·pi/2`.

    Built from the conjugation action rather than from named stim gates, so it
    covers generators no named gate spells (mixed two-qubit `XZ`, weight >= 3)
    with one rule. For `U = (I - i·s·P)/sqrt(2)` (the `k` odd case, `s = +1` for
    `k = 1` and `-1` for `k = 3`):

        U Q U^dagger = Q                 if [P, Q] = 0
                     = -i·s·P·Q          if P and Q anticommute

    and for `k = 2`, `U = -i·P`, so `Q -> +Q` / `-Q` on the same split. The
    construction reproduces stim's own `S`, `SQRT_X`, `SQRT_Y`, `SQRT_ZZ`,
    `SQRT_ZZ_DAG`, ... exactly, which is what
    `test_examples_oracles.py::test_clifford_rotation_tableaux_match_stim_named_gates`
    asserts.
    """
    width = len(pauli)
    generator = stim.PauliString(pauli)
    images: dict[str, list] = {"X": [], "Z": []}
    for j in range(width):
        for kind in ("X", "Z"):
            chars = ["_"] * width
            chars[j] = kind
            q = stim.PauliString("".join(chars))
            if k == 0 or generator.commutes(q):
                images[kind].append(q)
            elif k == 2:
                images[kind].append(-q)
            else:
                s = 1 if k == 1 else -1
                images[kind].append((-1j * s) * (generator * q))
    return stim.Tableau.from_conjugated_generators(xs=images["X"], zs=images["Z"])


def _apply_spec_gate_to_tableau_sim(stim, sim, gate: Mapping[str, Any]) -> None:
    name = str(gate["name"])
    qubits = _gate_qubits(gate)
    if name in _NOISE_GATE_NAMES:
        raise NonCliffordGate(
            f"stim_clifford_exact is an exact *unitary* Clifford oracle, but the "
            f"circuit contains the noise channel {name!r} on qubits {list(qubits)}. A "
            "tableau simulation of a noise channel samples one Pauli error rather than "
            "averaging over them, so its expectation value would be a sample, not the "
            "channel's. Use light_cone_exact's Pauli path for noisy circuits."
        )
    if name in _ALWAYS_CLIFFORD:
        # stim's simulator spells `S^dagger` `s_dag`; every other name matches.
        getattr(sim, _STIM_SIM_METHOD.get(name, name))(*qubits)
        return
    if name in _ROTATION_AXIS:
        axis = _ROTATION_AXIS[name]
        k = _clifford_quarter_turns(float(gate["theta"]), f"{name} on qubit {qubits[0]}")
        if k == 0:
            return
        sim.do_tableau(_clifford_rotation_tableau(stim, axis, k), list(qubits))
        return
    if name == "pauli_rotation":
        pauli = str(gate["pauli"])
        k = _clifford_quarter_turns(
            float(gate["theta"]), f"pauli_rotation({pauli!r}, {list(qubits)})"
        )
        if k == 0:
            return
        sim.do_tableau(_clifford_rotation_tableau(stim, pauli, k), list(qubits))
        return
    if name in ("unitary_1q", "unitary_2q"):
        matrix = np.asarray(gate["matrix"], dtype=complex)
        try:
            # `endian="big"` = first target is the most significant factor,
            # which is this library's `unitary_2q(q0, q1, m)` convention.
            tableau = stim.Tableau.from_unitary_matrix(matrix, endian="big")
        except ValueError as exc:
            raise NonCliffordGate(
                f"{name} on qubits {list(qubits)} is not a Clifford unitary "
                f"(stim: {exc})"
            ) from exc
        sim.do_tableau(tableau, list(qubits))
        return
    raise NonCliffordGate(  # pragma: no cover - _validate_gate rejects unknown names
        f"no stim translation for gate {name!r}"
    )


def _load_stim_source(stim, src):
    """`src` as a `stim.Circuit`, or `None` if it is not a stim source.

    Mirrors `interop._load_stim_circuit`'s admission rules (a `stim.Circuit`, a
    path, or program text), but returns `None` instead of raising so the caller
    can fall through to the `CircuitSpec` path.
    """
    if isinstance(src, stim.Circuit):
        return src
    if isinstance(src, os.PathLike):
        return stim.Circuit.from_file(str(src))
    if isinstance(src, str):
        path = Path(src)
        if path.exists():
            return stim.Circuit.from_file(str(path))
        return stim.Circuit(src)
    return None


def _check_stim_circuit_is_clifford(stim, circuit) -> None:
    for index, instruction in enumerate(circuit):
        name = instruction.name
        data = stim.gate_data(name)
        if data.is_unitary:
            continue
        if name in ("TICK", "QUBIT_COORDS", "SHIFT_COORDS", "OBSERVABLE_INCLUDE", "DETECTOR"):
            continue
        why = (
            "a noise channel"
            if data.is_noisy_gate
            else "a measurement" if data.produces_measurements
            else "a reset" if data.is_reset
            else "not a unitary Clifford operation"
        )
        raise NonCliffordGate(
            f"stim instruction {name!r} (#{index}) is {why}; stim_clifford_exact "
            "evaluates an exact unitary Clifford circuit, and a tableau simulation of "
            "either would return a sample rather than an expectation value."
        )


def stim_clifford_exact(
    circuit_or_stim_file: Any,
    observable: Any = None,
    *,
    initial_state: Any = None,
) -> complex:
    """Exact `<psi| U^dagger O U |psi>` for a Clifford circuit, at any `n`.

    `circuit_or_stim_file` is either

    - a `CircuitSpec` / `paulistrings.Circuit` -- each gate is applied to
      `stim.TableauSimulator`; a gate outside the Clifford group raises
      `NonCliffordGate` naming it, and noise channels are refused (a tableau
      simulation would *sample* an error, not average over the channel); or
    - a `.stim` file path, a `stim.Circuit`, or stim program text -- fed to the
      tableau simulator directly, so one checked-in `.stim` file drives both
      the engine (via `paulistrings.interop.circuit_from_stim`) and this oracle
      with no second transcription (plan §7 rule 6).

    `observable` is anything `pauli_terms` accepts. It may be `None` **only**
    for a stim source, in which case the file's `OBSERVABLE_INCLUDE` targets are
    used -- read through `interop.circuit_from_stim`, the same importer the
    engine uses, so the observable is shared too.

    Each Pauli term is evaluated with `TableauSimulator.
    peek_observable_expectation`, which returns exactly `+1`, `-1` or `0`; the
    result is the coefficient-weighted sum, so a single-term unit-coefficient
    observable at a Clifford point comes back as an exact integer.
    """
    stim = _import_stim()
    stim_circuit = _load_stim_source(stim, circuit_or_stim_file)

    if stim_circuit is None:
        spec = as_circuit_spec(circuit_or_stim_file)
        if observable is None:
            raise TypeError(
                "observable is required unless circuit_or_stim_file is a stim source "
                "carrying OBSERVABLE_INCLUDE"
            )
        n_circuit = spec.num_qubits
    else:
        stim_circuit = stim_circuit.flattened()
        _check_stim_circuit_is_clifford(stim, stim_circuit)
        n_circuit = stim_circuit.num_qubits
        if observable is None:
            # Read through the same importer the engine uses, so the observable
            # is shared rather than transcribed. That importer accepts a
            # narrower instruction set than the tableau simulator does (it has
            # no spelling for `SQRT_X`, `ISWAP`, ...), so its refusal is turned
            # into a pointer at the explicit-observable escape hatch.
            try:
                _imported, observable = interop.circuit_from_stim(stim_circuit)
            except ValueError as exc:
                raise OracleError(
                    "could not read the observable out of the stim source through "
                    f"paulistrings.interop.circuit_from_stim ({exc}). The tableau "
                    "oracle itself can run this circuit; pass `observable=` "
                    "explicitly to skip the importer."
                ) from exc
            if observable is None:
                raise OracleError(
                    "the stim source carries no OBSERVABLE_INCLUDE instruction, so "
                    "there is no observable to evaluate; pass one explicitly."
                )
        spec = None

    terms = pauli_terms(observable)
    if not terms:
        return 0j
    n = max(n_circuit, len(terms[0][0]))

    sim = stim.TableauSimulator()
    sim.set_num_qubits(n)
    for gate in _state_prep_gates(_normalize_initial_state(initial_state, n)):
        _apply_spec_gate_to_tableau_sim(stim, sim, gate)
    if spec is not None:
        for gate in spec.gates:
            _apply_spec_gate_to_tableau_sim(stim, sim, gate)
    else:
        sim.do_circuit(stim_circuit)

    total = 0j
    for label, coefficient in terms:
        padded = label + "I" * (n - len(label))
        total += coefficient * sim.peek_observable_expectation(stim.PauliString(padded))
    return total


# =============================================================================
# 3. Light-cone oracle
# =============================================================================


@dataclass(frozen=True)
class LightCone:
    """The backward causal cone of an observable through a gate list.

    `qubits` is the sorted set of qubits the cone covers and `gate_indices` the
    (ascending) indices of the gates inside it. `n_steps` is optional context
    recorded by the caller -- the Trotter-step count of the circuit, say -- and
    is echoed in diagnostics; it never affects the computation, which reads only
    the gate list. `commutation_aware` records which of the two cones in
    `light_cone` produced this one.

    Either way the cone is an **over-approximation**, so restricting to it is
    exact; it is not the tightest possible cone (that would be Pauli
    propagation, the thing being checked).
    """

    qubits: tuple[int, ...]
    gate_indices: tuple[int, ...]
    source_num_qubits: int
    n_steps: int | None = None
    commutation_aware: bool = True

    @property
    def size(self) -> int:
        return len(self.qubits)

    @property
    def index_map(self) -> dict[int, int]:
        """Original qubit index -> reduced index `0..size-1`."""
        return {q: i for i, q in enumerate(self.qubits)}

    def describe(self) -> str:
        steps = "" if self.n_steps is None else f", n_steps={self.n_steps}"
        return (
            f"cone of {self.size}/{self.source_num_qubits} qubits, "
            f"{len(self.gate_indices)} gates{steps}"
        )


def light_cone(
    circuit: Any,
    observable: Any,
    n_steps: int | None = None,
    *,
    commutation_aware: bool = True,
) -> LightCone:
    """Compute the observable's backward light cone through `circuit`.

    Two cones, both over-approximations, so restricting to either is exact.
    Both walk the gate list in reverse and are computed from the gate list,
    never tabulated -- the plan's Part 0.3 requirement that this reference be
    "computed, not hard-coded".

    **Support cone** (`commutation_aware=False`). A gate whose support meets the
    cone is kept and its whole support joins the cone; a gate disjoint from the
    cone is dropped. Dropping is exact for any initial state: in the Heisenberg
    picture `V^dagger O V = O` whenever `V` acts only on qubits outside the
    already-back-evolved support.

    **Commutation-aware cone** (the default). Instead of a set of qubits, this
    tracks per qubit the set of local Paulis the back-evolved operator can carry
    -- `{"I"}` means provably identity there, so "in the cone" is "not
    `{"I"}`". A Pauli rotation `exp(-i·theta·P/2)` is then dropped whenever
    every reachable local Pauli commutes with `P` site-wise (`reach[q]` is
    inside `{"I", P[q]}` for every `q` in `P`'s support), which is a sufficient
    condition for `V^dagger O V = O`. When it does act, each site's reachable
    set gains `reach[q] · P[q]`. Anything that is not a Pauli rotation (Clifford
    gates, raw unitaries, noise channels) falls back to the support rule and
    then admits all four Paulis on its support.

    The difference is large, and on the headline benchmark it decides whether a
    dense reference is reachable at all. One Trotter step's `ZZ` rotations all
    commute with each other but are *emitted* as three disjoint-support colour
    classes, so the support cone grows about three hops per step while the
    operator's true support grows one. At five steps on the 127-qubit heavy-hex
    kicked Ising, for the weight-1 / weight-10 / weight-17 observables:

    | cone | sizes |
    |---|---|
    | commutation-aware (default) | 19 / 30 / 59 |
    | radius-5 ball, i.e. one hop per step | 31 / 37 / 68 |
    | support-only (`commutation_aware=False`) | 87 / 72 / 122 |

    The middle row is what Kim et al.'s SI §VII B reports, and
    `test_examples_oracles.py` recomputes it from the edge list as a check on
    the lattice and the observable supports. The default cone is one layer
    tighter than that (the trailing `ZZ` layer commutes through), and 19 is
    small enough that `light_cone_exact` can answer the weight-1 observable at
    five steps with a dense statevector on 19 qubits.
    """
    spec = as_circuit_spec(circuit)
    terms = pauli_terms(observable, spec.num_qubits)
    if commutation_aware:
        reach = _reachable_paulis(terms, spec.num_qubits)
        kept = _walk_commutation_aware(spec, reach)
        qubits = tuple(q for q in range(spec.num_qubits) if reach[q] != frozenset("I"))
    else:
        cone = _observable_support(terms)
        kept = []
        for index in range(len(spec.gates) - 1, -1, -1):
            support = spec.support(index)
            if cone.isdisjoint(support):
                continue
            cone.update(support)
            kept.append(index)
        kept.reverse()
        qubits = tuple(sorted(cone))
    return LightCone(
        qubits=qubits,
        gate_indices=tuple(kept),
        source_num_qubits=spec.num_qubits,
        n_steps=n_steps,
        commutation_aware=commutation_aware,
    )


# Pauli multiplication up to phase: the phase is irrelevant to "which local
# Pauli can appear here", which is all the cone walk tracks.
_PAULI_PRODUCT: dict[tuple[str, str], str] = {
    (a, b): (b if a == "I" else a if b == "I" else "I" if a == b else
             next(iter(set("XYZ") - {a, b})))
    for a in "IXYZ"
    for b in "IXYZ"
}

_ALL_PAULIS = frozenset("IXYZ")
_ONLY_IDENTITY = frozenset("I")


def _reachable_paulis(
    terms: Sequence[tuple[str, complex]], num_qubits: int
) -> list[frozenset[str]]:
    """Per-qubit set of local Paulis present in `terms`.

    The seed of the commutation-aware walk. A single-term `Z_62` seeds
    `{"Z"}` at qubit 62 and `{"I"}` everywhere else -- note `{"Z"}` without
    `"I"`, which is what lets the first `ZZ` layer be dropped outright.
    """
    reach = [set() for _ in range(num_qubits)]
    for label, _ in terms:
        for q, ch in enumerate(label):
            reach[q].add(ch)
    return [frozenset(s) if s else _ONLY_IDENTITY for s in reach]


def _walk_commutation_aware(
    spec: CircuitSpec, reach: list[frozenset[str]]
) -> list[int]:
    """Reverse walk updating `reach` in place; returns the kept gate indices."""
    kept: list[int] = []
    for index in range(len(spec.gates) - 1, -1, -1):
        gate = spec.gates[index]
        name = str(gate["name"])
        support = _gate_qubits(gate)
        generator = (
            str(gate["pauli"])
            if name == "pauli_rotation"
            else _ROTATION_AXIS.get(name)
        )
        if generator is not None:
            if all(
                reach[q] <= (_ONLY_IDENTITY | {axis})
                for q, axis in zip(support, generator, strict=True)
            ):
                continue
            for q, axis in zip(support, generator, strict=True):
                reach[q] = reach[q] | {_PAULI_PRODUCT[(a, axis)] for a in reach[q]}
        else:
            if all(reach[q] == _ONLY_IDENTITY for q in support):
                continue
            for q in support:
                reach[q] = _ALL_PAULIS
        kept.append(index)
    kept.reverse()
    return kept


def _restrict_terms(
    terms: Sequence[tuple[str, complex]], qubits: Sequence[int]
) -> list[tuple[str, complex]]:
    keep = set(qubits)
    for label, _ in terms:
        outside = [i for i, ch in enumerate(label) if ch != "I" and i not in keep]
        if outside:
            raise OracleError(
                f"observable term {label!r} has support on qubit(s) {outside} outside "
                "the cone; the cone is seeded from the observable's support, so this "
                "cannot happen unless the two were computed from different observables."
            )
    return [("".join(label[q] for q in qubits), c) for label, c in terms]


def _pauli_propagation_exact(
    spec: CircuitSpec, terms: Sequence[tuple[str, complex]], labels: str
) -> complex:
    """Untruncated Heisenberg propagation followed by a product-state contraction."""
    observable = PauliSum.from_strings(
        {label: coefficient for label, coefficient in terms}, num_qubits=spec.num_qubits
    )
    evolved = observable.propagate(spec.to_circuit(), None, direction="heisenberg")
    return complex(evolved.expectation(_engine_state_argument(labels)))


def light_cone_exact(
    circuit: Any,
    observable: Any,
    n_steps: int | None = None,
    *,
    initial_state: Any = "z+",
    method: str = "auto",
    commutation_aware: bool = True,
    max_statevector_qubits: int = DEFAULT_MAX_STATEVECTOR_QUBITS,
    atol: float = 1e-10,
) -> complex:
    """Exact `<psi| U^dagger O U |psi>` by causal-cone reduction, at any `n`.

    The cone is computed by `light_cone` (see it for why dropping out-of-cone
    gates is exact), the circuit and observable are restricted and renumbered
    onto it, and the reduced problem is evaluated **exactly**:

    - `method="statevector"` -- dense Aer simulation of the reduced circuit.
      Exact up to floating point at any depth, but capped by
      `max_statevector_qubits` (memory `2**cone * 16` bytes). Measured on the
      reference host at five kicked-Ising steps, `theta_h = 0.6`: the weight-1
      `Z_62` cone (19 qubits) takes 1.7 s, and raising the cap to 30 puts the
      weight-10 cone in reach at 125 s and 16.1 GiB peak RSS. The weight-17 cone
      (59 qubits) is out of reach for this path and, at that depth, for the
      Pauli path too.
    - `method="pauli"` -- untruncated Pauli propagation (`policy=None`) on the
      cone, contracted with `PauliSum.expectation`. Exact up to floating point
      at any cone size; the cost is the *term count*, which grows with the
      number of anticommuting rotations inside the cone and is bounded above
      only by `4**cone`. So this path is "fine at any n" exactly while the term
      count stays inside memory, and **depth, not n, is the budget**: measured
      on the reference host, `Z_62` through 5 kicked-Ising Trotter steps at
      `theta_h = 0.6` peaks at 8.256e7 terms (~3 GiB, 13.4 s), against 1.7 s for
      the same answer on the statevector path over its 19-qubit cone (the two
      agree to 8.0e-15).
      Two special angles are cheap, and only two: `theta = 0` and `theta = pi`,
      where `sin(theta/2)` / `cos(theta/2)` is *exactly* zero and the engine's
      merge drops the dead branch. A Clifford `theta = ±pi/2` is **not** cheap
      here -- the branch coefficient is `cos(pi/2) = 6.1e-17`, not `0`, so the
      sum fans out even though the exact answer is a single Pauli string. Use
      `stim_clifford_exact` at the Clifford points; that is what it is for.
    - `method="both"` -- run both and require agreement within `atol`, raising
      `OracleError` on a mismatch. Only usable when the cone fits the
      statevector cap; this is the strongest form of the check, since the two
      paths share no simulation code.
    - `method="auto"` (default) -- `"statevector"` if the cone fits under
      `max_statevector_qubits` and the circuit is unitary, else `"pauli"`.

    `n_steps` is optional context (see `LightCone.n_steps`): it is carried into
    the cone report and into the text of a `ConeTooLarge`, and never changes the
    result. `commutation_aware` selects which of `light_cone`'s two cones to
    reduce onto; both are exact, and the default is much the tighter one.

    Noise channels are allowed only on the Pauli path (the statevector oracle is
    unitary-only, so `method="auto"` routes a noisy circuit there). Product
    states, uniform or not, work on both.
    """
    if method not in ("auto", "statevector", "pauli", "both"):
        raise ValueError(
            f"method must be 'auto', 'statevector', 'pauli' or 'both', got {method!r}"
        )
    spec = as_circuit_spec(circuit)
    terms = pauli_terms(observable, spec.num_qubits)
    if not terms:
        return 0j
    labels = _normalize_initial_state(initial_state, spec.num_qubits)
    cone = light_cone(spec, terms, n_steps, commutation_aware=commutation_aware)

    if cone.size == 0:
        # An identity observable: every gate is outside the cone, and the
        # expectation is the coefficient sum in any state.
        return sum(c for _, c in terms)

    reduced_spec = spec.restrict(cone.qubits)
    reduced_terms = _restrict_terms(terms, cone.qubits)
    reduced_labels = "".join(labels[q] for q in cone.qubits)

    if method == "auto":
        method = (
            "statevector"
            if cone.size <= max_statevector_qubits and reduced_spec.is_unitary
            else "pauli"
        )

    values: dict[str, complex] = {}
    if method in ("statevector", "both"):
        if cone.size > max_statevector_qubits:
            raise ConeTooLarge(
                f"light_cone_exact: {cone.describe()} exceeds "
                f"max_statevector_qubits={max_statevector_qubits}. Use method='pauli' "
                "(untruncated propagation, bounded by term count rather than 2**n), or "
                "raise the cap deliberately."
            )
        values["statevector"] = statevector_expectation(
            reduced_spec,
            reduced_terms,
            reduced_labels,
            max_qubits=max_statevector_qubits,
        )
    if method in ("pauli", "both"):
        values["pauli"] = _pauli_propagation_exact(
            reduced_spec, reduced_terms, reduced_labels
        )

    if method == "both":
        difference = abs(values["statevector"] - values["pauli"])
        if difference > atol:
            raise OracleError(
                "light_cone_exact method='both' disagreement: statevector "
                f"{values['statevector']!r} vs untruncated Pauli propagation "
                f"{values['pauli']!r} (|difference| = {difference:.3e} > atol={atol:.3e}) "
                f"for {cone.describe()}"
            )
        return values["statevector"]
    return values[method]


# =============================================================================
# 4. Published reference loader
# =============================================================================

#: Header fields every reference file must carry. Plan §6 Part 0.3:
#: "CSV/JSON with mandatory provenance header: source, method, accuracy".
REQUIRED_PROVENANCE_FIELDS = ("source", "method", "accuracy")


@dataclass(frozen=True)
class PublishedReference:
    """One provenance-tagged reference file.

    `provenance` always carries at least `source`, `method` and `accuracy` --
    the loader refuses the file otherwise. `rows` is the tabular payload: for a
    CSV, one dict per data row keyed by the header row; for a JSON file, the
    contents of its `"data"` list (or `[]` when it holds something else, which
    `payload` then exposes unchanged).
    """

    name: str
    path: Path
    provenance: dict[str, Any]
    rows: tuple[Mapping[str, Any], ...] = ()
    payload: Any = None
    fields: tuple[str, ...] = field(default_factory=tuple)

    def column(self, name: str, dtype=float) -> np.ndarray:
        """One column of `rows`, converted with `dtype` (default `float`)."""
        try:
            return np.array([dtype(row[name]) for row in self.rows])
        except KeyError as exc:
            raise KeyError(
                f"{self.path.name}: no column {name!r}; columns are {list(self.fields)}"
            ) from exc


def _reference_candidates(directory: Path, name: str) -> list[Path]:
    given = Path(name)
    if given.suffix:
        return [directory / given]
    return [directory / f"{name}{suffix}" for suffix in (".json", ".csv")]


def _parse_csv_reference(path: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    """Split a reference CSV into its `# key: value` header and its data rows."""
    provenance: dict[str, Any] = {}
    data_lines: list[str] = []
    for raw in path.read_text().splitlines():
        if raw.startswith("#"):
            body = raw.lstrip("#").strip()
            if not body:
                continue
            if ":" not in body:
                raise OracleError(
                    f"{path}: header comment {raw!r} is not of the form '# key: value'"
                )
            key, value = body.split(":", 1)
            provenance[key.strip().lower()] = value.strip()
        elif raw.strip():
            data_lines.append(raw)
    reader = csv.DictReader(data_lines)
    rows = [dict(row) for row in reader]
    return provenance, rows, list(reader.fieldnames or [])


def load_published_reference(
    name: str, *, directory: Path | str | None = None
) -> PublishedReference:
    """Load a provenance-tagged reference file from `examples/data/references/`.

    `name` is the file's stem (`"begusic2023_exact"`) or its full name
    (`"begusic2023_exact.csv"`); with no suffix, `.json` then `.csv` are tried.

    **The provenance header is mandatory.** A JSON file must have a top-level
    `"provenance"` object, a CSV file leading `# key: value` comment lines, and
    either must record `source`, `method` and `accuracy`. A file missing any of
    them is refused with an `OracleError` -- the loader is the enforcement point
    for the suite's global rule 1, so a reference value can never enter a plot
    or a test without a traceable origin.

    The directory ships with a README and **no data files**: nothing has been
    fetched into this repo. See `examples/data/references/README.md` for the
    upstream pointers and the exact header formats.
    """
    directory = REFERENCES_DIR if directory is None else Path(directory)
    candidates = _reference_candidates(directory, name)
    path = next((c for c in candidates if c.exists()), None)
    if path is None:
        available = (
            sorted(p.name for p in directory.iterdir() if p.suffix in (".json", ".csv"))
            if directory.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"no reference file for {name!r} in {directory} (tried "
            f"{[c.name for c in candidates]}); available: {available or '(none)'}. "
            "Reference files are not checked in -- see the directory's README.md for "
            "the upstream sources and the required provenance header."
        )

    if path.suffix == ".json":
        document = json.loads(path.read_text())
        if not isinstance(document, Mapping):
            raise OracleError(f"{path}: top level must be a JSON object")
        provenance = document.get("provenance")
        if not isinstance(provenance, Mapping):
            raise OracleError(
                f'{path}: missing the mandatory top-level "provenance" object'
            )
        provenance = {str(k).lower(): v for k, v in provenance.items()}
        payload = document.get("data")
        rows = tuple(payload) if isinstance(payload, list) else ()
        fields = tuple(rows[0].keys()) if rows and isinstance(rows[0], Mapping) else ()
    elif path.suffix == ".csv":
        provenance, row_list, field_names = _parse_csv_reference(path)
        rows = tuple(row_list)
        payload = rows
        fields = tuple(field_names)
    else:
        raise OracleError(
            f"{path}: unsupported reference format {path.suffix!r}; expected .json or .csv"
        )

    missing = [f for f in REQUIRED_PROVENANCE_FIELDS if not str(provenance.get(f, "")).strip()]
    if missing:
        raise OracleError(
            f"{path}: provenance header is missing {missing} (required: "
            f"{list(REQUIRED_PROVENANCE_FIELDS)}). Every reference value in this suite "
            "must be traceable to a fetched source, the method that produced it, and a "
            "stated accuracy; see examples/data/references/README.md."
        )

    return PublishedReference(
        name=path.stem,
        path=path,
        provenance=dict(provenance),
        rows=rows,
        payload=payload,
        fields=fields,
    )


# =============================================================================
# 5. Optional: tsim low-magic oracle
# =============================================================================


def tsim_low_magic_exact(circuit: Any, observable: Any, *, initial_state: Any = None):
    """Low-magic exact oracle via `tsim` -- **optional**, and not wired up.

    `tsim` is not a dependency of this repo and is not installed in the
    development environment, so this function does exactly two honest things:

    - `SkipOracle` when `tsim` cannot be imported. That is the supported path:
      the suite runs without this oracle, and a caller should treat the skip the
      way `pytest.importorskip` treats a missing package.
    - `NotImplementedError` when it *can* be imported. Writing the call against
      an API nobody here has run would be a fabricated binding, which the
      suite's global rule 5 forbids ("missing capabilities are escalated as
      named dependencies, never silently approximated"). Wiring it up means
      installing tsim, checking its actual entry points, and pinning the result
      against `statevector_expectation` on a small low-magic circuit in the same
      commit.
    """
    try:
        import tsim  # noqa: F401
    except ImportError as exc:
        raise SkipOracle(
            "tsim is not installed, so the low-magic oracle is unavailable. It is an "
            "optional cross-check: every benchmark in this suite has a statevector, "
            "stabilizer, or light-cone oracle that does not need it."
        ) from exc
    raise NotImplementedError(
        "tsim is installed but this oracle is not wired up: the binding was left "
        "unwritten rather than guessed against an unrun API (plan §7 rule 5). "
        "Implement it here, and pin it against statevector_expectation on a small "
        "low-magic circuit in the same commit."
    )
