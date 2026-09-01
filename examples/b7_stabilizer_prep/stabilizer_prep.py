"""Reusable pieces for showcase B7 -- stabilizer preparation, then estimation.

Handoff item B7; adapted spec in
`research/plans/2026-08-31-examples-benchmarks-suite.md` §6 Part B (decision
D13, and §3's A8-ii row). The narrative lives in `README.md` next to this file
and the driver in `run_b7.py`; this module holds everything both the driver and
the CI gate (`python/paulistrings/tests/test_showcase_b7.py`) need, so neither
copy-pastes a lattice, a stabilizer formula or a dense reference.

Three groups of things:

**Lattices and stabilizer preparations.** `grid_edges` / `grid_adjacency` build
an open `rows x cols` square lattice; `cluster_state_stabilizers` writes down
its cluster-state generators `K_q = X_q prod_{q' in N(q)} Z_{q'}` *from the edge
list* (never tabulated), and `cluster_prep_stim` builds the stim circuit that
prepares that state (`H` everywhere, then one `CZ` per edge, grouped into
disjoint-support colour layers by `common.circuits.heavy_hex_edge_coloring`).
`cluster_prep_circuit` is the same preparation as a `paulistrings.Circuit`, for
the paths that must run without `stim`. `identity_padding` appends
provably-identity Clifford rounds -- `H H`, `S S S S`, `CNOT CNOT`, `CZ CZ` --
which is how the driver grows the *depth* of a preparation without changing the
*state* it prepares, and `random_clifford_prep` builds an unstructured deep
Clifford preparation for contrast.

**The dense reference.** `dense_state` runs a `paulistrings.Circuit`'s gate list
on a `2^n` state vector with nothing but numpy, `dense_pauli_expectation`
contracts a Pauli label against it, and `dense_expectation` composes the two
into `<0| C^dagger O C |0>` for a composed preparation-plus-tail circuit. The
gate list comes from `Circuit.gates`, so the *same* circuit object the engine
propagates drives the reference -- no second transcription. `dense_projector_state`
is the independent second route for the *generators* alone: the rank-1
projector `Pi = prod_i (I + s_i G_i)/2`, which reads the signed generator
strings and never looks at the circuit that produced them.

Conventions (both inherited, both easy to get silently wrong):

* **Qubit 0 is the most significant tensor factor**, matching
  `PauliSum.from_strings` and `Circuit.unitary_2q(q0, q1, m)` -- so qubit `q`'s
  bit in a basis index is bit `n-1-q`, and axis `q` of the reshaped
  `[2]*n` tensor. `examples/common/oracles.py`'s trap 1 is the qiskit-facing
  half of the same asymmetry; nothing here talks to qiskit.
* **The Hermitian-Y convention** (CLAUDE.md §Known gaps): `Y` is
  `[[0, -i], [i, 0]]` with no phase factor, in labels, generators and gate
  matrices alike. stim uses the same Hermitian `Y`, so
  `interop.stabilizers_from_stim` needs no phase reconciliation and neither
  does anything here.
* `pauli_rotation(pauli, qubits, theta)` is `U = exp(-i theta P / 2)` with
  `pauli[k]` on `qubits[k]`, which is `cos(theta/2) I - i sin(theta/2) P` --
  how `_gate_matrix` builds it.

Everything here is numpy-only at import time; `stim` is imported lazily inside
the two functions that need it, so the CI gate can exercise the dense routes
with no optional dependency installed.
"""

from __future__ import annotations

import functools
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from paulistrings import Circuit, PauliSum

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from common.circuits import heavy_hex_edge_coloring  # noqa: E402

__all__ = [
    "MAX_DENSE_QUBITS",
    "MAX_PROJECTOR_QUBITS",
    "cluster_prep_circuit",
    "cluster_prep_stim",
    "cluster_state_stabilizers",
    "dense_expectation",
    "dense_pauli_expectation",
    "dense_projector_state",
    "dense_state",
    "grid_adjacency",
    "grid_edges",
    "identity_padding",
    "pauli_label",
    "random_clifford_prep",
    "single_z_generators",
]

#: Refusal threshold for every dense route here. A `2^n` complex state vector
#: is 16 bytes per amplitude, so n=20 is 16 MiB and n=24 is 256 MiB -- but the
#: projector route in `dense_projector_state` is a `2^n x 2^n` *matrix*, i.e.
#: 16 GiB at n=15. The bound is deliberately low: nothing in this showcase
#: needs a dense answer above 12 qubits, and the point of the whole exercise is
#: that the Pauli-propagation path does not pay `2^n` at all.
MAX_DENSE_QUBITS = 14

#: Refusal threshold for `dense_projector_state`, which builds a `2^n x 2^n`
#: matrix rather than a vector: 67 MiB at n=11, 1.1 GiB at n=13.
MAX_PROJECTOR_QUBITS = 11


# =============================================================================
# Lattices and stabilizer preparations
# =============================================================================


def grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    """Edges of an open `rows x cols` square lattice, row-major qubit indexing.

    Qubit `r*cols + c` sits at row `r`, column `c`; each edge is `(min, max)`
    and the list is sorted, so it is a canonical description of the lattice.
    A 2D lattice (rather than a chain) is the point: the cluster state on it is
    a universal resource state and is *not* a product state in any local basis,
    so contracting against it genuinely needs the stabilizer path rather than
    `expectation(state=...)`.
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be >= 1, got {rows}x{cols}")
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return sorted(edges)


def grid_adjacency(
    num_qubits: int, edges: Sequence[tuple[int, int]]
) -> dict[int, list[int]]:
    """Neighbour lists for `edges`, every qubit present (isolated ones empty)."""
    adj: dict[int, list[int]] = {q: [] for q in range(num_qubits)}
    for a, b in edges:
        if not 0 <= a < num_qubits or not 0 <= b < num_qubits:
            raise ValueError(f"edge ({a}, {b}) is outside qubits 0..{num_qubits - 1}")
        adj[a].append(b)
        adj[b].append(a)
    return {q: sorted(v) for q, v in adj.items()}


def pauli_label(support: Mapping[int, str], num_qubits: int) -> str:
    """A full-length Pauli label with `support[q]` on qubit `q`, `I` elsewhere.

    The same spelling `PauliSum.from_strings` and
    `PauliSum.expectation_stabilizer` take (character `i` on qubit `i`).
    """
    chars = ["I"] * num_qubits
    for q, ch in support.items():
        if not 0 <= q < num_qubits:
            raise ValueError(f"qubit {q} is outside 0..{num_qubits - 1}")
        if ch not in "IXYZ":
            raise ValueError(f"unexpected Pauli character {ch!r} on qubit {q}")
        chars[q] = ch
    return "".join(chars)


def cluster_state_stabilizers(
    num_qubits: int, edges: Sequence[tuple[int, int]]
) -> list[str]:
    """Signed cluster-state generators `K_q = X_q prod_{q' in N(q)} Z_{q'}`.

    Derived from the edge list, not tabulated: the graph state prepared by
    `H` on every qubit followed by one `CZ` per edge is stabilized by exactly
    these `n` operators, since `CZ_{ab}` conjugates `X_a` into `X_a Z_b` and
    leaves every `Z` alone. All signs are `+`.

    Returned in `expectation_stabilizer`'s signed-string form, so a caller can
    hand them straight to the contraction *or* compare them against what
    `interop.stabilizers_from_stim` reads out of the preparation circuit. They
    are two independent descriptions of one state: the second is stim's
    tableau, the first is this formula.
    """
    adj = grid_adjacency(num_qubits, edges)
    generators = []
    for q in range(num_qubits):
        support = {q: "X"}
        for neighbour in adj[q]:
            support[neighbour] = "Z"
        generators.append("+" + pauli_label(support, num_qubits))
    return generators


def single_z_generators(num_qubits: int) -> list[str]:
    """The `|0...0>` generators `+Z_q`, one per qubit.

    The degenerate case of a stabilizer state: contracting against these must
    reproduce `PauliSum.expectation(state="z+")` exactly, which is the
    special-case check in `run_b7.py` Part 1 and in the CI gate.
    """
    return [
        "+" + pauli_label({q: "Z"}, num_qubits) for q in range(num_qubits)
    ]


def cluster_prep_stim(
    num_qubits: int,
    edges: Sequence[tuple[int, int]],
    *,
    color_layers: bool = True,
):
    """The stim circuit preparing the cluster state on `edges`.

    `H` on every qubit, then one `CZ` per edge. With `color_layers=True`
    (default) the `CZ`s are emitted grouped into proper matchings by
    `common.circuits.heavy_hex_edge_coloring`, i.e. as disjoint-support
    hardware layers with a `TICK` between them -- all the `CZ`s commute, so the
    grouping cannot change the state, only the depth the circuit reads as.

    Needs `stim` (imported here, not at module scope).
    """
    import stim

    circuit = stim.Circuit()
    circuit.append("H", list(range(num_qubits)))
    circuit.append("TICK", [])
    groups = (
        heavy_hex_edge_coloring(edges) if color_layers else [sorted(edges)]
    )
    for group in groups:
        for a, b in group:
            circuit.append("CZ", [a, b])
        circuit.append("TICK", [])
    return circuit


def cluster_prep_circuit(
    num_qubits: int,
    edges: Sequence[tuple[int, int]],
    *,
    color_layers: bool = True,
) -> Circuit:
    """The same preparation as `cluster_prep_stim`, as a `paulistrings.Circuit`.

    Two spellings of one circuit: this one needs no optional dependency, so the
    CI gate can exercise the dense routes and the closed form without `stim`
    installed, while `run_b7.py` goes through
    `interop.circuit_from_stim(cluster_prep_stim(...))` -- which additionally
    exercises the importer. That the two agree gate for gate is pinned by
    `test_showcase_b7.py::test_the_two_cluster_preparations_are_the_same_circuit`.
    """
    circuit = Circuit(num_qubits)
    for q in range(num_qubits):
        circuit.h(q)
    groups = heavy_hex_edge_coloring(edges) if color_layers else [sorted(edges)]
    for group in groups:
        for a, b in group:
            circuit.cz(a, b)
    return circuit


def identity_padding(num_qubits: int, rounds: int, rng: np.random.Generator):
    """`rounds` Clifford rounds that multiply to the identity, as a stim circuit.

    Each round picks one of four provably-identity patterns and a random target
    (or target pair): `H H`, `S S S S`, `CNOT CNOT`, `CZ CZ`. Appending this to
    a preparation circuit therefore leaves the *prepared state* -- and so every
    stabilizer generator, and so every expectation value downstream -- exactly
    unchanged while growing the circuit's depth without bound. That is what
    makes it the honest way to measure "how much does preparation depth cost
    the estimate?": the answer must be *nothing at all*, and any drift would be
    a bug rather than physics.

    Needs `stim` (imported here, not at module scope).
    """
    import stim

    if rounds < 0:
        raise ValueError(f"rounds must be >= 0, got {rounds}")
    if num_qubits < 2:
        raise ValueError(f"identity padding needs >= 2 qubits, got {num_qubits}")
    circuit = stim.Circuit()
    for _ in range(rounds):
        kind = int(rng.integers(0, 4))
        if kind == 0:
            q = int(rng.integers(0, num_qubits))
            circuit.append("H", [q])
            circuit.append("H", [q])
        elif kind == 1:
            q = int(rng.integers(0, num_qubits))
            for _ in range(4):
                circuit.append("S", [q])
        else:
            a = int(rng.integers(0, num_qubits))
            b = int(rng.integers(0, num_qubits - 1))
            if b >= a:
                b += 1
            name = "CNOT" if kind == 2 else "CZ"
            circuit.append(name, [a, b])
            circuit.append(name, [a, b])
    return circuit


def random_clifford_prep(num_qubits: int, depth: int, rng: np.random.Generator):
    """An unstructured depth-`depth` Clifford preparation, as a stim circuit.

    Per round: a uniform choice from `{H, S, X, Z, I}` on every qubit, then a
    coin-flipped `CNOT` on each even brickwork rung and `CZ` on each odd one.
    Deliberately *not* a uniformly random Clifford (`stim.Tableau.random`) --
    the point is a circuit with a tunable gate count, so `run_b7.py` can show
    the preparation cost growing with depth while the estimate's cost does not.

    Needs `stim` (imported here, not at module scope).
    """
    import stim

    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    circuit = stim.Circuit()
    for _ in range(depth):
        for q in range(num_qubits):
            gate = ["H", "S", "X", "Z", "I"][int(rng.integers(0, 5))]
            if gate != "I":
                circuit.append(gate, [q])
        for start, name in ((0, "CNOT"), (1, "CZ")):
            for q in range(start, num_qubits - 1, 2):
                if rng.integers(0, 2):
                    circuit.append(name, [q, q + 1])
        circuit.append("TICK", [])
    return circuit


# =============================================================================
# The dense reference
# =============================================================================

_PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    # Hermitian Y, no phase factor (CLAUDE.md §Known gaps).
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

_INV_SQRT2 = 1.0 / math.sqrt(2.0)

#: Fixed Clifford matrices, in the "first listed qubit is the most significant
#: tensor factor" convention (see the module docstring).
_FIXED_GATES: dict[str, np.ndarray] = {
    "h": _INV_SQRT2 * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex),
    "s": np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex),
    "sdg": np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=complex),
    "x": _PAULI_MATRICES["X"],
    "y": _PAULI_MATRICES["Y"],
    "z": _PAULI_MATRICES["Z"],
    "cnot": np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=complex,
    ),
    "cz": np.diag(np.array([1.0, 1.0, 1.0, -1.0], dtype=complex)),
    "swap": np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=complex,
    ),
}

_AXIS_OF = {"rx": "X", "ry": "Y", "rz": "Z"}


def _pauli_matrix(label: str) -> np.ndarray:
    """`kron` of `label`'s characters, first character most significant."""
    return functools.reduce(
        np.kron, [_PAULI_MATRICES[ch] for ch in label]
    )


def _rotation_matrix(pauli: str, theta: float) -> np.ndarray:
    """`exp(-i theta P / 2) = cos(theta/2) I - i sin(theta/2) P`."""
    dim = 2 ** len(pauli)
    return (
        math.cos(theta / 2.0) * np.eye(dim, dtype=complex)
        - 1.0j * math.sin(theta / 2.0) * _pauli_matrix(pauli)
    )


def _gate_matrix(gate: Mapping[str, Any]) -> np.ndarray:
    """The unitary of one `Circuit.gates` entry, as a `2^k x 2^k` matrix.

    Covers exactly the unitary half of task-JSON schema v1's vocabulary (see
    `examples/common/oracles.py`'s `_GATE_SPECS`); a noise channel has no
    unitary and raises, since a dense *state-vector* reference cannot represent
    one.
    """
    name = str(gate["name"])
    if name in _FIXED_GATES:
        return _FIXED_GATES[name]
    if name in _AXIS_OF:
        return _rotation_matrix(_AXIS_OF[name], float(gate["theta"]))
    if name == "pauli_rotation":
        return _rotation_matrix(str(gate["pauli"]), float(gate["theta"]))
    if name in ("unitary_1q", "unitary_2q"):
        matrix = gate["matrix"]
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(
                [[complex(re, im) for re, im in row] for row in matrix], dtype=complex
            )
        return np.asarray(matrix, dtype=complex)
    raise ValueError(
        f"stabilizer_prep's dense reference has no unitary for gate {name!r}; it is a "
        "state-vector reference, so noise channels are out of scope (use a "
        "density-matrix route, which this module does not provide)"
    )


def _apply_gate(
    tensor: np.ndarray, matrix: np.ndarray, qubits: Sequence[int]
) -> np.ndarray:
    """Apply `matrix` to axes `qubits` of a `[2]*n` state tensor.

    `qubits[0]` is the matrix's most significant tensor factor, matching
    `Circuit.unitary_2q(q0, q1, m)` and `pauli_rotation`'s `pauli[k]` ->
    `qubits[k]` correspondence.
    """
    k = len(qubits)
    operator = matrix.reshape([2] * (2 * k))
    # Contract the operator's *input* axes (the second half) with the state's
    # qubit axes; tensordot puts the surviving output axes first, so they are
    # moved back into place afterwards.
    out = np.tensordot(operator, tensor, axes=(range(k, 2 * k), tuple(qubits)))
    return np.moveaxis(out, range(k), tuple(qubits))


def dense_state(circuit: Circuit, initial_state: np.ndarray | None = None) -> np.ndarray:
    """Run `circuit` forward on a `2^n` state vector with numpy alone.

    Reads the gate list through `Circuit.gates` -- the same circuit object the
    engine propagates, so this reference shares its *input* with the thing it
    checks and differs only in method. `initial_state` defaults to `|0...0>`.

    Refuses above `MAX_DENSE_QUBITS`; this is the `2^n` cost the showcase
    exists to avoid, kept around only as a small-`n` oracle.
    """
    n = circuit.num_qubits
    if n > MAX_DENSE_QUBITS:
        raise ValueError(
            f"dense_state refuses n={n} > MAX_DENSE_QUBITS={MAX_DENSE_QUBITS}: a dense "
            f"state vector would need {2**n * 16 / 2**20:.1f} MiB. The whole point of "
            "the stabilizer contraction is that it never pays this."
        )
    if initial_state is None:
        tensor = np.zeros([2] * n, dtype=complex)
        tensor[(0,) * n] = 1.0
    else:
        tensor = np.asarray(initial_state, dtype=complex).reshape([2] * n)
    for gate in circuit.gates:
        qubits = [int(q) for q in gate["qubits"]]
        tensor = _apply_gate(tensor, _gate_matrix(gate), qubits)
    return tensor.reshape(-1)


def dense_pauli_expectation(state: np.ndarray, label: str) -> complex:
    """`<state| P |state>` for the Pauli label `P`, by index arithmetic.

    `O(2^n)` work and no `2^n x 2^n` matrix: a Pauli string maps basis state
    `i` to a single basis state `i ^ flip` with a phase, so the whole
    contraction is one gather plus a dot product.
    """
    n = len(label)
    state = np.asarray(state, dtype=complex).reshape(-1)
    if state.size != 1 << n:
        raise ValueError(
            f"state has {state.size} amplitudes but the label is {n} qubits "
            f"({1 << n} expected)"
        )
    bad = sorted(set(label) - set("IXYZ"))
    if bad:
        raise ValueError(f"Pauli label {label!r} has unexpected character(s) {bad}")

    flip = 0
    for q, ch in enumerate(label):
        if ch in "XY":
            flip |= 1 << (n - 1 - q)  # qubit 0 is the most significant factor

    # `P|src> = phase(src) |src ^ flip>`, so the amplitude landing in slot `j`
    # comes from `src = j ^ flip` and carries *that* index's phase -- evaluating
    # the phase at `j` instead would flip the sign once per `Y` (whose flip bit
    # makes the two indices differ exactly there).
    source = np.arange(state.size, dtype=np.int64) ^ flip
    phase = np.ones(state.size, dtype=complex)
    for q, ch in enumerate(label):
        if ch in "IX":
            continue
        bit = 1 << (n - 1 - q)
        occupied = (source & bit) != 0
        # Z|b> = (-1)^b |b>; the Hermitian Y|b> = i(-1)^b |b^1>.
        phase = phase * np.where(occupied, -1.0, 1.0)
        if ch == "Y":
            phase = phase * 1.0j
    return complex(np.vdot(state, phase * state[source]))


def dense_expectation(circuit: Circuit, observable: Any) -> complex:
    """`<0| C^dagger O C |0>` for a `Circuit` `C` and a Pauli-sum observable.

    `observable` is a `PauliSum`, a `{label: coefficient}` mapping or a bare
    label. This is the showcase's small-`n` ground truth: it knows nothing
    about stabilizer groups, generators or Pauli propagation -- it runs the
    composed preparation-plus-tail circuit on a state vector and contracts the
    observable against it.
    """
    state = dense_state(circuit)
    terms = _observable_terms(observable, circuit.num_qubits)
    return sum(
        coefficient * dense_pauli_expectation(state, label)
        for label, coefficient in terms
    )


def _observable_terms(observable: Any, num_qubits: int) -> list[tuple[str, complex]]:
    if isinstance(observable, PauliSum):
        from common.oracles import pauli_terms

        return pauli_terms(observable, num_qubits)
    if isinstance(observable, str):
        return [(observable, 1.0 + 0.0j)]
    if isinstance(observable, Mapping):
        return [(str(k), complex(v)) for k, v in observable.items()]
    raise TypeError(
        f"unsupported observable {type(observable).__name__}; expected a PauliSum, a "
        "{label: coefficient} mapping, or a label string"
    )


def dense_projector_state(generators: Iterable[str]) -> np.ndarray:
    """The state fixed by `generators`, via `Pi = prod_i (I + s_i G_i)/2`.

    A second dense route, independent of any circuit: it reads only the signed
    generator strings. For `n` independent commuting generators the projector is
    rank 1, so any column of maximal norm is the state up to normalization --
    and the rank-1 assertion is itself a check that the generator set really
    does define a unique state. Costs a `2^n x 2^n` matrix, hence the low
    `MAX_DENSE_QUBITS` ceiling; use it at `n <= 10`.

    Same construction as `python/paulistrings/tests/test_stabilizer.py`'s dense
    reference, kept here so `run_b7.py` and the showcase gate can both reach it.
    """
    generators = list(generators)
    if not generators:
        raise ValueError("dense_projector_state needs at least one generator")
    signs_labels = [_split_sign(spec) for spec in generators]
    n = len(signs_labels[0][1])
    if len(generators) != n:
        raise ValueError(
            f"a stabilizer state on {n} qubits needs exactly {n} generators, got "
            f"{len(generators)}"
        )
    if n > MAX_PROJECTOR_QUBITS:
        raise ValueError(
            f"dense_projector_state refuses n={n} > MAX_PROJECTOR_QUBITS="
            f"{MAX_PROJECTOR_QUBITS}: the projector is a 2^{n} x 2^{n} matrix "
            f"({4**n * 16 / 2**30:.2f} GiB). Use dense_state (a vector) instead, or "
            "the engine's own contraction, which pays neither."
        )
    dim = 1 << n
    projector = np.eye(dim, dtype=complex)
    for sign, label in signs_labels:
        if len(label) != n:
            raise ValueError(f"generator {label!r} is not {n} characters long")
        projector = projector @ (
            (np.eye(dim, dtype=complex) + sign * _pauli_matrix(label)) / 2.0
        )
    trace = complex(np.trace(projector))
    if abs(trace - 1.0) > 1e-9:
        raise ValueError(
            f"the projector has trace {trace!r}, not 1: {generators} do not define a "
            "single stabilizer state (dependent or anticommuting generators)"
        )
    column = int(np.argmax(np.linalg.norm(projector, axis=0)))
    state = projector[:, column]
    return state / np.linalg.norm(state)


def _split_sign(spec: str) -> tuple[float, str]:
    """`("-XX")` -> `(-1.0, "XX")`; a bare label is `+`."""
    if not isinstance(spec, str) or not spec:
        raise TypeError(f"a generator must be a non-empty string, got {spec!r}")
    if spec[0] in "+-":
        return (-1.0 if spec[0] == "-" else 1.0), spec[1:]
    return 1.0, spec
