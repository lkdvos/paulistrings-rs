"""Observable builders for the `examples/` showcase suite.

Handoff item P0b; see `research/plans/2026-08-31-examples-benchmarks-suite.md`
§6 Part 0.2.

Everything here is built through `PauliSum.from_strings`, so the convention is
the repo-wide Hermitian one (CLAUDE.md §Known gaps): a key is a string of
`I/X/Y/Z` with character `i` addressing **qubit `i`**, `Y` maps to the symplectic
key `(x=1, z=1)` with no phase factor, and the coefficient multiplies the
literal Pauli string. A Hermitian observable therefore has real coefficients.

Published operator supports
---------------------------
The kicked-Ising observables of the IBM 127-qubit utility experiment -- Y. Kim
et al., "Evidence for the utility of quantum computing before fault tolerance",
Nature 618, 500-505 (2023), doi 10.1038/s41586-023-06096-3 -- are **not written
down in this module**. They are loaded from
`examples/data/kim2023_observables.json`, whose `provenance` block records the
URLs that were fetched, the figure panel title each support was read from, four
corroborating reproduction papers, and the retrieval date.
`examples/data/README.md` carries the same record in prose.

This split is deliberate, and enforced by `kim2023_operator` raising rather than
falling back: per the suite's global rule 1 ("no fabricated reference values"),
a published support that cannot be traced to a fetched source must fail loudly
at the point of use, not be reconstructed from memory into a literal that later
readers will mistake for a citation.

Four observables are recorded, on the Eagle 0..126 numbering: `weight_1_z62`
(Fig. 4b, 20 steps), `weight_10` (Fig. 3b), `weight_17` (Fig. 3c) and
`weight_17_modified` (Fig. 4a). The last two share their X support and differ by
swapping the Y and Z sets -- they are RX(pi/2) conjugates -- so confusing them
gives a silently wrong observable of the same weight.

The three stabilizer entries are not merely transcribed: the paper states they
are stabilizers of the theta_h = pi/2 Clifford circuit obtained by evolving
`Z_13` / `Z_58` for five Trotter steps, so
`python/paulistrings/tests/test_examples_circuits.py` re-derives every support
and sign by propagating those seeds through `circuits.heavy_hex_kicked_ising`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

from paulistrings import PauliSum

__all__ = [
    "KIM2023_OBSERVABLES_PATH",
    "PAULI_CHARS",
    "canonical_z_127",
    "kim2023_operator",
    "kim2023_provenance",
    "pauli_string",
    "pauli_sum_from_support",
    "pauli_weight",
    "single_z",
    "sparse_pauli_sum",
    "weight_10_operator",
    "weight_17_modified_operator",
    "weight_17_operator",
    "xxz_hamiltonian",
]

PAULI_CHARS = frozenset("IXYZ")

#: Provenance-tagged supports of the published kicked-Ising observables.
KIM2023_OBSERVABLES_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "kim2023_observables.json"
)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def pauli_string(support: Mapping[int, str], num_qubits: int) -> str:
    """Render `{qubit: 'X'|'Y'|'Z'}` as a full-length `I/X/Y/Z` string.

    Character `i` of the result addresses qubit `i` -- the convention
    `PauliSum.from_strings` uses.
    """
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
    chars = ["I"] * num_qubits
    for qubit, op in support.items():
        if not 0 <= qubit < num_qubits:
            raise ValueError(f"qubit {qubit} is outside 0..{num_qubits - 1}")
        letter = str(op).upper()
        if letter not in PAULI_CHARS:
            raise ValueError(f"qubit {qubit}: expected one of I/X/Y/Z, got {op!r}")
        chars[qubit] = letter
    return "".join(chars)


def pauli_weight(string: str) -> int:
    """Number of non-identity characters in a Pauli string."""
    return sum(1 for ch in string if ch != "I")


def pauli_sum_from_support(
    support: Mapping[int, str], num_qubits: int, coefficient: complex = 1.0
) -> PauliSum:
    """A one-term `PauliSum`: `coefficient` times the string given by `support`."""
    return PauliSum.from_strings(
        {pauli_string(support, num_qubits): coefficient}, num_qubits=num_qubits
    )


def single_z(q: int, n: int) -> PauliSum:
    """The single-qubit observable `Z_q` on `n` qubits, coefficient 1."""
    return pauli_sum_from_support({q: "Z"}, n)


# --------------------------------------------------------------------------
# Published kicked-Ising observables
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _kim2023_data() -> dict:
    if not KIM2023_OBSERVABLES_PATH.exists():
        raise FileNotFoundError(
            f"{KIM2023_OBSERVABLES_PATH} is missing. The published weight-10 / "
            "weight-17 supports are only ever loaded from that provenance-tagged "
            "file; they are deliberately not hard-coded in this module (see the "
            "module docstring and examples/data/README.md)."
        )
    return json.loads(KIM2023_OBSERVABLES_PATH.read_text())


def kim2023_provenance() -> dict:
    """The `provenance` block of `examples/data/kim2023_observables.json`.

    Report writers should embed this verbatim next to any number derived from
    these observables.
    """
    return dict(_kim2023_data()["provenance"])


def kim2023_operator(name: str, num_qubits: int = 127) -> PauliSum:
    """A published kicked-Ising observable, by its name in the data file.

    Names come from the data file's `observables` map (e.g. `"weight_1"`,
    `"weight_10"`, `"weight_17"`). Qubit indices in the file are the IBM Eagle
    device numbering `0..126`, the same numbering as
    `examples/data/heavy_hex_127.edges`, so an observable and the lattice it was
    measured on need no re-indexing.

    A name whose support could not be traced to a fetched source is absent from
    the file, and this call raises `KeyError` naming the provenance gap -- it is
    never approximated.
    """
    data = _kim2023_data()
    observables = data["observables"]
    if name not in observables:
        gaps = data.get("unverified", {})
        detail = gaps.get(name)
        available = ", ".join(sorted(observables)) or "(none)"
        message = (
            f"{name!r} is not in {KIM2023_OBSERVABLES_PATH.name}; available: {available}."
        )
        if detail:
            message += f" Recorded provenance gap: {detail}"
        else:
            message += (
                " Add it only with a fetched source recorded in the file's "
                "`provenance` block -- never from recollection."
            )
        raise KeyError(message)
    entry = observables[name]
    support = {int(q): op for q, op in entry["support"].items()}
    return pauli_sum_from_support(support, num_qubits)


def canonical_z_127() -> PauliSum:
    """`Z_62` on 127 qubits -- the weight-1 observable of the paper's Fig. 4b.

    Loaded through the provenance-tagged data file like the other published
    observables (`single_z(62, 127)` builds the same operator without the
    provenance detour).
    """
    return kim2023_operator("weight_1_z62")


def weight_10_operator(num_qubits: int = 127) -> PauliSum:
    """The weight-10 observable of the Kim et al. utility experiment (Fig. 3b).

    `X_{13,29,31} Y_{9,30} Z_{8,12,17,28,32}`, the five-step Clifford evolution
    of `Z_13`; a stabilizer with eigenvalue +1 at `theta_h = pi/2`.
    """
    return kim2023_operator("weight_10", num_qubits)


def weight_17_operator(num_qubits: int = 127) -> PauliSum:
    """The weight-17 observable of the Kim et al. utility experiment (Fig. 3c).

    The five-step Clifford evolution of `Z_58`; a stabilizer with eigenvalue -1
    at `theta_h = pi/2`. Not to be confused with `weight_17_modified_operator`
    (Fig. 4a), which shares its X support.
    """
    return kim2023_operator("weight_17", num_qubits)


def weight_17_modified_operator(num_qubits: int = 127) -> PauliSum:
    """The *modified* weight-17 observable of the paper's Fig. 4a.

    The Fig. 3c operator with its Y and Z sets swapped -- the stabilizer of the
    five-step circuit plus one further single-qubit X-rotation layer
    (`heavy_hex_kicked_ising(..., final_x_layer=True)`), eigenvalue -1.
    """
    return kim2023_operator("weight_17_modified", num_qubits)


# --------------------------------------------------------------------------
# Hamiltonians
# --------------------------------------------------------------------------


def xxz_hamiltonian(n: int, Jz: float = 1.0, *, coupling: float = 1.0) -> PauliSum:
    """The open XXZ chain Hamiltonian as a sparse `PauliSum`.

        H = coupling * sum_{i=0}^{n-2} ( X_i X_{i+1} + Y_i Y_{i+1} + Jz · Z_i Z_{i+1} )

    Matches the Hamiltonian `circuits.xxz_chain_trotter(n, ..., Jz=Jz)`
    Trotterizes (there with `coupling = 1`). `Jz = 0` gives the free XX/XY
    chain. The `Jz · Z Z` terms are omitted entirely when `Jz == 0` -- a
    zero-coefficient term would be dropped by the engine's merge phase anyway,
    so including it would only make the term count depend on how the sum was
    built.

    Term count is `3·(n-1)` for `Jz != 0`, `2·(n-1)` for `Jz == 0`.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2 for a chain with at least one bond, got {n}")
    terms: dict[str, complex] = {}
    for i in range(n - 1):
        for op, weight in (("X", coupling), ("Y", coupling), ("Z", coupling * Jz)):
            if weight == 0.0:
                continue
            terms[pauli_string({i: op, i + 1: op}, n)] = complex(weight)
    return PauliSum.from_strings(terms, num_qubits=n)


def sparse_pauli_sum(
    supports: Sequence[Mapping[int, str]],
    coefficients: Sequence[complex],
    num_qubits: int,
) -> PauliSum:
    """A `PauliSum` from parallel sequences of supports and coefficients."""
    if len(supports) != len(coefficients):
        raise ValueError(
            f"got {len(supports)} supports and {len(coefficients)} coefficients"
        )
    terms: dict[str, complex] = {}
    for support, coefficient in zip(supports, coefficients):
        key = pauli_string(support, num_qubits)
        if key in terms:
            raise ValueError(f"duplicate Pauli string {key!r}")
        terms[key] = complex(coefficient)
    return PauliSum.from_strings(terms, num_qubits=num_qubits)
