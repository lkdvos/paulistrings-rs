"""Parameterized circuit builders for the `examples/` showcase suite.

Handoff item P0a; see `research/plans/2026-08-31-examples-benchmarks-suite.md`
§6 Part 0.1 for the adapted specification.

Two rules govern everything in this module.

**One gate per channel.** Every builder pushes exactly one gate per `Circuit`
channel and never bundles several gates into one push (plan §5, decision D10).
This engine truncates after every channel, so the channel decomposition *is*
the truncation schedule: bundling would silently change where truncation
happens and destroy per-layer term-count parity with a per-gate-truncating
reference implementation.

**Rotation convention.** `paulistrings.gates.pauli_rotation(pauli, qubits,
theta)` implements

    U = exp(-i · theta · P / 2)

(pinned by `python/paulistrings/tests/test_pauli_rotation.py`). Every angle
below is therefore *twice* the coefficient it multiplies in the generator:
`exp(-i·s·P)` is `theta = 2s`, and the kicked-Ising bond `exp(+i·(pi/4)·ZZ)` is
`theta_zz = -pi/2`. Each builder's docstring states the unitary it realizes in
`exp(...)` form so the mapping is checkable without reading the code.

The 127-qubit heavy-hex edge list is loaded from the *generated*
`examples/data/heavy_hex_127.edges` (see `examples/data/generate_heavy_hex.py`
for its provenance); it is never hard-coded here.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from paulistrings import Circuit, gates

__all__ = [
    "HEAVY_HEX_127_PATH",
    "KICKED_ISING_CLIFFORD_THETA_ZZ",
    "KICKED_ISING_DEFAULT_ORDER",
    "Edge",
    "haar_su4",
    "hardware_efficient_ansatz",
    "hardware_efficient_ansatz_num_params",
    "heavy_hex_127_edges",
    "heavy_hex_edge_coloring",
    "heavy_hex_kicked_ising",
    "heavy_hex_sublattice",
    "load_edge_list",
    "qaoa",
    "random_su4_staircase",
    "xxz_chain_trotter",
]

Edge = tuple[int, int]

#: The checked-in, generated Eagle r3 coupling map.
HEAVY_HEX_127_PATH = Path(__file__).resolve().parents[1] / "data" / "heavy_hex_127.edges"

#: The kicked-Ising ZZ angle at the Clifford point: `theta_zz = -pi/2` realizes
#: `exp(+i·(pi/4)·Z_i Z_j)`, the entangler of the IBM utility experiment.
KICKED_ISING_CLIFFORD_THETA_ZZ = -math.pi / 2


# --------------------------------------------------------------------------
# Heavy-hex lattice plumbing
# --------------------------------------------------------------------------


def load_edge_list(path: Path | str) -> list[Edge]:
    """Parse an `examples/data/*.edges` file: `# comments` + one `lo hi` per line."""
    edges: list[Edge] = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{lineno}: expected 'lo hi', got {raw!r}")
        a, b = (int(p) for p in parts)
        if a == b:
            raise ValueError(f"{path}:{lineno}: self-loop on qubit {a}")
        edges.append((min(a, b), max(a, b)))
    if len(set(edges)) != len(edges):
        dupes = sorted(e for e, c in Counter(edges).items() if c > 1)
        raise ValueError(f"{path}: duplicate edges {dupes}")
    return edges


_HEAVY_HEX_127_CACHE: list[Edge] | None = None


def heavy_hex_127_edges() -> list[Edge]:
    """The 144 undirected edges of the 127-qubit Eagle r3 heavy-hex lattice.

    Read once from `examples/data/heavy_hex_127.edges` and cached; the returned
    list is a fresh copy, so callers may sort or filter it in place.
    """
    global _HEAVY_HEX_127_CACHE
    if _HEAVY_HEX_127_CACHE is None:
        _HEAVY_HEX_127_CACHE = load_edge_list(HEAVY_HEX_127_PATH)
    return list(_HEAVY_HEX_127_CACHE)


def _adjacency(edges: Iterable[Edge]) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _is_connected(num_qubits: int, edges: Sequence[Edge]) -> bool:
    if num_qubits <= 1:
        return True
    adj = _adjacency(edges)
    seen = {0}
    stack = [0]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == num_qubits


def heavy_hex_sublattice(n: int, *, require_connected: bool = True) -> list[Edge]:
    """Edges of the heavy-hex sublattice induced on device qubits `0..n-1`.

    This is the scaling knob for the kicked-Ising showcases: a real sub-piece of
    the Eagle map rather than a synthetic lattice, so small-`n` validation runs
    and the `n=127` headline share one topology and one qubit numbering.

    A handful of `n` leave the induced subgraph disconnected (device qubits 37,
    75 and 113 have no lower-indexed neighbour, so they sit isolated until their
    neighbours are included). Connectivity is *computed*, not tabulated; with
    `require_connected=True` those sizes raise rather than silently handing back
    a lattice with a free qubit.
    """
    if not 1 <= n <= 127:
        raise ValueError(f"n must be in 1..127 (the Eagle device size), got {n}")
    edges = [(a, b) for a, b in heavy_hex_127_edges() if a < n and b < n]
    if require_connected and not _is_connected(n, edges):
        isolated = sorted(set(range(n)) - set(_adjacency(edges)))
        raise ValueError(
            f"the heavy-hex sublattice on qubits 0..{n - 1} is disconnected "
            f"(isolated qubits: {isolated}). Pick a different n, or pass "
            f"require_connected=False if a disconnected lattice is intended."
        )
    return edges


def heavy_hex_edge_coloring(edges: Sequence[Edge]) -> list[list[Edge]]:
    """Partition `edges` into proper matchings ("hardware layers").

    A proper edge coloring is exactly a partition into matchings: within one
    class no two edges share a qubit, so its ZZ rotations act on disjoint
    supports — which is how the entangling layers are executed on hardware, and
    the order the kicked-Ising builder uses by default.

    Greedy first-fit in sorted edge order. By Vizing's theorem a graph of
    maximum degree 3 needs at most 4 colors, and greedy can need up to
    `2*Delta - 1`; on the Eagle map this particular order happens to achieve the
    optimal 3, which the structural test pins. Nothing here *requires* 3 — a
    caller that gets 4 classes still gets a correct circuit, just one extra
    (empty-of-conflicts) layer.
    """
    used: dict[int, set[int]] = defaultdict(set)
    classes: list[list[Edge]] = []
    for a, b in sorted(edges):
        color = 0
        while color in used[a] or color in used[b]:
            color += 1
        while len(classes) <= color:
            classes.append([])
        classes[color].append((a, b))
        used[a].add(color)
        used[b].add(color)
    return classes


# --------------------------------------------------------------------------
# Circuit builders
# --------------------------------------------------------------------------


#: Default Trotter-step layer order, matching Kim et al. (2023) SI Eq. (4)
#: `U(theta_h) = prod_{<i,j>} R_{Z_iZ_j}(-pi/2) · prod_i R_{X_i}(theta_h)`, in
#: which the rightmost factor -- the X layer -- acts first. See
#: `heavy_hex_kicked_ising` for why the default is this and not the reverse.
KICKED_ISING_DEFAULT_ORDER = "x-then-zz"


def heavy_hex_kicked_ising(
    n: int = 127,
    trotter_steps: int = 1,
    theta_h: float = 0.0,
    theta_zz: float = KICKED_ISING_CLIFFORD_THETA_ZZ,
    *,
    edges: Sequence[Edge] | None = None,
    color_layers: bool = True,
    order: str = KICKED_ISING_DEFAULT_ORDER,
    final_x_layer: bool = False,
) -> Circuit:
    """Kicked transverse-field Ising Trotter circuit on the heavy-hex lattice.

    One Trotter step applies an X-rotation layer and a ZZ-rotation layer:

        prod_{q}          exp(-i · theta_h  · X_q       / 2)      [`order` first half]
        prod_{(i,j) in E} exp(-i · theta_zz · Z_i Z_j   / 2)

    so the default `theta_zz = -pi/2` gives the utility experiment's entangler
    `exp(+i·(pi/4)·Z_i Z_j)`, and `theta_h` is the transverse-field kick angle
    passed straight to `rx`. Both `theta_h = 0` and `theta_h = pi/2` (at
    `theta_zz = -pi/2`) are Clifford points, which is what makes an exact
    stabilizer cross-check possible with no reference data.

    **Layer order.** `order` is `"x-then-zz"` (default) or `"zz-then-x"`. This
    is not cosmetic: for a depth-`D` circuit the two differ by a boundary layer,
    and a Z-type seed observable sees `"zz-then-x"`'s leading ZZ layer as a
    no-op, so `"zz-then-x"` at `D` steps is effectively half a step shallower
    than `"x-then-zz"` at `D` steps. The default is `"x-then-zz"` because it is
    the ordering of Kim et al. (2023) SI Eq. (4), and because it is the ordering
    under which this builder *reproduces the published weight-10 and weight-17
    operators exactly* -- see
    `python/paulistrings/tests/test_examples_circuits.py`, which evolves `Z_13`
    and `Z_58` for five steps at `theta_h = pi/2` and recovers the published
    supports and Clifford eigenvalues. Under `"zz-then-x"` the same evolution
    gives weight 6 and 15, not 10 and 17.

    `final_x_layer=True` appends one extra X-rotation layer after the last step.
    That is the "modified" circuit of the paper's Fig. 4a ("similar to Fig. 3c
    but with a further final layer of single-qubit Pauli rotations"), whose
    stabilizer is the modified weight-17 operator.

    Channel count is `trotter_steps · (len(E) + n)`, plus `n` when
    `final_x_layer`, one gate each. ZZ rotations are emitted grouped by
    `heavy_hex_edge_coloring` (disjoint-support hardware layers) unless
    `color_layers=False`, in which case they are emitted in sorted edge order.
    All ZZ rotations commute, so the grouping cannot change the exact result --
    it changes only the order in which per-channel truncation sees the sum, and
    the colored order is the physically faithful one.

    `edges` defaults to `heavy_hex_sublattice(n)`; pass it explicitly to run on
    a different topology with the same step structure.
    """
    if trotter_steps < 0:
        raise ValueError(f"trotter_steps must be >= 0, got {trotter_steps}")
    if order not in ("x-then-zz", "zz-then-x"):
        raise ValueError(f"order must be 'x-then-zz' or 'zz-then-x', got {order!r}")
    lattice = (
        heavy_hex_sublattice(n)
        if edges is None
        else [(min(a, b), max(a, b)) for a, b in edges]
    )
    for a, b in lattice:
        if not 0 <= a < n or not 0 <= b < n:
            raise ValueError(f"edge ({a}, {b}) is outside qubits 0..{n - 1}")

    if color_layers:
        zz_order = [e for group in heavy_hex_edge_coloring(lattice) for e in group]
    else:
        zz_order = sorted(lattice)

    circuit = Circuit(n)

    def push_x() -> None:
        for q in range(n):
            circuit.rx(theta_h, q)

    def push_zz() -> None:
        for a, b in zz_order:
            circuit.pauli_rotation("ZZ", [a, b], theta_zz)

    for _ in range(trotter_steps):
        if order == "x-then-zz":
            push_x()
            push_zz()
        else:
            push_zz()
            push_x()
    if final_x_layer:
        push_x()
    return circuit


def xxz_chain_trotter(
    n: int,
    trotter_steps: int,
    Jz: float = 1.0,
    dt: float = 0.1,
    *,
    bond_order: str = "even-odd",
) -> Circuit:
    """First-order Trotter circuit for the open XXZ chain.

    Hamiltonian (unit XY coupling, anisotropy `Jz`; `Jz = 0` is the free
    XX/XY case):

        H = sum_{i=0}^{n-2} ( X_i X_{i+1} + Y_i Y_{i+1} + Jz · Z_i Z_{i+1} )

    One Trotter step applies, bond by bond,

        exp(-i·dt·X_i X_{i+1}) exp(-i·dt·Y_i Y_{i+1}) exp(-i·dt·Jz·Z_i Z_{i+1})

    i.e. three `pauli_rotation` channels per bond with `theta = 2·dt`, `2·dt`
    and `2·dt·Jz`. The ZZ rotation is emitted even at `Jz = 0` (as the identity
    channel `theta = 0`) so the channel count -- and hence the truncation
    schedule -- does not depend on the value of `Jz`.

    `bond_order` is `"even-odd"` (the two-sublattice sweep: all even bonds, then
    all odd bonds -- gates within a sublattice have disjoint support) or
    `"sequential"` (bonds `0, 1, ... n-2` left to right).

    Channel count per step is `3 · (n - 1)`.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if trotter_steps < 0:
        raise ValueError(f"trotter_steps must be >= 0, got {trotter_steps}")
    bonds = list(range(n - 1))
    if bond_order == "even-odd":
        bonds = [i for i in bonds if i % 2 == 0] + [i for i in bonds if i % 2 == 1]
    elif bond_order != "sequential":
        raise ValueError(f"bond_order must be 'even-odd' or 'sequential', got {bond_order!r}")

    circuit = Circuit(n)
    for _ in range(trotter_steps):
        for i in bonds:
            pair = [i, i + 1]
            circuit.pauli_rotation("XX", pair, 2.0 * dt)
            circuit.pauli_rotation("YY", pair, 2.0 * dt)
            circuit.pauli_rotation("ZZ", pair, 2.0 * dt * Jz)
    return circuit


def haar_su4(rng: np.random.Generator) -> np.ndarray:
    """One Haar-random SU(4) matrix (4x4 complex ndarray).

    Method: QR of a complex Ginibre matrix with the phase fix of F. Mezzadri,
    "How to generate random matrices from the classical compact groups",
    Notices of the AMS 54, 592 (2007) (arXiv:math-ph/0609050), section 4. Draw
    `Z` with i.i.d. standard complex Gaussian entries, take `Z = Q R`, and
    replace `Q <- Q · diag(R_kk / |R_kk|)`; the phase fix is essential -- plain
    LAPACK QR is *not* Haar-distributed, because its `R` has positive-real
    diagonal only up to an unfixed phase convention. Finally divide by
    `det(Q)^(1/4)` to land in SU(4) rather than U(4).

    The global phase is irrelevant for the conjugation `Q -> U Q U†` this engine
    performs, so the determinant fix is cosmetic; it is done anyway so the name
    is honest and so a caller can compare against an independent SU(4) sampler.
    """
    z = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    diag = np.diagonal(r)
    q = q * (diag / np.abs(diag))
    return q / np.linalg.det(q) ** 0.25


def random_su4_staircase(n: int, depth: int, seed: int) -> Circuit:
    """Brickwork circuit of independent Haar-random SU(4) blocks.

    Layer `d` acts on the nearest-neighbour pairs `(i, i+1)` with `i ≡ d (mod
    2)`: even layers on `(0,1), (2,3), ...`, odd layers on `(1,2), (3,4), ...`.
    Blocks are drawn from `numpy.random.default_rng(seed)` in emission order
    (layer-major, then ascending `i`), so the circuit is a deterministic
    function of `(n, depth, seed)` -- the property benchmark E's checked-in
    circuit relies on.

    One `unitary_2q` channel per block; `gates.unitary_2q(q0, q1, m)` treats
    `q0` as the more significant tensor factor, and blocks are pushed as
    `(i, i+1)`.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2 for two-qubit blocks, got {n}")
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    rng = np.random.default_rng(seed)
    circuit = Circuit(n)
    for d in range(depth):
        for i in range(d % 2, n - 1, 2):
            circuit.append(gates.unitary_2q(i, i + 1, haar_su4(rng)))
    return circuit


def qaoa(
    graph_edges: Sequence[Edge],
    p: int,
    gammas: Sequence[float],
    betas: Sequence[float],
    *,
    num_qubits: int | None = None,
) -> Circuit:
    """MaxCut-style QAOA ansatz of depth `p` on `graph_edges`.

    Round `k` (`k = 0..p-1`) applies the cost layer then the mixer layer:

        prod_{(i,j) in E} exp(-i · gammas[k] · Z_i Z_j)     [theta = 2·gammas[k]]
        prod_{q}          exp(-i · betas[k]  · X_q)         [theta = 2·betas[k]]

    The usual MaxCut cost operator `C = sum_{(i,j)} (1 - Z_i Z_j)/2` differs
    from `sum Z_i Z_j` by an identity term and a factor `-1/2`; the identity is
    a global phase (invisible under conjugation) and the factor is absorbed into
    the caller's `gammas`, so `gamma` here is the coefficient of `Z_i Z_j`, not
    of `C`. State preparation (`|+>^n`) is not part of the circuit -- it is the
    `state="x+"` argument of `PauliSum.expectation`.

    `num_qubits` defaults to `max(node) + 1`. Channel count per round is
    `len(E) + num_qubits`, one gate each.
    """
    if p < 0:
        raise ValueError(f"p must be >= 0, got {p}")
    if len(gammas) != p or len(betas) != p:
        raise ValueError(f"expected p={p} gammas and betas, got {len(gammas)} and {len(betas)}")
    edges = [(min(a, b), max(a, b)) for a, b in graph_edges]
    for a, b in edges:
        if a == b:
            raise ValueError(f"self-loop on qubit {a}")
    if num_qubits is None:
        if not edges:
            raise ValueError("num_qubits must be given for an edgeless graph")
        num_qubits = max(max(e) for e in edges) + 1
    for a, b in edges:
        if b >= num_qubits or a < 0:
            raise ValueError(f"edge ({a}, {b}) is outside qubits 0..{num_qubits - 1}")

    circuit = Circuit(num_qubits)
    for k in range(p):
        for a, b in edges:
            circuit.pauli_rotation("ZZ", [a, b], 2.0 * gammas[k])
        for q in range(num_qubits):
            circuit.rx(2.0 * betas[k], q)
    return circuit


def hardware_efficient_ansatz_num_params(n: int, layers: int) -> int:
    """Length of the flat `params` vector `hardware_efficient_ansatz` expects."""
    return 2 * n * layers


def hardware_efficient_ansatz(
    n: int,
    layers: int,
    params: Sequence[float],
    *,
    entangler: str = "cnot",
) -> Circuit:
    """Rotation-plus-ladder hardware-efficient ansatz.

    Each of the `layers` layers applies, in order,

        exp(-i · ry_q · Y_q / 2)  then  exp(-i · rz_q · Z_q / 2)   for q = 0..n-1
        entangling ladder on (0,1), (1,2), ..., (n-2, n-1)

    `params` is a flat sequence of length `hardware_efficient_ansatz_num_params(n,
    layers) == 2·n·layers`, consumed in emission order: layer-major, then
    ascending qubit, `ry` before `rz`. The angles are passed straight to `ry` /
    `rz`, which use the same `exp(-i·theta·P/2)` convention as everything else
    here.

    `entangler` is `"cnot"` (default) or `"cz"`. The ladder is sequential, not
    brickwork -- adjacent rungs share a qubit -- because that is the shape of
    the ansatz as usually written; use `random_su4_staircase` when a
    disjoint-support brickwork is wanted instead.

    Channel count per layer is `2·n + (n - 1)`, one gate each.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if layers < 0:
        raise ValueError(f"layers must be >= 0, got {layers}")
    want = hardware_efficient_ansatz_num_params(n, layers)
    if len(params) != want:
        raise ValueError(f"expected {want} params for n={n}, layers={layers}, got {len(params)}")
    if entangler not in ("cnot", "cz"):
        raise ValueError(f"entangler must be 'cnot' or 'cz', got {entangler!r}")

    circuit = Circuit(n)
    idx = 0
    for _ in range(layers):
        for q in range(n):
            circuit.ry(float(params[idx]), q)
            circuit.rz(float(params[idx + 1]), q)
            idx += 2
        for q in range(n - 1):
            if entangler == "cnot":
                circuit.cnot(q, q + 1)
            else:
                circuit.cz(q, q + 1)
    return circuit
