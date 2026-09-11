"""Showcase B7 -- CI-safe correctness gates (adapted plan §6 Part B "B7").

`examples/b7_stabilizer_prep/run_b7.py` is the full showcase (narrative,
sweeps, committed figures); this file pins the properties of the
stabilizer-preparation -> Pauli-propagation pipeline that must never regress:

* **the dense cross-check** -- at `n = 6, 8, 12` the composed
  preparation-plus-tail circuit is run on a `2^n` state vector with numpy alone
  and the observable contracted against it, versus Heisenberg propagation
  contracted on the preparation's stabilizer generators. Agreement `1e-10`.
* **the closed form** -- for a cluster stabilizer `K_q` on an even-degree site
  and exactly two kicked-Ising Trotter steps, `<K_q> = cos^deg(q)(theta_h)`
  (derived in `run_b7.py`'s `CLOSED_FORM_TOL` comment). Asserted at every
  even-degree site of two lattices and eight kick angles, plus its corollary
  that a *single* step gives exactly zero.
* **the two special cases** -- a Clifford tail reproduces
  `oracles.stim_clifford_exact` on the composed circuit exactly, and the
  `|0...0>` generators `+Z_q` reproduce `expectation(state="z+")` on the same
  evolved sum.
* **the state is the state** -- identity-padded preparations of wildly
  different depth read out byte-identical generators and give an identical
  estimate; stim's readout agrees with the closed-form cluster generators.
* the dense routes' own conventions (the Hermitian-`Y` phase, qubit-0-is-most-
  significant) against explicit `numpy.kron` matrices, and the size guards.

Deliberately small (`n <= 12`, tail depth <= 3): the whole file is a couple of
seconds and a few MiB. Everything except the four stim-gated tests runs in the
numpy-only CI job -- `stabilizer_prep.py` imports numpy and `paulistrings`
only, and `cluster_prep_circuit` is the `stim`-free spelling of the
preparation.
"""

from __future__ import annotations

import functools
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from paulistrings import PauliSum, truncation

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_B7_DIR = _EXAMPLES_DIR / "b7_stabilizer_prep"
for _path in (str(_EXAMPLES_DIR), str(_B7_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import stabilizer_prep as sp  # noqa: E402

from common import circuits, oracles  # noqa: E402

#: The dense state-vector reference is exact up to accumulated rounding over
#: `O(gates * 2^n)` complex multiplications; 1e-10 is the plan's cross-check
#: bar and the observed gaps are ~1e-15.
TOL = 1e-10

#: The closed form is an identity, not an approximation, so it gets a tight
#: bound. Observed worst deviation: 1.1e-16.
CLOSED_FORM_TOL = 1e-12

#: Drops the floating-point dust the Clifford `ZZ` angle spawns (`cos(-pi/2)` is
#: 6.1e-17, not 0 -- plan §9(b)) and nothing physical.
DUST_CUTOFF = 1e-12

#: `(rows, cols)` lattices used throughout. 3x4 has interior sites of degree 4
#: and corners of degree 2; 2x3 and 2x4 have even-degree corners only.
LATTICES = ((2, 3), (2, 4), (3, 4))


# =============================================================================
# fixtures / helpers
# =============================================================================


def _lattice(rows: int, cols: int):
    n = rows * cols
    edges = sp.grid_edges(rows, cols)
    return n, edges, sp.grid_adjacency(n, edges)


def _stabilizer_observable(n: int, adjacency: dict[int, list[int]], q: int) -> PauliSum:
    """`K_q = X_q prod_{n in N(q)} Z_n` as a one-term `PauliSum`."""
    support = {q: "X"}
    for neighbour in adjacency[q]:
        support[neighbour] = "Z"
    return PauliSum.from_strings({sp.pauli_label(support, n): 1.0}, num_qubits=n)


def _tail(n: int, edges, steps: int, theta_h: float):
    return circuits.heavy_hex_kicked_ising(n, steps, theta_h, edges=edges)


def _estimate(n, edges, adjacency, q, steps, theta_h, generators=None):
    """`<K_q>` after `steps` Trotter steps, by the showcase's own pipeline."""
    if generators is None:
        generators = sp.cluster_state_stabilizers(n, edges)
    evolved = _stabilizer_observable(n, adjacency, q).propagate(
        _tail(n, edges, steps, theta_h), truncation.coeff(DUST_CUTOFF), direction="heisenberg"
    )
    return evolved.expectation_stabilizer(generators)


def _even_degree_sites(adjacency: dict[int, list[int]]) -> list[int]:
    return [q for q, neighbours in adjacency.items() if len(neighbours) % 2 == 0]


# =============================================================================
# 1. the closed form (numpy only)
# =============================================================================

_ANGLES = (0.0, 0.2, 0.6, math.pi / 8, math.pi / 4, 1.1, 3 * math.pi / 8, math.pi / 2)


@pytest.mark.parametrize("rows,cols", LATTICES)
def test_two_step_estimate_matches_the_closed_form(rows, cols):
    """`<K_q> = cos^deg(q)(theta_h)` at every even-degree site, every angle."""
    n, edges, adjacency = _lattice(rows, cols)
    sites = _even_degree_sites(adjacency)
    assert sites, f"the {rows}x{cols} lattice has no even-degree site to test"
    for q in sites:
        degree = len(adjacency[q])
        for theta in _ANGLES:
            got = _estimate(n, edges, adjacency, q, 2, theta)
            want = math.cos(theta) ** degree
            assert abs(got.imag) < CLOSED_FORM_TOL
            assert abs(got.real - want) < CLOSED_FORM_TOL, (
                f"{rows}x{cols} q={q} deg={degree} theta={theta}: {got.real} vs {want}"
            )


@pytest.mark.parametrize("rows,cols", LATTICES)
def test_one_step_estimate_is_exactly_zero(rows, cols):
    """The closed form's corollary: one step stops at `+-X_q`, not in the group.

    The `ZZ` layer's Clifford conjugation turns `K_q` into `+-X_q` (even
    degree), the `X` layer commutes with it, and a lone `X_q` is not a cluster
    group element -- so the contraction is exactly 0, with no cancellation
    needed. This is the sharpest available check that the contraction really is
    a membership test rather than an overlap-like quantity.
    """
    n, edges, adjacency = _lattice(rows, cols)
    for q in _even_degree_sites(adjacency):
        for theta in (0.3, 0.6, 1.2):
            got = _estimate(n, edges, adjacency, q, 1, theta)
            assert abs(got) < CLOSED_FORM_TOL, f"q={q} theta={theta}: {got}"


# =============================================================================
# 2. the dense cross-check (numpy only)
# =============================================================================


@pytest.mark.parametrize(
    "rows,cols,steps,theta_h",
    [
        (2, 3, 2, 0.6),
        (2, 3, 3, 0.6),
        (2, 4, 2, 0.35),
        (2, 4, 3, 1.0),
        (3, 4, 2, 0.6),
        (3, 4, 3, 0.6),
        (3, 4, 3, 1.25),
    ],
)
def test_pipeline_matches_a_dense_state_vector(rows, cols, steps, theta_h):
    """`<0| C^dagger O C |0>` densely, versus propagate-then-contract.

    The dense route shares only its *input* with the thing it checks: it reads
    the gate list off the composed `Circuit` (`Circuit.gates`) and runs it on a
    `2^n` state vector. It knows nothing about stabilizer groups.
    """
    n, edges, adjacency = _lattice(rows, cols)
    q = _even_degree_sites(adjacency)[-1]
    observable = _stabilizer_observable(n, adjacency, q)
    preparation = sp.cluster_prep_circuit(n, edges)
    tail = _tail(n, edges, steps, theta_h)

    evolved = observable.propagate(
        tail, truncation.coeff(DUST_CUTOFF), direction="heisenberg"
    )
    got = evolved.expectation_stabilizer(sp.cluster_state_stabilizers(n, edges))
    want = sp.dense_expectation(preparation + tail, observable)
    assert abs(got - want) < TOL, f"{got} vs dense {want}"


@pytest.mark.parametrize("rows,cols", [(2, 3), (2, 4)])
def test_projector_route_agrees_with_the_circuit_route(rows, cols):
    """`Pi = prod (I + s_i G_i)/2` (generators only) vs. the prepared state."""
    n, edges, _adjacency = _lattice(rows, cols)
    from_circuit = sp.dense_state(sp.cluster_prep_circuit(n, edges))
    from_projector = sp.dense_projector_state(sp.cluster_state_stabilizers(n, edges))
    # Equal up to a global phase, so compare the overlap magnitude.
    assert abs(abs(np.vdot(from_circuit, from_projector)) - 1.0) < TOL


def test_zero_state_generators_reproduce_the_product_state_path():
    """`+Z_q` generators == `expectation(state="z+")`, on one evolved sum.

    The degenerate stabilizer state. Uses a `Z`-type observable so the value is
    nowhere near zero and the agreement is not vacuous.
    """
    n, edges, _adjacency = _lattice(3, 4)
    observable = PauliSum.from_strings(
        {sp.pauli_label({5: "Z"}, n): 1.0}, num_qubits=n
    )
    evolved = observable.propagate(
        _tail(n, edges, 2, 0.6), truncation.coeff(DUST_CUTOFF), direction="heisenberg"
    )
    product = complex(evolved.expectation("z+"))
    generators = evolved.expectation_stabilizer(sp.single_z_generators(n))
    assert abs(product) > 0.1, "the probe must not be vacuously zero"
    assert abs(product - generators) < CLOSED_FORM_TOL


# =============================================================================
# 3. the dense helpers' own conventions (numpy only)
# =============================================================================

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _kron_pauli(label: str) -> np.ndarray:
    return functools.reduce(np.kron, [_PAULI[ch] for ch in label])


@pytest.mark.parametrize("n", [1, 2, 3])
def test_dense_pauli_expectation_matches_an_explicit_kron(n):
    """The `Y` phase and the qubit-0-most-significant convention, pinned.

    `dense_pauli_expectation` does index arithmetic rather than building a
    matrix; a sign error on `Y` (whose flip bit makes source and target indices
    differ exactly where its phase is evaluated) would be invisible in any test
    that only used `X` and `Z`.
    """
    rng = np.random.default_rng(0xB7 + n)
    state = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    state /= np.linalg.norm(state)
    for chars in itertools.product("IXYZ", repeat=n):
        label = "".join(chars)
        got = sp.dense_pauli_expectation(state, label)
        want = complex(state.conj() @ _kron_pauli(label) @ state)
        assert abs(got - want) < 1e-12, label


def test_dense_state_reproduces_a_known_bell_pair():
    """One hand-written case, so the gate-application machinery is anchored."""
    from paulistrings import Circuit

    circuit = Circuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    state = sp.dense_state(circuit)
    want = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2.0)
    assert np.allclose(state, want, atol=1e-14)


def test_dense_guards_refuse_instead_of_allocating():
    from paulistrings import Circuit

    with pytest.raises(ValueError, match="MAX_DENSE_QUBITS"):
        sp.dense_state(Circuit(sp.MAX_DENSE_QUBITS + 1))
    n = sp.MAX_PROJECTOR_QUBITS + 1
    with pytest.raises(ValueError, match="MAX_PROJECTOR_QUBITS"):
        sp.dense_projector_state(sp.single_z_generators(n))


def test_cluster_generators_are_a_valid_commuting_independent_set():
    """The engine validates generators; this pins that our formula passes it.

    `expectation_stabilizer` rejects anticommuting or GF(2)-dependent generator
    sets, so contracting a trivial observable against the formula's output is
    itself the check that `K_q` really is a stabilizer generator set.
    """
    for rows, cols in LATTICES:
        n, edges, _adjacency = _lattice(rows, cols)
        generators = sp.cluster_state_stabilizers(n, edges)
        identity = PauliSum.from_strings({"I" * n: 1.0}, num_qubits=n)
        assert identity.expectation_stabilizer(generators).real == pytest.approx(1.0)
        for spec in generators:
            element = PauliSum.from_strings({spec[1:]: 1.0}, num_qubits=n)
            assert element.expectation_stabilizer(generators).real == pytest.approx(1.0)


# =============================================================================
# 4. stim-gated: the importer, the Clifford-tail case, padding
# =============================================================================


@pytest.mark.parametrize("rows,cols", LATTICES)
def test_the_two_cluster_preparations_are_the_same_circuit(rows, cols):
    """`cluster_prep_stim` -> `interop.circuit_from_stim` == `cluster_prep_circuit`."""
    pytest.importorskip("stim")
    from paulistrings import interop

    n, edges, _adjacency = _lattice(rows, cols)
    imported, observable = interop.circuit_from_stim(sp.cluster_prep_stim(n, edges))
    assert observable is None
    assert imported.gates == sp.cluster_prep_circuit(n, edges).gates


@pytest.mark.parametrize("rows,cols", LATTICES)
def test_stim_readout_agrees_with_the_closed_form_generators(rows, cols):
    """Every formula `K_q` is a `+1` element of the group stim reads out.

    Two independent descriptions of one state -- stim's tableau and the graph
    formula -- checked against each other through the contraction, which is the
    only comparison that is invariant under the choice of generating set.
    """
    pytest.importorskip("stim")
    from paulistrings import interop

    n, edges, _adjacency = _lattice(rows, cols)
    generators = interop.stabilizers_from_stim(
        sp.cluster_prep_stim(n, edges), num_qubits=n
    )
    for spec in sp.cluster_state_stabilizers(n, edges):
        element = PauliSum.from_strings({spec[1:]: 1.0}, num_qubits=n)
        assert element.expectation_stabilizer(generators).real == pytest.approx(1.0)


@pytest.mark.parametrize("theta_h", [0.0, math.pi / 2])
@pytest.mark.parametrize("steps", [1, 2, 3])
def test_a_clifford_tail_matches_stim_on_the_composed_circuit(theta_h, steps):
    """Both Clifford points, cross-checked against the tableau oracle.

    The whole pipeline collapses to one stabilizer computation here, so stim can
    answer it on the *composed* preparation-plus-tail circuit -- and must give
    exactly the same number as propagate-then-contract.
    """
    pytest.importorskip("stim")
    from paulistrings import interop

    n, edges, adjacency = _lattice(3, 4)
    q = _even_degree_sites(adjacency)[-1]
    observable = _stabilizer_observable(n, adjacency, q)
    preparation = sp.cluster_prep_stim(n, edges)
    generators = interop.stabilizers_from_stim(preparation, num_qubits=n)
    prep_circuit, _ = interop.circuit_from_stim(preparation)
    tail = _tail(n, edges, steps, theta_h)

    evolved = observable.propagate(
        tail, truncation.coeff(DUST_CUTOFF), direction="heisenberg"
    )
    got = evolved.expectation_stabilizer(generators)
    want = oracles.stim_clifford_exact(prep_circuit + tail, observable)
    assert abs(got - want) < CLOSED_FORM_TOL, f"{got} vs stim {want}"
    # A Clifford circuit maps one Pauli to one Pauli, so the answer is an
    # integer: +-1 if the image is in the group, 0 if not.
    assert round(got.real, 9) in (-1.0, 0.0, 1.0)


def test_identity_padding_changes_neither_the_generators_nor_the_estimate():
    """Preparation depth is free -- the claim, as an assertion."""
    pytest.importorskip("stim")
    from paulistrings import interop

    n, edges, adjacency = _lattice(3, 4)
    q = _even_degree_sites(adjacency)[-1]
    observable = _stabilizer_observable(n, adjacency, q)
    evolved = observable.propagate(
        _tail(n, edges, 2, 0.6), truncation.coeff(DUST_CUTOFF), direction="heisenberg"
    )
    base = sp.cluster_prep_stim(n, edges)

    reference_generators = interop.stabilizers_from_stim(base, num_qubits=n)
    reference_value = evolved.expectation_stabilizer(reference_generators)
    for rounds in (1, 17, 500):
        padded = base + sp.identity_padding(n, rounds, np.random.default_rng(rounds))
        assert len(padded) > len(base)
        generators = interop.stabilizers_from_stim(padded, num_qubits=n)
        assert generators == reference_generators
        assert evolved.expectation_stabilizer(generators) == pytest.approx(
            reference_value
        )


def test_an_unstructured_random_preparation_still_reads_its_own_generators():
    """Any stabilizer state works, not just graph states.

    A deep random Clifford preparation's own generator has expectation exactly
    equal to its sign; a random Pauli string has expectation 0 with
    overwhelming probability (the group holds `2^n` of `4^n` strings), and both
    branches are exercised here.
    """
    pytest.importorskip("stim")
    from paulistrings import interop

    n = 12
    generators = interop.stabilizers_from_stim(
        sp.random_clifford_prep(n, 40, np.random.default_rng(0xB7)), num_qubits=n
    )
    signs = set()
    for spec in generators:
        element = PauliSum.from_strings({spec[1:]: 1.0}, num_qubits=n)
        value = element.expectation_stabilizer(generators).real
        expected = -1.0 if spec[0] == "-" else 1.0
        assert value == pytest.approx(expected)
        signs.add(spec[0])
    assert signs, "the preparation produced no generators"

    rng = np.random.default_rng(0xB7)
    zeros = 0
    for _ in range(12):
        label = "".join(rng.choice(list("IXYZ")) for _ in range(n))
        value = PauliSum.from_strings({label: 1.0}, num_qubits=n).expectation_stabilizer(
            generators
        )
        assert round(value.real, 9) in (-1.0, 0.0, 1.0)
        zeros += abs(value) < 1e-12
    assert zeros >= 10, "a random Pauli should almost never be a group element"
