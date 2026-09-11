"""Showcase B1 -- CI-safe correctness gates (adapted plan §6 Part B "B1").

`examples/b1_operator_scrambling/run_b1_1d.py` and `run_b1_2d.py` are the full
showcases (they produce the committed figures, the results JSON and the
narrative numbers); this file pins the properties those narratives rest on, at
sizes small enough to run in a fraction of a second:

1. **the OTOC formula** -- `C(r,t) = 2 sum_{P anti W_r} |c_P|^2` read off the
   evolved `PauliSum` equals `(1/2) <[W_r, O(t)], [W_r, O(t)]>` computed with
   dense `2^n x 2^n` matrices, for all three single-site probes;
2. **the support profile** -- the per-site weight read off the symplectic bit
   columns equals the one obtained from the dense single-qubit Pauli twirl;
3. **the probe-average identity** `mean_W C_W(r) = (4/3) w_r`, exact to
   rounding, which cross-checks the two bit-column accumulations against each
   other;
4. **a Clifford circuit gives a sharp cone** -- at `theta_h = pi/2` the evolved
   operator is a single Pauli string whose support is the exact causal cone, so
   the weight profile is 0/1-valued and its radius never exceeds one site per
   Trotter step;
5. **the two-point function** `G(r,t)` read out of the weight-one rows, both
   against the dense reference and against a hand-built sum, plus the square
   lattice's structure, the 2D path down to the same dense reference, and the
   chunk-size independence of the reading pass (it exists for `10^8`-term sums).

Everything here is numpy-only: `examples/common/{circuits,observables,oracles}.py`
pull qiskit/stim/matplotlib in only inside functions this file never calls, and
`scrambling.py`'s dense path is plain `numpy.kron`. So this is CI-visible with
no `importorskip` (plan §4).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from paulistrings import PauliSum, truncation

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from b1_operator_scrambling import scrambling as sc  # noqa: E402
from common import circuits, observables, oracles  # noqa: E402

#: Tiny by design: `n = 6` makes the dense reference a 64x64 matrix.
N_QUBITS = 6
STEPS = 3
THETA_H = 0.9
THETA_ZZ = circuits.KICKED_ISING_CLIFFORD_THETA_ZZ
CENTER = N_QUBITS // 2
TOLERANCE = 1e-10


def _chain_spec(n: int, steps: int, theta_h: float, theta_zz: float = THETA_ZZ):
    return oracles.record_gates(
        circuits.heavy_hex_kicked_ising,
        n,
        steps,
        theta_h,
        theta_zz,
        edges=sc.chain_edges(n),
    )


@pytest.fixture(scope="module")
def evolved_and_dense():
    """The same Heisenberg evolution down two independent paths.

    Returns `(evolved PauliSum, dense 2^n x 2^n operator, CircuitSpec)`.
    """
    spec = _chain_spec(N_QUBITS, STEPS, THETA_H)
    evolved = observables.single_z(CENTER, N_QUBITS).propagate(
        spec.to_circuit(), None, direction="heisenberg"
    )
    dense = sc.dense_heisenberg(
        spec, observables.pauli_string({CENTER: "Z"}, N_QUBITS)
    )
    return evolved, dense, spec


def test_evolution_is_nontrivial(evolved_and_dense):
    """Guard the whole file: a trivial evolution would pass every check below."""
    evolved, _dense, _spec = evolved_and_dense
    assert len(evolved) > 10
    profile = sc.support_profile(evolved)
    assert np.count_nonzero(profile > 1e-9) >= 3


def test_hs_norm_is_conserved_without_truncation(evolved_and_dense):
    evolved, dense, _spec = evolved_and_dense
    assert sc.hs_norm(evolved) == pytest.approx(1.0, abs=TOLERANCE)
    assert sc.dense_hs_norm(dense) == pytest.approx(1.0, abs=TOLERANCE)


def test_every_coefficient_matches_the_dense_trace(evolved_and_dense):
    """`c_P = Tr(P O(t)) / 2^n` term by term; the norms agreeing (above) is what
    rules out a *missing* term, which this check alone cannot see."""
    evolved, dense, _spec = evolved_and_dense
    worst = max(
        abs(coefficient - sc.dense_coefficient(dense, label))
        for label, coefficient in oracles.pauli_terms(evolved)
    )
    assert worst < TOLERANCE


def test_support_profile_matches_dense_twirl(evolved_and_dense):
    evolved, dense, _spec = evolved_and_dense
    engine = sc.support_profile(evolved)
    reference = sc.dense_support_profile(dense, N_QUBITS)
    assert np.allclose(engine, reference, atol=TOLERANCE)


@pytest.mark.parametrize("probe", ["X", "Y", "Z"])
def test_otoc_matches_dense_commutator(evolved_and_dense, probe):
    """The headline formula of the showcase, against an explicit commutator."""
    evolved, dense, _spec = evolved_and_dense
    engine = sc.otoc_profile(evolved, probe)
    reference = np.array(
        [sc.dense_otoc(dense, probe, r, N_QUBITS) for r in range(N_QUBITS)]
    )
    assert np.allclose(engine, reference, atol=TOLERANCE)
    # A probe on the seed site anticommutes with the seed, so at least one site
    # must have a nonzero squared commutator -- otherwise `allclose` above would
    # be comparing two arrays of zeros.
    assert np.max(reference) > 0.1


def test_probe_average_identity_is_exact(evolved_and_dense):
    """`mean_W C_W(r) = (4/3) w_r`: exact term by term, so this is a
    machine-precision self-test of the two bit-column accumulations."""
    evolved, _dense, _spec = evolved_and_dense
    assert sc.probe_average_gap(sc.site_sums(evolved)) < 1e-12


def test_two_point_function_matches_dense(evolved_and_dense):
    evolved, dense, _spec = evolved_and_dense
    engine = sc.single_pauli_coefficients(evolved, "Z")
    reference = np.array(
        [
            sc.dense_coefficient(
                dense, observables.pauli_string({r: "Z"}, N_QUBITS)
            ).real
            for r in range(N_QUBITS)
        ]
    )
    assert np.allclose(engine, reference, atol=TOLERANCE)


def test_site_sums_is_chunk_size_independent(evolved_and_dense):
    """The chunked pass exists for 10^8-term sums; chunking must not change it."""
    evolved, _dense, _spec = evolved_and_dense
    whole = sc.site_sums(evolved, chunk_rows=1 << 20)
    tiny = sc.site_sums(evolved, chunk_rows=3)
    assert tiny.norm == pytest.approx(whole.norm, abs=TOLERANCE)
    for name in ("x", "z", "xz"):
        assert np.allclose(getattr(tiny, name), getattr(whole, name), atol=TOLERANCE)


def test_seed_observable_has_a_single_site_support():
    seed = observables.single_z(CENTER, N_QUBITS)
    profile = sc.support_profile(seed)
    expected = np.zeros(N_QUBITS)
    expected[CENTER] = 1.0
    assert np.allclose(profile, expected, atol=TOLERANCE)
    assert sc.support_size(profile, 1e-6) == 1
    assert sc.front_position(profile, CENTER, 1e-6) == 0.0


def test_clifford_circuit_gives_a_sharp_cone():
    """`theta_h = pi/2` with `theta_zz = -pi/2` is Clifford: one Pauli string in,
    one out, and its support is the exact causal cone.

    The weight profile is therefore 0/1-valued -- a *sharp* cone, with no tail
    -- and the cone radius grows by at most one site per Trotter step, which is
    the hard bound every butterfly-velocity number in the showcase is measured
    against. The `min_abs_coeff = 1e-12` cutoff is not a physical truncation:
    at the Clifford point the vanishing branch has coefficient
    `cos(pi/2) = 6.1e-17` rather than an exact zero.
    """
    n, steps = 15, 5
    center = n // 2
    step = circuits.heavy_hex_kicked_ising(
        n, 1, math.pi / 2.0, THETA_ZZ, edges=sc.chain_edges(n)
    )
    dust = truncation.coeff(1e-12)
    evolved = observables.single_z(center, n)
    radii = []
    for t in range(1, steps + 1):
        evolved = evolved.propagate(step, dust, direction="heisenberg")
        assert len(evolved) == 1, f"Clifford step {t} produced {len(evolved)} strings"
        profile = sc.support_profile(evolved)
        occupied = profile > 1e-9
        # Sharp: every site in the support carries the whole weight.
        assert np.allclose(profile[occupied], 1.0, atol=TOLERANCE)
        assert np.allclose(profile[~occupied], 0.0, atol=TOLERANCE)
        radius = sc.front_position(profile, center, 0.5)
        assert radius <= t, f"cone radius {radius} at step {t} breaks the causal bound"
        radii.append(radius)
    # The cone really does grow (a bound that is never approached proves nothing).
    assert radii[-1] > radii[0]
    slope, _ = sc.front_velocity(range(1, steps + 1), radii)
    assert slope == pytest.approx(1.0, abs=1e-9)


def test_truncation_only_ever_discards_weight():
    """`N(t) <= 1` under truncation, and looser cutoffs keep less."""
    n, steps = 13, 6
    center = n // 2
    step = circuits.heavy_hex_kicked_ising(
        n, 1, THETA_H, THETA_ZZ, edges=sc.chain_edges(n)
    )
    norms = []
    for eps in (1e-1, 1e-2, 1e-3):
        evolved = observables.single_z(center, n)
        for _ in range(steps):
            evolved = evolved.propagate(step, truncation.coeff(eps), direction="heisenberg")
        norms.append(sc.hs_norm(evolved))
        assert norms[-1] <= 1.0 + TOLERANCE
    assert norms[0] < norms[-1], f"a looser cutoff kept more norm: {norms}"


def test_single_pauli_coefficients_reads_a_hand_built_sum():
    """Independent of any propagation: a sum built by hand, read back."""
    n = 5
    terms = {
        observables.pauli_string({1: "Z"}, n): 0.25,
        observables.pauli_string({3: "Z"}, n): -0.5,
        observables.pauli_string({0: "X"}, n): 0.75,
        observables.pauli_string({2: "Y"}, n): 0.125,
        # weight two: must be ignored by every axis
        observables.pauli_string({0: "Z", 4: "Z"}, n): 2.0,
    }
    summed = PauliSum.from_strings(terms, num_qubits=n)
    assert np.allclose(
        sc.single_pauli_coefficients(summed, "Z"), [0.0, 0.25, 0.0, -0.5, 0.0]
    )
    assert np.allclose(
        sc.single_pauli_coefficients(summed, "X"), [0.75, 0.0, 0.0, 0.0, 0.0]
    )
    assert np.allclose(
        sc.single_pauli_coefficients(summed, "Y"), [0.0, 0.0, 0.125, 0.0, 0.0]
    )


def test_square_lattice_structure():
    rows, cols = 4, 5
    edges = sc.square_lattice_edges(rows, cols)
    # 2D open lattice: rows*(cols-1) horizontal + (rows-1)*cols vertical bonds.
    assert len(edges) == rows * (cols - 1) + (rows - 1) * cols
    assert len(set(edges)) == len(edges)
    degree = np.zeros(rows * cols, dtype=int)
    for a, b in edges:
        assert a < b
        degree[a] += 1
        degree[b] += 1
    assert degree.max() == 4
    assert degree.min() == 2  # the corners
    # Round-trip the coordinate map, and check the distance metric.
    for q in range(rows * cols):
        r, c = sc.square_lattice_coords(rows, cols, q)
        assert sc.square_lattice_index(rows, cols, r, c) == q
    center = sc.square_lattice_index(rows, cols, 1, 2)
    distances = sc.square_lattice_distances(rows, cols, center)
    assert distances[center] == 0
    assert distances[sc.square_lattice_index(rows, cols, 3, 4)] == 4


def test_2d_lattice_evolution_matches_dense():
    """The 2D path down to the same dense reference.

    A 2x3 lattice: six qubits, so the dense reference is a 64x64 matrix and the
    test costs milliseconds, but the topology is genuinely two-dimensional (it
    has a four-cycle, which a chain does not) and the centre site has three
    neighbours. `run_b1_2d.py`'s own validation runs the same comparison on a
    3x3 lattice, where the dense path costs seconds rather than milliseconds.
    """
    rows, cols = 2, 3
    n = rows * cols
    center = sc.square_lattice_index(rows, cols, 1, 1)
    spec = oracles.record_gates(
        circuits.heavy_hex_kicked_ising,
        n,
        2,
        0.3,
        0.3,
        edges=sc.square_lattice_edges(rows, cols),
    )
    evolved = observables.single_z(center, n).propagate(
        spec.to_circuit(), None, direction="heisenberg"
    )
    dense = sc.dense_heisenberg(spec, observables.pauli_string({center: "Z"}, n))
    assert np.allclose(
        sc.support_profile(evolved),
        sc.dense_support_profile(dense, n),
        atol=TOLERANCE,
    )
    assert np.allclose(
        sc.otoc_profile(evolved, "X"),
        [sc.dense_otoc(dense, "X", r, n) for r in range(n)],
        atol=TOLERANCE,
    )
    # `<0...0|O(t)|0...0>` is the (0, 0) element of the dense operator, an
    # independent reference for the quench magnetization the 2D script reports.
    assert complex(evolved.expectation("z+")).real == pytest.approx(
        float(np.real(dense[0, 0])), abs=TOLERANCE
    )


def test_front_velocity_recovers_a_known_slope():
    times = [1, 2, 3, 4, 5]
    slope, intercept = sc.front_velocity(times, [0.5 * t + 2.0 for t in times])
    assert slope == pytest.approx(0.5)
    assert intercept == pytest.approx(2.0)
    with pytest.raises(ValueError):
        sc.front_velocity([1], [1.0])


def test_support_size_and_front_respect_their_floor():
    profile = np.array([0.0, 1e-8, 1e-3, 1.0, 1e-3, 1e-8, 0.0])
    assert sc.support_size(profile, 1e-6) == 3
    assert sc.support_size(profile, 1e-2) == 1
    # Sites above 1e-6 are 2, 3, 4 -- distances 1, 0, 1 from the centre.
    assert sc.front_position(profile, 3, 1e-6) == 1.0
    # Above 1e-9 the 1e-8 shoulders at 1 and 5 count too: distance 2.
    assert sc.front_position(profile, 3, 1e-9) == 2.0
    assert sc.front_position(profile, 3, 10.0) == 0.0
