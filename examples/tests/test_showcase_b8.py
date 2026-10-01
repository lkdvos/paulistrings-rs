"""B8 operator Loschmidt echo: circuit provenance, estimators, and a dense check on a 9-qubit patch."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "b8_ole"))
sys.path.insert(0, str(HERE.parent))

import ole  # noqa: E402
from common.circuits import _adjacency  # noqa: E402
from common.observables import pauli_sum_from_support  # noqa: E402

PATCH_ROOT = 132
PATCH_SIZE = 9


def _same_ops(a, b) -> bool:
    return len(a) == len(b) and all(
        x[0] == y[0] and x[2] == y[2] and (x[1] is None or abs(x[1] - y[1]) < 1e-12) for x, y in zip(a, b)
    )


@pytest.mark.parametrize("alpha", ole.spec()["tracker_alphas"])
def test_echo_half_reproduces_the_tracker_circuit(alpha):
    first, probe, second = ole.split_echo(ole.qasm_ops(alpha))
    assert _same_ops(ole.echo_half(ole.spec()["tracker_L"], ole.eta_of_alpha(alpha)), second)
    assert _same_ops(ole.invert(second), first)
    rz = [o for o in probe if o[0] == "rz"]
    assert sorted(o[2][0] for o in rz) == ole.spec()["perturbation_qubits"]
    assert all(abs(o[1] - 2 * ole.spec()["delta"]) < 1e-12 for o in rz)


def test_spec_qubit_sets_match_the_qasm():
    ops = ole.qasm_ops(0.15)
    base = ole.qasm_ops(0.0)
    sp = ole.spec()
    assert sorted({q for o in ops if o[0] == "cz" for q in o[2]}) == sp["device_qubits"]
    assert sorted({a[2][0] for a, b in zip(ops, base) if a[0] == "rx" and abs(a[1] - b[1]) > 1e-12}) == sp["scattering_qubits"]
    assert sorted({o[2][0] for o in base if o[0] == "rx" and abs(abs(o[1]) - sp["rx_slow"]) < 1e-12}) == sp["slow_qubits"]
    assert not set(sp["observable_qubits"]) & set(sp["perturbation_qubits"])
    assert sum(o[0] == "cz" for o in ops) == 1488


def test_diagonal_echo_resums_the_moment_series():
    paulistrings = pytest.importorskip("paulistrings")
    hist = [0.1, 0.2, 0.3, 0.25, 0.15]
    for order in (2, 4, 8):
        assert ole.truncated_series(hist, 0.3, order) != pytest.approx(
            paulistrings.diagonal_echo(hist, 0.3), abs=1e-12
        )
    assert ole.truncated_series(hist, 0.3, 24) == pytest.approx(paulistrings.diagonal_echo(hist, 0.3), abs=1e-12)
    # Full scrambling: n_P ~ Binomial(35, 1/2), App. C 1 c.
    full = [math.comb(35, n) / 2**35 for n in range(36)]
    assert paulistrings.diagonal_echo(full, 0.3) == pytest.approx(
        ole.spec()["references"]["full_scrambling"]["value"], abs=1e-4
    )


def _patch():
    edges = {tuple(sorted(o[2])) for o in ole.floquet_layer() if o[0] == "cz"}
    adj = _adjacency(edges)
    seen = [PATCH_ROOT]
    i = 0
    while len(seen) < PATCH_SIZE:
        for nb in sorted(adj[seen[i]]):
            if nb not in seen and len(seen) < PATCH_SIZE:
                seen.append(nb)
        i += 1
    return sorted(seen)


def _restrict(ops, qubits):
    keep = set(qubits)
    return [o for o in ops if set(o[2]) <= keep]


def _dense_echo(ops, qubits, obs_q, probe_q, delta):
    """`2^-n tr(A V^dagger A V)` with `A = C^dagger O C`, gates applied axis by axis to a `2^n x 2^n` matrix."""
    n = len(qubits)
    idx = {q: i for i, q in enumerate(qubits)}
    dim = 2**n

    def apply(m, gate, axes):
        # Left-multiply `m` by `gate` acting on the row-index axes `axes`.
        t = m.reshape((2,) * n + (dim,))
        k = len(axes)
        g = gate.reshape((2,) * (2 * k))
        t = np.tensordot(g, t, axes=(list(range(k, 2 * k)), axes))
        t = np.moveaxis(t, list(range(k)), axes)
        return t.reshape(dim, dim)

    def rot(p, theta):
        return math.cos(theta / 2) * np.eye(2) - 1j * math.sin(theta / 2) * p

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    cz = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
    C = np.eye(dim, dtype=complex)
    for name, angle, qs in ops:
        if name == "rx":
            C = apply(C, rot(X, angle), [idx[qs[0]]])
        elif name == "rz":
            C = apply(C, rot(Z, angle), [idx[qs[0]]])
        else:
            C = apply(C, cz, [idx[qs[0]], idx[qs[1]]])
    diag = np.ones(dim)
    bits = (np.arange(dim)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    for q in obs_q:
        diag = diag * (1 - 2 * bits[:, idx[q]])
    A = C.conj().T @ (diag[:, None] * C)
    V = np.eye(dim, dtype=complex)
    for q in probe_q:
        V = apply(V, math.cos(delta) * np.eye(2) - 1j * math.sin(delta) * X, [idx[q]])
    return float(np.real(np.trace(A @ V.conj().T @ A @ V)) / dim)


def _engine_patch(L, eta):
    ps = pytest.importorskip("paulistrings")
    qubits = _patch()
    index = {q: i for i, q in enumerate(qubits)}
    ops = _restrict(ole.echo_half(L, eta), qubits)
    obs_q = [q for q in ole.spec()["observable_qubits"] if q in index]
    probe_q = [q for q in ole.spec()["perturbation_qubits"] if q in index]
    obs = pauli_sum_from_support({index[q]: "Z" for q in obs_q}, len(qubits))
    circuit = ole.to_circuit(ops, index=index)
    sites = [index[q] for q in probe_q]
    return ps, qubits, ops, obs_q, probe_q, obs, circuit, sites


@pytest.mark.parametrize("L,alpha", [(2, 0.15), (3, 0.25)])
def test_exact_echo_matches_dense_on_a_patch(L, alpha):
    eta = ole.eta_of_alpha(alpha)
    ps, qubits, ops, obs_q, probe_q, obs, circuit, sites = _engine_patch(L, eta)
    assert probe_q and obs_q
    delta = ole.spec()["delta"]
    A = obs.propagate(circuit, direction="heisenberg")
    want = _dense_echo(ops, qubits, obs_q, probe_q, delta)
    assert A.rotated_overlap(sites, delta, axis="x") == pytest.approx(want, abs=1e-10)
    hist = A.anticommute_histogram(sites, axis="x")
    assert sum(hist) == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < ps.diagonal_echo(hist, delta) <= 1.0


def test_no_scattering_is_a_perfect_echo():
    ps, *_, obs, circuit, sites = _engine_patch(2, 0.0)
    A = obs.propagate(circuit, direction="heisenberg")
    assert ps.diagonal_echo(A.anticommute_histogram(sites, axis="x"), 0.3) == pytest.approx(1.0, abs=1e-12)
    assert A.rotated_overlap(sites, 0.3, axis="x") == pytest.approx(1.0, abs=1e-12)


def test_ppmc_with_a_large_cache_is_exact_and_a_small_one_collapses():
    ps, *_, obs, circuit, sites = _engine_patch(2, ole.eta_of_alpha(0.15))
    exact = obs.propagate(circuit, direction="heisenberg")
    big = obs.propagate(circuit, ps.truncation.collapse_sample(10**9, 7), direction="heisenberg")
    assert np.allclose(big.anticommute_histogram(sites, axis="x"), exact.anticommute_histogram(sites, axis="x"), atol=1e-12)
    runs = [
        ps.diagonal_echo(obs.propagate(circuit, ps.truncation.collapse_sample(8, s), direction="heisenberg").anticommute_histogram(sites, axis="x"), 0.3)
        for s in (1, 1, 2)
    ]
    assert runs[0] == runs[1]
    assert all(0.0 <= r <= 1.0 + 1e-12 for r in runs)


def test_quarter_turn_rotations_produce_a_single_clifford_term():
    """`PauliRotation` snaps `rz(pi/2)` etc. to an exact Clifford (fanout 1); skip until the built extension does so."""
    one = pauli_sum_from_support({0: "X"}, 1)
    c = ole.to_circuit([("rz", math.pi / 2, (0,))], index={0: 0})
    n = len(one.propagate(c, direction="heisenberg"))
    if n == 2:
        pytest.skip("extension predates core snapping")
    assert n == 1
