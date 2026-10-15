"""``assert_runs_agree`` and its fixtures, the Python twin of ``test_support::assert_runs_agree``."""

import math
import re

import numpy as np

from paulistrings import Circuit, PauliSum


def _terms(sum_):
    """{(x_words, z_words): coeff}, so comparisons ignore storage order."""
    xs, zs, cs = sum_.x_array(), sum_.z_array(), sum_.coefficients_array()
    return {
        (tuple(int(v) for v in xs[i]), tuple(int(v) for v in zs[i])): complex(cs[i])
        for i in range(len(sum_))
    }


def _has_layer_pass(policy):
    return bool(policy) and re.search(r"\b(TopN|ApproxTopN|CollapseSample)\(", repr(policy)) is not None


def _norm(terms):
    return math.sqrt(sum(abs(c) ** 2 for c in terms.values()))


def _assert_terms_close(a, b, tol, what):
    assert set(a) == set(b), (
        f"{what}: different keys: {len(set(a) - set(b))} only in got, "
        f"{len(set(b) - set(a))} only in want"
    )
    worst = max((abs(a[k] - b[k]) for k in a), default=0.0)
    assert worst < tol, f"{what}: largest coefficient difference {worst:.3e} exceeds {tol:.1e}"


def discarded_norm(sum_, circuit, policy, direction="forward"):
    """The per-layer norm of what ``policy`` drops from ``sum_`` through ``circuit``, summed over layers: the ``discarded`` of ``assert_runs_agree``."""
    direction = direction or "forward"
    order = range(len(circuit)) if direction == "forward" else reversed(range(len(circuit)))
    discarded = 0.0
    for i in order:
        layer = circuit[i : i + 1]
        full = _terms(sum_.propagate(layer, direction=direction))
        sum_ = sum_.propagate(layer, policy, direction=direction)
        kept = _terms(sum_)
        discarded += math.sqrt(sum(abs(c) ** 2 for k, c in full.items() if k not in kept))
    return discarded


def assert_runs_agree(got, want, policy, discarded, tol, what=""):
    """Two runs of ``policy`` from identical inputs, the reference having dropped ``discarded`` (``discarded_norm``), as ``test_support::assert_runs_agree`` compares them.

    Equal term counts; term for term to ``tol`` without a layer pass or when nothing was dropped.
    Otherwise each run lies within ``Δ = discarded + √len·tol`` of the untruncated operator, so ``1 − Re⟨got, want⟩ / (‖got‖ ‖want‖) ≤ 2Δ² / (‖got‖ ‖want‖)``; a cut too heavy for that bound to see a sign flip on the largest term fails.
    """
    assert len(got) == len(want), f"{what}: term count"
    a, b = _terms(got), _terms(want)
    if not _has_layer_pass(policy) or discarded == 0.0 or not b:
        _assert_terms_close(a, b, tol, what)
        return
    distance, bound, sees_a_sign_flip = _overlap(a, b, discarded, tol)
    assert sees_a_sign_flip, f"{what}: Δ = {discarded:e} of ‖want‖ = {_norm(b):e} is too heavy a cut to compare"
    assert distance <= bound, f"{what}: 1 − cos = {distance:e} exceeds 2Δ²/(‖got‖‖want‖) = {bound:e}"


def _overlap(a, b, discarded, tol):
    """``1 − cos`` between the runs, its bound from ``Δ``, and whether that bound could see a sign flip on the reference's largest term."""
    got_norm, want_norm = _norm(a), _norm(b)
    distance = sum(
        abs(a.get(k, 0.0) / got_norm - b.get(k, 0.0) / want_norm) ** 2 for k in a.keys() | b.keys()
    ) / 2
    slack = discarded + math.sqrt(len(b)) * tol
    bound = 2 * slack * slack / (got_norm * want_norm)
    teeth = 2 * max(abs(c) ** 2 for c in b.values()) / want_norm**2
    return distance, bound, bound < teeth


def assert_runs_agree_count_only(got, want, policy, discarded, what=""):
    """Two runs of ``policy``, a bare ``topn(n)`` or ``approx_topn(n)``, through a cut too heavy for ``assert_runs_agree``, as ``test_support::assert_runs_agree_count_only`` compares them: both keep exactly ``n`` terms, all finite; a cut light enough for ``assert_runs_agree`` fails."""
    match = re.fullmatch(r"Truncation\((?:Approx)?TopN\((\d+)\)\)", repr(policy))
    assert match, f"{what}: a count-only comparison needs a bare topn or approx_topn, got {policy!r}"
    assert len(got) == len(want), f"{what}: term count"
    assert len(want) == int(match[1]), f"{what}: the last layer must cut"
    a, b = _terms(got), _terms(want)
    assert all(np.isfinite(c) for c in [*a.values(), *b.values()]), f"{what}: a non-finite coefficient"
    assert not _overlap(a, b, discarded, 0.0)[2], f"{what}: Δ = {discarded:e} is light enough for assert_runs_agree"


def spread(coeffs, rng, octaves=40):
    """``coeffs`` each scaled by ``2^-k``, ``k`` drawn in ``0..octaves``: a cut through the low octaves drops little weight."""
    return coeffs * 2.0 ** -rng.integers(0, octaves, size=len(coeffs)).astype(np.float64)


def _random_keys(num_qubits, terms, rng):
    words = (num_qubits + 63) // 64
    x = rng.integers(0, 1 << 63, size=(terms, words), dtype=np.uint64)
    z = rng.integers(0, 1 << 63, size=(terms, words), dtype=np.uint64)
    tail = num_qubits - 64 * (words - 1)
    last = np.uint64((1 << tail) - 1) if tail < 64 else np.uint64((1 << 64) - 1)
    x[:, words - 1] &= last
    z[:, words - 1] &= last
    return x, z


def _haar_4(rng):
    q, r = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def single_cuts(num_qubits, seed=20261011):
    """One-layer ``(name, circuit, sum, n)`` cases whose single cut splits a boundary group and drops little weight: a Haar SU(4) on ``(0, 1)`` over ``spread`` complex coefficients, and a Hadamard on qubit 0 over magnitudes 1, 2⁻²⁰, 2⁻⁴⁰ with ``n`` inside the 2⁻²⁰ ties."""
    rng = np.random.default_rng(seed + num_qubits)
    x, z = _random_keys(num_qubits, 1_500, rng)
    coeffs = spread(rng.normal(size=1_500) + 1j * rng.normal(size=1_500), rng)
    dense = Circuit(num_qubits)
    dense.unitary_2q(0, 1, _haar_4(rng))
    dense_sum = PauliSum.from_arrays(x, z, coeffs, num_qubits=num_qubits)

    x, z = _random_keys(num_qubits, 2_000, rng)
    magnitudes = 2.0 ** (20.0 * rng.integers(0, 3, size=2_000) - 40.0)
    signs = rng.choice([-1.0, 1.0], size=2_000)
    hadamard = Circuit(num_qubits)
    hadamard.h(0)
    tied_sum = PauliSum.from_arrays(x, z, magnitudes * signs, num_qubits=num_qubits)
    return [
        ("dense cut", dense, dense_sum, 2_000),
        ("tied cut", hadamard, tied_sum, 1_300),
    ]


def heavy_cut(num_qubits, terms=800, seed=20261012):
    """``(circuit, sum)``: a TFIM Trotter step at angle 0.8 over four tied magnitudes; under a ``topn``/``approx_topn`` well below ``terms`` every layer cuts through a tie group or an octave, too heavily for ``assert_runs_agree``."""
    rng = np.random.default_rng(seed + num_qubits)
    x, z = _random_keys(num_qubits, terms, rng)
    magnitudes = np.array([1.0, 0.5, 0.25, 0.125])[np.arange(terms) % 4]
    circuit = Circuit(num_qubits)
    for q in range(num_qubits):
        circuit.pauli_rotation("ZZ", [q, (q + 1) % num_qubits], 0.8)
    for q in range(num_qubits):
        circuit.rx(0.8, q)
    return circuit, PauliSum.from_arrays(x, z, magnitudes, num_qubits=num_qubits)
