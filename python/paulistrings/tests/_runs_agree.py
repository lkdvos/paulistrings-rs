"""``assert_runs_agree`` and its single-cut fixtures, the Python twin of ``test_support::assert_runs_agree``."""

import re
import struct

import numpy as np

from paulistrings import Circuit, PauliSum


def _terms(sum_):
    """{(x_words, z_words): coeff}, so comparisons ignore storage order."""
    xs, zs, cs = sum_.x_array(), sum_.z_array(), sum_.coefficients_array()
    return {
        (tuple(int(v) for v in xs[i]), tuple(int(v) for v in zs[i])): complex(cs[i])
        for i in range(len(sum_))
    }


def _cut(policy):
    """``None``, ``"TopN"``, ``"ApproxTopN"`` or ``"other"``: the layer pass ``policy`` runs."""
    passes = re.findall(r"\b(TopN|ApproxTopN|CollapseSample)\(", repr(policy)) if policy else []
    if not passes:
        return None
    return passes[0] if passes in (["TopN"], ["ApproxTopN"]) else "other"


def _octave(norm_sqr):
    """``ApproxTopN``'s bin: the exponent bits of ``|c|²``."""
    return struct.unpack("<Q", struct.pack("<d", norm_sqr))[0] >> 52


def _norm_sqr(c):
    return c.real * c.real + c.imag * c.imag


def _assert_terms_close(a, b, tol, what):
    assert set(a) == set(b), (
        f"{what}: different keys: {len(set(a) - set(b))} only in got, "
        f"{len(set(b) - set(a))} only in want"
    )
    worst = max((abs(a[k] - b[k]) for k in a), default=0.0)
    assert worst < tol, f"{what}: largest coefficient difference {worst:.3e} exceeds {tol:.1e}"


def _assert_both_keep(other, source, above, tol, what):
    for key, c in source.items():
        if above(c):
            assert key in other, f"{what}: {key} (|c| = {abs(c)}) above the boundary is kept by one run only"
            assert abs(other[key] - c) < tol, f"{what}: {key} above the boundary: {c} vs {other[key]}"


def assert_runs_agree(got, want, policy, layers, tol, what=""):
    """Two runs of ``policy``, ``layers`` layer passes after identical inputs, as ``test_support::assert_runs_agree`` compares them.

    Equal keys, or no layer pass: term for term to ``tol``.
    Different keys after one cut: every term clear of the boundary matches, and the kept magnitudes agree (``topn``) or the smallest kept ``|c|²`` share an octave (``approx_topn``).
    Different keys after more than one cut: the count only, since the runs evolved different survivors.
    """
    assert len(got) == len(want), f"{what}: term count"
    a, b = _terms(got), _terms(want)
    cut = _cut(policy)
    if cut is None or set(a) == set(b) or layers == 0:
        _assert_terms_close(a, b, tol, what)
        return
    if layers > 1:
        return
    if cut == "TopN":
        g = sorted((abs(c) for c in a.values()), reverse=True)
        w = sorted((abs(c) for c in b.values()), reverse=True)
        worst = max(abs(x - y) for x, y in zip(g, w))
        assert worst < tol, (
            f"{what}: kept magnitudes differ by {worst:.3e}; the smallest kept are {g[-1]} vs {w[-1]}"
        )
        _assert_both_keep(a, b, lambda c: abs(c) > g[-1] + 2 * tol, tol, what)
        _assert_both_keep(b, a, lambda c: abs(c) > w[-1] + 2 * tol, tol, what)
    elif cut == "ApproxTopN":
        lowest_got = min(_octave(_norm_sqr(c)) for c in a.values())
        lowest_want = min(_octave(_norm_sqr(c)) for c in b.values())
        assert lowest_got == lowest_want, (
            f"{what}: the smallest kept |c|² lie in octaves {lowest_got} vs {lowest_want}"
        )
        edge = struct.unpack("<d", struct.pack("<Q", (lowest_want + 1) << 52))[0] ** 0.5
        _assert_both_keep(a, b, lambda c: abs(c) - tol >= edge, tol, what)
        _assert_both_keep(b, a, lambda c: abs(c) - tol >= edge, tol, what)


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
    """One-layer ``(name, circuit, sum, n)`` cases whose single cut splits a boundary group: a Haar SU(4) on ``(0, 1)`` over random complex coefficients, and a Hadamard on qubit 0 over coefficients ±1, ±2, ±3 with ``n`` inside the ±2 ties."""
    rng = np.random.default_rng(seed + num_qubits)
    x, z = _random_keys(num_qubits, 1_500, rng)
    coeffs = rng.normal(size=1_500) + 1j * rng.normal(size=1_500)
    dense = Circuit(num_qubits)
    dense.unitary_2q(0, 1, _haar_4(rng))
    dense_sum = PauliSum.from_arrays(x, z, coeffs, num_qubits=num_qubits)

    x, z = _random_keys(num_qubits, 2_000, rng)
    magnitudes = rng.integers(1, 4, size=2_000).astype(np.float64)
    signs = rng.choice([-1.0, 1.0], size=2_000)
    hadamard = Circuit(num_qubits)
    hadamard.h(0)
    tied_sum = PauliSum.from_arrays(x, z, magnitudes * signs, num_qubits=num_qubits)
    return [
        ("dense cut", dense, dense_sum, 2_000),
        ("tied cut", hadamard, tied_sum, 1_300),
    ]
