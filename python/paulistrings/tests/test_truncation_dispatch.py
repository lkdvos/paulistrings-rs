"""Host propagates run a tree's per-term filter as a monomorphized generic policy where one is equal to it, and the tree itself otherwise; both truncate alike."""

import os
import random
import sys

import numpy as np
import pytest

from paulistrings import Circuit, PauliSum, _paulistrings, truncation

from .test_from_arrays import _as_dict

TOL = 1e-12
NUM_QUBITS = 6


def _observable():
    rng = np.random.default_rng(20261013)
    x = rng.integers(0, 1 << NUM_QUBITS, size=(2_000, 1), dtype=np.uint64)
    z = rng.integers(0, 1 << NUM_QUBITS, size=(2_000, 1), dtype=np.uint64)
    coeffs = rng.normal(size=2_000) + 1j * rng.normal(size=2_000)
    return PauliSum.from_arrays(x, z, coeffs / 50, num_qubits=NUM_QUBITS)


def _circuit():
    rng = random.Random(20261013)
    circuit = Circuit(NUM_QUBITS)
    for _ in range(3):
        for q in range(NUM_QUBITS):
            getattr(circuit, rng.choice(("rz", "rx", "ry")))(rng.uniform(0.1, 1.4), q)
        for q in range(NUM_QUBITS - 1):
            circuit.cnot(q, q + 1)
    return circuit


def _both_or(policy):
    """``policy | policy``: the same filter, as a tree the dispatch leaves alone."""
    return policy | policy


# (dispatched, an equal tree the dispatch leaves alone)
CASES = {
    "none": (None, _both_or(truncation.weight(NUM_QUBITS))),
    "coeff": (truncation.coeff(1e-3), _both_or(truncation.coeff(1e-3))),
    "weight": (truncation.weight(4), _both_or(truncation.weight(4))),
    "coeff & weight": (
        truncation.coeff(1e-3) & truncation.weight(4),
        _both_or(truncation.coeff(1e-3)) & truncation.weight(4),
    ),
    "approx_topn": (
        truncation.approx_topn(3_000),
        truncation.approx_topn(3_000) & _both_or(truncation.weight(NUM_QUBITS)),
    ),
    "coeff & approx_topn": (
        truncation.coeff(1e-3) & truncation.approx_topn(3_000),
        _both_or(truncation.coeff(1e-3)) & truncation.approx_topn(3_000),
    ),
    "approx_topn & weight": (
        truncation.approx_topn(3_000) & truncation.weight(5),
        truncation.approx_topn(3_000) & _both_or(truncation.weight(5)),
    ),
    "topn & coeff": (
        truncation.topn(3_000) & truncation.coeff(1e-3),
        truncation.topn(3_000) & _both_or(truncation.coeff(1e-3)),
    ),
}

PARTITIONED = sys.platform == "linux" and len(os.sched_getaffinity(0)) >= 2


def _placements(name):
    yield {}
    if PARTITIONED and "topn &" not in name:
        cpus = sorted(os.sched_getaffinity(0))[:2]
        yield {"partitions": [[cpu] for cpu in cpus]}


def _run(s, c, policy, placement):
    out = s.propagate(c, policy, **placement)
    return out, _paulistrings._last_host_policy()


def _assert_close(got, want):
    a, b = _as_dict(got), _as_dict(want)
    assert set(a) == set(b)
    assert max((abs(a[k] - b[k]) for k in a), default=0.0) < TOL


@pytest.mark.parametrize("name", list(CASES))
def test_a_dispatched_tree_truncates_as_its_undispatched_equal(name):
    dispatched, fallback = CASES[name]
    s, c = _observable(), _circuit()
    untruncated = len(s.propagate(c))
    for placement in _placements(name):
        got, got_type = _run(s, c, dispatched, placement)
        want, want_type = _run(s, c, fallback, placement)
        assert "WithFilter" in got_type, (name, placement, got_type)
        assert want_type.endswith("BuiltinTruncation"), (name, placement, want_type)
        assert name == "none" or len(want) < untruncated, "the policy must bite"
        _assert_close(got, want)


def test_a_dispatched_collapse_sample_draws_the_undispatched_trajectory():
    s = PauliSum.from_strings({"Z": 1.0}, num_qubits=1)
    c = Circuit(1)
    for _ in range(8):
        c.rx(0.7, 0)
    for seed in range(5):
        got, got_type = _run(s, c, truncation.collapse_sample(1, seed), {})
        fallback = truncation.collapse_sample(1, seed) & _both_or(truncation.weight(1))
        want, want_type = _run(s, c, fallback, {})
        assert "WithFilter" in got_type and want_type.endswith("BuiltinTruncation")
        _assert_close(got, want)


def test_propagate_with_stats_dispatches_too():
    s, c = _observable(), _circuit()
    policy = truncation.coeff(1e-3)
    out, stats = s.propagate_with_stats(c, policy)
    assert "WithFilter" in _paulistrings._last_host_policy()
    _assert_close(out, s.propagate(c, _both_or(policy)))
    assert stats.final_terms == len(out)
