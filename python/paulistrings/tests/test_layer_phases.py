"""The per-layer, per-partition phase split on ``PartitionStats``.

The Rust side (``crates/paulistrings/tests/trace.rs``) pins that the phases fit inside each layer's ``nanos``; this file checks the Python shapes and the coarse/fine convention.
"""

import os

import numpy as np

from paulistrings import Circuit, PauliSum

COARSE = ("rebucket_ns", "prepare_ns", "collective_ns", "export_ns", "exchange_ns", "coset_loop_ns", "finalize_ns")
FINE = ("gather_ns", "sort_ns", "merge_ns", "append_ns", "chunk_wait_ns")
NUM_QUBITS = 16


def _observable(terms=4000, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 1 << NUM_QUBITS, size=(terms, 1), dtype=np.uint64)
    z = rng.integers(0, 1 << NUM_QUBITS, size=(terms, 1), dtype=np.uint64)
    return PauliSum.from_arrays(x, z, rng.normal(size=terms) + 0j, num_qubits=NUM_QUBITS)


def _circuit():
    """Cliffords and rotations that move rows across a partition boundary, and a depolarizing layer that moves none."""
    c = Circuit(NUM_QUBITS)
    for q in range(4):
        c.h(q)
        c.cnot(q, q + 1)
        c.rz(0.3, q + 1)
    c.depolarize(0.01, [5])
    return c


def _cpu_sets(count=2):
    return [[cpu] for cpu in sorted(os.sched_getaffinity(0))[:count]]


def _assert_phases(stats):
    part = stats.partition
    for name in COARSE:
        field = getattr(part, name)
        assert len(field) == stats.layers, name
        assert all(len(row) == part.partitions for row in field), name
    for k in range(stats.layers):
        for r in range(part.partitions):
            total = sum(getattr(part, name)[k][r] for name in COARSE)
            assert total <= part.nanos[k][r], (k, r)
        if part.local[k]:
            assert part.export_ns[k] == [0] * part.partitions, k
            assert part.exchange_ns[k] == [0] * part.partitions, k

    fine = [getattr(part, name) for name in FINE]
    if part.gather_ns is None:
        assert all(field is None for field in fine)
        return
    for field in fine:
        assert len(field) == stats.layers
        assert all(len(row) == part.partitions for row in field)
    for k in range(stats.layers):
        for r in range(part.partitions):
            assert part.chunk_wait_ns[k][r] <= part.append_ns[k][r] <= part.gather_ns[k][r], (k, r)


def test_unpartitioned_run_reports_one_partitions_phases():
    stats = _observable().propagate_with_stats(_circuit())
    _assert_phases(stats)
    part = stats.partition
    assert any(row[0] > 0 for row in part.coset_loop_ns)
    assert all(row == [0] for row in part.export_ns)


def test_partitioned_run_reports_each_partitions_phases():
    stats = _observable().propagate_with_stats(_circuit(), partitions=_cpu_sets(), pin_memory=False)
    _assert_phases(stats)
    part = stats.partition
    assert part.partitions == 2
    assert any(not local for local in part.local)
    for k in range(stats.layers):
        if not part.local[k]:
            assert all(ns > 0 for ns in part.export_ns[k]), k


def test_repr_leaves_the_phases_out():
    stats = _observable(terms=8).propagate_with_stats(_circuit())
    text = repr(stats.partition)
    assert all(name not in text for name in COARSE + FINE)
