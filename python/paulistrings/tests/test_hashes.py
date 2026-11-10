"""``paulistrings.hashes`` and the rows it feeds to ``PauliSum.with_hash`` and ``propagate(partition_rows=)``: the rows change speed, never results."""

import logging
import warnings

import numpy as np
import pytest

import paulistrings
from paulistrings import PauliSum, hashes, numa_nodes, propagated

from .test_bucket_knobs import _assert_terms_close, _fanout_circuit

N = 20


def _start(num_qubits=N):
    return PauliSum.from_strings({"Z" + "I" * (num_qubits - 1): 1.0, "IX" + "I" * (num_qubits - 2): 0.5}, num_qubits=num_qubits)


@pytest.mark.parametrize("build", [
    lambda rows, seed, **kw: hashes.dense(N, rows, seed, **kw),
    lambda rows, seed, **kw: hashes.sparse(N, rows, 3, seed, **kw),
    lambda rows, seed, **kw: hashes.projection(N, rows, seed, **kw),
])
def test_builders_give_full_rank_boolean_matrices_deterministically(build):
    rows = build(32, 7)
    assert rows.shape == (32, 2 * N) and rows.dtype == bool
    assert hashes.rank(rows) == 32
    np.testing.assert_array_equal(rows, build(32, 7))
    assert not np.array_equal(rows, build(32, 8))
    excluded = build(8, 7, exclude={"x": [0, 1], "z": list(range(N))})
    assert not excluded[:, [0, 1] + list(range(N, 2 * N))].any()
    assert hashes.rank(excluded) == 8


def test_sparse_rows_read_their_weight_and_even_weights_fall_one_short():
    assert (hashes.sparse(N, 10, 3, 1).sum(axis=1) == 3).all()
    even = hashes.sparse(N, 40, 2, 1)
    assert (even.sum(axis=1) == 2).all() and hashes.rank(even) == 39
    assert hashes.rank(hashes.sparse(N, 5, 40, 1)) == 1
    assert (hashes.projection(N, 45, 1).sum(axis=1) == 1).all()
    assert hashes.rank(hashes.projection(N, 45, 1)) == 40
    with pytest.raises(ValueError, match=r"must lie in 1\.\.=40"):
        hashes.sparse(N, 2, 41, 1)
    with pytest.raises(ValueError, match="outside"):
        hashes.dense(N, 2, 1, exclude={"z": [N]})


def test_with_hash_leads_with_the_given_rows_and_keeps_the_result():
    circuit = _fanout_circuit(N)
    start = _start()
    want = propagated(start, circuit)
    assert start.hash_rows().shape == (31, 2 * N)
    for rows in [hashes.sparse(N, 3, 3, 4), hashes.projection(N, 6, 4), hashes.dense(N, 31, 4), None]:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rehashed = start.with_hash(rows, seed=9)
        assert rehashed.hash_seed == 9 and start.hash_seed != 9, "with_hash returns a copy"
        if rows is not None:
            np.testing.assert_array_equal(rehashed.hash_rows()[: len(rows)], rows)
        assert hashes.rank(rehashed.hash_rows()) == 31
        got = propagated(rehashed, circuit)
        np.testing.assert_array_equal(got.hash_rows(), rehashed.hash_rows())
        assert got.num_buckets > 1 and got.bucket_lengths().sum() == len(got)
        _assert_terms_close(got, want, tol=1e-10)
    copied = start.with_hash(start.with_hash(seed=3).hash_rows())
    np.testing.assert_array_equal(copied.hash_rows(), start.with_hash(seed=3).hash_rows())
    assert start.with_hash(hashes.dense(N, 2, 1).astype(int)).hash_rows()[:2].sum() > 0, "0/1 entries are flags"


def test_dependent_hash_rows_warn_and_still_give_the_same_result():
    circuit = _fanout_circuit(N)
    rows = np.zeros((3, 2 * N), dtype=bool)
    rows[0, 3] = rows[1, 3] = True
    with pytest.warns(RuntimeWarning, match="31 rows have rank 29"):
        rehashed = _start().with_hash(rows)
    _assert_terms_close(propagated(rehashed, circuit), propagated(_start(), circuit), tol=1e-10)


@pytest.mark.parametrize("rows, message", [
    (np.zeros((1, 39), dtype=bool), r"shape \(k, 40\)"),
    (np.zeros((32, 40), dtype=bool), r"k <= 31"),
    (np.full((1, 40), 2), "booleans or 0/1"),
])
def test_bad_hash_rows_are_value_errors(rows, message):
    with pytest.raises(ValueError, match=message):
        _start().with_hash(rows)


needs_two_domains = pytest.mark.skipif(len(numa_nodes()) < 2, reason="needs two NUMA domains")


@needs_two_domains
@pytest.mark.parametrize("rows", [hashes.sparse(N, 1, 3, 2), hashes.projection(N, 1, 2), hashes.dense(N, 1, 2, exclude={"z": list(range(N))})])
def test_explicit_partition_rows_keep_the_result(rows):
    circuit = _fanout_circuit(N)
    want = _start()
    want.propagate(circuit)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = _start()
        got.propagate(circuit, partitions=2, partition_rows=rows)
    _assert_terms_close(got, want, tol=1e-10)


@needs_two_domains
def test_partition_rows_that_fall_short_warn():
    start = _start()
    with pytest.warns(RuntimeWarning, match="only 1 of the 2 partitions"):
        start.propagate(_fanout_circuit(N), partitions=2, partition_rows=np.zeros((1, 2 * N), dtype=bool))
    with pytest.warns(RuntimeWarning, match="only 1 of the 2 partitions"):
        start.propagate(_fanout_circuit(N), partitions=2, partition_row_exclude={"x": list(range(N)), "z": list(range(N))})


@needs_two_domains
def test_partition_rows_in_the_bucket_span_are_logged_once_the_run_reaches_them(caplog):
    start = _start()
    circuit = _fanout_circuit(N)
    with caplog.at_level(logging.WARNING, logger="paulistrings"):
        paulistrings.reset_log_cache()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = propagated(start, circuit, partitions=2, partition_rows=start.hash_rows()[:1])
    paulistrings.reset_log_cache()
    assert [r for r in caplog.records if "overlap the span of the bucket hash" in r.getMessage()]
    _assert_terms_close(got, propagated(start, circuit), tol=1e-10)


@pytest.mark.parametrize("kwargs, message", [
    (dict(partitions=2, partition_rows=np.zeros((2, 40), dtype=bool)), "partition_rows has 2 rows, but partitions=2 needs 1"),
    (dict(partitions=2, partition_rows=np.zeros((1, 40), dtype=bool), partition_row_seed=1), "gives the rows outright"),
    (dict(partitions=2, partition_rows=np.zeros((1, 39), dtype=bool)), r"shape \(k, 40\)"),
    (dict(partition_rows=np.zeros((1, 40), dtype=bool)), "partition_rows= needs partitions= or comm="),
])
def test_bad_partition_row_keywords_are_value_errors(kwargs, message):
    if kwargs.get("partitions", 1) > len(numa_nodes()):
        pytest.skip("needs two NUMA domains")
    with pytest.raises(ValueError, match=message):
        _start().propagate(_fanout_circuit(N), **kwargs)


@needs_two_domains
def test_remote_layers_predict_the_engines_plan():
    circuit = _fanout_circuit(N)
    deltas = circuit.deltas()
    assert len(deltas) == len(circuit) and all(d.shape[1] == 2 * N for d in deltas)
    for rows in [hashes.dense(N, 1, 3), hashes.projection(N, 1, 5), hashes.dense(N, 1, 4, exclude={"x": list(range(N))})]:
        predicted = hashes.remote_layers(rows, deltas)
        stats = _start().propagate_with_stats(circuit, partitions=2, partition_rows=rows)
        planned = [predicted[i] for i in stats.circuit_index]
        assert [not local for local in stats.partition.local] == planned


def _kicked_ising_4x4():
    """One kicked-Ising step on the open 4 x 4 square lattice: ``rx`` on every qubit (layers 0-15), then ``ZZ`` on every bond."""
    circuit = paulistrings.Circuit(16)
    for q in range(16):
        circuit.rx(0.4, q)
    bonds = [(q, q + 1) for q in range(16) if q % 4 < 3] + [(q, q + 4) for q in range(12)]
    for a, b in bonds:
        circuit.pauli_rotation("ZZ", [a, b], 0.9)
    return circuit, bonds


@needs_two_domains
def test_auto_partition_rows_follow_the_observable_and_keep_the_result():
    circuit, bonds = _kicked_ising_4x4()
    site = PauliSum.from_strings({"I" * 5 + "Z" + "I" * 10: 1.0}, num_qubits=16)
    magnetization = PauliSum.from_strings({"I" * q + "Z" + "I" * (15 - q): 1 / 16 for q in range(16)}, num_qubits=16)
    # A one-site observable gets x-parity rows over itself and its lowest neighbour, qubits 5 and 1: only their kicks exchange.
    # The magnetization gets a z-only cut between qubits 0-7 and 8-15: only the four bonds across it exchange.
    crossing = {16 + i for i, (a, b) in enumerate(bonds) if (a < 8) != (b < 8)}
    for start, remote in [(site, {1, 5}), (magnetization, crossing)]:
        stats = start.copy().propagate_with_stats(circuit, partitions=2, partition_rows="auto")
        assert {i for i, local in zip(stats.circuit_index, stats.partition.local) if not local} == remote
        got = start.copy()
        got.propagate(circuit, partitions=2, partition_rows="auto")
        _assert_terms_close(got, propagated(start, circuit), tol=1e-10)


def test_auto_partition_rows_need_a_circuit():
    with pytest.raises(ValueError, match="only propagate and propagate_with_stats take it"):
        _start().partition(partitions=1, partition_rows="auto")
    with pytest.raises(ValueError, match='not "cut"'):
        _start().propagate(_fanout_circuit(N), partitions=1, partition_rows="cut")
    with pytest.raises(ValueError, match="takes none of"):
        _start().propagate(_fanout_circuit(N), partitions=1, partition_rows="auto", partition_row_seed=1)
