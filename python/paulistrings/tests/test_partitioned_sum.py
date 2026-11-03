"""``PauliSum.partition`` and ``PartitionedPauliSum`` in one process: a split kept between calls, whose partition and bucket rows change mid-circuit, gives the serial answer."""

import warnings

import numpy as np
import pytest

from paulistrings import PartitionedPauliSum, PauliSum, hashes, numa_nodes, truncation

from .test_bucket_knobs import _assert_terms_close, _fanout_circuit

N = 20
needs_two_domains = pytest.mark.skipif(len(numa_nodes()) < 2, reason="needs two NUMA domains")


def _start():
    return PauliSum.from_strings({"Z" + "I" * (N - 1): 1.0, "IX" + "I" * (N - 2): 0.5}, num_qubits=N)


@pytest.mark.parametrize("partitions", [1, pytest.param(2, marks=needs_two_domains)])
def test_a_split_propagates_to_the_serial_answer(partitions):
    circuit = _fanout_circuit(N)
    want = _start().propagate(circuit)
    split = _start().partition(partitions=partitions)
    assert isinstance(split, PartitionedPauliSum) and split.num_partitions == partitions and split.rank is None
    split.propagate(circuit)
    _assert_terms_close(split.gather(), want, tol=1e-10)
    _assert_terms_close(split.local(), want, tol=1e-10)
    assert len(split) == len(want)
    assert split.expectation("z+") == pytest.approx(want.expectation("z+"), abs=1e-10)
    assert "PartitionedPauliSum" in repr(split)


@needs_two_domains
def test_partition_and_bucket_rows_switch_mid_circuit_without_changing_the_result():
    circuit = _fanout_circuit(N)
    first, second = circuit[: len(circuit) // 2], circuit[len(circuit) // 2:]
    want = _start().propagate(first).propagate(second)
    split = _start().partition(partitions=2, partition_rows=hashes.dense(N, 1, 1))
    split.propagate(first)
    new_rows = hashes.dense(N, 1, 2, exclude={"x": list(range(N))})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        split.repartition(partition_rows=new_rows)
        split.rehash(rows=hashes.sparse(N, 4, 3, 7), seed=11)
    np.testing.assert_array_equal(split.partition_rows(), new_rows)
    np.testing.assert_array_equal(split.hash_rows()[:4], hashes.sparse(N, 4, 3, 7))
    assert split.hash_seed == 11
    stats = split.propagate_with_stats(second)
    assert stats.partition.partitions == 2 and stats.layers == len(second)
    _assert_terms_close(split.gather(), want, tol=1e-10)


@needs_two_domains
def test_the_planned_layers_follow_the_rows_after_a_repartition():
    circuit = _fanout_circuit(N)
    split = _start().partition(partitions=2)
    rows = hashes.projection(N, 1, 5)
    split.repartition(partition_rows=rows)
    stats = split.propagate_with_stats(circuit)
    predicted = hashes.remote_layers(rows, circuit.deltas())
    assert [not local for local in stats.partition.local] == [predicted[i] for i in stats.circuit_index]


def test_the_source_sum_is_untouched_and_the_split_is_independent():
    start = _start()
    split = start.partition(partitions=1)
    split.propagate(_fanout_circuit(N))
    assert len(start) == 2


def test_dependent_rows_warn():
    split = _start().partition(partitions=1)
    rows = np.zeros((2, 2 * N), dtype=bool)
    rows[0, 3] = rows[1, 3] = True
    with pytest.warns(RuntimeWarning, match="rank"):
        split.rehash(rows=rows)


@pytest.mark.parametrize("call, error, message", [
    (lambda s: s.repartition(partition_rows=np.zeros((2, 2 * N), dtype=bool)), ValueError, "needs 0"),
    (lambda s: s.repartition(partition_row_seed=1, partition_row_blocks=[list(range(N))]), ValueError, "alternatives"),
    (lambda s: s.rehash(rows=np.zeros((1, 2 * N + 1), dtype=bool)), ValueError, "shape"),
    (lambda s: s.propagate(_fanout_circuit(N), truncation.topn(10)), NotImplementedError, "approx_topn"),
    (lambda s: s.expectation("0" * (N - 1)), ValueError, "one character per qubit"),
])
def test_bad_calls_raise_before_anything_moves(call, error, message):
    split = _start().partition(partitions=1)
    with pytest.raises(error, match=message):
        call(split)
    assert len(split) == 2


def test_comm_and_partitions_are_alternatives():
    with pytest.raises((ValueError, RuntimeError)):
        _start().partition(partitions=1, comm=object())
