"""``PauliSum.with_hash``: the bucket hash's family and seed, which change speed but never results."""

import warnings

import numpy as np
import pytest

from paulistrings import PauliSum, numa_nodes

from .test_bucket_knobs import _assert_terms_close, _fanout_circuit

FAMILIES = ["dense", "sparse:3", "sparse:2", "projection"]


def _start(num_qubits=20):
    return PauliSum.from_strings({"Z" + "I" * (num_qubits - 1): 1.0, "IX" + "I" * (num_qubits - 2): 0.5}, num_qubits=num_qubits)


def test_a_new_sum_carries_the_dense_family():
    assert _start().hash_family == "dense"


@pytest.mark.parametrize("family", FAMILIES)
def test_with_hash_keeps_the_terms_and_reports_the_family(family):
    start = _start()
    rehashed = start.with_hash(family, seed=7)
    assert (rehashed.hash_family, rehashed.hash_seed) == (family, 7)
    _assert_terms_close(rehashed, start)
    assert start.hash_family == "dense", "with_hash returns a copy"


def test_bucket_lengths_count_every_term():
    s = _start().propagate(_fanout_circuit(20)).with_hash("sparse:3", seed=5)
    lengths = s.bucket_lengths()
    assert len(lengths) == s.num_buckets
    assert lengths.sum() == len(s)


def test_the_seed_defaults_to_the_sums_own():
    start = _start()
    assert start.with_hash("projection").hash_seed == start.hash_seed


@pytest.mark.parametrize("family", FAMILIES[1:])
def test_propagation_keeps_the_family_and_the_result(family):
    circuit = _fanout_circuit(20)
    want = _start().propagate(circuit)
    got = _start().with_hash(family, seed=11).propagate(circuit)
    assert got.hash_family == family
    assert got.num_buckets > 1
    _assert_terms_close(got, want, tol=1e-10)


def test_partitions_keep_the_result_under_another_family():
    if len(numa_nodes()) < 2:
        pytest.skip("needs two NUMA domains")
    circuit = _fanout_circuit(20)
    want = _start().propagate(circuit)
    got = _start().with_hash("sparse:2", seed=3).propagate(circuit, partitions=2)
    _assert_terms_close(got, want, tol=1e-10)


@pytest.mark.parametrize("family, message", [
    ("sparse:0", r"must lie in 1\.\.=40"),
    ("sparse:41", r"must lie in 1\.\.=40"),
    ("sparse", "hash family must be"),
    ("sparse:x", "hash family must be"),
    ("xorshift", "hash family must be"),
])
def test_a_bad_family_is_a_value_error(family, message):
    with pytest.raises(ValueError, match=message):
        _start().with_hash(family)


def test_a_family_that_cannot_reach_full_rank_warns_and_still_works():
    small = PauliSum.from_strings({"ZIIII": 1.0})
    with pytest.warns(RuntimeWarning, match="rank 9"):
        rehashed = small.with_hash("sparse:2")
    assert rehashed.hash_family == "sparse:2"
    with pytest.warns(RuntimeWarning, match="rank 1"):
        _start().with_hash("sparse:40")


def _rows(num_qubits, coordinates):
    """A (len(coordinates), 2 * num_qubits) boolean matrix with row i reading coordinates[i]."""
    rows = np.zeros((len(coordinates), 2 * num_qubits), dtype=bool)
    for i, row in enumerate(coordinates):
        rows[i, row] = True
    return rows


def test_hash_rows_lay_out_the_rows_x_bits_first():
    rows = _start().hash_rows()
    assert rows.shape == (22, 40) and rows.dtype == bool
    assert rows.any(axis=1).all(), "no dense row is empty"


def test_explicit_rows_lead_and_dense_rows_complete_the_hash():
    given = _rows(20, [[0, 39], [5], [21, 3]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rehashed = _start().with_hash(rows=given, seed=9)
    assert rehashed.hash_family == "explicit:3"
    np.testing.assert_array_equal(rehashed.hash_rows()[:3], given)
    copied = _start().with_hash(rows=_start().with_hash("sparse:3", seed=4).hash_rows())
    np.testing.assert_array_equal(copied.hash_rows(), _start().with_hash("sparse:3", seed=4).hash_rows())


def test_dependent_explicit_rows_warn_and_still_give_the_same_result():
    circuit = _fanout_circuit(20)
    with pytest.warns(RuntimeWarning, match="rank 20"):
        rehashed = _start().with_hash(rows=_rows(20, [[3], [3], []]))
    _assert_terms_close(rehashed.propagate(circuit), _start().propagate(circuit), tol=1e-10)


@pytest.mark.parametrize("kwargs, message", [
    (dict(family="dense", rows=np.zeros((1, 40), dtype=bool)), "family= or rows=, not both"),
    (dict(rows=np.zeros((1, 39), dtype=bool)), r"shape \(k, 40\)"),
    (dict(rows=np.zeros((23, 40), dtype=bool)), r"k <= 22"),
])
def test_bad_explicit_rows_are_value_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _start().with_hash(**kwargs)


needs_two_domains = pytest.mark.skipif(len(numa_nodes()) < 2, reason="needs two NUMA domains")


@needs_two_domains
@pytest.mark.parametrize("kwargs", [
    dict(partition_row_family="sparse:3"),
    dict(partition_row_family="projection", partition_row_seed=7),
    dict(partition_row_family="sparse:5", partition_row_exclude={"z": list(range(20))}),
    dict(partition_rows=_rows(20, [[0, 1, 22]])),
])
def test_partition_rows_by_family_or_explicitly_keep_the_result(kwargs):
    circuit = _fanout_circuit(20)
    want = _start().propagate(circuit)
    stats_free = _start().propagate(circuit, partitions=2, **kwargs)
    _assert_terms_close(stats_free, want, tol=1e-10)


@needs_two_domains
def test_dependent_partition_rows_warn():
    with pytest.warns(RuntimeWarning, match="only 1 of the 2 partitions"):
        _start().propagate(_fanout_circuit(20), partitions=2, partition_rows=np.zeros((1, 40), dtype=bool))


@needs_two_domains
def test_partition_rows_in_the_bucket_hash_span_warn():
    start = _start()
    with pytest.warns(RuntimeWarning, match="overlap the span of the bucket hash"):
        start.propagate(_fanout_circuit(20), partitions=2, partition_rows=start.hash_rows()[:1])


@pytest.mark.parametrize("kwargs, message", [
    (dict(partitions=2, partition_rows=np.zeros((2, 40), dtype=bool)), "partition_rows has 2 rows, but partitions=2 needs 1"),
    (dict(partitions=2, partition_rows=np.zeros((1, 40), dtype=bool), partition_row_seed=1), "gives the rows outright"),
    (dict(partitions=2, partition_row_family="sparse:3", partition_row_blocks=[list(range(10)), list(range(10, 20))]), "partition_row_family= are alternatives"),
    (dict(partition_row_family="sparse:3"), "partition_row_family= needs partitions= or comm="),
    (dict(partition_rows=np.zeros((1, 40), dtype=bool)), "partition_rows= needs partitions= or comm="),
    (dict(partitions=2, partition_row_family="sparse:21", partition_row_exclude={"z": list(range(20))}), r"must lie in 1\.\.=20"),
])
def test_bad_partition_row_keywords_are_value_errors(kwargs, message):
    if kwargs.get("partitions", 1) > len(numa_nodes()):
        pytest.skip("needs two NUMA domains")
    with pytest.raises(ValueError, match=message):
        _start().propagate(_fanout_circuit(20), **kwargs)
