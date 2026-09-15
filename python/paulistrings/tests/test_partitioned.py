"""``partitions=`` / ``pin_memory=`` on the propagate surface.

The Rust side is covered by ``crates/paulistrings/tests/propagate_partitioned.rs``
(which pins the partitioned engine against the unpartitioned one, including a
``P = 1`` bitwise check) and by ``truncation_spec.rs``'s
``spec_finalize_partitioned_matches_core_builtins``. This file checks the Python
boundary: that the kwargs exist with the documented spellings and defaults, that
``partitions=None`` is today's path *exactly*, that a two-partition run agrees
with the unpartitioned one on both the terms and the per-layer term counts, that
the placement errors are clean ``ValueError``/``TypeError``s, that an exact
``topn`` is a ``NotImplementedError``, and that the trace surfaced as
``PropagationStats.partition`` is self-consistent.

Partitioned mode needs Linux CPU affinity, so the whole module skips elsewhere —
and on a single-CPU allocation, where there is nothing to place. Under
``taskset -c 0`` every test here is skipped rather than silently running
unpartitioned.

Two circuit fixtures, because term growth has to stay bounded at every width:
``_clifford_circuit`` is fanout-free (Cliffords plus coefficient-rescaling
noise), so the untruncated comparisons keep a fixed term count; the
fanout-carrying ``_mixed_circuit`` (generic-angle rotations, a multi-qubit
generator, a 2-qubit unitary, amplitude damping) is always run under a
truncation policy, which is also what exercises the *collective* layer
finalization on every layer.

Note CI does not run these (see CLAUDE.md); run them locally with
``maturin develop --release`` followed by ``pytest python/paulistrings/tests``.
"""

import os
import random
import sys

import numpy as np
import pytest

import paulistrings
from paulistrings import Circuit, PauliSum, numa_nodes, truncation


pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or len(os.sched_getaffinity(0)) < 2,
    reason="partitioned placement needs Linux CPU affinity and at least two CPUs",
)

# One per width band the bindings monomorphize: 8 -> W=1, 68 -> W=2, 130 -> W=4.
WIDTHS = [8, 68, 130]

# Coefficients agree to floating point, not bit for bit: the partitioned run
# sums equal keys in a different order (ARCHITECTURE.md §Determinism).
TOL = 1e-9

# Terms in the seeded observable — comfortably more than one bucket, so the
# scatter has something to split and each partition has real per-bucket work.
NUM_TERMS = 20_000

# Qubits the seeded circuits touch. The observable's support spans every word,
# so a narrow window still exercises the multi-word key path.
_WINDOW = 6


def _cpu_sets(count=2):
    """The first ``count`` CPUs of the affinity mask, one partition each.

    Explicit lists rather than ``"auto"``: a test must place the same way on a
    two-socket node and inside a one-core cgroup, and must not depend on the
    machine having as many NUMA nodes as it wants partitions.
    """
    return [[cpu] for cpu in sorted(os.sched_getaffinity(0))[:count]]


def _observable(num_qubits, terms=NUM_TERMS, seed=20260908):
    """A seeded random sum: keys spread over every word, complex coefficients."""
    rng = np.random.default_rng(seed + num_qubits)
    words = (num_qubits + 63) // 64
    x = rng.integers(0, 1 << 63, size=(terms, words), dtype=np.uint64)
    z = rng.integers(0, 1 << 63, size=(terms, words), dtype=np.uint64)
    # Clear the bits past `num_qubits` in the last word; `from_arrays` rejects
    # a key with support outside the register.
    tail = num_qubits - 64 * (words - 1)
    last = np.uint64((1 << tail) - 1) if tail < 64 else np.uint64((1 << 64) - 1)
    x[:, words - 1] &= last
    z[:, words - 1] &= last
    coeffs = rng.normal(size=terms) + 1j * rng.normal(size=terms)
    return PauliSum.from_arrays(x, z, coeffs, num_qubits=num_qubits)


def _clifford_circuit(num_qubits, layers=3, seed=20260908):
    """A deterministic fanout-free circuit: single- and two-qubit Cliffords
    (which relabel keys, so rows do cross partition boundaries) plus two
    coefficient-rescaling noise channels. The term count is bounded, so this is
    the fixture for the comparisons that run with no truncation at all."""
    rng = random.Random(seed)
    circuit = Circuit(num_qubits)
    for _ in range(layers):
        for q in range(_WINDOW):
            getattr(circuit, rng.choice(("h", "s", "sdg", "x", "z")))(q)
        for q in range(_WINDOW - 1):
            getattr(circuit, rng.choice(("cnot", "cz", "swap")))(q, q + 1)
        circuit.depolarize(0.02, [rng.randrange(_WINDOW)])
        circuit.dephase(0.01, [rng.randrange(_WINDOW)])
    return circuit


def _mixed_circuit(num_qubits, layers=2, seed=20260908):
    """A deterministic circuit with genuine fanout: generic-angle rotations, a
    weight-3 Pauli generator, a two-qubit unitary and amplitude damping. Run it
    under a truncation policy — untruncated it multiplies the term count by the
    window's key space."""
    rng = random.Random(seed)
    circuit = Circuit(num_qubits)
    swap = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    )
    for _ in range(layers):
        for q in range(_WINDOW):
            getattr(circuit, rng.choice(("rz", "rx", "ry")))(rng.uniform(0.1, 1.4), q)
        for q in range(_WINDOW - 1):
            circuit.cnot(q, q + 1)
        circuit.pauli_rotation("XYZ", [0, 2, 4], rng.uniform(0.1, 0.9))
        circuit.unitary_2q(1, 3, swap)
        circuit.amplitude_damping(0.05, [rng.randrange(_WINDOW)])
    return circuit


def _as_dict(sum_):
    """{(x_words, z_words): coeff}, so comparisons ignore storage order."""
    xs, zs, cs = sum_.x_array(), sum_.z_array(), sum_.coefficients_array()
    return {
        (tuple(int(v) for v in xs[i]), tuple(int(v) for v in zs[i])): complex(cs[i])
        for i in range(len(sum_))
    }


def _assert_terms_close(got, want, tol=TOL):
    a, b = _as_dict(got), _as_dict(want)
    assert set(a) == set(b), (
        f"different keys: {len(set(a) - set(b))} only in the first, "
        f"{len(set(b) - set(a))} only in the second"
    )
    worst = max((abs(a[k] - b[k]) for k in a), default=0.0)
    assert worst < tol, f"largest coefficient difference {worst:.3e} exceeds {tol:.1e}"


def _assert_counts_equal(got, want):
    assert got.layers == want.layers
    assert got.terms_in == want.terms_in
    assert got.terms_out == want.terms_out
    assert got.peak_terms == want.peak_terms
    assert got.final_terms == want.final_terms


# --------------------------------------------------------------------------
# The default is untouched


@pytest.mark.parametrize("partitions", [None, 1])
def test_none_and_one_are_the_classic_path(partitions):
    """``partitions=None``/``1`` must be today's path *bit for bit*, not merely
    close: the kwargs are additive, and ``1`` short-circuits before any pool is
    built."""
    s, c = _observable(WIDTHS[0]), _clifford_circuit(WIDTHS[0])
    assert _as_dict(s.propagate(c, partitions=partitions)) == _as_dict(s.propagate(c))


def test_unpartitioned_stats_report_no_partition():
    s, c = _observable(WIDTHS[0]), _clifford_circuit(WIDTHS[0])
    _, stats = s.propagate_with_stats(c)
    assert stats.partition is None
    _, stats = s.propagate_with_stats(c, partitions=1)
    assert stats.partition is None


# --------------------------------------------------------------------------
# Two partitions agree with none


@pytest.mark.parametrize("num_qubits", WIDTHS)
@pytest.mark.parametrize("direction", ["forward", "heisenberg"])
def test_two_partitions_match_unpartitioned(num_qubits, direction):
    s, c = _observable(num_qubits), _clifford_circuit(num_qubits)
    want, want_stats = s.propagate_with_stats(c, direction=direction)
    got, got_stats = s.propagate_with_stats(
        c, direction=direction, partitions=_cpu_sets()
    )
    _assert_terms_close(got, want)
    # The whole per-layer vectors, not just the totals: the partitioned layer
    # loop must apply the same truncation at the same points.
    _assert_counts_equal(got_stats, want_stats)
    assert got_stats.partition.partitions == 2


@pytest.mark.parametrize("num_qubits", WIDTHS)
@pytest.mark.parametrize("direction", ["forward", "heisenberg"])
def test_two_partitions_match_unpartitioned_with_fanout(num_qubits, direction):
    """The same check on a circuit with real fanout, under the collective
    ``approx_topn`` that bounds its growth."""
    s, c = _observable(num_qubits), _mixed_circuit(num_qubits)
    policy = truncation.approx_topn(NUM_TERMS)
    want, want_stats = s.propagate_with_stats(c, policy, direction=direction)
    got, got_stats = s.propagate_with_stats(
        c, policy, direction=direction, partitions=_cpu_sets()
    )
    _assert_terms_close(got, want)
    _assert_counts_equal(got_stats, want_stats)


def test_pin_memory_false_matches():
    """``pin_memory=False`` pins threads but not allocations — a placement
    knob, never a semantic one."""
    s, c = _observable(WIDTHS[1]), _clifford_circuit(WIDTHS[1])
    want = s.propagate(c)
    got = s.propagate(c, partitions=_cpu_sets(), pin_memory=False)
    _assert_terms_close(got, want)


def test_auto_places_one_partition_per_numa_node():
    s, c = _observable(WIDTHS[0]), _clifford_circuit(WIDTHS[0])
    got, stats = s.propagate_with_stats(c, partitions="auto")
    _assert_terms_close(got, s.propagate(c))
    # A single-node box (or an affinity mask inside one node) runs the classic
    # path and reports no partition at all; anything else places one partition
    # per node, rounded down to a power of two.
    nodes = len(numa_nodes())
    if stats.partition is None:
        assert nodes == 1
    else:
        assert stats.partition.partitions == 1 << (nodes.bit_length() - 1)
        assert stats.partition.partitions <= nodes


def test_the_kwargs_are_accepted_positionally_after_small_sum_threshold():
    s, c = _observable(WIDTHS[0]), _clifford_circuit(WIDTHS[0])
    positional = s.propagate(
        c, None, "forward", "sorted", 4096, None, None, _cpu_sets(), False
    )
    keyword = s.propagate(
        c,
        engine="sorted",
        small_sum_threshold=4096,
        partitions=_cpu_sets(),
        pin_memory=False,
    )
    assert _as_dict(positional) == _as_dict(keyword)


def test_a_trotter_loop_reuses_the_runtime():
    """Three calls in a row on one placement: the pinned pools are cached per
    config, so this must not spawn (and leak) a pool set per step. Nothing
    observable asserts the reuse — what is asserted is that stepping works and
    still agrees with the unpartitioned answer."""
    s, c = _observable(WIDTHS[0]), _mixed_circuit(WIDTHS[0], layers=1)
    policy = truncation.approx_topn(5_000)
    partitioned, plain = s, s
    for _ in range(3):
        partitioned = partitioned.propagate(c, policy, partitions=_cpu_sets())
        plain = plain.propagate(c, policy)
    _assert_terms_close(partitioned, plain)


# --------------------------------------------------------------------------
# Truncation


def test_approx_topn_agrees_with_unpartitioned():
    """The collective octave histogram: the union of what the partitions keep
    is what one partition holding everything would have kept."""
    s, c = _observable(WIDTHS[1]), _mixed_circuit(WIDTHS[1])
    policy = truncation.coeff(1e-12) & truncation.approx_topn(4_000)
    want, want_stats = s.propagate_with_stats(c, policy)
    got, got_stats = s.propagate_with_stats(c, policy, partitions=_cpu_sets())
    assert max(want_stats.terms_out) <= 4_000, "the policy must actually bite"
    _assert_terms_close(got, want)
    _assert_counts_equal(got_stats, want_stats)


@pytest.mark.parametrize(
    "policy",
    [
        truncation.topn(1_000),
        truncation.coeff(1e-9) & truncation.topn(1_000),
        truncation.topn(1_000) | truncation.coeff(1e-9),
    ],
    ids=["topn", "coeff & topn", "topn | coeff"],
)
def test_exact_topn_is_rejected_in_partitioned_mode(policy):
    s, c = _observable(WIDTHS[0]), _clifford_circuit(WIDTHS[0], layers=1)
    with pytest.raises(NotImplementedError, match="approx_topn"):
        s.propagate(c, policy, partitions=_cpu_sets())
    with pytest.raises(NotImplementedError, match="partitions="):
        s.propagate_with_stats(c, policy, partitions=_cpu_sets())
    # ... and it is a *partitioned*-mode restriction only: the same policy is
    # accepted unpartitioned. (Only the `&` and bare forms truncate there —
    # `Or`'s layer pass is the no-op default in both modes, which is exactly
    # why `spec_has_exact_topn` refuses it rather than honouring it in one.)
    assert len(s.propagate(c, policy)) > 0


# --------------------------------------------------------------------------
# Placement errors


@pytest.mark.parametrize(
    "partitions, message",
    [
        (3, "power of two"),
        (0, "at least one partition"),
        ([[0], [0]], "disjoint"),
        ([[]], "empty"),
        ([], "at least one CPU list"),
        ([[10**6]], "affinity mask"),
        ([[0], [1], [2]], "power of two"),
        ("foo", "'auto'"),
    ],
)
def test_bad_partitions_are_value_errors(partitions, message):
    s, c = _observable(WIDTHS[0], terms=64), _clifford_circuit(WIDTHS[0], layers=1)
    with pytest.raises(ValueError, match=message):
        s.propagate(c, partitions=partitions)


def test_more_partitions_than_nodes_is_rejected():
    """``partitions=k`` places one partition per NUMA node, so it refuses a
    count the machine cannot honour — and says how to place it anyway."""
    nodes = len(numa_nodes())
    too_many = 1 << nodes.bit_length()  # strictly more than `nodes`
    s, c = _observable(WIDTHS[0], terms=64), _clifford_circuit(WIDTHS[0], layers=1)
    with pytest.raises(ValueError, match="NUMA nodes"):
        s.propagate(c, partitions=too_many)


@pytest.mark.parametrize("partitions", [2.5, True, {"cpus": [0]}, [0, 1]])
def test_bad_partitions_types_are_type_errors(partitions):
    s, c = _observable(WIDTHS[0], terms=64), _clifford_circuit(WIDTHS[0], layers=1)
    with pytest.raises(TypeError):
        s.propagate(c, partitions=partitions)


# --------------------------------------------------------------------------
# The trace


def test_partition_stats_are_self_consistent():
    s, c = _observable(WIDTHS[0]), _clifford_circuit(WIDTHS[0])
    _, stats = s.propagate_with_stats(c, partitions=_cpu_sets())
    part = stats.partition

    assert part.partitions == 2
    for field in (
        part.local,
        part.rows_exported,
        part.bytes_exported,
        part.terms_in,
        part.terms_out,
        part.imbalance,
        part.nanos,
    ):
        assert len(field) == stats.layers

    for k in range(stats.layers):
        # A layer is local exactly when it exported nothing.
        assert part.local[k] == (part.rows_exported[k] == 0)
        assert (part.bytes_exported[k] == 0) == part.local[k]
        # The per-partition counts are a partition of the layer's counts.
        assert len(part.terms_in[k]) == part.partitions
        assert len(part.terms_out[k]) == part.partitions
        assert sum(part.terms_in[k]) == stats.terms_in[k]
        assert sum(part.terms_out[k]) == stats.terms_out[k]
        # Random partition rows on a 20k-term sum are near-perfectly balanced;
        # the bound that always holds is 1 <= imbalance <= P.
        assert 1.0 <= part.imbalance[k] <= part.partitions
        assert part.imbalance[k] < 1.2
        # The top-level `nanos` is the critical-rank proxy: the max over the
        # raw per-partition timings this record carries.
        assert len(part.nanos[k]) == part.partitions
        assert stats.nanos[k] == max(part.nanos[k])
        # `circuit_index`/`application_index`/`gate_name` are agreed collective
        # values, identical on every partition by construction.
        assert stats.circuit_index[k] == k
        assert stats.application_index[k] == k

    # This circuit does move rows: a partitioned run that never exchanged
    # anything would not be testing the exchange at all.
    assert any(not local for local in part.local)


def test_partition_stats_repr_names_all_fields():
    s, c = _observable(WIDTHS[0], terms=64), _clifford_circuit(WIDTHS[0], layers=1)
    _, stats = s.propagate_with_stats(c, partitions=_cpu_sets())
    text = repr(stats.partition)
    assert text.startswith("PartitionStats(")
    for field in (
        "partitions=",
        "local=",
        "rows_exported=",
        "bytes_exported=",
        "terms_in=",
        "terms_out=",
        "imbalance=",
    ):
        assert field in text
    assert f"partitions={stats.partition.partitions}" in text
    # Python spellings, so the line pastes back into a REPL.
    assert "true" not in text and "false" not in text


def test_propagation_stats_repr_is_unchanged_by_a_partitioned_run():
    """``PropagationStats.__repr__`` is pinned byte for byte
    (``test_propagation_stats.py``); the partition record is reached through
    ``stats.partition``, never through the repr."""
    s = PauliSum.from_strings({"X" + "I" * 7: 1.0}, num_qubits=8)
    c = Circuit(8)
    c.h(0)
    _, stats = s.propagate_with_stats(c, partitions=_cpu_sets())
    assert repr(stats) == (
        "PropagationStats(layers=1, terms_in=[1], terms_out=[1], "
        "peak_terms=1, final_terms=1)"
    )
    assert stats.partition is not None


# --------------------------------------------------------------------------
# Topology discovery


def test_numa_nodes_reports_cpu_lists():
    nodes = numa_nodes()
    assert nodes and all(cpus for cpus in nodes)
    allowed = os.sched_getaffinity(0)
    seen = set()
    for cpus in nodes:
        assert cpus == sorted(cpus), "each node's CPU list is ascending"
        assert set(cpus) <= allowed, "nodes are intersected with the affinity mask"
        assert not seen & set(cpus), "nodes do not overlap"
        seen |= set(cpus)


def test_numa_nodes_is_exported():
    assert paulistrings.numa_nodes is numa_nodes
    assert "numa_nodes" in paulistrings.__all__
    assert "PartitionStats" in paulistrings.__all__


# --------------------------------------------------------------------------
# partition_row_seed / partition_row_blocks (E8: random vs. designed-cut rows)
#
# Closes the campaign's E8 gap (quera-talk-data/campaign-2026-09-11/decisions.md
# #13): before this, `sum.rs` hardcoded `partition_row_seed: None` and there
# was no way at all to choose explicit rows from Python, so a random-vs-cut
# communication-volume comparison could not be collected.


def test_default_partition_rows_are_unchanged():
    """Neither new kwarg touches the result when both are left at `None` —
    the additive-kwarg contract every other partitioned knob already has."""
    s, c = _observable(WIDTHS[0]), _clifford_circuit(WIDTHS[0])
    want = s.propagate(c, partitions=_cpu_sets())
    got = s.propagate(c, partitions=_cpu_sets(), partition_row_seed=None, partition_row_blocks=None)
    assert _as_dict(got) == _as_dict(want)


def test_distinct_seeds_move_different_rows():
    """Two different `partition_row_seed`s draw two different GF(2) row sets,
    so the per-layer export volume differs -- proof the seed actually reaches
    the engine (it used to be silently dropped)."""
    s, c = _observable(WIDTHS[0]), _mixed_circuit(WIDTHS[0])
    policy = truncation.approx_topn(NUM_TERMS)
    _, stats_a = s.propagate_with_stats(
        c, policy, partitions=_cpu_sets(), partition_row_seed=1
    )
    _, stats_b = s.propagate_with_stats(
        c, policy, partitions=_cpu_sets(), partition_row_seed=2
    )
    assert stats_a.partition.rows_exported != stats_b.partition.rows_exported


@pytest.mark.parametrize("num_qubits", WIDTHS)
def test_seeded_and_cut_rows_agree_with_unpartitioned(num_qubits):
    """Both new row policies are placement knobs, not semantic ones: the
    returned sum and every per-layer term count still match the unpartitioned
    run, exactly like the existing seeded-random path."""
    s, c = _observable(num_qubits), _mixed_circuit(num_qubits)
    policy = truncation.approx_topn(NUM_TERMS)
    want, want_stats = s.propagate_with_stats(c, policy)

    half = num_qubits // 2
    blocks = [list(range(half)), list(range(half, num_qubits))]
    got_cut, stats_cut = s.propagate_with_stats(
        c, policy, partitions=_cpu_sets(), partition_row_blocks=blocks
    )
    _assert_terms_close(got_cut, want)
    _assert_counts_equal(stats_cut, want_stats)

    got_seed, stats_seed = s.propagate_with_stats(
        c, policy, partitions=_cpu_sets(), partition_row_seed=7
    )
    _assert_terms_close(got_seed, want)
    _assert_counts_equal(stats_seed, want_stats)


def test_cut_blocks_round_trip_to_the_named_partition():
    """A block's qubits actually land in that block's partition: a term whose
    only support is in block 1 always crosses when scattered from a sum built
    entirely of block-0 terms, and never crosses when its support matches the
    cut."""
    n = WIDTHS[0]
    half = n // 2
    blocks = [list(range(half)), list(range(half, n))]

    # An all-Z observable inside block 0 only: `PartitionRows::cut`'s own
    # tests establish that a term's partition is the XOR of the z-weight-odd
    # blocks, so a single Z in block 0 belongs to partition 0.
    s = PauliSum.from_strings({"Z" + "I" * (n - 1): 1.0}, num_qubits=n)
    c = Circuit(n)
    c.rz(0.3, 0)  # Z survives an RZ unchanged; keeps the term in one bucket.
    _, stats = s.propagate_with_stats(
        c, partitions=_cpu_sets(), partition_row_blocks=blocks
    )
    assert stats.partition.rows_exported == [0]


def test_partition_row_blocks_wrong_count_is_value_error():
    s, c = _observable(WIDTHS[0]), _clifford_circuit(WIDTHS[0])
    with pytest.raises(ValueError, match="partition_row_blocks"):
        s.propagate(c, partitions=_cpu_sets(), partition_row_blocks=[[0, 1]])


def test_partition_row_blocks_overlap_is_value_error():
    s, c = _observable(WIDTHS[0]), _clifford_circuit(WIDTHS[0])
    with pytest.raises(ValueError, match="more than one block"):
        s.propagate(
            c,
            partitions=_cpu_sets(),
            partition_row_blocks=[[0, 1], [1, 2, 3, 4, 5, 6, 7]],
        )


def test_partition_row_blocks_out_of_range_qubit_is_value_error():
    s, c = _observable(WIDTHS[0]), _clifford_circuit(WIDTHS[0])
    with pytest.raises(ValueError, match="out of range"):
        s.propagate(
            c, partitions=_cpu_sets(), partition_row_blocks=[[0, 99], [1, 2]]
        )


def test_partition_row_blocks_needs_partitions():
    s, c = _observable(WIDTHS[0]), _clifford_circuit(WIDTHS[0])
    with pytest.raises(ValueError, match="needs partitions"):
        s.propagate(c, partition_row_blocks=[[0, 1, 2, 3, 4, 5, 6, 7]])


def test_seed_and_blocks_are_mutually_exclusive():
    s, c = _observable(WIDTHS[0]), _clifford_circuit(WIDTHS[0])
    with pytest.raises(ValueError, match="alternatives"):
        s.propagate(
            c,
            partitions=_cpu_sets(),
            partition_row_seed=1,
            partition_row_blocks=[[0, 1, 2, 3], [4, 5, 6, 7]],
        )
