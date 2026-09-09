"""``comm=`` / ``result=`` on the propagate surface — the distributed path.

The Rust side is covered by ``crates/paulistrings/tests/mpi_ranks.rs``, the
multi-rank differential net that ``scripts/mpi-test.sh`` drives. This file
checks the Python boundary: that an mpi4py communicator is adopted correctly,
that ``result="gather"`` and ``result="local"`` hand back what they promise,
that the placement kwargs are mutually exclusive, and that the stats record
names the group.

Every test here is written to be **deterministic at any world size**, so the
same file is the whole net at one rank (a plain ``pytest`` run, where MPI
initializes as a singleton world) and at 2 or 4 ranks under::

    mpirun -n 4 python -m pytest python/paulistrings/tests/test_mpi.py -q \
        -p no:cacheprovider -p no:randomly

or, equivalently, ``scripts/mpi-test.sh --ranks 2,4 --python``.

Two rules keep it from deadlocking, and both are load-bearing:

* **No rank-dependent skip or branch around a collective.** Every rank reaches
  every ``propagate(comm=...)`` and every ``COMM.allreduce`` in the same order;
  the only rank-dependent code is an *assertion* after the collective has
  returned. The module-level skips are functions of the build and of the world
  size, which every rank agrees on.
* **Collective order is source order**, hence ``-p no:randomly``: a plugin that
  shuffles the test order would shuffle it independently per rank.

The input must be **replicated** — byte for byte the same terms on every rank —
because the scatter is a local filter of a sum every rank already holds, not a
distribution of rank 0's copy. Here that is a seeded NumPy generator, which is
the pattern to copy in a real script.
"""

import numpy as np
import pytest

import paulistrings
from paulistrings import Circuit, PauliSum, truncation

pytest.importorskip("mpi4py")

# Before `mpi4py.MPI` is imported, because that import calls MPI_Init: a
# default (non-MPI) build must not pay for it, or fail on a host whose MPI is
# only usable under a launcher.
if not paulistrings.mpi_available():
    pytest.skip(
        "built without the mpi feature; rebuild with `maturin develop --features mpi`",
        allow_module_level=True,
    )

import mpi4py  # noqa: E402

# Must be set *before* `mpi4py.MPI` is imported, which is what calls MPI_Init.
# SERIALIZED is the minimum the engine needs (its layer loop calls MPI from a
# pinned Rayon pool worker), so the tests run at exactly the minimum rather
# than at mpi4py's more permissive "multiple" default. A no-op if some other
# module already brought MPI up.
mpi4py.rc.thread_level = "serialized"

from mpi4py import MPI  # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
POWER_OF_TWO = SIZE & (SIZE - 1) == 0

if MPI.Query_thread() < MPI.THREAD_SERIALIZED:
    _SKIP = (
        f"MPI provides thread level {MPI.Query_thread()}, below "
        f"MPI_THREAD_SERIALIZED ({MPI.THREAD_SERIALIZED})"
    )
elif not POWER_OF_TWO:
    # Every propagate below would raise the same ValueError on every rank —
    # correct, but it says nothing this file is here to say. The mapping is
    # covered by `test_a_non_power_of_two_group_is_a_value_error`, which makes
    # its own three-rank group.
    _SKIP = f"launched with {SIZE} ranks; a partitioning needs a power-of-two world"
else:
    _SKIP = None

pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

# W = 2: two words per key, so the multi-word paths run on the wire too.
NUM_QUBITS = 68
NUM_TERMS = 20_000
# The circuit's fanout is bounded by this, and it is loose enough that the
# octave histogram keeps everything at the sizes here — so the comparisons are
# against an untruncated answer while still exercising the *collective* layer
# pass on every layer.
POLICY = truncation.approx_topn(NUM_TERMS * 8)
# A policy that really does cut, to exercise the collective octave reduction
# where the retained set is a strict subset.
TIGHT_POLICY = truncation.approx_topn(NUM_TERMS // 4)
# Coefficients agree to floating point, not bit for bit: the distributed run
# sums equal keys in a different order (ARCHITECTURE.md §Determinism).
TOL = 1e-9
# Qubits the circuit touches. The observable's support spans both words, so a
# narrow window still exercises the multi-word key path.
_WINDOW = 6


def _observable(num_qubits=NUM_QUBITS, terms=NUM_TERMS, seed=20260908):
    """A seeded random sum — identical on every rank, which is the contract."""
    rng = np.random.default_rng(seed + num_qubits)
    words = (num_qubits + 63) // 64
    x = rng.integers(0, 1 << 63, size=(terms, words), dtype=np.uint64)
    z = rng.integers(0, 1 << 63, size=(terms, words), dtype=np.uint64)
    tail = num_qubits - 64 * (words - 1)
    last = np.uint64((1 << tail) - 1) if tail < 64 else np.uint64((1 << 64) - 1)
    x[:, words - 1] &= last
    z[:, words - 1] &= last
    coeffs = rng.normal(size=terms) + 1j * rng.normal(size=terms)
    return PauliSum.from_arrays(x, z, coeffs, num_qubits=num_qubits)


def _circuit(num_qubits=NUM_QUBITS):
    """Rotation-heavy, with one `cnot` and one general two-qubit unitary.

    Generic angles, so every rotation splits the terms that anticommute with
    its generator and rows genuinely cross rank boundaries; the `unitary_2q` is
    a fixed non-Clifford two-qubit gate, which is the dense-PTM path."""
    circuit = Circuit(num_qubits)
    for q, angle in enumerate((0.37, 0.81, 1.13, 0.59)):
        getattr(circuit, ("rz", "rx", "ry", "rz")[q])(angle, q)
    circuit.pauli_rotation("XYZ", [0, 2, 4], 0.43)
    circuit.cnot(0, 1)
    # A fixed two-qubit unitary with no special structure: exp(-i theta XX/2)
    # in the computational basis, written out so the test needs no linear
    # algebra to build it.
    theta = 0.7
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    xx = np.array(
        [[c, 0, 0, s], [0, c, s, 0], [0, s, c, 0], [s, 0, 0, c]], dtype=np.complex128
    )
    circuit.unitary_2q(2, _WINDOW - 1, xx)
    circuit.depolarize(0.02, [3])
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


def _key_checksum(sum_):
    """An order-independent 64-bit fold of the *keys*: each row mixed into one
    word, the rows XORed together.

    XOR is associative and commutative, so disjoint shares fold to exactly what
    the whole sum folds to — which is how a ``result="local"`` run is checked
    for coverage without shipping a hundred thousand keys to rank 0."""
    xs, zs = sum_.x_array(), sum_.z_array()
    rows = np.zeros(len(sum_), dtype=np.uint64)
    golden = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        for col in range(xs.shape[1]):
            rows = (rows ^ xs[:, col]) * golden
            rows = (rows ^ zs[:, col]) * golden
    return int(np.bitwise_xor.reduce(rows)) if len(sum_) else 0


@pytest.fixture(scope="module")
def observable():
    return _observable()


@pytest.fixture(scope="module")
def circuit():
    return _circuit()


@pytest.fixture(scope="module")
def reference(observable, circuit):
    """The serial answer, computed identically on every rank. Every rank builds
    it — it is the oracle each of them compares against, and computing it only
    on rank 0 would make the fixture rank-dependent."""
    return observable.propagate(circuit, POLICY)


# --------------------------------------------------------------------------
# result="gather"


def test_gather_returns_the_whole_sum_on_rank_zero(observable, circuit, reference):
    got = observable.propagate(circuit, POLICY, comm=COMM)
    # Collectives are done; from here the assertions may differ per rank.
    assert got.num_qubits == NUM_QUBITS
    if RANK == 0:
        assert len(got) == len(reference)
        _assert_terms_close(got, reference)
    else:
        assert len(got) == 0


def test_gather_agrees_under_a_truncating_approx_topn(observable, circuit):
    """`approx_topn` all-reduces its octave histogram, so the retained set is
    exactly the serial one even when it really does cut. Spelled with an
    explicit `result="gather"`, which must mean what the default means."""
    want = observable.propagate(circuit, TIGHT_POLICY)
    got = observable.propagate(circuit, TIGHT_POLICY, comm=COMM, result="gather")
    assert len(want) < NUM_TERMS, "the tight policy must actually truncate"
    if RANK == 0:
        assert len(got) == len(want)
        _assert_terms_close(got, want)
    else:
        assert len(got) == 0


def test_gather_respects_the_direction(observable, circuit, reference):
    want = observable.propagate(circuit, POLICY, direction="heisenberg")
    got = observable.propagate(circuit, POLICY, direction="heisenberg", comm=COMM)
    if RANK == 0:
        _assert_terms_close(got, want)
        # Guard against this being the forward run by accident: applying the
        # adjoints in reverse must give a different answer.
        assert _as_dict(want) != _as_dict(reference)


# --------------------------------------------------------------------------
# result="local"


def test_local_shares_partition_the_gathered_sum(observable, circuit, reference):
    local = observable.propagate(circuit, POLICY, comm=COMM, result="local")
    # Every reduction below runs on every rank, unconditionally.
    total_terms = COMM.allreduce(len(local))
    total_coeff = COMM.allreduce(complex(local.coefficients_array().sum()))
    total_expectation = COMM.allreduce(local.expectation("z+"))

    assert local.num_qubits == NUM_QUBITS
    assert total_terms == len(reference)
    want_coeff = complex(reference.coefficients_array().sum())
    assert abs(total_coeff - want_coeff) < TOL * max(1.0, abs(want_coeff))
    assert abs(total_expectation - reference.expectation("z+")) < TOL


def test_local_shares_cover_exactly_the_gathered_keys(observable, circuit, reference):
    """The keys, not just the counts: XOR-folding every rank's share must give
    the fold of the whole sum, which with the equal term count above says the
    shares tile the result and overlap nowhere."""
    local = observable.propagate(circuit, POLICY, comm=COMM, result="local")
    fold = COMM.allreduce(_key_checksum(local), op=MPI.BXOR)
    assert fold == _key_checksum(reference)


# --------------------------------------------------------------------------
# Stats


def test_stats_name_the_group(observable, circuit, reference):
    got, stats = observable.propagate_with_stats(circuit, POLICY, comm=COMM)
    part = stats.partition
    assert part is not None
    assert part.size == SIZE
    assert part.rank == RANK
    assert part.partitions == SIZE
    # One entry per layer, in application order, and the per-partition
    # dimension is this rank's entry alone.
    assert stats.layers == len(circuit)
    assert len(part.terms_in) == len(circuit)
    assert all(len(row) == 1 for row in part.terms_in)
    assert all(len(row) == 1 for row in part.terms_out)
    assert len(part.rows_exported) == len(circuit)
    assert len(part.local) == len(circuit)
    assert part.local == [rows == 0 for rows in part.rows_exported]
    if RANK == 0:
        assert len(got) == len(reference)


def test_stats_repr_names_rank_and_size(observable, circuit):
    _, stats = observable.propagate_with_stats(circuit, POLICY, comm=COMM)
    text = repr(stats.partition)
    assert f"rank={RANK}" in text
    assert f"size={SIZE}" in text


def test_in_process_stats_still_report_no_rank(observable, circuit):
    """`rank`/`size` are `None` for a run that is not distributed — the phase-1
    records are unchanged."""
    _, stats = observable.propagate_with_stats(circuit, POLICY)
    assert stats.partition is None


def test_multi_rank_run_actually_exchanges(observable, circuit):
    """At more than one rank the circuit must move rows across a boundary; a
    run that never exchanged would not be testing the transport at all."""
    _, stats = observable.propagate_with_stats(circuit, POLICY, comm=COMM)
    exported = COMM.allreduce(int(sum(stats.partition.rows_exported)))
    if SIZE > 1:
        assert exported > 0
    else:
        assert exported == 0


# --------------------------------------------------------------------------
# Errors — every one of them raised before any collective, on every rank alike


def test_exact_topn_is_not_implemented(observable, circuit):
    with pytest.raises(NotImplementedError, match="topn"):
        observable.propagate(circuit, truncation.topn(1000), comm=COMM)


def test_comm_and_partitions_are_alternatives(observable, circuit):
    with pytest.raises(ValueError, match="alternatives"):
        observable.propagate(circuit, POLICY, partitions=2, comm=COMM)


def test_an_unknown_result_is_a_value_error(observable, circuit):
    with pytest.raises(ValueError, match="result must be"):
        observable.propagate(circuit, POLICY, comm=COMM, result="root")
    with pytest.raises(ValueError, match="result must be"):
        observable.propagate(circuit, POLICY, result="root")


def test_comm_must_be_a_communicator(observable, circuit):
    with pytest.raises(TypeError, match="mpi4py communicator"):
        observable.propagate(circuit, POLICY, comm="COMM_WORLD")


@pytest.mark.skipif(SIZE < 3, reason="needs at least three ranks to form a group of three")
def test_a_non_power_of_two_group_is_a_value_error(observable, circuit):
    """A partition is named by ``log2(P)`` GF(2) hash rows, so the group size
    must be a power of two.

    Split off a group of three rather than relaunching. The split is
    collective over `COMM`; after it, each rank talks only to its own
    subgroup, so the two branches below never wait on each other."""
    sub = COMM.Split(color=0 if RANK < 3 else 1, key=RANK)
    try:
        size = sub.Get_size()
        if size & (size - 1):
            with pytest.raises(ValueError, match="power of two"):
                observable.propagate(circuit, POLICY, comm=sub)
        else:
            # The leftover group is a legal size: this rank simply runs.
            observable.propagate(circuit, POLICY, comm=sub)
    finally:
        sub.Free()
