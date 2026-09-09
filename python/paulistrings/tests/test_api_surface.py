"""Smoke-test that the public API surface (ARCHITECTURE.md §Python-Bindings) is wired up.

Deliberately shallow: this module only asserts that the names exist and that the
constructors pick a width. Behavioral coverage lives in the sibling
``test_pauli_sum``, ``test_circuit``, ``test_truncation``, and ``test_numpy``
modules.
"""

import paulistrings
from paulistrings import Circuit, PauliSum, gates, noise, truncation


def test_top_level_names():
    assert hasattr(paulistrings, "PauliSum")
    assert hasattr(paulistrings, "Circuit")
    assert hasattr(paulistrings, "gates")
    assert hasattr(paulistrings, "noise")
    assert hasattr(paulistrings, "truncation")
    assert hasattr(paulistrings, "DEFAULT_SMALL_SUM_THRESHOLD")
    assert hasattr(paulistrings, "PropagationStats")
    assert hasattr(paulistrings, "PartitionStats")
    assert hasattr(paulistrings, "numa_nodes")
    assert hasattr(paulistrings, "mpi_available")


def test_mpi_available_answers_without_mpi4py():
    """``mpi_available()`` is a build-time fact, so it must answer in any
    process — and importing ``paulistrings`` must not drag in ``mpi4py``, which
    would call ``MPI_Init`` in every serial script.

    The import check runs in a subprocess: this one may already have imported
    ``mpi4py`` through a sibling test module."""
    import subprocess
    import sys

    assert isinstance(paulistrings.mpi_available(), bool)
    probe = subprocess.run(
        [sys.executable, "-c", "import paulistrings, sys; print('mpi4py' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "False"


def test_numa_nodes_answers_without_a_partitioned_run():
    """The placement `partitions="auto"` reads. One entry per NUMA node in the
    affinity mask, each a non-empty CPU list — and on a host with no NUMA
    information, one entry covering the whole mask."""
    nodes = paulistrings.numa_nodes()
    assert nodes and all(cpus for cpus in nodes)


def test_factory_module_names():
    for name in ("h", "sdg", "cnot", "rz", "pauli_rotation", "unitary_1q", "unitary_2q"):
        assert hasattr(gates, name)
    for name in (
        "depolarize",
        "dephase",
        "amplitude_damping",
        "pauli_channel",
        "depolarize2",
    ):
        assert hasattr(noise, name)
    for name in ("coeff", "weight", "topn", "approx_topn"):
        assert hasattr(truncation, name)


def test_constructors_pick_a_width():
    s = PauliSum(20)
    assert s.num_qubits == 20
    assert len(s) == 0

    c = Circuit(20)
    assert c.num_qubits == 20
    assert len(c) == 0
