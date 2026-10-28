"""The thread count from ``RAYON_NUM_THREADS``, read once per process, so every test sets it in a fresh interpreter.

The observable is the engine's INFO progress line on the ``paulistrings.propagate`` logger, which ends with ``P partitions [rank:cpusxN, ...]``, or ``1 partition [0:unpinned xN]`` for an unpartitioned call, where ``N`` is the thread count of each partition's pool.
A count above the CPUs this process may run on is a WARNING on ``paulistrings.partitioned``.
"""

import os
import re
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="thread placement reads the Linux CPU affinity mask"
)

NUM_QUBITS = 8

_PROGRESS = re.compile(r"\b(\d+) partitions? \[([^\]]*)\]")
_SLOT = re.compile(r"(\d+):(?:[\d,\-]+|unpinned )x(\d+)")


def _allowed():
    return len(os.sched_getaffinity(0))


def _cpu_sets(count=2, width=1):
    """The first ``count * width`` CPUs of the affinity mask, ``width`` per partition."""
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < count * width:
        pytest.skip(f"needs {count * width} CPUs in the affinity mask")
    return [cpus[i * width : (i + 1) * width] for i in range(count)]


def _run_with_rayon_num_threads(value, body):
    """Run ``body`` after a one-term setup in a fresh interpreter with ``RAYON_NUM_THREADS=value``, or with it removed for ``None``; returns the finished process."""
    script = textwrap.dedent(
        f"""
        import logging, sys
        def threads():
            with open("/proc/self/status") as status:
                return next(int(line.split()[1]) for line in status if line.startswith("Threads:"))
        import numpy as np
        at_start = threads()
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")
        from paulistrings import Circuit, PauliSum
        c = Circuit({NUM_QUBITS})
        c.rx(0.3, 0)
        s = PauliSum.from_arrays(np.array([[1]], dtype=np.uint64), np.array([[0]], dtype=np.uint64), np.array([1.0 + 0j]), num_qubits={NUM_QUBITS})
        """
    ) + textwrap.dedent(body)
    env = {key: item for key, item in os.environ.items() if key != "RAYON_NUM_THREADS"}
    if value is not None:
        env["RAYON_NUM_THREADS"] = value
    return subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)


def _plain_threads(probe):
    """The thread count of every one-partition progress line ``probe`` printed."""
    assert probe.returncode == 0, probe.stderr
    return {int(threads) for m in _PROGRESS.finditer(probe.stdout) if m.group(1) == "1" for _, threads in _SLOT.findall(m.group(2))}


def _partition_threads(value, partitions):
    probe = _run_with_rayon_num_threads(value, f"s.propagate(c, partitions={partitions!r})\n")
    assert probe.returncode == 0, probe.stderr
    m = _PROGRESS.search(probe.stdout)
    assert m, probe.stdout
    return [int(threads) for _, threads in _SLOT.findall(m.group(2))]


@pytest.mark.parametrize("value", [None, "0", "abc"])
def test_unset_zero_or_not_a_number_takes_every_allowed_cpu(value):
    assert _plain_threads(_run_with_rayon_num_threads(value, "s.propagate(c)\n")) == {_allowed()}


def test_unset_takes_each_cpu_list():
    assert _partition_threads(None, _cpu_sets(width=2)) == [2, 2]


def test_rayon_num_threads_sets_a_plain_call():
    assert _plain_threads(_run_with_rayon_num_threads("3", "s.propagate(c)\n")) == {3}


def test_rayon_num_threads_splits_evenly_across_partitions():
    assert _partition_threads("4", _cpu_sets(width=3)) == [2, 2]


def test_rayon_num_threads_is_capped_at_an_explicit_cpu_list():
    assert _partition_threads("4", _cpu_sets(width=1)) == [1, 1]


def test_rayon_num_threads_fewer_than_partitions_is_a_value_error():
    probe = _run_with_rayon_num_threads("1", f"s.propagate(c, partitions={_cpu_sets()!r})\n")
    assert probe.returncode != 0
    assert "ValueError" in probe.stderr and "RAYON_NUM_THREADS=1" in probe.stderr, probe.stderr


def test_more_threads_than_cpus_warns_and_runs():
    over = _allowed() + 1
    probe = _run_with_rayon_num_threads(str(over), "s.propagate(c)\n")
    assert _plain_threads(probe) == {over}
    warnings = [line for line in probe.stdout.splitlines() if line.startswith("WARNING")]
    assert any(str(over) in w and str(_allowed()) in w for w in warnings), probe.stdout


@pytest.mark.parametrize("value, pool", [("3", 3), (None, None)])
def test_readout_default_pool_follows_rayon_num_threads(value, pool):
    probe = _run_with_rayon_num_threads(value, "s.expectation()\nprint('gained', threads() - at_start)\n")
    assert probe.returncode == 0, probe.stderr
    assert f"gained {pool or _allowed()}" in probe.stdout, probe.stdout
