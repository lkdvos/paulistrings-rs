"""``threads=`` on the propagate and readout surface, and the default from ``RAYON_NUM_THREADS``.

The observable is the engine's INFO progress line on the ``paulistrings.propagate`` logger, which ends with ``P partitions [rank:cpusxN, ...]``, or ``1 partition [0:unpinned xN]`` for an unpartitioned call, where ``N`` is the thread count of each partition's pool.
A count above the CPUs this process may run on is a WARNING on ``paulistrings.partitioned``.
"""

import logging
import os
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import paulistrings
from paulistrings import Circuit, PauliSum, truncation

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="thread placement reads the Linux CPU affinity mask"
)

NUM_QUBITS = 8

TOL = 1e-9

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


def _observable(terms=2_000, seed=20261026):
    rng = np.random.default_rng(seed)
    mask = np.uint64((1 << NUM_QUBITS) - 1)
    x = rng.integers(0, 1 << 63, size=(terms, 1), dtype=np.uint64) & mask
    z = rng.integers(0, 1 << 63, size=(terms, 1), dtype=np.uint64) & mask
    coeffs = rng.normal(size=terms) + 1j * rng.normal(size=terms)
    return PauliSum.from_arrays(x, z, coeffs, num_qubits=NUM_QUBITS)


def _circuit():
    circuit = Circuit(NUM_QUBITS)
    for q in range(NUM_QUBITS):
        circuit.rx(0.3 + 0.1 * q, q)
    for q in range(NUM_QUBITS - 1):
        circuit.cnot(q, q + 1)
    circuit.depolarize(0.02, [0])
    return circuit


def _as_dict(sum_):
    xs, zs, cs = sum_.x_array(), sum_.z_array(), sum_.coefficients_array()
    return {
        (tuple(int(v) for v in xs[i]), tuple(int(v) for v in zs[i])): complex(cs[i])
        for i in range(len(sum_))
    }


def _assert_terms_close(got, want, tol=TOL):
    a, b = _as_dict(got), _as_dict(want)
    assert set(a) == set(b)
    worst = max((abs(a[k] - b[k]) for k in a), default=0.0)
    assert worst < tol, f"largest coefficient difference {worst:.3e} exceeds {tol:.1e}"


@pytest.fixture
def info_log(caplog):
    caplog.set_level(logging.INFO, logger="paulistrings")
    paulistrings.reset_log_cache()
    yield caplog
    paulistrings.reset_log_cache()


def _plain_counts(text):
    """The thread count of every one-partition progress line in ``text``."""
    return [int(threads) for m in _PROGRESS.finditer(text) if m.group(1) == "1" for _, threads in _SLOT.findall(m.group(2))]


def _plain_threads(caplog):
    counts = {n for r in caplog.records for n in _plain_counts(r.getMessage())}
    assert len(counts) == 1, f"expected one unpartitioned thread count, got {counts}"
    return counts.pop()


def _partition_threads(caplog):
    for record in caplog.records:
        if m := _PROGRESS.search(record.getMessage()):
            return [int(threads) for _, threads in _SLOT.findall(m.group(2))]
    pytest.fail("no partitioned progress line was logged")


def test_a_plain_call_takes_every_allowed_cpu_by_default(info_log, monkeypatch):
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    _observable().propagate(_circuit())
    assert _plain_threads(info_log) == _allowed()


def test_threads_sets_the_plain_pool(info_log):
    _observable().propagate(_circuit(), threads=4)
    assert _plain_threads(info_log) == 4


def test_threads_all_takes_every_allowed_cpu(info_log):
    _observable().propagate_with_stats(_circuit(), threads="all")
    assert _plain_threads(info_log) == _allowed()


def test_partitioned_default_takes_each_cpu_list(info_log, monkeypatch):
    monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
    _observable().propagate(_circuit(), partitions=_cpu_sets(width=2))
    assert _partition_threads(info_log) == [2, 2]


def test_threads_splits_evenly_across_partitions(info_log):
    _observable().propagate(_circuit(), partitions=_cpu_sets(width=2), threads=4)
    assert _partition_threads(info_log) == [2, 2]


def test_threads_is_capped_at_an_explicit_cpu_list(info_log):
    _observable().propagate(_circuit(), partitions=_cpu_sets(width=1), threads=4)
    assert _partition_threads(info_log) == [1, 1]


def test_threads_all_takes_each_cpu_list(info_log):
    _observable().propagate(_circuit(), partitions=_cpu_sets(width=2), threads="all")
    assert _partition_threads(info_log) == [2, 2]


def test_fewer_threads_than_partitions_is_a_value_error():
    with pytest.raises(ValueError, match="threads=1"):
        _observable().propagate(_circuit(), partitions=_cpu_sets(), threads=1)


@pytest.mark.parametrize("threads, error", [(0, ValueError), (-1, ValueError), ("some", ValueError), (True, TypeError), (1.5, TypeError)])
def test_bad_threads_raise(threads, error):
    s, c = _observable(), _circuit()
    with pytest.raises(error, match="threads"):
        s.propagate(c, threads=threads)
    with pytest.raises(error, match="threads"):
        s.propagate_with_stats(c, threads=threads)


def test_threads_with_device_is_a_value_error():
    with pytest.raises(ValueError, match="device="):
        _observable().propagate(_circuit(), device=0, threads=2)


def test_more_threads_than_cpus_warns_and_runs(info_log):
    over = _allowed() + 1
    s, c = _observable(), _circuit()
    _assert_terms_close(s.propagate(c, threads=over), s.propagate(c))
    warnings = [r.getMessage() for r in info_log.records if r.levelno == logging.WARNING]
    assert any(str(over) in w and str(_allowed()) in w for w in warnings), warnings


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
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
        import numpy as np
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


def test_rayon_num_threads_sets_the_default_and_threads_overrides_it():
    probe = _run_with_rayon_num_threads("3", "s.propagate(c)\nprint('override')\ns.propagate(c, threads=1)\n")
    assert probe.returncode == 0, probe.stderr
    default, override = probe.stdout.split("override")
    assert set(_plain_counts(default)) == {3}, probe.stdout
    assert set(_plain_counts(override)) == {1}, probe.stdout


@pytest.mark.parametrize("value", ["0", "abc"])
def test_rayon_num_threads_zero_or_not_a_number_is_all(value):
    probe = _run_with_rayon_num_threads(value, "s.propagate(c)\n")
    assert probe.returncode == 0, probe.stderr
    assert set(_plain_counts(probe.stdout)) == {_allowed()}, probe.stdout


def test_rayon_num_threads_fewer_than_partitions_is_a_value_error():
    cpus = _cpu_sets()
    probe = _run_with_rayon_num_threads("1", f"s.propagate(c, partitions={cpus!r})\n")
    assert probe.returncode != 0
    assert "ValueError" in probe.stderr and "RAYON_NUM_THREADS=1" in probe.stderr, probe.stderr


_Z_GENERATORS = ["".join("Z" if i == q else "I" for i in range(NUM_QUBITS)) for q in range(NUM_QUBITS)]


def _readouts(s, **kwargs):
    other = _observable(seed=7)
    x, z, coefficients = s.to_arrays(**kwargs)
    return (
        s.expectation(**kwargs),
        s.expectation("0+r1-l01", **kwargs),
        s.expectation_stabilizer(_Z_GENERATORS, **kwargs),
        s.overlap(other, **kwargs),
        x.tolist(),
        z.tolist(),
        coefficients.tolist(),
    )


@pytest.mark.parametrize("threads", [2, "all"])
def test_readouts_agree_across_thread_counts(threads):
    s = _observable()
    got, want = _readouts(s, threads=threads), _readouts(s)
    for value, reference in zip(got[:4], want[:4]):
        assert abs(value - reference) < TOL
    assert got[4:] == want[4:]


@pytest.mark.parametrize("threads, error", [(0, ValueError), ("x", ValueError), (True, TypeError)])
def test_bad_readout_threads_raise(threads, error):
    s = _observable()
    for readout in (lambda: s.expectation(threads=threads), lambda: s.expectation_stabilizer(_Z_GENERATORS, threads=threads), lambda: s.overlap(s, threads=threads), lambda: s.to_arrays(threads=threads)):
        with pytest.raises(error, match="threads"):
            readout()


def test_readout_threads_builds_a_pool_of_that_size():
    body = """
    s.expectation()
    before = threads()
    s.expectation(threads=5)
    print("gained", threads() - before)
    """
    probe = _run_with_rayon_num_threads(None, body)
    assert probe.returncode == 0, probe.stderr
    assert "gained 5" in probe.stdout, probe.stdout


@pytest.mark.parametrize("value, pool", [("3", 3), (None, None)])
def test_readout_default_pool_follows_rayon_num_threads(value, pool):
    probe = _run_with_rayon_num_threads(value, "s.expectation()\nprint('gained', threads() - at_start)\n")
    assert probe.returncode == 0, probe.stderr
    assert f"gained {pool or _allowed()}" in probe.stdout, probe.stdout


def test_results_agree_across_thread_counts():
    s, c = _observable(), _circuit()
    _assert_terms_close(s.propagate(c, threads=4), s.propagate(c, threads=1))
    policy = truncation.coeff(1e-3)
    _assert_terms_close(s.propagate(c, policy, threads=4), s.propagate(c, policy, threads=1))
