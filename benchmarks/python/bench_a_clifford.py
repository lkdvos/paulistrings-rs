"""Benchmark A -- Clifford gate, performance entries.

Correctness lives in `python/paulistrings/tests/test_benchmark_a_clifford.py`
(CI-safe, importorskips stim); this file is the **timed** half, following
`bench_baseline.py`'s idioms (`pytest.importorskip` per test, seeded/fixture
setup outside the timed region, `@pytest.mark.benchmark(group=...)` so
`pytest-benchmark`'s report places matching engines side by side,
assert-on-result rather than a bare timing). Not part of CI (CI runs
`pytest python/paulistrings/tests` only); run with::

    RAYON_NUM_THREADS=1 pytest benchmarks/python/bench_a_clifford.py \\
        --benchmark-only --benchmark-json=benchmarks/results/bench_a.json

Setup: heavy-hex kicked Ising, `n=127`, 5 Trotter steps, `theta_h=pi/2` (the
utility-experiment Clifford point), timing the Heisenberg propagation of the
published weight-10 and weight-17 observables. One group per observable
(plan §6 Part A's own table lists them together, but grouping keeps a
same-shape comparison in the HTML report). Each paulistrings entry
asserts the Clifford invariant on its own result (single term, coefficient
exactly `+-1.0`) so a correctness regression fails the *benchmark* run too,
not only the dedicated test file.

## The PauliPropagation.jl comparative entry

Schema-v1 task JSON built from the *same* gate list the paulistrings side
runs (`examples.common.oracles.record_gates` over the identical
`circuits.heavy_hex_kicked_ising` call), so neither engine gets a
transcription of the other's circuit. Per plan §7 rule 2 ("term-count parity
blocks timing"), `_require_layer_parity` runs **both** engines once, untimed,
and asserts every one of the 1355 per-layer term counts is identical (the
same per-gate-application-order comparison `test_julia_parity.py` uses,
reused here via `run_rust` rather than re-derived) before either engine's
timed entry is allowed to run.

**`min_abs_coeff` is `1e-8`, not a dyadic value.** `benchmarks/julia/README.md`
§P3 records a genuine, measured cross-engine divergence: this repo drops
`|c| <= eps` while PauliPropagation.jl keeps `|c| == eps`, so a coefficient
landing *exactly* on the threshold parity-fails for a reason that has nothing
to do with either engine's correctness. Clifford-point angles produce exact
dyadic coefficients (`sin(pi/2) == 1.0`, `cos(pi/2)` is the tiny residual —
see the correctness test file), so a dyadic cutoff (`2**-14`, ...) is exactly
where that boundary is likely to be hit bit-for-bit; `1e-8` is far from any
dyadic value and, being nine orders of magnitude above the `~6.1e-17`
residual this benchmark is truncating away, changes nothing about which
branch survives.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_BENCH_PY_DIR = Path(__file__).resolve().parent
for _p in (_EXAMPLES_DIR, _BENCH_PY_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

N_QUBITS = 127
TROTTER_STEPS = 5
THETA_H = math.pi / 2

#: Non-dyadic -- see the module docstring's boundary note.
EPS = 1e-8


# --------------------------------------------------------------------------
# Fixtures (built once, outside any timed region).
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kicked_ising_circuit():
    pytest.importorskip("paulistrings")
    from common import circuits

    return circuits.heavy_hex_kicked_ising(N_QUBITS, trotter_steps=TROTTER_STEPS, theta_h=THETA_H)


@pytest.fixture(scope="module")
def kicked_ising_gate_list():
    """The same circuit as a plain schema-v1 gate list (for the jl task JSON)."""
    pytest.importorskip("paulistrings")
    from common import circuits, oracles

    spec = oracles.record_gates(
        circuits.heavy_hex_kicked_ising, N_QUBITS, trotter_steps=TROTTER_STEPS, theta_h=THETA_H
    )
    return spec.gates


@pytest.fixture(scope="module")
def weight_10_observable():
    pytest.importorskip("paulistrings")
    from common import observables

    return observables.weight_10_operator(N_QUBITS)


@pytest.fixture(scope="module")
def weight_17_observable():
    pytest.importorskip("paulistrings")
    from common import observables

    return observables.weight_17_operator(N_QUBITS)


def _make_policy():
    from paulistrings import truncation

    return truncation.coeff(EPS)


def _observable_dict(observable):
    from common import oracles

    return dict(oracles.pauli_terms(observable))


# --------------------------------------------------------------------------
# paulistrings timing entries.
# --------------------------------------------------------------------------


@pytest.mark.benchmark(group="clifford_weight_10")
def test_bench_paulistrings_weight_10(benchmark, kicked_ising_circuit, weight_10_observable):
    pytest.importorskip("paulistrings")
    policy = _make_policy()
    result = benchmark(
        weight_10_observable.propagate,
        circuit=kicked_ising_circuit,
        policy=policy,
        direction="heisenberg",
    )
    assert len(result) == 1
    coeff = complex(result.coefficients_array()[0])
    assert coeff == 1.0 + 0j


@pytest.mark.benchmark(group="clifford_weight_17")
def test_bench_paulistrings_weight_17(benchmark, kicked_ising_circuit, weight_17_observable):
    pytest.importorskip("paulistrings")
    policy = _make_policy()
    result = benchmark(
        weight_17_observable.propagate,
        circuit=kicked_ising_circuit,
        policy=policy,
        direction="heisenberg",
    )
    assert len(result) == 1
    coeff = complex(result.coefficients_array()[0])
    assert coeff == -1.0 + 0j


# --------------------------------------------------------------------------
# PauliPropagation.jl comparative entries.
# --------------------------------------------------------------------------


def _jl_task(gate_list, observable, *, expected_sign: float):
    """Schema-v1 task for one observable: same gates, same truncation, `state="z+"`."""
    import julia_baseline as jb

    return jb.make_task(
        n_qubits=N_QUBITS,
        gates=gate_list,
        observable=_observable_dict(observable),
        direction="heisenberg",
        min_abs_coeff=EPS,
        threads=1,
        state="z+",
    )


def _require_layer_parity(task, *, label: str):
    """Blocking gate (plan §7 rule 2): every per-layer term count must match.

    Runs both engines once, untimed, mirroring `test_julia_parity.py::compare`
    (reusing its `run_rust`, rather than re-deriving the log-collector, since
    that file is the maintained reference implementation of this exact check).
    Raises `AssertionError` naming the first divergent layer -- a benchmark
    entry gated behind this must never report a timing for mismatched runs.
    """
    import julia_baseline as jb
    from test_julia_parity import run_rust

    jb.importorskip_julia()
    rust = run_rust(task)
    jl = jb.run_task(task, warm_repeats=0, layer_counts=True)

    rust_layers = rust["per_layer_terms"]
    jl_layers = jl.per_layer_terms
    assert jl_layers is not None, f"{label}: julia reported no per-layer term counts"
    assert len(rust_layers) == len(jl_layers), (
        f"{label}: layer count mismatch, rust={len(rust_layers)} jl={len(jl_layers)} "
        "(one gate object must be one channel on both sides)"
    )
    mismatches = [
        (i, a, b) for i, (a, b) in enumerate(zip(rust_layers, jl_layers)) if a != b
    ]
    assert not mismatches, (
        f"{label}: {len(mismatches)}/{len(rust_layers)} per-layer term counts differ "
        f"(first: layer {mismatches[0][0]}, rust={mismatches[0][1]} jl={mismatches[0][2]}); "
        "cross-engine timing must not be reported until this is resolved (plan §7 rule 2)"
    )
    assert rust["final_terms"] == jl.final_terms == 1, (
        f"{label}: expected both engines to collapse to the single Clifford stabilizer "
        f"term, got rust={rust['final_terms']} jl={jl.final_terms}"
    )
    return rust, jl


@pytest.mark.benchmark(group="clifford_weight_10")
def test_bench_julia_weight_10(benchmark, kicked_ising_gate_list, weight_10_observable):
    import julia_baseline as jb

    jb.importorskip_julia()
    task = _jl_task(kicked_ising_gate_list, weight_10_observable, expected_sign=1.0)
    _require_layer_parity(task, label="weight_10")

    holder: dict = {}

    def run_once():
        holder["result"] = jb.run_task(task, warm_repeats=3, layer_counts=False)

    # `benchmark.pedantic(..., rounds=1)`: a normal `benchmark(fn)` calibration
    # would re-launch the julia subprocess (and re-pay its JIT warmup) several
    # times over; the runner already does its own internal warm repeats, so
    # exactly one outer round is what "warm timing" means here.
    benchmark.pedantic(run_once, rounds=1, iterations=1, warmup_rounds=0)
    jl = holder["result"]
    assert jl.final_terms == 1
    assert jl.expectation is not None
    assert abs(jl.expectation.real - 1.0) < 1e-9
    assert abs(jl.expectation.imag) < 1e-9
    benchmark.extra_info["julia_wall_warm_s"] = jl.wall_warm_s
    benchmark.extra_info["julia_versions"] = jl.versions


@pytest.mark.benchmark(group="clifford_weight_17")
def test_bench_julia_weight_17(benchmark, kicked_ising_gate_list, weight_17_observable):
    import julia_baseline as jb

    jb.importorskip_julia()
    task = _jl_task(kicked_ising_gate_list, weight_17_observable, expected_sign=-1.0)
    _require_layer_parity(task, label="weight_17")

    holder: dict = {}

    def run_once():
        holder["result"] = jb.run_task(task, warm_repeats=3, layer_counts=False)

    benchmark.pedantic(run_once, rounds=1, iterations=1, warmup_rounds=0)
    jl = holder["result"]
    assert jl.final_terms == 1
    assert jl.expectation is not None
    assert abs(jl.expectation.real - (-1.0)) < 1e-9
    assert abs(jl.expectation.imag) < 1e-9
    benchmark.extra_info["julia_wall_warm_s"] = jl.wall_warm_s
    benchmark.extra_info["julia_versions"] = jl.versions
