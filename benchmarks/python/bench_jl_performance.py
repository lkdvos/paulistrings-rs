#!/usr/bin/env python3
"""Rigorous single-threaded head-to-head: this engine vs PauliPropagation.jl.

Results, protocol and caveats: ``benchmarks/python/jl_performance/README.md``.
That README is the document; this file is the instrument.

Why this driver exists when three suites already time both engines
------------------------------------------------------------------

Benchmarks C, D and E each produced *indicative* cross-engine timings as a
side effect of asking a different question, and they disagreed about who is
faster because they measured at different term counts. This driver asks only
the head-to-head question, and answers it under the one protocol
``benchmarks/PROFILING.md`` says can resolve differences near this host's noise
floor: **interleaved pairs, accepted on direction consistency, never on a
difference of two independently-noisy means.**

The protocol, in one place
--------------------------

1. **One task JSON is the single source of truth per configuration.** Both
   engines read the same schema-v1 file: Julia through
   ``benchmarks/julia/runner.jl``, this engine through
   ``paulistrings.interop.load_task``. Neither side rebuilds the circuit from
   a private description, so "the two engines ran the same circuit" is a
   property of the file, not of two code paths agreeing.

2. **Term-count parity gates every timed configuration.** Before any pair is
   run, both engines propagate once, untimed, with per-gate term counts
   collected, and *every* per-layer count must match. A parity failure
   disqualifies the configuration loudly — it is recorded with its mismatch
   and no timing for it is reported. (``benchmarks/julia/README.md`` §P3/§P9
   document the two known divergences this catches: the ``min_abs_coeff``
   boundary and exact-zero coefficients.)

3. **Warm in-process timing on both sides, construction excluded.** The Julia
   runner propagates once cold (paying JIT) and then times a warm
   propagation in the same process; this engine's leg does the same through
   ``harness.run_propagation``, which warms and then times. Input construction,
   contraction, oracles and logging are all outside the timed region on both
   sides. Only propagation is compared.

4. **Interleaved pairs, abba across pairs.** Each configuration runs
   ``--pairs`` pairs; each leg is its own process, and the within-pair order
   alternates (rust-first on even pairs, julia-first on odd) so a monotone
   drift in machine state cannot masquerade as a consistent win.

5. **Acceptance: direction consistency, not a p-value.** With a handful of
   pairs there is nothing statistically meaningful to compute and none is.
   Every pair must agree on *which engine was faster*; the median ratio is
   then the effect size. Pairs disagreeing in sign are reported as
   "indistinguishable" — not a small win, not a trend, not something to
   average over. This is ``PROFILING.md``'s A/B rule applied across engines
   instead of across two builds.

6. **Memory: each engine samples its own process.** Both read their own
   ``/proc/self/status``. A driver-side ``getrusage(RUSAGE_CHILDREN)`` conflates
   every child it has reaped, so a sibling engine's peak leaks into the other's
   number; it is never used here.

The ratio convention, fixed once
--------------------------------

``ratio = t_julia / t_paulistrings`` everywhere, in results.json
(``ratio_jl_over_rust``), in the figures, and in the README:

* ``ratio > 1`` — **this engine is faster** (Julia spends more time)
* ``ratio < 1`` — **Julia is faster**
* ``ratio = 1`` — the crossover

One number line, one direction, no per-sentence convention flips.

Usage
-----

::

    # the full study (hours; see --help for the leg selectors)
    RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py --all

    # one workload, a pilot at the loosest cutoff only
    RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py \
        --curves --workload kicked_ising --pilot

    # subprocess entry points (the driver invokes these; not for hand use)
    python benchmarks/python/bench_jl_performance.py --rust-timed-leg task.json
    python benchmarks/python/bench_jl_performance.py --rust-parity-leg task.json

``RAYON_NUM_THREADS=1`` must be in the environment **before the interpreter
starts**: Rayon builds its global pool once, at the first propagate, and never
resizes it (``examples/common/harness.py`` documents the measured behaviour).
The driver re-exports it to every leg it spawns and refuses to start a
core-comparison leg without it.

Module-level imports are stdlib-only on purpose: the protocol math below is
imported by ``python/paulistrings/tests/test_jl_performance_protocol.py``,
which runs in CI where neither Julia nor matplotlib is available. Engine,
plotting and harness imports are all inside functions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "jl_performance"

#: Julia 0.8.2 is single-threaded for propagation; the thread-scaling section
#: is this engine only, and says so. See the README's thread-scaling section.
JULIA_IS_SINGLE_THREADED = True

#: Default pairs per configuration. PROFILING.md's A/B harness defaults to 3;
#: the task-level bar here is >= 5, since a cross-engine ratio has two
#: independent runtimes in it rather than one.
DEFAULT_PAIRS = 5


# ==========================================================================
# Protocol math
#
# Pure functions over plain numbers: no engine, no Julia, no matplotlib, no
# filesystem. This is the part the CI smoke test exercises on synthetic data,
# so it must stay import-clean and side-effect-free.
# ==========================================================================


class ParityFailure(RuntimeError):
    """A configuration's two engines disagreed on per-layer term counts.

    Blocking by construction: the driver records the mismatch and reports no
    timing for that configuration. Never downgraded to a warning — a term-count
    divergence means the two engines did different amounts of work, so their
    runtimes are not comparable at all.
    """


def median(values: Sequence[float]) -> float:
    """Median of a non-empty sequence.

    Thin wrapper over ``statistics.median`` so the error on an empty input
    names the protocol reason rather than raising ``StatisticsError``.
    """
    if not values:
        raise ValueError("median of an empty sequence: a configuration with no pairs")
    return float(statistics.median(values))


def pair_ratios(pairs: Sequence[Mapping[str, float]]) -> list[float]:
    """``t_julia / t_paulistrings`` for each pair, in run order.

    Each pair is a mapping with ``rust_s`` and ``jl_s``. Ratios above 1 mean
    this engine was faster in that pair.
    """
    ratios = []
    for i, p in enumerate(pairs):
        rust_s = float(p["rust_s"])
        jl_s = float(p["jl_s"])
        if rust_s <= 0.0 or jl_s <= 0.0:
            raise ValueError(
                f"pair {i} has a non-positive runtime (rust_s={rust_s}, jl_s={jl_s}); "
                "a timed leg that measured zero did not measure anything"
            )
        ratios.append(jl_s / rust_s)
    return ratios


def rust_runs_first(pair_index: int) -> bool:
    """Within-pair leg order for pair ``pair_index`` — the abba rule.

    Alternating which engine runs first, instead of always running one of them
    first, is what stops a monotone drift in machine state (a warming cache, a
    frequency ramp, another tenant's load) from looking like a consistent win
    for whichever engine always ran second. Across pairs the sequence is
    ``ab | ba | ab | ba | ...``, which is ``PROFILING.md``'s ``--order abba``
    extended past two pairs.
    """
    return pair_index % 2 == 0


def analyze_pairs(pairs: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    """Apply PROFILING.md's acceptance rule to one configuration's pairs.

    Returns the per-pair ratios, the median as the effect size, and a
    ``verdict`` of ``"paulistrings"``, ``"julia"`` or ``"indistinguishable"``.

    **The rule.** Every pair must agree on which engine was faster. If they do,
    the difference is real and the median ratio is its size. If any pair
    disagrees in sign, the verdict is ``"indistinguishable"`` and the median is
    reported only as context — it is explicitly *not* a small win, and the
    README shades that term-count range rather than drawing a curve through it.

    A pair with exactly equal runtimes counts as a disagreement: it cannot
    support either direction.
    """
    ratios = pair_ratios(pairs)
    faster = ["paulistrings" if r > 1.0 else "julia" if r < 1.0 else "tie" for r in ratios]
    unique = set(faster)
    consistent = len(unique) == 1 and "tie" not in unique
    med = median(ratios)
    if consistent:
        verdict = unique.pop()
    else:
        verdict = "indistinguishable"
    return {
        "n_pairs": len(ratios),
        "ratio_jl_over_rust_per_pair": ratios,
        "median_ratio_jl_over_rust": med,
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "faster_per_pair": faster,
        "sign_consistent": consistent,
        "verdict": verdict,
        "rust_s_per_pair": [float(p["rust_s"]) for p in pairs],
        "jl_s_per_pair": [float(p["jl_s"]) for p in pairs],
        "rust_s_median": median([float(p["rust_s"]) for p in pairs]),
        "jl_s_median": median([float(p["jl_s"]) for p in pairs]),
    }


def indistinguishable_zone(
    points: Sequence[Mapping[str, Any]], terms_key: str = "final_terms"
) -> dict[str, Any] | None:
    """Term-count span over which the pairs never agreed on a direction.

    ``points`` are per-configuration dicts carrying ``terms_key`` and
    ``sign_consistent``. Returns ``{"lo": n, "hi": n, "n_configs": k}`` over the
    mixed-sign configurations, or ``None`` when every configuration was
    consistent. The README shades this span instead of drawing a crossover
    through it.

    ``terms_key`` selects the size axis. The study passes ``"peak_terms"``: the
    peak is what the engine actually had to hold and sort, so it is the size a
    runtime should be read against, and several configurations here truncate
    down to a handful of final terms after passing through millions.
    """
    mixed = [p for p in points if not p.get("sign_consistent", True)]
    if not mixed:
        return None
    terms = [int(p[terms_key]) for p in mixed]
    return {"lo": min(terms), "hi": max(terms), "n_configs": len(mixed)}


def interpolate_crossover(
    points: Sequence[Mapping[str, Any]], terms_key: str = "final_terms"
) -> dict[str, Any]:
    """Localize the term count where the median ratio crosses 1.

    ``points`` are per-configuration dicts with ``terms_key``,
    ``median_ratio_jl_over_rust`` and ``sign_consistent``. ``terms_key`` selects
    the size axis and must match the one the figures use (the study passes
    ``"peak_terms"`` — see :func:`indistinguishable_zone`). Only
    **sign-consistent** configurations can bracket a crossover: a mixed-sign
    point has no defined direction, so it cannot be one end of an interval that
    claims the direction changed across it.

    Interpolation is linear in ``log10(ratio)`` against ``log10(terms)``, solved
    for ``log10(ratio) = 0`` — the natural geometry for a log-log time-vs-terms
    curve, where both engines' costs are near power laws in the term count.

    Returns a dict with ``crossover_terms`` (``None`` when the sweep never
    changes direction), the bracketing configurations, and
    ``inside_indistinguishable_zone``: ``True`` when the interpolated point
    falls within a span where the pairs could not agree, which means the
    crossover is located but *not resolved* — the honest reading is "somewhere
    in this band", and the README says so.
    """
    usable = sorted(
        (p for p in points if p.get("sign_consistent", False)),
        key=lambda p: int(p[terms_key]),
    )
    zone = indistinguishable_zone(points, terms_key)
    out: dict[str, Any] = {
        "crossover_terms": None,
        "lower": None,
        "upper": None,
        "indistinguishable_zone": zone,
        "inside_indistinguishable_zone": False,
        "method": "linear in log10(ratio) vs log10(terms), solved for ratio = 1",
        "terms_key": terms_key,
    }
    if len(usable) < 2:
        out["note"] = (
            f"{len(usable)} sign-consistent configuration(s): a crossover needs two "
            "that bracket ratio = 1"
        )
        return out

    bracket = None
    for lo, hi in zip(usable, usable[1:]):
        r_lo = float(lo["median_ratio_jl_over_rust"])
        r_hi = float(hi["median_ratio_jl_over_rust"])
        if (r_lo - 1.0) * (r_hi - 1.0) < 0.0:
            bracket = (lo, hi)
            break
    if bracket is None:
        direction = (
            "paulistrings faster at every sign-consistent point"
            if float(usable[0]["median_ratio_jl_over_rust"]) > 1.0
            else "julia faster at every sign-consistent point"
        )
        out["note"] = f"no direction change across the swept range ({direction})"
        return out

    lo, hi = bracket
    x_lo, x_hi = math.log10(int(lo[terms_key])), math.log10(int(hi[terms_key]))
    y_lo = math.log10(float(lo["median_ratio_jl_over_rust"]))
    y_hi = math.log10(float(hi["median_ratio_jl_over_rust"]))
    x_cross = x_lo + (0.0 - y_lo) * (x_hi - x_lo) / (y_hi - y_lo)
    crossover = 10.0**x_cross
    out["crossover_terms"] = crossover
    out["lower"] = {
        terms_key: int(lo[terms_key]),
        "median_ratio": float(lo["median_ratio_jl_over_rust"]),
    }
    out["upper"] = {
        terms_key: int(hi[terms_key]),
        "median_ratio": float(hi["median_ratio_jl_over_rust"]),
    }
    if zone is not None and zone["lo"] <= crossover <= zone["hi"]:
        out["inside_indistinguishable_zone"] = True
    return out


def bytes_per_term(peak_kb: float | None, floor_kb: float | None, terms: int) -> float | None:
    """Floor-subtracted peak bytes per term, or ``None`` if unmeasurable.

    ``peak_kb`` is ``VmHWM``, ``floor_kb`` the engine's fixed per-process floor
    (the Julia runtime and loaded packages, or the Python interpreter plus
    numpy and the extension). Both engines carry a floor far larger than a
    small run's payload, so the raw peak divided by the term count is
    meaningless below ~10^5 terms; the floor-subtracted figure is the one that
    can be compared, and the README reports both.

    Returns ``None`` — never a negative or a zero — when the subtraction leaves
    nothing positive, which is what a run smaller than the allocator's slack
    looks like.
    """
    if peak_kb is None or floor_kb is None or terms <= 0:
        return None
    delta_kb = float(peak_kb) - float(floor_kb)
    if delta_kb <= 0.0:
        return None
    return delta_kb * 1024.0 / float(terms)


def check_parity(rust_layers: Sequence[int], jl_layers: Sequence[int] | None) -> list[str]:
    """Compare per-layer term counts; return a list of problems (empty = parity).

    Counts are compared in **application** order on both sides: Julia's
    ``@countpaulis`` records after every gate, and this engine's DEBUG line
    numbers application steps the same way, so for ``direction="heisenberg"``
    both lists run backwards through the task file and line up index by index
    with no reversal (``benchmarks/julia/README.md`` §P5).
    """
    problems: list[str] = []
    if jl_layers is None:
        return ["julia reported no per-layer term counts (PP_LAYER_COUNTS was off)"]
    if len(rust_layers) != len(jl_layers):
        problems.append(
            f"layer count: rust={len(rust_layers)} julia={len(jl_layers)} "
            "(one gate object must be one channel)"
        )
        return problems
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(rust_layers, jl_layers)) if a != b]
    if bad:
        head = ", ".join(f"layer {i + 1}: {a} vs {b}" for i, a, b in bad[:8])
        problems.append(
            f"{len(bad)}/{len(rust_layers)} per-layer term counts differ ({head})"
        )
    return problems


def is_dyadic(eps: float) -> bool:
    """Is ``eps`` an exact power of two?

    Reported per configuration because it says whether the one-ulp threshold
    mitigation below was *load-bearing* or merely harmless: dyadic cutoffs
    against Clifford-point angles (this suite's kicked-Ising workload has both)
    produce exactly dyadic coefficients that can land on the threshold
    bit-exactly, which is the one case where the engines' boundary rules
    actually diverge.
    """
    if eps <= 0.0 or not math.isfinite(eps):
        return False
    mantissa, _exponent = math.frexp(eps)
    return abs(mantissa) == 0.5


def julia_min_abs_coeff(eps: float) -> float:
    """The Julia-side cutoff that makes the two engines' boundary rules identical.

    ``benchmarks/julia/README.md`` §P3: this engine drops ``|c| <= eps``
    (inclusive), PauliPropagation.jl drops ``|c| < eps`` (strict), so Julia
    *keeps* a coefficient exactly equal to the threshold and this engine drops
    it. Truncation runs after every gate, so a single boundary hit changes term
    counts for the whole remaining circuit — it is not a rounding detail.

    Passing Julia ``nextafter(eps, +inf)`` closes the gap exactly: no float lies
    strictly between ``eps`` and its successor, so Julia's ``|c| < eps'`` **is**
    this engine's ``|c| <= eps``, bit for bit. The *threshold* moves by one ulp
    and no coefficient is touched anywhere.

    Applied unconditionally, dyadic cutoff or not — it is the exactly-right
    transformation in both cases, and making it unconditional removes a branch
    that could silently stop being taken. This is benchmark C's method
    (``bench_c_deep_trotter.py``), reused rather than reinvented.
    """
    if eps <= 0.0:
        raise ValueError(f"min_abs_coeff must be positive, got {eps}")
    return math.nextafter(eps, math.inf)


# ==========================================================================
# Task construction: one schema-v1 gate list per workload
#
# These mirror examples/common/circuits.py gate for gate, in the same emission
# order, but produce schema-v1 gate *dicts* instead of pushing onto a Circuit.
# That is what lets one description drive both engines (see the module
# docstring, protocol point 1). The mirroring is pinned by
# test_jl_performance_protocol.py, which rebuilds each workload's Circuit from
# the gate list via interop.circuit_from_json and checks the channel count
# against the corresponding circuits.py builder.
# ==========================================================================


def kicked_ising_gates(
    n: int = 127,
    trotter_steps: int = 1,
    theta_h: float = 0.0,
    theta_zz: float = -math.pi / 2,
    *,
    order: str = "x-then-zz",
) -> list[dict[str, Any]]:
    """Schema-v1 mirror of ``circuits.heavy_hex_kicked_ising``.

    One Trotter step is an X-rotation layer (``rx(theta_h)`` on every qubit)
    followed by a ZZ-rotation layer (``pauli_rotation("ZZ", edge, theta_zz)``
    over the heavy-hex edges, grouped by the same greedy edge coloring the
    circuits.py builder uses, so the truncation schedule is identical).

    Native ``pauli_rotation`` on both engines: this maps to jl's
    ``PauliRotation([:Z,:Z], qs, theta)``, its fast rotation path, not a
    transfer map.
    """
    sys.path.insert(0, str(REPO_ROOT / "examples"))
    from common.circuits import heavy_hex_edge_coloring, heavy_hex_sublattice

    if order not in ("x-then-zz", "zz-then-x"):
        raise ValueError(f"order must be 'x-then-zz' or 'zz-then-x', got {order!r}")
    lattice = heavy_hex_sublattice(n)
    zz_order = [e for group in heavy_hex_edge_coloring(lattice) for e in group]

    def x_layer() -> list[dict[str, Any]]:
        return [{"name": "rx", "qubits": [q], "theta": theta_h} for q in range(n)]

    def zz_layer() -> list[dict[str, Any]]:
        return [
            {"name": "pauli_rotation", "pauli": "ZZ", "qubits": [a, b], "theta": theta_zz}
            for a, b in zz_order
        ]

    gates: list[dict[str, Any]] = []
    for _ in range(trotter_steps):
        if order == "x-then-zz":
            gates.extend(x_layer())
            gates.extend(zz_layer())
        else:
            gates.extend(zz_layer())
            gates.extend(x_layer())
    return gates


def xxz_gates(
    n: int,
    trotter_steps: int,
    Jz: float = 0.5,
    dt: float = 0.1,
    *,
    bond_order: str = "even-odd",
) -> list[dict[str, Any]]:
    """Schema-v1 mirror of ``circuits.xxz_chain_trotter``.

    Three ``pauli_rotation`` channels per bond per step — ``XX``, ``YY``, ``ZZ``
    with ``theta = 2·dt``, ``2·dt`` and ``2·dt·Jz`` — over the even bonds then
    the odd bonds. Rotations only, so this exercises the same native rotation
    path as the kicked-Ising workload but with a non-Clifford angle and three
    generator types.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    bonds = list(range(n - 1))
    if bond_order == "even-odd":
        bonds = [i for i in bonds if i % 2 == 0] + [i for i in bonds if i % 2 == 1]
    elif bond_order != "sequential":
        raise ValueError(f"bond_order must be 'even-odd' or 'sequential', got {bond_order!r}")

    gates: list[dict[str, Any]] = []
    for _ in range(trotter_steps):
        for i in bonds:
            pair = [i, i + 1]
            gates.append({"name": "pauli_rotation", "pauli": "XX", "qubits": pair, "theta": 2.0 * dt})
            gates.append({"name": "pauli_rotation", "pauli": "YY", "qubits": pair, "theta": 2.0 * dt})
            gates.append(
                {"name": "pauli_rotation", "pauli": "ZZ", "qubits": pair, "theta": 2.0 * dt * Jz}
            )
    return gates


def su4_gates(n: int, depth: int, seed: int) -> list[dict[str, Any]]:
    """Schema-v1 mirror of ``circuits.random_su4_staircase``.

    Brickwork of independent Haar-random SU(4) blocks: layer ``d`` acts on the
    pairs ``(i, i+1)`` with ``i ≡ d (mod 2)``, blocks drawn from
    ``numpy.random.default_rng(seed)`` in emission order (layer-major, then
    ascending ``i``), so the circuit is a deterministic function of
    ``(n, depth, seed)`` — the same draw order ``circuits.random_su4_staircase``
    uses, hence the same matrices.

    ``unitary_2q`` is the interesting path here: it is this engine's general
    two-qubit channel and jl's ``TransferMapGate`` (its dense matrix-gate path,
    a 16x16 PTM per block), so this workload compares the two engines'
    *matrix* machinery rather than their rotation kernels.
    ``qubits = [q0, q1]`` with ``q0`` the more significant tensor factor on both
    sides (``benchmarks/julia/README.md`` §P7).
    """
    sys.path.insert(0, str(REPO_ROOT / "examples"))
    import numpy as np

    from common.circuits import haar_su4

    rng = np.random.default_rng(seed)
    gates: list[dict[str, Any]] = []
    for d in range(depth):
        for i in range(d % 2, n - 1, 2):
            m = haar_su4(rng)
            gates.append(
                {
                    "name": "unitary_2q",
                    "qubits": [i, i + 1],
                    "matrix": [[[c.real, c.imag] for c in row] for row in m],
                }
            )
    return gates


def make_task_payload(
    *,
    n_qubits: int,
    gates: Sequence[Mapping[str, Any]],
    observable: Mapping[str, complex | float],
    direction: str = "heisenberg",
    min_abs_coeff: float | None = None,
    max_weight: int | None = None,
    threads: int = 1,
    state: str | None = None,
) -> dict[str, Any]:
    """A schema-v1 task payload, validated by the shared wrapper.

    Delegates to ``julia_baseline.make_task`` so the schema validation is the
    same code the parity gate uses, then returns the plain payload dict.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from julia_baseline import make_task

    task = make_task(
        n_qubits=n_qubits,
        gates=gates,
        observable=observable,
        direction=direction,
        min_abs_coeff=min_abs_coeff,
        max_weight=max_weight,
        threads=threads,
        state=state,
    )
    return task.payload


# ==========================================================================
# Leg workers (subprocess entry points)
#
# One process per timed leg, on both engines. That is not incidental: it gives
# each leg a clean per-process VmHWM (the harness notes VmHWM is a
# process-lifetime high-water mark, so a second run in the same process
# inherits the first one's peak), and it makes the two engines symmetric --
# Julia has no choice but one process per invocation.
# ==========================================================================


def _status_kb(field: str) -> float | None:
    """One ``/proc/self/status`` field in KiB, or ``None`` if unreadable."""
    try:
        with open("/proc/self/status") as f:
            prefix = f"{field}:"
            for line in f:
                if line.startswith(prefix):
                    return float(line.split(":", 1)[1].split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return None


def rust_timed_leg(task_path: str | Path) -> dict[str, Any]:
    """One warm-then-timed propagation of ``task_path`` on this engine.

    Runs in its own process. Mirrors what ``runner.jl`` does on the Julia side:
    propagate once untimed, then time one propagation, in the same process,
    with construction and contraction outside the timed region.

    Timing comes from ``harness.run_propagation`` — the suite's canonical
    runner, which also enforces the thread pin and refuses to time anything
    while the engine's DEBUG logging is enabled (a clock read per layer).
    """
    sys.path.insert(0, str(REPO_ROOT / "examples"))
    from common import harness

    import paulistrings as ps
    from paulistrings.interop import load_task

    floor_kb = _status_kb("VmRSS")
    task = load_task(Path(task_path))
    if task.observable is None:
        raise ValueError(f"{task_path}: task has no observable")

    threads_env = os.environ.get("RAYON_NUM_THREADS")
    threads = int(threads_env) if threads_env and threads_env.isdigit() else None

    record = harness.run_propagation(
        task.circuit,
        task.observable,
        # The policy is taken from the task file, already built by load_task,
        # so the two engines cannot drift on truncation knobs.
        task.truncation if task.truncation is not None else None,
        task.direction,
        state=task.state,
        warmup=True,
        threads=threads,
        engine="paulistrings",
    )

    return {
        "engine": "paulistrings",
        "propagation_s": record.propagation_time_s,
        "contraction_s": record.contraction_time_s,
        "final_terms": record.final_terms,
        "peak_terms": record.peak_terms,
        "expectation": record.expectation_value,
        "memory": {
            "vmrss_start_kb": floor_kb,
            "vmrss_post_propagate_kb": record.extra.get("rss_kb_after"),
            "vmhwm_kb": record.peak_memory_kb,
            "source": "/proc/self/status",
        },
        "threads": threads,
        "observed_threads": record.extra.get("observed_threads"),
        "commit": record.provenance.commit,
        "engine_version": record.engine_version,
    }


def rust_parity_leg(task_path: str | Path) -> dict[str, Any]:
    """Per-layer term counts for ``task_path`` on this engine. Untimed.

    Per-layer counts come from the engine's DEBUG records on logger
    ``paulistrings.propagate`` (``layer {k}/{n} [name]: {before} -> {after}
    terms``), which is the only place they are exposed —
    ``PropagationStats`` carries only the final and peak counts. The DEBUG
    filter costs a clock read per layer, which is exactly why this is a
    separate, untimed process and never folded into a timed leg.
    """
    import logging
    import re

    sys.path.insert(0, str(REPO_ROOT / "examples"))
    import paulistrings as ps
    from paulistrings.interop import load_task

    layer_re = re.compile(
        r"^layer (?P<k>\d+)/(?P<n>\d+) \[(?P<name>[^\]]*)\]: "
        r"(?P<before>\d+) -> (?P<after>\d+) terms"
    )

    class Collector(logging.Handler):
        def __init__(self) -> None:
            super().__init__(level=logging.DEBUG)
            self.layers: list[tuple[int, int]] = []

        def emit(self, record: logging.LogRecord) -> None:
            m = layer_re.match(record.getMessage())
            if m is not None:
                self.layers.append((int(m["before"]), int(m["after"])))

    task = load_task(Path(task_path))
    if task.observable is None:
        raise ValueError(f"{task_path}: task has no observable")

    collector = Collector()
    logger = logging.getLogger("paulistrings.propagate")
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(collector)
    ps.reset_log_cache()
    try:
        evolved = task.observable.propagate(
            circuit=task.circuit, policy=task.truncation, direction=task.direction
        )
    finally:
        logger.removeHandler(collector)
        logger.setLevel(old_level)
        ps.reset_log_cache()

    expectation = None
    if task.state is not None:
        expectation = complex(evolved.expectation(state=task.state)).real

    return {
        "engine": "paulistrings",
        "input_terms": len(task.observable),
        "final_terms": len(evolved),
        "per_layer_terms": [after for _, after in collector.layers],
        "expectation": expectation,
    }


# ==========================================================================
# Leg drivers (parent side): spawn one process, parse its JSON
# ==========================================================================


def _spawn_rust_leg(
    task_path: Path, mode: str, *, threads: int = 1, timeout: float = 7200.0
) -> dict[str, Any]:
    """Run ``--rust-timed-leg`` / ``--rust-parity-leg`` in a fresh process."""
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = str(threads)
    env.pop("RUST_LOG", None)  # a debug filter costs a clock read per layer
    cmd = [sys.executable, str(Path(__file__).resolve()), f"--rust-{mode}-leg", str(task_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"rust {mode} leg exited {proc.returncode}\ncmd: {' '.join(cmd)}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"rust {mode} leg produced no stdout\nstderr:\n{proc.stderr[-2000:]}")
    return json.loads(lines[-1])


def _spawn_jl_leg(
    task_path: Path, *, timed: bool, threads: int = 1, timeout: float = 7200.0
) -> dict[str, Any]:
    """Run ``runner.jl`` in a fresh process; return the parsed result JSON.

    ``timed=True`` asks for exactly one warm propagation and no per-gate
    counting: the runner then does a cold run (paying JIT, discarded) and one
    timed warm run, matching the rust leg's warm-then-timed shape. ``PP_FUSED``
    stays off — its term-count parity is not established.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from julia_baseline import run_task

    payload = json.loads(task_path.read_text())
    res = run_task(
        payload,
        threads=threads,
        warm_repeats=1 if timed else 0,
        layer_counts=not timed,
        fused=False,
        timeout=timeout,
    )
    return res.raw


def _summarize_memory(samples: Sequence[Mapping[str, Any]], terms: int) -> dict[str, Any]:
    """Per-engine peak memory and bytes/term, medianed over the pairs."""

    def med(values: list[float]) -> float | None:
        vals = [v for v in values if v is not None]
        return median(vals) if vals else None

    rust_hwm = med([s["rust_memory"].get("vmhwm_kb") for s in samples])
    rust_floor = med([s["rust_memory"].get("vmrss_start_kb") for s in samples])
    jl_hwm = med([s["jl_memory"].get("vmhwm_kb") for s in samples])
    jl_floor = med([s["jl_memory"].get("vmrss_start_kb") for s in samples])
    return {
        "final_terms": terms,
        "rust_vmhwm_kb": rust_hwm,
        "rust_floor_kb": rust_floor,
        "rust_bytes_per_term": bytes_per_term(rust_hwm, rust_floor, terms),
        "jl_vmhwm_kb": jl_hwm,
        "jl_floor_kb": jl_floor,
        "jl_bytes_per_term": bytes_per_term(jl_hwm, jl_floor, terms),
    }


# ==========================================================================
# Workloads
#
# Four, chosen to separate what the engines actually differ at rather than to
# cover physics: two that exercise the native rotation path with different
# generator mixes, one that exercises the dense matrix-gate path on both sides,
# and one — `kicked_ising_deep` — that is the first workload's circuit taken to
# 20 Trotter steps purely to move a fixed term count away from the reachable
# set's closure, which is the saturation hypothesis's falsification test. Every
# configuration is Heisenberg, one gate per channel, single threaded, contracted
# against a state both engines can express.
# ==========================================================================


def _z_at(q: int, n: int) -> str:
    """Full-length Pauli label with ``Z`` on qubit ``q`` (leftmost = qubit 0)."""
    if not 0 <= q < n:
        raise ValueError(f"qubit {q} out of range for n={n}")
    return "I" * q + "Z" + "I" * (n - q - 1)


@dataclass(frozen=True)
class Workload:
    """One workload: a gate-list factory plus the sweep that varies its size."""

    key: str
    title: str
    n_qubits: int
    observable: dict[str, float]
    state: str
    cutoffs: tuple[float, ...]
    gates_factory: Callable[[], list[dict[str, Any]]]
    notes: str
    #: One extra configuration at the same term count reached by `max_weight`
    #: instead of `min_abs_coeff`, to show the two knobs are interchangeable as
    #: size dials. `(max_weight, min_abs_coeff)` — jl and this engine agree on
    #: the weight boundary exactly (README §P4), so no mitigation is needed.
    weight_variant: tuple[int, float] | None = None

    def gates(self) -> list[dict[str, Any]]:
        return self.gates_factory()


#: Benchmark E's seed, reused so the SU(4) blocks are the same matrices.
SU4_SEED = 20260831

#: Benchmark C's kicked-Ising angle. `theta_zz` stays at the Clifford point
#: `-pi/2`, which is what makes the cutoffs' dyadic straddle possible at all.
KICKED_ISING_THETA_H = 5.0 * math.pi / 16.0

#: The deep kicked-Ising angle, and the depth that goes with it. Benchmark C
#: proved per-layer parity at exactly this angle and depth — all 5420 layers
#: identical at `min_abs_coeff = 2^-14` (2 441 936 final / 3 108 582 peak
#: terms), which is what the thread-scaling section reuses — so the saturation
#: falsification test inherits a proven configuration rather than opening a new
#: untested one.
KICKED_ISING_DEEP_THETA_H = 7.0 * math.pi / 32.0
KICKED_ISING_DEEP_STEPS = 20


def workloads() -> dict[str, Workload]:
    """The four workloads, built lazily so `--workload` can pick one."""
    return {
        "kicked_ising": Workload(
            key="kicked_ising",
            title="kicked-Ising, heavy-hex 127q, 5 Trotter steps, theta_h = 5pi/16",
            n_qubits=127,
            observable={_z_at(62, 127): 1.0},
            state="z+",
            # Dyadic on purpose: benchmark C's grid, so its term counts and its
            # proven parity carry over. Dyadic cutoffs at a Clifford theta_zz
            # are exactly the case the one-ulp threshold mitigation exists for.
            cutoffs=(2.0**-4, 2.0**-6, 2.0**-8, 2.0**-10, 2.0**-12, 2.0**-14, 2.0**-16, 2.0**-18),
            gates_factory=lambda: kicked_ising_gates(
                127, trotter_steps=5, theta_h=KICKED_ISING_THETA_H
            ),
            weight_variant=(6, 2.0**-18),
            notes=(
                "Native ZZ pauli_rotations on both engines (jl PauliRotation, not a "
                "transfer map), 1355 channels (5 x (127 rx + 144 ZZ)), observable Z_62, "
                "theta_zz = -pi/2 at the Clifford point. Benchmark C's configuration."
            ),
        ),
        "kicked_ising_deep": Workload(
            key="kicked_ising_deep",
            title=(
                "kicked-Ising, heavy-hex 127q, 20 Trotter steps, theta_h = 7pi/32 "
                "(saturation falsification test)"
            ),
            n_qubits=127,
            observable={_z_at(62, 127): 1.0},
            state="z+",
            # Dyadic on purpose, same as the 5-step curve: this is the case the
            # one-ulp threshold mitigation exists for, so the falsification test
            # exercises it rather than dodging it with powers of ten. The two
            # tightest points (2^-13, 2^-14) are the ones that decide the
            # verdict; 2^-13 is inserted between the usual even exponents to put
            # a second measured point inside the 1e6-3e6 term band the 5-step
            # curve decayed across.
            cutoffs=(2.0**-8, 2.0**-10, 2.0**-12, 2.0**-13, 2.0**-14),
            gates_factory=lambda: kicked_ising_gates(
                127,
                trotter_steps=KICKED_ISING_DEEP_STEPS,
                theta_h=KICKED_ISING_DEEP_THETA_H,
            ),
            notes=(
                "The saturation falsification test for the 5-step curve's ratio decay "
                "(jl_performance/README.md, 'Hypothesis'). Same circuit family, same "
                "observable Z_62 and theta_zz = -pi/2, but 20 Trotter steps instead of "
                "5 — 5420 channels (20 x (127 rx + 144 ZZ)) — so that ~2-3e6 terms is "
                "far from the reachable Pauli set's closure instead of at it. If the "
                "ratio keeps rising with the term count and jl's per-term cost stays "
                "flat, the decay was a saturation regime; if it decays here too, it is "
                "a large-m property of this engine. Benchmark C's angle and depth, "
                "whose 5420 per-layer counts it already proved identical at 2^-14."
            ),
        ),
        "xxz": Workload(
            key="xxz",
            title="XXZ chain n=100, Jz=0.5, 6 Trotter steps, dt=0.1",
            n_qubits=100,
            observable={_z_at(50, 100): 1.0},
            # Domain wall |0...01...1>: a computational basis state, so jl can
            # contract it (overlapwithcomputational). |0...0> is an eigenstate
            # and |+...+> gives 0 by symmetry -- both vacuous.
            state="0" * 50 + "1" * 50,
            # Non-dyadic on purpose: the boundary divergence is measure-zero
            # here, so this workload is the control for the kicked-Ising one.
            cutoffs=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9),
            gates_factory=lambda: xxz_gates(100, 6, Jz=0.5, dt=0.1),
            notes=(
                "Rotations only, three generator types per bond (XX, YY, ZZ), 1782 "
                "channels (6 x 3 x 99), observable Z_50 against a domain-wall state. "
                "Benchmark D's configuration."
            ),
        ),
        "su4": Workload(
            key="su4",
            title="SU(4) brickwork n=36, depth 6",
            n_qubits=36,
            observable={_z_at(18, 36): 1.0},
            state="z+",
            cutoffs=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4),
            gates_factory=lambda: su4_gates(36, 6, SU4_SEED),
            notes=(
                "unitary_2q on this engine, TransferMapGate (a dense 16x16 PTM per "
                "block) on jl -- the matrix-gate path rather than the rotation kernel. "
                "Haar-random blocks from numpy default_rng(20260831) in layer-major "
                "emission order, so the circuit is a deterministic function of "
                "(n, depth, seed). Benchmark E's configuration and seed."
            ),
        ),
    }


# ==========================================================================
# Configuration bookkeeping
# ==========================================================================


@dataclass
class Config:
    """One point of a workload's sweep, and everything measured about it."""

    workload: str
    label: str
    n_qubits: int
    min_abs_coeff: float
    max_weight: int | None
    task_path: Path
    parity: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None
    cut: dict[str, Any] | None = None
    disqualified: str | None = None

    @property
    def knobs(self) -> dict[str, Any]:
        out: dict[str, Any] = {"min_abs_coeff": self.min_abs_coeff}
        if self.max_weight is not None:
            out["max_weight"] = self.max_weight
        return out


def write_task_pair(
    workload: Workload,
    *,
    out_dir: Path,
    min_abs_coeff: float,
    max_weight: int | None = None,
    threads: int = 1,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write the two task files for one configuration.

    Two files, not one, and this is the only asymmetry in the whole protocol:
    the Julia task carries ``nextafter(eps, +inf)`` as its ``min_abs_coeff`` so
    that its strict ``|c| < eps`` becomes this engine's inclusive
    ``|c| <= eps`` bit for bit (see :func:`julia_min_abs_coeff`). Everything
    else — every gate, the observable, the direction, the state, the weight
    cutoff — is identical, and the returned mapping records the perturbation so
    the results file carries it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gates = workload.gates()
    eps_jl = julia_min_abs_coeff(min_abs_coeff)

    common = dict(
        n_qubits=workload.n_qubits,
        gates=gates,
        observable=workload.observable,
        direction="heisenberg",
        max_weight=max_weight,
        threads=threads,
        state=workload.state,
    )
    rust_payload = make_task_payload(min_abs_coeff=min_abs_coeff, **common)
    jl_payload = make_task_payload(min_abs_coeff=eps_jl, **common)

    tag = f"{workload.key}-eps{min_abs_coeff:.6e}"
    if max_weight is not None:
        tag += f"-w{max_weight}"
    rust_path = out_dir / f"{tag}-rust.json"
    jl_path = out_dir / f"{tag}-jl.json"
    rust_path.write_text(json.dumps(rust_payload) + "\n")
    jl_path.write_text(json.dumps(jl_payload) + "\n")

    mitigation = {
        "min_abs_coeff_rust": min_abs_coeff,
        "min_abs_coeff_julia": eps_jl,
        "one_ulp_delta": eps_jl - min_abs_coeff,
        "cutoff_is_dyadic": is_dyadic(min_abs_coeff),
        "n_gates": len(gates),
    }
    return rust_path, jl_path, mitigation


def parity_gate(
    rust_task: Path, jl_task: Path, *, label: str, timeout: float = 7200.0
) -> dict[str, Any]:
    """Run both engines untimed and require identical per-layer term counts.

    Takes the two per-engine task files, so the one-ulp Julia threshold is in
    play — that is the configuration the timed legs will actually run, and
    therefore the one parity must hold for.

    Raises :class:`ParityFailure` on any mismatch. Returns both engines' counts
    and expectations, so a passing gate is recorded evidence the README quotes
    rather than a silent precondition.
    """
    rust = _spawn_rust_leg(rust_task, "parity", threads=1, timeout=timeout)
    jl = _spawn_jl_leg(jl_task, timed=False, threads=1, timeout=timeout)

    jl_layers = jl["result"]["per_layer_terms"]
    problems = check_parity(rust["per_layer_terms"], jl_layers)
    if rust["final_terms"] != jl["result"]["final_terms"]:
        problems.insert(
            0,
            f"final term count: rust={rust['final_terms']} julia={jl['result']['final_terms']}",
        )
    exp_jl_raw = jl["result"]["expectation"]
    exp_rust = rust["expectation"]
    exp_jl = None if exp_jl_raw is None else complex(exp_jl_raw["re"], exp_jl_raw["im"]).real
    exp_delta = None
    if exp_jl is not None and exp_rust is not None:
        exp_delta = abs(exp_jl - exp_rust)
        if exp_delta > 1e-9:
            problems.append(f"expectation differs by {exp_delta:.3e} (> 1e-9)")

    out = {
        "label": label,
        "ok": not problems,
        "problems": problems,
        "rust_final_terms": rust["final_terms"],
        "jl_final_terms": jl["result"]["final_terms"],
        "n_layers": len(rust["per_layer_terms"]),
        "rust_peak_terms": max([rust["input_terms"], *rust["per_layer_terms"]], default=0),
        "jl_peak_terms": jl["result"]["peak_terms"],
        "expectation_rust": exp_rust,
        "expectation_jl": exp_jl,
        "expectation_delta": exp_delta,
    }
    if problems:
        raise ParityFailure(
            f"[{label}] PARITY FAILED — configuration disqualified, no timing reported:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
    return out


def run_pairs(
    rust_task: Path,
    jl_task: Path,
    *,
    label: str,
    pairs: int = DEFAULT_PAIRS,
    threads: int = 1,
    timeout: float = 7200.0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run ``pairs`` interleaved (rust, julia) pairs and analyze them.

    The within-pair order alternates — rust first on even pairs, Julia first on
    odd — so a monotone drift in machine state cannot masquerade as a
    consistent win for whichever engine always ran second
    (``PROFILING.md``'s ``--order abba``).

    Legs run strictly sequentially: never two engines at once, and never
    alongside a build. On a quiet box that is what makes adjacent-in-time
    pairing meaningful.
    """
    samples: list[dict[str, Any]] = []
    for i in range(pairs):
        rust_first = rust_runs_first(i)
        if rust_first:
            r = _spawn_rust_leg(rust_task, "timed", threads=threads, timeout=timeout)
            j = _spawn_jl_leg(jl_task, timed=True, threads=1, timeout=timeout)
        else:
            j = _spawn_jl_leg(jl_task, timed=True, threads=1, timeout=timeout)
            r = _spawn_rust_leg(rust_task, "timed", threads=threads, timeout=timeout)
        rust_s = float(r["propagation_s"])
        jl_s = float(j["timing"]["wall_warm_s"])
        samples.append(
            {
                "pair": i,
                "order": "rust-first" if rust_first else "julia-first",
                "rust_s": rust_s,
                "jl_s": jl_s,
                "ratio_jl_over_rust": jl_s / rust_s,
                "rust_final_terms": r["final_terms"],
                "jl_final_terms": j["result"]["final_terms"],
                "rust_peak_terms": r["peak_terms"],
                "jl_peak_terms": j["result"]["peak_terms"],
                "rust_memory": r["memory"],
                "jl_memory": j.get("memory", {}),
                "jl_cold_s": j["timing"]["wall_cold_s"],
            }
        )
        log(
            f"    pair {i} ({samples[-1]['order']:>11}): "
            f"rust {rust_s:9.4f}s  jl {jl_s:9.4f}s  ratio {jl_s / rust_s:6.3f}"
        )

    analysis = analyze_pairs(samples)
    analysis["label"] = label
    analysis["pairs"] = samples
    rust_terms = {s["rust_final_terms"] for s in samples}
    jl_terms = {s["jl_final_terms"] for s in samples}
    if len(rust_terms) != 1 or len(jl_terms) != 1:
        raise ParityFailure(
            f"[{label}] term counts varied across legs (rust {sorted(rust_terms)}, "
            f"julia {sorted(jl_terms)}) — propagation is deterministic in term count, "
            "so this means the legs did not run the same task"
        )
    if rust_terms != jl_terms:
        raise ParityFailure(
            f"[{label}] engines disagree on final term count during timing "
            f"(rust {rust_terms}, julia {jl_terms})"
        )
    analysis["final_terms"] = rust_terms.pop()
    peaks = {s["rust_peak_terms"] for s in samples if s["rust_peak_terms"] is not None}
    analysis["peak_terms"] = max(peaks) if peaks else analysis["final_terms"]
    analysis["jl_peak_terms"] = samples[0]["jl_peak_terms"]
    analysis["peak_memory"] = _summarize_memory(samples, analysis["final_terms"])
    analysis["peak_memory"]["peak_terms"] = analysis["peak_terms"]
    analysis["peak_memory"]["rust_bytes_per_peak_term"] = bytes_per_term(
        analysis["peak_memory"]["rust_vmhwm_kb"],
        analysis["peak_memory"]["rust_floor_kb"],
        analysis["peak_terms"],
    )
    analysis["peak_memory"]["jl_bytes_per_peak_term"] = bytes_per_term(
        analysis["peak_memory"]["jl_vmhwm_kb"],
        analysis["peak_memory"]["jl_floor_kb"],
        analysis["peak_terms"],
    )
    return analysis


# ==========================================================================
# Pilot / projection: decide what to cut before spending hours on it
# ==========================================================================

#: Cut a leg whose projected single Julia run (cold + warm, since a timed leg
#: pays both) exceeds this. The task-level rule.
JL_RUN_BUDGET_S = 30 * 60.0

#: Cut a leg whose projected Julia peak exceeds this fraction of free RAM.
FREE_RAM_FRACTION = 0.5

#: jl's measured dict-backend cost, benchmark C's per-process refit
#: (deep_trotter/README.md §6.1): ~0.7 GiB fixed + 0.44-0.74 KiB per resident
#: term. The conservative slope is used for projections.
JL_BASE_GIB = 0.7
JL_KIB_PER_TERM = 0.74


def free_ram_gib() -> float | None:
    """``MemAvailable`` from ``/proc/meminfo``, in GiB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError):
        return None
    return None


def project_leg(
    *,
    terms: int,
    ref_terms: int,
    ref_rust_s: float,
    ref_ratio: float,
) -> dict[str, Any]:
    """Project a heavier leg's cost from a measured lighter one.

    Linear in the term count for time (both engines are near-linear per layer
    at fixed channel count) and jl's measured per-term memory model for space.
    A projection is a *decision aid*, recorded with the decision it drove; it
    is never reported as a measurement.
    """
    scale = terms / max(ref_terms, 1)
    rust_s = ref_rust_s * scale
    jl_warm_s = rust_s * ref_ratio
    # A timed jl leg pays a cold run (JIT + a full propagation) and a warm one.
    jl_leg_s = 2.0 * jl_warm_s
    jl_gib = JL_BASE_GIB + JL_KIB_PER_TERM * terms / (1024.0 * 1024.0)
    return {
        "projected_terms": terms,
        "from_terms": ref_terms,
        "projected_rust_s": rust_s,
        "projected_jl_warm_s": jl_warm_s,
        "projected_jl_leg_s": jl_leg_s,
        "projected_jl_peak_gib": jl_gib,
        "model": (
            "time linear in term count from the pilot point; jl memory = "
            f"{JL_BASE_GIB} GiB + {JL_KIB_PER_TERM} KiB/term (benchmark C's per-process refit)"
        ),
    }


def should_cut(projection: Mapping[str, Any], free_gib: float | None) -> str | None:
    """``None`` to run the leg, else the reason to cut it."""
    if projection["projected_jl_leg_s"] > JL_RUN_BUDGET_S:
        return (
            f"a single julia leg projects {projection['projected_jl_leg_s'] / 60.0:.1f} min "
            f"(> {JL_RUN_BUDGET_S / 60.0:.0f} min budget)"
        )
    if free_gib is not None and projection["projected_jl_peak_gib"] > FREE_RAM_FRACTION * free_gib:
        return (
            f"julia projects {projection['projected_jl_peak_gib']:.1f} GiB "
            f"(> {FREE_RAM_FRACTION:.0%} of {free_gib:.0f} GiB free)"
        )
    return None


# ==========================================================================
# Section 1: time-vs-term-count curves
# ==========================================================================


def run_curve(
    workload: Workload,
    *,
    task_dir: Path,
    pairs: int,
    pilot_only: bool,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Sweep one workload's cutoffs, gating parity and pairing every timing.

    Order of operations per cutoff, loosest first: write the two task files,
    run the parity gate, project the next-heavier leg from what this one
    measured, and only then run the timed pairs. Sweeping loosest-first is what
    makes the projection possible — every decision to cut a heavy leg is made
    from a measurement, not a guess.
    """
    log("")
    log(f"=== curve: {workload.title} ===")
    log(f"    {workload.notes}")
    free_gib = free_ram_gib()
    log(f"    free RAM: {free_gib:.0f} GiB" if free_gib else "    free RAM: unknown")

    configs: list[Config] = []
    points: list[dict[str, Any]] = []
    cuts: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    last: dict[str, Any] | None = None

    sweep: list[tuple[float, int | None]] = [(eps, None) for eps in workload.cutoffs]
    if workload.weight_variant is not None and not pilot_only:
        w, eps = workload.weight_variant
        sweep.append((eps, w))

    for eps, max_weight in sweep:
        knob = f"eps={eps:.3e}" + (f" max_weight={max_weight}" if max_weight else "")
        label = f"{workload.key} {knob}"
        rust_task, jl_task, mitigation = write_task_pair(
            workload, out_dir=task_dir, min_abs_coeff=eps, max_weight=max_weight
        )
        cfg = Config(
            workload=workload.key,
            label=label,
            n_qubits=workload.n_qubits,
            min_abs_coeff=eps,
            max_weight=max_weight,
            task_path=rust_task,
        )
        cfg_extra = {"mitigation": mitigation}

        # --- projection / cut decision, from the previous (lighter) leg ------
        if last is not None:
            # The term count at this cutoff is not known before running it, so
            # project it from the previous leg's growth. Deliberately crude and
            # deliberately recorded: it only ever decides whether to spend the
            # hours, and the decision is written down either way.
            projected_terms = int(last["final_terms"] * last.get("growth", 4.0))
            projection = project_leg(
                terms=projected_terms,
                ref_terms=last["final_terms"],
                ref_rust_s=last["rust_s_median"],
                ref_ratio=last["median_ratio"],
            )
            reason = should_cut(projection, free_gib)
            # Recorded whether or not it triggered a cut: a projection that
            # *authorized* a leg is as much a part of the record as one that
            # killed it, and without it "nothing was cut" is unfalsifiable.
            projections.append(
                {"label": label, "cut": reason is not None, "reason": reason, **projection}
            )
            if reason is not None:
                log(f"  CUT {label}: {reason}")
                cfg.cut = {"reason": reason, "projection": projection}
                cuts.append({"label": label, "reason": reason, "projection": projection})
                configs.append(cfg)
                break

        # --- parity gate -----------------------------------------------------
        log(f"  {label}: parity gate ...")
        try:
            parity = parity_gate(rust_task, jl_task, label=label)
        except ParityFailure as exc:
            log(f"  !! {exc}")
            cfg.disqualified = str(exc)
            configs.append(cfg)
            continue
        cfg.parity = parity
        log(
            f"    parity OK: {parity['n_layers']} layers identical, "
            f"{parity['rust_final_terms']} terms, "
            f"|dE| = {parity['expectation_delta']:.2e}"
            if parity["expectation_delta"] is not None
            else f"    parity OK: {parity['n_layers']} layers identical"
        )

        # --- timed pairs -----------------------------------------------------
        n_pairs = 1 if pilot_only else pairs
        timing = run_pairs(
            rust_task, jl_task, label=label, pairs=n_pairs, log=log
        )
        timing.update(cfg_extra)
        timing["truncation"] = cfg.knobs
        timing["parity"] = parity
        cfg.timing = timing
        configs.append(cfg)

        prev_terms = last["final_terms"] if last else None
        last = {
            "final_terms": timing["final_terms"],
            "rust_s_median": timing["rust_s_median"],
            "median_ratio": timing["median_ratio_jl_over_rust"],
            "growth": (
                timing["final_terms"] / prev_terms
                if prev_terms and prev_terms > 0 and timing["final_terms"] > prev_terms
                else 4.0
            ),
        }
        points.append(
            {
                "label": label,
                "max_weight": max_weight,
                "final_terms": timing["final_terms"],
                "peak_terms": timing["peak_terms"],
                "median_ratio_jl_over_rust": timing["median_ratio_jl_over_rust"],
                "sign_consistent": timing["sign_consistent"],
                "verdict": timing["verdict"],
                "rust_s_median": timing["rust_s_median"],
                "jl_s_median": timing["jl_s_median"],
            }
        )
        verdict = timing["verdict"]
        log(
            f"    -> {timing['final_terms']:>9} terms  median ratio "
            f"{timing['median_ratio_jl_over_rust']:6.3f}  [{verdict}]"
        )

    # The max_weight point is a knob-*equivalence* demonstration, not a point on
    # the min_abs_coeff size curve, so it must not bracket the crossover: a
    # crossover is only meaningful along a single-parameter family, and mixing
    # two knobs on one axis can manufacture a bracket where the sweep has none.
    curve_points = [p for p in points if p.get("max_weight") is None]
    crossover = interpolate_crossover(curve_points, "peak_terms")
    if crossover["crossover_terms"] is not None:
        log(
            f"  crossover at ~{crossover['crossover_terms']:.3g} terms"
            + ("  (inside the indistinguishable zone)" if crossover["inside_indistinguishable_zone"] else "")
        )
    else:
        log(f"  no crossover: {crossover.get('note', '')}")

    return {
        "workload": workload.key,
        "title": workload.title,
        "notes": workload.notes,
        "n_qubits": workload.n_qubits,
        "observable": workload.observable,
        "state": workload.state,
        "points": points,
        "crossover": crossover,
        "cuts": cuts,
        "projections": projections,
        "free_ram_gib_at_start": free_gib,
        "disqualified": [
            {"label": c.label, "reason": c.disqualified} for c in configs if c.disqualified
        ],
        "configs": [
            {
                "label": c.label,
                "truncation": c.knobs,
                "parity": c.parity,
                "timing": c.timing,
                "cut": c.cut,
                "disqualified": c.disqualified,
            }
            for c in configs
        ],
    }


# ==========================================================================
# Section 2: time to fixed accuracy
# ==========================================================================


@dataclass(frozen=True)
class AccuracyReference:
    """A claimable reference point: an exact oracle plus the grid to sweep to it.

    ``oracle`` values are taken from the committed summaries of benchmarks B
    and C, where each was produced by an *exact* method — a stim Clifford
    tableau, or two independent simulations over a causal cone required to
    agree — and marked claimable there. They are reference *data*, quoted with
    their provenance, not recomputed here.
    """

    key: str
    title: str
    theta_h: float
    theta_label: str
    steps: int
    oracle: float
    oracle_source: str
    cutoffs: tuple[float, ...]


ACCURACY_REFERENCES = (
    AccuracyReference(
        key="ki_7pi32_s5",
        title="kicked-Ising 127q, 5 steps, theta_h = 7pi/32",
        theta_h=7.0 * math.pi / 32.0,
        theta_label="7pi/32",
        steps=5,
        oracle=0.655563050749494,
        oracle_source=(
            "benchmarks/python/deep_trotter/summary.json — exact, "
            "light_cone_exact:both (19-qubit cone, Aer statevector and untruncated "
            "Pauli propagation required to agree); claimable row"
        ),
        cutoffs=(2.0**-8, 2.0**-10, 2.0**-12, 2.0**-14, 2.0**-16),
    ),
    AccuracyReference(
        key="ki_5pi16_s5",
        title="kicked-Ising 127q, 5 steps, theta_h = 5pi/16",
        theta_h=5.0 * math.pi / 16.0,
        theta_label="5pi/16",
        steps=5,
        oracle=0.2384771180185389,
        oracle_source=(
            "benchmarks/python/deep_trotter/summary.json — exact, "
            "light_cone_exact:both (19-qubit cone); claimable row"
        ),
        cutoffs=(2.0**-8, 2.0**-10, 2.0**-12),
    ),
    AccuracyReference(
        key="ki_pi4_s5",
        title="kicked-Ising 127q, 5 steps, theta_h = pi/4",
        theta_h=math.pi / 4.0,
        theta_label="pi/4",
        steps=5,
        oracle=0.5194110175524836,
        oracle_source=(
            "benchmarks/python/theta_sweep/summary.json — exact, "
            "light_cone_exact:both (19-qubit cone); exact row"
        ),
        # Powers of ten: non-dyadic, so this reference is the control against
        # the two dyadic ones above.
        cutoffs=(1e-2, 1e-3, 1e-4, 1e-5),
    ),
)

#: The two accuracy bars. Strict `<`, so each reads "error below this".
ACCURACY_BARS = (1e-2, 1e-3)


def run_time_to_accuracy(
    reference: AccuracyReference,
    *,
    task_dir: Path,
    pairs: int,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Sweep ``reference``'s grid and report each engine's cheapest passing config.

    The error at a given truncation is engine-*independent* — the parity gate
    proves both engines produce the same expectation to ~1e-16 — so the two
    engines pass each bar at the same grid points. What differs is the wall time
    at those points, which is exactly the question: at equal accuracy, who is
    faster. "Cheapest" is selected per engine from that engine's own median
    time, so a grid where the two engines order the passing configs differently
    is reported as such rather than assumed away.
    """
    log("")
    log(f"=== time to fixed accuracy: {reference.title} ===")
    log(f"    oracle {reference.oracle!r}")
    log(f"    source {reference.oracle_source}")

    workload = Workload(
        key=reference.key,
        title=reference.title,
        n_qubits=127,
        observable={_z_at(62, 127): 1.0},
        state="z+",
        cutoffs=reference.cutoffs,
        gates_factory=lambda: kicked_ising_gates(
            127, trotter_steps=reference.steps, theta_h=reference.theta_h
        ),
        notes="",
    )

    rows: list[dict[str, Any]] = []
    for eps in reference.cutoffs:
        label = f"{reference.key} eps={eps:.3e}"
        rust_task, jl_task, mitigation = write_task_pair(
            workload, out_dir=task_dir, min_abs_coeff=eps
        )
        try:
            parity = parity_gate(rust_task, jl_task, label=label)
        except ParityFailure as exc:
            log(f"  !! {exc}")
            rows.append({"min_abs_coeff": eps, "disqualified": str(exc)})
            continue
        timing = run_pairs(rust_task, jl_task, label=label, pairs=pairs, log=log)
        expectation = parity["expectation_rust"]
        error = abs(expectation - reference.oracle) if expectation is not None else None
        rows.append(
            {
                "min_abs_coeff": eps,
                "cutoff_is_dyadic": mitigation["cutoff_is_dyadic"],
                "min_abs_coeff_julia": mitigation["min_abs_coeff_julia"],
                "final_terms": timing["final_terms"],
                "peak_terms": timing["peak_terms"],
                "expectation": expectation,
                "expectation_jl": parity["expectation_jl"],
                "absolute_error": error,
                "rust_s_median": timing["rust_s_median"],
                "jl_s_median": timing["jl_s_median"],
                "median_ratio_jl_over_rust": timing["median_ratio_jl_over_rust"],
                "sign_consistent": timing["sign_consistent"],
                "verdict": timing["verdict"],
                "n_pairs": timing["n_pairs"],
                "ratio_per_pair": timing["ratio_jl_over_rust_per_pair"],
                "parity": parity,
            }
        )
        log(
            f"    eps={eps:.3e}: {timing['final_terms']:>9} terms  "
            f"|err| = {error:.3e}  rust {timing['rust_s_median']:.4f}s  "
            f"jl {timing['jl_s_median']:.4f}s  ratio {timing['median_ratio_jl_over_rust']:.3f}"
        )

    bars: dict[str, Any] = {}
    for bar in ACCURACY_BARS:
        passing = [
            r for r in rows
            if r.get("absolute_error") is not None and r["absolute_error"] < bar
        ]
        entry: dict[str, Any] = {"bar": bar, "reached": bool(passing)}
        if passing:
            cheap_rust = min(passing, key=lambda r: r["rust_s_median"])
            cheap_jl = min(passing, key=lambda r: r["jl_s_median"])
            entry["paulistrings"] = {
                "min_abs_coeff": cheap_rust["min_abs_coeff"],
                "final_terms": cheap_rust["final_terms"],
                "absolute_error": cheap_rust["absolute_error"],
                "wall_s": cheap_rust["rust_s_median"],
            }
            entry["julia"] = {
                "min_abs_coeff": cheap_jl["min_abs_coeff"],
                "final_terms": cheap_jl["final_terms"],
                "absolute_error": cheap_jl["absolute_error"],
                "wall_s": cheap_jl["jl_s_median"],
            }
            entry["same_configuration"] = (
                cheap_rust["min_abs_coeff"] == cheap_jl["min_abs_coeff"]
            )
            entry["speedup_rust_over_jl"] = (
                cheap_jl["jl_s_median"] / cheap_rust["rust_s_median"]
            )
            # Only a sign-consistent point supports a directional claim.
            entry["claimable"] = bool(
                cheap_rust["sign_consistent"] and cheap_jl["sign_consistent"]
            )
        else:
            entry["note"] = "no swept configuration reached this bar"
        bars[f"{bar:g}"] = entry
        if passing:
            log(
                f"  |err| < {bar:g}: rust {entry['paulistrings']['wall_s']:.4f}s vs "
                f"jl {entry['julia']['wall_s']:.4f}s "
                f"({entry['speedup_rust_over_jl']:.2f}x)"
            )
        else:
            log(f"  |err| < {bar:g}: not reached in the swept grid")

    return {
        "key": reference.key,
        "title": reference.title,
        "theta_h": reference.theta_h,
        "theta_label": reference.theta_label,
        "steps": reference.steps,
        "oracle": reference.oracle,
        "oracle_source": reference.oracle_source,
        "rows": rows,
        "bars": bars,
    }


# ==========================================================================
# Section 3: thread scaling — THIS ENGINE ONLY
#
# PauliPropagation.jl 0.8.2's propagation is single-threaded: `propagate`
# takes a `thread` keyword, but the dict backend this suite benchmarks has no
# threaded path, and the array backend (`VectorPauliSum`) has no established
# term-count parity here. So there is nothing to pair against, and this section
# is deliberately NOT a comparison. It never appears in the same figure or the
# same table as the head-to-head above.
# ==========================================================================

THREAD_COUNTS = (1, 2, 4, 8, 16, 32)


def run_thread_scaling(
    *,
    task_dir: Path,
    theta_h: float,
    theta_label: str,
    steps: int,
    min_abs_coeff: float,
    thread_counts: Sequence[int] = THREAD_COUNTS,
    log: Callable[[str], None],
) -> dict[str, Any]:
    """Scale one heavy, parity-proven configuration across Rayon thread counts.

    One process per thread count, ``RAYON_NUM_THREADS`` set before it starts —
    Rayon builds its global pool once, at the first propagate, and never
    resizes it, so the count cannot be changed inside a live process.
    """
    log("")
    log(f"=== thread scaling (this engine only): kicked-Ising theta_h={theta_label}, "
        f"{steps} steps, eps={min_abs_coeff:.3e} ===")
    log("    PauliPropagation.jl 0.8.2 has no threaded dict-backend path; "
        "this section is not a comparison.")

    workload = Workload(
        key=f"threadscale_{theta_label.replace('/', '')}",
        title="thread scaling",
        n_qubits=127,
        observable={_z_at(62, 127): 1.0},
        state="z+",
        cutoffs=(min_abs_coeff,),
        gates_factory=lambda: kicked_ising_gates(127, trotter_steps=steps, theta_h=theta_h),
        notes="",
    )
    rust_task, _jl_task, _mit = write_task_pair(
        workload, out_dir=task_dir, min_abs_coeff=min_abs_coeff
    )

    rows: list[dict[str, Any]] = []
    baseline: float | None = None
    for t in thread_counts:
        res = _spawn_rust_leg(rust_task, "timed", threads=t)
        wall = float(res["propagation_s"])
        if baseline is None:
            baseline = wall
        speedup = baseline / wall
        rows.append(
            {
                "threads": t,
                "propagation_s": wall,
                "speedup": speedup,
                "efficiency": speedup / t,
                "final_terms": res["final_terms"],
                "peak_terms": res["peak_terms"],
                "vmhwm_kb": res["memory"].get("vmhwm_kb"),
                "observed_threads": res.get("observed_threads"),
            }
        )
        log(
            f"    {t:>2} threads: {wall:9.3f}s  speedup {speedup:6.2f}x  "
            f"efficiency {speedup / t:5.1%}"
        )
    return {
        "configuration": {
            "workload": "kicked_ising",
            "theta_label": theta_label,
            "theta_h": theta_h,
            "trotter_steps": steps,
            "min_abs_coeff": min_abs_coeff,
            "n_qubits": 127,
        },
        "julia_note": (
            "PauliPropagation.jl 0.8.2's dict backend (`PauliSum`, the storage this "
            "study benchmarks) has no threaded propagation path, so there is no "
            "Julia curve to put beside this one. Its array backend "
            "(`VectorPauliSum`, PP_BACKEND=vector) does take a `thread` keyword, but "
            "term-count parity has not been established for it here, so it is out of "
            "scope rather than quietly substituted. Multithreading and GPU support "
            "are on the PauliPropagation.jl roadmap; when a threaded backend lands "
            "with parity, this section becomes a comparison and the head-to-head "
            "above stays single-threaded regardless."
        ),
        "rows": rows,
    }


# ==========================================================================
# Reporting
# ==========================================================================


def extension_provenance() -> dict[str, Any]:
    """Where the compiled extension actually came from.

    ``report.collect_provenance`` records the commit of the *working tree the
    driver runs in*, which is not necessarily the commit the ``.so`` was built
    from — a study can legitimately run from a branch that has no build of its
    own, against an extension built elsewhere. Recording only the former would
    attribute the measurements to code that never executed.

    So this resolves the extension the interpreter actually imported, finds the
    checkout it lives in, and reads *that* checkout's HEAD. When the two differ
    the README is obliged to say why the difference is harmless — the honest
    version of which is "the propagation source is identical, here is the diff".
    """
    out: dict[str, Any] = {}
    try:
        import paulistrings

        pkg_init = Path(paulistrings.__file__).resolve()
        out["package_path"] = str(pkg_init)
        ext = pkg_init.parent / "_paulistrings.abi3.so"
        if ext.exists():
            out["extension_path"] = str(ext)
            out["extension_mtime"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(ext.stat().st_mtime)
            )
        checkout = pkg_init.parents[2]
        out["checkout"] = str(checkout)
        for key, cmd in (
            ("commit", ["git", "rev-parse", "HEAD"]),
            ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ):
            try:
                out[key] = subprocess.run(
                    cmd, cwd=checkout, capture_output=True, text=True, check=True
                ).stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                out[key] = "unknown"
    except Exception as exc:  # pragma: no cover - provenance must never break a run
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def build_run_records(curves: Sequence[Mapping[str, Any]]) -> list[Any]:
    """One `report.RunRecord` per (configuration, engine), from the pair medians.

    The committed `results.json` follows the suite's convention — a flat JSON
    array of `RunRecord` dicts — so it feeds `report.py`'s plot helpers
    unchanged. The protocol-level structure (per-pair ratios, crossovers,
    parity evidence, cuts) lives in `summary.json`, as in benchmark B.
    """
    sys.path.insert(0, str(REPO_ROOT / "examples"))
    from common.report import RunRecord, collect_provenance

    provenance = collect_provenance(
        seeds={"su4_circuit": SU4_SEED},
        thread_count=1,
        repo_root=REPO_ROOT,
    )
    records: list[Any] = []
    for curve in curves:
        for cfg in curve["configs"]:
            timing = cfg.get("timing")
            if not timing:
                continue
            mem = timing["peak_memory"]
            shared = dict(
                n_qubits=curve["n_qubits"],
                direction="heisenberg",
                truncation=dict(cfg["truncation"]),
                final_terms=timing["final_terms"],
                peak_terms=timing["peak_terms"],
                provenance=provenance,
            )
            common_extra = {
                "workload": curve["workload"],
                "label": cfg["label"],
                "n_pairs": timing["n_pairs"],
                "median_ratio_jl_over_rust": timing["median_ratio_jl_over_rust"],
                "ratio_jl_over_rust_per_pair": timing["ratio_jl_over_rust_per_pair"],
                "sign_consistent": timing["sign_consistent"],
                "verdict": timing["verdict"],
                "mitigation": timing["mitigation"],
            }
            records.append(
                RunRecord(
                    engine="paulistrings",
                    engine_version=provenance.library_versions.get("paulistrings", "unknown"),
                    propagation_time_s=timing["rust_s_median"],
                    expectation_value=timing["parity"]["expectation_rust"],
                    peak_memory_kb=mem["rust_vmhwm_kb"],
                    extra={
                        **common_extra,
                        "wall_s_per_pair": timing["rust_s_per_pair"],
                        "memory_floor_kb": mem["rust_floor_kb"],
                        "bytes_per_final_term": mem["rust_bytes_per_term"],
                        "bytes_per_peak_term": mem["rust_bytes_per_peak_term"],
                    },
                    **shared,
                )
            )
            records.append(
                RunRecord(
                    engine="PauliPropagation.jl",
                    engine_version="0.8.2",
                    propagation_time_s=timing["jl_s_median"],
                    expectation_value=timing["parity"]["expectation_jl"],
                    peak_memory_kb=mem["jl_vmhwm_kb"],
                    extra={
                        **common_extra,
                        "wall_s_per_pair": timing["jl_s_per_pair"],
                        "memory_floor_kb": mem["jl_floor_kb"],
                        "bytes_per_final_term": mem["jl_bytes_per_term"],
                        "bytes_per_peak_term": mem["jl_bytes_per_peak_term"],
                        "julia_version": "1.12.6",
                    },
                    **shared,
                )
            )
    return records


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Head-to-head vs PauliPropagation.jl. See the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rust-timed-leg", metavar="TASK", help=argparse.SUPPRESS)
    parser.add_argument("--rust-parity-leg", metavar="TASK", help=argparse.SUPPRESS)
    parser.add_argument("--all", action="store_true", help="every section")
    parser.add_argument("--curves", action="store_true", help="time-vs-term-count curves")
    parser.add_argument("--accuracy", action="store_true", help="time to fixed accuracy")
    parser.add_argument("--threads", action="store_true", help="thread scaling (this engine only)")
    parser.add_argument("--figures", action="store_true", help="re-render figures from results.json")
    parser.add_argument(
        "--workload", action="append", choices=sorted(workloads()), help="restrict --curves"
    )
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS, help=f"default {DEFAULT_PAIRS}")
    parser.add_argument("--pilot", action="store_true", help="1 pair per config, no weight variant")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    if args.rust_timed_leg:
        print(json.dumps(rust_timed_leg(args.rust_timed_leg)))
        return 0
    if args.rust_parity_leg:
        print(json.dumps(rust_parity_leg(args.rust_parity_leg)))
        return 0

    if args.pairs < 1:
        parser.error("--pairs must be >= 1")
    if not (args.all or args.curves or args.accuracy or args.threads or args.figures):
        parser.error("pick a section: --all, --curves, --accuracy, --threads or --figures")
    if os.environ.get("RAYON_NUM_THREADS") != "1":
        parser.error(
            "RAYON_NUM_THREADS must be '1' in the environment before the interpreter "
            "starts (Rayon builds its global pool once, at the first propagate, and "
            "never resizes it). Re-run as:\n"
            "    RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py ..."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_dir = out_dir / "tasks"
    log_path = out_dir / "run.log"
    log_file = log_path.open("a")

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    started = time.time()
    log("")
    log(f"### bench_jl_performance start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"    pairs={args.pairs} pilot={args.pilot} sections=" + " ".join(
        s for s, on in (
            ("curves", args.all or args.curves),
            ("accuracy", args.all or args.accuracy),
            ("threads", args.all or args.threads),
        ) if on
    ))

    summary: dict[str, Any] = {
        "protocol": {
            "ratio_convention": "ratio = t_julia / t_paulistrings; > 1 means paulistrings faster",
            "pairs_per_configuration": args.pairs,
            "within_pair_order": "abba — rust-first on even pairs, julia-first on odd",
            "acceptance": (
                "direction consistency across every pair; median ratio is the effect "
                "size. Mixed signs are reported as 'indistinguishable', never as a "
                "small win (benchmarks/PROFILING.md)."
            ),
            "direction": "heisenberg",
            "threads": {"rayon": 1, "julia": 1},
            "timing": (
                "warm in-process on both sides: one untimed propagation, then one timed "
                "propagation, in the same process. Construction, contraction, oracles "
                "and logging are all outside the timed region."
            ),
            "parity_gate": (
                "every timed configuration first passes a per-layer term-count "
                "comparison; a failure disqualifies the configuration and no timing "
                "for it is reported."
            ),
            "cutoff_mitigation": (
                "the julia task carries nextafter(eps, +inf) so its strict |c| < eps "
                "equals this engine's inclusive |c| <= eps bit for bit; the threshold "
                "moves by one ulp and no coefficient is touched."
            ),
            "memory": (
                "each engine samples its own /proc/self/status (VmRSS floor, VmHWM "
                "peak). getrusage(RUSAGE_CHILDREN) is never used — it conflates "
                "siblings."
            ),
        },
        "curves": [],
        "accuracy": [],
        "thread_scaling": None,
        "cuts": [],
    }

    all_workloads = workloads()
    selected = args.workload or list(all_workloads)

    if args.all or args.curves:
        for key in selected:
            curve = run_curve(
                all_workloads[key],
                task_dir=task_dir,
                pairs=args.pairs,
                pilot_only=args.pilot,
                log=log,
            )
            summary["curves"].append(curve)
            summary["cuts"].extend(curve["cuts"])

    if args.all or args.accuracy:
        references = ACCURACY_REFERENCES
        if args.pilot:
            # One reference, its two loosest points: enough to exercise the bar
            # selection and the figure without paying for the tight tail.
            first = ACCURACY_REFERENCES[0]
            references = (
                AccuracyReference(**{**first.__dict__, "cutoffs": first.cutoffs[:2]}),
            )
        for reference in references:
            summary["accuracy"].append(
                run_time_to_accuracy(
                    reference, task_dir=task_dir, pairs=1 if args.pilot else args.pairs, log=log
                )
            )

    if args.all or args.threads:
        # The heaviest kicked-Ising configuration whose per-layer parity is
        # already proven: benchmark C established all 5420 layers identical at
        # theta_h = 7pi/32, 20 steps, min_abs_coeff = 2^-14 (2 441 936 final /
        # 3 108 582 peak terms). Heavy enough that fixed overheads do not
        # dominate the 32-thread point, which the 5-step configurations are too
        # small to satisfy.
        summary["thread_scaling"] = run_thread_scaling(
            task_dir=task_dir,
            theta_h=7.0 * math.pi / 32.0,
            theta_label="7pi/32",
            steps=5 if args.pilot else 20,
            min_abs_coeff=2.0**-10 if args.pilot else 2.0**-14,
            thread_counts=(1, 2) if args.pilot else THREAD_COUNTS,
            log=log,
        )

    summary["wall_clock_s"] = time.time() - started
    summary["free_ram_gib_at_end"] = free_ram_gib()
    summary["extension_provenance"] = extension_provenance()

    # results.json is a snapshot, regenerated wholesale: report.write_results
    # appends by design (right for a gitignored campaign directory, wrong for a
    # committed artifact), so the file is removed first — benchmark B's idiom.
    if summary["curves"]:
        sys.path.insert(0, str(REPO_ROOT / "examples"))
        from common import report

        results_path = out_dir / "results.json"
        if results_path.exists():
            results_path.unlink()
        report.write_results(build_run_records(summary["curves"]), out_dir, name="results")
        log(f"wrote {results_path}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    log(f"wrote {summary_path}")

    if args.figures or args.all or args.curves or args.accuracy or args.threads:
        try:
            from jl_performance_figures import render_all

            render_all(summary, out_dir)
            log(f"wrote figures to {out_dir}")
        except ImportError:
            log("figures module not importable; skipping (re-run with --figures)")

    log(f"### done in {summary['wall_clock_s'] / 60.0:.1f} min")
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
