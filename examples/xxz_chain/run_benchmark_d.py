#!/usr/bin/env python3
"""Benchmark D -- 1D Trotterized XXZ chain: scaling, growth law, self-check.

Adapted plan `research/plans/2026-08-31-examples-benchmarks-suite.md` §6 Part A
row **D**: `n = 20..100`, `Jz in {0, 0.5}`, a central `Z` and a weight-2
`Z_c Z_{c+1}`, statevector reference for `n <= 26`, the analytic term-growth law
at `Jz = 0`, and time/memory-vs-`n` for both engines.

Physics
-------
`common.circuits.xxz_chain_trotter(n, steps, Jz, dt)` first-order-Trotterizes

    H = sum_i ( X_i X_{i+1} + Y_i Y_{i+1} + Jz Z_i Z_{i+1} )

as three `pauli_rotation` channels per bond (even bonds, then odd bonds). Two
regimes:

* **`Jz = 0`** -- the XX+YY chain is a *free-fermion* model: Jordan-Wigner maps
  it to hopping Majorana bilinears, and every Trotter gate is Gaussian. A single
  `Z_c = -i g_{2c} g_{2c+1}` therefore stays a sum of Majorana **bilinears** for
  all time, and each bilinear `g_a g_b` is one Pauli string. The number of
  bilinears reachable in `s` steps is (cone width)^2, and the cone widens by a
  fixed number of sites per step -- so the non-zero Pauli-term count grows
  **quadratically in the number of Trotter steps**, until the cone hits the
  chain boundary. The `growth` mode measures the log-log slope instead of
  asserting counts (see `README.md` for the measured numbers).
* **`Jz = 0.5`** -- interacting; the same seed spreads over Majorana strings of
  every even order and the untruncated count grows exponentially (measured: 40,
  9512, 2.45e6 terms at `s = 1, 2, 3`). This is the regime truncation exists
  for, and it has no cheap large-`n` reference: `statevector` covers `n <= 26`
  and `convergence` shows self-convergence at `n = 60, 100`.

Initial state
-------------
A **domain wall** `|0...01...1>` (`state="0"*n//2 + "1"*(n-n//2)`, the A4
per-qubit label form). `|0...0>` and `|+...+>` are useless here: the first is an
eigenstate of `H` at every `Jz`, so every expectation is a constant, and the
second gives `<Z_c> = 0` by symmetry. The domain wall is the standard melting
setup, is a computational basis state (so PauliPropagation.jl can contract it
too -- `benchmarks/julia/README.md` "Known gaps"), and gives an observable that
actually moves.

Modes
-----
    growth       untruncated term count vs Trotter steps; log-log slope fit
    statevector  qiskit-Aer agreement at n <= 26, tight truncation, both Jz
    scaling      time and peak memory vs n, one subprocess per (n, regime) point
    convergence  error-vs-runtime at fixed n vs a statevector reference,
                 plus self-convergence panels at n = 60, 100
    julia        PauliPropagation.jl at matched truncation, per-layer parity
                 gate first (blocking), then warm times
    figures      regenerate every SVG from the committed JSON
    all          every mode above, in that order

Each mode overwrites `results/<mode>.json` (committed, so a rerun's diff *is*
the measurement's reproducibility record) and the figures it owns.

Running
-------
`RAYON_NUM_THREADS=1` must be exported **before** the interpreter starts: Rayon
builds its global pool once, at the first propagate, and never resizes it. The
script refuses to run otherwise.

    RAYON_NUM_THREADS=1 python examples/xxz_chain/run_benchmark_d.py all
    RAYON_NUM_THREADS=1 python examples/xxz_chain/run_benchmark_d.py growth

Why the timed sweeps live here and not in a `pytest-benchmark` file (unlike
benchmarks A/B/E): D's deliverable is a *curve* -- 36 (n, regime, observable)
points plus truncation sweeps -- and `pytest-benchmark`'s per-point calibration
would re-run each point several times to a target precision, which for a sweep
whose points span three orders of magnitude in cost is both slow and pointless.
The memory curve additionally *requires* one process per point, because `VmHWM`
is a process-lifetime high-water mark. Correctness is gated in CI by
`python/paulistrings/tests/test_benchmark_d_xxz.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_BENCH_PY_DIR = _REPO_ROOT / "benchmarks" / "python"
for _p in (_EXAMPLES_DIR, _BENCH_PY_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

# --- Fixed experiment parameters -------------------------------------------

DT = 0.1
#: The two regimes: free (Jordan-Wigner-Gaussian) and interacting.
JZ_FREE = 0.0
JZ_INTERACTING = 0.5
REGIMES = (JZ_FREE, JZ_INTERACTING)

#: Trotter depth of the scaling and convergence sweeps.
SCALING_STEPS = 6
#: Chain lengths of the scaling sweep (plan §6: n = 20..100).
SCALING_N = (20, 30, 40, 50, 60, 70, 80, 90, 100)
#: Matched cutoff for the scaling sweep. Non-dyadic and strictly positive, the
#: two conditions `benchmarks/julia/README.md` §P3/§P9 require of any cutoff
#: used in a cross-engine comparison.
SCALING_EPS = 1e-6

#: Tight cutoff for the statevector agreement panel: small enough to be
#: numerically inert at these depths, large enough to kill jl's exact zeros.
TIGHT_EPS = 1e-12
#: Agreement bar for the statevector panel.
STATEVECTOR_TOL = 1e-9

#: Truncation grid of the error-vs-runtime and convergence panels.
EPS_GRID = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
#: `n` of the error-vs-runtime panel (statevector-referenced).
ERROR_VS_RUNTIME_N = 24
#: `n` of the self-converged panels (no exact reference at this size).
SELF_CONVERGED_N = (60, 100)

DIRECTION = "heisenberg"

#: Measured on ccqlin038 while writing this driver, and the reason the two
#: oracle-using modes run **every engine propagation before the first oracle
#: call**: one `oracles.statevector_expectation` takes the process from 32 to 97
#: threads (qiskit Aer's own OpenMP pool), and `harness.assert_single_threaded`
#: counts threads gained since import -- it cannot tell Aer's pool from Rayon's,
#: so a `threads=1` run after an Aer call fails the pin assertion. Ordering the
#: passes keeps the assertion meaningful instead of weakening it to
#: `threads=None`; Aer's own thread count is irrelevant because the oracle is
#: never timed.
AER_THREADS_NOTE = (
    "qiskit Aer spawns ~2 threads per core on its first run, which "
    "harness.assert_single_threaded (a relative thread-count check) cannot "
    "distinguish from Rayon workers. Every timed paulistrings run in a process "
    "therefore happens before the first statevector call."
)


# --- Small shared helpers ---------------------------------------------------


def domain_wall_state(n: int) -> str:
    """`|0...01...1>` as an A4 per-qubit label string."""
    return "0" * (n // 2) + "1" * (n - n // 2)


def center(n: int) -> int:
    return n // 2


def observable(kind: str, n: int):
    """`"z1"` -> central `Z_c`; `"z2"` -> `Z_c Z_{c+1}` (weight 2)."""
    from common import observables

    c = center(n)
    if kind == "z1":
        return observables.single_z(c, n)
    if kind == "z2":
        return observables.pauli_sum_from_support({c: "Z", c + 1: "Z"}, n)
    raise ValueError(f"unknown observable kind {kind!r}")


OBSERVABLE_LABELS = {"z1": "Z_c", "z2": "Z_c Z_c+1"}


def unsaturated_max_steps(n: int) -> int:
    """Largest Trotter depth whose light cone still fits inside the chain.

    One step's even-then-odd bond sweep moves support by at most two sites in
    each direction, so after `s` steps a seed on `{c, c+1}` lives inside
    `[c - 2s, c + 1 + 2s]`. Requiring both ends inside `[0, n-1]` with
    `c = n // 2` gives `s <= n/4 - 1`. Beyond it the count is boundary-limited
    and must not enter the growth-law fit.
    """
    return max(0, n // 4 - 1)


def loglog_slope(steps: list[int], counts: list[int]) -> float:
    """Least-squares slope of `log(count)` against `log(steps)`."""
    import numpy as np

    x = np.log(np.asarray(steps, dtype=float))
    y = np.log(np.asarray(counts, dtype=float))
    return float(np.polyfit(x, y, 1)[0])


def require_pinned() -> None:
    if os.environ.get("RAYON_NUM_THREADS") != "1":
        raise SystemExit(
            "RAYON_NUM_THREADS must be '1' in the environment before this "
            "interpreter starts (Rayon builds its global pool once, at the first "
            "propagate, and never resizes it).\n"
            f"    RAYON_NUM_THREADS=1 python {Path(__file__).name} <mode>"
        )


def _provenance_dict() -> dict[str, Any]:
    from common import harness, report

    prov = report.collect_provenance(thread_count=1, repo_root=_REPO_ROOT)
    out = asdict(prov)
    out["import_thread_count"] = harness.IMPORT_THREAD_COUNT
    return out


def write_json(name: str, payload: dict[str, Any]) -> Path:
    """Overwrite `results/<name>.json`.

    Deliberately *not* `report.write_results`, which appends: these files are
    committed, so a rerun must reproduce the file rather than grow it.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)
    print(f"wrote {path.relative_to(_REPO_ROOT)}")
    return path


def read_json(name: str) -> dict[str, Any]:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run the '{name}' mode first")
    return json.loads(path.read_text())


def records_of(payload: dict[str, Any]):
    from common.report import RunRecord

    return [RunRecord.from_dict(r) for r in payload["records"]]


def _figure_path(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / name


# ===========================================================================
# 1. Growth law
# ===========================================================================

#: `(n, max_steps)` of the free-regime growth measurement. `n = 40` is included
#: precisely because its cone saturates inside the range, so the figure shows
#: the law *and* its boundary-limited breakdown.
GROWTH_FREE = ((40, 12), (60, 12), (80, 12), (100, 12))
#: The interacting regime is untruncated too, so it can only go a few steps.
GROWTH_INTERACTING = ((40, 3),)
GROWTH_SLOPE_TOL = 0.05


def mode_growth(args) -> dict[str, Any]:
    """Untruncated term count vs Trotter steps; fit the exponent."""
    from common import circuits, harness

    series: list[dict[str, Any]] = []
    for kind in ("z1", "z2"):
        for Jz, grid in ((JZ_FREE, GROWTH_FREE), (JZ_INTERACTING, GROWTH_INTERACTING)):
            for n, max_steps in grid:
                # The weight-2 seed is a Majorana *quartic*, so its untruncated
                # count is the square of the bilinear one -- cap the depth.
                cap = max_steps if kind == "z1" else min(max_steps, 8)
                if kind == "z2" and Jz != JZ_FREE:
                    cap = min(cap, 2)
                obs = observable(kind, n)
                steps_list: list[int] = []
                counts: list[int] = []
                times: list[float] = []
                for s in range(1, cap + 1):
                    circuit = circuits.xxz_chain_trotter(n, s, Jz=Jz, dt=DT)
                    rec = harness.run_propagation(
                        circuit,
                        obs,
                        None,
                        DIRECTION,
                        warmup=False,
                        threads=1,
                        extra={"Jz": Jz, "n": n, "steps": s, "observable": kind},
                    )
                    steps_list.append(s)
                    counts.append(rec.final_terms)
                    times.append(rec.propagation_time_s)
                    print(
                        f"  Jz={Jz} n={n:3d} {OBSERVABLE_LABELS[kind]:10s} "
                        f"steps={s:2d} terms={rec.final_terms:10d} "
                        f"{1e3 * rec.propagation_time_s:8.1f} ms"
                    )
                fit_max = unsaturated_max_steps(n)
                fit_steps = [s for s in steps_list if 2 <= s <= fit_max]
                entry: dict[str, Any] = {
                    "Jz": Jz,
                    "n": n,
                    "observable": kind,
                    "steps": steps_list,
                    "terms": counts,
                    "propagation_time_s": times,
                    "unsaturated_max_steps": fit_max,
                    "fit_steps": fit_steps,
                }
                if len(fit_steps) >= 3:
                    idx = [steps_list.index(s) for s in fit_steps]
                    entry["loglog_slope"] = loglog_slope(
                        fit_steps, [counts[i] for i in idx]
                    )
                series.append(entry)

    free_z1 = [
        e
        for e in series
        if e["Jz"] == JZ_FREE and e["observable"] == "z1" and "loglog_slope" in e
    ]
    slopes = [e["loglog_slope"] for e in free_z1]
    verdict = {
        "claim": "at Jz=0 the untruncated non-zero Pauli-term count of a "
        "weight-1 seed grows quadratically in Trotter steps",
        "measured_loglog_slopes": {str(e["n"]): e["loglog_slope"] for e in free_z1},
        "slope_tolerance": GROWTH_SLOPE_TOL,
        "supported": bool(slopes)
        and all(abs(s - 2.0) <= GROWTH_SLOPE_TOL for s in slopes),
        "exact_identity": "terms(s) == 16 * s**2 for every unsaturated point "
        "measured (independent of n, dt and seed site)",
        "exact_identity_holds": all(
            c == 16 * s * s
            for e in series
            if e["Jz"] == JZ_FREE and e["observable"] == "z1"
            for s, c in zip(e["steps"], e["terms"])
            if s <= e["unsaturated_max_steps"]
        ),
    }
    quartic = [
        e
        for e in series
        if e["Jz"] == JZ_FREE and e["observable"] == "z2" and "loglog_slope" in e
    ]
    if quartic:
        verdict["weight_2_loglog_slopes"] = {
            str(e["n"]): e["loglog_slope"] for e in quartic
        }
        verdict["weight_2_note"] = (
            "the weight-2 seed is a Majorana quartic, so its count is the square "
            "of the bilinear count -- exponent 4 asymptotically, approached from "
            "below at these depths (measured counts are exactly "
            "(8 s^2 + 6 s - 1)^2)"
        )

    payload = {
        "benchmark": "D",
        "mode": "growth",
        "dt": DT,
        "direction": DIRECTION,
        "truncation": "none (untruncated)",
        "verdict": verdict,
        "series": series,
        "provenance": _provenance_dict(),
    }
    write_json("growth", payload)
    print(f"\ngrowth-law verdict: supported={verdict['supported']} slopes={slopes}")
    if not args.no_figures:
        figure_growth(payload)
    return payload


def figure_growth(payload: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.report import _MUTED_TEXT, _PALETTE, _style_axes

    fig, (ax_free, ax_int) = plt.subplots(1, 2, figsize=(9.5, 4))

    free = [
        e
        for e in payload["series"]
        if e["Jz"] == JZ_FREE and e["observable"] == "z1"
    ]
    for i, e in enumerate(sorted(free, key=lambda e: e["n"])):
        color = _PALETTE[i % len(_PALETTE)]
        ax_free.plot(
            e["steps"],
            e["terms"],
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=color,
            label=f"n = {e['n']}",
        )
        sat = e["unsaturated_max_steps"]
        after = [(s, c) for s, c in zip(e["steps"], e["terms"]) if s > sat]
        if after:
            xs, ys = zip(*after)
            ax_free.plot(xs, ys, linestyle="none", marker="x", markersize=6, color=color)
    ref_steps = [s for s in range(1, 13)]
    ax_free.plot(
        ref_steps,
        [16 * s * s for s in ref_steps],
        linestyle="--",
        linewidth=1.2,
        color=_MUTED_TEXT,
        label=r"$16\,s^2$",
    )
    ax_free.set_xscale("log")
    ax_free.set_yscale("log")
    ax_free.set_xlabel("Trotter steps $s$")
    ax_free.set_ylabel("non-zero Pauli terms (untruncated)")
    ax_free.set_title("$J_z = 0$, seed $Z_c$: quadratic", fontsize=10)
    _style_axes(ax_free)
    ax_free.legend(frameon=False, fontsize=8)

    for i, e in enumerate(payload["series"]):
        if e["observable"] != "z1":
            continue
        color = _PALETTE[0] if e["Jz"] == JZ_FREE else _PALETTE[1]
        if e["n"] != 40:
            continue
        ax_int.plot(
            e["steps"],
            e["terms"],
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=color,
            label=f"$J_z = {e['Jz']}$",
        )
    ax_int.set_yscale("log")
    ax_int.set_xlabel("Trotter steps $s$")
    ax_int.set_ylabel("non-zero Pauli terms (untruncated)")
    ax_int.set_title("$n = 40$: free vs interacting", fontsize=10)
    _style_axes(ax_int)
    ax_int.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    path = _figure_path("term-growth.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(_REPO_ROOT)}")


# ===========================================================================
# 2. Statevector agreement
# ===========================================================================

#: `(n, Jz, observable, steps)`. The omissions are cost, not doubt: at `n = 26`
#: an `s = 3` interacting run with a weight-2 seed keeps ~4e7 untruncated terms.
STATEVECTOR_CASES = tuple(
    (n, Jz, kind, s)
    for n in (20, 24, 26)
    for Jz in REGIMES
    for kind in ("z1", "z2")
    for s in (1, 2, 3)
    if not (Jz == JZ_INTERACTING and kind == "z2" and s == 3)
    if not (n == 26 and s == 3)
)


def mode_statevector(args) -> dict[str, Any]:
    """Every engine run first, every Aer call after -- see `AER_THREADS_NOTE`."""
    from common import circuits, harness, oracles

    rows: list[dict[str, Any]] = []
    records = []
    specs = []
    for n, Jz, kind, s in STATEVECTOR_CASES:
        spec = oracles.record_gates(circuits.xxz_chain_trotter, n, s, Jz=Jz, dt=DT)
        obs = observable(kind, n)
        state = domain_wall_state(n)
        rec = harness.run_propagation(
            spec.to_circuit(),
            obs,
            {"min_abs_coeff": TIGHT_EPS},
            DIRECTION,
            state=state,
            warmup=False,
            threads=1,
            extra={"Jz": Jz, "n": n, "steps": s, "observable": kind},
        )
        records.append(rec)
        specs.append((spec, obs, state))
        print(
            f"  engine n={n:2d} Jz={Jz} {OBSERVABLE_LABELS[kind]:10s} s={s} "
            f"value={rec.expectation_value:+.12f} terms={rec.final_terms}"
        )

    worst = 0.0
    for (n, Jz, kind, s), rec, (spec, obs, state) in zip(
        STATEVECTOR_CASES, records, specs
    ):
        exact = oracles.statevector_expectation(spec, obs, state)
        rec.absolute_error = abs(rec.expectation_value - exact.real)
        worst = max(worst, rec.absolute_error)
        rows.append(
            {
                "n": n,
                "Jz": Jz,
                "observable": kind,
                "steps": s,
                "statevector": exact.real,
                "paulistrings": rec.expectation_value,
                "absolute_error": rec.absolute_error,
                "final_terms": rec.final_terms,
            }
        )
        print(
            f"  n={n:2d} Jz={Jz} {OBSERVABLE_LABELS[kind]:10s} s={s} "
            f"exact={exact.real:+.12f} pp={rec.expectation_value:+.12f} "
            f"|d|={rec.absolute_error:.2e} terms={rec.final_terms}"
        )

    payload = {
        "benchmark": "D",
        "mode": "statevector",
        "oracle": "examples.common.oracles.statevector_expectation (qiskit Aer)",
        "initial_state": "domain wall |0...01...1>",
        "truncation": {"min_abs_coeff": TIGHT_EPS},
        "thread_note": AER_THREADS_NOTE,
        "tolerance": STATEVECTOR_TOL,
        "max_absolute_error": worst,
        "passed": worst <= STATEVECTOR_TOL,
        "rows": rows,
        "records": [r.to_dict() for r in records],
        "provenance": _provenance_dict(),
    }
    write_json("statevector", payload)
    print(f"\nworst |error| = {worst:.3e} (bar {STATEVECTOR_TOL:g})")
    if worst > STATEVECTOR_TOL:
        raise SystemExit("statevector agreement FAILED")
    return payload


# ===========================================================================
# 3. Scaling: time and peak memory vs n, one subprocess per point
# ===========================================================================


def mode_scaling_point(args) -> dict[str, Any]:
    """One (n, Jz, observable) point. Prints one JSON line on stdout.

    Run in its own process by `mode_scaling` so `VmHWM` -- a process-lifetime
    high-water mark -- is this point's own peak and nothing else's.
    """
    from common import circuits, harness

    n, Jz, kind = args.n, args.jz, args.observable
    circuit = circuits.xxz_chain_trotter(n, SCALING_STEPS, Jz=Jz, dt=DT)
    obs = observable(kind, n)
    rec = harness.run_propagation(
        circuit,
        obs,
        {"min_abs_coeff": SCALING_EPS},
        DIRECTION,
        state=domain_wall_state(n),
        warmup=True,
        threads=1,
        extra={
            "Jz": Jz,
            "n": n,
            "steps": SCALING_STEPS,
            "observable": kind,
            "channels": len(circuit),
        },
    )
    print("JSON " + json.dumps(rec.to_dict(), default=str))
    return rec.to_dict()


def mode_scaling(args) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = "1"
    env.pop("RUST_LOG", None)
    for kind in ("z1", "z2"):
        for Jz in REGIMES:
            for n in SCALING_N:
                cmd = [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "scaling-point",
                    "--n",
                    str(n),
                    "--jz",
                    repr(Jz),
                    "--observable",
                    kind,
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=1800,
                    check=False,
                )
                if proc.returncode != 0:
                    raise SystemExit(
                        f"scaling point n={n} Jz={Jz} {kind} failed:\n{proc.stderr}"
                    )
                line = next(
                    ln for ln in reversed(proc.stdout.splitlines())
                    if ln.startswith("JSON ")
                )
                rec = json.loads(line[len("JSON "):])
                records.append(rec)
                print(
                    f"  n={n:3d} Jz={Jz} {OBSERVABLE_LABELS[kind]:10s} "
                    f"prop={rec['propagation_time_s']:8.4f} s "
                    f"terms={rec['final_terms']:8d} "
                    f"peak_delta={rec['extra'].get('peak_memory_kb_delta', 0):9.0f} KiB "
                    f"hwm={rec['peak_memory_kb']:9.0f} KiB"
                )

    payload = {
        "benchmark": "D",
        "mode": "scaling",
        "steps": SCALING_STEPS,
        "truncation": {"min_abs_coeff": SCALING_EPS},
        "initial_state": "domain wall |0...01...1>",
        "one_process_per_point": True,
        "memory_note": "peak_memory_kb is VmHWM (process lifetime, includes the "
        "interpreter + numpy baseline); extra.peak_memory_kb_delta is the growth "
        "caused by this run alone and is what the figure plots.",
        "records": records,
        "provenance": _provenance_dict(),
    }
    write_json("scaling", payload)
    if not args.no_figures:
        figure_scaling(payload)
    return payload


def figure_scaling(payload: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.report import _PALETTE, _style_axes

    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(9.5, 4))
    series: dict[tuple[float, str], list[tuple[int, float, float]]] = {}
    for rec in payload["records"]:
        key = (rec["extra"]["Jz"], rec["extra"]["observable"])
        series.setdefault(key, []).append(
            (
                rec["extra"]["n"],
                rec["propagation_time_s"] + (rec["contraction_time_s"] or 0.0),
                float(rec["extra"].get("peak_memory_kb_delta") or 0.0),
            )
        )
    for i, (key, points) in enumerate(sorted(series.items())):
        Jz, kind = key
        points.sort()
        ns = [p[0] for p in points]
        color = _PALETTE[i % len(_PALETTE)]
        label = f"$J_z={Jz}$, {OBSERVABLE_LABELS[kind]}"
        ax_t.plot(
            ns, [p[1] for p in points], marker="o", markersize=4, linewidth=1.4,
            color=color, label=label,
        )
        ax_m.plot(
            ns, [max(p[2], 1.0) for p in points], marker="o", markersize=4,
            linewidth=1.4, color=color, label=label,
        )

    for ax, ylabel, title in (
        (ax_t, "wall time (s)", "propagation + contraction"),
        (ax_m, "peak memory growth (KiB)", "VmHWM delta, one process per point"),
    ):
        ax.set_yscale("log")
        ax.set_xlabel("chain length $n$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"XXZ chain, {payload['steps']} Trotter steps, "
        f"min_abs_coeff = {payload['truncation']['min_abs_coeff']:g}",
        fontsize=10,
    )
    fig.tight_layout()
    path = _figure_path("time-memory-vs-n.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(_REPO_ROOT)}")


# ===========================================================================
# 4. Convergence: error vs runtime, and self-converged large-n panels
# ===========================================================================


def mode_convergence(args) -> dict[str, Any]:
    from common import circuits, harness, oracles

    records = []
    panels: list[dict[str, Any]] = []

    # (a) error vs runtime at fixed n, against the exact statevector value.
    # Timed runs first, Aer afterwards -- see AER_THREADS_NOTE.
    n = ERROR_VS_RUNTIME_N
    state = domain_wall_state(n)
    exact_panels: list[tuple[float, list, Any, Any]] = []
    for Jz in REGIMES:
        spec = oracles.record_gates(
            circuits.xxz_chain_trotter, n, SCALING_STEPS, Jz=Jz, dt=DT
        )
        obs = observable("z1", n)
        circuit = spec.to_circuit()
        panel_records = []
        for eps in EPS_GRID:
            rec = harness.run_propagation(
                circuit,
                obs,
                {"min_abs_coeff": eps},
                DIRECTION,
                state=state,
                warmup=False,
                threads=1,
                extra={"Jz": Jz, "n": n, "steps": SCALING_STEPS, "panel": "exact"},
            )
            panel_records.append(rec)
            print(
                f"  n={n} Jz={Jz} eps={eps:g} value={rec.expectation_value:+.12f} "
                f"terms={rec.final_terms:8d} {rec.propagation_time_s:.4f} s"
            )
        exact_panels.append((Jz, panel_records, spec, obs))

    # (b) self-convergence at sizes with no exact reference.
    for n_big in SELF_CONVERGED_N:
        state_big = domain_wall_state(n_big)
        for Jz in (JZ_INTERACTING,):
            circuit = circuits.xxz_chain_trotter(n_big, SCALING_STEPS, Jz=Jz, dt=DT)
            obs = observable("z1", n_big)
            panel_records = []
            for eps in EPS_GRID:
                rec = harness.run_propagation(
                    circuit,
                    obs,
                    {"min_abs_coeff": eps},
                    DIRECTION,
                    state=state_big,
                    warmup=False,
                    threads=1,
                    extra={
                        "Jz": Jz,
                        "n": n_big,
                        "steps": SCALING_STEPS,
                        "panel": "self-converged",
                    },
                )
                panel_records.append(rec)
                print(
                    f"  n={n_big} Jz={Jz} eps={eps:g} "
                    f"value={rec.expectation_value:+.12f} "
                    f"terms={rec.final_terms:8d} {rec.propagation_time_s:.4f} s"
                )
            tightest = panel_records[-1].expectation_value
            for rec in panel_records:
                rec.absolute_error = abs(rec.expectation_value - tightest)
            records.extend(panel_records)
            panels.append(
                {
                    "panel": "self-converged",
                    "n": n_big,
                    "Jz": Jz,
                    "reference": tightest,
                    "reference_kind": (
                        f"self-converged: the tightest point of this sweep "
                        f"(min_abs_coeff = {EPS_GRID[-1]:g}); NOT an independent oracle"
                    ),
                    "eps_grid": list(EPS_GRID),
                    "spread_over_last_three_points": max(
                        abs(r.expectation_value - tightest) for r in panel_records[-3:]
                    ),
                }
            )

    # Now the oracle calls (Aer's thread pool no longer matters).
    for Jz, panel_records, spec, obs in exact_panels:
        exact = oracles.statevector_expectation(spec, obs, state).real
        for rec in panel_records:
            rec.absolute_error = abs(rec.expectation_value - exact)
            print(
                f"  n={n} Jz={Jz} eps={rec.truncation['min_abs_coeff']:g} "
                f"|err|={rec.absolute_error:.3e} "
                f"terms={rec.final_terms:8d} {rec.propagation_time_s:.4f} s"
            )
        records.extend(panel_records)
        panels.append(
            {
                "panel": "exact",
                "n": n,
                "Jz": Jz,
                "reference": exact,
                "reference_kind": "statevector (qiskit Aer)",
                "eps_grid": list(EPS_GRID),
            }
        )

    payload = {
        "benchmark": "D",
        "mode": "convergence",
        "steps": SCALING_STEPS,
        "thread_note": AER_THREADS_NOTE,
        "initial_state": "domain wall |0...01...1>",
        "observable": "z1",
        "panels": panels,
        "records": [r.to_dict() for r in records],
        "provenance": _provenance_dict(),
    }
    write_json("convergence", payload)
    if not args.no_figures:
        figure_convergence(payload)
    return payload


def figure_convergence(payload: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import report
    from common.report import _style_axes

    records = records_of(payload)
    exact_panels = [p for p in payload["panels"] if p["panel"] == "exact"]
    self_panels = [p for p in payload["panels"] if p["panel"] == "self-converged"]

    # error vs runtime, one curve per regime (engine field is the series key the
    # report helper groups on, so relabel a copy rather than the stored records)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for i, p in enumerate(exact_panels):
        recs = [
            r
            for r in records
            if r.extra.get("panel") == "exact" and r.extra.get("Jz") == p["Jz"]
        ]
        for r in recs:
            r.engine = f"$J_z = {p['Jz']}$"
        report.plot_error_vs_runtime(recs, ax=ax)
    ax.set_title(
        f"error vs runtime, n = {exact_panels[0]['n']}, "
        f"{payload['steps']} steps (statevector reference)",
        fontsize=10,
    )
    _style_axes(ax)
    path = _figure_path("error-vs-runtime.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(_REPO_ROOT)}")

    # self-convergence panels, one subplot per size
    fig, axes = plt.subplots(1, max(1, len(self_panels)), figsize=(4.8 * len(self_panels), 4))
    if len(self_panels) == 1:
        axes = [axes]
    for ax, p in zip(axes, self_panels):
        recs = [
            r
            for r in records
            if r.extra.get("panel") == "self-converged"
            and r.extra.get("n") == p["n"]
        ]
        for r in recs:
            r.engine = "paulistrings"
        report.plot_convergence_panel(recs, reference_value=p["reference"], ax=ax)
        ax.set_title(
            f"n = {p['n']}, $J_z = {p['Jz']}$ (self-converged)", fontsize=10
        )
    fig.tight_layout()
    path = _figure_path("self-convergence.svg")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(_REPO_ROOT)}")


# ===========================================================================
# 5. PauliPropagation.jl comparison
# ===========================================================================

#: `(n, Jz, steps, min_abs_coeff)`. Cutoffs are non-dyadic and strictly
#: positive: `benchmarks/julia/README.md` §P3 (jl keeps `|c| == eps`, this
#: engine drops it) and §P9 (jl keeps exact zeros) are both measured
#: divergences, and both are avoided rather than fudged.
#: The last row is deliberately the largest one this pair of engines can be
#: compared on cheaply: the cross-engine ranking *changes sign* with the size of
#: the tracked set (see `README.md` §5), so a comparison reported at one size
#: only would be misleading in whichever direction that size happened to favour.
JULIA_CASES = (
    (40, JZ_FREE, 6, 1e-6),
    (20, JZ_INTERACTING, 3, 1e-5),
    (40, JZ_INTERACTING, 4, 1e-6),
    (40, JZ_INTERACTING, 6, 1e-6),
)


def mode_julia(args) -> dict[str, Any]:
    import julia_baseline as jb

    reason = jb.skip_reason()
    if reason is not None:
        print(f"SKIP: {reason}")
        return {"skipped": reason}

    from common import circuits, harness, observables, oracles
    from test_julia_parity import run_rust

    rows: list[dict[str, Any]] = []
    for n, Jz, steps, eps in JULIA_CASES:
        spec = oracles.record_gates(circuits.xxz_chain_trotter, n, steps, Jz=Jz, dt=DT)
        obs = observable("z1", n)
        label = observables.pauli_string({center(n): "Z"}, n)
        state = domain_wall_state(n)
        task = jb.make_task(
            n_qubits=n,
            gates=spec.to_circuit_json()["gates"],
            observable={label: 1.0},
            direction=DIRECTION,
            min_abs_coeff=eps,
            threads=1,
            state=state,
        )

        # --- blocking parity gate, untimed on both sides ------------------
        rust = run_rust(task)
        jl_counts = jb.run_task(task, warm_repeats=0, layer_counts=True)
        problems: list[str] = []
        if rust["final_terms"] != jl_counts.final_terms:
            problems.append(
                f"final terms rust={rust['final_terms']} jl={jl_counts.final_terms}"
            )
        jl_layers = jl_counts.per_layer_terms
        if jl_layers is None:
            problems.append("jl reported no per-layer counts")
        elif len(jl_layers) != len(rust["per_layer_terms"]):
            problems.append(
                f"layer count rust={len(rust['per_layer_terms'])} jl={len(jl_layers)}"
            )
        else:
            bad = [
                (i, a, b)
                for i, (a, b) in enumerate(zip(rust["per_layer_terms"], jl_layers))
                if a != b
            ]
            if bad:
                problems.append(
                    f"{len(bad)}/{len(jl_layers)} per-layer counts differ, first "
                    f"{bad[:5]}"
                )
        d_exp = None
        if rust["expectation"] is not None and jl_counts.expectation is not None:
            d_exp = abs(rust["expectation"] - jl_counts.expectation)
            if d_exp > 1e-12:
                problems.append(f"expectation |delta| = {d_exp:.3e} > 1e-12")

        row: dict[str, Any] = {
            "n": n,
            "Jz": Jz,
            "steps": steps,
            "min_abs_coeff": eps,
            "channels": len(spec),
            "layers_compared": len(jl_layers or []),
            "final_terms_paulistrings": rust["final_terms"],
            "final_terms_julia": jl_counts.final_terms,
            "expectation_paulistrings": rust["expectation"].real
            if rust["expectation"] is not None
            else None,
            "expectation_julia": jl_counts.expectation.real
            if jl_counts.expectation is not None
            else None,
            "expectation_abs_delta": d_exp,
            "parity_ok": not problems,
            "parity_problems": problems,
            "julia_versions": jl_counts.versions,
        }

        if problems:
            print(f"  PARITY FAILURE n={n} Jz={Jz}: {problems}")
            row["timing"] = "withheld: parity blocks timing (plan §7 rule 2)"
        else:
            rec = harness.run_propagation(
                spec.to_circuit(),
                obs,
                {"min_abs_coeff": eps},
                DIRECTION,
                state=state,
                warmup=True,
                threads=1,
                extra={"Jz": Jz, "n": n, "steps": steps},
            )
            jl_timed = jb.run_task(task, warm_repeats=3, layer_counts=False)
            row["paulistrings_propagation_s"] = rec.propagation_time_s
            row["paulistrings_total_s"] = rec.total_time_s
            row["julia_wall_warm_s"] = jl_timed.wall_warm_s
            row["julia_wall_cold_s"] = jl_timed.wall_cold_s
            row["speedup_vs_julia_warm"] = (
                jl_timed.wall_warm_s / rec.propagation_time_s
                if jl_timed.wall_warm_s
                else None
            )
            print(
                f"  n={n:3d} Jz={Jz} s={steps} eps={eps:g}: parity OK on "
                f"{row['layers_compared']} layers, terms={rust['final_terms']}, "
                f"|dE|={d_exp:.2e}; rust={rec.propagation_time_s:.4f}s "
                f"jl_warm={jl_timed.wall_warm_s:.4f}s "
                f"({row['speedup_vs_julia_warm']:.2f}x)"
            )
        rows.append(row)

    payload = {
        "benchmark": "D",
        "mode": "julia",
        "note": "cutoffs are non-dyadic and strictly positive; see "
        "benchmarks/julia/README.md §P3 and §P9",
        "parity_gate": "per-layer term counts must match index-by-index before "
        "any timing is recorded (plan §7 rule 2)",
        "rows": rows,
        "provenance": _provenance_dict(),
    }
    write_json("julia", payload)
    return payload


# ===========================================================================
# figures / all
# ===========================================================================


def mode_figures(args) -> dict[str, Any]:
    figure_growth(read_json("growth"))
    figure_scaling(read_json("scaling"))
    figure_convergence(read_json("convergence"))
    return {}


def mode_all(args) -> dict[str, Any]:
    for fn in (mode_growth, mode_statevector, mode_scaling, mode_convergence, mode_julia):
        print(f"\n=== {fn.__name__.removeprefix('mode_')} ===")
        fn(args)
    return {}


MODES = {
    "growth": mode_growth,
    "statevector": mode_statevector,
    "scaling": mode_scaling,
    "scaling-point": mode_scaling_point,
    "convergence": mode_convergence,
    "julia": mode_julia,
    "figures": mode_figures,
    "all": mode_all,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--n", type=int, help="scaling-point only")
    parser.add_argument("--jz", type=float, help="scaling-point only")
    parser.add_argument("--observable", choices=("z1", "z2"), help="scaling-point only")
    parser.add_argument(
        "--no-figures", action="store_true", help="skip SVG regeneration"
    )
    args = parser.parse_args(argv)
    if args.mode != "figures":
        require_pinned()
    MODES[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
