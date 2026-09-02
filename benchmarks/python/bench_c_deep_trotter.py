"""Benchmark C — deep Trotter, time-to-fixed-accuracy.

The headline entry of the suite. Heavy-hex kicked Ising, `n = 127`,
`theta_zz = -pi/2`, up to **20 Trotter steps**, kick angles in the hard interior
`theta_h in {7pi/32, 5pi/16}` (`0.6872`, `0.9817` — both on the published
`pi/32` grid, see `PUBLISHED_ANCHOR`), observable `Z_62` — the marquee point of
Kim et al. (2023) Fig. 4b is `Z_62` at 20 steps. For each
`(theta_h, trotter_steps)` the driver

1. establishes a **self-converged reference** with documented convergence
   evidence, reusing Benchmark B's plateau test verbatim (`_plateau_is_real`,
   imported, see §Reference strategy),
2. sweeps a **dyadic** `min_abs_coeff` grid loosest-to-tightest and
   records one `report.RunRecord` per grid point (warm, single-threaded, quiet
   logging — all enforced by `harness.run_propagation`),
3. checks the tracked-set size against the sanity envelope of 1.2e6-9.3e6
   unique strings at the sweep's cutoffs *before* any timing is reported,
4. optionally runs the same task through PauliPropagation.jl — behind a
   **memory gate**, because jl's dict backend needed 67.6 GiB at 2.85e6 terms in
   Benchmark B — and checks **per-layer** term-count parity at matched
   truncation before any cross-engine number is written: this is a blocking
   gate, and
5. writes the records as JSON and the figures as SVG.

This module is a driver, not a pytest module: it defines no `test_*` function,
so `pytest benchmarks/python` imports it and collects nothing. The CI-safe
correctness gate for the same physics lives in
`python/paulistrings/tests/test_benchmark_c_deep.py`, on a 20-qubit heavy-hex
sublattice where a dense statevector covers every point at the full 20 steps.

Reference strategy
------------------

There is **no exact reference at 20 steps**, and not for want of trying: the
commutation-aware backward cone of `Z_62` after 20 kicked-Ising steps is the
whole 127-qubit lattice, so the cone reduction that gives Benchmark B its exact
`Z_62` and weight-10 references (19 and 30 qubits at 5 steps) buys nothing here.
Dense simulation is out at 127 qubits and untruncated Pauli propagation fans out
without bound. Nor is this repo alone in that: the published exact-benchmark file
for these circuits covers the 5-step observables and `<Z_62>` at 9 steps and has
**no column for `<Z_62>` at 20 steps** (`PUBLISHED_ANCHOR`). So the 20-step
reference is **self-converged**, exactly as plan D5 specifies, and it is labelled
`self_converged` — never "exact" — in every record. At 5 steps, where the cone is
19 qubits, the reference *is* exact.

The self-convergence machinery is **imported from Benchmark B**
(`bench_b_theta_sweep.self_converged_reference` / `._plateau_is_real`), not
re-implemented. That is deliberate: B measured that the obvious criterion
("two successive values agree") is **wrong**, declaring convergence with an
estimated uncertainty of exactly zero while the value was still 5.6e-7 from the
truth, because at a small kick angle the expectation can sit bit-identical
across four decades of cutoff while the sum keeps growing. `_plateau_is_real`
requires the two small differences **and** either a saturated term count or two
strictly-nonzero differences, and rejects a zero-term sum outright. Re-typing
that logic here would risk regressing it; `test_benchmark_c_deep.py` asserts
this module uses B's function object.

Two things *are* retuned for C, and both are recorded in `summary.json`:

`SELF_CONVERGENCE_TOL`
    `1e-3`, against B's `1e-5`. The accuracy bar here is **0.01**, so
    a plateau resolved to 1e-3 leaves a 10x margin; B's 1e-5 is unreachable at
    20 steps at any affordable cutoff (the measured 2^-12 -> 2^-14 difference at
    `theta_h = 0.7`, 20 steps is 1.0e-2, and each further dyadic step costs
    ~15x).

`SELF_CONVERGENCE_GRID`
    Dyadic, and extended **two powers past the tightest timed grid point**, so
    that the error of every timed run is measured against something tighter than
    itself. Where the extension is unaffordable the reference is reported with
    `converged=False` and its own uncertainty, and the resulting circularity is
    stated in the README rather than papered over.

The dyadic cutoffs, and the one-ulp mitigation
----------------------------------------------

This benchmark fixes `min_abs_coeff in {2**-14, 2**-16, 2**-18}`. Those are exact
dyadics, which is the one case where this engine and PauliPropagation.jl
provably disagree: this repo drops `|c| <= eps`, jl keeps `|c| == eps`
(`benchmarks/julia/README.md` §P3), and at a Clifford `theta_zz` the
coefficients are exact dyadics too, so an exact straddle is *not* a measure-zero
event. The documented mitigation is to perturb the threshold by one ulp on one
side and report it — never to touch a coefficient:

* **paulistrings runs use the dyadic verbatim.** `2**-14` is `2**-14`.
* **jl runs get `math.nextafter(eps, inf)`.** jl drops `|c| < eps'`, and there is
  no float strictly between `eps` and `eps'`, so `|c| < eps'` is exactly
  `|c| <= eps` — jl's rule becomes this engine's rule, bit for bit.

`julia_min_abs_coeff` does that conversion and `summary.json` records both
numbers for every parity case.

Memory, and why a jl leg may be skipped
---------------------------------------

Benchmark B measured PauliPropagation.jl's dict backend at **67.6 GiB** RSS on a
2.85e6-term sum, against ~1.2 GiB for this engine's bucketed columns on the same
sum. Benchmark C's envelope starts where B's heaviest case ended, so a jl leg at
a tight cutoff can exceed the box: the driver therefore

1. runs a **pilot** jl leg at a loose cutoff and measures the child's peak RSS
   (`resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`, which is the max over
   reaped children — honest, and it needs no cooperation from the runner),
2. divides by the peak term count to get bytes/term, and
3. runs a tighter leg only when `bytes_per_term * peak_terms` fits inside
   `JULIA_MEMORY_HEADROOM * MemAvailable`.

A leg that does not fit is **skipped and reported with its projection**. That
asymmetry is a measurement, not a gap in the report.

The Aer thread trap
-------------------

Nothing in this driver calls a dense oracle, so the `qiskit-aer` OpenMP pool
that Benchmark B has to isolate in a child process never appears here. The
reference *is* still computed in a spawned child, for a different reason: a
reference is an oracle, not a timing measurement, so it may use
`REFERENCE_THREADS` Rayon workers to reach a tighter cutoff, and Rayon builds
its global pool once and never resizes it. A `spawn` child gets a fresh process
with the overridden variable; the parent's pinned pool is untouched.

Usage
-----

::

    RAYON_NUM_THREADS=1 python benchmarks/python/bench_c_deep_trotter.py --help
    RAYON_NUM_THREADS=1 python benchmarks/python/bench_c_deep_trotter.py        # full run
    RAYON_NUM_THREADS=1 python benchmarks/python/bench_c_deep_trotter.py \
        --thetas 7pi/32 --steps 5 --no-julia --out-dir /tmp/pilot             # quick look

`RAYON_NUM_THREADS=1` must be in the environment **before** the interpreter
starts: Rayon builds its global pool at the first propagate and never resizes it
(`harness` module docstring). The driver refuses to run otherwise.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))
# `julia_baseline` and `bench_b_theta_sweep` are this file's siblings; the driver
# may be run from anywhere.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import bench_b_theta_sweep as bench_b  # noqa: E402
from common import circuits, harness, observables, oracles, report  # noqa: E402

# --------------------------------------------------------------------------
# The benchmark's fixed parameters
# --------------------------------------------------------------------------

N_QUBITS = 127
STATE = "z+"
DIRECTION = "heisenberg"

#: The observable. `Z_62` is the weight-1 operator of Kim et al. (2023) Fig. 4b,
#: whose 20-step point is the marquee number of the utility experiment. Loaded
#: through the provenance-tagged data file like every published support in this
#: suite (`examples/common/observables.py`).
OBSERVABLE_NAME = "z62"
OBSERVABLE_BUILDERS: dict[str, Callable[[], Any]] = {
    "z62": observables.canonical_z_127,
}

#: Kick angles, in the hard interior `theta_h ~ 0.6-1.0`. Two values,
#: not three: the third would have cost a full grid plus a reference (~40 min
#: measured at these depths) and told the same story — see the README's cuts
#: section.
#:
#: **Both sit on the published `pi/32` grid** (`7*pi/32 = 0.687223...` and
#: `10*pi/32 = 5*pi/16 = 0.981748...`), which is a deliberate choice over the
#: round 0.7 / 1.0 sometimes used as example angles. Begušić, Gray & Chan
#: (arXiv:2308.05077) publish their exact kicked-Ising benchmarks on exactly that
#: grid, so choosing grid points makes every number here directly comparable to
#: the literature *by construction* — see `PUBLISHED_ANCHOR`. `7pi/32` sits just
#: under `pi/4`, where Benchmark B's `Z_62` sweep is at its most expensive;
#: `5pi/16` is deeper into the mixing regime.
THETA_POINTS: tuple[tuple[str, float], ...] = (
    ("7pi/32", 7 * math.pi / 32),
    ("5pi/16", 5 * math.pi / 16),
)

#: The depth ladder. 20 is the headline depth; the shallower rungs are what
#: make "deep" a measurement rather than an adjective — the same observable, the
#: same angles, the same cutoffs, four depths.
#:
#: **9, not 10.** The published exact benchmark for `<Z_62>` is at 9 Trotter
#: steps (`PUBLISHED_ANCHOR`), and 9 vs 10 makes no difference to anything else
#: here, so the rung is placed where an external anchor exists.
STEP_POINTS: tuple[int, ...] = (5, 9, 15, 20)

#: The truncation grid for this benchmark: `min_abs_coeff in {2^-14, 2^-16, 2^-18}`.
PLAN_COEFF_GRID: tuple[float, ...] = (2.0**-14, 2.0**-16, 2.0**-18)

#: The grid the driver actually sweeps, **loosest first** — the ordering contract
#: `harness.time_to_accuracy` documents for its `first` selection. Three looser
#: dyadics are prepended to `PLAN_COEFF_GRID`'s three: "time to fixed accuracy" asks for
#: the *cheapest* truncation that reaches the bar, and at 20 steps that answer
#: can be looser than `2^-14`, which a grid starting at `2^-14` could never
#: report. Same dyadic family, so the one-ulp story below is unchanged.
COEFF_GRID: tuple[float, ...] = (
    2.0**-8, 2.0**-10, 2.0**-12, *PLAN_COEFF_GRID,
)

#: Accuracy bar for `harness.time_to_accuracy` — the stated target for
#: this benchmark ("Reproduces reference within 0.01").
ACCURACY_EPSILON = 0.01

#: Where a published exact `<Z_62>` benchmark lives, what its columns mean, and
#: **why nothing was checked in**. Recorded here rather than as a data file
#: because `examples/data/references/README.md`'s rule is explicit: "the header
#: is the citation, so it must be written from the fetch, not from memory."
#:
#: What was established (2026-08-31):
#:
#: * The file is reachable: `https://raw.githubusercontent.com/tbegusic/
#:   arxiv-2308.05077-data/main/exact.csv` (repo `tbegusic/arxiv-2308.05077-data`,
#:   also Zenodo doi 10.5281/zenodo.10223349), for Begušić, Gray & Chan, "Fast
#:   and converged classical simulations of evidence for the utility of quantum
#:   computing before fault tolerance" (arXiv:2308.05077). Its README calls it
#:   "Exact benchmarks for Figs 4a-4d, 5a"; its header row is
#:   `theta_h,4a,4b,4c,4d,5a` and it carries 16 data rows on a `k*pi/32` grid.
#: * From that paper's figure captions: Fig. 4a-d are the 5-step observables
#:   (magnetization, weight-10, weight-17, weight-17-modified) and **Fig. 5a is
#:   `<Z_62>` after 9 steps, Fig. 5b after 20 steps**.
#: * `exact.csv` has a `5a` column and **no `5b` column**. So the paper that
#:   introduced the "<0.01 absolute accuracy" bar this benchmark is scored
#:   against publishes *no exact 20-step value either* — independent
#:   corroboration that C's 20-step reference has to be
#:   self-converged, and of `CONE_SIZES` below.
#: * A rendering of column `4b` reproduced this repo's own independent exact
#:   weight-10 references (`benchmarks/python/theta_sweep/README.md` §3.1) to 12
#:   significant figures at `theta_h = pi/8, pi/4, 3pi/8`, and the Clifford
#:   endpoints of `4b`/`4c`/`4d` reproduced the exact `0`/`+1`/`-1` integers.
#:   That is strong evidence the upstream file uses this suite's conventions
#:   (lattice, `theta_zz = -pi/2`, layer order, Hermitian `Y`, `|0...0>`).
#:
#: Why no file: the only egress available here was a summarizing fetch, not a
#: byte-exact one (`curl`/`wget` are blocked in this environment), and a
#: reference file transcribed through a summarizer is not a citation. The
#: actionable follow-up is therefore one step: fetch the file byte-exactly, drop
#: it in as `examples/data/references/begusic2023_exact.csv` with the header that
#: directory's README specifies, and compare column `5a` against this
#: benchmark's own 9-step rung — which is why the rung is at 9 steps and both
#: angles are on the `pi/32` grid.
PUBLISHED_ANCHOR = {
    "paper": (
        "T. Begušić, J. Gray, G. K.-L. Chan, 'Fast and converged classical "
        "simulations of evidence for the utility of quantum computing before fault "
        "tolerance', arXiv:2308.05077"
    ),
    "data_url": (
        "https://raw.githubusercontent.com/tbegusic/arxiv-2308.05077-data/main/exact.csv"
    ),
    "doi": "10.5281/zenodo.10223349",
    "header_row": "theta_h,4a,4b,4c,4d,5a",
    "theta_grid": "k*pi/32, k = 0..16",
    "columns": {
        "4a": "magnetization M_Z, 5 Trotter steps",
        "4b": "weight-10 operator, 5 Trotter steps",
        "4c": "weight-17 operator, 5 Trotter steps",
        "4d": "weight-17 modified, 5 steps + one extra single-qubit layer",
        "5a": "<Z_62>, 9 Trotter steps  <-- this benchmark's 9-step rung",
        "5b": "ABSENT from exact.csv (<Z_62> at 20 steps has no exact benchmark)",
    },
    "checked_in": False,
    "why_not_checked_in": (
        "only a summarizing fetch was available in this environment (curl/wget "
        "blocked); a reference file transcribed through a summarizer is not a "
        "citation, and examples/data/references/README.md requires the header to be "
        "written from the fetch"
    ),
    "conventions_cross_check": (
        "a rendering of column 4b matched this repo's independent exact weight-10 "
        "references to 12 significant figures at theta_h = pi/8, pi/4, 3pi/8, and the "
        "Clifford endpoints of 4b/4c/4d matched the exact 0/+1/-1 integers"
    ),
}

#: The sanity envelope on the tracked set: at these cutoffs the
#: number of unique Pauli strings should land in this range. A count *below* it
#: is the semantics red flag the check exists for (a wrong direction, a
#: commuting-seed no-op, an emptied sum); a count above it at a *tighter* cutoff
#: is the expected direction and is reported as cost, not as a fault.
TERM_COUNT_ENVELOPE = (1_200_000, 9_300_000)

#: A run whose looser neighbour already took longer than this is timed without
#: the extra untimed warm-up pass (which would double its cost). Recorded per
#: record as `extra["warm"]`. Same rule and same constant as Benchmark B.
WARMUP_TIME_BUDGET_S = 3.0


# --------------------------------------------------------------------------
# Recorded cuts (pilot, project, then shrink and record)
# --------------------------------------------------------------------------

#: Measured pilot on ccqlin038, `Z_62`, `n = 127`, single-threaded unless marked,
#: one gate per channel. `benchmarks/python/deep_trotter/README.md` §5 has the
#: full table; this is the part the cuts below are derived from.
#: The pilot was run at the round angles 0.7 and 1.0 before the grid-aligned
#: 7pi/32 = 0.6872 and 5pi/16 = 0.9817 were adopted (see THETA_POINTS). The gap
#: is 2% in theta_h and the cost projections it feeds are order-of-magnitude, so
#: it was not re-measured; the rows are labelled with the angle actually used.
_MEASURED_PILOT = """\
theta_h  steps  cutoff   <Z_62>        final terms   peak terms   wall
0.7      5      2^-14    +0.639626084      389 804      389 804     1.0 s
0.7      5      2^-16    +0.638708946    1 011 254    1 011 254     1.2 s
0.7      5      2^-18    +0.638708946    2 079 048    2 079 048     2.3 s
0.7      10     2^-14    +0.487348444    2 235 674    2 582 120      64 s
0.7      20     2^-8     +0.493356018          372        2 012    0.31 s
0.7      20     2^-10    +0.380652188        7 787       19 336     1.3 s
0.7      20     2^-12    +0.238520762      133 109      219 016      16 s
0.7      20     2^-14    +0.228420592    2 399 125    3 237 089     246 s
0.7      20     2^-14    +0.228420592    2 399 125    3 237 089      22 s  [32 threads]
0.7      20     2^-16    +0.368476045   38 840 616   47 644 820     404 s  [32 threads]
1.0      5      2^-14    +0.215511154    1 543 616    1 543 616     1.7 s
1.0      5      2^-16    +0.215535472    2 072 871    2 072 871     2.3 s
1.0      5      2^-18    +0.215535472    2 146 412    2 146 412     2.5 s
1.0      10     2^-14    +0.081905150    1 437 964   15 288 166     256 s
1.0      20     2^-8     +0.000000000            6        6 625    0.20 s
1.0      20     2^-10    +0.009333665           84       82 868     1.5 s
1.0      20     2^-12    +0.010138239        1 570    1 112 920      30 s
1.0      20     2^-14    +0.010388188       20 140   15 288 166     355 s

At 5 steps the sum *saturates*: 2^-16 and 2^-18 give the same value to 1e-15,
which is also the exact light-cone answer, so the full grid is affordable
there (2.5 s at the tightest point) and is kept.

At 20 steps it does not. Dyad-to-dyad ratios at theta_h = 0.7, 20 steps: wall
4x, 12x, 15.5x, 16.4x and peak terms 9.6x, 11x, 15x, 15x per factor of four in
the cutoff. So 2^-18 projects to ~6.6e3 s at 32 threads (~7e4 s single-threaded)
and ~7e8 resident terms (~37 GiB of columns) at that point -- out of budget
for a single grid point, never mind eight.
"""

#: Per-`(theta_h, steps)` **recorded cuts** to the coefficient grid: the tightest
#: cutoff that leg is allowed to reach. A cut only ever *shortens* the grid, and
#: only at the tight, expensive end, so `--coeff-grid` still governs the loose
#: end everywhere.
#:
#: The full grid (through `2^-18`) is kept at **5 steps** for both angles —
#: projected 230 s and 400 s for the `2^-18` point, which is affordable once and
#: honors the full grid at one depth. At every deeper rung the timed grid stops
#: at `2^-14`: the `2^-16` point alone projects to ~3 800 s single-threaded at
#: 10 steps and was *measured* at 404 s with 32 threads at 20 steps, i.e. ~1.1 h
#: pinned. The `2^-16` value at 20 steps is still reported — the reference sweep
#: reaches it, with threads, because a reference is an oracle and not a timing
#: measurement.
COEFF_GRID_CUTS: dict[tuple[str, int], float] = {
    ("7pi/32", 9): 2.0**-14,
    ("7pi/32", 15): 2.0**-14,
    ("7pi/32", 20): 2.0**-14,
    ("5pi/16", 9): 2.0**-14,
    ("5pi/16", 15): 2.0**-14,
    ("5pi/16", 20): 2.0**-14,
}


def coeff_grid_for(
    theta_label: str, steps: int, default: Sequence[float]
) -> list[float]:
    """The coefficient grid for one point, after the recorded cuts."""
    floor = COEFF_GRID_CUTS.get((theta_label, steps))
    if floor is None:
        return list(default)
    return [eps for eps in default if eps >= floor]


# --------------------------------------------------------------------------
# References — B's machinery, C's tuning
# --------------------------------------------------------------------------

#: Benchmark B's `Reference` record, reused so that C's JSON carries exactly the
#: same reference fields (`reference_exact`, `reference_uncertainty`, ...).
Reference = bench_b.Reference

#: B's corrected plateau test, imported rather than re-typed. See the module
#: docstring: the naive "two successive values agree" criterion was *measured*
#: wrong, and `test_benchmark_c_deep.py` asserts this alias is B's own function.
plateau_is_real = bench_b._plateau_is_real

#: Tolerance for the plateau test — see the module docstring for why it is 1e-3
#: here and 1e-5 in Benchmark B.
SELF_CONVERGENCE_TOL = 1e-3

#: How many dyadic powers past the tightest **timed** grid point the reference
#: sweep is allowed to reach. Two, so that every timed run — including the
#: tightest — is scored against something strictly tighter than itself. Where the
#: extension is unaffordable the reference stops early and says so.
SELF_CONVERGENCE_EXTRA_POWERS = 2

#: Budget guards on the self-convergence loop, passed straight to B's
#: `self_converged_reference` (which checks them by *projection* before running
#: the next, several-times-costlier point). These are the reference child's
#: budgets, not the timed sweep's: the child gets `REFERENCE_THREADS` workers.
SELF_CONVERGENCE_MAX_TERMS: int | None = 400_000_000
SELF_CONVERGENCE_MAX_SECONDS: float | None = 900.0

#: Rayon workers the reference child is allowed. A reference is an *oracle*, not
#: a timing measurement, so the single-thread rule for timed runs, which exists
#: to make cross-engine wall times comparable, does not bind it — nothing about
#: the reference's wall time is reported as a benchmark number. What the threads
#: buy is *reach*: a tighter cutoff, hence a better-converged reference.
REFERENCE_THREADS = 16


def self_convergence_grid(coeff_grid: Sequence[float]) -> tuple[float, ...]:
    """The dyadic reference grid for a timed grid: same points, extended tighter.

    Extending by `SELF_CONVERGENCE_EXTRA_POWERS` dyadic powers is what keeps the
    error of the tightest timed run from being zero by construction. The loose
    end is kept because the plateau test needs the *sequence* — B's measured
    failure mode is a flat value with a growing sum, which is only visible along
    a sweep.
    """
    if not coeff_grid:
        raise ValueError("coeff_grid is empty")
    tightest = min(coeff_grid)
    extra = [tightest * 4.0**-(k + 1) for k in range(SELF_CONVERGENCE_EXTRA_POWERS)]
    return tuple(sorted([*coeff_grid, *extra], reverse=True))


#: **Measured** commutation-aware backward-cone size of `Z_62` through the
#: kicked-Ising circuit, recomputed from the gate list by `oracles.light_cone` on
#: every run (these numbers are here so the routing below reads as a decision
#: rather than a magic constant):
#:
#: | steps | cone | gates in the reduced circuit |
#: |---|---|---|
#: | 5  | **19 q** | 83 |
#: | 9  | 65 q | 471 |
#: | 10 | 81 q | 638 |
#: | 15 | 127 q | 1 823 |
#: | 20 | 127 q | 3 178 |
#:
#: So an **exact** reference exists at 5 steps and nowhere deeper: by 10 steps the
#: cone is already 81 qubits (`2**81 * 16` bytes of statevector) and by 15 steps
#: it is the whole device, which is the structural reason plan D5 makes C's
#: reference self-converged. Corroborated from outside this repo: the published
#: exact-benchmark file (`PUBLISHED_ANCHOR`) covers Figs 4a-4d and 5a and has no
#: 5b column, i.e. no exact 20-step `<Z_62>` exists in the literature either.
#:
#: The 5-step rung is therefore the methodology validation: the self-convergence
#: procedure can be scored against a *known* answer at the real system size, not
#: only on the 20-qubit sublattice the CI gate uses.
CONE_SIZES = {5: 19, 9: 65, 10: 81, 15: 127, 20: 127}

#: Depths that get an exact light-cone reference, and the statevector cap for it.
#: `method="both"` runs *two* independent simulations over the same cone — an Aer
#: statevector and an untruncated Pauli propagation — and requires them to agree,
#: which is what Benchmark B established as affordable for this 19-qubit cone.
EXACT_REFERENCE_STEPS = frozenset({5})
LIGHT_CONE_STATEVECTOR_CAP = 19


def _reference_worker(
    theta: float, steps: int, grid: tuple[float, ...], kind: str = "auto"
) -> dict[str, Any]:
    """Child-process entry point. Returns a `Reference`'s fields as plain data.

    `kind="self_converged"` forces the self-convergence path even at a depth
    where an exact reference exists — that is how `--validate-convergence`
    scores the procedure against a known answer at `n = 127`.
    """
    observable = OBSERVABLE_BUILDERS[OBSERVABLE_NAME]()
    if kind == "auto" and steps in EXACT_REFERENCE_STEPS:
        spec = oracles.record_gates(
            circuits.heavy_hex_kicked_ising, N_QUBITS, steps, theta
        )
        start = time.perf_counter()
        value = complex(
            oracles.light_cone_exact(
                spec,
                observable,
                steps,
                initial_state=STATE,
                method="both",
                max_statevector_qubits=LIGHT_CONE_STATEVECTOR_CAP,
            )
        )
        seconds = time.perf_counter() - start
        if abs(value.imag) > 1e-12:  # pragma: no cover - Hermitian observable
            raise AssertionError(f"the light-cone oracle returned {value!r}")
        cone = oracles.light_cone(spec, observable, steps)
        return {
            "value": value.real,
            "method": "light_cone_exact:both",
            "exact": True,
            "uncertainty": None,
            "seconds": seconds,
            "evidence": {
                "cone_qubits": len(cone.qubits),
                "cone_gates": len(cone.gate_indices),
                "agreement": (
                    "Aer statevector and untruncated Pauli propagation over the same "
                    "cone, required to agree by oracles.light_cone_exact(method='both')"
                ),
            },
        }
    circuit = circuits.heavy_hex_kicked_ising(
        N_QUBITS, trotter_steps=steps, theta_h=theta
    )
    reference = bench_b.self_converged_reference(
        circuit,
        observable,
        grid=grid,
        state=STATE,
        direction=DIRECTION,
        tol=SELF_CONVERGENCE_TOL,
        max_terms=SELF_CONVERGENCE_MAX_TERMS,
        max_seconds=SELF_CONVERGENCE_MAX_SECONDS,
    )
    return {
        "value": reference.value,
        "method": reference.method,
        "exact": reference.exact,
        "uncertainty": reference.uncertainty,
        "seconds": reference.seconds,
        "evidence": reference.evidence,
    }


def resolve_reference(
    theta: float,
    steps: int,
    grid: Sequence[float],
    *,
    kind: str = "auto",
    in_process: bool = False,
    threads: int = REFERENCE_THREADS,
    log=None,
) -> Reference:
    """The reference for one `(theta_h, steps)` point — exact where one exists.

    Routing, not a fallback chain: 5 steps gets `light_cone_exact` because the
    cone is 19 qubits, everything deeper gets `self_converged_reference` because
    the cone is 81 qubits or the whole device (`CONE_SIZES`). Nothing here
    silently degrades an exact reference into a self-converged one.

    Computed in a spawned child for two reasons: it lets the self-convergence
    sweep use `threads` Rayon workers without disturbing the parent's pinned pool
    (Rayon builds its global pool once, at the first propagate, and never resizes
    it), and it confines the persistent OpenMP pool that qiskit-aer's statevector
    simulator leaves behind, which would otherwise trip
    `harness.assert_single_threaded` on every later timed run.
    """
    reference_grid = self_convergence_grid(grid)
    start = time.perf_counter()
    if in_process:
        payload = _reference_worker(theta, steps, reference_grid, kind)
    else:
        context = multiprocessing.get_context("spawn")
        saved = os.environ.get("RAYON_NUM_THREADS")
        os.environ["RAYON_NUM_THREADS"] = str(threads)
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=1, mp_context=context
            ) as pool:
                payload = pool.submit(
                    _reference_worker, theta, steps, reference_grid, kind
                ).result()
        finally:
            if saved is None:  # pragma: no cover - the driver always sets it
                os.environ.pop("RAYON_NUM_THREADS", None)
            else:
                os.environ["RAYON_NUM_THREADS"] = saved
    reference = Reference(**payload)
    if log is not None:
        points = reference.evidence.get("points", [])
        for point in points:
            delta = point["delta_vs_previous"]
            delta_text = "        —" if delta is None else f"{delta:9.2e}"
            log(
                f"      eps=2^{_dyadic_exponent(point['min_abs_coeff'])}  "
                f"<O>={point['value']:+.9f}  Δ={delta_text}  "
                f"terms={point['final_terms']:>10}  {point['seconds']:8.2f}s"
            )
        stopped = reference.evidence.get("stopped_early")
        if stopped:
            log(f"      stopped before convergence — {stopped}")
        status = (
            f"EXACT (cone {reference.evidence.get('cone_qubits')} qubits)"
            if reference.exact
            else (
                f"converged={reference.evidence.get('converged')}, uncertainty="
                f"{'n/a' if reference.uncertainty is None else format(reference.uncertainty, '.2e')}"
                f", claimable={reference_is_claimable(reference)}"
            )
        )
        log(
            f"      {reference.method}: {reference.value:+.12f} ({status}, "
            f"{time.perf_counter() - start:.1f} s wall, {threads} Rayon workers, "
            f"{'in-process' if in_process else 'isolated child'})"
        )
    return reference


def reference_is_claimable(reference: Reference) -> bool:
    """Is this reference resolved well enough to score the `ACCURACY_EPSILON` bar?

    An **exact** reference always is — that is what `exact` means here: an
    oracle with no truncation anywhere (at 5 steps, a 19-qubit causal cone
    evaluated two independent ways and required to agree).

    A self-converged one needs two conditions, and both are load-bearing:

    1. the plateau test declared convergence, and
    2. the reported uncertainty is comfortably (2x) inside the accuracy bar.

    Without (2) an "achieved" row is the circularity Benchmark B flagged for
    weight-17: the reference is this engine's own tightest run, so the tightest
    timed point agrees with it by construction and the error it reports says
    nothing about the truth.

    And a warning that is measured, not hypothetical: at full depth in the hard
    interior the successive-difference estimate is **not a bound in either
    direction**. On the 20-qubit sublattice at 20 steps, where the exact answer
    is known (`test_benchmark_c_deep.py`), the estimate *overstates* the true
    error by ~2.4x at `theta_h = 0.7` and *understates* it by ~16x at
    `theta_h = 1.0`. So condition (2) is a filter on how much a reference is
    worth quoting, never a proof that it is right; the README states the
    residual risk for every row it quotes.
    """
    if reference.exact:
        return True
    return bool(
        reference.evidence.get("converged")
        and reference.uncertainty is not None
        and reference.uncertainty < ACCURACY_EPSILON / 2.0
    )


def _dyadic_exponent(eps: float) -> int:
    """`-14` for `2**-14`. Used only for log/report labels."""
    return round(math.log2(eps))


def dyadic_label(eps: float) -> str:
    """`'2^-14'` when `eps` is an exact dyadic, else its `%g` form."""
    exponent = _dyadic_exponent(eps)
    return f"2^{exponent}" if 2.0**exponent == eps else f"{eps:g}"


# --------------------------------------------------------------------------
# The sanity envelope: outside it ⇒ semantics investigation, no timings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvelopeCheck:
    """Tracked-set size at the sweep's cutoffs, against the sanity envelope."""

    theta_label: str
    steps: int
    min_abs_coeff: float
    final_terms: int
    peak_terms: int | None
    #: The count actually scored against the envelope — `peak_terms` when
    #: available. It is the *peak resident* set, not the final one, that the
    #: envelope is about: it is what the run has to hold and merge, and at these
    #: angles the two differ by three orders of magnitude (measured: θ_h = 1.0,
    #: 20 steps, `2^-14` peaks at 1.53e7 terms and lands on 2.0e4).
    scored_terms: int
    inside: bool
    #: Does this reading demand a semantics investigation before its timings are
    #: reported? Only a count below the floor **at the headline depth** does. A
    #: shallower rung is *expected* below the floor (the envelope is a
    #: 20-step statement), and any count above the ceiling is expected too.
    needs_investigation: bool
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "theta_h_label": self.theta_label,
            "trotter_steps": self.steps,
            "min_abs_coeff": self.min_abs_coeff,
            "min_abs_coeff_label": dyadic_label(self.min_abs_coeff),
            "final_terms": self.final_terms,
            "peak_terms": self.peak_terms,
            "scored_terms": self.scored_terms,
            "inside_envelope": self.inside,
            "needs_investigation": self.needs_investigation,
            "verdict": self.verdict,
        }


def check_envelope(
    record: report.RunRecord, headline_steps: int = max(STEP_POINTS)
) -> EnvelopeCheck | None:
    """Score one record against `TERM_COUNT_ENVELOPE`, or `None` if not applicable.

    Only the three cutoffs in `PLAN_COEFF_GRID` are scored: the envelope is a
    statement about *those* cutoffs, and the three looser dyadics this driver
    prepends to the grid are expected to sit below it by construction.

    **Depth matters.** The 1.2e6-9.3e6 envelope is a statement about the
    *headline* depth, and the check says so rather than pretending otherwise:

    - below the floor at `headline_steps` — the failure the check exists for (an
      emptied sum, a commuting seed that never spread, a reversed direction).
      `needs_investigation`, and no timing from that point is reportable until
      it is explained.
    - below the floor at a shallower rung — *expected*: fewer steps means a
      smaller reachable set, and the 5-step rung really does peak at 3.9e5 terms
      at `2^-14`. Reported, not flagged.
    - above the ceiling — also expected, at a cutoff tighter than the one the
      envelope was quoted for. A cost fact, not a semantics fault.
    """
    eps = record.truncation.get("min_abs_coeff")
    if eps is None or not any(eps == plan_eps for plan_eps in PLAN_COEFF_GRID):
        return None
    low, high = TERM_COUNT_ENVELOPE
    peak = record.peak_terms if record.peak_terms is not None else record.final_terms
    steps = int(record.extra.get("trotter_steps", 0))
    inside = low <= peak <= high
    needs_investigation = False
    if inside:
        verdict = f"inside [{low:.1e}, {high:.1e}]"
    elif peak < low:
        if steps >= headline_steps:
            needs_investigation = True
            verdict = (
                f"BELOW the envelope floor {low:.1e} at the headline depth "
                f"({headline_steps} steps) — investigate semantics (direction, "
                "contraction state, an emptied sum) before reporting timings"
            )
        else:
            verdict = (
                f"below the envelope floor {low:.1e}, expected at {steps} of "
                f"{headline_steps} steps (a shallower circuit reaches fewer strings; "
                "the envelope is a headline-depth statement)"
            )
    else:
        verdict = (
            f"above the envelope ceiling {high:.1e} (expected: a tighter cutoff "
            "keeps more; a cost fact, not a semantics fault)"
        )
    return EnvelopeCheck(
        theta_label=str(record.extra.get("theta_h_label")),
        steps=steps,
        min_abs_coeff=eps,
        final_terms=record.final_terms,
        peak_terms=record.peak_terms,
        scored_terms=peak,
        inside=inside,
        needs_investigation=needs_investigation,
        verdict=verdict,
    )


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------


def _extra(theta_label: str, theta: float, steps: int, reference: Reference,
           sweep: str) -> dict[str, Any]:
    return {
        "benchmark": "C",
        "observable": OBSERVABLE_NAME,
        "theta_h_label": theta_label,
        "theta_h": theta,
        "trotter_steps": steps,
        "sweep": sweep,
        **reference.as_dict(),
    }


def sweep_one_point(
    theta_label: str,
    theta: float,
    steps: int,
    reference: Reference,
    *,
    coeff_grid: Sequence[float],
    library_versions: dict[str, str],
    log=None,
) -> tuple[list[report.RunRecord], harness.AccuracyResult | None]:
    """The dyadic coefficient sweep for one `(theta_h, steps)` point.

    Returns every record plus the `AccuracyResult` of the sweep (the "time to
    |error| < ACCURACY_EPSILON" selection), or `None` when the grid is empty.
    """
    circuit = circuits.heavy_hex_kicked_ising(
        N_QUBITS, trotter_steps=steps, theta_h=theta
    )
    observable = OBSERVABLE_BUILDERS[OBSERVABLE_NAME]()

    # `warmup` is switched off once a run gets long enough that a doubled cost
    # is not worth a warm allocator; the flag is recorded on the record.
    state = {"warm": True}

    def build_run(spec: harness.TruncationSpec) -> report.RunRecord:
        warm = state["warm"]
        record = harness.run_propagation(
            circuit,
            observable,
            spec,
            DIRECTION,
            state=STATE,
            warmup=warm,
            oracle_value=reference.value,
            threads=1,
            library_versions=library_versions,
            extra={
                **_extra(theta_label, theta, steps, reference, "min_abs_coeff"),
                "warm": warm,
                "min_abs_coeff_label": dyadic_label(spec.min_abs_coeff or 0.0),
            },
        )
        if record.propagation_time_s > WARMUP_TIME_BUDGET_S:
            state["warm"] = False
        if log is not None:
            log(
                f"      eps={dyadic_label(spec.min_abs_coeff or 0.0):<7} "
                f"<O>={record.expectation_value:+.9f} "
                f"err={record.absolute_error:.2e} "
                f"final={record.final_terms:>10} peak={record.peak_terms:>10} "
                f"{record.total_time_s:8.2f}s"
            )
        return record

    grid = coeff_grid_for(theta_label, steps, coeff_grid)
    if not grid:
        return [], None
    if log is not None:
        cut = "" if len(grid) == len(coeff_grid) else (
            f" [cut from {len(coeff_grid)} points, see COEFF_GRID_CUTS]"
        )
        log(f"    min_abs_coeff sweep (loosest first){cut}:")
    accuracy = harness.time_to_accuracy(
        build_run,
        reference.value,
        ACCURACY_EPSILON,
        [harness.TruncationSpec(min_abs_coeff=eps) for eps in grid],
    )
    return list(accuracy.records), accuracy


# --------------------------------------------------------------------------
# Cross-engine parity behind a memory gate
# --------------------------------------------------------------------------

#: Fraction of `MemAvailable` a projected jl leg may claim. jl's dict backend was
#: measured at 67.6 GiB on a 2.85e6-term sum in Benchmark B, this box has 251 GiB
#: total and is shared with other work (something was OOM-killed at 247 GiB
#: during this branch's development), so a leg gets at most this share of what is
#: actually free at the moment it is about to run.
JULIA_MEMORY_HEADROOM = 0.5

#: Bytes per resident term to assume for jl before two legs have been measured.
#: 24 KiB/term is Benchmark B's 67.6 GiB / 2.85e6 terms — deliberately the
#: pessimistic prior, so a thin pilot cannot let a large leg through.
JULIA_FALLBACK_BYTES_PER_TERM = 24_000.0

#: Julia's fixed footprint (runtime, JIT'd code, the task JSON) before any Pauli
#: sum exists, in GiB. Measured 3.68 GiB on a 1 925-term task, so a single small
#: pilot leg says almost nothing about the *per-term* cost. This is why the gate
#: fits an **affine** model, `base + slope * peak_terms`, rather than dividing one
#: measurement by its term count — which on that pilot would have read
#: 2 002 KiB/term and refused every later leg.
JULIA_FALLBACK_BASE_GIB = 4.0


@dataclass(frozen=True)
class JuliaMemoryModel:
    """`gib(peak_terms) = base_gib + slope_bytes_per_term * peak_terms`.

    Refit after every jl leg from the two most recent measurements, so the
    projection that gates the *next* leg is data from the same machine, same
    build, same backend. With fewer than two measurements the slope is the
    pessimistic prior above.
    """

    base_gib: float = JULIA_FALLBACK_BASE_GIB
    slope_bytes_per_term: float = JULIA_FALLBACK_BYTES_PER_TERM
    fitted: bool = False

    def projected_gib(self, peak_terms: int) -> float:
        return self.base_gib + self.slope_bytes_per_term * peak_terms / 2.0**30

    def refit(self, samples: Sequence[tuple[int, float]]) -> JuliaMemoryModel:
        """`samples` are `(peak_terms, measured_gib)`, in the order measured."""
        usable = [(t, g) for t, g in samples if g > 0.0]
        if not usable:
            return self
        if len(usable) == 1:
            terms, gib = usable[0]
            # One point pins the intercept only: keep the pessimistic slope and
            # let the measured footprint raise the base if it was larger.
            return JuliaMemoryModel(
                base_gib=max(
                    gib - self.slope_bytes_per_term * terms / 2.0**30, JULIA_FALLBACK_BASE_GIB
                ),
                slope_bytes_per_term=self.slope_bytes_per_term,
                fitted=False,
            )
        (t0, g0), (t1, g1) = usable[-2], usable[-1]
        if t1 == t0:
            return self
        slope = (g1 - g0) * 2.0**30 / (t1 - t0)
        if slope <= 0.0:
            # jl's footprint did not rise with the sum (GC, or the base
            # dominates): keep the prior rather than projecting downward.
            return self
        return JuliaMemoryModel(
            base_gib=max(g1 - slope * t1 / 2.0**30, 0.0),
            slope_bytes_per_term=slope,
            fitted=True,
        )

#: Julia is slow to start (JIT) and a 5 420-gate task is not cheap; one warm
#: repeat is enough for a parity check, which is about term counts.
PARITY_WARM_REPEATS = 1

#: `(theta_h label, steps)` the parity leg runs at, and the dyadic cutoffs it is
#: attempted at, loosest first. The loosest is the **pilot**: it is what measures
#: jl's bytes/term, so it must be cheap enough to be certain of.
PARITY_POINT: tuple[str, int] = ("7pi/32", 20)
PARITY_COEFFS: tuple[float, ...] = (2.0**-10, 2.0**-12, 2.0**-14)


def julia_min_abs_coeff(eps: float) -> float:
    """The threshold to hand PauliPropagation.jl so its rule matches this repo's.

    This engine drops `|c| <= eps`; jl drops `|c| < eps` and therefore **keeps**
    a coefficient exactly equal to the threshold (`benchmarks/julia/README.md`
    §P3). With `eps' = nextafter(eps, inf)` there is no float strictly between
    `eps` and `eps'`, so jl's `|c| < eps'` is exactly `|c| <= eps`: the two rules
    coincide, for every input, with no coefficient touched.

    This is a perturbed-eps comparison, reported as a finding, never fudged.
    It matters here and not in Benchmark B because B deliberately used
    powers of ten (non-dyadic, so an exact straddle is measure-zero) while C's
    grid is dyadic, where coefficients really do land on the cutoff.
    """
    if eps <= 0.0:
        raise ValueError(f"min_abs_coeff must be positive, got {eps}")
    return math.nextafter(eps, math.inf)


def mem_available_gib() -> float | None:
    """`MemAvailable` from `/proc/meminfo`, in GiB, or `None` off Linux."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1048576.0
    except OSError:  # pragma: no cover - non-Linux
        return None
    return None


def _runner_rss_kb() -> float:
    """Summed `VmRSS` of this user's running `runner.jl` processes, in KiB.

    Scans `/proc`. `getrusage(RUSAGE_CHILDREN).ru_maxrss` would be simpler but is
    a *process-lifetime running maximum over all reaped children*, and this
    driver's reference children are themselves multi-gigabyte — so a jl leg that
    used less than a reference child would read as "0 GiB", and one that used
    more would read as the difference. Sampling the Julia process itself gives
    the figure that is actually wanted, and needs no cooperation from
    `runner.jl`.
    """
    total = 0.0
    uid = os.getuid()
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            if b"runner.jl" not in (entry / "cmdline").read_bytes():
                continue
            with (entry / "status").open() as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += float(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):
            continue
    return total


class JuliaRssSampler:
    """Background sampler for the peak RSS of the `runner.jl` subprocess.

    A daemon thread polling `/proc` every `interval` seconds while `run_task`
    blocks. Sampling can only *under*-report (a peak between two samples is
    missed), so the figure is a lower bound on jl's high-water mark — which is
    the safe direction for a memory gate that decides whether to attempt the
    next, larger leg.
    """

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.peak_kb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_kb = max(self.peak_kb, _runner_rss_kb())

    def __enter__(self) -> JuliaRssSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @property
    def peak_gib(self) -> float:
        return self.peak_kb / 1048576.0


@dataclass
class ParityOutcome:
    """One matched-truncation comparison against PauliPropagation.jl."""

    theta_label: str
    steps: int
    min_abs_coeff: float
    julia_min_abs_coeff: float | None
    ran: bool
    ok: bool
    detail: str
    rust_record: report.RunRecord | None = None
    julia_record: report.RunRecord | None = None
    layers_compared: int = 0
    first_layer_mismatch: int | None = None
    rust_layers: list[int] = field(default_factory=list)
    julia_layers: list[int] = field(default_factory=list)
    julia_peak_rss_gib: float | None = None
    projected_julia_gib: float | None = None
    mem_available_gib: float | None = None
    bytes_per_term: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "theta_h_label": self.theta_label,
            "trotter_steps": self.steps,
            "min_abs_coeff": self.min_abs_coeff,
            "min_abs_coeff_label": dyadic_label(self.min_abs_coeff),
            "julia_min_abs_coeff": self.julia_min_abs_coeff,
            "one_ulp_perturbation": (
                None
                if self.julia_min_abs_coeff is None
                else self.julia_min_abs_coeff - self.min_abs_coeff
            ),
            "ran": self.ran,
            "ok": self.ok,
            "detail": self.detail,
            "layers_compared": self.layers_compared,
            "first_layer_mismatch": self.first_layer_mismatch,
            "rust_final_terms": (
                None if self.rust_record is None else self.rust_record.final_terms
            ),
            "julia_final_terms": (
                None if self.julia_record is None else self.julia_record.final_terms
            ),
            "rust_peak_terms": (
                None if self.rust_record is None else self.rust_record.peak_terms
            ),
            "julia_peak_terms": (
                None if self.julia_record is None else self.julia_record.peak_terms
            ),
            "rust_propagation_time_s": (
                None if self.rust_record is None else self.rust_record.propagation_time_s
            ),
            "julia_propagation_time_s": (
                None
                if self.julia_record is None
                else self.julia_record.propagation_time_s
            ),
            "julia_peak_rss_gib": self.julia_peak_rss_gib,
            "projected_julia_gib": self.projected_julia_gib,
            "mem_available_gib": self.mem_available_gib,
            "bytes_per_term": self.bytes_per_term,
        }


def _rust_leg(
    theta_label: str,
    theta: float,
    steps: int,
    reference: Reference,
    eps: float,
) -> tuple[report.RunRecord, list[int], Any]:
    """The engine side of one parity case: a timed record plus per-layer counts."""
    spec = oracles.record_gates(
        circuits.heavy_hex_kicked_ising, N_QUBITS, steps, theta
    )
    observable = OBSERVABLE_BUILDERS[OBSERVABLE_NAME]()
    circuit = spec.to_circuit()
    record = harness.run_propagation(
        circuit,
        observable,
        harness.TruncationSpec(min_abs_coeff=eps),
        DIRECTION,
        state=STATE,
        warmup=False,
        oracle_value=reference.value,
        threads=1,
        extra={
            **_extra(theta_label, theta, steps, reference, "parity"),
            "warm": False,
            "min_abs_coeff_label": dyadic_label(eps),
        },
    )
    _, stats = observable.propagate_with_stats(
        circuit, harness.make_policy(min_abs_coeff=eps), direction=DIRECTION
    )
    return record, list(stats.terms_out), spec


def julia_parity(
    theta_label: str,
    theta: float,
    steps: int,
    reference: Reference,
    eps: float,
    *,
    model: JuliaMemoryModel,
    timeout_s: float = 7200.0,
    log=None,
) -> ParityOutcome:
    """Run one matched-truncation task on both engines and compare per layer.

    Per-layer counts, not just the final count: a divergence that cancels by the
    end is exactly the coefficient-boundary or truncation-schedule bug the
    comparison exists to catch (`benchmarks/julia/README.md` §P3, §P5). Both
    lists are in *application* order on both engines, so they line up index by
    index with no reversal.

    The engine leg always runs (it is a single-engine measurement). The jl leg
    runs only if `model.projected_gib(rust_peak_terms)` fits inside
    `JULIA_MEMORY_HEADROOM * MemAvailable`; otherwise the outcome comes back
    with `ran=False` and the projection, which the README reports as a measured
    asymmetry.

    Never raises for a parity failure: the driver's contract is to *report* it
    and withhold the cross-engine timing, not to abort.
    """
    import julia_baseline

    rust, rust_layers, spec = _rust_leg(theta_label, theta, steps, reference, eps)
    peak = rust.peak_terms or rust.final_terms
    projected_gib = model.projected_gib(peak)
    available = mem_available_gib()
    budget = None if available is None else JULIA_MEMORY_HEADROOM * available

    if budget is not None and projected_gib > budget:
        detail = (
            f"jl leg SKIPPED for memory: {peak:.3g} resident terms at "
            f"{model.base_gib:.1f} GiB + {model.slope_bytes_per_term / 1024:.2f} KiB/term "
            f"({'measured' if model.fitted else 'prior'}) projects to "
            f"{projected_gib:.0f} GiB, over the {budget:.0f} GiB budget "
            f"({JULIA_MEMORY_HEADROOM:.0%} of {available:.0f} GiB available). "
            "This engine ran the same task in "
            f"{(rust.peak_memory_kb or 0.0) / 1048576.0:.2f} GiB."
        )
        if log is not None:
            log(f"      {detail}")
        return ParityOutcome(
            theta_label, steps, eps, None, False, False, detail,
            rust_record=rust, rust_layers=rust_layers,
            projected_julia_gib=projected_gib, mem_available_gib=available,
            bytes_per_term=model.slope_bytes_per_term,
        )

    observable = OBSERVABLE_BUILDERS[OBSERVABLE_NAME]()
    terms = oracles.pauli_terms(observable, N_QUBITS)
    jl_eps = julia_min_abs_coeff(eps)
    task = julia_baseline.make_task(
        n_qubits=N_QUBITS,
        gates=spec.to_circuit_json()["gates"],
        observable={label: coeff for label, coeff in terms},
        direction=DIRECTION,
        min_abs_coeff=jl_eps,
        threads=1,
        state=STATE,
    )

    try:
        with JuliaRssSampler() as sampler:
            result = julia_baseline.run_task(
                task,
                threads=1,
                warm_repeats=PARITY_WARM_REPEATS,
                layer_counts=True,
                timeout=timeout_s,
            )
    except julia_baseline.JuliaBaselineError as exc:
        return ParityOutcome(
            theta_label, steps, eps, jl_eps, False, False,
            f"PauliPropagation.jl runner failed: {exc}",
            rust_record=rust, rust_layers=rust_layers,
            projected_julia_gib=projected_gib, mem_available_gib=available,
            bytes_per_term=model.slope_bytes_per_term,
        )
    julia_peak_gib = sampler.peak_gib

    jl_versions = result.versions
    julia_record = report.RunRecord(
        engine="PauliPropagation.jl",
        engine_version=jl_versions.get("PauliPropagation", "unknown"),
        n_qubits=N_QUBITS,
        direction=DIRECTION,
        # The *engine-equivalent* label, so `check_term_parity`'s
        # matched-truncation test compares like with like; the one-ulp
        # perturbation actually passed to jl is in `extra`.
        truncation={"min_abs_coeff": eps},
        propagation_time_s=result.wall_warm_s or result.wall_cold_s,
        final_terms=result.final_terms,
        provenance=report.collect_provenance(
            thread_count=1,
            extra_library_versions=jl_versions,
            repo_root=_REPO_ROOT,
        ),
        peak_terms=result.peak_terms,
        expectation_value=None if result.expectation is None else result.expectation.real,
        absolute_error=(
            None
            if result.expectation is None
            else abs(result.expectation.real - reference.value)
        ),
        peak_memory_kb=julia_peak_gib * 1048576.0,
        extra={
            **_extra(theta_label, theta, steps, reference, "parity"),
            "warm": result.wall_warm_s is not None,
            "wall_cold_s": result.wall_cold_s,
            "min_abs_coeff_label": dyadic_label(eps),
            "julia_min_abs_coeff": jl_eps,
            "one_ulp_perturbation": jl_eps - eps,
            "julia_notes": result.notes,
        },
    )

    jl_layers = result.per_layer_terms or []
    first_mismatch = None
    compared = min(len(rust_layers), len(jl_layers))
    for index in range(compared):
        if rust_layers[index] != jl_layers[index]:
            first_mismatch = index
            break

    record_parity = harness.check_term_parity(rust, julia_record, coeff_tol=1e-9)
    reasons = list(record_parity.reasons)
    if len(rust_layers) != len(jl_layers):
        reasons.append(
            f"per-layer count lists have different lengths: {len(rust_layers)} "
            f"(paulistrings) vs {len(jl_layers)} (PauliPropagation.jl)"
        )
    if first_mismatch is not None:
        reasons.append(
            f"per-layer term counts first differ at applied layer {first_mismatch}: "
            f"{rust_layers[first_mismatch]} vs {jl_layers[first_mismatch]}"
        )

    ok = not reasons
    detail = (
        f"{compared}/{compared} per-layer term counts identical; final "
        f"{rust.final_terms} terms on both; |Δ⟨O⟩| = "
        f"{abs((julia_record.expectation_value or 0.0) - (rust.expectation_value or 0.0)):.3e}"
        if ok
        else "; ".join(reasons)
    )
    if log is not None:
        log(f"      {'PARITY OK' if ok else 'PARITY FAILED'}: {detail}")
        log(
            f"      jl peak RSS {julia_peak_gib:.2f} GiB at {peak} resident terms "
            f"(projected {projected_gib:.2f} GiB); paulistrings "
            f"{(rust.peak_memory_kb or 0.0) / 1048576.0:.2f} GiB "
            f"({(rust.extra.get('peak_memory_kb_delta') or 0.0) / 1048576.0:.2f} GiB "
            "of it this run)"
        )
    return ParityOutcome(
        theta_label,
        steps,
        eps,
        jl_eps,
        True,
        ok,
        detail,
        rust_record=rust,
        julia_record=julia_record,
        layers_compared=compared,
        first_layer_mismatch=first_mismatch,
        rust_layers=rust_layers,
        julia_layers=jl_layers,
        julia_peak_rss_gib=julia_peak_gib,
        projected_julia_gib=projected_gib,
        mem_available_gib=available,
        bytes_per_term=model.slope_bytes_per_term,
    )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

#: The validated categorical palette `report.py` also uses (dataviz skill,
#: `references/palette.md`). Kept as a local constant rather than reaching into
#: `report`'s private `_PALETTE`, since here the categorical dimension is
#: `(theta_h, steps)`, not the engine.
_SERIES_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
_MUTED = "#898781"
_GRID = "#e1e0d9"

#: Error floor for a log plot: an exactly-zero error cannot be drawn on a log
#: axis, so it is clamped here and the clamp is stated in the axis label.
_ERROR_FLOOR = 1e-17


def _style(ax) -> None:
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED)


def _series_color(steps: int, steps_order: Sequence[int]) -> str:
    return _SERIES_COLORS[list(steps_order).index(steps) % len(_SERIES_COLORS)]


def _legend_on_a_populated_panel(axes) -> None:
    """One legend, on the first panel that actually drew something."""
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8)
            return


def _panel_records(records, theta_label: str, sweep: str, steps: int):
    return [
        r
        for r in records
        if r.extra.get("theta_h_label") == theta_label
        and r.extra.get("sweep") == sweep
        and r.extra.get("trotter_steps") == steps
        and r.engine == "paulistrings"
    ]


def plot_error_vs_runtime(
    records: Sequence[report.RunRecord],
    theta_labels: Sequence[str],
    steps_order: Sequence[int],
    *,
    save_path: Path,
):
    """**The headline figure**: |error| vs warm wall time, with the 0.01 bar.

    One panel per `theta_h`, one curve per Trotter depth, points ordered by wall
    time. The dashed horizontal line is `ACCURACY_EPSILON` — reading the figure
    is "how far right must this depth's curve run before it drops under the
    line", which is exactly time-to-fixed-accuracy.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(theta_labels), figsize=(4.4 * len(theta_labels), 3.9), squeeze=False
    )
    for ax, theta_label in zip(axes[0], theta_labels):
        for steps in steps_order:
            points = sorted(
                (r.total_time_s, max(r.absolute_error, _ERROR_FLOOR))
                for r in _panel_records(records, theta_label, "min_abs_coeff", steps)
                if r.absolute_error is not None
            )
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(
                xs, ys, marker="o", markersize=4, linewidth=1.4,
                color=_series_color(steps, steps_order), label=f"{steps} steps",
            )
        ax.axhline(
            ACCURACY_EPSILON, color=_MUTED, linewidth=1.2, linestyle="--",
            label=f"target {ACCURACY_EPSILON:g}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("warm wall time (s), single-threaded")
        ax.set_ylabel(f"|error| vs reference (floored at {_ERROR_FLOOR:g})")
        ax.set_title(f"θ_h = {theta_label}", color=_MUTED, fontsize=10)
        _style(ax)
    _legend_on_a_populated_panel(list(axes[0]))
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_term_count_vs_truncation(
    records: Sequence[report.RunRecord],
    theta_labels: Sequence[str],
    steps_order: Sequence[int],
    *,
    save_path: Path,
):
    """Peak resident terms vs `min_abs_coeff`, with the sanity envelope shaded.

    The envelope band is 1.2e6-9.3e6, and `PLAN_COEFF_GRID`'s three cutoffs
    are marked: a curve that crosses the band inside those cutoffs is the setup
    behaving as expected.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(theta_labels), figsize=(4.4 * len(theta_labels), 3.9), squeeze=False
    )
    low, high = TERM_COUNT_ENVELOPE
    for ax, theta_label in zip(axes[0], theta_labels):
        ax.axhspan(low, high, color=_GRID, alpha=0.8, zorder=0)
        for steps in steps_order:
            points = sorted(
                (
                    r.truncation["min_abs_coeff"],
                    r.peak_terms if r.peak_terms is not None else r.final_terms,
                )
                for r in _panel_records(records, theta_label, "min_abs_coeff", steps)
                if "min_abs_coeff" in r.truncation
            )
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(
                xs, ys, marker="o", markersize=4, linewidth=1.4,
                color=_series_color(steps, steps_order), label=f"{steps} steps",
            )
        for eps in PLAN_COEFF_GRID:
            ax.axvline(eps, color=_MUTED, linewidth=0.7, linestyle=":", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("min_abs_coeff (tightening to the right; dotted: 2⁻¹⁴/⁻¹⁶/⁻¹⁸)")
        ax.set_ylabel(f"peak resident terms (band: {low:.1e}–{high:.1e} envelope)")
        ax.set_title(f"θ_h = {theta_label}", color=_MUTED, fontsize=10)
        _style(ax)
    _legend_on_a_populated_panel(list(axes[0]))
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_convergence_panel(
    records: Sequence[report.RunRecord],
    references: dict[tuple[str, int], Reference],
    theta_labels: Sequence[str],
    steps_order: Sequence[int],
    *,
    save_path: Path,
):
    """Convergence panel: ⟨Z_62⟩ vs cutoff, reference dashed."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(theta_labels), figsize=(4.4 * len(theta_labels), 3.9), squeeze=False
    )
    for ax, theta_label in zip(axes[0], theta_labels):
        for steps in steps_order:
            points = sorted(
                (r.truncation["min_abs_coeff"], r.expectation_value)
                for r in _panel_records(records, theta_label, "min_abs_coeff", steps)
                if "min_abs_coeff" in r.truncation and r.expectation_value is not None
            )
            if not points:
                continue
            xs, ys = zip(*points)
            color = _series_color(steps, steps_order)
            ax.plot(
                xs, ys, marker="o", markersize=4, linewidth=1.4, color=color,
                label=f"{steps} steps",
            )
            reference = references.get((theta_label, steps))
            if reference is not None:
                ax.axhline(
                    reference.value, color=color, linewidth=0.9, linestyle="--",
                    alpha=0.7,
                )
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("min_abs_coeff (tightening to the right)")
        ax.set_ylabel("⟨Z₆₂⟩ (dashed: self-converged reference)")
        ax.set_title(f"θ_h = {theta_label}", color=_MUTED, fontsize=10)
        _style(ax)
    _legend_on_a_populated_panel(list(axes[0]))
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_parity_layers(outcomes: Sequence[ParityOutcome], *, save_path: Path):
    """Per-layer term counts, both engines, for every parity case that ran.

    The two curves are drawn thick-solid (paulistrings) under thin-dashed
    (PauliPropagation.jl) precisely so that *identical* lists look like one line
    with a dashed overlay — a visible orange excursion is a real divergence, not
    a rendering artefact.
    """
    import matplotlib.pyplot as plt

    drawable = [o for o in outcomes if o.rust_layers and o.julia_layers]
    if not drawable:
        return None
    fig, axes = plt.subplots(
        1, len(drawable), figsize=(4.2 * len(drawable), 3.6), squeeze=False
    )
    for ax, outcome in zip(axes[0], drawable):
        ax.plot(range(len(outcome.rust_layers)), outcome.rust_layers, linewidth=2.6,
                color="#2a78d6", label="paulistrings")
        ax.plot(range(len(outcome.julia_layers)), outcome.julia_layers, linewidth=1.0,
                linestyle="--", color="#eb6834", label="PauliPropagation.jl")
        ax.set_yscale("log")
        ax.set_xlabel("applied layer (one gate = one channel)")
        ax.set_ylabel("resident terms")
        ax.set_title(
            f"θ_h = {outcome.theta_label}, {outcome.steps} steps, "
            f"min_abs_coeff = {dyadic_label(outcome.min_abs_coeff)}",
            color=_MUTED, fontsize=9,
        )
        _style(ax)
    _legend_on_a_populated_panel(list(axes[0]))
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


DEFAULT_OUT_DIR = _REPO_ROOT / "benchmarks" / "python" / "deep_trotter"


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name, key in (("qiskit", "qiskit"), ("stim", "stim"), ("numpy", "numpy")):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        versions[key] = getattr(module, "__version__", "unknown")
    return versions


def _published_cross_check(log) -> dict[str, Any]:
    """Try `oracles.load_published_reference`; never fabricate on failure.

    A published cross-check is used only if obtainable with clean provenance.
    `examples/data/references/` ships with **no data files**.
    `PUBLISHED_ANCHOR` records exactly where the relevant one lives, which of its
    columns is this benchmark's observable and depth, and why it was not checked
    in from this environment — so this call reports what it found (normally
    nothing) and the report leans on `PUBLISHED_ANCHOR` for the pointer rather
    than on a transcription.
    """
    for name in ("begusic2023_exact", "kim2023_experiment"):
        try:
            published = oracles.load_published_reference(name)
        except FileNotFoundError as exc:
            log(f"  published cross-check {name}: not available — {exc}".split("\n")[0])
            continue
        except oracles.OracleError as exc:
            log(f"  published cross-check {name}: REFUSED by the loader — {exc}")
            return {"name": name, "status": "refused", "detail": str(exc)}
        log(f"  published cross-check {name}: loaded from {published.path}")
        return {
            "name": name,
            "status": "loaded",
            "path": str(published.path),
            "provenance": published.provenance,
            "rows": len(published.rows),
        }
    return {
        "status": "unavailable",
        "detail": (
            "no provenance-tagged reference file is checked in. See "
            "`PUBLISHED_ANCHOR` for the located upstream file, its column-to-"
            "observable mapping, the conventions cross-check that was possible "
            "without it, and why it was not retrieved byte-exactly here. References "
            "in this run are exact (5 steps, causal cone) or self-converged "
            "(deeper), per plan D5."
        ),
        "anchor": PUBLISHED_ANCHOR,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark C: deep Trotter time-to-fixed-accuracy on the 127-qubit "
            "heavy hex."
        )
    )
    parser.add_argument(
        "--thetas", nargs="+", default=[label for label, _ in THETA_POINTS],
        choices=[label for label, _ in THETA_POINTS],
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", default=list(STEP_POINTS), choices=list(STEP_POINTS),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--coeff-grid", type=float, nargs="*", default=list(COEFF_GRID),
        help="min_abs_coeff grid, loosest first (dyadic)",
    )
    parser.add_argument("--no-julia", action="store_true", help="skip the parity leg")
    parser.add_argument(
        "--parity-coeffs", type=float, nargs="*", default=list(PARITY_COEFFS),
        help="dyadic min_abs_coeff values the jl parity check is attempted at",
    )
    parser.add_argument(
        "--parity-theta", default=PARITY_POINT[0],
        choices=[label for label, _ in THETA_POINTS],
    )
    parser.add_argument(
        "--parity-steps", type=int, default=PARITY_POINT[1], choices=list(STEP_POINTS),
    )
    parser.add_argument(
        "--julia-timeout", type=float, default=1800.0,
        help=(
            "wall-clock cap per PauliPropagation.jl leg. The runner does a cold "
            "propagation, PARITY_WARM_REPEATS warm ones and one untimed counting pass, "
            "so a tight cutoff at 20 steps is several propagations of a multi-million-"
            "term sum; a timeout is reported as a skipped leg, not as a parity failure"
        ),
    )
    parser.add_argument(
        "--validate-convergence", action="store_true",
        help=(
            "at every depth with an exact reference (5 steps), also run the "
            "self-convergence procedure and score its estimate against the truth"
        ),
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--reference-threads", type=int, default=REFERENCE_THREADS,
        help="Rayon workers the reference child may use (a reference is an oracle)",
    )
    parser.add_argument(
        "--in-process-references", action="store_true",
        help="compute references in this process (single-threaded); faster to debug",
    )
    args = parser.parse_args(argv)

    if os.environ.get("RAYON_NUM_THREADS") != "1":
        parser.error(
            "export RAYON_NUM_THREADS=1 before starting the interpreter: Rayon builds "
            "its global pool at the first propagate and never resizes it"
        )
    harness.assert_logging_quiet()
    bad = [eps for eps in args.coeff_grid if eps < bench_b.MIN_SAFE_COEFF]
    if bad:
        parser.error(
            f"min_abs_coeff values {bad} are below Benchmark B's "
            f"MIN_SAFE_COEFF={bench_b.MIN_SAFE_COEFF:g}; the cos(pi/2) residual branch "
            "then survives truncation at a Clifford angle and the propagation fans out"
        )

    def log(message: str) -> None:
        print(message, flush=True)

    theta_points = [(l, v) for l, v in THETA_POINTS if l in args.thetas]
    steps_points = [s for s in STEP_POINTS if s in args.steps]
    library_versions = _library_versions()
    started = time.perf_counter()

    records: list[report.RunRecord] = []
    references: dict[tuple[str, int], Reference] = {}
    accuracy_rows: list[dict[str, Any]] = []
    envelope_rows: list[EnvelopeCheck] = []

    log("=== published-reference cross-check (plan D5: only with clean provenance) ===")
    published = _published_cross_check(log)

    for theta_label, theta in theta_points:
        for steps in steps_points:
            log(f"\n=== θ_h = {theta_label}, {steps} Trotter steps ===")
            grid = coeff_grid_for(theta_label, steps, args.coeff_grid)
            if not grid:
                log("    grid fully cut — skipped (see COEFF_GRID_CUTS)")
                continue
            log("    reference (self-converged, B's plateau criterion):")
            reference = resolve_reference(
                theta, steps, grid,
                in_process=args.in_process_references,
                threads=args.reference_threads,
                log=log,
            )
            references[(theta_label, steps)] = reference

            point_records, accuracy = sweep_one_point(
                theta_label, theta, steps, reference,
                coeff_grid=args.coeff_grid,
                library_versions=library_versions,
                log=log,
            )
            records.extend(point_records)
            headline_steps = max(steps_points)
            for record in point_records:
                check = check_envelope(record, headline_steps)
                if check is not None:
                    envelope_rows.append(check)
                    if not check.inside:
                        log(
                            f"      envelope @ {dyadic_label(check.min_abs_coeff)}: "
                            f"{check.verdict}"
                        )

            if accuracy is not None:
                # "Achieved" means a grid point's error against *this* reference
                # was under the bar. That is only a *claim* about accuracy if the
                # reference itself is resolved well inside the bar — otherwise it
                # is the circularity Benchmark B flagged for weight-17: the
                # reference is this engine's own tightest run, so the tightest
                # timed point agrees with it by construction. `claimable` is the
                # honest gate, and the README quotes only claimable rows.
                claimable = accuracy.achieved and reference_is_claimable(reference)
                accuracy_rows.append(
                    {
                        "theta_h_label": theta_label,
                        "trotter_steps": steps,
                        "epsilon": accuracy.epsilon,
                        "achieved": accuracy.achieved,
                        "claimable": claimable,
                        "reference_value": reference.value,
                        "reference_uncertainty": reference.uncertainty,
                        "reference_converged": reference.evidence.get("converged"),
                        "first_truncation": (
                            None if accuracy.first_spec is None
                            else accuracy.first_spec.as_dict()
                        ),
                        "first_time_s": (
                            None if accuracy.first is None else accuracy.first.total_time_s
                        ),
                        "first_terms": (
                            None if accuracy.first is None else accuracy.first.final_terms
                        ),
                        "cheapest_truncation": (
                            None if accuracy.cheapest_spec is None
                            else accuracy.cheapest_spec.as_dict()
                        ),
                        "cheapest_time_s": (
                            None if accuracy.cheapest is None
                            else accuracy.cheapest.total_time_s
                        ),
                        "cheapest_terms": (
                            None if accuracy.cheapest is None
                            else accuracy.cheapest.final_terms
                        ),
                        "coeff_grid_used": [s.min_abs_coeff for s in accuracy.specs],
                        "coeff_grid_cut": (theta_label, steps) in COEFF_GRID_CUTS,
                    }
                )
                log("    " + accuracy.describe().replace("\n", "\n    "))

    # --- methodology validation, where the exact answer is known ------------
    validation_rows: list[dict[str, Any]] = []
    if args.validate_convergence:
        log("\n=== self-convergence methodology validation (n = 127, exact cone) ===")
        for theta_label, theta in theta_points:
            for steps in steps_points:
                exact = references.get((theta_label, steps))
                if exact is None or not exact.exact:
                    continue
                log(
                    f"  validate θ_h = {theta_label}, {steps} steps "
                    f"(exact: {exact.method})"
                )
                approx = resolve_reference(
                    theta, steps,
                    coeff_grid_for(theta_label, steps, args.coeff_grid),
                    kind="self_converged",
                    in_process=args.in_process_references,
                    threads=args.reference_threads,
                    log=log,
                )
                true_error = abs(approx.value - exact.value)
                validation_rows.append(
                    {
                        "theta_h_label": theta_label,
                        "trotter_steps": steps,
                        "exact_value": exact.value,
                        "exact_method": exact.method,
                        "self_converged_value": approx.value,
                        "estimated_uncertainty": approx.uncertainty,
                        "true_error": true_error,
                        "converged": approx.evidence.get("converged"),
                        # The same honesty bar Benchmark B asserts: a
                        # successive-difference estimate is a heuristic, not a
                        # bound, so the question is whether it understates the
                        # true error by more than an order of magnitude.
                        "conservative": (
                            approx.uncertainty is not None
                            and true_error
                            <= max(
                                approx.uncertainty * bench_b._UNCERTAINTY_SLACK,
                                bench_b._FP_NOISE_FLOOR,
                            )
                        ),
                        "uncertainty_slack": bench_b._UNCERTAINTY_SLACK,
                    }
                )
                log(
                    f"    exact={exact.value:+.12f}  self-converged={approx.value:+.12f}"
                    f"  true error={true_error:.2e}  estimated="
                    f"{'n/a' if approx.uncertainty is None else format(approx.uncertainty, '.2e')}"
                )

    # --- cross-engine parity, behind the memory gate ------------------------
    parity_outcomes: list[ParityOutcome] = []
    julia_skip_reason: str | None = None
    if not args.no_julia:
        import julia_baseline

        julia_skip_reason = julia_baseline.skip_reason()
        if julia_skip_reason is not None:
            log(f"\nPauliPropagation.jl parity skipped: {julia_skip_reason}")
        else:
            theta_label, steps = args.parity_theta, args.parity_steps
            theta = dict(THETA_POINTS)[theta_label]
            reference = references.get((theta_label, steps))
            if reference is None:
                log(
                    f"\nparity θ_h={theta_label}, {steps} steps: no reference in this "
                    "run, skipped"
                )
            else:
                model = JuliaMemoryModel()
                samples: list[tuple[int, float]] = []
                for eps in sorted(args.parity_coeffs, reverse=True):
                    log(
                        f"\n=== parity: θ_h = {theta_label}, {steps} steps, "
                        f"min_abs_coeff = {dyadic_label(eps)} "
                        f"(jl gets +1 ulp) ==="
                    )
                    outcome = julia_parity(
                        theta_label, theta, steps, reference, eps,
                        model=model, timeout_s=args.julia_timeout, log=log,
                    )
                    parity_outcomes.append(outcome)
                    if outcome.ran and outcome.julia_peak_rss_gib and outcome.rust_record:
                        # Refit the gate's memory model from this leg, so the
                        # projection for the next (larger) one is measured.
                        peak = (
                            outcome.rust_record.peak_terms
                            or outcome.rust_record.final_terms
                        )
                        samples.append((peak, outcome.julia_peak_rss_gib))
                        model = model.refit(samples)
                        log(
                            f"      memory model now {model.base_gib:.2f} GiB + "
                            f"{model.slope_bytes_per_term / 1024:.2f} KiB/term "
                            f"({'fitted' if model.fitted else 'prior slope'})"
                        )
                    # The *engine* record is always kept (it is a
                    # single-engine measurement), but the jl record — the only
                    # thing that turns the pair into a cross-engine claim — is
                    # written only when parity holds.
                    if outcome.rust_record is not None:
                        records.append(outcome.rust_record)
                    if outcome.julia_record is not None and outcome.ok:
                        records.append(outcome.julia_record)

    # --- outputs ------------------------------------------------------------
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    # The committed artifact is a *snapshot*, regenerated wholesale, so the file
    # is removed before `report.write_results` (which appends by design, the
    # right discipline for the gitignored campaign directory) recreates it.
    if results_path.exists():
        results_path.unlink()
    report.write_results(records, out_dir, name="results")
    log(f"\nwrote {len(records)} records to {results_path}")

    summary = {
        "benchmark": "C",
        "n_qubits": N_QUBITS,
        "observable": OBSERVABLE_NAME,
        "theta_zz": circuits.KICKED_ISING_CLIFFORD_THETA_ZZ,
        "state": STATE,
        "direction": DIRECTION,
        "step_points": steps_points,
        "theta_points": [(l, v) for l, v in theta_points],
        "plan_coeff_grid": list(PLAN_COEFF_GRID),
        "coeff_grid": list(args.coeff_grid),
        "coeff_grid_labels": [dyadic_label(eps) for eps in args.coeff_grid],
        "coeff_grid_cuts": {
            f"{theta}@{steps}steps": floor
            for (theta, steps), floor in COEFF_GRID_CUTS.items()
        },
        "accuracy_epsilon": ACCURACY_EPSILON,
        "term_count_envelope": list(TERM_COUNT_ENVELOPE),
        "self_convergence_tol": SELF_CONVERGENCE_TOL,
        "self_convergence_extra_powers": SELF_CONVERGENCE_EXTRA_POWERS,
        "self_convergence_criterion": (
            "bench_b_theta_sweep._plateau_is_real, imported verbatim: two successive "
            f"|Δ⟨O⟩| below tol={SELF_CONVERGENCE_TOL:g} AND (saturated term count OR "
            "both differences strictly nonzero); a zero-term sum is rejected outright"
        ),
        "reference_threads": args.reference_threads,
        "cone_sizes": CONE_SIZES,
        "exact_reference_steps": sorted(EXACT_REFERENCE_STEPS),
        "self_convergence_validation": validation_rows,
        "julia_memory_headroom": JULIA_MEMORY_HEADROOM,
        "julia_memory_model_prior": {
            "base_gib": JULIA_FALLBACK_BASE_GIB,
            "slope_bytes_per_term": JULIA_FALLBACK_BYTES_PER_TERM,
            "provenance": (
                "base measured on a 1925-term task on this host; slope is Benchmark "
                "B's 67.6 GiB / 2.85e6 terms, the pessimistic prior"
            ),
        },
        "julia_one_ulp_mitigation": (
            "this engine drops |c| <= eps, PauliPropagation.jl drops |c| < eps "
            "(benchmarks/julia/README.md §P3). Rust runs use the dyadic verbatim; jl "
            "runs get math.nextafter(eps, inf), which makes jl's rule exactly this "
            "engine's. Both numbers are recorded per parity case."
        ),
        "kim2023_provenance": observables.kim2023_provenance(),
        "published_cross_check": published,
        "published_anchor": PUBLISHED_ANCHOR,
        "references": {
            f"{theta}@{steps}steps": ref.as_dict()
            for (theta, steps), ref in references.items()
        },
        "time_to_accuracy": accuracy_rows,
        "envelope_checks": [check.as_dict() for check in envelope_rows],
        "julia_parity": [o.as_dict() for o in parity_outcomes],
        "julia_parity_point": [args.parity_theta, args.parity_steps],
        "julia_skip_reason": julia_skip_reason,
        "mem_available_gib_at_end": mem_available_gib(),
        "wall_clock_s": time.perf_counter() - started,
        "measured_pilot": _MEASURED_PILOT,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    log(f"wrote {summary_path}")

    if not args.no_figures:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            log("matplotlib not installed; figures skipped")
        else:
            theta_labels = [label for label, _ in theta_points]
            for path in (
                plot_error_vs_runtime(
                    records, theta_labels, steps_points,
                    save_path=out_dir / "error-vs-runtime.svg",
                ),
                plot_term_count_vs_truncation(
                    records, theta_labels, steps_points,
                    save_path=out_dir / "term-count-vs-truncation.svg",
                ),
                plot_convergence_panel(
                    records, references, theta_labels, steps_points,
                    save_path=out_dir / "convergence-vs-truncation.svg",
                ),
                plot_parity_layers(
                    parity_outcomes, save_path=out_dir / "parity-per-layer-terms.svg"
                ),
            ):
                if path is not None:
                    log(f"wrote {path}")

    below = [c for c in envelope_rows if c.needs_investigation]
    if below:
        log(
            f"\n{len(below)} envelope check(s) landed BELOW the floor at the headline "
            "depth — investigate semantics before treating those timings as meaningful"
        )
    failures = [o for o in parity_outcomes if o.ran and not o.ok]
    if failures:
        log(
            f"\n{len(failures)} parity case(s) FAILED — cross-engine timings for those "
            "cases are withheld"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
