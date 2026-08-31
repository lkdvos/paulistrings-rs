"""Showcase B2 — noisy simulation and quantum-utility verification.

Handoff item B2; see `research/plans/2026-08-31-examples-benchmarks-suite.md`
§6 Part B for the adapted specification. The headline claim, and the one worth
checking hardest, is the *opposite* of what density-matrix methods do:

    **noise makes this simulation cheaper, not more expensive.**

A density-matrix or Kraus-channel method pays for noise: the object it carries
doubles in dimension (`rho` is `4^n`, not `2^n`) and every channel is more work
than a gate. Pauli propagation pays *negatively*. Each single-qubit
depolarizing channel is a pure coefficient rescale in the Pauli basis — a
weight-`w` string on the channel's support loses a factor `(1 - 4p/3)` per hit —
so after `d` noise layers a string's coefficient is down by
`(1 - 4p/3)^(hits)`, `hits` growing with both depth and the string's weight.
A fixed `min_abs_coeff` therefore acts as a *weight- and depth-dependent* filter
that gets sharper as `p` grows: the tracked set collapses, and the answer gets
cheaper the noisier the device is.

Five parts, run in order by `main()`:

1. **`run_noise_sweep`** — the marquee measurement. 127-qubit heavy-hex kicked
   Ising, `theta_h = 5pi/16` (Benchmark C's hard interior), 20 Trotter steps,
   `Z_62`, Heisenberg picture against `|0...0>`, and a per-gate depolarizing
   strength sweep `p in P_GRID` at a fixed `min_abs_coeff = MARQUEE_COEFF`.
   Reports peak/final term counts, wall time and `<Z_62>` per `p`, plus the
   whole dyadic-cutoff sweep behind each point — which is both the mandatory
   convergence panel (plan §7 rule 4) and the second half of the headline:
   at `p = 3e-2` the *tightest* cutoff in the grid is affordable, while at
   `p = 0` it is out of reach by two dyadic powers.
2. **`run_channel_variants`** — the same circuit with `noise.amplitude_damping`
   (the one channel here whose Heisenberg dual is not the identity map on the
   key: `Z -> (1-gamma)Z + gamma I`, fixed in commit e42095c) and with an
   asymmetric `noise.pauli_channel`, to show the collapse is a property of
   Pauli-basis noise generally, not of the depolarizing special case.
3. **`run_noiseless_limit`** — at `p = 0` exactly, and at `p = 1e-6` (the
   `p -> 0` limit), the driver must reproduce **Benchmark C's committed
   reference values**. Only C's `claimable` rows are used, and they are read out
   of `benchmarks/python/deep_trotter/summary.json` at run time rather than
   transcribed (`claimable_references`), so a change to C's labels breaks this
   check instead of silently invalidating it.
4. **`run_reachability_boundary`** — Benchmark C's clearest negative result
   (`theta_h = 7pi/32` at 20 steps: its cutoff sweep never plateaued, and the
   0.01 bar projects to ~1e10 terms and ~17 h at 32 threads) run *with* noise.
   Measured: at `p = 1e-2` and the same `2^-16` cutoff the last difference is
   9.3e-4 on 9.4e5 peak terms, against C's 1.44e-1 on 4.5e7 — 48x fewer terms
   and 154x smaller — though the strict plateau test still says *unresolved*,
   because the second-to-last difference is 1.08e-3. It also answers a
   different question (the noisy channel's expectation, not the unitary
   circuit's), and the part says both things rather than presenting it as a
   shortcut.
5. **`run_verification`** — the utility-verification framing: ingest the
   experiment's circuit (heavy-hex edge list from the checked-in
   `examples/data/heavy_hex_127.edges`), return the converged classical answer
   for a *claimable* configuration with its convergence evidence, and state
   which configurations are **not** claimable — C measured that `theta_h =
   7pi/32` at 20 steps is not resolvable to 0.01 on a workstation, and this
   showcase does not pretend otherwise.

The noise model, precisely
-------------------------
`noisy_kicked_ising` builds, per Trotter step,

    for q in 0..n-1:              rx(theta_h, q)             then  N(q)
    for (a, b) in colored E:      pauli_rotation(ZZ, ab)     then  N(a), N(b)

i.e. **one single-qubit noise channel on every qubit in the support of the gate
that just ran**, immediately after it. So a qubit takes one channel per
single-qubit gate and one per two-qubit gate it participates in:
`deg(q) + 1` channels per step, which on the Eagle lattice is 2 to 4. Channel
count per step is `2n + 3|E|` = 686 at `n = 127` (against 271 for the noiseless
circuit), one gate per channel throughout (plan §5, decision D10), so the noise
channels are *also* truncation points — which is the whole mechanism.

`p = 0` is included in the grid as `depolarize(0.0)` channels rather than as a
different circuit: `1 - 4*0/3` is exactly `1.0`, the rescale is exactly
key-preserving, and a truncation pass repeated on an unchanged sum drops
nothing new, so the `p = 0` leg is *identical* to the noiseless circuit while
keeping the channel schedule of the noisy ones.
`test_showcase_b2.py::test_p_zero_leg_matches_the_noiseless_circuit` pins that.

Running it
----------
`RAYON_NUM_THREADS=1` must be exported **before** the interpreter starts (Rayon
builds its global pool at the first propagate and never resizes it); the driver
refuses to run otherwise::

    source .venv/bin/activate
    RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py
    RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py --quick
    python examples/b2_noisy_verification/run_b2.py --figures-only

`--quick` runs the same code paths on a 20-qubit sublattice at 6 steps (a second
or two) and writes nothing; it exists so the script can be exercised without
paying for the headline. `--figures-only` re-renders the three SVGs from an
existing `summary.json`, so the committed figures are reproducible from the
committed data without repeating the ~40 minutes of propagation. The CI-safe
correctness gate — the dense noisy density-matrix cross-check at `n <= 8` and
the reference-citation check — is
`python/paulistrings/tests/test_showcase_b2.py`.

Wall times here are **indicative**. They are single-threaded and warm-free
(`warmup=False`: the tight-cutoff `p = 0` leg costs minutes, so a discarded
warm-up pass would double the whole run), taken on a shared workstation whose
single-thread campaign noise is ±5-8 % (CLAUDE.md §Performance discipline).
Term counts, expectation values and convergence outcomes are load-independent
and are the numbers to quote.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))
# Benchmark B's plateau criterion is imported, not re-implemented: B *measured*
# that the obvious "two successive values agree" test declares convergence with
# an uncertainty of exactly zero while the value is still wrong (see its
# docstring). Benchmark C reuses it for the same reason, and
# `test_showcase_b2.py` asserts this module uses B's function object.
_BENCH_DIR = str(_REPO_ROOT / "benchmarks" / "python")
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from paulistrings import Circuit  # noqa: E402

import bench_b_theta_sweep as bench_b  # noqa: E402
from common import circuits, harness, observables, report  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

#: Benchmark C's committed measurements (commits e024d8b / 01a057c). The
#: `summary.json` there is the only source of a C reference value in this file.
DEEP_TROTTER_DIR = _REPO_ROOT / "benchmarks" / "python" / "deep_trotter"

# --------------------------------------------------------------------------
# The circuit under study — Benchmark C's, with noise
# --------------------------------------------------------------------------

N_QUBITS = 127
OBSERVABLE_QUBIT = 62
THETA_ZZ = circuits.KICKED_ISING_CLIFFORD_THETA_ZZ
STATE = "z+"
DIRECTION = "heisenberg"

#: The two kick angles Benchmark C measured, on the published `k*pi/32` grid.
THETA_H: dict[str, float] = {
    "7pi/32": 7.0 * math.pi / 32.0,
    "5pi/16": 5.0 * math.pi / 16.0,
}

#: `min_abs_coeff >= 1e-12` everywhere, inherited from Benchmark B's
#: `MIN_SAFE_COEFF`: `cos(pi/2) == 6.1e-17`, not zero, so at the Clifford
#: `theta_zz` every rotation leaves a numerically-dead residual branch and an
#: untruncated 127-qubit propagation fans out without bound.
MIN_SAFE_COEFF = 1e-12


@dataclass(frozen=True)
class NoiseModel:
    """One per-gate noise channel, as a `(kind, strength)` pair.

    `kind` is a `paulistrings.noise` factory name; `strength` is that
    channel's parameter (`p`, `gamma`, or the `(px, py, pz)` triple). The
    `apply` method takes the broadcast `Circuit` methods, so one channel is
    pushed per qubit — one gate per channel, like everything else in the suite.
    """

    kind: str
    strength: Any

    def apply(self, circuit: Any, qubits: Sequence[int]) -> None:
        qubits = list(qubits)
        if self.kind == "none":
            return
        if self.kind == "depolarize":
            circuit.depolarize(float(self.strength), qubits)
        elif self.kind == "dephase":
            circuit.dephase(float(self.strength), qubits)
        elif self.kind == "amplitude_damping":
            circuit.amplitude_damping(float(self.strength), qubits)
        elif self.kind == "pauli_channel":
            px, py, pz = (float(v) for v in self.strength)
            circuit.pauli_channel(px, py, pz, qubits)
        else:
            raise ValueError(
                f"unknown noise kind {self.kind!r}; this showcase uses "
                "'none', 'depolarize', 'dephase', 'amplitude_damping' or "
                "'pauli_channel' (one channel per qubit of the preceding gate)"
            )

    @property
    def channels_per_qubit_hit(self) -> int:
        """Noise channels pushed per qubit of a gate's support (0 or 1)."""
        return 0 if self.kind == "none" else 1

    @property
    def label(self) -> str:
        if self.kind == "none":
            return "noiseless"
        if self.kind == "pauli_channel":
            px, py, pz = self.strength
            return f"pauli_channel(px={px:g}, py={py:g}, pz={pz:g})"
        return f"{self.kind}({float(self.strength):g})"

    def as_dict(self) -> dict[str, Any]:
        strength = (
            [float(v) for v in self.strength]
            if self.kind == "pauli_channel"
            else (None if self.kind == "none" else float(self.strength))
        )
        return {"noise_kind": self.kind, "noise_strength": strength}


def depolarizing(p: float) -> NoiseModel:
    """The sweep's model: single-qubit depolarizing with error probability `p`."""
    return NoiseModel("depolarize", p)


NOISELESS = NoiseModel("none", None)


def noisy_kicked_ising(
    n: int,
    trotter_steps: int,
    theta_h: float,
    model: NoiseModel,
    *,
    theta_zz: float = THETA_ZZ,
    edges: Sequence[tuple[int, int]] | None = None,
) -> Circuit:
    """The kicked-Ising Trotter circuit of `circuits.heavy_hex_kicked_ising`,
    with one `model` channel after every qubit of every gate.

    Layer order, lattice and angles are `circuits.heavy_hex_kicked_ising`'s
    defaults (`"x-then-zz"`, the colored heavy-hex edge order, `theta_zz =
    -pi/2`), so at `model = NOISELESS` this builds exactly that circuit — see
    `test_showcase_b2.py::test_noiseless_model_reproduces_the_shared_builder`.
    It is a separate builder rather than a flag on the shared one because the
    noise placement is this showcase's subject, and `examples/common/circuits.py`
    is shared infrastructure.

    Channel count is `trotter_steps * (n + |E| + m*(n + 2|E|))` with `m = 1` for
    a noise model and `0` for `NOISELESS`.
    """
    if trotter_steps < 0:
        raise ValueError(f"trotter_steps must be >= 0, got {trotter_steps}")
    lattice = (
        circuits.heavy_hex_sublattice(n)
        if edges is None
        else [(min(a, b), max(a, b)) for a, b in edges]
    )
    zz_order = [e for group in circuits.heavy_hex_edge_coloring(lattice) for e in group]

    circuit = Circuit(n)
    for _ in range(trotter_steps):
        for q in range(n):
            circuit.rx(theta_h, q)
            model.apply(circuit, [q])
        for a, b in zz_order:
            circuit.pauli_rotation("ZZ", [a, b], theta_zz)
            model.apply(circuit, [a, b])
    return circuit


def channels_per_step(n: int, num_edges: int, model: NoiseModel) -> int:
    """Channels one Trotter step pushes — the truncation schedule's length."""
    m = model.channels_per_qubit_hit
    return n + num_edges + m * (n + 2 * num_edges)


# --------------------------------------------------------------------------
# Benchmark C's committed references
# --------------------------------------------------------------------------

#: The `(theta_h_label, trotter_steps)` rows of Benchmark C this showcase is
#: allowed to be scored against. Every one must come back `claimable` from
#: `claimable_references`; the driver asserts that rather than assuming it.
CITED_C_ROWS: tuple[tuple[str, int], ...] = (
    ("7pi/32", 5),
    ("5pi/16", 5),
    ("5pi/16", 20),
)


def claimable_references(summary_path: Path | None = None) -> dict[tuple[str, int], dict]:
    """Benchmark C's `claimable` reference rows, keyed `(theta_h_label, steps)`.

    C's `summary.json` carries a reference value for every
    `(theta_h, trotter_steps)` it measured, plus — in its `time_to_accuracy`
    block — an `achieved` and a **`claimable`** flag per row. `claimable` is the
    one to read: a self-converged reference may be *achieved* by construction
    (the tightest timed run is the reference, so it agrees with itself) while
    saying nothing about the truth, and C's §3.2 documents two rows that "pass"
    that way. This loader keeps only the claimable rows, and refuses a file
    whose schema has moved rather than silently returning nothing.

    Returns, per key: `value`, `uncertainty` (`None` for an exact reference),
    `method`, `exact`, `converged`.
    """
    path = Path(summary_path) if summary_path is not None else DEEP_TROTTER_DIR / "summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Benchmark C's committed measurements are the only "
            "source of a reference value in this showcase; re-run "
            "`benchmarks/python/bench_c_deep_trotter.py` or check out the commit "
            "that carries them (e024d8b / 01a057c)."
        )
    summary = json.loads(path.read_text())
    for key in ("references", "time_to_accuracy"):
        if key not in summary:
            raise ValueError(f"{path} has no {key!r} block; C's schema has changed")

    out: dict[tuple[str, int], dict] = {}
    for row in summary["time_to_accuracy"]:
        if not row.get("claimable"):
            continue
        key = (str(row["theta_h_label"]), int(row["trotter_steps"]))
        ref_key = f"{key[0]}@{key[1]}steps"
        reference = summary["references"].get(ref_key)
        if reference is None:
            raise ValueError(f"{path}: claimable row {key} has no references[{ref_key!r}]")
        out[key] = {
            "value": float(reference["reference_value"]),
            "uncertainty": (
                None
                if reference["reference_uncertainty"] is None
                else float(reference["reference_uncertainty"])
            ),
            "method": str(reference["reference_method"]),
            "exact": bool(reference["reference_exact"]),
            "converged": row.get("reference_converged"),
            "summary_path": str(path.relative_to(_REPO_ROOT)),
        }
    if not out:
        raise ValueError(f"{path} reports no claimable row; nothing here can be scored")
    return out


def reference_tolerance(reference: dict) -> float:
    """The bar a `p -> 0` run must meet against one of C's claimable rows.

    An **exact** reference (C's 5-step causal-cone rows, cross-checked between an
    Aer statevector and an untruncated Pauli propagation) is compared against the
    engine's own truncation error at the cutoff used, so the bar is the plan's
    accuracy target, 0.01. A **self-converged** reference is only as good as its
    reported uncertainty, so that is the bar — C's §2.3 is explicit that the
    estimate is not a bound, which is why nothing tighter is claimed here.
    """
    if reference["exact"]:
        return 0.01
    uncertainty = reference["uncertainty"]
    if uncertainty is None:
        raise ValueError("a self-converged reference with no uncertainty cannot be scored")
    return float(uncertainty)


# --------------------------------------------------------------------------
# Grids, and the recorded cuts
# --------------------------------------------------------------------------

MARQUEE_THETA_LABEL = "5pi/16"
MARQUEE_STEPS = 20

#: The fixed cutoff of the headline sweep — Benchmark C's middle dyadic, the one
#: whose 20-step tracked set (3.1e6 peak terms at 7pi/32) sits dead centre of the
#: handoff's 1.2e6-9.3e6 envelope.
MARQUEE_COEFF = 2.0**-14

#: Per-gate depolarizing probabilities. `1e-3` is the order of a good
#: two-qubit gate error on hardware; `1e-2` and `3e-2` are deliberately past it,
#: to show the collapse rather than to model a device.
P_GRID: tuple[float, ...] = (0.0, 1e-3, 5e-3, 1e-2, 3e-2)

#: The dyadic cutoff sweep behind every `p`, loosest first.
COEFF_GRID: tuple[float, ...] = tuple(2.0**-k for k in (8, 10, 12, 14, 16, 18, 20))

COEFF_GRID_LABELS: dict[float, str] = {2.0**-k: f"2^-{k}" for k in range(4, 32)}


def coeff_label(eps: float) -> str:
    return COEFF_GRID_LABELS.get(eps, f"{eps:.3e}")


#: The tightest cutoff run per `p`, and why the sweep stops there. Plan §6/D15's
#: time-box policy is "pilot, project, then shrink the grid and record the cut";
#: these are a table rather than an adaptive stop, so they are reviewable and do
#: not change shape with machine load. Pilot numbers are in `MEASURED_PILOT`.
COEFF_GRID_CUTS: dict[float, tuple[float, str]] = {
    0.0: (
        2.0**-14,
        "Benchmark C measured this exact point (5pi/16, 20 steps, 2^-14) at 1.44e7 "
        "peak terms and stopped there, its plateau test satisfied; measured here at "
        "~9 min single-threaded. The next pair projects from this leg's own measured "
        "growth (13.4x in peak terms and 14.1x in wall time from 2^-12 to 2^-14) to "
        "~1.9e8 terms — ~9 GiB of columns — and ~2 h single-threaded, past this "
        "showcase's whole time box for one grid point",
    ),
    1e-3: (
        2.0**-14,
        "measured 380 s / 9.7e6 peak terms at 2^-14 -- the most expensive point in "
        "the sweep after p=0; 2^-16 projects past the p=0 cut for no extra story",
    ),
    5e-3: (
        2.0**-14,
        "measured 84 s / 2.8e6 peak terms at 2^-14; 2^-16 projects to ~10 min on "
        "the measured 7x-per-dyadic-pair growth, which the p=1e-2 leg buys more "
        "cheaply",
    ),
    1e-2: (
        2.0**-16,
        "measured 167 s / 6.4e6 peak terms at 2^-16, and it moved the value by only "
        "5.1e-5 -- one dyadic pair tighter than the p<=5e-3 legs could reach",
    ),
    3e-2: (
        2.0**-20,
        "measured 12 s / 7.4e5 peak terms at 2^-18: three dyadic pairs past the p=0 "
        "cut, for 3 % of its cost. This is the headline of the sweep",
    ),
}

#: Pilot measurements the cuts come from (ccqlin038, single-threaded,
#: `theta_h = 5pi/16` unless noted, 20 steps, 13 720 channels), recorded so the
#: projections above are auditable:
#: `(p, cutoff, value, final terms, peak terms, wall s)`.
MEASURED_PILOT: tuple[tuple[float, float, float, int, int, float], ...] = (
    (0.0, 2.0**-10, 0.014729671765, 119, 79_029, 2.45),
    (0.0, 2.0**-12, 0.015481385131, 2_543, 1_071_093, 37.92),
    (1e-3, 2.0**-14, 0.012998050638, 20_098, 9_710_246, 380.14),
    (5e-3, 2.0**-14, 0.006280876086, 3_521, 2_818_675, 83.52),
    (1e-2, 2.0**-14, 0.002432260559, 408, 869_299, 20.23),
    (1e-2, 2.0**-16, 0.002483689686, 5_063, 6_445_313, 167.25),
    (3e-2, 2.0**-14, 0.0, 0, 22_105, 0.52),
    (3e-2, 2.0**-18, 0.000077529951, 70, 738_435, 12.40),
)

#: The two non-depolarizing channels, at the marquee cutoff. `amplitude_damping`
#: is the interesting one: it is the only channel here that is not self-adjoint
#: and not key-preserving (its Heisenberg dual sends `Z -> (1-gamma)Z + gamma I`,
#: so it *fans out* while it damps), and its `apply`/`apply_adjoint` orientation
#: was wrong until commit e42095c.
CHANNEL_VARIANTS: tuple[NoiseModel, ...] = (
    NoiseModel("amplitude_damping", 1e-2),
    NoiseModel("pauli_channel", (0.002, 0.002, 0.008)),
    NoiseModel("dephase", 1e-2),
)

#: `--quick`: the same code paths on a heavy-hex sublattice small enough to run
#: in about a minute. Not a measurement — nothing is written.
QUICK_N = 20
QUICK_STEPS = 6
QUICK_COEFF_GRID = (2.0**-8, 2.0**-10, 2.0**-12)

#: The convergence verdict's tolerance. The plan's accuracy target is 0.01, so a
#: plateau resolved to 1e-3 leaves 10x headroom — the same choice, for the same
#: reason, as Benchmark C's `SELF_CONVERGENCE_TOL`.
CONVERGENCE_TOL = 1e-3

#: Term-count cap for the weight histogram: reading `x_array`/`z_array` copies
#: the columns, so the diagnostic is skipped on the multi-million-term sums where
#: it would cost more memory than the propagation.
WEIGHT_PROFILE_MAX_TERMS = 3_000_000


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def weight_stats(pauli_sum: Any) -> dict[str, Any] | None:
    """`{max_weight, mean_weight, terms}` of a `PauliSum`, or `None` if skipped.

    The Pauli weight of a row is `popcount(x | z)` over the symplectic bit
    columns — a string is non-identity on qubit `q` exactly when one of the two
    bits is set. Read-only, straight off the numpy export (plan decision D12's
    approach for B6), and skipped above `WEIGHT_PROFILE_MAX_TERMS`.
    """
    terms = len(pauli_sum)
    if terms == 0:
        return {"terms": 0, "max_weight": 0, "mean_weight": 0.0}
    if terms > WEIGHT_PROFILE_MAX_TERMS:
        return None
    x = np.asarray(pauli_sum.x_array())
    z = np.asarray(pauli_sum.z_array())
    weights = np.bitwise_count(x | z).sum(axis=1)
    return {
        "terms": terms,
        "max_weight": int(weights.max()),
        "mean_weight": float(weights.mean()),
    }


def convergence_verdict(
    values: Sequence[float], term_counts: Sequence[int], tol: float = CONVERGENCE_TOL
) -> dict[str, Any]:
    """Is a cutoff sweep's tail a plateau worth believing, and how wide?

    The criterion is Benchmark B's `_plateau_is_real`, called on B's own
    `points` shape — imported rather than re-implemented, because B *measured*
    that the obvious version of this test reports an uncertainty of exactly zero
    on a value that is still wrong (see its docstring, and C §2.1).
    `uncertainty` is the last successive difference, which C §2.3 shows is not a
    bound — it is reported as an estimate and nothing here is claimed tighter
    than the reference it is scored against.
    """
    values = [float(v) for v in values]
    deltas = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    points = [{"final_terms": int(t)} for t in term_counts]
    converged = bench_b._plateau_is_real(points, deltas, tol)
    return {
        "converged": bool(converged),
        "uncertainty": (deltas[-1] if deltas else None),
        "deltas": deltas,
        "tol": tol,
    }


# --------------------------------------------------------------------------
# Part 1 — the noise sweep
# --------------------------------------------------------------------------


def _log(message: str = "") -> None:
    print(message, flush=True)


def cutoff_grid_for(p: float, grid: Sequence[float], cuts: dict[float, tuple[float, str]]):
    """The dyadic grid for one `p`, truncated at its recorded cut."""
    tightest = cuts.get(p, (grid[-1], "no cut recorded"))[0]
    return tuple(eps for eps in grid if eps >= tightest)


def run_one(
    *,
    n: int,
    steps: int,
    theta_h: float,
    theta_h_label: str,
    model: NoiseModel,
    eps: float,
    edges: Sequence[tuple[int, int]] | None = None,
    warmup: bool = False,
) -> report.RunRecord:
    """One propagation, as a `report.RunRecord` with this showcase's extras."""
    if eps < MIN_SAFE_COEFF:
        raise ValueError(
            f"min_abs_coeff={eps:g} is below MIN_SAFE_COEFF={MIN_SAFE_COEFF:g}: at the "
            "Clifford theta_zz every rotation leaves a cos(pi/2)=6.1e-17 residual "
            "branch, so a cutoff that small does not truncate at all"
        )
    circuit = noisy_kicked_ising(n, steps, theta_h, model, edges=edges)
    # `Z_62` at the full lattice size; on a `--quick` sublattice qubit 62 may not
    # exist, so fall back to the middle one. Every reported number is at n = 127.
    seed_qubit = OBSERVABLE_QUBIT if n > OBSERVABLE_QUBIT else n // 2
    observable = observables.single_z(seed_qubit, n)
    weights: dict[str, Any] | None = None

    def contract(evolved: Any) -> complex:
        nonlocal weights
        weights = weight_stats(evolved)
        return evolved.expectation(STATE)

    record = harness.run_propagation(
        circuit,
        observable,
        harness.TruncationSpec(min_abs_coeff=eps),
        DIRECTION,
        contract=contract,
        warmup=warmup,
        threads=1,
        extra={
            "theta_h_label": theta_h_label,
            "theta_h": theta_h,
            "trotter_steps": steps,
            "channels": len(circuit),
            "coeff_label": coeff_label(eps),
            "state": STATE,
            "showcase": "b2",
            **model.as_dict(),
        },
    )
    if weights is not None:
        record.extra["weight_stats"] = weights
    return record


def run_noise_sweep(
    *,
    n: int = N_QUBITS,
    steps: int = MARQUEE_STEPS,
    theta_h_label: str = MARQUEE_THETA_LABEL,
    p_grid: Sequence[float] = P_GRID,
    coeff_grid: Sequence[float] = COEFF_GRID,
    cuts: dict[float, tuple[float, str]] | None = None,
    marquee_coeff: float = MARQUEE_COEFF,
    edges: Sequence[tuple[int, int]] | None = None,
) -> tuple[list[report.RunRecord], list[dict[str, Any]]]:
    """The marquee measurement: term count and cost against noise strength.

    Returns `(records, legs)`; one leg per `p`, carrying its whole cutoff sweep,
    its convergence verdict and the marquee-cutoff row.
    """
    cuts = COEFF_GRID_CUTS if cuts is None else cuts
    theta_h = THETA_H[theta_h_label]
    num_edges = len(circuits.heavy_hex_sublattice(n) if edges is None else list(edges))

    _log("=" * 78)
    _log("Part 1 -- noise accelerates truncation")
    _log("=" * 78)
    _log(
        f"n={n} theta_h={theta_h_label} ({theta_h:.9f}) theta_zz=-pi/2 steps={steps} "
        f"edges={num_edges} observable=Z_{OBSERVABLE_QUBIT if n > OBSERVABLE_QUBIT else n // 2} "
        f"state={STATE} direction={DIRECTION}"
    )
    _log(
        f"channels/step: {channels_per_step(n, num_edges, NOISELESS)} noiseless, "
        f"{channels_per_step(n, num_edges, depolarizing(0.0))} with per-gate noise"
    )
    _log()

    records: list[report.RunRecord] = []
    legs: list[dict[str, Any]] = []
    for p in p_grid:
        model = depolarizing(p)
        grid = cutoff_grid_for(p, coeff_grid, cuts)
        leg_records: list[report.RunRecord] = []
        _log(f"p = {p:g}   cutoffs {[coeff_label(e) for e in grid]}")
        for eps in grid:
            record = run_one(
                n=n,
                steps=steps,
                theta_h=theta_h,
                theta_h_label=theta_h_label,
                model=model,
                eps=eps,
                edges=edges,
            )
            leg_records.append(record)
            weights = record.extra.get("weight_stats") or {}
            _log(
                f"  {coeff_label(eps):>7}  <Z> = {record.expectation_value:+.12f}  "
                f"final = {record.final_terms:>10,}  peak = {record.peak_terms:>10,}  "
                f"w_max = {weights.get('max_weight', '?'):>4}  "
                f"wall = {record.propagation_time_s:8.2f} s"
            )
        records.extend(leg_records)

        verdict = convergence_verdict(
            [r.expectation_value for r in leg_records],
            [r.final_terms for r in leg_records],
        )
        marquee = next(
            (r for r in leg_records if r.truncation.get("min_abs_coeff") == marquee_coeff),
            None,
        )
        tightest = leg_records[-1]
        legs.append(
            {
                "p": p,
                "noise": model.label,
                "cutoffs": [coeff_label(e) for e in grid],
                "cutoff_values": [float(e) for e in grid],
                "tightest_cutoff": coeff_label(grid[-1]),
                "cut_reason": cuts.get(p, (None, "no cut recorded"))[1],
                "marquee_cutoff": coeff_label(marquee_coeff),
                "marquee_value": None if marquee is None else marquee.expectation_value,
                "marquee_final_terms": None if marquee is None else marquee.final_terms,
                "marquee_peak_terms": None if marquee is None else marquee.peak_terms,
                "marquee_wall_s": None if marquee is None else marquee.propagation_time_s,
                "tightest_value": tightest.expectation_value,
                "tightest_peak_terms": tightest.peak_terms,
                "values": [r.expectation_value for r in leg_records],
                "final_terms": [r.final_terms for r in leg_records],
                "peak_terms": [r.peak_terms for r in leg_records],
                "wall_s": [r.propagation_time_s for r in leg_records],
                "weight_stats": [r.extra.get("weight_stats") for r in leg_records],
                **verdict,
            }
        )
        if verdict["uncertainty"] is None:
            _log("  -> a single grid point: no successive difference to score")
        else:
            _log(
                f"  -> converged={verdict['converged']} "
                f"uncertainty={verdict['uncertainty']:.3e} at {coeff_label(grid[-1])}"
            )
        _log()

    _summarize_sweep(legs, marquee_coeff)
    return records, legs


def _summarize_sweep(legs: Sequence[dict[str, Any]], marquee_coeff: float) -> None:
    _log(f"marquee cutoff {coeff_label(marquee_coeff)} -- the headline table:")
    _log(
        f"{'p':>8} {'<Z_62>':>16} {'final terms':>13} {'peak terms':>13} "
        f"{'wall (s)':>10} {'tightest':>9} {'peak @ tightest':>16}"
    )
    for leg in legs:
        if leg["marquee_value"] is None:
            continue
        _log(
            f"{leg['p']:>8g} {leg['marquee_value']:>+16.12f} "
            f"{leg['marquee_final_terms']:>13,} {leg['marquee_peak_terms']:>13,} "
            f"{leg['marquee_wall_s']:>10.2f} {leg['tightest_cutoff']:>9} "
            f"{leg['tightest_peak_terms']:>16,}"
        )
    baseline = next((leg for leg in legs if leg["p"] == 0.0), None)
    if baseline is not None and baseline["marquee_peak_terms"]:
        _log()
        for leg in legs:
            if leg["p"] == 0.0 or leg["marquee_peak_terms"] is None:
                continue
            ratio = baseline["marquee_peak_terms"] / max(leg["marquee_peak_terms"], 1)
            _log(
                f"  p={leg['p']:g}: {ratio:,.1f}x fewer peak terms than p=0 at "
                f"{coeff_label(marquee_coeff)}"
            )
    _log()


# --------------------------------------------------------------------------
# Part 2 — the other channels
# --------------------------------------------------------------------------


def run_channel_variants(
    *,
    n: int = N_QUBITS,
    steps: int = MARQUEE_STEPS,
    theta_h_label: str = MARQUEE_THETA_LABEL,
    variants: Sequence[NoiseModel] = CHANNEL_VARIANTS,
    eps: float = MARQUEE_COEFF,
    edges: Sequence[tuple[int, int]] | None = None,
) -> list[report.RunRecord]:
    """The same circuit with amplitude damping, a general Pauli channel and
    dephasing — one run each, at the marquee cutoff."""
    _log("=" * 78)
    _log("Part 2 -- the collapse is not specific to depolarizing noise")
    _log("=" * 78)
    theta_h = THETA_H[theta_h_label]
    records = []
    for model in variants:
        record = run_one(
            n=n,
            steps=steps,
            theta_h=theta_h,
            theta_h_label=theta_h_label,
            model=model,
            eps=eps,
            edges=edges,
        )
        records.append(record)
        weights = record.extra.get("weight_stats") or {}
        _log(
            f"  {model.label:<42} <Z> = {record.expectation_value:+.12f}  "
            f"final = {record.final_terms:>10,}  peak = {record.peak_terms:>10,}  "
            f"w_max = {weights.get('max_weight', '?'):>4}  "
            f"wall = {record.propagation_time_s:8.2f} s"
        )
    _log()
    return records


# --------------------------------------------------------------------------
# Part 3 — the noiseless limit
# --------------------------------------------------------------------------

#: `p` for the "`p -> 0`" leg: small enough that its effect on `<Z_62>` is far
#: below C's reference uncertainty, large enough not to be `0.0` in disguise.
P_LIMIT = 1e-6

#: Cutoff for the 5-step legs. C measured that `2^-18` reproduces both exact
#: 5-step references to 3.9e-15 and 2.2e-16 in ~2.4 s.
NOISELESS_LIMIT_5_STEP_COEFF = 2.0**-18


def run_noiseless_limit(
    *,
    n: int = N_QUBITS,
    sweep_legs: Sequence[dict[str, Any]] | None = None,
    references: dict[tuple[str, int], dict] | None = None,
    edges: Sequence[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """At `p = 0` (and `p = P_LIMIT`), reproduce Benchmark C's claimable rows.

    The 5-step rows are re-run here at `2^-18` because they are cheap and their
    references are *exact*. The 20-step row reuses the `p = 0` leg of
    `run_noise_sweep` when one is supplied, rather than paying for the most
    expensive propagation in this showcase twice.
    """
    references = claimable_references() if references is None else references
    _log("=" * 78)
    _log("Part 3 -- the noiseless limit recovers Benchmark C")
    _log("=" * 78)
    missing = [row for row in CITED_C_ROWS if row not in references]
    if missing:
        raise ValueError(
            f"Benchmark C's summary.json no longer reports {missing} as claimable; "
            "this showcase may only cite claimable rows (see claimable_references)"
        )
    _log(f"references: {references[CITED_C_ROWS[0]]['summary_path']} (claimable rows only)")

    rows: list[dict[str, Any]] = []
    for theta_h_label, steps in CITED_C_ROWS:
        reference = references[(theta_h_label, steps)]
        bar = reference_tolerance(reference)
        if steps == MARQUEE_STEPS:
            # Never re-run the deep row here: at `NOISELESS_LIMIT_5_STEP_COEFF`
            # a 20-step noiseless propagation is hours of work — that is two
            # dyadic pairs past the sweep's p=0 cut, which already costs ~9 min
            # on its own — so the sweep's p=0 leg is required, not optional.
            leg = next((lg for lg in (sweep_legs or ()) if lg["p"] == 0.0), None)
            if leg is None:
                raise ValueError(
                    f"the {MARQUEE_STEPS}-step row must reuse run_noise_sweep's p=0 "
                    "leg; re-running it at a tighter cutoff here would cost hours"
                )
            value = leg["tightest_value"]
            source = f"p=0 leg of the sweep at {leg['tightest_cutoff']}"
            eps = None
            wall = sum(leg["wall_s"])
            p_used = 0.0
        else:
            eps = NOISELESS_LIMIT_5_STEP_COEFF
            record = run_one(
                n=n,
                steps=steps,
                theta_h=THETA_H[theta_h_label],
                theta_h_label=theta_h_label,
                model=depolarizing(0.0),
                eps=eps,
                edges=edges,
            )
            value = record.expectation_value
            source = f"p=0 run at {coeff_label(eps)}"
            wall = record.propagation_time_s
            p_used = 0.0
        gap = abs(value - reference["value"])
        rows.append(
            {
                "theta_h_label": theta_h_label,
                "trotter_steps": steps,
                "p": p_used,
                "source": source,
                "cutoff": None if eps is None else coeff_label(eps),
                "value": value,
                "reference": reference["value"],
                "reference_method": reference["method"],
                "reference_exact": reference["exact"],
                "reference_uncertainty": reference["uncertainty"],
                "bar": bar,
                "gap": gap,
                "agrees": gap <= bar,
                "wall_s": wall,
            }
        )
        _log(
            f"  {theta_h_label:>7} {steps:>3} steps  {source:<34} "
            f"value = {value:+.12f}  C = {reference['value']:+.12f}  "
            f"gap = {gap:.3e}  bar = {bar:.3e}  "
            f"{'OK' if gap <= bar else 'FAIL'}"
        )

    failures = [row for row in rows if not row["agrees"]]
    if failures:
        raise AssertionError(
            "the noiseless limit does not reproduce Benchmark C's claimable rows: "
            + "; ".join(
                f"{r['theta_h_label']}@{r['trotter_steps']}: gap {r['gap']:.3e} > "
                f"bar {r['bar']:.3e}"
                for r in failures
            )
        )
    _log()
    return rows


def run_p_to_zero(
    *,
    n: int = N_QUBITS,
    theta_h_label: str = MARQUEE_THETA_LABEL,
    steps: int = 5,
    p_values: Sequence[float] = (0.0, P_LIMIT, 1e-4),
    eps: float = NOISELESS_LIMIT_5_STEP_COEFF,
    references: dict[tuple[str, int], dict] | None = None,
    edges: Sequence[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """`<Z_62>` against an *exact* reference as `p` is taken to zero.

    The 5-step rung is the only place in this showcase where a noisy answer can
    be scored against an exact number, because it is the only depth where an
    exact reference exists at all (C §2: the commutation-aware backward cone of
    `Z_62` is 19 qubits at 5 steps and the whole lattice by 9).
    """
    references = claimable_references() if references is None else references
    reference = references[(theta_h_label, steps)]
    _log(f"  p -> 0 at {theta_h_label}, {steps} steps, {coeff_label(eps)} "
         f"(exact reference {reference['value']:+.12f}):")
    rows = []
    for p in p_values:
        record = run_one(
            n=n,
            steps=steps,
            theta_h=THETA_H[theta_h_label],
            theta_h_label=theta_h_label,
            model=depolarizing(p),
            eps=eps,
            edges=edges,
        )
        gap = abs(record.expectation_value - reference["value"])
        rows.append(
            {
                "p": p,
                "value": record.expectation_value,
                "gap_vs_exact": gap,
                "final_terms": record.final_terms,
                "peak_terms": record.peak_terms,
                "wall_s": record.propagation_time_s,
            }
        )
        _log(
            f"    p={p:<8g} <Z> = {record.expectation_value:+.12f}  "
            f"|gap| = {gap:.3e}  final = {record.final_terms:>9,}  "
            f"wall = {record.propagation_time_s:7.2f} s"
        )
    gaps = [row["gap_vs_exact"] for row in rows]
    if gaps[0] > gaps[-1]:
        raise AssertionError(
            f"the p=0 leg must be at least as close to the exact reference as the "
            f"noisiest one: gaps {gaps}"
        )
    _log()
    return rows


# --------------------------------------------------------------------------
# Part 4 — the reachability boundary, with and without noise
# --------------------------------------------------------------------------

#: Benchmark C's clearest negative result: `theta_h = 7pi/32` at 20 steps. Its
#: reference sweep reached 3.9e7 terms at `2^-16` and the value still swung by
#: 1.44e-1 on the last tightening, and C projects `2^-20`-`2^-22`, ~1e10 terms
#: and ~17 h at 32 threads to reach the plan's 0.01 bar (C README §3.1, §3.3).
REACHABILITY_THETA_LABEL = "7pi/32"
REACHABILITY_P = 1e-2
REACHABILITY_COEFF_GRID: tuple[float, ...] = (2.0**-10, 2.0**-12, 2.0**-14, 2.0**-16)


def run_reachability_boundary(
    *,
    n: int = N_QUBITS,
    steps: int = MARQUEE_STEPS,
    theta_h_label: str = REACHABILITY_THETA_LABEL,
    p: float = REACHABILITY_P,
    coeff_grid: Sequence[float] = REACHABILITY_COEFF_GRID,
    edges: Sequence[tuple[int, int]] | None = None,
) -> tuple[dict[str, Any], list[report.RunRecord]]:
    """The noiseless-unreachable point of Benchmark C, run *with* noise.

    C measured that the noiseless answer at `theta_h = 7pi/32`, 20 steps is out
    of reach: its cutoff sweep never plateaued and the projected cost of the
    plan's 0.01 bar is ~1e10 terms. Adding per-gate depolarizing noise at
    `p = 1e-2` moves the *same circuit's* sweep a long way towards resolution —
    measured, at the same `2^-16` cutoff: last difference 9.3e-4 on 9.4e5 peak
    terms in 50 s, against C's 1.44e-1 on 4.5e7 peak terms (3.9e7 of them still
    resident at the end) in 276 s at 16 threads. It does **not** cross the
    plateau test: the second-to-last difference is 1.08e-3, just over
    `CONVERGENCE_TOL`, so the verdict stays `converged = false` and the value
    stays unclaimable. One further dyadic pair (`2^-18`) would very likely
    settle it, at a projected ~6 min; it was cut, because the comparison this
    part exists to make is the one above and does not need it.

    **Nor does it resolve C's question.** It answers a different one: the
    expectation of `Z_62` under the *noisy* channel, not under the unitary
    circuit. The two agree only as `p -> 0`, and at `p = 1e-2` after 20 steps
    they are far apart. The point of running it is that this is the regime a
    hardware experiment is actually in — a noisy device's answer is the one a
    verification claim is about — and that regime is the cheap one here.
    """
    _log("=" * 78)
    _log("Part 4 -- the reachability boundary: C's unresolvable point, with noise")
    _log("=" * 78)
    theta_h = THETA_H[theta_h_label]
    records = []
    for eps in coeff_grid:
        record = run_one(
            n=n,
            steps=steps,
            theta_h=theta_h,
            theta_h_label=theta_h_label,
            model=depolarizing(p),
            eps=eps,
            edges=edges,
        )
        records.append(record)
        _log(
            f"  {coeff_label(eps):>7}  <Z> = {record.expectation_value:+.12f}  "
            f"final = {record.final_terms:>10,}  peak = {record.peak_terms:>10,}  "
            f"wall = {record.propagation_time_s:8.2f} s"
        )
    verdict = convergence_verdict(
        [r.expectation_value for r in records], [r.final_terms for r in records]
    )
    _log(
        f"  -> converged={verdict['converged']} "
        f"uncertainty={verdict['uncertainty']:.3e} at {coeff_label(coeff_grid[-1])}"
    )
    _log(
        "  noiseless, the same point is NOT resolvable: Benchmark C's sweep swung by "
        "1.44e-1 on its last tightening (3.9e7 terms) and projects ~1e10 terms for "
        "the 0.01 bar. The noisy answer above is a different quantity, not a "
        "cheaper route to C's."
    )
    _log()
    return {
        "theta_h_label": theta_h_label,
        "trotter_steps": steps,
        "p": p,
        "cutoffs": [coeff_label(e) for e in coeff_grid],
        "cutoff_values": [float(e) for e in coeff_grid],
        "values": [r.expectation_value for r in records],
        "final_terms": [r.final_terms for r in records],
        "peak_terms": [r.peak_terms for r in records],
        "wall_s": [r.propagation_time_s for r in records],
        "noiseless_status": (
            "not claimable -- Benchmark C's reference sweep never plateaued here "
            "(uncertainty 1.44e-1 at 2^-16, 3.9e7 terms) and projects ~1e10 terms / "
            "~560 GiB / ~17 h at 32 threads for the plan's 0.01 bar"
        ),
        **verdict,
    }, records


# --------------------------------------------------------------------------
# Part 5 — the verification framing
# --------------------------------------------------------------------------


def run_verification(
    *,
    sweep_legs: Sequence[dict[str, Any]],
    limit_rows: Sequence[dict[str, Any]],
    reachability: dict[str, Any] | None = None,
    references: dict[tuple[str, int], dict] | None = None,
) -> dict[str, Any]:
    """The utility-verification statement, claimable and non-claimable halves.

    Nothing is computed here that was not computed above; this is the part that
    says out loud *which* question the showcase answers and which it does not.
    """
    references = claimable_references() if references is None else references
    _log("=" * 78)
    _log("Part 5 -- what this verifies, and what it does not")
    _log("=" * 78)

    claimable_20 = references[(MARQUEE_THETA_LABEL, MARQUEE_STEPS)]
    row_20 = next(
        (
            row
            for row in limit_rows
            if row["trotter_steps"] == MARQUEE_STEPS
            and row["theta_h_label"] == MARQUEE_THETA_LABEL
        ),
        None,
    )
    noisy_legs = [leg for leg in sweep_legs if leg["p"] > 0.0]
    resolved = [leg for leg in noisy_legs if leg["converged"]]

    out = {
        "circuit_source": str(circuits.HEAVY_HEX_127_PATH.relative_to(_REPO_ROOT)),
        "claimable_configuration": {
            "theta_h": MARQUEE_THETA_LABEL,
            "trotter_steps": MARQUEE_STEPS,
            "noise": "p = 0 (noiseless)",
            "value": None if row_20 is None else row_20["value"],
            "reference": claimable_20["value"],
            "reference_method": claimable_20["method"],
            "reference_uncertainty": claimable_20["uncertainty"],
        },
        "resolved_noisy_legs": [
            {"p": leg["p"], "value": leg["tightest_value"], "uncertainty": leg["uncertainty"],
             "cutoff": leg["tightest_cutoff"]}
            for leg in resolved
        ],
        "unresolved_noisy_legs": [
            {"p": leg["p"], "uncertainty": leg["uncertainty"], "cutoff": leg["tightest_cutoff"]}
            for leg in noisy_legs
            if not leg["converged"]
        ],
        "not_claimable": {
            "configuration": "theta_h = 7pi/32, 20 Trotter steps, noiseless",
            "finding": (
                "Benchmark C's reference sweep reached 3.9e7 terms at 2^-16 and the "
                "value still moved by 1.44e-1 on the last tightening; C projects "
                "~1e10 terms / ~560 GiB / ~17 h at 32 threads to reach the plan's "
                "0.01 bar there. C reports it as not claimable and so does this "
                "showcase -- see benchmarks/python/deep_trotter/README.md §3.1-3.3."
            ),
            "with_noise": (
                None
                if reachability is None
                else {
                    "p": reachability["p"],
                    "converged": reachability["converged"],
                    "value": reachability["values"][-1],
                    "uncertainty": reachability["uncertainty"],
                    "caveat": (
                        "this is the noisy channel's expectation, a different "
                        "quantity from the unitary circuit's -- not a cheaper route "
                        "to C's number"
                    ),
                }
            ),
        },
    }

    _log(f"  circuit ingested from {out['circuit_source']} (144 edges, generated, not typed)")
    if row_20 is not None:
        _log(
            f"  claimable: theta_h={MARQUEE_THETA_LABEL}, {MARQUEE_STEPS} steps, "
            f"noiseless -> <Z_62> = {row_20['value']:+.12f}"
        )
        _log(
            f"             C's claimable reference {claimable_20['value']:+.12f} "
            f"+- {claimable_20['uncertainty']:.2e} ({claimable_20['method']}), "
            f"gap {row_20['gap']:.3e}"
        )
    for leg in resolved:
        _log(
            f"  claimable: p={leg['p']:g} -> <Z_62> = {leg['tightest_value']:+.12f} "
            f"+- {leg['uncertainty']:.2e} (plateau at {leg['tightest_cutoff']})"
        )
    for leg in noisy_legs:
        if not leg["converged"]:
            _log(
                f"  NOT claimable: p={leg['p']:g} -- the cutoff sweep never plateaued "
                f"(last difference {leg['uncertainty']:.2e} at {leg['tightest_cutoff']})"
            )
    _log(f"  NOT claimable: {out['not_claimable']['configuration']}")
    _log(f"                 {out['not_claimable']['finding']}")
    if reachability is not None and reachability["converged"]:
        _log(
            f"                 with p={reachability['p']:g} the same circuit's sweep "
            f"*does* plateau ({reachability['values'][-1]:+.12f} +- "
            f"{reachability['uncertainty']:.2e}) -- a different quantity, see Part 4"
        )
    _log()
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
_GRID_COLOR = "#e1e0d9"
_MUTED = "#898781"


def _style(ax) -> None:
    ax.grid(True, color=_GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED)


def _p_axis(values: Sequence[float]) -> list[float]:
    """`p = 0` on a log axis: plotted at a decade below the smallest nonzero p."""
    nonzero = [v for v in values if v > 0]
    floor = min(nonzero) / 10.0 if nonzero else 1e-4
    return [floor if v == 0 else v for v in values]


def plot_terms_and_time_vs_noise(legs: Sequence[dict[str, Any]], save_path: Path) -> None:
    """The headline figure: tracked-set size and wall time against noise strength."""
    import matplotlib.pyplot as plt

    rows = [leg for leg in legs if leg["marquee_peak_terms"] is not None]
    ps = [leg["p"] for leg in rows]
    xs = _p_axis(ps)
    fig, (ax_terms, ax_time) = plt.subplots(1, 2, figsize=(9.5, 4))

    ax_terms.plot(xs, [leg["marquee_peak_terms"] for leg in rows], marker="o", markersize=5,
                  linewidth=1.6, color=_PALETTE[0], label="peak resident terms")
    # A leg whose sum empties out has no place on a log axis: plot the nonzero
    # points and say so in words, rather than drawing a fictitious "0.5 terms".
    finite = [(x, leg) for x, leg in zip(xs, rows) if leg["marquee_final_terms"] > 0]
    if finite:
        ax_terms.plot([x for x, _ in finite], [lg["marquee_final_terms"] for _, lg in finite],
                      marker="s", markersize=5, linewidth=1.6, color=_PALETTE[1],
                      linestyle="--", label="final terms")
    for x, leg in zip(xs, rows):
        if leg["marquee_final_terms"] == 0:
            ax_terms.annotate(
                "final = 0\n(sum emptied)", xy=(x, leg["marquee_peak_terms"]),
                xytext=(0, -28), textcoords="offset points", ha="center", fontsize=8,
                color=_MUTED,
            )
    ax_terms.set_xscale("log")
    ax_terms.set_yscale("log")
    ax_terms.set_xlabel("per-gate depolarizing probability p")
    ax_terms.set_ylabel("Pauli strings")
    ax_terms.legend(frameon=False)
    _style(ax_terms)

    ax_time.plot(xs, [leg["marquee_wall_s"] for leg in rows], marker="o", markersize=5,
                 linewidth=1.6, color=_PALETTE[2])
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("per-gate depolarizing probability p")
    ax_time.set_ylabel("wall time, 1 thread (s)")
    _style(ax_time)

    for ax in (ax_terms, ax_time):
        ax.set_xticks(xs)
        ax.set_xticklabels(["0" if p == 0 else f"{p:g}" for p in ps])
        ax.minorticks_off()

    fig.suptitle(
        f"noise accelerates truncation -- 127q kicked Ising, {MARQUEE_STEPS} steps, "
        f"min_abs_coeff = {coeff_label(MARQUEE_COEFF)}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")


def plot_observable_decay(legs: Sequence[dict[str, Any]], save_path: Path) -> None:
    """`<Z_62>` against `p`, with the cutoff sweep's spread as the error bar."""
    import matplotlib.pyplot as plt

    rows = [leg for leg in legs if leg["marquee_value"] is not None]
    ps = [leg["p"] for leg in rows]
    xs = _p_axis(ps)
    values = [leg["tightest_value"] for leg in rows]
    errs = [leg["uncertainty"] or 0.0 for leg in rows]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(xs, values, yerr=errs, marker="o", markersize=5, linewidth=1.6,
                color=_PALETTE[0], ecolor=_MUTED, capsize=3,
                label="tightest affordable cutoff")
    ax.plot(xs, [leg["marquee_value"] for leg in rows], marker="s", markersize=4,
            linewidth=1.2, linestyle="--", color=_PALETTE[1],
            label=f"fixed {coeff_label(MARQUEE_COEFF)}")
    ax.set_xscale("log")
    # symlog, not log: the fixed-cutoff curve is exactly 0 wherever the tracked
    # set emptied out, and the signal spans 2-3 decades across the grid.
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylim(bottom=0.0)  # every value here is >= 0; drop symlog's mirror half
    ax.set_xticks(xs)
    ax.set_xticklabels(["0" if p == 0 else f"{p:g}" for p in ps])
    ax.minorticks_off()
    ax.set_xlabel("per-gate depolarizing probability p")
    ax.set_ylabel(f"<Z_{OBSERVABLE_QUBIT}> after {MARQUEE_STEPS} steps (symlog)")
    for x, leg in zip(xs, rows):
        if leg["marquee_value"] == 0.0:
            ax.annotate(
                f"0 at {coeff_label(MARQUEE_COEFF)}", xy=(x, 0.0), xytext=(-8, 10),
                textcoords="offset points", ha="right", fontsize=8, color=_MUTED,
            )
    ax.legend(frameon=False, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")


def plot_convergence_vs_cutoff(legs: Sequence[dict[str, Any]], save_path: Path) -> None:
    """Plan §7 rule 4's convergence panel, one curve per `p`, plus the cutoff
    reach the collapse buys."""
    import matplotlib.pyplot as plt

    fig, (ax_value, ax_terms) = plt.subplots(1, 2, figsize=(9.5, 4))
    for index, leg in enumerate(legs):
        color = _PALETTE[index % len(_PALETTE)]
        xs = leg["cutoff_values"]
        label = f"p = {leg['p']:g}" + ("" if leg["converged"] else " (unresolved)")
        style = "-" if leg["converged"] else "--"
        ax_value.plot(xs, leg["values"], marker="o", markersize=4, linewidth=1.5,
                      linestyle=style, color=color, label=label)
        ax_terms.plot(xs, [max(t, 1) for t in leg["peak_terms"]], marker="o", markersize=4,
                      linewidth=1.5, linestyle=style, color=color, label=label)

    for ax in (ax_value, ax_terms):
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("min_abs_coeff (tighter to the right)")
        _style(ax)
    ax_value.axvline(MARQUEE_COEFF, color=_MUTED, linewidth=1.0, linestyle=":")
    # symlog so all five legs are legible at once: <Z_62> spans 0.016 down to
    # 7.8e-5 across the grid, and is exactly 0 where a leg's sum emptied out.
    ax_value.set_yscale("symlog", linthresh=1e-5)
    ax_value.set_ylim(bottom=0.0)
    ax_value.set_ylabel(f"<Z_{OBSERVABLE_QUBIT}> (symlog)")
    # One legend for both panels: they share the color assignment, and on the
    # left there is no free corner that does not sit on a curve.
    ax_terms.set_yscale("log")
    ax_terms.set_ylabel("peak resident terms")
    ax_terms.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "convergence, and the cutoff reach noise buys "
        f"(127q, {MARQUEE_STEPS} steps, theta_h = {MARQUEE_THETA_LABEL})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")


def legs_from_summary(summary_path: Path) -> list[dict[str, Any]]:
    """The sweep legs of a previous run, for `--figures-only`.

    `summary.json` carries everything the three figures read, so a figure can be
    restyled and re-rendered from a recorded run without paying for the ~40 min
    of propagation again — which is also what keeps the committed SVGs
    reproducible from the committed data.
    """
    summary = json.loads(Path(summary_path).read_text())
    if "legs" not in summary:
        raise ValueError(f"{summary_path} has no 'legs' block; it is not a B2 summary")
    return list(summary["legs"])


def write_figures(legs: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    paths = []
    for name, fn in (
        ("terms-and-time-vs-noise.svg", plot_terms_and_time_vs_noise),
        ("observable-decay-vs-noise.svg", plot_observable_decay),
        ("convergence-vs-cutoff.svg", plot_convergence_vs_cutoff),
    ):
        path = out_dir / name
        fn(legs, path)
        paths.append(path)
        _log(f"wrote {path}")
    return paths


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"run the same code paths on a {QUICK_N}-qubit sublattice at "
        f"{QUICK_STEPS} steps and write nothing (about a minute)",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="re-render the figures from an existing summary.json and exit "
        "(no propagation, no thread pin needed)",
    )
    args = parser.parse_args(argv)

    if args.figures_only:
        out_dir = Path(args.out_dir)
        write_figures(legs_from_summary(out_dir / "summary.json"), out_dir)
        return 0

    if os.environ.get("RAYON_NUM_THREADS") != "1":
        parser.error(
            "export RAYON_NUM_THREADS=1 before starting the interpreter: Rayon builds "
            "its global pool at the first propagate and never resizes it, so setting "
            "it from inside the process is too late"
        )
    harness.assert_single_threaded()
    harness.assert_logging_quiet()

    started = time.perf_counter()
    references = claimable_references()

    if args.quick:
        n, steps = QUICK_N, QUICK_STEPS
        edges = circuits.heavy_hex_sublattice(n)
        _log(f"--quick: n={n}, {steps} steps, cutoffs "
             f"{[coeff_label(e) for e in QUICK_COEFF_GRID]}; nothing is written")
        cuts = {p: (QUICK_COEFF_GRID[-1], "quick run") for p in P_GRID}
        records, legs = run_noise_sweep(
            n=n, steps=steps, coeff_grid=QUICK_COEFF_GRID, cuts=cuts,
            marquee_coeff=QUICK_COEFF_GRID[-1], edges=edges,
        )
        run_channel_variants(n=n, steps=steps, eps=QUICK_COEFF_GRID[-1], edges=edges)
        run_reachability_boundary(
            n=n, steps=steps, coeff_grid=QUICK_COEFF_GRID, edges=edges
        )
        # Parts 3 and 5 are skipped: they are scored against Benchmark C's
        # 127-qubit reference values, which say nothing about a 20-qubit
        # sublattice. `claimable_references()` above still ran, so the citation
        # path is exercised.
        _log(f"quick run done in {time.perf_counter() - started:.1f} s "
             f"({len(records)} records, not written)")
        return 0

    records, legs = run_noise_sweep()
    variant_records = run_channel_variants()
    limit_rows = run_noiseless_limit(sweep_legs=legs, references=references)
    p_to_zero = run_p_to_zero(references=references)
    reachability, reachability_records = run_reachability_boundary()
    verification = run_verification(
        sweep_legs=legs,
        limit_rows=limit_rows,
        reachability=reachability,
        references=references,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records = records + variant_records + reachability_records
    results_path = out_dir / "results.json"
    if results_path.exists():
        results_path.unlink()
    report.write_results(all_records, out_dir, name="results")
    _log(f"wrote {results_path}")

    wall = time.perf_counter() - started
    summary = {
        "showcase": "b2",
        "n_qubits": N_QUBITS,
        "observable": f"Z_{OBSERVABLE_QUBIT}",
        "theta_zz": THETA_ZZ,
        "state": STATE,
        "direction": DIRECTION,
        "theta_h": {k: v for k, v in THETA_H.items()},
        "marquee": {
            "theta_h_label": MARQUEE_THETA_LABEL,
            "trotter_steps": MARQUEE_STEPS,
            "min_abs_coeff": MARQUEE_COEFF,
            "min_abs_coeff_label": coeff_label(MARQUEE_COEFF),
        },
        "noise_model": (
            "one single-qubit channel on every qubit of the gate that just ran; "
            f"channels/step = 2n + 3|E| = "
            f"{channels_per_step(N_QUBITS, 144, depolarizing(0.0))}"
        ),
        "p_grid": list(P_GRID),
        "coeff_grid": list(COEFF_GRID),
        "coeff_grid_labels": [coeff_label(e) for e in COEFF_GRID],
        "coeff_grid_cuts": {
            str(p): {"tightest": coeff_label(eps), "reason": reason}
            for p, (eps, reason) in COEFF_GRID_CUTS.items()
        },
        "convergence_tol": CONVERGENCE_TOL,
        "min_safe_coeff": MIN_SAFE_COEFF,
        "measured_pilot": [
            {
                "p": p,
                "cutoff": coeff_label(eps),
                "value": value,
                "final_terms": final,
                "peak_terms": peak,
                "wall_s": wall_s,
            }
            for p, eps, value, final, peak, wall_s in MEASURED_PILOT
        ],
        "cited_benchmark_c_rows": [
            {"theta_h_label": t, "trotter_steps": s, **references[(t, s)]}
            for t, s in CITED_C_ROWS
        ],
        "legs": legs,
        "channel_variants": [
            {
                "noise": r.extra["noise_kind"],
                "strength": r.extra["noise_strength"],
                "cutoff": r.extra["coeff_label"],
                "value": r.expectation_value,
                "final_terms": r.final_terms,
                "peak_terms": r.peak_terms,
                "wall_s": r.propagation_time_s,
                "weight_stats": r.extra.get("weight_stats"),
            }
            for r in variant_records
        ],
        "noiseless_limit": limit_rows,
        "p_to_zero": p_to_zero,
        "reachability_boundary": reachability,
        "verification": verification,
        "wall_clock_s": wall,
        "provenance": report.collect_provenance(
            thread_count=1, repo_root=_REPO_ROOT
        ).__dict__,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    _log(f"wrote {summary_path}")

    if not args.no_figures:
        write_figures(legs, out_dir)

    _log()
    _log(f"done in {wall / 60.0:.1f} min ({len(all_records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
