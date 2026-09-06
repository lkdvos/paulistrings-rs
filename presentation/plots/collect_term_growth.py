"""Collect term-growth data for the talk's figure F0.

Builds the 127-qubit heavy-hex kicked-Ising circuit (`examples/common/circuits.py
::heavy_hex_kicked_ising(127, trotter_steps=5, theta_h=5*pi/16, theta_zz=-pi/2)`),
propagates the observable `Z_62` in the Heisenberg direction under a sweep of
`paulistrings.truncation.coeff(eps)` cutoffs (`eps = 2**-k`, `k = 6..14`), and
records the per-layer term count (post-truncation) from each run's
`PropagationStats`.

Also records an *untruncated* growth curve. `PauliSum.propagate_with_stats`
runs a whole `Circuit` in one non-interruptible Rust call, so there is no way
to ask it to stop mid-circuit once the sum gets large. Instead this script
exploits `Circuit.__getitem__` slicing (`circuit[:m]` returns a fresh prefix
`Circuit`) to grow `m` one channel at a time, re-propagating a fresh copy of
the observable through `circuit[:m]` at each `m`: cheap while the sum is
small, and it naturally lands on the largest `m` whose peak term count is
still under the cap, without ever running an uncontrolled blow-up. It stops
as soon as one call's `peak_terms` exceeds `UNTRUNCATED_TERM_CAP` or the
cumulative wall time of the whole untruncated sweep exceeds
`UNTRUNCATED_TIME_BUDGET_S`, and reports the last call that stayed under the
cap. (Measured on ccqlin038, 2026-09-06: the heavy-hex kicked-Ising circuit's
untruncated growth is flat for most of a Trotter step and jumps sharply at
one particular channel offset within each step — channel 63 of 271 — so the
run reaches deep into Trotter step 5 (m=1146 of 1355 channels) before the
next jump would send peak_terms from ~1e5 to ~8e7 in one channel; the whole
sweep costs well under the 60 s budget.)

Run with (from the repo root, after `maturin develop --release` and
`source .venv/bin/activate`)::

    python presentation/plots/collect_term_growth.py

Threads: `RAYON_NUM_THREADS` deliberately left unset (default Rayon pool) --
this is a term-count measurement, not a timing campaign, so thread pinning
does not matter; `wall_s` is recorded for reference only. `RUST_LOG` must
also be unset (CLAUDE.md Progress logging / Performance discipline): an
enabled DEBUG filter on `paulistrings::propagate` adds a clock read per
layer.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from common import circuits, observables  # noqa: E402
from paulistrings import truncation  # noqa: E402

OUT_PATH = _REPO_ROOT / "presentation" / "data" / (
    "term_growth.jsonl" if os.environ.get("TROTTER_STEPS", "5") == "5" else f"term_growth_{os.environ['TROTTER_STEPS']}steps.jsonl"
)

#: `k` values of the `eps = 2**-k` truncation sweep (CLAUDE.md's coeff-cutoff
#: knob, `truncation.coeff`).
K_VALUES = list(range(6, 15))

#: Per-run stop conditions for the truncated sweep (task spec: "stop early if
#: a run exceeds ~60 s or 5e6 peak terms"). Measured 2026-09-06: no k in
#: K_VALUES actually trips these (worst case k=14: peak_terms ~1.5e6, ~0.3 s),
#: so in practice every point in the grid gets run.
RUN_TIME_LIMIT_S = 60.0
RUN_PEAK_TERM_LIMIT = 5_000_000

#: Stop conditions for the untruncated growth curve (task spec: "cap by wall
#: time 60 s", "as many layers as stay under 2e6 terms").
UNTRUNCATED_TERM_CAP = 2_000_000
UNTRUNCATED_TIME_BUDGET_S = 60.0

N_QUBITS = 127
TROTTER_STEPS = int(os.environ.get("TROTTER_STEPS", "5"))
THETA_H = 5 * math.pi / 16
THETA_ZZ = -math.pi / 2
OBSERVABLE_QUBIT = 62


def _git_short_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _paulistrings_version() -> str:
    try:
        import importlib.metadata as _metadata

        return _metadata.version("paulistrings")
    except Exception:
        return "unknown"


def _provenance() -> dict:
    return {
        "host": platform.node().split(".", 1)[0],
        "date": time.strftime("%Y-%m-%d"),
        "commit": _git_short_commit(),
        "python": platform.python_version(),
        "paulistrings": _paulistrings_version(),
        "circuit": {
            "n_qubits": N_QUBITS,
            "trotter_steps": TROTTER_STEPS,
            "theta_h": THETA_H,
            "theta_zz": THETA_ZZ,
            "observable": f"Z_{OBSERVABLE_QUBIT}",
            "direction": "heisenberg",
        },
    }


def _build_circuit():
    return circuits.heavy_hex_kicked_ising(
        N_QUBITS,
        trotter_steps=TROTTER_STEPS,
        theta_h=THETA_H,
        theta_zz=THETA_ZZ,
    )


def _truncated_sweep(circuit) -> list[dict]:
    """One record per `eps = 2**-k` in `K_VALUES`, in increasing `k` order."""
    records = []
    for k in K_VALUES:
        eps = 2.0**-k
        observable = observables.single_z(OBSERVABLE_QUBIT, N_QUBITS)
        start = time.perf_counter()
        _evolved, stats = observable.propagate_with_stats(
            circuit, truncation.coeff(eps), direction="heisenberg"
        )
        wall_s = time.perf_counter() - start
        records.append(
            {
                "eps": eps,
                "k": k,
                "layers": stats.layers,
                "terms_per_layer": list(stats.terms_out),
                "peak_terms": stats.peak_terms,
                "final_terms": stats.final_terms,
                "wall_s": wall_s,
            }
        )
        print(
            f"k={k:2d} eps={eps:.6g}: layers={stats.layers} "
            f"peak_terms={stats.peak_terms} final_terms={stats.final_terms} "
            f"wall_s={wall_s:.3f}"
        )
        if wall_s > RUN_TIME_LIMIT_S or stats.peak_terms > RUN_PEAK_TERM_LIMIT:
            print(
                f"  stopping sweep early: wall_s={wall_s:.1f} or "
                f"peak_terms={stats.peak_terms} exceeded the stop condition"
            )
            break
    return records


def _untruncated_growth(circuit) -> dict:
    """Grow the prefix length `m` one channel at a time (see module docstring)."""
    total_channels = len(circuit)
    last_good: dict | None = None
    cumulative_wall_s = 0.0
    stop_reason = "reached the end of the circuit"
    m = 1
    while m <= total_channels:
        prefix = circuit[:m]
        observable = observables.single_z(OBSERVABLE_QUBIT, N_QUBITS)
        start = time.perf_counter()
        _evolved, stats = observable.propagate_with_stats(
            prefix, direction="heisenberg"
        )
        dt = time.perf_counter() - start
        cumulative_wall_s += dt

        if stats.peak_terms > UNTRUNCATED_TERM_CAP:
            stop_reason = (
                f"peak_terms {stats.peak_terms} at m={m} exceeded the "
                f"{UNTRUNCATED_TERM_CAP} cap"
            )
            break
        last_good = {
            "m": m,
            "layers": stats.layers,
            "terms_per_layer": list(stats.terms_out),
            "peak_terms": stats.peak_terms,
            "final_terms": stats.final_terms,
        }
        if cumulative_wall_s > UNTRUNCATED_TIME_BUDGET_S:
            stop_reason = (
                f"cumulative wall time {cumulative_wall_s:.1f}s exceeded the "
                f"{UNTRUNCATED_TIME_BUDGET_S}s budget"
            )
            break
        m += 1

    assert last_good is not None, "even the single-channel prefix exceeded the cap"
    print(
        f"untruncated: kept {last_good['m']}/{total_channels} channels, "
        f"peak_terms={last_good['peak_terms']}, "
        f"cumulative_wall_s={cumulative_wall_s:.2f} ({stop_reason})"
    )
    return {
        "untruncated": True,
        "layers": last_good["layers"],
        "terms_per_layer": last_good["terms_per_layer"],
        "peak_terms": last_good["peak_terms"],
        "final_terms": last_good["final_terms"],
        "wall_s": cumulative_wall_s,
        "channels_covered": last_good["m"],
        "channels_total": total_channels,
        "stop_reason": stop_reason,
        "method": (
            "propagate_with_stats cannot stop mid-circuit, so this curve was "
            "built by re-propagating circuit[:m] from scratch for increasing "
            "m (Circuit slicing) until peak_terms exceeded the cap or the "
            "time budget ran out; see module docstring"
        ),
    }


def main() -> None:
    if os.environ.get("RUST_LOG"):
        print(
            "warning: RUST_LOG is set; per-layer logging may perturb term-growth "
            "timings (CLAUDE.md Progress logging / Performance discipline)",
            file=sys.stderr,
        )

    circuit = _build_circuit()
    print(f"circuit: {len(circuit)} channels ({N_QUBITS} qubits, {TROTTER_STEPS} Trotter steps)")

    truncated_records = _truncated_sweep(circuit)
    untruncated_record = _untruncated_growth(circuit)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        f.write(f"# provenance: {json.dumps(_provenance())}\n")
        for record in truncated_records:
            f.write(json.dumps(record) + "\n")
        f.write(json.dumps(untruncated_record) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
