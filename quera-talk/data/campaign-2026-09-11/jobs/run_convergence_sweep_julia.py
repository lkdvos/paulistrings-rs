"""Per-Trotter-step PauliPropagation.jl expectation-value sweep, for campaign-2026-09-11.

The Rust-side `run_convergence_sweep.py` gets a real per-step `<O>` trajectory
"for free" by slicing `Circuit`. Julia has no such shortcut: `runner.jl`'s new
`PP_LAYER_EXPECTATION` diagnostic (see its module header comment) must
genuinely re-propagate every growing circuit prefix from scratch, because a
longer Heisenberg-picture prefix prepends new gates at the *innermost*
conjugation position -- a longer prefix's result is provably not an extension
of a shorter prefix's, not merely an API gap. Checked against
`PauliPropagation.jl` 0.8.2's own `propagate!`/`propagate!(::
AbstractPauliPropagationCache, ...)` source before writing this driver.

This makes the sweep's total cost roughly `sum_k cost(prefix_k)`, not
`n_steps * cost(final)` -- expensive but, per real Rust term-count data
(`raw/2026-09-14-worker7172-convergence/convergence.jsonl`), the term-count
trajectory saturates well before step 20 for the achievable cutoffs, so the
total is a single-digit multiple of one full run's cost, not `n_steps`-times
worse. See `decisions.md`'s Julia-convergence entry for the real/estimated
per-cutoff cost this grid was chosen against; `eps=2^-18` is excluded because
even the RUST engine's own single-thread full run at that cutoff has twice
failed to complete inside an 8h wall cap (job-ledger.jsonl 7030090/7031789),
and Julia is ~3x slower per gate single-threaded (decisions.md #10/#27).

One runner.jl PROCESS per cutoff (not per step): `PP_TROTTER_STEPS` drives an
internal loop over all `n_steps` prefixes in one Julia invocation, so JIT
compilation of the specialized propagate methods is paid once per cutoff, not
once per (cutoff, step). `PP_WARM_REPEATS=0` and `PP_LAYER_COUNTS=0` skip the
runner's own (redundant, for this driver's purposes) full-circuit warm-repeat
and per-gate-term-count passes -- neither is needed here and both cost
roughly one more full propagation each.

CLI: `python run_convergence_sweep_julia.py --out-dir <dir> --min-abs-coeff <eps> [<eps> ...]`.
Writes one JSONL record per (min_abs_coeff, trotter_step) point to
`<out-dir>/convergence_julia.jsonl` -- `JULIA_CONVERGENCE_FIELDS` reuses every
field name from `run_convergence_sweep.py`'s `CONVERGENCE_FIELDS` whose
meaning matches, plus `engine`/`runtime_version` to identify the Julia leg
when the two files are concatenated for plotting.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

_JOBS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _JOBS_DIR.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "examples"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "python"))
sys.path.insert(0, str(_JOBS_DIR))

from common import oracles  # noqa: E402
import julia_baseline  # noqa: E402
import preflight  # noqa: E402
import run_cell  # noqa: E402
from run_cell import CellSpec, _git_provenance  # noqa: E402
from run_cell_julia import (  # noqa: E402
    _jl_min_abs_coeff,
    _julia_compiler_version,
    _pauli_propagation_runtime_version,
    _slurm_job_id,
)

CONVERGENCE_SCHEMA_VERSION = 1
CAMPAIGN_ID = "campaign-2026-09-11"

#: Field order of one convergence-point record. Shares every name with
#: `run_convergence_sweep.py`'s `CONVERGENCE_FIELDS` whose meaning matches
#: (deliberately NOT `analysis/schema.py`'s `RUN_FIELDS`: one row is one
#: (cutoff, step) point on a curve, not a campaign cell, exactly as for the
#: Rust-side sweep).
JULIA_CONVERGENCE_FIELDS = (
    "schema_version", "campaign_id", "sweep_id", "task_id", "source_commit",
    "dirty", "compiler_version", "runtime_version", "slurm_job_id",
    "node_class", "hardware_valid", "engine", "n_qubits", "theta_h",
    "trotter_steps", "direction", "state", "observable", "min_abs_coeff",
    "trotter_step", "expectation_re", "expectation_im", "final_terms",
    "wall_time_s", "status", "failure_reason",
)

ENGINE = "pauli_propagation_jl"


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def run_sweep(
    *,
    out_dir: Path,
    min_abs_coeffs: list[float],
    n_qubits: int = 127,
    trotter_steps: int = 20,
    theta_h: float = 0.6872233929727672,
    direction: str = "heisenberg",
    state: str = "z+",
    observable_name: str = "canonical_z_127",
    max_weight: int | None = None,
    timeout: float = 86400.0,
) -> list[dict[str, Any]]:
    """Run the full (min_abs_coeff x trotter_step) grid through PauliPropagation.jl.

    One `runner.jl` process per `min_abs_coeff` (never per step -- see module
    docstring); every record from that process's `per_layer_expectation` is
    appended before moving to the next cutoff, so a later cutoff's crash or
    timeout does not lose earlier cutoffs' real data.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_id = str(uuid.uuid4())
    commit, dirty = _git_provenance()
    common = {
        "schema_version": CONVERGENCE_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "sweep_id": sweep_id,
        "task_id": "T02-canonical",
        "source_commit": commit,
        "dirty": dirty,
        "compiler_version": _julia_compiler_version(),
        "runtime_version": _pauli_propagation_runtime_version(),
        "slurm_job_id": _slurm_job_id(),
        "engine": ENGINE,
        "n_qubits": n_qubits,
        "theta_h": theta_h,
        "trotter_steps": trotter_steps,
        "direction": direction,
        "state": state,
        "observable": observable_name,
    }

    hardware = preflight.run_preflight()
    if not hardware["preflight_passed"]:
        record = {
            **common,
            "node_class": hardware["node_class_guess"],
            "hardware_valid": False,
            "min_abs_coeff": None,
            "trotter_step": None,
            "expectation_re": None,
            "expectation_im": None,
            "final_terms": None,
            "wall_time_s": None,
            "status": "invalid_hardware",
            "failure_reason": (
                f"preflight failed: node_class_guess={hardware['node_class_guess']!r} "
                f"(contract wants {preflight.HARDWARE_CONTRACT_ID!r}), "
                f"smt_siblings_present={hardware['smt_siblings_present']}, "
                f"affinity_is_physical_core_only={hardware['affinity_is_physical_core_only']}"
            ),
        }
        assert set(record) == set(JULIA_CONVERGENCE_FIELDS)
        _append_jsonl(out_dir / "convergence_julia.jsonl", [record])
        return [record]

    skip_reason = julia_baseline.skip_reason()
    if skip_reason is not None:
        record = {
            **common,
            "node_class": hardware["node_class_guess"],
            "hardware_valid": True,
            "min_abs_coeff": None,
            "trotter_step": None,
            "expectation_re": None,
            "expectation_im": None,
            "final_terms": None,
            "wall_time_s": None,
            "status": "other",
            "failure_reason": f"julia unavailable: {skip_reason}",
        }
        assert set(record) == set(JULIA_CONVERGENCE_FIELDS)
        _append_jsonl(out_dir / "convergence_julia.jsonl", [record])
        return [record]

    all_records: list[dict[str, Any]] = []
    for eps in min_abs_coeffs:
        spec = CellSpec(
            variant_id="bucketed_current",
            theta_h=theta_h,
            trotter_steps=trotter_steps,
            min_abs_coeff=eps,
            direction=direction,
            state=state,
            observable=observable_name,
            threads=1,
            partitions=None,
            repetition_index=0,
            n_qubits=n_qubits,
            max_weight=max_weight,
        )
        circuit = run_cell._build_circuit(spec)
        observable = run_cell._build_observable(spec)
        circuit_spec = oracles.as_circuit_spec(circuit)
        terms = dict(oracles.pauli_terms(observable, n_qubits))
        jl_eps = _jl_min_abs_coeff(eps)

        task = julia_baseline.make_task(
            n_qubits=n_qubits,
            gates=circuit_spec.to_circuit_json()["gates"],
            observable=terms,
            direction=direction,
            min_abs_coeff=jl_eps,
            max_weight=max_weight,
            threads=1,
            state=state,
        )

        print(f"== eps={eps} (jl_eps={jl_eps}): starting runner.jl", flush=True)
        try:
            result = julia_baseline.run_task(
                task,
                threads=1,
                warm_repeats=0,
                layer_counts=False,
                timeout=timeout,
                extra_env={
                    "PP_LAYER_EXPECTATION": "1",
                    "PP_TROTTER_STEPS": str(trotter_steps),
                },
            )
        except julia_baseline.JuliaBaselineError as exc:
            msg = str(exc)
            status = "timeout" if "timed out after" in msg else "other"
            record = {
                **common,
                "node_class": hardware["node_class_guess"],
                "hardware_valid": True,
                "min_abs_coeff": eps,
                "trotter_step": None,
                "expectation_re": None,
                "expectation_im": None,
                "final_terms": None,
                "wall_time_s": None,
                "status": status,
                "failure_reason": f"runner.jl failed: {msg}",
            }
            assert set(record) == set(JULIA_CONVERGENCE_FIELDS)
            all_records.append(record)
            _append_jsonl(out_dir / "convergence_julia.jsonl", [record])
            print(f"  eps={eps} FAILED: {msg}", flush=True)
            continue

        traj = result.per_layer_expectation
        if traj is None:
            record = {
                **common,
                "node_class": hardware["node_class_guess"],
                "hardware_valid": True,
                "min_abs_coeff": eps,
                "trotter_step": None,
                "expectation_re": None,
                "expectation_im": None,
                "final_terms": None,
                "wall_time_s": None,
                "status": "other",
                "failure_reason": (
                    "runner.jl produced no per_layer_expectation -- PP_LAYER_EXPECTATION "
                    "was not honored (stale runner.jl on the julia --project=benchmarks/julia "
                    "path?)"
                ),
            }
            assert set(record) == set(JULIA_CONVERGENCE_FIELDS)
            all_records.append(record)
            _append_jsonl(out_dir / "convergence_julia.jsonl", [record])
            print(f"  eps={eps} FAILED: no per_layer_expectation", flush=True)
            continue

        step_records = []
        for point in traj:
            record = {
                **common,
                "node_class": hardware["node_class_guess"],
                "hardware_valid": True,
                "min_abs_coeff": eps,
                "trotter_step": point["trotter_step"],
                "expectation_re": point["re"],
                "expectation_im": point["im"],
                "final_terms": point["final_terms"],
                "wall_time_s": point["wall_time_s"],
                "status": "completed",
                "failure_reason": None,
            }
            assert set(record) == set(JULIA_CONVERGENCE_FIELDS)
            step_records.append(record)
        all_records.extend(step_records)
        _append_jsonl(out_dir / "convergence_julia.jsonl", step_records)
        for r in step_records:
            print(
                f"  eps={eps} step={r['trotter_step']}/{trotter_steps} "
                f"terms={r['final_terms']} wall_time_s={r['wall_time_s']:.3f} "
                f"exp={r['expectation_re']:.6f}",
                flush=True,
            )

    return all_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-abs-coeff", type=float, nargs="+", required=True)
    parser.add_argument("--n-qubits", type=int, default=127)
    parser.add_argument("--trotter-steps", type=int, default=20)
    parser.add_argument("--theta-h", type=float, default=0.6872233929727672)
    parser.add_argument("--direction", default="heisenberg")
    parser.add_argument("--state", default="z+")
    parser.add_argument("--observable", default="canonical_z_127")
    parser.add_argument(
        "--timeout", type=float, default=86400.0,
        help="seconds before giving up on one cutoff's runner.jl subprocess (default 24h -- "
             "a single cutoff's whole prefix sweep is one subprocess call, not one per step)",
    )
    args = parser.parse_args(argv)

    records = run_sweep(
        out_dir=args.out_dir,
        min_abs_coeffs=args.min_abs_coeff,
        n_qubits=args.n_qubits,
        trotter_steps=args.trotter_steps,
        theta_h=args.theta_h,
        direction=args.direction,
        state=args.state,
        observable_name=args.observable,
        timeout=args.timeout,
    )
    return 0 if records and records[0]["status"] not in ("invalid_hardware",) else 1


if __name__ == "__main__":
    raise SystemExit(main())
