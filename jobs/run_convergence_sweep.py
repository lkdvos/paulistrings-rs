"""Per-Trotter-step expectation-value convergence sweep for campaign-2026-09-11.

The single-point accuracy comparison in `run_cell.py`/`run_cell_julia.py`
(one final expectation value per run, decisions.md #26-27) makes a poor plot:
one dot per engine, nothing to show a trend against. This script instead
propagates the canonical circuit's PREFIX of length `1, 2, ..., trotter_steps`
Trotter steps -- via `Circuit.__getitem__`'s slice support, never a new
engine feature -- and records `observable.expectation(state)` after each
prefix, for every `min_abs_coeff` in a small cutoff grid. One line per cutoff
in the resulting figure shows the observable's trajectory across time and how
truncation error grows (or doesn't) as the circuit deepens.

A circuit prefix of length `k*channels_per_step` truncates exactly as a full
`k`-step circuit would up to that point (truncation only ever depends on the
sum's own history, never on gates not yet applied), so this is a real,
correctness-preserving way to get per-step values from the existing public
API -- at the cost of redoing the shared prefix work for every step, since
there is no incremental/checkpointed propagate call to build on instead.

CLI: `python run_convergence_sweep.py --out-dir <dir> --min-abs-coeff <eps> [<eps> ...]`.
Writes one JSONL record per (min_abs_coeff, trotter_step) point to
`<out-dir>/convergence.jsonl` -- a different, simpler shape than
`analysis/schema.py`'s `RUN_FIELDS` (this is one point on a curve, not one
campaign cell), documented inline rather than forced into that schema.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_JOBS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_JOBS_DIR))

from run_cell import CellSpec, _build_circuit, _build_observable, _compiler_version, _git_provenance, _slurm_job_id  # noqa: E402
from common import harness  # noqa: E402
import preflight  # noqa: E402

CONVERGENCE_SCHEMA_VERSION = 1
CAMPAIGN_ID = "campaign-2026-09-11"

#: Field order of one convergence-point record.
CONVERGENCE_FIELDS = (
    "schema_version", "campaign_id", "sweep_id", "task_id", "source_commit",
    "dirty", "compiler_version", "slurm_job_id", "node_class",
    "hardware_valid", "n_qubits", "theta_h", "trotter_steps", "direction",
    "state", "observable", "min_abs_coeff", "trotter_step",
    "expectation_re", "expectation_im", "final_terms", "wall_time_s",
    "status", "failure_reason",
)


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
) -> list[dict[str, Any]]:
    """Run the full (min_abs_coeff x trotter_step) grid, returning every record written."""
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
        "compiler_version": _compiler_version(),
        "slurm_job_id": _slurm_job_id(),
        "n_qubits": n_qubits,
        "theta_h": theta_h,
        "trotter_steps": trotter_steps,
        "direction": direction,
        "state": state,
        "observable": observable_name,
    }

    hardware = preflight.run_preflight()
    all_records: list[dict[str, Any]] = []
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
        assert set(record) == set(CONVERGENCE_FIELDS)
        _append_jsonl(out_dir / "convergence.jsonl", [record])
        return [record]

    for eps in min_abs_coeffs:
        spec = CellSpec(
            variant_id="bucketed_current",
            theta_h=theta_h,
            trotter_steps=trotter_steps,
            min_abs_coeff=eps,
            direction=direction,
            state=state,
            observable=observable_name,
            threads=0,  # unused by _build_circuit/_build_observable
            partitions=None,
            repetition_index=0,
            n_qubits=n_qubits,
            max_weight=max_weight,
        )
        circuit = _build_circuit(spec)
        observable = _build_observable(spec)
        policy = harness.make_policy(max_weight=max_weight, min_abs_coeff=eps)
        channels_per_step = len(circuit) // trotter_steps

        for step in range(1, trotter_steps + 1):
            prefix = circuit[: step * channels_per_step]
            start = time.perf_counter()
            evolved = observable.propagate(prefix, policy, direction=direction)
            wall_time_s = time.perf_counter() - start
            exp = evolved.expectation(state)
            record = {
                **common,
                "node_class": hardware["node_class_guess"],
                "hardware_valid": True,
                "min_abs_coeff": eps,
                "trotter_step": step,
                "expectation_re": exp.real,
                "expectation_im": exp.imag,
                "final_terms": len(evolved),
                "wall_time_s": wall_time_s,
                "status": "completed",
                "failure_reason": None,
            }
            assert set(record) == set(CONVERGENCE_FIELDS)
            all_records.append(record)
            _append_jsonl(out_dir / "convergence.jsonl", [record])
            print(
                f"  eps={eps} step={step}/{trotter_steps} "
                f"terms={record['final_terms']} wall_time_s={wall_time_s:.3f} "
                f"exp={exp.real:.6f}",
                flush=True,
            )

    return all_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-abs-coeff", type=float, nargs="+", required=True)
    parser.add_argument("--n-qubits", type=int, default=127)
    parser.add_argument("--trotter-steps", type=int, default=20)
    parser.add_argument("--theta-h", type=float, default=0.6872233929727672)
    parser.add_argument("--direction", default="heisenberg")
    parser.add_argument("--state", default="z+")
    parser.add_argument("--observable", default="canonical_z_127")
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
    )
    return 0 if records and records[0]["status"] != "invalid_hardware" else 1


if __name__ == "__main__":
    raise SystemExit(main())
