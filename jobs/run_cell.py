"""Single-cell run driver for campaign-2026-09-11 (T07).

CLI: `python run_cell.py <cell.json> --out-dir <dir>`. One `cell.json` is one
point of the campaign matrix; see `CellSpec` for its fields. Writes one run
record to `<out-dir>/runs.jsonl` (append) and, for a rank-0 (or unpartitioned)
run, per-gate records to `<out-dir>/gates.rank-0.jsonl` (append) — schema v1,
field-for-field what T06's validator expects (see the module docstring's
`RUN_FIELDS` / `GATE_FIELDS`).

Only `variant_id == "bucketed_current"` runs the real engine in this pass; any
other `variant_id` names a historical revision from `tasks/T01-variants.json`,
whose worktree-checkout build machinery is explicitly deferred (see T07's
final report) — such a cell writes a `status="skipped_variant"` record and
does no propagation.

`RAYON_NUM_THREADS` must already be set correctly in the environment before
this process starts when `threads == 1` is required
(`examples/common/harness.py` module docstring) — this driver does not
re-exec itself to fix that; a `threads=1` cell run under the wrong
environment fails loudly via `harness.assert_single_threaded`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_JOBS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _JOBS_DIR.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "examples"))
sys.path.insert(0, str(_JOBS_DIR))

from common import circuits, harness, observables, report  # noqa: E402
import preflight  # noqa: E402

SCHEMA_VERSION = 1
CAMPAIGN_ID = "campaign-2026-09-11"

#: Field order of one run record, pinned to match T06's validator (contract.md
#: "Schema version"). Every field is present in every record; unavailable
#: values are `None` (JSON `null`), never omitted or fabricated.
RUN_FIELDS = (
    "schema_version", "campaign_id", "run_id", "task_id", "config_id",
    "variant_id", "repetition_index", "pair_index", "source_commit", "dirty",
    "build_features", "compiler_version", "runtime_version", "n_qubits",
    "direction", "state", "min_abs_coeff", "max_weight", "policy", "engine",
    "partitions", "threads", "ranks", "slurm_job_id", "node_class",
    "hardware_contract_id", "hardware_valid", "trace_enabled", "wall_time_s",
    "setup_time_s", "scatter_time_s", "gather_time_s", "initial_terms",
    "final_terms", "peak_terms", "peak_rss_kb", "peak_rss_provenance",
    "status", "failure_reason", "log_path", "gate_trace_path",
)

#: Field order of one per-gate record.
GATE_FIELDS = (
    "schema_version", "run_id", "rank_id", "application_index",
    "circuit_index", "trotter_step", "gate_name", "support_weight", "terms_in",
    "terms_out", "nanos", "bucket_bits", "rows_exported", "bytes_exported",
    "partner_count",
)


@dataclass(frozen=True)
class CellSpec:
    """One point of the campaign matrix, as read from `cell.json`."""

    variant_id: str
    theta_h: float
    trotter_steps: int
    min_abs_coeff: float
    direction: str
    state: str
    observable: str
    threads: int
    partitions: int | None
    repetition_index: int
    task_id: str = "T02-canonical"
    config_id: str | None = None
    pair_index: int | None = None
    max_weight: int | None = None
    # Debug-only override, NOT part of the frozen campaign schema
    # (contract.md pins n_qubits=127): a smoke-test cell may set this to a
    # small value to prove the plumbing without running the real workload.
    # Any value other than 127 is recorded in the run record's `status`
    # metadata is not needed since n_qubits itself is a normal field, but
    # `observable` must then be "debug_single_z" (see `_build_observable`).
    n_qubits: int = 127

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CellSpec":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"cell.json has unknown fields {sorted(unknown)}")
        return cls(**data)


def _build_circuit(spec: CellSpec):
    """`circuits.heavy_hex_kicked_ising` per `contract.md`'s frozen parameters.

    `theta_zz` is left at its default (`KICKED_ISING_CLIFFORD_THETA_ZZ =
    -pi/2`), `final_x_layer` at its default `False`, `order` at its default
    `"x-then-zz"` — none of these are cell.json fields because the contract
    freezes them for every cell in this campaign.
    """
    return circuits.heavy_hex_kicked_ising(
        n=spec.n_qubits,
        trotter_steps=spec.trotter_steps,
        theta_h=spec.theta_h,
    )


def _build_observable(spec: CellSpec):
    if spec.observable == "canonical_z_127":
        if spec.n_qubits != 127:
            raise ValueError(
                "observable='canonical_z_127' requires n_qubits=127 (contract.md); "
                f"got n_qubits={spec.n_qubits}. Use observable='debug_single_z' for "
                "a smoke-test cell at a smaller n_qubits."
            )
        return observables.canonical_z_127()
    if spec.observable == "debug_single_z":
        # Smoke-test-only path: a synthetic weight-1 Z observable at the
        # register's midpoint, valid for any n_qubits. Never used for a real
        # campaign cell.
        return observables.single_z(spec.n_qubits // 2, spec.n_qubits)
    raise ValueError(
        f"unknown observable {spec.observable!r}; expected 'canonical_z_127' "
        "(the real campaign cell) or 'debug_single_z' (smoke test only)"
    )


def _git_provenance() -> tuple[str, bool | None]:
    return report._git_commit_and_dirty(_REPO_ROOT)


def _runtime_version() -> str | None:
    try:
        import paulistrings

        return getattr(paulistrings, "__version__", "unknown")
    except Exception:
        return None


def _compiler_version() -> str | None:
    """`rustc`'s own version string, or `None` if it's not on `PATH`.

    Genuinely available (the extension was just built with it) but never
    wired up before this fix; see the campaign's decisions log.
    """
    try:
        out = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip() or None
    except Exception:
        return None


#: `campaign-genoa.sbatch`'s `maturin develop --release` call passes no
#: `--features`, so the built extension's feature set is always empty today.
#: Update this alongside the sbatch template if that ever changes.
BUILD_FEATURES: list[str] = []


def _empty_run_record(
    spec: CellSpec,
    run_id: str,
    status: str,
    failure_reason: str | None,
    hardware: dict[str, Any] | None,
) -> dict[str, Any]:
    commit, dirty = _git_provenance()
    node_class = hardware["node_class_guess"] if hardware else None
    hardware_valid = hardware["preflight_passed"] if hardware else False
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "run_id": run_id,
        "task_id": spec.task_id,
        "config_id": spec.config_id,
        "variant_id": spec.variant_id,
        "repetition_index": spec.repetition_index,
        "pair_index": spec.pair_index,
        "source_commit": commit,
        "dirty": dirty,
        "build_features": list(BUILD_FEATURES),
        "compiler_version": _compiler_version(),
        "runtime_version": _runtime_version(),
        "n_qubits": spec.n_qubits,
        "direction": spec.direction,
        "state": spec.state,
        "min_abs_coeff": spec.min_abs_coeff,
        "max_weight": spec.max_weight,
        "policy": None,
        "engine": "unpartitioned" if spec.partitions is None else "partitioned",
        "partitions": spec.partitions if spec.partitions is not None else 1,
        "threads": spec.threads,
        "ranks": 1,
        "slurm_job_id": _slurm_job_id(),
        "node_class": node_class,
        "hardware_contract_id": preflight.HARDWARE_CONTRACT_ID,
        "hardware_valid": hardware_valid,
        "trace_enabled": False,
        "wall_time_s": None,
        "setup_time_s": None,
        "scatter_time_s": None,
        "gather_time_s": None,
        "initial_terms": None,
        "final_terms": None,
        "peak_terms": None,
        "peak_rss_kb": None,
        "peak_rss_provenance": None,
        "status": status,
        "failure_reason": failure_reason,
        "log_path": None,
        "gate_trace_path": None,
    }


def _slurm_job_id() -> str | None:
    import os

    return os.environ.get("SLURM_JOB_ID") or None


def run_cell(spec: CellSpec, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one cell, returning `(run_record, gate_records)`. Never raises for
    a cell that legitimately cannot run (bad hardware, deferred variant) —
    those come back as a `status`-labeled record instead; only a genuine bug
    (bad cell.json, engine exception) propagates.
    """
    run_id = str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_trace_path = out_dir / "gates.rank-0.jsonl"

    if spec.variant_id != "bucketed_current":
        record = _empty_run_record(
            spec,
            run_id,
            status="skipped_variant",
            failure_reason=(
                f"variant_id={spec.variant_id!r} needs a historical-revision worktree "
                "checkout, deferred in this pass (T07 report, E1/E2 note)"
            ),
            hardware=None,
        )
        return record, []

    hardware = preflight.run_preflight()
    if not hardware["preflight_passed"]:
        record = _empty_run_record(
            spec,
            run_id,
            status="invalid_hardware",
            failure_reason=(
                f"preflight failed: node_class_guess={hardware['node_class_guess']!r} "
                f"(contract wants {preflight.HARDWARE_CONTRACT_ID!r}), "
                f"smt_siblings_present={hardware['smt_siblings_present']}, "
                f"affinity_is_physical_core_only={hardware['affinity_is_physical_core_only']}"
            ),
            hardware=hardware,
        )
        return record, []

    circuit = _build_circuit(spec)
    observable = _build_observable(spec)
    policy = harness.make_policy(max_weight=spec.max_weight, min_abs_coeff=spec.min_abs_coeff)

    # One untraced propagate for the authoritative wall time — no stats
    # object in the timed region — mirroring bench_c_deep_trotter.py's
    # `_rust_leg` pattern of keeping the timed call free of tracing overhead.
    start = time.perf_counter()
    evolved = observable.propagate(circuit, policy, direction=spec.direction)
    wall_time_s = time.perf_counter() - start
    if spec.state:
        evolved.expectation(spec.state)

    # A second, separately timed call for the per-gate trace; its own wall
    # time is diagnostic only and is not written into `wall_time_s`.
    _, stats = observable.propagate_with_stats(circuit, policy, direction=spec.direction)

    commit, dirty = _git_provenance()
    run_record = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "run_id": run_id,
        "task_id": spec.task_id,
        "config_id": spec.config_id,
        "variant_id": spec.variant_id,
        "repetition_index": spec.repetition_index,
        "pair_index": spec.pair_index,
        "source_commit": commit,
        "dirty": dirty,
        "build_features": list(BUILD_FEATURES),
        "compiler_version": _compiler_version(),
        "runtime_version": _runtime_version(),
        "n_qubits": spec.n_qubits,
        "direction": spec.direction,
        "state": spec.state,
        "min_abs_coeff": spec.min_abs_coeff,
        "max_weight": spec.max_weight,
        "policy": repr(policy) if policy is not None else None,
        "engine": "unpartitioned" if spec.partitions is None else "partitioned",
        "partitions": spec.partitions if spec.partitions is not None else 1,
        "threads": spec.threads,
        "ranks": 1,
        "slurm_job_id": _slurm_job_id(),
        "node_class": hardware["node_class_guess"],
        "hardware_contract_id": preflight.HARDWARE_CONTRACT_ID,
        "hardware_valid": True,
        "trace_enabled": True,
        "wall_time_s": wall_time_s,
        # Not separately measured by this driver: run_propagation's warmup
        # discipline and this driver's own two-call structure don't isolate a
        # distinct "setup" phase. Genuinely unavailable, not fabricated.
        "setup_time_s": None,
        "scatter_time_s": None,
        "gather_time_s": None,
        "initial_terms": stats.terms_in[0] if stats.terms_in else None,
        "final_terms": stats.final_terms,
        "peak_terms": stats.peak_terms,
        "peak_rss_kb": harness.peak_memory_kb(),
        "peak_rss_provenance": "proc_self_status_vmhwm" if harness.peak_memory_kb() is not None else None,
        "status": "completed",
        "failure_reason": None,
        "log_path": None,
        "gate_trace_path": str(gate_trace_path),
    }

    gate_records: list[dict[str, Any]] = []
    layers = stats.layers
    partition = stats.partition
    # `trotter_step` is never stored by the engine (Circuit has no notion of
    # steps) -- derived here since this driver knows both circuit length and
    # `spec.trotter_steps`, and `analysis/schema.py::validate_gate` checks the
    # derivation itself when given `channels_per_step`.
    channels_per_step = len(circuit) // spec.trotter_steps
    for k in range(layers):
        circuit_index = int(stats.circuit_index[k])
        gate_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "rank_id": 0,
                "application_index": int(stats.application_index[k]),
                "circuit_index": circuit_index,
                "trotter_step": circuit_index // channels_per_step,
                "gate_name": stats.gate_name[k],
                # Not exposed by PropagationStats/PartitionStats today (no
                # per-gate support-weight getter on the Python binding) —
                # genuinely unavailable, not derivable from what the CLI
                # sees; see T07's report.
                "support_weight": None,
                "terms_in": stats.terms_in[k],
                "terms_out": stats.terms_out[k],
                "nanos": int(stats.nanos[k]),
                # bucket_bits/partner_count are phase_breakdown.rs-only
                # concepts today, not on PropagationStats/PartitionStats.
                "bucket_bits": None,
                "rows_exported": int(partition.rows_exported[k]) if partition is not None else None,
                "bytes_exported": int(partition.bytes_exported[k]) if partition is not None else None,
                "partner_count": None,
            }
        )

    return run_record, gate_records


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell_json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    cell_data = json.loads(args.cell_json.read_text())
    spec = CellSpec.from_dict(cell_data)

    run_record, gate_records = run_cell(spec, args.out_dir)

    _append_jsonl(args.out_dir / "runs.jsonl", [run_record])
    _append_jsonl(args.out_dir / "gates.rank-0.jsonl", gate_records)

    print(json.dumps({"run_id": run_record["run_id"], "status": run_record["status"]}))
    return 0 if run_record["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
