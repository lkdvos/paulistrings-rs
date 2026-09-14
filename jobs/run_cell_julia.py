"""Single-cell PauliPropagation.jl driver for campaign-2026-09-11 (E0, external baseline).

CLI: `python run_cell_julia.py <cell.json> --out-dir <dir>`. Same `cell.json`
shape `run_cell.py` consumes (`run_cell.CellSpec`, reused verbatim rather than
a second dataclass) -- but a cell's `variant_id` names a Rust commit-registry
entry (`tasks/T01-variants.json`) and has no meaning for this leg, so it is
ignored for that purpose; the emitted run record always stamps
`variant_id="external_pauli_propagation_jl"` and preserves `config_id`
verbatim so a Rust cell and its Julia counterpart join on `config_id`.

Reuses, rather than reimplements: `run_cell._build_circuit`/`_build_observable`
for the physics, `common.oracles.as_circuit_spec`/`pauli_terms` to translate
the engine's `Circuit`/`PauliSum` into task-JSON schema v1, and
`benchmarks/python/julia_baseline.py` (`make_task`/`run_task`) to build the
task file and invoke `benchmarks/julia/runner.jl` as a subprocess. No new
task-JSON construction lives here.

Writes one run record to `<out-dir>/runs.jsonl` (append), schema v1
(`analysis/schema.py::validate_run`). Never writes a gate-trace file: per
decision #10, PauliPropagation.jl has no per-gate wall-time instrumentation,
and `validate_gate` requires `nanos` as a real non-null int, so there is no
honest per-gate record this leg can produce. `trace_enabled` is always
`False` and `gate_trace_path` always `null`. Per-layer term counts
(`PP_LAYER_COUNTS=1`, on by default) are real and are carried in the run
record's open-ended `extra` field instead, since they have no gate-record home.

Preflight-gated exactly like `run_cell.py`: an `invalid_hardware` record on a
non-`genoa` host, no Julia subprocess started. This is the same physical node
in a real campaign run, so the same hardware contract applies to both legs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from math import nextafter, inf
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
from run_cell import CellSpec  # noqa: E402

SCHEMA_VERSION = 1
CAMPAIGN_ID = "campaign-2026-09-11"
VARIANT_ID = "external_pauli_propagation_jl"

#: Same field order as `run_cell.RUN_FIELDS` -- kept as a separate tuple
#: (identical today) so a deliberate future divergence between the two
#: drivers' schemas doesn't have to fight a shared name.
RUN_FIELDS = run_cell.RUN_FIELDS


def _git_provenance() -> tuple[str, bool | None]:
    return run_cell._git_provenance()


def _slurm_job_id() -> str | None:
    return run_cell._slurm_job_id()


def _julia_compiler_version() -> str:
    """`julia --version`'s own string, or `"unknown"`.

    Queried independent of whether a task later succeeds, mirroring
    `run_cell._compiler_version()` -- this is the toolchain, not the run
    outcome, and schema v1 requires a non-null string here even on a failed
    or hardware-gated record.
    """
    julia = julia_baseline.find_julia()
    if julia is None:
        return "unknown"
    try:
        out = subprocess.run(
            [julia, "--version"], capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _pauli_propagation_runtime_version() -> str:
    """The pinned `PauliPropagation.jl` version and tree-hash, from
    `Manifest.toml` directly -- no package load needed, so this is cheap
    enough to call unconditionally (including on an `invalid_hardware`
    record). `contract.md` pins the tree-sha, not the (unverifiable) version
    string alone, so both are reported.
    """
    manifest = julia_baseline.JULIA_PROJECT / "Manifest.toml"
    try:
        import tomllib

        data = tomllib.loads(manifest.read_text())
        entries = data.get("deps", {}).get("PauliPropagation")
        if not entries:
            return "unknown"
        entry = entries[0]
        version = entry.get("version", "unknown")
        tree_sha1 = entry.get("git-tree-sha1", "unknown")
        return f"PauliPropagation.jl {version} (tree {tree_sha1})"
    except Exception:
        return "unknown"


def _jl_min_abs_coeff(eps: float) -> float:
    """The threshold to hand `runner.jl` so its truncation rule matches this
    engine's, per `runner.jl`'s own header comment and `bench_c_deep_trotter.
    julia_min_abs_coeff` (not imported: that module pulls in
    `bench_b_theta_sweep` and is a plotting driver, too heavy a dependency for
    this CLI's production path over one `nextafter` call).

    This engine drops `|c| <= eps`; jl drops `|c| < eps` and therefore keeps a
    coefficient exactly at the threshold. `eps' = nextafter(eps, inf)` has no
    float strictly between it and `eps`, so jl's `|c| < eps'` is exactly
    `|c| <= eps` -- the two rules coincide for every input.
    """
    if eps <= 0.0:
        return 0.0
    return nextafter(eps, inf)


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
        "variant_id": VARIANT_ID,
        "repetition_index": spec.repetition_index,
        "pair_index": spec.pair_index,
        "source_commit": commit,
        "dirty": dirty,
        "build_features": [],
        "compiler_version": _julia_compiler_version(),
        "runtime_version": _pauli_propagation_runtime_version(),
        "n_qubits": spec.n_qubits,
        "direction": spec.direction,
        "state": spec.state,
        "min_abs_coeff": spec.min_abs_coeff,
        "max_weight": spec.max_weight,
        "policy": None,
        "engine": "pauli_propagation_jl",
        "partitions": 1,
        "threads": spec.threads,
        "ranks": 1,
        "slurm_job_id": _slurm_job_id(),
        "node_class": node_class,
        "hardware_contract_id": preflight.HARDWARE_CONTRACT_ID,
        "hardware_valid": hardware_valid,
        "trace_enabled": False,
        "wall_time_s": None,
        # validate_run requires a "<field>_reason" string whenever wall_time_s
        # is null; the cell never ran, so the reason is the same failure_reason
        # (mirrors the identical fix in run_cell.py, found via the same real
        # validate_campaign.py run -- see decisions.md).
        "wall_time_s_reason": failure_reason,
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
        "extra": None,
    }


def run_cell_julia(
    spec: CellSpec,
    out_dir: Path,
    *,
    warm_repeats: int | None = None,
    timeout: float = 3600.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one cell through PauliPropagation.jl, returning `(run_record, [])`.

    The second element is always `[]`: this leg never produces gate records
    (module docstring). Never raises for a cell that legitimately cannot run
    (bad hardware, no Julia, runner failure) -- those come back as a
    `status`-labeled record instead; only a genuine bug in this driver's own
    task construction propagates.
    """
    run_id = str(uuid.uuid4())
    out_dir.mkdir(parents=True, exist_ok=True)

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

    skip_reason = julia_baseline.skip_reason()
    if skip_reason is not None:
        record = _empty_run_record(
            spec, run_id, status="other", failure_reason=f"julia unavailable: {skip_reason}",
            hardware=hardware,
        )
        return record, []

    circuit = run_cell._build_circuit(spec)
    observable = run_cell._build_observable(spec)
    circuit_spec = oracles.as_circuit_spec(circuit)
    terms = dict(oracles.pauli_terms(observable, spec.n_qubits))
    jl_eps = _jl_min_abs_coeff(spec.min_abs_coeff)

    task = julia_baseline.make_task(
        n_qubits=spec.n_qubits,
        gates=circuit_spec.to_circuit_json()["gates"],
        observable=terms,
        direction=spec.direction,
        min_abs_coeff=jl_eps,
        max_weight=spec.max_weight,
        threads=spec.threads,
        state=spec.state,
    )

    commit, dirty = _git_provenance()
    start = time.perf_counter()
    try:
        result = julia_baseline.run_task(
            task,
            threads=spec.threads,
            warm_repeats=warm_repeats,
            layer_counts=True,
            timeout=timeout,
        )
    except julia_baseline.JuliaBaselineError as exc:
        msg = str(exc)
        status = "timeout" if "timed out after" in msg else "other"
        record = _empty_run_record(
            spec, run_id, status=status, failure_reason=f"runner.jl failed: {msg}",
            hardware=hardware,
        )
        return record, []
    driver_wall_s = time.perf_counter() - start  # diagnostic only, not written as wall_time_s

    wall_time_s = result.wall_warm_s if result.wall_warm_s is not None else result.wall_cold_s
    exp = result.expectation
    policy_desc = (
        f"min_abs_coeff={spec.min_abs_coeff!r} (jl_eps={jl_eps!r}), "
        f"max_weight={spec.max_weight!r}"
    )

    run_record = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "run_id": run_id,
        "task_id": spec.task_id,
        "config_id": spec.config_id,
        "variant_id": VARIANT_ID,
        "repetition_index": spec.repetition_index,
        "pair_index": spec.pair_index,
        "source_commit": commit,
        "dirty": dirty,
        "build_features": [],
        "compiler_version": f"julia {result.versions.get('julia', 'unknown')}",
        "runtime_version": (
            f"PauliPropagation.jl {result.versions.get('PauliPropagation', 'unknown')}"
        ),
        "n_qubits": spec.n_qubits,
        "direction": spec.direction,
        "state": spec.state,
        "min_abs_coeff": spec.min_abs_coeff,
        "max_weight": spec.max_weight,
        "policy": policy_desc,
        "engine": "pauli_propagation_jl",
        "partitions": 1,
        "threads": spec.threads,
        "ranks": 1,
        "slurm_job_id": _slurm_job_id(),
        "node_class": hardware["node_class_guess"],
        "hardware_contract_id": preflight.HARDWARE_CONTRACT_ID,
        "hardware_valid": True,
        "trace_enabled": False,
        "wall_time_s": wall_time_s,
        # Julia's runner has no isolated setup/scatter/gather phase (module
        # docstring); genuinely unavailable, not fabricated.
        "setup_time_s": None,
        "scatter_time_s": None,
        "gather_time_s": None,
        "initial_terms": result.raw["result"]["input_terms"],
        "final_terms": result.final_terms,
        "peak_terms": result.peak_terms,
        "peak_rss_kb": result.peak_memory_kb,
        "peak_rss_provenance": (
            "julia_proc_self_status_vmhwm" if result.peak_memory_kb is not None else None
        ),
        "status": "completed",
        "failure_reason": None,
        "log_path": None,
        "gate_trace_path": None,
        "extra": {
            "per_layer_terms": result.per_layer_terms,
            "expectation_re": None if exp is None else exp.real,
            "expectation_im": None if exp is None else exp.imag,
            "expectation_method": result.raw["result"]["expectation_method"],
            "julia_min_abs_coeff": jl_eps,
            "one_ulp_perturbation": jl_eps - spec.min_abs_coeff,
            "wall_cold_s": result.wall_cold_s,
            "wall_warm_all_s": result.raw["timing"]["wall_warm_all_s"],
            "julia_threads": result.raw["config"]["julia_threads"],
            "julia_backend": result.raw["config"]["backend"],
            "julia_notes": result.notes,
            "driver_wall_s": driver_wall_s,
        },
    }
    return run_record, []


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
    parser.add_argument(
        "--warm-repeats", type=int, default=None,
        help="override runner.jl's PP_WARM_REPEATS (default: runner.jl's own default, 3)",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)

    cell_data = json.loads(args.cell_json.read_text())
    spec = CellSpec.from_dict(cell_data)

    run_record, gate_records = run_cell_julia(
        spec, args.out_dir, warm_repeats=args.warm_repeats, timeout=args.timeout
    )

    _append_jsonl(args.out_dir / "runs.jsonl", [run_record])
    _append_jsonl(args.out_dir / "gates.rank-0.jsonl", gate_records)

    print(json.dumps({"run_id": run_record["run_id"], "status": run_record["status"]}))
    return 0 if run_record["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
