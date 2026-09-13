"""Distributed (`comm=`) single-cell run driver for campaign-2026-09-11.

Launched under `mpirun`/`srun`, one process per rank, alongside `run_cell.py`
(the unpartitioned/in-process driver) rather than replacing it: the two share
`CellSpec`/`RUN_FIELDS`/`GATE_FIELDS`/the builder helpers by import, but this
file owns its own `main` because MPI's startup order is load-bearing
(`mpi4py.rc.thread_level` must be set before `mpi4py.MPI` is imported) and
because every collective call below must be reached by every rank in the same
order — see `python/paulistrings/tests/test_mpi.py`'s module docstring for the
two rules this file follows: no rank-dependent skip or branch around a
collective, and collective order is source order.

CLI: `mpirun -n <ranks> python run_cell_distributed.py <cell.json> --out-dir <dir>`.
Only rank 0 writes `runs.jsonl` (avoids concurrent writers on one shared
file); every rank writes its own `gates.rank-<r>.jsonl`.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

# Must be set before `mpi4py.MPI` is imported (that import calls MPI_Init).
# SERIALIZED is the minimum the engine needs, matching test_mpi.py.
import mpi4py

mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI  # noqa: E402

_JOBS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _JOBS_DIR.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "examples"))
sys.path.insert(0, str(_JOBS_DIR))

from common import harness  # noqa: E402
import preflight  # noqa: E402
from run_cell import (  # noqa: E402
    BUILD_FEATURES,
    CAMPAIGN_ID,
    GATE_FIELDS,
    RUN_FIELDS,
    SCHEMA_VERSION,
    CellSpec,
    _append_jsonl,
    _build_circuit,
    _build_observable,
    _compiler_version,
    _git_provenance,
    _runtime_version,
    _slurm_job_id,
)

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def _empty_run_record(
    spec: CellSpec, run_id: str, status: str, failure_reason: str | None
) -> dict:
    commit, dirty = _git_provenance()
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
        "engine": "distributed",
        "partitions": SIZE,
        "threads": spec.threads,
        "ranks": SIZE,
        "slurm_job_id": _slurm_job_id(),
        "node_class": None,
        "hardware_contract_id": preflight.HARDWARE_CONTRACT_ID,
        "hardware_valid": False,
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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or "--out-dir" not in argv:
        if RANK == 0:
            print("usage: run_cell_distributed.py <cell.json> --out-dir <dir>", file=sys.stderr)
        return 2
    cell_path = Path(argv[0])
    out_dir = Path(argv[argv.index("--out-dir") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_path = out_dir / f"gates.rank-{RANK}.jsonl"

    spec = CellSpec.from_dict(json.loads(cell_path.read_text()))

    # run_id is generated once, on rank 0, and broadcast -- every rank's gate
    # records and the one run record must share it.
    run_id = str(uuid.uuid4()) if RANK == 0 else None
    run_id = COMM.bcast(run_id, root=0)

    # Every rank runs its own preflight (a local, non-collective check), but
    # whether the GROUP proceeds is a collective decision every rank reaches
    # identically -- no rank branches on its own local hardware verdict alone.
    local_ok = preflight.run_preflight()["preflight_passed"]
    group_ok = COMM.allreduce(local_ok, op=MPI.LAND)

    if spec.variant_id != "bucketed_current":
        if RANK == 0:
            _append_jsonl(
                out_dir / "runs.jsonl",
                [
                    _empty_run_record(
                        spec,
                        run_id,
                        "skipped_variant",
                        f"variant_id={spec.variant_id!r} needs a historical-revision "
                        "worktree checkout, deferred (same as run_cell.py's note)",
                    )
                ],
            )
        return 0

    if not group_ok:
        if RANK == 0:
            _append_jsonl(
                out_dir / "runs.jsonl",
                [
                    _empty_run_record(
                        spec,
                        run_id,
                        "invalid_hardware",
                        f"preflight failed on at least one of {SIZE} ranks "
                        "(contract wants every rank on the frozen hardware class)",
                    )
                ],
            )
        return 0

    circuit = _build_circuit(spec)
    observable = _build_observable(spec)
    policy = harness.make_policy(max_weight=spec.max_weight, min_abs_coeff=spec.min_abs_coeff)

    # One untraced propagate for the authoritative wall time, `result="local"`
    # so no rank gathers the full sum (the handoff's explicit distributed-
    # capacity rule) -- mirrors run_cell.py's untraced-call pattern.
    start = time.perf_counter()
    evolved_local = observable.propagate(
        circuit, policy, direction=spec.direction, comm=COMM, result="local"
    )
    # A local partial sum's expectation is one disjoint term of the true
    # value by linearity (matches PartitionedSum::expectation_product_state's
    # documented contract); summing it across ranks recovers the real
    # expectation without ever gathering the full operator onto one rank.
    local_exp = evolved_local.expectation(spec.state) if spec.state else None
    total_exp = COMM.allreduce(local_exp, op=MPI.SUM) if spec.state else None
    wall_time_s = time.perf_counter() - start
    del total_exp  # computed for realistic timing parity with run_cell.py; not yet a schema field

    # A separate, untimed-for-wall_time_s-purposes call for the per-gate
    # trace -- same two-call structure as run_cell.py.
    _, stats = observable.propagate_with_stats(circuit, policy, direction=spec.direction, comm=COMM)

    # Diagnostic only, computed after the timed region: the group's
    # critical-rank proxy, since ranks are not synchronized mid-layer.
    max_wall_time_s = COMM.allreduce(wall_time_s, op=MPI.MAX)

    channels_per_step = len(circuit) // spec.trotter_steps
    partition = stats.partition
    gate_records = []
    for k in range(stats.layers):
        circuit_index = int(stats.circuit_index[k])
        gate_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "rank_id": RANK,
                "application_index": int(stats.application_index[k]),
                "circuit_index": circuit_index,
                "trotter_step": circuit_index // channels_per_step,
                "gate_name": stats.gate_name[k],
                "support_weight": None,
                "terms_in": stats.terms_in[k],
                "terms_out": stats.terms_out[k],
                "nanos": int(stats.nanos[k]),
                "bucket_bits": None,
                "rows_exported": int(partition.rows_exported[k]) if partition is not None else None,
                "bytes_exported": int(partition.bytes_exported[k]) if partition is not None else None,
                "partner_count": None,
            }
        )
    _append_jsonl(gate_path, gate_records)

    # Peak terms/RSS are summed across ranks: each rank holds a disjoint
    # share, so the group's peak resident terms is the sum of each rank's own
    # peak -- not a simultaneous global peak (they need not all peak at the
    # same layer), named accordingly per the handoff's data-contract rule.
    local_peak = stats.peak_terms
    local_final = stats.final_terms
    local_initial = stats.terms_in[0] if stats.terms_in else None
    local_rss = harness.peak_memory_kb() or 0.0
    peak_terms = COMM.allreduce(local_peak, op=MPI.SUM)
    final_terms = COMM.allreduce(local_final, op=MPI.SUM)
    initial_terms = COMM.allreduce(local_initial or 0, op=MPI.SUM)
    peak_rss_kb_sum = COMM.allreduce(local_rss, op=MPI.SUM)
    node_class = preflight.run_preflight()["node_class_guess"]

    if RANK == 0:
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
            "engine": "distributed",
            "partitions": SIZE,
            "threads": spec.threads,
            "ranks": SIZE,
            "slurm_job_id": _slurm_job_id(),
            "node_class": node_class,
            "hardware_contract_id": preflight.HARDWARE_CONTRACT_ID,
            "hardware_valid": True,
            "trace_enabled": True,
            "wall_time_s": max_wall_time_s,
            "setup_time_s": None,
            "scatter_time_s": None,
            "gather_time_s": None,
            "initial_terms": initial_terms,
            "final_terms": final_terms,
            "peak_terms": peak_terms,
            "peak_rss_kb": peak_rss_kb_sum,
            "peak_rss_provenance": "sum_of_per_rank_proc_self_status_vmhwm",
            "status": "completed",
            "failure_reason": None,
            "log_path": None,
            "gate_trace_path": str(out_dir / "gates.rank-*.jsonl"),
        }
        assert set(run_record) == set(RUN_FIELDS), (
            f"run record field set drifted from run_cell.py's RUN_FIELDS: "
            f"{set(run_record) ^ set(RUN_FIELDS)}"
        )
        assert not gate_records or set(gate_records[0]) == set(GATE_FIELDS), (
            f"gate record field set drifted from run_cell.py's GATE_FIELDS: "
            f"{set(gate_records[0]) ^ set(GATE_FIELDS)}"
        )
        _append_jsonl(out_dir / "runs.jsonl", [run_record])
        print(json.dumps({"run_id": run_id, "status": "completed", "ranks": SIZE}))

    return 0


if __name__ == "__main__":
    sys.exit(main())
