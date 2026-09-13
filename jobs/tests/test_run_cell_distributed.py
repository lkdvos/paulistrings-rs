"""`run_cell_distributed.py` under a real MPI world.

Like `python/paulistrings/tests/test_mpi.py`, this file is the whole net at
one rank (a plain `pytest` run, MPI initializes as a singleton world) and at
2 or 4 ranks under `mpirun`::

    mpirun -n 4 .venv-mpi/bin/python -m pytest \
        quera-talk-data/campaign-2026-09-11/jobs/tests/test_run_cell_distributed.py \
        -q -p no:cacheprovider -p no:randomly

Same two rules as `test_mpi.py`: no rank-dependent skip or branch around a
collective, and collective order is source order (`-p no:randomly`).
Requires the `mpi` feature build in `.venv-mpi` (see CLAUDE.md's MPI section)
and `mpi4py`; skips cleanly otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mpi4py")

import mpi4py  # noqa: E402

mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI  # noqa: E402

_JOBS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_JOBS_DIR))

import preflight  # noqa: E402
import run_cell_distributed  # noqa: E402
from run_cell import GATE_FIELDS, RUN_FIELDS  # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
POWER_OF_TWO = SIZE & (SIZE - 1) == 0

if not POWER_OF_TWO:
    _SKIP = f"launched with {SIZE} ranks; needs a power-of-two world"
else:
    _SKIP = None


def _passing_hardware_report():
    return {
        "hardware_contract_id": preflight.HARDWARE_CONTRACT_ID,
        "cpu_model": "synthetic genoa for smoke test",
        "cpu_vendor_id": "AuthenticAMD",
        "cpu_family": 25,
        "cpu_model_number": 17,
        "node_class_guess": "genoa",
        "node_class_fingerprint_note": "synthetic",
        "physical_cores_available": 8,
        "smt_siblings_present": False,
        "affinity_is_physical_core_only": True,
        "affinity_cpu_count": 8,
        "affinity_cpus_missing_from_lscpu": [],
        "preflight_passed": True,
    }


def _smoke_cell(tmp_path: Path, **overrides) -> Path:
    data = {
        "variant_id": "bucketed_current",
        "theta_h": 0.6872233929727672,
        "trotter_steps": 1,
        "min_abs_coeff": 0.0,
        "direction": "heisenberg",
        "state": "z+",
        "observable": "debug_single_z",
        "threads": 1,
        "partitions": None,
        "repetition_index": 0,
        "config_id": "smoke",
        "n_qubits": 8,
        **overrides,
    }
    # Every rank must agree on the path (a shared tmp_path from pytest's own
    # per-test fixture is already rank-local under mpirun -- broadcast rank
    # 0's to keep every rank pointed at the same file).
    path = tmp_path / "cell.json"
    if RANK == 0:
        path.write_text(json.dumps(data))
    path = COMM.bcast(path if RANK == 0 else None, root=0)
    return path


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_completed_run_matches_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "run_preflight", _passing_hardware_report)
    cell_path = _smoke_cell(tmp_path)
    out_dir = tmp_path / "out"
    out_dir = COMM.bcast(out_dir if RANK == 0 else None, root=0)

    rc = run_cell_distributed.main([str(cell_path), "--out-dir", str(out_dir)])
    assert rc == 0

    COMM.Barrier()  # every rank must have finished writing its own gate file
    if RANK == 0:
        runs = [json.loads(l) for l in (out_dir / "runs.jsonl").read_text().splitlines()]
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"
        assert runs[0]["ranks"] == SIZE
        assert runs[0]["engine"] == "distributed"
        assert set(runs[0]) == set(RUN_FIELDS)

        for r in range(SIZE):
            gate_path = out_dir / f"gates.rank-{r}.jsonl"
            assert gate_path.exists(), f"rank {r} never wrote its gate file"
            rows = [json.loads(l) for l in gate_path.read_text().splitlines()]
            assert rows, f"rank {r} wrote an empty gate file"
            assert all(row["rank_id"] == r for row in rows)
            assert set(rows[0]) == set(GATE_FIELDS)

        # Every rank saw the same circuit, so every rank's gate count agrees.
        counts = {
            r: len((out_dir / f"gates.rank-{r}.jsonl").read_text().splitlines())
            for r in range(SIZE)
        }
        assert len(set(counts.values())) == 1, f"rank gate counts disagree: {counts}"


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_invalid_hardware_on_any_rank_fails_the_whole_group(tmp_path, monkeypatch):
    """One rank's failed preflight must stop the *group*, not just that rank
    -- otherwise the survivors would deadlock waiting in a collective the
    failed rank never reaches."""

    def _report():
        # Rank 0 alone fails; every rank still reaches the same `allreduce`
        # and the same branch on its result, so this cannot deadlock.
        return {**_passing_hardware_report(), "preflight_passed": RANK != 0}

    monkeypatch.setattr(preflight, "run_preflight", _report)
    cell_path = _smoke_cell(tmp_path)
    out_dir = tmp_path / "out"
    out_dir = COMM.bcast(out_dir if RANK == 0 else None, root=0)

    rc = run_cell_distributed.main([str(cell_path), "--out-dir", str(out_dir)])
    assert rc == 0

    COMM.Barrier()
    if RANK == 0:
        runs = [json.loads(l) for l in (out_dir / "runs.jsonl").read_text().splitlines()]
        assert len(runs) == 1
        assert runs[0]["status"] == "invalid_hardware"
        assert not (out_dir / "gates.rank-0.jsonl").exists()
