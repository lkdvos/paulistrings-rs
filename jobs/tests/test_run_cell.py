"""Plumbing smoke tests for `run_cell.py`.

This host is not `genoa` (verified by `preflight.py` itself — see
`test_preflight.py`), so the *real* CLI path on this host always produces an
`invalid_hardware` record, which is exercised directly below with no
monkeypatching: that is the correct, honest behavior being tested. The
"completed" record / gate-trace path is exercised by monkeypatching
`preflight.run_preflight` to return a passing report and running a tiny
synthetic circuit (`n_qubits=8`, `trotter_steps=1`) — never the real
127-qubit campaign circuit. Both are labeled smoke tests: neither writes
anything that should be mistaken for real campaign data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import preflight  # noqa: E402
import run_cell  # noqa: E402


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


def _smoke_spec(**overrides):
    data = dict(
        variant_id="bucketed_current",
        theta_h=0.6872233929727672,
        trotter_steps=1,
        min_abs_coeff=1e-6,
        direction="heisenberg",
        state="z+",
        observable="debug_single_z",
        threads=1,
        partitions=None,
        repetition_index=0,
        n_qubits=8,
    )
    data.update(overrides)
    return run_cell.CellSpec.from_dict(data)


def test_real_preflight_on_this_host_yields_invalid_hardware(tmp_path):
    """No monkeypatch: this host really is not genoa, so this is the honest
    real path, not a smoke fixture."""
    spec = _smoke_spec()
    record, gates = run_cell.run_cell(spec, tmp_path)
    assert record["status"] == "invalid_hardware"
    assert record["hardware_valid"] is False
    assert gates == []
    for field in run_cell.RUN_FIELDS:
        assert field in record


def test_deferred_variant_is_skipped_without_touching_hardware(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(preflight, "run_preflight", lambda: called.append(1) or _passing_hardware_report())
    spec = _smoke_spec(variant_id="bucketed_engine_serial")
    record, gates = run_cell.run_cell(spec, tmp_path)
    assert record["status"] == "skipped_variant"
    assert gates == []
    assert not called  # preflight is never even run for a deferred variant


def test_completed_run_on_tiny_synthetic_circuit(tmp_path, monkeypatch):
    """The full engine path: tiny 8-qubit circuit, one Trotter step, real
    `propagate` + `propagate_with_stats` calls, real record/gate-file writes.
    `preflight.run_preflight` is monkeypatched to a passing genoa report so
    this exercises the 'completed' branch on non-genoa test hardware."""
    monkeypatch.setattr(preflight, "run_preflight", _passing_hardware_report)
    spec = _smoke_spec()

    record, gates = run_cell.run_cell(spec, tmp_path)

    assert record["status"] == "completed"
    assert record["hardware_valid"] is True
    assert record["n_qubits"] == 8
    assert record["wall_time_s"] is not None and record["wall_time_s"] >= 0.0
    assert record["final_terms"] is not None
    for field in run_cell.RUN_FIELDS:
        assert field in record

    assert gates, "an 8-edge-free tiny lattice at 1 step still applies >=1 gate"
    for gate in gates:
        for field in run_cell.GATE_FIELDS:
            assert field in gate
        assert gate["run_id"] == record["run_id"]
        assert gate["rank_id"] == 0

    run_cell._append_jsonl(tmp_path / "runs.jsonl", [record])
    run_cell._append_jsonl(tmp_path / "gates.rank-0.jsonl", gates)

    written_runs = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    written_gates = [json.loads(line) for line in (tmp_path / "gates.rank-0.jsonl").read_text().splitlines()]
    assert len(written_runs) == 1
    assert len(written_gates) == len(gates)


def test_cell_json_rejects_unknown_fields():
    import pytest

    with pytest.raises(ValueError):
        run_cell.CellSpec.from_dict({"variant_id": "bucketed_current", "typo_field": 1})


# --------------------------------------------------------------------------
# partition_row_policy (E8)


def test_unpartitioned_cell_has_null_partition_row_policy(monkeypatch):
    """`partitions=None` never touches the row-policy machinery at all; the
    record's `partition_row_policy` is `None`, matching `engine="unpartitioned"`
    (schema.py rejects any other combination)."""
    monkeypatch.setattr(preflight, "run_preflight", _passing_hardware_report)
    spec = _smoke_spec(partition_row_policy="cut")  # ignored: partitions is None
    record, _ = run_cell.run_cell(spec, Path("/tmp"))
    assert record["engine"] == "unpartitioned"
    assert record["partition_row_policy"] is None


def test_partitioned_cell_records_its_row_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "run_preflight", _passing_hardware_report)
    spec = _smoke_spec(partitions=2, partition_row_policy="random")
    record, gates = run_cell.run_cell(spec, tmp_path)
    assert record["status"] == "completed"
    assert record["engine"] == "partitioned"
    assert record["partition_row_policy"] == "random"
    assert gates, "a partitioned completed run still emits per-layer gate records"


def test_cut_policy_uses_the_heavy_hex_lattice_and_differs_from_random(tmp_path, monkeypatch):
    """The real point of E8: a 'cut' cell and a 'random' cell on the same
    otherwise-identical spec move a different number of rows across the
    partition boundary."""
    monkeypatch.setattr(preflight, "run_preflight", _passing_hardware_report)

    random_spec = _smoke_spec(
        partitions=2, partition_row_policy="random", trotter_steps=3
    )
    cut_spec = _smoke_spec(partitions=2, partition_row_policy="cut", trotter_steps=3)

    random_record, random_gates = run_cell.run_cell(random_spec, tmp_path)
    cut_record, cut_gates = run_cell.run_cell(cut_spec, tmp_path)

    assert random_record["partition_row_policy"] == "random"
    assert cut_record["partition_row_policy"] == "cut"

    random_rows = [g["rows_exported"] for g in random_gates]
    cut_rows = [g["rows_exported"] for g in cut_gates]
    assert random_rows != cut_rows, (
        "random and cut policies must move a genuinely different row pattern, "
        "or the schema/driver plumbing isn't reaching the engine"
    )


def test_unknown_partition_row_policy_raises(monkeypatch):
    monkeypatch.setattr(preflight, "run_preflight", _passing_hardware_report)
    spec = _smoke_spec(partitions=2, partition_row_policy="not_a_real_policy")
    import pytest

    with pytest.raises(ValueError, match="partition_row_policy"):
        run_cell.run_cell(spec, Path("/tmp"))


def test_cut_blocks_partition_all_qubits_disjointly():
    spec = _smoke_spec(partitions=4)
    blocks = run_cell._cut_blocks(spec, 4)
    assert len(blocks) == 4
    seen: set[int] = set()
    for block in blocks:
        assert not (seen & set(block)), "cut blocks must be disjoint"
        seen |= set(block)
    assert seen == set(range(spec.n_qubits)), "cut blocks must cover every qubit"
