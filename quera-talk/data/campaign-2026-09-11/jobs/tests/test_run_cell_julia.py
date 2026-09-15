"""Plumbing smoke tests for `run_cell_julia.py`, run against the real
`julia`/PauliPropagation.jl installation on this host (no mocking of the
runner itself -- only `preflight.run_preflight` is monkeypatched, exactly as
`test_run_cell.py` does for the Rust driver, since this host is not `genoa`).

Every "completed" test below really shells out to
`julia --project=benchmarks/julia benchmarks/julia/runner.jl` on a tiny
synthetic circuit (`n_qubits=8`, 1 Trotter step) -- these are slow (~1-2s
cold-start) real subprocess tests, not fixtures, and are skipped outright if
`julia_baseline.skip_reason()` is non-`None` on this host.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_JOBS_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _JOBS_DIR.parents[2]
sys.path.insert(0, str(_JOBS_DIR))
sys.path.insert(0, str(_JOBS_DIR.parent / "analysis"))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "python"))

import julia_baseline  # noqa: E402
import preflight  # noqa: E402
import run_cell_julia as rcj  # noqa: E402
import schema  # noqa: E402

_JULIA_SKIP_REASON = julia_baseline.skip_reason()
requires_julia = pytest.mark.skipif(
    _JULIA_SKIP_REASON is not None, reason=_JULIA_SKIP_REASON or ""
)


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
    return rcj.CellSpec.from_dict(data)


def test_real_preflight_on_this_host_yields_invalid_hardware(tmp_path):
    """No monkeypatch, no julia subprocess: this host really is not genoa."""
    spec = _smoke_spec()
    record, gates = rcj.run_cell_julia(spec, tmp_path)
    assert record["status"] == "invalid_hardware"
    assert record["hardware_valid"] is False
    assert record["engine"] == "pauli_propagation_jl"
    assert record["variant_id"] == "external_pauli_propagation_jl"
    assert gates == []
    for field in rcj.RUN_FIELDS:
        assert field in record
    assert schema.validate_run(record) == []


@requires_julia
def test_completed_run_on_tiny_synthetic_circuit(tmp_path, monkeypatch):
    """The full path: tiny 8-qubit circuit, one Trotter step, a real
    `runner.jl` subprocess, a real run-record write -- schema-clean."""
    monkeypatch.setattr(preflight, "run_preflight", _passing_hardware_report)
    spec = _smoke_spec()

    record, gates = rcj.run_cell_julia(spec, tmp_path, warm_repeats=1, timeout=120.0)

    assert record["status"] == "completed"
    assert record["hardware_valid"] is True
    assert record["engine"] == "pauli_propagation_jl"
    assert record["n_qubits"] == 8
    assert record["wall_time_s"] is not None and record["wall_time_s"] >= 0.0
    assert record["final_terms"] is not None
    assert record["trace_enabled"] is False
    assert record["gate_trace_path"] is None
    assert gates == [], "the Julia leg never emits gate records (decision #10)"
    assert record["extra"]["per_layer_terms"], "PP_LAYER_COUNTS=1 by default"

    for field in rcj.RUN_FIELDS:
        assert field in record
    problems = schema.validate_run(record)
    assert problems == [], problems

    rcj._append_jsonl(tmp_path / "runs.jsonl", [record])
    written = [line for line in (tmp_path / "runs.jsonl").read_text().splitlines() if line]
    assert len(written) == 1


@requires_julia
def test_completed_run_matches_direct_rust_leg_term_count(tmp_path, monkeypatch):
    """Cross-check: the same tiny circuit through both drivers agrees on
    final term count -- a miniature version of decision #10's 5-step pilot,
    kept cheap (1 step, 8 qubits) so it runs on every test invocation rather
    than needing a real cluster allocation."""
    monkeypatch.setattr(preflight, "run_preflight", _passing_hardware_report)
    import run_cell

    spec = _smoke_spec()
    rust_record, _ = run_cell.run_cell(spec, tmp_path / "rust")
    jl_record, _ = rcj.run_cell_julia(spec, tmp_path / "jl", warm_repeats=1, timeout=120.0)

    assert rust_record["status"] == "completed"
    assert jl_record["status"] == "completed"
    assert rust_record["final_terms"] == jl_record["final_terms"]


def test_cell_json_rejects_unknown_fields():
    with pytest.raises(ValueError):
        rcj.CellSpec.from_dict({"variant_id": "bucketed_current", "typo_field": 1})
