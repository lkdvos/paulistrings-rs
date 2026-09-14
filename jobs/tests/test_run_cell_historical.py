"""Tests for `run_cell_historical.py`.

Fast tests (registry shape, cell.json parsing, hardware gating) run every
time. A real historical build (`git worktree add` + `cargo build`/`maturin
develop` against one of the four frozen commits) takes tens of seconds to
minutes and is genuinely slow, in the same spirit as `test_run_cell_
distributed.py`'s MPI-build tests -- skipped by default, gated on the
`PYTEST_RUN_SLOW_HISTORICAL_BUILDS=1` environment variable (no `slow` pytest
marker is registered anywhere else in this suite, so a skipif env-var gate
matches the existing convention instead of introducing one).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_RUN_SLOW = os.environ.get("PYTEST_RUN_SLOW_HISTORICAL_BUILDS") == "1"
_SLOW_SKIP_REASON = (
    "real worktree + cargo/maturin build; set PYTEST_RUN_SLOW_HISTORICAL_BUILDS=1 to run"
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import preflight  # noqa: E402
import run_cell_historical as rch  # noqa: E402


def _passing_hardware_report():
    return {
        "hardware_contract_id": preflight.HARDWARE_CONTRACT_ID,
        "node_class_guess": "genoa",
        "smt_siblings_present": False,
        "affinity_is_physical_core_only": True,
        "preflight_passed": True,
    }


def _spec(**overrides):
    data = dict(
        variant_id="naive_baseline",
        min_abs_coeff=1e-6,
        direction="heisenberg",
        repetition_index=0,
    )
    data.update(overrides)
    return rch.HistoricalCellSpec.from_dict(data)


def test_registry_covers_the_four_historical_variants():
    """Exactly the four variants the task requires; the fifth
    (presentation_bench_crate_variants) is explicitly out of scope."""
    assert set(rch.VARIANT_REGISTRY) == {
        "naive_baseline",
        "direct_small_sum_path",
        "bucketed_engine_serial",
        "bucketed_engine_parallel",
    }
    for entry in rch.VARIANT_REGISTRY.values():
        assert entry.strategy in ("rust_harness", "pyo3_handrolled")
        assert len(entry.commit_sha) == 40


def test_registry_shas_match_contract_md():
    """Cross-check against tasks/T01-variants.json's frozen hashes, so a typo
    here can't silently drift from the campaign's frozen variant registry."""
    expected = {
        "naive_baseline": "d410f4e5985ad917146867be31511143fde8f893",
        "direct_small_sum_path": "e56f021e54f3f64c3ddb8e2f688c39d91433d721",
        "bucketed_engine_serial": "f08db7df8bcd771f25383db0120e111cfb018bd2",
        "bucketed_engine_parallel": "ef037012e645d4f63013f26eaf3dbd6ce6299660",
    }
    for variant_id, sha in expected.items():
        assert rch.VARIANT_REGISTRY[variant_id].commit_sha == sha


def test_cell_json_rejects_unknown_fields():
    with pytest.raises(ValueError):
        rch.HistoricalCellSpec.from_dict({"variant_id": "naive_baseline", "bogus": 1})


def test_unknown_variant_id_is_recorded_not_raised(tmp_path):
    spec = _spec(variant_id="not_a_real_variant")
    record = rch.run_cell(spec, tmp_path, tmp_path / "scratch")
    assert record["status"] == "other"
    assert "not_a_real_variant" in record["failure_reason"]
    for field in rch.RUN_FIELDS:
        assert field in record


def test_real_preflight_on_this_host_yields_invalid_hardware(tmp_path):
    """No monkeypatch, no PS_HIST_SKIP_PREFLIGHT: this host is not genoa, so
    this is the honest real gating path, not a smoke fixture."""
    spec = _spec()
    record = rch.run_cell(spec, tmp_path, tmp_path / "scratch")
    assert record["status"] == "invalid_hardware"
    assert record["hardware_valid"] is False


def test_skip_preflight_env_var_bypasses_gate_before_the_build(monkeypatch, tmp_path):
    """PS_HIST_SKIP_PREFLIGHT is documented as local-dev-only; verify it does
    what it claims without actually invoking cargo/maturin (monkeypatch the
    strategy functions so this test stays fast)."""
    monkeypatch.setenv("PS_HIST_SKIP_PREFLIGHT", "1")
    called = {}

    def fake_naive(spec, worktree, scratch):
        called["ran"] = True
        return {"wall_time_s": 0.001, "initial_terms": 1, "final_terms": 3, "peak_rss_kb": 1000}

    monkeypatch.setattr(rch, "_run_naive_rust_harness", fake_naive)
    # Avoid a real `git worktree add` against the live repo in a unit test.
    monkeypatch.setattr(
        rch.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
    )
    spec = _spec()
    record = rch.run_cell(spec, tmp_path, tmp_path / "scratch")
    assert called.get("ran") is True
    assert record["status"] == "completed"
    assert record["final_terms"] == 3
    assert record["source_commit"] == rch.VARIANT_REGISTRY["naive_baseline"].commit_sha


def test_empty_run_record_has_every_run_field(tmp_path):
    record = rch._empty_run_record(_spec(), "run-id", "other", "some reason", None)
    for field in rch.RUN_FIELDS:
        assert field in record


@pytest.mark.skipif(not _RUN_SLOW, reason=_SLOW_SKIP_REASON)
@pytest.mark.parametrize("variant_id", sorted(rch.VARIANT_REGISTRY))
def test_real_historical_build_produces_a_schema_valid_record(variant_id, tmp_path, monkeypatch):
    """A real end-to-end run: git worktree checkout, real cargo/maturin build,
    real execution, real schema validation. Slow (a fresh PyO3 build can take
    a minute or more) -- opt in with `pytest -m slow`."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from analysis.schema import validate_run

    monkeypatch.setenv("PS_HIST_SKIP_PREFLIGHT", "1")
    spec = _spec(variant_id=variant_id, config_id=f"slow-test-{variant_id}")
    record = rch.run_cell(spec, tmp_path, tmp_path / "scratch")
    assert record["status"] == "completed", record.get("failure_reason")
    assert record["final_terms"] > 0
    assert validate_run(record) == []
