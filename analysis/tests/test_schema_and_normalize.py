import pytest

from normalize import efficiency_binned, rank_scaling, thread_scaling
from schema import validate_gate, validate_run


def make_run(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "campaign_id": "campaign-2026-09-11",
        "run_id": "run-0001",
        "task_id": "T06",
        "config_id": "cfg-0001",
        "variant_id": "bucketed_engine_parallel",
        "repetition_index": 0,
        "pair_index": None,
        "source_commit": "a" * 40,
        "dirty": False,
        "build_features": [],
        "compiler_version": "rustc 1.82.0",
        "runtime_version": "python 3.11.11",
        "n_qubits": 127,
        "direction": "heisenberg",
        "state": "z+",
        "min_abs_coeff": 1e-6,
        "max_weight": None,
        "policy": "approx_topn",
        "engine": "unpartitioned",
        "partitions": 1,
        "threads": 8,
        "ranks": 1,
        "slurm_job_id": None,
        "node_class": "genoa",
        "hardware_contract_id": "genoa-ccq-v1",
        "hardware_valid": True,
        "trace_enabled": True,
        "wall_time_s": 12.5,
        "setup_time_s": 0.1,
        "scatter_time_s": None,
        "gather_time_s": None,
        "initial_terms": 1,
        "final_terms": 1000,
        "peak_terms": 1200,
        "peak_rss_kb": 500_000.0,
        "peak_rss_provenance": "getrusage",
        "status": "completed",
        "failure_reason": None,
        "log_path": None,
        "gate_trace_path": "gates.rank-0.jsonl",
    }
    base.update(overrides)
    return base


def make_gate(**overrides) -> dict:
    base = {
        "schema_version": 1,
        "run_id": "run-0001",
        "rank_id": 0,
        "application_index": 0,
        "circuit_index": 0,
        "trotter_step": None,
        "gate_name": "rx",
        "support_weight": 1,
        "terms_in": 10,
        "terms_out": 12,
        "nanos": 500,
        "bucket_bits": None,
        "rows_exported": None,
        "bytes_exported": None,
        "partner_count": None,
    }
    base.update(overrides)
    return base


def test_valid_run_record_passes():
    assert validate_run(make_run()) == []


def test_valid_gate_record_passes():
    assert validate_gate(make_gate()) == []


def test_run_missing_required_field_fails_with_clear_message():
    record = make_run()
    del record["n_qubits"]
    problems = validate_run(record)
    assert any("n_qubits" in p for p in problems)


def test_oom_run_with_nonnull_wall_time_is_flagged():
    record = make_run(
        status="oom",
        failure_reason="OOM killed by cgroup",
        wall_time_s=99.9,
    )
    problems = validate_run(record)
    assert any("wall_time_s" in p and "non-completed" in p for p in problems)


def test_oom_run_without_failure_reason_is_flagged():
    record = make_run(status="oom", wall_time_s=None, failure_reason=None)
    problems = validate_run(record)
    assert any("failure_reason" in p for p in problems)


def test_gate_trotter_step_derivation_mismatch_flagged():
    record = make_gate(circuit_index=7, trotter_step=0)
    problems = validate_gate(record, channels_per_step=2)
    assert any("trotter_step" in p for p in problems)


def test_gate_trotter_step_derivation_correct_passes():
    record = make_gate(circuit_index=7, trotter_step=3)
    problems = validate_gate(record, channels_per_step=2)
    assert problems == []


def test_efficiency_binned_weighted_aggregate():
    gates = [
        make_gate(gate_name="rx", terms_in=10, nanos=100),
        make_gate(gate_name="rx", terms_in=20, nanos=10),
    ]
    rows = efficiency_binned(gates, bin_edges=[0, 100])
    assert len(rows) == 1
    row = rows[0]
    assert row["gate_name"] == "rx"
    assert row["gate_count"] == 2
    assert row["sum_terms_in"] == 30
    assert row["sum_nanos"] == 110
    # weighted rate: sum(terms_in)/sum(nanos), not mean((10/100),(20/10))
    expected_rate = 30 / 110 * 1e9
    assert row["rate"] == pytest.approx(expected_rate)
    mean_of_rates = ((10 / 100 * 1e9) + (20 / 10 * 1e9)) / 2
    assert row["rate"] != pytest.approx(mean_of_rates)


def test_efficiency_binned_keeps_gate_families_separate():
    gates = [
        make_gate(gate_name="rx", terms_in=10, nanos=100),
        make_gate(gate_name="pauli_rotation", terms_in=10, nanos=100),
    ]
    rows = efficiency_binned(gates, bin_edges=[0, 100])
    assert len(rows) == 2
    names = {r["gate_name"] for r in rows}
    assert names == {"rx", "pauli_rotation"}


def test_thread_scaling_missing_baseline_raises():
    runs = [make_run(threads=8, wall_time_s=10.0)]
    with pytest.raises(ValueError, match="1-thread baseline"):
        thread_scaling(runs, variant_id="bucketed_engine_parallel", min_abs_coeff=1e-6)


def test_thread_scaling_speedup_and_efficiency():
    runs = [
        make_run(run_id="r1", threads=1, wall_time_s=100.0),
        make_run(run_id="r2", threads=4, wall_time_s=30.0),
    ]
    rows = thread_scaling(runs, variant_id="bucketed_engine_parallel", min_abs_coeff=1e-6)
    row4 = next(r for r in rows if r["threads"] == 4)
    assert row4["speedup"] == pytest.approx(100.0 / 30.0)
    assert row4["efficiency"] == pytest.approx((100.0 / 30.0) / 4)


def test_rank_scaling_capacity_extension_vs_overlap():
    runs = [
        make_run(run_id="r1", ranks=1, wall_time_s=100.0, peak_terms=1000),
        make_run(run_id="r2", ranks=2, wall_time_s=60.0, peak_terms=900),
        make_run(run_id="r3", ranks=4, wall_time_s=200.0, peak_terms=5000),
    ]
    rows = rank_scaling(runs, variant_id="bucketed_engine_parallel", min_abs_coeff=1e-6)
    row_overlap = next(r for r in rows if r["ranks"] == 2)
    row_capacity = next(r for r in rows if r["ranks"] == 4)
    assert row_overlap["regime"] == "overlap"
    assert row_overlap["speedup"] == pytest.approx(100.0 / 60.0)
    assert row_capacity["regime"] == "capacity_extension"
    assert row_capacity["speedup"] is None
    assert row_capacity["efficiency"] is None
