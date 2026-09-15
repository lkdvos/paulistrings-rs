import pytest

from normalize import efficiency_binned, hash_communication, rank_scaling, thread_scaling
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
        "partition_row_policy": None,
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


def test_partition_row_policy_null_on_unpartitioned_passes():
    assert validate_run(make_run(engine="unpartitioned", partition_row_policy=None)) == []


def test_partition_row_policy_random_on_partitioned_passes():
    record = make_run(engine="partitioned", partitions=2, partition_row_policy="random")
    assert validate_run(record) == []


def test_partition_row_policy_cut_on_distributed_passes():
    record = make_run(
        engine="distributed", partitions=2, ranks=2, partition_row_policy="cut"
    )
    assert validate_run(record) == []


def test_partition_row_policy_bad_value_flagged():
    record = make_run(engine="partitioned", partitions=2, partition_row_policy="quantum")
    problems = validate_run(record)
    assert any("partition_row_policy" in p for p in problems)


def test_partition_row_policy_nonnull_on_unpartitioned_flagged():
    record = make_run(engine="unpartitioned", partition_row_policy="random")
    problems = validate_run(record)
    assert any("partition_row_policy" in p and "unpartitioned" in p for p in problems)


def test_partition_row_policy_missing_field_flagged():
    record = make_run(engine="partitioned", partitions=2, partition_row_policy="random")
    del record["partition_row_policy"]
    problems = validate_run(record)
    assert any("partition_row_policy" in p for p in problems)


# --------------------------------------------------------------------------
# hash_communication (E8)


def make_partitioned_run(*, run_id, policy, config_id="cfg-e8", rows_exported, bytes_exported, min_abs_coeff=1e-6):
    """A completed partitioned run plus its own gate records, both schema-shaped."""
    run = make_run(
        run_id=run_id,
        config_id=config_id,
        engine="partitioned",
        partitions=2,
        partition_row_policy=policy,
        min_abs_coeff=min_abs_coeff,
    )
    gates = [
        make_gate(
            run_id=run_id,
            circuit_index=k,
            application_index=k,
            rows_exported=r,
            bytes_exported=b,
        )
        for k, (r, b) in enumerate(zip(rows_exported, bytes_exported))
    ]
    return run, gates


def test_hash_communication_raises_with_no_policy_tagged_runs():
    runs = [make_run(engine="unpartitioned", partition_row_policy=None)]
    with pytest.raises(ValueError, match="partition_row_policy"):
        hash_communication(runs, [])


def test_hash_communication_compares_random_vs_cut_by_config():
    random_run, random_gates = make_partitioned_run(
        run_id="r-random", policy="random", rows_exported=[100, 200], bytes_exported=[1000, 2000]
    )
    cut_run, cut_gates = make_partitioned_run(
        run_id="r-cut", policy="cut", rows_exported=[10, 20], bytes_exported=[100, 200]
    )
    rows = hash_communication([random_run, cut_run], random_gates + cut_gates)
    assert len(rows) == 2
    by_policy = {r["partition_row_policy"]: r for r in rows}
    assert by_policy["random"]["config_id"] == "cfg-e8"
    assert by_policy["random"]["total_rows_exported"] == 300
    assert by_policy["random"]["total_bytes_exported"] == 3000
    assert by_policy["cut"]["total_rows_exported"] == 30
    assert by_policy["cut"]["total_bytes_exported"] == 300
    # The whole point of E8: the cut policy exports less than the random draw.
    assert by_policy["cut"]["total_rows_exported"] < by_policy["random"]["total_rows_exported"]
    assert by_policy["random"]["min_abs_coeff"] == 1e-6


def test_hash_communication_sorts_by_cutoff_across_configs():
    """Rows carry min_abs_coeff so a caller can build a cutoff-sweep figure,
    not just a per-config_id bar chart (the same "single point is a weak plot"
    upgrade as accuracy -> convergence)."""
    loose_run, loose_gates = make_partitioned_run(
        run_id="r-loose", policy="random", config_id="cfg-loose",
        rows_exported=[10], bytes_exported=[100], min_abs_coeff=1e-4,
    )
    tight_run, tight_gates = make_partitioned_run(
        run_id="r-tight", policy="random", config_id="cfg-tight",
        rows_exported=[1000], bytes_exported=[10000], min_abs_coeff=1e-8,
    )
    rows = hash_communication([tight_run, loose_run], tight_gates + loose_gates)
    assert [r["min_abs_coeff"] for r in rows] == [1e-8, 1e-4]


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
