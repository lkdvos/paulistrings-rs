"""Schema v1 for run and gate records.

Plain-dict validators, not pydantic (not installed in .venv).
`validate_run`/`validate_gate` never raise; they return a list of problem strings.
"""

import math

SCHEMA_VERSION = 1

_DIRECTIONS = {"forward", "heisenberg"}
_ENGINES = {"unpartitioned", "partitioned", "distributed"}
_STATUSES = {
    "completed",
    "invalid_hardware",
    "numerical_mismatch",
    "oom",
    "timeout",
    "scheduler_failure",
    "build_failure",
    "cancelled",
    "other",
}

# Fields whose value is a real measurement of run duration/memory/term counts.
# A run whose status != "completed" should not carry a real number here;
# an exact 0 or NaN on such a field is treated as a suspicious placeholder
# rather than a genuine measurement, since the handoff's rule is "null +
# reason, never 0/NaN as a stand-in for unavailable".
_SUSPICIOUS_ZERO_ON_FAILURE = (
    "wall_time_s",
    "setup_time_s",
    "scatter_time_s",
    "gather_time_s",
    "peak_rss_kb",
)


def _is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def _require(record, field):
    return field in record and record[field] is not None


def _check_type(record, field, types, problems):
    if field not in record:
        problems.append(f"missing required field '{field}'")
        return False
    if record[field] is None:
        problems.append(f"field '{field}' is null but is required")
        return False
    if not isinstance(record[field], types):
        problems.append(
            f"field '{field}' has type {type(record[field]).__name__}, expected {types}"
        )
        return False
    return True


def _check_nullable_type(record, field, types, problems):
    """Field must be present (possibly null); if non-null, must match types."""
    if field not in record:
        problems.append(f"missing field '{field}' (may be null but must be present)")
        return False
    if record[field] is None:
        return True
    if not isinstance(record[field], types):
        problems.append(
            f"field '{field}' has type {type(record[field]).__name__}, expected {types} or null"
        )
        return False
    return True


def _check_nonneg(record, field, problems, allow_null=True):
    if field not in record:
        problems.append(f"missing field '{field}'")
        return
    v = record[field]
    if v is None:
        if not allow_null:
            problems.append(f"field '{field}' must not be null")
        return
    if _is_nan(v):
        problems.append(f"field '{field}' is NaN, use null with a '<field>_reason' instead")
        return
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        problems.append(f"field '{field}' must be numeric, got {type(v).__name__}")
        return
    if v < 0:
        problems.append(f"field '{field}' must be >= 0, got {v}")


def _check_reason_pair(record, field, problems):
    """If field is null, a '<field>_reason' string should explain why."""
    if field in record and record[field] is None:
        reason_key = f"{field}_reason"
        reason = record.get(reason_key)
        if not isinstance(reason, str) or not reason.strip():
            problems.append(
                f"field '{field}' is null but '{reason_key}' is missing or empty"
            )


def validate_run(record: dict) -> list[str]:
    """Validate a run record (one row of runs.jsonl). Returns a list of problems, empty if valid."""
    problems: list[str] = []

    if record.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version must be {SCHEMA_VERSION}, got {record.get('schema_version')!r}"
        )

    for field in ("campaign_id", "run_id", "task_id", "config_id", "variant_id"):
        _check_type(record, field, str, problems)

    _check_type(record, "repetition_index", int, problems)
    if isinstance(record.get("repetition_index"), int) and record["repetition_index"] < 0:
        problems.append("repetition_index must be >= 0")

    _check_nullable_type(record, "pair_index", int, problems)

    if _check_type(record, "source_commit", str, problems):
        sc = record["source_commit"]
        if sc != "unknown" and not (len(sc) == 40 and all(c in "0123456789abcdef" for c in sc.lower())):
            problems.append("source_commit must be 40-hex or 'unknown'")

    _check_nullable_type(record, "dirty", bool, problems)

    if _check_type(record, "build_features", list, problems):
        if not all(isinstance(f, str) for f in record["build_features"]):
            problems.append("build_features must be a list of str")

    for field in ("compiler_version", "runtime_version"):
        _check_type(record, field, str, problems)

    if _check_type(record, "n_qubits", int, problems) and record["n_qubits"] <= 0:
        problems.append("n_qubits must be > 0")

    if _check_type(record, "direction", str, problems) and record["direction"] not in _DIRECTIONS:
        problems.append(f"direction must be one of {_DIRECTIONS}, got {record['direction']!r}")

    _check_type(record, "state", str, problems)

    _check_nullable_type(record, "min_abs_coeff", (int, float), problems)
    if record.get("min_abs_coeff") is not None and record["min_abs_coeff"] < 0:
        problems.append("min_abs_coeff must be >= 0")

    _check_nullable_type(record, "max_weight", int, problems)

    _check_type(record, "policy", str, problems)

    if _check_type(record, "engine", str, problems) and record["engine"] not in _ENGINES:
        problems.append(f"engine must be one of {_ENGINES}, got {record['engine']!r}")

    for field in ("partitions", "threads", "ranks"):
        if _check_type(record, field, int, problems) and record[field] < 1:
            problems.append(f"{field} must be >= 1")

    _check_nullable_type(record, "slurm_job_id", str, problems)
    _check_type(record, "node_class", str, problems)
    _check_type(record, "hardware_contract_id", str, problems)
    _check_type(record, "hardware_valid", bool, problems)
    _check_type(record, "trace_enabled", bool, problems)

    status = record.get("status")
    if _check_type(record, "status", str, problems) and status not in _STATUSES:
        problems.append(f"status must be one of {_STATUSES}, got {status!r}")

    _check_nullable_type(record, "failure_reason", str, problems)
    if status is not None and status != "completed" and not record.get("failure_reason"):
        problems.append("failure_reason must be non-null when status != 'completed'")
    if status == "completed" and record.get("failure_reason"):
        problems.append("failure_reason should be null when status == 'completed'")
    if status is not None and status != "completed" and record.get("wall_time_s") is not None:
        problems.append(
            f"wall_time_s must be null on a non-completed run (status={status!r}); "
            "never fabricate a duration for a failed/incomplete run"
        )

    for field in ("wall_time_s", "setup_time_s", "scatter_time_s", "gather_time_s"):
        _check_nullable_type(record, field, (int, float), problems)
        _check_nonneg(record, field, problems)
    _check_reason_pair(record, "wall_time_s", problems)

    for field in ("initial_terms", "final_terms", "peak_terms"):
        _check_nullable_type(record, field, int, problems)
        _check_nonneg(record, field, problems)

    _check_nullable_type(record, "peak_rss_kb", (int, float), problems)
    _check_nonneg(record, "peak_rss_kb", problems)
    _check_nullable_type(record, "peak_rss_provenance", str, problems)

    for field in ("log_path", "gate_trace_path"):
        _check_nullable_type(record, field, str, problems)

    # defensive 0/NaN-as-placeholder check: a non-completed run's timing/memory
    # fields should not carry an exact numeric 0 (a real measurement of "no
    # progress at all" is vanishingly unlikely and almost certainly a sentinel).
    if status is not None and status != "completed":
        for field in _SUSPICIOUS_ZERO_ON_FAILURE:
            v = record.get(field)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
                problems.append(
                    f"field '{field}' is exactly 0 on a non-completed run (status={status!r}); "
                    "use null + '<field>_reason' instead of 0 as a stand-in"
                )

    # extra is schema v1's one open-ended field (see normalize.py:accuracy).
    if "extra" in record and record["extra"] is not None and not isinstance(record["extra"], dict):
        problems.append("extra, if present, must be a dict or null")

    return problems


def validate_gate(record: dict, *, channels_per_step: int | None = None) -> list[str]:
    """Validate a gate record (one row of gates.rank-N.jsonl). Returns a list of problems."""
    problems: list[str] = []

    if record.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version must be {SCHEMA_VERSION}, got {record.get('schema_version')!r}"
        )

    _check_type(record, "run_id", str, problems)

    if _check_type(record, "rank_id", int, problems) and record["rank_id"] < 0:
        problems.append("rank_id must be >= 0")

    for field in ("application_index", "circuit_index"):
        if _check_type(record, field, int, problems) and record[field] < 0:
            problems.append(f"{field} must be >= 0")

    _check_nullable_type(record, "trotter_step", int, problems)
    if record.get("trotter_step") is not None and record["trotter_step"] < 0:
        problems.append("trotter_step must be >= 0")
    if (
        channels_per_step is not None
        and record.get("trotter_step") is not None
        and record.get("circuit_index") is not None
    ):
        expected = record["circuit_index"] // channels_per_step
        if record["trotter_step"] != expected:
            problems.append(
                f"trotter_step {record['trotter_step']} does not match "
                f"circuit_index // channels_per_step = {expected}"
            )

    _check_type(record, "gate_name", str, problems)
    _check_nullable_type(record, "support_weight", int, problems)
    _check_nonneg(record, "support_weight", problems)

    for field in ("terms_in", "terms_out"):
        if _check_type(record, field, int, problems) and record[field] < 0:
            problems.append(f"{field} must be >= 0")

    if _check_type(record, "nanos", int, problems) and record["nanos"] < 0:
        problems.append("nanos must be >= 0")

    _check_nullable_type(record, "bucket_bits", int, problems)
    _check_nonneg(record, "bucket_bits", problems)
    for field in ("rows_exported", "bytes_exported", "partner_count"):
        _check_nullable_type(record, field, int, problems)
        _check_nonneg(record, field, problems)

    return problems
