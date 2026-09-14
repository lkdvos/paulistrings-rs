"""Pure normalization functions: list[dict] in, list[dict] rows out.

Inputs are already-parsed, already-validated run/gate records.
No file I/O here; that lives in validate_campaign.py.
"""

from bisect import bisect_right


def efficiency_gates(gate_records: list[dict]) -> list[dict]:
    """One row per gate record, with a derived updates/sec rate."""
    rows = []
    for g in gate_records:
        nanos = g["nanos"]
        rate = None if nanos == 0 else g["terms_in"] / nanos * 1e9
        rows.append(
            {
                "run_id": g["run_id"],
                "application_index": g["application_index"],
                "circuit_index": g["circuit_index"],
                "gate_name": g["gate_name"],
                "terms_in": g["terms_in"],
                "nanos": nanos,
                "rate": rate,
            }
        )
    return rows


def efficiency_binned(gate_records: list[dict], bin_edges: list[float]) -> list[dict]:
    """Bin gates by terms_in into caller-given edges, per gate_name.

    Rate is the weighted aggregate sum(terms_in)/sum(nanos), never a mean of
    per-gate rates (contract.md's timing-semantics rule).
    Bins are half-open [edges[i], edges[i+1]); values below edges[0] or at/above
    edges[-1] are dropped (bin_edges must bracket the data if that's undesired).
    """
    edges = sorted(bin_edges)
    agg: dict[tuple[str, int], dict] = {}
    for g in gate_records:
        terms_in = g["terms_in"]
        idx = bisect_right(edges, terms_in) - 1
        if idx < 0 or idx >= len(edges) - 1:
            continue
        key = (g["gate_name"], idx)
        entry = agg.setdefault(
            key,
            {
                "gate_name": g["gate_name"],
                "bin_low": edges[idx],
                "bin_high": edges[idx + 1],
                "gate_count": 0,
                "sum_terms_in": 0,
                "sum_nanos": 0,
            },
        )
        entry["gate_count"] += 1
        entry["sum_terms_in"] += terms_in
        entry["sum_nanos"] += g["nanos"]

    rows = []
    for entry in agg.values():
        sum_nanos = entry["sum_nanos"]
        rate = None if sum_nanos == 0 else entry["sum_terms_in"] / sum_nanos * 1e9
        rows.append({**entry, "rate": rate})
    rows.sort(key=lambda r: (r["gate_name"], r["bin_low"]))
    return rows


def runtime_tolerance(run_records: list[dict]) -> list[dict]:
    """One row per run (completed or not); non-completed rows carry a null wall_time_s."""
    rows = []
    for r in run_records:
        wall_time_s = r["wall_time_s"] if r["status"] == "completed" else None
        rows.append(
            {
                "run_id": r["run_id"],
                "variant_id": r["variant_id"],
                "min_abs_coeff": r["min_abs_coeff"],
                "wall_time_s": wall_time_s,
                "peak_terms": r["peak_terms"],
                "peak_rss_kb": r["peak_rss_kb"],
                "status": r["status"],
            }
        )
    return rows


def thread_scaling(
    run_records: list[dict], *, variant_id: str, min_abs_coeff: float | None
) -> list[dict]:
    """One row per threads value for a fixed (variant_id, min_abs_coeff), completed runs only."""
    filtered = [
        r
        for r in run_records
        if r["variant_id"] == variant_id
        and r["min_abs_coeff"] == min_abs_coeff
        and r["status"] == "completed"
    ]
    baseline = [r for r in filtered if r["threads"] == 1]
    if not baseline:
        raise ValueError(
            f"thread_scaling: no 1-thread baseline for variant_id={variant_id!r}, "
            f"min_abs_coeff={min_abs_coeff!r}"
        )
    baseline_time = baseline[0]["wall_time_s"]

    rows = []
    for r in filtered:
        threads = r["threads"]
        wall_time_s = r["wall_time_s"]
        speedup = None if wall_time_s in (None, 0) else baseline_time / wall_time_s
        efficiency = None if speedup is None else speedup / threads
        rows.append(
            {
                "threads": threads,
                "wall_time_s": wall_time_s,
                "speedup": speedup,
                "efficiency": efficiency,
            }
        )
    rows.sort(key=lambda r: r["threads"])
    return rows


def rank_scaling(
    run_records: list[dict], *, variant_id: str, min_abs_coeff: float | None
) -> list[dict]:
    """One row per ranks value for a fixed (variant_id, min_abs_coeff), completed runs only.

    A row is only speedup/efficiency-bearing (regime="overlap") when its peak_terms
    does not exceed the max peak_terms reached by any 1-rank run in the filtered set;
    otherwise it is regime="capacity_extension" and speedup/efficiency are left null,
    per contract.md's "keep fixed-problem speedup distinct from capacity extension" rule.
    """
    filtered = [
        r
        for r in run_records
        if r["variant_id"] == variant_id
        and r["min_abs_coeff"] == min_abs_coeff
        and r["status"] == "completed"
    ]
    baseline = [r for r in filtered if r["ranks"] == 1]
    if not baseline:
        raise ValueError(
            f"rank_scaling: no 1-rank baseline for variant_id={variant_id!r}, "
            f"min_abs_coeff={min_abs_coeff!r}"
        )
    baseline_time = baseline[0]["wall_time_s"]
    one_rank_max_terms = max(
        (r["peak_terms"] for r in baseline if r["peak_terms"] is not None), default=None
    )

    rows = []
    for r in filtered:
        ranks = r["ranks"]
        wall_time_s = r["wall_time_s"]
        peak_terms = r["peak_terms"]
        is_capacity_extension = (
            one_rank_max_terms is not None
            and peak_terms is not None
            and peak_terms > one_rank_max_terms
        )
        regime = "capacity_extension" if is_capacity_extension else "overlap"
        if is_capacity_extension:
            speedup = None
            efficiency = None
        else:
            speedup = None if wall_time_s in (None, 0) else baseline_time / wall_time_s
            efficiency = None if speedup is None else speedup / ranks
        rows.append(
            {
                "ranks": ranks,
                "wall_time_s": wall_time_s,
                "peak_terms": peak_terms,
                "regime": regime,
                "speedup": speedup,
                "efficiency": efficiency,
            }
        )
    rows.sort(key=lambda r: r["ranks"])
    return rows


def hash_communication(run_records: list[dict], gate_records: list[dict]) -> list[dict]:
    """One row per completed, `partition_row_policy`-tagged run: total rows/bytes
    exported across all its layers, for a random-vs-cut communication-volume
    comparison (E8, contract.md's "Partition rows are drawn at random by
    default" + `decisions.md` #13).

    Grouped by `run_id` (not averaged/merged across runs) so the caller can
    pair same-`config_id` runs that differ only in `partition_row_policy` --
    the otherwise-identical-cell comparison this evidence slot needs.
    Raises `ValueError` if no run in `run_records` carries a non-null
    `partition_row_policy` at all -- schema/driver support with zero
    real data is a different failure than "the field doesn't exist yet"
    (the old placeholder), so this still refuses to fabricate a plot, but
    with a precise cause.
    """
    tagged = [
        r
        for r in run_records
        if r.get("partition_row_policy") is not None and r["status"] == "completed"
    ]
    if not tagged:
        raise ValueError(
            "hash_communication: no completed run in run_records carries a non-null "
            "partition_row_policy -- run at least one 'random' and one 'cut' cell "
            "(jobs/run_cell.py's partition_row_policy field) before this table has "
            "anything to compare"
        )

    gates_by_run: dict[str, list[dict]] = {}
    for g in gate_records:
        gates_by_run.setdefault(g["run_id"], []).append(g)

    rows = []
    for r in tagged:
        gates = gates_by_run.get(r["run_id"], [])
        total_rows_exported = sum(g["rows_exported"] or 0 for g in gates)
        total_bytes_exported = sum(g["bytes_exported"] or 0 for g in gates)
        rows.append(
            {
                "run_id": r["run_id"],
                "config_id": r["config_id"],
                "min_abs_coeff": r["min_abs_coeff"],
                "partition_row_policy": r["partition_row_policy"],
                "partitions": r["partitions"],
                "layers": len(gates),
                "total_rows_exported": total_rows_exported,
                "total_bytes_exported": total_bytes_exported,
            }
        )
    rows.sort(key=lambda r: (r["min_abs_coeff"], r["partition_row_policy"], r["run_id"]))
    return rows


def accuracy(run_records: list[dict]) -> list[dict]:
    """One row per (Rust, Julia) pair of completed runs of the same cell.

    Neither driver's own record holds both a "reference" and an "observed"
    value -- `jobs/run_cell.py` and `jobs/run_cell_julia.py` each write
    `extra.expectation_re`/`expectation_im` for their own engine's run only
    (decisions.md #20). This pairs a `pauli_propagation_jl` run (the external
    reference) with every other completed run sharing its `(min_abs_coeff,
    n_qubits)` -- the two campaign quantities that change the observable's
    true value; `theta_h`/`trotter_steps` are fixed campaign-wide -- and
    reports the observed-minus-reference delta. Runs with no matching partner
    on either side are silently skipped, not padded with `None`s: a lone
    Rust or Julia run proves nothing about accuracy by itself.
    """
    def _exp(r: dict) -> complex | None:
        extra = r.get("extra") or {}
        re, im = extra.get("expectation_re"), extra.get("expectation_im")
        return None if re is None else complex(re, im or 0.0)

    completed = [r for r in run_records if r["status"] == "completed" and _exp(r) is not None]
    references = [r for r in completed if r["engine"] == "pauli_propagation_jl"]
    observed = [r for r in completed if r["engine"] != "pauli_propagation_jl"]

    rows = []
    for ref in references:
        key = (ref["min_abs_coeff"], ref["n_qubits"])
        for obs in observed:
            if (obs["min_abs_coeff"], obs["n_qubits"]) != key:
                continue
            ref_val, obs_val = _exp(ref), _exp(obs)
            rows.append(
                {
                    "run_id": obs["run_id"],
                    "reference_run_id": ref["run_id"],
                    "min_abs_coeff": key[0],
                    "n_qubits": key[1],
                    "reference_value": ref_val.real,
                    "observed_value": obs_val.real,
                    "abs_delta": abs(obs_val - ref_val),
                }
            )
    rows.sort(key=lambda r: (r["min_abs_coeff"], r["run_id"]))
    return rows


def memory_model(run_records: list[dict]) -> list[dict]:
    """One row per completed run: bytes_per_term = peak_rss_kb*1024/peak_terms."""
    rows = []
    for r in run_records:
        if r["status"] != "completed":
            continue
        peak_rss_kb = r["peak_rss_kb"]
        peak_terms = r["peak_terms"]
        bytes_per_term = (
            None
            if peak_rss_kb is None or not peak_terms
            else peak_rss_kb * 1024 / peak_terms
        )
        rows.append(
            {
                "run_id": r["run_id"],
                "n_qubits": r["n_qubits"],
                "peak_terms": peak_terms,
                "peak_rss_kb": peak_rss_kb,
                "bytes_per_term": bytes_per_term,
            }
        )
    return rows
