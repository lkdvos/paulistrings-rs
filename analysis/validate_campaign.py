#!/usr/bin/env python3
"""CLI validation for a campaign's runs.jsonl and gates.rank-N.jsonl files.

Usage: validate_campaign.py <runs.jsonl> <gates.jsonl>...
The first argument is treated as the run-record file, every remaining
argument as a gate-record file (any number, e.g. one per rank).
Prints one problem per line, then a pass/fail summary; exits 1 if any
problem was found, else 0.
"""

import json
import sys

from schema import validate_gate, validate_run


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                records.append({"__parse_error__": f"{path}:{lineno}: {e}"})
    return records


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1

    runs_path = argv[0]
    gates_paths = argv[1:]

    problems: list[str] = []

    run_records = _load_jsonl(runs_path)
    runs_by_id: dict[str, dict] = {}
    for i, r in enumerate(run_records):
        if "__parse_error__" in r:
            problems.append(r["__parse_error__"])
            continue
        for p in validate_run(r):
            problems.append(f"{runs_path}#{i} (run_id={r.get('run_id')!r}): {p}")
        run_id = r.get("run_id")
        if run_id in runs_by_id:
            problems.append(f"{runs_path}#{i}: duplicate run_id {run_id!r}")
        elif run_id is not None:
            runs_by_id[run_id] = r

    gate_records: list[dict] = []
    for path in gates_paths:
        recs = _load_jsonl(path)
        for i, g in enumerate(recs):
            if "__parse_error__" in g:
                problems.append(g["__parse_error__"])
                continue
            for p in validate_gate(g):
                problems.append(f"{path}#{i} (run_id={g.get('run_id')!r}): {p}")
            gate_records.append(g)

    # Orphan check: every gate record's run_id must join to a known run.
    gates_per_run: dict[str, list[dict]] = {}
    for i, g in enumerate(gate_records):
        run_id = g.get("run_id")
        if run_id is not None and run_id not in runs_by_id:
            problems.append(
                f"orphaned gate record (run_id={run_id!r} not found in {runs_path})"
            )
        gates_per_run.setdefault(run_id, []).append(g)

    # Gate-trace length vs. the run's own final_terms/layer count, where derivable.
    # We can only check something concrete here: a run with trace_enabled=True and
    # a completed status should have at least one gate record; a run with
    # trace_enabled=False should have none.
    for run_id, run in runs_by_id.items():
        recs = gates_per_run.get(run_id, [])
        if run.get("trace_enabled") is True and run.get("status") == "completed" and not recs:
            problems.append(
                f"run_id={run_id!r}: trace_enabled=True and status=completed but no gate records found"
            )
        if run.get("trace_enabled") is False and recs:
            problems.append(
                f"run_id={run_id!r}: trace_enabled=False but {len(recs)} gate records found"
            )

    # status != completed must never carry a speedup-bearing wall_time_s downstream;
    # here we check the run record itself already respects that (defense in depth
    # with schema.validate_run's own check).
    for run_id, run in runs_by_id.items():
        if run.get("status") != "completed" and run.get("wall_time_s") is not None:
            problems.append(
                f"run_id={run_id!r}: status={run.get('status')!r} but wall_time_s is non-null "
                "(must be null so it never enters a speedup ratio)"
            )

    for p in problems:
        print(p)

    n_runs = len(run_records)
    n_gates = len(gate_records)
    if problems:
        print(f"FAIL: {len(problems)} problem(s) found across {n_runs} run(s), {n_gates} gate record(s)")
        return 1
    print(f"PASS: {n_runs} run(s), {n_gates} gate record(s), 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
