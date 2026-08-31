#!/usr/bin/env python3
"""Summarize and diff Criterion 0.5 benchmark output for paulistrings-rs.

Criterion 0.5 lays out one directory per benchmark under
``target/criterion/``, at a nesting depth that varies with how the bench was
grouped:

    target/criterion/<group>/<bench-id>/new/estimates.json
    target/criterion/<group>/new/estimates.json

(confirmed against this repo's own tree: e.g.
``apply_layer_bucketed/depolarizing_1000000/new`` is 2 levels deep,
``thread_scaling_bucketed_rotation_1e6/32/new`` is 2 levels deep with a
numeric parameter directory, and ``pauli_string_mul_assign/W=1/new`` is a
group/function pair -- there is no fixed depth). This tool therefore walks
*any* directory named ``new`` that contains an ``estimates.json`` and, for
each one, prefers the ``full_id`` recorded in the sibling ``new/benchmark.json``
(e.g. ``"apply_layer_bucketed/depolarizing/1000000"``) over a path derived
from the directory names (whose components use ``_`` where ``full_id`` uses
``/``, so they are not interchangeable -- the JSON is only a fallback for
older Criterion output that lacks ``benchmark.json``).

``estimates.json`` timings are nanoseconds throughout (Criterion's own unit).

Subcommands:
  snapshot OUT.json [--filter SUBSTR] [--merge]
      Walk target/criterion and write a JSON map of
      full_id -> {median_ns, mean_ns, stddev_ns, throughput_elems, melem_per_s}.
      With --merge, an existing OUT.json is loaded first and updated with
      the newly collected entries (new entries win on key collisions)
      before writing back, instead of overwriting OUT.json outright. Tip:
      run snapshot once per --filter group and --merge them into the same
      file -- this is what bench-campaign.sh does for its criterion:
      items, so one campaign's snapshot JSON accumulates every group
      instead of the last group clobbering the earlier ones.

  compare OLD.json NEW.json [--threshold 5.0]
      Print a markdown comparison table. This is a reporting tool, not a
      gate: it always exits 0, regardless of how many regressions it finds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CRITERION_DIR = REPO_ROOT / "target" / "criterion"


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def collect_snapshot(criterion_dir: Path, filter_substr: Optional[str] = None) -> dict:
    """Walk ``criterion_dir`` and return {full_id: metrics} for every bench found."""
    results = {}
    if not criterion_dir.is_dir():
        return results

    for new_dir in sorted(criterion_dir.rglob("new")):
        if not new_dir.is_dir():
            continue
        estimates_path = new_dir / "estimates.json"
        estimates = _read_json(estimates_path)
        if estimates is None:
            continue

        full_id = None
        throughput_elems = None
        bench = _read_json(new_dir / "benchmark.json")
        if bench is not None:
            full_id = bench.get("full_id")
            throughput = bench.get("throughput") or {}
            if isinstance(throughput, dict):
                throughput_elems = throughput.get("Elements")

        if full_id is None:
            # Fallback for a run with no benchmark.json: derive an id from
            # the directory path relative to target/criterion (the parent
            # of `new`). Not guaranteed to match Criterion's own full_id
            # convention (directory names collapse "/" to "_"), but it's
            # stable and unique per bench directory.
            rel = new_dir.parent.relative_to(criterion_dir)
            full_id = "/".join(rel.parts)

        if filter_substr and filter_substr not in full_id:
            continue

        median_ns = (estimates.get("median") or {}).get("point_estimate")
        mean_ns = (estimates.get("mean") or {}).get("point_estimate")
        stddev_ns = (estimates.get("std_dev") or {}).get("point_estimate")

        melem_per_s = None
        if throughput_elems is not None and median_ns:
            melem_per_s = throughput_elems / median_ns * 1000.0

        results[full_id] = {
            "median_ns": median_ns,
            "mean_ns": mean_ns,
            "stddev_ns": stddev_ns,
            "throughput_elems": throughput_elems,
            "melem_per_s": melem_per_s,
        }

    return results


def cmd_snapshot(args: argparse.Namespace) -> int:
    new_results = collect_snapshot(CRITERION_DIR, args.filter)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = new_results
    merged_note = ""
    if args.merge and out_path.is_file():
        existing = _read_json(out_path)
        if existing is None:
            print(
                f"criterion-report snapshot: --merge requested but {out_path} "
                "could not be parsed as JSON -- overwriting it",
                file=sys.stderr,
            )
        else:
            existing.update(new_results)
            results = existing
            merged_note = (
                f", merged into {len(existing)} total existing at {out_path}"
            )

    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(
        f"criterion-report snapshot: captured {len(new_results)} bench(es) "
        f"from {CRITERION_DIR} -> {out_path}{merged_note}",
        file=sys.stderr,
    )
    return 0


def _human_time(ns: Optional[float]) -> str:
    if ns is None:
        return "n/a"
    ans = abs(ns)
    if ans < 1e3:
        return f"{ns:.2f} ns"
    if ans < 1e6:
        return f"{ns / 1e3:.2f} µs"
    if ans < 1e9:
        return f"{ns / 1e6:.2f} ms"
    return f"{ns / 1e9:.2f} s"


def cmd_compare(args: argparse.Namespace) -> int:
    old = json.loads(Path(args.old).read_text())
    new = json.loads(Path(args.new).read_text())
    threshold = args.threshold

    common = sorted(set(old) & set(new))
    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))

    lines = ["| bench | old median | new median | Δ% | note |",
             "|---|---|---|---|---|"]
    for bench in common:
        o = (old[bench] or {}).get("median_ns")
        n = (new[bench] or {}).get("median_ns")
        if o is None or n is None or o == 0:
            delta_str = "n/a"
            note = ""
        else:
            delta = (n - o) / o * 100.0
            delta_str = f"{delta:+.2f}%"
            if delta > threshold:
                note = "**REGRESSION**"
            elif delta < -threshold:
                note = "**IMPROVED**"
            else:
                note = ""
        lines.append(
            f"| {bench} | {_human_time(o)} | {_human_time(n)} | {delta_str} | {note} |"
        )

    print("\n".join(lines))

    if only_old or only_new:
        print()
        print("### Only present in one file")
        print()
        if only_old:
            print(f"- only in {args.old}: {', '.join(only_old)}")
        if only_new:
            print(f"- only in {args.new}: {', '.join(only_new)}")

    # Reporting tool, not a gate: always exit 0.
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="criterion-report.py",
        description=(
            "Summarize and diff Criterion 0.5 output under target/criterion/ "
            "for paulistrings-rs."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_snap = sub.add_parser(
        "snapshot",
        help="Snapshot target/criterion into a JSON file.",
        description=(
            "Walk target/criterion and emit "
            "{full_id: {median_ns, mean_ns, stddev_ns, throughput_elems, "
            "melem_per_s}} sorted by key. melem_per_s = "
            "throughput_elems / median_ns * 1000."
        ),
    )
    p_snap.add_argument("out", help="Path to write the snapshot JSON to.")
    p_snap.add_argument(
        "--filter",
        default=None,
        help=(
            "Only include benches whose full_id contains this SUBSTRING "
            "(plain substring match, not a regex or glob). Tip: run "
            "snapshot once per group and --merge them into one file -- "
            "this is what bench-campaign.sh's criterion: items do."
        ),
    )
    p_snap.add_argument(
        "--merge",
        action="store_true",
        help=(
            "If OUT.json already exists, load it and update it with the "
            "newly collected entries (new entries win on key collisions) "
            "instead of overwriting it outright."
        ),
    )
    p_snap.set_defaults(func=cmd_snapshot)

    p_cmp = sub.add_parser(
        "compare",
        help="Compare two snapshot JSON files.",
        description=(
            "Print a markdown table comparing two snapshot JSON files "
            "(old vs new median). Always exits 0 -- this is a reporting "
            "tool, not a CI gate."
        ),
    )
    p_cmp.add_argument("old", help="Path to the baseline snapshot JSON.")
    p_cmp.add_argument("new", help="Path to the candidate snapshot JSON.")
    p_cmp.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help=(
            "Percent |delta| above which a row is flagged REGRESSION/IMPROVED "
            "(default 5.0, the measured single-thread noise floor on the "
            "reference host; see benchmarks/PROFILING.md for the full noise "
            "figures and the A/B protocol for sub-noise effects)."
        ),
    )
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
