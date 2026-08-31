#!/usr/bin/env python3
"""Paired per-run report over two phase_breakdown JSONL sidecars (A vs B).

This is the reporting half of the interleaved A/B protocol
(``scripts/ab-compare.sh``). On the reference host, single-shot campaign
noise is ~±5-8% at 1 thread and ~±10-26% at 8/32 threads -- untouched code
has moved that much between campaigns -- so a difference of two independent
campaign means cannot resolve the ~5-10% effects the engine work produces.
What can: run the two prebuilt binaries alternated adjacent in time, then
compare *pairs* (run i of A against run i of B) and ask whether every pair
moved the same way.

Input: two files written by ``phase_breakdown --json-out FILE``, one JSON
object per line, one line per ``(layer, threads)`` cell per invocation.
Runs are paired within a cell by their order of appearance in the file, so
each file must come from a single A/B campaign (ab-compare.sh rotates stale
sidecars aside for exactly this reason). Unequal run counts pair up to the
minimum; a cell present in only one file is listed, not compared.

Usage:
    python3 scripts/ab-report.py A.jsonl B.jsonl [--field wall_ns]
        [--all-phases] [--label-a NAME] [--label-b NAME]

Re-invokable at any time on archived sidecars -- it reads nothing but the
two files. Stdlib only; runs no benchmarks and no cargo.

Exit status: 0 always (a report, not a gate), except 2 when an input file is
unreadable or contains no usable JSON lines.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from typing import Optional

# Worker busy-time phases of the coset loop, reported by --all-phases. These
# are summed across every Rayon worker (see benchmarks/PROFILING.md), so they
# do not sum to wall time -- they are where a traffic/working-set change
# shows up first, which is why the results notes quote them alongside wall.
PHASE_FIELDS = ["gather_ns", "sort_ns", "merge_ns"]


class InputError(Exception):
    """An input file was unreadable or carried no usable JSON lines."""


def load_runs(path: str) -> list[dict]:
    """Parse one JSONL sidecar into a list of cell dicts, in file order.

    Malformed lines are reported on stderr and skipped: a probe killed
    mid-write should cost one cell, not the whole report.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw_lines = handle.readlines()
    except OSError as exc:
        raise InputError(f"cannot read '{path}': {exc}") from exc

    runs: list[dict] = []
    for lineno, line in enumerate(raw_lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(
                f"ab-report: {path}:{lineno}: skipping malformed JSON line ({exc.msg})",
                file=sys.stderr,
            )
            continue
        if not isinstance(obj, dict):
            print(
                f"ab-report: {path}:{lineno}: skipping non-object JSON line",
                file=sys.stderr,
            )
            continue
        runs.append(obj)

    if not runs:
        raise InputError(f"'{path}' contains no usable JSON lines")
    return runs


def cell_key(run: dict) -> tuple[str, str]:
    """Cell identity: the probe's (layer, threads) pair, as strings."""
    layer = run.get("layer")
    threads = run.get("threads")
    return (
        str(layer) if layer is not None else "?",
        str(threads) if threads is not None else "?",
    )


def group_by_cell(runs: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """{cell: [run, ...]} preserving file order within each cell."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        cells.setdefault(cell_key(run), []).append(run)
    return cells


def numeric(value) -> Optional[float]:
    """A finite number, or None (missing field, null, string, NaN, ...)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def per_layer(run: dict, field: str, value: float) -> tuple[float, str]:
    """Scale an absolute for display: ns fields become ms per layer.

    The probe's ``*_ns`` fields cover the whole cell (``layers`` layers),
    and it is ms/layer that the research notes quote. Δ% is computed on the
    raw field, never on this, so a hypothetical A/B disagreement in
    ``layers`` cannot leak into the effect size.
    """
    layers = numeric(run.get("layers"))
    if field.endswith("_ns"):
        if layers and layers > 0:
            return value / layers / 1e6, "ms/layer"
        return value / 1e6, "ms"
    return value, ""


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    ans = abs(value)
    if ans >= 1e4:
        # Counts (terms_in, rows_gathered, vmhwm_kb) live up here; keep them
        # exact rather than rounding them into scientific notation.
        return f"{value:.0f}"
    if ans < 1e-3 and ans != 0.0:
        return f"{value:.3e}"
    return f"{value:.3f}"


def consistency(deltas: list[float]) -> str:
    """One line on direction agreement -- the actual acceptance criterion."""
    n = len(deltas)
    if n == 0:
        return "no comparable pairs"
    neg = sum(1 for d in deltas if d < 0)
    pos = sum(1 for d in deltas if d > 0)
    if neg == 0 and pos == 0:
        return f"{n}/{n} pairs exactly equal (no change)"
    if neg == n:
        return f"{n}/{n} pairs negative (consistent: B lower)"
    if pos == n:
        return f"{n}/{n} pairs positive (consistent: B higher)"
    tied = n - neg - pos
    tied_note = f", {tied}/{n} exactly equal" if tied else ""
    return f"{neg}/{n} negative, {pos}/{n} positive{tied_note} — no consistent change"


def pair_deltas(
    runs_a: list[dict], runs_b: list[dict], field: str
) -> tuple[list[Optional[float]], list[tuple[Optional[float], Optional[float]]], str]:
    """Per-pair Δ% = (b-a)/a*100 plus the display absolutes and their unit.

    A pair whose field is missing, non-numeric, or has a==0 yields None --
    kept in place so the pair indices in the table stay honest.
    """
    n_pairs = min(len(runs_a), len(runs_b))
    deltas: list[Optional[float]] = []
    absolutes: list[tuple[Optional[float], Optional[float]]] = []
    unit = ""
    for i in range(n_pairs):
        run_a, run_b = runs_a[i], runs_b[i]
        val_a, val_b = numeric(run_a.get(field)), numeric(run_b.get(field))
        show_a = show_b = None
        if val_a is not None:
            show_a, unit = per_layer(run_a, field, val_a)
        if val_b is not None:
            show_b, unit_b = per_layer(run_b, field, val_b)
            unit = unit or unit_b
        absolutes.append((show_a, show_b))
        if val_a is None or val_b is None or val_a == 0.0:
            deltas.append(None)
        else:
            deltas.append((val_b - val_a) / val_a * 100.0)
    return deltas, absolutes, unit


def summarize(deltas: list[Optional[float]]) -> Optional[tuple[float, float, float, int]]:
    """(median, min, max, count) over the comparable pairs, or None."""
    valid = [d for d in deltas if d is not None]
    if not valid:
        return None
    return statistics.median(valid), min(valid), max(valid), len(valid)


def report_cell(
    cell: tuple[str, str],
    runs_a: list[dict],
    runs_b: list[dict],
    field: str,
    extra_fields: list[str],
    label_a: str,
    label_b: str,
) -> None:
    layer, threads = cell
    print(f"=== layer={layer}  threads={threads} ===")

    n_pairs = min(len(runs_a), len(runs_b))
    if len(runs_a) != len(runs_b):
        print(
            f"  note: {len(runs_a)} run(s) on A, {len(runs_b)} on B — "
            f"pairing the first {n_pairs}"
        )
    if n_pairs == 0:
        print("  no pairs to compare")
        print()
        return

    deltas, absolutes, unit = pair_deltas(runs_a, runs_b, field)
    unit_note = f" [{unit}]" if unit else ""
    print(f"  {field}{unit_note}   A: {label_a}   B: {label_b}")
    print(f"  {'pair':>5}  {'A':>12}  {'B':>12}  {'Δ%':>9}")
    for i, delta in enumerate(deltas):
        show_a, show_b = absolutes[i]
        delta_str = "n/a" if delta is None else f"{delta:+.2f}"
        print(f"  {i + 1:>5}  {fmt(show_a):>12}  {fmt(show_b):>12}  {delta_str:>9}")

    stats = summarize(deltas)
    if stats is None:
        print(f"  field '{field}' missing or unusable in every pair of this cell")
    else:
        median, lo, hi, count = stats
        print(
            f"  median Δ% {median:+.2f}   min {lo:+.2f}   max {hi:+.2f}   "
            f"({count} comparable pair(s))"
        )
        print(f"  {consistency([d for d in deltas if d is not None])}")

    for extra in extra_fields:
        if extra == field:
            continue
        extra_deltas, _, _ = pair_deltas(runs_a, runs_b, extra)
        extra_stats = summarize(extra_deltas)
        if extra_stats is None:
            print(f"  {extra:>12}:  not present in these runs")
            continue
        median, lo, hi, _ = extra_stats
        print(
            f"  {extra:>12}:  median Δ% {median:+.2f}  "
            f"(min {lo:+.2f}, max {hi:+.2f})  "
            f"{consistency([d for d in extra_deltas if d is not None])}"
        )
    print()


HONESTY_RULE = """\
Reading this report
  With a handful of pairs there is no p-value worth quoting, and none is
  computed here. The acceptance criterion is DIRECTION CONSISTENCY: every
  pair in a cell must move the same way, and the median Δ% is then the
  effect size to report. A cell whose pairs disagree in sign is "no
  consistent change" — not a small win, not a trend, not something to
  quote a mean over.
  Pairs are adjacent in time by construction (see scripts/ab-compare.sh)
  precisely because this host's campaign-to-campaign noise is ~±5-8% at 1
  thread and ~±10-26% at 8/32 threads: absolute values from different
  campaigns are not comparable, paired deltas from one campaign are.
  --all-phases busy fields (gather/sort/merge) are summed over all Rayon
  workers, so they do not sum to wall time; use them to explain a wall
  effect, not as one."""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ab-report.py",
        description=(
            "Paired per-run A/B report over two phase_breakdown --json-out "
            "sidecars. Always exits 0 unless an input is unreadable/empty -- "
            "this is a reporting tool, not a gate."
        ),
        epilog="Written for scripts/ab-compare.sh; re-runnable on archived sidecars.",
    )
    parser.add_argument("a", help="JSONL sidecar for side A (the baseline).")
    parser.add_argument("b", help="JSONL sidecar for side B (the candidate).")
    parser.add_argument(
        "--field",
        default="wall_ns",
        help=(
            "Probe field to compare per pair (default: wall_ns). Any numeric "
            "field of the sidecar works, e.g. coset_loop_ns, merge_ns, "
            "vmhwm_kb, terms_out."
        ),
    )
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help=(
            "Also summarize the coset-loop worker busy fields "
            f"({', '.join(PHASE_FIELDS)}) for every cell, one line each, "
            "where present."
        ),
    )
    parser.add_argument("--label-a", default="A", help="Label for side A in the header.")
    parser.add_argument("--label-b", default="B", help="Label for side B in the header.")
    args = parser.parse_args(argv)

    try:
        runs_a = load_runs(args.a)
        runs_b = load_runs(args.b)
    except InputError as exc:
        print(f"ab-report: {exc}", file=sys.stderr)
        return 2

    cells_a = group_by_cell(runs_a)
    cells_b = group_by_cell(runs_b)

    print("Interleaved A/B report")
    print(f"  A: {args.a}  ({len(runs_a)} run row(s), {len(cells_a)} cell(s))  [{args.label_a}]")
    print(f"  B: {args.b}  ({len(runs_b)} run row(s), {len(cells_b)} cell(s))  [{args.label_b}]")
    print(f"  field: {args.field}" + ("  +all-phases" if args.all_phases else ""))
    print()

    extra_fields = PHASE_FIELDS if args.all_phases else []

    # Cells in A's order of first appearance (the probe's own layer/thread
    # order), then any cell only B has.
    ordered = list(cells_a.keys()) + [c for c in cells_b if c not in cells_a]
    for cell in ordered:
        report_cell(
            cell,
            cells_a.get(cell, []),
            cells_b.get(cell, []),
            args.field,
            extra_fields,
            args.label_a,
            args.label_b,
        )

    only_a = [c for c in cells_a if c not in cells_b]
    only_b = [c for c in cells_b if c not in cells_a]
    if only_a or only_b:
        print("Cells present in only one file (not compared)")
        for cell in only_a:
            print(f"  only in A: layer={cell[0]} threads={cell[1]}")
        for cell in only_b:
            print(f"  only in B: layer={cell[0]} threads={cell[1]}")
        print()

    print(HONESTY_RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
