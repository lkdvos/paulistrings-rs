#!/usr/bin/env python3
"""Per-phase strong-scaling table from phase_breakdown --json-out sidecars.

    python3 scripts/scaling-report.py SIDECAR.jsonl [...]

One table per (layer, n, truncation, partitions), one row per thread count T, relative to the smallest T present.
Columns: wall per layer and its speedup and efficiency; the serial share (wall outside the coset loop, export and exchange); the coset loop's
parallel efficiency, sum(busy) / (coset_loop x T), whose shortfall is idle workers (imbalance, fork-join); and each busy
phase's ns per input term summed over workers, whose growth with T is contention inside that phase.
A phase that scales perfectly keeps its busy ns/term flat; the first column to grow is the phase that stops scaling first.
"""
import json
import sys
from collections import defaultdict

BUSY = ("gather", "sort", "merge", "swap", "size", "clear")


def load(paths):
    cells = defaultdict(dict)
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["layer"], row["n"], row["truncation"], row.get("partitions", 1))
                cells[key][row["threads"]] = row  # a re-run overwrites, as the other reporters do
    return cells


def report(key, rows):
    layer, n, truncation, partitions = key
    threads = sorted(rows)
    base = rows[threads[0]]
    per_term = lambda row, field: row[field] / max(row["terms_in"], 1)
    print(f"\n{layer}  n={n}  {truncation}  P={partitions}  layers={base['layers']}  terms_in/layer={base['terms_in'] / max(base['layers'], 1):.3g}")
    header = f"{'T':>4} {'ms/layer':>10} {'speedup':>8} {'eff':>6} {'serial%':>8} {'loop_eff':>8}"
    header += "".join(f" {name + ' ns/t':>13}" for name in BUSY)
    print(header)
    for t in threads:
        row = rows[t]
        layers = max(row["layers"], 1)
        wall = row["wall_ns"]
        speedup = base["wall_ns"] / wall
        efficiency = speedup * threads[0] / t
        exchange = row.get("export_ns", 0) + row.get("exchange_ns", 0)
        serial = (wall - row["coset_loop_ns"] - exchange) / wall * 100
        busy = sum(row[f"{name}_ns"] for name in BUSY)
        loop_efficiency = busy / (row["coset_loop_ns"] * t) if row["coset_loop_ns"] else 0.0
        line = f"{t:>4} {wall / layers / 1e6:>10.3f} {speedup:>8.2f} {efficiency:>6.2f} {serial:>7.1f}% {loop_efficiency:>8.2f}"
        line += "".join(f" {per_term(row, name + '_ns'):>13.2f}" for name in BUSY)
        if partitions > 1:
            line += f"   export+exchange {exchange / wall * 100:.1f}% of wall, {row['remote_layers']}/{row['layers']} layers remote ({row['partition_rows']} rows)"
        print(line)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for key, rows in sorted(load(sys.argv[1:]).items(), key=lambda item: tuple(map(str, item[0]))):
        report(key, rows)


if __name__ == "__main__":
    main()
