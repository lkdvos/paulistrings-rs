#!/usr/bin/env python3
"""Tabulate `lanczos_probe` cells as a markdown table: one row per cell from its rank-0 (or only) JSON sidecar.

Usage: scripts/lanczos-report.py <sidecar path without .rank0> ...
Group-wide columns (`shipped`, `imbal`, `sent_GB`, `rss_GB`) are what the probe all-reduced; times are the slowest partition's.
"""

import json
import os
import sys


def read(path):
    for candidate in (path + ".rank0", path):
        if os.path.exists(candidate):
            with open(candidate) as f:
                return [json.loads(line) for line in f if line.strip()]
    return None


def main(paths):
    print("| cell | partitions | steps | terms | total s | last step s | last xchg s | xchg % | shipped | imbal | sent GB | peak RSS GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        rows = read(path)
        if not rows:
            print(f"| {os.path.basename(path)} | missing | | | | | | | | | | |")
            continue
        last = rows[-1]
        total = sum(r["wall_ns"] for r in rows) / 1e9
        product = sum(r["product_ns_max"] for r in rows)
        exchange = sum(r["export_exchange_ns_max"] for r in rows)
        sent = sum(r["bytes_sent_group"] for r in rows) / 1e9
        peak = max(r["rss_kb_group"] for r in rows) / 1e6
        share = 100.0 * exchange / product if product else 0.0
        name = last["label"] or os.path.basename(path)
        print(
            f"| {name} | {last['partitions']} | {last['step']} | {last['terms']:.3e} | {total:.2f} "
            f"| {last['wall_ns'] / 1e9:.2f} | {last['export_exchange_ns_max'] / 1e9:.2f} | {share:.1f} "
            f"| {last['shipped']:.3f} | {last['imbalance']:.3f} | {sent:.2f} | {peak:.1f} |"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
