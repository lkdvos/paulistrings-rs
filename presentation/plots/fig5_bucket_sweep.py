"""Figure F5: the bucket-size lever, ns/term-layer and cache miss rates vs. bucket size.

Reads `presentation/data/bucket_sweep.jsonl` (timing, 1 and 16 threads) and
`bucket_sweep_perf.jsonl` (the same grid plus perf counters) and draws three
stacked panels sharing a log-x axis of `target_bucket_len` (labeled "target
bucket length (terms)" -- the realised terms/bucket track this directly once
`n / target_bucket_len` clears `min_buckets`, so the task brief's more
elaborate formula collapses to just plotting the sweep parameter):

1. ns per term-layer (`wall_ns / terms_in`, falling back to
   `wall_ns / (n * layers)` when `terms_in` is absent), one line for 1 thread
   and one for 16 threads.
2. L2 miss rate from the perf-counter file.
3. LLC miss rate from the perf-counter file.

Two vertical bands mark where the gather run's working set
(`2 * target_bucket_len * 48 bytes` -- the read+write halves of the SoA
columns) crosses the L2 cache size (1 MiB) and the LLC/socket size (24.75 MiB,
`research/notes/2026-08-30-bandwidth-ceiling-ccqlin038.md`).

Perf-counter key names are looked up case-insensitively by substring (per the
task brief -- exact naming from the real collector may differ slightly):
`l2` + `miss` + `rate` for the L2 miss rate, `llc` + `miss` + `rate` for LLC.
Falls back to computing `misses / refs` from the raw counter fields if no
precomputed rate field exists.

Run (from the repo root)::

    python presentation/plots/fig5_bucket_sweep.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

BYTES_PER_TERM_ROW = 48  # SoA column width per CLAUDE.md
GATHER_FACTOR = 2  # read + write halves of one gather pass
L2_BYTES = 1 * 1024**2
LLC_BYTES = 24.75 * 1024**2


def _find_key_ci(row: dict, *substrings: str) -> str | None:
    """First key in `row` containing every substring, case-insensitively."""
    for key in row:
        low = key.lower()
        if all(s in low for s in substrings):
            return key
    return None


def _ns_per_term_layer(rows: list[dict]) -> list[float]:
    out = []
    for r in rows:
        terms_in = r.get("terms_in")
        denom = terms_in if terms_in else (r["n"] * r["layers"])
        out.append(r["wall_ns"] / denom)
    return out


def _timing_curves(records: list[dict]) -> dict[int, dict[int, tuple[float, float, float]]]:
    """threads -> {target_bucket_len: (median, lo, hi)} of ns/term-layer."""
    by_threads: dict[int, list[dict]] = {}
    for r in records:
        by_threads.setdefault(r["threads"], []).append(r)

    out: dict[int, dict[int, tuple[float, float, float]]] = {}
    for threads, rows in by_threads.items():
        by_bucket: dict[int, list[float]] = {}
        for r, ns in zip(rows, _ns_per_term_layer(rows)):
            by_bucket.setdefault(r["target_bucket_len"], []).append(ns)
        out[threads] = {
            b: (float(np.median(v)), float(np.min(v)), float(np.max(v))) for b, v in by_bucket.items()
        }
    return out


def _miss_rate_curves(records: list[dict], prefix: str) -> dict[int, dict[int, tuple[float, float, float]]]:
    """`prefix` is "l2" or "llc". Prefers a precomputed `<prefix>...miss...rate`
    field; falls back to `<prefix>...miss[es] / <prefix>...ref|load`."""
    rate_key = _find_key_ci(records[0], prefix, "miss", "rate") if records else None
    by_threads: dict[int, list[dict]] = {}
    for r in records:
        by_threads.setdefault(r["threads"], []).append(r)

    out: dict[int, dict[int, tuple[float, float, float]]] = {}
    for threads, rows in by_threads.items():
        by_bucket: dict[int, list[float]] = {}
        for r in rows:
            if rate_key is not None and r.get(rate_key) is not None:
                rate = r[rate_key]
            else:
                misses_key = _find_key_ci(r, prefix, "miss")
                refs_key = _find_key_ci(r, prefix, "ref") or _find_key_ci(r, prefix, "load")
                if misses_key is None or refs_key is None or not r.get(refs_key):
                    continue
                rate = r[misses_key] / r[refs_key]
            by_bucket.setdefault(r["target_bucket_len"], []).append(rate)
        out[threads] = {
            b: (float(np.median(v)), float(np.min(v)), float(np.max(v))) for b, v in by_bucket.items()
        }
    return out


def plot_bucket_sweep(timing_records: list[dict], perf_records: list[dict]) -> plt.Figure:
    common.apply_rcparams()
    fig, (ax_ns, ax_l2, ax_llc) = plt.subplots(
        3, 1, figsize=(6.4, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0, 1.0], "hspace": 0.12},
    )

    thread_colors = {1: common.VARIANT_COLORS["bucketed-coarse"], 16: common.VARIANT_COLORS["bucketed"]}
    thread_styles = {1: "-", 16: "-"}

    timing = _timing_curves(timing_records)
    for threads in sorted(timing):
        curve = timing[threads]
        xs = sorted(curve)
        medians = [curve[x][0] for x in xs]
        los = [curve[x][1] for x in xs]
        his = [curve[x][2] for x in xs]
        color = thread_colors.get(threads, common._MUTED_TEXT)
        ax_ns.plot(xs, medians, marker="o", markersize=4, linewidth=1.6, color=color,
                    label=f"{threads} thread{'s' if threads != 1 else ''}", zorder=3)
        ax_ns.fill_between(xs, los, his, color=color, alpha=0.15, zorder=2)

    l2_curves = _miss_rate_curves(perf_records, "l2")
    for threads in sorted(l2_curves):
        curve = l2_curves[threads]
        xs = sorted(curve)
        color = thread_colors.get(threads, common._MUTED_TEXT)
        ax_l2.plot(xs, [100 * curve[x][0] for x in xs], marker="o", markersize=4,
                    linewidth=1.4, color=color, zorder=3)

    llc_curves = _miss_rate_curves(perf_records, "llc")
    for threads in sorted(llc_curves):
        curve = llc_curves[threads]
        xs = sorted(curve)
        color = thread_colors.get(threads, common._MUTED_TEXT)
        ax_llc.plot(xs, [100 * curve[x][0] for x in xs], marker="o", markersize=4,
                     linewidth=1.4, color=color, zorder=3)

    l2_x = L2_BYTES / (GATHER_FACTOR * BYTES_PER_TERM_ROW)
    llc_x = LLC_BYTES / (GATHER_FACTOR * BYTES_PER_TERM_ROW)
    for ax in (ax_ns, ax_l2, ax_llc):
        ax.axvspan(l2_x / 1.15, l2_x * 1.15, color=common._MUTED_TEXT, alpha=0.22, zorder=1)
        ax.axvspan(llc_x / 1.15, llc_x * 1.15, color=common._MUTED_TEXT, alpha=0.10, zorder=1)
        ax.set_xscale("log", base=2)
        common._style_axes(ax)

    ax_ns.annotate("gather crosses L2", xy=(l2_x, ax_ns.get_ylim()[1]), xytext=(4, -10),
                    textcoords="offset points", fontsize=7.5, color=common._MUTED_TEXT, ha="left")
    ax_ns.annotate("gather crosses LLC", xy=(llc_x, ax_ns.get_ylim()[1]), xytext=(4, -10),
                    textcoords="offset points", fontsize=7.5, color=common._MUTED_TEXT, ha="left")

    ax_ns.set_ylabel("ns / term-layer")
    ax_ns.legend(frameon=False, fontsize=8, loc="upper left")
    ax_l2.set_ylabel("L2 miss\nrate (%)")
    ax_llc.set_ylabel("LLC miss\nrate (%)")
    ax_llc.set_xlabel("target bucket length (terms, log2)")

    fig.align_ylabels([ax_ns, ax_l2, ax_llc])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    timing_records = common.load_jsonl(data_dir / "bucket_sweep.jsonl")
    perf_records = common.load_jsonl(data_dir / "bucket_sweep_perf.jsonl")

    fig = plot_bucket_sweep(timing_records, perf_records)
    common.save(fig, "fig5_bucket_sweep")


if __name__ == "__main__":
    main()
