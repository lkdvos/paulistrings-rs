"""Figure F4: fine vs. coarse buckets, speedup normalized to the coarse baseline.

Reads a thread-scaling data file (`thread_scaling.jsonl` for the moderate-`n`
run, `thread_scaling_large.jsonl` for the `eps=2**-15` / ~4e6-term run) and
plots speedup vs. threads for `bucketed` and `bucketed-coarse`, *both*
normalized to the coarse variant's own 1-thread median wall time -- so a fine
bucket run that is already faster than coarse at 1 thread starts above 1.0 on
the y-axis, which is the point: fine buckets keep more of the working set in
L2 even before any threads are added, and the effect either persists or grows
once cosets are farmed out across threads.

A dashed ideal-linear line is drawn from the fine-bucketed 1-thread point (its
own speedup=1 would be a different reference; this panel's "1.0" is the
coarse baseline, so "ideal" here is diagonal growth from wherever fine
buckets start). The bottom, stacked panel is parallel efficiency defined per
`PhaseStats` as `busy_total_ns / (coset_loop_ns * threads)` for the bucketed
rows -- a different quantity from F3's speedup/threads efficiency, so it gets
its own label.

Run (from the repo root)::

    python presentation/plots/fig4_superlinear.py --data thread_scaling.jsonl --out fig4_superlinear
    python presentation/plots/fig4_superlinear.py --data thread_scaling_large.jsonl --out fig4_superlinear_large
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _by_variant_threads(records: list[dict]) -> dict[str, dict[int, list[dict]]]:
    out: dict[str, dict[int, list[dict]]] = {}
    for r in records:
        v = common.variant_key(r)
        if v not in ("bucketed", "bucketed-coarse"):
            continue
        out.setdefault(v, {}).setdefault(r["threads"], []).append(r)
    return out


def _median(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(vals)) if vals else float("nan")


def plot_superlinear(records: list[dict]) -> plt.Figure:
    common.apply_rcparams()
    fig, (ax_top, ax_eff) = plt.subplots(
        2, 1, figsize=(common.FIGSIZE[0], 5.2), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.4], "hspace": 0.08},
    )

    by_variant = _by_variant_threads(records)
    if "bucketed-coarse" not in by_variant or 1 not in by_variant["bucketed-coarse"]:
        raise SystemExit("no bucketed-coarse, 1-thread rows to normalize against")
    coarse_base = _median(by_variant["bucketed-coarse"][1], "wall_ns")

    all_threads = sorted({t for curve in by_variant.values() for t in curve})
    ax_top.plot(all_threads, all_threads, color=common._MUTED_TEXT, linewidth=1.2,
                linestyle="--", label="ideal (linear in threads)", zorder=1)

    for variant in ("bucketed", "bucketed-coarse"):
        if variant not in by_variant:
            continue
        threads_map = by_variant[variant]
        threads = sorted(threads_map)
        speedups, los, his, efficiencies = [], [], [], []
        for t in threads:
            rows = threads_map[t]
            median, lo, hi = common.aggregate_median(rows, group_keys=(), value_key="wall_ns")[()]
            speedups.append(coarse_base / median)
            los.append(coarse_base / hi)
            his.append(coarse_base / lo)

            busy_total = _median(rows, "busy_total_ns")
            coset_loop = _median(rows, "coset_loop_ns")
            efficiencies.append(busy_total / (coset_loop * t) if coset_loop else float("nan"))

        color = common.VARIANT_COLORS[variant]
        label = common.VARIANT_LABELS[variant]
        ax_top.plot(threads, speedups, marker="o", markersize=4.5, linewidth=1.6,
                     color=color, label=label, zorder=3)
        ax_top.fill_between(threads, los, his, color=color, alpha=0.12, zorder=2)
        ax_eff.plot(threads, efficiencies, marker="o", markersize=4.5, linewidth=1.6,
                     color=color, zorder=3)

    ax_top.axhline(1.0, color=common._MUTED_TEXT, linewidth=0.8, linestyle=":", zorder=1)
    ax_top.set_xscale("log", base=2)
    ax_top.set_yscale("log", base=2)
    ax_top.set_xticks(all_threads)
    ax_top.set_xticklabels([str(t) for t in all_threads])
    ax_top.set_ylabel("speedup vs.\ncoarse, 1 thread (log2)")
    common._style_axes(ax_top)
    ax_top.legend(frameon=False, fontsize=8, loc="upper left")

    ax_eff.axhline(1.0, color=common._MUTED_TEXT, linewidth=1.0, linestyle="--", zorder=1)
    ax_eff.set_xlabel("threads (log2)")
    ax_eff.set_ylabel("parallel efficiency\n(busy / (coset x threads))")
    ax_eff.set_ylim(0, 1.15)
    common._style_axes(ax_eff)

    fig.align_ylabels([ax_top, ax_eff])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="thread_scaling.jsonl")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out", default="fig4_superlinear")
    args = parser.parse_args()

    records = common.load_jsonl(Path(args.data_dir) / args.data)
    fig = plot_superlinear(records)
    common.save(fig, args.out)


if __name__ == "__main__":
    main()
