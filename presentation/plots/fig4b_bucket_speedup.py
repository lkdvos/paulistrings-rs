"""Figure F4b: what the bucket size does to 16 threads.

From `bucket_sweep.jsonl` (1 and 16 threads at every target bucket length):
speedup of the 16-thread run over the 1-thread run of the SAME bucket size
(solid), and over the 1-thread run at the DEFAULT bucket size (dotted) -- the
two normalisations coincide because the single-thread cost is flat in the bucket
size (see F5). Bottom panel: the realised bucket count and the parallel
efficiency busy / (coset_loop x threads) from the engine's phase counters, so the
audience can see how much of the collapse at coarse buckets is task starvation.

    python presentation/plots/fig4b_bucket_speedup.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
L2_BYTES = 1 << 20
ROW_BYTES = 48
FANOUT = 2  # a Pauli rotation's gather run is two bucket-lengths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    rows = common.load_jsonl(Path(args.data_dir) / "bucket_sweep.jsonl")
    agg = common.aggregate_median(rows, ("target_bucket_len", "threads"), "wall_ns")
    targets = sorted({r["target_bucket_len"] for r in rows})
    one = {t: agg[(t, 1)][0] for t in targets if (t, 1) in agg}
    many = {t: agg[(t, args.threads)] for t in targets if (t, args.threads) in agg}
    default_one = one.get(1024, min(one.values()))
    xs = [t for t in targets if t in one and t in many]
    own = [one[t] / many[t][0] for t in xs]
    own_lo = [one[t] / many[t][2] for t in xs]
    own_hi = [one[t] / many[t][1] for t in xs]
    vs_default = [default_one / many[t][0] for t in xs]

    eff = {}
    buckets = {}
    for r in rows:
        if r["threads"] == args.threads and r.get("busy_total_ns") and r.get("coset_loop_ns"):
            eff.setdefault(r["target_bucket_len"], []).append(r["busy_total_ns"] / (r["coset_loop_ns"] * args.threads))
        if r["threads"] == args.threads and r.get("buckets"):
            buckets[r["target_bucket_len"]] = r["buckets"]

    common.apply_rcparams()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.12})
    c = common.VARIANT_COLORS["bucketed"]
    ax.fill_between(xs, own_lo, own_hi, color=c, alpha=0.15, linewidth=0)
    ax.plot(xs, own, "-o", color=c, label=f"{args.threads} threads vs 1 thread, same bucket size")
    ax.plot(xs, vs_default, ":", color=common.VARIANT_COLORS["bucketed-coarse"], label="vs 1 thread at the default (1024)")
    ax.axhline(args.threads, color=common._MUTED_TEXT, linestyle="--", linewidth=1, label=f"ideal ({args.threads}×)")
    x_l2 = L2_BYTES / (ROW_BYTES * FANOUT)
    for a in (ax, ax2):
        a.axvspan(x_l2 / 1.3, x_l2 * 1.3, color=common._GRID_COLOR, alpha=0.7, linewidth=0)
    ax.text(x_l2, args.threads * 0.97, "gather run\ncrosses L2", ha="center", va="top", fontsize=8, color=common._MUTED_TEXT)
    ax.set_xscale("log", base=2)
    ax.set_ylabel("speedup")
    ax.set_ylim(0, args.threads * 1.1)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    common._style_axes(ax)

    if eff:
        ex = [t for t in xs if t in eff]
        ax2.plot(ex, [float(np.median(eff[t])) for t in ex], "-o", color=c)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("parallel\nefficiency")
    ax2.set_xlabel("target bucket length (terms, log2)")
    common._style_axes(ax2)
    for t in xs:
        if t in buckets:
            ax2.annotate(f"B={buckets[t]}", (t, 0.08), ha="center", fontsize=7.5, color=common._MUTED_TEXT)
    fig.align_ylabels()
    common.save(fig, "fig4b_bucket_speedup")
    print("bucket size -> speedup:", {t: round(v, 1) for t, v in zip(xs, own)})


if __name__ == "__main__":
    main()
