"""Figure F3: thread scaling, speedup vs. threads with a stacked efficiency panel.

Reads `presentation/data/thread_scaling.jsonl` and, per variant, plots
speedup vs. threads (each variant's own 1-thread median wall time is the
speedup=1 reference -- a variant is never compared to another variant's
baseline here). Two invocations are the deliverables:

- `--only threadmaps,mergesort` -- just the two old multithreading attempts,
  for the talk's Act 2 slide; written as `fig3_old_attempts.svg`.
- (no `--only`) -- every variant in the file, with the two old attempts
  greyed out so the bucketed curves read as the figure; written as
  `fig3_all.svg`.

A dashed diagonal marks ideal linear speedup on the top panel. The bottom
panel (stacked below, never a right-hand axis) is parallel efficiency
(speedup / threads); ideal is the horizontal line at 1.0.

Run (from the repo root)::

    python presentation/plots/fig3_thread_scaling.py --only threadmaps,mergesort
    python presentation/plots/fig3_thread_scaling.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

OLD_ATTEMPTS = ["threadmaps", "mergesort"]
VARIANT_ORDER = ["threadmaps", "mergesort", "bucketed", "bucketed-coarse"]


def _speedup_by_variant(records: list[dict]) -> dict[str, dict[int, tuple[float, float, float]]]:
    """variant -> {threads: (median_speedup, lo, hi)}, normalized to that variant's own 1-thread median."""
    by_variant: dict[str, list[dict]] = {}
    for r in records:
        by_variant.setdefault(common.variant_key(r), []).append(r)

    out: dict[str, dict[int, tuple[float, float, float]]] = {}
    for variant, recs in by_variant.items():
        by_threads = common.aggregate_median(recs, group_keys=("threads",), value_key="wall_ns")
        if (1,) not in by_threads:
            continue
        base_median, _lo, _hi = by_threads[(1,)]
        curve = {}
        for (threads,), (median, lo, hi) in by_threads.items():
            # speedup = base_wall / wall; min/max wall map (with a flip) to max/min speedup.
            curve[threads] = (base_median / median, base_median / hi, base_median / lo)
        out[variant] = curve
    return out


def plot_thread_scaling(curves: dict[str, dict[int, tuple[float, float, float]]], variants: list[str], grey: set[str]) -> plt.Figure:
    common.apply_rcparams()
    fig, (ax_top, ax_eff) = plt.subplots(
        2, 1, figsize=(common.FIGSIZE[0], 5.2), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.4], "hspace": 0.08},
    )

    all_threads = sorted({t for curve in curves.values() for t in curve})
    if all_threads:
        ax_top.plot(all_threads, all_threads, color=common._MUTED_TEXT, linewidth=1.2,
                    linestyle="--", label="ideal", zorder=1)
    ax_eff.axhline(1.0, color=common._MUTED_TEXT, linewidth=1.2, linestyle="--", zorder=1)

    for variant in variants:
        if variant not in curves:
            continue
        curve = curves[variant]
        threads = sorted(curve)
        medians = np.array([curve[t][0] for t in threads])
        los = np.array([curve[t][1] for t in threads])
        his = np.array([curve[t][2] for t in threads])
        is_grey = variant in grey
        color = common._MUTED_TEXT if is_grey else common.VARIANT_COLORS[variant]
        alpha = 0.55 if is_grey else 1.0
        label = common.VARIANT_LABELS.get(variant, variant)

        ax_top.plot(threads, medians, marker="o", markersize=4.5, linewidth=1.6,
                     color=color, alpha=alpha, label=label, zorder=3)
        ax_top.fill_between(threads, los, his, color=color, alpha=0.12 * alpha, zorder=2)

        efficiency = medians / np.array(threads, dtype=float)
        ax_eff.plot(threads, efficiency, marker="o", markersize=4.5, linewidth=1.6,
                     color=color, alpha=alpha, zorder=3)

    ax_top.set_xscale("log", base=2)
    ax_top.set_yscale("log", base=2)
    ax_top.set_xticks(all_threads)
    ax_top.set_xticklabels([str(t) for t in all_threads])
    ax_top.set_ylabel("speedup vs. 1 thread\n(own baseline, log2)")
    common._style_axes(ax_top)
    ax_top.legend(frameon=False, fontsize=8, loc="upper left")

    ax_eff.set_xlabel("threads (log2)")
    ax_eff.set_ylabel("parallel\nefficiency")
    ax_eff.set_ylim(0, 1.15)
    common._style_axes(ax_eff)

    fig.align_ylabels([ax_top, ax_eff])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="comma-separated variant list")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    records = common.load_jsonl(Path(args.data_dir) / "thread_scaling.jsonl")
    curves = _speedup_by_variant(records)

    if args.only:
        requested = [v.strip() for v in args.only.split(",")]
        variants = [v for v in requested if v in curves]
        grey: set[str] = set()
        default_name = "fig3_old_attempts" if requested == OLD_ATTEMPTS else "fig3_" + "_".join(requested)
    else:
        variants = [v for v in VARIANT_ORDER if v in curves]
        grey = {v for v in OLD_ATTEMPTS if v in curves}
        default_name = "fig3_all"

    if not variants:
        raise SystemExit(f"no requested variants found in {args.data_dir}/thread_scaling.jsonl")

    fig = plot_thread_scaling(curves, variants, grey)
    common.save(fig, args.out_name or default_name)


if __name__ == "__main__":
    main()
