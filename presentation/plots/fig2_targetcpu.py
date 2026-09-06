"""Figure F2: paired target-cpu=native vs. default delta, one dot per rep.

Reads `presentation/data/targetcpu_default.jsonl` and
`targetcpu_native.jsonl` -- the same grid of (variant, threads, rep) run
twice, once per `RUSTFLAGS` build -- and, per variant, pairs each default row
with the native row sharing the same `rep` (run order). Draws a dot plot of
the per-pair percent change

    delta% = 100 * (native_wall_ns - default_wall_ns) / default_wall_ns

(negative = native faster) against a zero reference line, one horizontal
strip per variant, jittered vertically so overlapping reps stay visible.
Each strip is annotated with the median delta% and "k/N pairs same sign" --
the number of pairs whose delta% shares the sign of the strip's median,
which is the acceptance signal CLAUDE.md's ab-compare convention cares about
(direction consistency, not the average magnitude).

Run (from the repo root, once both targetcpu_*.jsonl files exist)::

    python presentation/plots/fig2_targetcpu.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _pair_by_rep(default_records: list[dict], native_records: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """variant -> [(default_wall_ns, native_wall_ns), ...] paired on (variant, threads, rep)."""
    native_index: dict[tuple[str, int, int], float] = {}
    for r in native_records:
        key = (common.variant_key(r), r["threads"], r["rep"])
        native_index[key] = r["wall_ns"]

    pairs: dict[str, list[tuple[float, float]]] = {}
    for r in default_records:
        variant = common.variant_key(r)
        key = (variant, r["threads"], r["rep"])
        if key not in native_index:
            continue
        pairs.setdefault(variant, []).append((r["wall_ns"], native_index[key]))
    return pairs


def _order(pairs: dict[str, list]) -> list[str]:
    order = ["naive", "threadmaps", "mergesort", "bucketed", "bucketed-coarse"]
    return [v for v in order if v in pairs] + [v for v in pairs if v not in order]


def plot_targetcpu(pairs: dict[str, list[tuple[float, float]]]) -> plt.Figure:
    common.apply_rcparams()
    variants = _order(pairs)
    fig, ax = plt.subplots(figsize=(common.FIGSIZE[0], 0.9 + 0.6 * len(variants)))

    rng = np.random.default_rng(0)
    for row, variant in enumerate(variants):
        deltas = np.array(
            [100.0 * (native - default) / default for default, native in pairs[variant]]
        )
        median = float(np.median(deltas))
        sign = 1.0 if median >= 0 else -1.0
        same_sign = int(np.sum(np.sign(deltas) == sign)) if median != 0 else int(np.sum(deltas == 0))
        n = len(deltas)

        y_jitter = row + rng.uniform(-0.18, 0.18, size=n)
        color = common.VARIANT_COLORS[variant]
        ax.scatter(deltas, y_jitter, s=22, color=color, alpha=0.8, zorder=3, edgecolor="none")
        ax.plot([median, median], [row - 0.32, row + 0.32], color=color, linewidth=2.2, zorder=4)
        ax.annotate(
            f"median {median:+.1f}%   ({same_sign}/{n} pairs same sign)",
            xy=(median, row),
            xytext=(6, 22),
            textcoords="offset points",
            fontsize=8,
            color=common._MUTED_TEXT,
            ha="left",
        )

    ax.axvline(0.0, color=common._MUTED_TEXT, linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([common.VARIANT_LABELS.get(v, v) for v in variants])
    ax.set_ylim(-0.6, len(variants) - 0.4 + 0.35)
    ax.set_xlabel(r"$\Delta$% wall time, native vs. default ($< 0$ = native faster)")
    common._style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    default_records = common.load_jsonl(data_dir / "targetcpu_default.jsonl")
    native_records = common.load_jsonl(data_dir / "targetcpu_native.jsonl")
    pairs = _pair_by_rep(default_records, native_records)
    if not pairs:
        raise SystemExit("no matching (variant, threads, rep) pairs between default/native files")

    fig = plot_targetcpu(pairs)
    common.save(fig, "fig2_targetcpu")

    print("variant            median Δ%   same-sign pairs")
    for variant in _order(pairs):
        deltas = np.array([100.0 * (native - default) / default for default, native in pairs[variant]])
        median = float(np.median(deltas))
        sign = 1.0 if median >= 0 else -1.0
        same_sign = int(np.sum(np.sign(deltas) == sign))
        print(f"{variant:<18} {median:+8.2f}%   {same_sign}/{len(deltas)}")


if __name__ == "__main__":
    main()
