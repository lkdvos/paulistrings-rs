"""Figure F6: peak memory per term, by variant.

Reads `presentation/data/memory.jsonl` and computes, per row,

    bytes_per_peak_term = (vmhwm_kb - baseline_kb) * 1024 / peak_terms

with `baseline_kb = min(vmrss_kb over the whole file)` (0 if the file has no
`vmrss_kb` values at all) -- a fixed per-file baseline, not a per-row one, so
every variant's bar is measuring against the same "process just started"
floor. Bars are medians across reps, horizontal, one per variant, with a
reference dotted line at 48 bytes (the SoA per-term row size: x + z + coeff
columns, per CLAUDE.md).

Prints the baseline value and the definition above to stdout, since the
"bytes per peak term" quantity depends on that choice and isn't otherwise
recorded in the figure.

Run (from the repo root)::

    python presentation/plots/fig6_memory.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import common

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SOA_ROW_BYTES = 48


def _bytes_per_peak_term(records: list[dict]) -> dict[str, tuple[float, float, float]]:
    vmrss_values = [r["vmrss_kb"] for r in records if r.get("vmrss_kb") is not None]
    baseline_kb = min(vmrss_values) if vmrss_values else 0.0

    derived = []
    for r in records:
        if r.get("vmhwm_kb") is None or not r.get("peak_terms"):
            continue
        bytes_per_term = (r["vmhwm_kb"] - baseline_kb) * 1024.0 / r["peak_terms"]
        derived.append({**r, "_bytes_per_peak_term": bytes_per_term})

    by_variant: dict[str, list[float]] = {}
    for r in derived:
        by_variant.setdefault(common.variant_key(r), []).append(r["_bytes_per_peak_term"])

    agg = common.aggregate_median(
        [{"variant": v, "value": x} for v, xs in by_variant.items() for x in xs],
        group_keys=("variant",),
        value_key="value",
    )
    return {variant: stats for (variant,), stats in agg.items()}, baseline_kb


def _order(values: dict[str, tuple]) -> list[str]:
    order = ["naive", "threadmaps", "mergesort", "bucketed", "bucketed-coarse"]
    return [v for v in order if v in values] + [v for v in values if v not in order]


def plot_memory(values: dict[str, tuple[float, float, float]]) -> plt.Figure:
    common.apply_rcparams()
    variants = _order(values)
    fig, ax = plt.subplots(figsize=(common.FIGSIZE[0], 0.9 + 0.55 * len(variants)))

    y_positions = list(range(len(variants)))[::-1]
    for y, variant in zip(y_positions, variants):
        median, lo, hi = values[variant]
        color = common.VARIANT_COLORS[variant]
        ax.barh(y, median, height=0.6, color=color, zorder=2)
        ax.errorbar(median, y, xerr=[[median - lo], [hi - median]], color=common._MUTED_TEXT,
                    linewidth=1.0, capsize=3, zorder=3)
        ax.text(max(median, hi) * 1.03, y, f"{median:,.0f} B", ha="left", va="center",
                fontsize=8.5, color=common._MUTED_TEXT)

    ax.axvline(SOA_ROW_BYTES, color=common._MUTED_TEXT, linewidth=1.2, linestyle=":", zorder=1)
    ax.annotate(
        f"{SOA_ROW_BYTES} B (SoA row size)",
        xy=(SOA_ROW_BYTES, 1.0), xycoords=("data", "axes fraction"),
        xytext=(4, -10), textcoords="offset points",
        fontsize=8, color=common._MUTED_TEXT, va="top",
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([common.VARIANT_LABELS.get(v, v) for v in variants])
    ax.set_ylim(-0.7, len(variants) - 0.3)
    ax.set_xlabel("peak memory per term (bytes, above baseline VmRSS)")
    common._style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    records = common.load_jsonl(Path(args.data_dir) / "memory.jsonl")
    values, baseline_kb = _bytes_per_peak_term(records)
    if not values:
        raise SystemExit("no rows with both vmhwm_kb and peak_terms in memory.jsonl")

    print(
        "bytes_per_peak_term = (vmhwm_kb - baseline_kb) * 1024 / peak_terms, "
        f"baseline_kb = min(vmrss_kb) over the file = {baseline_kb:.0f} KiB"
    )
    for variant in _order(values):
        median, lo, hi = values[variant]
        print(f"{variant:<18} median {median:8.1f} B  (range {lo:.1f}-{hi:.1f})")

    fig = plot_memory(values)
    common.save(fig, "fig6_memory")


if __name__ == "__main__":
    main()
