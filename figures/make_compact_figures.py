"""Compact single-panel figures: thread scaling, hash communication, accuracy.

Same SYNTHETIC-placeholder rule as `make_recurring_figure.py`: nothing here is
real campaign data yet, and every figure's title says so.
"""

from __future__ import annotations

from typing import Sequence

_ACCENT = "#2a78d6"
_IDEAL = "#898781"


def _style_axes(ax) -> None:
    ax.grid(True, color="#e1e0d9", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#898781")
    ax.tick_params(colors="#898781")


def make_thread_scaling_figure(rows: Sequence[dict]):
    """One curve: thread count (x, log2-spaced) vs. speedup (y), from `thread_scaling()` rows.

    `speedup` is already relative to the filtered set's 1-thread row
    (`normalize.thread_scaling`'s contract); this function plots that column
    verbatim rather than recomputing anything against total/absolute time, so
    a y=x line at threads=1 always starts at speedup=1 by construction, never
    an absolute wall-clock baseline.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))

    points = sorted(
        (r["threads"], r["speedup"]) for r in rows if r.get("speedup") is not None
    )
    if points:
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, color=_ACCENT, label="measured")
        ideal_xs = xs
        ideal_ys = [x / xs[0] for x in xs]
        ax.plot(ideal_xs, ideal_ys, linestyle="--", linewidth=1.0, color=_IDEAL, label="ideal (y=x)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup vs. 1 thread")
    ax.set_title("SYNTHETIC — placeholder: thread scaling")
    _style_axes(ax)
    if points:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def make_hash_communication_figure(rows: Sequence[dict]):
    """Export-volume comparison between random and cut-based partition row selection.

    `rows` is `normalize.hash_communication()`'s output: one row per completed,
    `partition_row_policy`-tagged run, carrying `total_rows_exported` (this
    function's y-axis) grouped by `partition_row_policy` (the bar color/label)
    across `config_id` (the x-axis categories -- one otherwise-identical cell
    run under both policies per category).

    `rows` empty still raises `NotImplementedError` rather than plotting
    nothing: `normalize.hash_communication` itself already raises when there is
    no `partition_row_policy`-tagged data at all, so an empty list reaching
    here means every tagged run's gates were empty too (e.g. a zero-layer
    circuit) -- still "no data to plot", never fabricated.
    """
    if not rows:
        raise NotImplementedError(
            "make_hash_communication_figure: no rows to plot -- every "
            "'partition_row_policy'-tagged run had zero gate records. Run at "
            "least one 'random' and one 'cut' cell with a non-trivial circuit "
            "(jobs/run_cell.py's partition_row_policy field) before this figure "
            "has anything to plot."
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))

    config_ids = sorted({r["config_id"] or r["run_id"] for r in rows})
    policies = sorted({r["partition_row_policy"] for r in rows})
    colors = {"random": _IDEAL, "cut": _ACCENT}
    width = 0.8 / max(len(policies), 1)
    x = range(len(config_ids))

    for i, policy in enumerate(policies):
        by_config = {r["config_id"] or r["run_id"]: r for r in rows if r["partition_row_policy"] == policy}
        ys = [by_config[c]["total_rows_exported"] if c in by_config else 0 for c in config_ids]
        offsets = [xi + (i - (len(policies) - 1) / 2) * width for xi in x]
        ax.bar(offsets, ys, width=width, label=policy, color=colors.get(policy, _ACCENT))

    ax.set_xticks(list(x))
    ax.set_xticklabels(config_ids, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("total rows exported")
    ax.set_title("Hash communication: random vs. cut partition rows")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def make_accuracy_figure(rows: Sequence[dict]):
    """Reference vs. observed value scatter, one point per `accuracy()` row.

    Each point is labeled by `run_id` (the only matched-parameter identifier
    `normalize.accuracy()` carries per row); a y=x line marks perfect agreement.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))

    points = [
        (r["run_id"], r["reference_value"], r["observed_value"])
        for r in rows
        if r.get("reference_value") is not None and r.get("observed_value") is not None
    ]
    if points:
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        ax.scatter(xs, ys, color=_ACCENT, zorder=3)
        for run_id, x, y in points:
            ax.annotate(run_id, (x, y), fontsize=7, color="#898781", xytext=(4, 4), textcoords="offset points")
        lo = min(xs + ys)
        hi = max(xs + ys)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color=_IDEAL, label="reference = observed")
        ax.legend(frameon=False)

    ax.set_xlabel("reference value")
    ax.set_ylabel("observed value")
    ax.set_title("SYNTHETIC — placeholder: accuracy / convergence")
    _style_axes(ax)
    fig.tight_layout()
    return fig
