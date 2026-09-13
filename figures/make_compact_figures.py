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

    `normalize.hash_communication()` is a documented empty-list placeholder
    (contract.md: "the run-record schema has no `partition_row_policy` field
    yet"). This function does not invent that field to make a plot appear --
    it raises `NotImplementedError` naming the missing schema field, which the
    caller is expected to catch, rather than crashing on an empty-list `.plot()`
    call somewhere deep inside matplotlib.
    """
    if not rows:
        raise NotImplementedError(
            "make_hash_communication_figure: hash_communication() returned no rows "
            "because the run-record schema has no 'partition_row_policy' field yet "
            "(see contract.md and normalize.hash_communication's docstring); "
            "add that field to the schema before this figure has anything to plot."
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    # Left uninhabited until the schema/normalize gap above is closed; no row
    # shape is assumed here since none has ever been produced.
    ax.set_title("SYNTHETIC — placeholder: hash communication")
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
