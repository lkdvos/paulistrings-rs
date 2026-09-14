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


def make_hash_communication_vs_cutoff_figure(rows: Sequence[dict]):
    """Export volume vs. `min_abs_coeff`, one line per `partition_row_policy`.

    Same underlying data as `make_hash_communication_figure` (`normalize.
    hash_communication()`'s output, now carrying `min_abs_coeff` per row) but
    plotted as a cutoff sweep rather than a per-`config_id` bar chart -- the
    same "a single point/category is a weak plot" upgrade requested for the
    accuracy figure (see `make_convergence_figure`), applied to E8: does the
    cut policy's advantage over random hold, grow, or shrink as the cutoff
    tightens and the sum grows?
    """
    if not rows:
        raise NotImplementedError(
            "make_hash_communication_vs_cutoff_figure: no rows to plot -- see "
            "make_hash_communication_figure's docstring for the same underlying "
            "'no partition_row_policy-tagged data' cause."
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))

    colors = {"random": _IDEAL, "cut": _ACCENT}
    policies = sorted({r["partition_row_policy"] for r in rows})
    for policy in policies:
        pts = sorted((r["min_abs_coeff"], r["total_rows_exported"]) for r in rows if r["partition_row_policy"] == policy)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, color=colors.get(policy, _ACCENT), label=policy)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("min_abs_coeff")
    ax.set_ylabel("total rows exported")
    ax.set_title("Hash communication vs. truncation cutoff")
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


def make_convergence_figure(rows: Sequence[dict], julia_points: Sequence[dict] = ()):
    """One line per `min_abs_coeff`: the observable's expectation value at
    every Trotter step, from `jobs/run_convergence_sweep.py`'s
    `convergence.jsonl` records.

    `rows` is that file's records directly (not a `normalize.py` function's
    output -- the sweep driver already writes one row per point, so there is
    nothing to normalize). Only `status == "completed"` rows are plotted;
    an `invalid_hardware` sweep raises `NotImplementedError` rather than
    silently plotting nothing, matching this module's other "real data or an
    explicit reason" contract.

    `julia_points` optionally overlays PauliPropagation.jl reference points
    (`{"min_abs_coeff", "trotter_step", "expectation_re"}` dicts, e.g. built
    from `jobs/run_cell_julia.py`'s run records) as black stars -- Julia's
    `runner.jl` only computes the observable's expectation once, at the end
    of the whole circuit (see its `PP_LAYER_COUNTS` docs: per-layer *term
    counts* are available, per-layer *expectation values* are not), so this
    is real endpoint(s), not a Julia trajectory line. Each point is matched
    to the Rust line of the same `min_abs_coeff` when one exists.
    """
    completed = [r for r in rows if r.get("status") == "completed"]
    if not completed:
        raise NotImplementedError(
            "make_convergence_figure: no completed rows to plot -- the sweep "
            "either never ran or failed preflight (see the row's status/"
            "failure_reason)."
        )

    import math

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))

    by_eps: dict[float, list[dict]] = {}
    for r in completed:
        by_eps.setdefault(r["min_abs_coeff"], []).append(r)

    cmap = plt.get_cmap("viridis")
    epsilons = sorted(by_eps)
    for i, eps in enumerate(epsilons):
        pts = sorted(by_eps[eps], key=lambda r: r["trotter_step"])
        xs = [p["trotter_step"] for p in pts]
        ys = [p["expectation_re"] for p in pts]
        color = cmap(i / max(len(epsilons) - 1, 1))
        # log2(eps) for a compact, campaign-native label (every cutoff here is dyadic).
        label = f"eps=2^{round(math.log2(eps))}" if eps > 0 else "eps=0"
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, color=color, label=label)

    if julia_points:
        jxs = [p["trotter_step"] for p in julia_points]
        jys = [p["expectation_re"] for p in julia_points]
        ax.scatter(
            jxs, jys, marker="*", s=140, color="black", zorder=5,
            label="PauliPropagation.jl", edgecolors="white", linewidths=0.5,
        )

    ax.set_xlabel("Trotter step")
    ax.set_ylabel("<O>")
    ax.set_title("Observable trajectory vs. truncation cutoff")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    return fig
