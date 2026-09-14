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


# --- Deck v2 theme -----------------------------------------------------------
# White/transparent background, dark-navy text on white, light-gray grid, per
# the updated 960x540pt deck spec. Founders Grotesk / Tiempos Headline (the
# deck's real fonts) are NOT installed in this environment (checked via
# matplotlib.font_manager.fontManager.ttflist); "DejaVu Sans" is matplotlib's
# own bundled sans-serif and is used here as an honestly-labeled fallback,
# never claimed to be the real typeface. Math text uses matplotlib's built-in
# 'stix' mathtext font set, a real, license-clean STIX approximation that
# needs no external font file.
_DECK_NAVY = "#1c2954"
_DECK_GRID = "#d7dbe6"
_DECK_BODY_FONT = "DejaVu Sans"
_DECK_FONT_PT = 18

# Each series is distinguished by BOTH color and marker/linestyle, never color
# alone.
_DECK_SERIES = [
    {"color": "#1c2954", "marker": "o", "linestyle": "-"},
    {"color": "#c1543a", "marker": "^", "linestyle": "--"},
    {"color": "#2f8f7a", "marker": "s", "linestyle": ":"},
    {"color": "#a67c1c", "marker": "D", "linestyle": "-."},
    {"color": "#5b5f97", "marker": "v", "linestyle": "-"},
    # Added for the baseline-eps-scaling figure's 7-series progressive reveal (page 16) --
    # additive only, existing figures only ever index 0-4 and are unaffected.
    {"color": "#8a3b6b", "marker": "P", "linestyle": "--"},
    {"color": "#3a7ca5", "marker": "X", "linestyle": ":"},
]


def _deck_rc_params(font_pt: float = _DECK_FONT_PT) -> dict:
    """rcParams for the deck theme, meant for a `matplotlib.rc_context`.

    Font sizes are in matplotlib "points", the same typographic point
    `export_deck_figure` sizes the canvas in -- see that function's
    docstring for why that makes these sizes land correctly at final
    placement.
    """
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [_DECK_BODY_FONT],
        "mathtext.fontset": "stix",
        "font.size": font_pt,
        "axes.labelsize": font_pt,
        "axes.titlesize": font_pt,
        "legend.fontsize": font_pt * 0.75,
        "xtick.labelsize": font_pt * 0.8,
        "ytick.labelsize": font_pt * 0.8,
        "text.color": _DECK_NAVY,
        "axes.labelcolor": _DECK_NAVY,
        "xtick.color": _DECK_NAVY,
        "ytick.color": _DECK_NAVY,
        "axes.edgecolor": _DECK_NAVY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }


def _style_axes_deck(ax) -> None:
    ax.set_facecolor("none")
    ax.grid(True, color=_DECK_GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_DECK_NAVY)
    ax.tick_params(colors=_DECK_NAVY)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(_DECK_NAVY)


def export_deck_figure(fig, base_path: str, width_pt: float, height_pt: float, dpi: int = 200) -> dict:
    """Export `fig` at an exact deck-point size to SVG + PDF + PNG.

    `width_pt`/`height_pt` are PDF/deck points (1 pt = 1/72 in, the same unit
    PDF and matplotlib's own font-size both use). Sizing the canvas as
    inches = points/72 means a figure placed UNSCALED on the deck reproduces
    its matplotlib font sizes as identical PDF points at final placement --
    the sizing contract every deck-themed figure in this module relies on.
    `dpi` only affects the PNG preview; SVG/PDF are vector and scale-free.
    Returns the three written paths keyed by extension.
    """
    fig.set_size_inches(width_pt / 72.0, height_pt / 72.0)
    paths = {}
    for ext in ("svg", "pdf", "png"):
        path = f"{base_path}.{ext}"
        fig.savefig(path, format=ext, dpi=dpi, transparent=True)
        paths[ext] = path
    return paths


def make_thread_scaling_figure(
    rows: Sequence[dict],
    other_rows: Sequence[dict] = (),
    other_label: str = "other",
    *,
    theme: str = "legacy",
    figsize_pt: tuple[float, float] | None = None,
    title: str | None = None,
):
    """One curve: thread count (x, log2-spaced) vs. speedup (y), from `thread_scaling()` rows.

    `speedup` is already relative to the filtered set's 1-thread row
    (`normalize.thread_scaling`'s contract); this function plots that column
    verbatim rather than recomputing anything against total/absolute time, so
    a y=x line at threads=1 always starts at speedup=1 by construction, never
    an absolute wall-clock baseline.

    `other_rows` optionally overlays a second engine's thread-scaling curve
    (same row shape, e.g. `thread_scaling()` called with `variant_id=
    "external_pauli_propagation_jl"` for a Julia comparison) -- each series is
    normalized against its OWN 1-thread time, so the comparison is of relative
    parallel efficiency, not absolute wall-clock speed. One shared "ideal"
    reference line covers both, since y=x on a log-log plot is independent of
    either series' absolute baseline.

    `theme="legacy"` (default) is this function's original parchment styling,
    exercised by the existing tests, and is byte-for-byte unchanged.
    `theme="deck"` switches to the v2 deck theme (`_DECK_*`, `_style_axes_deck`)
    -- distinct marker/linestyle per series, not just color -- and honors
    `figsize_pt`/`title` for exact deck sizing (see `export_deck_figure`).
    """
    import matplotlib.pyplot as plt

    deck = theme == "deck"
    figsize = (figsize_pt[0] / 72.0, figsize_pt[1] / 72.0) if (deck and figsize_pt) else (5.5, 4)

    def _build():
        fig, ax = plt.subplots(figsize=figsize)

        def _plot_series(series_rows, idx, legacy_color, label):
            pts = sorted(
                (r["threads"], r["speedup"]) for r in series_rows if r.get("speedup") is not None
            )
            if pts:
                xs, ys = zip(*pts)
                if deck:
                    st = _DECK_SERIES[idx % len(_DECK_SERIES)]
                    ax.plot(
                        xs, ys, marker=st["marker"], markersize=6, linewidth=1.8,
                        linestyle=st["linestyle"], color=st["color"], label=label,
                    )
                else:
                    ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, color=legacy_color, label=label)
            return pts

        points = _plot_series(rows, 0, _ACCENT, "measured")
        other_points = _plot_series(other_rows, 1, "#eb6834", other_label) if other_rows else []

        all_points = points or other_points
        if all_points:
            xs = sorted({p[0] for p in points} | {p[0] for p in other_points})
            ideal_ys = [x / xs[0] for x in xs]
            if deck:
                ax.plot(xs, ideal_ys, linestyle=(0, (1, 1)), linewidth=1.2, color="#8a8f9c", label="ideal (y=x)")
            else:
                ax.plot(xs, ideal_ys, linestyle="--", linewidth=1.0, color=_IDEAL, label="ideal (y=x)")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup vs. 1 thread")
        if deck:
            if title:
                ax.set_title(title, fontsize=_DECK_FONT_PT, color=_DECK_NAVY)
            _style_axes_deck(ax)
        else:
            ax.set_title("SYNTHETIC — placeholder: thread scaling")
            _style_axes(ax)
        if all_points:
            ax.legend(frameon=False)
        fig.tight_layout()
        return fig

    if deck:
        import matplotlib as mpl

        with mpl.rc_context(_deck_rc_params()):
            fig = _build()
    else:
        fig = _build()
    return fig


def make_attempts_figure(attempt_rows: Sequence[dict], current_engine_row: dict | None = None):
    """Wall time vs. threads, one line per historically-attempted parallelization strategy.

    `attempt_rows` is pre-aggregated (one row per (label, threads), e.g. median
    of repetitions): `{"label": str, "description": str, "threads": int,
    "wall_time_s": float}`. Source is the recovered `presentation` branch's
    `bench/{threadmaps,mergesort}.rs` reconstructions plus its own in-file
    `bucketed` reference point (`raw/recovered-presentation-branch-attempts/`)
    -- real measured wall-clock time on `ccqlin038`, NOT this campaign's
    genoa/rocky9 node class. This is cross-architecture historical reference
    data, plotted honestly as such (title + caption note), never as a
    same-hardware head-to-head.

    `current_engine_row` optionally adds a single labeled reference point for
    THIS campaign's own bucketed-engine 1-thread wall time (e.g. from
    `figures/real/thread_scaling.jsonl`'s 1-thread row), drawn as a distinct
    star marker with its own label -- explicitly not connected into any line,
    since it is a different host/architecture and must never look like part
    of the same scaling curve.
    """
    if not attempt_rows:
        raise NotImplementedError(
            "make_attempts_figure: no rows to plot -- recover "
            "raw/recovered-presentation-branch-attempts/thread_scaling.jsonl "
            "and aggregate it into (label, threads, wall_time_s) rows first."
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))

    labels = sorted({r["label"] for r in attempt_rows})
    palette = [_ACCENT, "#eb6834", "#5a8f3c", "#a15fb5"]
    colors = {label: palette[i % len(palette)] for i, label in enumerate(labels)}
    descriptions = {r["label"]: r.get("description", "") for r in attempt_rows}

    for label in labels:
        pts = sorted(
            (r["threads"], r["wall_time_s"])
            for r in attempt_rows
            if r["label"] == label and r.get("wall_time_s") is not None
        )
        if not pts:
            continue
        xs, ys = zip(*pts)
        legend_label = label
        if descriptions.get(label):
            legend_label = f"{label} ({descriptions[label]})"
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, color=colors[label], label=legend_label)

    if current_engine_row is not None and current_engine_row.get("wall_time_s") is not None:
        ax.scatter(
            [current_engine_row["threads"]],
            [current_engine_row["wall_time_s"]],
            marker="*",
            s=160,
            color="black",
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
            label=current_engine_row.get("label", "current engine (this campaign, different node class)"),
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("threads")
    ax.set_ylabel("wall time (s)")
    ax.set_title("Attempted parallel strategies — historical, cross-architecture reference")
    ax.legend(frameon=False, fontsize=7, loc="best")
    _style_axes(ax)
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


def make_hash_communication_vs_cutoff_figure(
    rows: Sequence[dict],
    *,
    theme: str = "legacy",
    figsize_pt: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Export volume vs. `min_abs_coeff`, one line per `partition_row_policy`.

    Same underlying data as `make_hash_communication_figure` (`normalize.
    hash_communication()`'s output, now carrying `min_abs_coeff` per row) but
    plotted as a cutoff sweep rather than a per-`config_id` bar chart -- the
    same "a single point/category is a weak plot" upgrade requested for the
    accuracy figure (see `make_convergence_figure`), applied to E8: does the
    cut policy's advantage over random hold, grow, or shrink as the cutoff
    tightens and the sum grows?

    `theme="legacy"` (default, unchanged) vs `theme="deck"` -- see
    `make_thread_scaling_figure`'s docstring for the shared contract.
    """
    if not rows:
        raise NotImplementedError(
            "make_hash_communication_vs_cutoff_figure: no rows to plot -- see "
            "make_hash_communication_figure's docstring for the same underlying "
            "'no partition_row_policy-tagged data' cause."
        )

    import matplotlib.pyplot as plt

    deck = theme == "deck"
    figsize = (figsize_pt[0] / 72.0, figsize_pt[1] / 72.0) if (deck and figsize_pt) else (5.5, 4)
    # Random vs. cut are always the same two policies here, so a fixed
    # (marker, linestyle) pair per policy keeps them distinguishable by shape
    # too, not just color, across every deck-themed export.
    deck_style = {"random": {"color": "#c1543a", "marker": "^", "linestyle": "--"},
                  "cut": {"color": _DECK_NAVY, "marker": "o", "linestyle": "-"}}

    def _build():
        fig, ax = plt.subplots(figsize=figsize)

        colors = {"random": _IDEAL, "cut": _ACCENT}
        policies = sorted({r["partition_row_policy"] for r in rows})
        for policy in policies:
            pts = sorted((r["min_abs_coeff"], r["total_rows_exported"]) for r in rows if r["partition_row_policy"] == policy)
            if not pts:
                continue
            xs, ys = zip(*pts)
            if deck:
                st = deck_style.get(policy, _DECK_SERIES[0])
                ax.plot(xs, ys, marker=st["marker"], markersize=6, linewidth=1.8,
                         linestyle=st["linestyle"], color=st["color"], label=policy)
            else:
                ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, color=colors.get(policy, _ACCENT), label=policy)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("min_abs_coeff")
        ax.set_ylabel("total rows exported")
        if deck:
            if title:
                ax.set_title(title, fontsize=_DECK_FONT_PT, color=_DECK_NAVY)
            _style_axes_deck(ax)
        else:
            ax.set_title("Hash communication vs. truncation cutoff")
            _style_axes(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        return fig

    if deck:
        import matplotlib as mpl

        with mpl.rc_context(_deck_rc_params()):
            fig = _build()
    else:
        fig = _build()
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


def make_convergence_figure(
    rows: Sequence[dict],
    julia_rows: Sequence[dict] = (),
    *,
    theme: str = "legacy",
    figsize_pt: tuple[float, float] | None = None,
    title: str | None = None,
    show_diff: bool = False,
):
    """One line per `min_abs_coeff`: the observable's expectation value at
    every Trotter step, from `jobs/run_convergence_sweep.py`'s
    `convergence.jsonl` records.

    `rows` is that file's records directly (not a `normalize.py` function's
    output -- the sweep driver already writes one row per point, so there is
    nothing to normalize). Only `status == "completed"` rows are plotted;
    an `invalid_hardware` sweep raises `NotImplementedError` rather than
    silently plotting nothing, matching this module's other "real data or an
    explicit reason" contract.

    `julia_rows` optionally overlays PauliPropagation.jl data in the SAME row
    shape as `rows` -- one row per `(min_abs_coeff, trotter_step)`, e.g. from
    `jobs/run_convergence_sweep_julia.py`'s `convergence_julia.jsonl`
    (`runner.jl`'s `PP_LAYER_EXPECTATION` diagnostic; see its module header
    for why this needed a genuine per-step re-propagation, not a cheap
    per-gate hook like `PP_LAYER_COUNTS`). A cutoff with more than one Julia
    point gets a real dashed line, in the SAME color as the Rust line at that
    cutoff so the pair reads as one comparison; a cutoff with exactly one
    point (e.g. `jobs/run_cell_julia.py`'s single final-step run record,
    decisions.md #27/#33, which has no `trotter_step` of its own and is
    conventionally given the final step) falls back to a black star, the
    original single-endpoint overlay this function shipped with.

    `show_diff=True` adds a second, shorter panel below the trajectory: the
    signed difference (Rust `expectation_re` minus Julia's, per matched
    `(min_abs_coeff, trotter_step)` pair) -- same color per cutoff as the top
    panel. Two nearly-overlapping lines in the same color are hard to compare
    by eye; the raw gap between them, on its own axis, is not. Only cutoffs
    with a REAL multi-point Julia trajectory (not the single-endpoint star)
    get a diff line, since a single point has no trajectory to difference
    against meaningfully across steps.
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

    deck = theme == "deck"
    figsize = (figsize_pt[0] / 72.0, figsize_pt[1] / 72.0) if (deck and figsize_pt) else (5.5, 4)

    by_eps: dict[float, list[dict]] = {}
    for r in completed:
        by_eps.setdefault(r["min_abs_coeff"], []).append(r)

    jl_completed = [r for r in julia_rows if r.get("status", "completed") == "completed"]
    jl_by_eps: dict[float, list[dict]] = {}
    for r in jl_completed:
        jl_by_eps.setdefault(r["min_abs_coeff"], []).append(r)

    epsilons = sorted(set(by_eps) | set(jl_by_eps))

    def _label(eps: float) -> str:
        # log2(eps) for a compact, campaign-native label (every cutoff here is dyadic),
        # rendered as real mathtext rather than plain "eps=2^-16" text.
        return rf"$\varepsilon=2^{{{round(math.log2(eps))}}}$" if eps > 0 else r"$\varepsilon=0$"

    # A diff panel only makes sense for cutoffs with a REAL multi-point Julia
    # trajectory (the single-endpoint star has nothing to difference across steps).
    diff_eps = sorted(eps for eps, pts in jl_by_eps.items() if len(pts) > 1) if show_diff else []

    def _build():
        if diff_eps:
            fig, (ax, ax_diff) = plt.subplots(
                2, 1, figsize=figsize, sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            )
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax_diff = None

        if deck:
            # Deck theme: one (color, marker/linestyle) pair per cutoff from
            # the shared _DECK_SERIES palette, instead of a continuous
            # colormap -- keeps every line distinguishable by shape too.
            colors = {eps: _DECK_SERIES[i % len(_DECK_SERIES)]["color"] for i, eps in enumerate(epsilons)}
            styles = {eps: _DECK_SERIES[i % len(_DECK_SERIES)] for i, eps in enumerate(epsilons)}
        else:
            cmap = plt.get_cmap("viridis")
            colors = {eps: cmap(i / max(len(epsilons) - 1, 1)) for i, eps in enumerate(epsilons)}
            styles = None

        for eps in sorted(by_eps):
            pts = sorted(by_eps[eps], key=lambda r: r["trotter_step"])
            xs = [p["trotter_step"] for p in pts]
            ys = [p["expectation_re"] for p in pts]
            if deck:
                st = styles[eps]
                ax.plot(
                    xs, ys, marker=st["marker"], markersize=5, linewidth=1.6,
                    linestyle=st["linestyle"], color=colors[eps], label=_label(eps),
                )
            else:
                ax.plot(
                    xs, ys, marker="o", markersize=3, linewidth=1.2, color=colors[eps],
                    label=_label(eps),
                )

        star_labeled = False
        for eps in sorted(jl_by_eps):
            jl_pts = sorted(jl_by_eps[eps], key=lambda r: r["trotter_step"])
            if len(jl_pts) > 1:
                jxs = [p["trotter_step"] for p in jl_pts]
                jys = [p["expectation_re"] for p in jl_pts]
                ax.plot(
                    jxs, jys, linestyle="--", marker="^", markersize=4, linewidth=1.2,
                    color=colors[eps], label=f"PauliPropagation.jl {_label(eps)}",
                )
            else:
                p = jl_pts[0]
                star_color = _DECK_NAVY if deck else "black"
                ax.scatter(
                    [p["trotter_step"]], [p["expectation_re"]], marker="*", s=140,
                    color=star_color, zorder=5,
                    label=None if star_labeled else "PauliPropagation.jl",
                    edgecolors="white", linewidths=0.5,
                )
                star_labeled = True

        if ax_diff is not None:
            for eps in diff_eps:
                rust_by_step = {p["trotter_step"]: p["expectation_re"] for p in by_eps.get(eps, [])}
                jl_by_step = {p["trotter_step"]: p["expectation_re"] for p in jl_by_eps[eps]}
                shared_steps = sorted(set(rust_by_step) & set(jl_by_step))
                dxs = shared_steps
                dys = [rust_by_step[s] - jl_by_step[s] for s in shared_steps]
                if deck:
                    st = styles[eps]
                    ax_diff.plot(
                        dxs, dys, marker=st["marker"], markersize=4, linewidth=1.4,
                        linestyle=st["linestyle"], color=colors[eps],
                    )
                else:
                    ax_diff.plot(dxs, dys, marker="o", markersize=3, linewidth=1.0, color=colors[eps])
            ax_diff.axhline(0.0, color=_DECK_NAVY if deck else "#898781", linewidth=0.8, linestyle=":")
            ax_diff.set_xlabel("Trotter step")
            ax_diff.set_ylabel("Rust $-$ Julia", fontsize=(_DECK_FONT_PT * 0.7 if deck else 8))
            if deck:
                _style_axes_deck(ax_diff)
            else:
                _style_axes(ax_diff)
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Trotter step")
        ax.set_ylabel(r"$\langle O \rangle$")
        if deck:
            if title:
                ax.set_title(title, fontsize=_DECK_FONT_PT, color=_DECK_NAVY)
            _style_axes_deck(ax)
        else:
            ax.set_title("Observable trajectory vs. truncation cutoff")
            _style_axes(ax)
        # A plain loc="best" legend picked a spot that overlapped real plotted data once the
        # diff panel changed this figure's aspect ratio (caught on an actual exported figure,
        # not a theoretical concern) -- outside the axes, to the right, is layout-independent.
        ax.legend(
            frameon=False, fontsize=8 if not deck else _DECK_FONT_PT * 0.75,
            loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
        )
        if ax_diff is not None:
            # tight_layout() doesn't fully account for a sharex two-row gridspec (it warns
            # and can under-reserve room for ax_diff's own x-label) -- a real clipped label
            # was caught on an actual exported figure, not a theoretical concern. An explicit
            # margin after tight_layout's best-effort pass fixes both that and legend room.
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.22, right=0.62, hspace=0.12)
        else:
            fig.tight_layout()
            fig.subplots_adjust(right=0.62)
        return fig

    if deck:
        import matplotlib as mpl

        with mpl.rc_context(_deck_rc_params()):
            fig = _build()
    else:
        fig = _build()
    return fig


def make_distributed_capacity_figure(
    rows: Sequence[dict],
    *,
    theme: str = "legacy",
    figsize_pt: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Distributed-engine (`DistributedSum`, `ARCHITECTURE.md`'s MPI layer)
    runtime and capacity view: one point per `(ranks, min_abs_coeff)` cell,
    from hand-assembled `raw/*/runs.jsonl` records (E6/E7 in `evidence.md`).

    There is no `normalize.py` helper for this exact shape: `rank_scaling()`
    requires a completed 1-rank baseline at the SAME `min_abs_coeff` to
    classify a row as "overlap" vs "capacity_extension", and several real
    points here genuinely have no such baseline (`decisions.md` #22-23) --
    an untested 1-rank point, and an OOM'd intermediate rank count. This
    function accepts that pre-classified reality directly rather than forcing
    it through a helper built for the cleaner case.

    Each row: `ranks` (int), `min_abs_coeff` (float), `wall_time_s` (float,
    ONLY when `status == "completed"`), `peak_terms` (int or None),
    `peak_rss_kb` (float or None), `status` ("completed" | "oom" | "timeout"
    | "untested"), and an optional `note` (str) rendered as a point
    annotation -- e.g. whether a capacity boundary is measured or inferred,
    and which memory allocation it refers to, per this figure's own
    reporting requirement. A non-completed row carrying a real `wall_time_s`
    is a caller bug: a failed or untested run has no honest runtime, and this
    function refuses to plot one rather than silently accepting it.
    """
    if not rows:
        raise NotImplementedError(
            "make_distributed_capacity_figure: no rows to plot -- assemble at "
            "least one distributed-engine run record (E6/E7 in evidence.md) "
            "before this figure has anything to show."
        )
    for r in rows:
        if r["status"] != "completed" and r.get("wall_time_s") is not None:
            raise ValueError(
                f"make_distributed_capacity_figure: non-completed row (status={r['status']!r}, "
                f"ranks={r.get('ranks')!r}) carries a wall_time_s -- a failed/untested run must "
                "never plot a fabricated runtime"
            )

    import math
    from collections import Counter

    import matplotlib.pyplot as plt

    deck = theme == "deck"
    figsize = (figsize_pt[0] / 72.0, figsize_pt[1] / 72.0) if (deck and figsize_pt) else (6.5, 4.5)

    def _label(eps: float) -> str:
        return rf"$\varepsilon=2^{{{round(math.log2(eps))}}}$" if eps > 0 else r"$\varepsilon=0$"

    def _sci(n: int) -> str:
        # 3-significant-figure scientific notation, mathtext content ONLY (no $ delimiters --
        # callers embed this inside their own math span), e.g. 8923556570 -> "8.92\times10^{9}".
        # Both shorter (helps the legend actually fit) and matches the "numbers in LaTeX math"
        # request, rather than a long comma-grouped integer.
        exp = len(str(n)) - 1
        mantissa = n / (10**exp)
        return rf"{mantissa:.2f}\times10^{{{exp}}}"

    def _build():
        fig, ax = plt.subplots(figsize=figsize)

        by_eps: dict[float, list[dict]] = {}
        for r in rows:
            by_eps.setdefault(r["min_abs_coeff"], []).append(r)
        epsilons = sorted(by_eps)

        any_plotted = False
        for i, eps in enumerate(epsilons):
            if deck:
                style = _DECK_SERIES[i % len(_DECK_SERIES)]
            else:
                style = {"color": _ACCENT if i == 0 else _IDEAL, "marker": "o", "linestyle": "-"}
            annotation_color = _DECK_NAVY if deck else "#898781"

            series_rows = sorted(by_eps[eps], key=lambda r: r["ranks"])
            completed = [r for r in series_rows if r["status"] == "completed"]
            if completed:
                any_plotted = True
                xs = [r["ranks"] for r in completed]
                ys = [r["wall_time_s"] for r in completed]
                # peak_terms is stated once in the legend label rather than at every point, to
                # avoid repeating "N=..." densely along a short log-log line. Uses the most
                # common value across the series (a single-rank reference point can legitimately
                # report final_terms rather than peak_terms and disagree slightly -- accepted
                # per the user's explicit call, not hidden as if it matched).
                n_counts = Counter(r["peak_terms"] for r in completed if r.get("peak_terms"))
                label = _label(eps)
                if n_counts:
                    label = rf"{label}, $N={_sci(n_counts.most_common(1)[0][0])}$"
                # Peak memory is reported once, as the series' MAXIMUM, in the legend label --
                # not per-point -- per the user's explicit call: individual per-point memory
                # annotations crowded the plot, and the max across the series is the number that
                # matters for a capacity story anyway.
                rss_values = [r["peak_rss_kb"] for r in completed if r.get("peak_rss_kb")]
                if rss_values:
                    label = f"{label}, max {max(rss_values) / 1e9:.2f} TB"
                ax.plot(
                    xs, ys, marker=style["marker"], markersize=7, linewidth=1.8,
                    linestyle=style["linestyle"] if len(xs) > 1 else "none",
                    color=style["color"], label=label,
                )
                # No per-point text annotations at all (not even a "note") -- every qualifying
                # remark (measured vs. inferred, OOM/untested provenance) is stated by the
                # presenter verbally, per the user's explicit call; the figure shows only the
                # real plotted points and the legend.
            # Non-completed rows (oom/timeout/untested) are validated above (no fabricated
            # wall_time_s) but deliberately NOT drawn -- the presenter states those verbally
            # rather than having the figure show sentinel markers for unmeasured points.

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("ranks")
        ax.set_ylabel("wall time (s)")
        if deck:
            if title:
                ax.set_title(title, fontsize=_DECK_FONT_PT, color=_DECK_NAVY)
            _style_axes_deck(ax)
        else:
            ax.set_title("Distributed engine: runtime and capacity by rank count")
            _style_axes(ax)

        if any_plotted:
            handles, labels = ax.get_legend_handles_labels()
            seen: set[str] = set()
            dedup_handles, dedup_labels = [], []
            for h, l in zip(handles, labels):
                if l in seen:
                    continue
                seen.add(l)
                dedup_handles.append(h)
                dedup_labels.append(l)
            # Along the bottom, outside the axes: one line if the real rendered width actually
            # fits the figure, otherwise one entry per line (ncol=1) -- no intermediate column
            # count, per the user's explicit either/or. matplotlib does NOT auto-wrap a fixed
            # ncol when it's too wide, it just overflows past the canvas edge (confirmed the
            # hard way: an earlier version of this code assumed wrapping and clipped real text
            # off a real exported figure) -- so this measures the legend's actual on-canvas
            # width after a real draw rather than assuming a fixed ncol will fit.
            fontsize = 7 if not deck else _DECK_FONT_PT * 0.7
            legend = ax.legend(
                dedup_handles, dedup_labels, frameon=False, fontsize=fontsize,
                loc="upper center", bbox_to_anchor=(0.5, -0.18), borderaxespad=0.0,
                ncol=len(dedup_labels),
            )
            fig.canvas.draw()
            fits_one_line = legend.get_window_extent().width <= fig.get_window_extent().width
            ncol = len(dedup_labels) if fits_one_line else 1
            if not fits_one_line:
                legend.remove()
                legend = ax.legend(
                    dedup_handles, dedup_labels, frameon=False, fontsize=fontsize,
                    loc="upper center", bbox_to_anchor=(0.5, -0.18), borderaxespad=0.0,
                    ncol=1,
                )
            n_rows = math.ceil(len(dedup_labels) / ncol)
            fig.tight_layout()
            # tight_layout() doesn't know the legend lives outside the axes -- it would
            # otherwise shrink the bottom margin back down and reclip the legend it just fit.
            fig.subplots_adjust(bottom=0.14 + 0.09 * n_rows)
        else:
            fig.tight_layout()
        return fig

    if deck:
        import matplotlib as mpl

        with mpl.rc_context(_deck_rc_params()):
            fig = _build()
    else:
        fig = _build()
    return fig


def make_memory_diagnosis_figure(
    phase_rows: Sequence[dict],
    traffic: dict,
    *,
    theme: str = "legacy",
    figsize_pt: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Two-panel memory/bandwidth diagnosis view: phase time-share (A) plus a
    payload/traffic/peak-memory summary (B), for `phase_breakdown --features
    phase-timing` runs (`ARCHITECTURE.md` `coset_loop` phase names).

    `phase_rows`: one dict per measured thread count, each `{"threads": int,
    "wall_ms_per_layer": float, "phases": {"permute": float, "coset_loop":
    float, "unpermute": float, "recount": float, "other": float}}` -- ms/layer
    values, `"other"` already pre-summed by the caller (rebucket + prepare +
    span_plan + finalize), the same "small phases folded into one bucket"
    convention the probe's own `.txt`/HTML report uses. Panel A stacks each
    row's phases as a time-SHARE bar (`phase / wall_ms_per_layer`), not raw
    ms, so a 1-thread and a 96-thread row are visually comparable despite a
    ~3x different wall time per layer; the real per-layer ms total is
    annotated next to each bar so the absolute magnitude is not lost.

    `traffic` carries the three numbers this figure exists to keep DISTINCT
    -- never conflated into one metric:
      - `payload_bytes_per_term` (int): the fixed `T=16W+16` payload fact
        (48 for W=2, Complex64), independent of any measurement.
      - `modeled_traffic_gbps` (float) and `modeled_bytes_per_term` (float):
        a MODELED traffic-per-update estimate -- derived from the probe's
        real gather/sort/merge/terms counts divided by the real wall time of
        `traffic_scope_label` -- deliberately larger than the raw payload
        because it counts gather streams, sort/merge temporaries, and
        coeff-only metadata rows, not just the final resident term.
      - `bandwidth_ceiling_gbps` (float or None): the REAL measured ceiling
        at a matching thread count, from THIS host's own `bandwidth.txt`
        ONLY -- never a different architecture's number. When `None`,
        `bandwidth_unavailable_reason` (str) must explain why, and this
        function renders that reason instead of fabricating a % of ceiling.
      - `peak_vmhwm_kb` (float): peak resident memory (`VmHWM`), reported
        on its own, never divided by anything or folded into the traffic
        number above.

    Panel B is deliberately NOT a bar chart of these three numbers together
    -- they are different units (bytes/term, GB/s, kB) and plotting them on
    one shared axis would visually imply they are comparable magnitudes.
    Instead it renders three text "stat tiles", the same
    distinct-units-distinct-tiles idea `make_distributed_capacity_figure`
    uses for its legend-only memory annotation, just made primary here since
    memory diagnosis IS this figure's subject rather than a footnote.

    `theme="legacy"` (default) vs `theme="deck"` -- see
    `make_thread_scaling_figure`'s docstring for the shared contract.
    """
    if not phase_rows:
        raise NotImplementedError(
            "make_memory_diagnosis_figure: no phase rows to plot -- run "
            "phase_breakdown --features phase-timing for at least one thread count."
        )

    import matplotlib.pyplot as plt

    deck = theme == "deck"
    figsize = (figsize_pt[0] / 72.0, figsize_pt[1] / 72.0) if (deck and figsize_pt) else (9.5, 4)
    # Below ~250pt tall there is no room for panel B's sub-captions and the
    # bandwidth-unavailable note without them colliding -- same "compact
    # drops qualifying detail, presenter states it verbally" precedent as
    # `make_distributed_capacity_figure`'s half-height compact export.
    compact = bool(figsize_pt) and figsize_pt[1] < 250

    phase_order = ["permute", "coset_loop", "unpermute", "recount", "other"]
    phase_labels = {"permute": "permute", "coset_loop": "coset_loop", "unpermute": "unpermute",
                    "recount": "recount", "other": "other (serial)"}

    def _build():
        fig, (ax_phase, ax_stats) = plt.subplots(1, 2, figsize=figsize)

        # --- panel A: phase time-share, one stacked bar per thread count ----
        rows = sorted(phase_rows, key=lambda r: r["threads"])
        ys = range(len(rows))
        left = [0.0] * len(rows)
        for i, phase in enumerate(phase_order):
            color = _DECK_SERIES[i % len(_DECK_SERIES)]["color"] if deck else None
            shares = [100.0 * r["phases"].get(phase, 0.0) / r["wall_ms_per_layer"] for r in rows]
            ax_phase.barh(list(ys), shares, left=left, height=0.55, color=color, label=phase_labels[phase])
            left = [l + s for l, s in zip(left, shares)]

        text_color = _DECK_NAVY if deck else "#333333"
        for i, r in enumerate(rows):
            ax_phase.annotate(
                f"{r['wall_ms_per_layer']:.2f} ms/layer",
                (101.0, i), xycoords=("data", "data"), va="center", ha="left",
                fontsize=8 if not deck else _DECK_FONT_PT * 0.6, color=text_color,
            )
        ax_phase.set_yticks(list(ys))
        ax_phase.set_yticklabels([f"{r['threads']} thread{'s' if r['threads'] != 1 else ''}" for r in rows])
        ax_phase.set_xlim(0, 100)
        ax_phase.set_xlabel("share of wall time (%)")
        ax_phase.legend(frameon=False, fontsize=7 if not deck else _DECK_FONT_PT * 0.55,
                         loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=len(phase_order))

        # --- panel B: three distinct stat tiles, never one shared axis ------
        # Manual `textwrap` (not matplotlib's `wrap=True`, which wraps at the
        # FIGURE edge, not the narrow ~1/3-width column each tile actually
        # has -- confirmed the hard way, an earlier version left three tiles'
        # text overlapping into one unreadable smear) at a width tuned to
        # this panel's column count (3) and font size.
        import textwrap

        ax_stats.axis("off")
        navy = _DECK_NAVY if deck else "#1c2954"
        muted = "#8a8f9c" if deck else "#898781"
        big_fs = (_DECK_FONT_PT * (0.65 if compact else 0.85)) if deck else 13
        label_fs = (_DECK_FONT_PT * (0.42 if compact else 0.55)) if deck else 8.5
        sub_fs = (_DECK_FONT_PT * (0.36 if compact else 0.48)) if deck else 7.5
        wrap_width = (14 if compact else 20) if deck else 24

        # Sub-captions are single short lines by design -- the fuller prose
        # (traffic scope, bandwidth-unavailable reasoning) lives in the ONE
        # shared note below, not repeated per tile, so three short tiles plus
        # one note fit the 340pt full box without collision.
        tiles = [
            ("payload" if compact else "payload (fixed)",
             f"{traffic['payload_bytes_per_term']:.0f} B/term",
             "W=2, Complex64"),
            ("traffic" if compact else "modeled traffic",
             f"{traffic['modeled_traffic_gbps']:.2f} GB/s",
             f"{traffic['modeled_bytes_per_term']:.0f} B/term-update"),
            ("peak RSS" if compact else "peak resident (VmHWM)",
             f"{traffic['peak_vmhwm_kb'] / 1e6:.2f} GB",
             f"{traffic['peak_vmhwm_kb']:,.0f} kB"),
        ]
        label_y, value_y, sub_y = (0.86, 0.52, 0.18) if compact else (0.92, 0.62, 0.36)
        for i, (label, value, sub) in enumerate(tiles):
            x = (i + 0.5) / 3.0
            ax_stats.text(x, label_y, label, ha="center", va="center",
                          fontsize=label_fs, color=navy, fontweight="bold",
                          transform=ax_stats.transAxes)
            ax_stats.text(x, value_y, value, ha="center", va="center", fontsize=big_fs,
                          color=navy, transform=ax_stats.transAxes)
            if not compact:
                ax_stats.text(x, sub_y, sub, ha="center", va="center", fontsize=sub_fs,
                              color=muted, transform=ax_stats.transAxes)

        # The bandwidth-unavailable/ceiling caveat is long prose -- in the
        # compact box it is dropped from the figure itself (same precedent as
        # above) and belongs in the caption/MANIFEST/presenter's voice instead.
        if not compact:
            if traffic.get("bandwidth_ceiling_gbps") is not None:
                pct = 100.0 * traffic["modeled_traffic_gbps"] / traffic["bandwidth_ceiling_gbps"]
                ceiling_note = (
                    f"traffic scope: {traffic.get('traffic_scope_label', '')}. "
                    f"ceiling {traffic['bandwidth_ceiling_gbps']:.1f} GB/s "
                    f"({pct:.1f}% of measured ceiling, this host)"
                )
            else:
                ceiling_note = (
                    f"traffic scope: {traffic.get('traffic_scope_label', '')}. "
                    + traffic.get(
                        "bandwidth_unavailable_reason",
                        "no bandwidth ceiling available for this host/thread count",
                    )
                )
            ax_stats.text(
                0.5, 0.04, "\n".join(textwrap.wrap(ceiling_note, wrap_width * 3)),
                ha="center", va="bottom", fontsize=sub_fs, color=muted,
                transform=ax_stats.transAxes,
            )

        if deck:
            if title:
                fig.suptitle(title, fontsize=_DECK_FONT_PT, color=_DECK_NAVY)
            _style_axes_deck(ax_phase)
            ax_phase.spines["left"].set_visible(False)
            ax_phase.grid(axis="y", visible=False)
        else:
            ax_phase.set_title("Phase time-share")
            ax_stats.set_title("Payload / traffic / peak memory", color="#333333")
            _style_axes(ax_phase)
            ax_phase.spines["left"].set_visible(False)
            ax_phase.grid(axis="y", visible=False)
        fig.tight_layout()
        return fig

    if deck:
        import matplotlib as mpl

        with mpl.rc_context(_deck_rc_params()):
            fig = _build()
    else:
        fig = _build()
    return fig


def make_bucket_size_figure(
    rows: Sequence[dict],
    *,
    theme: str = "legacy",
    figsize_pt: tuple[float, float] | None = None,
    title: str | None = None,
    single_panel: bool = False,
    l2_cache_bytes: float | None = None,
    bytes_per_term: float = 48.0,
):
    """Two panels vs. `target_bucket_len` (x, log2-spaced, one fixed cell per point):
    throughput (left) and occupancy distribution + empty-bucket fraction (right).

    `single_panel=True` renders throughput alone (no occupancy panel) -- the
    occupancy view stays available via the default two-panel mode, this is
    an alternate rendering of the SAME rows, not a different dataset.

    `l2_cache_bytes`, when given, draws a vertical reference line at the
    `target_bucket_len` whose bucket working set (`target_bucket_len *
    bytes_per_term`) equals that many bytes -- i.e. `l2_cache_bytes /
    bytes_per_term`. `bytes_per_term` defaults to 48 (W=2, Complex64, this
    repo's own fixed payload fact, `crates/paulistrings/src/bucket/sum.rs`).
    Per this module's own established discipline (see this function's design
    note below on why fig4b/fig5's cache bands were NOT borrowed originally):
    only pass a REAL, measured `l2_cache_bytes` for the actual host these
    rows were measured on (e.g. `lscpu -C` output from the same job) -- never
    a spec sheet or another architecture's number. A line on this plot is
    evidence consistent with locality if a throughput peak sits near it,
    never proof of cache residency by itself.

    `rows` are the probe's raw JSON sidecar objects directly (no `normalize.py`
    step -- there is no existing helper for this row shape and every other
    figure that lacks one, e.g. `make_distributed_capacity_figure`, takes raw
    dicts too), each augmented with a `strings_per_s` key: the probe's JSON
    does not carry `strings/s` (checked directly against the sidecar's own
    keys), only its sibling `.txt` phase-breakdown report does, on a
    `strings/s = ...` line, so the caller must read and attach that field
    before calling this function. Required keys per row: `target_bucket_len`,
    `strings_per_s`, `num_buckets`, `empty_buckets`, `occupancy_median`,
    `occupancy_p95`, `occupancy_max`.

    This is REAL data from exactly one cell configuration (fixed workload,
    threads=1, fixed hash, fixed `min_abs_coeff`, fixed depth -- only
    `target_bucket_len` varies), 5 points, job 7035853. Occupancy is sampled
    at the final step of a genuine 20-step trajectory, after a real
    depth-doubling bug in the occupancy-sampling path was found and fixed
    (`crates/paulistrings/examples/phase_breakdown.rs` git history: "fix
    occupancy sampling's silent depth-doubling").

    Over the 5 tested values (256..4096) throughput rises MONOTONICALLY with
    no peak in range -- this function's title/caption must never claim an
    optimum, a flattening, or cache residency; the honest statement is that
    the curve is still rising at the largest tested value. A performance
    optimum near an estimated cache size would be evidence consistent with
    locality, not proof of cache residency -- moot here anyway, since there
    is no optimum in the tested range at all.

    The empty-bucket fraction (`empty_buckets / num_buckets`) is rendered as
    its own bar series, never folded into the occupancy percentiles: a large
    empty fraction lowers the *mean* occupancy of all buckets but says
    nothing about how full the occupied ones are, which is what
    `occupancy_median/p95/max` already describe correctly by excluding empty
    buckets from their sample.

    Design note (compared against the older `presentation` branch's
    fig4b_bucket_speedup.py / fig5_bucket_sweep.py, which the user pointed at
    as "the figure I need"): those two figures measure genuinely different
    things -- fig4b is 16-vs-1-thread speedup and parallel efficiency vs.
    bucket size, fig5 is ns/term-layer and L2/LLC cache-miss rate vs. bucket
    size, both from a 2026-09-06 ccqlin038 (Cascade Lake) sweep with no
    `--occupancy-at` support at all. Neither shows an occupancy distribution,
    so neither is a structural match for this deck's actual page-30 ask
    ("throughput vs. target bucket size and occupancy distribution") -- this
    two-panel throughput+occupancy layout already is that content, and this
    campaign's own deck-v2 theme (`_DECK_SERIES`/`_DECK_NAVY`) is what every
    other figure in this deck uses, so re-theming to fig4b/fig5's unrelated
    `common.py` palette would break consistency with the rest of THIS deck.
    The one design idea borrowed from fig4b is real and cheap to keep
    honest: annotating the realised `num_buckets` under each throughput
    point, the same "B={buckets}" convention fig4b uses under its bottom
    panel. No cache-crossing vertical bands were added (fig4b/fig5 both
    have them): this campaign's host is AMD Genoa (EPYC 9474F), and
    `research/HARDWARE.md` has no measured L2/LLC size for that host, only
    for `ccqlin038` -- adding a band would mean an unmeasured spec number,
    which this repo's own convention (measured facts over spec, see
    `research/HARDWARE.md`'s DDR4 spec-vs-measured note) argues against.
    """
    if not rows:
        raise NotImplementedError(
            "make_bucket_size_figure: no rows to plot -- run the "
            "bucketsize-tbl{...}-probe cell for at least one target_bucket_len."
        )

    import matplotlib.pyplot as plt

    deck = theme == "deck"
    if single_panel:
        figsize = (figsize_pt[0] / 72.0, figsize_pt[1] / 72.0) if (deck and figsize_pt) else (5.5, 4)
    else:
        figsize = (figsize_pt[0] / 72.0, figsize_pt[1] / 72.0) if (deck and figsize_pt) else (9.5, 4)

    def _build():
        if single_panel:
            fig, ax_thr = plt.subplots(figsize=figsize)
            ax_occ = None
        else:
            fig, (ax_thr, ax_occ) = plt.subplots(1, 2, figsize=figsize)

        pts = sorted(rows, key=lambda r: r["target_bucket_len"])
        xs = [r["target_bucket_len"] for r in pts]

        # --- left/only panel: throughput ---------------------------------
        ys_thr = [r["strings_per_s"] for r in pts]
        if deck:
            st = _DECK_SERIES[0]
            ax_thr.plot(
                xs, ys_thr, marker=st["marker"], markersize=6, linewidth=1.8,
                linestyle=st["linestyle"], color=st["color"],
            )
        else:
            ax_thr.plot(xs, ys_thr, marker="o", markersize=5, linewidth=1.5, color=_ACCENT)
        ax_thr.set_xscale("log", base=2)
        ax_thr.set_xlabel("target bucket size (terms)")
        ax_thr.set_ylabel("strings/s")

        if l2_cache_bytes is not None:
            critical_x = l2_cache_bytes / bytes_per_term
            line_color = _DECK_NAVY if deck else "#898781"
            ax_thr.axvline(critical_x, color=line_color, linewidth=1.2, linestyle=(0, (4, 2)))
            ax_thr.annotate(
                f"L2 ({l2_cache_bytes / 1024:.0f} KiB)",
                (critical_x, 1.0), xycoords=ax_thr.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7 if not deck else _DECK_FONT_PT * 0.6,
                color=line_color,
            )

        # Realised bucket count under each point -- borrowed from the older
        # `presentation` deck's fig4b_bucket_speedup.py, which annotates
        # "B={buckets}" the same way; it lets the audience read off that
        # `num_buckets` is itself derived (n / target_bucket_len, floored at
        # `min_buckets`) rather than an independent axis. `get_xaxis_transform`
        # keeps the label at a fixed vertical fraction regardless of the
        # strings/s scale.
        label_color = _DECK_NAVY if deck else "#898781"
        for r in pts:
            ax_thr.annotate(
                f"B={r['num_buckets']}",
                (r["target_bucket_len"], 0.04),
                xycoords=ax_thr.get_xaxis_transform(),
                ha="center", va="bottom",
                fontsize=7 if not deck else _DECK_FONT_PT * 0.55,
                color=label_color,
            )

        # --- right panel: occupancy distribution + empty-bucket fraction --
        if ax_occ is None:
            if deck:
                if title:
                    ax_thr.set_title(title, fontsize=_DECK_FONT_PT, color=_DECK_NAVY)
                _style_axes_deck(ax_thr)
            else:
                ax_thr.set_title("Throughput vs. target bucket size")
                _style_axes(ax_thr)
            fig.tight_layout()
            return fig

        medians = [r["occupancy_median"] for r in pts]
        p95s = [r["occupancy_p95"] for r in pts]
        maxs = [r["occupancy_max"] for r in pts]
        empty_frac = [r["empty_buckets"] / r["num_buckets"] for r in pts]

        if deck:
            median_style = _DECK_SERIES[0]
            p95_style = _DECK_SERIES[1]
            max_style = _DECK_SERIES[2]
        else:
            median_style = {"color": _ACCENT, "marker": "o", "linestyle": "-"}
            p95_style = {"color": "#eb6834", "marker": "^", "linestyle": "--"}
            max_style = {"color": "#5a8f3c", "marker": "s", "linestyle": ":"}

        ax_occ.plot(xs, medians, marker=median_style["marker"], markersize=6, linewidth=1.8,
                    linestyle=median_style["linestyle"], color=median_style["color"], label="median")
        ax_occ.plot(xs, p95s, marker=p95_style["marker"], markersize=6, linewidth=1.8,
                    linestyle=p95_style["linestyle"], color=p95_style["color"], label="p95")
        ax_occ.plot(xs, maxs, marker=max_style["marker"], markersize=6, linewidth=1.8,
                    linestyle=max_style["linestyle"], color=max_style["color"], label="max")
        ax_occ.set_xscale("log", base=2)
        ax_occ.set_xlabel("target bucket size (terms)")
        ax_occ.set_ylabel("occupied strings per non-empty bucket")

        # Empty-bucket fraction on its own twin axis, as thin bars -- a
        # distinct visual channel from the occupancy lines so a reader sees
        # both "how full are non-empty buckets" and "what fraction are
        # empty" without one number diluting the other.
        ax_empty = ax_occ.twinx()
        bar_color = _DECK_NAVY if deck else "#898781"
        # Bar width in log-x data units: a fixed fraction of each point's own x.
        widths = [x * 0.12 for x in xs]
        ax_empty.bar(xs, empty_frac, width=widths, color=bar_color, alpha=0.25, zorder=1, label="empty fraction")
        ax_empty.set_ylim(0, 1)
        ax_empty.set_ylabel("empty-bucket fraction")
        if deck:
            ax_empty.tick_params(colors=_DECK_NAVY)
            ax_empty.spines["right"].set_color(_DECK_NAVY)
            for label in ax_empty.get_yticklabels():
                label.set_color(_DECK_NAVY)
        else:
            ax_empty.tick_params(colors="#898781")

        lines, labels = ax_occ.get_legend_handles_labels()
        bars, bar_labels = ax_empty.get_legend_handles_labels()
        ax_occ.legend(lines + bars, labels + bar_labels, frameon=False, fontsize=8 if not deck else _DECK_FONT_PT * 0.7)

        if deck:
            if title:
                fig.suptitle(title, fontsize=_DECK_FONT_PT, color=_DECK_NAVY)
            _style_axes_deck(ax_thr)
            _style_axes_deck(ax_occ)
        else:
            ax_thr.set_title("Throughput vs. target bucket size")
            ax_occ.set_title("Occupancy distribution vs. target bucket size")
            _style_axes(ax_thr)
            _style_axes(ax_occ)
        fig.tight_layout()
        return fig

    if deck:
        import matplotlib as mpl

        with mpl.rc_context(_deck_rc_params()):
            fig = _build()
    else:
        fig = _build()
    return fig


def make_baseline_pivot_figure(
    rows: Sequence[dict],
    *,
    theme: str = "legacy",
    figsize_pt: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Three real wall-clock points marking "before this work" (deck page 16).

    Supersedes the `naive_baseline`-tolerance-sweep framing of the old
    `baseline_v2` figure (`make_recurring_figure(..., highlight_variant=
    "naive_baseline")`): `decisions.md` #42 found that commit's merge phase
    never wires truncation in at all, so every point of a sweep comes back
    with the identical `final_terms` -- it cannot show a tolerance sweep, and
    is truncation-inert rather than a real "before" baseline. The three
    points the user asked for instead: PauliPropagation.jl single-thread,
    PauliPropagation.jl 96-thread, and THIS engine's own single-bucket point
    (`min_buckets=1`, forced back to the pre-bucketing regime) -- all at the
    SAME `eps=2^-16`, 127-qubit canonical config.

    Each row: `{"label": str, "wall_time_s": float, "threads": int, "engine":
    "julia" | "current_engine", "final_terms": int | None}`.

    This is a plain 3-point bar chart, deliberately NOT a scaling curve: the
    x-axis is categorical (`label`), never a numeric thread axis, and no line
    connects the three bars. The two Julia points ARE a directly comparable
    pair (same engine, same code, 1 vs. 96 threads); the current-engine point
    is a DIFFERENT implementation, back at 1 thread again -- 1 -> 96 -> 1 is
    not a monotonic thread progression, and drawing it as one continuous
    series would misleadingly imply otherwise. Julia's two bars share one
    color; the current-engine bar gets both a distinct color AND a hatch
    pattern, so "this one is not part of the Julia pair" survives grayscale
    or color-blind viewing, not just a color difference.
    """
    if not rows:
        raise NotImplementedError(
            "make_baseline_pivot_figure: no rows to plot -- need the three "
            "real wall-clock points (Julia 1-thread, Julia 96-thread, "
            "current-engine 1-bucket) before this figure has anything to show."
        )

    import matplotlib.pyplot as plt

    deck = theme == "deck"
    figsize = (figsize_pt[0] / 72.0, figsize_pt[1] / 72.0) if (deck and figsize_pt) else (6, 4.2)

    def _build():
        fig, ax = plt.subplots(figsize=figsize)

        walls = [r["wall_time_s"] for r in rows]
        xs = list(range(len(rows)))

        julia_color = _DECK_SERIES[1]["color"] if deck else "#eb6834"
        engine_color = _DECK_SERIES[0]["color"] if deck else _ACCENT
        edge_color = _DECK_NAVY if deck else "#333333"
        colors = [engine_color if r.get("engine") == "current_engine" else julia_color for r in rows]

        bars = ax.bar(xs, walls, color=colors, width=0.55, edgecolor=edge_color, linewidth=0.8)
        hatch_color = "#ffffff" if deck else "#f5f4ef"
        for bar, r in zip(bars, rows):
            if r.get("engine") == "current_engine":
                bar.set_hatch("///")
                # matplotlib draws hatch lines in the patch's edgecolor, which
                # is the same navy as this bar's own outline -- invisible
                # against a navy fill without an explicit lighter override.
                bar.set_edgecolor(hatch_color)
                bar.set_linewidth(1.2)

        text_color = _DECK_NAVY if deck else "#333333"
        muted = _DECK_GRID if deck else "#898781"
        for xi, r in zip(xs, rows):
            ax.annotate(
                f"{r['wall_time_s']:.1f} s",
                (xi, r["wall_time_s"]), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9 if not deck else _DECK_FONT_PT * 0.65,
                color=text_color,
            )
            thread_label = f"{r['threads']} thread{'s' if r['threads'] != 1 else ''}"
            ax.annotate(
                thread_label,
                (xi, 0.02), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=8 if not deck else _DECK_FONT_PT * 0.55,
                color=muted,
            )

        ax.set_xticks(xs)
        ax.set_xticklabels([r["label"] for r in rows], fontsize=9 if not deck else _DECK_FONT_PT * 0.7)
        ax.set_ylabel("wall time (s)")
        ax.set_ylim(0, max(walls) * 1.22)

        # Julia's own 1-thread -> 96-thread speedup, stated once as an explicit
        # number rather than left for the reader to compute from the two bar
        # heights -- only drawn when both Julia points are actually present.
        julia_idx = [i for i, r in enumerate(rows) if r.get("engine") != "current_engine"]
        if len(julia_idx) >= 2:
            i0, i1 = julia_idx[0], julia_idx[-1]
            if rows[i1]["wall_time_s"] > 0:
                speedup = rows[i0]["wall_time_s"] / rows[i1]["wall_time_s"]
                ax.annotate(
                    f"{speedup:.2f}x",
                    ((i0 + i1) / 2, max(rows[i0]["wall_time_s"], rows[i1]["wall_time_s"]) * 1.10),
                    ha="center", va="bottom", fontsize=9 if not deck else _DECK_FONT_PT * 0.6,
                    color=text_color, fontweight="bold",
                )

        if deck:
            if title:
                ax.set_title(title, fontsize=_DECK_FONT_PT, color=_DECK_NAVY)
            _style_axes_deck(ax)
        else:
            ax.set_title("SYNTHETIC — placeholder: baseline pivot")
            _style_axes(ax)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        return fig

    if deck:
        import matplotlib as mpl

        with mpl.rc_context(_deck_rc_params()):
            fig = _build()
    else:
        fig = _build()
    return fig


def make_baseline_eps_scaling_figure(
    rows: Sequence[dict],
    *,
    series_order: list[str] | None = None,
    theme: str = "legacy",
    figsize_pt: tuple[float, float] | None = None,
    title: str | None = None,
    speedup_baseline: str | None = None,
):
    """One line per named series: wall time (y, log) vs. `min_abs_coeff` (x, log2-spaced), the
    eps-sweep redesign of the deck page 16 baseline story.

    `speedup_baseline`, when given (a label present in `rows`), adds a second panel: speedup
    relative to that label AT THE SAME eps -- `wall_time_s[speedup_baseline][eps] /
    wall_time_s[label][eps]` -- for every drawn series, including the baseline itself (a flat
    line at 1.0, a visible sanity check). This is a per-eps ratio, not a single fixed baseline
    value, since the baseline's own wall time varies across the eps grid. Requires the baseline
    label to have a row at every eps value any drawn series has one at, else that series/eps
    point is silently skipped (no fabricated ratio from a missing denominator) -- callers should
    ensure the baseline series is complete across the grid before relying on this panel.

    Supersedes `make_baseline_pivot_figure` (`baseline_v3`, a single-eps 3-bar chart) as page
    16's figure -- MANIFEST marks `baseline_v3` superseded, not deleted, same as this campaign's
    other supersessions (`bucketed_1t_v2` -> `bucketed_1t_v3`). Each row: `{"label": str,
    "min_abs_coeff": float, "wall_time_s": float}`.

    Progressive reveal: `series_order` picks which labels are DRAWN, and in what legend order;
    `None` draws every label present in `rows`. Unlike `make_recurring_figure`'s `stage=`, which
    indexes a hardcoded `STAGE_VARIANTS` list, this function has no baked-in knowledge of which
    labels exist -- the caller passes a growing prefix of the full label list across successive
    calls as more series land. Two things are deliberately keyed off `rows` (the full label set),
    NEVER off `series_order` (the subset drawn), so the SAME rows produce the SAME axes/style at
    every stage and only the drawn lines change:
      - each label's color/marker/linestyle, fixed by that label's first-appearance position in
        `rows` -- a label plotted alone at stage 1 keeps the exact same look once stage 2 reveals
        a second line next to it;
      - the x/y axis limits, taken from every row in `rows` regardless of `series_order` -- so a
        caller who always passes the full, current `rows` (trimming only `series_order`) gets a
        deck build where the axes never jump between reveals, only new lines appear on them.
    A future stage's caller is expected to pass `rows` containing that stage's new series too;
    this function does not know or care how many stages there will eventually be.
    """
    if not rows:
        raise NotImplementedError(
            "make_baseline_eps_scaling_figure: no rows to plot -- need at least one "
            "(label, min_abs_coeff, wall_time_s) series before this figure has anything to show."
        )

    import math

    import matplotlib.pyplot as plt

    deck = theme == "deck"
    figsize = (figsize_pt[0] / 72.0, figsize_pt[1] / 72.0) if (deck and figsize_pt) else (5.5, 4)

    # First-appearance order in `rows`, never `series_order` -- see docstring: this is what
    # keeps a label's color/marker stable across a growing `series_order` prefix.
    all_labels: list[str] = []
    for r in rows:
        if r["label"] not in all_labels:
            all_labels.append(r["label"])

    if series_order is None:
        draw_labels = list(all_labels)
    else:
        unknown = [label for label in series_order if label not in all_labels]
        if unknown:
            raise ValueError(
                f"make_baseline_eps_scaling_figure: series_order names label(s) not present in "
                f"rows: {unknown}"
            )
        draw_labels = list(series_order)

    if speedup_baseline is not None and speedup_baseline not in all_labels:
        raise ValueError(
            f"make_baseline_eps_scaling_figure: speedup_baseline={speedup_baseline!r} is not a "
            f"label present in rows: {all_labels}"
        )

    def _eps_label(eps: float) -> str:
        return rf"$\varepsilon=2^{{{round(math.log2(eps))}}}$" if eps > 0 else r"$\varepsilon=0$"

    legacy_palette = [_ACCENT, "#eb6834", "#5a8f3c", "#a15fb5", "#5b5f97"]

    def _build():
        if speedup_baseline is not None:
            fig, (ax, ax_speedup) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax_speedup = None

        style_map = {label: _DECK_SERIES[i % len(_DECK_SERIES)] for i, label in enumerate(all_labels)}
        legacy_color = {label: legacy_palette[i % len(legacy_palette)] for i, label in enumerate(all_labels)}

        by_label: dict[str, list[dict]] = {}
        for r in rows:
            by_label.setdefault(r["label"], []).append(r)

        for label in draw_labels:
            pts = sorted((r["min_abs_coeff"], r["wall_time_s"]) for r in by_label[label])
            if not pts:
                continue
            xs, ys = zip(*pts)
            if deck:
                st = style_map[label]
                ax.plot(xs, ys, marker=st["marker"], markersize=6, linewidth=1.8,
                         linestyle=st["linestyle"], color=st["color"], label=label)
            else:
                ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, color=legacy_color[label], label=label)

        # Axis range from every row in `rows`, not just `draw_labels` -- see docstring: this is
        # what keeps the axes from jumping when a later stage's `series_order` grows.
        all_eps = [r["min_abs_coeff"] for r in rows]
        all_wall = [r["wall_time_s"] for r in rows]
        ax.set_xlim(min(all_eps) / 1.6, max(all_eps) * 1.6)
        ax.set_ylim(min(all_wall) / 1.6, max(all_wall) * 1.6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        # Dyadic cutoffs get real mathtext ticks (the established $\varepsilon=2^{-16}$
        # convention, same `_eps_label` shape `make_convergence_figure` uses for its legend)
        # rather than plain log-scale number ticks -- there are only ever a handful of distinct
        # eps values actually measured, so labeling exactly those points is more legible than a
        # dense automatic log grid.
        #
        # Ticks come from `draw_labels` only (which series are ACTUALLY DRAWN this stage), not
        # every row in `rows` -- per user request: a tighter eps like 2^-18/2^-20 should only
        # get a tick once some drawn series actually reaches it, not merely because a later
        # stage's series will. The x-AXIS LIMITS stay keyed off the full `rows` (see above),
        # so the framing itself still doesn't jump between reveals -- only which positions get
        # a labeled tick changes.
        eps_ticks = sorted({r["min_abs_coeff"] for label in draw_labels for r in by_label[label]})
        ax.set_xticks(eps_ticks)
        ax.set_xticklabels([_eps_label(e) for e in eps_ticks], fontsize=7 if not deck else _DECK_FONT_PT * 0.6)
        ax.minorticks_off()

        ax.set_xlabel(r"$\varepsilon$ (min_abs_coeff)")
        ax.set_ylabel("wall time (s)")
        if deck:
            if title:
                ax.set_title(title, fontsize=_DECK_FONT_PT, color=_DECK_NAVY)
            _style_axes_deck(ax)
        else:
            ax.set_title("SYNTHETIC — placeholder: baseline eps scaling")
            _style_axes(ax)

        if ax_speedup is not None:
            baseline_by_eps = {r["min_abs_coeff"]: r["wall_time_s"] for r in by_label[speedup_baseline]}
            for label in draw_labels:
                pts = []
                for r in by_label[label]:
                    eps = r["min_abs_coeff"]
                    if eps in baseline_by_eps:
                        pts.append((eps, baseline_by_eps[eps] / r["wall_time_s"]))
                if not pts:
                    continue
                pts.sort()
                xs, ys = zip(*pts)
                if deck:
                    st = style_map[label]
                    ax_speedup.plot(xs, ys, marker=st["marker"], markersize=6, linewidth=1.8,
                                     linestyle=st["linestyle"], color=st["color"])
                else:
                    ax_speedup.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, color=legacy_color[label])

            ax_speedup.axhline(1.0, color=(_DECK_NAVY if deck else "#898781"), linewidth=0.8, linestyle=":")
            ax_speedup.set_xlim(ax.get_xlim())
            ax_speedup.set_xscale("log", base=2)
            # Linear, not log, per explicit user request (unlike the left panel's wall time,
            # which stays log -- that choice is unchanged).
            ax_speedup.set_xticks(eps_ticks)
            ax_speedup.set_xticklabels([_eps_label(e) for e in eps_ticks], fontsize=7 if not deck else _DECK_FONT_PT * 0.6)
            ax_speedup.minorticks_off()
            ax_speedup.set_xlabel(r"$\varepsilon$ (min_abs_coeff)")
            # A rotated y-label as long as the full baseline name (e.g. "speedup vs. current
            # engine, 1 bucket") ran past the top of the canvas on an actual render at this
            # figure's height -- real clipping, not theoretical. Just "speedup" keeps the axis
            # legible; which baseline it's relative to is stated in the figure's own title/
            # caption and in MANIFEST.md, not silently dropped from the deliverable.
            ax_speedup.set_ylabel("speedup")
            if deck:
                _style_axes_deck(ax_speedup)
            else:
                ax_speedup.set_title("SYNTHETIC — placeholder: speedup")
                _style_axes(ax_speedup)

        if draw_labels:
            # A `loc="best"` legend sits INSIDE the axes and, with 4 long labels, can overlap
            # the plotted lines rather than truly clip -- caught on an actual rendered PNG at
            # the 450x340 compact size, not a theoretical concern. Below the axes, spanning the
            # full canvas width, is the same layout-independent fix `make_distributed_capacity_
            # figure` uses: one row if the real rendered width fits, else one entry per row.
            #
            # A FIXED bottom-margin fraction (this module's other below-axis legends use one)
            # turned out not to generalize across this figure's own size range -- caught on an
            # actual render too: 0.20 + 0.10*n_rows collided with the x-axis label at 900x170,
            # and starved the plot area at 450x340 once 4 rows were needed. Measuring the
            # legend's real rendered height and adding it on top of whatever bottom margin
            # `tight_layout()` already reserved for the x-axis label/ticks (rather than
            # guessing both from a row count) is what actually holds at every size tried.
            fontsize = 7 if not deck else _DECK_FONT_PT * 0.6
            fig.tight_layout()
            base_bottom = fig.subplotpars.bottom
            # fig.legend() (figure coordinates), not ax.legend() (axes coordinates): with a
            # second panel (`speedup_baseline` given), an axes-relative legend centers under
            # only the LEFT panel, not the whole figure -- real problem, only visible once a
            # second panel exists, so this must be figure-relative unconditionally.
            handles, labels = ax.get_legend_handles_labels()
            # Jumping straight from "all in one row" to "exactly one column" (as an earlier
            # version of this code did) makes a 7-entry legend seven rows tall -- real problem
            # hit at 7 series: it overflowed the bottom-margin cap and overlapped the x-axis
            # label on an actual rendered figure. Searching downward from ncol=len(labels) for
            # the widest column count that actually fits keeps the legend far shorter.
            ncol = len(draw_labels)
            legend = None
            while ncol >= 1:
                if legend is not None:
                    legend.remove()
                legend = fig.legend(
                    handles, labels, frameon=False, fontsize=fontsize, loc="lower center",
                    bbox_to_anchor=(0.5, 0.0), borderaxespad=0.0, ncol=ncol,
                )
                fig.canvas.draw()
                fig_bbox = fig.get_window_extent()
                if legend.get_window_extent().width <= fig_bbox.width or ncol == 1:
                    break
                ncol -= 1
            legend_height_frac = legend.get_window_extent().height / fig_bbox.height
            # No hard cap: a legend that genuinely needs more room gets it, rather than being
            # silently clipped by an arbitrary ceiling (the earlier 0.85 cap's real failure mode).
            fig.subplots_adjust(bottom=min(base_bottom + legend_height_frac + 0.03, 0.97))
            return fig
        fig.tight_layout()
        return fig

    if deck:
        import matplotlib as mpl

        with mpl.rc_context(_deck_rc_params()):
            fig = _build()
    else:
        fig = _build()
    return fig


def make_single_bucket_comparison_figure(
    rows: Sequence[dict],
    *,
    theme: str = "legacy",
    figsize_pt: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Two-bar comparison: current engine, single-thread, default bucket
    config vs. forced to a single bucket (deck page 29).

    Supersedes the dropped `bucketed_1t_v2` figure (`decisions.md` #46):
    that figure compared `bucketed_engine_serial` vs. `bucketed_engine_
    parallel`, two DIFFERENT historical commits, which never supported page
    29's actual claim. The real claim is narrower and entirely within the
    CURRENT engine, single-threaded: does bucket-splitting itself help,
    independent of threading? Both rows here are the same engine, same
    commit, same thread count (1), same `eps=2^-16`/127-qubit config -- only
    `min_buckets`/`target_bucket_len` differ.

    Each row: `{"label": str, "wall_time_s": float, "final_terms": int |
    None, "expectation_re": float | None}`. Exactly two rows are expected
    (default config, forced single bucket); this function does not refuse a
    different count, but its "N% slower" annotation only draws when there
    are at least two.
    """
    if not rows:
        raise NotImplementedError(
            "make_single_bucket_comparison_figure: no rows to plot -- need "
            "the default-bucket and single-bucket wall-clock points (same "
            "config, same thread count) before this figure has anything to "
            "show."
        )

    import matplotlib.pyplot as plt

    deck = theme == "deck"
    figsize = (figsize_pt[0] / 72.0, figsize_pt[1] / 72.0) if (deck and figsize_pt) else (5, 4.2)

    def _build():
        fig, ax = plt.subplots(figsize=figsize)

        walls = [r["wall_time_s"] for r in rows]
        xs = list(range(len(rows)))
        palette = [_DECK_SERIES[0]["color"], _DECK_SERIES[1]["color"]] if deck else [_ACCENT, "#eb6834"]
        colors = [palette[i % len(palette)] for i in xs]
        edge_color = _DECK_NAVY if deck else "#333333"

        ax.bar(xs, walls, color=colors, width=0.5, edgecolor=edge_color, linewidth=0.8)

        text_color = _DECK_NAVY if deck else "#333333"
        muted = _DECK_GRID if deck else "#898781"
        for xi, r in zip(xs, rows):
            ax.annotate(
                f"{r['wall_time_s']:.1f} s", (xi, r["wall_time_s"]),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                fontsize=9 if not deck else _DECK_FONT_PT * 0.65, color=text_color,
            )

        if len(rows) >= 2 and rows[0]["wall_time_s"] > 0:
            pct = 100.0 * (rows[1]["wall_time_s"] / rows[0]["wall_time_s"] - 1.0)
            ax.annotate(
                f"{pct:+.1f}%",
                (0.5, max(walls) * 1.16),
                xycoords=("axes fraction", "data"), ha="center", va="bottom",
                fontsize=10 if not deck else _DECK_FONT_PT * 0.7, color=text_color, fontweight="bold",
            )

        ax.set_xticks(xs)
        ax.set_xticklabels([r["label"] for r in rows], fontsize=9 if not deck else _DECK_FONT_PT * 0.7)
        ax.set_ylabel("wall time (s)")
        ax.set_ylim(0, max(walls) * 1.3)

        # Same-correctness note: identical final_terms across both bars means
        # this is a pure wall-clock effect, never a correctness difference.
        terms = {r["final_terms"] for r in rows if r.get("final_terms") is not None}
        if len(terms) == 1:
            (n,) = terms
            ax.annotate(
                f"final_terms identical: {n:,}",
                (0.5, -0.20), xycoords="axes fraction", ha="center", va="top",
                fontsize=7 if not deck else _DECK_FONT_PT * 0.5, color=muted,
            )

        if deck:
            if title:
                ax.set_title(title, fontsize=_DECK_FONT_PT, color=_DECK_NAVY)
            _style_axes_deck(ax)
        else:
            ax.set_title("SYNTHETIC — placeholder: bucket config comparison")
            _style_axes(ax)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        if len(terms) == 1:
            fig.subplots_adjust(bottom=0.24)
        return fig

    if deck:
        import matplotlib as mpl

        with mpl.rc_context(_deck_rc_params()):
            fig = _build()
    else:
        fig = _build()
    return fig
