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
        # log2(eps) for a compact, campaign-native label (every cutoff here is dyadic).
        return f"eps=2^{round(math.log2(eps))}" if eps > 0 else "eps=0"

    def _build():
        fig, ax = plt.subplots(figsize=figsize)

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

        ax.set_xlabel("Trotter step")
        ax.set_ylabel("<O>")
        if deck:
            if title:
                ax.set_title(title, fontsize=_DECK_FONT_PT, color=_DECK_NAVY)
            _style_axes_deck(ax)
        else:
            ax.set_title("Observable trajectory vs. truncation cutoff")
            _style_axes(ax)
        ax.legend(frameon=False, fontsize=8 if not deck else _DECK_FONT_PT * 0.75)
        fig.tight_layout()
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

    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms

    deck = theme == "deck"
    figsize = (figsize_pt[0] / 72.0, figsize_pt[1] / 72.0) if (deck and figsize_pt) else (6.5, 4.5)
    _STATUS_MARKER = {"oom": "X", "timeout": "P", "untested": "$?$"}

    def _label(eps: float) -> str:
        return f"eps=2^{round(math.log2(eps))}" if eps > 0 else "eps=0"

    def _build():
        fig, ax = plt.subplots(figsize=figsize)
        blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

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
                ax.plot(
                    xs, ys, marker=style["marker"], markersize=7, linewidth=1.8,
                    linestyle=style["linestyle"] if len(xs) > 1 else "none",
                    color=style["color"], label=_label(eps),
                )
                for r in completed:
                    parts = []
                    if r.get("peak_terms"):
                        parts.append(f"N={r['peak_terms']:,}")
                    if r.get("peak_rss_kb"):
                        parts.append(f"{r['peak_rss_kb'] / 1e9:.2f} TB")
                    if r.get("note"):
                        parts.append(r["note"])
                    if parts:
                        ax.annotate(
                            "\n".join(parts), (r["ranks"], r["wall_time_s"]), fontsize=7,
                            color=annotation_color, xytext=(6, 6), textcoords="offset points",
                        )

            for r in series_rows:
                if r["status"] == "completed":
                    continue
                any_plotted = True
                marker = _STATUS_MARKER.get(r["status"], "X")
                ax.plot(
                    [r["ranks"]], [0.95], marker=marker, markersize=10,
                    color=style["color"], linestyle="none", transform=blended,
                    label=f"{_label(eps)} ({r['status']})",
                )
                if r.get("note"):
                    ax.annotate(
                        r["note"], (r["ranks"], 0.95), xycoords=blended, fontsize=7,
                        color=annotation_color, xytext=(6, -12), textcoords="offset points",
                    )

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
            ax.legend(
                dedup_handles, dedup_labels, frameon=False,
                fontsize=7 if not deck else _DECK_FONT_PT * 0.7, loc="lower right",
            )
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
):
    """Two panels vs. `target_bucket_len` (x, log2-spaced, one fixed cell per point):
    throughput (left) and occupancy distribution + empty-bucket fraction (right).

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
    figsize = (figsize_pt[0] / 72.0, figsize_pt[1] / 72.0) if (deck and figsize_pt) else (9.5, 4)

    def _build():
        fig, (ax_thr, ax_occ) = plt.subplots(1, 2, figsize=figsize)

        pts = sorted(rows, key=lambda r: r["target_bucket_len"])
        xs = [r["target_bucket_len"] for r in pts]

        # --- left panel: throughput -------------------------------------
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
        ax_thr.set_xlabel("target_bucket_len")
        ax_thr.set_ylabel("strings/s")

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
        ax_occ.set_xlabel("target_bucket_len")
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
