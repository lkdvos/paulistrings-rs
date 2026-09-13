"""The recurring two-panel talk figure: efficiency (left) vs. calculation cost (right).

Every figure this module draws is a placeholder over synthetic data until real
`runs.jsonl`/`gates.rank-N.jsonl` exist (T07) and flow through T06's `normalize.py`.
`make_recurring_figure` never touches a file; callers pass already-normalized rows.

Row shapes and the join this module assumes
---------------------------------------------

`tolerance_rows` are exactly `normalize.runtime_tolerance()`'s output: each row
already carries `variant_id`, `min_abs_coeff`, `wall_time_s`, `peak_terms`,
`peak_rss_kb`, `status`.

`efficiency_rows` are `normalize.efficiency_gates()` or `efficiency_binned()`
output, *enriched* with `variant_id` (and optionally `threads`, `ranks`) joined
in from the owning run record by `run_id` -- neither normalize function emits
that field itself (a gate/bin row has no notion of which run produced it beyond
`run_id`), so the join is this module's caller's job, not normalize.py's.
An `efficiency_gates` row's x is `terms_in`; an `efficiency_binned` row's x is
`bin_low`. Both carry `rate` and `gate_name`.

Stage -> variant mapping
------------------------

The seven stages are the seven benchmarked variants in
`contract.md`'s variant registry (the eighth, `attempted_rejected_variants`, is
narrative-only and never plotted), in the pedagogical order the talk reveals
them -- not their chronological order. `direct_small_sum_path` has no
memory-specific role tag in the registry; it is placed at stage 4
("memory-annotated") by elimination against the other six exact-match role
strings, since it is the one variant slot left once the other six are pinned
to their registry roles. Documented here rather than left implicit.
"""

from __future__ import annotations

from typing import Any, Sequence

STAGE_VARIANTS = [
    "naive_baseline",  # stage 1: baseline + external refs, contract.md "E0 internal baseline"
    "jcc_erratum_and_branch_prediction",  # stage 2: kernel improvement, "E1 kernel-improvement candidate"
    "presentation_bench_crate_variants",  # stage 3: historical threading attempt, "threadmaps ... E1/E2"
    "direct_small_sum_path",  # stage 4: memory-annotated (elimination, see module docstring)
    "bucketed_engine_serial",  # stage 5: bucketed single-thread, "GF(2) bucketing, serial -- E4"
    "bucketed_engine_parallel",  # stage 6: bucketed multi-thread, "Rayon parallel -- E5"
    "partitioned_numa_engine",  # stage 7: distributed + hash, "NUMA + MPI -- E6/E7/E8"
]

_NON_COMPLETED_MARKER = "X"
_MUTED_ALPHA = 0.35

_PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]


def _color_for_variant(variant_id: str) -> str:
    """Stable palette slot keyed by position in `STAGE_VARIANTS`, not first-seen order.

    Unlike `examples/common/report.py`'s `_color_for_engine` cache, this stays stable
    across separate figure calls (different stages) without a shared mutable cache.
    """
    try:
        idx = STAGE_VARIANTS.index(variant_id)
    except ValueError:
        idx = hash(variant_id) % len(_PALETTE)
    return _PALETTE[idx % len(_PALETTE)]


def _style_axes(ax) -> None:
    ax.grid(True, color="#e1e0d9", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#898781")
    ax.tick_params(colors="#898781")


def _group_by(rows: Sequence[dict], key: str) -> dict[Any, list[dict]]:
    grouped: dict[Any, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def _draw_efficiency_panel(ax, efficiency_rows, visible_variants, highlight_variant) -> None:
    grouped = _group_by(efficiency_rows, "variant_id")
    highlighted_points = None
    for variant_id in visible_variants:
        rows = grouped.get(variant_id)
        if not rows:
            continue
        points = []
        for row in rows:
            x = row["terms_in"] if "terms_in" in row else row.get("bin_low")
            y = row.get("rate")
            if x is None or y is None:
                continue
            points.append((x, y))
        if not points:
            continue
        points.sort()
        is_highlight = variant_id == highlight_variant
        color = _color_for_variant(variant_id)
        alpha = 1.0 if is_highlight else _MUTED_ALPHA
        zorder = 3 if is_highlight else 1
        xs, ys = zip(*points)
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=5,
            linewidth=1.5,
            color=color,
            alpha=alpha,
            zorder=zorder,
            label=variant_id,
        )
        if is_highlight:
            highlighted_points = rows

    if highlighted_points:
        sample = highlighted_points[0]
        gate_name = sample.get("gate_name", "?")
        threads = sample.get("threads")
        ranks = sample.get("ranks")
        annotation = f"gate={gate_name}"
        if threads is not None:
            annotation += f", threads={threads}"
        if ranks is not None:
            annotation += f", ranks={ranks}"
        ax.text(
            0.02,
            0.02,
            annotation,
            transform=ax.transAxes,
            fontsize=8,
            color="#898781",
            va="bottom",
            ha="left",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("strings entering gate")
    ax.set_ylabel("input-string updates / sec")
    ax.set_title("Efficiency")
    _style_axes(ax)
    if grouped:
        ax.legend(frameon=False, fontsize=7)


def _draw_cost_panel(ax, tolerance_rows, visible_variants, highlight_variant, stage) -> None:
    import matplotlib.transforms as mtransforms

    grouped = _group_by(tolerance_rows, "variant_id")
    any_series = False
    any_oom = False
    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    for variant_id in visible_variants:
        rows = grouped.get(variant_id)
        if not rows:
            continue
        is_highlight = variant_id == highlight_variant
        color = _color_for_variant(variant_id)
        alpha = 1.0 if is_highlight else _MUTED_ALPHA
        zorder = 3 if is_highlight else 1

        completed = sorted(
            (r["min_abs_coeff"], r["wall_time_s"])
            for r in rows
            if r["status"] == "completed" and r["wall_time_s"] is not None
        )
        if completed:
            any_series = True
            xs, ys = zip(*completed)
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=5,
                linewidth=1.5,
                color=color,
                alpha=alpha,
                zorder=zorder,
                label=variant_id,
            )
            if is_highlight:
                for x, y, r in zip(xs, ys, [r for r in rows if r["status"] == "completed"]):
                    if x in (xs[0], xs[-1]):
                        note = f"N={r.get('peak_terms')}"
                        if r.get("peak_rss_kb") is not None:
                            note += f"\n{r['peak_rss_kb'] / 1024:.0f} MB"
                        ax.annotate(
                            note,
                            (x, y),
                            fontsize=7,
                            color="#898781",
                            xytext=(0, 8),
                            textcoords="offset points",
                            ha="center",
                        )

        non_completed = [r for r in rows if r["status"] != "completed"]
        for r in non_completed:
            any_oom = True
            ax.plot(
                r["min_abs_coeff"],
                0.95,
                marker=_NON_COMPLETED_MARKER,
                markersize=7,
                color=color,
                alpha=alpha,
                zorder=zorder + 1,
                transform=blended,
                linestyle="none",
                label=f"{variant_id} ({r['status']})",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("coefficient tolerance")
    ax.set_ylabel("wall time (s)")
    ax.set_title("Calculation cost")
    _style_axes(ax)
    ax.invert_xaxis()

    if stage == 7:
        xlim = ax.get_xlim()
        # Widen visibly to show the distributed extension reaching tighter tolerances.
        ax.set_xlim(xlim[0], xlim[1] / 100)

    if any_series or any_oom:
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        dedup_handles, dedup_labels = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            dedup_handles.append(h)
            dedup_labels.append(l)
        ax.legend(dedup_handles, dedup_labels, frameon=False, fontsize=7)


def make_recurring_figure(
    efficiency_rows: Sequence[dict],
    tolerance_rows: Sequence[dict],
    *,
    highlight_variant: str,
    stage: int,
):
    """Build the recurring two-panel figure (Efficiency | Calculation cost).

    `stage` selects a cumulative prefix of `STAGE_VARIANTS` (1..7 inclusive);
    every variant in that prefix is drawn muted except `highlight_variant`,
    drawn in full color and on top. `highlight_variant` must itself be inside
    that prefix -- highlighting a variant the talk hasn't introduced yet at
    this stage is a story error, not a rendering choice.
    Returns the `Figure`; callers save or embed it (SYNTHETIC placeholder
    figures render under `figures/_synth/`, never committed as real data).
    """
    import matplotlib.pyplot as plt

    if not 1 <= stage <= len(STAGE_VARIANTS):
        raise ValueError(f"stage must be in 1..{len(STAGE_VARIANTS)}, got {stage}")
    visible_variants = STAGE_VARIANTS[:stage]
    if highlight_variant not in visible_variants:
        raise ValueError(
            f"highlight_variant={highlight_variant!r} is not yet introduced at stage={stage} "
            f"(visible variants: {visible_variants!r})"
        )

    fig, (ax_eff, ax_cost) = plt.subplots(1, 2, figsize=(10, 4))
    _draw_efficiency_panel(ax_eff, efficiency_rows, visible_variants, highlight_variant)
    _draw_cost_panel(ax_cost, tolerance_rows, visible_variants, highlight_variant, stage)
    fig.suptitle(f"SYNTHETIC — placeholder (stage {stage}/{len(STAGE_VARIANTS)})", fontsize=9, color="#898781")
    fig.tight_layout()
    return fig
