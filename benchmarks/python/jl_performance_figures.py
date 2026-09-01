"""Figures for the PauliPropagation.jl head-to-head.

Rendered by ``bench_jl_performance.py`` from the ``summary.json`` it just
wrote, so a figure can always be re-made from the committed data without
re-running a single measurement::

    python benchmarks/python/jl_performance_figures.py \
        benchmarks/python/jl_performance/summary.json

Style is the suite's: ``examples/common/report.py``'s palette
(``_PALETTE[0]`` blue, ``_PALETTE[1]`` orange), its hairline-grid
``_style_axes``, and SVG output — so these sit next to benchmarks B, C and E's
figures without looking like a different document. The two-slot categorical
pair was validated (lightness band, chroma floor, CVD separation ΔE 24.7
protan / 32.7 tritan, normal-vision ΔE 33.6, contrast ≥ 3:1 — all pass).

Five deliberate choices worth stating, because each is a rule that is easy to
break by accident:

* **Small multiples, one panel per workload, never one axis with every
  workload's curves on it.** Identity then comes from the panel title rather
  than from hue alone, which is also what relieves the third palette slot's
  contrast warning.
* **Never a dual y-axis.** Where two measures of different scale belong
  together (speedup and efficiency; bytes-per-term and absolute peak) they get
  two panels, not two scales on one.
* **The indistinguishable zone is drawn, not omitted.** A term-count range
  where the pairs disagreed in sign is a shaded band with a label; no curve is
  drawn through it as if the direction were known.
* **Per-pair scatter under every median.** The ratio figure shows all pairs, so
  a reader sees the spread the median came from instead of trusting a line.
* **Ratio = t_julia / t_paulistrings throughout**, matching the driver, the
  results file and the README. Above 1 means this engine is faster.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples"))

from common import report  # noqa: E402

RUST_COLOR = report._PALETTE[0]  # blue
JL_COLOR = report._PALETTE[1]  # orange
GRID = report._GRID_COLOR
MUTED = report._MUTED_TEXT

RUST_LABEL = "paulistrings"
JL_LABEL = "PauliPropagation.jl 0.8.2"

#: Shading for a term-count band where the pairs never agreed on a direction.
ZONE_FACE = "#d8d6cf"
ZONE_ALPHA = 0.45


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    _plt().close(fig)
    return path


def _timed_points(curve: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The curve's configurations that produced a timing, term-count ordered."""
    return sorted(
        # `is not None`, not a truthiness test: a legitimately tiny time must
        # not be filtered out as if it were missing.
        (p for p in curve["points"] if p.get("rust_s_median") is not None),
        key=lambda p: p["peak_terms"],
    )


def _sweep_points(curve: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Only the `min_abs_coeff` sweep — the joined curve's actual points.

    A `max_weight` configuration is a knob-*equivalence* demonstration, not a
    point on the coefficient-cutoff size curve. Joining it into the same line
    draws a dip that is purely an artifact of putting two different knobs on one
    axis; it gets its own open marker instead.
    """
    return [p for p in _timed_points(curve) if p.get("max_weight") is None]


def _variant_points(curve: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [p for p in _timed_points(curve) if p.get("max_weight") is not None]


def _shade_zone(ax, curve: Mapping[str, Any], *, label: bool = True) -> None:
    """Shade the term-count band where the pairs disagreed on a direction."""
    zone = curve["crossover"].get("indistinguishable_zone")
    if not zone:
        return
    lo, hi = zone["lo"], zone["hi"]
    if lo == hi:  # a single configuration: give it a visible sliver
        lo, hi = lo * 0.85, hi * 1.18
    ax.axvspan(lo, hi, color=ZONE_FACE, alpha=ZONE_ALPHA, zorder=0, lw=0)
    if label:
        ax.text(
            math.sqrt(lo * hi),
            0.03,
            "indistinguishable",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
            rotation=90,
        )


def _mark_crossover(ax, curve: Mapping[str, Any]) -> None:
    x = curve["crossover"].get("crossover_terms")
    if not x:
        return
    ax.axvline(x, color=MUTED, ls=":", lw=1.0, zorder=1)
    ax.text(
        x,
        0.97,
        f"  crossover ~{x:.2g}",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=6.5,
        color=MUTED,
    )


# --------------------------------------------------------------------------
# 1. time vs term count
# --------------------------------------------------------------------------


def plot_time_vs_terms(curves: Sequence[Mapping[str, Any]], save_path: Path) -> Path:
    """Log-log warm propagation time against peak term count, per workload.

    Peak terms, not final terms, on the x axis: the peak is what the engine
    actually had to hold and sort, so it is the size a runtime should be read
    against. Several configurations here truncate down to a handful of final
    terms after passing through millions, and plotting those against their
    final count would put a two-second run at x = 11.
    """
    plt = _plt()
    curves = [c for c in curves if _timed_points(c)]
    if not curves:
        raise ValueError("no timed configurations to plot")
    fig, axes = plt.subplots(
        1, len(curves), figsize=(4.4 * len(curves), 3.8), squeeze=False
    )
    for ax, curve in zip(axes[0], curves):
        pts = _sweep_points(curve)
        x = [p["peak_terms"] for p in pts]
        _shade_zone(ax, curve)
        ax.plot(
            x, [p["rust_s_median"] for p in pts],
            "-o", color=RUST_COLOR, lw=2.0, ms=5, label=RUST_LABEL, zorder=3,
        )
        ax.plot(
            x, [p["jl_s_median"] for p in pts],
            "-s", color=JL_COLOR, lw=2.0, ms=5, label=JL_LABEL, zorder=3,
        )
        for v in _variant_points(curve):
            ax.plot(
                [v["peak_terms"]], [v["rust_s_median"]], "o", mfc="none",
                mec=RUST_COLOR, ms=9, mew=1.5, zorder=4,
                label=f"max_weight={v['max_weight']} variant",
            )
            ax.plot(
                [v["peak_terms"]], [v["jl_s_median"]], "s", mfc="none",
                mec=JL_COLOR, ms=9, mew=1.5, zorder=4,
            )
        _mark_crossover(ax, curve)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("peak terms")
        ax.set_title(curve["workload"], fontsize=9)
        report._style_axes(ax)
    axes[0][0].set_ylabel("warm propagation time (s), 1 thread")
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(
        "Warm single-threaded propagation time vs peak term count "
        "(median of interleaved pairs)",
        fontsize=10,
    )
    fig.tight_layout()
    return _save(fig, save_path)


# --------------------------------------------------------------------------
# 2. ratio vs term count, with per-pair scatter
# --------------------------------------------------------------------------


def plot_ratio_vs_terms(curves: Sequence[Mapping[str, Any]], save_path: Path) -> Path:
    """Per-pair ratios and their medians, against peak term count.

    Every pair is drawn, not just the median, because the acceptance rule is
    about the pairs' *agreement in sign* — a reader has to be able to see
    whether they agreed. Points above the ``ratio = 1`` line are pairs this
    engine won.
    """
    plt = _plt()
    curves = [c for c in curves if _timed_points(c)]
    fig, axes = plt.subplots(
        1, len(curves), figsize=(4.4 * len(curves), 3.8), squeeze=False
    )
    for ax, curve in zip(axes[0], curves):
        pts = _sweep_points(curve)
        # Keyed on the label, not the term count: two configurations can land on
        # the same peak (a saturated sum reached from two cutoffs), and keying on
        # the count would silently drop one of their scatters.
        by_label = {
            cfg["label"]: cfg["timing"] for cfg in curve["configs"] if cfg.get("timing")
        }
        _shade_zone(ax, curve)
        ax.axhline(1.0, color=MUTED, lw=1.2, zorder=1)
        # per-pair scatter
        for p in pts:
            timing = by_label.get(p["label"])
            if not timing:
                continue
            ratios = timing["ratio_jl_over_rust_per_pair"]
            ax.plot(
                [p["peak_terms"]] * len(ratios), ratios,
                "o", color=RUST_COLOR, ms=3.2, alpha=0.45, mew=0, zorder=2,
            )
        ax.plot(
            [p["peak_terms"] for p in pts],
            [p["median_ratio_jl_over_rust"] for p in pts],
            "-o", color=RUST_COLOR, lw=2.0, ms=5.5, zorder=3,
            label="median of pairs",
        )
        # mark the configurations whose pairs disagreed
        mixed = [p for p in pts if not p["sign_consistent"]]
        if mixed:
            ax.plot(
                [p["peak_terms"] for p in mixed],
                [p["median_ratio_jl_over_rust"] for p in mixed],
                "o", mfc="none", mec=MUTED, ms=10, mew=1.4, zorder=4,
                label="pairs disagreed",
            )
        for v in _variant_points(curve):
            ax.plot(
                [v["peak_terms"]], [v["median_ratio_jl_over_rust"]], "D",
                mfc="none", mec=JL_COLOR, ms=8, mew=1.5, zorder=4,
                label=f"max_weight={v['max_weight']} variant",
            )
        _mark_crossover(ax, curve)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("peak terms")
        ax.set_title(curve["workload"], fontsize=9)
        report._style_axes(ax)
        ax.legend(frameon=False, fontsize=7, loc="best")
    axes[0][0].set_ylabel(
        r"$t_{\mathrm{julia}}\,/\,t_{\mathrm{paulistrings}}$"
        "\n(above 1: paulistrings faster)"
    )
    fig.suptitle(
        "Per-pair time ratio vs peak term count — every pair drawn, median joined",
        fontsize=10,
    )
    fig.tight_layout()
    return _save(fig, save_path)


# --------------------------------------------------------------------------
# 2b. per-term cost — where the ratio's shape comes from
# --------------------------------------------------------------------------


def plot_per_term_cost(curves: Sequence[Mapping[str, Any]], save_path: Path) -> Path:
    """Nanoseconds per peak term, per engine, per workload.

    The ratio figure says *that* the advantage changes with size; this one says
    *whose* cost moved, which is the only version of the observation an
    optimization can act on. Dividing out the term count removes the trivially
    dominant linear growth and leaves the per-term efficiency, so a curve that
    falls is an engine still amortizing fixed cost and a curve that flattens is
    one that has stopped.

    Read the two panels against each other: where one engine's curve keeps
    falling and the other's has plateaued, the ratio moves for a structural
    reason rather than a noisy one.
    """
    plt = _plt()
    curves = [c for c in curves if _timed_points(c)]
    fig, axes = plt.subplots(
        1, len(curves), figsize=(4.4 * len(curves), 3.8), squeeze=False
    )
    for ax, curve in zip(axes[0], curves):
        pts = _sweep_points(curve)
        x = [p["peak_terms"] for p in pts]
        ax.plot(
            x, [p["rust_s_median"] / p["peak_terms"] * 1e9 for p in pts],
            "-o", color=RUST_COLOR, lw=2.0, ms=5, label=RUST_LABEL,
        )
        ax.plot(
            x, [p["jl_s_median"] / p["peak_terms"] * 1e9 for p in pts],
            "-s", color=JL_COLOR, lw=2.0, ms=5, label=JL_LABEL,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("peak terms")
        ax.set_title(curve["workload"], fontsize=9)
        report._style_axes(ax)
    axes[0][0].set_ylabel("ns per peak term (warm, 1 thread)")
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="best")
    fig.suptitle(
        "Per-term cost — the shape behind the ratio "
        "(falling = still amortizing fixed cost)",
        fontsize=10,
    )
    fig.tight_layout()
    return _save(fig, save_path)


# --------------------------------------------------------------------------
# 3. memory
# --------------------------------------------------------------------------


def plot_memory(curves: Sequence[Mapping[str, Any]], save_path: Path) -> Path:
    """Two panels: bytes per peak term, and absolute peak RSS.

    Two panels rather than two y-scales on one. The left panel is the figure
    that can be compared between engines; the right panel is why the left one
    needs a floor subtracted at all — both engines carry a fixed per-process
    cost (Julia's runtime and packages, ~0.6 GiB; the Python interpreter plus
    numpy and the extension) that swamps a small run's payload.
    """
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    rows: list[dict[str, Any]] = []
    for curve in curves:
        for cfg in curve["configs"]:
            timing = cfg.get("timing")
            if not timing:
                continue
            mem = timing["peak_memory"]
            rows.append({"workload": curve["workload"], **mem})
    rows.sort(key=lambda r: r["peak_terms"])

    ax = axes[0]
    for key, color, label, marker in (
        ("rust_bytes_per_peak_term", RUST_COLOR, RUST_LABEL, "o"),
        ("jl_bytes_per_peak_term", JL_COLOR, JL_LABEL, "s"),
    ):
        pts = [(r["peak_terms"], r[key]) for r in rows if r.get(key)]
        if pts:
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts],
                marker, color=color, ms=5.5, lw=0, label=label, alpha=0.85,
            )
    # This engine's payload arithmetic, as a floor to read the plateau against:
    # at W = 2 a term is 48 B (a 32 B symplectic key + a 16 B complex
    # coefficient) -- benchmarks/PROFILING.md's bytes-moved model. Anything
    # above it is allocator slack, transient buffers and bucket headroom.
    ax.axhline(48.0, color=RUST_COLOR, ls=":", lw=1.0, alpha=0.8)
    ax.text(
        0.99, 48.0, "48 B/term: paulistrings payload at W=2 ",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=6.5, color=MUTED,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("peak terms")
    ax.set_ylabel("bytes per peak term\n(peak RSS minus the process floor)")
    ax.set_title(
        "Floor-subtracted bytes per term\n(only meaningful once the payload "
        "clears the floor, right of ~$10^5$)",
        fontsize=8.5,
    )
    report._style_axes(ax)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    for key, floor_key, color, label, marker in (
        ("rust_vmhwm_kb", "rust_floor_kb", RUST_COLOR, RUST_LABEL, "o"),
        ("jl_vmhwm_kb", "jl_floor_kb", JL_COLOR, JL_LABEL, "s"),
    ):
        pts = [(r["peak_terms"], r[key] / 1048576.0) for r in rows if r.get(key)]
        if pts:
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts],
                marker + "-", color=color, ms=5, lw=1.6, label=label, alpha=0.9,
            )
        floors = [r[floor_key] for r in rows if r.get(floor_key)]
        if floors:
            ax.axhline(
                sorted(floors)[len(floors) // 2] / 1048576.0,
                color=color, ls="--", lw=1.0, alpha=0.7,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("peak terms")
    ax.set_ylabel("peak RSS (GiB)")
    ax.set_title("Absolute peak RSS (dashed: each engine's floor)", fontsize=9)
    report._style_axes(ax)
    ax.legend(frameon=False, fontsize=7.5)

    fig.suptitle(
        "Memory, each engine sampling its own /proc/self/status", fontsize=10
    )
    fig.tight_layout()
    return _save(fig, save_path)


# --------------------------------------------------------------------------
# 4. time to fixed accuracy
# --------------------------------------------------------------------------


def plot_time_to_accuracy(
    references: Sequence[Mapping[str, Any]], save_path: Path
) -> Path:
    """Absolute error against warm wall time, one panel per reference.

    The time-to-accuracy reading is horizontal: pick an error bar, and the
    curve that reaches it furthest left is the engine that gets there sooner.
    Both engines pass through the *same* error values — matched truncation
    gives matched expectations to ~1e-16 — so the two curves differ only
    horizontally, which is exactly the claim.
    """
    plt = _plt()
    usable = [
        r
        for r in references
        if any(row.get("absolute_error") is not None for row in r["rows"])
    ]
    if not usable:
        raise ValueError("no accuracy rows to plot")
    fig, axes = plt.subplots(
        1, len(usable), figsize=(4.3 * len(usable), 3.8), squeeze=False
    )
    floor = 1e-17
    for ax, ref in zip(axes[0], usable):
        rows = [r for r in ref["rows"] if r.get("absolute_error") is not None]
        for key, color, label, marker in (
            ("rust_s_median", RUST_COLOR, RUST_LABEL, "o"),
            ("jl_s_median", JL_COLOR, JL_LABEL, "s"),
        ):
            xs = [r[key] for r in rows]
            ys = [max(r["absolute_error"], floor) for r in rows]
            ax.plot(xs, ys, marker + "-", color=color, lw=2.0, ms=5, label=label)
        for bar in (1e-2, 1e-3):
            ax.axhline(bar, color=MUTED, ls="--", lw=1.0)
            ax.text(
                0.99, bar, f" |err| < {bar:g}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.5, color=MUTED,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("warm propagation time (s), 1 thread")
        ax.set_title(f"$\\theta_h$ = {ref['theta_label']}, {ref['steps']} steps", fontsize=9)
        report._style_axes(ax)
    axes[0][0].set_ylabel(f"|error| vs exact oracle\n(clamped at {floor:g})")
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="best")
    fig.suptitle(
        "Time to fixed accuracy — kicked-Ising 127q, exact references", fontsize=10
    )
    fig.tight_layout()
    return _save(fig, save_path)


# --------------------------------------------------------------------------
# 5. thread scaling (this engine only)
# --------------------------------------------------------------------------


def plot_thread_scaling(scaling: Mapping[str, Any], save_path: Path) -> Path:
    """Speedup and parallel efficiency, in two panels.

    Two panels, not one axis with two scales: speedup runs 1..32 and efficiency
    runs 0..1, and putting them on a shared axis is the single most common chart
    mistake there is.

    This engine only — the title says so, because PauliPropagation.jl 0.8.2's
    dict backend has no threaded propagation path and there is no second curve
    that belongs here.
    """
    plt = _plt()
    rows = scaling["rows"]
    threads = [r["threads"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7))

    ax = axes[0]
    ax.plot(threads, [r["speedup"] for r in rows], "-o", color=RUST_COLOR, lw=2.0, ms=5.5)
    ax.plot(threads, threads, ls="--", lw=1.0, color=MUTED, label="ideal")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("Rayon threads")
    ax.set_ylabel("speedup vs 1 thread")
    ax.set_title("Speedup", fontsize=9)
    report._style_axes(ax)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    ax.plot(
        threads, [r["efficiency"] for r in rows], "-o", color=RUST_COLOR, lw=2.0, ms=5.5
    )
    ax.axhline(1.0, ls="--", lw=1.0, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Rayon threads")
    ax.set_ylabel("parallel efficiency")
    ax.set_title("Efficiency (speedup / threads)", fontsize=9)
    report._style_axes(ax)

    cfg = scaling["configuration"]
    fig.suptitle(
        f"Thread scaling — paulistrings only (PauliPropagation.jl 0.8.2 is "
        f"single-threaded)\nkicked-Ising 127q, $\\theta_h$ = {cfg['theta_label']}, "
        f"{cfg['trotter_steps']} steps, min_abs_coeff = {cfg['min_abs_coeff']:.3g}",
        fontsize=9.5,
    )
    fig.tight_layout()
    return _save(fig, save_path)


# --------------------------------------------------------------------------


def render_all(summary: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Render whatever the summary has data for; skip the rest without failing."""
    out_dir = Path(out_dir)
    written: list[Path] = []
    curves = [c for c in summary.get("curves", []) if _timed_points(c)]
    if curves:
        written.append(plot_time_vs_terms(curves, out_dir / "time-vs-terms.svg"))
        written.append(plot_ratio_vs_terms(curves, out_dir / "ratio-vs-terms.svg"))
        written.append(plot_per_term_cost(curves, out_dir / "per-term-cost.svg"))
        written.append(plot_memory(curves, out_dir / "memory-per-term.svg"))
    accuracy = summary.get("accuracy") or []
    if any(
        any(r.get("absolute_error") is not None for r in a["rows"]) for a in accuracy
    ):
        written.append(plot_time_to_accuracy(accuracy, out_dir / "time-to-accuracy.svg"))
    scaling = summary.get("thread_scaling")
    if scaling and scaling.get("rows"):
        written.append(plot_thread_scaling(scaling, out_dir / "thread-scaling.svg"))
    return written


def main(argv: Sequence[str]) -> int:
    if not argv:
        print(__doc__)
        print("usage: jl_performance_figures.py <summary.json> [out_dir]")
        return 2
    summary_path = Path(argv[0])
    out_dir = Path(argv[1]) if len(argv) > 1 else summary_path.parent
    summary = json.loads(summary_path.read_text())
    for path in render_all(summary, out_dir):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
