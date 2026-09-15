"""Regenerate every figure the talk uses, in the Organic design system.

Writes `presentation/figures/organic/` plus `sizes.json`, a manifest of each
figure's true size in stage points. The deck places each image at exactly that
size, so 1 figure pt == 1 stage pt and a 24pt tick label is 24pt on the slide.

    python presentation/plots/organic_figures.py [--only NAME,NAME]

The presentation scripts (`fig*.py`) are run unmodified through `runpy`; the
palette reaches them through `organic_style.apply(common)`, and the figure
size, the 24pt type floor and the save path through the interceptors below.
The three generators under `docs/figures/` do not use `common.py`, so their
module-level color constants are rebound by hand.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "figures" / "organic"
os.environ["PS_FIGURES_DIR"] = str(OUT)
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import common  # noqa: E402
import organic_style as org  # noqa: E402

SIZES: dict[str, tuple[float, float]] = {}


# ── interceptors ─────────────────────────────────────────────────────────────
@contextlib.contextmanager
def figsize(w_pt: float, h_pt: float):
    """Force every `plt.subplots` in the wrapped call to the authored size."""
    orig = plt.subplots
    size = org.stage_figsize(w_pt, h_pt)

    def subplots(*a, **kw):
        kw["figsize"] = size
        return orig(*a, **kw)

    plt.subplots = subplots
    try:
        yield
    finally:
        plt.subplots = orig


@contextlib.contextmanager
def saves(tight: bool = False, strip_titles: bool = False, post=None):
    """Redirect every savefig into `OUT`, drop the tight crop (it silently
    rescales the type), lift the figure to the 24pt floor, and record the
    size actually written."""
    orig = Figure.savefig

    def savefig(self, fname, **kw):
        name = Path(str(fname)).stem
        if strip_titles:
            for ax in self.axes:
                ax.set_title("")
        if post is not None:
            post(self)
        org.enforce_min_type(self)
        org.enlarge_markers(self)
        for ax in self.axes:
            org.thicken(ax)
        if tight:
            kw["bbox_inches"] = "tight"
        else:
            kw.pop("bbox_inches", None)
            with contextlib.suppress(Exception):
                self.tight_layout()
        kw["facecolor"] = org.BG
        kw.pop("format", None)
        OUT.mkdir(parents=True, exist_ok=True)
        for ext in ("svg", "pdf"):
            orig(self, OUT / f"{name}.{ext}", format=ext, **kw)
        SIZES[name] = svg_size(OUT / f"{name}.svg")

    Figure.savefig = savefig
    try:
        yield
    finally:
        Figure.savefig = orig


def svg_size(path: Path) -> tuple[float, float]:
    """The rendered size in pt, read back from the SVG header."""
    head = path.read_text()[:600]
    w = re.search(r'width="([\d.]+)pt"', head)
    h = re.search(r'height="([\d.]+)pt"', head)
    return (round(float(w.group(1)), 1), round(float(h.group(1)), 1)) if w and h else (0.0, 0.0)


def _fig5_legend(fig) -> None:
    ax = fig.axes[0]
    if ax.get_legend() is not None:
        ax.legend(loc="center left", frameon=False, fontsize=org.T_AXIS)
    # The band annotations are left-anchored on their band; the rightmost one
    # runs off the canvas at this type size. Flip it to hang left of its band.
    lo, hi = ax.get_xlim()
    for t in ax.texts:
        # These are Annotations: the anchor is `.xy` in data coords, while
        # `get_position()` is the offset in points.
        anchor = getattr(t, "xy", (lo, 0))[0]
        if t.get_text().startswith("gather crosses") and anchor > lo + 0.75 * (hi - lo):
            t.set_horizontalalignment("right")
            t.set_position((-6, t.get_position()[1]))
    # This one figure keeps the tight crop: its band annotation overhangs the
    # axes and no alignment keeps it inside a fixed canvas. The deck places it
    # at the size in `sizes.json`, so the type scale is still 1:1.


def run_script(script: str, argv: list[str], w: float, h: float, env: dict | None = None,
               tight: bool = False, pre=None, post=None) -> None:
    for k, v in (env or {}).items():
        os.environ[k] = v
    org.apply(common, figsize_pt=(w, h))
    if pre is not None:
        pre()
    sys.argv = [script, *argv, "--data-dir", str(HERE.parent / "data")]
    with figsize(w, h), saves(tight=tight, post=post):
        runpy.run_path(str(HERE / script), run_name="__main__")
    plt.close("all")


# ── the ladder, redrawn in the system's chart grammar ────────────────────────
def ladder(w_pt: float = 1400, h_pt: float = 560, name: str = "organic_ladder",
           upto: int | None = None) -> None:
    """F1. Horizontal bars are drawn as round-capped lines: the system's
    "fully pill" rule, and the only way to round an end on a log axis without
    fighting the data transform."""
    import numpy as np

    org.apply(common)
    common.apply_rcparams()
    import fig1_engine_ladder as f

    stages = f.build_stages(HERE.parent / "data")
    if upto is not None:
        stages = stages[:upto]
    labels = [s[0] for s in stages]
    values = [s[1] / 1e9 for s in stages]
    keys = [s[2] for s in stages]
    full = [s[1] / 1e9 for s in f.build_stages(HERE.parent / "data")]
    baseline = full[0]

    fig, ax = plt.subplots(figsize=org.stage_figsize(w_pt, h_pt))
    # Rows are positioned against the *full* ladder, so a bar sits at the same
    # height in every reveal and the chart grows downward instead of the
    # revealed bars sliding to the bottom of the axis.
    y = np.arange(len(full))[::-1][: len(values)]
    for yi, v, k in zip(y, values, keys):
        ax.plot([min(full) * 0.42, v], [yi, yi], color=org.VARIANT_COLORS[k],
                linewidth=30, solid_capstyle="round", zorder=2)
        ax.text(v * 1.26, yi, f"{v:.1f} s" + ("" if v == baseline else f"   {baseline / v:.1f}×"),
                va="center", ha="left", fontsize=org.T_AXIS, color=org.INK_55, zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(min(full) * 0.42, max(full) * 3.0)
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.8, len(full) - 0.2)
    ax.set_xlabel("median propagate wall time (s, log scale)")
    ax.tick_params(axis="y", length=0, pad=20)
    ax.grid(axis="x", color=org.DIVIDER, linewidth=org.LW_HAIRLINE)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(org.DIVIDER)
    with saves():
        fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


# ── the three generators under docs/figures/ ─────────────────────────────────
def docs_figures() -> None:
    """These do not import `common`; rebind their module-level constants."""
    org.apply(common)
    matplotlib.rcParams.update(org.rcparams())
    sys.path.insert(0, str(ROOT / "docs" / "figures" / "design"))
    sys.path.insert(0, str(ROOT / "docs" / "figures" / "comparisons"))

    import bucket_cosets as bc

    # Four cosets: the highlighted one takes the lead accent, the rest recede
    # into sage and neutral. Same order as the script's `reps`.
    bc.PALETTE = [org.SAGE[500], org.ACCENT, org.NEUTRAL[400], org.CLAY[300]]

    def rounded(xy, w, h, **kw):
        """The schematic's cells are the one place the system's "round
        everything" rule can be applied to someone else's figure: swap the
        Rectangle for a FancyBboxPatch with a 0.1-unit corner."""
        from matplotlib.patches import FancyBboxPatch

        kw.setdefault("edgecolor", org.BG)
        return FancyBboxPatch((xy[0] + 0.06, xy[1] + 0.06), w - 0.12, h - 0.12,
                              boxstyle="round,pad=0,rounding_size=0.14", **kw)

    bc.Rectangle = rounded
    with figsize(1000, 620), saves(strip_titles=True):
        bc.main()

    import performance_plots as pp

    pp.BLUE, pp.ORANGE, pp.AQUA = org.CLAY[600], org.SAGE[600], org.NEUTRAL[500]
    pp.INK, pp.MUTED = org.INK, org.INK_55
    with figsize(900, 560), saves(strip_titles=True):
        pp.roofline_threads()
    with figsize(900, 440), saves(strip_titles=True):
        pp.phase_shares()

    import baseline_ops as bo

    bo.BLUE, bo.ORANGE, bo.AQUA = org.CLAY[600], org.SAGE[600], org.NEUTRAL[500]
    bo.INK, bo.MUTED = org.INK, org.INK_55
    # `LIBS` captured the old colors at import time, so rebinding the names
    # above is not enough -- rewrite the tuple's color slot too.
    bo.LIBS = tuple((k, n, c, m) for (k, n, _, m), c in
                    zip(bo.LIBS, (org.CLAY[600], org.SAGE[600], org.NEUTRAL[500])))
    with figsize(1400, 520), saves(tight=True):
        bo.main()
    plt.close("all")


# ── the set the deck uses ────────────────────────────────────────────────────
JOBS = {
    # fig0 samples `plt.cm.viridis` by name for its 9 ordered curves; swap the
    # ramp for the system's own sage -> terracotta sequence.
    "fig0": lambda: run_script("fig0_term_growth.py", [], 1180, 520, env={"TROTTER_STEPS": "10"},
                               pre=lambda: setattr(plt.cm, "viridis", org.sequential_cmap())),
    "ladder": lambda: (ladder(1400, 560, "organic_ladder", upto=1),
                       ladder(1400, 560, "organic_ladder2", upto=2),
                       ladder(1400, 560, "organic_ladder5", upto=5),
                       ladder(1400, 560, "organic_ladder7"),
                       ladder(900, 520, "organic_ladder_col")),
    "fig2": lambda: run_script("fig2_targetcpu.py", [], 900, 380),
    "fig3old": lambda: run_script("fig3_thread_scaling.py", ["--only", "threadmaps,mergesort"], 1100, 560),
    "fig3all": lambda: run_script("fig3_thread_scaling.py", [], 1100, 560),
    "fig4": lambda: run_script("fig4_superlinear.py", ["--data", "thread_scaling.jsonl", "--out", "fig4_superlinear"], 900, 520),
    "fig4l": lambda: run_script("fig4_superlinear.py", ["--data", "thread_scaling_large.jsonl", "--out", "fig4_superlinear_large"], 900, 520),
    "fig4b": lambda: run_script("fig4b_bucket_speedup.py", [], 1100, 620),
    # Wider than the rest: the top panel carries two band annotations that run
    # to the right edge, and at 28pt they need the room. Its legend is moved
    # off the curves after the script builds it.
    "fig5": lambda: run_script("fig5_bucket_sweep.py", [], 1340, 640, post=_fig5_legend, tight=True),
    "fig6": lambda: run_script("fig6_memory.py", [], 900, 480),
    "docs": docs_figures,
}


def ensure_fonts() -> None:
    """matplotlib caches the font list, so a newly installed Figtree is
    invisible until the cache is rebuilt -- and the figures fall back to
    Carlito without saying so."""
    import matplotlib.font_manager as fm

    if not any(f.name == "Figtree" for f in fm.fontManager.ttflist):
        # `_load_fontmanager` returns a fresh manager, it does not rebind the
        # module global -- calling it alone looks like it worked and changes
        # nothing.
        fm.fontManager = fm._load_fontmanager(try_read_cache=False)
        found = any(f.name == "Figtree" for f in fm.fontManager.ttflist)
        print(f"rebuilt matplotlib font cache; Figtree found: {found}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(JOBS)

    ensure_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT / "sizes.json").read_text()) if (OUT / "sizes.json").exists() else {}
    for name in names:
        print(f"==> {name}")
        JOBS[name]()
    manifest.update({k: list(v) for k, v in SIZES.items()})
    (OUT / "sizes.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {len(SIZES)} figures to {OUT}")


if __name__ == "__main__":
    main()
