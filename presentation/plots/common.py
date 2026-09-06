"""Shared plotting helpers for the presentation figures.

Palette, grid/axis styling and the SVG-saving convention are copied (not
imported) from `examples/common/report.py`, so this module has no dependency
on the examples/benchmarks suite and can evolve independently for slide-deck
needs. `load_jsonl` reads the `# provenance: {...}` + one-JSON-object-per-line
format `collect_term_growth.py` writes.

Fonts: DejaVu Sans (matplotlib's default, always available). Base font size
11pt. Default figure size (6.4, 3.6) inches -- a 16:9 slide crop. Axes never
carry a second (right-hand) y-scale: a genuinely different measure gets its
own subplot instead of a dual axis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Palette (copied from examples/common/report.py, not imported)
# --------------------------------------------------------------------------

_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
_GRID_COLOR = "#e1e0d9"
_MUTED_TEXT = "#898781"

FIGSIZE = (6.4, 3.6)
FONT_SIZE = 11

import os as _os

FIGURES_DIR = Path(_os.environ.get("PS_FIGURES_DIR") or (Path(__file__).resolve().parents[1] / "figures"))

# --------------------------------------------------------------------------
# Variant identity, fixed across every figure in the F1-F7 set (fig0's eps
# sweep is a separate, ordered-quantity colour scheme -- see its docstring).
# --------------------------------------------------------------------------

VARIANT_COLORS = {
    "naive": "#898781",  # muted grey
    "threadmaps": "#eb6834",  # orange
    "mergesort": "#eda100",  # yellow
    "bucketed": "#2a78d6",  # blue
    "bucketed-coarse": "#4a3aa7",  # violet
}

VARIANT_LABELS = {
    "naive": "naive hash map",
    "threadmaps": "per-thread maps",
    "mergesort": "parallel mergesort",
    "bucketed": "bucketed",
    "bucketed-coarse": "bucketed, coarse buckets",
}


def _style_axes(ax) -> None:
    """Recessive grid/axes: hairline grid, muted spines. Never a right-hand y-axis."""
    ax.grid(True, color=_GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED_TEXT)
    ax.tick_params(colors=_MUTED_TEXT)


def apply_rcparams() -> None:
    """Base font family/size for every figure in this directory."""
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["font.size"] = FONT_SIZE


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a `collect_term_growth.py`-style JSONL file, skipping `#` lines."""
    records = []
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append(json.loads(line))
    return records


COARSE_TARGET_BUCKET_LEN = 16384


def variant_key(record: dict[str, Any]) -> str:
    """Map a data row to one of `VARIANT_COLORS`' keys.

    Every layer name other than `"bucketed"` is used as-is. A `"bucketed"` row
    splits into `"bucketed"` (default `target_bucket_len`) vs
    `"bucketed-coarse"` (`target_bucket_len == COARSE_TARGET_BUCKET_LEN`), since
    the two are drawn as separate series everywhere in this figure set.
    """
    layer = record["layer"]
    if layer != "bucketed":
        return layer
    if record.get("target_bucket_len") == COARSE_TARGET_BUCKET_LEN:
        return "bucketed-coarse"
    return "bucketed"


def aggregate_median(
    records: list[dict[str, Any]],
    group_keys: tuple[str, ...],
    value_key: str,
) -> dict[tuple[Any, ...], tuple[float, float, float]]:
    """Group `records` by `group_keys` and reduce `value_key` to (median, min, max).

    Reps land in the same group when every `group_keys` field matches; the
    three-tuple is what an errorbar plot wants directly (median as the point,
    min/max as the low/high whisker).
    """
    import numpy as np

    groups: dict[tuple[Any, ...], list[float]] = {}
    for r in records:
        key = tuple(r.get(k) for k in group_keys)
        if r.get(value_key) is None:
            continue
        groups.setdefault(key, []).append(r[value_key])

    out = {}
    for key, values in groups.items():
        arr = np.asarray(values, dtype=float)
        out[key] = (float(np.median(arr)), float(arr.min()), float(arr.max()))
    return out


def save(fig, name: str) -> tuple[Path, Path]:
    """Write `figures/<name>.svg` and `figures/<name>.pdf`, `bbox_inches="tight"`."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = FIGURES_DIR / f"{name}.svg"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    return svg_path, pdf_path
