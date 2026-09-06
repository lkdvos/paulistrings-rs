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

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


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


def save(fig, name: str) -> tuple[Path, Path]:
    """Write `figures/<name>.svg` and `figures/<name>.pdf`, `bbox_inches="tight"`."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = FIGURES_DIR / f"{name}.svg"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    return svg_path, pdf_path
