"""Organic design-system styling for the presentation figures.

The deck's stage is 1920pt x 1080pt, and Typst's pt is matplotlib's pt, so a
figure sized `stage_figsize(w_pt, h_pt)` renders 1:1 on the slide: a 24pt tick
label is 24pt on the stage. Every size below is therefore the system's own
number from `slides/organic/BRIEF.md`, unconverted.

Use as an overlay on `common.py` -- `apply(common)` rebinds the palette hooks
the figure scripts read, so the existing scripts keep working unchanged.
"""

from __future__ import annotations

# ── tokens (slides/organic/organic.typ) ──────────────────────────────────────
BG = "#f5ead8"
SURFACE = "#ebddc5"
INK = "#201e1d"
ACCENT = "#c67139"        # terracotta, the lead
ACCENT2 = "#7a8a5e"       # sage, the second voice

NEUTRAL = {300: "#dcd3c4", 400: "#c0b6a5", 500: "#a19786", 600: "#82796a", 700: "#645c50"}
CLAY = {300: "#ffc6a5", 400: "#f6a06b", 500: "#d67f48", 600: "#b2622d", 700: "#8c491a", 800: "#643312", 900: "#402310"}
SAGE = {300: "#ccdbb2", 500: "#8fa073", 600: "#728157", 750: "#56633f", 800: "#3d472b"}

INK_55 = "#201e1d8c"      # captions, axis text
DIVIDER = "#dcd3c4"       # 1pt hairline grid (neutral-300, opaque:
                          # matplotlib multiplies an 8-digit hex by any alpha= kwarg)

T_AXIS = 28               # axis and value text. The brief's kicker is 24pt
                          # and that is the floor; the deck's reading sizes were
                          # raised, so figure type comes up with them (70 % of body).
T_CAPTION = 30
LW_LEAD = 4               # terracotta leads solid at 4pt
LW_HAIRLINE = 1

# Five engine variants against a two-accent system: the two rejected attempts
# take neutral and sage, the shipped engine owns terracotta in two tones.
VARIANT_COLORS = {
    "naive": NEUTRAL[500],
    "threadmaps": NEUTRAL[700],
    "mergesort": SAGE[600],
    "bucketed": CLAY[600],
    "bucketed-coarse": CLAY[900],
}


def stage_figsize(w_pt: float, h_pt: float) -> tuple[float, float]:
    """Figure size in inches such that 1 matplotlib pt == 1 stage pt."""
    return (w_pt / 72.0, h_pt / 72.0)


def sequential_cmap():
    """Ordered-quantity ramp: sage dark -> sage -> terracotta -> clay light.

    Monotone in lightness so an ordered sweep reads as ordered, and it stays
    inside the system's two hues instead of importing viridis.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "organic_seq", [SAGE[800], SAGE[600], CLAY[600], CLAY[400]]
    )


def rcparams(base: int = T_AXIS) -> dict:
    return {
        "font.family": ["Figtree", "Carlito", "DejaVu Sans"],   # the deck's body face;
                                                    # DejaVu carries the math glyphs Figtree lacks
        "font.size": base,
        "axes.labelsize": base,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "text.color": INK,
        "axes.labelcolor": INK_55,
        "xtick.color": INK_55,
        "ytick.color": INK_55,
        "axes.edgecolor": DIVIDER,
        "grid.color": DIVIDER,
        "grid.linewidth": LW_HAIRLINE,
        "lines.linewidth": LW_LEAD,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def save_exact(common):
    """`common.save` crops with `bbox_inches="tight"`, which silently rescales
    the figure when the slide places it at a fixed width -- 24pt axis type
    lands at 14pt on the stage. This saver keeps the authored figure size, so
    `image(path, width: <authored w>pt)` is 1:1 and the type is the type."""

    def save(fig, name: str):
        fig.tight_layout()
        common.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        svg = common.FIGURES_DIR / f"{name}.svg"
        pdf = common.FIGURES_DIR / f"{name}.pdf"
        fig.savefig(svg, format="svg")
        fig.savefig(pdf, format="pdf")
        return svg, pdf

    return save


def apply(common, base: int = T_AXIS, figsize_pt: tuple[float, float] = (1180, 620)) -> None:
    """Rebind `common`'s palette hooks and rcparams to the Organic system."""
    import matplotlib as mpl

    common._GRID_COLOR = DIVIDER
    common._MUTED_TEXT = INK_55
    common._PALETTE = [ACCENT, ACCENT2, NEUTRAL[600], CLAY[700], SAGE[800], CLAY[300], NEUTRAL[400], SAGE[500]]
    common.VARIANT_COLORS = dict(VARIANT_COLORS)
    common.FIGSIZE = stage_figsize(*figsize_pt)
    common.FONT_SIZE = base

    def apply_rcparams() -> None:
        mpl.rcParams.update(rcparams(base))

    common.apply_rcparams = apply_rcparams
    common.save = save_exact(common)


def enforce_min_type(fig, floor: int = T_AXIS) -> None:
    """The figure scripts hardcode `fontsize=8..9` on legends and annotations.
    On the 1920pt stage those are 8pt of type. The system's floor is 24pt --
    nothing on the stage goes below it, figures included."""
    import matplotlib.text

    for t in fig.findobj(matplotlib.text.Text):
        if t.get_fontsize() < floor:
            t.set_fontsize(floor)


def enlarge_markers(fig, floor: float = 13.0) -> None:
    """Marker-only series (the cross-library comparison) keep matplotlib's
    default 6pt marker, which is a speck on a 1400pt figure. The system draws
    every mark as a filled circle -- give them a size that reads."""
    import matplotlib.lines

    for line in fig.findobj(matplotlib.lines.Line2D):
        if line.get_marker() not in (None, "None", "", " "):
            line.set_markersize(max(line.get_markersize(), floor))


def thicken(ax, width: float = LW_LEAD) -> None:
    """Raise matplotlib's 1.0-1.5pt data lines to the system's 4pt lead.

    Only mid-weight lines: a hairline grid stays a hairline, and a bar drawn
    as a 30pt round-capped line stays a bar. (An earlier version clamped
    everything to 4pt and quietly turned the ladder's pills into threads.)
    """
    for line in ax.lines:
        lw = line.get_linewidth()
        if LW_HAIRLINE * 1.2 < lw < width:
            line.set_linewidth(width)
        if lw <= width:
            line.set_solid_capstyle("round")
