#!/usr/bin/env python3
"""Performance figures for the Design → Performance page.

Two figures, both plotting numbers copied verbatim from the committed fact
sheet ``research/notes/2026-09-01-roofline-ccqlin038.md`` (host ccqlin038,
commit 94b3364, ``--qubits 128``, W = 2):

* ``roofline-threads.svg`` — attributable DRAM traffic of the ``su4`` probe
  layer against thread count, with the host's measured read/write bandwidth
  ceilings drawn as bands (the band spans the 16-phys and 32-thread
  placements). The write series sits on its ceiling from 16 threads up.
* ``phase-shares.svg`` — share of summed worker busy time in gather / sort /
  merge per layer class at one thread.

Regenerate the committed SVGs with:

    .venv/bin/python docs/figures/design/performance_plots.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

BLUE, ORANGE, AQUA = "#4C72B0", "#DD8452", "#1baf7a"
INK, MUTED = "#222222", "#666666"

# fact sheet Table B: su4, m = 1.41e7
THREADS = [1, 8, 16, 32]
READ_GBPS = [0.61, 25.9, 39.3, 39.5]
WRITE_GBPS = [0.27, 18.2, 28.1, 27.6]
# ceilings (bandwidth fact sheet): both sockets, 16 phys vs 32 threads
READ_CEIL = (45.0, 48.8)
WRITE_CEIL = (23.1, 25.3)

# fact sheet phase-share table, 1 thread, % of busy in gather/sort/merge
SHARES = {
    "rotation_zz\n(m = 4.5e6)": (54, 9, 37),
    "gu2q\n(m = 3.0e6)": (38, 33, 29),
    "su4\n(m = 4.24e7)": (41, 51, 8),
}


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def roofline_threads() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = range(len(THREADS))

    ax.axhspan(*READ_CEIL, color=MUTED, alpha=0.14, lw=0)
    ax.axhspan(*WRITE_CEIL, color=MUTED, alpha=0.14, lw=0)
    ax.text(0.02, sum(READ_CEIL) / 2, "measured read ceiling", va="center",
            fontsize=8.5, color=MUTED)
    ax.text(0.02, sum(WRITE_CEIL) / 2, "measured write ceiling", va="center",
            fontsize=8.5, color=MUTED)

    ax.plot(x, READ_GBPS, "-o", color=BLUE, lw=2, ms=7,
            mec="white", mew=1.5, label="read", zorder=3)
    ax.plot(x, WRITE_GBPS, "-s", color=ORANGE, lw=2, ms=7,
            mec="white", mew=1.5, label="write", zorder=3)
    for xi, v in zip(x, READ_GBPS):
        dx, dy, ha = (8, 2, "left") if xi == 0 else (0, 8, "center")
        ax.annotate(f"{v:g}", (xi, v), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=8.5, color=INK)
    for xi, v in zip(x, WRITE_GBPS):
        dx, dy, ha = (8, -9, "left") if xi == 0 else (0, -14, "center")
        ax.annotate(f"{v:g}", (xi, v), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=8.5, color=INK)

    ax.set_xticks(list(x), [str(t) for t in THREADS])
    ax.set_xlabel("threads", color=INK)
    ax.set_ylabel("attributable DRAM traffic (GB/s)", color=INK)
    ax.set_ylim(0, 55)
    ax.set_title("su4 sits on the write ceiling from 16 threads",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28),
              ncols=2, fontsize=9)
    ax.grid(axis="y", color=MUTED, alpha=0.2, lw=0.6)
    ax.set_axisbelow(True)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(HERE / "roofline-threads.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def phase_shares() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    labels = list(SHARES)
    phases = ("gather", "sort", "merge")
    colors = (ORANGE, BLUE, AQUA)

    for row, name in enumerate(labels):
        left = 0.0
        for share, color, phase in zip(SHARES[name], colors, phases):
            ax.barh(row, share, left=left, height=0.55, color=color,
                    edgecolor="white", linewidth=2,
                    label=phase if row == 0 else None)
            if share >= 6:
                ax.text(left + share / 2, row, f"{share}%", ha="center",
                        va="center", fontsize=9, color="white", fontweight="bold")
            left += share

    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.set_title("Share of worker busy time, one thread",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncols=3, fontsize=9)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, labelcolor=INK)

    fig.tight_layout()
    fig.savefig(HERE / "phase-shares.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    roofline_threads()
    phase_shares()
    print("wrote", HERE / "roofline-threads.svg", "and", HERE / "phase-shares.svg")
