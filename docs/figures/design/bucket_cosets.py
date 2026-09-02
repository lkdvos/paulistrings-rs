#!/usr/bin/env python3
"""Schematic of the coset decomposition of bucket-index space.

B = 16 buckets drawn as a 4x4 grid (index = 4*row + col). The channel's
realized bucket deltas h(D) = {0b0101, 0b1010} span a rank-2 subspace
S = {0, 5, 10, 15}; its cosets partition the 16 bucket indices into four
closed groups of four. Cells are colored by coset, and the arrows show one
coset's data movement: from each member bucket i to i XOR 5 and i XOR 10 -
every edge stays inside the coset, which is why a coset is an independent,
write-disjoint task.

Regenerate the committed SVG with:

    .venv/bin/python docs/figures/design/bucket_cosets.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

DELTAS = (0b0101, 0b1010)  # h(D) generators, rank 2
SPAN = sorted({a ^ b for a in (0, *DELTAS) for b in (0, *DELTAS)} | {DELTAS[0] ^ DELTAS[1]})
B = 16

# Coset id per bucket index: canonical representative = min of the coset.
coset_of = {}
for i in range(B):
    coset_of[i] = min(i ^ s for s in SPAN)
reps = sorted(set(coset_of.values()))

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
HIGHLIGHT = reps[1]  # the coset whose arrows are drawn

def cell_xy(i: int) -> tuple[float, float]:
    row, col = divmod(i, 4)
    return col, 3 - row  # index 0 top-left


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    for i in range(B):
        x, y = cell_xy(i)
        rep = coset_of[i]
        color = PALETTE[reps.index(rep)]
        emphasized = rep == HIGHLIGHT
        ax.add_patch(
            Rectangle(
                (x, y),
                1,
                1,
                facecolor=color,
                alpha=0.85 if emphasized else 0.28,
                edgecolor="white",
                linewidth=2,
            )
        )
        ax.text(
            x + 0.08,
            y + 0.80,
            f"{i:04b}",
            fontsize=9,
            family="monospace",
            color="white" if emphasized else "#333333",
        )

    # Arrows i -> i^delta within the highlighted coset, one style per delta.
    members = sorted(i for i in range(B) if coset_of[i] == HIGHLIGHT)
    seen = set()
    for delta, style in zip(DELTAS, ("-", (0, (4, 3)))):
        for i in members:
            j = i ^ delta
            if (j, i) in seen:
                continue
            seen.add((i, j))
            xi, yi = cell_xy(i)
            xj, yj = cell_xy(j)
            ax.add_patch(
                FancyArrowPatch(
                    (xi + 0.5, yi + 0.5),
                    (xj + 0.5, yj + 0.5),
                    arrowstyle="<->",
                    mutation_scale=13,
                    linewidth=1.8,
                    linestyle=style,
                    color="#1a1a1a",
                    shrinkA=10,
                    shrinkB=10,
                    connectionstyle="arc3,rad=0.18",
                )
            )

    for rep, color in zip(reps, PALETTE):
        ax.scatter([], [], marker="s", s=90, color=color, alpha=0.7, label=f"coset of {rep:04b}")
    ax.plot([], [], "-", color="#1a1a1a", label=f"i ⊕ {DELTAS[0]:04b}")
    ax.plot([], [], linestyle=(0, (4, 3)), color="#1a1a1a", label=f"i ⊕ {DELTAS[1]:04b}")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)

    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(-0.35, 4.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "16 buckets, h(D) spanning {0101, 1010}: four closed cosets\n"
        "arrows = one coset's complete data movement",
        fontsize=10,
    )

    out = Path(__file__).with_name("bucket-cosets.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
