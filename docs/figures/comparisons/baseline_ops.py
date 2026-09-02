#!/usr/bin/env python3
"""Cross-library baseline figure for the comparisons page.

Dot plot of the committed pytest-benchmark medians in
``benchmarks/python/baseline_comparison/results.json`` (read directly, so the
figure regenerates from the data): construct and Clifford-conjugate medians
for paulistrings, qiskit ``SparsePauliOp`` and openfermion ``QubitOperator``
at 100 / 1 000 / 10 000 terms. Log time axis; competitor dots carry their
ratio to paulistrings.

Regenerate the committed SVG with:

    .venv/bin/python docs/figures/comparisons/baseline_ops.py
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "../../../benchmarks/python/baseline_comparison/results.json"

BLUE, ORANGE, AQUA = "#4C72B0", "#DD8452", "#1baf7a"
INK, MUTED = "#222222", "#666666"
LIBS = (  # (key in test name, display name, color, marker)
    ("paulistrings", "paulistrings", BLUE, "o"),
    ("qiskit", "qiskit SparsePauliOp", ORANGE, "s"),
    ("openfermion", "openfermion QubitOperator", AQUA, "^"),
)
SIZES = (100, 1000, 10000)


def load_medians() -> dict:
    data = json.loads(RESULTS.read_text())
    out = {}
    for b in data["benchmarks"]:
        m = re.match(r"test_(construct|conjugate)_(\w+)\[(\d+)\]", b["name"])
        if m:
            out[(m.group(1), m.group(2), int(m.group(3)))] = b["stats"]["median"] * 1e6
    return out


def fmt(us: float) -> str:
    return f"{us / 1000:.3g} ms" if us >= 1000 else f"{us:.3g} µs"


def main() -> None:
    med = load_medians()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)

    for ax, group, title in zip(
        axes, ("construct", "conjugate"),
        ("Construct from string terms", "Conjugate by a Clifford layer"),
    ):
        dodge = {"paulistrings": 0.0, "qiskit": -0.16, "openfermion": 0.16}
        for row, n in enumerate(SIZES):
            ax.axhline(row, color=MUTED, alpha=0.25, lw=0.7, zorder=1)
            base = med[(group, "paulistrings", n)]
            for key, _, color, marker in LIBS:
                if (group, key, n) not in med:
                    continue
                v = med[(group, key, n)]
                y = row + dodge[key]
                ax.plot(v, y, marker, color=color, ms=8, mec="white",
                        mew=1.3, zorder=3)
                label = fmt(v) if key == "paulistrings" else f"{v / base:.0f}×"
                dy = -13 if key == "openfermion" else 9
                ax.annotate(label, (v, y), xytext=(0, dy),
                            textcoords="offset points", ha="center",
                            fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_yticks(range(len(SIZES)), [f"{n:,} terms" for n in SIZES])
        ax.invert_yaxis()
        ax.set_ylim(len(SIZES) - 0.5, -0.6)
        ax.set_xlabel("median time (µs, log scale)", color=INK)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK, left=False)
        ax.grid(axis="x", color=MUTED, alpha=0.15, lw=0.6)
        ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], color=c, marker=m, ls="", ms=8, mec="white", mew=1.3,
                   label=name)
        for _, name, c, m in LIBS
    ]
    fig.legend(handles=handles, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncols=3, fontsize=9)

    fig.tight_layout()
    fig.savefig(HERE / "baseline-ops.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", HERE / "baseline-ops.svg")


if __name__ == "__main__":
    main()
