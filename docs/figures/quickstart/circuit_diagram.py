#!/usr/bin/env python3
"""Circuit diagram for the landing-page and Manual quickstart example.

Four qubit wires, three gates in the order the code pushes them: `rz(pi/8)`
on qubit 0, `cnot(0, 1)`, then `h` on qubit 2. Qubit 3 is idle -- it carries a
seed term in the observable but the circuit never touches it, which is the
point the accompanying prose makes about which terms survive untouched.

Regenerate the committed SVG with:

    .venv/bin/python docs/figures/quickstart/circuit_diagram.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

WIRE_COLOR = "#333333"
GATE_COLOR = "#4C72B0"
N_QUBITS = 4
X_RZ, X_CNOT, X_H = 1.1, 2.7, 4.0
X_END = 5.0


def gate_box(ax, x, y, label, width=0.64, fontsize=11):
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - 0.28),
            width,
            0.56,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=GATE_COLOR,
            edgecolor="none",
        )
    )
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=fontsize, fontweight="bold")


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.6, 2.6))

    for q in range(N_QUBITS):
        y = N_QUBITS - 1 - q
        ax.plot([0, X_END], [y, y], color=WIRE_COLOR, linewidth=1.2, zorder=1)
        ax.text(-0.25, y, f"q{q}", ha="right", va="center", fontsize=10, family="monospace")

    y0, y1, y2 = N_QUBITS - 1, N_QUBITS - 2, N_QUBITS - 3

    gate_box(ax, X_RZ, y0, "Rz(π/8)", width=1.1, fontsize=10)

    ax.plot([X_CNOT, X_CNOT], [y0, y1], color=WIRE_COLOR, linewidth=1.2, zorder=1)
    ax.add_patch(Circle((X_CNOT, y0), 0.07, facecolor=WIRE_COLOR, edgecolor="none", zorder=2))
    ax.add_patch(Circle((X_CNOT, y1), 0.16, facecolor="white", edgecolor=WIRE_COLOR, linewidth=1.4, zorder=2))
    ax.plot([X_CNOT - 0.16, X_CNOT + 0.16], [y1, y1], color=WIRE_COLOR, linewidth=1.4, zorder=3)
    ax.plot([X_CNOT, X_CNOT], [y1 - 0.16, y1 + 0.16], color=WIRE_COLOR, linewidth=1.4, zorder=3)

    gate_box(ax, X_H, y2, "H")

    ax.set_xlim(-0.6, X_END + 0.2)
    ax.set_ylim(-0.7, N_QUBITS - 0.3)
    ax.set_aspect("equal")
    ax.axis("off")

    out = Path(__file__).with_name("circuit.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
