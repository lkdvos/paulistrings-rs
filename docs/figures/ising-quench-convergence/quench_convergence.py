#!/usr/bin/env python3
"""Truncation-convergence study for the landing-page 2D Ising quench.

Heisenberg-propagates the average-X-magnetization observable through a
first-order Trotter circuit on an 8x8 periodic transverse-field Ising
lattice, at three field strengths `h` (color) and three coefficient
truncation thresholds each (opacity: solid is the tightest threshold, faint
is the loosest). Larger `h` drives a larger single-qubit rotation angle per
Trotter step, i.e. further from the Clifford points -- more of the curves for
a given `h` should stay bunched together for a family that stays cheap to
truncate, and split apart earlier where truncation is already deciding the
answer.

Runtime is dominated by the tightest threshold at the largest `h`: this
script takes several minutes single-shot, since it is a one-off landing-page
figure and not part of the doc-snippet CI checker.

Regenerate the committed SVG with:

    .venv/bin/python docs/figures/ising-quench-convergence/quench_convergence.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paulistrings import Circuit, PauliSum, truncation

LX, LY = 8, 8
J = 1.0
DT = 0.05
N_STEPS = 8
H_VALUES = [0.5, 1.0, 2.0]
H_COLORS = {0.5: "#2a78d6", 1.0: "#eb6834", 2.0: "#1baf7a"}
THRESHOLDS = [1e-3, 1e-5, 1e-7]
THRESHOLD_ALPHAS = {1e-3: 0.35, 1e-5: 0.65, 1e-7: 1.0}
TOPN_CAP = 300_000

GRID_COLOR = "#e1e0d9"
MUTED_TEXT = "#898781"


def x_magnetization(lx: int, ly: int) -> PauliSum:
    n = lx * ly
    terms = {}
    for site in range(n):
        key = ["I"] * n
        key[site] = "X"
        terms["".join(key)] = 1.0 / n
    return PauliSum.from_strings(terms, num_qubits=n)


def trotter_step(lx: int, ly: int, dt: float, j_coupling: float, h: float) -> Circuit:
    n = lx * ly
    circuit = Circuit(n)

    def idx(x: int, y: int) -> int:
        return (y % ly) * lx + (x % lx)

    for y in range(ly):
        for x in range(lx):
            i = idx(x, y)
            for nx, ny in ((x + 1, y), (x, y + 1)):
                j = idx(nx, ny)
                circuit.pauli_rotation("ZZ", [i, j], 2 * j_coupling * dt)
    for site in range(n):
        circuit.rx(2 * h * dt, site)
    return circuit


def run_series(h: float, eps: float) -> tuple[list[float], list[float]]:
    policy = truncation.coeff(eps) & truncation.topn(TOPN_CAP)
    step_circuit = trotter_step(LX, LY, DT, J, h)
    observable = x_magnetization(LX, LY)
    ts = [0.0]
    xs = [observable.expectation("x+").real]
    for k in range(1, N_STEPS + 1):
        observable = observable.propagate(step_circuit, policy, direction="heisenberg")
        ts.append(k * DT)
        xs.append(observable.expectation("x+").real)
    print(f"h={h} eps={eps:.0e}: terms={len(observable)} <X>_final={xs[-1]:+.6f}")
    return ts, xs


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for h in H_VALUES:
        for eps in THRESHOLDS:
            ts, xs = run_series(h, eps)
            ax.plot(
                ts,
                xs,
                color=H_COLORS[h],
                alpha=THRESHOLD_ALPHAS[eps],
                linewidth=2.0,
            )

    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_TEXT)
    ax.tick_params(colors=MUTED_TEXT)
    ax.set_xlabel("t")
    ax.set_ylabel("⟨X⟩")

    h_handles = [Line2D([0], [0], color=H_COLORS[h], linewidth=2.0, label=f"h = {h}") for h in H_VALUES]
    h_legend = ax.legend(handles=h_handles, title="field h", loc="upper right", frameon=False)
    ax.add_artist(h_legend)

    eps_handles = [
        Line2D([0], [0], color=MUTED_TEXT, alpha=THRESHOLD_ALPHAS[eps], linewidth=2.0, label=f"ε = {eps:.0e}")
        for eps in THRESHOLDS
    ]
    ax.legend(handles=eps_handles, title="coeff threshold", loc="lower left", frameon=False)

    out = Path(__file__).with_name("quench_convergence.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
