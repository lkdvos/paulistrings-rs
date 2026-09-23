#!/usr/bin/env python3
"""Truncation-convergence study for the landing-page 2D Ising quench.

Heisenberg-propagates the average-X-magnetization observable through a
first-order Trotter circuit on a 6x6 periodic transverse-field Ising
lattice, at two field strengths `h` (color) and five `TopN` term caps each
(opacity: solid is the largest cap, faint is the smallest). Larger `h`
drives a larger single-qubit rotation angle per Trotter step, i.e. more
"magic" injected per step, so the two curves separate at a lower term cap
than the weak-field pair does.

Runtime is dominated by the two largest caps, which saturate the sum and pay
full sort-and-truncate cost on every layer: this script takes on the order of
ten minutes single-shot, since it is a one-off landing-page figure and not
part of the doc-snippet CI checker.

Regenerate the committed SVG with:

    .venv/bin/python docs/figures/ising-quench-convergence/quench_convergence.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paulistrings import Circuit, PauliSum, truncation

LX, LY = 6, 6
J = 1.0
DT = 0.05
N_STEPS = 16
H_VALUES = [0.3, 3.1]
H_COLORS = {0.3: "#2a78d6", 3.1: "#eb6834"}
TOPN_CAPS = [100, 1_000, 10_000, 100_000, 300_000]
TOPN_ALPHAS = {100: 0.25, 1_000: 0.45, 10_000: 0.65, 100_000: 0.85, 300_000: 1.0}

GRID_COLOR = "#e1e0d9"
MUTED_TEXT = "#898781"


def topn_label(n_cap: int) -> str:
    """`10^d` mathtext, or `m×10^d` when `n_cap` is not a bare power of ten."""
    exponent = 0
    mantissa = n_cap
    while mantissa % 10 == 0:
        mantissa //= 10
        exponent += 1
    if mantissa == 1:
        return f"$10^{{{exponent}}}$"
    return f"${mantissa}\\times10^{{{exponent}}}$"


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


def run_series(h: float, n_cap: int) -> tuple[list[float], list[float]]:
    policy = truncation.topn(n_cap)
    step_circuit = trotter_step(LX, LY, DT, J, h)
    observable = x_magnetization(LX, LY)
    ts = [0.0]
    xs = [observable.expectation("x+").real]
    for k in range(1, N_STEPS + 1):
        observable = observable.propagate(step_circuit, policy, direction="heisenberg")
        ts.append(k * DT)
        xs.append(observable.expectation("x+").real)
    print(f"h={h} topn={n_cap}: terms={len(observable)} <X>_final={xs[-1]:+.6f}")
    return ts, xs


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for h in H_VALUES:
        for n_cap in TOPN_CAPS:
            ts, xs = run_series(h, n_cap)
            ax.plot(
                ts,
                xs,
                color=H_COLORS[h],
                alpha=TOPN_ALPHAS[n_cap],
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

    topn_handles = [
        Line2D([0], [0], color=MUTED_TEXT, alpha=TOPN_ALPHAS[n_cap], linewidth=2.0, label=f"TopN = {topn_label(n_cap)}")
        for n_cap in TOPN_CAPS
    ]
    ax.legend(handles=topn_handles, title="term cap", loc="lower left", frameon=False)

    out = Path(__file__).with_name("quench_convergence.svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
