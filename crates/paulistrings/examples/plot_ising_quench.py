"""Plot the 2D Ising quench data produced by `ising_2d_quench.rs`.

Reads CSVs from `crates/paulistrings/examples/output/`, draws both curves
on shared axes, and writes the result to
`crates/paulistrings/docs/examples/img/ising_quench.svg`. The committed
SVG is what docs.rs / GitHub Pages embeds, so regenerate it via this
script after any change to the Rust example.

Run from the repo root, after `cargo run --example ising_2d_quench --release`:

    python crates/paulistrings/examples/plot_ising_quench.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "output"
PLOT_DIR = HERE.parent / "docs" / "examples" / "img"


def load_series(path: Path) -> tuple[list[float], list[float]]:
    ts: list[float] = []
    ms: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(float(row["t"]))
            ms.append(float(row["m_x"]))
    return ts, ms


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    t4, m4 = load_series(DATA_DIR / "ising_4x4.csv")
    t6, m6 = load_series(DATA_DIR / "ising_6x6.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(t4, m4, label="4 × 4", marker="o", markersize=3, linewidth=1.4)
    ax.plot(t6, m6, label="6 × 6", marker="s", markersize=3, linewidth=1.4)
    ax.axhline(0.0, color="0.7", linewidth=0.6)
    ax.set_xlabel(r"Time  $t \cdot J$")
    ax.set_ylabel(r"$\langle X_{\mathrm{avg}} \rangle$")
    ax.set_title(
        r"Quench from $|+\rangle^{\otimes N}$  ·  "
        r"$H = -J\sum Z_iZ_j - h\sum X_i$  ·  $J = h = 1$"
    )
    ax.legend(loc="best", frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()

    out_path = PLOT_DIR / "ising_quench.svg"
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
