"""Figure F0: term growth under the coeff-truncation sweep.

Reads `presentation/data/term_growth.jsonl` (written by `collect_term_growth.py`)
and produces two figures:

- `fig0_term_growth`: terms surviving after each layer (log y) vs. layer index,
  one line per `eps = 2**-k`, plus the untruncated curve (dashed grey, partial --
  it stops where `collect_term_growth.py`'s cap did) and reference furniture:
  light vertical bands marking the 5 Trotter steps of the kicked-Ising circuit
  (271 channels/step: 127 X-rotations + 144 ZZ-rotations, confirmed from the
  provenance block's `layers=1355=5*271`), and a horizontal dotted line at 1e6
  labeled as the talk's working point.
- `fig0_term_growth_peak`: peak terms vs. `eps` (log-log), with a fitted power
  law `peak_terms ~ eps^alpha` annotated.

Colour choice: the 9 eps curves are colour-coded by a viridis sampling (one
sample per k, `matplotlib.cm.viridis` over `np.linspace(0.15, 0.95, len(k))`)
rather than the categorical 8-color palette in `common.py` -- there are 9
series that vary along one ordered quantity (truncation strength), which is
exactly what a sequential colormap communicates and a categorical palette
does not (and the palette only has 8 slots).

Run with (from the repo root, after `collect_term_growth.py` has written its
data file)::

    python presentation/plots/fig0_term_growth.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "term_growth.jsonl"

CHANNELS_PER_TROTTER_STEP = 271  # 127 X-rotations + 144 ZZ-rotations
WORKING_POINT_TERMS = 1_000_000


def _load():
    records = common.load_jsonl(DATA_PATH)
    eps_records = [r for r in records if not r.get("untruncated")]
    untruncated = next(r for r in records if r.get("untruncated"))
    eps_records.sort(key=lambda r: r["k"])
    return eps_records, untruncated


def plot_term_growth(eps_records, untruncated) -> plt.Figure:
    common.apply_rcparams()
    fig, ax = plt.subplots(figsize=common.FIGSIZE)

    n_curves = len(eps_records)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, n_curves))

    total_layers = eps_records[0]["layers"]
    n_steps = total_layers // CHANNELS_PER_TROTTER_STEP
    assert n_steps * CHANNELS_PER_TROTTER_STEP == total_layers, (
        f"layers={total_layers} is not a multiple of "
        f"{CHANNELS_PER_TROTTER_STEP} channels/Trotter step"
    )
    for step in range(n_steps):
        if step % 2 == 0:
            continue  # shade every other step so the bands read as alternating
        lo = step * CHANNELS_PER_TROTTER_STEP
        hi = lo + CHANNELS_PER_TROTTER_STEP
        ax.axvspan(lo, hi, color=common._GRID_COLOR, alpha=0.6, zorder=0)

    for color, record in zip(colors, eps_records):
        xs = np.arange(1, record["layers"] + 1)
        ys = np.array(record["terms_per_layer"], dtype=float)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.4,
            label=rf"$2^{{-{record['k']}}}$",
        )

    xs_u = np.arange(1, untruncated["layers"] + 1)
    ys_u = np.array(untruncated["terms_per_layer"], dtype=float)
    ax.plot(
        xs_u,
        ys_u,
        color=common._MUTED_TEXT,
        linewidth=1.4,
        linestyle="--",
        label="untruncated (partial)",
    )

    ax.axhline(
        WORKING_POINT_TERMS,
        color=common._MUTED_TEXT,
        linewidth=1.0,
        linestyle=":",
    )
    ax.annotate(
        "~1e6 terms: the talk's working point",
        xy=(total_layers, WORKING_POINT_TERMS),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color=common._MUTED_TEXT,
    )

    ax.set_yscale("log")
    ax.set_xlabel("layer (channel index)")
    ax.set_ylabel("terms after layer")
    ax.set_xlim(0, total_layers)
    common._style_axes(ax)
    ax.legend(
        title=r"$\varepsilon$",
        frameon=False,
        fontsize=8,
        title_fontsize=8,
        ncol=2,
        loc="upper left",
    )
    fig.tight_layout()
    return fig


def _power_law_fit(eps: np.ndarray, peak_terms: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of `log(peak_terms) = alpha*log(eps) + log(c)`."""
    log_eps = np.log(eps)
    log_peak = np.log(peak_terms)
    alpha, log_c = np.polyfit(log_eps, log_peak, 1)
    return alpha, np.exp(log_c)


def plot_peak_terms_vs_eps(eps_records) -> plt.Figure:
    common.apply_rcparams()
    fig, ax = plt.subplots(figsize=common.FIGSIZE)

    eps = np.array([r["eps"] for r in eps_records])
    peak = np.array([r["peak_terms"] for r in eps_records], dtype=float)

    ax.plot(
        eps,
        peak,
        marker="o",
        markersize=5,
        linewidth=1.4,
        color=common._PALETTE[0],
    )

    alpha, c = _power_law_fit(eps, peak)
    eps_fit = np.array([eps.min(), eps.max()])
    ax.plot(
        eps_fit,
        c * eps_fit**alpha,
        linewidth=1.2,
        linestyle="--",
        color=common._MUTED_TEXT,
    )
    ax.annotate(
        rf"peak terms $\propto \varepsilon^{{{alpha:.2f}}}$",
        xy=(eps_fit[0], c * eps_fit[0] ** alpha),
        xytext=(10, -12),
        textcoords="offset points",
        fontsize=9,
        color=common._MUTED_TEXT,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()  # tighter truncation (smaller eps) reads left-to-right as "more terms"
    ax.set_xlabel(r"truncation cutoff $\varepsilon$")
    ax.set_ylabel("peak terms")
    common._style_axes(ax)
    fig.tight_layout()
    return fig, alpha, c


def main() -> None:
    eps_records, untruncated = _load()

    fig = plot_term_growth(eps_records, untruncated)
    common.save(fig, "fig0_term_growth")

    fig_peak, alpha, c = plot_peak_terms_vs_eps(eps_records)
    common.save(fig_peak, "fig0_term_growth_peak")

    print(f"fitted power law: peak_terms ~ {c:.3g} * eps^{alpha:.3f}")
    print("peak-term table:")
    print(f"{'k':>3} {'eps':>12} {'peak_terms':>12} {'final_terms':>12} {'wall_s':>8}")
    for r in eps_records:
        print(
            f"{r['k']:>3} {r['eps']:>12.6g} {r['peak_terms']:>12d} "
            f"{r['final_terms']:>12d} {r['wall_s']:>8.3f}"
        )


if __name__ == "__main__":
    main()
