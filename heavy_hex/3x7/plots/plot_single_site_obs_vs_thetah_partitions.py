#!/usr/bin/env python3
"""
Plot the partitioning test written by single_site_obs_vs_thetah_partitions.py.

Usage:
    python3 heavy_hex/3x7/plots/plot_single_site_obs_vs_thetah_partitions.py --data heavy_hex/3x7/data/single_site_obs_vs_thetah_partitions.npz

Plots against k, one color per cutoff and one line style per P: the wall time
of the step, the rows exported between nodes (P > 1), and the speedup.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]


def latex_sci(x):
    """LaTeX string for x in scientific notation, e.g. 1e-4 -> 10^{-4}."""
    mantissa, exponent = f"{x:e}".split("e")
    mantissa, exponent = float(mantissa), int(exponent)
    return rf"10^{{{exponent}}}" if mantissa == 1 else rf"{mantissa:g} \times 10^{{{exponent}}}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="the .npz written by single_site_obs_vs_thetah_partitions.py")
    parser.add_argument("--style", default=Path(__file__).resolve().parents[0] / "paper.mplstyle", help="matplotlib style file")
    args = parser.parse_args()

    path = Path(args.data)
    plt.style.use(str(args.style))

    with np.load(path) as data:
        k, cutoffs, partition_counts = data["k"], data["cutoffs"], data["partition_counts"]
        expectation, terms = data["expectation"], data["terms"]
        seconds, exported = data["seconds"], data["exported"]

    colors = plt.colormaps["Blues"](np.linspace(0.3, 0.85, len(cutoffs)))

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), layout="constrained")
    for i, P in enumerate(partition_counts):
        for j, eps in enumerate(cutoffs):
            kwargs = dict(color=colors[j],  markerfacecolor=to_rgba(colors[j], alpha = 0.7), markeredgecolor = colors[j], marker=MARKERS[i % 4], linestyle=LINESTYLES[i % 4])
            axes[0].plot(k, seconds[i, j], **kwargs)
            if P > 1:
                axes[1].plot(k, exported[i, j], **kwargs)
                axes[2].plot(k, seconds[0, j] / seconds[i, j], **kwargs)

    axes[0].set_ylabel("wall time per step (s)")
    axes[0].set_yscale("log")
    axes[0].minorticks_on()
    axes[0].xaxis.set_tick_params(which='minor', bottom=False)
    axes[1].set_ylabel("terms exported per step")
    axes[1].set_yscale("log")
    axes[1].minorticks_on()
    axes[1].xaxis.set_tick_params(which='minor', bottom=False)
    axes[2].set_ylabel(r"speedup")
    axes[2].axhline(1, color="black", linewidth=1.5, linestyle="dashed")
    for ax in axes:
        ax.set_xlabel(r"$k$")

    cutoff_handles = [Line2D([], [], color=colors[j], marker="o", label=rf"$\varepsilon = {latex_sci(eps)}$") for j, eps in enumerate(cutoffs)]
    partition_handles = [Line2D([], [], color="black", marker=MARKERS[i % 4], linestyle=LINESTYLES[i % 4], label=rf"$P = {P}$") for i, P in enumerate(partition_counts)]
    axes[0].legend(handles=cutoff_handles)
    axes[1].legend(handles=partition_handles)
    fig.savefig(path.with_suffix(".pdf"))

    plt.show()


if __name__ == "__main__":
    main()