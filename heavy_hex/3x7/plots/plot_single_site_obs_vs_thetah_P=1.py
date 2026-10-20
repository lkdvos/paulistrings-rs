#!/usr/bin/env python3
"""
Plot the kicked-Ising time series written by single_site_obs_vs_thetah.py.

Usage:
    python3 heavy_hex/3x7/plots/plot_single_site_obs_vs_thetah_P=1.py --data heavy_hex/3x7/data/single_site_obs_vs_thetah_P=1.npz
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def latex_sci(x):
    """LaTeX string for x in scientific notation, e.g. 1e-4 -> 10^{-4}."""
    mantissa, exponent = f"{x:e}".split("e")
    mantissa, exponent = float(mantissa), int(exponent)
    return rf"10^{{{exponent}}}" if mantissa == 1 else rf"{mantissa:g} \times 10^{{{exponent}}}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="the .npz written by single_site_obs_vs_thetah.py")
    parser.add_argument("--style", default=Path(__file__).resolve().parents[0] / "paper.mplstyle", help="matplotlib style file")
    args = parser.parse_args()

    path = Path(args.data)
    plt.style.use(str(args.style))

    with np.load(path) as data:
        k, cutoffs = data["k"], data["cutoffs"]
        h, dt, site = float(data["h"]), float(data["dt"]), int(data["site"])
        expectation, terms, retained = data["expectation"], data["terms"], data["retained"]

    colors = plt.colormaps["Blues"](np.linspace(0.3, 0.85, len(cutoffs)))

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), layout="constrained")
    for j, eps in enumerate(cutoffs):
        kwargs = dict(color=colors[j],  markerfacecolor=to_rgba(colors[j], alpha = 0.7), markeredgecolor = colors[j], marker="o", label=rf"$\varepsilon = {latex_sci(eps)}$")
        axes[0].plot(k, expectation[0, j], **kwargs)
        axes[1].plot(k, terms[0, j], **kwargs)
        discarded = 1 - retained[0, j]
        axes[2].plot(k, np.where(discarded > 1e-14, discarded, np.nan), **kwargs)

    axes[0].set_ylabel(rf"$\langle Z_{{{site}}} \rangle_k$")
    axes[1].set_ylabel("$N_{\mathrm{max}}$")
    axes[1].set_yscale("log")
    axes[1].minorticks_on()
    axes[1].xaxis.set_tick_params(which='minor', bottom=False)
    axes[2].set_ylabel(r"$1 - \sum_s |c_s|^2$")
    axes[2].set_yscale("log")
    axes[2].minorticks_on()
    axes[2].xaxis.set_tick_params(which='minor', bottom=False)
    for ax in axes:
        ax.set_xlabel(r"$k$")

    axes[0].legend()
    fig.savefig(path.with_suffix(".pdf"))

    plt.show()


if __name__ == "__main__":
    main()