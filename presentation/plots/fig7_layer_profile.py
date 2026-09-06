"""Figure F7 (optional): per-layer profile of one bucketed run.

Searches the known data files under `--data-dir` (default: every `*.jsonl` in
`presentation/data/`) for the first row with a non-null `layer_wall_ns`
array -- normally a single hand-picked bucketed rep carries this detail, since
recording it for every rep of every file would be wasteful. Plots per-layer
wall time (ms) against layer (channel) index on top, and `terms_out` on a
stacked panel below (not a right-hand axis).

Run (from the repo root)::

    python presentation/plots/fig7_layer_profile.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _find_profiled_row(data_dir: Path) -> tuple[dict, Path] | None:
    for path in sorted(data_dir.glob("*.jsonl")):
        for r in common.load_jsonl(path):
            if r.get("layer_wall_ns") and r.get("layer") == "bucketed":
                return r, path
    return None


def plot_layer_profile(row: dict) -> plt.Figure:
    common.apply_rcparams()
    fig, (ax_time, ax_terms) = plt.subplots(
        2, 1, figsize=(common.FIGSIZE[0], 4.4), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0], "hspace": 0.1},
    )

    layer_wall_ms = np.asarray(row["layer_wall_ns"], dtype=float) / 1e6
    terms_out = np.asarray(row["terms_out"], dtype=float)
    xs = np.arange(1, len(layer_wall_ms) + 1)

    color = common.VARIANT_COLORS["bucketed"]
    ax_time.plot(xs, layer_wall_ms, color=color, linewidth=1.0, zorder=3)
    ax_time.set_ylabel("layer wall\ntime (ms)")
    common._style_axes(ax_time)

    ax_terms.plot(xs, terms_out, color=common.VARIANT_COLORS["bucketed-coarse"], linewidth=1.2, zorder=3)
    ax_terms.set_yscale("log")
    ax_terms.set_ylabel("terms out\n(log)")
    ax_terms.set_xlabel("layer (channel index)")
    common._style_axes(ax_terms)

    fig.align_ylabels([ax_time, ax_terms])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    found = _find_profiled_row(Path(args.data_dir))
    if found is None:
        raise SystemExit(
            f"no row with a non-null layer_wall_ns found under {args.data_dir}/*.jsonl "
            "-- F7 is optional and skipped without one"
        )
    row, path = found
    print(f"using {path.name}: layer={row['layer']} threads={row['threads']} rep={row['rep']}")

    fig = plot_layer_profile(row)
    common.save(fig, "fig7_layer_profile")


if __name__ == "__main__":
    main()
