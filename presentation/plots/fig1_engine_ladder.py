"""Figure F1: the engine ladder -- one horizontal-bar figure per stage reveal.

Reads `presentation/data/engine_ladder.jsonl` (and, optionally,
`targetcpu_default.jsonl` / `targetcpu_native.jsonl` for the "naive, native
target-cpu" stage) and draws the ordered stages of the talk's narrative arc as
horizontal bars of median total wall time (log x):

    1. naive, 1 thread
    2. naive, 1 thread, target-cpu=native           (only if targetcpu data present)
    3. per-thread maps ("threadmaps"), best of threads
    4. parallel mergesort ("mergesort"), best of threads
    5. bucketed, 1 thread
    6. bucketed, 32 threads, coarse buckets (target_bucket_len = 16384)
    7. bucketed, 32 threads, default bucket size

Each bar is labeled with its median wall time and speedup vs. the naive
baseline. `--upto K` draws only the first `K` stages; stages beyond `K` are
drawn as faint dashed outlines unless `--hide-future` removes them from the
axes entirely. `--all` renders every `K` from 1 to the stage count in one
invocation, writing `fig1_stage1.svg` ... `fig1_stageN.svg` (each also gets a
`.pdf`, via `common.save`).

Run (from the repo root, after the bench driver has written the ladder data)::

    python presentation/plots/fig1_engine_ladder.py --upto 5
    python presentation/plots/fig1_engine_ladder.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import common

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FIGSIZE = (5.6, 3.6)


def _median_wall_ns(records: list[dict], **filters) -> float | None:
    rows = [r for r in records if all(r.get(k) == v for k, v in filters.items())]
    if not rows:
        return None
    agg = common.aggregate_median(rows, group_keys=(), value_key="wall_ns")
    if not agg:
        return None
    ((median, _lo, _hi),) = agg.values()
    return median


def _best_of_threads(records: list[dict], layer: str) -> float | None:
    rows = [r for r in records if r["layer"] == layer]
    if not rows:
        return None
    by_threads = common.aggregate_median(rows, group_keys=("threads",), value_key="wall_ns")
    if not by_threads:
        return None
    return min(median for median, _lo, _hi in by_threads.values())


def build_stages(data_dir: Path) -> list[tuple[str, float | None, str]]:
    """Returns `[(label, median_wall_ns_or_None, color_key), ...]` in ladder order."""
    ladder_path = data_dir / "engine_ladder.jsonl"
    records = common.load_jsonl(ladder_path)

    stages: list[tuple[str, float | None, str]] = []
    stages.append(("naive, 1 thread", _median_wall_ns(records, layer="naive", threads=1), "naive"))

    native_default = data_dir / "targetcpu_default.jsonl"
    native_native = data_dir / "targetcpu_native.jsonl"
    if native_default.exists() and native_native.exists():
        native_records = common.load_jsonl(native_native)
        native_wall = _median_wall_ns(native_records, layer="naive", tag="native", threads=1)
        if native_wall is not None:
            stages.append(("naive, 1 thread, target-cpu=native", native_wall, "naive"))

    # The reconstructed baselines are measured in the scaling stage (thread_scaling.jsonl);
    # fall back to ladder rows if that file is absent.
    scaling_path = data_dir / "thread_scaling.jsonl"
    old_rows = common.load_jsonl(scaling_path) if scaling_path.exists() else records
    stages.append(("per-thread maps, best of threads", _best_of_threads(old_rows, "threadmaps"), "threadmaps"))
    stages.append(("parallel mergesort, best of threads", _best_of_threads(old_rows, "mergesort"), "mergesort"))

    bucketed = [r for r in records if r["layer"] == "bucketed"]
    bucketed_fine = [r for r in bucketed if common.variant_key(r) == "bucketed"]
    stages.append((
        "bucketed, 1 thread",
        _median_wall_ns(bucketed_fine, threads=1),
        "bucketed",
    ))
    stages.append((
        "bucketed, 32 threads, coarse buckets",
        _median_wall_ns([r for r in bucketed if common.variant_key(r) == "bucketed-coarse"], threads=32),
        "bucketed-coarse",
    ))
    stages.append((
        "bucketed, 32 threads, default",
        _median_wall_ns(bucketed_fine, threads=32),
        "bucketed",
    ))

    return [(label, wall, color_key) for label, wall, color_key in stages if wall is not None]


def plot_ladder(stages: list[tuple[str, float | None, str]], upto: int, hide_future: bool) -> plt.Figure:
    common.apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    n = len(stages)
    upto = min(upto, n)
    naive_wall = stages[0][1]

    visible = stages if not hide_future else stages[:upto]
    y_positions = list(range(len(visible)))[::-1]

    # Labels are real y-tick labels (outside the axes, left of every bar) so
    # they never collide with the bar fill or the value text regardless of
    # how short a bar is; only the value + speedup annotation sits inside the
    # plot, to the right of each bar's end.
    for idx, (y, (label, wall, color_key)) in enumerate(zip(y_positions, visible)):
        is_future = idx >= upto
        color = common.VARIANT_COLORS[color_key]

        if is_future:
            ax.barh(y, wall / 1e9, height=0.6, facecolor="none", edgecolor=color,
                    linewidth=1.2, linestyle="--", alpha=0.45, zorder=2)
            continue

        ax.barh(y, wall / 1e9, height=0.6, color=color, zorder=2)
        seconds = wall / 1e9
        if idx == 0:
            value_str = f"{seconds:,.2f} s (baseline)"
        else:
            value_str = f"{seconds:,.2f} s ({naive_wall / wall:.1f}x)"
        ax.text(
            seconds * 1.08, y, value_str,
            ha="left", va="center", fontsize=8.5, color=common._MUTED_TEXT, zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("median total propagate wall time (s, log scale)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for label, _wall, _ck in visible], fontsize=8.5)
    m = len(visible)
    for tick_label in ax.get_yticklabels():
        y_val = round(tick_label.get_position()[1])
        idx = m - 1 - y_val  # y_positions are m-1..0, top to bottom -> stage index
        if idx >= upto:
            tick_label.set_color(common._MUTED_TEXT)
            tick_label.set_alpha(0.6)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(visible) - 0.3)

    xmax = max(wall for _, wall, _ in stages) / 1e9
    ax.set_xlim(xmax * 3e-3, xmax * 40)

    common._style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upto", type=int, default=None, help="reveal only the first K stages")
    parser.add_argument("--all", action="store_true", help="render every K from 1..N")
    parser.add_argument("--hide-future", action="store_true", help="omit stages beyond K instead of ghosting them")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    stages = build_stages(Path(args.data_dir))
    if not stages:
        raise SystemExit(f"no usable rows found under {args.data_dir}")
    n = len(stages)

    if args.all:
        for k in range(1, n + 1):
            fig = plot_ladder(stages, upto=k, hide_future=args.hide_future)
            common.save(fig, f"fig1_stage{k}")
            plt.close(fig)
        return

    upto = args.upto if args.upto is not None else n
    fig = plot_ladder(stages, upto=upto, hide_future=args.hide_future)
    common.save(fig, "fig1_engine_ladder")


if __name__ == "__main__":
    main()
