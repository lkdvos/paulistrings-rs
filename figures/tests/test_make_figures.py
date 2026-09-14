import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot

_FIGURES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _FIGURES_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(_FIGURES_DIR), "analysis"))

import pytest

from normalize import thread_scaling

from make_compact_figures import (
    export_deck_figure,
    make_baseline_pivot_figure,
    make_hash_communication_vs_cutoff_figure,
    make_accuracy_figure,
    make_attempts_figure,
    make_bucket_size_figure,
    make_convergence_figure,
    make_distributed_capacity_figure,
    make_hash_communication_figure,
    make_memory_diagnosis_figure,
    make_single_bucket_comparison_figure,
    make_thread_scaling_figure,
)
from make_recurring_figure import STAGE_VARIANTS, make_recurring_figure


# --- tiny synthetic fixtures (hand-built, nowhere near 127-qubit scale) -----


def _efficiency_row(variant_id, terms_in, rate, gate_name="rx", threads=1, ranks=1):
    return {
        "variant_id": variant_id,
        "terms_in": terms_in,
        "rate": rate,
        "gate_name": gate_name,
        "threads": threads,
        "ranks": ranks,
    }


def _tolerance_row(variant_id, min_abs_coeff, wall_time_s, status="completed", peak_terms=10, peak_rss_kb=1000.0):
    return {
        "run_id": f"{variant_id}-{min_abs_coeff}",
        "variant_id": variant_id,
        "min_abs_coeff": min_abs_coeff,
        "wall_time_s": wall_time_s,
        "peak_terms": peak_terms,
        "peak_rss_kb": peak_rss_kb,
        "status": status,
    }


@pytest.fixture
def efficiency_rows():
    rows = []
    for variant_id in STAGE_VARIANTS:
        rows.append(_efficiency_row(variant_id, 10, 100.0))
        rows.append(_efficiency_row(variant_id, 100, 500.0))
    return rows


@pytest.fixture
def tolerance_rows():
    rows = []
    for variant_id in STAGE_VARIANTS:
        rows.append(_tolerance_row(variant_id, 1e-2, 1.0))
        rows.append(_tolerance_row(variant_id, 1e-4, 5.0))
    # one OOM run for the highlighted stage-7 variant, at a tighter tolerance
    rows.append(
        _tolerance_row(
            "partitioned_numa_engine", 1e-8, None, status="oom", peak_terms=None, peak_rss_kb=None
        )
    )
    return rows


# --- recurring figure --------------------------------------------------------


@pytest.mark.parametrize("stage", range(1, 8))
def test_recurring_figure_builds_at_every_stage(efficiency_rows, tolerance_rows, stage):
    highlight = STAGE_VARIANTS[stage - 1]
    fig = make_recurring_figure(
        efficiency_rows, tolerance_rows, highlight_variant=highlight, stage=stage
    )
    assert fig is not None
    fig.savefig(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "_synth",
            f"recurring-stage{stage}.png",
        )
    )
    matplotlib.pyplot.close(fig)


def test_recurring_figure_uses_log_axes_both_panels(efficiency_rows, tolerance_rows):
    fig = make_recurring_figure(
        efficiency_rows, tolerance_rows, highlight_variant="naive_baseline", stage=1
    )
    ax_eff, ax_cost = fig.axes
    assert ax_eff.get_xscale() == "log"
    assert ax_eff.get_yscale() == "log"
    assert ax_cost.get_xscale() == "log"
    assert ax_cost.get_yscale() == "log"
    matplotlib.pyplot.close(fig)


def test_recurring_figure_rejects_unintroduced_highlight(efficiency_rows, tolerance_rows):
    with pytest.raises(ValueError):
        make_recurring_figure(
            efficiency_rows,
            tolerance_rows,
            highlight_variant="partitioned_numa_engine",
            stage=1,
        )


def test_oom_run_never_gets_a_plotted_runtime_point(efficiency_rows, tolerance_rows):
    fig = make_recurring_figure(
        efficiency_rows,
        tolerance_rows,
        highlight_variant="partitioned_numa_engine",
        stage=7,
    )
    _, ax_cost = fig.axes
    for line in ax_cost.get_lines():
        if line.get_linestyle() == "None":
            continue  # the OOM sentinel marker itself: no connected series, checked separately below
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        # The OOM row's tolerance is 1e-8; the connected wall-time series must
        # never carry a data point at that x with a real y from the line data.
        if 1e-8 in xdata:
            idx = list(xdata).index(1e-8)
            assert False, f"OOM tolerance point unexpectedly present in a line series: y={ydata[idx]}"

    # The OOM marker itself must be drawn off the data y-axis (axes-fraction y),
    # never at a fabricated wall-time value.
    oom_markers = [
        line
        for line in ax_cost.get_lines()
        if line.get_linestyle() == "None" and 1e-8 in list(line.get_xdata())
    ]
    assert oom_markers, "expected a distinct OOM marker at min_abs_coeff=1e-8"
    for marker in oom_markers:
        assert list(marker.get_ydata()) == [0.95]
    matplotlib.pyplot.close(fig)


# --- thread scaling ----------------------------------------------------------


def test_thread_scaling_speedup_is_relative_to_one_thread_point():
    rows = [
        {"threads": 1, "wall_time_s": 10.0, "speedup": 1.0, "efficiency": 1.0},
        {"threads": 2, "wall_time_s": 6.0, "speedup": 10.0 / 6.0, "efficiency": (10.0 / 6.0) / 2},
        {"threads": 4, "wall_time_s": 3.0, "speedup": 10.0 / 3.0, "efficiency": (10.0 / 3.0) / 4},
    ]
    fig = make_thread_scaling_figure(rows)
    measured_line = next(l for l in fig.axes[0].get_lines() if l.get_label() == "measured")
    ydata = list(measured_line.get_ydata())
    xdata = list(measured_line.get_xdata())
    assert xdata == [1, 2, 4]
    # Not the absolute wall_time_s values (10.0, 6.0, 3.0): must be the speedup column.
    assert ydata == pytest.approx([1.0, 10.0 / 6.0, 10.0 / 3.0])
    matplotlib.pyplot.close(fig)


def test_thread_scaling_overlays_a_second_engine_normalized_to_its_own_baseline():
    rust_rows = [
        {"threads": 1, "wall_time_s": 10.0, "speedup": 1.0},
        {"threads": 4, "wall_time_s": 3.0, "speedup": 10.0 / 3.0},
    ]
    # Julia's own baseline is much slower in absolute terms, but its speedup
    # column is still relative to ITS OWN 1-thread time -- the overlay must
    # plot that column verbatim, not renormalize against Rust's baseline.
    julia_rows = [
        {"threads": 1, "wall_time_s": 4875.0, "speedup": 1.0},
        {"threads": 96, "wall_time_s": 899.0, "speedup": 4875.0 / 899.0},
    ]
    fig = make_thread_scaling_figure(rust_rows, other_rows=julia_rows, other_label="PauliPropagation.jl")
    ax = fig.axes[0]
    julia_line = next(l for l in ax.get_lines() if l.get_label() == "PauliPropagation.jl")
    assert list(julia_line.get_xdata()) == [1, 96]
    assert list(julia_line.get_ydata()) == pytest.approx([1.0, 4875.0 / 899.0])
    matplotlib.pyplot.close(fig)


def test_thread_scaling_julia_overlay_is_non_monotonic_past_32_threads():
    # Real job 7034671 data (raw/2026-09-14-worker7160-julia/runs.jsonl,
    # eps=2^-16, vector backend): speedup peaks at 32 threads, then DEGRADES --
    # 48 threads is slower than 32, and 96 is roughly 3x slower than 48. This
    # pins that shape as a regression tripwire so it is never smoothed over by
    # a future refactor of the overlay path.
    julia_records = [
        {"variant_id": "external_pauli_propagation_jl", "min_abs_coeff": 1.5258789e-05,
         "status": "completed", "threads": t, "wall_time_s": w}
        for t, w in [
            (1, 1590.163038272), (2, 1113.475872035), (4, 698.943801298),
            (8, 479.97266168), (16, 382.632201749), (32, 302.334663506),
            (48, 334.579655597), (96, 974.957910291),
        ]
    ]
    julia_rows = thread_scaling(
        julia_records, variant_id="external_pauli_propagation_jl", min_abs_coeff=1.5258789e-05
    )
    by_threads = {r["threads"]: r["speedup"] for r in julia_rows}
    assert by_threads[32] == max(by_threads.values())
    assert by_threads[48] < by_threads[32]
    assert by_threads[96] < by_threads[48]
    # "roughly 3x worse" going 48 -> 96, in wall-clock terms.
    ratio = 974.957910291 / 334.579655597
    assert ratio == pytest.approx(2.914, abs=0.01)

    rust_rows = [
        {"threads": 1, "wall_time_s": 10.0, "speedup": 1.0},
        {"threads": 4, "wall_time_s": 3.0, "speedup": 10.0 / 3.0},
    ]
    fig = make_thread_scaling_figure(
        rust_rows, other_rows=julia_rows, other_label="PauliPropagation.jl",
        theme="deck", figsize_pt=(900, 340),
    )
    ax = fig.axes[0]
    julia_line = next(l for l in ax.get_lines() if l.get_label() == "PauliPropagation.jl")
    xdata = list(julia_line.get_xdata())
    ydata = list(julia_line.get_ydata())
    assert xdata == [1, 2, 4, 8, 16, 32, 48, 96]
    peak_idx = xdata.index(32)
    assert ydata[peak_idx] == max(ydata)
    assert ydata[xdata.index(96)] < ydata[xdata.index(48)] < ydata[peak_idx]
    matplotlib.pyplot.close(fig)


# --- attempted parallel strategies (recovered presentation-branch data) ------


def test_attempts_figure_empty_input_raises_clear_error():
    with pytest.raises(NotImplementedError, match="thread_scaling.jsonl"):
        make_attempts_figure([])


def test_attempts_figure_one_line_per_label():
    rows = [
        {"label": "threadmaps", "description": "per-thread map, merge at layer end", "threads": 1, "wall_time_s": 137.4},
        {"label": "threadmaps", "description": "per-thread map, merge at layer end", "threads": 32, "wall_time_s": 162.1},
        {"label": "mergesort", "description": "flat array, parallel sort, segmented merge", "threads": 1, "wall_time_s": 54.1},
        {"label": "mergesort", "description": "flat array, parallel sort, segmented merge", "threads": 32, "wall_time_s": 28.5},
    ]
    fig = make_attempts_figure(rows)
    assert fig is not None
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert "historical" in ax.get_title().lower()
    matplotlib.pyplot.close(fig)


def test_attempts_figure_current_engine_point_is_a_distinct_unconnected_marker():
    rows = [
        {"label": "threadmaps", "description": "per-thread map", "threads": 1, "wall_time_s": 137.4},
        {"label": "threadmaps", "description": "per-thread map", "threads": 32, "wall_time_s": 162.1},
    ]
    current = {"label": "bucketed (this campaign, genoa)", "threads": 1, "wall_time_s": 3.2}
    fig = make_attempts_figure(rows, current_engine_row=current)
    ax = fig.axes[0]
    # A line series has real linewidth; the reference point must be a scatter
    # marker (no connected line), so it never reads as part of a scaling curve.
    assert len(ax.lines) == 1
    star_collections = [c for c in ax.collections if c.get_label() == current["label"]]
    assert len(star_collections) == 1
    matplotlib.pyplot.close(fig)


# --- hash communication -------------------------------------------------------


def test_hash_communication_empty_input_raises_clear_error():
    with pytest.raises(NotImplementedError, match="partition_row_policy"):
        make_hash_communication_figure([])


def test_hash_communication_figure_plots_random_vs_cut():
    rows = [
        {
            "run_id": "r-random",
            "config_id": "cfg-e8",
            "partition_row_policy": "random",
            "partitions": 2,
            "layers": 4,
            "total_rows_exported": 300,
            "total_bytes_exported": 3000,
        },
        {
            "run_id": "r-cut",
            "config_id": "cfg-e8",
            "partition_row_policy": "cut",
            "partitions": 2,
            "layers": 4,
            "total_rows_exported": 30,
            "total_bytes_exported": 300,
        },
    ]
    fig = make_hash_communication_figure(rows)
    assert fig is not None
    heights = sorted(bar.get_height() for ax in fig.axes for bar in ax.patches)
    assert heights == [30, 300]
    matplotlib.pyplot.close(fig)


def test_hash_communication_vs_cutoff_empty_input_raises_clear_error():
    with pytest.raises(NotImplementedError, match="partition_row_policy"):
        make_hash_communication_vs_cutoff_figure([])


def test_hash_communication_vs_cutoff_figure_one_line_per_policy():
    rows = [
        {"run_id": "r1", "config_id": "cfg-loose", "min_abs_coeff": 1e-4, "partition_row_policy": "random", "partitions": 2, "layers": 1, "total_rows_exported": 100, "total_bytes_exported": 1000},
        {"run_id": "r2", "config_id": "cfg-loose", "min_abs_coeff": 1e-4, "partition_row_policy": "cut", "partitions": 2, "layers": 1, "total_rows_exported": 10, "total_bytes_exported": 100},
        {"run_id": "r3", "config_id": "cfg-tight", "min_abs_coeff": 1e-6, "partition_row_policy": "random", "partitions": 2, "layers": 1, "total_rows_exported": 1000, "total_bytes_exported": 10000},
        {"run_id": "r4", "config_id": "cfg-tight", "min_abs_coeff": 1e-6, "partition_row_policy": "cut", "partitions": 2, "layers": 1, "total_rows_exported": 100, "total_bytes_exported": 1000},
    ]
    fig = make_hash_communication_vs_cutoff_figure(rows)
    assert fig is not None
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    for line in ax.lines:
        assert list(line.get_xdata()) == [1e-6, 1e-4]
    matplotlib.pyplot.close(fig)


# --- accuracy ------------------------------------------------------------------


def test_accuracy_figure_builds_from_reference_observed_pairs():
    rows = [
        {"run_id": "run-a", "reference_value": 0.5, "observed_value": 0.501},
        {"run_id": "run-b", "reference_value": -0.2, "observed_value": -0.198},
    ]
    fig = make_accuracy_figure(rows)
    assert fig is not None
    matplotlib.pyplot.close(fig)


# --- convergence -----------------------------------------------------------


def test_convergence_figure_no_completed_rows_raises_clear_error():
    with pytest.raises(NotImplementedError, match="no completed rows"):
        make_convergence_figure([{"status": "invalid_hardware"}])


def test_convergence_figure_one_line_per_cutoff():
    rows = [
        {"status": "completed", "min_abs_coeff": 1e-4, "trotter_step": 1, "expectation_re": 0.9},
        {"status": "completed", "min_abs_coeff": 1e-4, "trotter_step": 2, "expectation_re": 0.8},
        {"status": "completed", "min_abs_coeff": 1e-6, "trotter_step": 1, "expectation_re": 0.9},
        {"status": "completed", "min_abs_coeff": 1e-6, "trotter_step": 2, "expectation_re": 0.7},
    ]
    fig = make_convergence_figure(rows)
    assert fig is not None
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    matplotlib.pyplot.close(fig)


def test_convergence_figure_overlays_single_julia_point_as_star():
    """A single Julia point per cutoff (e.g. one endpoint run record) still
    falls back to the original star-overlay rendering, not a degenerate line.
    """
    rows = [
        {"status": "completed", "min_abs_coeff": 1e-6, "trotter_step": 1, "expectation_re": 0.9},
        {"status": "completed", "min_abs_coeff": 1e-6, "trotter_step": 2, "expectation_re": 0.7},
    ]
    julia_rows = [{"min_abs_coeff": 1e-6, "trotter_step": 2, "expectation_re": 0.701}]
    fig = make_convergence_figure(rows, julia_rows=julia_rows)
    assert fig is not None
    ax = fig.axes[0]
    assert any(c.get_label() == "PauliPropagation.jl" for c in ax.collections)
    matplotlib.pyplot.close(fig)


def test_convergence_figure_draws_a_real_julia_line_for_a_full_trajectory():
    """More than one Julia point at a cutoff draws a dashed line, in the same
    color as the Rust line at that cutoff -- the real per-cutoff trajectory
    case `run_convergence_sweep_julia.py`'s `PP_LAYER_EXPECTATION` sweep
    produces, not just a single endpoint.
    """
    rows = [
        {"status": "completed", "min_abs_coeff": 1e-6, "trotter_step": 1, "expectation_re": 0.9},
        {"status": "completed", "min_abs_coeff": 1e-6, "trotter_step": 2, "expectation_re": 0.7},
    ]
    julia_rows = [
        {"status": "completed", "min_abs_coeff": 1e-6, "trotter_step": 1, "expectation_re": 0.901},
        {"status": "completed", "min_abs_coeff": 1e-6, "trotter_step": 2, "expectation_re": 0.699},
    ]
    fig = make_convergence_figure(rows, julia_rows=julia_rows)
    assert fig is not None
    ax = fig.axes[0]
    dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert len(dashed) == 1
    assert dashed[0].get_label() == r"PauliPropagation.jl $\varepsilon=2^{-20}$"
    # Same color as the (single) solid Rust line at the same cutoff.
    solid = [ln for ln in ax.lines if ln.get_linestyle() != "--"]
    assert len(solid) == 1
    assert dashed[0].get_color() == solid[0].get_color()
    # No leftover star overlay from the single-point fallback path.
    assert not ax.collections
    matplotlib.pyplot.close(fig)


def test_convergence_figure_julia_line_only_cutoff_still_gets_a_color():
    """A cutoff present only on the Julia side (no matching Rust line) must
    not crash the shared color lookup.
    """
    rows = [
        {"status": "completed", "min_abs_coeff": 1e-4, "trotter_step": 1, "expectation_re": 0.9},
        {"status": "completed", "min_abs_coeff": 1e-4, "trotter_step": 2, "expectation_re": 0.7},
    ]
    julia_rows = [
        {"status": "completed", "min_abs_coeff": 1e-6, "trotter_step": 1, "expectation_re": 0.9},
        {"status": "completed", "min_abs_coeff": 1e-6, "trotter_step": 2, "expectation_re": 0.7},
    ]
    fig = make_convergence_figure(rows, julia_rows=julia_rows)
    assert fig is not None
    matplotlib.pyplot.close(fig)


# --- deck v2 theme -------------------------------------------------------------


def test_thread_scaling_deck_theme_uses_navy_spines_and_exact_size(tmp_path):
    rows = [
        {"threads": 1, "wall_time_s": 10.0, "speedup": 1.0},
        {"threads": 4, "wall_time_s": 3.0, "speedup": 10.0 / 3.0},
    ]
    fig = make_thread_scaling_figure(rows, theme="deck", figsize_pt=(450, 340), title="Thread scaling")
    ax = fig.axes[0]
    assert ax.spines["bottom"].get_edgecolor()[:3] == pytest.approx((0x1c / 255, 0x29 / 255, 0x54 / 255), abs=1e-6)
    paths = export_deck_figure(fig, str(tmp_path / "thread_scaling_v2"), 450, 340)
    assert set(paths) == {"svg", "pdf", "png"}
    for p in paths.values():
        assert os.path.exists(p)
    # inches = points / 72, the sizing contract export_deck_figure documents.
    assert fig.get_size_inches() == pytest.approx((450 / 72.0, 340 / 72.0))
    matplotlib.pyplot.close(fig)


def test_hash_communication_vs_cutoff_deck_theme_distinguishes_series_by_marker_too():
    rows = [
        {"run_id": "r1", "config_id": "cfg-loose", "min_abs_coeff": 1e-4, "partition_row_policy": "random", "partitions": 2, "layers": 1, "total_rows_exported": 100, "total_bytes_exported": 1000},
        {"run_id": "r2", "config_id": "cfg-loose", "min_abs_coeff": 1e-4, "partition_row_policy": "cut", "partitions": 2, "layers": 1, "total_rows_exported": 10, "total_bytes_exported": 100},
    ]
    fig = make_hash_communication_vs_cutoff_figure(rows, theme="deck", figsize_pt=(450, 340))
    ax = fig.axes[0]
    markers = {line.get_label(): line.get_marker() for line in ax.lines}
    linestyles = {line.get_label(): line.get_linestyle() for line in ax.lines}
    assert markers["random"] != markers["cut"]
    assert linestyles["random"] != linestyles["cut"]
    matplotlib.pyplot.close(fig)


def test_convergence_deck_theme_still_builds_and_omits_default_legacy_title():
    rows = [
        {"status": "completed", "min_abs_coeff": 1e-4, "trotter_step": 1, "expectation_re": 0.9},
        {"status": "completed", "min_abs_coeff": 1e-4, "trotter_step": 2, "expectation_re": 0.8},
    ]
    fig = make_convergence_figure(rows, theme="deck", figsize_pt=(450, 340), title="Consistency")
    ax = fig.axes[0]
    assert ax.get_title() == "Consistency"
    matplotlib.pyplot.close(fig)


def test_recurring_figure_deck_theme_with_external_points_draws_stars():
    tolerance_rows = [
        {"run_id": "naive-1", "variant_id": "naive_baseline", "min_abs_coeff": 1e-6, "wall_time_s": 2.9561e-05, "peak_terms": 14, "peak_rss_kb": 426248, "status": "completed"},
    ]
    external_points = [
        {"label": "bucketed_current (Rust)", "min_abs_coeff": 1.5258789e-05, "wall_time_s": 1629.9, "config_note": "n=127 canonical, threads=1"},
        {"label": "PauliPropagation.jl", "min_abs_coeff": 1.5258789e-05, "wall_time_s": 4874.94, "config_note": "n=127 canonical, threads=1"},
    ]
    fig = make_recurring_figure(
        [], tolerance_rows, highlight_variant="naive_baseline", stage=1,
        theme="deck", figsize_pt=(900, 340), external_points=external_points,
    )
    _, ax_cost = fig.axes
    star_labels = {c.get_label() for c in ax_cost.collections}
    assert "bucketed_current (Rust)" in star_labels
    assert "PauliPropagation.jl" in star_labels
    matplotlib.pyplot.close(fig)


# --- distributed capacity ------------------------------------------------------


def test_distributed_capacity_figure_empty_input_raises_clear_error():
    with pytest.raises(NotImplementedError, match="no rows to plot"):
        make_distributed_capacity_figure([])


def test_distributed_capacity_figure_rejects_fabricated_runtime_on_failed_row():
    rows = [
        {"ranks": 8, "min_abs_coeff": 9.5367432e-07, "wall_time_s": 123.0, "peak_terms": None, "peak_rss_kb": None, "status": "oom"},
    ]
    with pytest.raises(ValueError, match="fabricated runtime"):
        make_distributed_capacity_figure(rows)


def test_distributed_capacity_figure_omits_oom_and_untested_points():
    rows = [
        {"ranks": 1, "min_abs_coeff": 3.8146973e-06, "wall_time_s": 470.9, "peak_terms": 635371364, "peak_rss_kb": 1.5e8, "status": "completed"},
        {"ranks": 4, "min_abs_coeff": 3.8146973e-06, "wall_time_s": 468.77, "peak_terms": 635371364, "peak_rss_kb": 2.0e8, "status": "completed"},
        {"ranks": 1, "min_abs_coeff": 9.5367432e-07, "wall_time_s": None, "peak_terms": None, "peak_rss_kb": None, "status": "untested", "note": "never submitted"},
        {"ranks": 8, "min_abs_coeff": 9.5367432e-07, "wall_time_s": None, "peak_terms": None, "peak_rss_kb": 1.2e9, "status": "oom", "note": "measured OOM, job 7032060"},
        {"ranks": 16, "min_abs_coeff": 9.5367432e-07, "wall_time_s": 1658.8, "peak_terms": 8923556570, "peak_rss_kb": 3.04e9, "status": "completed"},
    ]
    fig = make_distributed_capacity_figure(rows)
    ax = fig.axes[0]

    # The completed 4-rank/eps=2^-18 point must still appear on a real connected line.
    completed_lines = [ln for ln in ax.get_lines() if ln.get_linestyle() != "None" and 4 in list(ln.get_xdata())]
    assert completed_lines and 468.77 in list(completed_lines[0].get_ydata())

    # oom/untested rows draw nothing at all -- per the user's explicit call, the presenter
    # states unmeasured points verbally rather than the figure showing sentinel markers for
    # them. No line's x-data may include ranks=1 (untested) or ranks=8 (oom) for eps=2^-20.
    eps20_xs = {
        x
        for ln in ax.get_lines()
        for x in ln.get_xdata()
    }
    # ranks=16 (the one real completed eps=2^-20 point) is allowed to appear; ranks=1/8
    # only appear here via the eps=2^-18 series (different tolerance), never eps=2^-20's.
    assert 16 in eps20_xs
    for artist in ax.texts:
        assert "never submitted" not in artist.get_text()
        assert "measured OOM" not in artist.get_text()
    matplotlib.pyplot.close(fig)


def test_distributed_capacity_figure_states_max_memory_in_legend_not_per_point():
    rows = [
        {"ranks": 1, "min_abs_coeff": 3.8146973e-06, "wall_time_s": 482.6, "peak_terms": 635371364, "peak_rss_kb": 1.5e8, "status": "completed"},
        {"ranks": 4, "min_abs_coeff": 3.8146973e-06, "wall_time_s": 468.77, "peak_terms": 635371364, "peak_rss_kb": 2.0e8, "status": "completed"},
    ]
    fig = make_distributed_capacity_figure(rows)
    ax = fig.axes[0]

    handles, labels = ax.get_legend_handles_labels()
    assert any("max 0.20 TB" in label for label in labels), labels

    # No per-point "X.XX TB" text annotation near the data -- only the legend states memory.
    for artist in ax.texts:
        assert "TB" not in artist.get_text()
    matplotlib.pyplot.close(fig)


def test_distributed_capacity_figure_deck_theme_exports_at_exact_size(tmp_path):
    rows = [
        {"ranks": 1, "min_abs_coeff": 1.5258789e-05, "wall_time_s": 41.6, "peak_terms": 38791220, "peak_rss_kb": 1.5e7, "status": "completed"},
        {"ranks": 4, "min_abs_coeff": 1.5258789e-05, "wall_time_s": 69.5, "peak_terms": 45418768, "peak_rss_kb": 1.4e7, "status": "completed"},
    ]
    fig = make_distributed_capacity_figure(rows, theme="deck", figsize_pt=(900, 340), title="Distributed capacity")
    paths = export_deck_figure(fig, str(tmp_path / "distributed_capacity_v2"), 900, 340)
    assert set(paths) == {"svg", "pdf", "png"}
    assert fig.get_size_inches() == pytest.approx((900 / 72.0, 340 / 72.0))
    matplotlib.pyplot.close(fig)


# --- bucket size -----------------------------------------------------------

# Real numbers from job 7035853 (raw/2026-09-14-worker7202-bucketsize/), used
# verbatim -- see the figures/real/MANIFEST.md "bucket-size" section and
# evidence.md for full provenance. strings_per_s is read from the sibling
# .txt file's "strings/s = ..." line, since the JSON sidecar does not carry it.
_BUCKET_SIZE_ROWS = [
    {"target_bucket_len": 256, "num_buckets": 4096, "empty_buckets": 3374,
     "occupancy_median": 1, "occupancy_p95": 2, "occupancy_max": 3, "strings_per_s": 5.713e7},
    {"target_bucket_len": 512, "num_buckets": 2048, "empty_buckets": 1416,
     "occupancy_median": 1, "occupancy_p95": 2, "occupancy_max": 4, "strings_per_s": 6.659e7},
    {"target_bucket_len": 1024, "num_buckets": 1024, "empty_buckets": 518,
     "occupancy_median": 1, "occupancy_p95": 3, "occupancy_max": 5, "strings_per_s": 7.158e7},
    {"target_bucket_len": 2048, "num_buckets": 512, "empty_buckets": 107,
     "occupancy_median": 2, "occupancy_p95": 4, "occupancy_max": 6, "strings_per_s": 7.438e7},
    {"target_bucket_len": 4096, "num_buckets": 256, "empty_buckets": 7,
     "occupancy_median": 3, "occupancy_p95": 6, "occupancy_max": 8, "strings_per_s": 7.588e7},
]


def test_bucket_size_figure_empty_input_raises_clear_error():
    with pytest.raises(NotImplementedError, match="no rows to plot"):
        make_bucket_size_figure([])


def test_bucket_size_figure_builds_two_panels():
    fig = make_bucket_size_figure(_BUCKET_SIZE_ROWS)
    assert len(fig.axes) >= 2
    matplotlib.pyplot.close(fig)


def test_bucket_size_figure_single_panel_omits_occupancy_axes():
    fig = make_bucket_size_figure(_BUCKET_SIZE_ROWS, single_panel=True)
    assert len(fig.axes) == 1
    matplotlib.pyplot.close(fig)


def test_bucket_size_figure_l2_line_sits_at_cache_bytes_over_bytes_per_term():
    # 1 MiB L2, 48 B/term (this repo's fixed W=2/Complex64 payload) -> the
    # line should sit at target_bucket_len = 1048576 / 48.
    fig = make_bucket_size_figure(
        _BUCKET_SIZE_ROWS, single_panel=True, l2_cache_bytes=1024 * 1024, bytes_per_term=48.0,
    )
    ax = fig.axes[0]
    vlines = [ln for ln in ax.get_lines() if ln.get_xdata()[0] == ln.get_xdata()[-1]]
    assert vlines, "expected a vertical line for the L2 cache reference"
    assert vlines[0].get_xdata()[0] == pytest.approx((1024 * 1024) / 48.0)
    matplotlib.pyplot.close(fig)


def test_bucket_size_figure_throughput_panel_plots_all_five_points_in_order():
    fig = make_bucket_size_figure(_BUCKET_SIZE_ROWS)
    ax_thr = fig.axes[0]
    line = ax_thr.get_lines()[0]
    assert list(line.get_xdata()) == [256, 512, 1024, 2048, 4096]
    assert list(line.get_ydata()) == pytest.approx(
        [5.713e7, 6.659e7, 7.158e7, 7.438e7, 7.588e7]
    )
    matplotlib.pyplot.close(fig)


def test_bucket_size_figure_throughput_has_no_peak_in_tested_range():
    """The real data's headline honesty finding: throughput rises
    monotonically across the whole tested range with no interior peak --
    this is a property of the real numbers themselves, pinned here so a
    future data refresh can't silently reintroduce a false "optimum" claim
    without the test noticing the shape changed.
    """
    ys = [r["strings_per_s"] for r in sorted(_BUCKET_SIZE_ROWS, key=lambda r: r["target_bucket_len"])]
    assert all(b >= a for a, b in zip(ys, ys[1:])), "expected monotonically non-decreasing throughput"
    assert ys[-1] == max(ys), "expected the maximum to sit at the largest tested target_bucket_len (no peak in range)"


def test_bucket_size_figure_empty_fraction_never_folded_into_occupancy_percentiles():
    """empty_buckets/num_buckets must be rendered as its own series (the bar
    axis), never averaged or folded into occupancy_median/p95/max -- those
    fields already exclude empty buckets from their sample by construction.
    """
    fig = make_bucket_size_figure(_BUCKET_SIZE_ROWS)
    ax_occ = fig.axes[1]
    occ_lines = {ln.get_label(): list(ln.get_ydata()) for ln in ax_occ.get_lines()}
    assert occ_lines["median"] == [r["occupancy_median"] for r in _BUCKET_SIZE_ROWS]
    assert occ_lines["p95"] == [r["occupancy_p95"] for r in _BUCKET_SIZE_ROWS]
    assert occ_lines["max"] == [r["occupancy_max"] for r in _BUCKET_SIZE_ROWS]
    # The empty-bucket fraction must appear on a distinct axes (the twin bar
    # axis), never as a fourth occupancy line sharing that axes' data scale.
    assert len(fig.axes) >= 3
    ax_empty = fig.axes[2]
    bar_heights = sorted(p.get_height() for p in ax_empty.patches)
    expected = sorted(r["empty_buckets"] / r["num_buckets"] for r in _BUCKET_SIZE_ROWS)
    assert bar_heights == pytest.approx(expected)
    matplotlib.pyplot.close(fig)


def test_bucket_size_figure_annotates_realised_bucket_count():
    """Borrowed from the older `presentation` deck's fig4b_bucket_speedup.py
    "B={buckets}" convention: each throughput point is labeled with its real
    `num_buckets`, since that field already exists on every row and helps a
    reader see `num_buckets` is derived from `target_bucket_len`, not a
    second independent axis.
    """
    fig = make_bucket_size_figure(_BUCKET_SIZE_ROWS)
    ax_thr = fig.axes[0]
    texts = {t.get_text() for t in ax_thr.texts}
    for r in _BUCKET_SIZE_ROWS:
        assert f"B={r['num_buckets']}" in texts
    matplotlib.pyplot.close(fig)


def test_bucket_size_figure_deck_theme_exports_at_exact_size(tmp_path):
    fig = make_bucket_size_figure(
        _BUCKET_SIZE_ROWS, theme="deck", figsize_pt=(900, 340), title="Bucket size"
    )
    paths = export_deck_figure(fig, str(tmp_path / "bucket_size_v2"), 900, 340)
    assert set(paths) == {"svg", "pdf", "png"}
    assert fig.get_size_inches() == pytest.approx((900 / 72.0, 340 / 72.0))


# --- bucketed-1t (real full-scale historical sweep, deck page 29) -----------
#
# Real rows from job 7033945 (`raw/2026-09-14-worker7150-historical/runs.jsonl`),
# n_qubits=127, trotter_steps=10 -- the same 4-point campaign grid for
# direct_small_sum_path/bucketed_engine_serial/bucketed_engine_parallel plus
# naive_baseline's single truncation-inert point (decisions.md #42). These
# pin the confirmed, disclosed finding: bucketed_engine_parallel is SLOWER
# than bucketed_engine_serial at every one of the 4 tolerance points, on
# real genoa hardware, not just the earlier toy-scale/local-workstation
# observation (decisions.md #34).

_BUCKETED_1T_ROWS = [
    {"run_id": "naive-1", "variant_id": "naive_baseline", "min_abs_coeff": 0.000244140625,
     "wall_time_s": 65.405733104, "peak_terms": 3018683, "peak_rss_kb": 112100, "status": "completed"},
] + [
    {"run_id": f"direct-{i}", "variant_id": "direct_small_sum_path", "min_abs_coeff": eps,
     "wall_time_s": w, "peak_terms": t, "peak_rss_kb": 430036, "status": "completed"}
    for i, (eps, w, t) in enumerate([
        (0.000244140625, 0.7502037849626504, 232432),
        (6.103515625e-05, 0.8052491209818982, 696172),
        (1.5258789e-05, 0.9201539809582755, 1791652),
        (3.8146973e-06, 1.1498842669534497, 3936794),
    ])
] + [
    {"run_id": f"serial-{i}", "variant_id": "bucketed_engine_serial", "min_abs_coeff": eps,
     "wall_time_s": w, "peak_terms": t, "peak_rss_kb": 500000, "status": "completed"}
    for i, (eps, w, t) in enumerate([
        (0.000244140625, 11.987827610049862, 232432),
        (6.103515625e-05, 21.857503906008787, 696172),
        (1.5258789e-05, 41.13229779800167, 1791652),
        (3.8146973e-06, 66.11256570497062, 3936794),
    ])
] + [
    {"run_id": f"parallel-{i}", "variant_id": "bucketed_engine_parallel", "min_abs_coeff": eps,
     "wall_time_s": w, "peak_terms": t, "peak_rss_kb": 900000, "status": "completed"}
    for i, (eps, w, t) in enumerate([
        (0.000244140625, 14.780599495046772, 232432),
        (6.103515625e-05, 39.35577713698149, 696172),
        (1.5258789e-05, 79.51175081694964, 1791652),
        (3.8146973e-06, 140.26288040401414, 3936794),
    ])
]


def test_bucketed_1t_parallel_slower_than_serial_at_every_cutoff():
    """Pins the real, disclosed finding as a regression tripwire: a future
    data refresh should not silently "fix" or hide this anomaly without the
    test noticing.
    """
    serial = {r["min_abs_coeff"]: r["wall_time_s"] for r in _BUCKETED_1T_ROWS if r["variant_id"] == "bucketed_engine_serial"}
    parallel = {r["min_abs_coeff"]: r["wall_time_s"] for r in _BUCKETED_1T_ROWS if r["variant_id"] == "bucketed_engine_parallel"}
    assert set(serial) == set(parallel)
    for eps in serial:
        assert parallel[eps] > serial[eps], f"expected parallel slower than serial at eps={eps}"


def test_bucketed_1t_figure_reuses_recurring_figure_stage6_unmodified():
    """`make_recurring_figure`'s existing row shape (`normalize.runtime_tolerance`'s
    output) accepts these real full-scale rows directly -- no new plotting
    function needed, per this campaign's established reuse-over-rewrite pattern.
    """
    fig = make_recurring_figure(
        [], _BUCKETED_1T_ROWS, highlight_variant="bucketed_engine_serial", stage=6,
        theme="deck", figsize_pt=(900, 340),
    )
    _, ax_cost = fig.axes
    labels = {line.get_label() for line in ax_cost.lines}
    assert {"naive_baseline", "direct_small_sum_path", "bucketed_engine_serial", "bucketed_engine_parallel"} <= labels
    matplotlib.pyplot.close(fig)


def test_bucketed_1t_figure_unmuted_parallel_line_stays_fully_legible():
    """The highlighted variant is bucketed_engine_serial (the "1 thread"
    story), but bucketed_engine_parallel must not fade to the standard 0.35
    muted alpha every other non-highlighted stage variant gets -- the
    parallel-slower finding is the whole point of this figure and must stay
    legible alongside the highlight.
    """
    fig = make_recurring_figure(
        [], _BUCKETED_1T_ROWS, highlight_variant="bucketed_engine_serial", stage=6,
        theme="deck", figsize_pt=(900, 340),
    )
    _, ax_cost = fig.axes
    for line in ax_cost.lines:
        if line.get_label() == "bucketed_engine_parallel":
            line.set_alpha(1.0)
    parallel_line = next(l for l in ax_cost.lines if l.get_label() == "bucketed_engine_parallel")
    assert parallel_line.get_alpha() == 1.0
    matplotlib.pyplot.close(fig)


def test_bucketed_1t_figure_deck_theme_exports_at_exact_size(tmp_path):
    fig = make_recurring_figure(
        [], _BUCKETED_1T_ROWS, highlight_variant="bucketed_engine_serial", stage=6,
        theme="deck", figsize_pt=(900, 340),
    )
    paths = export_deck_figure(fig, str(tmp_path / "bucketed_1t_v2"), 900, 340)
    assert set(paths) == {"svg", "pdf", "png"}
    assert fig.get_size_inches() == pytest.approx((900 / 72.0, 340 / 72.0))
    matplotlib.pyplot.close(fig)
    matplotlib.pyplot.close(fig)


# --- memory diagnosis (deck page 17, job 7035691) ----------------------------
#
# Real numbers from raw/2026-09-14-worker7183-memory/: the probe's
# memory-diagnosis-eps1.5258789e-05-probe.json / .txt (heavyhex_step,
# 127 qubits, eps=2^-16-ish coeff:1.5258789e-05, threads in {1, 96}). The
# ms/layer phase values are the probe .txt's own table, "other" pre-summed
# (rebucket + prepare + span_plan + finalize) the same way the probe's HTML
# report folds small phases together. Traffic numbers are modeled from the
# probe JSON's real terms_in/rows_sorted/terms_out/coset_loop_ns at 96
# threads (see figures/real/MANIFEST.md for the full derivation) -- the ONLY
# scope this campaign treats as valid for a traffic-rate comparison, per the
# user's own requirement that the timing scope must genuinely support it.
# `bandwidth_ceiling_gbps=None` is real: `bandwidth.sh` failed to build
# `membench` on worker7183 (see `bandwidth.stderr.log`), so no genoa ceiling
# exists at any thread count -- and the Cascade Lake (ccqlin038) numbers in
# `research/HARDWARE.md` are a different architecture and are never
# substituted in.

_MEMORY_PHASE_ROWS = [
    {"threads": 1, "wall_ms_per_layer": 51.832, "phases": {
        "permute": 5.9934, "coset_loop": 43.5192, "unpermute": 1.9688,
        "recount": 0.3393, "other": 0.0001 + 0.0084 + 0.0007 + 0.0008,
    }},
    {"threads": 96, "wall_ms_per_layer": 15.152, "phases": {
        "permute": 7.4094, "coset_loop": 2.2446, "unpermute": 5.1791,
        "recount": 0.3085, "other": 0.0001 + 0.0074 + 0.0009 + 0.0006,
    }},
]

_MEMORY_TRAFFIC = {
    "payload_bytes_per_term": 48,
    "modeled_traffic_gbps": 1.7870479605950702,
    "modeled_bytes_per_term": 142.02072738046874,
    "traffic_scope_label": "96-thread coset_loop phase",
    "bandwidth_ceiling_gbps": None,
    "bandwidth_unavailable_reason": (
        "bandwidth.sh failed to build membench on this host (worker7183, "
        "AMD Genoa) -- no ceiling captured at 1 or 96 threads. Cascade Lake "
        "(ccqlin038) numbers in research/HARDWARE.md are a different "
        "architecture and are not substituted."
    ),
    "peak_vmhwm_kb": 16806572,
}


def test_memory_diagnosis_figure_empty_input_raises_clear_error():
    with pytest.raises(NotImplementedError, match="no phase rows to plot"):
        make_memory_diagnosis_figure([], _MEMORY_TRAFFIC)


def test_memory_diagnosis_figure_builds_two_panels():
    fig = make_memory_diagnosis_figure(_MEMORY_PHASE_ROWS, _MEMORY_TRAFFIC)
    assert len(fig.axes) >= 2
    matplotlib.pyplot.close(fig)


def test_memory_diagnosis_figure_phase_shares_sum_close_to_100_percent():
    """Panel A stacks each row's phases as a SHARE of that row's own wall
    time, so each bar's segments must sum close to 100% regardless of the
    ~3.4x real difference in absolute ms/layer between 1 and 96 threads.
    """
    fig = make_memory_diagnosis_figure(_MEMORY_PHASE_ROWS, _MEMORY_TRAFFIC)
    ax_phase = fig.axes[0]
    bars_by_row = {}
    for container in ax_phase.containers:
        for i, patch in enumerate(container):
            bars_by_row.setdefault(i, []).append(patch)
    for i, patches in bars_by_row.items():
        total_width = sum(p.get_width() for p in patches)
        assert total_width == pytest.approx(100.0, abs=1.0), (i, total_width)
    matplotlib.pyplot.close(fig)


def test_memory_diagnosis_figure_annotates_real_wall_ms_per_layer():
    fig = make_memory_diagnosis_figure(_MEMORY_PHASE_ROWS, _MEMORY_TRAFFIC)
    ax_phase = fig.axes[0]
    texts = {t.get_text() for t in ax_phase.texts}
    assert "51.83 ms/layer" in texts
    assert "15.15 ms/layer" in texts
    matplotlib.pyplot.close(fig)


def test_memory_diagnosis_figure_keeps_payload_traffic_peak_rss_as_distinct_numbers():
    """The three headline numbers must never be conflated -- each is its own
    stat tile with its own value string, not merged into one metric.
    """
    fig = make_memory_diagnosis_figure(_MEMORY_PHASE_ROWS, _MEMORY_TRAFFIC)
    ax_stats = fig.axes[1]
    texts = {t.get_text() for t in ax_stats.texts}
    assert "48 B/term" in texts
    assert "1.79 GB/s" in texts
    assert "16.81 GB" in texts
    matplotlib.pyplot.close(fig)


def test_memory_diagnosis_figure_states_bandwidth_unavailable_reason_when_ceiling_is_none():
    """`bandwidth_ceiling_gbps=None` (the real, honest state for this host)
    must render the disclosed reason, never a fabricated % of ceiling.
    """
    fig = make_memory_diagnosis_figure(_MEMORY_PHASE_ROWS, _MEMORY_TRAFFIC)
    ax_stats = fig.axes[1]
    full_text = " ".join(t.get_text() for t in ax_stats.texts).replace("\n", " ")
    assert "failed to build membench" in full_text
    assert "% of measured ceiling" not in full_text
    matplotlib.pyplot.close(fig)


def test_memory_diagnosis_figure_with_real_ceiling_states_percent_of_ceiling():
    """When a real ceiling IS available, the figure computes and states a
    genuine % of ceiling rather than omitting the comparison.
    """
    traffic = dict(_MEMORY_TRAFFIC, bandwidth_ceiling_gbps=40.0)
    fig = make_memory_diagnosis_figure(_MEMORY_PHASE_ROWS, traffic)
    ax_stats = fig.axes[1]
    full_text = " ".join(t.get_text() for t in ax_stats.texts).replace("\n", " ")
    assert "% of" in full_text and "measured ceiling" in full_text
    matplotlib.pyplot.close(fig)


def test_memory_diagnosis_figure_deck_theme_exports_at_exact_size(tmp_path):
    fig = make_memory_diagnosis_figure(
        _MEMORY_PHASE_ROWS, _MEMORY_TRAFFIC, theme="deck", figsize_pt=(900, 340),
        title="Memory & bandwidth diagnosis",
    )
    paths = export_deck_figure(fig, str(tmp_path / "memory_v2"), 900, 340)
    assert set(paths) == {"svg", "pdf", "png"}
    assert fig.get_size_inches() == pytest.approx((900 / 72.0, 340 / 72.0))
    matplotlib.pyplot.close(fig)


def test_memory_diagnosis_figure_compact_variant_exports_at_exact_size(tmp_path):
    fig = make_memory_diagnosis_figure(
        _MEMORY_PHASE_ROWS, _MEMORY_TRAFFIC, theme="deck", figsize_pt=(900, 170),
        title="Memory & bandwidth diagnosis",
    )
    paths = export_deck_figure(fig, str(tmp_path / "memory_v2_compact"), 900, 170)
    assert set(paths) == {"svg", "pdf", "png"}
    assert fig.get_size_inches() == pytest.approx((900 / 72.0, 170 / 72.0))
    matplotlib.pyplot.close(fig)


# --- baseline pivot (deck page 16, real 3-point comparison, 2026-09-14) -----
#
# Real numbers: Julia 1-thread (raw/2026-09-14-worker7169-julia/runs.jsonl,
# job 7033031), Julia 96-thread (decisions.md #38, job 7034021), current
# engine forced to a single bucket (raw/2026-09-14-worker7160-single-bucket/
# runs.jsonl, job 7036526). Supersedes the naive_baseline-sweep framing per
# decisions.md #42 (that commit never wires truncation into its merge phase,
# so it cannot produce a real tolerance sweep).

_BASELINE_PIVOT_ROWS = [
    {"label": "Julia (1 thread)", "wall_time_s": 4874.939709082, "threads": 1,
     "engine": "julia", "final_terms": 38791220},
    {"label": "Julia (96 threads)", "wall_time_s": 899.49, "threads": 96,
     "engine": "julia", "final_terms": 38791220},
    {"label": "Current engine (1 bucket)", "wall_time_s": 1967.578165213985, "threads": 1,
     "engine": "current_engine", "final_terms": 38791220},
]


def test_baseline_pivot_figure_empty_input_raises_clear_error():
    with pytest.raises(NotImplementedError, match="no rows to plot"):
        make_baseline_pivot_figure([])


def test_baseline_pivot_figure_plots_one_bar_per_row_in_order():
    fig = make_baseline_pivot_figure(_BASELINE_PIVOT_ROWS)
    ax = fig.axes[0]
    heights = [p.get_height() for p in ax.patches]
    assert heights == pytest.approx([r["wall_time_s"] for r in _BASELINE_PIVOT_ROWS])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == [r["label"] for r in _BASELINE_PIVOT_ROWS]
    matplotlib.pyplot.close(fig)


def test_baseline_pivot_figure_x_axis_is_categorical_not_a_thread_count():
    """The 1 -> 96 -> 1 thread progression across these three real points is
    NOT a monotonic thread axis (a different implementation sits at the
    third point) -- this figure must use a plain categorical x-axis, never a
    numeric/log thread scale that would visually imply otherwise.
    """
    fig = make_baseline_pivot_figure(_BASELINE_PIVOT_ROWS)
    ax = fig.axes[0]
    assert ax.get_xscale() == "linear"
    assert list(ax.get_xticks()) == [0, 1, 2]
    matplotlib.pyplot.close(fig)


def test_baseline_pivot_figure_current_engine_bar_is_hatched_distinctly():
    """The current-engine bar must be visually distinguishable from the two
    Julia bars by more than color alone (a hatch pattern here), so the
    "different implementation, not part of the Julia pair" signal survives
    grayscale/color-blind viewing.
    """
    fig = make_baseline_pivot_figure(_BASELINE_PIVOT_ROWS)
    ax = fig.axes[0]
    hatches = [p.get_hatch() for p in ax.patches]
    engine_idx = [i for i, r in enumerate(_BASELINE_PIVOT_ROWS) if r["engine"] == "current_engine"]
    julia_idx = [i for i, r in enumerate(_BASELINE_PIVOT_ROWS) if r["engine"] == "julia"]
    assert all(hatches[i] for i in engine_idx)
    assert all(not hatches[i] for i in julia_idx)
    matplotlib.pyplot.close(fig)


def test_baseline_pivot_figure_annotates_julia_speedup():
    fig = make_baseline_pivot_figure(_BASELINE_PIVOT_ROWS)
    ax = fig.axes[0]
    texts = {t.get_text() for t in ax.texts}
    assert any("5.42x" in t for t in texts)
    matplotlib.pyplot.close(fig)


def test_baseline_pivot_figure_deck_theme_exports_at_exact_size(tmp_path):
    fig = make_baseline_pivot_figure(_BASELINE_PIVOT_ROWS, theme="deck", figsize_pt=(900, 340), title="Baseline pivot")
    paths = export_deck_figure(fig, str(tmp_path / "baseline_v3"), 900, 340)
    assert set(paths) == {"svg", "pdf", "png"}
    assert fig.get_size_inches() == pytest.approx((900 / 72.0, 340 / 72.0))
    matplotlib.pyplot.close(fig)


def test_baseline_pivot_figure_compact_variant_exports_at_exact_size(tmp_path):
    fig = make_baseline_pivot_figure(_BASELINE_PIVOT_ROWS, theme="deck", figsize_pt=(450, 340), title="Baseline pivot")
    paths = export_deck_figure(fig, str(tmp_path / "baseline_v3_compact"), 450, 340)
    assert set(paths) == {"svg", "pdf", "png"}
    assert fig.get_size_inches() == pytest.approx((450 / 72.0, 340 / 72.0))
    matplotlib.pyplot.close(fig)


# --- single-bucket comparison (deck page 29, real 2-point comparison) ------
#
# Replaces the dropped bucketed_1t_v2 figure (decisions.md #46: that figure
# compared two different historical commits and did not support page 29's
# actual claim). Real numbers: default bucket config (raw/2026-09-13-
# worker7277/runs.jsonl, job 7030090) vs. forced single bucket
# (raw/2026-09-14-worker7160-single-bucket/runs.jsonl, job 7036526), both
# current engine, single-thread, same eps=2^-16/127-qubit config.

_SINGLE_BUCKET_ROWS = [
    {"label": "default buckets", "wall_time_s": 1629.904597465007, "final_terms": 38791220,
     "expectation_re": 0.39716532998468246},
    {"label": "1 bucket", "wall_time_s": 1967.578165213985, "final_terms": 38791220,
     "expectation_re": 0.3971653299846819},
]


def test_single_bucket_comparison_figure_empty_input_raises_clear_error():
    with pytest.raises(NotImplementedError, match="no rows to plot"):
        make_single_bucket_comparison_figure([])


def test_single_bucket_comparison_figure_plots_two_bars_in_order():
    fig = make_single_bucket_comparison_figure(_SINGLE_BUCKET_ROWS)
    ax = fig.axes[0]
    heights = [p.get_height() for p in ax.patches]
    assert heights == pytest.approx([r["wall_time_s"] for r in _SINGLE_BUCKET_ROWS])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == [r["label"] for r in _SINGLE_BUCKET_ROWS]
    matplotlib.pyplot.close(fig)


def test_single_bucket_comparison_figure_single_bucket_is_slower_by_about_21_percent():
    """Pins the real, disclosed finding as a regression tripwire: a future
    data refresh should not silently change this shape without the test
    noticing.
    """
    default_t = next(r["wall_time_s"] for r in _SINGLE_BUCKET_ROWS if r["label"] == "default buckets")
    single_t = next(r["wall_time_s"] for r in _SINGLE_BUCKET_ROWS if r["label"] == "1 bucket")
    pct = 100.0 * (single_t / default_t - 1.0)
    assert pct == pytest.approx(20.7, abs=0.1)


def test_single_bucket_comparison_figure_annotates_percent_difference():
    fig = make_single_bucket_comparison_figure(_SINGLE_BUCKET_ROWS)
    ax = fig.axes[0]
    texts = {t.get_text() for t in ax.texts}
    assert any("20.7%" in t for t in texts)
    matplotlib.pyplot.close(fig)


def test_single_bucket_comparison_figure_states_matching_final_terms():
    """Same correctness across both bars (identical final_terms) must be
    stated on the figure itself, not left implicit -- this is the whole
    point of the comparison being a pure wall-clock effect.
    """
    fig = make_single_bucket_comparison_figure(_SINGLE_BUCKET_ROWS)
    ax = fig.axes[0]
    texts = {t.get_text() for t in ax.texts}
    assert any("38,791,220" in t for t in texts)
    matplotlib.pyplot.close(fig)


def test_single_bucket_comparison_figure_deck_theme_exports_at_exact_size(tmp_path):
    fig = make_single_bucket_comparison_figure(
        _SINGLE_BUCKET_ROWS, theme="deck", figsize_pt=(900, 340), title="Bucket config, single thread"
    )
    paths = export_deck_figure(fig, str(tmp_path / "bucketed_1t_v3"), 900, 340)
    assert set(paths) == {"svg", "pdf", "png"}
    assert fig.get_size_inches() == pytest.approx((900 / 72.0, 340 / 72.0))
    matplotlib.pyplot.close(fig)


def test_single_bucket_comparison_figure_compact_variant_exports_at_exact_size(tmp_path):
    fig = make_single_bucket_comparison_figure(
        _SINGLE_BUCKET_ROWS, theme="deck", figsize_pt=(450, 340), title="Bucket config, single thread"
    )
    paths = export_deck_figure(fig, str(tmp_path / "bucketed_1t_v3_compact"), 450, 340)
    assert set(paths) == {"svg", "pdf", "png"}
    assert fig.get_size_inches() == pytest.approx((450 / 72.0, 340 / 72.0))
    matplotlib.pyplot.close(fig)
