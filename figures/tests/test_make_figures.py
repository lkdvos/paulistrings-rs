import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from make_compact_figures import (
    make_accuracy_figure,
    make_hash_communication_figure,
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


# --- accuracy ------------------------------------------------------------------


def test_accuracy_figure_builds_from_reference_observed_pairs():
    rows = [
        {"run_id": "run-a", "reference_value": 0.5, "observed_value": 0.501},
        {"run_id": "run-b", "reference_value": -0.2, "observed_value": -0.198},
    ]
    fig = make_accuracy_figure(rows)
    assert fig is not None
    matplotlib.pyplot.close(fig)
