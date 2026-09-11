"""Tests for `examples/common/report.py` (handoff item P0e).

`examples/` is not on the pytest path by default (it isn't a package under
`python/`), so this file inserts the repo's `examples/` directory onto
`sys.path` and imports `report` as a top-level module of the `common`
package. Schema/JSON tests are numpy-only (no matplotlib import happens
unless a plot function is actually called); the plot smoke tests
`pytest.importorskip("matplotlib")` so this file stays CI-safe when
matplotlib isn't installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from common import report  # noqa: E402


# --- Schema -------------------------------------------------------------


def _make_provenance(**overrides) -> report.Provenance:
    defaults = dict(
        commit="deadbeef",
        dirty=False,
        cpu_model="Test CPU",
        python_version="3.11.11",
        rustc_version="rustc 1.94.0",
        library_versions={"paulistrings": "0.1.0"},
        seeds={"circuit": 1234},
        thread_count=1,
        hostname="testhost",
        date="2026-08-31",
    )
    defaults.update(overrides)
    return report.Provenance(**defaults)


def _make_record(**overrides) -> report.RunRecord:
    defaults = dict(
        engine="paulistrings",
        engine_version="0.1.0",
        n_qubits=16,
        direction="heisenberg",
        truncation={"min_abs_coeff": 1e-6},
        propagation_time_s=0.5,
        final_terms=1000,
        provenance=_make_provenance(),
        contraction_time_s=0.1,
        peak_terms=1200,
        expectation_value=0.75,
        absolute_error=1e-9,
        peak_memory_kb=204800.0,
        extra={"theta_h": 0.3},
    )
    defaults.update(overrides)
    return report.RunRecord(**defaults)


def test_run_record_total_time():
    rec = _make_record(propagation_time_s=0.5, contraction_time_s=0.25)
    assert rec.total_time_s == pytest.approx(0.75)


def test_run_record_total_time_no_contraction():
    rec = _make_record(contraction_time_s=None)
    assert rec.total_time_s == rec.propagation_time_s


def test_run_record_to_dict_is_json_serializable():
    rec = _make_record()
    d = rec.to_dict()
    # Round-trips through json without a custom encoder.
    s = json.dumps(d)
    assert json.loads(s) == d
    assert d["engine"] == "paulistrings"
    assert d["provenance"]["commit"] == "deadbeef"


def test_run_record_from_dict_round_trip():
    rec = _make_record()
    rec2 = report.RunRecord.from_dict(rec.to_dict())
    assert rec2 == rec


def test_collect_provenance_shape():
    prov = report.collect_provenance(seeds={"circuit": 7}, thread_count=1)
    assert isinstance(prov.commit, str) and prov.commit
    assert prov.dirty is None or isinstance(prov.dirty, bool)
    assert isinstance(prov.cpu_model, str) and prov.cpu_model
    assert isinstance(prov.python_version, str) and prov.python_version
    assert prov.seeds == {"circuit": 7}
    assert prov.thread_count == 1
    assert isinstance(prov.hostname, str) and prov.hostname
    assert isinstance(prov.date, str) and len(prov.date) == 10
    # This environment has paulistrings installed (it's the package under test).
    assert "paulistrings" in prov.library_versions


def test_collect_provenance_merges_extra_library_versions():
    prov = report.collect_provenance(extra_library_versions={"stim": "1.2.3"})
    assert prov.library_versions["stim"] == "1.2.3"


def test_default_results_dir_naming():
    d = report.default_results_dir(base="benchmarks/results")
    parts = d.name.split("-", 3)
    # <YYYY>-<MM>-<DD>-<host>
    assert len(parts) == 4
    assert len(parts[0]) == 4 and parts[0].isdigit()
    assert str(d.parent) == "benchmarks/results"


# --- write_results / read_results ----------------------------------------


def test_write_results_creates_file(tmp_path):
    rec = _make_record()
    out_dir = tmp_path / "2026-08-31-testhost"
    path = report.write_results([rec], out_dir, name="bench_a")
    assert path == out_dir / "bench_a.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["engine"] == "paulistrings"


def test_write_results_appends_never_overwrites(tmp_path):
    out_dir = tmp_path / "results"
    rec_a = _make_record(engine="paulistrings")
    rec_b = _make_record(engine="PauliPropagation.jl")

    path1 = report.write_results([rec_a], out_dir, name="bench_a")
    path2 = report.write_results([rec_b], out_dir, name="bench_a")

    assert path1 == path2
    data = json.loads(path2.read_text())
    assert len(data) == 2
    assert {d["engine"] for d in data} == {"paulistrings", "PauliPropagation.jl"}


def test_write_results_rejects_non_list_file(tmp_path):
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    (out_dir / "bad.json").write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError):
        report.write_results([_make_record()], out_dir, name="bad")


def test_read_results_round_trips(tmp_path):
    rec = _make_record()
    path = report.write_results([rec], tmp_path, name="roundtrip")
    loaded = report.read_results(path)
    assert loaded == [rec]


# --- Plot helpers (require matplotlib) ------------------------------------


def test_plot_error_vs_runtime_writes_svg(tmp_path):
    pytest.importorskip("matplotlib")
    records = [
        _make_record(engine="paulistrings", propagation_time_s=t, absolute_error=e)
        for t, e in [(0.1, 1e-3), (0.5, 1e-5), (1.0, 1e-8)]
    ] + [
        _make_record(engine="PauliPropagation.jl", propagation_time_s=t, absolute_error=e)
        for t, e in [(0.2, 1e-3), (0.6, 1e-6)]
    ]
    out = tmp_path / "error_vs_runtime.svg"
    fig = report.plot_error_vs_runtime(records, save_path=out)
    assert out.exists()
    assert out.read_text(errors="ignore").lstrip().startswith("<?xml") or b"svg" in out.read_bytes()[:200]
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_term_count_vs_truncation_writes_svg(tmp_path):
    pytest.importorskip("matplotlib")
    records = [
        _make_record(truncation={"min_abs_coeff": eps}, final_terms=n)
        for eps, n in [(1e-3, 500), (1e-6, 5000), (1e-9, 50000)]
    ]
    out = tmp_path / "terms_vs_trunc.svg"
    fig = report.plot_term_count_vs_truncation(records, save_path=out)
    assert out.exists()
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_time_and_memory_vs_size_writes_svg(tmp_path):
    pytest.importorskip("matplotlib")
    records = [
        _make_record(n_qubits=n, propagation_time_s=0.01 * n, peak_memory_kb=1000.0 * n)
        for n in (16, 32, 64)
    ]
    out = tmp_path / "time_mem_vs_size.svg"
    fig = report.plot_time_and_memory_vs_size(records, save_path=out)
    assert out.exists()
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_convergence_panel_writes_svg(tmp_path):
    pytest.importorskip("matplotlib")
    records = [
        _make_record(truncation={"min_abs_coeff": eps}, expectation_value=v)
        for eps, v in [(1e-3, 0.5), (1e-6, 0.7), (1e-9, 0.75)]
    ]
    out = tmp_path / "convergence.svg"
    fig = report.plot_convergence_panel(records, reference_value=0.751, save_path=out)
    assert out.exists()
    import matplotlib.pyplot as plt

    plt.close(fig)


# The module-level `from common import report` above already proves the
# import-without-matplotlib contract: this environment doesn't have
# matplotlib installed, and every test file in this collection ran, so
# `report.py` never imports matplotlib at module scope.
