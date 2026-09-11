"""CI gate on the jl head-to-head *protocol* — never on its measurements.

The driver is ``benchmarks/python/bench_jl_performance.py``; its results and
narrative live in ``benchmarks/python/jl_performance/``. Neither is imported by
the shipped package, and neither can run in CI: the comparison needs a Julia
toolchain with a pinned PauliPropagation.jl, and a timing number from a shared
CI runner would be worthless anyway.

What *can* be checked in CI, and is checked here, is that the protocol's logic
is right — because that logic is what turns runtimes into a claim, and a bug in
it would silently produce a confident, wrong answer:

* the acceptance rule (direction consistency across pairs, mixed signs reported
  as indistinguishable rather than as a small win),
* the crossover interpolation and the indistinguishable-zone bookkeeping,
* the parity gate's wiring — that a term-count mismatch really does raise and
  block a timing, rather than being logged and ignored,
* the interleaving order,
* the ``min_abs_coeff`` boundary transformation that makes the two engines'
  truncation rules identical,
* that the schema-v1 gate lists the driver feeds both engines still mirror
  ``examples/common/circuits.py`` gate for gate.

Everything is exercised on synthetic numbers or on circuit *construction*
(never propagation), so this module is fast and needs no Julia, no matplotlib,
and no timing.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER_PATH = REPO_ROOT / "benchmarks" / "python" / "bench_jl_performance.py"


@pytest.fixture(scope="module")
def driver():
    """Import the driver by path; it is not an installed module.

    Registering it in ``sys.modules`` before executing is required — its
    dataclasses look themselves up there while being processed.
    """
    if not DRIVER_PATH.exists():
        pytest.skip(f"driver not present at {DRIVER_PATH}")
    spec = importlib.util.spec_from_file_location("_bench_jl_performance", DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_jl_performance"] = module
    spec.loader.exec_module(module)
    return module


def pair(rust_s: float, jl_s: float) -> dict[str, float]:
    return {"rust_s": rust_s, "jl_s": jl_s}


# --------------------------------------------------------------------------
# The module must stay import-clean
# --------------------------------------------------------------------------


def test_driver_top_level_imports_are_stdlib_only():
    """The driver's *module-level* imports must all be stdlib.

    CI imports this driver purely for its protocol math, in a job that has
    neither Julia nor matplotlib. Engine, plotting and harness imports therefore
    have to stay inside functions.

    Checked by reading the file's AST rather than by inspecting
    ``sys.modules``: a ``sys.modules`` check is order-dependent — another test in
    the same session that imports matplotlib would make it pass or fail for
    reasons having nothing to do with this file — whereas the AST *is* the
    property.
    """
    import ast

    source = DRIVER_PATH.read_text()
    roots: set[str] = set()
    for node in ast.parse(source).body:  # module level only, not nested
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    non_stdlib = sorted(r for r in roots if r not in sys.stdlib_module_names)
    assert not non_stdlib, (
        f"{DRIVER_PATH.name} imports {non_stdlib} at module level; CI imports it "
        "for the protocol math alone, so engine/plotting imports must be inside "
        "functions"
    )


def test_protocol_bar_is_at_least_five_pairs(driver):
    assert driver.DEFAULT_PAIRS >= 5


# --------------------------------------------------------------------------
# Acceptance rule: direction consistency, never a p-value
# --------------------------------------------------------------------------


def test_all_pairs_agreeing_gives_a_directional_verdict(driver):
    result = driver.analyze_pairs([pair(1.0, 2.0), pair(1.1, 2.3), pair(0.9, 1.7)])
    assert result["sign_consistent"] is True
    assert result["verdict"] == "paulistrings"
    assert result["median_ratio_jl_over_rust"] == pytest.approx(2.0, rel=0.05)
    assert result["n_pairs"] == 3


def test_all_pairs_agreeing_the_other_way_names_julia(driver):
    result = driver.analyze_pairs([pair(2.0, 1.0), pair(2.2, 1.1), pair(1.8, 0.9)])
    assert result["sign_consistent"] is True
    assert result["verdict"] == "julia"
    assert result["median_ratio_jl_over_rust"] == pytest.approx(0.5, rel=0.05)


def test_one_disagreeing_pair_makes_the_result_indistinguishable(driver):
    """The core rule. Four pairs favour this engine, one favours Julia; the
    median is above 1. That must NOT be reported as a small win.
    """
    result = driver.analyze_pairs(
        [pair(1.0, 1.06), pair(1.0, 1.04), pair(1.0, 0.98), pair(1.0, 1.05), pair(1.0, 1.03)]
    )
    assert result["sign_consistent"] is False
    assert result["verdict"] == "indistinguishable"
    # the median is still reported, as context only
    assert result["median_ratio_jl_over_rust"] > 1.0


def test_an_exact_tie_counts_as_disagreement(driver):
    """A pair with equal runtimes supports neither direction, so it breaks
    consistency rather than being rounded to whichever side the others took."""
    result = driver.analyze_pairs([pair(1.0, 2.0), pair(1.0, 1.0), pair(1.0, 2.0)])
    assert result["sign_consistent"] is False
    assert result["verdict"] == "indistinguishable"


def test_ratios_use_the_documented_convention(driver):
    """ratio = t_julia / t_paulistrings, so above 1 means paulistrings faster."""
    assert driver.pair_ratios([pair(1.0, 3.0)]) == [3.0]
    assert driver.pair_ratios([pair(4.0, 1.0)]) == [0.25]


def test_a_zero_runtime_is_rejected_not_averaged(driver):
    with pytest.raises(ValueError, match="non-positive runtime"):
        driver.pair_ratios([pair(1.0, 2.0), pair(0.0, 2.0)])


def test_median_of_no_pairs_names_the_protocol_reason(driver):
    with pytest.raises(ValueError, match="no pairs"):
        driver.median([])


# --------------------------------------------------------------------------
# Interleaving order
# --------------------------------------------------------------------------


def test_within_pair_order_alternates_abba(driver):
    order = [driver.rust_runs_first(i) for i in range(6)]
    assert order == [True, False, True, False, True, False]


# --------------------------------------------------------------------------
# Crossover localization
# --------------------------------------------------------------------------


def test_crossover_interpolates_geometrically(driver):
    """Symmetric ratios about 1 put the crossing at the geometric mean.

    Interpolation is linear in log10(ratio) against log10(terms), so ratios of
    1/4 and 4 at 10^2 and 10^6 terms cross 1 at 10^4.
    """
    points = [
        {"final_terms": 100, "median_ratio_jl_over_rust": 0.25, "sign_consistent": True},
        {"final_terms": 1_000_000, "median_ratio_jl_over_rust": 4.0, "sign_consistent": True},
    ]
    out = driver.interpolate_crossover(points)
    assert out["crossover_terms"] == pytest.approx(10_000.0, rel=1e-9)
    assert out["lower"]["final_terms"] == 100
    assert out["upper"]["final_terms"] == 1_000_000


def test_crossover_can_interpolate_on_the_peak_term_axis(driver):
    """The study localizes on ``peak_terms``, because the peak is what the
    engine actually had to hold and sort — and it must be the same axis the
    figures use, or the marked crossover would not sit on the drawn curves.
    """
    points = [
        {
            "final_terms": 11,
            "peak_terms": 100,
            "median_ratio_jl_over_rust": 0.25,
            "sign_consistent": True,
        },
        {
            "final_terms": 900_000,
            "peak_terms": 1_000_000,
            "median_ratio_jl_over_rust": 4.0,
            "sign_consistent": True,
        },
    ]
    out = driver.interpolate_crossover(points, "peak_terms")
    assert out["crossover_terms"] == pytest.approx(10_000.0, rel=1e-9)
    assert out["terms_key"] == "peak_terms"
    assert out["lower"]["peak_terms"] == 100
    # the default axis gives a different answer, which is why it is explicit
    assert driver.interpolate_crossover(points)["crossover_terms"] != pytest.approx(
        10_000.0, rel=1e-6
    )


def test_indistinguishable_zone_honours_the_term_axis(driver):
    points = [
        {"final_terms": 5, "peak_terms": 700, "sign_consistent": False},
        {"final_terms": 50, "peak_terms": 900, "sign_consistent": False},
    ]
    assert driver.indistinguishable_zone(points, "peak_terms") == {
        "lo": 700,
        "hi": 900,
        "n_configs": 2,
    }


def test_no_direction_change_reports_no_crossover(driver):
    points = [
        {"final_terms": 100, "median_ratio_jl_over_rust": 1.5, "sign_consistent": True},
        {"final_terms": 10_000, "median_ratio_jl_over_rust": 2.5, "sign_consistent": True},
    ]
    out = driver.interpolate_crossover(points)
    assert out["crossover_terms"] is None
    assert "no direction change" in out["note"]
    assert "paulistrings faster" in out["note"]


def test_mixed_sign_points_cannot_bracket_a_crossover(driver):
    """A point whose pairs disagreed has no direction, so it must not serve as
    an endpoint of an interval claiming the direction changed across it."""
    points = [
        {"final_terms": 100, "median_ratio_jl_over_rust": 0.5, "sign_consistent": False},
        {"final_terms": 10_000, "median_ratio_jl_over_rust": 2.0, "sign_consistent": False},
    ]
    out = driver.interpolate_crossover(points)
    assert out["crossover_terms"] is None
    assert "sign-consistent" in out["note"]


def test_crossover_inside_an_indistinguishable_zone_is_flagged(driver):
    points = [
        {"final_terms": 100, "median_ratio_jl_over_rust": 0.25, "sign_consistent": True},
        {"final_terms": 10_000, "median_ratio_jl_over_rust": 1.02, "sign_consistent": False},
        {"final_terms": 1_000_000, "median_ratio_jl_over_rust": 4.0, "sign_consistent": True},
    ]
    out = driver.interpolate_crossover(points)
    assert out["crossover_terms"] == pytest.approx(10_000.0, rel=1e-9)
    assert out["inside_indistinguishable_zone"] is True
    assert out["indistinguishable_zone"] == {"lo": 10_000, "hi": 10_000, "n_configs": 1}


def test_indistinguishable_zone_is_none_when_every_point_agreed(driver):
    points = [
        {"final_terms": 100, "median_ratio_jl_over_rust": 0.5, "sign_consistent": True},
        {"final_terms": 10_000, "median_ratio_jl_over_rust": 2.0, "sign_consistent": True},
    ]
    assert driver.indistinguishable_zone(points) is None


def test_indistinguishable_zone_spans_the_mixed_points(driver):
    points = [
        {"final_terms": 500, "median_ratio_jl_over_rust": 1.0, "sign_consistent": False},
        {"final_terms": 90_000, "median_ratio_jl_over_rust": 1.0, "sign_consistent": False},
        {"final_terms": 1_000_000, "median_ratio_jl_over_rust": 2.0, "sign_consistent": True},
    ]
    assert driver.indistinguishable_zone(points) == {
        "lo": 500,
        "hi": 90_000,
        "n_configs": 2,
    }


# --------------------------------------------------------------------------
# The truncation-boundary transformation
# --------------------------------------------------------------------------


def test_julia_cutoff_makes_the_two_boundary_rules_identical(driver):
    """This engine drops ``|c| <= eps``; jl drops ``|c| < eps``.

    Handing jl ``nextafter(eps, +inf)`` closes the gap exactly, because no
    float lies strictly between ``eps`` and its successor — so jl's strict
    comparison against the successor accepts and rejects exactly what this
    engine's inclusive comparison against ``eps`` does.
    """
    for eps in (2.0**-14, 1e-4, 2.0**-8, 3e-7):
        eps_jl = driver.julia_min_abs_coeff(eps)
        assert eps_jl > eps
        assert math.nextafter(eps, math.inf) == eps_jl
        # nothing strictly between: the two rules coincide
        assert math.nextafter(eps_jl, -math.inf) == eps
        # a coefficient exactly on the cutoff: dropped by both
        assert not (abs(eps) < eps_jl) or True  # jl's rule
        assert abs(eps) < eps_jl  # jl drops it, matching our <=


def test_julia_cutoff_rejects_a_non_positive_threshold(driver):
    """min_abs_coeff = 0 is banned for comparative runs: jl keeps exact zeros
    and this engine's merge drops them (benchmarks/julia/README.md §P9)."""
    with pytest.raises(ValueError, match="must be positive"):
        driver.julia_min_abs_coeff(0.0)


def test_dyadic_detection_flags_the_load_bearing_cutoffs(driver):
    assert driver.is_dyadic(2.0**-14)
    assert driver.is_dyadic(0.5)
    assert driver.is_dyadic(1.0)
    assert not driver.is_dyadic(1e-4)
    assert not driver.is_dyadic(3e-3)
    assert not driver.is_dyadic(0.0)


# --------------------------------------------------------------------------
# Memory accounting
# --------------------------------------------------------------------------


def test_bytes_per_term_subtracts_the_process_floor(driver):
    # 1 GiB peak, 0.5 GiB floor, 1e6 terms -> 0.5 GiB / 1e6 terms
    got = driver.bytes_per_term(1024.0 * 1024.0, 512.0 * 1024.0, 1_000_000)
    assert got == pytest.approx(512.0 * 1024.0 * 1024.0 / 1e6)


def test_bytes_per_term_is_none_when_the_payload_is_below_the_floor(driver):
    assert driver.bytes_per_term(1000.0, 1000.0, 10) is None
    assert driver.bytes_per_term(900.0, 1000.0, 10) is None
    assert driver.bytes_per_term(None, 1000.0, 10) is None
    assert driver.bytes_per_term(2000.0, None, 10) is None
    assert driver.bytes_per_term(2000.0, 1000.0, 0) is None


# --------------------------------------------------------------------------
# Parity comparison
# --------------------------------------------------------------------------


def test_identical_per_layer_counts_are_parity(driver):
    assert driver.check_parity([1, 5, 20], [1, 5, 20]) == []


def test_a_differing_layer_is_reported_with_its_index(driver):
    problems = driver.check_parity([1, 5, 20], [1, 6, 20])
    assert len(problems) == 1
    assert "layer 2: 5 vs 6" in problems[0]


def test_a_layer_count_mismatch_names_the_one_gate_per_channel_rule(driver):
    problems = driver.check_parity([1, 5, 20], [1, 5])
    assert "layer count" in problems[0]
    assert "one gate object must be one channel" in problems[0]


def test_missing_julia_counts_are_a_parity_problem_not_a_pass(driver):
    problems = driver.check_parity([1, 2, 3], None)
    assert problems and "no per-layer term counts" in problems[0]


# --------------------------------------------------------------------------
# Parity-gate and pair-loop wiring (spawners stubbed; no Julia, no timing)
# --------------------------------------------------------------------------


def _rust_parity_result(layers, final=None, expectation=0.25):
    return {
        "engine": "paulistrings",
        "input_terms": 1,
        "final_terms": final if final is not None else layers[-1],
        "per_layer_terms": list(layers),
        "expectation": expectation,
    }


def _jl_parity_result(layers, final=None, expectation=0.25, peak=None):
    return {
        "result": {
            "final_terms": final if final is not None else layers[-1],
            "per_layer_terms": list(layers),
            "peak_terms": peak if peak is not None else max(layers),
            "expectation": {"re": expectation, "im": 0.0},
        }
    }


def test_parity_gate_passes_and_records_its_evidence(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(
        driver, "_spawn_rust_leg", lambda *a, **k: _rust_parity_result([2, 8, 30])
    )
    monkeypatch.setattr(
        driver, "_spawn_jl_leg", lambda *a, **k: _jl_parity_result([2, 8, 30])
    )
    out = driver.parity_gate(tmp_path / "r.json", tmp_path / "j.json", label="unit")
    assert out["ok"] is True
    assert out["problems"] == []
    assert out["n_layers"] == 3
    assert out["rust_final_terms"] == out["jl_final_terms"] == 30
    assert out["expectation_delta"] == pytest.approx(0.0)


def test_parity_gate_raises_and_blocks_on_a_layer_mismatch(driver, monkeypatch, tmp_path):
    """The gate must be *blocking*. A term-count divergence means the engines
    did different amounts of work, so their runtimes are not comparable at
    all — it can never be downgraded to a warning."""
    monkeypatch.setattr(
        driver, "_spawn_rust_leg", lambda *a, **k: _rust_parity_result([2, 8, 30])
    )
    monkeypatch.setattr(
        driver, "_spawn_jl_leg", lambda *a, **k: _jl_parity_result([2, 9, 30])
    )
    with pytest.raises(driver.ParityFailure) as excinfo:
        driver.parity_gate(tmp_path / "r.json", tmp_path / "j.json", label="unit")
    assert "PARITY FAILED" in str(excinfo.value)
    assert "disqualified" in str(excinfo.value)
    assert "layer 2: 8 vs 9" in str(excinfo.value)


def test_parity_gate_raises_on_a_diverging_expectation(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(
        driver,
        "_spawn_rust_leg",
        lambda *a, **k: _rust_parity_result([2, 8, 30], expectation=0.25),
    )
    monkeypatch.setattr(
        driver,
        "_spawn_jl_leg",
        lambda *a, **k: _jl_parity_result([2, 8, 30], expectation=0.26),
    )
    with pytest.raises(driver.ParityFailure, match="expectation differs"):
        driver.parity_gate(tmp_path / "r.json", tmp_path / "j.json", label="unit")


def _stub_legs(driver, monkeypatch, rust_times, jl_times, *, terms=1000, engines=None):
    """Stub both spawners, recording the order legs were requested in.

    ``engines`` is an optional list the stub appends each rust leg's requested
    layer engine to, so a test can assert the setting reached the subprocess
    boundary instead of stopping at the function that accepted it.
    """
    order: list[str] = []
    rust_iter = iter(rust_times)
    jl_iter = iter(jl_times)

    def rust(task, mode, *, threads=1, timeout=0.0, rust_engine="sorted"):
        order.append("rust")
        if engines is not None:
            engines.append(rust_engine)
        return {
            "propagation_s": next(rust_iter),
            "final_terms": terms,
            "peak_terms": terms,
            "memory": {"vmhwm_kb": 200000.0, "vmrss_start_kb": 40000.0},
        }

    def jl(task, *, timed, threads=1, timeout=0.0):
        order.append("jl")
        return {
            "result": {"final_terms": terms, "peak_terms": terms},
            "timing": {"wall_warm_s": next(jl_iter), "wall_cold_s": 1.5},
            "memory": {"vmhwm_kb": 900000.0, "vmrss_start_kb": 640000.0},
        }

    monkeypatch.setattr(driver, "_spawn_rust_leg", rust)
    monkeypatch.setattr(driver, "_spawn_jl_leg", jl)
    return order


def test_run_pairs_interleaves_abba(driver, monkeypatch, tmp_path):
    order = _stub_legs(driver, monkeypatch, [1.0] * 4, [2.0] * 4)
    driver.run_pairs(
        tmp_path / "r.json", tmp_path / "j.json", label="unit", pairs=4, log=lambda m: None
    )
    assert order == ["rust", "jl", "jl", "rust", "rust", "jl", "jl", "rust"]


def test_run_pairs_reports_every_pair_and_the_median(driver, monkeypatch, tmp_path):
    _stub_legs(driver, monkeypatch, [1.0, 1.0, 1.0], [3.0, 2.0, 4.0])
    out = driver.run_pairs(
        tmp_path / "r.json", tmp_path / "j.json", label="unit", pairs=3, log=lambda m: None
    )
    assert out["n_pairs"] == 3
    assert out["ratio_jl_over_rust_per_pair"] == [3.0, 2.0, 4.0]
    assert out["median_ratio_jl_over_rust"] == 3.0
    assert out["sign_consistent"] is True
    assert out["verdict"] == "paulistrings"
    assert len(out["pairs"]) == 3
    assert out["pairs"][0]["order"] == "rust-first"
    assert out["pairs"][1]["order"] == "julia-first"


def test_run_pairs_records_per_engine_memory_and_bytes_per_term(driver, monkeypatch, tmp_path):
    _stub_legs(driver, monkeypatch, [1.0], [2.0], terms=1_000_000)
    out = driver.run_pairs(
        tmp_path / "r.json", tmp_path / "j.json", label="unit", pairs=1, log=lambda m: None
    )
    mem = out["peak_memory"]
    # rust: (200000 - 40000) KiB over 1e6 terms
    assert mem["rust_bytes_per_term"] == pytest.approx(160000.0 * 1024.0 / 1e6)
    # jl: (900000 - 640000) KiB over 1e6 terms
    assert mem["jl_bytes_per_term"] == pytest.approx(260000.0 * 1024.0 / 1e6)
    assert mem["rust_floor_kb"] == 40000.0
    assert mem["jl_floor_kb"] == 640000.0


def test_run_pairs_rejects_engines_disagreeing_on_term_count(driver, monkeypatch, tmp_path):
    """A disagreement during timing means the parity gate missed something;
    the timing must not be reported."""

    def rust(task, mode, *, threads=1, timeout=0.0, rust_engine="sorted"):
        return {
            "propagation_s": 1.0,
            "final_terms": 1000,
            "peak_terms": 1000,
            "memory": {},
        }

    def jl(task, *, timed, threads=1, timeout=0.0):
        return {
            "result": {"final_terms": 1001, "peak_terms": 1001},
            "timing": {"wall_warm_s": 2.0, "wall_cold_s": 1.5},
            "memory": {},
        }

    monkeypatch.setattr(driver, "_spawn_rust_leg", rust)
    monkeypatch.setattr(driver, "_spawn_jl_leg", jl)
    with pytest.raises(driver.ParityFailure, match="disagree on final term count"):
        driver.run_pairs(
            tmp_path / "r.json", tmp_path / "j.json", label="unit", pairs=1,
            log=lambda m: None,
        )


def test_run_pairs_rejects_a_term_count_that_moved_between_legs(driver, monkeypatch, tmp_path):
    counts = iter([1000, 1000, 1234, 1234])

    def rust(task, mode, *, threads=1, timeout=0.0, rust_engine="sorted"):
        return {
            "propagation_s": 1.0,
            "final_terms": next(counts),
            "peak_terms": 1000,
            "memory": {},
        }

    def jl(task, *, timed, threads=1, timeout=0.0):
        return {
            "result": {"final_terms": next(counts), "peak_terms": 1000},
            "timing": {"wall_warm_s": 2.0, "wall_cold_s": 1.5},
            "memory": {},
        }

    monkeypatch.setattr(driver, "_spawn_rust_leg", rust)
    monkeypatch.setattr(driver, "_spawn_jl_leg", jl)
    with pytest.raises(driver.ParityFailure, match="varied across legs"):
        driver.run_pairs(
            tmp_path / "r.json", tmp_path / "j.json", label="unit", pairs=2,
            log=lambda m: None,
        )


# --------------------------------------------------------------------------
# The rust layer-engine selector (--engine)
# --------------------------------------------------------------------------


def test_the_default_layer_engine_is_the_one_every_committed_run_measured(driver):
    """`sorted` — the bucketed engine at every term count. Every result file
    committed before `--engine` existed was measured on it, so any other
    default would silently reinterpret the whole committed corpus."""
    assert driver.DEFAULT_RUST_ENGINE == "sorted"
    assert driver.DEFAULT_RUST_ENGINE in driver.RUST_ENGINES


def test_the_offered_engines_are_the_ones_the_binding_accepts(driver):
    """The driver must not offer a spelling `PauliSum.propagate` would reject,
    nor hide one it takes."""
    assert set(driver.RUST_ENGINES) == {"sorted", "auto", "direct"}


def test_run_pairs_forwards_the_layer_engine_to_every_rust_leg(driver, monkeypatch, tmp_path):
    engines: list[str] = []
    _stub_legs(driver, monkeypatch, [1.0] * 3, [2.0] * 3, engines=engines)
    out = driver.run_pairs(
        tmp_path / "r.json",
        tmp_path / "j.json",
        label="unit",
        pairs=3,
        rust_engine="auto",
        log=lambda m: None,
    )
    assert engines == ["auto", "auto", "auto"]
    assert out["rust_engine"] == "auto"


def test_run_pairs_defaults_to_the_sorting_engine(driver, monkeypatch, tmp_path):
    engines: list[str] = []
    _stub_legs(driver, monkeypatch, [1.0], [2.0], engines=engines)
    out = driver.run_pairs(
        tmp_path / "r.json", tmp_path / "j.json", label="unit", pairs=1, log=lambda m: None
    )
    assert engines == ["sorted"]
    assert out["rust_engine"] == "sorted"


def test_the_parity_gate_runs_the_engine_that_will_be_timed(driver, monkeypatch, tmp_path):
    """Gating parity on the *timed* engine is the point: a layer engine that
    changed per-layer term counts has to disqualify its own configuration, not
    be waved through by a gate that ran a different code path."""
    seen: list[str] = []

    def rust(task, mode, *, threads=1, timeout=0.0, rust_engine="sorted"):
        seen.append(rust_engine)
        return _rust_parity_result([2, 8, 30])

    monkeypatch.setattr(driver, "_spawn_rust_leg", rust)
    monkeypatch.setattr(driver, "_spawn_jl_leg", lambda *a, **k: _jl_parity_result([2, 8, 30]))
    out = driver.parity_gate(
        tmp_path / "r.json", tmp_path / "j.json", label="unit", rust_engine="auto"
    )
    assert seen == ["auto"]
    assert out["rust_engine"] == "auto"
    assert out["ok"] is True


# --------------------------------------------------------------------------
# Cutoff subsetting (--max-configs)
# --------------------------------------------------------------------------


def test_the_declared_grid_runs_when_no_subset_is_asked_for(driver):
    workload = driver.workloads()["kicked_ising"]
    points = driver.sweep_points(workload)
    assert [eps for eps, _w in points[:-1]] == list(workload.cutoffs)
    # the weight variant is the tail
    assert points[-1] == (workload.weight_variant[1], workload.weight_variant[0])


def test_max_configs_keeps_the_loosest_cutoffs_in_order(driver):
    """Loosest, not an arbitrary three: the sweep is declared loosest-first and
    the subset must be a prefix of it, so every kept configuration is the same
    configuration the full run measures."""
    workload = driver.workloads()["xxz"]
    points = driver.sweep_points(workload, max_configs=3)
    assert [eps for eps, _w in points] == list(workload.cutoffs[:3])
    assert all(w is None for _eps, w in points)


def test_max_configs_drops_the_weight_variant_with_the_tight_cutoffs(driver):
    workload = driver.workloads()["kicked_ising"]
    assert workload.weight_variant is not None
    points = driver.sweep_points(workload, max_configs=3)
    assert len(points) == 3
    assert all(w is None for _eps, w in points)


def test_max_configs_at_or_above_the_grid_size_changes_nothing(driver):
    workload = driver.workloads()["su4"]
    full = driver.sweep_points(workload)
    assert driver.sweep_points(workload, max_configs=len(workload.cutoffs)) == full
    assert driver.sweep_points(workload, max_configs=99) == full


def test_max_configs_below_one_is_rejected(driver):
    workload = driver.workloads()["xxz"]
    with pytest.raises(ValueError, match="max_configs"):
        driver.sweep_points(workload, max_configs=0)


def test_a_pilot_still_drops_the_weight_variant(driver):
    workload = driver.workloads()["kicked_ising"]
    points = driver.sweep_points(workload, include_weight_variant=False)
    assert [eps for eps, _w in points] == list(workload.cutoffs)


# --------------------------------------------------------------------------
# Cut / projection bookkeeping
# --------------------------------------------------------------------------


def test_a_leg_just_inside_the_time_budget_is_not_cut(driver):
    """40x the terms -> 400 s rust, 800 s jl warm, 1600 s per leg: under the
    1800 s (30 min) budget, so it runs. Pinned so the boundary is deliberate."""
    projection = driver.project_leg(
        terms=40_000_000, ref_terms=1_000_000, ref_rust_s=10.0, ref_ratio=2.0
    )
    assert projection["projected_jl_leg_s"] == pytest.approx(1600.0)
    assert driver.JL_RUN_BUDGET_S == pytest.approx(1800.0)
    assert driver.should_cut(projection, 1_000_000.0) is None


def test_an_over_budget_julia_leg_is_cut_with_a_reason(driver):
    """60x the terms -> 600 s rust, 1200 s jl warm, 2400 s per leg: over budget.

    Free RAM is set absurdly high so the *time* rule is what fires, not memory —
    a cut must name the reason it actually happened for.
    """
    projection = driver.project_leg(
        terms=60_000_000, ref_terms=1_000_000, ref_rust_s=10.0, ref_ratio=2.0
    )
    assert projection["projected_jl_leg_s"] == pytest.approx(2400.0)
    reason = driver.should_cut(projection, 1_000_000.0)
    assert reason is not None
    assert "budget" in reason and "min" in reason


def test_an_over_memory_julia_leg_is_cut_on_free_ram(driver):
    projection = driver.project_leg(
        terms=100_000_000, ref_terms=1_000_000, ref_rust_s=0.001, ref_ratio=1.0
    )
    # 100e6 terms * 0.74 KiB = ~70.6 GiB, over half of 100 GiB free
    reason = driver.should_cut(projection, 100.0)
    assert reason is not None and "GiB" in reason


def test_an_affordable_leg_is_not_cut(driver):
    projection = driver.project_leg(
        terms=2_000_000, ref_terms=1_000_000, ref_rust_s=1.0, ref_ratio=2.0
    )
    assert driver.should_cut(projection, 200.0) is None


# --------------------------------------------------------------------------
# The gate lists really do mirror examples/common/circuits.py
#
# Construction only, never propagation: building a Circuit is cheap, and the
# mirroring is exactly the invariant that lets ONE description drive both
# engines. If a builder in circuits.py changes and the mirror does not, the
# study would silently compare two different circuits.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def circuits_module():
    sys.path.insert(0, str(REPO_ROOT / "examples"))
    try:
        from common import circuits
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"examples/common not importable: {exc}")
    return circuits


def _channels_from_gate_list(gate_list, n):
    from paulistrings.interop import circuit_from_json

    return circuit_from_json({"gates": gate_list}, n)


def test_kicked_ising_gate_list_mirrors_the_circuit_builder(driver, circuits_module):
    theta_h = 5.0 * math.pi / 16.0
    gates = driver.kicked_ising_gates(127, trotter_steps=2, theta_h=theta_h)
    reference = circuits_module.heavy_hex_kicked_ising(
        127, trotter_steps=2, theta_h=theta_h
    )
    # 2 steps x (127 rx + 144 ZZ)
    assert len(gates) == 2 * (127 + 144)
    assert len(_channels_from_gate_list(gates, 127)) == len(reference) == len(gates)
    # An X layer first, then a ZZ layer -- Kim et al. SI Eq. (4) ordering, which
    # is not cosmetic (see the builder's docstring).
    assert [g["name"] for g in gates[:127]] == ["rx"] * 127
    assert gates[127]["name"] == "pauli_rotation"
    assert gates[127]["pauli"] == "ZZ"
    assert all(g["theta"] == theta_h for g in gates[:127])
    assert all(
        g["theta"] == pytest.approx(-math.pi / 2)
        for g in gates
        if g["name"] == "pauli_rotation"
    )


def test_kicked_ising_deep_workload_is_the_configuration_it_claims(driver, circuits_module):
    """The saturation falsification test's workload, pinned to its claim.

    ``kicked_ising_deep`` exists to move a *fixed* term count away from the
    reachable Pauli set's closure by deepening the circuit, so the only thing
    that may differ from ``kicked_ising`` is the depth and the kick angle — same
    lattice, same observable, same Clifford ``theta_zz``. It also reuses
    benchmark C's proven angle/depth (all 5420 per-layer counts identical at
    ``2^-14``), which is only true if the gate list really is 20 steps at
    ``7pi/32``: pin both here rather than trusting the workload's prose.
    """
    workloads = driver.workloads()
    deep = workloads["kicked_ising_deep"]
    shallow = workloads["kicked_ising"]

    assert driver.KICKED_ISING_DEEP_STEPS == 20
    assert deep.n_qubits == shallow.n_qubits == 127
    assert deep.observable == shallow.observable  # Z_62, same seed operator
    assert deep.state == shallow.state == "z+"
    # a weight variant would put a second knob on the size axis, and the
    # falsification test reads a single-parameter ratio-vs-terms trend
    assert deep.weight_variant is None
    # the two tightest points are what decide the verdict; 2^-13 is deliberately
    # between the even exponents so the decisive band has two measured points
    assert deep.cutoffs == (2.0**-8, 2.0**-10, 2.0**-12, 2.0**-13, 2.0**-14)
    assert all(driver.is_dyadic(eps) for eps in deep.cutoffs), (
        "dyadic on purpose: the one-ulp threshold mitigation must be load-bearing "
        "in the falsification test too, not dodged with powers of ten"
    )

    gates = deep.gates()
    assert len(gates) == 20 * (127 + 144) == 5420
    reference = circuits_module.heavy_hex_kicked_ising(
        127, trotter_steps=20, theta_h=driver.KICKED_ISING_DEEP_THETA_H
    )
    assert len(_channels_from_gate_list(gates, 127)) == len(reference) == len(gates)
    assert [g["name"] for g in gates[:127]] == ["rx"] * 127
    assert all(g["theta"] == pytest.approx(7.0 * math.pi / 32.0) for g in gates[:127])
    assert all(
        g["theta"] == pytest.approx(-math.pi / 2)
        for g in gates
        if g["name"] == "pauli_rotation"
    )
    # Depth only *appends* Trotter steps: the deep list's first five steps are
    # the five-step list at the same angle, gate for gate. That is what makes
    # "same circuit family, more steps" a true statement about the comparison
    # rather than about the prose.
    assert gates[: 5 * (127 + 144)] == driver.kicked_ising_gates(
        127, trotter_steps=5, theta_h=driver.KICKED_ISING_DEEP_THETA_H
    )


def test_xxz_gate_list_mirrors_the_circuit_builder(driver, circuits_module):
    gates = driver.xxz_gates(20, 2, Jz=0.5, dt=0.1)
    reference = circuits_module.xxz_chain_trotter(20, 2, Jz=0.5, dt=0.1)
    assert len(gates) == 2 * 3 * 19
    assert len(_channels_from_gate_list(gates, 20)) == len(reference) == len(gates)
    # XX, YY, ZZ per bond, with theta = 2*dt, 2*dt, 2*dt*Jz
    assert [g["pauli"] for g in gates[:3]] == ["XX", "YY", "ZZ"]
    assert gates[0]["theta"] == pytest.approx(0.2)
    assert gates[1]["theta"] == pytest.approx(0.2)
    assert gates[2]["theta"] == pytest.approx(0.1)
    # even bonds first, then odd
    assert gates[0]["qubits"] == [0, 1]
    assert gates[3]["qubits"] == [2, 3]


def test_su4_gate_list_mirrors_the_circuit_builder(driver, circuits_module):
    gates = driver.su4_gates(8, 4, driver.SU4_SEED)
    reference = circuits_module.random_su4_staircase(8, 4, driver.SU4_SEED)
    assert len(_channels_from_gate_list(gates, 8)) == len(reference) == len(gates)
    assert all(g["name"] == "unitary_2q" for g in gates)
    # brickwork: even layers on (0,1),(2,3),..., odd layers on (1,2),(3,4),...
    assert gates[0]["qubits"] == [0, 1]
    assert gates[4]["qubits"] == [1, 2]
    # 4x4 matrices, serialized as rows of [re, im] pairs
    for g in gates:
        assert len(g["matrix"]) == 4
        assert all(len(row) == 4 and all(len(c) == 2 for c in row) for row in g["matrix"])


def test_su4_gate_list_is_a_deterministic_function_of_the_seed(driver):
    a = driver.su4_gates(6, 3, 12345)
    b = driver.su4_gates(6, 3, 12345)
    c = driver.su4_gates(6, 3, 54321)
    assert a == b
    assert a != c


def test_su4_matrices_are_unitary(driver):
    """The Julia runner range-checks this and hard-errors, so catching a
    non-unitary block here is cheaper than discovering it mid-study."""
    import numpy as np

    for g in driver.su4_gates(6, 2, driver.SU4_SEED):
        m = np.array([[complex(re, im) for re, im in row] for row in g["matrix"]])
        assert np.allclose(m.conj().T @ m, np.eye(4), atol=1e-12)


# --------------------------------------------------------------------------
# Workload / reference declarations
# --------------------------------------------------------------------------


def test_every_workload_is_declared_coherently(driver):
    for key, workload in driver.workloads().items():
        assert workload.key == key
        assert workload.cutoffs, f"{key} has no cutoff sweep"
        # loosest first: the sweep must descend, so a heavy leg can be projected
        # from the lighter one before it and cut before it is run
        assert list(workload.cutoffs) == sorted(workload.cutoffs, reverse=True)
        assert all(eps > 0 for eps in workload.cutoffs), "0 cutoff banned (§P9)"
        for label in workload.observable:
            assert len(label) == workload.n_qubits
            assert set(label) <= set("IXYZ")


def test_accuracy_references_carry_an_oracle_and_its_provenance(driver):
    assert driver.ACCURACY_REFERENCES
    for ref in driver.ACCURACY_REFERENCES:
        assert ref.cutoffs
        assert list(ref.cutoffs) == sorted(ref.cutoffs, reverse=True)
        assert ref.oracle_source.strip(), f"{ref.key} has no oracle provenance"
        assert "summary.json" in ref.oracle_source
        assert math.isfinite(ref.oracle)


def test_accuracy_bars_are_the_two_stated_ones(driver):
    assert driver.ACCURACY_BARS == (1e-2, 1e-3)


def test_extension_provenance_resolves_the_binary_that_actually_ran(driver):
    """It must describe the imported ``.so``, not the driver's working tree.

    A study can run from a branch with no build of its own, against an
    extension built elsewhere; attributing the numbers to the branch would name
    code that never executed.
    """
    out = driver.extension_provenance()
    assert "error" not in out, out.get("error")
    assert out["package_path"].endswith("paulistrings/__init__.py")
    # resolved from the imported module, so it is the checkout really in use
    assert Path(out["checkout"]).is_dir()
    assert out["commit"] and out["branch"]


def test_thread_counts_start_at_one_and_are_powers_of_two(driver):
    assert driver.THREAD_COUNTS[0] == 1
    assert all(t & (t - 1) == 0 for t in driver.THREAD_COUNTS)
    assert list(driver.THREAD_COUNTS) == sorted(driver.THREAD_COUNTS)
